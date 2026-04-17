"""
Reflex model: frozen LLM backbone + autoregressive control head.

RV32I variant: emits 32-bit instructions as six classification heads over
the ISA's field decomposition (opcode, rd, funct3, rs1, rs2, funct7).
Six heads instead of CHIP-8's two — the model learns the ISA structure
directly.

Flow:
  instruction → frozen LLM → hidden states (understanding, same every step)
  previous-instruction history → K/V tokens (one per emitted instruction)
  cross-attention: instruction queries instruction history
  GRU: carries hidden state across steps, takes previous instruction fields as input
  six output heads: opcode, rd, funct3, rs1, rs2, funct7
"""

import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np

BACKBONE_DIM = 1536
BACKBONE_ID = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16"

MAX_TOKENS = 32    # pad/truncate instruction token IDs
TID_VOCAB = 4096
TID_DIM = 192

# RV32I field class counts (what each head predicts).
N_OPCODE = 128   # 7 bits
N_RD     = 32    # 5 bits
N_FUNCT3 = 8     # 3 bits
N_RS1    = 32    # 5 bits
N_RS2    = 32    # 5 bits
N_FUNCT7 = 128   # 7 bits

FIELD_NAMES = ("opcode", "rd", "funct3", "rs1", "rs2", "funct7")
FIELD_CLASSES = (N_OPCODE, N_RD, N_FUNCT3, N_RS1, N_RS2, N_FUNCT7)

# Per-field embedding dimensions — roughly proportional to bit-widths.
# Summed: 32+16+16+16+16+32 = 128.
FIELD_EMBED_DIMS = (32, 16, 16, 16, 16, 32)
PREV_OP_DIM = sum(FIELD_EMBED_DIMS)  # 128

# Cross-attention K/V: one slot per emitted instruction + 1 start token.
# Programs top out at ~16 ops in our dataset; 32 leaves plenty of headroom.
MAX_KV_LEN = 32

# History-field embedding dims (separate from the GRU-input embeddings so
# the two pathways can specialise).
HIST_FIELD_EMBED_DIMS = (32, 16, 16, 16, 16, 32)
HIST_CONCAT_DIM = sum(HIST_FIELD_EMBED_DIMS)  # 128


def load_backbone():
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    from mlx_lm import load as mlx_load
    print(f"  Loading backbone: {BACKBONE_ID}")
    model, tokenizer = mlx_load(BACKBONE_ID)
    model.freeze()
    return model, tokenizer


def encode_instruction(text: str, backbone, tokenizer):
    """Returns (hidden_states [1, seq_len, 1536], token_ids [MAX_TOKENS] as ints)."""
    tokens = tokenizer.encode(text)
    hidden = backbone.model(mx.array([tokens]))
    tid = np.zeros(MAX_TOKENS, dtype=np.int32)
    for i, t in enumerate(tokens[:MAX_TOKENS]):
        tid[i] = t % TID_VOCAB
    return hidden.astype(mx.float32), tid


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.Wz = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.Wr = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.Wh = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def __call__(self, x, h):
        xh = mx.concatenate([x, h], axis=-1)
        z = mx.sigmoid(self.Wz(xh))
        r = mx.sigmoid(self.Wr(xh))
        xrh = mx.concatenate([x, r * h], axis=-1)
        return (1 - z) * h + z * mx.tanh(self.Wh(xrh))


def _split_fields(instr: mx.array) -> tuple[mx.array, ...]:
    """Split a tensor of 32-bit instructions into (opcode, rd, funct3, rs1, rs2, funct7)."""
    return (
        instr & 0x7F,
        (instr >> 7) & 0x1F,
        (instr >> 12) & 0x7,
        (instr >> 15) & 0x1F,
        (instr >> 20) & 0x1F,
        (instr >> 25) & 0x7F,
    )


class ReflexModel(nn.Module):
    """
    Autoregressive control head. Six classification heads for RV32I's six
    field-position groupings (opcode / rd / funct3 / rs1 / rs2 / funct7).
    For non-R-type instructions those bit positions carry immediate bits
    instead — the model learns the mapping.

    Pathways (unchanged from CHIP-8 build apart from the head count):
      - Cross-attention over emitted-instruction history (K/V).
      - Token-ID MLP for operand-precision detail.
      - GRU with scheduled-sampling-friendly previous-instruction input.

    Input (per step t):
      - backbone_hidden: [B, seq_len, BACKBONE_DIM]
      - prev_instrs: [B, MAX_KV_LEN] int32 — emitted-instruction history
      - kv_valid_count: t + 1 (start token + t emitted)
      - token_ids: [B, MAX_TOKENS]
      - prev_fields: tuple of 6 [B] int32 arrays — last emitted instruction
      - h_state: [B, dim] GRU hidden state

    Output: (logits_per_field: tuple of 6, new_h_state)
    """
    def __init__(self, dim=512, n_heads=8):
        super().__init__()
        self.dim = dim

        # Instruction hidden-states → queries
        self.instr_norm = nn.RMSNorm(BACKBONE_DIM)
        self.instr_proj = nn.Linear(BACKBONE_DIM, dim)

        # Token-ID pathway
        self.tid_embed = nn.Embedding(TID_VOCAB, TID_DIM)
        self.tid_proj = nn.Linear(MAX_TOKENS * TID_DIM, dim)

        # History embeddings — one per field.
        self.hist_embeds = [
            nn.Embedding(cls, d)
            for cls, d in zip(FIELD_CLASSES, HIST_FIELD_EMBED_DIMS)
        ]
        self.hist_proj = nn.Linear(HIST_CONCAT_DIM, dim)
        self.hist_pos_embed = nn.Embedding(MAX_KV_LEN + 1, dim)
        self.kv_start_embed = nn.Embedding(1, dim)

        # Cross / self attention
        self.cross_attn = nn.MultiHeadAttention(dim, n_heads)
        self.cross_norm = nn.RMSNorm(dim)
        self.self_attn = nn.MultiHeadAttention(dim, n_heads)
        self.self_norm = nn.RMSNorm(dim)
        self.out_norm = nn.RMSNorm(dim)

        # Previous-instruction field embeddings (separate weights from history).
        self.prev_embeds = [
            nn.Embedding(cls, d)
            for cls, d in zip(FIELD_CLASSES, FIELD_EMBED_DIMS)
        ]

        # GRU
        self.gru = GRUCell(dim + PREV_OP_DIM, dim)

        # Token-ID → output MLP
        self.tid_out = nn.Sequential(
            nn.Linear(MAX_TOKENS * TID_DIM, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        # Output MLP + six heads
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        self.heads = [nn.Linear(dim, cls) for cls in FIELD_CLASSES]

    def _embed_history(self, prev_instrs: mx.array) -> mx.array:
        """prev_instrs [B, MAX_KV_LEN] int32 → [B, MAX_KV_LEN, dim]."""
        fields = _split_fields(prev_instrs)
        parts = [emb(f) for emb, f in zip(self.hist_embeds, fields)]
        concat = mx.concatenate(parts, axis=-1)           # [B, KV, HIST_CONCAT_DIM]
        return self.hist_proj(concat)                     # [B, KV, dim]

    def _build_kv(self, prev_instrs: mx.array, kv_valid_count: int) -> tuple[mx.array, mx.array]:
        B = prev_instrs.shape[0]
        hist = self._embed_history(prev_instrs)           # [B, MAX_KV_LEN, dim]
        pos = self.hist_pos_embed(mx.arange(1, MAX_KV_LEN + 1))[None]
        hist = hist + pos
        start = self.kv_start_embed(mx.zeros((B, 1), dtype=mx.int32))
        start = start + self.hist_pos_embed(mx.zeros((1,), dtype=mx.int32))[None]
        kv = mx.concatenate([start, hist], axis=1)        # [B, MAX_KV_LEN+1, dim]
        valid = mx.arange(MAX_KV_LEN + 1) < kv_valid_count
        mask = mx.where(valid, mx.array(0.0), mx.array(-1e9))
        mask = mask[None, None, None, :]
        return kv, mask

    def __call__(self, backbone_hidden, prev_instrs, kv_valid_count, token_ids,
                 prev_fields=None, h_state=None):
        B = backbone_hidden.shape[0]

        q = self.instr_proj(self.instr_norm(backbone_hidden))
        tid_embedded = self.tid_embed(token_ids).reshape(B, -1)
        q = q + self.tid_proj(tid_embedded)[:, None, :]

        kv, attn_mask = self._build_kv(prev_instrs, kv_valid_count)
        h = self.cross_norm(q)
        h = q + self.cross_attn(h, kv, kv, mask=attn_mask)

        r = self.self_norm(h)
        h = h + self.self_attn(r, r, r)

        ctx = self.out_norm(h).mean(axis=1)

        if prev_fields is None:
            prev_fields = tuple(mx.zeros((B,), dtype=mx.int32) for _ in FIELD_CLASSES)
        prev_parts = [emb(f) for emb, f in zip(self.prev_embeds, prev_fields)]
        prev_embed = mx.concatenate(prev_parts, axis=-1)

        if h_state is None:
            h_state = mx.zeros((B, self.dim))
        gru_input = mx.concatenate([ctx, prev_embed], axis=-1)
        h_state = self.gru(gru_input, h_state)

        out = h_state + self.mlp(h_state) + self.tid_out(tid_embedded)
        logits = tuple(head(out) for head in self.heads)
        return logits, h_state
