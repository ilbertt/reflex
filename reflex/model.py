"""
Reflex model: frozen LLM backbone + autoregressive control head.

Flow:
  instruction → frozen LLM → hidden states (understanding, same every step)
  machine state at step t → K/V tokens
  cross-attention: instruction queries machine state
  GRU: carries hidden state across steps, takes previous opcode as input
  two output heads: high byte, low byte (next opcode)
"""

import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .chip8 import DISPLAY_SIZE

BACKBONE_DIM = 1536
BACKBONE_ID = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16"

STATE_DIM = DISPLAY_SIZE + 16 + 2 + 2  # 2068 (display + registers + I + prev_opcode)
STATE_TOKENS = 32
MAX_TOKENS = 32    # pad/truncate instruction token IDs
TID_VOCAB = 4096   # hash embedding table size for token IDs
TID_DIM = 64       # embedding dimension per token

N_HIGH = 256
N_LOW = 256

# Previous-opcode encoding in the GRU input
PREV_OP_DIM = 128
PREV_HALF = PREV_OP_DIM // 2


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
        tid[i] = t % TID_VOCAB  # hash to embedding table size
    return hidden.astype(mx.float32), tid


class GRUCell(nn.Module):
    """Gated Recurrent Unit — gives the control head autoregressive memory."""
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


class ReflexModel(nn.Module):
    """
    Autoregressive control head with two pathways and GRU memory.

    Cross-attention pathway (the core): LLM hidden states → queries,
      machine state → keys/values. Handles program structure — "what
      kind of opcode comes next given where we are."
    Token ID pathway: hash-embedded token IDs → learned MLP, fed directly
      into the output. Carries operand-level detail (which specific digit,
      which specific sprite name) that mean-pooled LLM states can't
      distinguish.
    GRU memory: carries hidden state across opcode predictions within a
      program. Previous opcode is an explicit input, enabling scheduled-
      sampling training (teacher-forced prev_opcode vs. model's own
      argmax) which closes the exposure-bias gap at inference.

    Input (per step t):
      - backbone_hidden: [B, seq_len, BACKBONE_DIM] — same every step
      - state: [B, STATE_DIM] — machine state at step t
      - token_ids: [B, MAX_TOKENS] — same every step
      - prev_hi, prev_lo: [B] int32 — previous opcode (None/zeros at step 0)
      - h_state: [B, dim] GRU hidden state (None at step 0)

    Output: (high_byte_logits, low_byte_logits, new_h_state)
    """
    def __init__(self, dim=512, n_heads=8):
        super().__init__()
        self.dim = dim

        # Instruction tokens → queries
        self.instr_norm = nn.RMSNorm(BACKBONE_DIM)
        self.instr_proj = nn.Linear(BACKBONE_DIM, dim)

        # Token ID embeddings
        self.tid_embed = nn.Embedding(TID_VOCAB, TID_DIM)
        self.tid_proj = nn.Linear(MAX_TOKENS * TID_DIM, dim)

        # Machine state → K/V tokens
        self.state_proj = nn.Linear(STATE_DIM, STATE_TOKENS * dim)

        # Cross-attention
        self.cross_attn = nn.MultiHeadAttention(dim, n_heads)
        self.cross_norm = nn.RMSNorm(dim)

        # Self-attention
        self.self_attn = nn.MultiHeadAttention(dim, n_heads)
        self.self_norm = nn.RMSNorm(dim)

        # Pool
        self.out_norm = nn.RMSNorm(dim)

        # Previous-opcode embeddings (one per byte)
        self.hi_embed = nn.Embedding(N_HIGH, PREV_HALF)
        self.lo_embed = nn.Embedding(N_LOW, PREV_HALF)

        # GRU: input = [context (dim), prev_op_embed (PREV_OP_DIM)]
        self.gru = GRUCell(dim + PREV_OP_DIM, dim)

        # Token ID pathway: direct to output
        self.tid_out = nn.Sequential(
            nn.Linear(MAX_TOKENS * TID_DIM, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        # Output
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        self.high_head = nn.Linear(dim, N_HIGH)
        self.low_head = nn.Linear(dim, N_LOW)

    def __call__(self, backbone_hidden, state, token_ids,
                 prev_hi=None, prev_lo=None, h_state=None):
        B = state.shape[0]

        # Instruction tokens → queries (with token-ID bias)
        q = self.instr_proj(self.instr_norm(backbone_hidden))
        tid_embedded = self.tid_embed(token_ids).reshape(B, -1)
        tid_feat = self.tid_proj(tid_embedded)[:, None, :]
        q = q + tid_feat

        # Machine state → K/V tokens
        kv = self.state_proj(state).reshape(B, STATE_TOKENS, -1)

        # Cross-attention (instruction looks at machine)
        h = self.cross_norm(q)
        h = q + self.cross_attn(h, kv, kv)

        # Self-attention
        r = self.self_norm(h)
        h = h + self.self_attn(r, r, r)

        # Pool → context vector
        ctx = self.out_norm(h).mean(axis=1)  # [B, dim]

        # Encode previous opcode (zeros at step 0)
        if prev_hi is None:
            prev_hi = mx.zeros((B,), dtype=mx.int32)
            prev_lo = mx.zeros((B,), dtype=mx.int32)
        prev_embed = mx.concatenate(
            [self.hi_embed(prev_hi), self.lo_embed(prev_lo)], axis=-1
        )

        # GRU step
        if h_state is None:
            h_state = mx.zeros((B, self.dim))
        gru_input = mx.concatenate([ctx, prev_embed], axis=-1)
        h_state = self.gru(gru_input, h_state)

        # Output (h_state + residual MLP + token-ID pathway)
        out = h_state + self.mlp(h_state) + self.tid_out(tid_embedded)
        return self.high_head(out), self.low_head(out), h_state
