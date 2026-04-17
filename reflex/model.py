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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .chip8 import DISPLAY_SIZE

BACKBONE_DIM = 1536
BACKBONE_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

STATE_DIM = DISPLAY_SIZE + 16 + 2 + 2  # 2068 (display + registers + I + prev_opcode)
STATE_TOKENS = 32
MAX_TOKENS = 20  # pad/truncate instruction token IDs
TID_VOCAB = 4096  # hash embedding table size for token IDs
TID_DIM = 64  # embedding dimension per token

N_HIGH = 256
N_LOW = 256

# Previous-opcode encoding in the GRU input
PREV_OP_DIM = 128
PREV_HALF = PREV_OP_DIM // 2


def load_backbone(device="cuda"):
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading backbone: {BACKBONE_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BACKBONE_ID,
        dtype=torch.bfloat16,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cuda":
        model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def encode_instruction(text: str, backbone, tokenizer, device="cuda"):
    """Returns (hidden_states [1, seq_len, 1536], token_ids [MAX_TOKENS] as ints)."""
    tokens = tokenizer.encode(text, return_tensors="pt")

    with torch.no_grad():
        if device != "cpu":
            tokens = tokens.to(device)
        outputs = backbone(tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # Get last layer hidden states

    tid = np.zeros(MAX_TOKENS, dtype=np.int32)
    token_list = tokens[0].cpu().numpy()
    for i, t in enumerate(token_list[:MAX_TOKENS]):
        tid[i] = int(t % TID_VOCAB)  # hash to embedding table size

    return hidden.float().to("cpu"), tid


class GRUCell(nn.Module):
    """Gated Recurrent Unit — gives the control head autoregressive memory."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.Wz = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.Wr = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.Wh = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, h):
        xh = torch.cat([x, h], dim=-1)
        z = torch.sigmoid(self.Wz(xh))
        r = torch.sigmoid(self.Wr(xh))
        xrh = torch.cat([x, r * h], dim=-1)
        return (1 - z) * h + z * torch.tanh(self.Wh(xrh))


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
        self.instr_norm = nn.LayerNorm(BACKBONE_DIM)
        self.instr_proj = nn.Linear(BACKBONE_DIM, dim)

        # Token ID embeddings
        self.tid_embed = nn.Embedding(TID_VOCAB, TID_DIM)
        self.tid_proj = nn.Linear(MAX_TOKENS * TID_DIM, dim)

        # Machine state → K/V tokens
        self.state_proj = nn.Linear(STATE_DIM, STATE_TOKENS * dim)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)

        # Pool
        self.out_norm = nn.LayerNorm(dim)

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

    def forward(
        self,
        backbone_hidden,
        state,
        token_ids,
        prev_hi=None,
        prev_lo=None,
        h_state=None,
    ):
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
        h_attn, _ = self.cross_attn(h, kv, kv)
        h = q + h_attn

        # Self-attention
        r = self.self_norm(h)
        h_attn, _ = self.self_attn(r, r, r)
        h = h + h_attn

        # Pool → context vector
        ctx = self.out_norm(h).mean(dim=1)  # [B, dim]

        # Encode previous opcode (zeros at step 0)
        if prev_hi is None:
            prev_hi = torch.zeros((B,), dtype=torch.int32, device=state.device)
            prev_lo = torch.zeros((B,), dtype=torch.int32, device=state.device)
        prev_embed = torch.cat([self.hi_embed(prev_hi), self.lo_embed(prev_lo)], dim=-1)

        # GRU step
        if h_state is None:
            h_state = torch.zeros((B, self.dim), device=state.device)
        gru_input = torch.cat([ctx, prev_embed], dim=-1)
        h_state = self.gru(gru_input, h_state)

        # Output (h_state + residual MLP + token-ID pathway)
        out = h_state + self.mlp(h_state) + self.tid_out(tid_embedded)
        return self.high_head(out), self.low_head(out), h_state
