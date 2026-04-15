"""
Reflex model: LLM backbone + flipped cross-attention control head.

Instruction tokens = Q (the instruction looks at the machine)
Machine state = K/V (the machine is the knowledge base)

The LLM understands the instruction. The control head cross-attends
to the machine state to decide the next opcode.
"""

import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from chip8 import DISPLAY_SIZE

BACKBONE_DIM = 1536
BACKBONE_ID = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16"

PC_WINDOW = 32
STATE_DIM = DISPLAY_SIZE + 16 + 2 + PC_WINDOW + 2  # 2100
STATE_TOKENS = 32  # project state into this many K/V tokens
MAX_TOKENS = 20    # pad/truncate instruction token IDs

N_HIGH = 256
N_LOW = 256


def load_backbone():
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    from mlx_lm import load as mlx_load
    print(f"  Loading backbone: {BACKBONE_ID}")
    model, tokenizer = mlx_load(BACKBONE_ID)
    model.freeze()
    return model, tokenizer


def encode_instruction(text: str, backbone, tokenizer):
    """Returns (hidden_states [1, seq_len, 1536], token_ids [MAX_TOKENS])."""
    tokens = tokenizer.encode(text)
    hidden = backbone.model(mx.array([tokens]))
    tid = np.zeros(MAX_TOKENS, dtype=np.float32)
    for i, t in enumerate(tokens[:MAX_TOKENS]):
        tid[i] = t / 150000.0
    return hidden.astype(mx.float32), tid


class ReflexModel(nn.Module):
    """
    Flipped cross-attention: instruction queries the machine state.

    Input:
      - backbone_hidden: [B, seq_len, BACKBONE_DIM] — LLM hidden states
      - state: [B, STATE_DIM] — raw machine state
      - token_ids: [B, MAX_TOKENS] — raw token IDs for operand precision

    Output: (high_byte_logits, low_byte_logits) — next opcode
    """
    def __init__(self, dim=256, n_heads=4):
        super().__init__()
        # Instruction tokens → queries
        self.instr_norm = nn.RMSNorm(BACKBONE_DIM)
        self.instr_proj = nn.Linear(BACKBONE_DIM, dim)

        # Token IDs → additional query features
        self.tid_proj = nn.Linear(MAX_TOKENS, dim)

        # Machine state → K/V tokens
        self.state_proj = nn.Linear(STATE_DIM, STATE_TOKENS * dim)

        # Cross-attention: instruction looks at machine
        self.cross_attn = nn.MultiHeadAttention(dim, n_heads)
        self.cross_norm = nn.RMSNorm(dim)

        # Self-attention over instruction tokens
        self.self_attn = nn.MultiHeadAttention(dim, n_heads)
        self.self_norm = nn.RMSNorm(dim)

        # Output
        self.out_norm = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        self.high_head = nn.Linear(dim, N_HIGH)
        self.low_head = nn.Linear(dim, N_LOW)

    def __call__(self, backbone_hidden, state, token_ids):
        B = state.shape[0]

        # Instruction tokens → queries
        q = self.instr_proj(self.instr_norm(backbone_hidden))  # [B, seq_len, dim]

        # Add token ID info as bias
        tid_feat = self.tid_proj(token_ids)[:, None, :]  # [B, 1, dim]
        q = q + tid_feat

        # Machine state → K/V tokens
        kv = self.state_proj(state).reshape(B, STATE_TOKENS, -1)  # [B, STATE_TOKENS, dim]

        # Cross-attention: instruction looks at machine
        h = self.cross_norm(q)
        h = q + self.cross_attn(h, kv, kv)

        # Self-attention
        r = self.self_norm(h)
        h = h + self.self_attn(r, r, r)

        # Pool + output
        h = self.out_norm(h).mean(axis=1)
        h = h + self.mlp(h)

        return self.high_head(h), self.low_head(h)
