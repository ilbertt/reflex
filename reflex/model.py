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

from .chip8 import DISPLAY_SIZE

BACKBONE_DIM = 1536
BACKBONE_ID = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16"

STATE_DIM = DISPLAY_SIZE + 16 + 2 + 2  # 2068 (display + registers + I + prev_opcode)
STATE_TOKENS = 32
MAX_TOKENS = 20    # pad/truncate instruction token IDs
TID_VOCAB = 4096   # hash embedding table size for token IDs
TID_DIM = 64       # embedding dimension per token

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
    """Returns (hidden_states [1, seq_len, 1536], token_ids [MAX_TOKENS] as ints)."""
    tokens = tokenizer.encode(text)
    hidden = backbone.model(mx.array([tokens]))
    tid = np.zeros(MAX_TOKENS, dtype=np.int32)
    for i, t in enumerate(tokens[:MAX_TOKENS]):
        tid[i] = t % TID_VOCAB  # hash to embedding table size
    return hidden.astype(mx.float32), tid


class ReflexModel(nn.Module):
    """
    Flipped cross-attention: instruction queries the machine state.

    Two pathways merge before the output heads:
      1. Cross-attention pathway (the core): LLM hidden states → queries,
         machine state → keys/values. Handles program structure — "what kind
         of opcode comes next given where we are."
      2. Token ID pathway: raw token IDs → learned embeddings → MLP. Handles
         operand precision — "which specific digit, which specific sprite."
         Needed because mean-pooled LLM embeddings don't separate "draw digit 1"
         from "draw digit 7" well enough (0.9943 cosine similarity).

    Input:
      - backbone_hidden: [B, seq_len, BACKBONE_DIM] — LLM hidden states
      - state: [B, STATE_DIM] — raw machine state
      - token_ids: [B, MAX_TOKENS] — integer token IDs

    Output: (high_byte_logits, low_byte_logits)
    """
    def __init__(self, dim=512, n_heads=8):
        super().__init__()

        # Instruction tokens → queries
        self.instr_norm = nn.RMSNorm(BACKBONE_DIM)
        self.instr_proj = nn.Linear(BACKBONE_DIM, dim)

        # Token IDs → learned embeddings (not normalized floats!)
        self.tid_embed = nn.Embedding(TID_VOCAB, TID_DIM)
        self.tid_proj = nn.Linear(MAX_TOKENS * TID_DIM, dim)

        # Machine state → K/V tokens
        self.state_proj = nn.Linear(STATE_DIM, STATE_TOKENS * dim)

        # Cross-attention: instruction looks at machine
        self.cross_attn = nn.MultiHeadAttention(dim, n_heads)
        self.cross_norm = nn.RMSNorm(dim)

        # Self-attention over instruction tokens
        self.self_attn = nn.MultiHeadAttention(dim, n_heads)
        self.self_norm = nn.RMSNorm(dim)

        # Pool
        self.out_norm = nn.RMSNorm(dim)

        # Token ID pathway: direct to output
        # Embeddings carry operand-level detail that mean-pooled LLM states can't
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

    def __call__(self, backbone_hidden, state, token_ids):
        B = state.shape[0]

        # Instruction tokens → queries
        q = self.instr_proj(self.instr_norm(backbone_hidden))  # [B, seq_len, dim]

        # Embed token IDs and add as query bias
        tid_embedded = self.tid_embed(token_ids).reshape(B, -1)  # [B, MAX_TOKENS * TID_DIM]
        tid_feat = self.tid_proj(tid_embedded)[:, None, :]  # [B, 1, dim]
        q = q + tid_feat

        # Machine state → K/V tokens
        kv = self.state_proj(state).reshape(B, STATE_TOKENS, -1)  # [B, STATE_TOKENS, dim]

        # Cross-attention: instruction looks at machine
        h = self.cross_norm(q)
        h = q + self.cross_attn(h, kv, kv)

        # Self-attention
        r = self.self_norm(h)
        h = h + self.self_attn(r, r, r)

        # Pool + merge token ID signal
        h = self.out_norm(h).mean(axis=1)
        tid_signal = self.tid_out(tid_embedded)  # [B, dim]
        h = h + tid_signal
        h = h + self.mlp(h)

        return self.high_head(h), self.low_head(h)
