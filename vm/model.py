"""
Reflex model: the action head that converts understanding into syscall signals.

Architecture:
  - Frozen Qwen backbone → full hidden states (not just last token)
  - Syscall head: classification (which syscall)
  - Arg head: classification (exact arg values from vocabulary)
  - Buffer decoder: small autoregressive transformer that cross-attends
    to backbone hidden states and generates buffer bytes one at a time

The buffer decoder can generate novel content because it attends to
the full instruction — "foo.py" and "print goodbye" are in the tokens.
"""

import struct
import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .kernel import SYS, STATE_DIM, pack_syscall


# ── Constants ──────────────────────────────────────────────────────────────

SYSCALL_TABLE = list(SYS.keys())
N_SYSCALLS = len(SYSCALL_TABLE)

ARG_DIM = 6
BUF_MAX = 256
CONTEXT_LEN = 4
BACKBONE_DIM = 1536
INNER_DIM = 512
PREV_ACTION_DIM = N_SYSCALLS + ARG_DIM

MODEL_ID = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16"

# Arg vocabulary
_arg_vocab = None


def build_arg_vocab(all_arg_values: set[int]) -> list[int]:
    global _arg_vocab
    _arg_vocab = sorted(all_arg_values)
    return _arg_vocab


def get_arg_vocab() -> list[int]:
    return _arg_vocab


def set_arg_vocab(vocab: list[int]):
    global _arg_vocab
    _arg_vocab = vocab


def arg_to_idx(val: int) -> int:
    try:
        return _arg_vocab.index(val)
    except ValueError:
        return min(range(len(_arg_vocab)), key=lambda i: abs(_arg_vocab[i] - val))


def idx_to_arg(idx: int) -> int:
    return _arg_vocab[idx]


# ── Backbone ──────────────────────────────────────────────────────────────

def load_backbone():
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    from mlx_lm import load as mlx_load
    print(f"  Loading backbone: {MODEL_ID}")
    model, tokenizer = mlx_load(MODEL_ID)
    model.freeze()
    return model, tokenizer


def encode_instruction_full(text: str, backbone, tokenizer) -> mx.array:
    """Encode instruction → full hidden states [1, seq_len, 1536]."""
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])
    hidden = backbone.model(input_ids)
    return hidden.astype(mx.float32)


def encode_instruction_last(text: str, backbone, tokenizer) -> mx.array:
    """Encode instruction → last hidden state [1, 1536]. For backward compat."""
    hidden = encode_instruction_full(text, backbone, tokenizer)
    return hidden[:, -1, :]


# ── Buffer decoder ────────────────────────────────────────────────────────

class BufferHead(nn.Module):
    """Per-byte classification from fused context. Simple and effective."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(INNER_DIM, BUF_MAX * 256)

    def __call__(self, ctx):
        """ctx: [B, INNER_DIM] → [B, BUF_MAX, 256]"""
        return self.proj(ctx).reshape(-1, BUF_MAX, 256)


# ── Action head ───────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiHeadAttention(dim, n_heads)
        self.norm = nn.RMSNorm(dim)

    def __call__(self, x):
        h = self.norm(x)
        h = self.attn(h, h, h)
        return x + h


class ReflexHead(nn.Module):
    """
    Full reflex action head:
      - Syscall classification
      - Arg classification
      - Autoregressive buffer generation (cross-attends to backbone)
    """
    def __init__(self, n_arg_vocab: int = 32):
        super().__init__()
        self.n_arg_vocab = n_arg_vocab

        # State processing
        self.state_proj = nn.Linear(STATE_DIM, INNER_DIM)
        self.temporal = TemporalAttention(INNER_DIM, n_heads=4)

        # Instruction: last-token embedding for syscall/arg heads
        self.instr_proj = nn.Linear(BACKBONE_DIM, INNER_DIM)
        self.prev_action_proj = nn.Linear(PREV_ACTION_DIM, INNER_DIM)

        # Fusion
        self.fuse = nn.Linear(INNER_DIM * 3, INNER_DIM)
        self.hidden = nn.Linear(INNER_DIM, INNER_DIM)

        # Discrete heads
        self.syscall_head = nn.Linear(INNER_DIM, N_SYSCALLS)
        self.arg_head = nn.Linear(INNER_DIM, ARG_DIM * n_arg_vocab)

        # Buffer head
        self.buf_head = BufferHead()

    def __call__(self, instr_last, states, prev_action):
        """Full forward: instruction embedding + state + prev action → all outputs."""
        h = nn.relu(self.state_proj(states))
        h = self.temporal(h)
        h_state = h[:, -1, :]
        h_instr = nn.relu(self.instr_proj(instr_last))
        h_prev = nn.relu(self.prev_action_proj(prev_action))
        h = nn.relu(self.fuse(mx.concatenate([h_state, h_instr, h_prev], axis=-1)))
        ctx = nn.relu(self.hidden(h))

        syscall_logits = self.syscall_head(ctx)
        arg_logits = self.arg_head(ctx).reshape(-1, ARG_DIM, self.n_arg_vocab)
        buf_logits = self.buf_head(ctx)

        return syscall_logits, arg_logits, buf_logits


# ── Syscall encoding/decoding ─────────────────────────────────────────────

def encode_syscall_target(raw_bytes: bytes) -> tuple[int, np.ndarray, np.ndarray]:
    nr = struct.unpack_from("<H", raw_bytes, 0)[0]
    buf_len = struct.unpack_from("<H", raw_bytes, 2)[0]
    args = list(struct.unpack_from("<6q", raw_bytes, 4))
    buffer_data = raw_bytes[52:52 + buf_len] if buf_len > 0 else b""

    syscall_name = None
    for name, num in SYS.items():
        if num == nr:
            syscall_name = name
            break
    if syscall_name is None:
        raise ValueError(f"Unknown syscall number: {nr}")
    syscall_idx = SYSCALL_TABLE.index(syscall_name)

    arg_indices = np.array([arg_to_idx(a) for a in args], dtype=np.int32)

    buf_target = np.zeros(BUF_MAX, dtype=np.int32)
    for i, b in enumerate(buffer_data[:BUF_MAX]):
        buf_target[i] = b

    return syscall_idx, arg_indices, buf_target


def encode_prev_action(syscall_idx: int, arg_indices: np.ndarray) -> np.ndarray:
    prev = np.zeros(PREV_ACTION_DIM, dtype=np.float32)
    if syscall_idx >= 0:
        prev[syscall_idx] = 1.0
        prev[N_SYSCALLS:N_SYSCALLS + ARG_DIM] = arg_indices / max(len(_arg_vocab), 1)
    return prev


def decode_syscall(syscall_idx: int, arg_indices: np.ndarray, buf_bytes: np.ndarray) -> bytes:
    syscall_name = SYSCALL_TABLE[syscall_idx]
    nr = SYS[syscall_name]
    args = [idx_to_arg(int(idx)) for idx in arg_indices]

    if syscall_name == "write":
        buf_len = args[2]
    elif syscall_name in ("openat", "mkdirat", "unlinkat", "renameat2"):
        buf_len = 0
        for i in range(len(buf_bytes)):
            buf_len = i + 1
            if buf_bytes[i] == 0:
                break
    elif syscall_name in ("close", "getpid"):
        buf_len = 0
    else:
        buf_len = 0
        for i in range(len(buf_bytes) - 1, -1, -1):
            if buf_bytes[i] != 0:
                buf_len = i + 1
                break

    buffer_data = bytes(buf_bytes[:buf_len].tolist())
    return pack_syscall(nr, args, buffer_data)
