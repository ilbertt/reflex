"""
Reflex model: the action head that converts understanding into syscall signals.

Architecture:
  - Frozen Qwen backbone → last hidden state (1536-dim)
  - Syscall head: classification (which syscall)
  - Arg head: classification (exact arg values from vocabulary)
  - Buffer head: per-byte classification + copy pointer
    Each buffer position gets logits from BOTH:
    1. Generate distribution (256 classes)
    2. Copy distribution (pointer into instruction bytes)
    These are mixed via a learned gate. The model learns to copy
    filenames from the instruction and generate syntax/code.
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
INSTR_BYTES_MAX = 128
CONTEXT_LEN = 4
BACKBONE_DIM = 1536
INNER_DIM = 512
PREV_ACTION_DIM = N_SYSCALLS + ARG_DIM

MODEL_ID = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16"

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
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])
    hidden = backbone.model(input_ids)
    return hidden.astype(mx.float32)

def encode_instruction_last(text: str, backbone, tokenizer) -> mx.array:
    hidden = encode_instruction_full(text, backbone, tokenizer)
    return hidden[:, -1, :]

def instruction_to_bytes(text: str) -> np.ndarray:
    raw = text.encode("utf-8")[:INSTR_BYTES_MAX]
    result = np.zeros(INSTR_BYTES_MAX, dtype=np.int32)
    for i, b in enumerate(raw):
        result[i] = b
    return result


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
    Action head with copy mechanism for buffer bytes.

    Outputs:
      - syscall_logits: [B, N_SYSCALLS]
      - arg_logits: [B, ARG_DIM, n_arg_vocab]
      - gen_logits: [B, BUF_MAX, 256] — generate distribution
      - copy_logits: [B, BUF_MAX, INSTR_BYTES_MAX] — copy pointer distribution
      - gate: [B, BUF_MAX, 1] — copy probability (sigmoid)
    """
    def __init__(self, n_arg_vocab: int = 32):
        super().__init__()
        self.n_arg_vocab = n_arg_vocab

        self.state_proj = nn.Linear(STATE_DIM, INNER_DIM)
        self.temporal = TemporalAttention(INNER_DIM, n_heads=4)
        self.instr_proj = nn.Linear(BACKBONE_DIM, INNER_DIM)
        self.prev_action_proj = nn.Linear(PREV_ACTION_DIM, INNER_DIM)

        self.fuse = nn.Linear(INNER_DIM * 3, INNER_DIM)
        self.hidden = nn.Linear(INNER_DIM, INNER_DIM)

        self.syscall_head = nn.Linear(INNER_DIM, N_SYSCALLS)
        self.arg_head = nn.Linear(INNER_DIM, ARG_DIM * n_arg_vocab)

        # Buffer: generate + copy + gate
        self.buf_gen = nn.Linear(INNER_DIM, BUF_MAX * 256)
        self.buf_copy = nn.Linear(INNER_DIM, BUF_MAX * INSTR_BYTES_MAX)
        self.buf_gate = nn.Linear(INNER_DIM, BUF_MAX)

    def __call__(self, instr_last, states, prev_action):
        h = nn.relu(self.state_proj(states))
        h = self.temporal(h)
        h_state = h[:, -1, :]
        h_instr = nn.relu(self.instr_proj(instr_last))
        h_prev = nn.relu(self.prev_action_proj(prev_action))
        h = nn.relu(self.fuse(mx.concatenate([h_state, h_instr, h_prev], axis=-1)))
        ctx = nn.relu(self.hidden(h))

        syscall_logits = self.syscall_head(ctx)
        arg_logits = self.arg_head(ctx).reshape(-1, ARG_DIM, self.n_arg_vocab)

        gen_logits = self.buf_gen(ctx).reshape(-1, BUF_MAX, 256)
        copy_logits = self.buf_copy(ctx).reshape(-1, BUF_MAX, INSTR_BYTES_MAX)
        gate = self.buf_gate(ctx).reshape(-1, BUF_MAX, 1)

        return syscall_logits, arg_logits, gen_logits, copy_logits, gate


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


def make_copy_targets(buf_target: np.ndarray, instr_bytes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Determine which buffer positions should copy from instruction."""
    copy_gate = np.zeros(BUF_MAX, dtype=np.float32)
    copy_source = np.zeros(BUF_MAX, dtype=np.int32)

    instr_len = 0
    for i in range(INSTR_BYTES_MAX):
        if instr_bytes[i] == 0:
            instr_len = i
            break
    else:
        instr_len = INSTR_BYTES_MAX

    if instr_len == 0:
        return copy_gate, copy_source

    buf_len = 0
    for i in range(BUF_MAX):
        if buf_target[i] == 0 and (i == 0 or buf_target[i-1] == 0):
            break
        buf_len = i + 1

    i = 0
    while i < buf_len:
        best_start = -1
        best_len = 0
        for j in range(instr_len):
            match_len = 0
            while (i + match_len < buf_len and
                   j + match_len < instr_len and
                   buf_target[i + match_len] == instr_bytes[j + match_len]):
                match_len += 1
            if match_len > best_len:
                best_len = match_len
                best_start = j
        if best_len >= 2:
            for k in range(best_len):
                copy_gate[i + k] = 1.0
                copy_source[i + k] = best_start + k
            i += best_len
        else:
            i += 1

    return copy_gate, copy_source


def encode_prev_action(syscall_idx: int, arg_indices: np.ndarray) -> np.ndarray:
    prev = np.zeros(PREV_ACTION_DIM, dtype=np.float32)
    if syscall_idx >= 0:
        prev[syscall_idx] = 1.0
        prev[N_SYSCALLS:N_SYSCALLS + ARG_DIM] = arg_indices / max(len(_arg_vocab), 1)
    return prev


def decode_buffer(gen_logits, copy_logits, gate, instr_bytes: np.ndarray) -> np.ndarray:
    """Decode buffer using mixed generate/copy."""
    gate_np = np.array(mx.sigmoid(gate[:, 0]))
    gen_np = np.array(gen_logits)
    copy_np = np.array(copy_logits)

    result = np.zeros(BUF_MAX, dtype=np.uint8)
    for i in range(BUF_MAX):
        if gate_np[i] > 0.5:
            src_idx = int(np.argmax(copy_np[i]))
            if src_idx < len(instr_bytes):
                result[i] = instr_bytes[src_idx]
        else:
            result[i] = int(np.argmax(gen_np[i]))

    return result


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
