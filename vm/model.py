"""
Reflex model: transformer that reads raw machine bytes and emits control signals.

Shared between train.py and demo.py.
"""

import subprocess

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ── Raw state: fixed-size byte windows from the machine ────────────────

RAW_SOURCES = [
    ("/proc/stat", 256),
    ("/proc/meminfo", 256),
    ("/proc/loadavg", 64),
]
RAW_STATE_DIM = sum(size for _, size in RAW_SOURCES)  # 576 bytes

# Action space
N_PID_BUCKETS = 8
N_PRIORITY_BUCKETS = 5  # nice values: -10, -5, 0, 5, 10
PRIORITIES = [-10, -5, 0, 5, 10]

CONTAINER_NAME = "reflex-raw"


# ── Machine interface ──────────────────────────────────────────────────

def read_raw_state(container: str) -> np.ndarray:
    """Read raw bytes from the machine. No parsing. No interpretation."""
    state = np.zeros(RAW_STATE_DIM, dtype=np.float32)
    offset = 0

    for path, size in RAW_SOURCES:
        try:
            r = subprocess.run(
                ["docker", "exec", container, "dd", f"if={path}",
                 f"bs={size}", "count=1", "status=none"],
                capture_output=True, timeout=2,
            )
            raw = r.stdout[:size]
            for i, b in enumerate(raw):
                state[offset + i] = b / 255.0
            offset += size
        except Exception:
            offset += size

    return state


def get_pids(container: str) -> list[int]:
    """Get running process IDs from the container."""
    r = subprocess.run(
        ["docker", "exec", container, "ps", "-eo", "pid", "--no-headers"],
        capture_output=True, text=True, timeout=2,
    )
    pids = []
    for line in r.stdout.strip().split("\n"):
        try:
            pids.append(int(line.strip()))
        except ValueError:
            pass
    return pids[:N_PID_BUCKETS]


def apply_action(container: str, pid_bucket: int, priority_bucket: int, pids: list[int]):
    """Apply the model's action: renice a process."""
    if pid_bucket < len(pids):
        pid = pids[pid_bucket]
        nice = PRIORITIES[priority_bucket]
        subprocess.run(
            ["docker", "exec", container, "renice", str(nice), "-p", str(pid)],
            capture_output=True, timeout=2,
        )
        return pid, nice
    return None, None


# ── Model ──────────────────────────────────────────────────────────────

class RawReflexModel(nn.Module):
    """Transformer that attends over raw bytes and outputs control signals.

    The bytes have structure — 'cpu ' prefix, newlines, digits — and
    attention learns that structure, just like it learns structure in text.
    But these aren't tokens. They're raw machine state.
    """
    def __init__(self, dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.byte_emb = nn.Embedding(256, dim)
        self.pos_emb = nn.Embedding(RAW_STATE_DIM, dim)

        self.layers = []
        for _ in range(n_layers):
            self.layers.append((
                nn.MultiHeadAttention(dim, n_heads),
                nn.RMSNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
                nn.RMSNorm(dim),
            ))

        self.out_norm = nn.RMSNorm(dim)
        self.pid_head = nn.Linear(dim, N_PID_BUCKETS)
        self.priority_head = nn.Linear(dim, N_PRIORITY_BUCKETS)

    def __call__(self, x):
        """x: [B, RAW_STATE_DIM] floats in [0,1] — raw bytes normalized."""
        byte_ids = mx.clip((x * 255).astype(mx.int32), 0, 255)
        h = self.byte_emb(byte_ids) + self.pos_emb(mx.arange(RAW_STATE_DIM))

        for attn, norm1, ff1, act, ff2, norm2 in self.layers:
            r = norm1(h)
            h = h + attn(r, r, r)
            r = norm2(h)
            h = h + ff2(act(ff1(r)))

        h = self.out_norm(h).mean(axis=1)
        return self.pid_head(h), self.priority_head(h)
