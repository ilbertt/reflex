"""
Reflex models: transformers that read raw machine bytes and emit control signals.

Shared between train.py, demo.py, and security.py.
"""

import subprocess

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ── Constants ──────────────────────────────────────────────────────────

RAW_SOURCES = [
    ("/proc/stat", 256),
    ("/proc/meminfo", 256),
    ("/proc/loadavg", 64),
]
RAW_STATE_DIM = sum(size for _, size in RAW_SOURCES)  # 576

PROC_FDS_SIZE = 256
PROC_CMDLINE_SIZE = 256
SECURITY_STATE_DIM = RAW_STATE_DIM + PROC_FDS_SIZE + PROC_CMDLINE_SIZE  # 1088

N_PID_BUCKETS = 8
N_PRIORITY_BUCKETS = 5
PRIORITIES = [-10, -5, 0, 5, 10]

CONTAINER_NAME = "reflex-raw"


# ── Machine interface ──────────────────────────────────────────────────

def read_raw_state(container: str) -> np.ndarray:
    """Read raw bytes from /proc. No parsing. No interpretation."""
    state = np.zeros(RAW_STATE_DIM, dtype=np.float32)
    offset = 0
    for path, size in RAW_SOURCES:
        try:
            r = subprocess.run(
                ["docker", "exec", container, "dd", f"if={path}",
                 f"bs={size}", "count=1", "status=none"],
                capture_output=True, timeout=2,
            )
            for i, b in enumerate(r.stdout[:size]):
                state[offset + i] = b / 255.0
        except Exception:
            pass
        offset += size
    return state


def read_security_state(container: str) -> np.ndarray:
    """Read raw bytes including process file descriptors and command lines."""
    state = np.zeros(SECURITY_STATE_DIM, dtype=np.float32)
    # Base /proc state
    base = read_raw_state(container)
    state[:RAW_STATE_DIM] = base
    offset = RAW_STATE_DIM

    # Process fd listing
    try:
        r = subprocess.run(
            ["docker", "exec", container, "sh", "-c",
             "ls -la /proc/[0-9]*/fd 2>/dev/null | head -30"],
            capture_output=True, timeout=1,
        )
        for i, b in enumerate(r.stdout[:PROC_FDS_SIZE]):
            state[offset + i] = b / 255.0
    except Exception:
        pass
    offset += PROC_FDS_SIZE

    # Process command lines
    try:
        r = subprocess.run(
            ["docker", "exec", container, "sh", "-c",
             "cat /proc/[0-9]*/cmdline 2>/dev/null | head -c 256"],
            capture_output=True, timeout=1,
        )
        for i, b in enumerate(r.stdout[:PROC_CMDLINE_SIZE]):
            state[offset + i] = b / 255.0
    except Exception:
        pass

    return state


def get_pids(container: str) -> list[int]:
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


def get_attacker_pid(container: str) -> int | None:
    r = subprocess.run(
        ["docker", "exec", container, "sh", "-c",
         "ps -eo pid,args --no-headers | grep -E 'passwd|shadow' | grep -v grep | head -1 | awk '{print $1}'"],
        capture_output=True, text=True, timeout=1,
    )
    try:
        return int(r.stdout.strip())
    except ValueError:
        return None


def apply_action(container: str, pid_bucket: int, priority_bucket: int, pids: list[int]):
    if pid_bucket < len(pids):
        pid = pids[pid_bucket]
        nice = PRIORITIES[priority_bucket]
        subprocess.run(
            ["docker", "exec", container, "renice", str(nice), "-p", str(pid)],
            capture_output=True, timeout=2,
        )
        return pid, nice
    return None, None


def kill_process(container: str, pid: int):
    try:
        subprocess.run(
            ["docker", "exec", container, "kill", "-9", str(pid)],
            capture_output=True, timeout=3,
        )
    except subprocess.TimeoutExpired:
        pass


def boot_container():
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    subprocess.Popen(
        ["docker", "run", "--rm", "--name", CONTAINER_NAME,
         "alpine", "sh", "-c", "apk add --no-cache procps > /dev/null 2>&1 && sleep infinity"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


# ── Base transformer ──────────────────────────────────────────────────

class ByteTransformer(nn.Module):
    """Transformer backbone over raw byte sequences."""
    def __init__(self, state_dim: int, dim: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.byte_emb = nn.Embedding(256, dim)
        self.pos_emb = nn.Embedding(state_dim, dim)
        self.state_dim = state_dim

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

    def __call__(self, x):
        byte_ids = mx.clip((x * 255).astype(mx.int32), 0, 255)
        h = self.byte_emb(byte_ids) + self.pos_emb(mx.arange(self.state_dim))
        for attn, norm1, ff1, act, ff2, norm2 in self.layers:
            r = norm1(h)
            h = h + attn(r, r, r)
            r = norm2(h)
            h = h + ff2(act(ff1(r)))
        return self.out_norm(h).mean(axis=1)


# ── Task-specific models ──────────────────────────────────────────────

class ProcessControlModel(nn.Module):
    """Raw bytes → process scheduling decisions."""
    def __init__(self):
        super().__init__()
        self.backbone = ByteTransformer(RAW_STATE_DIM)
        self.pid_head = nn.Linear(128, N_PID_BUCKETS)
        self.priority_head = nn.Linear(128, N_PRIORITY_BUCKETS)

    def __call__(self, x):
        h = self.backbone(x)
        return self.pid_head(h), self.priority_head(h)


class SecurityModel(nn.Module):
    """Raw bytes → threat detection."""
    def __init__(self):
        super().__init__()
        self.backbone = ByteTransformer(SECURITY_STATE_DIM)
        self.threat_head = nn.Linear(128, 2)

    def __call__(self, x):
        return self.threat_head(self.backbone(x))
