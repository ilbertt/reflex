"""
The MVP: raw bytes in, raw bytes out, tight loop.

The model reads raw machine state (bytes from /proc, fd table, etc.)
and writes raw control bytes (syscalls). No parsing. No text.
The model learns to interpret the bytes, like a vision model learns pixels.

This is the proof: a neural net can read and control a machine
through raw bytes alone.

Usage:
    uv run python -m vm.raw_loop
"""

import time
import subprocess

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# ── Raw state: fixed-size byte windows from the machine ────────────────

# We read raw bytes from these sources. The model sees ONLY bytes.
# No parsing, no normalization, no human interpretation.
RAW_SOURCES = [
    ("/proc/stat", 256),       # CPU state: raw bytes
    ("/proc/meminfo", 256),    # Memory state: raw bytes
    ("/proc/loadavg", 64),     # Load: raw bytes
]
# Total raw state size
RAW_STATE_DIM = sum(size for _, size in RAW_SOURCES)  # 576 bytes

# Action: which process to renice (pid selection + priority)
# This is the simplest meaningful control: adjust process scheduling
ACTION_DIM = 2  # [pid_bucket, priority_bucket]
N_PID_BUCKETS = 8      # model selects one of 8 process slots
N_PRIORITY_BUCKETS = 5  # nice values: -10, -5, 0, 5, 10

CONTAINER_NAME = "reflex-raw"


def read_raw_state(container: str) -> np.ndarray:
    """Read raw bytes from the machine. No parsing. No interpretation.

    The model receives these bytes as-is and must learn what they mean.
    """
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
                state[offset + i] = b / 255.0  # only normalization: bytes to [0,1]
            offset += size
        except Exception:
            offset += size

    return state


def apply_action(container: str, pid_bucket: int, priority_bucket: int, pids: list[int]):
    """Apply the model's action: renice a process."""
    priorities = [-10, -5, 0, 5, 10]
    if pid_bucket < len(pids):
        pid = pids[pid_bucket]
        nice = priorities[priority_bucket]
        subprocess.run(
            ["docker", "exec", container, "renice", str(nice), "-p", str(pid)],
            capture_output=True, timeout=2,
        )
        return pid, nice
    return None, None


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


# ── Model: raw bytes → action ─────────────────────────────────────────

class RawReflexModel(nn.Module):
    """Transformer that attends over raw bytes and outputs control signals.

    The bytes have structure — 'cpu ' prefix, newlines, digits — and
    attention learns that structure, just like it learns structure in text.
    But these aren't tokens. They're raw machine state.
    """
    def __init__(self, dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.byte_emb = nn.Embedding(256, dim)  # each byte value gets an embedding
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
        # Convert back to byte indices for embedding
        byte_ids = (x * 255).astype(mx.int32)  # [B, RAW_STATE_DIM]
        byte_ids = mx.clip(byte_ids, 0, 255)

        h = self.byte_emb(byte_ids) + self.pos_emb(mx.arange(RAW_STATE_DIM))

        for attn, norm1, ff1, act, ff2, norm2 in self.layers:
            r = norm1(h)
            h = h + attn(r, r, r)
            r = norm2(h)
            h = h + ff2(act(ff1(r)))

        # Pool: mean over all byte positions
        h = self.out_norm(h).mean(axis=1)  # [B, dim]

        return self.pid_head(h), self.priority_head(h)


# ── Training: learn from raw state → optimal action pairs ─────────────

def collect_training_data(container: str, n_episodes: int = 100):
    """Collect (raw_state, optimal_action) pairs.

    We create load scenarios and record what the optimal action would be:
    - High CPU on a process → lower its priority (nice +10)
    - Low CPU overall → keep priorities balanced (nice 0)
    - Memory pressure → lower priority of biggest consumer
    """
    states, pid_targets, pri_targets = [], [], []

    for ep in range(n_episodes):
        # Alternate between load and no-load episodes
        has_load = ep % 2 == 0

        if has_load:
            # Spawn CPU burners
            n_workers = np.random.randint(1, 5)
            for _ in range(n_workers):
                subprocess.run(
                    ["docker", "exec", "-d", container, "sh", "-c",
                     "yes > /dev/null 2>&1"],
                    capture_output=True, timeout=2,
                )
            time.sleep(3)  # /proc/stat needs time to accumulate CPU ticks
        else:
            time.sleep(1)

        # Read raw state
        state = read_raw_state(container)
        pids = get_pids(container)

        # Read loadavg to determine load level (from the raw bytes we already have)
        r = subprocess.run(
            ["docker", "exec", container, "cat", "/proc/loadavg"],
            capture_output=True, text=True, timeout=2,
        )
        running_procs = 0
        load_1m = 0.0
        if r.returncode == 0:
            parts = r.stdout.strip().split()
            try:
                load_1m = float(parts[0])
                running_procs = int(parts[3].split("/")[0])
            except (ValueError, IndexError):
                pass

        # Find busiest process (use /proc/[pid]/stat for reliable CPU info)
        busiest_pid = None
        if has_load and len(pids) > 1:
            # The yes processes are the ones we spawned — find them
            r2 = subprocess.run(
                ["docker", "exec", container, "sh", "-c",
                 "ps -eo pid,comm --no-headers | grep yes | head -1 | awk '{print $1}'"],
                capture_output=True, text=True, timeout=2,
            )
            try:
                busiest_pid = int(r2.stdout.strip())
            except ValueError:
                pass

        # Optimal policy based on running process count and load:
        # - Many running processes (>5) → system is loaded, throttle
        # - Few running (<3) → system is idle, do nothing
        if busiest_pid and running_procs > 5 and busiest_pid in pids:
            pid_target = pids.index(busiest_pid)
            pri_target = 4  # nice +10
        elif busiest_pid and running_procs > 3 and busiest_pid in pids:
            pid_target = pids.index(busiest_pid)
            pri_target = 3  # nice +5
        else:
            pid_target = 0
            pri_target = 2  # nice 0

        states.append(state)
        pid_targets.append(pid_target)
        pri_targets.append(pri_target)

        # Kill workers after recording
        if has_load:
            subprocess.run(
                ["docker", "exec", container, "sh", "-c", "killall yes 2>/dev/null"],
                capture_output=True, timeout=2,
            )
            time.sleep(0.3)

        if (ep + 1) % 10 == 0:
            n_throttle = sum(1 for p in pri_targets[-10:] if p > 2)
            print(f"  Episode {ep+1}/{n_episodes}  load={load_1m:.1f}  running={running_procs}  "
                  f"last 10: {n_throttle} throttle, {10-n_throttle} idle")

    return (np.stack(states),
            np.array(pid_targets, dtype=np.int32),
            np.array(pri_targets, dtype=np.int32))


def train_model(S, PT, PrT, steps=2000):
    model = RawReflexModel()
    optimizer = optim.Adam(learning_rate=1e-3)

    Sm = mx.array(S)
    PTm = mx.array(PT)
    PrTm = mx.array(PrT)
    n = len(S)

    def loss_fn(model, s, pt, prt):
        pid_logits, pri_logits = model(s)
        return (nn.losses.cross_entropy(pid_logits, pt).mean() +
                nn.losses.cross_entropy(pri_logits, prt).mean())

    for step in range(steps):
        idx = mx.array(np.random.choice(n, min(64, n), replace=False))
        loss, grads = nn.value_and_grad(model, loss_fn)(
            model, Sm[idx], PTm[idx], PrTm[idx]
        )
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 200 == 0:
            pl, prl = model(Sm)
            pid_acc = (mx.argmax(pl, axis=1) == PTm).mean().item()
            pri_acc = (mx.argmax(prl, axis=1) == PrTm).mean().item()
            print(f"  step {step:4d}  loss={loss.item():.4f}  "
                  f"pid_acc={pid_acc:.1%}  pri_acc={pri_acc:.1%}")

    return model


# ── Main loop ──────────────────────────────────────────────────────────

B = "\033[1m"
D = "\033[2m"
G = "\033[32m"
Y = "\033[33m"
R = "\033[31m"
N = "\033[0m"


def main():
    print(f"""
{B}╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   Reflex: raw bytes in, raw bytes out                         ║
║                                                               ║
║   The model reads /proc as raw bytes.                         ║
║   No parsing. No metrics. No text.                            ║
║   It learns what the bytes mean and how to respond.           ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝{N}
""")

    # Boot container
    print(f"{D}Booting container...{N}")
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    proc = subprocess.Popen(
        ["docker", "run", "--rm", "--name", CONTAINER_NAME,
         "alpine", "sh", "-c", "apk add --no-cache procps > /dev/null 2>&1 && sleep infinity"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(3)
    print(f"{D}Container ready.{N}")

    # Test raw state reading
    state = read_raw_state(CONTAINER_NAME)
    non_zero = (state > 0).sum()
    print(f"{D}Raw state: {RAW_STATE_DIM} bytes, {non_zero} non-zero{N}")
    print(f"{D}First 32 bytes of /proc/stat as the model sees them:{N}")
    print(f"  {D}{state[:32].tolist()}{N}")
    print(f"{D}(these are raw byte values / 255 — the model gets NO interpretation){N}")

    # Collect training data
    print(f"\n{D}Collecting training data (creating load scenarios)...{N}")
    S, PT, PrT = collect_training_data(CONTAINER_NAME, n_episodes=100)
    print(f"{D}Collected {len(S)} episodes.{N}")

    # Train
    print(f"\n{D}Training model (raw bytes → control action)...{N}")
    model = train_model(S, PT, PrT, steps=2000)

    # Live control loop
    print(f"\n{B}{'═' * 62}{N}")
    print(f"{B}  LIVE: raw bytes → neural net → machine control{N}")
    print(f"{B}{'═' * 62}{N}")
    print(f"{D}Spawning load to control...{N}\n")

    # Spawn load that varies over time
    subprocess.run(
        ["docker", "exec", "-d", CONTAINER_NAME, "sh", "-c",
         "while true; do yes > /dev/null 2>&1 & sleep 3; killall yes 2>/dev/null; sleep 2; done"],
        capture_output=True,
    )
    time.sleep(1)

    try:
        for cycle in range(60):
            t0 = time.perf_counter()

            # READ raw bytes from the machine
            state = read_raw_state(CONTAINER_NAME)

            # MODEL forward pass
            state_mx = mx.array(state[None])
            pid_logits, pri_logits = model(state_mx)
            mx.eval(pid_logits, pri_logits)

            pid_bucket = int(mx.argmax(pid_logits[0]).item())
            pri_bucket = int(mx.argmax(pri_logits[0]).item())

            us = (time.perf_counter() - t0) * 1e6

            # WRITE control signal to the machine
            pids = get_pids(CONTAINER_NAME)
            pid, nice = apply_action(CONTAINER_NAME, pid_bucket, pri_bucket, pids)

            # Show what's happening
            priorities = [-10, -5, 0, 5, 10]
            if pid:
                action_str = f"renice pid={pid} to {nice:+d}"
            else:
                action_str = "no action"

            raw_preview = ''.join(f'{int(b*255):02x}' for b in state[:8])
            print(f"  {D}[{cycle:3d}]{N}  "
                  f"raw=[{raw_preview}...]  "
                  f"→ {Y}{action_str:30s}{N}  "
                  f"{D}({us:.0f}µs){N}")

            time.sleep(0.5)

    except KeyboardInterrupt:
        pass

    # Cleanup
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    print(f"\n{D}Done.{N}")


if __name__ == "__main__":
    main()
