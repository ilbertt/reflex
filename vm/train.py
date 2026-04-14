"""
Train the reflex model on raw machine bytes.

Collects (raw_state, optimal_action) pairs by creating load scenarios
inside a Docker container, then trains a transformer to map raw /proc
bytes to process control signals.

Usage:
    uv run train
"""

import time
import subprocess

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .model import (
    RawReflexModel, RAW_STATE_DIM, N_PID_BUCKETS,
    CONTAINER_NAME, read_raw_state, get_pids,
)


# ── Data collection ────────────────────────────────────────────────────

def collect_training_data(container: str, n_episodes: int = 100):
    """Collect (raw_state, optimal_action) pairs from load scenarios."""
    states, pid_targets, pri_targets = [], [], []

    for ep in range(n_episodes):
        has_load = ep % 2 == 0

        if has_load:
            n_workers = np.random.randint(1, 5)
            for _ in range(n_workers):
                subprocess.run(
                    ["docker", "exec", "-d", container, "sh", "-c",
                     "yes > /dev/null 2>&1"],
                    capture_output=True, timeout=2,
                )
            time.sleep(3)
        else:
            time.sleep(1)

        state = read_raw_state(container)
        pids = get_pids(container)

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

        busiest_pid = None
        if has_load and len(pids) > 1:
            r2 = subprocess.run(
                ["docker", "exec", container, "sh", "-c",
                 "ps -eo pid,comm --no-headers | grep yes | head -1 | awk '{print $1}'"],
                capture_output=True, text=True, timeout=2,
            )
            try:
                busiest_pid = int(r2.stdout.strip())
            except ValueError:
                pass

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


# ── Training ───────────────────────────────────────────────────────────

def train(S, PT, PrT, steps=2000):
    model = RawReflexModel()
    optimizer = optim.Adam(learning_rate=1e-3)

    Sm, PTm, PrTm = mx.array(S), mx.array(PT), mx.array(PrT)
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


# ── Main ───────────────────────────────────────────────────────────────

B = "\033[1m"
D = "\033[2m"
N = "\033[0m"


def boot_container():
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    subprocess.Popen(
        ["docker", "run", "--rm", "--name", CONTAINER_NAME,
         "alpine", "sh", "-c", "apk add --no-cache procps > /dev/null 2>&1 && sleep infinity"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(3)


def main():
    print(f"{B}Reflex — Training{N}\n")

    print(f"{D}Booting container...{N}")
    boot_container()

    state = read_raw_state(CONTAINER_NAME)
    non_zero = (state > 0).sum()
    print(f"{D}Raw state: {RAW_STATE_DIM} bytes, {non_zero} non-zero{N}")

    print(f"\n{D}Collecting training data...{N}")
    S, PT, PrT = collect_training_data(CONTAINER_NAME, n_episodes=100)
    print(f"{D}Collected {len(S)} episodes.{N}")

    print(f"\n{D}Training (raw bytes → control action)...{N}")
    model = train(S, PT, PrT, steps=2000)

    weights_path = "vm/weights.npz"
    mx.savez(weights_path, **dict(tree_flatten(model.parameters())))
    print(f"\n{D}Saved: {weights_path}{N}")

    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    print(f"{D}Done. Now run: uv run demo{N}")


if __name__ == "__main__":
    main()
