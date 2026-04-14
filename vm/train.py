"""
Train reflex models on raw machine bytes.

Trains both:
  1. Process control model (raw /proc bytes → renice decisions)
  2. Security model (raw /proc bytes → threat detection)

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
    ProcessControlModel, SecurityModel, CONTAINER_NAME,
    read_raw_state, read_security_state, get_pids, get_attacker_pid,
    kill_process, boot_container,
)

B = "\033[1m"
D = "\033[2m"
N = "\033[0m"


# ── Process control data collection ───────────────────────────────────

def collect_process_data(container: str, n_episodes: int = 100):
    states, pid_targets, pri_targets = [], [], []

    for ep in range(n_episodes):
        has_load = ep % 2 == 0

        if has_load:
            for _ in range(np.random.randint(1, 5)):
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
            pri_target = 4
        elif busiest_pid and running_procs > 3 and busiest_pid in pids:
            pid_target = pids.index(busiest_pid)
            pri_target = 3
        else:
            pid_target = 0
            pri_target = 2

        states.append(state)
        pid_targets.append(pid_target)
        pri_targets.append(pri_target)

        if has_load:
            subprocess.run(
                ["docker", "exec", container, "sh", "-c", "killall yes 2>/dev/null"],
                capture_output=True, timeout=2,
            )
            time.sleep(0.3)

        if (ep + 1) % 20 == 0:
            n_throttle = sum(1 for p in pri_targets[-20:] if p > 2)
            print(f"    Episode {ep+1}/{n_episodes}  load={load_1m:.1f}  "
                  f"running={running_procs}  throttle={n_throttle}/20")

    return (np.stack(states),
            np.array(pid_targets, dtype=np.int32),
            np.array(pri_targets, dtype=np.int32))


# ── Security data collection ──────────────────────────────────────────

def collect_security_data(container: str, n_episodes: int = 100):
    states, labels = [], []

    for ep in range(n_episodes):
        is_attack = ep % 2 == 0

        if is_attack:
            subprocess.run(
                ["docker", "exec", "-d", container, "sh", "-c",
                 "exec 3</etc/passwd; sleep 10; exec 3>&-"],
                capture_output=True, timeout=2,
            )
            time.sleep(1)

        state = read_security_state(container)
        states.append(state)
        labels.append(1 if is_attack else 0)

        if is_attack:
            pid = get_attacker_pid(container)
            if pid:
                kill_process(container, pid)
            time.sleep(0.3)
        else:
            time.sleep(0.5)

        if (ep + 1) % 20 == 0:
            n_threats = sum(labels[-20:])
            print(f"    Episode {ep+1}/{n_episodes}  threats={n_threats}/20")

    return np.stack(states), np.array(labels, dtype=np.int32)


# ── Generic training loop ─────────────────────────────────────────────

def train_model(model, loss_fn, acc_fn, data, steps=2000):
    optimizer = optim.Adam(learning_rate=1e-3)
    arrays = [mx.array(d) for d in data]
    n_samples = len(data[0])
    perfect_count = 0

    for step in range(steps):
        idx = mx.array(np.random.choice(n_samples, min(64, n_samples), replace=False))
        batch = [a[idx] for a in arrays]

        loss, grads = nn.value_and_grad(model, loss_fn)(model, *batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 200 == 0:
            full_loss = loss_fn(model, *arrays)
            acc = acc_fn(model, *arrays)
            print(f"    step {step:4d}  loss={full_loss.item():.4f}  acc={acc:.1%}")

            if acc == 1.0:
                perfect_count += 1
                if perfect_count >= 2:
                    print(f"    Converged at step {step}.")
                    break
            else:
                perfect_count = 0

    return model


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print(f"{B}Reflex — Training{N}\n")

    print(f"{D}Booting container...{N}")
    boot_container()
    time.sleep(3)

    # 1. Process control
    print(f"\n{B}[1/2] Process control model{N}")
    print(f"{D}  Collecting data...{N}")
    S, PT, PrT = collect_process_data(CONTAINER_NAME, n_episodes=100)

    print(f"{D}  Training...{N}")
    pc_model = ProcessControlModel()

    def pc_loss(model, s, pt, prt):
        pl, prl = model(s)
        return (nn.losses.cross_entropy(pl, pt).mean() +
                nn.losses.cross_entropy(prl, prt).mean())

    def pc_acc(model, s, pt, prt):
        pl, prl = model(s)
        pid_ok = (mx.argmax(pl, axis=1) == pt).mean().item()
        pri_ok = (mx.argmax(prl, axis=1) == prt).mean().item()
        return min(pid_ok, pri_ok)

    train_model(pc_model, pc_loss, pc_acc, [S, PT, PrT])
    mx.savez("vm/process_weights.npz", **dict(tree_flatten(pc_model.parameters())))
    print(f"  Saved: vm/process_weights.npz")

    # 2. Security
    print(f"\n{B}[2/2] Security model{N}")
    print(f"{D}  Collecting data...{N}")
    SS, SL = collect_security_data(CONTAINER_NAME, n_episodes=100)

    print(f"{D}  Training...{N}")
    sec_model = SecurityModel()

    def sec_loss(model, s, l):
        return nn.losses.cross_entropy(model(s), l).mean()

    def sec_acc(model, s, l):
        return (mx.argmax(model(s), axis=1) == l).mean().item()

    train_model(sec_model, sec_loss, sec_acc, [SS, SL])
    mx.savez("vm/security_weights.npz", **dict(tree_flatten(sec_model.parameters())))
    print(f"  Saved: vm/security_weights.npz")

    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    print(f"\n{D}Done. Run: uv run demo / uv run security{N}")


if __name__ == "__main__":
    main()
