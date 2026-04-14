"""
Reflex controller: neural net keeps a web server healthy.

Reads metrics at high frequency, emits config adjustments.
No text generation, no reasoning — pure perception → action loop.

The model input is a state vector (5 floats).
The model output is an action vector (3 values: workers, connections, rate_limit).

Usage:
    uv run python -m vm.server.controller
"""

import time
import json
import urllib.request
import subprocess
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

DIR = os.path.dirname(os.path.abspath(__file__))

# State: 5 metrics normalized to [0, 1]
STATE_DIM = 5
# Action: 3 discrete choices
WORKER_OPTIONS = [1, 2, 4, 8]
CONN_OPTIONS = [64, 128, 256, 512, 1024]
RATE_OPTIONS = [50, 100, 200, 500, 1000]


class ReflexController(nn.Module):
    """Tiny controller: 5 floats in, 3 classifications out."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.worker_head = nn.Linear(64, len(WORKER_OPTIONS))
        self.conn_head = nn.Linear(64, len(CONN_OPTIONS))
        self.rate_head = nn.Linear(64, len(RATE_OPTIONS))

    def __call__(self, x):
        h = self.net(x)
        return self.worker_head(h), self.conn_head(h), self.rate_head(h)


def read_metrics(host="127.0.0.1", port=9100) -> np.ndarray:
    """Read server metrics and return as normalized state vector."""
    try:
        resp = urllib.request.urlopen(f"http://{host}:{port}/", timeout=1)
        data = json.loads(resp.read())
        state = np.array([
            min(data["request_rate"] / 1000.0, 1.0),    # normalize to [0,1]
            min(data["p99_latency"] / 0.5, 1.0),        # 500ms = 1.0
            min(data["error_rate"], 1.0),
            min(data["active_connections"] / 500.0, 1.0),
            data["worker_count"] / 8.0,
        ], dtype=np.float32)
        return state, data
    except Exception as e:
        return np.zeros(STATE_DIM, dtype=np.float32), {}


def apply_action(worker_idx, conn_idx, rate_idx, container_name="reflex-server"):
    """Apply the chosen configuration to the server."""
    workers = WORKER_OPTIONS[worker_idx]
    connections = CONN_OPTIONS[conn_idx]
    rate = RATE_OPTIONS[rate_idx]

    subprocess.run(
        ["docker", "exec", container_name, "python3", "/apply_config.py",
         f"workers={workers}", f"connections={connections}", f"rate={rate}"],
        capture_output=True, timeout=3,
    )
    return workers, connections, rate


# ── Training via simulation ───────────────────────────────────────────────

def simulate_reward(state, action_workers, action_conns, action_rate):
    """Reward function: low latency + low error rate + efficient resource use."""
    latency = state[1]      # 0 = good, 1 = bad
    error_rate = state[2]   # 0 = good, 1 = bad
    connections = state[3]  # load indicator

    # Penalty for high latency and errors
    reward = 1.0 - latency - 2.0 * error_rate

    # Small penalty for over-provisioning (using too many workers when not needed)
    if connections < 0.2 and action_workers > 1:
        reward -= 0.1 * action_workers

    return reward


def train_controller(steps=5000):
    """Train the controller using simulated scenarios."""
    model = ReflexController()
    optimizer = optim.Adam(learning_rate=1e-3)

    # Generate diverse training scenarios
    # State: [req_rate, p99_latency, error_rate, active_conns, worker_count]
    # We create (state, best_action) pairs based on heuristic optimal policy
    states = []
    targets_w, targets_c, targets_r = [], [], []

    for _ in range(10000):
        # Random state
        req_rate = np.random.uniform(0, 1)
        latency = np.random.uniform(0, 1)
        error_rate = np.random.uniform(0, 0.5)
        conns = np.random.uniform(0, 1)
        workers = np.random.choice([0.125, 0.25, 0.5, 1.0])

        state = np.array([req_rate, latency, error_rate, conns, workers], dtype=np.float32)
        states.append(state)

        # Optimal policy (heuristic — the model should learn to match or beat this)
        # High load → more workers, more connections, keep rate limit reasonable
        # Low load → fewer workers, lower connections
        # High errors → increase capacity
        load = max(req_rate, conns, error_rate)

        if load > 0.7:
            targets_w.append(3)  # 8 workers
            targets_c.append(4)  # 1024 connections
        elif load > 0.4:
            targets_w.append(2)  # 4 workers
            targets_c.append(3)  # 512 connections
        elif load > 0.2:
            targets_w.append(1)  # 2 workers
            targets_c.append(2)  # 256 connections
        else:
            targets_w.append(0)  # 1 worker
            targets_c.append(1)  # 128 connections

        # Rate limit: allow more when capacity is high, restrict when overwhelmed
        if error_rate > 0.3:
            targets_r.append(0)  # 50 r/s — protect the server
        elif latency > 0.5:
            targets_r.append(1)  # 100 r/s — ease off
        elif load > 0.5:
            targets_r.append(2)  # 200 r/s
        else:
            targets_r.append(3)  # 500 r/s — let it flow

    Sm = mx.array(np.stack(states))
    Wm = mx.array(np.array(targets_w, dtype=np.int32))
    Cm = mx.array(np.array(targets_c, dtype=np.int32))
    Rm = mx.array(np.array(targets_r, dtype=np.int32))

    def loss_fn(model, sm, wm, cm, rm):
        wl, cl, rl = model(sm)
        return (nn.losses.cross_entropy(wl, wm).mean() +
                nn.losses.cross_entropy(cl, cm).mean() +
                nn.losses.cross_entropy(rl, rm).mean())

    batch_size = 256
    n = len(states)

    for step in range(steps):
        idx = mx.array(np.random.choice(n, batch_size, replace=False))
        loss, grads = nn.value_and_grad(model, loss_fn)(
            model, Sm[idx], Wm[idx], Cm[idx], Rm[idx]
        )
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 500 == 0:
            wl, cl, rl = model(Sm[:1000])
            w_acc = (mx.argmax(wl, axis=1) == Wm[:1000]).mean().item()
            c_acc = (mx.argmax(cl, axis=1) == Cm[:1000]).mean().item()
            r_acc = (mx.argmax(rl, axis=1) == Rm[:1000]).mean().item()
            print(f"  step {step:4d}  loss={loss.item():.4f}  "
                  f"workers={w_acc:.1%}  conns={c_acc:.1%}  rate={r_acc:.1%}")

    return model


# ── Colors ─────────────────────────────────────────────────────────────────

B = "\033[1m"
D = "\033[2m"
G = "\033[32m"
Y = "\033[33m"
R = "\033[31m"
N = "\033[0m"


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"""
{B}╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   Reflex: neural net keeps a web server healthy               ║
║                                                               ║
║   No text. No reasoning. Pure perception → action loop.       ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝{N}
""")

    weights_path = os.path.join(DIR, "controller_weights.npz")

    if not os.path.exists(weights_path):
        print(f"{D}Training controller on simulated scenarios...{N}")
        model = train_controller(steps=3000)
        mx.savez(weights_path, **dict(tree_flatten(model.parameters())))
        print(f"{D}Saved: {weights_path}{N}\n")
    else:
        model = ReflexController()
        model.load_weights(list(mx.load(weights_path).items()))
        print(f"{D}Loaded controller weights.{N}\n")

    # Check server is running
    print(f"{D}Connecting to server metrics...{N}")
    state, raw = read_metrics()
    if not raw:
        print(f"{R}Cannot reach metrics at http://127.0.0.1:9100/")
        print(f"Start the server first:{N}")
        print(f"  docker build -t reflex-server -f vm/server/Dockerfile vm/server/")
        print(f"  docker run --rm -p 80:80 -p 9100:9100 --name reflex-server reflex-server")
        return

    print(f"{G}Server online.{N}")
    print(f"\n{B}Starting reflex control loop (Ctrl+C to stop){N}")
    print(f"{D}{'─' * 62}{N}")

    prev_action = None
    cycle = 0

    try:
        while True:
            t0 = time.perf_counter()

            # Perceive
            state, raw = read_metrics()
            if not raw:
                time.sleep(0.5)
                continue

            # Decide (single forward pass)
            state_mx = mx.array(state[None])
            wl, cl, rl = model(state_mx)
            mx.eval(wl, cl, rl)

            w_idx = int(mx.argmax(wl[0]).item())
            c_idx = int(mx.argmax(cl[0]).item())
            r_idx = int(mx.argmax(rl[0]).item())

            us = (time.perf_counter() - t0) * 1e6

            # Act (only if action changed)
            action = (w_idx, c_idx, r_idx)
            if action != prev_action:
                workers, conns, rate = apply_action(w_idx, c_idx, r_idx)
                prev_action = action
                changed = f"→ {Y}workers={workers} conns={conns} rate={rate}r/s{N}"
            else:
                changed = ""

            # Display
            if cycle % 5 == 0:
                latency_color = G if raw.get("p99_latency", 0) < 0.1 else (Y if raw.get("p99_latency", 0) < 0.3 else R)
                print(f"  {D}[{cycle:4d}]{N} "
                      f"req={raw.get('request_rate', 0):6.0f}/s  "
                      f"p99={latency_color}{raw.get('p99_latency', 0)*1000:5.0f}ms{N}  "
                      f"err={raw.get('error_rate', 0)*100:4.1f}%  "
                      f"conns={raw.get('active_connections', 0):3d}  "
                      f"{D}({us:.0f}µs){N}  {changed}")

            cycle += 1
            time.sleep(0.1)  # 10Hz control loop

    except KeyboardInterrupt:
        print(f"\n{D}Stopped.{N}")


if __name__ == "__main__":
    main()
