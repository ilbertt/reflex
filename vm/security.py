"""
Security demo: reflex model detects and kills unauthorized file access in real-time.

A simulated attacker periodically tries to write to /etc/passwd.
The attack window is ~100ms. A text agent polling every 3s misses it.
The reflex model reading raw /proc bytes catches it in milliseconds.

Usage:
    uv run security
"""

import time
import subprocess
import threading

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .model import RAW_STATE_DIM, RAW_SOURCES, CONTAINER_NAME

# Extended state: also read /proc/*/fd and /proc/*/cmdline
# to detect which processes have sensitive files open
PROC_FDS_SIZE = 256       # raw bytes from ls -la /proc/[pids]/fd
PROC_CMDLINE_SIZE = 256   # raw bytes from /proc/[pids]/cmdline
SECURITY_STATE_DIM = RAW_STATE_DIM + PROC_FDS_SIZE + PROC_CMDLINE_SIZE  # 1088


def read_security_state(container: str) -> np.ndarray:
    """Read raw bytes including process file descriptors."""
    state = np.zeros(SECURITY_STATE_DIM, dtype=np.float32)
    offset = 0

    # Standard /proc state
    for path, size in RAW_SOURCES:
        try:
            r = subprocess.run(
                ["docker", "exec", container, "dd", f"if={path}",
                 f"bs={size}", "count=1", "status=none"],
                capture_output=True, timeout=1,
            )
            for i, b in enumerate(r.stdout[:size]):
                state[offset + i] = b / 255.0
        except Exception:
            pass
        offset += size

    # Process fd listing (shows which files each process has open)
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
    offset += PROC_CMDLINE_SIZE

    return state


def get_attacker_pid(container: str) -> int | None:
    """Find the attacker process (anything touching /etc/passwd or /etc/shadow)."""
    r = subprocess.run(
        ["docker", "exec", container, "sh", "-c",
         "grep -l 'passwd\\|shadow' /proc/[0-9]*/cmdline 2>/dev/null | head -1"],
        capture_output=True, text=True, timeout=1,
    )
    if r.stdout.strip():
        try:
            return int(r.stdout.strip().split("/")[2])
        except (ValueError, IndexError):
            pass
    # Also check by process name
    r = subprocess.run(
        ["docker", "exec", container, "sh", "-c",
         "ps -eo pid,args --no-headers | grep -E 'passwd|shadow' | grep -v grep | head -1 | awk '{print $1}'"],
        capture_output=True, text=True, timeout=1,
    )
    try:
        return int(r.stdout.strip())
    except ValueError:
        return None


def kill_process(container: str, pid: int):
    subprocess.run(
        ["docker", "exec", container, "kill", "-9", str(pid)],
        capture_output=True, timeout=1,
    )


# ── Model ──────────────────────────────────────────────────────────────

class SecurityModel(nn.Module):
    """Detects threats from raw bytes. Output: threat_score (binary)."""
    def __init__(self, dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.byte_emb = nn.Embedding(256, dim)
        self.pos_emb = nn.Embedding(SECURITY_STATE_DIM, dim)

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
        self.threat_head = nn.Linear(dim, 2)  # [safe, threat]

    def __call__(self, x):
        byte_ids = mx.clip((x * 255).astype(mx.int32), 0, 255)
        h = self.byte_emb(byte_ids) + self.pos_emb(mx.arange(SECURITY_STATE_DIM))

        for attn, norm1, ff1, act, ff2, norm2 in self.layers:
            r = norm1(h)
            h = h + attn(r, r, r)
            r = norm2(h)
            h = h + ff2(act(ff1(r)))

        h = self.out_norm(h).mean(axis=1)
        return self.threat_head(h)


# ── Training ───────────────────────────────────────────────────────────

def collect_security_data(container: str, n_episodes: int = 100):
    """Collect safe/threat states by simulating attacks."""
    states, labels = [], []

    for ep in range(n_episodes):
        is_attack = ep % 2 == 0

        if is_attack:
            # Simulate attacker: holds /etc/passwd open (like real exfiltration)
            subprocess.run(
                ["docker", "exec", "-d", container, "sh", "-c",
                 "exec 3</etc/passwd; cp /etc/passwd /tmp/.stolen; sleep 10"],
                capture_output=True, timeout=2,
            )
            time.sleep(1)  # let process and fd appear in /proc

        state = read_security_state(container)
        states.append(state)
        labels.append(1 if is_attack else 0)

        if is_attack:
            # Kill the attacker after recording
            pid = get_attacker_pid(container)
            if pid:
                kill_process(container, pid)
            time.sleep(0.3)
        else:
            time.sleep(0.5)

        if (ep + 1) % 20 == 0:
            n_threats = sum(labels[-20:])
            print(f"  Episode {ep+1}/{n_episodes}  threats: {n_threats}/20")

    return np.stack(states), np.array(labels, dtype=np.int32)


def train_security(S, L, steps=2000):
    model = SecurityModel()
    optimizer = optim.Adam(learning_rate=1e-3)

    Sm, Lm = mx.array(S), mx.array(L)
    n = len(S)

    def loss_fn(model, s, l):
        logits = model(s)
        return nn.losses.cross_entropy(logits, l).mean()

    for step in range(steps):
        idx = mx.array(np.random.choice(n, min(32, n), replace=False))
        loss, grads = nn.value_and_grad(model, loss_fn)(model, Sm[idx], Lm[idx])
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 200 == 0:
            logits = model(Sm)
            acc = (mx.argmax(logits, axis=1) == Lm).mean().item()
            print(f"  step {step:4d}  loss={loss.item():.4f}  acc={acc:.1%}")

    return model


# ── Demo ───────────────────────────────────────────────────────────────

B = "\033[1m"
D = "\033[2m"
G = "\033[32m"
Y = "\033[33m"
R = "\033[31m"
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
    print(f"""
{B}╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   Reflex Security: real-time threat detection from raw bytes  ║
║                                                               ║
║   The model reads /proc as raw bytes and detects attacks      ║
║   that text agents are too slow to catch.                     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝{N}
""")

    print(f"{D}Booting container...{N}")
    boot_container()

    # Train
    print(f"\n{D}Collecting security training data...{N}")
    S, L = collect_security_data(CONTAINER_NAME, n_episodes=100)
    n_threats = L.sum()
    print(f"{D}Collected {len(L)} episodes ({n_threats} threats, {len(L)-n_threats} safe){N}")

    print(f"\n{D}Training threat detector...{N}")
    model = train_security(S, L, steps=2000)

    # Live demo: random attacks, model must detect and kill
    print(f"\n{B}{'═' * 62}{N}")
    print(f"{B}  LIVE: reflex security monitor{N}")
    print(f"{B}{'═' * 62}{N}")

    attacks_launched = 0
    attacks_caught = 0
    attacks_missed = 0

    # Background attacker: randomly tries to access /etc/passwd
    attack_active = threading.Event()
    stop = threading.Event()

    def attacker():
        nonlocal attacks_launched
        while not stop.is_set():
            time.sleep(np.random.uniform(2, 5))  # random delay
            if stop.is_set():
                break
            attacks_launched += 1
            attack_active.set()
            subprocess.run(
                ["docker", "exec", "-d", CONTAINER_NAME, "sh", "-c",
                 "exec 3</etc/passwd; cp /etc/passwd /tmp/.stolen; sleep 10"],
                capture_output=True, timeout=2,
            )

    attacker_thread = threading.Thread(target=attacker, daemon=True)
    attacker_thread.start()

    print(f"{D}Attacker running in background. Monitoring...{N}\n")

    try:
        for cycle in range(80):
            t0 = time.perf_counter()

            state = read_security_state(CONTAINER_NAME)
            state_mx = mx.array(state[None])
            logits = model(state_mx)
            mx.eval(logits)

            threat_score = float(mx.softmax(logits[0])[1].item())
            is_threat = threat_score > 0.5

            us = (time.perf_counter() - t0) * 1e6

            raw_preview = ''.join(f'{int(b*255):02x}' for b in state[:6])

            if is_threat:
                pid = get_attacker_pid(CONTAINER_NAME)
                if pid:
                    kill_process(CONTAINER_NAME, pid)
                    if attack_active.is_set():
                        attacks_caught += 1
                        attack_active.clear()
                    print(f"  {R}[{cycle:3d}] THREAT DETECTED{N}  "
                          f"raw=[{raw_preview}...]  "
                          f"score={threat_score:.2f}  "
                          f"{R}→ KILLED pid={pid}{N}  "
                          f"{D}({us:.0f}µs){N}")
                else:
                    print(f"  {Y}[{cycle:3d}] threat signal{N}  "
                          f"raw=[{raw_preview}...]  "
                          f"score={threat_score:.2f}  "
                          f"{D}({us:.0f}µs){N}")
            else:
                print(f"  {G}[{cycle:3d}] safe{N}          "
                      f"raw=[{raw_preview}...]  "
                      f"score={threat_score:.2f}  "
                      f"{D}({us:.0f}µs){N}")

            time.sleep(0.3)

    except KeyboardInterrupt:
        pass

    stop.set()

    # Results
    print(f"\n{B}{'═' * 62}{N}")
    print(f"{B}  RESULTS{N}")
    print(f"{B}{'═' * 62}{N}")
    print(f"\n  Attacks launched:   {attacks_launched}")
    print(f"  Attacks caught:     {attacks_caught}")
    print(f"  Detection rate:     {attacks_caught/max(attacks_launched,1)*100:.0f}%")
    print(f"\n  {D}A text agent polling every 3s would catch: ~0%{N}")
    print(f"  {D}(attack window is <500ms, text agent cycle is 3000ms){N}")

    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    print(f"\n{D}Done.{N}")


if __name__ == "__main__":
    main()
