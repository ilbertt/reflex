"""
DEMO: real-time threat detection from raw bytes.

Usage:
    uv run security
"""

import time
import subprocess
import threading

import mlx.core as mx
import numpy as np

from .model import (
    SecurityModel, CONTAINER_NAME,
    read_security_state, get_attacker_pid, kill_process, boot_container,
)

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
║   Reflex Security: real-time threat detection from raw bytes  ║
║                                                               ║
║   The model reads /proc as raw bytes and detects attacks      ║
║   that text agents are too slow to catch.                     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝{N}
""")

    print(f"{D}Loading model...{N}")
    model = SecurityModel()
    try:
        model.load_weights(list(mx.load("vm/security_weights.npz").items()))
    except FileNotFoundError:
        print("No weights found. Run: uv run train")
        return

    print(f"{D}Booting container...{N}")
    boot_container()
    time.sleep(3)

    print(f"\n{B}{'═' * 62}{N}")
    print(f"{B}  LIVE: reflex security monitor{N}")
    print(f"{B}{'═' * 62}{N}")

    attacks_launched = 0
    attacks_caught = 0
    attack_active = threading.Event()
    stop = threading.Event()

    def attacker():
        nonlocal attacks_launched
        while not stop.is_set():
            time.sleep(np.random.uniform(2, 5))
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
            logits = model(mx.array(state[None]))
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

    print(f"\n{B}{'═' * 62}{N}")
    print(f"{B}  RESULTS{N}")
    print(f"{B}{'═' * 62}{N}")
    print(f"\n  Attacks launched:   {attacks_launched}")
    print(f"  Attacks caught:     {attacks_caught}")
    if attacks_launched > 0:
        print(f"  Detection rate:     {attacks_caught/attacks_launched*100:.0f}%")
    print(f"\n  {D}A text agent polling every 3s would miss most of these.{N}")

    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    print(f"\n{D}Done.{N}")


if __name__ == "__main__":
    main()
