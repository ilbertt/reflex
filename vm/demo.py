"""
DEMO: process control from raw bytes.

Usage:
    uv run demo
"""

import time
import subprocess

import mlx.core as mx

from .model import (
    ProcessControlModel, CONTAINER_NAME,
    read_raw_state, get_pids, apply_action, boot_container,
)

B = "\033[1m"
D = "\033[2m"
G = "\033[32m"
Y = "\033[33m"
N = "\033[0m"


def main():
    print(f"""
{B}╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   Reflex: raw bytes in, raw bytes out                         ║
║                                                               ║
║   The model reads /proc as raw bytes.                         ║
║   No parsing. No metrics. No text.                            ║
║   It learned what the bytes mean and how to respond.          ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝{N}
""")

    print(f"{D}Loading model...{N}")
    model = ProcessControlModel()
    try:
        model.load_weights(list(mx.load("vm/process_weights.npz").items()))
    except FileNotFoundError:
        print("No weights found. Run: uv run train")
        return

    print(f"{D}Booting container...{N}")
    boot_container()
    time.sleep(3)

    print(f"\n{B}{'═' * 62}{N}")
    print(f"{B}  LIVE: raw bytes → neural net → machine control{N}")
    print(f"{B}{'═' * 62}{N}")
    print(f"{D}Spawning variable load...{N}\n")

    subprocess.run(
        ["docker", "exec", "-d", CONTAINER_NAME, "sh", "-c",
         "while true; do yes > /dev/null 2>&1 & sleep 3; killall yes 2>/dev/null; sleep 2; done"],
        capture_output=True,
    )
    time.sleep(1)

    prev_action = None
    try:
        for cycle in range(60):
            t0 = time.perf_counter()

            state = read_raw_state(CONTAINER_NAME)
            state_mx = mx.array(state[None])
            pid_logits, pri_logits = model(state_mx)
            mx.eval(pid_logits, pri_logits)

            pid_bucket = int(mx.argmax(pid_logits[0]).item())
            pri_bucket = int(mx.argmax(pri_logits[0]).item())
            us = (time.perf_counter() - t0) * 1e6

            pids = get_pids(CONTAINER_NAME)
            action = (pid_bucket, pri_bucket)
            changed = ""
            if action != prev_action:
                pid, nice = apply_action(CONTAINER_NAME, pid_bucket, pri_bucket, pids)
                prev_action = action
                if pid:
                    changed = f"  {Y}→ renice pid={pid} to {nice:+d}{N}"

            raw_preview = ''.join(f'{int(b*255):02x}' for b in state[:8])
            print(f"  {D}[{cycle:3d}]{N}  "
                  f"raw=[{raw_preview}...]  "
                  f"{D}({us:.0f}µs){N}{changed}")

            time.sleep(0.5)

    except KeyboardInterrupt:
        pass

    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    print(f"\n{D}Done.{N}")


if __name__ == "__main__":
    main()
