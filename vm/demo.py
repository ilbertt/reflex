"""
DEMO: a neural net reads raw machine bytes and controls processes.

No text. No parsing. Raw bytes in, control signals out.
The model learned what /proc bytes mean by itself.

Usage:
    uv run demo
"""

import time
import subprocess

import mlx.core as mx
import numpy as np

from .model import (
    RawReflexModel, RAW_STATE_DIM,
    CONTAINER_NAME, PRIORITIES,
    read_raw_state, get_pids, apply_action,
)

B = "\033[1m"
D = "\033[2m"
G = "\033[32m"
Y = "\033[33m"
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
    weights_path = "vm/weights.npz"

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
    model = RawReflexModel()
    try:
        model.load_weights(list(mx.load(weights_path).items()))
    except FileNotFoundError:
        print(f"No weights found. Run: uv run train")
        return

    print(f"{D}Booting container...{N}")
    boot_container()

    state = read_raw_state(CONTAINER_NAME)
    non_zero = (state > 0).sum()
    print(f"{D}Raw state: {RAW_STATE_DIM} bytes, {non_zero} non-zero{N}")

    print(f"\n{B}{'═' * 62}{N}")
    print(f"{B}  LIVE: raw bytes → neural net → machine control{N}")
    print(f"{B}{'═' * 62}{N}")
    print(f"{D}Spawning variable load...{N}\n")

    # Spawn load that cycles on and off
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
