"""
Demo for the autoregressive (GRU + scheduled sampling) model.

Usage:
    uv run demo-gru       # preset test cases
    uv run demo-gru -i    # interactive
"""

import sys
import time

import mlx.core as mx
import numpy as np

from .chip8 import Chip8
from .model import ReflexModelGRU, load_backbone, encode_instruction

B = "\033[1m"
D = "\033[2m"
G = "\033[32m"
Y = "\033[33m"
R = "\033[31m"
N = "\033[0m"


def render_display(display: np.ndarray, width: int = 64, indent: str = "  ") -> str:
    lines = [indent + "┌" + "─" * width + "┐"]
    for row in range(0, 32, 2):
        line = indent + "│"
        for col in range(width):
            top = display[row * width + col]
            bot = display[(row + 1) * width + col] if row + 1 < 32 else 0
            if top and bot:
                line += "█"
            elif top:
                line += "▀"
            elif bot:
                line += "▄"
            else:
                line += " "
        line += "│"
        lines.append(line)
    lines.append(indent + "└" + "─" * width + "┘")
    return "\n".join(lines)


def run_instruction(chip, model, backbone, tokenizer, instruction, max_steps=20):
    chip.reset()

    h, tid = encode_instruction(instruction, backbone, tokenizer)
    mx.eval(h)

    print(f"\n{B}━━━ \"{instruction}\" ━━━{N}\n")

    total_us = 0
    h_state = None
    prev_hi = prev_lo = None
    for step in range(max_steps):
        state = chip.get_state()

        t0 = time.perf_counter()
        hi_l, lo_l, h_state = model(
            h, mx.array(state[None]), mx.array(tid[None]),
            prev_hi, prev_lo, h_state,
        )
        mx.eval(hi_l, lo_l, h_state)
        us = (time.perf_counter() - t0) * 1e6
        total_us += us

        hi = int(mx.argmax(hi_l[0]).item())
        lo = int(mx.argmax(lo_l[0]).item())
        opcode = (hi << 8) | lo

        if opcode == 0x0000:
            print(f"  {D}step {step:2d}  STOP{N}  {D}({us:.0f}µs){N}")
            break

        print(f"  {D}step {step:2d}{N}  opcode={Y}0x{opcode:04X}{N}  {D}({us:.0f}µs){N}")
        chip.step(opcode)
        prev_hi = mx.array([hi], dtype=mx.int32)
        prev_lo = mx.array([lo], dtype=mx.int32)

    pixels = int(chip.display.sum())
    if pixels > 0:
        print(f"\n{G}Result ({pixels} pixels, {total_us/1000:.1f}ms total):{N}")
        print(render_display(chip.display))
    else:
        print(f"\n  {R}No pixels drawn{N}")


def main():
    interactive = "-i" in sys.argv or "--interactive" in sys.argv

    print(f"""
{B}╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  Reflex (GRU): autoregressive control head                     ║
║                                                                ║
║  Same cross-attention + token-ID pathway, plus a GRU that      ║
║  carries hidden state across opcodes. Trained with scheduled   ║
║  sampling to handle its own errors.                            ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝{N}
""")

    print(f"{D}Loading...{N}")
    backbone, tokenizer = load_backbone()
    model = ReflexModelGRU()
    try:
        model.load_weights(list(mx.load("weights_gru.npz").items()))
    except FileNotFoundError:
        print("No weights_gru.npz. Run: uv run train-gru")
        return

    chip = Chip8()

    if interactive:
        print(f"\n{D}Type an instruction. Type 'quit' to exit.{N}\n")
        while True:
            try:
                instruction = input(f"{B}> {N}").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not instruction or instruction == "quit":
                break
            run_instruction(chip, model, backbone, tokenizer, instruction)
            print()
    else:
        for instruction in [
            "draw a smiley",
            "draw digit 7 at position 15 10",
            "draw a circle",
            "draw digit 5 at position 30 15",
        ]:
            run_instruction(chip, model, backbone, tokenizer, instruction)

    print(f"\n{D}Done.{N}")


if __name__ == "__main__":
    main()
