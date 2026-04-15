"""
DEMO: LLM understands, control head generates opcodes. No pre-loaded programs.

The model reads the instruction through the frozen backbone, cross-attends
to the machine state, and generates opcodes step by step. Every opcode
comes from the model's understanding — nothing is pre-loaded in memory.

Usage:
    uv run demo       # preset test cases
    uv run demo -i    # interactive: type anything
"""

import sys
import time

import mlx.core as mx
import numpy as np

from .chip8 import Chip8
from .model import ReflexModel, load_backbone, encode_instruction

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
    """Model generates opcodes from instruction alone. No pre-loaded program."""
    chip.reset()

    h, tid = encode_instruction(instruction, backbone, tokenizer)
    mx.eval(h)

    print(f"\n{B}━━━ \"{instruction}\" ━━━{N}\n")

    total_us = 0
    for step in range(max_steps):
        state = chip.get_state()

        t0 = time.perf_counter()
        hi_l, lo_l = model(h, mx.array(state[None]), mx.array(tid[None]))
        mx.eval(hi_l, lo_l)
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

    # Show result
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
║  Reflex: LLM understands → control head generates opcodes      ║
║                                                                ║
║  No pre-loaded programs. Every opcode comes from the model.    ║
║  The LLM backbone understands the instruction.                 ║
║  The control head generates CHIP-8 opcodes to execute it.      ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝{N}
""")

    print(f"{D}Loading...{N}")
    backbone, tokenizer = load_backbone()
    model = ReflexModel()
    try:
        model.load_weights(list(mx.load("weights.npz").items()))
    except FileNotFoundError:
        print("No weights. Run: uv run train")
        return

    chip = Chip8()

    if interactive:
        print(f"\n{D}Type an instruction. The model generates opcodes from understanding alone.{N}")
        print(f"{D}Try:{N}")
        print(f"{D}  draw a smiley              draw a circle{N}")
        print(f"{D}  draw digit 7 at position 15 10{N}")
        print(f"{D}  draw digit A at position 30 5{N}")
        print(f"{D}  draw digit 0 at position 10 20{N}")
        print(f"{D}Digits 0-F, positions 5-50 x 5-25. Type 'quit' to exit.{N}\n")

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
