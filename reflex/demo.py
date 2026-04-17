"""
Demo: type an instruction, the model emits CHIP-8 opcodes to execute it.
No pre-loaded programs — every opcode comes from the model's understanding.

Usage:
    uv run demo       # preset test cases
    uv run demo -i    # interactive: type anything
"""

import sys
import time

import torch
import numpy as np

from .chip8 import Chip8
from .model import ReflexModel, encode_instruction, load_backbone

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


def run_instruction(
    chip, model, backbone, tokenizer, instruction, max_steps=20, device="cuda"
):
    chip.reset()

    h, tid = encode_instruction(instruction, backbone, tokenizer, device)

    print(f'\n{B}━━━ "{instruction}" ━━━{N}\n')

    total_us = 0
    h_state = None
    prev_hi = prev_lo = None

    h_tensor = (
        h.to(device)
        if hasattr(h, "to")
        else torch.tensor(h[None], dtype=torch.float32, device=device)
    )
    tid_tensor = torch.tensor(tid[None], dtype=torch.int32, device=device)

    for step in range(max_steps):
        state = chip.get_state()
        state_tensor = torch.tensor(state[None], dtype=torch.float32, device=device)

        t0 = time.perf_counter()
        with torch.no_grad():
            hi_l, lo_l, h_state = model(
                h_tensor,
                state_tensor,
                tid_tensor,
                prev_hi,
                prev_lo,
                h_state,
            )
        us = (time.perf_counter() - t0) * 1e6
        total_us += us

        hi = int(torch.argmax(hi_l[0]).item())
        lo = int(torch.argmax(lo_l[0]).item())
        opcode = (hi << 8) | lo

        if opcode == 0x0000:
            print(f"  {D}step {step:2d}  STOP{N}  {D}({us:.0f}µs){N}")
            break

        print(
            f"  {D}step {step:2d}{N}  opcode={Y}0x{opcode:04X}{N}  {D}({us:.0f}µs){N}"
        )
        chip.step(opcode)
        prev_hi = torch.tensor([hi], dtype=torch.int32, device=device)
        prev_lo = torch.tensor([lo], dtype=torch.int32, device=device)

    pixels = int(chip.display.sum())
    if pixels > 0:
        print(f"\n{G}Result ({pixels} pixels, {total_us / 1000:.1f}ms total):{N}")
        print(render_display(chip.display))
    else:
        print(f"\n  {R}No pixels drawn{N}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    backbone, tokenizer = load_backbone(device)
    model = ReflexModel().to(device)

    try:
        model.load_state_dict(torch.load("weights.pth", map_location=device))
    except FileNotFoundError:
        print("No weights.pth. Run: uv run train")
        return

    chip = Chip8()

    if interactive:
        print(
            f"\n{D}Type an instruction. The model generates opcodes from understanding alone.{N}"
        )
        print(f"{D}Works best with these patterns:{N}")
        print(
            f"{D}  Sprites:   draw a smiley    |  draw a heart    |  draw a circle{N}"
        )
        print(
            f"{D}             draw a star      |  draw a cross    |  draw a diamond{N}"
        )
        print(f"{D}             smiley           |  circle          |  a heart{N}")
        print(
            f"{D}  Digits:    draw digit 7     |  draw digit A    |  digit 3 at 20 20{N}"
        )
        print(f"{D}             draw digit 5 at position 30 15{N}")
        print(f"{D}             draw digits A and B{N}")
        print(f"{D}  Math:      3 + 5            |  add three and five{N}")
        print(f"{D}             compute 3 plus 5 and draw result{N}")
        print(f"{D}Type 'quit' to exit.{N}\n")

        while True:
            try:
                instruction = input(f"{B}> {N}").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not instruction or instruction == "quit":
                break
            run_instruction(
                chip, model, backbone, tokenizer, instruction, device=device
            )
            print()
    else:
        for instruction in [
            "draw a smiley",
            "draw digit 7 at position 15 10",
            "draw a circle",
            "draw digit 5 at position 30 15",
        ]:
            run_instruction(
                chip, model, backbone, tokenizer, instruction, device=device
            )

    print(f"\n{D}Done.{N}")


if __name__ == "__main__":
    main()
