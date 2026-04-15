"""
DEMO: LLM understands the instruction, control head acts on the machine.

Usage:
    uv run demo
"""

import time

import mlx.core as mx
import numpy as np

from chip8 import Chip8, DISPLAY_SIZE, PROGRAM_START
from model import ReflexModel, load_backbone, encode_instruction
from train import prog_draw_digit, prog_draw_two_digits, prog_add_and_draw

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


def run_goal(chip, model, backbone_hidden, token_ids, goal_display, max_steps=20):
    steps_taken = 0
    for step in range(max_steps):
        state = chip.get_state()

        t0 = time.perf_counter()
        hi_logits, lo_logits = model(
            backbone_hidden, mx.array(state[None]), mx.array(token_ids[None]))
        mx.eval(hi_logits, lo_logits)
        us = (time.perf_counter() - t0) * 1e6

        hi = int(mx.argmax(hi_logits[0]).item())
        lo = int(mx.argmax(lo_logits[0]).item())
        opcode = (hi << 8) | lo

        print(f"  {D}step {step:2d}{N}  opcode={Y}0x{opcode:04X}{N}  {D}({us:.0f}µs){N}")

        if opcode == 0x0000:
            break
        chip.step(opcode)
        steps_taken += 1

        if np.array_equal(chip.display.astype(np.float32), goal_display):
            return steps_taken

    return steps_taken


def main():
    print(f"""
{B}╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  Reflex: the LLM understands, the control head acts            ║
║                                                                ║
║  Frozen Qwen2.5-Coder-1.5B encodes the instruction.           ║
║  Flipped cross-attention: instruction queries machine state.   ║
║  Zero tokens generated.                                        ║
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

    test_cases = [
        ("draw digit 7 at position 15 10", prog_draw_digit(7, 15, 10)),
        ("draw digit A at position 25 15", prog_draw_digit(0xA, 25, 15)),
        ("draw digits 4 and 2", prog_draw_two_digits(4, 2)),
        ("compute 3 plus 5 and draw result", prog_add_and_draw(3, 5)),
    ]

    for instruction, program in test_cases:
        print(f"\n{B}━━━ \"{instruction}\" ━━━{N}")

        # Goal
        chip.load_program(program)
        for _ in range(len(program) // 2):
            if chip.pc < PROGRAM_START or chip.pc >= PROGRAM_START + len(program):
                break
            hi = chip.memory[chip.pc]
            lo = chip.memory[chip.pc + 1]
            chip.step((int(hi) << 8) | int(lo))
        goal = chip.get_display().copy()
        print(f"{D}Goal:{N}")
        print(render_display(goal))

        # Encode instruction
        hidden, tid = encode_instruction(instruction, backbone, tokenizer)
        mx.eval(hidden)

        # Model controls
        chip.load_program(program)
        print(f"\n{D}Model:{N}")
        steps = run_goal(chip, model, hidden, tid, goal)

        result = chip.get_display()
        match = np.array_equal(result.astype(np.float32), goal)
        if match:
            print(f"\n  {G}✓ Goal reached!{N}  ({steps} opcodes)")
        else:
            print(f"\n  {R}✗ Display mismatch{N}  ({steps} opcodes)")
            print(f"\n{D}Got:{N}")
            print(render_display(result))

    print(f"\n{D}Done.{N}")


if __name__ == "__main__":
    main()
