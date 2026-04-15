"""
DEMO: the model controls a CHIP-8 machine from raw bytes.

Given a goal (target display), the model reads the current machine state
and emits opcodes to reach the goal. No text. No instruction set manual.
It learned what opcodes do by watching the machine.

Usage:
    uv run demo
"""

import time

import mlx.core as mx
import numpy as np

from chip8 import Chip8
from model import ReflexModel

B = "\033[1m"
D = "\033[2m"
G = "\033[32m"
Y = "\033[33m"
R = "\033[31m"
N = "\033[0m"


def render_display(display: np.ndarray, width: int = 64, indent: str = "  ") -> str:
    """Render display using block characters (2 rows per line). Always shows full display."""
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


def run_goal(chip: Chip8, model: ReflexModel, goal_display: np.ndarray,
             max_steps: int = 20) -> int:
    """Let the model control the machine toward the goal display."""
    steps_taken = 0

    for step in range(max_steps):
        state = chip.get_state()
        model_input = np.concatenate([state, goal_display])

        t0 = time.perf_counter()
        hi_logits, lo_logits = model(mx.array(model_input[None]))
        mx.eval(hi_logits, lo_logits)
        us = (time.perf_counter() - t0) * 1e6

        hi = int(mx.argmax(hi_logits[0]).item())
        lo = int(mx.argmax(lo_logits[0]).item())
        opcode = (int(hi) << 8) | int(lo)

        opcode_str = f"0x{opcode:04X}"
        print(f"  {D}step {step:2d}{N}  opcode={Y}{opcode_str}{N}  {D}({us:.0f}µs){N}")

        if opcode == 0x0000:
            break  # NOP — program ended

        chip.step(opcode)
        steps_taken += 1

        # Check if display matches goal
        if np.array_equal(chip.display.astype(np.float32), goal_display):
            return steps_taken

    return steps_taken


def main():
    print(f"""
{B}╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  Reflex: a neural net controls a CHIP-8 from raw bytes         ║
║                                                                ║
║  The model reads raw machine state + goal display as bytes.     ║
║  It emits 2-byte opcodes. No instruction manual. No text.      ║
║  It learned what opcodes do by watching the machine.           ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝{N}
""")

    print(f"{D}Loading model...{N}")
    model = ReflexModel()
    try:
        model.load_weights(list(mx.load("weights.npz").items()))
    except FileNotFoundError:
        print("No weights found. Run: uv run train")
        return

    chip = Chip8()

    # Generate some goals by running known programs
    from chip8 import PROGRAM_START
    from train import prog_draw_digit, prog_draw_two_digits, prog_add_and_draw

    test_cases = [
        ("Draw digit 7 at (15, 10)", prog_draw_digit(7, 15, 10)),
        ("Draw digit A at (25, 15)", prog_draw_digit(0xA, 25, 15)),
        ("Draw pair: 4 2", prog_draw_two_digits(4, 2)),
        ("Compute 3 + 5, draw result", prog_add_and_draw(3, 5)),
    ]

    for title, program in test_cases:
        print(f"\n{B}━━━ {title} ━━━{N}")

        # Get goal by running the program normally
        chip.load_program(program)
        n_ops = len(program) // 2
        for _ in range(n_ops):
            if chip.pc < PROGRAM_START or chip.pc >= PROGRAM_START + len(program):
                break
            hi = chip.memory[chip.pc]
            lo = chip.memory[chip.pc + 1]
            chip.step((int(hi) << 8) | int(lo))
        goal = chip.get_display().copy()

        print(f"{D}Goal:{N}")
        print(render_display(goal))

        # Now let the model try to reach this goal from scratch
        chip.load_program(program)  # reset state (program in memory but display clear)
        print(f"\n{D}Model controlling the machine:{N}")
        steps = run_goal(chip, model, goal)

        # Show result
        result = chip.get_display()
        match = np.array_equal(result.astype(np.float32), goal)
        print(f"\n  {'%s✓ Goal reached!' % G if match else '%s✗ Display mismatch' % R}{N}"
              f"  ({steps} opcodes)")

        if not match:
            print(f"\n{D}Got:{N}")
            print(render_display(result))

    print(f"\n{D}Done.{N}")


if __name__ == "__main__":
    main()
