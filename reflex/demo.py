"""
DEMO: LLM understands the instruction, control head acts on the machine.

Usage:
    uv run demo       # preset test cases
    uv run demo -i    # interactive: type your own instructions
"""

import sys
import time

import mlx.core as mx
import numpy as np

from .chip8 import Chip8, PROGRAM_START
from .model import ReflexModel, load_backbone, encode_instruction
from .train import prog_draw_digit, prog_draw_two_digits, prog_add_and_draw

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


def run_goal(chip, model, backbone_hidden, token_ids, max_steps=20):
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

    return steps_taken


def parse_instruction(instr: str):
    """Try to parse instruction into a program. Returns (program, n_expected_steps) or None."""
    instr = instr.strip().lower()

    # "draw digit X at position Y Z"
    if instr.startswith("draw digit ") and " at position " in instr:
        parts = instr.split()
        try:
            digit = int(parts[2], 16)
            x = int(parts[5])
            y = int(parts[6])
            return prog_draw_digit(digit, x, y)
        except (ValueError, IndexError):
            pass

    # "draw digits X and Y"
    if instr.startswith("draw digits ") and " and " in instr:
        parts = instr.split()
        try:
            d1 = int(parts[2], 16)
            d2 = int(parts[4], 16)
            return prog_draw_two_digits(d1, d2)
        except (ValueError, IndexError):
            pass

    # "compute X plus Y and draw result"
    if instr.startswith("compute ") and " plus " in instr:
        parts = instr.split()
        try:
            a = int(parts[1])
            b = int(parts[3])
            return prog_add_and_draw(a, b)
        except (ValueError, IndexError):
            pass

    return None


def run_preset(backbone, tokenizer, model, chip):
    test_cases = [
        ("draw digit 7 at position 15 10", prog_draw_digit(7, 15, 10)),
        ("draw digit A at position 25 15", prog_draw_digit(0xA, 25, 15)),
        ("draw digits 4 and 2", prog_draw_two_digits(4, 2)),
        ("compute 3 plus 5 and draw result", prog_add_and_draw(3, 5)),
    ]

    for instruction, program in test_cases:
        run_instruction(instruction, program, backbone, tokenizer, model, chip)


def run_instruction(instruction, program, backbone, tokenizer, model, chip):
    print(f"\n{B}━━━ \"{instruction}\" ━━━{N}")

    # Get goal by running normally
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

    # Encode instruction through backbone
    hidden, tid = encode_instruction(instruction, backbone, tokenizer)
    mx.eval(hidden)

    # Model controls the machine
    chip.load_program(program)
    print(f"\n{D}Model:{N}")
    steps = run_goal(chip, model, hidden, tid)

    result = chip.get_display()
    match = np.array_equal(result.astype(np.float32), goal)
    if match:
        print(f"\n  {G}✓ Goal reached!{N}  ({steps} opcodes)")
    else:
        print(f"\n  {R}✗ Display mismatch{N}  ({steps} opcodes)")
        print(f"\n{D}Got:{N}")
        print(render_display(result))


def run_interactive(backbone, tokenizer, model, chip):
    print(f"\n{D}Type an instruction. Examples:{N}")
    print(f"  {D}draw digit 7 at position 15 10{N}")
    print(f"  {D}draw digit A at position 25 15{N}")
    print(f"  {D}draw digits 4 and 2{N}")
    print(f"  {D}compute 3 plus 5 and draw result{N}")
    print(f"  {D}quit{N}\n")

    while True:
        try:
            instr = input(f"{B}> {N}").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not instr or instr == "quit":
            break

        program = parse_instruction(instr)
        if program is None:
            print(f"  {R}Can't parse. Try: draw digit 7 at position 15 10{N}")
            continue

        run_instruction(instr, program, backbone, tokenizer, model, chip)
        print()


def main():
    interactive = "-i" in sys.argv or "--interactive" in sys.argv

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

    if interactive:
        run_interactive(backbone, tokenizer, model, chip)
    else:
        run_preset(backbone, tokenizer, model, chip)

    print(f"\n{D}Done.{N}")


if __name__ == "__main__":
    main()
