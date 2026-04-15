"""
Train the reflex model on CHIP-8 execution traces.

Generates programs that draw things on screen, runs them through
the emulator, records (state + goal_display, opcode) pairs.
The model learns: given current machine state and desired screen,
what opcode brings me closer to the goal?

Usage:
    uv run train
"""

import time
import struct

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from chip8 import Chip8, PROGRAM_START
from model import ReflexModel


# ── Program generators ────────────────────────────────────────────────
#
# Each generator creates a CHIP-8 program (raw bytes) that draws
# something on screen. The model learns from these execution traces.

def prog_clear_screen() -> bytes:
    """Just clear the display."""
    return struct.pack(">H", 0x00E0)


def prog_draw_digit(digit: int, x: int = 10, y: int = 10) -> bytes:
    """Draw a hex digit (0-F) at position (x, y)."""
    ops = [
        0x6000 | (0 << 8) | (x & 0xFF),   # V0 = x
        0x6000 | (1 << 8) | (y & 0xFF),   # V1 = y
        0x6000 | (2 << 8) | (digit & 0xF), # V2 = digit
        0xF229,                             # I = sprite addr for V2
        0xD015,                             # Draw 5-row sprite at (V0, V1)
    ]
    return b"".join(struct.pack(">H", op) for op in ops)


def prog_draw_two_digits(d1: int, d2: int) -> bytes:
    """Draw two digits side by side."""
    ops = [
        0x00E0,                             # Clear
        0x6000 | (0 << 8) | 10,            # V0 = 10
        0x6000 | (1 << 8) | 10,            # V1 = 10
        0x6000 | (2 << 8) | (d1 & 0xF),   # V2 = d1
        0xF229,                             # I = sprite for V2
        0xD015,                             # Draw at (V0, V1)
        0x6000 | (0 << 8) | 18,            # V0 = 18
        0x6000 | (2 << 8) | (d2 & 0xF),   # V2 = d2
        0xF229,                             # I = sprite for V2
        0xD015,                             # Draw at (V0, V1)
    ]
    return b"".join(struct.pack(">H", op) for op in ops)


def prog_draw_line(x: int, y: int, length: int) -> bytes:
    """Draw a horizontal line using individual pixels."""
    ops = [0x00E0]  # Clear
    # Store a single-pixel sprite (0x80) at a known address
    # We'll use address 0x300 for our custom sprite
    ops.append(0xA300)  # I = 0x300
    ops.append(0x6000 | (3 << 8) | 0x80)  # V3 = 0x80
    ops.append(0xF355)  # Store V0..V3 at I (stores V3=0x80 at 0x303)
    # Actually let's just draw using the "0" font sprite which has top row = 0xF0
    ops.append(0x6000 | (0 << 8) | (x & 0xFF))  # V0 = x
    ops.append(0x6000 | (1 << 8) | (y & 0xFF))  # V1 = y
    ops.append(0x6000 | (2 << 8) | 0)            # V2 = 0 (digit "0")
    ops.append(0xF229)                             # I = font sprite for "0"
    ops.append(0xD011)                             # Draw 1-row sprite (top row of "0" = 0xF0)
    return b"".join(struct.pack(">H", op) for op in ops)


def prog_add_and_draw(a: int, b: int) -> bytes:
    """Compute a + b, display the result digit."""
    result = (a + b) & 0xF
    ops = [
        0x00E0,
        0x6000 | (0 << 8) | (a & 0xFF),   # V0 = a
        0x7000 | (0 << 8) | (b & 0xFF),   # V0 += b
        # Now draw V0 as a digit at (20, 10)
        0x6000 | (1 << 8) | 20,            # V1 = 20 (x)
        0x6000 | (2 << 8) | 10,            # V2 = 10 (y)
        0xF029,                             # I = font sprite for V0
        0xD125,                             # Draw 5-row sprite at (V1, V2)
    ]
    return b"".join(struct.pack(">H", op) for op in ops)


def generate_programs() -> list[tuple[str, bytes]]:
    """Generate diverse CHIP-8 programs for training."""
    programs = []

    # Draw each hex digit at various positions
    for digit in range(16):
        for x in [5, 15, 25, 35, 45]:
            for y in [5, 10, 15, 20]:
                programs.append((f"digit_{digit}_at_{x}_{y}",
                                 prog_draw_digit(digit, x, y)))

    # Draw pairs
    for d1 in range(8):
        for d2 in range(8):
            programs.append((f"pair_{d1}_{d2}",
                             prog_draw_two_digits(d1, d2)))

    # Addition
    for a in range(8):
        for b in range(8):
            programs.append((f"add_{a}_{b}",
                             prog_add_and_draw(a, b)))

    # Lines at different positions
    for x in range(0, 50, 10):
        for y in range(0, 25, 5):
            programs.append((f"line_{x}_{y}",
                             prog_draw_line(x, y, 8)))

    return programs


# ── Data collection ────────────────────────────────────────────────────

def collect_traces(programs: list[tuple[str, bytes]]) -> tuple:
    """Run programs, record (state + goal, opcode) at each step."""
    chip = Chip8()
    inputs, high_targets, low_targets = [], [], []

    for name, program in programs:
        # First: run the program to get the final display (the goal)
        chip.load_program(program)
        n_opcodes = len(program) // 2
        for _ in range(n_opcodes):
            if chip.pc < PROGRAM_START or chip.pc >= PROGRAM_START + len(program):
                break
            hi = chip.memory[chip.pc]
            lo = chip.memory[chip.pc + 1]
            opcode = (int(hi) << 8) | int(lo)
            chip.step(opcode)
        goal_display = chip.get_display().copy()

        # Second: replay and record each step
        chip.load_program(program)
        for step_i in range(n_opcodes):
            if chip.pc < PROGRAM_START or chip.pc >= PROGRAM_START + len(program):
                break

            state = chip.get_state()
            model_input = np.concatenate([state, goal_display])

            hi = chip.memory[chip.pc]
            lo = chip.memory[chip.pc + 1]
            opcode = (int(hi) << 8) | int(lo)

            inputs.append(model_input)
            high_targets.append(hi)
            low_targets.append(lo)

            chip.step(opcode)

    return (np.stack(inputs),
            np.array(high_targets, dtype=np.int32),
            np.array(low_targets, dtype=np.int32))


# ── Training ───────────────────────────────────────────────────────────

def train(X, HT, LT, steps=5000):
    model = ReflexModel()
    scheduler = optim.cosine_decay(1e-3, steps, end=1e-5)
    optimizer = optim.Adam(learning_rate=scheduler)

    Xm, HTm, LTm = mx.array(X), mx.array(HT), mx.array(LT)
    n = len(X)
    batch_size = min(64, n)
    perfect = 0

    def loss_fn(model, x, ht, lt):
        hi, lo = model(x)
        return (nn.losses.cross_entropy(hi, ht).mean() +
                nn.losses.cross_entropy(lo, lt).mean())

    def acc_fn(model, x, ht, lt):
        hi, lo = model(x)
        hi_ok = (mx.argmax(hi, axis=1) == ht).mean().item()
        lo_ok = (mx.argmax(lo, axis=1) == lt).mean().item()
        return min(hi_ok, lo_ok)

    for step in range(steps):
        idx = mx.array(np.random.choice(n, batch_size, replace=False))
        loss, grads = nn.value_and_grad(model, loss_fn)(model, Xm[idx], HTm[idx], LTm[idx])
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 200 == 0:
            acc = acc_fn(model, Xm, HTm, LTm)
            print(f"  step {step:4d}  loss={loss_fn(model, Xm, HTm, LTm).item():.4f}  acc={acc:.1%}")
            if acc == 1.0:
                perfect += 1
                if perfect >= 2:
                    print(f"  Converged.")
                    break
            else:
                perfect = 0

    return model


# ── Main ───────────────────────────────────────────────────────────────

B = "\033[1m"
D = "\033[2m"
N = "\033[0m"


def main():
    print(f"{B}Reflex — CHIP-8 Training{N}\n")

    print(f"{D}Generating programs...{N}")
    programs = generate_programs()
    print(f"  {len(programs)} programs")

    print(f"\n{D}Collecting execution traces...{N}")
    t0 = time.time()
    X, HT, LT = collect_traces(programs)
    print(f"  {len(X)} trace steps")
    print(f"  Input dim: {X.shape[1]} (state + goal display)")
    print(f"  Unique opcodes: {len(set(zip(HT.tolist(), LT.tolist())))}")
    print(f"  Collected in {time.time()-t0:.1f}s")

    print(f"\n{D}Training...{N}")
    t0 = time.time()
    model = train(X, HT, LT, steps=20000)
    print(f"  Trained in {time.time()-t0:.1f}s")

    weights_path = "weights.npz"
    mx.savez(weights_path, **dict(tree_flatten(model.parameters())))
    print(f"\n  Saved: {weights_path}")
    print(f"  Run: uv run demo")


if __name__ == "__main__":
    main()
