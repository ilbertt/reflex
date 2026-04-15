"""
Train the reflex model on CHIP-8 execution traces.

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

def prog_draw_digit(digit: int, x: int = 10, y: int = 10) -> bytes:
    ops = [
        0x6000 | (0 << 8) | (x & 0xFF),
        0x6000 | (1 << 8) | (y & 0xFF),
        0x6000 | (2 << 8) | (digit & 0xF),
        0xF229,
        0xD015,
    ]
    return b"".join(struct.pack(">H", op) for op in ops)


def prog_draw_two_digits(d1: int, d2: int) -> bytes:
    ops = [
        0x00E0,
        0x6000 | (0 << 8) | 10,
        0x6000 | (1 << 8) | 10,
        0x6000 | (2 << 8) | (d1 & 0xF),
        0xF229, 0xD015,
        0x6000 | (0 << 8) | 18,
        0x6000 | (2 << 8) | (d2 & 0xF),
        0xF229, 0xD015,
    ]
    return b"".join(struct.pack(">H", op) for op in ops)


def prog_add_and_draw(a: int, b: int) -> bytes:
    ops = [
        0x00E0,
        0x6000 | (0 << 8) | (a & 0xFF),
        0x7000 | (0 << 8) | (b & 0xFF),
        0x6000 | (1 << 8) | 20,
        0x6000 | (2 << 8) | 10,
        0xF029,
        0xD125,
    ]
    return b"".join(struct.pack(">H", op) for op in ops)


def generate_tasks() -> list[tuple[str, bytes]]:
    tasks = []
    for digit in range(16):
        for x in [5, 15, 25, 35, 45]:
            for y in [5, 10, 15, 20]:
                tasks.append((f"draw digit {digit:X} at position {x} {y}",
                              prog_draw_digit(digit, x, y)))
    for d1 in range(8):
        for d2 in range(8):
            tasks.append((f"draw digits {d1} and {d2}",
                          prog_draw_two_digits(d1, d2)))
    for a in range(8):
        for b in range(8):
            tasks.append((f"compute {a} plus {b} and draw result",
                          prog_add_and_draw(a, b)))
    return tasks


# ── Data collection ────────────────────────────────────────────────────

def collect_traces(tasks):
    chip = Chip8()
    inputs, high_targets, low_targets = [], [], []

    for _, program in tasks:
        # Run once to get goal display
        chip.load_program(program)
        for _ in range(len(program) // 2):
            if chip.pc < PROGRAM_START or chip.pc >= PROGRAM_START + len(program):
                break
            hi = chip.memory[chip.pc]
            lo = chip.memory[chip.pc + 1]
            chip.step((int(hi) << 8) | int(lo))
        goal_display = chip.get_display().copy()

        # Replay and record
        chip.load_program(program)
        for _ in range(len(program) // 2):
            if chip.pc < PROGRAM_START or chip.pc >= PROGRAM_START + len(program):
                break
            state = chip.get_state()
            model_input = np.concatenate([state, goal_display])

            hi = chip.memory[chip.pc]
            lo = chip.memory[chip.pc + 1]

            inputs.append(model_input)
            high_targets.append(int(hi))
            low_targets.append(int(lo))

            chip.step((int(hi) << 8) | int(lo))

    return (np.stack(inputs),
            np.array(high_targets, dtype=np.int32),
            np.array(low_targets, dtype=np.int32))


# ── Training ───────────────────────────────────────────────────────────

def train(X, HT, LT, steps=20000):
    model = ReflexModel()
    scheduler = optim.cosine_decay(3e-3, steps, end=1e-5)
    optimizer = optim.Adam(learning_rate=scheduler)

    Xm, HTm, LTm = mx.array(X), mx.array(HT), mx.array(LT)
    n = len(X)
    batch_size = min(64, n)
    perfect = 0

    def loss_fn(model, x, ht, lt):
        hi, lo = model(x)
        return (nn.losses.cross_entropy(hi, ht).mean() +
                nn.losses.cross_entropy(lo, lt).mean())

    for step in range(steps):
        idx = mx.array(np.random.choice(n, batch_size, replace=False))
        loss, grads = nn.value_and_grad(model, loss_fn)(model, Xm[idx], HTm[idx], LTm[idx])
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 500 == 0:
            hi, lo = model(Xm)
            acc = min((mx.argmax(hi, axis=1) == HTm).mean().item(),
                      (mx.argmax(lo, axis=1) == LTm).mean().item())
            print(f"  step {step:5d}  loss={loss_fn(model, Xm, HTm, LTm).item():.4f}  acc={acc:.1%}")
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
    print(f"{B}Reflex — Training{N}\n")

    print(f"{D}Generating tasks...{N}")
    tasks = generate_tasks()
    print(f"  {len(tasks)} tasks")

    print(f"\n{D}Collecting execution traces...{N}")
    t0 = time.time()
    X, HT, LT = collect_traces(tasks)
    print(f"  {len(X)} trace steps")
    print(f"  Unique opcodes: {len(set(zip(HT.tolist(), LT.tolist())))}")
    print(f"  Collected in {time.time()-t0:.1f}s")

    print(f"\n{D}Training...{N}")
    t0 = time.time()
    model = train(X, HT, LT, steps=20000)
    print(f"  Trained in {time.time()-t0:.1f}s")

    mx.savez("weights.npz", **dict(tree_flatten(model.parameters())))
    print(f"\n  Saved: weights.npz")
    print(f"  Run: uv run demo")


if __name__ == "__main__":
    main()
