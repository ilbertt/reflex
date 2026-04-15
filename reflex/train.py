"""
Train: LLM backbone + flipped cross-attention → CHIP-8 opcodes.

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

from .chip8 import Chip8, PROGRAM_START
from .model import ReflexModel, BACKBONE_DIM, load_backbone, encode_instruction


# ── Programs ───────────────────────────────────────────────────────────

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


def prog_draw_sprite(name: str, sprite_bytes: list[int], x: int, y: int) -> bytes:
    """Draw a custom sprite. Stores sprite data at 0x300, then draws it."""
    height = len(sprite_bytes)
    ops = [0x00E0]  # clear

    # Store sprite bytes in memory at 0x300 using registers + Fx55
    # Load each byte into V0..Vn, set I=0x300, store
    for i, b in enumerate(sprite_bytes[:8]):  # max 8 rows (V0-V7)
        ops.append(0x6000 | (i << 8) | (b & 0xFF))  # Vi = byte
    ops.append(0xA300)  # I = 0x300
    ops.append(0xF055 | ((min(height, 8) - 1) << 8))  # store V0..Vn at I

    # Set draw position
    ops.append(0x6000 | (8 << 8) | (x & 0xFF))   # V8 = x
    ops.append(0x6000 | (9 << 8) | (y & 0xFF))   # V9 = y
    ops.append(0xA300)  # I = 0x300

    # Draw sprite: D89n where n = height
    ops.append(0xD890 | (min(height, 8) & 0xF))

    return b"".join(struct.pack(">H", op) for op in ops)


# Named sprites (8 pixels wide, variable height)
SPRITES = {
    "horizontal line": [0xFF],
    "vertical line": [0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80],
    "box": [0xFF, 0x81, 0x81, 0x81, 0x81, 0x81, 0x81, 0xFF],
    "cross": [0x18, 0x18, 0xFF, 0x18, 0x18, 0xFF, 0x18, 0x18],
    "diamond": [0x10, 0x28, 0x44, 0x82, 0x44, 0x28, 0x10],
    "arrow right": [0x10, 0x18, 0xFC, 0xFE, 0xFC, 0x18, 0x10],
    "arrow down": [0x10, 0x10, 0x10, 0xFE, 0x7C, 0x38, 0x10],
    "smiley": [0x3C, 0x42, 0xA5, 0x81, 0xA5, 0x99, 0x42, 0x3C],
    "heart": [0x66, 0xFF, 0xFF, 0xFF, 0x7E, 0x3C, 0x18],
    "star": [0x10, 0x38, 0xFE, 0x38, 0x6C, 0x44],
    "triangle": [0x10, 0x28, 0x28, 0x44, 0x44, 0x82, 0xFE],
    "snake": [0x00, 0x7E, 0x02, 0x3E, 0x20, 0x3F, 0x01, 0x7F],
    "zigzag": [0xC0, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x03],
    "checkerboard": [0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55],
    "circle": [0x3C, 0x42, 0x81, 0x81, 0x81, 0x81, 0x42, 0x3C],
    "x shape": [0x81, 0x42, 0x24, 0x18, 0x18, 0x24, 0x42, 0x81],
    "inverted triangle": [0xFE, 0x82, 0x44, 0x44, 0x28, 0x28, 0x10],
    "letter L": [0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xFE],
    "letter T": [0xFE, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10],
    "letter H": [0x82, 0x82, 0x82, 0xFE, 0x82, 0x82, 0x82],
}


def generate_tasks() -> list[tuple[str, bytes]]:
    tasks = []

    # Digit drawing
    for digit in range(16):
        for x in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
            for y in [5, 10, 15, 20, 25]:
                tasks.append((f"draw digit {digit:X} at position {x} {y}",
                              prog_draw_digit(digit, x, y)))

    # Two digits
    for d1 in range(16):
        for d2 in range(16):
            tasks.append((f"draw digits {d1:X} and {d2:X}",
                          prog_draw_two_digits(d1, d2)))

    # Arithmetic
    for a in range(10):
        for b in range(10):
            tasks.append((f"compute {a} plus {b} and draw result",
                          prog_add_and_draw(a, b)))

    # Custom sprites at various positions
    for name, sprite in SPRITES.items():
        for x in [5, 15, 25, 35]:
            for y in [3, 10, 18]:
                tasks.append((f"draw {name} at position {x} {y}",
                              prog_draw_sprite(name, sprite, x, y)))

    # Alternative phrasings for sprites
    for name, sprite in SPRITES.items():
        tasks.append((f"draw a {name}", prog_draw_sprite(name, sprite, 20, 10)))
        tasks.append((f"show me a {name}", prog_draw_sprite(name, sprite, 20, 10)))
        tasks.append((f"display {name}", prog_draw_sprite(name, sprite, 20, 10)))

    return tasks


# ── Data collection ────────────────────────────────────────────────────

def collect_traces(tasks, backbone, tokenizer):
    chip = Chip8()

    print("  Encoding instructions...")
    instr_cache = {}
    for instr, _ in tasks:
        if instr not in instr_cache:
            h, tid = encode_instruction(instr, backbone, tokenizer)
            mx.eval(h)
            instr_cache[instr] = (np.array(h[0]), tid)
    print(f"  {len(instr_cache)} unique instructions")

    states, hiddens, tids, high_targets, low_targets = [], [], [], [], []

    for instr, program in tasks:
        hidden, tid = instr_cache[instr]

        chip.load_program(program)
        for _ in range(len(program) // 2):
            if chip.pc < PROGRAM_START or chip.pc >= PROGRAM_START + len(program):
                break
            state = chip.get_state()
            hi = int(chip.memory[chip.pc])
            lo = int(chip.memory[chip.pc + 1])

            states.append(state)
            hiddens.append(hidden)
            tids.append(tid)
            high_targets.append(hi)
            low_targets.append(lo)

            chip.step((hi << 8) | lo)

    # Pad hiddens
    max_seq = max(h.shape[0] for h in hiddens)
    H = np.zeros((len(hiddens), max_seq, BACKBONE_DIM), dtype=np.float32)
    for i, h in enumerate(hiddens):
        H[i, :h.shape[0], :] = h

    return (np.stack(states), H, np.stack(tids),
            np.array(high_targets, dtype=np.int32),
            np.array(low_targets, dtype=np.int32))


# ── Training ───────────────────────────────────────────────────────────

def train(S, H, T, HT, LT, steps=40000):
    model = ReflexModel()
    scheduler = optim.cosine_decay(1e-3, steps, end=1e-5)
    optimizer = optim.Adam(learning_rate=scheduler)

    Sm, Hm, Tm = mx.array(S), mx.array(H), mx.array(T)
    HTm, LTm = mx.array(HT), mx.array(LT)
    n = len(S)
    batch_size = min(64, n)
    perfect = 0

    def loss_fn(model, h, s, t, ht, lt):
        hi, lo = model(h, s, t)
        return (nn.losses.cross_entropy(hi, ht).mean() +
                nn.losses.cross_entropy(lo, lt).mean())

    for step in range(steps):
        idx = mx.array(np.random.choice(n, batch_size, replace=False))
        loss, grads = nn.value_and_grad(model, loss_fn)(
            model, Hm[idx], Sm[idx], Tm[idx], HTm[idx], LTm[idx])
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 500 == 0:
            hi, lo = model(Hm, Sm, Tm)
            acc = min((mx.argmax(hi, axis=1) == HTm).mean().item(),
                      (mx.argmax(lo, axis=1) == LTm).mean().item())
            print(f"  step {step:5d}  loss={loss_fn(model, Hm, Sm, Tm, HTm, LTm).item():.4f}  acc={acc:.1%}")
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
    print(f"{D}Frozen backbone + flipped cross-attention control head{N}\n")

    backbone, tokenizer = load_backbone()

    print(f"\n{D}Generating tasks...{N}")
    tasks = generate_tasks()
    print(f"  {len(tasks)} tasks")

    print(f"\n{D}Collecting traces...{N}")
    t0 = time.time()
    S, H, T, HT, LT = collect_traces(tasks, backbone, tokenizer)
    print(f"  {len(S)} trace steps")
    print(f"  Collected in {time.time()-t0:.1f}s")

    print(f"\n{D}Training...{N}")
    t0 = time.time()
    model = train(S, H, T, HT, LT, steps=40000)
    print(f"  Trained in {time.time()-t0:.1f}s")

    mx.savez("weights.npz", **dict(tree_flatten(model.parameters())))
    print(f"\n  Saved: weights.npz")
    print(f"  Run: uv run demo")


if __name__ == "__main__":
    main()
