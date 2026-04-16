"""
Train: frozen LLM backbone + autoregressive control head → CHIP-8 opcodes.

Trained with scheduled sampling so pure-inference accuracy tracks
teacher-forced accuracy (closes the exposure-bias gap).

Usage:
    uv run train
"""

import hashlib
import os
import struct
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .chip8 import Chip8, PROGRAM_START
from .model import (
    BACKBONE_DIM,
    MAX_TOKENS,
    STATE_DIM,
    ReflexModel,
    encode_instruction,
    load_backbone,
)


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


def prog_draw_sprite(sprite_bytes: list[int], x: int, y: int) -> bytes:
    """Draw a custom sprite. Stores sprite data at 0x300, then draws it."""
    height = len(sprite_bytes)
    ops = [0x00E0]
    for i, b in enumerate(sprite_bytes[:8]):
        ops.append(0x6000 | (i << 8) | (b & 0xFF))
    ops.append(0xA300)
    ops.append(0xF055 | ((min(height, 8) - 1) << 8))
    ops.append(0x6000 | (8 << 8) | (x & 0xFF))
    ops.append(0x6000 | (9 << 8) | (y & 0xFF))
    ops.append(0xA300)
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

DIGIT_NAMES = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
}


# ── Phrasing generation ──────────────────────────────────────────────

def digit_phrasings(digit: int, x: int, y: int) -> list[str]:
    d = f"{digit:X}"
    phrases = [
        f"draw digit {d} at position {x} {y}",
        f"draw digit {d} at {x} {y}",
        f"digit {d} at position {x} {y}",
        f"digit {d} at {x} {y}",
        f"put digit {d} at {x} {y}",
        f"show digit {d} at position {x} {y}",
    ]
    if x == 10 and y == 10:
        phrases += [
            f"draw digit {d}",
            f"draw a {d}",
            f"draw {d}",
            f"show {d}",
            f"display {d}",
            f"{d}",
            f"render {d}",
            f"number {d}",
            f"show me {d}",
            f"put {d} on screen",
            f"the number {d}",
        ]
        if digit in DIGIT_NAMES:
            name = DIGIT_NAMES[digit]
            phrases += [
                f"draw {name}",
                f"draw a {name}",
                f"show {name}",
                f"the number {name}",
                f"{name}",
            ]
    return phrases


def arithmetic_phrasings(a: int, b: int) -> list[str]:
    return [
        f"compute {a} plus {b} and draw result",
        f"add {a} and {b}",
        f"calculate {a} + {b}",
        f"{a} + {b}",
        f"{a} plus {b}",
        f"compute {a}+{b}",
        f"what is {a} + {b}",
        f"sum of {a} and {b}",
        f"add {a} to {b}",
        f"show {a} + {b}",
        f"draw {a} + {b}",
        f"result of {a} plus {b}",
    ]


def sprite_phrasings(name: str) -> list[str]:
    return [
        f"draw a {name}",
        f"draw {name}",
        f"show me a {name}",
        f"show a {name}",
        f"show {name}",
        f"display {name}",
        f"display a {name}",
        f"render a {name}",
        f"render {name}",
        f"create a {name}",
        f"make a {name}",
        f"put a {name}",
        f"{name}",
        f"a {name}",
        f"draw me a {name}",
        f"can you draw a {name}",
        f"please draw a {name}",
        f"i want a {name}",
        f"show me {name}",
    ]


def sprite_position_phrasings(name: str, x: int, y: int) -> list[str]:
    return [
        f"draw {name} at position {x} {y}",
        f"draw a {name} at position {x} {y}",
        f"draw {name} at {x} {y}",
        f"{name} at position {x} {y}",
        f"put a {name} at {x} {y}",
        f"show {name} at {x} {y}",
    ]


def two_digit_phrasings(d1: int, d2: int) -> list[str]:
    a, b = f"{d1:X}", f"{d2:X}"
    return [
        f"draw digits {a} and {b}",
        f"show digits {a} and {b}",
        f"display {a} and {b}",
        f"draw {a} and {b}",
        f"{a} and {b}",
        f"show {a} {b}",
        f"digits {a} {b}",
    ]


def generate_tasks() -> list[tuple[str, bytes]]:
    tasks = []

    for digit in range(16):
        for x in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
            for y in [5, 10, 15, 20, 25]:
                for phrase in digit_phrasings(digit, x, y):
                    tasks.append((phrase, prog_draw_digit(digit, x, y)))

    for d1 in range(16):
        for d2 in range(16):
            for phrase in two_digit_phrasings(d1, d2):
                tasks.append((phrase, prog_draw_two_digits(d1, d2)))

    for a in range(10):
        for b in range(10):
            for phrase in arithmetic_phrasings(a, b):
                tasks.append((phrase, prog_add_and_draw(a, b)))

    for name, sprite in SPRITES.items():
        for x in [5, 15, 25, 35]:
            for y in [3, 10, 18]:
                for phrase in sprite_position_phrasings(name, x, y):
                    tasks.append((phrase, prog_draw_sprite(sprite, x, y)))

    for name, sprite in SPRITES.items():
        for phrase in sprite_phrasings(name):
            tasks.append((phrase, prog_draw_sprite(sprite, 20, 10)))

    return tasks


# ── Data collection ────────────────────────────────────────────────────

def load_or_encode(tasks, backbone, tokenizer):
    """Cache backbone encodings to disk — saves ~5 min per run."""
    task_hash = hashlib.md5(str(sorted(set(i for i, _ in tasks))).encode()).hexdigest()[:8]
    cache_file = f"encoding_cache_{task_hash}.npz"

    if os.path.exists(cache_file):
        print(f"  Loading cached encodings from {cache_file}...")
        data = np.load(cache_file, allow_pickle=True)
        return dict(data["cache"].item())

    print("  Encoding instructions through backbone...")
    instr_cache = {}
    unique = sorted(set(i for i, _ in tasks))
    for idx, instr in enumerate(unique):
        h, tid = encode_instruction(instr, backbone, tokenizer)
        mx.eval(h)
        instr_cache[instr] = (np.array(h[0]), tid)
        if (idx + 1) % 500 == 0:
            print(f"    {idx + 1}/{len(unique)} encoded...")

    np.savez(cache_file, cache=instr_cache)
    print(f"  Cached {len(instr_cache)} encodings to {cache_file}")
    return instr_cache


def collect_sequences(tasks, instr_cache):
    """Collect per-program sequences for autoregressive training."""
    chip = Chip8()
    programs = []

    for instr, program in tasks:
        hidden, tid = instr_cache[instr]
        states, hi_targets, lo_targets = [], [], []

        chip.load_program(program)
        for _ in range(len(program) // 2):
            if chip.pc < PROGRAM_START or chip.pc >= PROGRAM_START + len(program):
                break
            state = chip.get_state()
            hi = int(chip.memory[chip.pc])
            lo = int(chip.memory[chip.pc + 1])
            states.append(state)
            hi_targets.append(hi)
            lo_targets.append(lo)
            chip.step((hi << 8) | lo)

        # STOP token
        states.append(chip.get_state())
        hi_targets.append(0)
        lo_targets.append(0)

        programs.append((hidden, tid, states, hi_targets, lo_targets))

    max_steps = max(len(s) for _, _, s, _, _ in programs)
    max_seq = max(h.shape[0] for h, _, _, _, _ in programs)
    n = len(programs)

    H = np.zeros((n, max_seq, BACKBONE_DIM), dtype=np.float32)
    T = np.zeros((n, MAX_TOKENS), dtype=np.int32)
    S = np.zeros((n, max_steps, STATE_DIM), dtype=np.float32)
    HT = np.zeros((n, max_steps), dtype=np.int32)
    LT = np.zeros((n, max_steps), dtype=np.int32)
    M = np.zeros((n, max_steps), dtype=np.float32)

    for i, (hidden, tid, states, hi_targets, lo_targets) in enumerate(programs):
        H[i, :hidden.shape[0], :] = hidden
        T[i] = tid
        for j in range(len(states)):
            S[i, j] = states[j]
            HT[i, j] = hi_targets[j]
            LT[i, j] = lo_targets[j]
            M[i, j] = 1.0

    total = int(M.sum())
    print(f"  {n} programs, {total} total steps, max {max_steps} steps/program")
    return H, T, S, HT, LT, M, max_steps


# ── Training ───────────────────────────────────────────────────────────

def linear_epsilon(step, total_steps, start=1.0, end=0.1):
    """Linear decay for scheduled sampling probability."""
    t = min(step / total_steps, 1.0)
    return start + (end - start) * t


def train(H, T, S, HT, LT, M, max_steps, steps=80000):
    model = ReflexModel()
    scheduler = optim.cosine_decay(3e-4, steps, end=1e-6)
    optimizer = optim.Adam(learning_rate=scheduler)

    Hm = mx.array(H)
    Tm = mx.array(T)
    Sm = mx.array(S)
    HTm = mx.array(HT)
    LTm = mx.array(LT)
    Mm = mx.array(M)
    n = len(H)
    batch_size = min(32, n)
    perfect = 0

    def loss_fn(model, h, s, t, ht, lt, mask, epsilon):
        B = s.shape[0]
        h_state = mx.zeros((B, model.dim))
        prev_hi = mx.zeros((B,), dtype=mx.int32)
        prev_lo = mx.zeros((B,), dtype=mx.int32)
        total_loss = mx.array(0.0)

        for step_t in range(max_steps):
            hi_logits, lo_logits, h_state = model(
                h, s[:, step_t], t, prev_hi, prev_lo, h_state
            )
            loss_hi = nn.losses.cross_entropy(hi_logits, ht[:, step_t]) * mask[:, step_t]
            loss_lo = nn.losses.cross_entropy(lo_logits, lt[:, step_t]) * mask[:, step_t]
            total_loss = total_loss + (loss_hi + loss_lo).sum()

            # Scheduled sampling: GT with prob ε, else model's own argmax (detached)
            use_gt = mx.random.uniform(shape=(B,)) < epsilon
            pred_hi = mx.argmax(mx.stop_gradient(hi_logits), axis=-1).astype(mx.int32)
            pred_lo = mx.argmax(mx.stop_gradient(lo_logits), axis=-1).astype(mx.int32)
            prev_hi = mx.where(use_gt, ht[:, step_t], pred_hi)
            prev_lo = mx.where(use_gt, lt[:, step_t], pred_lo)

        return total_loss / mask.sum()

    def eval_accuracy(epsilon):
        """Evaluate at given ε (1.0 = teacher forcing, 0.0 = pure inference)."""
        chunk = 64
        correct_hi = correct_lo = total = 0
        for i in range(0, n, chunk):
            h = Hm[i:i+chunk]
            s = Sm[i:i+chunk]
            t = Tm[i:i+chunk]
            ht = HTm[i:i+chunk]
            lt = LTm[i:i+chunk]
            mask = Mm[i:i+chunk]
            B = h.shape[0]
            h_state = mx.zeros((B, model.dim))
            prev_hi = mx.zeros((B,), dtype=mx.int32)
            prev_lo = mx.zeros((B,), dtype=mx.int32)
            for step_t in range(max_steps):
                hi_logits, lo_logits, h_state = model(
                    h, s[:, step_t], t, prev_hi, prev_lo, h_state
                )
                m = mask[:, step_t]
                pred_hi = mx.argmax(hi_logits, axis=-1).astype(mx.int32)
                pred_lo = mx.argmax(lo_logits, axis=-1).astype(mx.int32)
                correct_hi += ((pred_hi == ht[:, step_t]) * m).sum().item()
                correct_lo += ((pred_lo == lt[:, step_t]) * m).sum().item()
                total += m.sum().item()

                if epsilon >= 1.0:
                    prev_hi = ht[:, step_t]
                    prev_lo = lt[:, step_t]
                else:
                    use_gt = mx.random.uniform(shape=(B,)) < epsilon
                    prev_hi = mx.where(use_gt, ht[:, step_t], pred_hi)
                    prev_lo = mx.where(use_gt, lt[:, step_t], pred_lo)
        return min(correct_hi / total, correct_lo / total)

    print(f"  Training with scheduled sampling (ε: 1.0 → 0.1)")

    for step in range(steps):
        epsilon = linear_epsilon(step, steps)
        idx = mx.array(np.random.choice(n, batch_size, replace=False))
        loss, grads = nn.value_and_grad(model, loss_fn)(
            model, Hm[idx], Sm[idx], Tm[idx], HTm[idx], LTm[idx], Mm[idx], epsilon)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 250 == 0:
            tf_acc = eval_accuracy(1.0)
            inf_acc = eval_accuracy(0.0)
            print(f"  step {step:5d}  ε={epsilon:.2f}  loss={loss.item():.4f}  "
                  f"tf_acc={tf_acc:.1%}  inf_acc={inf_acc:.1%}")
            if inf_acc >= 0.9999:
                mx.savez("weights.npz", **dict(tree_flatten(model.parameters())))
                print(f"  Saved weights (inf_acc={inf_acc:.4%})")
            if inf_acc == 1.0:
                perfect += 1
                if perfect >= 2:
                    print(f"  Converged.")
                    return model
            else:
                perfect = 0

    return model


# ── Main ───────────────────────────────────────────────────────────────

B = "\033[1m"
D = "\033[2m"
N = "\033[0m"


def main():
    print(f"{B}Reflex — Training{N}\n")
    print(f"{D}Autoregressive control head + scheduled sampling{N}\n")

    backbone, tokenizer = load_backbone()

    print(f"\n{D}Generating tasks...{N}")
    tasks = generate_tasks()
    print(f"  {len(tasks)} tasks")

    print(f"\n{D}Collecting sequences...{N}")
    t0 = time.time()
    instr_cache = load_or_encode(tasks, backbone, tokenizer)
    print(f"  {len(instr_cache)} unique instructions")
    H, T, S, HT, LT, M, max_steps = collect_sequences(tasks, instr_cache)
    print(f"  Collected in {time.time()-t0:.1f}s")

    print(f"\n{D}Training...{N}")
    t0 = time.time()
    model = train(H, T, S, HT, LT, M, max_steps, steps=80000)
    print(f"  Trained in {time.time()-t0:.1f}s")

    mx.savez("weights.npz", **dict(tree_flatten(model.parameters())))
    print(f"  Saved: weights.npz")
    print(f"  Run: uv run demo")


if __name__ == "__main__":
    main()
