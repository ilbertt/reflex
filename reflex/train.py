"""
Train: frozen LLM backbone + autoregressive control head → CHIP-8 programs
WITH CONTROL FLOW (loops, conditionals, subroutines, calls/returns).

Each program is emitted as a static byte sequence, then loaded into the
emulator and executed. Trained with scheduled sampling so pure-inference
accuracy tracks teacher-forced accuracy.

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
    MAX_KV_LEN,
    MAX_TOKENS,
    OPC_BYTE_DIM,
    TID_DIM,
    ReflexModel,
    encode_instruction,
    load_backbone,
)


# ── Opcode encoders ────────────────────────────────────────────────────
# Tiny helpers so the templates below read like assembly.

def _b(*ops: int) -> bytes:
    """Pack a list of 16-bit opcodes into a CHIP-8 program (big-endian)."""
    return b"".join(struct.pack(">H", op & 0xFFFF) for op in ops)

def cls() -> int: return 0x00E0
def ret() -> int: return 0x00EE
def jump(nnn: int) -> int: return 0x1000 | (nnn & 0xFFF)
def call(nnn: int) -> int: return 0x2000 | (nnn & 0xFFF)
def skip_eq(x: int, nn: int) -> int: return 0x3000 | (x << 8) | (nn & 0xFF)
def skip_ne(x: int, nn: int) -> int: return 0x4000 | (x << 8) | (nn & 0xFF)
def skip_eqv(x: int, y: int) -> int: return 0x5000 | (x << 8) | (y << 4)
def mov(x: int, nn: int) -> int: return 0x6000 | (x << 8) | (nn & 0xFF)
def addn(x: int, nn: int) -> int: return 0x7000 | (x << 8) | (nn & 0xFF)
def movv(x: int, y: int) -> int: return 0x8000 | (x << 8) | (y << 4)
def addv(x: int, y: int) -> int: return 0x8004 | (x << 8) | (y << 4)
def subv(x: int, y: int) -> int: return 0x8005 | (x << 8) | (y << 4)
def seti(nnn: int) -> int: return 0xA000 | (nnn & 0xFFF)
def rnd(x: int, nn: int) -> int: return 0xC000 | (x << 8) | (nn & 0xFF)
def draw(x: int, y: int, n: int) -> int: return 0xD000 | (x << 8) | (y << 4) | (n & 0xF)
def get_dt(x: int) -> int: return 0xF007 | (x << 8)
def set_dt(x: int) -> int: return 0xF015 | (x << 8)
def font(x: int) -> int: return 0xF029 | (x << 8)
def bcd(x: int) -> int: return 0xF033 | (x << 8)
def store(x: int) -> int: return 0xF055 | (x << 8)
def load(x: int) -> int: return 0xF065 | (x << 8)


def _addr(ops: list[int]) -> int:
    """Address of the NEXT op to be appended."""
    return PROGRAM_START + 2 * len(ops)


def _halt(ops: list[int]) -> None:
    """Append a self-jump so the emulator halts cleanly at this point."""
    ops.append(jump(_addr(ops)))


# ── Program templates ─────────────────────────────────────────────────
# Each category has multiple structural variants. Phrasings stay agnostic
# — diversity comes from cross-product (params × variant × phrasing).

# 1. LOOPS — count from `start` to `end-1`, drawing each digit at
#    advancing x positions.

def prog_count_forward_v0(start: int, end: int) -> bytes:
    """V2 counter, V0/V1 coords, x step 8, with cls."""
    ops = [cls(), mov(0, 10), mov(1, 10), mov(2, start)]
    loop = _addr(ops)
    ops += [font(2), draw(0, 1, 5), addn(0, 8),
            addn(2, 1), skip_eq(2, end), jump(loop)]
    _halt(ops)
    return _b(*ops)


def prog_count_forward_v1(start: int, end: int) -> bytes:
    """V5 counter, V3/V4 coords, x step 7, no cls, different y."""
    ops = [mov(3, 8), mov(4, 12), mov(5, start)]
    loop = _addr(ops)
    ops += [font(5), draw(3, 4, 5), addn(3, 7),
            addn(5, 1), skip_eq(5, end), jump(loop)]
    _halt(ops)
    return _b(*ops)


def prog_count_forward_v2(start: int, end: int) -> bytes:
    """V4 counter, V0/V1 coords, x step 6, with cls, y=15."""
    ops = [cls(), mov(0, 6), mov(1, 15), mov(4, start)]
    loop = _addr(ops)
    ops += [font(4), draw(0, 1, 5), addn(0, 6),
            addn(4, 1), skip_eq(4, end), jump(loop)]
    _halt(ops)
    return _b(*ops)


def prog_count_backward_v0(start: int, end: int) -> bytes:
    """Count down from `end-1` to `start` using 8XY5 sub. V2 counter,
    V3=1 used as decrement constant."""
    ops = [cls(), mov(0, 10), mov(1, 10), mov(2, end - 1), mov(3, 1)]
    loop = _addr(ops)
    ops += [font(2), draw(0, 1, 5), addn(0, 8),
            subv(2, 3),
            # skip if V2 < start (V2 wraps below start when done)
            # Simpler: loop until V2 == start - 1 (i.e. underflow check
            # via skip_eq with the wrapped value). For start=1, end=10:
            # V2 goes 9..1, then subv makes it 0; we exit when V2 == 0.
            skip_eq(2, start - 1 if start > 0 else 0xFF),
            jump(loop)]
    _halt(ops)
    return _b(*ops)


def prog_count_backward_v1(start: int, end: int) -> bytes:
    """V5 counter, V3/V4 coords, x step 7, V6=1 decrement."""
    ops = [mov(3, 8), mov(4, 8), mov(5, end - 1), mov(6, 1)]
    loop = _addr(ops)
    ops += [font(5), draw(3, 4, 5), addn(3, 7),
            subv(5, 6),
            skip_eq(5, start - 1 if start > 0 else 0xFF),
            jump(loop)]
    _halt(ops)
    return _b(*ops)


COUNT_FWD_VARIANTS = [prog_count_forward_v0, prog_count_forward_v1, prog_count_forward_v2]
COUNT_BWD_VARIANTS = [prog_count_backward_v0, prog_count_backward_v1]


# 2. LOOPS — blink the screen N times by XOR-redrawing the digit-5 sprite.

def prog_blink_v0(n: int) -> bytes:
    """V0 counter, V1 target=2N, V2/V3 coords, V4=5. Loop does 2N XOR
    draws (ON-OFF-...-OFF), then one final draw so the result ends
    visible (verifiable). 'Blink N times' reads as N flicker cycles."""
    ops = [mov(0, 0), mov(1, 2 * n), mov(2, 28), mov(3, 12),
           mov(4, 5), font(4)]
    loop = _addr(ops)
    ops += [draw(2, 3, 5), addn(0, 1), skip_eqv(0, 1), jump(loop)]
    ops.append(draw(2, 3, 5))  # final ON
    _halt(ops)
    return _b(*ops)


def prog_blink_v1(n: int) -> bytes:
    """V5 counter, V6 target, V3/V4 coords, V7=5, x off-center."""
    ops = [mov(5, 0), mov(6, 2 * n), mov(3, 20), mov(4, 14),
           mov(7, 5), font(7)]
    loop = _addr(ops)
    ops += [draw(3, 4, 5), addn(5, 1), skip_eqv(5, 6), jump(loop)]
    ops.append(draw(3, 4, 5))  # final ON
    _halt(ops)
    return _b(*ops)


def prog_blink_v2(n: int) -> bytes:
    """V0 counter compared by skip_eq against literal target."""
    ops = [mov(0, 0), mov(1, 30), mov(2, 14),
           mov(3, 5), font(3)]
    loop = _addr(ops)
    ops += [draw(1, 2, 5), addn(0, 1), skip_eq(0, 2 * n), jump(loop)]
    ops.append(draw(1, 2, 5))  # final ON
    _halt(ops)
    return _b(*ops)


BLINK_VARIANTS = [prog_blink_v0, prog_blink_v1, prog_blink_v2]


# 3. CONDITIONALS — set Vx, then branch based on equality with NN.

def prog_if_eq_v0(test_val: int, threshold: int, then_d: int, else_d: int) -> bytes:
    """`4XNN` (skip-if-not-equal). V0 holds test_val, V1 holds the digit
    we draw. V2/V3 are coords."""
    ops = [cls(), mov(0, test_val), mov(2, 26), mov(3, 12),
           skip_ne(0, threshold)]
    # If V0 == threshold, fall through to "then" branch.
    after_then = None  # placeholder; we'll fix once we know its address
    then_jump_idx = len(ops)
    ops.append(0)  # reserve slot for jump-to-end
    # else branch:
    ops += [mov(1, else_d), font(1), draw(2, 3, 5)]
    end = _addr(ops) + 2  # +2 because next op is "jump end"
    ops.append(jump(end))
    # then branch:
    then_addr = _addr(ops)
    ops[then_jump_idx] = jump(then_addr)
    ops += [mov(1, then_d), font(1), draw(2, 3, 5)]
    _halt(ops)
    return _b(*ops)


def prog_if_eq_v1(test_val: int, threshold: int, then_d: int, else_d: int) -> bytes:
    """`3XNN` (skip-if-equal). V0 test, V4 digit, V5/V6 coords."""
    ops = [cls(), mov(0, test_val), mov(5, 26), mov(6, 12),
           skip_eq(0, threshold)]
    # If V0 == threshold, skip the "go to else" jump → fall through to then.
    else_jump_idx = len(ops)
    ops.append(0)  # placeholder
    # then branch:
    ops += [mov(4, then_d), font(4), draw(5, 6, 5)]
    end_addr = _addr(ops) + 2  # for the "jump end" we'll add
    ops.append(jump(end_addr))
    # else branch:
    else_addr = _addr(ops)
    ops[else_jump_idx] = jump(else_addr)
    ops += [mov(4, else_d), font(4), draw(5, 6, 5)]
    _halt(ops)
    return _b(*ops)


def prog_if_eq_v2(test_val: int, threshold: int, then_d: int, else_d: int) -> bytes:
    """V3 test register, V2 digit, V0/V1 coords, skip_ne."""
    ops = [cls(), mov(3, test_val), mov(0, 26), mov(1, 12),
           skip_ne(3, threshold)]
    then_jump_idx = len(ops)
    ops.append(0)
    ops += [mov(2, else_d), font(2), draw(0, 1, 5)]
    end = _addr(ops) + 2
    ops.append(jump(end))
    ops[then_jump_idx] = jump(_addr(ops))
    ops += [mov(2, then_d), font(2), draw(0, 1, 5)]
    _halt(ops)
    return _b(*ops)


IF_EQ_VARIANTS = [prog_if_eq_v0, prog_if_eq_v1, prog_if_eq_v2]


# 4. ARITHMETIC — add a + b, draw result. For result > 9, BCD + draw two
# digits. Source value lives in V3+ to avoid the FX65 V0..V2 clobber.

def _draw_two_digits(ops: list[int], src_reg: int, x_reg: int, y_reg: int,
                     scratch_addr: int, x_step: int) -> None:
    """Append BCD + load-into-V0..V2 + draw tens + draw ones."""
    ops += [seti(scratch_addr), bcd(src_reg), seti(scratch_addr), load(2)]
    # V0=hundreds, V1=tens, V2=ones; draw tens then ones.
    ops += [font(1), draw(x_reg, y_reg, 5),
            addn(x_reg, x_step),
            font(2), draw(x_reg, y_reg, 5)]


def _draw_one_digit(ops: list[int], src_reg: int, x_reg: int, y_reg: int) -> None:
    ops += [font(src_reg), draw(x_reg, y_reg, 5)]


def prog_add_show_v0(a: int, b: int) -> bytes:
    """V3=a, V0=b, V3+=V0; if result>9, BCD via 0x300; else draw V3."""
    result = a + b
    ops = [cls(), mov(3, a), mov(0, b), addv(3, 0),
           mov(4, 22), mov(5, 12)]
    if result > 9:
        _draw_two_digits(ops, src_reg=3, x_reg=4, y_reg=5,
                         scratch_addr=0x300, x_step=8)
    else:
        _draw_one_digit(ops, src_reg=3, x_reg=4, y_reg=5)
    _halt(ops)
    return _b(*ops)


def prog_add_show_v1(a: int, b: int) -> bytes:
    """V5=a, addn-immediate b, draw at V6/V7."""
    result = a + b
    ops = [cls(), mov(5, a), addn(5, b),
           mov(6, 24), mov(7, 14)]
    if result > 9:
        _draw_two_digits(ops, src_reg=5, x_reg=6, y_reg=7,
                         scratch_addr=0x320, x_step=7)
    else:
        _draw_one_digit(ops, src_reg=5, x_reg=6, y_reg=7)
    _halt(ops)
    return _b(*ops)


def prog_add_show_v2(a: int, b: int) -> bytes:
    """V4=a, V5=b, addv. Different scratch and step."""
    result = a + b
    ops = [cls(), mov(4, a), mov(5, b), addv(4, 5),
           mov(6, 20), mov(7, 10)]
    if result > 9:
        _draw_two_digits(ops, src_reg=4, x_reg=6, y_reg=7,
                         scratch_addr=0x340, x_step=9)
    else:
        _draw_one_digit(ops, src_reg=4, x_reg=6, y_reg=7)
    _halt(ops)
    return _b(*ops)


ADD_VARIANTS = [prog_add_show_v0, prog_add_show_v1, prog_add_show_v2]


# Multiply via repeated addition.
def prog_mul_show_v0(a: int, b: int) -> bytes:
    """V3=a (multiplicand), V4=b (multiplier), V5=acc, V6=loop counter."""
    result = a * b
    ops = [cls(), mov(3, a), mov(4, b), mov(5, 0), mov(6, 0)]
    loop = _addr(ops)
    ops += [addv(5, 3), addn(6, 1), skip_eqv(6, 4), jump(loop)]
    ops += [mov(0, 22), mov(1, 12)]
    if result > 9:
        _draw_two_digits(ops, src_reg=5, x_reg=0, y_reg=1,
                         scratch_addr=0x300, x_step=8)
    else:
        _draw_one_digit(ops, src_reg=5, x_reg=0, y_reg=1)
    _halt(ops)
    return _b(*ops)


def prog_mul_show_v1(a: int, b: int) -> bytes:
    """V7=mult, V8=mer, V9=acc, VA=ctr; draw at V0/V1."""
    result = a * b
    ops = [cls(), mov(7, a), mov(8, b), mov(9, 0), mov(0xA, 0)]
    loop = _addr(ops)
    ops += [addv(9, 7), addn(0xA, 1), skip_eqv(0xA, 8), jump(loop)]
    ops += [movv(5, 9),  # copy acc to V5 for BCD source (not strictly needed)
            mov(0, 24), mov(1, 14)]
    if result > 9:
        _draw_two_digits(ops, src_reg=5, x_reg=0, y_reg=1,
                         scratch_addr=0x320, x_step=7)
    else:
        _draw_one_digit(ops, src_reg=5, x_reg=0, y_reg=1)
    _halt(ops)
    return _b(*ops)


MUL_VARIANTS = [prog_mul_show_v0, prog_mul_show_v1]


# 5. SUBROUTINES — store a star sprite via FX55, then call a draw
# subroutine at two positions.

STAR = [0x10, 0x38, 0xFE, 0xFE, 0x6C, 0x44]  # 6 rows


def prog_subroutine_v0(x1: int, y1: int, x2: int, y2: int) -> bytes:
    """Subroutine placed AFTER main body; main reaches a self-jump halt
    that sits BEFORE the subroutine, so execution never falls into it."""
    ops = [
        mov(0, STAR[0]), mov(1, STAR[1]), mov(2, STAR[2]),
        mov(3, STAR[3]), mov(4, STAR[4]), mov(5, STAR[5]),
        seti(0x300), store(5),
    ]
    ops += [mov(6, x1), mov(7, y1)]
    sub_call_idx_1 = len(ops); ops.append(0)  # placeholder for call1
    ops += [mov(6, x2), mov(7, y2)]
    sub_call_idx_2 = len(ops); ops.append(0)  # placeholder for call2
    halt_addr = _addr(ops)
    ops.append(jump(halt_addr))               # self-jump halt
    sub_addr = _addr(ops)
    ops += [seti(0x300), draw(6, 7, len(STAR)), ret()]
    ops[sub_call_idx_1] = call(sub_addr)
    ops[sub_call_idx_2] = call(sub_addr)
    return _b(*ops)


def prog_subroutine_v1(x1: int, y1: int, x2: int, y2: int) -> bytes:
    """Subroutine placed BEFORE main; main starts with a JUMP that
    skips over the subroutine body."""
    # Layout:
    # [jump main] [subroutine: seti 0x300, draw, ret] [main body]
    ops = [0]  # placeholder for "jump main"
    sub_addr = _addr(ops)
    ops += [seti(0x300), draw(6, 7, len(STAR)), ret()]
    main_addr = _addr(ops)
    ops[0] = jump(main_addr)
    ops += [mov(0, STAR[0]), mov(1, STAR[1]), mov(2, STAR[2]),
            mov(3, STAR[3]), mov(4, STAR[4]), mov(5, STAR[5]),
            seti(0x300), store(5),
            mov(6, x1), mov(7, y1), call(sub_addr),
            mov(6, x2), mov(7, y2), call(sub_addr)]
    _halt(ops)
    return _b(*ops)


SUBROUTINE_VARIANTS = [prog_subroutine_v0, prog_subroutine_v1]


# 6. MEMORY — store a value at an address, load it back, optionally
# display via BCD.

def prog_store_load_v0(val: int, addr: int) -> bytes:
    """V0=val, store(0) writes mem[addr]=V0=val, load(0) reads it back.
    Copy to V3 for BCD display so the load result survives the FX65 in
    the BCD path."""
    ops = [cls(), mov(0, val), seti(addr), store(0),
           seti(addr), load(0), movv(3, 0),
           mov(4, 24), mov(5, 12)]
    if val > 9:
        _draw_two_digits(ops, src_reg=3, x_reg=4, y_reg=5,
                         scratch_addr=(addr + 0x10) & 0xFFF, x_step=8)
    else:
        _draw_one_digit(ops, src_reg=3, x_reg=4, y_reg=5)
    _halt(ops)
    return _b(*ops)


def prog_store_load_v1(val: int, addr: int) -> bytes:
    """Same V0=val store/load. Different display register layout (V5 source)."""
    ops = [cls(), mov(0, val), seti(addr), store(0),
           seti(addr), load(0), movv(5, 0),
           mov(6, 22), mov(7, 14)]
    if val > 9:
        _draw_two_digits(ops, src_reg=5, x_reg=6, y_reg=7,
                         scratch_addr=(addr + 0x20) & 0xFFF, x_step=7)
    else:
        _draw_one_digit(ops, src_reg=5, x_reg=6, y_reg=7)
    _halt(ops)
    return _b(*ops)


def prog_store_load_v2(val: int, addr: int) -> bytes:
    """V0=val, but copy to V7 BEFORE store (different register flow).
    Still satisfies 'load into V0'."""
    ops = [cls(), mov(0, val), movv(7, 0), seti(addr), store(0),
           seti(addr), load(0), movv(3, 0),
           mov(4, 26), mov(5, 10)]
    if val > 9:
        _draw_two_digits(ops, src_reg=3, x_reg=4, y_reg=5,
                         scratch_addr=(addr + 0x30) & 0xFFF, x_step=9)
    else:
        _draw_one_digit(ops, src_reg=3, x_reg=4, y_reg=5)
    _halt(ops)
    return _b(*ops)


STORE_LOAD_VARIANTS = [prog_store_load_v0, prog_store_load_v1, prog_store_load_v2]


# 7. TIMERS — set delay timer, busy-loop until expired, then draw digit.

def prog_wait_draw_v0(ticks: int, digit: int) -> bytes:
    """V0=ticks, set_dt, wait loop with V0=get_dt + skip_eq(0,0)."""
    ops = [cls(), mov(0, ticks), set_dt(0)]
    wait = _addr(ops)
    ops += [get_dt(0), skip_eq(0, 0), jump(wait)]
    ops += [mov(1, digit), font(1), mov(2, 28), mov(3, 12), draw(2, 3, 5)]
    _halt(ops)
    return _b(*ops)


def prog_wait_draw_v1(ticks: int, digit: int) -> bytes:
    """V5=ticks, V6 in wait loop, different draw register layout."""
    ops = [cls(), mov(5, ticks), set_dt(5)]
    wait = _addr(ops)
    ops += [get_dt(6), skip_eq(6, 0), jump(wait)]
    ops += [mov(7, digit), font(7), mov(0, 24), mov(1, 14), draw(0, 1, 5)]
    _halt(ops)
    return _b(*ops)


def prog_wait_draw_v2(ticks: int, digit: int) -> bytes:
    """skip_ne flipped logic: skip if V0 != 0 (still ticking) → loop;
    fall through when V0 == 0."""
    ops = [cls(), mov(0, ticks), set_dt(0)]
    wait = _addr(ops)
    ops += [get_dt(0), skip_ne(0, 0)]
    # placeholder for "jump after_wait" — taken when timer expired.
    after_idx = len(ops); ops.append(0)
    ops.append(jump(wait))
    ops[after_idx] = jump(_addr(ops))
    ops += [mov(2, digit), font(2), mov(3, 26), mov(4, 12), draw(3, 4, 5)]
    _halt(ops)
    return _b(*ops)


WAIT_VARIANTS = [prog_wait_draw_v0, prog_wait_draw_v1, prog_wait_draw_v2]


# 8. RANDOM — draw a random font sprite (digit 0..15).

def prog_random_v0() -> bytes:
    """Cap to 0..15 with `& 0x0F`."""
    ops = [cls(), rnd(0, 0x0F),
           mov(1, 28), mov(2, 12), font(0), draw(1, 2, 5)]
    _halt(ops)
    return _b(*ops)


def prog_random_v1() -> bytes:
    """Cap to 0..7 with `& 0x07`, different draw position."""
    ops = [cls(), rnd(3, 0x07),
           mov(4, 24), mov(5, 14), font(3), draw(4, 5, 5)]
    _halt(ops)
    return _b(*ops)


def prog_random_v2() -> bytes:
    """Random 0..15 in V5, copy to V3 for font lookup, draw at (V0,V1)."""
    ops = [cls(), rnd(5, 0x0F), movv(3, 5),
           mov(0, 26), mov(1, 10), font(3), draw(0, 1, 5)]
    _halt(ops)
    return _b(*ops)


RANDOM_VARIANTS = [prog_random_v0, prog_random_v1, prog_random_v2]


# ── Phrasing pools ────────────────────────────────────────────────────

NUM_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 20: "twenty", 30: "thirty", 42: "forty-two",
    45: "forty-five", 60: "sixty",
}


def _word(n: int) -> str:
    return NUM_WORDS.get(n, str(n))


def count_phrasings(start: int, end: int) -> list[str]:
    """`end` is exclusive (display start..end-1)."""
    s, e = start, end - 1
    sw, ew = _word(s), _word(e)
    return [
        f"count from {s} to {e} and display each digit",
        f"count from {s} to {e} and show each digit",
        f"count from {s} to {e}",
        f"count from {sw} to {ew}",
        f"draw digits {s} through {e} in a row",
        f"draw digits {s} to {e}",
        f"display digits {s} to {e}",
        f"show digits {s} through {e}",
        f"loop from {s} to {e} and draw the digits",
        f"iterate from {s} to {e} drawing each number",
        f"render digits {s} through {e}",
        f"print {s} to {e} on screen",
        f"show me a count from {s} to {e}",
        f"go from {s} to {e} drawing each digit",
        f"step from {s} to {e} and draw each",
        f"draw the digits {s} to {e} side by side",
        f"display every digit from {s} to {e}",
        f"counting up from {s} to {e}",
        f"count {s} to {e} on the screen",
        f"show numbers {s} through {e}",
        f"draw {sw} through {ew}",
        f"display the digits from {sw} to {ew}",
        f"loop {s} to {e} drawing each",
        f"the digits from {s} up to {e}",
        f"a count from {s} to {e}",
        f"counting from {s} to {e} digit by digit",
        f"draw a row of digits {s} through {e}",
        f"please show digits {s} to {e}",
        f"show the numbers {s} to {e}",
        f"display {s} {s+1} ... {e}",
    ]


def count_down_phrasings(start: int, end: int) -> list[str]:
    """Backward-count: display end-1 down to start."""
    hi, lo = end - 1, start
    hw, lw = _word(hi), _word(lo)
    return [
        f"count down from {hi} to {lo}",
        f"count from {hi} down to {lo}",
        f"countdown from {hi} to {lo}",
        f"draw digits {hi} down to {lo}",
        f"display digits from {hi} down to {lo}",
        f"show digits {hi} to {lo} going down",
        f"loop from {hi} down to {lo}",
        f"iterate from {hi} down to {lo} drawing each",
        f"count {hi} down to {lo} on the screen",
        f"go from {hi} to {lo} backwards",
        f"reverse count from {hi} to {lo}",
        f"step backward from {hi} to {lo}",
        f"draw {hi} {hi-1} ... {lo}",
        f"display each digit from {hi} down to {lo}",
        f"countdown {hw} to {lw}",
        f"count down from {hw} to {lw}",
        f"draw digits in descending order from {hi} to {lo}",
        f"render digits {hi} through {lo} descending",
        f"a countdown from {hi} to {lo}",
        f"counting down {hi} to {lo}",
        f"backwards from {hi} to {lo}",
        f"show me a countdown from {hi} to {lo}",
        f"display digits {hi} downto {lo}",
        f"show {hi} {hi-1} {hi-2} down to {lo}",
        f"the digits from {hi} down to {lo}",
        f"please count down from {hi} to {lo}",
        f"draw the descending count from {hi} to {lo}",
        f"go backwards from {hi} drawing each digit until {lo}",
        f"countdown of digits {hi} to {lo}",
        f"reverse loop {hi} to {lo}",
    ]


def blink_phrasings(n: int) -> list[str]:
    nw = _word(n)
    return [
        f"blink the screen {n} times",
        f"blink the display {n} times",
        f"flash the screen {n} times",
        f"flash {n} times",
        f"blink {n} times",
        f"toggle the sprite {n} times",
        f"flicker {n} times",
        f"flash the digit {n} times",
        f"blink {nw} times",
        f"flash the screen {nw} times",
        f"do {n} blinks",
        f"do {nw} blinks",
        f"draw and erase {n} times",
        f"toggle on and off {n} times",
        f"blink a sprite {n} times",
        f"make the screen blink {n} times",
        f"flash a sprite on the screen {n} times",
        f"on/off the sprite {n} times",
        f"flash the pixels {n} times",
        f"blink the pixels {n} times",
        f"toggle display {n} times",
        f"alternate the sprite {n} times",
        f"flash the figure {n} times",
        f"blink {n}x",
        f"draw and clear {n} times",
        f"{n} blinks",
        f"a sprite blinking {n} times",
        f"please blink the screen {n} times",
        f"have the sprite blink {n} times",
        f"do a {n}-times blink",
    ]


def if_eq_phrasings(test_val: int, threshold: int, then_d: int, else_d: int) -> list[str]:
    """Every phrasing names `test_val` — the program byte that gets it
    into V0 — so each phrasing maps unambiguously to one program."""
    return [
        f"set V0 to {test_val}; if V0 equals {threshold} draw a {then_d} else draw a {else_d}",
        f"V0={test_val}; if V0 == {threshold} draw {then_d} otherwise draw {else_d}",
        f"with V0={test_val}, if the register equals {threshold} draw {then_d} else draw {else_d}",
        f"set V0 to {test_val} and check if V0 is {threshold}; if so draw {then_d}, else draw {else_d}",
        f"set V0 to {test_val}; if V0 == {threshold} draw {then_d} else {else_d}",
        f"V0 is {test_val}: when V0 equals {threshold} show {then_d} otherwise show {else_d}",
        f"V0={test_val}; if V0 = {threshold} display {then_d}, else display {else_d}",
        f"V0={test_val}; draw {then_d} when V0 is {threshold}, else draw {else_d}",
        f"V0={test_val}; branch on V0 == {threshold}: then {then_d} else {else_d}",
        f"V0={test_val}; if value is {threshold} draw a {then_d} otherwise a {else_d}",
        f"V0 starts at {test_val}: compare V0 to {threshold}; equal -> {then_d}, not equal -> {else_d}",
        f"V0={test_val}; if equal to {threshold} render {then_d} else {else_d}",
        f"with V0 set to {test_val}, branch: equal {threshold} draws {then_d}, else {else_d}",
        f"V0={test_val}; branch when V0 = {threshold} to draw {then_d} otherwise {else_d}",
        f"V0={test_val}; display {then_d} if V0 == {threshold} otherwise display {else_d}",
        f"V0={test_val}; if-else: V0 == {threshold} ? {then_d} : {else_d}",
        f"V0 = {test_val}; equal-to-{threshold} branch draws {then_d} else {else_d}",
        f"V0={test_val}; draw {then_d} if V0 is {threshold}, else {else_d}",
        f"V0={test_val}; check V0=={threshold}; draw {then_d} or {else_d}",
        f"V0={test_val}; conditional draw: V0 ?= {threshold} -> {then_d} else {else_d}",
        f"V0={test_val}; if V0 holds {threshold} draw a {then_d}, otherwise a {else_d}",
        f"set V0={test_val} and branch on equality with {threshold}: {then_d} or {else_d}",
        f"V0={test_val}: equal {threshold} -> draw {then_d}; not equal -> draw {else_d}",
        f"V0={test_val}; depending on V0 vs {threshold}, draw {then_d} (eq) or {else_d} (ne)",
        f"V0={test_val}; if equals {threshold} then {then_d} else {else_d}",
        f"V0 set to {test_val}; show {then_d} on equality with {threshold}, else {else_d}",
        f"V0={test_val}; draw {then_d} when V0 matches {threshold} else {else_d}",
        f"branch program V0={test_val}: V0=={threshold}? draw {then_d}, else draw {else_d}",
        f"if-then-else with V0={test_val} comparing {threshold} -> {then_d} or {else_d}",
        f"V0 = {test_val}, draw {then_d} if equal to {threshold}, otherwise {else_d}",
    ]


def add_phrasings(a: int, b: int) -> list[str]:
    aw, bw = _word(a), _word(b)
    return [
        f"add {a} and {b} and show the result",
        f"add {a} and {b} and draw the result",
        f"compute {a} plus {b} and display it",
        f"compute {a} + {b} and draw the result",
        f"calculate {a} + {b} and show it",
        f"draw {a} + {b}",
        f"show {a} plus {b}",
        f"show me {a} + {b}",
        f"sum {a} and {b} and display",
        f"the sum of {a} and {b}",
        f"what is {a} plus {b}",
        f"display {a} + {b}",
        f"draw the sum of {a} and {b}",
        f"add {aw} and {bw} and draw the result",
        f"compute {aw} plus {bw}",
        f"plus: {a} {b} -> draw",
        f"{a} plus {b} on screen",
        f"render {a} + {b}",
        f"show the result of {a} + {b}",
        f"draw the result of adding {a} and {b}",
        f"{a}+{b}",
        f"{a} + {b} = ?",
        f"please show {a} + {b}",
        f"work out {a} + {b} and draw it",
        f"{a} added to {b}",
    ]


def mul_phrasings(a: int, b: int) -> list[str]:
    aw, bw = _word(a), _word(b)
    return [
        f"compute {a} times {b} and show it",
        f"multiply {a} by {b} and display it",
        f"draw {a} * {b}",
        f"show {a} times {b}",
        f"calculate {a} * {b} and draw the result",
        f"compute {a} x {b}",
        f"the product of {a} and {b}",
        f"what is {a} times {b}",
        f"draw the product of {a} and {b}",
        f"render {a} multiplied by {b}",
        f"compute {aw} times {bw}",
        f"multiply {aw} and {bw} and display",
        f"{a}*{b}",
        f"{a} x {b} = ?",
        f"show me {a} times {b}",
        f"display {a} * {b}",
        f"please show {a} * {b}",
        f"draw {a} multiplied by {b}",
        f"work out {a} times {b}",
        f"product of {a} and {b}",
    ]


def subroutine_phrasings(x1: int, y1: int, x2: int, y2: int) -> list[str]:
    return [
        "draw a star using a subroutine called twice",
        "use a subroutine to draw a star at two positions",
        "draw two stars using a subroutine",
        "call a draw-star subroutine twice",
        "draw a star, then draw another star, using a subroutine",
        f"draw a star at ({x1},{y1}) and ({x2},{y2}) via a subroutine",
        "use a subroutine to render two stars",
        "subroutine that draws a star, called twice",
        "draw two stars by calling a routine twice",
        "have a subroutine and use it twice to draw stars",
        "make a star-drawing subroutine and call it twice",
        "two stars rendered via the same subroutine",
        "call the star routine twice",
        f"draw a star at {x1} {y1} and a star at {x2} {y2} using a subroutine",
        "use a function to draw a star, twice",
        "render two stars with one subroutine",
        "subroutine: draw star, called twice",
        "twice-called star subroutine",
        "draw a star then call again to draw another",
        "two-star program using a subroutine",
        "star sprite drawn via subroutine, twice",
        "factor the star draw into a subroutine, call twice",
        "abstract the star draw into a function, use it twice",
        "use 2NNN to call a star-draw routine twice",
        "draw stars using a callable routine, twice",
        "draw a star with a subroutine, twice",
        "two invocations of a star subroutine",
        "draw two stars by reusing a subroutine",
        "call/return: draw two stars",
        "subroutine-based two-star draw",
    ]


def store_load_phrasings(val: int, addr: int) -> list[str]:
    vw = _word(val)
    return [
        f"store {val} at address 0x{addr:03X} and load it into V0",
        f"store {val} at 0x{addr:03X} then load to V0",
        f"write {val} to memory at 0x{addr:03X} and read it back",
        f"put {val} at memory location 0x{addr:03X} and load it",
        f"save {val} at 0x{addr:03X} and load back",
        f"store the value {val} at address 0x{addr:03X} and read it",
        f"write {val} into 0x{addr:03X} and load it",
        f"store {vw} at address 0x{addr:03X} and load it",
        f"memory store {val} at 0x{addr:03X}, then load",
        f"store {val} into memory at 0x{addr:03X} and read into V0",
        f"put {val} in memory at 0x{addr:03X} and load",
        f"write the byte {val} to 0x{addr:03X} and load it back",
        f"store {val} at addr {addr} and load",
        f"save the number {val} at address 0x{addr:03X} and read it back",
        f"persist {val} at 0x{addr:03X} and load",
        f"store {val} at 0x{addr:03X}, retrieve and display",
        f"write {val} at memory address 0x{addr:03X} then load",
        f"value {val} -> 0x{addr:03X} -> load -> display",
        f"store value {val} into 0x{addr:03X} and load back",
        f"memory write {val} at 0x{addr:03X} and read",
    ]


def wait_phrasings(ticks: int, digit: int) -> list[str]:
    tw, dw = _word(ticks), _word(digit)
    return [
        f"wait for {ticks} ticks then draw a {digit}",
        f"wait {ticks} ticks then show a {digit}",
        f"delay {ticks} ticks then draw {digit}",
        f"after {ticks} ticks draw a {digit}",
        f"pause {ticks} ticks then draw {digit}",
        f"wait {ticks} ticks before drawing {digit}",
        f"wait {tw} ticks then draw a {dw}",
        f"countdown {ticks} ticks then draw {digit}",
        f"set delay timer to {ticks} then draw {digit}",
        f"wait {ticks} cycles then display {digit}",
        f"sleep {ticks} ticks and draw {digit}",
        f"after a {ticks}-tick wait draw {digit}",
        f"wait {ticks} ticks; draw {digit}",
        f"delay for {ticks} ticks then show {digit}",
        f"timer {ticks} then draw {digit}",
        f"hold for {ticks} ticks before drawing {digit}",
        f"wait {ticks} timer ticks then draw a {digit}",
        f"after the delay timer hits 0 (started at {ticks}), draw {digit}",
        f"start delay {ticks}, on expiry draw {digit}",
        f"set the delay to {ticks} and then draw {digit} when it expires",
    ]


def random_phrasings() -> list[str]:
    return [
        "draw a random digit",
        "show a random digit",
        "render a random digit on screen",
        "display a random number",
        "draw a random number from the font",
        "pick a random digit and draw it",
        "random digit on screen",
        "draw any random digit",
        "show me a random digit",
        "draw a random font sprite",
        "draw a random digit 0-15",
        "random font digit",
        "pick and display a random digit",
        "draw a digit at random",
        "display a random digit on the screen",
        "render any digit at random",
        "show one random digit",
        "draw a random hex digit",
        "random digit please",
        "give me a random digit",
        "any random digit on screen",
        "render a random one of the digits",
        "select and draw a random digit",
        "draw a single random digit",
        "show a random hex digit",
    ]


# ── Verification ──────────────────────────────────────────────────────

def verify_program(program: bytes, expected: dict, max_cycles: int = 2000,
                   chip: Chip8 | None = None) -> tuple[bool, str]:
    """Run program in a fresh emulator. Halt detected via 1NNN self-jump.

    `expected` keys (all optional):
      - display_pixel_min: int
      - v_register: (idx, val) — V[idx] must equal val at halt
      - memory: (addr, val) — memory[addr] must equal val at halt
      - stack_balanced: True — sp must be 0 at halt
    Returns (ok, reason).
    """
    if chip is None:
        chip = Chip8()
    chip.load_program(program)
    halted = False
    for _ in range(max_cycles):
        pc = chip.pc
        if pc + 1 >= len(chip.memory):
            return False, "pc out of bounds"
        opcode = (int(chip.memory[pc]) << 8) | int(chip.memory[pc + 1])
        # Self-jump halt detection.
        if (opcode & 0xF000) == 0x1000 and (opcode & 0x0FFF) == pc:
            halted = True
            break
        chip.step(opcode)
    if not halted:
        return False, "did not halt within cycle budget"

    if "display_pixel_min" in expected:
        pixels = int(chip.display.sum())
        if pixels < expected["display_pixel_min"]:
            return False, f"only {pixels} pixels (need {expected['display_pixel_min']})"
    if "v_register" in expected:
        idx, val = expected["v_register"]
        if int(chip.V[idx]) != (val & 0xFF):
            return False, f"V{idx:X}={int(chip.V[idx])} != {val}"
    if "memory" in expected:
        addr, val = expected["memory"]
        if int(chip.memory[addr]) != (val & 0xFF):
            return False, f"mem[{addr:03X}]={int(chip.memory[addr])} != {val}"
    if expected.get("stack_balanced"):
        if int(chip.sp) != 0:
            return False, f"sp={int(chip.sp)} (stack not balanced)"
    return True, "ok"


# ── Task generation ───────────────────────────────────────────────────

def _add_tasks(tasks: list, label: str, rejected: dict, phrasings: list[str],
               variants: list, args: tuple, expected: dict) -> None:
    """Distribute phrasings across structural variants round-robin. Each
    (phrasing, args) maps to exactly one program — no label ambiguity."""
    valid = []
    for v in variants:
        program = v(*args)
        ok, _ = verify_program(program, expected)
        if ok:
            valid.append(program)
        else:
            rejected[label] = rejected.get(label, 0) + 1
    if not valid:
        return
    for i, phrase in enumerate(phrasings):
        tasks.append((phrase, valid[i % len(valid)]))


def generate_tasks() -> list[tuple[str, bytes]]:
    tasks: list[tuple[str, bytes]] = []
    rejected: dict[str, int] = {}

    # 1a. Forward count loops.
    # (start, end) — end exclusive. start in 0..2, end in 4..10.
    fwd_count = [(s, e) for s in (0, 1, 2) for e in range(4, 11) if e - s >= 3]
    for s, e in fwd_count:
        phrasings = count_phrasings(s, e)
        # Expect at least ~3 pixels per drawn digit.
        expected = {"display_pixel_min": max(1, 3 * (e - s))}
        _add_tasks(tasks, "count_fwd", rejected, phrasings,
                   COUNT_FWD_VARIANTS, (s, e), expected)

    # 1b. Backward count loops.
    bwd_count = [(s, e) for s in (1, 2) for e in range(4, 10) if e - s >= 3]
    for s, e in bwd_count:
        phrasings = count_down_phrasings(s, e)
        expected = {"display_pixel_min": max(1, 3 * (e - s))}
        _add_tasks(tasks, "count_bwd", rejected, phrasings,
                   COUNT_BWD_VARIANTS, (s, e), expected)

    # 2. Blink the screen N times. Each variant ends with a final ON draw,
    # so the sprite is always visible at halt regardless of N parity.
    for n in (1, 2, 3, 4, 5, 6):
        phrasings = blink_phrasings(n)
        expected = {"display_pixel_min": 5}
        _add_tasks(tasks, "blink", rejected, phrasings,
                   BLINK_VARIANTS, (n,), expected)

    # 3. Conditionals.
    cond_cases = []
    for test_val in (0, 1, 3, 5, 7, 10, 15):
        for threshold in (0, 1, 5, 10):
            for then_d, else_d in [(1, 0), (3, 7), (5, 2)]:
                cond_cases.append((test_val, threshold, then_d, else_d))
    for test_val, threshold, then_d, else_d in cond_cases:
        phrasings = if_eq_phrasings(test_val, threshold, then_d, else_d)
        expected = {"display_pixel_min": 5}
        _add_tasks(tasks, "if_eq", rejected, phrasings,
                   IF_EQ_VARIANTS, (test_val, threshold, then_d, else_d), expected)

    # 4a. Add and show.
    add_pairs = [(a, b) for a in range(0, 10) for b in range(0, 10) if a + b <= 18]
    for a, b in add_pairs:
        phrasings = add_phrasings(a, b)
        # Source register varies across variants; pixel-count is the
        # common ground (BCD load clobbers V0..V2 anyway).
        expected = {"display_pixel_min": 4}
        _add_tasks(tasks, "add", rejected, phrasings,
                   ADD_VARIANTS, (a, b), expected)

    # 4b. Multiply and show.
    mul_pairs = [(a, b) for a in range(2, 6) for b in range(2, 6)]
    for a, b in mul_pairs:
        phrasings = mul_phrasings(a, b)
        expected = {"display_pixel_min": 4}
        _add_tasks(tasks, "mul", rejected, phrasings,
                   MUL_VARIANTS, (a, b), expected)

    # 5. Subroutines. Phrasings are mostly position-agnostic; distribute
    # them round-robin across (position × variant) so each phrasing maps
    # to a single program (no label ambiguity).
    sub_positions = [
        (4, 4, 32, 4),
        (8, 8, 40, 8),
        (5, 12, 30, 12),
        (10, 6, 36, 18),
        (6, 6, 26, 18),
        (12, 4, 44, 4),
    ]
    sub_expected = {"display_pixel_min": 6, "stack_balanced": True}
    sub_combos: list[bytes] = []
    for (x1, y1, x2, y2) in sub_positions:
        for variant in SUBROUTINE_VARIANTS:
            program = variant(x1, y1, x2, y2)
            ok, _ = verify_program(program, sub_expected)
            if ok:
                sub_combos.append(program)
            else:
                rejected["subroutine"] = rejected.get("subroutine", 0) + 1
    sub_phrasings = subroutine_phrasings(0, 0, 0, 0)  # position-agnostic baseline
    for i, phrase in enumerate(sub_phrasings):
        tasks.append((phrase, sub_combos[i % len(sub_combos)]))
    # Position-specific phrasings: one per (position) tied to a specific variant.
    for pi, (x1, y1, x2, y2) in enumerate(sub_positions):
        program = SUBROUTINE_VARIANTS[pi % len(SUBROUTINE_VARIANTS)](x1, y1, x2, y2)
        ok, _ = verify_program(program, sub_expected)
        if not ok:
            continue
        for phrase in [
            f"draw a star at ({x1},{y1}) and at ({x2},{y2}) via a subroutine",
            f"draw stars at {x1},{y1} and {x2},{y2} using a subroutine",
            f"call the star subroutine at positions ({x1},{y1}) and ({x2},{y2})",
            f"two stars at ({x1},{y1}) and ({x2},{y2}) drawn by a subroutine",
        ]:
            tasks.append((phrase, program))

    # 6. Memory store/load.
    store_cases = []
    for val in (0, 5, 9, 12, 25, 42, 50, 99, 128, 200, 255):
        for addr in (0x300, 0x320, 0x350, 0x380):
            store_cases.append((val, addr))
    for val, addr in store_cases:
        phrasings = store_load_phrasings(val, addr)
        # Expected: memory[addr] == val. The display value also reflects val.
        expected = {"memory": (addr, val), "display_pixel_min": 4}
        _add_tasks(tasks, "store_load", rejected, phrasings,
                   STORE_LOAD_VARIANTS, (val, addr), expected)

    # 7. Wait then draw.
    wait_cases = []
    for ticks in (5, 10, 15, 20, 30, 45, 60):
        for digit in (0, 1, 3, 5, 7, 9):
            wait_cases.append((ticks, digit))
    for ticks, digit in wait_cases:
        phrasings = wait_phrasings(ticks, digit)
        expected = {"display_pixel_min": 5}
        _add_tasks(tasks, "wait", rejected, phrasings,
                   WAIT_VARIANTS, (ticks, digit), expected)

    # 8. Random. Verify each variant over multiple seeds (the draw is
    # non-deterministic but should always produce SOME pixels).
    rand_combos: list[bytes] = []
    for variant in RANDOM_VARIANTS:
        program = variant()
        ok = True
        for seed in range(8):
            np.random.seed(seed)
            chip = Chip8()
            ok2, _ = verify_program(program, {"display_pixel_min": 2}, chip=chip)
            if not ok2:
                ok = False
                break
        if ok:
            rand_combos.append(program)
        else:
            rejected["random"] = rejected.get("random", 0) + 1
    rand_phrasings = random_phrasings()
    for i, phrase in enumerate(rand_phrasings):
        tasks.append((phrase, rand_combos[i % len(rand_combos)]))

    print(f"  Generated {len(tasks)} tasks")
    if rejected:
        print(f"  Rejected variants: {rejected}")
    else:
        print(f"  All variants verified")
    # Per-category counts.
    by_cat: dict[str, int] = {}
    for instr, prog in tasks:
        # Heuristic: bucket by program-length-class for quick sanity.
        by_cat[f"{len(prog)//2}-op"] = by_cat.get(f"{len(prog)//2}-op", 0) + 1
    print(f"  Programs by length: {dict(sorted(by_cat.items()))}")
    return tasks


# ── Data collection ───────────────────────────────────────────────────

def load_or_encode(tasks, backbone, tokenizer):
    """Cache backbone encodings to disk. Hash includes MAX_TOKENS so a
    shape change invalidates the cache (silent truncation would be a bug)."""
    instr_set = sorted(set(i for i, _ in tasks))
    hash_input = f"MAX_TOKENS={MAX_TOKENS}|" + "\n".join(instr_set)
    task_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    cache_file = f"encoding_cache_{task_hash}.npz"

    if os.path.exists(cache_file):
        print(f"  Loading cached encodings from {cache_file}...")
        data = np.load(cache_file, allow_pickle=True)
        return dict(data["cache"].item())

    print("  Encoding instructions through backbone...")
    instr_cache = {}
    for idx, instr in enumerate(instr_set):
        h, tid = encode_instruction(instr, backbone, tokenizer)
        mx.eval(h)
        instr_cache[instr] = (np.array(h[0]), tid)
        if (idx + 1) % 500 == 0:
            print(f"    {idx + 1}/{len(instr_set)} encoded...")

    np.savez(cache_file, cache=instr_cache)
    print(f"  Cached {len(instr_cache)} encodings to {cache_file}")
    return instr_cache


def collect_sequences(tasks, instr_cache):
    """Linear-emit sequences. The model emits the program in memory order;
    K/V at each step is the model's own opcode history (built inside the
    train loop), so we no longer carry a state tensor here."""
    programs = []
    for instr, program in tasks:
        hidden, tid = instr_cache[instr]
        hi_targets, lo_targets = [], []
        for emit_pc in range(PROGRAM_START, PROGRAM_START + len(program), 2):
            offset = emit_pc - PROGRAM_START
            hi_targets.append(int(program[offset]))
            lo_targets.append(int(program[offset + 1]))
        # STOP token
        hi_targets.append(0)
        lo_targets.append(0)
        programs.append((hidden, tid, hi_targets, lo_targets))

    max_steps = max(len(h) for _, _, h, _ in programs)
    max_seq = max(h.shape[0] for h, _, _, _ in programs)
    n = len(programs)

    H = np.zeros((n, max_seq, BACKBONE_DIM), dtype=np.float32)
    T = np.zeros((n, MAX_TOKENS), dtype=np.int32)
    HT = np.zeros((n, max_steps), dtype=np.int32)
    LT = np.zeros((n, max_steps), dtype=np.int32)
    M = np.zeros((n, max_steps), dtype=np.float32)

    for i, (hidden, tid, hi_targets, lo_targets) in enumerate(programs):
        H[i, :hidden.shape[0], :] = hidden
        T[i] = tid
        for j in range(len(hi_targets)):
            HT[i, j] = hi_targets[j]
            LT[i, j] = lo_targets[j]
            M[i, j] = 1.0

    total = int(M.sum())
    print(f"  {n} programs, {total} total steps, max {max_steps} steps/program")
    if max_steps > MAX_KV_LEN:
        raise ValueError(f"max_steps ({max_steps}) exceeds MAX_KV_LEN ({MAX_KV_LEN})")
    return H, T, HT, LT, M, max_steps


# ── Training ──────────────────────────────────────────────────────────

def linear_epsilon(step, total_steps, start=1.0, end=0.1):
    """Linear decay for scheduled sampling probability."""
    t = min(step / total_steps, 1.0)
    return start + (end - start) * t


def _update_history(history: mx.array, step_t: int, new_op: mx.array) -> mx.array:
    """Functional update: write `new_op` (shape [B]) into column step_t of
    `history` (shape [B, MAX_KV_LEN])."""
    col_mask = mx.arange(MAX_KV_LEN) == step_t            # [MAX_KV_LEN] bool
    new_col = mx.broadcast_to(new_op[:, None], history.shape)
    return mx.where(col_mask[None, :], new_col, history)


def train(H, T, HT, LT, M, max_steps, steps=80000):
    model = ReflexModel()
    scheduler = optim.cosine_decay(3e-4, steps, end=1e-6)
    optimizer = optim.Adam(learning_rate=scheduler)

    Hm = mx.array(H)
    Tm = mx.array(T)
    HTm = mx.array(HT)
    LTm = mx.array(LT)
    Mm = mx.array(M)
    n = len(H)
    batch_size = min(32, n)
    perfect = 0

    def loss_fn(model, h, t, ht, lt, mask, epsilon):
        B = h.shape[0]
        h_state = mx.zeros((B, model.dim))
        prev_hi = mx.zeros((B,), dtype=mx.int32)
        prev_lo = mx.zeros((B,), dtype=mx.int32)
        history = mx.zeros((B, MAX_KV_LEN), dtype=mx.int32)
        total_loss = mx.array(0.0)

        for step_t in range(max_steps):
            # K/V valid count = start token (1) + opcodes emitted so far (step_t)
            hi_logits, lo_logits, h_state = model(
                h, history, step_t + 1, t, prev_hi, prev_lo, h_state
            )
            loss_hi = nn.losses.cross_entropy(hi_logits, ht[:, step_t]) * mask[:, step_t]
            loss_lo = nn.losses.cross_entropy(lo_logits, lt[:, step_t]) * mask[:, step_t]
            total_loss = total_loss + (loss_hi + loss_lo).sum()

            # Scheduled sampling: GT with prob ε, else model's own argmax (detached).
            use_gt = mx.random.uniform(shape=(B,)) < epsilon
            pred_hi = mx.argmax(mx.stop_gradient(hi_logits), axis=-1).astype(mx.int32)
            pred_lo = mx.argmax(mx.stop_gradient(lo_logits), axis=-1).astype(mx.int32)
            prev_hi = mx.where(use_gt, ht[:, step_t], pred_hi)
            prev_lo = mx.where(use_gt, lt[:, step_t], pred_lo)
            new_op = (prev_hi << 8) | prev_lo
            history = _update_history(history, step_t, new_op)

        return total_loss / mask.sum()

    # Eval on a fixed random subset to keep iteration time tractable.
    eval_subset_size = min(1024, n)
    np.random.seed(0)
    eval_idx = np.sort(np.random.choice(n, eval_subset_size, replace=False))
    Hev = Hm[mx.array(eval_idx)]
    Tev = Tm[mx.array(eval_idx)]
    HTev = HTm[mx.array(eval_idx)]
    LTev = LTm[mx.array(eval_idx)]
    Mev = Mm[mx.array(eval_idx)]
    n_ev = eval_subset_size

    def eval_accuracy(epsilon):
        """Evaluate at given ε (1.0 = teacher forcing, 0.0 = pure inference)
        on a fixed 1024-program subset."""
        chunk = 128
        correct_hi = correct_lo = total = 0
        for i in range(0, n_ev, chunk):
            h = Hev[i:i+chunk]
            t = Tev[i:i+chunk]
            ht = HTev[i:i+chunk]
            lt = LTev[i:i+chunk]
            mask = Mev[i:i+chunk]
            B = h.shape[0]
            h_state = mx.zeros((B, model.dim))
            prev_hi = mx.zeros((B,), dtype=mx.int32)
            prev_lo = mx.zeros((B,), dtype=mx.int32)
            history = mx.zeros((B, MAX_KV_LEN), dtype=mx.int32)
            for step_t in range(max_steps):
                hi_logits, lo_logits, h_state = model(
                    h, history, step_t + 1, t, prev_hi, prev_lo, h_state
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
                new_op = (prev_hi << 8) | prev_lo
                history = _update_history(history, step_t, new_op)
        return min(correct_hi / total, correct_lo / total)

    print(f"  Training with scheduled sampling (ε: 1.0 → 0.1)")

    for step in range(steps):
        epsilon = linear_epsilon(step, steps)
        idx = mx.array(np.random.choice(n, batch_size, replace=False))
        loss, grads = nn.value_and_grad(model, loss_fn)(
            model, Hm[idx], Tm[idx], HTm[idx], LTm[idx], Mm[idx], epsilon)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 500 == 0:
            tf_acc = eval_accuracy(1.0)
            inf_acc = eval_accuracy(0.0)
            print(f"  step {step:5d}  ε={epsilon:.2f}  loss={loss.item():.4f}  "
                  f"tf_acc={tf_acc:.1%}  inf_acc={inf_acc:.1%}", flush=True)
            # Save every 1000 steps once we have a non-trivially trained model
            # (anything > 50% inf_acc is more useful than no checkpoint at all).
            if step > 0 and step % 1000 == 0 and inf_acc > 0.5:
                mx.savez("weights.npz", **dict(tree_flatten(model.parameters())))
                print(f"  Checkpoint saved (inf_acc={inf_acc:.4%})", flush=True)
            if inf_acc >= 0.999:
                mx.savez("weights.npz", **dict(tree_flatten(model.parameters())))
                print(f"  Converged at inf_acc={inf_acc:.4%}", flush=True)
                perfect += 1
                if perfect >= 2:
                    return model
            else:
                perfect = 0

    return model


# ── Main ──────────────────────────────────────────────────────────────

B = "\033[1m"
D = "\033[2m"
N = "\033[0m"


def main():
    print(f"{B}Reflex — Training (program synthesis with control flow){N}\n")
    print(f"{D}Linear-emit, zero-state. GRU + scheduled sampling + token-ID pathway.{N}\n")

    backbone, tokenizer = load_backbone()

    print(f"\n{D}Generating tasks...{N}")
    tasks = generate_tasks()

    print(f"\n{D}Collecting sequences...{N}")
    t0 = time.time()
    instr_cache = load_or_encode(tasks, backbone, tokenizer)
    print(f"  {len(instr_cache)} unique instructions")
    H, T, HT, LT, M, max_steps = collect_sequences(tasks, instr_cache)
    print(f"  Collected in {time.time()-t0:.1f}s")

    print(f"\n{D}Training...{N}")
    t0 = time.time()
    model = train(H, T, HT, LT, M, max_steps, steps=15000)
    print(f"  Trained in {time.time()-t0:.1f}s")

    mx.savez("weights.npz", **dict(tree_flatten(model.parameters())))
    print(f"  Saved: weights.npz")
    print(f"  Run: uv run demo")


if __name__ == "__main__":
    main()
