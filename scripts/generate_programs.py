"""
Generate a large corpus of verified RV32I program families for training data.

Writes one JSON file per family to programs/<category>/<family>.json.
Each program is verified by running it through unicorn; rejected variants
are counted and reported.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

# Make repo importable
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from reflex.riscv import (  # noqa: E402
    DATA_BASE,
    add,
    addi,
    and_,
    andi,
    auipc,
    bge,
    beq,
    blt,
    bne,
    halt,
    jal,
    jalr,
    lb,
    lbu,
    lui,
    lw,
    or_,
    ori,
    pack,
    sb,
    sll,
    slli,
    slt,
    slti,
    srli,
    sub,
    sw,
    xor_,
    xori,
)
from reflex.programs import verify_program  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _addr(ops: list[int]) -> int:
    return 4 * len(ops)


def _load_addr(ops: list[int], rd: int, addr: int) -> None:
    """Materialise an absolute address into rd via LUI(+ADDI)."""
    hi = (addr + 0x800) >> 12
    lo = addr - (hi << 12)
    ops.append(lui(rd, hi & 0xFFFFF))
    if (lo & 0xFFF) != 0:
        ops.append(addi(rd, rd, lo))


def _load_imm(ops: list[int], rd: int, value: int) -> None:
    """Load a signed 32-bit immediate into rd.
    In [-2048, 2047] → single addi. Otherwise lui+addi with sign-fix."""
    v = value & 0xFFFFFFFF
    # Treat as signed
    s = v - (1 << 32) if v & 0x80000000 else v
    if -2048 <= s <= 2047:
        ops.append(addi(rd, 0, s))
        return
    hi = (v + 0x800) >> 12
    lo = v - (hi << 12)
    # sign-adjust lo to signed 12
    lo_s = lo & 0xFFF
    if lo_s & 0x800:
        lo_s = lo_s - 0x1000
    ops.append(lui(rd, hi & 0xFFFFF))
    if lo_s != 0:
        ops.append(addi(rd, rd, lo_s))


NUM_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 20: "twenty", 25: "twenty-five",
    30: "thirty", 42: "forty-two", 50: "fifty", 100: "one hundred",
}


def _w(n: int) -> str:
    if n < 0:
        return f"minus {_w(-n)}"
    return NUM_WORDS.get(n, str(n))


# ═══════════════════════════════════════════════════════════════════════
# Builders — grouped by category. Each family has 2-3 register layouts.
# ═══════════════════════════════════════════════════════════════════════

# ── arithmetic ──────────────────────────────────────────────────────

def build_add(a, b, layout):
    r1, r2, r3, rp = layout
    ops = []
    _load_imm(ops, r1, a)
    _load_imm(ops, r2, b)
    ops.append(add(r3, r1, r2))
    ops.append(lui(rp, DATA_BASE >> 12))
    ops.append(sw(r3, rp, 0))
    ops.append(halt())
    return pack(*ops)


def build_sub(a, b, layout):
    r1, r2, r3, rp = layout
    ops = []
    _load_imm(ops, r1, a)
    _load_imm(ops, r2, b)
    ops.append(sub(r3, r1, r2))
    ops.append(lui(rp, DATA_BASE >> 12))
    ops.append(sw(r3, rp, 0))
    ops.append(halt())
    return pack(*ops)


def build_mul_by_repeated_add(a, b, layout):
    """compute a * b via repeated addition of a, b times (b small, positive)."""
    ra, rb, acc, rp = layout
    ops = []
    _load_imm(ops, ra, a)
    _load_imm(ops, rb, b)
    ops.append(addi(acc, 0, 0))
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [
        add(acc, acc, ra),
        addi(rb, rb, -1),
    ]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(acc, rp, 0), halt()]
    ops[exit_idx] = beq(rb, 0, done - 4 * exit_idx)
    return pack(*ops)


def build_abs(a, layout):
    r1, r2, rp = layout
    ops = []
    _load_imm(ops, r1, a)
    # if r1 < 0 → negate
    ops.append(0); branch_idx = len(ops) - 1  # bge r1, x0, positive
    # else: r1 = 0 - r1
    ops.append(sub(r1, 0, r1))
    positive = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(r1, rp, 0), halt()]
    ops[branch_idx] = bge(r1, 0, positive - 4 * branch_idx)
    return pack(*ops)


def build_negate(a, layout):
    r1, r2, rp = layout
    ops = []
    _load_imm(ops, r1, a)
    ops.append(sub(r2, 0, r1))
    ops += [lui(rp, DATA_BASE >> 12), sw(r2, rp, 0), halt()]
    return pack(*ops)


# ── bitwise ──────────────────────────────────────────────────────────

def build_bitop(a, b, op_fn, layout):
    r1, r2, r3, rp = layout
    ops = []
    _load_imm(ops, r1, a)
    _load_imm(ops, r2, b)
    ops.append(op_fn(r3, r1, r2))
    ops += [lui(rp, DATA_BASE >> 12), sw(r3, rp, 0), halt()]
    return pack(*ops)


def build_shl(a, sh, layout):
    r1, r2, rp = layout
    ops = []
    _load_imm(ops, r1, a)
    ops.append(slli(r2, r1, sh))
    ops += [lui(rp, DATA_BASE >> 12), sw(r2, rp, 0), halt()]
    return pack(*ops)


def build_shr(a, sh, layout):
    r1, r2, rp = layout
    ops = []
    _load_imm(ops, r1, a)
    ops.append(srli(r2, r1, sh))
    ops += [lui(rp, DATA_BASE >> 12), sw(r2, rp, 0), halt()]
    return pack(*ops)


def build_popcount(a, layout):
    """count set bits in the low 32 using shift+and loop, 32 iterations."""
    rv, rc, rt, ri, rp = layout
    ops = []
    _load_imm(ops, rv, a)
    ops.append(addi(rc, 0, 0))        # count = 0
    ops.append(addi(ri, 0, 32))       # i = 32
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [
        andi(rt, rv, 1),
        add(rc, rc, rt),
        srli(rv, rv, 1),
        addi(ri, ri, -1),
    ]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(rc, rp, 0), halt()]
    ops[exit_idx] = beq(ri, 0, done - 4 * exit_idx)
    return pack(*ops)


def build_is_power_of_2(a, layout):
    """result = 1 if a > 0 and (a & (a-1)) == 0 else 0."""
    rv, rm, rt, rr, rp = layout
    ops = []
    _load_imm(ops, rv, a)
    ops.append(addi(rr, 0, 0))                # result = 0
    # if rv == 0 → skip
    ops.append(0); b_zero = len(ops) - 1
    ops.append(addi(rm, rv, -1))
    ops.append(and_(rt, rv, rm))
    # if rt != 0 → skip (not power)
    ops.append(0); b_notpow = len(ops) - 1
    ops.append(addi(rr, 0, 1))
    end = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(rr, rp, 0), halt()]
    ops[b_zero] = beq(rv, 0, end - 4 * b_zero)
    ops[b_notpow] = bne(rt, 0, end - 4 * b_notpow)
    return pack(*ops)


# ── comparison ──────────────────────────────────────────────────────

def build_min(a, b, layout):
    r1, r2, r3, rp = layout
    ops = []
    _load_imm(ops, r1, a)
    _load_imm(ops, r2, b)
    ops.append(0); br = len(ops) - 1           # blt r1,r2 → take r1 (r1 < r2)
    # else r3 = r2
    ops.append(add(r3, r2, 0))
    ops.append(0); skip = len(ops) - 1
    take_r1 = _addr(ops)
    ops.append(add(r3, r1, 0))
    end = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(r3, rp, 0), halt()]
    ops[br] = blt(r1, r2, take_r1 - 4 * br)
    ops[skip] = jal(0, end - 4 * skip)
    return pack(*ops)


def build_max(a, b, layout):
    r1, r2, r3, rp = layout
    ops = []
    _load_imm(ops, r1, a)
    _load_imm(ops, r2, b)
    ops.append(0); br = len(ops) - 1           # bge r1,r2 → take r1
    ops.append(add(r3, r2, 0))
    ops.append(0); skip = len(ops) - 1
    take_r1 = _addr(ops)
    ops.append(add(r3, r1, 0))
    end = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(r3, rp, 0), halt()]
    ops[br] = bge(r1, r2, take_r1 - 4 * br)
    ops[skip] = jal(0, end - 4 * skip)
    return pack(*ops)


def build_sign(a, layout):
    """result = 1 if a>0, -1 if a<0, 0 if a==0."""
    r1, rr, rp = layout
    ops = []
    _load_imm(ops, r1, a)
    ops.append(addi(rr, 0, 0))
    # if r1 == 0 → end
    ops.append(0); b_zero = len(ops) - 1
    # if r1 < 0 → rr = -1, jump end
    ops.append(0); b_neg = len(ops) - 1
    ops.append(addi(rr, 0, 1))
    ops.append(0); skip = len(ops) - 1
    neg_ = _addr(ops)
    ops.append(addi(rr, 0, -1))
    end = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(rr, rp, 0), halt()]
    ops[b_zero] = beq(r1, 0, end - 4 * b_zero)
    ops[b_neg] = blt(r1, 0, neg_ - 4 * b_neg)
    ops[skip] = jal(0, end - 4 * skip)
    return pack(*ops)


def build_equal(a, b, layout):
    r1, r2, rr, rp = layout
    ops = []
    _load_imm(ops, r1, a)
    _load_imm(ops, r2, b)
    ops.append(addi(rr, 0, 0))
    ops.append(0); br = len(ops) - 1
    ops.append(addi(rr, 0, 1))
    end = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(rr, rp, 0), halt()]
    ops[br] = bne(r1, r2, end - 4 * br)
    return pack(*ops)


def build_not_equal(a, b, layout):
    r1, r2, rr, rp = layout
    ops = []
    _load_imm(ops, r1, a)
    _load_imm(ops, r2, b)
    ops.append(addi(rr, 0, 1))
    ops.append(0); br = len(ops) - 1
    ops.append(addi(rr, 0, 0))
    end = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(rr, rp, 0), halt()]
    ops[br] = bne(r1, r2, end - 4 * br)
    return pack(*ops)


def build_clamp(x, lo, hi, layout):
    """clamp x to [lo, hi]."""
    rx, rlo, rhi, rp = layout
    ops = []
    _load_imm(ops, rx, x)
    _load_imm(ops, rlo, lo)
    _load_imm(ops, rhi, hi)
    # if rx < rlo → rx = rlo
    ops.append(0); b1 = len(ops) - 1
    ops.append(add(rx, rlo, 0))
    after_lo = _addr(ops)
    # if rx > rhi (rhi < rx) → rx = rhi
    ops.append(0); b2 = len(ops) - 1
    ops.append(add(rx, rhi, 0))
    after_hi = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(rx, rp, 0), halt()]
    ops[b1] = bge(rx, rlo, after_lo - 4 * b1)
    ops[b2] = bge(rhi, rx, after_hi - 4 * b2)
    return pack(*ops)


# ── loops ───────────────────────────────────────────────────────────

def build_count_up(n, layout):
    """store 1,2,...,n at DATA_BASE."""
    ri, rp, rn = layout
    ops = [addi(ri, 0, 1), addi(rn, 0, n + 1)]
    ops.append(lui(rp, DATA_BASE >> 12))
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [
        sw(ri, rp, 0),
        addi(rp, rp, 4),
        addi(ri, ri, 1),
    ]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops.append(halt())
    ops[exit_idx] = beq(ri, rn, done - 4 * exit_idx)
    return pack(*ops)


def build_count_down(n, layout):
    """store n, n-1,...,1 at DATA_BASE."""
    rv, rp = layout
    ops = [addi(rv, 0, n), lui(rp, DATA_BASE >> 12)]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [sw(rv, rp, 0), addi(rp, rp, 4), addi(rv, rv, -1)]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops.append(halt())
    ops[exit_idx] = beq(rv, 0, done - 4 * exit_idx)
    return pack(*ops)


def build_sum_range(lo, hi, layout):
    """sum from lo..hi inclusive."""
    rs, ri, rh, rp = layout
    ops = []
    ops.append(addi(rs, 0, 0))
    ops.append(addi(ri, 0, lo))
    ops.append(addi(rh, 0, hi + 1))
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [add(rs, rs, ri), addi(ri, ri, 1)]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(rs, rp, 0), halt()]
    ops[exit_idx] = beq(ri, rh, done - 4 * exit_idx)
    return pack(*ops)


def build_sum_evens(n, layout):
    """sum of even numbers 2,4,...,2n."""
    rs, ri, rh, rp = layout
    ops = [addi(rs, 0, 0), addi(ri, 0, 2), addi(rh, 0, 2 * n + 2)]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [add(rs, rs, ri), addi(ri, ri, 2)]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(rs, rp, 0), halt()]
    ops[exit_idx] = beq(ri, rh, done - 4 * exit_idx)
    return pack(*ops)


def build_sum_odds(n, layout):
    """sum of odd numbers 1,3,...,2n-1."""
    rs, ri, rh, rp = layout
    ops = [addi(rs, 0, 0), addi(ri, 0, 1), addi(rh, 0, 2 * n + 1)]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [add(rs, rs, ri), addi(ri, ri, 2)]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(rs, rp, 0), halt()]
    ops[exit_idx] = beq(ri, rh, done - 4 * exit_idx)
    return pack(*ops)


def build_product_range_small(lo, hi, layout):
    """product lo..hi via nested repeated add."""
    # Use initial acc=1, then for i in lo..hi: acc = acc * i (repeated add)
    racc, ri, rh, rtmp, rctr, rp = layout
    ops = [
        addi(racc, 0, 1),
        addi(ri, 0, lo),
        addi(rh, 0, hi + 1),
    ]
    outer = _addr(ops)
    ops.append(0); outer_exit = len(ops) - 1
    # tmp = 0; ctr = 0
    ops += [addi(rtmp, 0, 0), addi(rctr, 0, 0)]
    inner = _addr(ops)
    ops.append(0); inner_exit = len(ops) - 1
    ops += [add(rtmp, rtmp, racc), addi(rctr, rctr, 1)]
    ops.append(jal(0, inner - _addr(ops)))
    inner_done = _addr(ops)
    ops += [add(racc, rtmp, 0), addi(ri, ri, 1)]
    ops.append(jal(0, outer - _addr(ops)))
    done = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(racc, rp, 0), halt()]
    ops[outer_exit] = beq(ri, rh, done - 4 * outer_exit)
    ops[inner_exit] = beq(rctr, ri, inner_done - 4 * inner_exit)
    return pack(*ops)


# ── arrays ──────────────────────────────────────────────────────────
# Use DATA_BASE+0x100 as input array, DATA_BASE as output single/other.

ARR_OFFSET = 0x100
DST_OFFSET = 0x200


def build_sum_array(n, layout):
    rp, rs, rn, ri, rt = layout
    ops = []
    _load_addr(ops, rp, DATA_BASE + ARR_OFFSET)
    ops += [addi(rs, 0, 0), addi(ri, 0, 0), addi(rn, 0, n)]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [lw(rt, rp, 0), add(rs, rs, rt), addi(rp, rp, 4), addi(ri, ri, 1)]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    # store sum at DATA_BASE
    ops.append(lui(ri, DATA_BASE >> 12))
    ops += [sw(rs, ri, 0), halt()]
    ops[exit_idx] = beq(ri, rn, done - 4 * exit_idx)
    return pack(*ops)


def build_max_array(n, layout):
    rp, rm, rn, ri, rt = layout
    ops = []
    _load_addr(ops, rp, DATA_BASE + ARR_OFFSET)
    ops.append(lw(rm, rp, 0))
    ops += [addi(ri, 0, 1), addi(rn, 0, n), addi(rp, rp, 4)]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops.append(lw(rt, rp, 0))
    ops.append(0); b_skip = len(ops) - 1  # if rm >= rt skip update
    ops.append(add(rm, rt, 0))
    after = _addr(ops)
    ops += [addi(rp, rp, 4), addi(ri, ri, 1)]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops.append(lui(ri, DATA_BASE >> 12))
    ops += [sw(rm, ri, 0), halt()]
    ops[exit_idx] = beq(ri, rn, done - 4 * exit_idx)
    ops[b_skip] = bge(rm, rt, after - 4 * b_skip)
    return pack(*ops)


def build_min_array(n, layout):
    rp, rm, rn, ri, rt = layout
    ops = []
    _load_addr(ops, rp, DATA_BASE + ARR_OFFSET)
    ops.append(lw(rm, rp, 0))
    ops += [addi(ri, 0, 1), addi(rn, 0, n), addi(rp, rp, 4)]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops.append(lw(rt, rp, 0))
    ops.append(0); b_skip = len(ops) - 1  # if rt >= rm skip update
    ops.append(add(rm, rt, 0))
    after = _addr(ops)
    ops += [addi(rp, rp, 4), addi(ri, ri, 1)]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops.append(lui(ri, DATA_BASE >> 12))
    ops += [sw(rm, ri, 0), halt()]
    ops[exit_idx] = beq(ri, rn, done - 4 * exit_idx)
    ops[b_skip] = bge(rt, rm, after - 4 * b_skip)
    return pack(*ops)


def build_reverse_array(n, layout):
    """reverse src array (DATA_BASE+ARR_OFFSET) into dst (DATA_BASE+DST_OFFSET)."""
    rsrc, rdst, ri, rn, rt = layout
    ops = []
    _load_addr(ops, rsrc, DATA_BASE + ARR_OFFSET)
    _load_addr(ops, rdst, DATA_BASE + DST_OFFSET + 4 * (n - 1))
    ops += [addi(ri, 0, 0), addi(rn, 0, n)]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [lw(rt, rsrc, 0), sw(rt, rdst, 0),
            addi(rsrc, rsrc, 4), addi(rdst, rdst, -4), addi(ri, ri, 1)]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops.append(halt())
    ops[exit_idx] = beq(ri, rn, done - 4 * exit_idx)
    return pack(*ops)


def build_copy_array(n, layout):
    rsrc, rdst, ri, rn, rt = layout
    ops = []
    _load_addr(ops, rsrc, DATA_BASE + ARR_OFFSET)
    _load_addr(ops, rdst, DATA_BASE + DST_OFFSET)
    ops += [addi(ri, 0, 0), addi(rn, 0, n)]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [lw(rt, rsrc, 0), sw(rt, rdst, 0),
            addi(rsrc, rsrc, 4), addi(rdst, rdst, 4), addi(ri, ri, 1)]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops.append(halt())
    ops[exit_idx] = beq(ri, rn, done - 4 * exit_idx)
    return pack(*ops)


def build_fill_array(n, value, layout):
    rp, rv, ri, rn = layout
    ops = []
    _load_addr(ops, rp, DATA_BASE + ARR_OFFSET)
    _load_imm(ops, rv, value)
    ops += [addi(ri, 0, 0), addi(rn, 0, n)]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [sw(rv, rp, 0), addi(rp, rp, 4), addi(ri, ri, 1)]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops.append(halt())
    ops[exit_idx] = beq(ri, rn, done - 4 * exit_idx)
    return pack(*ops)


def build_find_element(n, target, layout):
    """Find first index of target in array. -1 if not found (stored as 0xFFFFFFFF).
    Store at DATA_BASE."""
    rp, rt, ri, rn, rv, rres = layout
    ops = []
    _load_addr(ops, rp, DATA_BASE + ARR_OFFSET)
    _load_imm(ops, rt, target)
    ops += [addi(ri, 0, 0), addi(rn, 0, n), addi(rres, 0, -1)]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops.append(lw(rv, rp, 0))
    # if rv == rt → set res=ri, jump done
    ops.append(0); b_found = len(ops) - 1
    ops += [addi(rp, rp, 4), addi(ri, ri, 1)]
    ops.append(jal(0, loop - _addr(ops)))
    # found branch: set res and fall through to done
    found_tgt = _addr(ops)
    ops.append(add(rres, ri, 0))
    done = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(rres, rp, 0), halt()]
    ops[exit_idx] = beq(ri, rn, done - 4 * exit_idx)
    ops[b_found] = beq(rv, rt, found_tgt - 4 * b_found)
    return pack(*ops)


def build_count_occurrences(n, target, layout):
    rp, rt, ri, rn, rv, rc = layout
    ops = []
    _load_addr(ops, rp, DATA_BASE + ARR_OFFSET)
    _load_imm(ops, rt, target)
    ops += [addi(ri, 0, 0), addi(rn, 0, n), addi(rc, 0, 0)]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops.append(lw(rv, rp, 0))
    ops.append(0); b_neq = len(ops) - 1  # if rv != rt skip inc
    ops.append(addi(rc, rc, 1))
    after = _addr(ops)
    ops += [addi(rp, rp, 4), addi(ri, ri, 1)]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(rc, rp, 0), halt()]
    ops[exit_idx] = beq(ri, rn, done - 4 * exit_idx)
    ops[b_neq] = bne(rv, rt, after - 4 * b_neq)
    return pack(*ops)


# ── math ────────────────────────────────────────────────────────────

def build_factorial(n, layout):
    rn, racc, ri, rtmp, rj, rp = layout
    ops = [addi(rn, 0, n), addi(racc, 0, 1), addi(ri, 0, 2)]
    outer = _addr(ops)
    ops.append(0); outer_idx = len(ops) - 1
    ops += [addi(rtmp, 0, 0), addi(rj, 0, 0)]
    inner = _addr(ops)
    ops.append(0); inner_idx = len(ops) - 1
    ops += [add(rtmp, rtmp, racc), addi(rj, rj, 1)]
    ops.append(jal(0, inner - _addr(ops)))
    inner_done = _addr(ops)
    ops += [add(racc, rtmp, 0), addi(ri, ri, 1)]
    ops.append(jal(0, outer - _addr(ops)))
    done = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(racc, rp, 0), halt()]
    ops[outer_idx] = blt(rn, ri, done - 4 * outer_idx)
    ops[inner_idx] = bge(rj, ri, inner_done - 4 * inner_idx)
    return pack(*ops)


def build_fibonacci(n, layout):
    ra, rb, rcnt, rtmp, rp = layout
    ops = [addi(ra, 0, 0), addi(rb, 0, 1), addi(rcnt, 0, n),
           lui(rp, DATA_BASE >> 12)]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [sw(ra, rp, 0), add(rtmp, ra, rb), add(ra, rb, 0),
            add(rb, rtmp, 0), addi(rp, rp, 4), addi(rcnt, rcnt, -1)]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops.append(halt())
    ops[exit_idx] = beq(rcnt, 0, done - 4 * exit_idx)
    return pack(*ops)


def build_gcd(a, b, layout):
    """Euclidean gcd via repeated subtraction."""
    ra, rb, rp = layout
    ops = []
    _load_imm(ops, ra, a)
    _load_imm(ops, rb, b)
    loop = _addr(ops)
    # if ra == rb → done
    ops.append(0); b_done = len(ops) - 1
    # if ra < rb → rb -= ra, else ra -= rb
    ops.append(0); b_lt = len(ops) - 1
    ops.append(sub(ra, ra, rb))
    ops.append(jal(0, loop - _addr(ops)))
    branch_lt = _addr(ops)
    ops.append(sub(rb, rb, ra))
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(ra, rp, 0), halt()]
    ops[b_done] = beq(ra, rb, done - 4 * b_done)
    ops[b_lt] = blt(ra, rb, branch_lt - 4 * b_lt)
    return pack(*ops)


def build_power(base, exp, layout):
    """base^exp via repeated multiplication (repeated add inner)."""
    rbase, rexp, racc, rtmp, rj, rp = layout
    ops = []
    _load_imm(ops, rbase, base)
    _load_imm(ops, rexp, exp)
    ops.append(addi(racc, 0, 1))
    outer = _addr(ops)
    ops.append(0); outer_idx = len(ops) - 1
    # multiply racc *= rbase via repeated add (rbase times)
    ops += [addi(rtmp, 0, 0), addi(rj, 0, 0)]
    inner = _addr(ops)
    ops.append(0); inner_idx = len(ops) - 1
    ops += [add(rtmp, rtmp, racc), addi(rj, rj, 1)]
    ops.append(jal(0, inner - _addr(ops)))
    inner_done = _addr(ops)
    ops += [add(racc, rtmp, 0), addi(rexp, rexp, -1)]
    ops.append(jal(0, outer - _addr(ops)))
    done = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(racc, rp, 0), halt()]
    ops[outer_idx] = beq(rexp, 0, done - 4 * outer_idx)
    ops[inner_idx] = beq(rj, rbase, inner_done - 4 * inner_idx)
    return pack(*ops)


def build_triangular(n, layout):
    rs, ri, rn, rp = layout
    ops = [addi(rs, 0, 0), addi(ri, 0, 1), addi(rn, 0, n + 1)]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [add(rs, rs, ri), addi(ri, ri, 1)]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(rs, rp, 0), halt()]
    ops[exit_idx] = beq(ri, rn, done - 4 * exit_idx)
    return pack(*ops)


def build_isqrt(n, layout):
    """integer square root: largest i such that i*i <= n."""
    rn, ri, rsq, rtmp, rj, rp = layout
    ops = [addi(ri, 0, 0)]
    _load_imm(ops, rn, n)
    # loop: compute (ri+1)^2 via repeated add; if > n, break; else ri++
    outer = _addr(ops)
    # square = 0; j = 0; next = ri + 1
    ops += [addi(rsq, 0, 0), addi(rj, 0, 0), addi(rtmp, ri, 1)]
    # inner: repeat (ri+1) times, add rtmp to rsq
    inner = _addr(ops)
    ops.append(0); inner_idx = len(ops) - 1
    ops += [add(rsq, rsq, rtmp), addi(rj, rj, 1)]
    ops.append(jal(0, inner - _addr(ops)))
    inner_done = _addr(ops)
    # if rsq > rn → done, else ri++ and continue
    ops.append(0); b_done = len(ops) - 1  # blt rn, rsq → done
    ops.append(addi(ri, ri, 1))
    ops.append(jal(0, outer - _addr(ops)))
    done = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(ri, rp, 0), halt()]
    ops[inner_idx] = beq(rj, rtmp, inner_done - 4 * inner_idx)
    ops[b_done] = blt(rn, rsq, done - 4 * b_done)
    return pack(*ops)


# ── strings ────────────────────────────────────────────────────────
# We seed a byte-string at DATA_BASE+ARR_OFFSET terminated by '\0'.
# seed values are u32 words; we pack 4 chars per word LE.

def pack_string_seed(s: str) -> list[int]:
    """Pack s + null terminator into u32 words, little-endian. Pads with 0."""
    data = s.encode("ascii") + b"\x00"
    while len(data) % 4 != 0:
        data += b"\x00"
    return [int.from_bytes(data[i:i+4], "little") for i in range(0, len(data), 4)]


def build_strlen(layout):
    rp, rc, rch = layout
    ops = []
    _load_addr(ops, rp, DATA_BASE + ARR_OFFSET)
    ops.append(addi(rc, 0, 0))
    loop = _addr(ops)
    ops.append(lbu(rch, rp, 0))
    # if rch == 0 → done
    ops.append(0); b_done = len(ops) - 1
    ops += [addi(rp, rp, 1), addi(rc, rc, 1)]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops.append(lui(rp, DATA_BASE >> 12))
    ops += [sw(rc, rp, 0), halt()]
    ops[b_done] = beq(rch, 0, done - 4 * b_done)
    return pack(*ops)


def build_count_char(ch, layout):
    rp, rc, rch, rt = layout
    ops = []
    _load_addr(ops, rp, DATA_BASE + ARR_OFFSET)
    _load_imm(ops, rt, ch)
    ops.append(addi(rc, 0, 0))
    loop = _addr(ops)
    ops.append(lbu(rch, rp, 0))
    # if rch == 0 → done
    ops.append(0); b_done = len(ops) - 1
    # if rch != rt skip inc
    ops.append(0); b_ne = len(ops) - 1
    ops.append(addi(rc, rc, 1))
    after = _addr(ops)
    ops.append(addi(rp, rp, 1))
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops.append(lui(rp, DATA_BASE >> 12))
    ops += [sw(rc, rp, 0), halt()]
    ops[b_done] = beq(rch, 0, done - 4 * b_done)
    ops[b_ne] = bne(rch, rt, after - 4 * b_ne)
    return pack(*ops)


# ── memory ─────────────────────────────────────────────────────────

def build_memcpy(n, layout):
    rsrc, rdst, ri, rn, rt = layout
    ops = []
    _load_addr(ops, rsrc, DATA_BASE + ARR_OFFSET)
    _load_addr(ops, rdst, DATA_BASE + DST_OFFSET)
    ops += [addi(ri, 0, 0), addi(rn, 0, n)]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [lw(rt, rsrc, 0), sw(rt, rdst, 0),
            addi(rsrc, rsrc, 4), addi(rdst, rdst, 4), addi(ri, ri, 1)]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops.append(halt())
    ops[exit_idx] = beq(ri, rn, done - 4 * exit_idx)
    return pack(*ops)


def build_memset(n, value, layout):
    rp, rv, ri, rn = layout
    ops = []
    _load_addr(ops, rp, DATA_BASE + ARR_OFFSET)
    _load_imm(ops, rv, value)
    ops += [addi(ri, 0, 0), addi(rn, 0, n)]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [sw(rv, rp, 0), addi(rp, rp, 4), addi(ri, ri, 1)]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops.append(halt())
    ops[exit_idx] = beq(ri, rn, done - 4 * exit_idx)
    return pack(*ops)


def build_swap_two_values(a, b, layout):
    """Store a at DATA_BASE, b at DATA_BASE+4, then swap them."""
    ra, rb, rtmp, rp = layout
    ops = []
    _load_imm(ops, ra, a)
    _load_imm(ops, rb, b)
    ops.append(lui(rp, DATA_BASE >> 12))
    ops += [sw(ra, rp, 0), sw(rb, rp, 4)]
    # swap
    ops += [lw(ra, rp, 0), lw(rb, rp, 4), add(rtmp, ra, 0),
            sw(rb, rp, 0), sw(rtmp, rp, 4), halt()]
    return pack(*ops)


# ── display ─────────────────────────────────────────────────────────

def build_write_digits(n, layout):
    """store decimal digits of N as separate words (most significant first).
    Extract digits by repeated sub of 10."""
    rn, rten, rd, rq, rcnt, rp = layout
    # We need a buffer. Compute digits by extracting mod 10 (via sub-loop)
    # and pushing onto stack of 6 slots max; then reverse into DATA_BASE.
    # Keep it simple: handle small n (<= 999) with at most 3 digits, use
    # scratch area at DATA_BASE+0x300.
    ops = []
    _load_imm(ops, rn, n)
    ops.append(addi(rten, 0, 10))
    ops.append(addi(rcnt, 0, 0))
    # scratch buffer grows at DATA_BASE+0x300
    _load_addr(ops, rp, DATA_BASE + 0x300)
    # extract digits: do { d = rn % 10; sw d; rn /= 10; cnt++ } while rn > 0
    extract = _addr(ops)
    # compute q = rn / 10 via repeated sub; d = rn - q*10... repeated-add is long.
    # Easier: repeatedly subtract 10 until rn < 10 → that's last digit; count each sub
    # But we need digit per iteration. Use: q = 0; while rn >= 10: rn -= 10; q++; d = rn; rn = q.
    ops.append(addi(rq, 0, 0))
    div_loop = _addr(ops)
    ops.append(0); b_end_div = len(ops) - 1  # blt rn, rten → end
    ops += [sub(rn, rn, rten), addi(rq, rq, 1)]
    ops.append(jal(0, div_loop - _addr(ops)))
    div_done = _addr(ops)
    # rn now holds digit, rq holds quotient
    ops += [sw(rn, rp, 0), addi(rp, rp, 4), addi(rcnt, rcnt, 1),
            add(rn, rq, 0)]
    # if rn != 0 → loop extract
    ops.append(0); b_more = len(ops) - 1
    # Now we have rcnt digits stored at DATA_BASE+0x300..+0x300+4*(cnt-1)
    # reverse them into DATA_BASE
    # rp is currently end+4 (one past last written). Move back 4 at a time.
    # Use ri = 0; dst ptr = DATA_BASE; src ptr = DATA_BASE+0x300 + 4*(cnt-1)
    # compute source end: rp -= 4
    ops.append(addi(rp, rp, -4))
    # dst ptr in rten (reuse register)
    _load_addr(ops, rten, DATA_BASE)
    ops.append(addi(rq, 0, 0))   # i
    rev = _addr(ops)
    ops.append(0); b_rev_done = len(ops) - 1  # beq rq, rcnt → done
    ops += [lw(rn, rp, 0), sw(rn, rten, 0),
            addi(rp, rp, -4), addi(rten, rten, 4), addi(rq, rq, 1)]
    ops.append(jal(0, rev - _addr(ops)))
    done = _addr(ops)
    ops.append(halt())
    ops[b_end_div] = blt(rn, rten, div_done - 4 * b_end_div)
    ops[b_more] = bne(rn, 0, extract - 4 * b_more)
    ops[b_rev_done] = beq(rq, rcnt, done - 4 * b_rev_done)
    return pack(*ops)


def build_write_ascii_text(text: str, layout):
    """Store `text` bytes at DATA_BASE using sb."""
    rp, rc = layout
    ops = []
    _load_addr(ops, rp, DATA_BASE)
    for i, ch in enumerate(text):
        ops.append(addi(rc, 0, ord(ch)))
        ops.append(sb(rc, rp, i))
    # null terminator
    ops.append(sb(0, rp, len(text)))
    ops.append(halt())
    return pack(*ops)


# ── control_flow ───────────────────────────────────────────────────

def build_if_else_max(a, b, layout):
    # Same as max but styled as if/else
    return build_max(a, b, layout)


def build_nested_if_sign(a, layout):
    return build_sign(a, layout)


def build_switch_like_grade(score, layout):
    """90+→4, 80+→3, 70+→2, 60+→1, else 0."""
    rs, rt, rr, rp = layout
    ops = []
    _load_imm(ops, rs, score)
    ops.append(addi(rr, 0, 0))  # default 0
    ops.append(addi(rt, 0, 90))
    ops.append(0); b90 = len(ops) - 1   # if rs < 90 → skip
    ops.append(addi(rr, 0, 4))
    ops.append(0); j_end1 = len(ops) - 1
    after90 = _addr(ops)
    ops.append(addi(rt, 0, 80))
    ops.append(0); b80 = len(ops) - 1
    ops.append(addi(rr, 0, 3))
    ops.append(0); j_end2 = len(ops) - 1
    after80 = _addr(ops)
    ops.append(addi(rt, 0, 70))
    ops.append(0); b70 = len(ops) - 1
    ops.append(addi(rr, 0, 2))
    ops.append(0); j_end3 = len(ops) - 1
    after70 = _addr(ops)
    ops.append(addi(rt, 0, 60))
    ops.append(0); b60 = len(ops) - 1
    ops.append(addi(rr, 0, 1))
    after60 = _addr(ops)
    end = _addr(ops)
    ops += [lui(rp, DATA_BASE >> 12), sw(rr, rp, 0), halt()]
    ops[b90] = blt(rs, rt, after90 - 4 * b90)
    ops[j_end1] = jal(0, end - 4 * j_end1)
    ops[b80] = blt(rs, rt, after80 - 4 * b80)
    ops[j_end2] = jal(0, end - 4 * j_end2)
    ops[b70] = blt(rs, rt, after70 - 4 * b70)
    ops[j_end3] = jal(0, end - 4 * j_end3)
    ops[b60] = blt(rs, rt, after60 - 4 * b60)
    return pack(*ops)


def build_early_return_zero(a, layout):
    """If a == 0, store 0 and halt early; else store a*2."""
    r1, r2, rp = layout
    ops = []
    _load_imm(ops, r1, a)
    ops.append(lui(rp, DATA_BASE >> 12))
    # if a == 0 → store 0, halt
    ops.append(0); b_zero = len(ops) - 1
    # compute 2a
    ops.append(add(r2, r1, r1))
    ops.append(sw(r2, rp, 0))
    ops.append(halt())
    zero_branch = _addr(ops)
    ops.append(sw(0, rp, 0))
    ops.append(halt())
    ops[b_zero] = beq(r1, 0, zero_branch - 4 * b_zero)
    return pack(*ops)


# ── function ────────────────────────────────────────────────────────

def build_double_fn(n, layout):
    ra0, rp, ra = layout
    ops = [addi(ra0, 0, n)]
    call_idx = len(ops); ops.append(0)
    ops += [lui(rp, DATA_BASE >> 12), sw(ra0, rp, 0), halt()]
    fn = _addr(ops)
    ops += [add(ra0, ra0, ra0), jalr(0, ra, 0)]
    ops[call_idx] = jal(ra, fn - 4 * call_idx)
    return pack(*ops)


def build_triple_fn(n, layout):
    ra0, rp, ra, rt = layout
    ops = [addi(ra0, 0, n)]
    call_idx = len(ops); ops.append(0)
    ops += [lui(rp, DATA_BASE >> 12), sw(ra0, rp, 0), halt()]
    fn = _addr(ops)
    ops += [add(rt, ra0, ra0), add(ra0, rt, ra0), jalr(0, ra, 0)]
    ops[call_idx] = jal(ra, fn - 4 * call_idx)
    return pack(*ops)


def build_abs_fn(n, layout):
    ra0, rp, ra = layout
    ops = []
    _load_imm(ops, ra0, n)
    call_idx = len(ops); ops.append(0)
    ops += [lui(rp, DATA_BASE >> 12), sw(ra0, rp, 0), halt()]
    fn = _addr(ops)
    # if ra0 >= 0 → return
    ops.append(0); b_pos = len(ops) - 1
    ops.append(sub(ra0, 0, ra0))
    ret = _addr(ops)
    ops.append(jalr(0, ra, 0))
    ops[call_idx] = jal(ra, fn - 4 * call_idx)
    ops[b_pos] = bge(ra0, 0, ret - 4 * b_pos)
    return pack(*ops)


# ── sorting ─────────────────────────────────────────────────────────

def build_bubble_sort_small(n, layout):
    """Bubble sort n elements in place at DATA_BASE+ARR_OFFSET.
    Seed the array; sorts in-place. n is 3 or 4."""
    rp, ri, rj, ra, rb, rlim = layout
    ops = []
    # outer: for i in 0..n-1
    ops.append(addi(ri, 0, 0))
    outer = _addr(ops)
    ops.append(0); outer_exit = len(ops) - 1   # beq ri, n → done
    # inner: for j in 0..n-1-i
    _load_addr(ops, rp, DATA_BASE + ARR_OFFSET)
    ops.append(addi(rj, 0, 0))
    # compute limit = n - 1 - i
    ops.append(addi(rlim, 0, n - 1))
    ops.append(sub(rlim, rlim, ri))
    inner = _addr(ops)
    ops.append(0); inner_exit = len(ops) - 1
    ops.append(lw(ra, rp, 0))
    ops.append(lw(rb, rp, 4))
    # if ra <= rb → skip
    ops.append(0); b_skip = len(ops) - 1  # bge rb, ra → skip (rb >= ra)
    # swap
    ops += [sw(rb, rp, 0), sw(ra, rp, 4)]
    after = _addr(ops)
    ops += [addi(rp, rp, 4), addi(rj, rj, 1)]
    ops.append(jal(0, inner - _addr(ops)))
    inner_done = _addr(ops)
    ops.append(addi(ri, ri, 1))
    ops.append(jal(0, outer - _addr(ops)))
    done = _addr(ops)
    ops.append(halt())
    # outer_exit: when ri == n → done
    # use rlim-relative compare: load constant n into unused reg? easier: compare with addi result
    # Simplest: precompute n in a free reg. We'll reuse rlim by inserting a setup -- but rlim is used.
    # Use rb for comparison target since we can reload.
    # Actually ri vs (reg holding n). We need a persistent reg. Allocate fresh by overloading: use 'rb'.
    # Need to set rb=n before exit test. Do this at outer start.
    # We'll insert addi(rb,0,n) at loop head before the beq, reusing rb (gets clobbered inside, that's fine).
    # Replace ops[outer_exit] to compare ri vs rb after setting rb=n.
    # But rb is set inside the inner; by the time we return to outer, it's clobbered but we reset at outer head.
    # Restructure: we already have space — let's just hack by setting rb=n before the test.
    # But we already placed beq at outer_exit position referring to ri, rb. Need to set rb=n first.
    # Rewrite by patching: use rlim temporarily — rlim is also clobbered; fine, we set it again below.
    # Final: set rlim=n at outer head. Compare ri with rlim.
    # This doesn't work because of the existing insertions. Let me just rebuild cleanly:
    return _build_bubble_sort_small_clean(n, layout)


def _build_bubble_sort_small_clean(n, layout):
    rp, ri, rj, ra, rb, rlim = layout
    ops = []
    ops.append(addi(ri, 0, 0))
    outer = _addr(ops)
    # test ri == n: use rb as scratch
    ops.append(addi(rb, 0, n))
    ops.append(0); outer_exit = len(ops) - 1   # beq ri, rb → done
    # inner setup
    _load_addr(ops, rp, DATA_BASE + ARR_OFFSET)
    ops.append(addi(rj, 0, 0))
    ops.append(addi(rlim, 0, n - 1))
    ops.append(sub(rlim, rlim, ri))
    inner = _addr(ops)
    # test rj == rlim
    ops.append(0); inner_exit = len(ops) - 1
    ops.append(lw(ra, rp, 0))
    ops.append(lw(rb, rp, 4))
    ops.append(0); b_skip = len(ops) - 1  # bge rb, ra → skip swap
    ops += [sw(rb, rp, 0), sw(ra, rp, 4)]
    after = _addr(ops)
    ops += [addi(rp, rp, 4), addi(rj, rj, 1)]
    ops.append(jal(0, inner - _addr(ops)))
    inner_done = _addr(ops)
    ops.append(addi(ri, ri, 1))
    ops.append(jal(0, outer - _addr(ops)))
    done = _addr(ops)
    ops.append(halt())
    ops[outer_exit] = beq(ri, rb, done - 4 * outer_exit)
    ops[inner_exit] = beq(rj, rlim, inner_done - 4 * inner_exit)
    ops[b_skip] = bge(rb, ra, after - 4 * b_skip)
    return pack(*ops)


# ── stack ───────────────────────────────────────────────────────────

def build_push_pop_sequence(values, layout):
    """Push values onto a stack (DATA_BASE+0x300 growing up), then pop to DATA_BASE.
    Result: memory at DATA_BASE..+4*n holds values reversed."""
    rsp, rt, rp = layout
    ops = []
    _load_addr(ops, rsp, DATA_BASE + 0x300)
    # push each value
    for v in values:
        _load_imm(ops, rt, v)
        ops.append(sw(rt, rsp, 0))
        ops.append(addi(rsp, rsp, 4))
    # pop in reverse into DATA_BASE..
    ops.append(lui(rp, DATA_BASE >> 12))
    for i in range(len(values)):
        ops.append(addi(rsp, rsp, -4))
        ops.append(lw(rt, rsp, 0))
        ops.append(sw(rt, rp, 0))
        ops.append(addi(rp, rp, 4))
    ops.append(halt())
    return pack(*ops)


# ═══════════════════════════════════════════════════════════════════════
# Phrasings
# ═══════════════════════════════════════════════════════════════════════

def ph_add(a, b):
    return [f"add {a} and {b}", f"compute {a} + {b}", f"{a} plus {b}",
            f"{_w(a)} plus {_w(b)}", f"sum of {a} and {b}", f"{a}+{b}",
            f"store {a} + {b}", f"save {a}+{b}"]


def ph_sub(a, b):
    return [f"subtract {b} from {a}", f"compute {a} - {b}", f"{a} minus {b}",
            f"{_w(a)} minus {_w(b)}", f"{a}-{b}", f"store {a} - {b}",
            f"difference of {a} and {b}", f"{a} take away {b}"]


def ph_mul(a, b):
    return [f"multiply {a} by {b}", f"compute {a} * {b}", f"{a} times {b}",
            f"{_w(a)} times {_w(b)}", f"product of {a} and {b}",
            f"{a}*{b}", f"store {a} * {b}", f"calculate {a} x {b}"]


def ph_abs(a):
    return [f"absolute value of {a}", f"|{a}|", f"abs({a})",
            f"compute abs of {a}", f"take absolute value of {a}",
            f"magnitude of {a}", f"abs of {_w(a)}"]


def ph_neg(a):
    return [f"negate {a}", f"compute -{a}", f"-{a}", f"minus {a}",
            f"flip sign of {a}", f"store -{a}", f"the negation of {a}"]


def ph_bitop(a, b, name, sym):
    return [f"compute {a} {sym} {b}", f"bitwise {name} of {a} and {b}",
            f"{a} {name} {b}", f"{_w(a)} {name} {_w(b)}",
            f"store {a} {sym} {b}", f"{a} {sym} {b}",
            f"bitwise {sym} of {a}, {b}"]


def ph_shl(a, sh):
    return [f"shift {a} left by {sh}", f"{a} << {sh}",
            f"left shift {a} by {sh} bits", f"{a} shifted left {sh}",
            f"compute {a} << {sh}", f"shl({a}, {sh})", f"{a} lsl {sh}"]


def ph_shr(a, sh):
    return [f"shift {a} right by {sh}", f"{a} >> {sh}",
            f"right shift {a} by {sh} bits", f"{a} shifted right {sh}",
            f"compute {a} >> {sh}", f"shr({a}, {sh})", f"{a} lsr {sh}"]


def ph_popcount(a):
    return [f"count set bits of {a}", f"popcount({a})",
            f"number of 1s in {a}", f"Hamming weight of {a}",
            f"bits set in {a}", f"count ones in binary of {a}"]


def ph_power_of_2(a):
    return [f"is {a} a power of 2?", f"check if {a} is a power of two",
            f"power-of-2 test on {a}", f"test: {a} is pow2",
            f"is {_w(a)} a power of 2", f"does {a} equal 2^k for some k"]


def ph_min(a, b):
    return [f"min of {a} and {b}", f"smaller of {a} and {b}",
            f"min({a}, {b})", f"the lesser of {a} and {b}",
            f"{_w(a)} or {_w(b)}, smaller", f"store min({a}, {b})",
            f"pick the smaller: {a} or {b}"]


def ph_max(a, b):
    return [f"max of {a} and {b}", f"larger of {a} and {b}",
            f"max({a}, {b})", f"the greater of {a} and {b}",
            f"{_w(a)} or {_w(b)}, bigger", f"store max({a}, {b})",
            f"pick the larger: {a} or {b}"]


def ph_sign(a):
    return [f"sign of {a}", f"sgn({a})", f"signum of {a}",
            f"compute sign of {a}", f"sign({a})",
            f"is {a} positive, zero, or negative"]


def ph_eq(a, b):
    return [f"is {a} equal to {b}", f"{a} == {b}",
            f"does {a} equal {b}", f"eq({a}, {b})",
            f"check {a} == {b}", f"{_w(a)} equals {_w(b)}?"]


def ph_ne(a, b):
    return [f"is {a} != {b}", f"{a} != {b}",
            f"not equal: {a}, {b}", f"ne({a}, {b})",
            f"check {a} != {b}", f"{_w(a)} not equal {_w(b)}?"]


def ph_clamp(x, lo, hi):
    return [f"clamp {x} to [{lo}, {hi}]", f"clip {x} between {lo} and {hi}",
            f"bound {x} in [{lo},{hi}]", f"restrict {x} to {lo}..{hi}",
            f"clamp({x}, {lo}, {hi})", f"saturate {x} between {lo} and {hi}"]


def ph_count_up(n):
    return [f"count up from 1 to {n}", f"store 1..{n}",
            f"write 1,2,...,{n}", f"enumerate 1 through {n}",
            f"{n} values ascending from 1", f"count up to {_w(n)}"]


def ph_count_down(n):
    return [f"count down from {n} to 1", f"store {n}, {n-1}, ..., 1",
            f"countdown from {n}", f"descending from {n} to 1",
            f"reverse count from {_w(n)}", f"list {n} down to 1"]


def ph_sum_range(lo, hi):
    return [f"sum from {lo} to {hi}", f"sum of {lo}..{hi}",
            f"{lo}+{lo+1}+...+{hi}", f"add integers {lo} through {hi}",
            f"total from {lo} to {hi}", f"sum {_w(lo)} to {_w(hi)}"]


def ph_sum_evens(n):
    return [f"sum of first {n} even numbers", f"2+4+...+{2*n}",
            f"sum {n} evens", f"add the first {_w(n)} even integers",
            f"total of first {n} evens", f"sum evens 2..{2*n}"]


def ph_sum_odds(n):
    return [f"sum of first {n} odd numbers", f"1+3+...+{2*n-1}",
            f"sum {n} odds", f"add the first {_w(n)} odd integers",
            f"total of first {n} odds", f"sum odds 1..{2*n-1}"]


def ph_product_range(lo, hi):
    return [f"product of {lo} to {hi}", f"{lo}*{lo+1}*...*{hi}",
            f"multiply integers {lo} through {hi}",
            f"product from {_w(lo)} to {_w(hi)}",
            f"compute prod({lo}..{hi})", f"{lo} times ... times {hi}"]


def ph_sum_array(n):
    return [f"sum the array of {n} elements", f"add up all {n} values",
            f"total of the {n}-element array", f"compute array sum, len {n}",
            f"sum(arr) with {n} items", f"sum of {_w(n)} values in memory"]


def ph_max_array(n):
    return [f"max of {n}-element array", f"largest of {n} values",
            f"find the maximum in {n} slots", f"max(arr), len {n}",
            f"biggest of {_w(n)} values", f"array max over {n} items"]


def ph_min_array(n):
    return [f"min of {n}-element array", f"smallest of {n} values",
            f"find the minimum in {n} slots", f"min(arr), len {n}",
            f"smallest of {_w(n)} values", f"array min over {n} items"]


def ph_reverse_array(n):
    return [f"reverse the {n}-element array", f"flip array of {n} items",
            f"reverse {n} values", f"mirror an array of {_w(n)} items",
            f"reverse(arr) length {n}", f"invert array order, {n} items"]


def ph_copy_array(n):
    return [f"copy {n}-element array", f"duplicate {n} values",
            f"clone array of {n} items", f"copy arr of {_w(n)}",
            f"replicate {n}-word array", f"copy(arr) length {n}"]


def ph_fill_array(n, v):
    return [f"fill {n} slots with {v}", f"initialize {n} words to {v}",
            f"set {n} elements to {v}", f"memset-like fill, {n}x {v}",
            f"fill(arr, {v}) length {n}", f"write {v} to {n} slots"]


def ph_find(n, tgt):
    return [f"find index of {tgt} in {n}-array",
            f"locate {tgt} in the {n}-slot buffer",
            f"indexOf({tgt}) in array of {n}",
            f"search for {tgt} among {n} values",
            f"first index of {tgt} in {_w(n)} items",
            f"find {_w(tgt)} in the array"]


def ph_count_occ(n, tgt):
    return [f"count {tgt} in {n}-array",
            f"how many times does {tgt} appear in {n} items",
            f"occurrences of {tgt} in array of {n}",
            f"count({tgt}) over {n} slots",
            f"tally {tgt} in {_w(n)} values",
            f"how many {_w(tgt)}s in the array"]


def ph_factorial(n):
    return [f"{n} factorial", f"{n}!", f"compute factorial of {n}",
            f"factorial({n})", f"{_w(n)} factorial",
            f"find {n}!", f"the value of {n}!"]


def ph_fib(n):
    return [f"first {n} Fibonacci numbers", f"{n} fib terms",
            f"Fibonacci({n})", f"fib sequence, {n} terms",
            f"first {_w(n)} Fibonacci values", f"generate {n} Fibonacci numbers"]


def ph_gcd(a, b):
    return [f"gcd({a}, {b})", f"greatest common divisor of {a} and {b}",
            f"gcd of {a} and {b}", f"hcf({a}, {b})",
            f"gcd of {_w(a)} and {_w(b)}", f"find gcd({a},{b})"]


def ph_power(base, exp):
    return [f"{base}^{exp}", f"{base} to the power {exp}",
            f"pow({base}, {exp})", f"{base} raised to {exp}",
            f"{_w(base)} to the {_w(exp)}", f"compute {base}**{exp}"]


def ph_tri(n):
    return [f"triangular number {n}", f"T_{n}",
            f"compute T({n})", f"nth triangular with n={n}",
            f"the {_w(n)}th triangular number", f"tri({n})"]


def ph_isqrt(n):
    return [f"isqrt({n})", f"integer square root of {n}",
            f"floor(sqrt({n}))", f"largest k with k*k <= {n}",
            f"int sqrt of {_w(n)}", f"compute isqrt({n})"]


def ph_strlen(s):
    return [f"length of string '{s}'", f"strlen('{s}')",
            f"count chars in '{s}'", f"how long is '{s}'",
            f"len('{s}')", f"number of characters in '{s}'"]


def ph_count_char(s, ch):
    return [f"count '{chr(ch)}' in '{s}'",
            f"how many '{chr(ch)}' in '{s}'",
            f"occurrences of '{chr(ch)}' in '{s}'",
            f"tally '{chr(ch)}' in the string '{s}'",
            f"count char {chr(ch)!r} in {s!r}",
            f"character count of '{chr(ch)}' in '{s}'"]


def ph_memcpy(n):
    return [f"memcpy {n} words", f"copy {n} ints src→dst",
            f"copy block of {n} words", f"transfer {n} memory words",
            f"duplicate {_w(n)} words", f"memory copy length {n}"]


def ph_memset(n, v):
    return [f"memset {n} words to {v}", f"fill {n} words with {v}",
            f"set {n} memory slots to {v}", f"memset(dst, {v}, {n})",
            f"zero-like fill {n}x{v}", f"initialize {_w(n)} slots to {v}"]


def ph_swap(a, b):
    return [f"swap {a} and {b} in memory", f"exchange values {a} and {b}",
            f"swap({a}, {b})", f"flip two memory words: {a}, {b}",
            f"interchange {_w(a)} and {_w(b)}", f"exchange two words, {a} and {b}"]


def ph_write_digits(n):
    return [f"write the decimal digits of {n}",
            f"store each digit of {n} as a word",
            f"split {n} into digits", f"digits of {_w(n)}",
            f"decompose {n} digit-by-digit", f"store {n}'s decimal digits"]


def ph_write_text(s):
    return [f"write the text '{s}' to memory",
            f"store the string '{s}' as bytes",
            f"display '{s}'", f"output the text '{s}'",
            f"put '{s}' in memory", f"render '{s}' as bytes"]


def ph_if_else_max(a, b):
    return [f"if {a} > {b} take {a} else {b}",
            f"if-else max of {a} and {b}",
            f"branch: return larger of {a} or {b}",
            f"if/else: bigger of {a}, {b}",
            f"conditional max({a},{b})",
            f"pick larger via if-else: {a} vs {b}"]


def ph_nested_if_sign(a):
    return [f"nested if for sign of {a}",
            f"use if-else-if to classify {a}",
            f"sign-check via nested conditionals for {a}",
            f"compute sign of {a} with branches",
            f"nested branch sign({a})",
            f"sign classification of {_w(a)}"]


def ph_switch_grade(s):
    return [f"grade for score {s}", f"letter grade of {s}",
            f"switch on {s} to grade", f"compute grade bucket for {s}",
            f"grade({s}) as 0-4 scale",
            f"score {s} → grade"]


def ph_early_ret(a):
    return [f"early return 0 if {a} is 0 else 2*{a}",
            f"if {a}==0 return 0 else double it",
            f"early-return check for {a}",
            f"zero-shortcut on {a}, else 2x",
            f"short-circuit on zero: {a}",
            f"test {_w(a)} then double"]


def ph_fn_double(n):
    return [f"double {n} via function call",
            f"call doubler({n})", f"subroutine 2*{n}",
            f"function: double {n}", f"jal to doubler with {_w(n)}",
            f"invoke double({n})"]


def ph_fn_triple(n):
    return [f"triple {n} via function", f"call tripler({n})",
            f"function 3*{n}", f"subroutine: triple {n}",
            f"invoke triple({n})", f"3x of {_w(n)} via function"]


def ph_fn_abs(n):
    return [f"absolute value of {n} via function",
            f"call abs({n})", f"subroutine abs of {n}",
            f"function |{n}|", f"invoke abs_fn({n})",
            f"abs of {_w(n)} via subroutine"]


def ph_bubble(n):
    return [f"bubble sort {n} elements", f"sort {n} values ascending",
            f"in-place bubble sort, {n} items",
            f"sort array of {n} using bubble sort",
            f"bubble_sort(arr) length {n}", f"sort {_w(n)} numbers"]


def ph_push_pop(n):
    return [f"push {n} values then pop them",
            f"stack push/pop sequence with {n} items",
            f"LIFO push-pop of {n} values",
            f"reverse {n} values via stack",
            f"push {n}, pop {n}",
            f"{_w(n)}-element stack exercise"]


# ═══════════════════════════════════════════════════════════════════════
# Family registration
# ═══════════════════════════════════════════════════════════════════════

# Each entry: (category, family_name, list_of_variants)
# Each variant: (phrasings, program_bytes, expected_dict, seed_dict_or_None)

FAMILIES: list[tuple[str, str, list]] = []
REJECTED: dict[str, int] = defaultdict(int)


def _try_build(label, builder, args, layouts, phrasings, expected, seed=None):
    """Try each layout; return first verified program bytes, or None."""
    progs = []
    for layout in layouts:
        try:
            p = builder(*args, layout)
        except Exception as e:
            REJECTED[label] += 1
            continue
        ok, _reason = verify_program(p, expected, seed=seed)
        if ok:
            progs.append(p)
        else:
            REJECTED[label] += 1
    if not progs:
        return None
    return progs[0]


def add_family(category, name, variants_data):
    """variants_data: list of dicts with keys:
         instruction, instruction_variants, bytes, expected_result, num_ops"""
    if variants_data:
        FAMILIES.append((category, name, variants_data))


# Build all families
def collect_all():
    # ── arithmetic ──────────────────────────────────────────────
    for fam_name, builder, ph_fn, pairs in [
        ("add", build_add, ph_add,
            [(a, b) for a in (0, 1, 3, 5, 7, 10, 20, 42, 100, 200)
                   for b in (0, 1, 3, 7, 15, 25, 50) if a + b <= 500][:28]),
        ("sub", build_sub, ph_sub,
            [(a, b) for a in (5, 10, 20, 30, 42, 50, 100, 200)
                   for b in (1, 3, 5, 7, 10, 25)][:28]),
    ]:
        layouts = [(5, 6, 7, 10), (15, 16, 17, 18), (8, 9, 11, 12)]
        vs = []
        for a, b in pairs:
            exp_val = (a + b) if fam_name == "add" else (a - b)
            expected = {"mem_word": (DATA_BASE, exp_val & 0xFFFFFFFF)}
            prog = _try_build(fam_name, builder, (a, b), layouts,
                              ph_fn(a, b), expected)
            if prog is None:
                continue
            vs.append(_mk_entry(ph_fn(a, b)[0], ph_fn(a, b), prog,
                                {"mem_word": (DATA_BASE, exp_val & 0xFFFFFFFF)}))
        add_family("arithmetic", fam_name, vs)

    # mul_by_repeated_add
    layouts = [(5, 6, 7, 10), (15, 16, 17, 18), (8, 9, 11, 12)]
    pairs = [(a, b) for a in (1, 2, 3, 4, 5, 6, 7, 10, 12) for b in (1, 2, 3, 5, 7)][:25]
    vs = []
    for a, b in pairs:
        expected = {"mem_word": (DATA_BASE, a * b)}
        prog = _try_build("mul_by_repeated_add", build_mul_by_repeated_add,
                          (a, b), layouts, ph_mul(a, b), expected)
        if prog:
            vs.append(_mk_entry(ph_mul(a, b)[0], ph_mul(a, b), prog, expected))
    add_family("arithmetic", "mul_by_repeated_add", vs)

    # abs / negate
    layouts = [(5, 6, 10), (15, 16, 17), (8, 9, 11)]
    vs = []
    for a in (-100, -50, -25, -10, -5, -3, -1, 0, 1, 3, 5, 10, 25, 50, 100, 200, 300, -200):
        expected = {"mem_word": (DATA_BASE, abs(a) & 0xFFFFFFFF)}
        prog = _try_build("abs", build_abs, (a,), layouts, ph_abs(a), expected)
        if prog:
            vs.append(_mk_entry(ph_abs(a)[0], ph_abs(a), prog, expected))
    add_family("arithmetic", "abs", vs)

    layouts = [(5, 6, 10), (15, 16, 17), (8, 9, 11)]
    vs = []
    for a in (-100, -50, -25, -10, -5, -3, -1, 0, 1, 3, 5, 10, 25, 50, 100, 200):
        expected = {"mem_word": (DATA_BASE, (-a) & 0xFFFFFFFF)}
        prog = _try_build("negate", build_negate, (a,), layouts, ph_neg(a), expected)
        if prog:
            vs.append(_mk_entry(ph_neg(a)[0], ph_neg(a), prog, expected))
    add_family("arithmetic", "negate", vs)

    # ── bitwise ─────────────────────────────────────────────────
    layouts = [(5, 6, 7, 10), (15, 16, 17, 18), (8, 9, 11, 12)]
    for fam_name, op_fn, sym, py_op in [
        ("and", and_, "&", lambda a, b: a & b),
        ("or", or_, "|", lambda a, b: a | b),
        ("xor", xor_, "^", lambda a, b: a ^ b),
    ]:
        vs = []
        pairs = [(a, b) for a in (0, 1, 3, 5, 7, 15, 31, 63, 127, 255, 0xFF0, 0x0FF)
                        for b in (0, 1, 3, 5, 15, 0xF0, 0xAA)][:25]
        for a, b in pairs:
            exp_val = py_op(a, b) & 0xFFFFFFFF
            expected = {"mem_word": (DATA_BASE, exp_val)}
            prog = _try_build(fam_name, build_bitop, (a, b, op_fn), layouts,
                              ph_bitop(a, b, fam_name, sym), expected)
            if prog:
                vs.append(_mk_entry(ph_bitop(a, b, fam_name, sym)[0],
                                    ph_bitop(a, b, fam_name, sym), prog, expected))
        add_family("bitwise", fam_name, vs)

    # shl / shr
    layouts = [(5, 6, 10), (15, 16, 17), (8, 9, 11)]
    vs = []
    for a in (1, 2, 3, 5, 7, 15, 31, 100, 255):
        for sh in (0, 1, 2, 3, 4, 5, 8, 12):
            exp_val = ((a << sh) & 0xFFFFFFFF)
            expected = {"mem_word": (DATA_BASE, exp_val)}
            prog = _try_build("shl", build_shl, (a, sh), layouts, ph_shl(a, sh), expected)
            if prog:
                vs.append(_mk_entry(ph_shl(a, sh)[0], ph_shl(a, sh), prog, expected))
    add_family("bitwise", "shl", vs[:25])

    vs = []
    for a in (0xFF, 0xFFFF, 0xF0F0, 100, 1000, 65535, 32, 256):
        for sh in (0, 1, 2, 3, 4, 5, 8):
            exp_val = ((a & 0xFFFFFFFF) >> sh)
            expected = {"mem_word": (DATA_BASE, exp_val)}
            prog = _try_build("shr", build_shr, (a, sh), layouts, ph_shr(a, sh), expected)
            if prog:
                vs.append(_mk_entry(ph_shr(a, sh)[0], ph_shr(a, sh), prog, expected))
    add_family("bitwise", "shr", vs[:25])

    # popcount
    layouts = [(5, 6, 7, 8, 10), (15, 16, 17, 18, 19), (20, 21, 22, 23, 24)]
    vs = []
    for a in (0, 1, 2, 3, 5, 7, 15, 16, 31, 42, 63, 100, 127, 255, 256, 511, 1023, 0xAAAA, 0x5555, 0xFFFF):
        expected = {"mem_word": (DATA_BASE, bin(a).count("1"))}
        prog = _try_build("popcount", build_popcount, (a,), layouts, ph_popcount(a), expected)
        if prog:
            vs.append(_mk_entry(ph_popcount(a)[0], ph_popcount(a), prog, expected))
    add_family("bitwise", "popcount", vs)

    # is_power_of_2
    layouts = [(5, 6, 7, 8, 10), (15, 16, 17, 18, 19), (20, 21, 22, 23, 24)]
    vs = []
    for a in (0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 31, 32, 63, 64, 100, 128, 256, 512, 1024):
        is_pow = 1 if (a > 0 and (a & (a - 1)) == 0) else 0
        expected = {"mem_word": (DATA_BASE, is_pow)}
        prog = _try_build("is_power_of_2", build_is_power_of_2, (a,), layouts,
                          ph_power_of_2(a), expected)
        if prog:
            vs.append(_mk_entry(ph_power_of_2(a)[0], ph_power_of_2(a), prog, expected))
    add_family("bitwise", "is_power_of_2", vs)

    # ── comparison ─────────────────────────────────────────────
    layouts = [(5, 6, 7, 10), (15, 16, 17, 18), (8, 9, 11, 12)]
    pairs = [(a, b) for a in (0, 1, 3, 5, 10, 20, 50, 100)
                    for b in (0, 2, 4, 7, 15, 25, 75) if a != b][:25]
    vs = []
    for a, b in pairs:
        expected = {"mem_word": (DATA_BASE, min(a, b))}
        prog = _try_build("min", build_min, (a, b), layouts, ph_min(a, b), expected)
        if prog:
            vs.append(_mk_entry(ph_min(a, b)[0], ph_min(a, b), prog, expected))
    add_family("comparison", "min", vs)

    vs = []
    for a, b in pairs:
        expected = {"mem_word": (DATA_BASE, max(a, b))}
        prog = _try_build("max", build_max, (a, b), layouts, ph_max(a, b), expected)
        if prog:
            vs.append(_mk_entry(ph_max(a, b)[0], ph_max(a, b), prog, expected))
    add_family("comparison", "max", vs)

    layouts = [(5, 6, 10), (15, 16, 17), (8, 9, 11)]
    vs = []
    for a in (-100, -50, -10, -5, -1, 0, 1, 5, 10, 50, 100, 200, -200, 25, -25, 42, -42):
        sg = 1 if a > 0 else (-1 if a < 0 else 0)
        expected = {"mem_word": (DATA_BASE, sg & 0xFFFFFFFF)}
        prog = _try_build("sign", build_sign, (a,), layouts, ph_sign(a), expected)
        if prog:
            vs.append(_mk_entry(ph_sign(a)[0], ph_sign(a), prog, expected))
    add_family("comparison", "sign", vs)

    layouts = [(5, 6, 7, 10), (15, 16, 17, 18), (8, 9, 11, 12)]
    pairs = [(0, 0), (5, 5), (10, 10), (42, 42), (100, 100),
             (0, 1), (5, 7), (10, 20), (42, 43), (100, 99),
             (-5, -5), (-10, -10), (7, -7), (-3, 3), (1, -1),
             (25, 25), (50, 50), (3, 5)]
    vs = []
    for a, b in pairs:
        expected = {"mem_word": (DATA_BASE, 1 if a == b else 0)}
        prog = _try_build("equal", build_equal, (a, b), layouts, ph_eq(a, b), expected)
        if prog:
            vs.append(_mk_entry(ph_eq(a, b)[0], ph_eq(a, b), prog, expected))
    add_family("comparison", "equal", vs)

    vs = []
    for a, b in pairs:
        expected = {"mem_word": (DATA_BASE, 0 if a == b else 1)}
        prog = _try_build("not_equal", build_not_equal, (a, b), layouts, ph_ne(a, b), expected)
        if prog:
            vs.append(_mk_entry(ph_ne(a, b)[0], ph_ne(a, b), prog, expected))
    add_family("comparison", "not_equal", vs)

    layouts = [(5, 6, 7, 10), (15, 16, 17, 18), (8, 9, 11, 12)]
    triples = [(0, 1, 10), (5, 1, 10), (15, 1, 10), (-5, 0, 10), (100, 0, 50),
               (7, 5, 20), (42, 10, 50), (25, 0, 20), (3, 5, 15), (8, 2, 7),
               (-3, -1, 5), (50, 10, 40), (1, 1, 100), (99, 0, 100),
               (60, 50, 70), (0, -5, 5), (10, 10, 20), (30, 20, 25)]
    vs = []
    for x, lo, hi in triples:
        c = max(lo, min(hi, x))
        expected = {"mem_word": (DATA_BASE, c & 0xFFFFFFFF)}
        prog = _try_build("clamp", build_clamp, (x, lo, hi), layouts,
                          ph_clamp(x, lo, hi), expected)
        if prog:
            vs.append(_mk_entry(ph_clamp(x, lo, hi)[0], ph_clamp(x, lo, hi),
                                prog, expected))
    add_family("comparison", "clamp", vs)

    # ── loops ──────────────────────────────────────────────────
    layouts = [(5, 6, 7), (15, 16, 17), (8, 9, 11)]
    vs = []
    for n in (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20):
        expected = {"mem_words": (DATA_BASE, list(range(1, n + 1)))}
        prog = _try_build("count_up", build_count_up, (n,), layouts, ph_count_up(n), expected)
        if prog:
            vs.append(_mk_entry(ph_count_up(n)[0], ph_count_up(n), prog, expected))
    add_family("loops", "count_up", vs)

    layouts = [(5, 6), (15, 16), (8, 9)]
    vs = []
    for n in (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20):
        expected = {"mem_words": (DATA_BASE, list(range(n, 0, -1)))}
        prog = _try_build("count_down", build_count_down, (n,), layouts, ph_count_down(n), expected)
        if prog:
            vs.append(_mk_entry(ph_count_down(n)[0], ph_count_down(n), prog, expected))
    add_family("loops", "count_down", vs)

    layouts = [(5, 6, 7, 10), (15, 16, 17, 18), (8, 9, 11, 12)]
    vs = []
    for lo, hi in [(1, 5), (1, 10), (2, 8), (3, 7), (5, 15), (1, 20),
                   (10, 20), (1, 3), (4, 9), (2, 6), (5, 10), (1, 30),
                   (7, 12), (1, 100), (20, 40), (1, 50)]:
        expected = {"mem_word": (DATA_BASE, sum(range(lo, hi + 1)))}
        prog = _try_build("sum_range", build_sum_range, (lo, hi), layouts,
                          ph_sum_range(lo, hi), expected)
        if prog:
            vs.append(_mk_entry(ph_sum_range(lo, hi)[0], ph_sum_range(lo, hi), prog, expected))
    add_family("loops", "sum_range", vs)

    vs = []
    for n in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50):
        expected = {"mem_word": (DATA_BASE, sum(range(2, 2 * n + 1, 2)))}
        prog = _try_build("sum_evens", build_sum_evens, (n,), layouts, ph_sum_evens(n), expected)
        if prog:
            vs.append(_mk_entry(ph_sum_evens(n)[0], ph_sum_evens(n), prog, expected))
    add_family("loops", "sum_evens", vs)

    vs = []
    for n in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50):
        expected = {"mem_word": (DATA_BASE, sum(range(1, 2 * n, 2)))}
        prog = _try_build("sum_odds", build_sum_odds, (n,), layouts, ph_sum_odds(n), expected)
        if prog:
            vs.append(_mk_entry(ph_sum_odds(n)[0], ph_sum_odds(n), prog, expected))
    add_family("loops", "sum_odds", vs)

    layouts = [(5, 6, 7, 8, 9, 10), (15, 16, 17, 18, 19, 20), (21, 22, 23, 24, 25, 26)]
    vs = []
    for lo, hi in [(1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (1, 6),
                   (3, 5), (1, 2), (2, 3), (3, 4), (4, 5), (1, 7),
                   (2, 6), (3, 6), (1, 8)]:
        prod = 1
        for k in range(lo, hi + 1):
            prod *= k
        expected = {"mem_word": (DATA_BASE, prod)}
        prog = _try_build("product_range_small", build_product_range_small,
                          (lo, hi), layouts, ph_product_range(lo, hi), expected)
        if prog:
            vs.append(_mk_entry(ph_product_range(lo, hi)[0], ph_product_range(lo, hi),
                                prog, expected))
    add_family("loops", "product_range_small", vs)

    # ── arrays ─────────────────────────────────────────────────
    layouts = [(5, 6, 7, 8, 10), (15, 16, 17, 18, 19), (20, 21, 22, 23, 24)]
    for fam_name, builder, ph_fn, py_reduce in [
        ("sum_array", build_sum_array, ph_sum_array, sum),
        ("max_array", build_max_array, ph_max_array, max),
        ("min_array", build_min_array, ph_min_array, min),
    ]:
        vs = []
        for n in (2, 3, 4, 5, 6, 8, 10):
            for base in ([1, 7, 3, 9, 2, 5, 8, 4, 6, 10],
                         [10, 3, 8, 1, 5, 7, 2, 9, 4, 6],
                         [5, 5, 2, 8, 1, 3, 9, 7, 4, 6]):
                arr = base[:n]
                expected = {"mem_word": (DATA_BASE, py_reduce(arr))}
                seed = {DATA_BASE + ARR_OFFSET: arr}
                prog = _try_build(fam_name, builder, (n,), layouts,
                                  ph_fn(n), expected, seed=seed)
                if prog:
                    # Attach seed info — for output we include seed into expected?
                    # task: expected_result as dict. We'll keep expected but add seed?
                    # Spec says only mem/reg in expected_result. We store seed separately.
                    vs.append(_mk_entry(ph_fn(n)[0], ph_fn(n), prog, expected,
                                        seed=seed))
        add_family("arrays", fam_name, vs[:25])

    # reverse_array
    layouts = [(5, 6, 7, 8, 10), (15, 16, 17, 18, 19), (20, 21, 22, 23, 24)]
    vs = []
    for n in (2, 3, 4, 5, 6, 7, 8, 10):
        for base in ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                     [7, 3, 9, 1, 5, 8, 2, 6, 4, 10]):
            arr = base[:n]
            expected = {"mem_words": (DATA_BASE + DST_OFFSET, list(reversed(arr)))}
            seed = {DATA_BASE + ARR_OFFSET: arr}
            prog = _try_build("reverse_array", build_reverse_array, (n,), layouts,
                              ph_reverse_array(n), expected, seed=seed)
            if prog:
                vs.append(_mk_entry(ph_reverse_array(n)[0], ph_reverse_array(n),
                                    prog, expected, seed=seed))
    add_family("arrays", "reverse_array", vs[:25])

    # copy_array
    vs = []
    for n in (2, 3, 4, 5, 6, 8, 10):
        for base in ([11, 22, 33, 44, 55, 66, 77, 88, 99, 100],
                     [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]):
            arr = base[:n]
            expected = {"mem_words": (DATA_BASE + DST_OFFSET, arr)}
            seed = {DATA_BASE + ARR_OFFSET: arr}
            prog = _try_build("copy_array", build_copy_array, (n,), layouts,
                              ph_copy_array(n), expected, seed=seed)
            if prog:
                vs.append(_mk_entry(ph_copy_array(n)[0], ph_copy_array(n),
                                    prog, expected, seed=seed))
    add_family("arrays", "copy_array", vs[:20])

    # fill_array
    layouts_fill = [(5, 6, 7, 8), (15, 16, 17, 18), (20, 21, 22, 23)]
    vs = []
    for n in (2, 3, 4, 5, 6, 8, 10):
        for v in (0, 1, 7, 42, 100, 255):
            expected = {"mem_words": (DATA_BASE + ARR_OFFSET, [v] * n)}
            prog = _try_build("fill_array", build_fill_array, (n, v), layouts_fill,
                              ph_fill_array(n, v), expected)
            if prog:
                vs.append(_mk_entry(ph_fill_array(n, v)[0], ph_fill_array(n, v),
                                    prog, expected))
    add_family("arrays", "fill_array", vs[:25])

    # find_element
    layouts = [(5, 6, 7, 8, 10, 11), (15, 16, 17, 18, 19, 20), (20, 21, 22, 23, 24, 25)]
    vs = []
    for n, tgt, arr, idx in [
        (5, 3, [1, 2, 3, 4, 5], 2),
        (5, 7, [7, 2, 3, 4, 5], 0),
        (5, 5, [1, 2, 3, 4, 5], 4),
        (5, 99, [1, 2, 3, 4, 5], -1),
        (4, 10, [5, 10, 15, 20], 1),
        (6, 42, [1, 2, 42, 4, 5, 6], 2),
        (3, 0, [0, 1, 2], 0),
        (4, 20, [20, 10, 20, 30], 0),
        (5, 1, [5, 4, 3, 2, 1], 4),
        (6, 100, [1, 2, 3, 4, 5, 6], -1),
        (4, 8, [2, 4, 6, 8], 3),
        (3, 7, [1, 7, 9], 1),
        (5, 3, [3, 1, 2, 4, 5], 0),
        (4, 1, [4, 3, 2, 1], 3),
        (6, 5, [1, 2, 3, 4, 5, 6], 4),
    ]:
        expected = {"mem_word": (DATA_BASE, idx & 0xFFFFFFFF)}
        seed = {DATA_BASE + ARR_OFFSET: arr}
        prog = _try_build("find_element", build_find_element, (n, tgt), layouts,
                          ph_find(n, tgt), expected, seed=seed)
        if prog:
            vs.append(_mk_entry(ph_find(n, tgt)[0], ph_find(n, tgt),
                                prog, expected, seed=seed))
    add_family("arrays", "find_element", vs)

    # count_occurrences
    vs = []
    for n, tgt, arr, cnt in [
        (5, 3, [3, 1, 3, 2, 3], 3),
        (4, 5, [5, 5, 5, 5], 4),
        (6, 1, [2, 3, 4, 5, 6, 7], 0),
        (5, 0, [0, 1, 0, 2, 0], 3),
        (4, 7, [1, 7, 2, 7], 2),
        (3, 42, [42, 42, 42], 3),
        (6, 10, [10, 20, 10, 30, 10, 40], 3),
        (5, 9, [1, 2, 3, 4, 5], 0),
        (4, 8, [8, 8, 1, 8], 3),
        (5, 2, [1, 2, 3, 2, 1], 2),
        (6, 5, [5, 4, 3, 2, 1, 5], 2),
        (3, 0, [0, 0, 1], 2),
        (4, 4, [4, 1, 4, 4], 3),
        (5, 100, [10, 20, 30, 40, 50], 0),
        (6, 1, [1, 1, 1, 1, 1, 1], 6),
    ]:
        expected = {"mem_word": (DATA_BASE, cnt)}
        seed = {DATA_BASE + ARR_OFFSET: arr}
        prog = _try_build("count_occurrences", build_count_occurrences, (n, tgt),
                          layouts, ph_count_occ(n, tgt), expected, seed=seed)
        if prog:
            vs.append(_mk_entry(ph_count_occ(n, tgt)[0], ph_count_occ(n, tgt),
                                prog, expected, seed=seed))
    add_family("arrays", "count_occurrences", vs)

    # ── math ──────────────────────────────────────────────────
    layouts = [(5, 6, 7, 8, 9, 10), (15, 16, 17, 18, 19, 20), (21, 22, 23, 24, 25, 26)]
    vs = []
    for n in (1, 2, 3, 4, 5, 6, 7):
        expected = {"mem_word": (DATA_BASE, math.factorial(n))}
        prog = _try_build("factorial", build_factorial, (n,), layouts, ph_factorial(n), expected)
        if prog:
            vs.append(_mk_entry(ph_factorial(n)[0], ph_factorial(n), prog, expected))
    add_family("math", "factorial", vs)

    layouts = [(5, 6, 7, 8, 10), (15, 16, 17, 18, 19), (20, 21, 22, 23, 24)]
    def fib_seq(n):
        a, b, out = 0, 1, []
        for _ in range(n):
            out.append(a); a, b = b, a + b
        return out
    vs = []
    for n in (2, 3, 4, 5, 6, 7, 8, 10, 12):
        expected = {"mem_words": (DATA_BASE, fib_seq(n))}
        prog = _try_build("fibonacci", build_fibonacci, (n,), layouts, ph_fib(n), expected)
        if prog:
            vs.append(_mk_entry(ph_fib(n)[0], ph_fib(n), prog, expected))
    add_family("math", "fibonacci", vs)

    layouts = [(5, 6, 10), (15, 16, 17), (8, 9, 11)]
    vs = []
    for a, b in [(12, 18), (24, 36), (17, 5), (100, 75), (48, 36),
                 (7, 14), (9, 12), (20, 15), (25, 10), (8, 12),
                 (30, 18), (21, 14), (6, 9), (50, 20), (11, 7),
                 (45, 60), (100, 40), (16, 24)]:
        expected = {"mem_word": (DATA_BASE, math.gcd(a, b))}
        prog = _try_build("gcd", build_gcd, (a, b), layouts, ph_gcd(a, b), expected)
        if prog:
            vs.append(_mk_entry(ph_gcd(a, b)[0], ph_gcd(a, b), prog, expected))
    add_family("math", "gcd", vs)

    layouts = [(5, 6, 7, 8, 9, 10), (15, 16, 17, 18, 19, 20), (21, 22, 23, 24, 25, 26)]
    vs = []
    for base, exp in [(2, 3), (2, 4), (2, 5), (2, 6), (2, 8), (3, 2), (3, 3), (3, 4),
                      (5, 2), (5, 3), (4, 3), (7, 2), (10, 2), (10, 3),
                      (6, 2), (8, 2), (9, 2), (4, 4), (2, 10)]:
        expected = {"mem_word": (DATA_BASE, base ** exp)}
        prog = _try_build("power", build_power, (base, exp), layouts, ph_power(base, exp), expected)
        if prog:
            vs.append(_mk_entry(ph_power(base, exp)[0], ph_power(base, exp), prog, expected))
    add_family("math", "power", vs)

    layouts = [(5, 6, 7, 10), (15, 16, 17, 18), (8, 9, 11, 12)]
    vs = []
    for n in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 50, 100):
        expected = {"mem_word": (DATA_BASE, n * (n + 1) // 2)}
        prog = _try_build("triangular", build_triangular, (n,), layouts, ph_tri(n), expected)
        if prog:
            vs.append(_mk_entry(ph_tri(n)[0], ph_tri(n), prog, expected))
    add_family("math", "triangular", vs)

    layouts = [(5, 6, 7, 8, 9, 10), (15, 16, 17, 18, 19, 20), (21, 22, 23, 24, 25, 26)]
    vs = []
    for n in (0, 1, 2, 3, 4, 5, 8, 9, 10, 15, 16, 17, 25, 26, 36, 50, 100, 121, 144):
        expected = {"mem_word": (DATA_BASE, int(math.isqrt(n)))}
        prog = _try_build("isqrt", build_isqrt, (n,), layouts, ph_isqrt(n), expected)
        if prog:
            vs.append(_mk_entry(ph_isqrt(n)[0], ph_isqrt(n), prog, expected))
    add_family("math", "isqrt", vs)

    # ── strings ───────────────────────────────────────────────
    layouts = [(5, 6, 7), (15, 16, 17), (8, 9, 11)]
    vs = []
    for s in ["hi", "hello", "world", "a", "abc", "test", "foo", "bar",
              "claude", "rv32i", "xyz", "longer", "programming", "foobar",
              "ok", "up", "down", "reflex"]:
        expected = {"mem_word": (DATA_BASE, len(s))}
        seed = {DATA_BASE + ARR_OFFSET: pack_string_seed(s)}
        prog = _try_build("strlen", build_strlen, (), layouts, ph_strlen(s),
                          expected, seed=seed)
        if prog:
            vs.append(_mk_entry(ph_strlen(s)[0], ph_strlen(s), prog, expected, seed=seed))
    add_family("strings", "strlen", vs)

    layouts = [(5, 6, 7, 8), (15, 16, 17, 18), (20, 21, 22, 23)]
    vs = []
    for s, ch in [("hello", ord("l")), ("banana", ord("a")),
                  ("test", ord("t")), ("mississippi", ord("s")),
                  ("programming", ord("m")), ("aaa", ord("a")),
                  ("abcdef", ord("z")), ("xyxyxy", ord("x")),
                  ("rv32i", ord("3")), ("countoccur", ord("o")),
                  ("aabbcc", ord("b")), ("hello world", ord("l")),
                  ("letters", ord("e")), ("rabbit", ord("b")),
                  ("zebra", ord("r")), ("apple pie", ord("p"))]:
        expected = {"mem_word": (DATA_BASE, s.count(chr(ch)))}
        seed = {DATA_BASE + ARR_OFFSET: pack_string_seed(s)}
        prog = _try_build("count_char", build_count_char, (ch,), layouts,
                          ph_count_char(s, ch), expected, seed=seed)
        if prog:
            vs.append(_mk_entry(ph_count_char(s, ch)[0], ph_count_char(s, ch),
                                prog, expected, seed=seed))
    add_family("strings", "count_char", vs)

    # ── memory ────────────────────────────────────────────────
    layouts = [(5, 6, 7, 8, 10), (15, 16, 17, 18, 19), (20, 21, 22, 23, 24)]
    vs = []
    for n in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12):
        arr = list(range(1, n + 1))
        expected = {"mem_words": (DATA_BASE + DST_OFFSET, arr)}
        seed = {DATA_BASE + ARR_OFFSET: arr}
        prog = _try_build("memcpy", build_memcpy, (n,), layouts,
                          ph_memcpy(n), expected, seed=seed)
        if prog:
            vs.append(_mk_entry(ph_memcpy(n)[0], ph_memcpy(n), prog, expected, seed=seed))
    add_family("memory", "memcpy", vs)

    layouts_ms = [(5, 6, 7, 8), (15, 16, 17, 18), (20, 21, 22, 23)]
    vs = []
    for n in (1, 2, 3, 4, 5, 6, 8, 10):
        for v in (0, 1, 42, 100, 255):
            expected = {"mem_words": (DATA_BASE + ARR_OFFSET, [v] * n)}
            prog = _try_build("memset", build_memset, (n, v), layouts_ms,
                              ph_memset(n, v), expected)
            if prog:
                vs.append(_mk_entry(ph_memset(n, v)[0], ph_memset(n, v), prog, expected))
    add_family("memory", "memset", vs[:20])

    layouts = [(5, 6, 7, 10), (15, 16, 17, 18), (8, 9, 11, 12)]
    vs = []
    for a, b in [(1, 2), (10, 20), (42, 99), (7, 13), (100, 200),
                 (0, 5), (25, 75), (50, 60), (3, 8), (11, 22),
                 (33, 77), (1, 1000), (5, 500), (15, 150), (6, 9), (2, 3)]:
        expected = {"mem_words": (DATA_BASE, [b, a])}
        prog = _try_build("swap_two_values", build_swap_two_values, (a, b), layouts,
                          ph_swap(a, b), expected)
        if prog:
            vs.append(_mk_entry(ph_swap(a, b)[0], ph_swap(a, b), prog, expected))
    add_family("memory", "swap_two_values", vs)

    # ── display ──────────────────────────────────────────────
    layouts = [(5, 6, 7, 8, 9, 10), (15, 16, 17, 18, 19, 20), (21, 22, 23, 24, 25, 26)]
    vs = []
    for n in (1, 5, 7, 9, 10, 12, 42, 99, 100, 123, 250, 500, 999, 3, 27, 88, 6):
        digits = [int(c) for c in str(n)]
        expected = {"mem_words": (DATA_BASE, digits)}
        prog = _try_build("write_digits", build_write_digits, (n,), layouts,
                          ph_write_digits(n), expected)
        if prog:
            vs.append(_mk_entry(ph_write_digits(n)[0], ph_write_digits(n), prog, expected))
    add_family("display", "write_digits", vs)

    layouts = [(5, 6), (15, 16), (8, 9)]
    vs = []
    for s in ["hi", "ok", "hello", "abc", "xy", "test",
              "reflex", "rv32", "bye", "foo", "bar", "claude",
              "ai", "go", "up", "down"]:
        bytes_ = s.encode("ascii") + b"\x00"
        while len(bytes_) % 4 != 0:
            bytes_ += b"\x00"
        words = [int.from_bytes(bytes_[i:i+4], "little")
                 for i in range(0, len(bytes_), 4)]
        expected = {"mem_words": (DATA_BASE, words)}
        prog = _try_build("write_ascii_text", build_write_ascii_text, (s,), layouts,
                          ph_write_text(s), expected)
        if prog:
            vs.append(_mk_entry(ph_write_text(s)[0], ph_write_text(s), prog, expected))
    add_family("display", "write_ascii_text", vs)

    # ── control_flow ─────────────────────────────────────────
    layouts = [(5, 6, 7, 10), (15, 16, 17, 18), (8, 9, 11, 12)]
    pairs = [(3, 5), (10, 7), (0, 0), (100, 99), (15, 25), (-5, 5),
             (7, 2), (42, 50), (1, 0), (50, 50), (20, 15), (-10, -3),
             (8, 12), (33, 11), (0, -1), (99, 100), (25, 24), (6, 9)]
    vs = []
    for a, b in pairs:
        expected = {"mem_word": (DATA_BASE, max(a, b) & 0xFFFFFFFF)}
        prog = _try_build("if_else_max", build_if_else_max, (a, b), layouts,
                          ph_if_else_max(a, b), expected)
        if prog:
            vs.append(_mk_entry(ph_if_else_max(a, b)[0], ph_if_else_max(a, b), prog, expected))
    add_family("control_flow", "if_else_max", vs)

    layouts = [(5, 6, 10), (15, 16, 17), (8, 9, 11)]
    vs = []
    for a in (-100, -50, -10, -1, 0, 1, 10, 50, 100, 200, -200, 25, -25, 42, -42, 5, -5):
        sg = 1 if a > 0 else (-1 if a < 0 else 0)
        expected = {"mem_word": (DATA_BASE, sg & 0xFFFFFFFF)}
        prog = _try_build("nested_if_sign", build_nested_if_sign, (a,), layouts,
                          ph_nested_if_sign(a), expected)
        if prog:
            vs.append(_mk_entry(ph_nested_if_sign(a)[0], ph_nested_if_sign(a), prog, expected))
    add_family("control_flow", "nested_if_sign", vs)

    layouts = [(5, 6, 7, 10), (15, 16, 17, 18), (8, 9, 11, 12)]
    vs = []
    for s in (0, 10, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 59, 61, 79, 89, 99, 45):
        if s >= 90: g = 4
        elif s >= 80: g = 3
        elif s >= 70: g = 2
        elif s >= 60: g = 1
        else: g = 0
        expected = {"mem_word": (DATA_BASE, g)}
        prog = _try_build("switch_like_grade", build_switch_like_grade, (s,), layouts,
                          ph_switch_grade(s), expected)
        if prog:
            vs.append(_mk_entry(ph_switch_grade(s)[0], ph_switch_grade(s), prog, expected))
    add_family("control_flow", "switch_like_grade", vs)

    layouts = [(5, 6, 10), (15, 16, 17), (8, 9, 11)]
    vs = []
    for a in (0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 42, 50, 100, 200, 500, 77):
        res = 0 if a == 0 else 2 * a
        expected = {"mem_word": (DATA_BASE, res & 0xFFFFFFFF)}
        prog = _try_build("early_return_zero", build_early_return_zero, (a,), layouts,
                          ph_early_ret(a), expected)
        if prog:
            vs.append(_mk_entry(ph_early_ret(a)[0], ph_early_ret(a), prog, expected))
    add_family("control_flow", "early_return_zero", vs)

    # ── function ─────────────────────────────────────────────
    # layout (ra0, rp, ra) — ra=1 by convention but customize
    layouts = [(10, 11, 1), (12, 13, 1), (14, 15, 1)]
    vs = []
    for n in (0, 1, 2, 3, 5, 7, 10, 15, 25, 42, 50, 100, 200, 500, 1000):
        expected = {"mem_word": (DATA_BASE, 2 * n)}
        prog = _try_build("double_fn", build_double_fn, (n,), layouts,
                          ph_fn_double(n), expected)
        if prog:
            vs.append(_mk_entry(ph_fn_double(n)[0], ph_fn_double(n), prog, expected))
    add_family("function", "double_fn", vs)

    layouts = [(10, 11, 1, 12), (13, 14, 1, 15), (16, 17, 1, 18)]
    vs = []
    for n in (0, 1, 2, 3, 5, 7, 10, 15, 25, 42, 50, 100, 200, 333, 666):
        expected = {"mem_word": (DATA_BASE, 3 * n)}
        prog = _try_build("triple_fn", build_triple_fn, (n,), layouts,
                          ph_fn_triple(n), expected)
        if prog:
            vs.append(_mk_entry(ph_fn_triple(n)[0], ph_fn_triple(n), prog, expected))
    add_family("function", "triple_fn", vs)

    layouts = [(10, 11, 1), (12, 13, 1), (14, 15, 1)]
    vs = []
    for n in (-100, -50, -25, -10, -5, -3, -1, 0, 1, 3, 5, 10, 25, 50, 100, 200):
        expected = {"mem_word": (DATA_BASE, abs(n) & 0xFFFFFFFF)}
        prog = _try_build("abs_fn", build_abs_fn, (n,), layouts,
                          ph_fn_abs(n), expected)
        if prog:
            vs.append(_mk_entry(ph_fn_abs(n)[0], ph_fn_abs(n), prog, expected))
    add_family("function", "abs_fn", vs)

    # ── sorting ──────────────────────────────────────────────
    layouts = [(5, 6, 7, 8, 9, 10), (15, 16, 17, 18, 19, 20), (20, 21, 22, 23, 24, 25)]
    vs = []
    test_arrays = [
        (3, [3, 1, 2]), (3, [1, 2, 3]), (3, [3, 2, 1]),
        (3, [5, 5, 1]), (3, [7, 3, 9]), (3, [10, 4, 6]),
        (4, [4, 3, 2, 1]), (4, [1, 3, 2, 4]), (4, [5, 2, 8, 1]),
        (4, [9, 7, 3, 5]), (4, [10, 1, 5, 2]), (4, [6, 6, 3, 1]),
        (3, [2, 1, 0]), (3, [0, 5, 3]), (4, [8, 4, 2, 1]),
        (4, [3, 3, 3, 3]), (3, [100, 50, 25]), (4, [7, 1, 4, 2]),
    ]
    for n, arr in test_arrays:
        expected = {"mem_words": (DATA_BASE + ARR_OFFSET, sorted(arr))}
        seed = {DATA_BASE + ARR_OFFSET: arr}
        prog = _try_build("bubble_sort_small", _build_bubble_sort_small_clean,
                          (n,), layouts, ph_bubble(n), expected, seed=seed)
        if prog:
            vs.append(_mk_entry(ph_bubble(n)[0], ph_bubble(n), prog, expected, seed=seed))
    add_family("sorting", "bubble_sort_small", vs)

    # ── stack ────────────────────────────────────────────────
    layouts = [(5, 6, 7), (15, 16, 17), (8, 9, 11)]
    vs = []
    for values in [
        [1, 2, 3], [10, 20, 30], [5, 10], [1, 2, 3, 4],
        [7, 14, 21, 28], [100, 200, 300], [42, 42], [1, 3, 5, 7, 9],
        [9, 8, 7, 6], [2, 4, 6, 8, 10], [11, 22, 33],
        [50, 40, 30, 20, 10], [1, 1, 1], [5, 6, 7, 8, 9],
        [100], [1, 2], [3, 6, 9, 12], [0, 1, 2, 3],
    ]:
        expected = {"mem_words": (DATA_BASE, list(reversed(values)))}
        prog = _try_build("push_pop_sequence", build_push_pop_sequence,
                          (values,), layouts, ph_push_pop(len(values)), expected)
        if prog:
            vs.append(_mk_entry(ph_push_pop(len(values))[0],
                                ph_push_pop(len(values)), prog, expected))
    add_family("stack", "push_pop_sequence", vs)


def _mk_entry(instr, phrasings, prog_bytes, expected, seed=None):
    # Convert expected dict → JSON shape
    result = {}
    if "mem_word" in expected:
        addr, val = expected["mem_word"]
        result[f"mem_0x{addr:X}"] = int(val) & 0xFFFFFFFF
    if "mem_words" in expected:
        addr, vals = expected["mem_words"]
        result[f"mem_0x{addr:X}_words"] = [int(v) & 0xFFFFFFFF for v in vals]
    if "reg" in expected:
        idx, val = expected["reg"]
        result[f"reg_{idx}"] = int(val) & 0xFFFFFFFF
    entry = {
        "instruction": instr,
        "instruction_variants": phrasings,
        "bytes": list(prog_bytes),
        "expected_result": result,
        "num_ops": len(prog_bytes) // 4,
    }
    if seed is not None:
        entry["seed"] = {f"0x{addr:X}": [int(v) & 0xFFFFFFFF for v in vals]
                         for addr, vals in seed.items()}
    return entry


def main():
    out_root = REPO / "programs"
    out_root.mkdir(exist_ok=True)

    print("Generating programs…")
    collect_all()

    # Write files
    by_category: dict[str, int] = defaultdict(int)
    by_family: dict[str, int] = {}
    total_progs = 0
    for category, name, progs in FAMILIES:
        cat_dir = out_root / category
        cat_dir.mkdir(exist_ok=True)
        path = cat_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump({"family": name, "programs": progs}, f, indent=2)
        by_category[category] += len(progs)
        by_family[f"{category}/{name}"] = len(progs)
        total_progs += len(progs)

    print(f"\n=== Summary ===")
    print(f"Families: {len(FAMILIES)}")
    print(f"Total programs: {total_progs}")
    print(f"\nBy category:")
    for cat, n in sorted(by_category.items()):
        print(f"  {cat:<14} {n}")
    print(f"\nBy family:")
    for fam, n in sorted(by_family.items()):
        rej = REJECTED.get(fam.split("/")[-1], 0)
        attempted = n + rej
        rate = (rej / attempted * 100) if attempted > 0 else 0
        flag = " <<<HIGH" if rate > 5 else ""
        print(f"  {fam:<40} kept={n:<4} rejected={rej:<3} ({rate:.1f}%){flag}")
    print(f"\nRejection totals: {dict(REJECTED)}")


if __name__ == "__main__":
    main()
