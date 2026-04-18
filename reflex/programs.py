"""
RV32I program templates + natural-language phrasings + task generator.

Each template is 5-20 instructions. The verifier runs every candidate
through unicorn and checks the post-halt register/memory state; variants
that don't match are rejected before they ever reach training.
"""

from .riscv import (
    DATA_BASE, HALT_INSTR, PROGRAM_START, Rv32i,
    add, addi, bge, beq, blt, bne, halt, jal, jalr, lui, lw, pack, sub, sw,
)


def _addr(ops: list[int]) -> int:
    """Byte offset of the next op to be appended."""
    return 4 * len(ops)


# ── 1. ADD TWO NUMBERS ───────────────────────────────────────────────
# Compute a+b, store at DATA_BASE.

def prog_add_v0(a: int, b: int) -> bytes:
    ops = [
        addi(5, 0, a),
        addi(6, 0, b),
        add(7, 5, 6),
        lui(10, DATA_BASE >> 12),
        sw(7, 10, 0),
        halt(),
    ]
    return pack(*ops)


def prog_add_v1(a: int, b: int) -> bytes:
    """addi-immediate form: skip the second addi, use addi on the first reg."""
    ops = [
        addi(5, 0, a),
        addi(7, 5, b),               # x7 = a + b via immediate add
        lui(11, DATA_BASE >> 12),
        sw(7, 11, 0),
        halt(),
    ]
    return pack(*ops)


def prog_add_v2(a: int, b: int) -> bytes:
    """Different register layout: x15/x16 → x17."""
    ops = [
        addi(15, 0, a),
        addi(16, 0, b),
        add(17, 15, 16),
        lui(18, DATA_BASE >> 12),
        sw(17, 18, 0),
        halt(),
    ]
    return pack(*ops)


ADD_VARIANTS = [prog_add_v0, prog_add_v1, prog_add_v2]


# ── 1b. SUBTRACT TWO NUMBERS ─────────────────────────────────────────
# Compute a-b, store at DATA_BASE. Mirrors the ADD family.

def prog_sub_v0(a: int, b: int) -> bytes:
    ops = [
        addi(5, 0, a),
        addi(6, 0, b),
        sub(7, 5, 6),
        lui(10, DATA_BASE >> 12),
        sw(7, 10, 0),
        halt(),
    ]
    return pack(*ops)


def prog_sub_v1(a: int, b: int) -> bytes:
    """Different register layout: x15/x16 → x17."""
    ops = [
        addi(15, 0, a),
        addi(16, 0, b),
        sub(17, 15, 16),
        lui(18, DATA_BASE >> 12),
        sw(17, 18, 0),
        halt(),
    ]
    return pack(*ops)


SUB_VARIANTS = [prog_sub_v0, prog_sub_v1]


# ── 1c. DISPLAY BUFFER (ASCII) ───────────────────────────────────────
# Writes each character of `text` as one 32-bit word at DISPLAY_BASE + 4*i.
# Uses lui to materialise the base pointer; then (addi imm) + sw for each
# character. No loops — fixed unrolled program.
DISPLAY_BASE = 0x6000                    # 0x5000 + 0x1000; inside data region
DISPLAY_OFFSET = DISPLAY_BASE - DATA_BASE


def prog_display_v0(text: str) -> bytes:
    """Unroll: load display ptr into x10, then for each char: addi then sw."""
    ops = [lui(10, DISPLAY_BASE >> 12)]
    for i, ch in enumerate(text):
        ops.append(addi(5, 0, ord(ch)))
        ops.append(sw(5, 10, 4 * i))
    ops.append(halt())
    return pack(*ops)


def prog_display_v1(text: str) -> bytes:
    """Different register layout (x15 for val, x16 for ptr)."""
    ops = [lui(16, DISPLAY_BASE >> 12)]
    for i, ch in enumerate(text):
        ops.append(addi(15, 0, ord(ch)))
        ops.append(sw(15, 16, 4 * i))
    ops.append(halt())
    return pack(*ops)


DISPLAY_VARIANTS = [prog_display_v0, prog_display_v1]


# ── 2. FACTORIAL ─────────────────────────────────────────────────────
# n! via nested loops. Outer iterates i=2..n; inner multiplies via
# repeated addition (no MUL in RV32I).

def prog_factorial_v0(n: int) -> bytes:
    ops = [addi(5, 0, n), addi(6, 0, 1), addi(7, 0, 2)]
    outer = _addr(ops)
    ops.append(0)                             # placeholder: outer exit
    outer_idx = len(ops) - 1
    ops += [addi(8, 0, 0), addi(9, 0, 0)]     # tmp=0, j=0
    inner = _addr(ops)
    ops.append(0)                             # placeholder: inner exit
    inner_idx = len(ops) - 1
    ops += [add(8, 8, 6), addi(9, 9, 1)]
    ops.append(jal(0, inner - _addr(ops)))    # jump back to inner
    inner_done = _addr(ops)
    ops += [add(6, 8, 0), addi(7, 7, 1)]
    ops.append(jal(0, outer - _addr(ops)))    # jump back to outer
    done = _addr(ops)
    ops += [lui(10, DATA_BASE >> 12), sw(6, 10, 0), halt()]
    ops[outer_idx] = blt(5, 7, done - 4 * outer_idx)
    ops[inner_idx] = bge(9, 7, inner_done - 4 * inner_idx)
    return pack(*ops)


def prog_factorial_v1(n: int) -> bytes:
    """Different register choices: t-regs shifted by 10."""
    ops = [addi(15, 0, n), addi(16, 0, 1), addi(17, 0, 2)]
    outer = _addr(ops)
    ops.append(0); outer_idx = len(ops) - 1
    ops += [addi(18, 0, 0), addi(19, 0, 0)]
    inner = _addr(ops)
    ops.append(0); inner_idx = len(ops) - 1
    ops += [add(18, 18, 16), addi(19, 19, 1)]
    ops.append(jal(0, inner - _addr(ops)))
    inner_done = _addr(ops)
    ops += [add(16, 18, 0), addi(17, 17, 1)]
    ops.append(jal(0, outer - _addr(ops)))
    done = _addr(ops)
    ops += [lui(20, DATA_BASE >> 12), sw(16, 20, 0), halt()]
    ops[outer_idx] = blt(15, 17, done - 4 * outer_idx)
    ops[inner_idx] = bge(19, 17, inner_done - 4 * inner_idx)
    return pack(*ops)


FACTORIAL_VARIANTS = [prog_factorial_v0, prog_factorial_v1]


# ── 3. FIBONACCI ─────────────────────────────────────────────────────
# Store first N terms at DATA_BASE..DATA_BASE+4*(N-1).

def prog_fib_v0(n: int) -> bytes:
    ops = [
        addi(5, 0, 0),                     # a = 0
        addi(6, 0, 1),                     # b = 1
        addi(7, 0, 0),                     # i
        addi(8, 0, n),                     # limit
        lui(9, DATA_BASE >> 12),           # ptr
    ]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1  # placeholder: exit
    ops += [
        sw(5, 9, 0),                       # store a
        add(10, 5, 6),                     # tmp = a + b
        add(5, 6, 0),                      # a = b
        add(6, 10, 0),                     # b = tmp
        addi(9, 9, 4),                     # ptr += 4
        addi(7, 7, 1),                     # i++
    ]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops.append(halt())
    ops[exit_idx] = beq(7, 8, done - 4 * exit_idx)
    return pack(*ops)


def prog_fib_v1(n: int) -> bytes:
    """Use bne-exit form and different registers."""
    ops = [
        addi(15, 0, 0),
        addi(16, 0, 1),
        addi(17, 0, n),                    # remaining count
        lui(18, DATA_BASE >> 12),
    ]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [
        sw(15, 18, 0),
        add(19, 15, 16),
        add(15, 16, 0),
        add(16, 19, 0),
        addi(18, 18, 4),
        addi(17, 17, -1),
    ]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops.append(halt())
    ops[exit_idx] = beq(17, 0, done - 4 * exit_idx)
    return pack(*ops)


FIB_VARIANTS = [prog_fib_v0, prog_fib_v1]


# ── 4. COUNTDOWN ─────────────────────────────────────────────────────
# Store N, N-1, ..., 1 consecutively at DATA_BASE.

def prog_countdown_v0(n: int) -> bytes:
    ops = [
        addi(5, 0, n),
        lui(6, DATA_BASE >> 12),
    ]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [
        sw(5, 6, 0),
        addi(6, 6, 4),
        addi(5, 5, -1),
    ]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops.append(halt())
    ops[exit_idx] = beq(5, 0, done - 4 * exit_idx)
    return pack(*ops)


def prog_countdown_v1(n: int) -> bytes:
    """Use blt-exit: x5 < 1 → done. Pointer advances after the store."""
    ops = [
        addi(15, 0, n),
        addi(16, 0, 1),
        lui(17, DATA_BASE >> 12),
    ]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [
        sw(15, 17, 0),
        addi(17, 17, 4),
        addi(15, 15, -1),
    ]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops.append(halt())
    ops[exit_idx] = blt(15, 16, done - 4 * exit_idx)   # if x15 < 1 → done
    return pack(*ops)


COUNTDOWN_VARIANTS = [prog_countdown_v0, prog_countdown_v1]


# ── 5. SUM 1..N ──────────────────────────────────────────────────────
# sum = 1 + 2 + ... + N; stored at DATA_BASE.

def prog_sum_v0(n: int) -> bytes:
    ops = [
        addi(5, 0, 0),           # sum
        addi(6, 0, 1),           # i
        addi(7, 0, n + 1),       # limit
    ]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [
        add(5, 5, 6),
        addi(6, 6, 1),
    ]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops += [lui(8, DATA_BASE >> 12), sw(5, 8, 0), halt()]
    ops[exit_idx] = beq(6, 7, done - 4 * exit_idx)
    return pack(*ops)


def prog_sum_v1(n: int) -> bytes:
    """Count down from N to 1, summing. Different reg layout."""
    ops = [
        addi(15, 0, 0),          # sum
        addi(16, 0, n),          # i
    ]
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [
        add(15, 15, 16),
        addi(16, 16, -1),
    ]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops += [lui(17, DATA_BASE >> 12), sw(15, 17, 0), halt()]
    ops[exit_idx] = beq(16, 0, done - 4 * exit_idx)
    return pack(*ops)


SUM_VARIANTS = [prog_sum_v0, prog_sum_v1]


# ── 6. MAX OF TWO ────────────────────────────────────────────────────
# max(a, b), stored at DATA_BASE.

def prog_max_v0(a: int, b: int) -> bytes:
    ops = [addi(5, 0, a), addi(6, 0, b)]
    ops.append(0); branch_idx = len(ops) - 1   # placeholder: blt → else
    # then: x7 = x5 (a ≥ b → max = a)
    ops += [add(7, 5, 0)]
    ops.append(0); skip_idx = len(ops) - 1     # placeholder: jal → end
    else_ = _addr(ops)
    ops += [add(7, 6, 0)]                      # x7 = x6
    end = _addr(ops)
    ops += [lui(8, DATA_BASE >> 12), sw(7, 8, 0), halt()]
    ops[branch_idx] = blt(5, 6, else_ - 4 * branch_idx)   # if a<b goto else
    ops[skip_idx] = jal(0, end - 4 * skip_idx)
    return pack(*ops)


def prog_max_v1(a: int, b: int) -> bytes:
    """bge instead of blt; 'else' is first in memory."""
    ops = [addi(15, 0, a), addi(16, 0, b)]
    ops.append(0); branch_idx = len(ops) - 1   # placeholder: bge → then
    # "else" first: x17 = x16 (because a < b → max = b)
    ops += [add(17, 16, 0)]
    ops.append(0); skip_idx = len(ops) - 1
    then_ = _addr(ops)
    ops += [add(17, 15, 0)]                    # x17 = x15
    end = _addr(ops)
    ops += [lui(18, DATA_BASE >> 12), sw(17, 18, 0), halt()]
    ops[branch_idx] = bge(15, 16, then_ - 4 * branch_idx)
    ops[skip_idx] = jal(0, end - 4 * skip_idx)
    return pack(*ops)


MAX_VARIANTS = [prog_max_v0, prog_max_v1]


# ── 7. COPY MEMORY BLOCK ─────────────────────────────────────────────
# Copy N 32-bit words from src=DATA_BASE+0x100 to dst=DATA_BASE+0x200.
# Verifier pre-seeds src with values 1..N; program is a pure memcpy loop.

SRC_OFFSET = 0x100
DST_OFFSET = 0x200


def _load_addr(ops: list[int], rd: int, addr: int) -> None:
    """Materialise a 32-bit absolute address into rd via LUI + ADDI.
    Handles the sign-extension on ADDI immediate correctly."""
    hi = (addr + 0x800) >> 12
    lo = addr - (hi << 12)
    lo_u = lo & 0xFFF
    ops.append(lui(rd, hi))
    if lo_u != 0:
        ops.append(addi(rd, rd, lo))


def prog_memcpy_v0(n: int) -> bytes:
    ops = []
    _load_addr(ops, 5, DATA_BASE + SRC_OFFSET)
    _load_addr(ops, 6, DATA_BASE + DST_OFFSET)
    ops.append(addi(7, 0, 0))               # i = 0
    ops.append(addi(8, 0, n))               # limit
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [
        lw(9, 5, 0),
        sw(9, 6, 0),
        addi(5, 5, 4),
        addi(6, 6, 4),
        addi(7, 7, 1),
    ]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops.append(halt())
    ops[exit_idx] = beq(7, 8, done - 4 * exit_idx)
    return pack(*ops)


def prog_memcpy_v1(n: int) -> bytes:
    """Count-down form: i = n → 0."""
    ops = []
    _load_addr(ops, 15, DATA_BASE + SRC_OFFSET)
    _load_addr(ops, 16, DATA_BASE + DST_OFFSET)
    ops.append(addi(17, 0, n))              # remaining
    loop = _addr(ops)
    ops.append(0); exit_idx = len(ops) - 1
    ops += [
        lw(18, 15, 0),
        sw(18, 16, 0),
        addi(15, 15, 4),
        addi(16, 16, 4),
        addi(17, 17, -1),
    ]
    ops.append(jal(0, loop - _addr(ops)))
    done = _addr(ops)
    ops.append(halt())
    ops[exit_idx] = beq(17, 0, done - 4 * exit_idx)
    return pack(*ops)


MEMCPY_VARIANTS = [prog_memcpy_v0, prog_memcpy_v1]


# ── 8. FUNCTION CALL AND RETURN ──────────────────────────────────────
# Double a value via a subroutine: fn(x) = x + x. Store fn(n) at DATA_BASE.

def prog_call_v0(n: int) -> bytes:
    ops = [
        addi(10, 0, n),                        # a0 = n
    ]
    call_idx = len(ops); ops.append(0)         # placeholder: jal ra, fn
    ops += [
        lui(11, DATA_BASE >> 12),
        sw(10, 11, 0),
        halt(),
    ]
    fn = _addr(ops)
    ops += [
        add(10, 10, 10),                       # a0 *= 2
        jalr(0, 1, 0),                         # return (ret = jalr x0, ra, 0)
    ]
    ops[call_idx] = jal(1, fn - 4 * call_idx)
    return pack(*ops)


def prog_call_v1(n: int) -> bytes:
    """Subroutine before main: initial jal skips over it."""
    ops = [0]                                  # placeholder: jal to main
    entry_skip = 0
    fn = _addr(ops)
    ops += [
        add(10, 10, 10),
        jalr(0, 1, 0),
    ]
    main = _addr(ops)
    ops[entry_skip] = jal(0, main - 0)         # from addr 0 to main
    ops.append(addi(10, 0, n))
    call_idx = len(ops); ops.append(0)
    ops += [
        lui(11, DATA_BASE >> 12),
        sw(10, 11, 0),
        halt(),
    ]
    ops[call_idx] = jal(1, fn - 4 * call_idx)
    return pack(*ops)


CALL_VARIANTS = [prog_call_v0, prog_call_v1]


# ── Phrasings ────────────────────────────────────────────────────────

NUM_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 20: "twenty", 25: "twenty-five", 30: "thirty",
    42: "forty-two", 50: "fifty", 100: "one hundred",
}


def _w(n: int) -> str:
    return NUM_WORDS.get(n, str(n))


def add_phrasings(a: int, b: int) -> list[str]:
    aw, bw = _w(a), _w(b)
    return [
        # A-bucket: raw arithmetic forms
        f"add {a} and {b} and store the result",
        f"add {a} plus {b} and store it",
        f"compute {a} + {b} and save it to memory",
        f"{a} + {b}",
        f"{a}+{b}",
        f"{a} plus {b}",
        f"what is {a} + {b}",
        f"{a}+{b} = ?",
        f"{a} + {b} = ?",
        f"evaluate {a} + {b}",
        f"solve {a} + {b}",
        f"resolve the expression {a} + {b}",
        f"simplify {a} + {b}",
        f"compute the sum {a} + {b}",
        f"work out {a} + {b}",
        f"calculate {a} + {b}",
        # B-bucket: English sentence forms
        f"add {a} to {b}",
        f"sum {a} and {b} and write it out",
        f"store the sum of {a} and {b}",
        f"add the numbers {a} and {b}",
        f"sum of {a} and {b}",
        f"compute {a} plus {b}",
        f"the result of adding {a} and {b}",
        f"the sum of {a} and {b}",
        f"the total of {a} and {b}",
        f"combine {a} and {b}",
        f"please add {a} and {b}",
        f"please compute {a} + {b}",
        f"kindly add {a} and {b}",
        f"can you add {a} and {b} for me",
        # C-bucket: destination-explicit
        f"store {a}+{b}",
        f"save {a} + {b} to memory",
        f"put {a} + {b} in memory",
        f"write {a} + {b} to memory",
        f"record {a} + {b} to memory",
        f"persist {a} + {b} to memory",
        f"log the sum {a} + {b}",
        f"save addition result for {a} and {b}",
        f"deposit {a}+{b} to the data region",
        f"commit {a} + {b} to memory",
        # D-bucket: number-word forms
        f"add {aw} and {bw}",
        f"{aw} plus {bw}",
        f"add {aw} plus {bw}",
        f"sum {aw} and {bw}",
        f"what's {aw} plus {bw}",
        f"compute {aw} + {bw}",
        f"the sum {aw} plus {bw}",
        # E-bucket: programming / pseudo-code
        f"simple addition: {a} + {b}",
        f"addition: {a}, {b}",
        f"add(a={a}, b={b})",
        f"add({a}, {b})",
        f"sum({a}, {b})",
        f"compute add({a},{b})",
        f"let result = {a} + {b}; save",
        f"x = {a} + {b}; store x",
        f"result := {a} + {b}",
        f"out = {a} + {b}",
        # F-bucket: word problems / scenarios
        f"{a} added to {b}",
        f"{a} incremented by {b}",
        f"take {a} and add {b}",
        f"take {a}, then add {b}",
        f"starting from {a} add {b}",
        f"begin with {a} and add {b}",
        f"{a} together with {b}",
        f"aggregate {a} with {b}",
        f"{a} combined with {b}",
        f"unite {a} and {b} by addition",
        # G-bucket: casual / terse / stylistic
        f"hey add {a} and {b}",
        f"just add {a} + {b}",
        f"quick addition {a} + {b}",
        f"quickly sum {a} and {b}",
        f"do an add of {a} and {b}",
        f"execute addition of {a} and {b}",
        f"perform {a} + {b}",
        f"carry out {a} + {b}",
        f"run the addition {a} + {b}",
        f"addition op with operands {a}, {b}",
    ]


def sub_phrasings(a: int, b: int) -> list[str]:
    """a - b. Note the asymmetry: order of operands matters."""
    aw, bw = _w(a), _w(b)
    return [
        # A: "a minus b" direct
        f"subtract {b} from {a} and store the result",
        f"subtract {b} from {a}",
        f"{a} minus {b}",
        f"{a} - {b}",
        f"compute {a} - {b}",
        f"compute {a} minus {b}",
        f"what is {a} - {b}",
        f"what is {a} minus {b}",
        f"evaluate {a} - {b}",
        f"solve {a} - {b}",
        # B: save / store focused
        f"save {a} - {b} to memory",
        f"store {a} minus {b}",
        f"write {a} - {b} to memory",
        f"record {a} - {b}",
        f"persist {a} - {b} to memory",
        f"output {a} - {b}",
        f"commit {a} - {b} to memory",
        f"deposit {a}-{b}",
        f"log the difference {a} - {b}",
        f"store the difference of {a} and {b}",
        # C: difference framing
        f"the difference between {a} and {b}",
        f"the difference of {a} and {b}",
        f"{a} take away {b}",
        f"{a} less {b}",
        f"{a} decreased by {b}",
        f"{a} reduced by {b}",
        f"from {a} remove {b}",
        f"take {b} away from {a}",
        f"remove {b} from {a}",
        # D: programming idioms
        f"sub({a}, {b})",
        f"subtract(a={a}, b={b})",
        f"difference({a}, {b})",
        f"{a} - {b} = ?",
        f"let r = {a} - {b}; save r",
        f"result = {a} - {b}",
        f"out = {a} - {b}",
        # E: number-word
        f"{aw} minus {bw}",
        f"subtract {bw} from {aw}",
        f"{aw} take away {bw}",
        f"the difference between {aw} and {bw}",
        f"compute {aw} minus {bw}",
        f"{aw} less {bw}",
        # F: polite / imperative
        f"please subtract {b} from {a}",
        f"kindly subtract {b} from {a}",
        f"perform subtraction: {a} - {b}",
        f"execute {a} - {b}",
        f"run the subtraction {a} - {b}",
        f"do a subtract {a} - {b}",
        f"calculate {a} - {b}",
        f"work out {a} - {b}",
        # G: sentence forms
        f"starting from {a} subtract {b}",
        f"begin with {a} and subtract {b}",
        f"start at {a}, remove {b}",
        f"given {a}, subtract {b}",
        f"with x = {a}, compute x - {b}",
        f"decrement {a} by {b}",
        f"{a} decremented by {b}",
        f"reduce {a} by {b}",
        # H: short/casual
        f"{a}-{b}",
        f"minus: {a}, {b}",
        f"diff {a} {b}",
        f"sub {a} {b}",
        f"{a} sub {b}",
        f"subtract {b} off {a}",
        f"hey subtract {b} from {a}",
        f"quick subtraction {a} - {b}",
        f"simple subtraction: {a} - {b}",
        f"subtraction op with operands {a}, {b}",
    ]


def display_phrasings(text: str) -> list[str]:
    """Phrasings for: write `text` to the display buffer as ASCII."""
    # Quote-or-not variants to expose the tokenizer to multiple forms.
    t = text
    qt = f'"{text}"'
    return [
        # Direct
        f"say {t}",
        f"display {t}",
        f"print {t}",
        f"show {t}",
        f"write {t}",
        f"output {t}",
        f"say {qt}",
        f"print {qt}",
        f"display {qt}",
        f"show {qt}",
        # Explicit destination
        f"write {t} to the screen",
        f"display {t} on screen",
        f"print {t} on screen",
        f"show {t} on the display",
        f"write {t} to display",
        f"put {t} on the display",
        f"write {t} to the display buffer",
        f"output {t} to the display",
        f"render {t} on screen",
        f"draw {t} on screen",
        # ASCII / byte framing
        f"write the ASCII for {t}",
        f"emit ASCII {t}",
        f"output the ASCII bytes for {t}",
        f"put the characters of {t} in the display buffer",
        f"write each character of {t} one per word",
        f"store ASCII bytes: {t}",
        f"emit the characters of {t}",
        # Imperative / polite
        f"please display {t}",
        f"please print {t}",
        f"kindly display {t}",
        f"can you show {t}",
        f"let's display {t}",
        f"I want to see {t}",
        # Short / quirky
        f"{qt}",
        f"screen: {t}",
        f"display := {t}",
        f"stdout: {t}",
        f"echo {t}",
        f"echo {qt}",
        f"println({qt})",
        f"print({qt})",
        f"puts({qt})",
    ]


def factorial_phrasings(n: int) -> list[str]:
    nw = _w(n)
    return [
        # A-bucket: direct
        f"compute {n} factorial and store it",
        f"calculate {n}!",
        f"store {n}!",
        f"compute factorial of {n}",
        f"factorial of {n}",
        f"what is {n} factorial",
        f"{n}!",
        f"{n} factorial",
        f"find {n}!",
        f"evaluate {n}!",
        # B-bucket: calculate/save style
        f"calculate the factorial of {n}",
        f"save factorial of {n}",
        f"write {n}! to memory",
        f"compute the factorial {n}!",
        f"save the factorial of {n}",
        f"the factorial of {n}",
        f"store the factorial of {n}",
        f"work out {n} factorial",
        f"please compute {n}!",
        f"factorial({n})",
        # C-bucket: number-word + misc
        f"compute {nw} factorial",
        f"{nw}!",
        f"{nw} factorial",
        f"calculate {nw} factorial and save",
        f"the factorial of {nw}",
        f"store {nw} factorial",
        f"what is {nw} factorial",
        f"compute factorial of {nw}",
        # D-bucket: programming idioms
        f"fact({n})",
        f"fact({n}) = ?",
        f"n! with n = {n}",
        f"n! for n={n}",
        f"compute n! where n = {n}",
        f"let n={n}; compute n!",
        f"def fact: return n!; fact({n})",
        f"reduce(*, 1..{n})",
        f"product from 1 to {n}",
        f"multiply 1 through {n}",
        f"the product of integers from 1 to {n}",
        # E-bucket: symbolic / quirky
        f"{n} ! =",
        f"{n} !",
        f"the value of {n}!",
        f"the result of {n}!",
        f"evaluate the factorial {n}!",
        f"simplify {n}!",
        f"resolve {n}!",
        f"{n} bang",
        f"{n} factorial please",
        f"kindly compute {n}!",
        # F-bucket: math description
        f"compute {n} × ({n}-1) × ... × 1",
        f"multiply the first {n} positive integers",
        f"save the product 1*2*...*{n}",
        f"store 1 times 2 times ... times {n}",
        f"the product 1·2·…·{n}",
        f"compute the descending product starting at {n}",
        f"iterate 1..{n} and multiply",
        f"fold (*) over [1..{n}]",
        f"store the {n}-factorial",
        f"compute {n} factorial via repeated addition",
        # G-bucket: longer sentences
        f"find the factorial of {n} and save it to memory",
        f"please calculate the factorial of {n}",
        f"compute the factorial for the number {n}",
        f"given n = {n}, store n!",
        f"the factorial of the number {n}",
        f"what's the factorial of {n}",
        f"output {n}! to memory",
        f"deposit {n}! in memory",
        f"persist the factorial of {n}",
        f"compute and save {n}!",
        # H: programming-prompt flavor
        f"run the factorial program with n={n}",
        f"execute factorial({n})",
        f"invoke factorial with argument {n}",
        f"call factorial on {n}",
        f"compute factorial(n={n})",
    ]


def fib_phrasings(n: int) -> list[str]:
    nw = _w(n)
    return [
        # A: "first N" style
        f"store the first {n} Fibonacci numbers",
        f"write the first {n} Fibonacci numbers to memory",
        f"first {n} Fibonacci numbers",
        f"the first {n} Fibonacci numbers",
        f"the opening {n} Fibonacci numbers",
        f"first {n} terms of Fibonacci",
        f"the first {n} fib terms",
        f"first {n} of the Fibonacci sequence",
        f"save the first {n} Fibonacci terms",
        f"save the first {nw} Fibonacci numbers",
        # B: "N terms" style
        f"compute {n} terms of the Fibonacci sequence",
        f"generate {n} Fibonacci terms",
        f"Fibonacci {n} terms",
        f"Fibonacci, {n} terms",
        f"compute the Fibonacci sequence, {n} terms",
        f"please compute {n} Fibonacci terms",
        f"the Fibonacci series, first {n}",
        f"produce the first {n} fib terms",
        f"emit {n} Fibonacci numbers",
        f"dump {n} Fibonacci numbers",
        # C: shorthand / programming
        f"fibonacci({n})",
        f"fib({n})",
        f"fib seq n={n}",
        f"fib sequence length {n}",
        f"store {n} fib numbers",
        f"write {n} terms of fib to memory",
        f"compute {n} fib",
        f"compute {n} Fib",
        f"{n}-term Fibonacci",
        f"Fibonacci: first {n} entries",
        # D: number-word
        f"the first {nw} Fibonacci terms",
        f"first {nw} Fibonacci numbers",
        f"store {nw} Fibonacci numbers",
        f"compute {nw} Fibonacci terms",
        f"save the first {nw} Fibonacci terms",
        f"compute the first {nw} fib terms",
        f"the opening {nw} Fibonacci numbers",
        # E: descriptive
        f"make a Fibonacci sequence of length {n}",
        f"build a Fibonacci sequence with {n} numbers",
        f"construct the Fibonacci sequence for {n} terms",
        f"generate a Fibonacci series of length {n}",
        f"produce a Fibonacci list of size {n}",
        f"the Fibonacci array of length {n}",
        f"the Fibonacci prefix of size {n}",
        f"the Fibonacci prefix of length {n}",
        # F: imperative
        f"compute fib for {n} terms and save",
        f"run fib for n = {n}",
        f"run fibonacci with count {n}",
        f"fibonacci with {n} iterations",
        f"iterate fib {n} times",
        f"execute fib({n})",
        f"call fibonacci({n})",
        f"invoke fib with size {n}",
        # G: math description
        f"each term is sum of two previous, length {n}",
        f"a_i = a_{{i-1}} + a_{{i-2}}, {n} terms",
        f"F(i) = F(i-1) + F(i-2), compute {n} terms",
        f"additive recurrence, first {n} terms",
        f"the Fibonacci recurrence, {n} outputs",
        f"Fibonacci recurrence for {n} steps",
        f"linear recurrence F_n, length {n}",
        f"golden-ratio sequence, {n} terms",
        # H: store-focused
        f"save the Fibonacci sequence up to {n} terms",
        f"persist {n} Fibonacci numbers",
        f"deposit {n} Fibonacci numbers",
        f"record the first {n} Fibonacci values",
        f"commit {n} fib numbers to memory",
        f"write {n} fib values",
        f"output {n} Fibonacci numbers",
        f"log {n} fib numbers",
        f"dump the first {n} fib numbers",
        f"store the fib prefix for n={n}",
        f"fib sequence: save {n} elements",
    ]


def countdown_phrasings(n: int) -> list[str]:
    nw = _w(n)
    return [
        # A: classic countdown
        f"count down from {n} to 1 and store each value",
        f"count down from {n} to one",
        f"count down from {n} to 1",
        f"count {n} down to 1",
        f"counting down from {n}",
        f"counting down from {nw}",
        f"count down starting at {n}",
        f"countdown from {n} to 1",
        f"countdown {n}",
        f"countdown {nw} to one",
        # B: ranges and sequences
        f"from {n} down to 1",
        f"{n} down to 1",
        f"{n} downto 1",
        f"store the countdown from {n}",
        f"a countdown {n} to 1",
        f"numbers from {n} down to 1",
        f"descending sequence {n} to 1",
        f"reverse count from {n} to 1",
        f"range {n}..1 descending",
        f"range {n} to 1 reversed",
        # C: imperative/store
        f"write the values {n} down to 1 to memory",
        f"dump {n} down to 1 in memory",
        f"list {n} down to 1 in memory",
        f"write {n}, {n-1}, ..., 1 to memory",
        f"store {n} {n-1} ... 1",
        f"save a countdown starting at {n}",
        f"save the countdown {n}..1",
        f"persist countdown from {n}",
        f"deposit countdown {n} to 1",
        f"output {n} down to 1",
        # D: loop-focused
        f"loop from {n} to 1 decrementing",
        f"decrement from {n} to 1 and save each",
        f"decrement {n} down to 1",
        f"decrement loop {n}..1",
        f"descending loop from {n}",
        f"iterate {n} down to 1",
        f"iterate from {n} decrementing",
        f"for i = {n}; i >= 1; i--",
        f"for i from {n} to 1 step -1",
        f"while n > 0 starting at {n}",
        # E: phrasing variety
        f"please count down from {n}",
        f"kindly count down from {n}",
        f"please store countdown {n}..1",
        f"run a countdown of {n}",
        f"do a countdown from {n}",
        f"execute countdown({n})",
        f"call countdown with n={n}",
        f"invoke a countdown starting at {n}",
        # F: number-word
        f"count down from {nw} to one",
        f"countdown {nw} to 1",
        f"from {nw} down to one",
        f"store {nw} down to 1",
        f"save a countdown of {nw}",
        f"write {nw} down to 1",
        # G: programming idioms
        f"reversed(range(1, {n+1}))",
        f"range({n}, 0, -1)",
        f"[{n}..1]",
        f"n..1 where n = {n}",
        f"for i in {n}..1",
        f"decr loop, start = {n}",
        f"desc count n={n}",
        # H: math/task description
        f"write a descending integer sequence from {n}",
        f"emit a descending run of {n} integers starting at {n}",
        f"output integers from {n} down to 1 in order",
        f"save the integers {n}, {n-1}, ..., 1",
        f"store decrementing integers starting at {n}",
        f"reverse enumeration from {n} to 1",
        f"a descending integer stream from {n} to 1",
        f"descending range starting at {n}",
        f"count back from {n}",
        f"count backward from {n} to 1",
    ]


def sum_phrasings(n: int) -> list[str]:
    nw = _w(n)
    return [
        # A-bucket: explicit series
        f"compute 1 + 2 + ... + {n} and store the sum",
        f"1 + 2 + ... + {n}",
        f"save 1+2+...+{n}",
        f"what is 1+2+...+{n}",
        f"evaluate 1 + 2 + ... + {n}",
        f"reduce 1 + 2 + ... + {n}",
        f"the series 1 + 2 + ... + {n}",
        f"series total 1 + 2 + ... + {n}",
        f"store 1+2+3+...+{n}",
        f"compute the progression 1..{n}",
        # B-bucket: range-style
        f"sum the numbers from 1 to {n}",
        f"store the sum 1 through {n}",
        f"compute the sum 1..{n}",
        f"sum of 1 to {n}",
        f"add up 1 through {n}",
        f"total of 1 through {n}",
        f"sum 1 to {n}",
        f"sum from 1 to {nw}",
        f"compute sum of 1..{n}",
        f"sum from 1 up to {n}",
        f"1 through {n} sum",
        f"integer sum 1..{n}",
        f"range sum [1,{n}]",
        f"reduce range 1 to {n} with addition",
        f"accumulate from 1 to {n}",
        # C-bucket: triangular-number framing
        f"triangular number {n}",
        f"the {n}th triangular number",
        f"compute triangular({n})",
        f"Gauss sum for n={n}",
        f"{n} choose 2 plus {n}",
        f"({n}*({n}+1))/2",
        f"n(n+1)/2 for n={n}",
        f"closed form: n(n+1)/2 at n={n}",
        f"partial sum of first {n} integers",
        f"sum of first {n} positive integers",
        # D-bucket: natural language
        f"the sum of integers 1 to {n}",
        f"add all numbers from 1 to {n}",
        f"add every integer from 1 through {n}",
        f"add each number from 1 to {n}",
        f"aggregate integers 1 to {n}",
        f"total the integers from 1 to {n}",
        f"count up and total 1 through {n}",
        f"roll up the sum from 1 to {n}",
        f"compute the total 1 through {n}",
        f"addition of 1 to {n}",
        f"sum integers 1 to {n}",
        # E-bucket: number-word / imperative
        f"sum from one to {nw}",
        f"add one through {nw}",
        f"compute sum(1..{nw})",
        f"the sum one plus two plus ... plus {nw}",
        f"please sum 1 to {n}",
        f"kindly total 1 through {n}",
        f"do a sum from 1 to {n}",
        f"run a sum over 1..{n}",
        f"execute sum 1 through {n}",
        # F-bucket: programming idioms
        f"store sum(1..{n})",
        f"sum(range(1, {n+1}))",
        f"reduce(+, 1..{n})",
        f"fold (+) over [1..{n}]",
        f"sigma from i=1 to {n} of i",
        f"Σ i for i in 1..{n}",
        f"for i in 1..{n}: total += i",
        f"accumulator over 1..{n}",
        f"running total 1 through {n}",
        f"iterate 1 to {n}, sum into memory",
        # G-bucket: save/store-focused
        f"sum 1..{n} and save it",
        f"write the sum 1..{n} to memory",
        f"persist sum(1..{n})",
        f"commit 1+2+...+{n} to memory",
        f"deposit the total of 1..{n}",
        f"log the sum for 1..{n}",
        f"record the triangular number {n}",
        f"save the integer sum 1 to {n}",
        f"output 1 + 2 + ... + {n} to memory",
        f"store the arithmetic series total for 1..{n}",
    ]


def max_phrasings(a: int, b: int) -> list[str]:
    aw, bw = _w(a), _w(b)
    return [
        # A: direct
        f"find the max of {a} and {b} and store it",
        f"max of {a} and {b}",
        f"max({a}, {b})",
        f"compute max({a}, {b})",
        f"save max({a}, {b})",
        f"maximum({a}, {b})",
        f"maximum of {a} and {b}",
        f"the maximum of {a} and {b}",
        f"the max of {a} and {b}",
        f"{a} max {b}",
        # B: larger/greater
        f"store the larger of {a} and {b}",
        f"larger of {a} and {b}",
        f"the larger of {a} and {b}",
        f"the larger value of {a} and {b}",
        f"store the greater of {a} and {b}",
        f"the greater of {a} and {b}",
        f"the greater number of {a} and {b}",
        f"store the greater number {a} or {b}",
        f"pick the greater of {a} and {b}",
        f"pick the larger of {a} or {b}",
        # C: which is bigger
        f"which is bigger, {a} or {b}; store it",
        f"which of {a} and {b} is greater; save it",
        f"which is larger, {a} or {b}",
        f"which is greater: {a} or {b}",
        f"is {a} or {b} bigger? save it",
        f"bigger: {a} or {b}",
        f"bigger of {a} and {b}",
        f"bigger number between {a} and {b}",
        f"bigger value between {a} and {b}",
        f"whichever is bigger, {a} or {b}",
        # D: comparison framing
        f"branch on {a} vs {b} and keep the max",
        f"{a} vs {b}: keep the larger",
        f"{a} or {b}: store the larger",
        f"compare {a} and {b} and save the larger",
        f"if {a} > {b} else {b}",
        f"({a} if {a} > {b} else {b})",
        f"ternary max: {a}, {b}",
        f"branchy max({a}, {b})",
        f"conditional max of {a} and {b}",
        f"if-else max for {a} and {b}",
        # E: number-word
        f"larger of {aw} and {bw}",
        f"the greater of {aw} and {bw}",
        f"max of {aw} and {bw}",
        f"maximum of {aw} and {bw}",
        f"the bigger of {aw} and {bw}",
        f"which is bigger: {aw} or {bw}",
        f"store the larger of {aw} and {bw}",
        # F: imperative / save
        f"please compute max({a},{b})",
        f"kindly compute max({a},{b})",
        f"store whichever is greater, {a} or {b}",
        f"store the max of two numbers {a} and {b}",
        f"save the max of {a} and {b}",
        f"write max({a}, {b}) to memory",
        f"deposit max({a}, {b})",
        f"commit max({a}, {b})",
        f"output max({a}, {b})",
        f"log max({a}, {b})",
        # G: terse / stylized
        f"max {a} {b}",
        f"pick max {a} {b}",
        f"max, {a}, {b}",
        f"max → {a} | {b}",
        f"[{a}, {b}].max",
        f"max {a},{b}",
        f"max of ({a}, {b})",
        # H: return-style
        f"return max of {a} and {b}",
        f"return the larger of {a} and {b}",
        f"compute maximum of {a}, {b}",
        f"compute and return max({a},{b})",
        f"determine the max of {a} and {b}",
        f"find the larger number between {a} and {b}",
        f"find whichever is greater: {a} or {b}",
        f"pick the max between {a} and {b}",
        f"evaluate max({a}, {b})",
        f"resolve max({a}, {b})",
    ]


def memcpy_phrasings(n: int) -> list[str]:
    nw = _w(n)
    return [
        # A-bucket: src→dst explicit
        f"copy {n} words from source to destination",
        f"transfer {n} words from src to dst",
        f"copy {n} words from src buffer to dst buffer",
        f"move {n} words from source to destination",
        f"shuttle {n} words from src to dst",
        f"pipe {n} words from src to dst",
        f"forward {n} words from src to dst",
        f"migrate {n} words source → destination",
        f"relay {n} words src → dst",
        f"push {n} words from source into destination",
        # B-bucket: memcpy idiom
        f"memcpy {n} elements",
        f"memcpy size {n}",
        f"{n}-word memcpy",
        f"memory copy, {n} words",
        f"do a memcpy of {n} words",
        f"perform a memcpy of length {n}",
        f"memcpy(dst, src, {n})",
        f"call memcpy with count {n}",
        f"invoke memcpy({n})",
        f"execute memcpy for {n} words",
        # C-bucket: block/bulk
        f"copy a block of {n} words",
        f"block copy of {n} ints",
        f"bulk copy {n} elements",
        f"bulk transfer {n} words",
        f"block move of {n} words",
        f"batch copy {n} memory cells",
        f"batch transfer of {n} words",
        f"en masse copy {n} words",
        f"wholesale copy {n} words",
        f"in-bulk move {n} words",
        # D-bucket: generic verbs
        f"copy {n} memory words",
        f"move {n} words of memory",
        f"duplicate {n} words of data",
        f"replicate {n} words of memory",
        f"clone {n} memory slots",
        f"shift {n} words to destination",
        f"mirror {n} words into destination",
        f"propagate {n} words into destination",
        f"reproduce {n} words at destination",
        f"echo {n} words from src to dst",
        # E-bucket: size/length framing
        f"copy {n} values from one buffer to another",
        f"copy {n} 32-bit values",
        f"copy {n} 4-byte words",
        f"copy a buffer of {n} words",
        f"move a buffer of {n} ints",
        f"transfer a block of {n} words",
        f"array copy, length {n}",
        f"copy an array of {n} words",
        f"copy the first {n} words",
        f"copy {n} words to the destination buffer",
        # F-bucket: number-word / imperative
        f"copy {nw} words",
        f"move {nw} words",
        f"copy {nw} memory words",
        f"copy over {n} words of data",
        f"please copy {n} words",
        f"kindly copy {n} words",
        f"can you copy {n} words from src to dst",
        f"{n} words: copy src → dst",
        f"dst := src for {n} words",
        f"for i in 1..{n}: dst[i] = src[i]",
        # G-bucket: programming idioms
        f"copy_mem(src, dst, count={n})",
        f"memmove {n} words",
        f"loop copy {n} words",
        f"word-by-word copy, count {n}",
        f"iterate {n} times: load then store",
        f"load-store loop for {n} words",
        f"unaligned copy of {n} words",
        f"linear copy of {n} elements",
        f"sequential word copy, {n} iterations",
        f"copy exactly {n} machine words",
        f"buffer-to-buffer copy of {n} words",
        f"slide {n} words over to destination",
        f"relocate {n} words to destination",
        f"deploy {n} words to dst",
        f"ship {n} words from src to dst",
    ]


def call_phrasings(n: int) -> list[str]:
    nw = _w(n)
    return [
        # A: call-a-function explicit
        f"call a function that doubles {n} and store the result",
        f"use a subroutine to double {n}",
        f"double {n} via a function call",
        f"call fn({n}) where fn doubles its argument",
        f"subroutine call: double {n}",
        f"double {n} using a subroutine",
        f"please double {n} using a subroutine",
        f"double-via-subroutine with input {n}",
        f"call/return: double {n}",
        f"call a doubling subroutine with {n}",
        # B: jal/jalr phrasing
        f"jal to a doubler with {n}",
        f"use jal/jalr to double {n}",
        f"jal into the doubler with arg {n}",
        f"jalr-return doubler on {n}",
        f"use a jal to invoke the doubler for {n}",
        f"jump-and-link to a function that doubles {n}",
        f"function call via jal, arg = {n}",
        f"jal doubler({n})",
        # C: function-returns framing
        f"function returning 2x, arg = {n}",
        f"a subroutine that returns 2x, called with {n}",
        f"function call with arg {n} that returns 2x",
        f"a function returning 2*input, called with {n}",
        f"return 2n from fn, with n={n}",
        f"fn(x) = 2x, compute fn({n})",
        f"function f(n) = 2n, compute f({n})",
        f"call f(x)=2x on {n}",
        # D: invoke/call terse
        f"invoke a doubling function on {n}",
        f"invoke doubler({n})",
        f"call the doubler on {n}",
        f"call doubler({n})",
        f"call double({n})",
        f"doubling function called with {n}",
        f"function call to double {n}",
        f"invoke a function that doubles {n}",
        f"run the doubler on {n}",
        f"execute doubler({n})",
        # E: x*2 style
        f"{n} * 2 via a function",
        f"2 * {n} via a function call",
        f"{n} times 2 via subroutine",
        f"2x of {n} via function",
        f"compute 2*{n} using a function",
        f"compute {n}*2 with a subroutine",
        f"compute 2·{n} via call",
        f"compute {n}+{n} via a function",
        f"store 2 times {n} using a function",
        f"store 2*{n} via a subroutine",
        # F: pass arg style
        f"pass {n} to a doubler and save the result",
        f"pass arg {n} to a doubling fn",
        f"apply a doubling function to {n}",
        f"apply fn(x)=2x with x={n}",
        f"feed {n} into the doubler",
        f"pass {n} into a subroutine returning 2x",
        # G: number-word
        f"double {nw} using a subroutine",
        f"call a doubler on {nw}",
        f"invoke a doubling function on {nw}",
        f"a function call to double {nw}",
        f"use a subroutine to double {nw}",
        f"pass {nw} to a doubler",
        # H: programming idiom
        f"double({n}) via function",
        f"subroutine double({n})",
        f"proc double: return 2n; call on {n}",
        f"def double(x): return 2*x; double({n})",
        f"fn double(x) returns 2*x; invoked with {n}",
        f"ret = double({n}); save ret",
        f"let ret = fn({n}); store ret",
        f"call: doubler; arg: {n}",
        f"subroutine doubler; input {n}",
        f"calling convention double({n})",
    ]


# ── Verification ─────────────────────────────────────────────────────

def verify_program(program: bytes, expected: dict, max_cycles: int = 5000,
                   seed: dict | None = None) -> tuple[bool, str]:
    """Run program in unicorn, halt on self-jump (jal x0, 0).

    `expected` keys (all optional):
      - mem_word: (addr, val) — memory[addr:addr+4] as uint32 must equal val
      - reg: (idx, val) — x[idx] must equal val (unsigned)
      - mem_words: (addr, [v0, v1, ...]) — N consecutive words
    `seed`: {addr: [words...]} — pre-fill memory before running.
    """
    cpu = Rv32i()
    cpu.load_program(program)
    if seed:
        for addr, words in seed.items():
            data = b"".join(int(w & 0xFFFFFFFF).to_bytes(4, "little") for w in words)
            cpu.uc.mem_write(addr, data)
    halted = False
    for _ in range(max_cycles):
        try:
            instr = cpu.fetch()
        except Exception as e:
            return False, f"fetch error: {e}"
        if instr == HALT_INSTR:
            halted = True
            break
        if instr == 0x00000000:
            return False, f"hit 0x0 at pc={cpu.pc:#x}"
        try:
            cpu.step()
        except Exception as e:
            return False, f"step error at pc={cpu.pc:#x}: {e}"
    if not halted:
        return False, "did not halt within cycle budget"

    if "mem_word" in expected:
        addr, val = expected["mem_word"]
        got = cpu.mem_word(addr)
        if got != (val & 0xFFFFFFFF):
            return False, f"mem[{addr:#x}]={got} != {val}"
    if "mem_words" in expected:
        addr, vals = expected["mem_words"]
        for i, v in enumerate(vals):
            got = cpu.mem_word(addr + 4 * i)
            if got != (v & 0xFFFFFFFF):
                return False, f"mem[{addr+4*i:#x}]={got} != {v}"
    if "reg" in expected:
        idx, val = expected["reg"]
        got = cpu.reg(idx)
        if got != (val & 0xFFFFFFFF):
            return False, f"x{idx}={got} != {val}"
    return True, "ok"


# ── Task generation ──────────────────────────────────────────────────

def _add_tasks(tasks: list, label: str, rejected: dict,
               phrasings: list[str], variants: list, args: tuple,
               expected: dict, seed: dict | None = None) -> None:
    valid = []
    for v in variants:
        prog = v(*args)
        ok, reason = verify_program(prog, expected, seed=seed)
        if ok:
            valid.append(prog)
        else:
            rejected[label] = rejected.get(label, 0) + 1
    if not valid:
        return
    for i, phrase in enumerate(phrasings):
        tasks.append((phrase, valid[i % len(valid)]))


def generate_tasks() -> list[tuple[str, bytes]]:
    import math

    tasks: list[tuple[str, bytes]] = []
    rejected: dict[str, int] = {}

    # 1. ADD TWO
    add_pairs = [(a, b) for a in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                  15, 17, 20, 25, 30, 42, 50, 75, 100)
                        for b in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15,
                                  20, 25) if a + b <= 200]
    for a, b in add_pairs:
        expected = {"mem_word": (DATA_BASE, a + b)}
        _add_tasks(tasks, "add", rejected, add_phrasings(a, b),
                   ADD_VARIANTS, (a, b), expected)

    # 1b. SUBTRACT TWO (minimal: 25 pairs × 2 variants × ~75 phrasings)
    sub_pairs = [(10, 3), (12, 4), (15, 7), (20, 5), (25, 10), (30, 12),
                 (50, 17), (100, 25), (7, 2), (8, 5), (9, 1), (13, 4),
                 (17, 8), (20, 11), (25, 3), (30, 7), (40, 15), (50, 22),
                 (60, 33), (75, 40), (80, 25), (90, 45), (100, 50),
                 (11, 6), (14, 9)]
    for a, b in sub_pairs:
        expected = {"mem_word": (DATA_BASE, a - b)}
        _add_tasks(tasks, "sub", rejected, sub_phrasings(a, b),
                   SUB_VARIANTS, (a, b), expected)

    # 2. FACTORIAL
    for n in range(1, 11):                       # 1..10
        expected = {"mem_word": (DATA_BASE, math.factorial(n))}
        _add_tasks(tasks, "factorial", rejected, factorial_phrasings(n),
                   FACTORIAL_VARIANTS, (n,), expected)

    # 3. FIBONACCI
    def fib_seq(n):
        a, b, out = 0, 1, []
        for _ in range(n):
            out.append(a); a, b = b, a + b
        return out
    for n in range(3, 16):                       # 3..15
        expected = {"mem_words": (DATA_BASE, fib_seq(n))}
        _add_tasks(tasks, "fib", rejected, fib_phrasings(n),
                   FIB_VARIANTS, (n,), expected)

    # 4. COUNTDOWN
    for n in range(2, 26):                       # 2..25
        expected = {"mem_words": (DATA_BASE, list(range(n, 0, -1)))}
        _add_tasks(tasks, "countdown", rejected, countdown_phrasings(n),
                   COUNTDOWN_VARIANTS, (n,), expected)

    # 5. SUM 1..N
    for n in range(2, 26):                       # 2..25
        expected = {"mem_word": (DATA_BASE, n * (n + 1) // 2)}
        _add_tasks(tasks, "sum", rejected, sum_phrasings(n),
                   SUM_VARIANTS, (n,), expected)

    # 6. MAX OF TWO
    max_vals = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 20, 25, 30,
                40, 50, 75, 100)
    max_pairs = [(a, b) for a in max_vals for b in max_vals if a != b]
    for a, b in max_pairs:
        expected = {"mem_word": (DATA_BASE, max(a, b))}
        _add_tasks(tasks, "max", rejected, max_phrasings(a, b),
                   MAX_VARIANTS, (a, b), expected)

    # 7. MEMCPY — pre-seed src with values 1..n, verify dst matches.
    for n in range(2, 13):                       # 2..12
        src_addr = DATA_BASE + SRC_OFFSET
        dst_addr = DATA_BASE + DST_OFFSET
        seed_vals = list(range(1, n + 1))
        expected = {"mem_words": (dst_addr, seed_vals)}
        _add_tasks(tasks, "memcpy", rejected, memcpy_phrasings(n),
                   MEMCPY_VARIANTS, (n,), expected,
                   seed={src_addr: seed_vals})

    # 8. FUNCTION CALL
    for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 25, 30, 35, 40,
              42, 50, 60, 70, 75, 80, 90, 100):
        expected = {"mem_word": (DATA_BASE, 2 * n)}
        _add_tasks(tasks, "call", rejected, call_phrasings(n),
                   CALL_VARIANTS, (n,), expected)

    # 9. DISPLAY BUFFER — write ASCII bytes of `text` at 0x6000.
    display_texts = [
        "hi", "OK", "hello", "hey", "yes", "no", "bye", "42",
        "wow", "cat", "dog", "sun", "moon", "star", "fire",
        "1", "7", "0", "9", "12", "99", "100",
        "H", "A", "Z", "q",
        "hola", "salut", "test", "abc", "xyz", "code", "data",
        "ready", "done", "fail", "error", "ok!", "go",
        "pi", "e", "id", "net", "cpu", "mem", "bit",
        "ship", "ping", "pong", "foo", "bar", "baz",
        "1234", "2025", "year", "time", "day", "now",
        "H!", "OK.", "END", "RUN", "HI5", "ACK", "nak",
        "name", "type", "flag", "call", "exit",
        "3.14", "root", "home", "user", "x5",
        "A0", "AZ", "99!", "yeah", "nope", "ugh", "hmm",
    ]
    for text in display_texts:
        chars = [ord(c) for c in text]
        expected = {"mem_words": (DISPLAY_BASE, chars)}
        _add_tasks(tasks, "display", rejected, display_phrasings(text),
                   DISPLAY_VARIANTS, (text,), expected)

    print(f"  Generated {len(tasks)} tasks")
    if rejected:
        print(f"  Rejected variants: {rejected}")
    else:
        print(f"  All variants verified")
    by_len: dict[str, int] = {}
    for _, prog in tasks:
        key = f"{len(prog)//4}-op"
        by_len[key] = by_len.get(key, 0) + 1
    print(f"  Programs by length: {dict(sorted(by_len.items(), key=lambda kv: int(kv[0].split('-')[0])))}")
    return tasks
