"""
RV32I program templates + natural-language phrasings + task generator.

Each template is 5-20 instructions. The verifier runs every candidate
through unicorn and checks the post-halt register/memory state; variants
that don't match are rejected before they ever reach training.
"""

from .riscv import (
    DATA_BASE, HALT_INSTR, PROGRAM_START, Rv32i,
    add, addi, bge, beq, blt, bne, halt, jal, jalr, lui, lw, pack, sw,
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
        f"add {a} and {b} and store the result",
        f"add {a} plus {b} and store it",
        f"compute {a} + {b} and save it to memory",
        f"add {a} to {b}",
        f"sum {a} and {b} and write it out",
        f"what is {a} + {b}",
        f"store the sum of {a} and {b}",
        f"{a} + {b}",
        f"{a} plus {b}",
        f"add the numbers {a} and {b}",
        f"add {aw} and {bw}",
        f"sum of {a} and {b}",
        f"work out {a} + {b}",
        f"compute {a} plus {b}",
        f"calculate {a} + {b}",
        f"please add {a} and {b}",
        f"the result of adding {a} and {b}",
        f"{aw} plus {bw}",
        f"store {a}+{b}",
        f"save {a} + {b} to memory",
        f"put {a} + {b} in memory",
        f"compute the sum {a} + {b}",
        f"simple addition: {a} + {b}",
        f"{a} added to {b}",
        f"{a}+{b} = ?",
    ]


def factorial_phrasings(n: int) -> list[str]:
    nw = _w(n)
    return [
        f"compute {n} factorial and store it",
        f"calculate {n}!",
        f"store {n}!",
        f"compute factorial of {n}",
        f"factorial of {n}",
        f"what is {n} factorial",
        f"compute {nw} factorial",
        f"{n}!",
        f"{n} factorial",
        f"calculate the factorial of {n}",
        f"find {n}!",
        f"save factorial of {n}",
        f"write {n}! to memory",
        f"compute n! for n = {n}",
        f"compute the factorial {n}!",
        f"save the factorial of {n}",
        f"the factorial of {n}",
        f"{nw}!",
        f"store the factorial of {n}",
        f"work out {n} factorial",
        f"please compute {n}!",
        f"factorial({n})",
        f"calculate {nw} factorial and save",
        f"{n} ! =",
        f"the value of {n}!",
    ]


def fib_phrasings(n: int) -> list[str]:
    nw = _w(n)
    return [
        f"store the first {n} Fibonacci numbers",
        f"write the first {n} Fibonacci numbers to memory",
        f"compute {n} terms of the Fibonacci sequence",
        f"generate {n} Fibonacci terms",
        f"first {n} Fibonacci numbers",
        f"the first {nw} Fibonacci terms",
        f"save the Fibonacci sequence up to {n} terms",
        f"Fibonacci {n} terms",
        f"fibonacci({n})",
        f"compute the Fibonacci sequence, {n} terms",
        f"store {n} fib numbers",
        f"fib sequence length {n}",
        f"write {n} terms of fib to memory",
        f"produce the first {n} fib terms",
        f"dump {n} Fibonacci numbers",
        f"Fibonacci: first {n} entries",
        f"compute {n} Fib",
        f"the Fibonacci series, first {n}",
        f"emit {n} Fibonacci numbers",
        f"make a Fibonacci sequence of length {n}",
        f"save the first {nw} Fibonacci numbers",
        f"please compute {n} Fibonacci terms",
        f"first {n} of the Fibonacci sequence",
        f"the opening {n} Fibonacci numbers",
        f"Fibonacci, {n} terms",
    ]


def countdown_phrasings(n: int) -> list[str]:
    nw = _w(n)
    return [
        f"count down from {n} to 1 and store each value",
        f"count down from {n} to one",
        f"write the values {n} down to 1 to memory",
        f"store the countdown from {n}",
        f"countdown from {n} to 1",
        f"dump {n} down to 1 in memory",
        f"count {n} down to 1",
        f"loop from {n} to 1 decrementing",
        f"decrement from {n} to 1 and save each",
        f"save a countdown starting at {n}",
        f"countdown {n}",
        f"counting down from {nw}",
        f"list {n} down to 1 in memory",
        f"numbers from {n} down to 1",
        f"reverse count from {n} to 1",
        f"descending sequence {n} to 1",
        f"write {n}, {n-1}, ..., 1 to memory",
        f"from {n} down to 1",
        f"n down to 1 where n = {n}",
        f"decrement {n} down to 1",
        f"please count down from {n}",
        f"countdown {nw} to one",
        f"store {n} {n-1} ... 1",
        f"a countdown {n} to 1",
        f"{n} downto 1",
    ]


def sum_phrasings(n: int) -> list[str]:
    nw = _w(n)
    return [
        f"compute 1 + 2 + ... + {n} and store the sum",
        f"sum the numbers from 1 to {n}",
        f"store the sum 1 through {n}",
        f"compute the sum 1..{n}",
        f"sum of 1 to {n}",
        f"1 + 2 + ... + {n}",
        f"add up 1 through {n}",
        f"total of 1 through {n}",
        f"sum 1 to {n}",
        f"sum from 1 to {nw}",
        f"compute sum of 1..{n}",
        f"the sum of integers 1 to {n}",
        f"add all numbers from 1 to {n}",
        f"triangular number {n}",
        f"save 1+2+...+{n}",
        f"what is 1+2+...+{n}",
        f"sum 1..{n} and save it",
        f"compute the total 1 through {n}",
        f"1 through {n} sum",
        f"sum from 1 up to {n}",
        f"please sum 1 to {n}",
        f"store sum(1..{n})",
        f"addition of 1 to {n}",
        f"sum integers 1 to {n}",
        f"sum of first {n} positive integers",
    ]


def max_phrasings(a: int, b: int) -> list[str]:
    aw, bw = _w(a), _w(b)
    return [
        f"find the max of {a} and {b} and store it",
        f"store the larger of {a} and {b}",
        f"compute max({a}, {b})",
        f"which is bigger, {a} or {b}; store it",
        f"the maximum of {a} and {b}",
        f"max of {a} and {b}",
        f"bigger of {a} and {b}",
        f"save max({a}, {b})",
        f"larger of {aw} and {bw}",
        f"max({a}, {b})",
        f"pick the larger of {a} or {b}",
        f"branch on {a} vs {b} and keep the max",
        f"store the greater of {a} and {b}",
        f"return max of {a} and {b}",
        f"compute maximum of {a}, {b}",
        f"which of {a} and {b} is greater; save it",
        f"the greater of {aw} and {bw}",
        f"bigger: {a} or {b}",
        f"{a} vs {b}: keep the larger",
        f"larger number of {a} and {b}",
        f"please compute max({a},{b})",
        f"store whichever is greater, {a} or {b}",
        f"pick max {a} {b}",
        f"store the max of two numbers {a} and {b}",
        f"bigger value between {a} and {b}",
    ]


def memcpy_phrasings(n: int) -> list[str]:
    nw = _w(n)
    return [
        f"copy {n} words from source to destination",
        f"copy {n} memory words",
        f"memcpy {n} elements",
        f"copy a block of {n} words",
        f"transfer {n} words from src to dst",
        f"move {n} words of memory",
        f"duplicate {n} words of data",
        f"copy {nw} words",
        f"copy {n} values from one buffer to another",
        f"memcpy size {n}",
        f"block copy of {n} ints",
        f"copy the first {n} words",
        f"do a memcpy of {n} words",
        f"copy {n} 32-bit values",
        f"transfer a block of {n} words",
        f"move a buffer of {n} ints",
        f"{n}-word memcpy",
        f"copy {n} words to the destination buffer",
        f"replicate {n} words of memory",
        f"please copy {n} words",
        f"copy over {n} words of data",
        f"move {nw} words",
        f"memory copy, {n} words",
        f"clone {n} memory slots",
        f"shift {n} words to destination",
    ]


def call_phrasings(n: int) -> list[str]:
    nw = _w(n)
    return [
        f"call a function that doubles {n} and store the result",
        f"use a subroutine to double {n}",
        f"double {n} via a function call",
        f"call fn({n}) where fn doubles its argument",
        f"subroutine call: double {n}",
        f"jal to a doubler with {n}",
        f"function call with arg {n} that returns 2x",
        f"double {nw} using a subroutine",
        f"invoke a doubling function on {n}",
        f"call the doubler on {n}",
        f"use jal/jalr to double {n}",
        f"double({n}) via function",
        f"2 * {n} via a function call",
        f"pass {n} to a doubler and save the result",
        f"a subroutine that returns 2x, called with {n}",
        f"call/return: double {n}",
        f"function returning 2x, arg = {n}",
        f"store 2 times {n} using a function",
        f"subroutine double({n})",
        f"invoke doubler({n})",
        f"please double {n} using a subroutine",
        f"{n} * 2 via a function",
        f"doubling function called with {n}",
        f"function call to double {n}",
        f"double-via-subroutine with input {n}",
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
    add_pairs = [(a, b) for a in (0, 1, 2, 3, 5, 7, 10, 15, 25, 42, 50, 100)
                        for b in (0, 1, 3, 5, 7, 10, 25) if a + b <= 200]
    for a, b in add_pairs:
        expected = {"mem_word": (DATA_BASE, a + b)}
        _add_tasks(tasks, "add", rejected, add_phrasings(a, b),
                   ADD_VARIANTS, (a, b), expected)

    # 2. FACTORIAL
    for n in (1, 2, 3, 4, 5):
        expected = {"mem_word": (DATA_BASE, math.factorial(n))}
        _add_tasks(tasks, "factorial", rejected, factorial_phrasings(n),
                   FACTORIAL_VARIANTS, (n,), expected)

    # 3. FIBONACCI
    def fib_seq(n):
        a, b, out = 0, 1, []
        for _ in range(n):
            out.append(a); a, b = b, a + b
        return out
    for n in (3, 4, 5, 6, 8):
        expected = {"mem_words": (DATA_BASE, fib_seq(n))}
        _add_tasks(tasks, "fib", rejected, fib_phrasings(n),
                   FIB_VARIANTS, (n,), expected)

    # 4. COUNTDOWN
    for n in (3, 4, 5, 6, 8, 10):
        expected = {"mem_words": (DATA_BASE, list(range(n, 0, -1)))}
        _add_tasks(tasks, "countdown", rejected, countdown_phrasings(n),
                   COUNTDOWN_VARIANTS, (n,), expected)

    # 5. SUM 1..N
    for n in (3, 4, 5, 6, 8, 10, 15):
        expected = {"mem_word": (DATA_BASE, n * (n + 1) // 2)}
        _add_tasks(tasks, "sum", rejected, sum_phrasings(n),
                   SUM_VARIANTS, (n,), expected)

    # 6. MAX OF TWO
    max_pairs = [(a, b) for a in (0, 1, 3, 5, 7, 10, 15, 20)
                        for b in (0, 2, 4, 6, 9, 12, 18) if a != b]
    for a, b in max_pairs:
        expected = {"mem_word": (DATA_BASE, max(a, b))}
        _add_tasks(tasks, "max", rejected, max_phrasings(a, b),
                   MAX_VARIANTS, (a, b), expected)

    # 7. MEMCPY — pre-seed src with values 1..n, verify dst matches.
    for n in (2, 3, 4, 5, 6):
        src_addr = DATA_BASE + SRC_OFFSET
        dst_addr = DATA_BASE + DST_OFFSET
        seed_vals = list(range(1, n + 1))
        expected = {"mem_words": (dst_addr, seed_vals)}
        _add_tasks(tasks, "memcpy", rejected, memcpy_phrasings(n),
                   MEMCPY_VARIANTS, (n,), expected,
                   seed={src_addr: seed_vals})

    # 8. FUNCTION CALL
    for n in (1, 2, 3, 5, 7, 10, 25, 42, 50):
        expected = {"mem_word": (DATA_BASE, 2 * n)}
        _add_tasks(tasks, "call", rejected, call_phrasings(n),
                   CALL_VARIANTS, (n,), expected)

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
