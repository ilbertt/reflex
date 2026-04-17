"""
Demo: type an instruction. The model emits a complete RV32I program
(linear-emit, zero-state). The program loads into unicorn and runs
end-to-end.

Two phases visible to the viewer:
  1. EMIT    — neural control head produces 32-bit instructions.
  2. EXECUTE — unicorn runs the program until it self-jumps to halt.

Usage:
    uv run demo       # preset test cases (one per category)
    uv run demo -i    # interactive: type anything
"""

import sys
import threading
import time
from collections import deque

import mlx.core as mx
import numpy as np

from .model import FIELD_CLASSES, MAX_KV_LEN, ReflexModel, encode_instruction, load_backbone
from .programs import DST_OFFSET, SRC_OFFSET
from .riscv import (
    DATA_BASE, HALT_INSTR, PROGRAM_START, Rv32i, compose, decompose,
)

B = "\033[1m"
D = "\033[2m"
G = "\033[32m"
Y = "\033[33m"
R = "\033[31m"
C = "\033[36m"
N = "\033[0m"


# ── Disassembler ──────────────────────────────────────────────────────

def _sign_extend(val: int, bits: int) -> int:
    """Interpret `val` as a `bits`-bit two's-complement number."""
    mask = (1 << bits) - 1
    val &= mask
    if val & (1 << (bits - 1)):
        return val - (1 << bits)
    return val


def _i_imm(instr: int) -> int:
    return _sign_extend(instr >> 20, 12)


def _s_imm(instr: int) -> int:
    hi = (instr >> 25) & 0x7F
    lo = (instr >> 7) & 0x1F
    return _sign_extend((hi << 5) | lo, 12)


def _b_imm(instr: int) -> int:
    b12 = (instr >> 31) & 0x1
    b11 = (instr >> 7) & 0x1
    b10_5 = (instr >> 25) & 0x3F
    b4_1 = (instr >> 8) & 0xF
    raw = (b12 << 12) | (b11 << 11) | (b10_5 << 5) | (b4_1 << 1)
    return _sign_extend(raw, 13)


def _u_imm(instr: int) -> int:
    return (instr >> 12) & 0xFFFFF


def _j_imm(instr: int) -> int:
    b20 = (instr >> 31) & 0x1
    b19_12 = (instr >> 12) & 0xFF
    b11 = (instr >> 20) & 0x1
    b10_1 = (instr >> 21) & 0x3FF
    raw = (b20 << 20) | (b19_12 << 12) | (b11 << 11) | (b10_1 << 1)
    return _sign_extend(raw, 21)


def disassemble(instr: int) -> str:
    """One-line RV32I mnemonic."""
    if instr == 0x00000000:
        return "STOP"
    if instr == HALT_INSTR:
        return "HALT (jal x0, 0)"
    opcode = instr & 0x7F
    rd = (instr >> 7) & 0x1F
    funct3 = (instr >> 12) & 0x7
    rs1 = (instr >> 15) & 0x1F
    rs2 = (instr >> 20) & 0x1F
    funct7 = (instr >> 25) & 0x7F

    if opcode == 0x13:   # OP-IMM
        imm = _i_imm(instr)
        name = {0: "ADDI", 2: "SLTI", 4: "XORI", 6: "ORI", 7: "ANDI",
                1: "SLLI", 5: "SRLI"}.get(funct3, f"?I{funct3}")
        return f"{name:<6} x{rd}, x{rs1}, {imm}"
    if opcode == 0x33:   # OP
        key = (funct3, funct7)
        name = {(0, 0): "ADD", (0, 0x20): "SUB", (1, 0): "SLL",
                (2, 0): "SLT", (4, 0): "XOR", (6, 0): "OR",
                (7, 0): "AND", (5, 0): "SRL", (5, 0x20): "SRA"}.get(key, f"?R{funct3}.{funct7}")
        return f"{name:<6} x{rd}, x{rs1}, x{rs2}"
    if opcode == 0x37:   # LUI
        return f"LUI    x{rd}, 0x{_u_imm(instr):05X}"
    if opcode == 0x17:   # AUIPC
        return f"AUIPC  x{rd}, 0x{_u_imm(instr):05X}"
    if opcode == 0x03:   # LOAD
        imm = _i_imm(instr)
        name = {0: "LB", 1: "LH", 2: "LW", 4: "LBU", 5: "LHU"}.get(funct3, f"?L{funct3}")
        return f"{name:<6} x{rd}, {imm}(x{rs1})"
    if opcode == 0x23:   # STORE
        imm = _s_imm(instr)
        name = {0: "SB", 1: "SH", 2: "SW"}.get(funct3, f"?S{funct3}")
        return f"{name:<6} x{rs2}, {imm}(x{rs1})"
    if opcode == 0x63:   # BRANCH
        imm = _b_imm(instr)
        name = {0: "BEQ", 1: "BNE", 4: "BLT", 5: "BGE", 6: "BLTU", 7: "BGEU"}.get(funct3, f"?B{funct3}")
        return f"{name:<6} x{rs1}, x{rs2}, {imm:+d}"
    if opcode == 0x6F:   # JAL
        imm = _j_imm(instr)
        return f"JAL    x{rd}, {imm:+d}"
    if opcode == 0x67:   # JALR
        imm = _i_imm(instr)
        return f"JALR   x{rd}, x{rs1}, {imm}"
    return f"?op{opcode:02x}"


def render_program(program: bytes, current_pc: int | None = None) -> str:
    lines = []
    for i in range(0, len(program), 4):
        addr = PROGRAM_START + i
        instr = int.from_bytes(program[i:i+4], "little")
        marker = f"{C}>{N}" if addr == current_pc else " "
        lines.append(f"  {marker} 0x{addr:04X}  {instr:08X}  {disassemble(instr)}")
    return "\n".join(lines)


# ── Panels ────────────────────────────────────────────────────────────

REG_PANEL_W = 36


def register_lines(cpu: Rv32i) -> list[str]:
    """Four columns of 8 registers each."""
    lines = [f"{D}registers{N}"]
    for row in range(8):
        cells = []
        for col in range(4):
            r = row + 8 * col
            v = cpu.reg(r)
            cells.append(f"x{r:>2}={v:08X}")
        lines.append("  " + "  ".join(cells))
    return lines


def data_lines(cpu: Rv32i, offset: int = 0, n_words: int = 8,
               label: str = "data") -> list[str]:
    addr = DATA_BASE + offset
    lines = [f"{D}{label} @ 0x{addr:04X}{N}"]
    for i in range(n_words):
        v = cpu.mem_word(addr + 4 * i)
        lines.append(f"  +{4*i:02d}  {v:08X}  ({v})")
    return lines


# ── Emit ──────────────────────────────────────────────────────────────

EMIT_MAX_STEPS = 20
EXEC_MAX_CYCLES = 5000


def emit_program_stream(model, h, tid, max_steps: int = EMIT_MAX_STEPS):
    """Yields (instr_int, emit_us) per step. Stops on STOP (all fields 0) or max_steps."""
    h_state = None
    prev_fields = None
    history = mx.zeros((1, MAX_KV_LEN), dtype=mx.int32)
    tid_m = mx.array(tid[None])
    for step_t in range(min(max_steps, MAX_KV_LEN)):
        t0 = time.perf_counter()
        logits, h_state = model(h, history, step_t + 1, tid_m, prev_fields, h_state)
        mx.eval(*logits, h_state)
        dt_us = (time.perf_counter() - t0) * 1e6
        field_vals = [int(mx.argmax(lg[0]).item()) for lg in logits]
        instr = compose(*field_vals)
        if instr == 0x00000000:
            return
        yield instr, dt_us
        prev_fields = tuple(mx.array([v], dtype=mx.int32) for v in field_vals)
        col_mask = mx.arange(MAX_KV_LEN) == step_t
        # Convert to signed int32 bit pattern; instructions with funct7 ≥ 64
        # have bit 31 set and overflow signed int32 as a Python positive int.
        instr_i32 = instr - (1 << 32) if instr >= (1 << 31) else instr
        new_col = mx.broadcast_to(
            mx.array([instr_i32], dtype=mx.int32)[:, None], history.shape)
        history = mx.where(col_mask[None, :], new_col, history)


def emit_program(model, h, tid, max_steps: int = EMIT_MAX_STEPS) -> tuple[bytes, float]:
    program = bytearray()
    total_us = 0.0
    for instr, dt in emit_program_stream(model, h, tid, max_steps):
        program += int(instr & 0xFFFFFFFF).to_bytes(4, "little")
        total_us += dt
    return bytes(program), total_us


# ── Execute ───────────────────────────────────────────────────────────

def _seed_memcpy_src(cpu: Rv32i, n: int = 8) -> None:
    """Pre-fill memcpy source region so demo memcpy programs have data
    to copy regardless of the instruction phrasing."""
    data = b"".join(int(i).to_bytes(4, "little") for i in range(1, n + 1))
    cpu.uc.mem_write(DATA_BASE + SRC_OFFSET, data)


def execute_program(cpu: Rv32i, program: bytes, max_cycles: int = EXEC_MAX_CYCLES,
                    seed_src: bool = True, trace: bool = True) -> tuple[int, bool, float, str]:
    cpu.load_program(program)
    if seed_src:
        _seed_memcpy_src(cpu)
    program_end = PROGRAM_START + len(program)
    total_us = 0.0
    halted = False
    aborted = ""
    cycle = 0
    for cycle in range(max_cycles):
        pc = cpu.pc
        if pc + 3 >= PROGRAM_START + 0x10000 or pc < PROGRAM_START:
            aborted = f"pc=0x{pc:X} out of bounds"
            break
        if pc >= program_end:
            aborted = f"pc=0x{pc:X} past program end (0x{program_end:X})"
            break
        try:
            instr = cpu.fetch()
        except Exception as e:
            aborted = f"fetch: {e}"
            break
        if instr == HALT_INSTR:
            halted = True
            break
        if instr == 0x00000000:
            aborted = f"hit 0x0 (STOP) at pc=0x{pc:X}"
            break
        if trace and cycle < 60:
            print(f"    {D}cyc {cycle:3d}  pc=0x{pc:04X}  {instr:08X}  "
                  f"{disassemble(instr)}{N}")
        elif trace and cycle == 60:
            print(f"    {D}... (further cycles suppressed){N}")
        t0 = time.perf_counter()
        try:
            cpu.step()
        except Exception as e:
            aborted = f"step at pc=0x{pc:X}: {e}"
            break
        total_us += (time.perf_counter() - t0) * 1e6
    return cycle + (1 if halted else 0), halted, total_us, aborted


def run_instruction(cpu: Rv32i, model, backbone, tokenizer, instruction: str) -> None:
    h, tid = encode_instruction(instruction, backbone, tokenizer)
    mx.eval(h)

    print(f"\n{B}━━━ \"{instruction}\" ━━━{N}\n")

    print(f"  {D}EMIT  (neural control head, zero machine state){N}")
    program, emit_us = emit_program(model, h, tid)
    n_ops = len(program) // 4
    print(f"  {D}{n_ops} instructions in {emit_us/1000:.1f}ms "
          f"({emit_us/max(n_ops,1):.0f}µs/op){N}\n")

    if n_ops == 0:
        print(f"  {R}No program emitted (immediate STOP).{N}")
        return

    print(f"{D}  Emitted program:{N}")
    print(render_program(program))
    print()

    print(f"  {D}EXECUTE  (unicorn RV32I){N}")
    cycles, halted, exec_us, aborted = execute_program(cpu, program, trace=True)
    status = (f"{R}{aborted}{N}" if aborted
              else (f"{G}halted{N}" if halted else f"{Y}timed out{N}"))
    print()
    print(f"  {D}emit:{N} {n_ops} ops in {emit_us/1000:.1f}ms  "
          f"{D}|{N}  {D}execute:{N} {cycles} cycles in {exec_us/1000:.1f}ms  "
          f"{D}|{N}  {status}")

    print()
    for line in register_lines(cpu):
        print(line)
    print()
    for line in data_lines(cpu, 0, 8, "data"):
        print(line)
    dst = cpu.mem_word(DATA_BASE + DST_OFFSET)
    if dst != 0:
        print()
        for line in data_lines(cpu, DST_OFFSET, 8, "memcpy dst"):
            print(line)


# ── Persistent TUI ────────────────────────────────────────────────────

EXAMPLES = [
    ("Add",         "add 7 and 8 and store the result"),
    ("Factorial",   "compute 5 factorial and store it"),
    ("Fibonacci",   "store the first 6 Fibonacci numbers"),
    ("Countdown",   "count down from 5 to 1 and store each value"),
    ("Sum",         "compute 1 + 2 + ... + 10 and store the sum"),
    ("Max",         "find the max of 7 and 12 and store it"),
    ("Memcpy",      "copy 4 words from source to destination"),
    ("Call/return", "call a function that doubles 25 and store the result"),
]

TUI_PANEL_HEIGHT = 12
TUI_FRAME_DELAY_MS = 40
PULSE_FRAMES = ["·  ", "•  ", "•· ", "•·•", " ·•", "  •", "  ·"]


def _pulse(frame: int) -> str:
    return PULSE_FRAMES[frame % len(PULSE_FRAMES)]


def _animate_during(action_fn, redraw_fn, frame_ms: int = 70):
    box = {"result": None, "error": None}

    def worker():
        try:
            box["result"] = action_fn()
        except BaseException as e:
            box["error"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    frame = 0
    while t.is_alive():
        redraw_fn(frame)
        frame += 1
        t.join(timeout=frame_ms / 1000.0)
    if box["error"] is not None:
        raise box["error"]
    return box["result"]


def _render_tui(cpu: Rv32i, program: bytes, current_pc: int | None,
                waterfall: deque, summary: str, status: str,
                instruction: str, show_prompt: bool, show_examples: bool) -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.write(f"{B}Reflex — instruction → RV32I program → execute{N}   "
                     f"{D}(emit: neural | execute: unicorn){N}\n\n")

    # Top panel: program listing | registers | data
    prog_lines: list[str] = []
    if program:
        for i in range(0, len(program), 4):
            addr = PROGRAM_START + i
            instr = int.from_bytes(program[i:i+4], "little")
            marker = f"{C}>{N}" if addr == current_pc else " "
            prog_lines.append(f" {marker} 0x{addr:04X}  {instr:08X}  {disassemble(instr)}")
    else:
        prog_lines = [f"  {D}(no program yet){N}"]

    reg_lines = register_lines(cpu)
    data_lines_ = data_lines(cpu, 0, 6, "data")

    height = max(len(prog_lines), len(reg_lines) + len(data_lines_) + 1, TUI_PANEL_HEIGHT)
    right_lines = reg_lines + [""] + data_lines_

    for i in range(height):
        left = prog_lines[i] if i < len(prog_lines) else ""
        right = right_lines[i] if i < len(right_lines) else ""
        sys.stdout.write(f"  {left:<52}   {right}\n")
    sys.stdout.write("\n")

    # Waterfall
    wf_list = list(waterfall)
    if wf_list:
        sys.stdout.write(f"  {D}exec waterfall{N}\n")
        for cyc, p, op, mn in wf_list[-6:]:
            sys.stdout.write(f"    {cyc:>3}  0x{p:04X}  {op:08X}  {mn}\n")
        sys.stdout.write("\n")

    instr_line = (f"{B}> {N}{instruction}" if instruction
                  else f"{D}(idle — type an instruction below){N}")
    sys.stdout.write(f"  {instr_line}\n")
    sys.stdout.write(f"  {D}{status}{N}\n")
    if summary:
        sys.stdout.write(f"  {summary}\n")
    sys.stdout.write("\n")
    if show_examples:
        sys.stdout.write(f"  {D}Examples (paraphrasing works — the model generalises):{N}\n")
        for cat, ex in EXAMPLES:
            sys.stdout.write(f"    {D}{cat:<12}{N} {ex}\n")
        sys.stdout.write("\n")
    sys.stdout.write(f"  {D}('?' or 'help' for examples · empty line or 'quit' to exit){N}\n\n")
    if show_prompt:
        sys.stdout.write(f"{B}> {N}")
    sys.stdout.flush()


def _show_examples_screen(wait_for_enter: bool = True) -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.write(f"{B}Reflex — example instructions{N}\n\n")
    sys.stdout.write(f"  {D}Each phrasing below emits a complete RV32I program.{N}\n\n")
    for cat, ex in EXAMPLES:
        sys.stdout.write(f"    {D}{cat:<12}{N} {ex}\n")
    if wait_for_enter:
        sys.stdout.write(f"\n  {B}press enter to start{N} ")
        sys.stdout.flush()
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass


def run_persistent_tui(cpu: Rv32i, model, backbone, tokenizer) -> None:
    waterfall: deque = deque(maxlen=200)
    summary = ""
    instr_cache: dict = {}
    last_instruction = ""
    first_render = True
    program = b""

    while True:
        _render_tui(cpu, program, None, waterfall, summary,
                    status="", instruction=last_instruction,
                    show_prompt=True, show_examples=first_render)
        first_render = False
        try:
            instruction = input().strip()
        except (EOFError, KeyboardInterrupt):
            sys.stdout.write("\n")
            break
        if not instruction or instruction == "quit":
            break
        if instruction in ("?", "help", "examples"):
            _show_examples_screen(wait_for_enter=True)
            continue
        last_instruction = instruction
        summary = ""

        # Encode
        if instruction in instr_cache:
            h, tid = instr_cache[instruction]
            cached = True
        else:
            def do_encode():
                hh, tt = encode_instruction(instruction, backbone, tokenizer)
                mx.eval(hh)
                return hh, tt

            def draw_encoding(frame):
                pulse = _pulse(frame)
                _render_tui(cpu, program, None, waterfall, summary,
                            status=f"{C}{pulse}{N}  ENCODE  Qwen2.5-Coder-1.5B",
                            instruction=instruction, show_prompt=False, show_examples=False)

            h, tid = _animate_during(do_encode, draw_encoding)
            instr_cache[instruction] = (h, tid)
            cached = False

        # Emit (streaming)
        cpu.reset()
        _seed_memcpy_src(cpu)
        waterfall.clear()
        program_bytes = bytearray()
        emit_us = 0.0
        for i, (instr, dt) in enumerate(emit_program_stream(model, h, tid)):
            program_bytes += int(instr & 0xFFFFFFFF).to_bytes(4, "little")
            emit_us += dt
            _render_tui(cpu, bytes(program_bytes), None, waterfall, summary,
                        status=f"{C}{_pulse(i)}{N}  EMIT  op {i+1}  {dt/1000:.0f}ms  "
                               f"{instr:08X}  {disassemble(instr)}",
                        instruction=instruction, show_prompt=False, show_examples=False)
        program = bytes(program_bytes)
        n_ops = len(program) // 4

        if n_ops == 0:
            summary = f"{R}\"{instruction}\" → 0 ops (immediate STOP){N}"
            continue

        # Execute (animated per cycle)
        cpu.load_program(program)
        _seed_memcpy_src(cpu)
        program_end = PROGRAM_START + len(program)
        exec_us = 0.0
        halted = False
        aborted = ""
        cycles_run = 0
        for cycle in range(EXEC_MAX_CYCLES):
            pc = cpu.pc
            if pc < PROGRAM_START or pc >= program_end:
                aborted = f"pc=0x{pc:X} out of program"
                break
            try:
                instr = cpu.fetch()
            except Exception as e:
                aborted = f"fetch: {e}"
                break
            if instr == HALT_INSTR:
                halted = True
                break
            if instr == 0x00000000:
                aborted = f"hit 0x0 at pc=0x{pc:X}"
                break
            waterfall.append((cycle, pc, instr, disassemble(instr)))
            t1 = time.perf_counter()
            try:
                cpu.step()
            except Exception as e:
                aborted = f"step at pc=0x{pc:X}: {e}"
                break
            exec_us += (time.perf_counter() - t1) * 1e6
            cycles_run = cycle + 1
            _render_tui(cpu, program, pc, waterfall, summary,
                        status=f"EXEC  cyc {cycle}  pc=0x{pc:04X}  {instr:08X}  "
                               f"{disassemble(instr)}",
                        instruction=instruction, show_prompt=False, show_examples=False)
            time.sleep(TUI_FRAME_DELAY_MS / 1000.0)

        status = (f"{R}{aborted}{N}" if aborted
                  else (f"{G}halted{N}" if halted else f"{Y}timed out{N}"))
        cache_tag = f"{D}(cached){N}" if cached else f"{D}(fresh encode){N}"
        summary = (f"{cache_tag}  "
                   f"emit {n_ops} ops {emit_us/1000:.0f}ms │ "
                   f"exec {cycles_run} cyc {exec_us/1000:.1f}ms │ {status}")


def main():
    interactive = "-i" in sys.argv or "--interactive" in sys.argv

    print(f"""
{B}╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  Reflex: instruction → complete RV32I program → executed       ║
║                                                                ║
║  Phase 1 (neural):    control head emits instructions.         ║
║  Phase 2 (unicorn):   the emitted program runs end-to-end.     ║
║                                                                ║
║  Programs cover: arithmetic, loops (sum, countdown, Fibonacci, ║
║  factorial), conditionals (max), memory (store/load/memcpy),   ║
║  and function call/return.                                     ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝{N}
""")

    print(f"{D}Loading...{N}")
    backbone, tokenizer = load_backbone()
    model = ReflexModel()
    try:
        model.load_weights(list(mx.load("weights.npz").items()))
    except FileNotFoundError:
        print("No weights.npz. Run: uv run train")
        return

    cpu = Rv32i()

    if interactive:
        print(f"{D}Warming up...{N}")
        _h, _tid = encode_instruction("warmup", backbone, tokenizer)
        mx.eval(_h)
        _hist = mx.zeros((1, MAX_KV_LEN), dtype=mx.int32)
        _logits, _hs = model(_h, _hist, 1, mx.array(_tid[None]))
        mx.eval(*_logits, _hs)
        run_persistent_tui(cpu, model, backbone, tokenizer)
    else:
        for _, instruction in EXAMPLES:
            run_instruction(cpu, model, backbone, tokenizer, instruction)

    print(f"\n{D}Done.{N}")


if __name__ == "__main__":
    main()
