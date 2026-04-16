"""
Demo: type an instruction. The model emits a complete CHIP-8 program
(linear-emit, zero-state). The program is then loaded into the emulator
and executed end-to-end.

Two phases visible to the viewer:
  1. EMIT — neural control head produces opcodes, ~3ms each.
  2. EXECUTE — emulator runs the program until it halts.

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

from .chip8 import Chip8, DISPLAY_SIZE, PROGRAM_START
from .model import STATE_DIM, ReflexModel, encode_instruction, load_backbone

B = "\033[1m"
D = "\033[2m"
G = "\033[32m"
Y = "\033[33m"
R = "\033[31m"
C = "\033[36m"
N = "\033[0m"


# ── Disassembler ──────────────────────────────────────────────────────

def disassemble(opcode: int) -> str:
    """One-line mnemonic for a single opcode."""
    op = (opcode >> 12) & 0xF
    x = (opcode >> 8) & 0xF
    y = (opcode >> 4) & 0xF
    n = opcode & 0xF
    nn = opcode & 0xFF
    nnn = opcode & 0xFFF
    if opcode == 0x0000: return "STOP"
    if opcode == 0x00E0: return "CLS"
    if opcode == 0x00EE: return "RET"
    if op == 0x1: return f"JUMP   0x{nnn:03X}"
    if op == 0x2: return f"CALL   0x{nnn:03X}"
    if op == 0x3: return f"SKE    V{x:X}, 0x{nn:02X}"
    if op == 0x4: return f"SKNE   V{x:X}, 0x{nn:02X}"
    if op == 0x5: return f"SKE    V{x:X}, V{y:X}"
    if op == 0x6: return f"MOV    V{x:X}, 0x{nn:02X}"
    if op == 0x7: return f"ADD    V{x:X}, 0x{nn:02X}"
    if op == 0x8:
        return {
            0x0: f"MOV    V{x:X}, V{y:X}",
            0x1: f"OR     V{x:X}, V{y:X}",
            0x2: f"AND    V{x:X}, V{y:X}",
            0x3: f"XOR    V{x:X}, V{y:X}",
            0x4: f"ADD    V{x:X}, V{y:X}",
            0x5: f"SUB    V{x:X}, V{y:X}",
            0x6: f"SHR    V{x:X}",
            0x7: f"SUBN   V{x:X}, V{y:X}",
            0xE: f"SHL    V{x:X}",
        }.get(n, f"?{opcode:04X}")
    if op == 0x9: return f"SKNE   V{x:X}, V{y:X}"
    if op == 0xA: return f"MOVI   0x{nnn:03X}"
    if op == 0xB: return f"JUMPV  0x{nnn:03X}"
    if op == 0xC: return f"RAND   V{x:X}, 0x{nn:02X}"
    if op == 0xD: return f"DRAW   V{x:X}, V{y:X}, {n}"
    if op == 0xF:
        return {
            0x07: f"GETDT  V{x:X}",
            0x0A: f"WAITK  V{x:X}",
            0x15: f"SETDT  V{x:X}",
            0x18: f"SETST  V{x:X}",
            0x1E: f"ADDI   V{x:X}",
            0x29: f"FONT   V{x:X}",
            0x33: f"BCD    V{x:X}",
            0x55: f"STORE  V0..V{x:X}",
            0x65: f"LOAD   V0..V{x:X}",
        }.get(nn, f"?{opcode:04X}")
    return f"?{opcode:04X}"


def render_program(program: bytes, current_pc: int | None = None) -> str:
    """Hex dump + disassembly. Marks current_pc with an arrow."""
    lines = []
    for i in range(0, len(program), 2):
        addr = PROGRAM_START + i
        opcode = (program[i] << 8) | program[i + 1]
        marker = f"{C}>{N}" if addr == current_pc else " "
        lines.append(f"  {marker} 0x{addr:03X}  {opcode:04X}   {disassemble(opcode)}")
    return "\n".join(lines)


# ── Display ───────────────────────────────────────────────────────────

def display_lines(display: np.ndarray, width: int = 64) -> list[str]:
    """Return display as a list of strings (no indent, no joining)."""
    lines = ["┌" + "─" * width + "┐"]
    for row in range(0, 32, 2):
        line = "│"
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
    lines.append("└" + "─" * width + "┘")
    return lines


def render_display(display: np.ndarray, width: int = 64, indent: str = "  ") -> str:
    return "\n".join(indent + ln for ln in display_lines(display, width))


DISPLAY_PANEL_HEIGHT = 18  # 1 top border + 16 row pairs + 1 bottom border


# ── Run one instruction ───────────────────────────────────────────────

EMIT_MAX_STEPS = 31
EXEC_MAX_CYCLES = 2000


def emit_program_stream(model, h, tid, max_steps: int = EMIT_MAX_STEPS):
    """Generator: yields (opcode_int, emit_us_for_this_op) per step.
    Stops on STOP token (0x0000) or after max_steps. Callee assembles bytes."""
    zero_state = mx.zeros((1, STATE_DIM))
    h_state = None
    prev_hi = prev_lo = None
    for _ in range(max_steps):
        t0 = time.perf_counter()
        hi_l, lo_l, h_state = model(
            h, zero_state, mx.array(tid[None]),
            prev_hi, prev_lo, h_state,
        )
        mx.eval(hi_l, lo_l, h_state)
        dt_us = (time.perf_counter() - t0) * 1e6
        hi = int(mx.argmax(hi_l[0]).item())
        lo = int(mx.argmax(lo_l[0]).item())
        opcode = (hi << 8) | lo
        if opcode == 0x0000:
            return
        yield opcode, dt_us
        prev_hi = mx.array([hi], dtype=mx.int32)
        prev_lo = mx.array([lo], dtype=mx.int32)


def emit_program(model, h, tid, max_steps: int = EMIT_MAX_STEPS) -> tuple[bytes, float]:
    """Phase 1: linear-emit, zero-state. Returns (program_bytes, total_us)."""
    program = bytearray()
    total_us = 0.0
    for opcode, dt in emit_program_stream(model, h, tid, max_steps):
        program.append((opcode >> 8) & 0xFF)
        program.append(opcode & 0xFF)
        total_us += dt
    return bytes(program), total_us


def execute_program(chip: Chip8, program: bytes, max_cycles: int = EXEC_MAX_CYCLES,
                    trace: bool = True) -> tuple[int, bool, float]:
    """Phase 2: load and run. Returns (cycles_run, halted_cleanly, total_us).
    Cleanly halted = self-jump 1NNN to current pc."""
    chip.load_program(program)
    program_end = PROGRAM_START + len(program)
    total_us = 0.0
    halted = False
    cycle = 0
    for cycle in range(max_cycles):
        pc = chip.pc
        if pc + 1 >= len(chip.memory):
            break
        if pc >= program_end:
            if trace:
                print(f"    {R}pc=0x{pc:03X} ran past program end (0x{program_end:03X}){N}")
            break
        opcode = (int(chip.memory[pc]) << 8) | int(chip.memory[pc + 1])
        if opcode == 0x0000:
            if trace:
                print(f"    {R}hit 0x0000 (undefined) at pc=0x{pc:03X}{N}")
            break
        if (opcode & 0xF000) == 0x1000 and (opcode & 0x0FFF) == pc:
            halted = True
            break
        # Stack overflow guard: a bad CALL chain would otherwise blow the stack.
        if (opcode & 0xF000) == 0x2000 and chip.sp >= len(chip.stack):
            print(f"    {R}stack overflow at pc=0x{pc:03X} — aborting{N}")
            break
        if trace and cycle < 60:
            print(f"    {D}cyc {cycle:3d}  pc=0x{pc:03X}  {opcode:04X}   "
                  f"{disassemble(opcode)}{N}")
        elif trace and cycle == 60:
            print(f"    {D}... (further cycles suppressed){N}")
        t0 = time.perf_counter()
        try:
            chip.step(opcode)
        except (IndexError, ValueError) as e:
            print(f"    {R}runtime error at pc=0x{pc:03X}: {e}{N}")
            break
        total_us += (time.perf_counter() - t0) * 1e6
    return cycle + (1 if halted else 0), halted, total_us


def run_instruction(chip, model, backbone, tokenizer, instruction: str) -> None:
    """Non-interactive single-shot: emit then execute, flat trace + final display.
    Used by the preset-cases path. The `-i` flag uses run_persistent_tui instead."""
    h, tid = encode_instruction(instruction, backbone, tokenizer)
    mx.eval(h)

    print(f"\n{B}━━━ \"{instruction}\" ━━━{N}\n")

    # Phase 1: EMIT
    print(f"  {D}EMIT  (neural control head, zero machine state){N}")
    program, emit_us = emit_program(model, h, tid)
    n_ops = len(program) // 2
    print(f"  {D}{n_ops} opcodes in {emit_us/1000:.1f}ms ({emit_us/max(n_ops,1):.0f}µs/op){N}\n")

    if n_ops == 0:
        print(f"  {R}No program emitted (immediate STOP).{N}")
        return

    print(f"{D}  Emitted program:{N}")
    print(render_program(program))
    print()

    # Phase 2: EXECUTE
    print(f"  {D}EXECUTE  (CHIP-8 emulator){N}")
    cycles, halted, exec_us = execute_program(chip, program, trace=True)
    pixels = int(chip.display.sum())

    status = f"{G}halted{N}" if halted else f"{Y}timed out{N}"
    print()
    print(f"  {D}emit:{N} {n_ops} opcodes in {emit_us/1000:.1f}ms (neural)  "
          f"{D}|{N}  {D}execute:{N} {cycles} cycles in {exec_us/1000:.1f}ms (emulator)  "
          f"{D}|{N}  {D}pixels:{N} {pixels}  {D}|{N}  {status}")

    if pixels > 0:
        print(f"\n{G}Display:{N}")
        print(render_display(chip.display))
    else:
        print(f"\n  {R}No pixels drawn{N}")


# ── Persistent TUI for interactive mode ──────────────────────────────

EXAMPLES = [
    ("Loops",        "count from 1 to 9 and display each digit"),
    ("",             "blink the screen 3 times"),
    ("",             "count down from 7 to 1"),
    ("Conditionals", "set V0 to 5; if V0 equals 5 draw a 1 else draw a 0"),
    ("Arithmetic",   "add 7 and 8 and show the result"),
    ("",             "compute 3 times 4 and show it"),
    ("Subroutines",  "draw a star using a subroutine called twice"),
    ("Memory",       "store 42 at address 0x300 and load it into V0"),
    ("Timers",       "wait for 30 ticks then draw a 1"),
    ("Random",       "draw a random digit"),
]
DISPLAY_W_CH = 66
TUI_PANEL_HEIGHT = DISPLAY_PANEL_HEIGHT  # 18
TUI_FRAME_DELAY_MS = 70

# A "synapse firing" indicator — pulses in/out so the eye catches it.
PULSE_FRAMES = ["·  ", "•  ", "•· ", "•·•", " ·•", "  •", "  ·"]


def _pulse(frame: int) -> str:
    return PULSE_FRAMES[frame % len(PULSE_FRAMES)]


def _animate_during(action_fn, redraw_fn, frame_ms: int = 70):
    """Run `action_fn()` in a daemon thread; while it runs, call
    `redraw_fn(frame_idx)` every `frame_ms`. Returns the action's return value;
    re-raises any exception thrown by the worker."""
    box = {"result": None, "error": None}

    def worker():
        try:
            box["result"] = action_fn()
        except BaseException as e:  # noqa: BLE001
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


def _render_panels(display_arr: np.ndarray, waterfall: deque) -> list[str]:
    """Build TUI_PANEL_HEIGHT lines of `<display>    <waterfall>`."""
    disp = display_lines(display_arr)
    wf_lines = [f"{D}cyc   pc     opcode   mnemonic{N}"]
    wf_list = list(waterfall)
    for idx, (cyc, p, op, mn) in enumerate(wf_list):
        is_current = idx == len(wf_list) - 1
        arrow = f"{C}>{N}" if is_current else " "
        wf_lines.append(f"{arrow} {cyc:>3}  0x{p:03X}  {op:04X}     {mn}")
    while len(wf_lines) < TUI_PANEL_HEIGHT:
        wf_lines.append("")
    out = []
    for i in range(TUI_PANEL_HEIGHT):
        left = disp[i] if i < len(disp) else " " * DISPLAY_W_CH
        right = wf_lines[i] if i < len(wf_lines) else ""
        out.append(f"  {left:<{DISPLAY_W_CH}}    {right}")
    return out


def _draw_tui(display_arr: np.ndarray, waterfall: deque,
              summary: str, status: str = "",
              instruction: str = "",
              show_prompt: bool = True,
              show_examples: bool = False) -> None:
    """Clear screen and redraw the entire UI from cursor home.
    `show_examples=True` inlines the example list above the prompt — used
    only on the first render of a session."""
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.write(f"{B}Reflex — instruction → CHIP-8 program → execute{N}   "
                     f"{D}(emit: neural | execute: emulator){N}\n\n")
    for line in _render_panels(display_arr, waterfall):
        sys.stdout.write(line + "\n")
    sys.stdout.write("\n")
    instr_line = (f"{B}> {N}{instruction}" if instruction
                  else f"{D}(idle — type an instruction below){N}")
    sys.stdout.write(f"  {instr_line}\n")
    sys.stdout.write(f"  {D}{status}{N}\n")
    if summary:
        sys.stdout.write(f"  {summary}\n")
    sys.stdout.write("\n")
    if show_examples:
        sys.stdout.write(f"  {D}Examples (try paraphrasing — the model generalises):{N}\n")
        for cat, ex in EXAMPLES:
            cat_str = f"{cat:<13}" if cat else " " * 13
            sys.stdout.write(f"    {D}{cat_str}{N} {ex}\n")
        sys.stdout.write("\n")
    sys.stdout.write(f"  {D}('?' or 'help' for examples · empty line or 'quit' to exit){N}\n\n")
    if show_prompt:
        sys.stdout.write(f"{B}> {N}")
    sys.stdout.flush()


def _show_examples_screen(wait_for_enter: bool = True) -> None:
    """Full-screen modal listing the example instructions, organised by
    category. On startup we use this as a welcome screen; later, the user
    can re-summon it by typing '?' / 'help' / 'examples'."""
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.write(f"{B}Reflex — example instructions{N}\n\n")
    sys.stdout.write(f"  {D}The model emits a complete CHIP-8 program for each. "
                     f"Try paraphrasing — it generalises.{N}\n\n")
    for cat, ex in EXAMPLES:
        cat_str = f"{cat:<13}" if cat else " " * 13
        sys.stdout.write(f"    {D}{cat_str}{N} {ex}\n")
    sys.stdout.write(f"\n  {D}Free-form is also fine — see the README for limits.{N}\n")
    if wait_for_enter:
        sys.stdout.write(f"\n  {B}press enter to start{N} ")
        sys.stdout.flush()
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        sys.stdout.flush()


def run_persistent_tui(chip: Chip8, model, backbone, tokenizer) -> None:
    """Interactive REPL with always-visible display + waterfall on top,
    examples + prompt at the bottom. Status line shows what the model is
    doing (encoding, emitting, executing) so the wait isn't silent."""
    display_arr = np.zeros(DISPLAY_SIZE, dtype=np.uint8)
    waterfall: deque = deque(maxlen=TUI_PANEL_HEIGHT - 1)
    summary = ""  # populated only after a run completes
    instr_cache: dict = {}  # encoded-instruction cache
    last_instruction = ""
    first_render = True

    while True:
        _draw_tui(display_arr, waterfall, summary,
                  instruction=last_instruction,
                  show_examples=first_render)
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
        # Clear the previous run's summary so it doesn't sit there stale
        # next to the new "encoding..." status.
        summary = ""

        # ── Encode instruction (cached for repeats; animated when fresh) ──
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
                _draw_tui(display_arr, waterfall, summary,
                          status=f"{C}{pulse}{N}  ENCODE  Qwen2.5-Coder-1.5B forward pass",
                          instruction=instruction, show_prompt=False)

            h, tid = _animate_during(do_encode, draw_encoding)
            instr_cache[instruction] = (h, tid)
            cached = False

        # ── Emit (streaming: each opcode appears live as the head produces it) ──
        waterfall.clear()
        display_arr[:] = 0
        program_bytes = bytearray()
        emit_us = 0.0
        for i, (opcode, dt) in enumerate(emit_program_stream(model, h, tid)):
            program_bytes.append((opcode >> 8) & 0xFF)
            program_bytes.append(opcode & 0xFF)
            emit_us += dt
            pc = PROGRAM_START + 2 * i
            waterfall.append((i, pc, opcode, disassemble(opcode)))
            pulse = _pulse(i)
            _draw_tui(display_arr, waterfall, summary,
                      status=f"{C}{pulse}{N}  EMIT (neural)  op {i+1}  {dt/1000:.0f}ms  "
                             f"{opcode:04X}  {disassemble(opcode)}",
                      instruction=instruction, show_prompt=False)
        program = bytes(program_bytes)
        n_ops = len(program) // 2

        if n_ops == 0:
            summary = f"{R}\"{instruction}\" → 0 opcodes (immediate STOP){N}"
            continue

        # ── Execute (animate panel updates per cycle) ──
        chip.reset()
        chip.load_program(program)
        program_end = PROGRAM_START + len(program)
        waterfall.clear()
        display_arr[:] = 0
        exec_us = 0.0
        halted = False
        cycles_run = 0
        aborted = ""
        for cycle in range(EXEC_MAX_CYCLES):
            pc = chip.pc
            if pc + 1 >= len(chip.memory):
                aborted = "pc out of memory bounds"
                break
            # Walked past the bytes the model emitted — almost always means
            # the program never reached a clean self-jump halt.
            if pc >= program_end:
                aborted = (f"pc=0x{pc:03X} ran past program end (0x{program_end:03X}) "
                           f"— no halt emitted")
                break
            opcode = (int(chip.memory[pc]) << 8) | int(chip.memory[pc + 1])
            # 0x0000 is undefined in CHIP-8; in our convention it's the STOP
            # token. Hitting it during execution means a malformed program.
            if opcode == 0x0000:
                aborted = f"hit 0x0000 (undefined / STOP) at pc=0x{pc:03X}"
                break
            if (opcode & 0xF000) == 0x1000 and (opcode & 0x0FFF) == pc:
                halted = True
                break
            if (opcode & 0xF000) == 0x2000 and chip.sp >= len(chip.stack):
                aborted = "stack overflow"
                break
            waterfall.append((cycle, pc, opcode, disassemble(opcode)))
            t1 = time.perf_counter()
            try:
                chip.step(opcode)
            except (IndexError, ValueError) as e:
                aborted = f"runtime: {e}"
                break
            exec_us += (time.perf_counter() - t1) * 1e6
            cycles_run = cycle + 1
            display_arr[:] = chip.display
            _draw_tui(display_arr, waterfall, summary,
                      status=f"EXEC (emulator)  cyc {cycle}  pc=0x{pc:03X}  "
                             f"{opcode:04X}  {disassemble(opcode)}",
                      instruction=instruction, show_prompt=False)
            time.sleep(TUI_FRAME_DELAY_MS / 1000.0)

        pixels = int(chip.display.sum())
        chip_status = (f"{R}{aborted}{N}" if aborted
                       else (f"{G}halted{N}" if halted else f"{Y}timed out{N}"))
        cache_tag = f"{D}(cached){N}" if cached else f"{D}(fresh encode){N}"
        summary = (f"{cache_tag}  "
                   f"emit {n_ops} ops {emit_us/1000:.0f}ms │ "
                   f"exec {cycles_run} cyc {exec_us/1000:.1f}ms │ "
                   f"{pixels}px │ {chip_status}")


def main():
    interactive = "-i" in sys.argv or "--interactive" in sys.argv

    print(f"""
{B}╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  Reflex: instruction → complete CHIP-8 program → executed      ║
║                                                                ║
║  Phase 1 (neural):    control head emits opcodes, zero state.  ║
║  Phase 2 (emulator):  the emitted program runs end-to-end.     ║
║                                                                ║
║  Programs use loops, conditionals, subroutines, calls,         ║
║  returns, BCD digit display, delay timer, random.              ║
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

    chip = Chip8()

    if interactive:
        # Warm up MLX so the very first instruction doesn't pay JIT-compile cost.
        print(f"{D}Warming up...{N}")
        _h, _tid = encode_instruction("warmup", backbone, tokenizer)
        mx.eval(_h)
        _hi, _lo, _hs = model(
            _h, mx.zeros((1, STATE_DIM)), mx.array(_tid[None]),
        )
        mx.eval(_hi, _lo, _hs)
        run_persistent_tui(chip, model, backbone, tokenizer)
    else:
        for instruction in [
            "count from 1 to 9 and display each digit",
            "blink the screen 3 times",
            "set V0 to 5; if V0 equals 5 draw a 1 else draw a 0",
            "add 7 and 8 and show the result",
            "draw a star using a subroutine called twice",
            "store 42 at address 0x300 and load it into V0",
            "wait for 30 ticks then draw a 1",
            "draw a random digit",
        ]:
            run_instruction(chip, model, backbone, tokenizer, instruction)

    print(f"\n{D}Done.{N}")


if __name__ == "__main__":
    main()
