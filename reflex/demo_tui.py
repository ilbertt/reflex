"""Side-by-side demo: Reflex (grounded CPU emission) vs plain text
generation from the same Qwen2.5-Coder-7B backbone.

Left panel shows the opcode stream, live register state, and display
buffer. Right panel shows the backbone's normal chat completion
streamed token-by-token, throttled to ~30 tok/s so the comparison
isn't dominated by A100 throughput. Bottom row: the shared prompt.

Usage:
    uv run python demo_split.py --ckpt reflex_coder7b.pt
"""
import argparse
import select
import shutil
import sys
import termios
import threading
import time
import tty
from collections import deque

import torch
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from unicorn.riscv_const import UC_RISCV_REG_PC

from reflex.demo import load
from reflex.model import SYSTEM_MESSAGE, code_region_halt_fill, extract_state
from reflex.programs import DISPLAY_BASE, SRC_OFFSET
from reflex.riscv import (
    DATA_BASE, HALT_INSTR, PROGRAM_START, Rv32i,
    add, addi, beq, bge, blt, bne, halt, jal, jalr, lui, lw, pack, sub, sw,
)


# ── Tiny RV32I disassembler (reused from eval_history) ───────────────
R3 = {0x0:('add','sub'),0x7:('and',None),0x6:('or',None),0x4:('xor',None),
      0x1:('sll',None),0x5:('srl','sra'),0x2:('slt',None),0x3:('sltu',None)}
I3 = {0x0:'addi',0x7:'andi',0x6:'ori',0x4:'xori',0x1:'slli',0x5:'srli',
      0x2:'slti',0x3:'sltiu'}
L3 = {0x0:'lb',0x1:'lh',0x2:'lw',0x4:'lbu',0x5:'lhu'}
S3 = {0x0:'sb',0x1:'sh',0x2:'sw'}
B3 = {0x0:'beq',0x1:'bne',0x4:'blt',0x5:'bge',0x6:'bltu',0x7:'bgeu'}


def _sx(v, bits):
    m = 1 << (bits - 1); return (v ^ m) - m


def disasm(w: int) -> str:
    op = w & 0x7f; rd = (w>>7) & 0x1f; f3 = (w>>12) & 0x7
    rs1 = (w>>15) & 0x1f; rs2 = (w>>20) & 0x1f; f7 = (w>>25) & 0x7f
    if op == 0x33:
        name, alt = R3.get(f3,('?',None))
        if f7 == 0x20 and alt: name = alt
        return f'{name:<6} x{rd},x{rs1},x{rs2}'
    if op == 0x13:
        imm = _sx((w>>20)&0xfff, 12)
        return f'{I3.get(f3,"?"):<6} x{rd},x{rs1},{imm}'
    if op == 0x03:
        imm = _sx((w>>20)&0xfff, 12)
        return f'{L3.get(f3,"?"):<6} x{rd},{imm}(x{rs1})'
    if op == 0x23:
        imm = _sx(((w>>25)<<5)|((w>>7)&0x1f), 12)
        return f'{S3.get(f3,"?"):<6} x{rs2},{imm}(x{rs1})'
    if op == 0x63:
        imm = _sx(((w>>31)<<12)|(((w>>7)&1)<<11)|(((w>>25)&0x3f)<<5)|
                  (((w>>8)&0xf)<<1), 13)
        return f'{B3.get(f3,"?"):<6} x{rs1},x{rs2},{imm}'
    if op == 0x37: return f'lui    x{rd},0x{(w>>12)&0xfffff:x}'
    if op == 0x17: return f'auipc  x{rd},0x{(w>>12)&0xfffff:x}'
    if op == 0x6f: return f'jal    x{rd}'
    if op == 0x67: return f'jalr   x{rd}'
    return f'??(0x{w:08x})'


# ── Shared state ──────────────────────────────────────────────────────
class DemoState:
    def __init__(self):
        self.prompt = ''
        self.phase = 'input'          # 'input' | 'running' | 'done'
        self.input_buffer = ''
        # Reflex
        self.reflex_ops = deque(maxlen=200)    # keep long history; render tails
        self.reflex_op_count = 0
        self.reflex_regs = [0] * 32
        self.reflex_display = ''
        self.reflex_start = None
        self.reflex_dt_ms = 0
        self.reflex_done = False
        self.reflex_final_mem = 0
        self.reflex_halted = False
        self.reflex_err = ''
        self.reflex_iter = 1                   # current refinement pass (1-based)
        self.reflex_total_iters = 1            # total passes actually run
        # Text
        self.text_out = ''
        self.text_tok_count = 0
        self.text_start = None
        self.text_dt_ms = 0
        self.text_done = False
        self.text_err = ''
        self.text_final_mem = 0
        self.text_display = ''
        self.text_halted = False
        self.text_ops_run = 0
        self.text_regs = [0] * 32


# Per-worker CUDA streams so the two threads can launch kernels
# concurrently without colliding on the default cuBLAS workspace.
REFLEX_STREAM = torch.cuda.Stream() if torch.cuda.is_available() else None
TEXT_STREAM = torch.cuda.Stream() if torch.cuda.is_available() else None


# ── Reflex worker ─────────────────────────────────────────────────────


@torch.no_grad()
def reflex_worker(state: DemoState, model, tok, device, max_instr_tokens=96,
                  cycles_per_iter: int = 200, max_iterations: int = 3):
  """Iterative refinement.

  Each iteration runs up to ``cycles_per_iter`` grounded cycles. When an
  iteration ends (halt / crash / budget), the **register file + data/
  display memory are preserved** and a fresh refinement pass begins with
  PC reset to 0x1000 and the code region wiped. The model sees the
  resumed machine state and gets another chance to finish the task (e.g.
  emit the `sw` that a crashed program never reached). The loop exits
  when either:
    • the model emits HALT on the first cycle of a refinement pass (it
      inspected the state and decided no more work is needed), or
    • we hit ``max_iterations`` passes.
  """
  try:
    from reflex.model import render_prompt
    text = render_prompt(tok, state.prompt)
    e = tok(text, padding='max_length', truncation=True,
            max_length=max_instr_tokens, return_tensors='pt').to(device)
    ids, amask = e.input_ids, e.attention_mask

    cpu = Rv32i()
    cpu.uc.mem_write(PROGRAM_START, code_region_halt_fill())
    seed = b''.join(int(i).to_bytes(4, 'little') for i in range(1, 9))
    cpu.uc.mem_write(DATA_BASE + SRC_OFFSET, seed)

    state.reflex_start = time.time()
    global_cyc = 0
    for iteration in range(max_iterations):
        state.reflex_iter = iteration + 1
        state.reflex_total_iters = iteration + 1
        # For refinement passes: reset PC + wipe code region but keep
        # registers and data memory intact.
        if iteration > 0:
            try:
                cpu.uc.reg_write(UC_RISCV_REG_PC, PROGRAM_START)
            except Exception:
                pass
            cpu.uc.mem_write(PROGRAM_START, code_region_halt_fill())
            cpu.uc.ctl_remove_cache(PROGRAM_START, PROGRAM_START + 0x1000)
            state.reflex_ops.append(
                (-1, 0, 0, f'--- refinement pass {iteration+1} ---'))

        iter_halted_first = False
        iter_halted = False
        for iter_cyc in range(cycles_per_iter):
            pc = cpu.pc
            st = extract_state(cpu)
            st_t = torch.from_numpy(st.astype('int64')).unsqueeze(0).to(device)
            if REFLEX_STREAM is not None:
                with torch.cuda.stream(REFLEX_STREAM):
                    logits = model(ids, amask, st_t)
                REFLEX_STREAM.synchronize()       # needed before CPU reads
            else:
                logits = model(ids, amask, st_t)
            bits = (logits > 0).long().squeeze(0).tolist()
            w = 0
            for i, b in enumerate(bits):
                w |= (int(b) & 1) << i

            state.reflex_ops.append((global_cyc, pc, w, disasm(w)))
            state.reflex_op_count = global_cyc + 1
            state.reflex_regs = [cpu.reg(i) for i in range(32)]
            state.reflex_dt_ms = int((time.time() - state.reflex_start) * 1000)
            dbytes = []
            for i in range(32):
                b = cpu.mem_word(DISPLAY_BASE + 4*i) & 0xff
                dbytes.append(chr(b) if 32 <= b < 127
                              else ('·' if b == 0 else '?'))
            state.reflex_display = ''.join(dbytes).rstrip('·') or '·'

            # Did the model signal "satisfied" on the first cycle of a
            # refinement pass?
            if iteration > 0 and iter_cyc == 0 and w == HALT_INSTR:
                iter_halted_first = True

            try:
                cpu.uc.mem_write(pc, int(w & 0xFFFFFFFF).to_bytes(4, 'little'))
                cpu.uc.ctl_remove_cache(pc, pc + 4)
            except Exception:
                global_cyc += 1; break
            if w == HALT_INSTR:
                state.reflex_halted = True
                iter_halted = True
                global_cyc += 1
                break
            if w == 0:
                global_cyc += 1; break
            try:
                cpu.step()
            except Exception:
                global_cyc += 1; break
            global_cyc += 1

        # Stop refining if the model declared itself done immediately.
        if iter_halted_first:
            break
        # Or if the task result already landed.
        if iter_halted and cpu.mem_word(DATA_BASE) != 0:
            break

    state.reflex_final_mem = cpu.mem_word(DATA_BASE)
  except Exception as e:
    import traceback
    state.reflex_err = f'{type(e).__name__}: {e}\n{traceback.format_exc()}'
  finally:
    state.reflex_done = True


# ── Text streamer (unthrottled — stream at full GPU speed) ───────────
class RichStreamer(TextStreamer):
    """Subclass of HuggingFace's TextStreamer that updates a DemoState
    on every token. ``put()`` is called once per decoder step with a
    single token id, so it gives us a true per-token count."""
    def __init__(self, tok, state: DemoState):
        super().__init__(tok, skip_prompt=True, skip_special_tokens=True)
        self.state = state

    def put(self, value):
        super().put(value)
        # value is a 1-D LongTensor of new token IDs for this step.
        try:
            n = value.numel()
        except Exception:
            n = 1
        self.state.text_tok_count += n

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.state.text_out += text
        self.state.text_dt_ms = int(
            (time.time() - self.state.text_start) * 1000)


# ── Tiny assembler for the text-mode output ──────────────────────────
import re as _re

ASM_SYS_PROMPT = (
    "Write a RV32I assembly program using only base integer instructions "
    "(no M extension — no mul, div, rem). Available registers: x5-x15. "
    "Arithmetic results go to mem[0x5000]. The display buffer is at "
    "0x6000: ASCII bytes, one character per 4-byte word, so `mem[0x6000]` "
    "is the first character, `mem[0x6004]` the second, etc. Use EBREAK "
    "to halt.\n\n"
    "Allowed mnemonics: addi, add, sub, lui, lw, sw, jal, jalr, beq, bne, "
    "blt, bge, ebreak.\n"
    "Pseudo-ops: li rd, imm · mv rd, rs · neg rd, rs · j label · ret · "
    "beqz/bnez/bltz/bgez rs, label.\n"
    "Labels like `loop:` are allowed; branches/jumps may reference them.\n"
    "Memory ops use `offset(base_reg)` syntax ONLY — e.g. `sw x7, 0(x10)`. "
    "To reach an absolute address, load the base with `lui`: `lui x10, 5` "
    "sets x10 = 0x5000; `lui x10, 6` sets x10 = 0x6000.\n"
    "Prompts like 'say X', 'display X', 'show X', 'print X' ask you to "
    "WRITE EACH CHARACTER of X as ASCII to the display — digits too "
    "(e.g. 'say 42' writes '4' (0x34) at 0x6000 and '2' (0x32) at 0x6004, "
    "NOT the number 42 at 0x5000). Case matters: 'h'=0x68, 'H'=0x48.\n"
    "Everything else (add, subtract, factorial, etc.) is arithmetic: "
    "compute the result and store it at 0x5000.\n\n"
    "Output ONLY the assembly. One instruction per line. NO explanations, "
    "NO numbered steps, NO 'Explanation:' section. Stop immediately after "
    "the `ebreak` halt — emit nothing else.\n\n"
    "Example — arithmetic 5 + 8 stored at 0x5000:\n"
    "  addi x5, x0, 5\n"
    "  addi x6, x0, 8\n"
    "  add x7, x5, x6\n"
    "  lui x10, 5\n"
    "  sw x7, 0(x10)\n"
    "  ebreak\n\n"
    "Example — 'say hi' writes characters to the display at 0x6000:\n"
    "  lui x10, 6\n"
    "  addi x5, x0, 0x68\n"
    "  sw x5, 0(x10)\n"
    "  addi x5, x0, 0x69\n"
    "  sw x5, 4(x10)\n"
    "  ebreak\n\n"
    "Example — multiply 3 and 4 by repeated addition (NO `mul` — it's not "
    "in the base ISA):\n"
    "  addi x5, x0, 3\n"
    "  addi x6, x0, 4\n"
    "  addi x7, x0, 0\n"
    "loop:\n"
    "  beqz x6, done\n"
    "  add x7, x7, x5\n"
    "  addi x6, x6, -1\n"
    "  j loop\n"
    "done:\n"
    "  lui x10, 5\n"
    "  sw x7, 0(x10)\n"
    "  ebreak"
)


def _reg(s: str) -> int:
    s = s.strip().lower().rstrip(',')
    if s.startswith('x'):
        return int(s[1:])
    raise ValueError(f'bad reg: {s}')


def _imm(s: str) -> int:
    s = s.strip().rstrip(',')
    # Character literal: `'S'` or `'\n'`.
    if len(s) >= 3 and s[0] == "'" and s[-1] == "'":
        body = s[1:-1]
        if body.startswith('\\') and len(body) == 2:
            esc = {'n': 10, 't': 9, 'r': 13, '0': 0,
                   '\\': 92, "'": 39, '"': 34}
            return esc.get(body[1], ord(body[1]))
        if len(body) == 1:
            return ord(body)
    return int(s, 0)


MEM_RE = _re.compile(r'^\s*(-?(?:0x)?[\da-f]+)\s*\(\s*x(\d+)\s*\)\s*$',
                     _re.IGNORECASE)
LABEL_RE = _re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _resolve(tok: str, pc: int, labels: dict) -> int:
    """Numeric literal or label → PC-relative byte offset."""
    tok = tok.strip().rstrip(',')
    if tok in labels:
        return labels[tok] * 4 - pc
    return int(tok, 0)


def assemble_line(line: str, pc: int = 0, labels: dict | None = None):
    """Encode one assembly line to a 32-bit int, or None if skippable."""
    labels = labels or {}
    line = line.split('#')[0].split('//')[0].strip().rstrip(';').rstrip(',')
    if not line:
        return None
    parts = _re.split(r'[\s,]+', line, maxsplit=1)
    op = parts[0].lower()
    tail = parts[1] if len(parts) > 1 else ''
    operands = [t.strip() for t in _re.split(r',\s*', tail) if t.strip()]
    if op == 'nop':
        return addi(0, 0, 0)
    if op in ('halt', 'ebreak'):
        return halt()
    if op == 'ret':                             # pseudo: jalr x0, x1, 0
        return jalr(0, 1, 0)
    if op == 'j' and len(operands) == 1:        # pseudo: jal x0, target
        return jal(0, _resolve(operands[0], pc, labels))
    if op == 'mv' and len(operands) == 2:       # pseudo: addi rd, rs, 0
        return addi(_reg(operands[0]), _reg(operands[1]), 0)
    if op == 'li' and len(operands) == 2:       # pseudo: addi rd, x0, imm
        return addi(_reg(operands[0]), 0, _imm(operands[1]))
    if op == 'addi' and len(operands) == 3:
        return addi(_reg(operands[0]), _reg(operands[1]), _imm(operands[2]))
    if op == 'add' and len(operands) == 3:
        return add(_reg(operands[0]), _reg(operands[1]), _reg(operands[2]))
    if op == 'sub' and len(operands) == 3:
        return sub(_reg(operands[0]), _reg(operands[1]), _reg(operands[2]))
    if op == 'lui' and len(operands) == 2:
        return lui(_reg(operands[0]), _imm(operands[1]))
    if op == 'lw' and len(operands) == 2:
        m = MEM_RE.match(operands[1])
        if m:
            return lw(_reg(operands[0]), int(m.group(2)), int(m.group(1), 0))
    if op == 'sw' and len(operands) == 2:
        m = MEM_RE.match(operands[1])
        if m:
            return sw(_reg(operands[0]), int(m.group(2)), int(m.group(1), 0))
    if op == 'jal':
        if len(operands) == 1:
            tok = operands[0].strip()
            # `jal rd` (unusual, offset 0) vs `jal target` (pseudo: rd=x1).
            if tok.lower().startswith('x') and tok[1:].rstrip(',').isdigit():
                return jal(_reg(tok), 0)
            return jal(1, _resolve(tok, pc, labels))
        if len(operands) == 2:
            return jal(_reg(operands[0]),
                       _resolve(operands[1], pc, labels))
    if op == 'jalr' and len(operands) >= 1:
        return jalr(_reg(operands[0]),
                    _reg(operands[1]) if len(operands) > 1 else 0,
                    _imm(operands[2]) if len(operands) > 2 else 0)
    if op == 'beq' and len(operands) == 3:
        return beq(_reg(operands[0]), _reg(operands[1]),
                   _resolve(operands[2], pc, labels))
    if op == 'bne' and len(operands) == 3:
        return bne(_reg(operands[0]), _reg(operands[1]),
                   _resolve(operands[2], pc, labels))
    if op == 'blt' and len(operands) == 3:
        return blt(_reg(operands[0]), _reg(operands[1]),
                   _resolve(operands[2], pc, labels))
    if op == 'bge' and len(operands) == 3:
        return bge(_reg(operands[0]), _reg(operands[1]),
                   _resolve(operands[2], pc, labels))
    # ── compare-with-zero pseudo-ops ────────────────────────────
    if op == 'beqz' and len(operands) == 2:
        return beq(_reg(operands[0]), 0, _resolve(operands[1], pc, labels))
    if op == 'bnez' and len(operands) == 2:
        return bne(_reg(operands[0]), 0, _resolve(operands[1], pc, labels))
    if op == 'bltz' and len(operands) == 2:
        return blt(_reg(operands[0]), 0, _resolve(operands[1], pc, labels))
    if op == 'bgez' and len(operands) == 2:
        return bge(_reg(operands[0]), 0, _resolve(operands[1], pc, labels))
    if op == 'neg' and len(operands) == 2:      # neg rd, rs → sub rd, x0, rs
        return sub(_reg(operands[0]), 0, _reg(operands[1]))
    raise ValueError(f'unsupported: {line!r}')


def _extract_asm_block(asm_text: str) -> list[str]:
    """If the text contains a ```...``` fenced block, return only its
    contents. Otherwise return all lines. Stops at the closing fence so
    trailing prose (like 'Explanation: 1. Load constants...') doesn't
    reach the assembler."""
    lines = asm_text.splitlines()
    start = end = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith('```'):
            if start is None:
                start = i + 1
            else:
                end = i
                break
    if start is not None:
        return lines[start:end] if end is not None else lines[start:]
    return lines


def assemble_program(asm_text: str):
    """Two-pass assembler: first pass collects label → instruction-index
    mapping, second pass emits words with label references resolved to
    PC-relative byte offsets. Returns (bytes, parsed_lines, errors)."""
    raw = _extract_asm_block(asm_text)
    # ── Pass 1: flatten to (line_text,) list, collecting labels. ──
    labels: dict[str, int] = {}
    body: list[str] = []
    for ln in raw:
        s = ln.split('#')[0].split('//')[0].strip()
        if not s:
            continue
        # A line may carry an inline label: `loop: addi x5, x0, 1`.
        while ':' in s:
            head, _, rest = s.partition(':')
            head = head.strip()
            if LABEL_RE.match(head):
                labels[head] = len(body)
                s = rest.strip()
                if not s:
                    break
                continue
            break
        if not s:
            continue
        body.append(s)

    # ── Pass 2: assemble with label context. ──
    ops, parsed, errors = [], [], []
    for i, line in enumerate(body):
        pc = i * 4
        try:
            w = assemble_line(line, pc, labels)
            if w is None:
                continue
            ops.append(w)
            parsed.append(line)
        except Exception as e:
            errors.append(f'{line}  → {e}')
    return pack(*ops) if ops else b'', parsed, errors


@torch.no_grad()
def text_worker(state: DemoState, causal_lm, tok, device, max_new_tokens=256):
    try:
        msgs = [{"role": "system", "content": ASM_SYS_PROMPT},
                {"role": "user", "content": state.prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        enc = tok(text, return_tensors='pt').to(device)
        state.text_start = time.time()
        streamer = RichStreamer(tok, state)
        if TEXT_STREAM is not None:
            with torch.cuda.stream(TEXT_STREAM):
                causal_lm.generate(
                    enc.input_ids, attention_mask=enc.attention_mask,
                    max_new_tokens=max_new_tokens, do_sample=False,
                    repetition_penalty=1.15,
                    pad_token_id=tok.pad_token_id, streamer=streamer)
        else:
            causal_lm.generate(
                enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tok.pad_token_id, streamer=streamer)

        # ── Assemble the streamed program and execute in a fresh CPU. ──
        asm_text = state.text_out            # snapshot before annotation
        state.text_out += '\n--- assembling ---\n'
        prog_bytes, parsed, errors = assemble_program(asm_text)
        if errors:
            state.text_out += f'assembler: {len(errors)} lines skipped\n'
        if not prog_bytes:
            state.text_out += 'no program assembled\n'
            return
        cpu = Rv32i()
        cpu.uc.mem_write(PROGRAM_START, code_region_halt_fill())
        seed = b''.join(int(i).to_bytes(4, 'little') for i in range(1, 9))
        cpu.uc.mem_write(DATA_BASE + SRC_OFFSET, seed)
        cpu.uc.mem_write(PROGRAM_START, prog_bytes)
        cpu.uc.ctl_remove_cache(PROGRAM_START, PROGRAM_START + len(prog_bytes))
        halted = False
        for _ in range(2000):
            try:
                instr = cpu.fetch()
            except Exception:
                break
            if instr == HALT_INSTR:
                halted = True; break
            if instr == 0:
                break
            try:
                cpu.step()
            except Exception:
                break
        state.text_final_mem = cpu.mem_word(DATA_BASE)
        state.text_regs = [cpu.reg(i) for i in range(32)]
        state.text_display = ''.join(
            (chr(b & 0xff) if 32 <= (b & 0xff) < 127 else ('·' if b == 0 else '?'))
            for b in [cpu.mem_word(DISPLAY_BASE + 4*i) for i in range(32)]
        ).rstrip('·') or '·'
        state.text_halted = halted
        state.text_ops_run = len(parsed)
    except Exception as e:
        import traceback
        state.text_err = f'{type(e).__name__}: {e}\n{traceback.format_exc()}'
    finally:
        state.text_done = True


# ── Rendering ────────────────────────────────────────────────────────
def render_layout(state: DemoState) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name='header', size=3),
        Layout(name='top', ratio=5),
        Layout(name='bottom', size=3),
    )
    layout['top'].split_row(Layout(name='left'), Layout(name='right'))
    # Both columns split vertically: flex stream on top, fixed-height
    # register table in the middle, and a dedicated display box below.
    for side in ('left', 'right'):
        layout[side].split_column(
            Layout(name=f'{side}_ops'),           # flex
            Layout(name=f'{side}_regs', size=12), # 8 reg rows + borders + pad
            Layout(name=f'{side}_disp', size=5),  # display line + mem line
        )

    # ── Header: current / most recent task ───────────────────────
    if state.prompt:
        header_body = Text.assemble(
            ('Task: ', 'bold cyan'),
            (state.prompt, 'bold white'),
        )
    else:
        header_body = Text('(no task yet — type one below)', style='dim')
    layout['header'].update(Panel(header_body, border_style='cyan'))

    # Helpers shared by both sides.
    def reg_grid(regs):
        t = Table.grid(padding=(0, 1))
        for _ in range(4):
            t.add_column(style='cyan', no_wrap=True)
            t.add_column(no_wrap=True)
        for row in range(8):
            cells = []
            for col in range(4):
                r = row + col * 8
                v = regs[r]
                cells.append(f'x{r:<2d}')
                cells.append(Text(f'0x{v:08x}',
                                  style='bold white' if v else 'dim'))
            t.add_row(*cells)
        return t

    def disp_panel(display, mem, border, halted=None, title='result'):
        body = Group(
            Text(f'display: [{display or "·"}]', style='bold yellow'),
            Text(f'mem[0x5000] = 0x{mem:x} = {mem}'
                 + (f'   halted={halted}' if halted is not None else ''),
                 style='magenta'),
        )
        return Panel(body, title=title, border_style=border)

    # ── Reflex (left) ────────────────────────────────────────────
    # Keep the newest N ops that fit the flex region.
    total_rows = shutil.get_terminal_size((120, 40)).lines
    ops_budget = max(5, total_rows - 3 - 3 - 12 - 5 - 6)   # header/bot/regs/disp/borders
    op_lines = Text()
    for cyc, pc, w, dis in list(state.reflex_ops)[-ops_budget:]:
        if cyc < 0:
            op_lines.append(f'\n{dis}\n', style='bold yellow')
            continue
        op_lines.append(f'{cyc:3d}  0x{pc:04x}  {w:08x}  {dis}\n',
                        style='green' if state.reflex_halted else None)

    status = ('HALTED' if state.reflex_halted else
              ('RUNNING' if not state.reflex_done else 'STOPPED'))
    iter_str = (f'iter {state.reflex_iter}/3'
                if state.reflex_total_iters > 1 or not state.reflex_done
                else '')
    # One forward pass = one emitted 32-bit instruction word — the same
    # "one forward = one token" accounting used for text mode, just with
    # a 4-byte machine-code token instead of a vocab token.
    parts = [f'{state.reflex_op_count} tokens (= ops)',
             f'{state.reflex_dt_ms} ms']
    if iter_str: parts.append(iter_str)
    parts.append(status)
    reflex_footer = ' · '.join(parts)

    layout['left_ops'].update(Panel(
        op_lines,
        title='[bold green]Reflex[/] — opcode stream',
        border_style='green', subtitle=reflex_footer))
    layout['left_regs'].update(Panel(
        reg_grid(state.reflex_regs),
        title='registers', border_style='green'))
    layout['left_disp'].update(disp_panel(
        state.reflex_display, state.reflex_final_mem, 'green',
        halted=state.reflex_halted))

    # ── Text mode (right) ────────────────────────────────────────
    if state.text_err:
        body_inner = Text(state.text_err, style='red')
        status_t = 'ERROR'
    elif not state.text_out and not state.text_done:
        wait = (time.time() - state.text_start) if state.text_start else 0
        body_inner = Text(f'[prefilling… {wait:.1f}s]', style='dim')
        status_t = 'PREFILL'
    else:
        body_inner = Text(state.text_out, style='white')
        status_t = 'DONE' if state.text_done else 'STREAMING'
    text_footer = (f'{int(state.text_tok_count)} tokens · '
                   f'{state.text_dt_ms} ms · '
                   f'{state.text_ops_run} asm ops · {status_t}')
    layout['right_ops'].update(Panel(
        body_inner,
        title='[bold blue]Text mode[/] — writes asm → assembled → run in Unicorn',
        border_style='blue', subtitle=text_footer))
    layout['right_regs'].update(Panel(
        reg_grid(state.text_regs),
        title='registers', border_style='blue'))
    layout['right_disp'].update(disp_panel(
        state.text_display, state.text_final_mem, 'blue',
        halted=state.text_halted if state.text_done else None))

    # ── Bottom ───────────────────────────────────────────────────
    if state.phase == 'input':
        blink = '▌' if int(time.time() * 2) % 2 == 0 else ' '
        body = Text.assemble(
            ('task> ', 'bold cyan'),
            (state.input_buffer, 'bold white'),
            (blink, 'bold white'),
        )
        title = '[dim]type a task · Enter = run · empty = quit[/]'
        border = 'cyan'
    else:
        body = Text(state.prompt, style='bold')
        if state.phase == 'running':
            title = '[dim]running…[/]'; border = 'yellow'
        else:
            title = '[dim]done · press Enter for next task[/]'
            border = 'green'
    layout['bottom'].update(Panel(body, title=title, border_style=border))
    return layout


# ── Raw-character input so the prompt renders inside the Live panel ──
def read_key(timeout: float = 0.1):
    """Non-blocking single-keystroke read. Returns '' on timeout.

    Arrow keys and other function keys send multi-char ESC sequences
    (``\\x1b [ A``, etc). Consume and discard them so they don't land
    in the input buffer as literal text.
    """
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if not r:
        return ''
    ch = sys.stdin.read(1)
    if ch == '\x1b':
        # Drain any follow-up bytes of an escape sequence.
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(1)
        return ''
    return ch


def input_phase(state: DemoState, live: Live) -> str:
    """Read a line into the bottom panel, returning the trimmed prompt."""
    state.phase = 'input'
    state.input_buffer = ''
    while True:
        ch = read_key(0.05)
        live.update(render_layout(state))
        if not ch:
            continue
        if ch in ('\n', '\r'):
            return state.input_buffer.strip()
        if ch == '\x7f':                          # backspace
            state.input_buffer = state.input_buffer[:-1]
        elif ch == '\x03':                        # Ctrl-C
            raise KeyboardInterrupt
        elif ch == '\x04':                        # Ctrl-D
            raise EOFError
        elif ch.isprintable():
            state.input_buffer += ch


def reset_for_run(state: DemoState) -> None:
    state.reflex_ops.clear()
    state.reflex_op_count = 0
    state.reflex_regs = [0] * 32
    state.reflex_display = ''
    state.reflex_dt_ms = 0; state.reflex_start = None
    state.reflex_done = False; state.reflex_halted = False
    state.reflex_final_mem = 0; state.reflex_err = ''
    state.reflex_iter = 1; state.reflex_total_iters = 1
    state.text_out = ''; state.text_tok_count = 0; state.text_dt_ms = 0
    state.text_start = None; state.text_done = False; state.text_err = ''
    state.text_final_mem = 0; state.text_display = ''
    state.text_halted = False; state.text_ops_run = 0
    state.text_regs = [0] * 32
    # Flush prior CUDA state so one run's allocations can't poison the next.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def run_prompt(state: DemoState, live: Live,
               model, tok, causal_lm, tok_text, device):
    """Launch both workers, update the Live display until both finish."""
    reset_for_run(state)
    state.phase = 'running'
    t_r = threading.Thread(target=reflex_worker,
                           args=(state, model, tok, device), daemon=True)
    t_t = threading.Thread(target=text_worker,
                           args=(state, causal_lm, tok_text, device), daemon=True)
    t_r.start(); t_t.start()
    while not (state.reflex_done and state.text_done):
        live.update(render_layout(state))
        time.sleep(0.05)
    state.phase = 'done'
    live.update(render_layout(state))


def main():
    ap = argparse.ArgumentParser(
        description='Reflex interactive demo: side-by-side grounded '
        'CPU emission vs text-mode assembly generation.')
    ap.add_argument('--ckpt', '--checkpoint', required=True,
                    dest='ckpt', metavar='PATH',
                    help='Path to the trained Reflex checkpoint.')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'

    console = Console()
    console.print(f'[dim]loading Reflex from {args.ckpt}...[/]')
    model, tok, cfg = load(args.ckpt, device)
    console.print(f'[dim]loading text-mode causal-LM copy of '
                  f'{cfg["backbone_id"]}...[/]')
    bb = AutoModelForCausalLM.from_pretrained(
        cfg['backbone_id'], torch_dtype=torch.bfloat16).to(device).eval()
    tok2 = AutoTokenizer.from_pretrained(cfg['backbone_id'])
    if tok2.pad_token is None:
        tok2.pad_token = tok2.eos_token
    console.print('[bold green]ready.[/]\n')

    state = DemoState()
    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        with Live(render_layout(state), refresh_per_second=20,
                  screen=True) as live:
            while True:
                try:
                    prompt = input_phase(state, live)
                except (EOFError, KeyboardInterrupt):
                    break
                if not prompt:
                    break
                state.prompt = prompt
                run_prompt(state, live, model, tok, bb, tok2, device)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)


if __name__ == '__main__':
    main()
