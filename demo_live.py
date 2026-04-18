"""Grounded demo with a live display buffer.

As the model emits opcodes, we step Unicorn one instruction at a time
and re-read the 32-word display buffer at 0x6000. Whenever a word
changes, we print the ASCII char that just appeared (or a hex byte if
it's non-printable), so the user sees letters light up in order.
"""
import argparse
import sys
import time

import torch

from reflex.demo import load
from reflex.model import CONTEXT_PREFIX, MAX_INSTR_TOKENS, extract_state
from reflex.programs import DISPLAY_BASE, SRC_OFFSET
from reflex.riscv import DATA_BASE, HALT_INSTR, PROGRAM_START, Rv32i
from reflex.model import code_region_halt_fill

DISPLAY_WORDS = 32               # how many bytes to render


def read_display(cpu: Rv32i) -> list[int]:
    return [cpu.mem_word(DISPLAY_BASE + 4 * i) for i in range(DISPLAY_WORDS)]


def pretty(b: int) -> str:
    b &= 0xFF
    if 32 <= b < 127:
        return chr(b)
    if b == 0:
        return '·'
    return f'\\x{b:02x}'


def render(buf: list[int]) -> str:
    return ''.join(pretty(w) for w in buf)


@torch.no_grad()
def run_live(model, tok, instruction, device, max_cycles=400,
             context_prefix=True, max_instr_tokens=128, verbose=False):
    text = (CONTEXT_PREFIX + instruction) if context_prefix else instruction
    e = tok(text, padding='max_length', truncation=True,
            max_length=max_instr_tokens, return_tensors='pt').to(device)
    ids, amask = e.input_ids, e.attention_mask

    cpu = Rv32i()
    cpu.uc.mem_write(PROGRAM_START, code_region_halt_fill())
    seed = b''.join(int(i).to_bytes(4, 'little') for i in range(1, 9))
    cpu.uc.mem_write(DATA_BASE + SRC_OFFSET, seed)

    prev_display = read_display(cpu)
    print(f'  screen: [{render(prev_display)}]')
    halted = False
    err = ''
    n_ops = 0
    for cycle in range(max_cycles):
        pc = cpu.pc
        state = extract_state(cpu)
        state_t = torch.from_numpy(state.astype('int64')).unsqueeze(0).to(device)
        logits = model(ids, amask, state_t)
        bits = (logits > 0).long().squeeze(0).tolist()
        instr_w = 0
        for i, b in enumerate(bits):
            instr_w |= (int(b) & 1) << i
        n_ops += 1
        try:
            cpu.uc.mem_write(pc, int(instr_w & 0xFFFFFFFF).to_bytes(4, 'little'))
            cpu.uc.ctl_remove_cache(pc, pc + 4)
        except Exception as e_:
            err = f'write@0x{pc:x}: {e_}'; break
        if instr_w == HALT_INSTR:
            halted = True; break
        if instr_w == 0:
            err = f'emitted 0x0 at 0x{pc:x}'; break
        try:
            cpu.step()
        except Exception as e_:
            err = f'step@0x{pc:x}: {e_}'; break

        disp = read_display(cpu)
        if disp != prev_display:
            for i, (old, new) in enumerate(zip(prev_display, disp)):
                if old != new:
                    print(f'    cyc {cycle:3d}  display[{i:2d}] = {pretty(new)} '
                          f'(0x{new & 0xFF:02x})  pc was 0x{pc:04x}')
            prev_display = disp
            print(f'  screen: [{render(disp)}]')
        elif verbose:
            print(f'    cyc {cycle:3d}  pc=0x{pc:04x}  instr=0x{instr_w:08x}')

    print(f'  final screen: [{render(read_display(cpu))}]  '
          f'ops={n_ops}  halted={halted}' + (f'  err={err}' if err else ''))
    return cpu, halted, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--instruction', default=None,
                    help='One prompt to run. If omitted, runs a canonical list.')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-cycles', type=int, default=400)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'

    model, tok = load(args.ckpt, device)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    cp = cfg.get('context_prefix', False)
    mt = cfg.get('max_instr_tokens', 32)
    del ckpt
    print(f'loaded.  context_prefix={cp}  max_instr_tokens={mt}\n')

    examples = [args.instruction] if args.instruction else [
        # Trained display prompts
        'say hi',
        'display OK',
        'print hello',
        'show 42',
        # Novel / OOD display prompts
        'draw a box',
        'write your name',
        'display the result of 3+4',
    ]
    for ex in examples:
        print(f'▶ {ex!r}')
        run_live(model, tok, ex, device, max_cycles=args.max_cycles,
                 context_prefix=cp, max_instr_tokens=mt,
                 verbose=args.verbose)
        print()


if __name__ == '__main__':
    main()
