"""
Grounded RV32I demo.

Encode instruction once. Fresh Rv32i (code region pre-filled with HALT).
Each cycle:
  read live state → model emits one 4-byte instr → write at pc → step.
Stops on HALT (jal x0, 0) or ``--max-cycles``.

Usage:
    uv run demo                                # 8 canonical examples
    uv run demo --instruction "add 7 and 8 and store the result"
"""
import argparse
import os
import sys

import torch

from .model_lora import (
    BACKBONE_ID, CTRL_DIM, FIELD_CLASSES, MAX_INSTR_TOKENS, N_XATTN_LAYERS,
    LoraReflex, build_backbone, code_region_halt_fill, extract_state,
)
from .programs import DST_OFFSET, SRC_OFFSET
from .riscv import DATA_BASE, HALT_INSTR, PROGRAM_START, Rv32i, compose

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

EXAMPLES = [
    'add 7 and 8 and store the result',
    'compute 5 factorial and store it',
    'store the first 6 Fibonacci numbers',
    'count down from 5 to 1 and store each value',
    'compute 1 + 2 + ... + 10 and store the sum',
    'find the max of 7 and 12 and store it',
    'copy 4 words from source to destination',
    'call a function that doubles 25 and store the result',
]


def load(ckpt_path: str, device: str = 'cuda'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    use_lora = not cfg.get('freeze_backbone', True)
    bb, tok, hidden = build_backbone(
        cfg['backbone_id'], use_lora=use_lora, dtype=torch.float16)
    bb = bb.to(device)
    model = LoraReflex(
        bb, cfg['hidden'], cfg['ctrl_dim'], cfg['n_xattn'],
        cfg.get('freeze_backbone', True)).to(device)
    model.load_state_dict(ckpt['state'], strict=False)
    model.eval()
    return model, tok


@torch.no_grad()
def run_grounded(model, tok, instruction: str, device: str = 'cuda',
                 max_cycles: int = 200, seed_memcpy: bool = True,
                 verbose: bool = False) -> tuple[Rv32i, list[int], bool, str]:
    e = tok(instruction, padding='max_length', truncation=True,
            max_length=MAX_INSTR_TOKENS, return_tensors='pt').to(device)
    ids, amask = e.input_ids, e.attention_mask
    with torch.autocast('cuda', dtype=torch.float16,
                        enabled=(device == 'cuda')):
        instr_h = model.encode_instruction(ids, amask)

    cpu = Rv32i()
    cpu.uc.mem_write(PROGRAM_START, code_region_halt_fill())
    if seed_memcpy:
        seed = b''.join(int(i).to_bytes(4, 'little') for i in range(1, 9))
        cpu.uc.mem_write(DATA_BASE + SRC_OFFSET, seed)

    emitted: list[int] = []
    h_state = torch.zeros(1, model.ctrl_dim, device=device)
    prev_fields = [torch.zeros(1, dtype=torch.long, device=device)
                   for _ in FIELD_CLASSES]

    halted = False
    err = ''
    for cycle in range(max_cycles):
        pc = cpu.pc
        state = extract_state(cpu)
        state_t = torch.from_numpy(state.astype('int64')).unsqueeze(0).to(device)
        with torch.autocast('cuda', dtype=torch.float16,
                            enabled=(device == 'cuda')):
            logits, h_state = model.decode_step(
                instr_h, amask, state_t, prev_fields, h_state)
        fields = [int(lg.argmax(-1).item()) for lg in logits]
        instr_w = compose(*fields)
        emitted.append(instr_w)
        if verbose:
            print(f'  cyc {cycle:3d}  pc=0x{pc:04X}  {instr_w:08X}')
        try:
            cpu.uc.mem_write(pc, int(instr_w & 0xFFFFFFFF).to_bytes(4, 'little'))
            cpu.uc.ctl_remove_cache(pc, pc + 4)
        except Exception as e_:
            err = f'write at pc=0x{pc:X}: {e_}'; break
        if instr_w == HALT_INSTR:
            halted = True; break
        if instr_w == 0:
            err = f'emitted 0x0 at pc=0x{pc:X}'; break
        try:
            cpu.step()
        except Exception as e_:
            err = f'step at pc=0x{pc:X}: {e_}'; break

        prev_fields = [torch.tensor([v], dtype=torch.long, device=device)
                       for v in fields]

    return cpu, emitted, halted, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='reflex_grounded.pt')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-cycles', type=int, default=200)
    ap.add_argument('--instruction', default=None)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    print(f'loading {args.ckpt} on {device}', flush=True)
    model, tok = load(args.ckpt, device)
    print('loaded.\n', flush=True)

    instrs = [args.instruction] if args.instruction else EXAMPLES
    for ex in instrs:
        cpu, emitted, halted, err = run_grounded(
            model, tok, ex, device, args.max_cycles, verbose=args.verbose)
        marker = '✓' if halted and not err else '✗'
        status = f'halted={halted}' + (f' err={err}' if err else '')
        print(f'{marker} {ex!r}  ops={len(emitted)}  {status}')
        if args.verbose:
            for i, w in enumerate(emitted):
                print(f'   cyc {i:3d}  {w:08X}')
        mem_data = [cpu.mem_word(DATA_BASE + 4*i) for i in range(8)]
        print(f'   mem[DATA_BASE..+32] = {mem_data}')
        dst_start = cpu.mem_word(DATA_BASE + DST_OFFSET)
        if dst_start != 0:
            dst = [cpu.mem_word(DATA_BASE + DST_OFFSET + 4*i) for i in range(8)]
            print(f'   mem[DST..+32]       = {dst}')
        print()


if __name__ == '__main__':
    main()
