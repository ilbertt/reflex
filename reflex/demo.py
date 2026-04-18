"""
Grounded RV32I demo (Flamingo-style fusion).

Fresh Rv32i (code region pre-filled with HALT). Each cycle:
  read live state → model (instruction + state fused inside the
  backbone) emits one 4-byte instr → write at pc → step.
Stops on HALT (jal x0, 0) or ``--max-cycles``.

Usage:
    uv run demo                                # 8 canonical examples
    uv run demo --instruction "add 7 and 8 and store the result"
"""
import argparse
import os

import torch

from .model import (
    INJECT_EVERY, MAX_INSTR_TOKENS, GroundedReflex, build_backbone,
    code_region_halt_fill, extract_state, render_prompt,
)
from .programs import DST_OFFSET, SRC_OFFSET
from .riscv import DATA_BASE, HALT_INSTR, PROGRAM_START, Rv32i

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
    """Load a checkpoint and return (model, tok, cfg).

    ``cfg`` exposes the prompt-rendering mode the checkpoint was trained
    with so callers can mirror it at inference. Defaults match the
    current chat-template training; legacy checkpoints (e.g.
    reflex_3b_ctx.pt) carry context_prefix=True and are routed through
    the plain-text prefix path by render_prompt.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    bb, tok, hidden = build_backbone(cfg['backbone_id'], dtype=dtype)
    bb = bb.to(device)
    model = GroundedReflex(
        bb, cfg['hidden'],
        inject_every=cfg.get('inject_every', INJECT_EVERY),
        adapter_mlp_ratio=cfg.get('adapter_mlp_ratio', 2),
        freeze_backbone=True).to(device)
    model.load_state_dict(ckpt['state'], strict=False)
    model.eval()
    return model, tok, cfg


@torch.no_grad()
def run_grounded(model, tok, instruction: str, device: str = 'cuda',
                 max_cycles: int = 200, seed_memcpy: bool = True,
                 verbose: bool = False,
                 max_instr_tokens: int = MAX_INSTR_TOKENS,
                 use_chat_template: bool = True,
                 use_context_prefix: bool = False,
                 ) -> tuple[Rv32i, list[int], bool, str]:
    text = render_prompt(tok, instruction,
                         use_chat_template=use_chat_template,
                         use_context_prefix=use_context_prefix)
    e = tok(text, padding='max_length', truncation=True,
            max_length=max_instr_tokens, return_tensors='pt').to(device)
    ids, amask = e.input_ids, e.attention_mask

    cpu = Rv32i()
    cpu.uc.mem_write(PROGRAM_START, code_region_halt_fill())
    if seed_memcpy:
        seed = b''.join(int(i).to_bytes(4, 'little') for i in range(1, 9))
        cpu.uc.mem_write(DATA_BASE + SRC_OFFSET, seed)

    emitted: list[int] = []
    halted = False
    err = ''
    for cycle in range(max_cycles):
        pc = cpu.pc
        state = extract_state(cpu)
        state_t = torch.from_numpy(state.astype('int64')).unsqueeze(0).to(device)
        logits = model(ids, amask, state_t)
        bits = (logits > 0).long().squeeze(0).tolist()
        instr_w = 0
        for i, b in enumerate(bits):
            instr_w |= (int(b) & 1) << i
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

    return cpu, emitted, halted, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='reflex.pt')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-cycles', type=int, default=200)
    ap.add_argument('--instruction', default=None)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    print(f'loading {args.ckpt} on {device}', flush=True)
    model, tok, cfg = load(args.ckpt, device)
    max_tok = cfg.get('max_instr_tokens', MAX_INSTR_TOKENS)
    # Legacy ckpts default to False/False; current ckpts to True/False.
    use_chat = cfg.get('chat_template', True)
    use_prefix = cfg.get('context_prefix', False)
    print(f'loaded.  chat_template={use_chat}  context_prefix={use_prefix}  '
          f'max_instr_tokens={max_tok}\n', flush=True)

    instrs = [args.instruction] if args.instruction else EXAMPLES
    for ex in instrs:
        cpu, emitted, halted, err = run_grounded(
            model, tok, ex, device, args.max_cycles, verbose=args.verbose,
            max_instr_tokens=max_tok,
            use_chat_template=use_chat, use_context_prefix=use_prefix)
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
