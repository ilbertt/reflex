"""Grounded RV32I inference helpers.

Library module shared by ``demo_tui`` and ``eval``. ``load()`` builds a
``GroundedReflex`` from a checkpoint; ``run_grounded()`` drives the
emit-one-cycle / step-Unicorn loop that is the model's inference path.
"""
import os

import torch

from .model import (
    EMBED_DIM, INJECT_EVERY, MAX_INSTR_TOKENS, GroundedReflex, build_backbone,
    code_region_halt_fill, extract_state, render_prompt,
)
from .programs import SRC_OFFSET
from .riscv import DATA_BASE, HALT_INSTR, PROGRAM_START, Rv32i

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


def load(ckpt_path: str, device: str = 'cuda'):
    """Load a checkpoint and return (model, tok, cfg).

    ``cfg`` exposes the prompt-rendering mode the checkpoint was trained
    with so callers can mirror it at inference.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    bb, tok, hidden = build_backbone(cfg['backbone_id'], dtype=dtype)
    bb = bb.to(device)
    model = GroundedReflex(
        bb, cfg['hidden'],
        num_instrs=cfg['num_instrs'],
        inject_every=cfg.get('inject_every', INJECT_EVERY),
        adapter_mlp_ratio=cfg.get('adapter_mlp_ratio', 2),
        embed_dim=cfg.get('embed_dim', EMBED_DIM),
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
    """Drive one grounded-emission session for ``instruction``. Returns
    ``(cpu, emitted_words, halted, err_msg)``. Stops on HALT, on an
    emitted zero word, or when ``max_cycles`` is exhausted."""
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
        pred = model(ids, amask, state_t)
        instr_w = int(model.decode_words(pred).item()) & 0xFFFFFFFF
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
