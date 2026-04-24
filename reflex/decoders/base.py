"""Shared helpers used by every decoder.

Kept intentionally small — no decoder-specific logic here. Anything more
than prompt tokenization, CPU setup, or a single model forward belongs
in the decoder that needs it.
"""
import numpy as np
import torch

from ..model import (
    MAX_INSTR_TOKENS, code_region_halt_fill, extract_state, render_prompt,
)
from ..programs import SRC_OFFSET
from ..riscv import DATA_BASE, HALT_INSTR, PROGRAM_START, Rv32i


def prep_prompt(tok, prompt: str, device: str, *,
                max_instr_tokens: int = MAX_INSTR_TOKENS,
                use_chat_template: bool = True,
                use_context_prefix: bool = False):
    """Tokenize once per task. Returns (input_ids, attention_mask)."""
    text = render_prompt(tok, prompt,
                         use_chat_template=use_chat_template,
                         use_context_prefix=use_context_prefix)
    e = tok(text, padding='max_length', truncation=True,
            max_length=max_instr_tokens, return_tensors='pt').to(device)
    return e.input_ids, e.attention_mask


def fresh_cpu(seed_memcpy: bool = True) -> Rv32i:
    """A fresh Rv32i with the HALT-filled code region and optional
    memcpy seed data. Matches ``run_grounded``'s initial setup."""
    cpu = Rv32i()
    cpu.uc.mem_write(PROGRAM_START, code_region_halt_fill())
    if seed_memcpy:
        seed = b''.join(int(i).to_bytes(4, 'little') for i in range(1, 9))
        cpu.uc.mem_write(DATA_BASE + SRC_OFFSET, seed)
    return cpu


@torch.no_grad()
def predict_topk(model, ids: torch.Tensor, amask: torch.Tensor,
                 cpu: Rv32i, device: str, topk: int = 1):
    """One model forward at the current CPU state. Returns
    ``(top_words, top_sims)`` where ``top_words`` is a list of int 32-bit
    words and ``top_sims`` is a list of cosine similarities, both sorted
    descending by similarity."""
    state = extract_state(cpu)
    state_t = torch.from_numpy(state.astype('int64')).unsqueeze(0).to(device)
    pred = model(ids, amask, state_t)
    sims = model.table_similarity(pred).squeeze(0)
    top = torch.topk(sims, topk)
    top_sims = [float(s) for s in top.values.tolist()]
    top_words = [int(model.instr_words[i].item()) & 0xFFFFFFFF
                 for i in top.indices.tolist()]
    return top_words, top_sims


def emit_and_step(cpu: Rv32i, pc: int, instr_w: int) -> tuple[bool, str]:
    """Write ``instr_w`` at ``pc``, invalidate Unicorn's decode cache,
    and step once. Returns ``(halted, err)``. The HALT instruction is
    treated as a clean exit; ``0x00000000`` is treated as an error to
    match the canonical ``run_grounded`` stop conditions."""
    try:
        cpu.uc.mem_write(pc, int(instr_w & 0xFFFFFFFF).to_bytes(4, 'little'))
        cpu.uc.ctl_remove_cache(pc, pc + 4)
    except Exception as e:
        return False, f'write@{pc:#x}: {e}'
    if instr_w == HALT_INSTR:
        return True, ''
    if instr_w == 0:
        return False, f'zero@{pc:#x}'
    try:
        cpu.step()
    except Exception as e:
        return False, f'step@{pc:#x}: {e}'
    return False, ''


def replay_committed(committed: list[int], seed_memcpy: bool) -> Rv32i | None:
    """Rebuild a CPU by replaying a list of committed instruction words
    from a fresh start. Used by decoders that need to fork from an
    intermediate committed state (exec_verify, beam, lookahead probes).
    Returns None if any replayed step fails — the caller should treat
    that as the candidate being unreachable from current state."""
    cpu = fresh_cpu(seed_memcpy)
    for w in committed:
        pc = cpu.pc
        halted, err = emit_and_step(cpu, pc, w)
        if err:
            return None
        if halted:
            return cpu
    return cpu
