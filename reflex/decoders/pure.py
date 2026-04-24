"""Canonical decoder — argmax over the JEPA codebook.

    state → backbone forward → pooled → 256-d embedding
         → cosine NN over 691 codebook rows
         → argmax → 32-bit word
         → write at PC, step Unicorn once

No interventions. No lookahead. No branching. One model forward per
cycle, one write, one step. This is the baseline the README describes
as the "thinnest possible bridge".

Exists in this package so callers can select it by name alongside the
alternatives — behaviour is otherwise identical to ``run_grounded``.
"""
from ..model import MAX_INSTR_TOKENS
from ..riscv import Rv32i
from .base import emit_and_step, fresh_cpu, predict_topk, prep_prompt


def run(model, tok, prompt: str, device: str, max_cycles: int, *,
        seed_memcpy: bool = True,
        max_instr_tokens: int = MAX_INSTR_TOKENS,
        use_chat_template: bool = True,
        use_context_prefix: bool = False,
        **_unused,
        ) -> tuple[Rv32i, list[int], bool, str, dict]:
    ids, amask = prep_prompt(tok, prompt, device,
                              max_instr_tokens=max_instr_tokens,
                              use_chat_template=use_chat_template,
                              use_context_prefix=use_context_prefix)
    cpu = fresh_cpu(seed_memcpy)
    emitted: list[int] = []
    halted, err = False, ''
    for cycle in range(max_cycles):
        pc = cpu.pc
        top_words, _ = predict_topk(model, ids, amask, cpu, device, topk=1)
        w = top_words[0]
        emitted.append(w)
        halted, err = emit_and_step(cpu, pc, w)
        if halted or err:
            break
    return cpu, emitted, halted, err, {'decoder': 'pure'}
