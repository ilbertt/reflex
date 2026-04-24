"""Margin-gated tiebreaker decoder.

Augments the canonical argmax path with two narrow rules, each triggered
only when the codebook's top-1 / top-2 similarities are close AND we
are past the first few cycles (which govern register allocation and are
brittle to override):

    R1 (halt-guard)
        If top-1 is HALT (``0x0000006F``) or ``0x00000000`` and the
        margin to top-2 is below ``halt_margin_eps``, prefer top-2.
        Rationale: committing to program termination on weak evidence is
        a high-cost error; the program can still recover from emitting
        a non-halt instruction, but cannot recover from a premature
        halt.

    R2 (one-step self-consistency)
        If the margin is below ``margin_eps``, for each of the top-``k``
        candidates, run the model forward ONE more time on a trial CPU
        that has been stepped under that candidate. Pick the candidate
        whose induced next-cycle top-1 similarity is highest.

Both rules are disabled before ``min_cycle`` (default 2). Early cycles
set up register conventions for the whole program; overriding there
was empirically destructive.

Empirical outcome on the released JEPA checkpoint's 23-task suite:
same 19/23 as the canonical decoder — tiebreakers neither rescued nor
regressed. Shipped anyway because the rules are defensible and the
infrastructure is useful for future experiments.
"""
from ..model import MAX_INSTR_TOKENS
from ..riscv import HALT_INSTR, Rv32i
from .base import (
    emit_and_step, fresh_cpu, predict_topk, prep_prompt, replay_committed,
)


def _one_step_lookahead_sim(model, ids, amask, committed: list[int],
                             pc: int, candidate_word: int, device: str,
                             seed_memcpy: bool) -> float | None:
    """Replay committed, apply candidate, step, return the next-cycle
    top-1 similarity. Returns None on replay/step failure."""
    trial = replay_committed(committed, seed_memcpy)
    if trial is None:
        return None
    _, err = emit_and_step(trial, pc, candidate_word)
    if err:
        return None
    _, sims = predict_topk(model, ids, amask, trial, device, topk=1)
    return sims[0]


def run(model, tok, prompt: str, device: str, max_cycles: int, *,
        seed_memcpy: bool = True,
        max_instr_tokens: int = MAX_INSTR_TOKENS,
        use_chat_template: bool = True,
        use_context_prefix: bool = False,
        margin_eps: float = 0.15,
        halt_margin_eps: float = 0.30,
        topk: int = 5,
        min_cycle: int = 2,
        max_interventions: int = 2,
        **_unused,
        ) -> tuple[Rv32i, list[int], bool, str, dict]:
    ids, amask = prep_prompt(tok, prompt, device,
                              max_instr_tokens=max_instr_tokens,
                              use_chat_template=use_chat_template,
                              use_context_prefix=use_context_prefix)
    cpu = fresh_cpu(seed_memcpy)
    committed: list[int] = []
    halted, err = False, ''
    overrides: list[dict] = []
    interventions_used = 0

    for cycle in range(max_cycles):
        pc = cpu.pc
        top_words, top_sims = predict_topk(model, ids, amask, cpu, device, topk=topk)
        top1_w = top_words[0]
        margin = top_sims[0] - top_sims[1] if len(top_sims) > 1 else 1.0
        chosen_w = top1_w

        eligible = cycle >= min_cycle and interventions_used < max_interventions

        if eligible and top1_w in (HALT_INSTR, 0) and margin < halt_margin_eps:
            chosen_w = top_words[1]
            interventions_used += 1
            overrides.append({'cycle': cycle, 'rule': 'halt_guard',
                              'from': f'0x{top1_w:08x}', 'to': f'0x{chosen_w:08x}',
                              'margin': round(margin, 4)})
        elif eligible and margin < margin_eps:
            best_sim, best_w = -1.0, top1_w
            for cand in top_words:
                if cand in (HALT_INSTR, 0):
                    continue
                s = _one_step_lookahead_sim(model, ids, amask, committed,
                                             pc, cand, device, seed_memcpy)
                if s is None:
                    continue
                if s > best_sim:
                    best_sim, best_w = s, cand
            if best_w != top1_w:
                chosen_w = best_w
                interventions_used += 1
                overrides.append({'cycle': cycle, 'rule': 'self_consistency',
                                  'from': f'0x{top1_w:08x}', 'to': f'0x{chosen_w:08x}',
                                  'margin': round(margin, 4),
                                  'next_sim': round(best_sim, 4)})

        committed.append(chosen_w)
        halted, err = emit_and_step(cpu, pc, chosen_w)
        if halted or err:
            break

    return cpu, committed, halted, err, {
        'decoder': 'tiebreak',
        'overrides': overrides,
        'interventions_used': interventions_used,
    }
