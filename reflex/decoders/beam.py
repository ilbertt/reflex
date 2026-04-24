"""Beam-search decoder.

Maintains ``beam_width`` parallel trajectories through the cycle tree.
Each beam is ``(committed_words, cumulative_log_margin, cpu)``. At each
cycle, every beam proposes its top-``branching`` candidates; the full
set of ``beam_width × branching`` extensions is scored and pruned back
to ``beam_width``. When a beam halts, it is retained as a completed
candidate. When every surviving beam halts (or ``max_cycles`` is
exhausted), the completed beam with the highest cumulative score is
returned.

Spirit deviation is the most pronounced in this package. The canonical
decoder emits one opcode per cycle to one CPU. Beam search runs
``beam_width`` CPUs in parallel and commits only the winning
trajectory's opcodes at the end. The loop is no longer per-cycle; it's
per-tree-level. See ``CONFESSION.md``.

This is the most expensive decoder and is intended for experimentation
and ceiling analysis, not deployment. A ``beam_width=1`` degenerates to
the canonical argmax decoder (modulo reconstruction overhead); useful
as a sanity check.
"""
import math
from collections import Counter

from ..model import MAX_INSTR_TOKENS
from ..riscv import HALT_INSTR, Rv32i
from .base import (
    emit_and_step, fresh_cpu, predict_topk, prep_prompt, replay_committed,
)


def _committed_terminal(committed: list[int]) -> tuple[bool, str]:
    """Check whether a just-appended committed word terminated the beam.

    ``replay_committed`` is memoryless about the terminal flag — it
    returns a CPU, not a ``(cpu, halted)`` pair. For beam search we need
    to know whether the last committed word was HALT (clean exit) or
    0x00000000 (faulted emission), since beams in those states should
    move to ``completed`` instead of being extended again. Inspecting
    the committed list is the cheapest way to recover that flag."""
    if not committed:
        return False, ''
    last = committed[-1]
    if last == HALT_INSTR:
        return True, ''
    if last == 0:
        return False, f'zero_emission'
    return False, ''


def run(model, tok, prompt: str, device: str, max_cycles: int, *,
        seed_memcpy: bool = True,
        max_instr_tokens: int = MAX_INSTR_TOKENS,
        use_chat_template: bool = True,
        use_context_prefix: bool = False,
        beam_width: int = 4,
        branching: int = 3,
        **_unused,
        ) -> tuple[Rv32i, list[int], bool, str, dict]:
    ids, amask = prep_prompt(tok, prompt, device,
                              max_instr_tokens=max_instr_tokens,
                              use_chat_template=use_chat_template,
                              use_context_prefix=use_context_prefix)

    # Each live beam: {'committed': [words], 'score': float, 'cpu': Rv32i}
    root_cpu = fresh_cpu(seed_memcpy)
    live = [{'committed': [], 'score': 0.0, 'cpu': root_cpu, 'halted': False, 'err': ''}]
    completed: list[dict] = []

    for cycle in range(max_cycles):
        # Extend every live beam by its top-``branching`` candidates.
        extensions: list[dict] = []
        for beam in live:
            if beam['halted'] or beam['err']:
                completed.append(beam)
                continue
            pc = beam['cpu'].pc
            top_words, top_sims = predict_topk(
                model, ids, amask, beam['cpu'], device, topk=branching)
            for cand_w, cand_s in zip(top_words, top_sims):
                # Score each extension by cumulative log-sim; clamp
                # small/negative sims to avoid -inf.
                step_score = math.log(max(cand_s, 1e-4))
                extensions.append({
                    'parent': beam,
                    'word': cand_w,
                    'step_score': step_score,
                    'new_score': beam['score'] + step_score,
                })

        if not extensions:
            break

        # Prune to ``beam_width`` best extensions.
        extensions.sort(key=lambda e: e['new_score'], reverse=True)
        extensions = extensions[:beam_width]

        # Materialize each surviving extension by replaying its parent's
        # committed trace + the new word. Replay is required in the
        # general case because Unicorn CPUs don't cheaply deep-copy; one
        # CPU per live beam. The fast path — single child of single
        # parent — can step the parent's CPU in place and skip replay.
        new_live: list[dict] = []
        # Map parent id → count of children for the fast-path check.
        child_count = Counter(id(ext['parent']) for ext in extensions)
        for ext in extensions:
            parent = ext['parent']
            committed_next = parent['committed'] + [ext['word']]
            single_child = child_count[id(parent)] == 1
            if single_child:
                cpu_next = parent['cpu']
                pc = cpu_next.pc
                halted, err = emit_and_step(cpu_next, pc, ext['word'])
            else:
                cpu_next = replay_committed(committed_next, seed_memcpy)
                if cpu_next is None:
                    halted, err = False, 'beam_replay_failed'
                else:
                    halted, err = _committed_terminal(committed_next)
            new_live.append({
                'committed': committed_next,
                'score': ext['new_score'],
                'cpu': cpu_next,
                'halted': halted,
                'err': err,
            })

        live = new_live
        # Harvest any halted/errored beams into completed.
        for beam in live[:]:
            if beam['halted'] or beam['err']:
                completed.append(beam)
        live = [b for b in live if not (b['halted'] or b['err'])]
        if not live:
            break

    # Any live beams at max_cycles join the completed pool.
    completed.extend(live)
    # Prefer cleanly halted beams; among those, highest cumulative score.
    def _key(b):
        return (1 if (b['halted'] and not b['err']) else 0,
                0 if b['err'] else 1,
                b['score'])
    completed.sort(key=_key, reverse=True)
    winner = completed[0]
    return (winner['cpu'], winner['committed'],
            winner['halted'], winner['err'] or '',
            {'decoder': 'beam',
             'beam_width': beam_width, 'branching': branching,
             'n_completed': len(completed),
             'winner_score': round(winner['score'], 4)})
