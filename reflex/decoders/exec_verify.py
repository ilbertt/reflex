"""Execution-verification decoder.

At low-margin cycles, fork each of the top-``k`` codebook candidates,
simulate ``lookahead_depth`` cycles of the resulting program in a
cloned CPU, and pick the candidate whose forward trajectory is the most
"well-formed" by a small set of execution signals:

  * cleanly halted within the lookahead window (preferred over not
    halting, tiebroken by shorter length)
  * no invalid memory write, no Unicorn step fault, no emitted zero
  * highest cumulative top-1 similarity across the simulated cycles

This is the heaviest decoder in the suite: at an intervention cycle it
performs ``k × lookahead_depth`` extra model forwards and steps. Cost
is bounded by gating interventions with ``margin_eps`` and
``max_interventions`` — on the 23-task suite most cycles have margin
≫ 0.15 and no intervention fires.

Spirit deviation from the canonical decoder is explicit and
substantial — the "thinnest possible bridge" becomes a depth-``d`` tree
of cloned CPUs at uncertain cycles, and the committed instruction is
chosen by a scoring function rather than direct argmax. See
``CONFESSION.md``.

Decoder metadata records each intervention with the trajectory scores
considered, so downstream analysis can inspect what the oracle saw.
"""
from ..model import MAX_INSTR_TOKENS
from ..riscv import HALT_INSTR, Rv32i
from .base import (
    emit_and_step, fresh_cpu, predict_topk, prep_prompt, replay_committed,
)


def _simulate_forward(model, ids, amask, trial_cpu: Rv32i,
                       device: str, depth: int) -> dict:
    """Run ``trial_cpu`` forward up to ``depth`` additional cycles by
    argmax snap. Returns a dict describing the trajectory.

    Tracks both cumulative top-1 absolute sim and cumulative margin
    (top1 - top2). The 2026 lookahead-decoding literature (PG-TD,
    NEST, Lookahead Decoding) consistently finds margin a stronger
    correctness signal than absolute similarity — a high-sim candidate
    whose top-2 is nearly tied is an ambiguous step; a modest-sim
    candidate that is unambiguous is a clean step."""
    cum_sim = 0.0
    cum_margin = 0.0
    halted, err = False, ''
    sim_steps = 0
    for _ in range(depth):
        pc = trial_cpu.pc
        top_words, top_sims = predict_topk(
            model, ids, amask, trial_cpu, device, topk=2)
        cum_sim += top_sims[0]
        cum_margin += (top_sims[0] - top_sims[1]) if len(top_sims) > 1 else top_sims[0]
        halted, err = emit_and_step(trial_cpu, pc, top_words[0])
        sim_steps += 1
        if halted or err:
            break
    return {
        'halted': halted,
        'err': err,
        'steps': sim_steps,
        'cum_sim': round(cum_sim, 4),
        'cum_margin': round(cum_margin, 4),
    }


def _score_trajectory(traj: dict, scoring: str = 'margin') -> tuple:
    """Sort key for trajectories (larger is better).

    Primary: no execution error (invalid write / step fault / emitted 0).
    Secondary: halted cleanly within the horizon.
    Tertiary: confidence signal — cumulative margin by default
             (stronger correctness signal per the 2026 literature), or
             cumulative top-1 sim if ``scoring='sim'`` for comparison.
    Quaternary: shorter halted trajectories (fewer ops to the answer).
    """
    no_err   = 0 if traj['err'] else 1
    halted   = 1 if traj['halted'] and not traj['err'] else 0
    confidence = traj['cum_margin'] if scoring == 'margin' else traj['cum_sim']
    short    = -traj['steps'] if traj['halted'] and not traj['err'] else 0
    return (no_err, halted, confidence, short)


def run(model, tok, prompt: str, device: str, max_cycles: int, *,
        seed_memcpy: bool = True,
        max_instr_tokens: int = MAX_INSTR_TOKENS,
        use_chat_template: bool = True,
        use_context_prefix: bool = False,
        margin_eps: float = 0.15,
        topk: int = 5,
        min_cycle: int = 2,
        max_interventions: int = 3,
        lookahead_depth: int = 6,
        scoring: str = 'margin',
        **_unused,
        ) -> tuple[Rv32i, list[int], bool, str, dict]:
    ids, amask = prep_prompt(tok, prompt, device,
                              max_instr_tokens=max_instr_tokens,
                              use_chat_template=use_chat_template,
                              use_context_prefix=use_context_prefix)
    cpu = fresh_cpu(seed_memcpy)
    committed: list[int] = []
    halted, err = False, ''
    interventions: list[dict] = []
    interventions_used = 0

    for cycle in range(max_cycles):
        pc = cpu.pc
        top_words, top_sims = predict_topk(model, ids, amask, cpu, device, topk=topk)
        top1_w = top_words[0]
        margin = top_sims[0] - top_sims[1] if len(top_sims) > 1 else 1.0
        chosen_w = top1_w

        eligible = (cycle >= min_cycle
                    and interventions_used < max_interventions
                    and margin < margin_eps)

        if eligible:
            trajectories = []
            for cand in top_words:
                trial = replay_committed(committed, seed_memcpy)
                if trial is None:
                    trajectories.append({'cand': cand, 'traj': {
                        'halted': False, 'err': 'replay_failed',
                        'steps': 0, 'cum_sim': 0.0, 'cum_margin': 0.0}})
                    continue
                rep_halted, rep_err = emit_and_step(trial, pc, cand)
                if rep_err:
                    trajectories.append({'cand': cand, 'traj': {
                        'halted': False, 'err': rep_err,
                        'steps': 0, 'cum_sim': 0.0, 'cum_margin': 0.0}})
                    continue
                if rep_halted:
                    # HALT candidate: a legitimate short trajectory of
                    # length 1. Do not extend into the lookahead — that
                    # would overwrite the halt and mis-attribute it.
                    trajectories.append({'cand': cand, 'traj': {
                        'halted': True, 'err': '',
                        'steps': 1, 'cum_sim': 0.0, 'cum_margin': 0.0}})
                    continue
                traj = _simulate_forward(
                    model, ids, amask, trial, device, lookahead_depth)
                trajectories.append({'cand': cand, 'traj': traj})

            scored = sorted(trajectories,
                             key=lambda t: _score_trajectory(t['traj'], scoring=scoring),
                             reverse=True)
            best = scored[0]
            if best['cand'] != top1_w:
                chosen_w = best['cand']
                interventions_used += 1
                interventions.append({
                    'cycle': cycle,
                    'from': f'0x{top1_w:08x}', 'to': f'0x{chosen_w:08x}',
                    'margin': round(margin, 4),
                    'trajectories': [{
                        'cand': f"0x{t['cand']:08x}",
                        'halted': t['traj']['halted'],
                        'err': t['traj']['err'],
                        'steps': t['traj']['steps'],
                        'cum_sim': t['traj']['cum_sim'],
                        'cum_margin': t['traj'].get('cum_margin', 0.0),
                    } for t in trajectories],
                })

        committed.append(chosen_w)
        halted, err = emit_and_step(cpu, pc, chosen_w)
        if halted or err:
            break

    return cpu, committed, halted, err, {
        'decoder': 'exec_verify',
        'interventions': interventions,
        'interventions_used': interventions_used,
    }
