"""Margin-gated tiebreaker: minimal spirit-preserving intervention.

Two tiny rules, both triggered only at low-margin cycles:

  R1 (early-halt guard). If top-1 is HALT (0x0000006F) or 0x00000000 and
     the margin to top-2 is below ``halt_margin_eps``, prefer top-2.
     Rationale: a low-confidence halt commits the program to stopping on
     weak evidence; weight is better spent continuing the program.

  R2 (one-step self-consistency, no execution forking). If margin <
     ``margin_eps``, run the model forward ONCE MORE per top-k candidate
     on a trial CPU that has been stepped under that candidate, and
     pick the candidate whose induced next-cycle top-1 sim is highest.
     Unlike the full lookahead decoder, this uses only ONE extra forward
     per candidate, no multi-step unrolling.

The loop structure is preserved: one 32-bit word per cycle written to
the CPU. No hidden-state thread, no tree search across cycles, no
retraining. Measures whether minimal tiebreakers recover display tasks.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import torch

from reflex.demo import load
from reflex.model import (
    MAX_INSTR_TOKENS, code_region_halt_fill, extract_state, render_prompt,
)
from reflex.programs import DISPLAY_BASE, DST_OFFSET, SRC_OFFSET
from reflex.riscv import DATA_BASE, HALT_INSTR, PROGRAM_START, Rv32i

from scripts.jepa_testbed import SHORT, MEDIUM, LONG, DISPLAY_TIER, _check, _display_read


def _fresh_cpu(seed_memcpy: bool) -> Rv32i:
    cpu = Rv32i()
    cpu.uc.mem_write(PROGRAM_START, code_region_halt_fill())
    if seed_memcpy:
        seed = b''.join(int(i).to_bytes(4, 'little') for i in range(1, 9))
        cpu.uc.mem_write(DATA_BASE + SRC_OFFSET, seed)
    return cpu


@torch.no_grad()
def _predict(model, ids, amask, cpu, device: str, topk: int):
    state = extract_state(cpu)
    state_t = torch.from_numpy(state.astype('int64')).unsqueeze(0).to(device)
    pred = model(ids, amask, state_t)
    sims = model.table_similarity(pred).squeeze(0)
    top = torch.topk(sims, topk)
    top_idx = top.indices.tolist()
    top_sims = [float(s) for s in top.values.tolist()]
    top_words = [int(model.instr_words[i].item()) & 0xFFFFFFFF
                 for i in top_idx]
    return top_words, top_sims


def _replay(prompt_ids, prompt_mask, committed: list[int],
            seed_memcpy: bool) -> Rv32i | None:
    cpu = _fresh_cpu(seed_memcpy)
    for w in committed:
        pc = cpu.pc
        try:
            cpu.uc.mem_write(pc, int(w).to_bytes(4, 'little'))
            cpu.uc.ctl_remove_cache(pc, pc + 4)
        except Exception:
            return None
        if w == HALT_INSTR or w == 0:
            return cpu
        try:
            cpu.step()
        except Exception:
            return None
    return cpu


@torch.no_grad()
def run_tiebreak(model, tok, prompt: str, device: str,
                 max_cycles: int, mit: int, use_ct: bool, use_cp: bool,
                 seed_memcpy: bool, *,
                 margin_eps: float, halt_margin_eps: float,
                 topk: int, max_interventions: int, min_cycle: int = 0):
    text = render_prompt(tok, prompt,
                          use_chat_template=use_ct, use_context_prefix=use_cp)
    e = tok(text, padding='max_length', truncation=True,
            max_length=mit, return_tensors='pt').to(device)
    ids, amask = e.input_ids, e.attention_mask

    cpu = _fresh_cpu(seed_memcpy)
    committed: list[int] = []
    halted, err = False, ''
    interventions_used = 0
    overrides = []

    for cycle in range(max_cycles):
        pc = cpu.pc
        top_words, top_sims = _predict(model, ids, amask, cpu, device, topk)
        top1_w = top_words[0]
        margin = top_sims[0] - top_sims[1]
        chosen_w = top1_w
        reason = None

        eligible = cycle >= min_cycle and interventions_used < max_interventions

        # R1: early-halt guard
        if eligible and top1_w in (HALT_INSTR, 0) and margin < halt_margin_eps:
            chosen_w = top_words[1]
            reason = 'halt_guard'
            interventions_used += 1
            overrides.append({'cycle': cycle, 'rule': reason,
                              'from': f'0x{top1_w:08x}',
                              'to': f'0x{chosen_w:08x}',
                              'margin': round(margin, 4)})

        # R2: one-step self-consistency on low margin (skipped if R1 fired)
        elif eligible and margin < margin_eps:
            best_next_sim = -1.0
            best_w = top1_w
            for w_cand in top_words:
                if w_cand in (HALT_INSTR, 0):
                    continue
                trial_cpu = _replay(ids, amask, committed, seed_memcpy)
                if trial_cpu is None:
                    continue
                try:
                    trial_cpu.uc.mem_write(pc, int(w_cand).to_bytes(4, 'little'))
                    trial_cpu.uc.ctl_remove_cache(pc, pc + 4)
                    trial_cpu.step()
                except Exception:
                    continue
                _, sims2 = _predict(model, ids, amask, trial_cpu, device, 1)
                s_next = sims2[0]
                if s_next > best_next_sim:
                    best_next_sim = s_next
                    best_w = w_cand
            if best_w != top1_w:
                chosen_w = best_w
                reason = 'self_consistency'
                interventions_used += 1
                overrides.append({'cycle': cycle, 'rule': reason,
                                  'from': f'0x{top1_w:08x}',
                                  'to': f'0x{chosen_w:08x}',
                                  'margin': round(margin, 4),
                                  'next_sim': round(best_next_sim, 4)})

        committed.append(chosen_w)
        try:
            cpu.uc.mem_write(pc, int(chosen_w).to_bytes(4, 'little'))
            cpu.uc.ctl_remove_cache(pc, pc + 4)
        except Exception as e_:
            err = f'write@{pc:#x}: {e_}'; break
        if chosen_w == HALT_INSTR:
            halted = True; break
        if chosen_w == 0:
            err = f'zero@{pc:#x}'; break
        try: cpu.step()
        except Exception as e_: err = f'step@{pc:#x}: {e_}'; break

    return cpu, committed, halted, err, overrides


@torch.no_grad()
def run_baseline(model, tok, prompt, device, max_cycles, mit, use_ct, use_cp,
                 seed_memcpy):
    text = render_prompt(tok, prompt,
                          use_chat_template=use_ct, use_context_prefix=use_cp)
    e = tok(text, padding='max_length', truncation=True,
            max_length=mit, return_tensors='pt').to(device)
    ids, amask = e.input_ids, e.attention_mask
    cpu = _fresh_cpu(seed_memcpy)
    emitted, halted, err = [], False, ''
    for cycle in range(max_cycles):
        pc = cpu.pc
        top_words, _ = _predict(model, ids, amask, cpu, device, 1)
        w = top_words[0]
        emitted.append(w)
        try:
            cpu.uc.mem_write(pc, int(w).to_bytes(4, 'little'))
            cpu.uc.ctl_remove_cache(pc, pc + 4)
        except Exception as e_:
            err = f'write@{pc:#x}: {e_}'; break
        if w == HALT_INSTR: halted = True; break
        if w == 0: err = f'zero@{pc:#x}'; break
        try: cpu.step()
        except Exception as e_: err = f'step@{pc:#x}: {e_}'; break
    return cpu, emitted, halted, err


def _pass(name, cpu, halted, err, row):
    if name == 'display':
        _, _, want, _ = row
        return bool(halted and not err and _display_read(cpu, len(want)) == want)
    _, _, kind, expected, _ = row
    return bool(halted and not err and _check(cpu, kind, expected))


def _got(name, cpu, row):
    if name == 'display':
        _, _, want, _ = row
        return _display_read(cpu, len(want))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='reflex.pt')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='/tmp/jepa_tiebreak.json')
    ap.add_argument('--margin-eps', type=float, default=0.15,
                    help='R2 (self-consistency) triggers when margin below this.')
    ap.add_argument('--halt-margin-eps', type=float, default=0.25,
                    help='R1 (halt guard) triggers when top1∈{halt,0} and margin below this.')
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--max-interventions', type=int, default=2,
                    help='Per-task cap on tiebreaker uses.')
    ap.add_argument('--min-cycle', type=int, default=0,
                    help='Skip tiebreakers on cycles before this; protects '
                    'early register-allocation choices.')
    ap.add_argument('--tiers', nargs='+', default=['short','medium','long','display'])
    args = ap.parse_args()

    print(f'Loading {args.ckpt} on {args.device}…', flush=True)
    model, tok, cfg = load(args.ckpt, args.device)
    mit = cfg.get('max_instr_tokens', MAX_INSTR_TOKENS)
    use_ct = bool(cfg.get('chat_template', True))
    use_cp = bool(cfg.get('context_prefix', False))
    print(f'eps={args.margin_eps}  halt_eps={args.halt_margin_eps}  '
          f'topk={args.topk}  max_interventions={args.max_interventions}\n',
          flush=True)

    tier_map = {'short': SHORT, 'medium': MEDIUM, 'long': LONG, 'display': DISPLAY_TIER}
    results = []
    for name in args.tiers:
        tasks = tier_map[name]
        print(f'\n=== {name} ({len(tasks)} tasks) ===', flush=True)
        rows = []
        base_pass, tb_pass = 0, 0
        for row in tasks:
            tag = row[0]
            prompt = row[1]
            max_cycles = row[-1]
            seed_memcpy = (name != 'display')

            t0 = time.time()
            cpu_b, em_b, h_b, er_b = run_baseline(
                model, tok, prompt, args.device, max_cycles, mit, use_ct, use_cp,
                seed_memcpy)
            pb = _pass(name, cpu_b, h_b, er_b, row)
            tb = time.time() - t0

            t1 = time.time()
            cpu_t, em_t, h_t, er_t, overrides = run_tiebreak(
                model, tok, prompt, args.device, max_cycles, mit, use_ct, use_cp,
                seed_memcpy,
                margin_eps=args.margin_eps,
                halt_margin_eps=args.halt_margin_eps,
                topk=args.topk,
                max_interventions=args.max_interventions,
                min_cycle=args.min_cycle)
            pt = _pass(name, cpu_t, h_t, er_t, row)
            tt = time.time() - t1

            base_pass += int(pb)
            tb_pass += int(pt)
            flip = ''
            if pb != pt:
                flip = ' ⇑ RESCUE' if pt else ' ⇓ REGRESSION'
            got_b = _got(name, cpu_b, row)
            got_t = _got(name, cpu_t, row)
            got_str = f'b={got_b!r} t={got_t!r}' if got_b is not None else ''
            print(f'  {tag:<18s}  base={"✓" if pb else "✗"}({tb:.1f}s)  '
                  f'tb={"✓" if pt else "✗"}({tt:.1f}s)  '
                  f'n={len(overrides)}{flip}  {got_str}',
                  flush=True)
            for o in overrides:
                print(f'      cyc {o["cycle"]}  rule={o["rule"]}  '
                      f'{o["from"]} → {o["to"]}  margin={o["margin"]}',
                      flush=True)
            rows.append({'tag': tag, 'baseline': pb, 'tiebreak': pt,
                         'overrides': overrides,
                         'baseline_elapsed_s': round(tb, 2),
                         'tiebreak_elapsed_s': round(tt, 2)})
        print(f'  → {name}: base {base_pass}/{len(tasks)}  '
              f'tiebreak {tb_pass}/{len(tasks)}', flush=True)
        results.append({'tier': name, 'baseline_passes': base_pass,
                        'tiebreak_passes': tb_pass,
                        'total': len(tasks), 'tasks': rows})

    print('\n=== OVERALL ===', flush=True)
    total = sum(r['total'] for r in results)
    base = sum(r['baseline_passes'] for r in results)
    tb = sum(r['tiebreak_passes'] for r in results)
    print(f'  baseline:  {base}/{total}  ({100*base/max(total,1):.0f}%)', flush=True)
    print(f'  tiebreak:  {tb}/{total}  ({100*tb/max(total,1):.0f}%)', flush=True)
    print(f'  delta:     {tb-base:+d}', flush=True)

    Path(args.out).write_text(json.dumps({
        'ckpt': args.ckpt, 'margin_eps': args.margin_eps,
        'halt_margin_eps': args.halt_margin_eps,
        'topk': args.topk, 'max_interventions': args.max_interventions,
        'summary': {'baseline': base, 'tiebreak': tb, 'total': total},
        'tiers': results,
    }, indent=2))
    print(f'\nwrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
