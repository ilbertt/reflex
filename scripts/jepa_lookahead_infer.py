"""Inference-time margin-aware lookahead: algorithmic self-consistency.

At each cycle, if the JEPA top-1 margin falls below a threshold, try each
of the top-k codebook candidates in a forked CPU (fresh replay of the
committed trace to that point + candidate + one step), measure the next
cycle's margin under each candidate, and commit the candidate whose
next-cycle margin is highest.

This is a pure-algorithmic "context" mechanism: it uses the model's own
*next* hidden-state confidence as a validator for the current choice.
No latent thread, no retraining, no extra parameters. Spirit-compatible
with the JEPA-era reflex design.

Runs the 4 tiers (short/medium/long/display) and the factorial-5
consistency probe, reports baseline vs. lookahead pass rates.
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


# Reuse the task tiers from the testbed
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


def _replay_to_cycle(prompt_ids, prompt_mask, committed: list[int],
                     seed_memcpy: bool) -> Rv32i | None:
    """Rebuild a fresh CPU by playing back `committed` instruction words.
    Returns the CPU positioned AFTER the last committed cycle (i.e. PC
    has advanced past all committed instrs). Returns None on failure."""
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
def run_with_lookahead(model, tok, prompt: str, device: str,
                        max_cycles: int, mit: int, use_ct: bool, use_cp: bool,
                        seed_memcpy: bool,
                        margin_threshold: float, topk: int,
                        max_lookaheads: int = 3):
    """Run with margin-aware top-k lookahead. `max_lookaheads` caps the
    number of low-margin cycles where we pay the per-candidate fork cost
    (to keep per-task wall time bounded)."""
    text = render_prompt(tok, prompt,
                          use_chat_template=use_ct, use_context_prefix=use_cp)
    e = tok(text, padding='max_length', truncation=True,
            max_length=mit, return_tensors='pt').to(device)
    ids, amask = e.input_ids, e.attention_mask

    cpu = _fresh_cpu(seed_memcpy)
    committed: list[int] = []
    halted, err = False, ''
    lookaheads_used = 0
    overrides_taken = []

    for cycle in range(max_cycles):
        pc = cpu.pc
        top_words, top_sims = _predict(model, ids, amask, cpu, device, topk)
        top1_w = top_words[0]
        margin = top_sims[0] - top_sims[1]
        chosen_w = top1_w

        if margin < margin_threshold and lookaheads_used < max_lookaheads:
            lookaheads_used += 1
            # Score each candidate by the next-cycle margin it induces
            best_next_margin = -1.0
            best_w = top1_w
            for w_cand in top_words:
                if w_cand == HALT_INSTR or w_cand == 0:
                    # Candidate that ends the program: skip unless it's top-1
                    # (we'd rather commit and halt than lookahead past zero)
                    continue
                # Replay to here, apply candidate, step, predict next
                trial_cpu = _replay_to_cycle(ids, amask, committed, seed_memcpy)
                if trial_cpu is None:
                    continue
                try:
                    trial_cpu.uc.mem_write(pc, int(w_cand).to_bytes(4, 'little'))
                    trial_cpu.uc.ctl_remove_cache(pc, pc + 4)
                    trial_cpu.step()
                except Exception:
                    continue
                _, sims2 = _predict(model, ids, amask, trial_cpu, device, 2)
                next_margin = sims2[0] - sims2[1]
                if next_margin > best_next_margin:
                    best_next_margin = next_margin
                    best_w = w_cand
            if best_w != top1_w:
                overrides_taken.append({
                    'cycle': cycle, 'from': f'0x{top1_w:08x}',
                    'to': f'0x{best_w:08x}',
                    'orig_margin': round(margin, 4),
                    'lookahead_next_margin': round(best_next_margin, 4),
                })
            chosen_w = best_w

        # Commit
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
        try:
            cpu.step()
        except Exception as e_:
            err = f'step@{pc:#x}: {e_}'; break

    return cpu, committed, halted, err, overrides_taken, lookaheads_used


@torch.no_grad()
def run_baseline(model, tok, prompt: str, device: str,
                 max_cycles: int, mit: int, use_ct: bool, use_cp: bool,
                 seed_memcpy: bool):
    """Plain top-1 greedy (matches eval.py's run_grounded)."""
    text = render_prompt(tok, prompt,
                          use_chat_template=use_ct, use_context_prefix=use_cp)
    e = tok(text, padding='max_length', truncation=True,
            max_length=mit, return_tensors='pt').to(device)
    ids, amask = e.input_ids, e.attention_mask
    cpu = _fresh_cpu(seed_memcpy)
    emitted, halted, err = [], False, ''
    for cycle in range(max_cycles):
        pc = cpu.pc
        top_words, _ = _predict(model, ids, amask, cpu, device, 2)
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


def _tier_evaluator(tier_name: str, tasks: list):
    def _eval_task(cpu, halted, err, row):
        if tier_name == 'display':
            tag, prompt, want, max_cycles = row
            got = _display_read(cpu, len(want))
            return {'passed': bool(halted and not err and got == want),
                    'want': want, 'got': got, 'max_cycles': max_cycles,
                    'tag': tag, 'prompt': prompt}
        tag, prompt, kind, expected, max_cycles = row
        return {'passed': bool(halted and not err and _check(cpu, kind, expected)),
                'expected': expected, 'kind': kind, 'max_cycles': max_cycles,
                'tag': tag, 'prompt': prompt}
    return _eval_task


def run_tier(name: str, tasks: list, *, model, tok, device, mit,
             use_ct, use_cp, margin_threshold, topk, max_lookaheads):
    ev = _tier_evaluator(name, tasks)
    base_pass, look_pass = 0, 0
    rows = []
    print(f'\n=== {name} ({len(tasks)} tasks) ===', flush=True)
    for row in tasks:
        tag = row[0]
        max_cycles = row[-1]
        seed_memcpy = (name != 'display')

        t0 = time.time()
        prompt = row[1]
        cpu_b, em_b, h_b, er_b = run_baseline(model, tok, prompt, device,
                                               max_cycles, mit, use_ct, use_cp,
                                               seed_memcpy)
        rb = ev(cpu_b, h_b, er_b, row)
        t_base = time.time() - t0

        t1 = time.time()
        cpu_l, em_l, h_l, er_l, overrides, _ = run_with_lookahead(
            model, tok, prompt, device, max_cycles, mit, use_ct, use_cp,
            seed_memcpy, margin_threshold=margin_threshold,
            topk=topk, max_lookaheads=max_lookaheads)
        rl = ev(cpu_l, h_l, er_l, row)
        t_look = time.time() - t1

        base_pass += int(rb['passed'])
        look_pass += int(rl['passed'])
        flip = '' if rb['passed'] == rl['passed'] else (' ⇑ RESCUE' if rl['passed'] else ' ⇓ REGRESSION')
        got_l = rl.get('got', rl.get('expected'))
        got_b = rb.get('got', rb.get('expected'))
        print(f'  {tag:<18s}  base={"✓" if rb["passed"] else "✗"} ({t_base:.1f}s)  '
              f'look={"✓" if rl["passed"] else "✗"} ({t_look:.1f}s)  '
              f'overrides={len(overrides)}{flip}', flush=True)
        for o in overrides:
            print(f'      cyc {o["cycle"]}  {o["from"]} → {o["to"]}  '
                  f'orig_margin={o["orig_margin"]}  next_margin={o["lookahead_next_margin"]}',
                  flush=True)
        rows.append({'tag': tag, 'prompt': prompt,
                     'baseline_passed': rb['passed'], 'baseline_elapsed_s': round(t_base, 2),
                     'lookahead_passed': rl['passed'], 'lookahead_elapsed_s': round(t_look, 2),
                     'overrides': overrides})
    print(f'  → {name}: base {base_pass}/{len(tasks)}  '
          f'lookahead {look_pass}/{len(tasks)}', flush=True)
    return {'tier': name, 'total': len(tasks),
            'baseline_passes': base_pass, 'lookahead_passes': look_pass,
            'tasks': rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='reflex.pt')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='/tmp/jepa_lookahead.json')
    ap.add_argument('--margin-threshold', type=float, default=0.50,
                    help='Trigger lookahead when top1-top2 margin is below this.')
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--max-lookaheads', type=int, default=3,
                    help='Per-task cap on low-margin cycles that pay the fork cost.')
    ap.add_argument('--tiers', nargs='+', default=['short','medium','long','display'])
    args = ap.parse_args()

    print(f'Loading {args.ckpt} on {args.device}…', flush=True)
    model, tok, cfg = load(args.ckpt, args.device)
    mit = cfg.get('max_instr_tokens', MAX_INSTR_TOKENS)
    use_ct = bool(cfg.get('chat_template', True))
    use_cp = bool(cfg.get('context_prefix', False))
    print(f'cfg: num_instrs={cfg.get("num_instrs")} '
          f'embed_dim={cfg.get("embed_dim")}\n'
          f'margin_threshold={args.margin_threshold}  '
          f'topk={args.topk}  max_lookaheads={args.max_lookaheads}\n',
          flush=True)

    tier_map = {'short': SHORT, 'medium': MEDIUM, 'long': LONG, 'display': DISPLAY_TIER}
    results = []
    for name in args.tiers:
        results.append(run_tier(name, tier_map[name],
                                model=model, tok=tok, device=args.device,
                                mit=mit, use_ct=use_ct, use_cp=use_cp,
                                margin_threshold=args.margin_threshold,
                                topk=args.topk, max_lookaheads=args.max_lookaheads))

    print('\n=== OVERALL ===', flush=True)
    total = sum(r['total'] for r in results)
    base = sum(r['baseline_passes'] for r in results)
    look = sum(r['lookahead_passes'] for r in results)
    print(f'  baseline:  {base}/{total}  ({100*base/max(total,1):.0f}%)', flush=True)
    print(f'  lookahead: {look}/{total}  ({100*look/max(total,1):.0f}%)', flush=True)
    print(f'  delta:     {look-base:+d}', flush=True)

    Path(args.out).write_text(json.dumps({
        'ckpt': args.ckpt, 'device': args.device,
        'margin_threshold': args.margin_threshold, 'topk': args.topk,
        'max_lookaheads': args.max_lookaheads,
        'summary': {'baseline': base, 'lookahead': look, 'total': total},
        'tiers': results,
    }, indent=2))
    print(f'\nwrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
