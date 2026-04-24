"""JEPA testbed — per-cycle diagnostics for the codebook-head checkpoint.

This is the Reflex test suite adapted to the JEPA meta (PR #7). The old
latent-recurrence testbed is obsolete: the model no longer threads a
hidden state, and codebook snap already absorbs the single-bit noise
that latent recurrence was meant to paper over.

What we surface now:

  * per-cycle TOP-5 cosine similarities against the 691-row codebook
  * margin = sim(top1) - sim(top2)  — confidence proxy
  * opcode coherence — fraction of top-5 that share top-1's opcode
  * pass/fail per task (final memory state correctness)
  * aggregate: mean margin, margin quantiles, top-5 opcode purity,
    emitted-opcode histogram, failures grouped by tier

Usage:
  python scripts/jepa_testbed.py --ckpt reflex.pt --device cuda \
      --out /tmp/jepa_testbed.json
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import torch

from reflex.demo import load
from reflex.eval import IN_DIST, OOD, DISPLAY, NOVEL
from reflex.model import (
    MAX_INSTR_TOKENS, code_region_halt_fill, extract_state, render_prompt,
)
from reflex.programs import DISPLAY_BASE, DST_OFFSET, SRC_OFFSET
from reflex.riscv import DATA_BASE, HALT_INSTR, PROGRAM_START, Rv32i


# ── task tiers & cycle budgets ────────────────────────────────────────
# Group eval tasks by expected depth. Display tier is narrow on purpose —
# it's where JEPA's byte-constant neighbour problem shows up worst.
SHORT = [
    ('add 7+8',     'add 7 and 8 and store the result',         'mem', 15,     40),
    ('max(7,12)',   'find the max of 7 and 12 and store it',    'mem', 12,     40),
    ('abs(-5)',     'compute the absolute value of -5 and store it', 'mem', 5, 40),
    ('min(7,3,9)',  'find the minimum of 7, 3, and 9 and store it',  'mem', 3, 40),
    ('subtract 25-10','subtract 10 from 25 and store the result', 'mem', 15,  40),
    ('double 25',   'call a function that doubles 25 and store the result','mem',50,40),
]

MEDIUM = [
    ('multiply 7*8','multiply 7 and 8 and store the result',    'mem', 56,  100),
    ('sum 1..10',   'compute 1 + 2 + ... + 10 and store the sum','mem', 55, 100),
    ('countdown 5', 'count down from 5 to 1 and store each value','seq',[5,4,3,2,1],100),
    ('memcpy 4',    'copy 4 words from source to destination',   'dst',[1,2,3,4],100),
    ('fib 6',       'store the first 6 Fibonacci numbers',       'seq',[0,1,1,2,3,5],150),
    ('factorial 5', 'compute 5 factorial and store it',          'mem', 120, 150),
]

LONG = [
    ('sum 1..20',      'compute 1+2+...+20 and store the sum',   'mem', 210, 250),
    ('factorial 7',    'compute 7 factorial and store it',       'mem', 5040,300),
    ('fib 10',         'store the first 10 fibonacci numbers',   'seq',[0,1,1,2,3,5,8,13],200),
    ('countdown 20',   'count down from 20 to 1 and store each value','seq',[20,19,18,17,16,15,14,13],200),
    ('power 2^5',      'compute 2 to the power of 5 and store the result','mem',32,150),
    ('popcount 255',   'count the number of 1-bits in 255 and store it','mem',8,400),
]

DISPLAY_TIER = [
    ('say hi',    'say hi',     'hi',   40),
    ('say wow',   'say wow',    'wow',  40),
    ('display OK','display OK', 'OK',   40),
    ('show 42',   'show 42',    '42',   40),
    ('print hello','print hello','hello',50),
]

TIERS = [
    ('short',   SHORT),
    ('medium',  MEDIUM),
    ('long',    LONG),
    ('display', DISPLAY_TIER),
]


# ── output checking ───────────────────────────────────────────────────
def _check(cpu, kind, expected) -> bool:
    if kind == 'mem':
        return cpu.mem_word(DATA_BASE) == expected
    if kind == 'seq':
        return [cpu.mem_word(DATA_BASE + 4*i) for i in range(len(expected))] == expected
    if kind == 'dst':
        return [cpu.mem_word(DATA_BASE + DST_OFFSET + 4*i) for i in range(len(expected))] == expected
    if kind == 'disp':  # display tier uses raw string compare elsewhere
        return False
    return False


def _display_read(cpu, n: int) -> str:
    out = []
    for i in range(n):
        b = cpu.mem_word(DISPLAY_BASE + 4*i) & 0xFF
        out.append(chr(b) if 0x20 <= b < 0x7F else '·')
    return ''.join(out)


def _opcode(word: int) -> int:
    return word & 0x7F


OPCODE_NAMES = {
    0x03: 'LOAD', 0x13: 'OP-IMM', 0x17: 'AUIPC', 0x23: 'STORE',
    0x33: 'OP', 0x37: 'LUI', 0x63: 'BRANCH', 0x67: 'JALR', 0x6F: 'JAL',
    0x73: 'SYSTEM',
}


# ── traced run ────────────────────────────────────────────────────────
@torch.no_grad()
def run_traced(model, tok, instruction: str, device: str,
               max_cycles: int, max_instr_tokens: int,
               use_chat_template: bool, use_context_prefix: bool,
               seed_memcpy: bool = True, topk: int = 5):
    """Like run_grounded, but returns per-cycle diagnostics.

    Returns ``(cpu, emitted, halted, err, trace)`` where ``trace`` is a
    list of dicts: ``{pc, word, top_idx, top_sims, top_words, opcode,
    top_opcodes, margin}``.
    """
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
    trace: list[dict] = []
    halted, err = False, ''
    for cycle in range(max_cycles):
        pc = cpu.pc
        state = extract_state(cpu)
        state_t = torch.from_numpy(state.astype('int64')).unsqueeze(0).to(device)
        pred = model(ids, amask, state_t)                    # [1, embed_dim]
        sims = model.table_similarity(pred).squeeze(0)       # [num_instrs]
        top = torch.topk(sims, topk)
        top_idx = top.indices.tolist()
        top_sims = [round(float(s), 4) for s in top.values.tolist()]
        top_words = [int(model.instr_words[i].item()) & 0xFFFFFFFF for i in top_idx]
        instr_w = top_words[0]
        emitted.append(instr_w)
        trace.append({
            'cycle': cycle, 'pc': pc, 'word': instr_w,
            'top_words': [f'0x{w:08x}' for w in top_words],
            'top_sims': top_sims,
            'margin': round(top_sims[0] - top_sims[1], 4),
            'opcode': _opcode(instr_w),
            'top_opcodes': [_opcode(w) for w in top_words],
        })
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

    return cpu, emitted, halted, err, trace


# ── aggregate helpers ─────────────────────────────────────────────────
def _margin_stats(trace: list[dict]) -> dict:
    if not trace:
        return {'mean': 0.0, 'p10': 0.0, 'p50': 0.0, 'p90': 0.0, 'min': 0.0}
    ms = sorted(t['margin'] for t in trace)
    n = len(ms)
    def q(p): return ms[min(int(p*n), n-1)]
    return {
        'mean': round(sum(ms) / n, 4),
        'p10': q(0.10), 'p50': q(0.50), 'p90': q(0.90),
        'min': ms[0],
    }


def _top5_same_opcode_purity(trace: list[dict]) -> float:
    """Fraction of cycles where all top-5 share top-1's opcode."""
    if not trace:
        return 0.0
    hit = sum(1 for t in trace
              if all(o == t['opcode'] for o in t['top_opcodes']))
    return round(hit / len(trace), 4)


def _opcode_histogram(trace: list[dict]) -> dict:
    c = Counter(t['opcode'] for t in trace)
    return {OPCODE_NAMES.get(op, f'0x{op:02x}'): n for op, n in c.most_common()}


# ── runners per tier ──────────────────────────────────────────────────
def run_code_tier(model, tok, device: str, tier_name: str, tasks: list,
                  mit: int, use_ct: bool, use_cp: bool) -> dict:
    results = []
    passes = 0
    t_start = time.time()
    for row in tasks:
        tag, prompt, kind, expected, max_cycles = row
        t0 = time.time()
        try:
            cpu, emitted, halted, err, trace = run_traced(
                model, tok, prompt, device, max_cycles, mit, use_ct, use_cp)
            ok = _check(cpu, kind, expected)
        except Exception as e:
            cpu, emitted, halted, err, trace = None, [], False, f'EXC: {e}', []
            ok = False
        dt = round(time.time() - t0, 2)
        passed = bool(ok and halted and not err)
        if passed:
            passes += 1
        margin = _margin_stats(trace)
        purity = _top5_same_opcode_purity(trace)
        print(f'  {"✓" if passed else "✗"} {tag:<18s}  '
              f'ops={len(emitted):4d}  halt={str(halted):5s}  '
              f'margin={margin["mean"]:.3f}  purity={purity:.2f}  '
              f'{dt:.1f}s  {err[:40]}',
              flush=True)
        results.append({
            'tag': tag, 'prompt': prompt, 'expected': expected, 'kind': kind,
            'passed': passed, 'halted': halted, 'err': err,
            'ops': len(emitted),
            'max_cycles': max_cycles,
            'margin_stats': margin,
            'top5_same_opcode_purity': purity,
            'opcode_histogram': _opcode_histogram(trace),
            'elapsed_s': dt,
            'trace': trace,
        })
    wall = round(time.time() - t_start, 2)
    print(f'  → {tier_name}: {passes}/{len(tasks)}  wall={wall}s\n',
          flush=True)
    return {
        'tier': tier_name,
        'passes': passes, 'total': len(tasks),
        'pass_rate': round(passes / max(len(tasks), 1), 3),
        'wall_s': wall,
        'tasks': results,
    }


def run_display_tier(model, tok, device: str, tasks: list, mit: int,
                     use_ct: bool, use_cp: bool) -> dict:
    results = []
    passes = 0
    t_start = time.time()
    for tag, prompt, want, max_cycles in tasks:
        t0 = time.time()
        try:
            cpu, emitted, halted, err, trace = run_traced(
                model, tok, prompt, device, max_cycles, mit, use_ct, use_cp,
                seed_memcpy=False)
            got = _display_read(cpu, len(want))
            ok = (got == want)
        except Exception as e:
            cpu, emitted, halted, err, trace = None, [], False, f'EXC: {e}', []
            got, ok = '', False
        dt = round(time.time() - t0, 2)
        passed = bool(ok and halted and not err)
        if passed: passes += 1
        margin = _margin_stats(trace)
        purity = _top5_same_opcode_purity(trace)
        print(f'  {"✓" if passed else "✗"} {tag:<14s}  got={got!r:<10s}  '
              f'want={want!r:<8s}  ops={len(emitted):3d}  '
              f'margin={margin["mean"]:.3f}  purity={purity:.2f}  '
              f'{dt:.1f}s', flush=True)
        results.append({
            'tag': tag, 'prompt': prompt, 'want': want, 'got': got,
            'passed': passed, 'halted': halted, 'err': err,
            'ops': len(emitted), 'max_cycles': max_cycles,
            'margin_stats': margin,
            'top5_same_opcode_purity': purity,
            'opcode_histogram': _opcode_histogram(trace),
            'elapsed_s': dt,
            'trace': trace,
        })
    wall = round(time.time() - t_start, 2)
    print(f'  → display: {passes}/{len(tasks)}  wall={wall}s\n', flush=True)
    return {
        'tier': 'display',
        'passes': passes, 'total': len(tasks),
        'pass_rate': round(passes / max(len(tasks), 1), 3),
        'wall_s': wall,
        'tasks': results,
    }


def run_consistency(model, tok, device: str, trials: int, mit: int,
                    use_ct: bool, use_cp: bool) -> dict:
    """Determinism probe: factorial 5 × N. JEPA checkpoint claims 10/10."""
    prompt = 'compute 5 factorial and store it'
    runs = []
    t_start = time.time()
    for i in range(trials):
        try:
            cpu, emitted, halted, err, trace = run_traced(
                model, tok, prompt, device, 200, mit, use_ct, use_cp)
            mem = cpu.mem_word(DATA_BASE) if cpu else None
            passed = bool(halted and not err and mem == 120)
        except Exception as e:
            emitted, halted, err, trace = [], False, f'EXC: {e}', []
            mem, passed = None, False
        runs.append({
            'i': i+1, 'passed': passed, 'ops': len(emitted),
            'mem': mem, 'halted': halted, 'err': err,
            'margin_mean': _margin_stats(trace)['mean'],
        })
        print(f'  run {i+1:2d}  ops={len(emitted):3d}  mem={mem}  '
              f'halt={halted}  margin={runs[-1]["margin_mean"]:.3f}',
              flush=True)
    wall = round(time.time() - t_start, 2)
    passes = sum(1 for r in runs if r['passed'])
    op_counts = [r['ops'] for r in runs if r['passed']]
    op_deterministic = len(set(op_counts)) == 1 if op_counts else False
    print(f'  → consistency: {passes}/{trials}  '
          f'op-counts={sorted(set(op_counts))}  wall={wall}s\n',
          flush=True)
    return {
        'passes': passes, 'trials': trials,
        'pass_rate': round(passes / trials, 3),
        'op_counts_seen': sorted(set(r['ops'] for r in runs)),
        'op_deterministic': op_deterministic,
        'wall_s': wall,
        'runs': runs,
    }


# ── aggregate across tiers ────────────────────────────────────────────
def _flat_margin(tier: dict) -> list[float]:
    out = []
    for t in tier.get('tasks', []):
        for c in t.get('trace', []):
            out.append(c['margin'])
    return out


def _overall_purity(tier: dict) -> float:
    all_traces = []
    for t in tier.get('tasks', []):
        all_traces.extend(t.get('trace', []))
    return _top5_same_opcode_purity(all_traces)


def _overall_opcode_hist(tier: dict) -> dict:
    all_traces = []
    for t in tier.get('tasks', []):
        all_traces.extend(t.get('trace', []))
    return _opcode_histogram(all_traces)


def build_summary(tier_results: list[dict], consistency: dict) -> dict:
    passes_total = sum(t['passes'] for t in tier_results)
    total_total = sum(t['total'] for t in tier_results)
    all_margins = []
    for tr in tier_results:
        all_margins.extend(_flat_margin(tr))
    m_sorted = sorted(all_margins)
    def q(p):
        if not m_sorted: return 0.0
        return round(m_sorted[min(int(p * len(m_sorted)), len(m_sorted)-1)], 4)
    return {
        'tasks_passed': passes_total, 'tasks_total': total_total,
        'pass_rate': round(passes_total / max(total_total, 1), 3),
        'consistency_pass_rate': consistency.get('pass_rate', 0.0),
        'consistency_op_deterministic': consistency.get('op_deterministic', False),
        'margin_all_tiers': {
            'n_cycles': len(all_margins),
            'mean': round(sum(all_margins)/max(len(all_margins),1), 4),
            'p10': q(0.10), 'p50': q(0.50), 'p90': q(0.90),
            'min': q(0.0), 'max': q(1.0),
        },
        'top5_same_opcode_purity_by_tier': {
            tr['tier']: _overall_purity(tr) for tr in tier_results
        },
        'opcode_histogram_by_tier': {
            tr['tier']: _overall_opcode_hist(tr) for tr in tier_results
        },
        'failures_by_tier': {
            tr['tier']: [t['tag'] for t in tr['tasks'] if not t['passed']]
            for tr in tier_results
        },
    }


# ── main ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='reflex.pt')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='/tmp/jepa_testbed.json')
    ap.add_argument('--trials', type=int, default=5,
                    help='Consistency trials (factorial 5)')
    ap.add_argument('--skip-consistency', action='store_true')
    ap.add_argument('--tiers', nargs='+', default=None,
                    help='Subset of tiers to run. Default: all.')
    args = ap.parse_args()

    print(f'Loading {args.ckpt} on {args.device}…', flush=True)
    model, tok, cfg = load(args.ckpt, args.device)
    mit = cfg.get('max_instr_tokens', MAX_INSTR_TOKENS)
    use_ct = bool(cfg.get('chat_template', True))
    use_cp = bool(cfg.get('context_prefix', False))
    print(f'cfg: num_instrs={cfg.get("num_instrs")} embed_dim={cfg.get("embed_dim")} '
          f'chat_template={use_ct} context_prefix={use_cp}\n', flush=True)

    selected = TIERS
    if args.tiers:
        selected = [(n, t) for (n, t) in TIERS if n in args.tiers]

    tier_results = []
    for tier_name, tasks in selected:
        print(f'=== {tier_name} ({len(tasks)} tasks) ===', flush=True)
        if tier_name == 'display':
            tr = run_display_tier(model, tok, args.device, tasks, mit, use_ct, use_cp)
        else:
            tr = run_code_tier(model, tok, args.device, tier_name, tasks,
                               mit, use_ct, use_cp)
        tier_results.append(tr)

    consistency = {'pass_rate': None}
    if not args.skip_consistency:
        print(f'=== consistency (factorial 5 × {args.trials}) ===', flush=True)
        consistency = run_consistency(model, tok, args.device, args.trials,
                                       mit, use_ct, use_cp)

    summary = build_summary(tier_results, consistency)
    print('=== SUMMARY ===', flush=True)
    print(f'  pass rate:               {summary["tasks_passed"]}/{summary["tasks_total"]}  '
          f'({summary["pass_rate"]*100:.0f}%)', flush=True)
    print(f'  consistency pass rate:   {summary["consistency_pass_rate"]}  '
          f'op-deterministic={summary["consistency_op_deterministic"]}', flush=True)
    print(f'  margin across all cycles: mean={summary["margin_all_tiers"]["mean"]}  '
          f'p10={summary["margin_all_tiers"]["p10"]}  '
          f'p50={summary["margin_all_tiers"]["p50"]}  '
          f'p90={summary["margin_all_tiers"]["p90"]}', flush=True)
    print(f'  top-5 same-opcode purity by tier:  {summary["top5_same_opcode_purity_by_tier"]}',
          flush=True)
    print(f'  failures by tier:        {summary["failures_by_tier"]}', flush=True)

    out = {
        'ckpt': args.ckpt, 'device': args.device,
        'cfg': {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))},
        'summary': summary,
        'tiers': tier_results,
        'consistency': consistency,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f'\nwrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
