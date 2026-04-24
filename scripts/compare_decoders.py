"""Run the 23-task JEPA testbed across every registered decoder and
report pass rates side by side.

Usage:
    python scripts/compare_decoders.py --ckpt reflex.pt --device cuda
        [--decoders pure tiebreak exec_verify beam]
        [--tiers short medium long display]

Does not re-implement the task lists — pulls them from
``scripts.jepa_testbed`` so the decoders are compared on the same grid
as the primary testbed. Each decoder's extra kwargs are threaded
through a small per-decoder config table.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from reflex.decoders import DECODERS, get as get_decoder
from reflex.demo import load
from reflex.model import MAX_INSTR_TOKENS
from scripts.jepa_testbed import (
    SHORT, MEDIUM, LONG, DISPLAY_TIER, _check, _display_read,
)


# Per-decoder default hyperparameters. Override any of these at the
# command line if needed.
DECODER_DEFAULTS = {
    'pure':        {},
    'tiebreak':    {'margin_eps': 0.15, 'halt_margin_eps': 0.30,
                    'topk': 5, 'min_cycle': 2, 'max_interventions': 2},
    'exec_verify': {'margin_eps': 0.15, 'topk': 4, 'min_cycle': 2,
                    'max_interventions': 3, 'lookahead_depth': 6},
    'beam':        {'beam_width': 4, 'branching': 3},
}


def _pass(tier_name: str, cpu, halted, err, row) -> tuple[bool, str]:
    """Returns (passed, display_str_if_relevant)."""
    if tier_name == 'display':
        _, _, want, _ = row
        got = _display_read(cpu, len(want))
        return bool(halted and not err and got == want), got
    _, _, kind, expected, _ = row
    return bool(halted and not err and _check(cpu, kind, expected)), ''


def _build_kwargs(decoder_name: str, cfg: dict, overrides: dict) -> dict:
    """Merge defaults + user overrides, and thread the checkpoint
    config's prompt-rendering flags so legacy checkpoints still work."""
    kw = dict(DECODER_DEFAULTS.get(decoder_name, {}))
    kw.update(overrides)
    kw['max_instr_tokens'] = cfg.get('max_instr_tokens', MAX_INSTR_TOKENS)
    kw['use_chat_template'] = bool(cfg.get('chat_template', True))
    kw['use_context_prefix'] = bool(cfg.get('context_prefix', False))
    return kw


def run_decoder_on_suite(decoder_name: str, model, tok, cfg, device: str,
                         tiers: dict[str, list], overrides: dict) -> dict:
    fn = get_decoder(decoder_name)
    kwargs = _build_kwargs(decoder_name, cfg, overrides)
    print(f'\n####  {decoder_name}  ####  kwargs={kwargs}', flush=True)
    results = []
    total_passes = 0
    total_tasks = 0
    grand_t0 = time.time()
    for tier_name, tasks in tiers.items():
        print(f'  === {tier_name} ({len(tasks)} tasks) ===', flush=True)
        tier_passes = 0
        tier_rows = []
        for row in tasks:
            tag = row[0]; prompt = row[1]; max_cycles = row[-1]
            seed_memcpy = (tier_name != 'display')
            t0 = time.time()
            try:
                cpu, emitted, halted, err, meta = fn(
                    model, tok, prompt, device, max_cycles,
                    seed_memcpy=seed_memcpy, **kwargs)
                passed, got = _pass(tier_name, cpu, halted, err, row)
            except Exception as e:
                cpu, emitted, halted, err = None, [], False, f'EXC: {e}'
                passed, got, meta = False, '', {'decoder': decoder_name, 'exc': str(e)}
            dt = time.time() - t0
            tier_passes += int(passed)
            total_passes += int(passed)
            total_tasks += 1
            extra = ''
            if tier_name == 'display':
                extra = f"  got={got!r}"
            print(f'    {"✓" if passed else "✗"} {tag:<18s}  '
                  f'ops={len(emitted):4d}  {dt:.1f}s{extra}', flush=True)
            tier_rows.append({'tag': tag, 'passed': passed, 'ops': len(emitted),
                              'halted': halted, 'err': err, 'got': got,
                              'elapsed_s': round(dt, 2),
                              'meta': meta})
        print(f'    → {tier_name}: {tier_passes}/{len(tasks)}', flush=True)
        results.append({'tier': tier_name,
                         'passes': tier_passes, 'total': len(tasks),
                         'tasks': tier_rows})
    wall = round(time.time() - grand_t0, 2)
    print(f'  TOTAL: {total_passes}/{total_tasks}  wall={wall}s', flush=True)
    return {'decoder': decoder_name, 'kwargs': kwargs,
            'passes': total_passes, 'total': total_tasks,
            'wall_s': wall, 'tiers': results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='reflex.pt')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--decoders', nargs='+',
                    default=['pure', 'tiebreak', 'exec_verify', 'beam'])
    ap.add_argument('--tiers', nargs='+',
                    default=['short', 'medium', 'long', 'display'])
    ap.add_argument('--out', default='/tmp/compare_decoders.json')
    # Pass-through overrides for every decoder. Only applied to decoders
    # whose defaults have a matching key.
    ap.add_argument('--margin-eps', type=float, default=None)
    ap.add_argument('--halt-margin-eps', type=float, default=None)
    ap.add_argument('--topk', type=int, default=None)
    ap.add_argument('--min-cycle', type=int, default=None)
    ap.add_argument('--max-interventions', type=int, default=None)
    ap.add_argument('--lookahead-depth', type=int, default=None)
    ap.add_argument('--beam-width', type=int, default=None)
    ap.add_argument('--branching', type=int, default=None)
    args = ap.parse_args()

    overrides = {}
    for key in ('margin_eps','halt_margin_eps','topk','min_cycle',
                'max_interventions','lookahead_depth','beam_width','branching'):
        val = getattr(args, key)
        if val is not None:
            overrides[key] = val

    for name in args.decoders:
        if name not in DECODERS:
            raise SystemExit(f'unknown decoder {name!r}; '
                             f'choose from {sorted(DECODERS)}')

    print(f'Loading {args.ckpt} on {args.device}…', flush=True)
    model, tok, cfg = load(args.ckpt, args.device)
    print(f'cfg: num_instrs={cfg.get("num_instrs")} '
          f'embed_dim={cfg.get("embed_dim")}\n', flush=True)

    tier_map = {'short': SHORT, 'medium': MEDIUM, 'long': LONG,
                'display': DISPLAY_TIER}
    tiers = {n: tier_map[n] for n in args.tiers}

    per_decoder = []
    for name in args.decoders:
        per_decoder.append(run_decoder_on_suite(
            name, model, tok, cfg, args.device, tiers, overrides))

    print('\n=== COMPARISON ===', flush=True)
    header = f'  {"decoder":<12s}  {"passes":>7s}   pass_rate'
    print(header, flush=True)
    for r in per_decoder:
        rate = r['passes'] / max(r['total'], 1)
        print(f"  {r['decoder']:<12s}  {r['passes']:>3d}/{r['total']:<3d}   "
              f"{100*rate:5.1f}%  ({r['wall_s']}s)", flush=True)

    Path(args.out).write_text(json.dumps({
        'ckpt': args.ckpt, 'device': args.device,
        'cfg': {k: v for k, v in cfg.items()
                if isinstance(v, (int, float, str, bool))},
        'overrides': overrides,
        'decoders': per_decoder,
    }, indent=2))
    print(f'\nwrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
