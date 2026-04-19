"""Full eval suite for the Coder-7B checkpoint.

Categories:
  1. In-dist     — 8 canonical tasks
  2. OOD         — 10 extrapolation tasks
  3. Display     — 4 display prompts
  4. Novel       — 9 zero-shot novel tasks
  5. Consistency — factorial 5 × 10 runs, count exact matches

Prints a summary table and writes eval_results.json alongside.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from reflex.demo import load, run_grounded
from reflex.model import MAX_INSTR_TOKENS
from reflex.programs import DISPLAY_BASE, DST_OFFSET
from reflex.riscv import DATA_BASE


IN_DIST = [
    ('add 7+8',     'add 7 and 8 and store the result',                   'mem', 15),
    ('factorial 5', 'compute 5 factorial and store it',                   'mem', 120),
    ('fib 6',       'store the first 6 Fibonacci numbers',                'seq', [0,1,1,2,3,5]),
    ('countdown 5', 'count down from 5 to 1 and store each value',        'seq', [5,4,3,2,1]),
    ('sum 1..10',   'compute 1 + 2 + ... + 10 and store the sum',         'mem', 55),
    ('max(7,12)',   'find the max of 7 and 12 and store it',              'mem', 12),
    ('memcpy 4',    'copy 4 words from source to destination',            'dst', [1,2,3,4]),
    ('double 25',   'call a function that doubles 25 and store the result','mem', 50),
]

OOD = [
    ('factorial 7',    'compute 7 factorial and store it',                   'mem', 5040),
    ('fib 10',         'store the first 10 fibonacci numbers',               'seq', [0,1,1,2,3,5,8,13]),
    ('countdown 20',   'count down from 20 to 1 and store each value',       'seq', [20,19,18,17,16,15,14,13]),
    ('add 100+200',    'add 100 and 200 and store the result',               'mem', 300),
    ('max(3,3)',       'find the max of 3 and 3 and store it',               'mem', 3),
    ('memcpy 8',       'copy 8 words from source to destination',            'dst', [1,2,3,4,5,6,7,8]),
    ('sum 1..20',      'compute 1+2+...+20 and store the sum',               'mem', 210),
    ('subtract 25-10', 'subtract 10 from 25 and store the result',           'mem', 15),
    ('double 100',     'double 100',                                         'mem', 200),
    ('fib 3',          'store the first 3 fibonacci numbers',                'seq', [0,1,1]),
]

DISPLAY = [
    ('say hi',    'say hi',     'hi'),
    ('display OK','display OK', 'OK'),
    ('show 42',   'show 42',    '42'),
    ('print hello','print hello','hello'),
]

NOVEL = [
    ('multiply 3*4', 'multiply 3 and 4 and store the result', 'mem', 12),
    ('multiply 7*8', 'multiply 7 and 8 and store the result', 'mem', 56),
    ('power 2^5',    'compute 2 to the power of 5 and store the result', 'mem', 32),
    ('min(7,3,9)',   'find the minimum of 7, 3, and 9 and store it',     'mem', 3),
    ('abs(-5)',      'compute the absolute value of -5 and store it',    'mem', 5),
    ('popcount 255', 'count the number of 1-bits in 255 and store it',   'mem', 8),
    ('say wow',      'say wow',                                          'disp', 'wow'),
    ('say 42',       'say the number 42',                                'disp', '42'),
    ('count up 1..5','count up from 1 to 5 and store each value',        'seq', [1,2,3,4,5]),
]


def check(cpu, kind, expected):
    if kind == 'mem':
        got = cpu.mem_word(DATA_BASE)
        return got == expected, f'mem={got}'
    if kind == 'seq':
        got = [cpu.mem_word(DATA_BASE + 4*i) for i in range(len(expected))]
        return got == expected, f'seq={got}'
    if kind == 'dst':
        got = [cpu.mem_word(DATA_BASE + DST_OFFSET + 4*i) for i in range(len(expected))]
        return got == expected, f'dst={got}'
    if kind == 'disp':
        got = [cpu.mem_word(DISPLAY_BASE + 4*i) for i in range(len(expected))]
        expected_bytes = [ord(c) for c in expected]
        disp = ''.join(chr(b) if 32<=b<127 else '·' for b in got)
        return got == expected_bytes, f'display={disp!r}'
    return False, '?'


def run_one(model, tok, prompt, max_tok, kind):
    """Run grounded emission on a single prompt."""
    cpu, emitted, halted, err = run_grounded(
        model, tok, prompt, 'cuda', max_cycles=400, max_instr_tokens=max_tok)
    return cpu, emitted, halted, err


def run_section(label, tasks, model, tok, max_tok, kind_col=2):
    print(f'\n=== {label} ===')
    rows = []
    passed = 0
    for entry in tasks:
        tag = entry[0]
        prompt = entry[1]
        kind = entry[kind_col]
        expected = entry[kind_col + 1]
        cpu, emitted, halted, err = run_one(model, tok, prompt, max_tok, kind)
        ok, msg = check(cpu, kind, expected)
        pass_ = ok and halted and not err
        mark = '✓' if pass_ else '✗'
        tail = f'  err={err}' if err else ''
        print(f'  {mark} {tag:<18} ops={len(emitted):>4} halt={str(halted):<5} '
              f'{msg} want={expected}{tail}')
        rows.append({
            'tag': tag, 'prompt': prompt, 'expected': expected,
            'result': msg, 'ops': len(emitted), 'halted': halted,
            'err': err, 'pass': pass_,
        })
        if pass_:
            passed += 1
    print(f'{label}: {passed}/{len(tasks)}')
    return {'passed': passed, 'total': len(tasks), 'rows': rows}


def run_consistency(model, tok, max_tok, n_trials=10):
    """factorial 5 × 10 runs; check all return 120."""
    label = 'consistency: factorial 5 × 10'
    print(f'\n=== {label} ===')
    results = []
    passed = 0
    for i in range(n_trials):
        cpu, emitted, halted, err = run_one(
            model, tok, 'compute 5 factorial and store it', max_tok, 'mem')
        got = cpu.mem_word(DATA_BASE)
        pass_ = (got == 120) and halted and not err
        mark = '✓' if pass_ else '✗'
        tail = f'  err={err}' if err else ''
        print(f'  {mark} run {i+1:2d}  ops={len(emitted):>3} halt={str(halted):<5} '
              f'mem={got}{tail}')
        results.append({'run': i+1, 'got': got, 'ops': len(emitted),
                        'halted': halted, 'err': err, 'pass': pass_})
        if pass_:
            passed += 1
    print(f'{label}: {passed}/{n_trials}')
    return {'passed': passed, 'total': n_trials, 'rows': results}


def main():
    ap = argparse.ArgumentParser(
        description='Reflex headless eval suite: in-dist + OOD + display '
        '+ novel zero-shot + consistency.')
    ap.add_argument('--ckpt', '--checkpoint', required=True,
                    dest='ckpt', metavar='PATH',
                    help='Path to the trained Reflex checkpoint.')
    ap.add_argument('--out', default='eval_results.json')
    args = ap.parse_args()

    print(f'loading {args.ckpt}')
    model, tok, cfg = load(args.ckpt, 'cuda')
    max_tok = cfg.get('max_instr_tokens', MAX_INSTR_TOKENS)

    t0 = time.time()
    report = {
        'ckpt': args.ckpt,
        'config': {k: cfg[k] for k in cfg if isinstance(cfg[k], (str, int, bool, float))},
        'sections': {},
    }
    report['sections']['in_dist'] = run_section('IN-DIST', IN_DIST, model, tok, max_tok)
    report['sections']['ood'] = run_section('OOD', OOD, model, tok, max_tok)
    report['sections']['display'] = run_section(
        'DISPLAY',
        [(t[0], t[1], 'disp', t[2]) for t in DISPLAY],
        model, tok, max_tok)
    report['sections']['novel'] = run_section('NOVEL', NOVEL, model, tok, max_tok)
    report['sections']['consistency'] = run_consistency(model, tok, max_tok)

    report['wall_seconds'] = time.time() - t0

    print('\n=== SUMMARY ===')
    for name, sec in report['sections'].items():
        print(f'  {name:<14} {sec["passed"]}/{sec["total"]}')
    print(f'  wall time: {report["wall_seconds"]:.0f}s')

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f'\nresults → {args.out}')


if __name__ == '__main__':
    main()
