"""Combined 8-in-dist + 10-OOD eval with optional context-prefix toggle."""
import argparse
import torch

from reflex.demo import load, run_grounded
from reflex.programs import DST_OFFSET, SRC_OFFSET
from reflex.riscv import DATA_BASE

IN_DIST = [
    ('add 7+8',            'add 7 and 8 and store the result',                   'mem',  15),
    ('factorial 5',        'compute 5 factorial and store it',                   'mem',  120),
    ('fib 6',              'store the first 6 Fibonacci numbers',                'seq',  [0,1,1,2,3,5]),
    ('countdown 5',        'count down from 5 to 1 and store each value',        'seq',  [5,4,3,2,1]),
    ('sum 1..10',          'compute 1 + 2 + ... + 10 and store the sum',         'mem',  55),
    ('max(7,12)',          'find the max of 7 and 12 and store it',              'mem',  12),
    ('memcpy 4',           'copy 4 words from source to destination',            'dst',  [1,2,3,4]),
    ('double 25',          'call a function that doubles 25 and store the result','mem', 50),
]

OOD = [
    ('factorial 7',        'compute 7 factorial and store it',                   'mem',  5040),
    ('fib 10',             'store the first 10 fibonacci numbers',               'seq',  [0,1,1,2,3,5,8,13]),
    ('countdown 20',       'count down from 20 to 1 and store each value',       'seq',  [20,19,18,17,16,15,14,13]),
    ('add 100+200',        'add 100 and 200 and store the result',               'mem',  300),
    ('max(3,3)',           'find the max of 3 and 3 and store it',               'mem',  3),
    ('memcpy 8',           'copy 8 words from source to destination',            'dst',  [1,2,3,4,5,6,7,8]),
    ('sum 1..20',          'compute 1+2+...+20 and store the sum',               'mem',  210),
    ('subtract 25-10',     'subtract 10 from 25 and store the result',           'mem',  15),
    ('double 100',         'double 100',                                         'mem',  200),
    ('fib 3',              'store the first 3 fibonacci numbers',                'seq',  [0,1,1]),
]


def run_one(model, tok, prompt, device, context_prefix, max_tok):
    return run_grounded(model, tok, prompt, device, max_cycles=400,
                        context_prefix=context_prefix,
                        max_instr_tokens=max_tok)


def check(cpu, kind, expected):
    if kind == 'mem':
        got = cpu.mem_word(DATA_BASE)
        return got == expected, f'mem[0]={got}'
    if kind == 'seq':
        got = [cpu.mem_word(DATA_BASE + 4*i) for i in range(len(expected))]
        return got == expected, f'mem={got}'
    if kind == 'dst':
        got = [cpu.mem_word(DATA_BASE + DST_OFFSET + 4*i) for i in range(len(expected))]
        return got == expected, f'dst={got}'
    return False, '?'


def section(label, tasks, model, tok, device, context_prefix, max_tok):
    print(f'\n=== {label} ===')
    correct = 0
    for tag, prompt, kind, expected in tasks:
        cpu, emitted, halted, err = run_one(
            model, tok, prompt, device, context_prefix, max_tok)
        ok, result = check(cpu, kind, expected)
        mark = '✓' if ok and halted and not err else '✗'
        tail = f'err={err}' if err else ''
        print(f'{mark} {tag:<18} ops={len(emitted):>3} halt={str(halted):<5} {result}  {tail}')
        if mark == '✓':
            correct += 1
    print(f'{label}: {correct}/{len(tasks)}')
    return correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--context-prefix', action='store_true', default=False,
                    help='Override: prepend context prefix at inference.')
    ap.add_argument('--max-instr-tokens', type=int, default=None,
                    help='Override tokenizer max length.')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    print(f'loading {args.ckpt}')
    model, tok = load(args.ckpt, device)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    # If the checkpoint was trained with a prefix, auto-enable at inference.
    cp = args.context_prefix or cfg.get('context_prefix', False)
    mt = args.max_instr_tokens or cfg.get('max_instr_tokens', 32)
    print(f'context_prefix={cp}  max_instr_tokens={mt}')
    del ckpt

    n_in = section('IN-DIST', IN_DIST, model, tok, device, cp, mt)
    n_ood = section('OOD', OOD, model, tok, device, cp, mt)
    print(f'\nTOTAL: in-dist {n_in}/8  ood {n_ood}/10')


if __name__ == '__main__':
    main()
