"""Run the same OOD prompts as eval_ood.py but against the LoRA+head model."""
import sys
import torch

from reflex.demo_lora import load, run_grounded
from reflex.programs import DST_OFFSET
from reflex.riscv import DATA_BASE

PROMPTS = [
    ('factorial 7',        'compute 7 factorial and store it',                   'mem',     5040),
    ('fib 10',             'store the first 10 fibonacci numbers',               'mem10',   None),
    ('countdown 20',       'count down from 20 to 1 and store each value',       'mem20',   None),
    ('add 100+200',        'add 100 and 200 and store the result',               'mem',     300),
    ('max(3,3)',           'find the max of 3 and 3 and store it',               'mem',     3),
    ('memcpy 8',           'copy 8 words from source to destination',            'dst8',    None),
    ('sum 1..20',          'compute 1+2+...+20 and store the sum',               'mem',     210),
    ('subtract 25-10',     'subtract 10 from 25 and store the result',           'mem',     15),
    ('double 100',         'double 100',                                         'mem',     200),
    ('fib 3',              'store the first 3 fibonacci numbers',                'mem3',    None),
]


def expected(label: str, val):
    if label == 'mem':    return [val] + [0]*7
    if label == 'mem3':   return [0, 1, 1, 0, 0, 0, 0, 0]
    if label == 'mem10':  return [0, 1, 1, 2, 3, 5, 8, 13]
    if label == 'mem20':  return [20, 19, 18, 17, 16, 15, 14, 13]


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else 'reflex_lora.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'loading {ckpt}', flush=True)
    model, tok = load(ckpt, device)
    print(f'loaded.', flush=True)

    print(f'\n{"task":<22} {"ops":>4} {"halt":>5}  result')
    print('-'*90)
    for tag, prompt, kind, val in PROMPTS:
        cpu, emitted, halted, err = run_grounded(model, tok, prompt, device, max_cycles=400)
        if kind == 'dst8':
            got = [cpu.mem_word(DATA_BASE + DST_OFFSET + 4*i) for i in range(8)]
            ok = got == [1,2,3,4,5,6,7,8]
            result = f'dst={got}'
        else:
            got = [cpu.mem_word(DATA_BASE + 4*i) for i in range(8)]
            exp = expected(kind, val)
            ok = got[0] == val if kind == 'mem' else got[:len(exp)] == exp[:len(exp)]
            result = f'mem={got[:8]}'
        mark = '✓' if ok and halted and not err else '✗'
        tail = err if err else ''
        print(f'{mark} {tag:<20} {len(emitted):>4} {str(halted):>5}  {result}  {tail}')


if __name__ == '__main__':
    main()
