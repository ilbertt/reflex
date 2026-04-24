"""Probe whether top-k re-rank could rescue failing display tasks.

For each failing task, for each cycle c, for each alternative k in top-2
through top-5, replay the task with that single substitution and see if
the final output is correct. If any (c, k) override flips the task
from fail → pass, re-rank has signal and a margin-aware inference
policy is worth building.

Pure-algorithmic: uses only the existing JEPA codebook top-5 decode, no
hidden-state thread, no training.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import torch

from reflex.demo import load
from reflex.model import (
    MAX_INSTR_TOKENS, code_region_halt_fill, extract_state, render_prompt,
)
from reflex.programs import DISPLAY_BASE
from reflex.riscv import HALT_INSTR, PROGRAM_START, Rv32i


DISPLAY_TESTS = [
    ('display OK', 'display OK', 'OK', 40),
    ('show 42',    'show 42',    '42', 40),
    ('print hello','print hello','hello', 50),
]


def _display_read(cpu, n: int) -> str:
    out = []
    for i in range(n):
        b = cpu.mem_word(DISPLAY_BASE + 4*i) & 0xFF
        out.append(chr(b) if 0x20 <= b < 0x7F else '·')
    return ''.join(out)


@torch.no_grad()
def _fresh_cpu() -> Rv32i:
    cpu = Rv32i()
    cpu.uc.mem_write(PROGRAM_START, code_region_halt_fill())
    return cpu


@torch.no_grad()
def run_with_override(model, tok, prompt: str, device: str,
                       max_cycles: int, mit: int, use_ct: bool, use_cp: bool,
                       override: tuple[int, int] | None = None,
                       capture_topk: int = 5):
    """Run grounded; if override=(c, w), at cycle c emit word w instead
    of top-1. Returns (cpu, emitted, halted, err, traces) where traces
    is a list of (cycle, top_words[topk])."""
    text = render_prompt(tok, prompt,
                          use_chat_template=use_ct, use_context_prefix=use_cp)
    e = tok(text, padding='max_length', truncation=True,
            max_length=mit, return_tensors='pt').to(device)
    ids, amask = e.input_ids, e.attention_mask
    cpu = _fresh_cpu()
    emitted = []
    traces = []
    halted, err = False, ''
    for cycle in range(max_cycles):
        pc = cpu.pc
        state = extract_state(cpu)
        import numpy as np
        state_t = torch.from_numpy(state.astype('int64')).unsqueeze(0).to(device)
        pred = model(ids, amask, state_t)
        sims = model.table_similarity(pred).squeeze(0)
        top = torch.topk(sims, capture_topk)
        top_words = [int(model.instr_words[i].item()) & 0xFFFFFFFF
                     for i in top.indices.tolist()]
        if override is not None and override[0] == cycle:
            instr_w = override[1] & 0xFFFFFFFF
        else:
            instr_w = top_words[0]
        emitted.append(instr_w)
        traces.append((cycle, top_words))
        try:
            cpu.uc.mem_write(pc, int(instr_w).to_bytes(4, 'little'))
            cpu.uc.ctl_remove_cache(pc, pc + 4)
        except Exception as e_:
            err = f'write@{pc:#x}: {e_}'; break
        if instr_w == HALT_INSTR:
            halted = True; break
        if instr_w == 0:
            err = f'zero@{pc:#x}'; break
        try:
            cpu.step()
        except Exception as e_:
            err = f'step@{pc:#x}: {e_}'; break
    return cpu, emitted, halted, err, traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='reflex.pt')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    print(f'Loading {args.ckpt} on {args.device}…', flush=True)
    model, tok, cfg = load(args.ckpt, args.device)
    mit = cfg.get('max_instr_tokens', MAX_INSTR_TOKENS)
    use_ct = bool(cfg.get('chat_template', True))
    use_cp = bool(cfg.get('context_prefix', False))

    for tag, prompt, want, max_cycles in DISPLAY_TESTS:
        print(f'\n=== {tag}  want={want!r} ===', flush=True)
        # Baseline run (no override) to get the top-5 candidates per cycle
        t0 = time.time()
        cpu, emitted, halted, err, traces = run_with_override(
            model, tok, prompt, args.device, max_cycles, mit, use_ct, use_cp)
        got = _display_read(cpu, len(want))
        print(f'  baseline: got={got!r}  ops={len(emitted)}  halted={halted}  '
              f'err={err or "-"}  {time.time()-t0:.1f}s', flush=True)
        if got == want:
            print('  (already passes, skipping)'); continue

        # Try each single-cycle top-k override
        rescues = []
        total_trials = 0
        t1 = time.time()
        for c, top_words in traces:
            baseline_w = top_words[0]
            for k in range(1, len(top_words)):
                alt = top_words[k]
                if alt == baseline_w: continue
                total_trials += 1
                cpu2, em2, h2, er2, _ = run_with_override(
                    model, tok, prompt, args.device, max_cycles, mit, use_ct, use_cp,
                    override=(c, alt))
                g2 = _display_read(cpu2, len(want))
                if g2 == want and h2 and not er2:
                    rescues.append({'cycle': c, 'top_k': k+1,
                                    'from_word': f'0x{baseline_w:08x}',
                                    'to_word': f'0x{alt:08x}',
                                    'ops': len(em2)})
                    print(f'    ✓ rescue: cyc {c}  top-{k+1}  '
                          f'{baseline_w:#010x} → {alt:#010x}  ops={len(em2)}',
                          flush=True)
        print(f'  → {len(rescues)} single-cycle rescues out of '
              f'{total_trials} trials  ({time.time()-t1:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
