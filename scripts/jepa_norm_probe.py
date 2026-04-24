"""Probe whether pre-head norms carry inference-time confidence signal
independent of the cosine margin.

The JEPA head reads the backbone's pooled last-token hidden state,
runs it through a 2-layer MLP, then a linear projection to 256-d, and
compares that 256-d vector (by cosine similarity) to a 691-row
codebook. Two norms are discarded along the way:

  * ``||pooled||``       — the raw pooled hidden state
  * ``||pred||``         — the 256-d projection BEFORE cosine norm

Cosine keeps only direction. The norms are available but unused by the
decoder. Question: do failing cycles look different in these norms?

The script runs the canonical decoder over the 23-task suite and,
per cycle, records ``(tag, cycle, top1_sim, margin, ||pooled||,
||pred||, bias_proj)``. ``bias_proj`` is the L2-distance between
``pred`` and the embed-head bias — the MHA-bias-dominance residual
flagged in the latent-recurrence work. At small ``||pred||`` the
direction of ``pred`` is dominated by the bias; ``bias_proj`` being
near zero on a cycle is a structural "the model has nothing to say"
signal.

Output: JSON file of per-cycle records + a printed summary
contrasting the norm distribution on passing vs failing cycles
(failing = cycles in the programs that failed the suite).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import torch

from reflex.decoders.base import (
    emit_and_step, fresh_cpu, prep_prompt,
)
from reflex.demo import load
from reflex.model import MAX_INSTR_TOKENS, extract_state
from reflex.programs import DISPLAY_BASE
from reflex.riscv import HALT_INSTR, Rv32i
from scripts.jepa_testbed import (
    SHORT, MEDIUM, LONG, DISPLAY_TIER, _check, _display_read,
)


def _pass(tier: str, cpu: Rv32i, halted: bool, err: str, row) -> bool:
    if tier == 'display':
        _, _, want, _ = row
        return bool(halted and not err and _display_read(cpu, len(want)) == want)
    _, _, kind, expected, _ = row
    return bool(halted and not err and _check(cpu, kind, expected))


@torch.no_grad()
def trace_one(model, tok, prompt: str, device: str,
              max_cycles: int, mit: int, use_ct: bool, use_cp: bool,
              seed_memcpy: bool):
    ids, amask = prep_prompt(tok, prompt, device,
                              max_instr_tokens=mit,
                              use_chat_template=use_ct,
                              use_context_prefix=use_cp)
    cpu = fresh_cpu(seed_memcpy)
    # Embed-head bias vector, materialised once per run.
    bias = model.embed_head.bias.detach().float().cpu() \
        if model.embed_head.bias is not None else None

    records = []
    halted, err = False, ''
    for cycle in range(max_cycles):
        pc = cpu.pc
        state = extract_state(cpu)
        state_t = torch.from_numpy(state.astype('int64')).unsqueeze(0).to(device)
        # Replicate model.forward while capturing intermediates.
        kv = model.kv_norm(model.state_encoder(state_t))
        model._current_kv = kv
        try:
            out = model.backbone(input_ids=ids, attention_mask=amask,
                                  output_hidden_states=False,
                                  return_dict=True, use_cache=False)
        finally:
            model._current_kv = None
        head_dtype = next(model.head_mlp.parameters()).dtype
        h = out.last_hidden_state.to(head_dtype)
        last_idx = amask.sum(dim=1).long() - 1
        pooled = h[torch.arange(h.size(0), device=h.device), last_idx]  # [1, H]
        feat = model.head_mlp(pooled)
        pred = model.embed_head(feat)                                   # [1, 256]

        pooled_norm = float(pooled.float().norm())
        pred_norm = float(pred.float().norm())
        bias_dist = float((pred.float().cpu() - bias).norm()) if bias is not None else None

        sims = model.table_similarity(pred).squeeze(0)
        top = torch.topk(sims, 2)
        top_sims = [float(s) for s in top.values.tolist()]
        top_idx = top.indices.tolist()
        instr_w = int(model.instr_words[top_idx[0]].item()) & 0xFFFFFFFF
        margin = top_sims[0] - top_sims[1]

        records.append({
            'cycle': cycle, 'pc': pc, 'word': f'0x{instr_w:08x}',
            'top1_sim': round(top_sims[0], 4),
            'margin': round(margin, 4),
            'pooled_norm': round(pooled_norm, 4),
            'pred_norm': round(pred_norm, 4),
            'bias_dist': round(bias_dist, 4) if bias_dist is not None else None,
        })

        halted, err = emit_and_step(cpu, pc, instr_w)
        if halted or err:
            break
    return cpu, records, halted, err


def _quantiles(xs: list[float]) -> dict:
    if not xs:
        return {'n': 0, 'mean': 0, 'p10': 0, 'p50': 0, 'p90': 0, 'min': 0, 'max': 0}
    s = sorted(xs)
    def q(p): return s[min(int(p * len(s)), len(s) - 1)]
    return {'n': len(xs), 'mean': round(sum(xs) / len(xs), 4),
            'p10': q(0.10), 'p50': q(0.50), 'p90': q(0.90),
            'min': s[0], 'max': s[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='reflex.pt')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='/tmp/jepa_norm_probe.json')
    args = ap.parse_args()

    model, tok, cfg = load(args.ckpt, args.device)
    mit = cfg.get('max_instr_tokens', MAX_INSTR_TOKENS)
    use_ct = bool(cfg.get('chat_template', True))
    use_cp = bool(cfg.get('context_prefix', False))

    all_tasks = [('short', SHORT), ('medium', MEDIUM),
                 ('long', LONG), ('display', DISPLAY_TIER)]
    passing_cycles = {'top1_sim': [], 'margin': [],
                      'pooled_norm': [], 'pred_norm': [], 'bias_dist': []}
    failing_cycles = {'top1_sim': [], 'margin': [],
                      'pooled_norm': [], 'pred_norm': [], 'bias_dist': []}
    all_records = []
    for tier_name, tasks in all_tasks:
        for row in tasks:
            tag = row[0]
            prompt = row[1]
            max_cycles = row[-1]
            seed_memcpy = (tier_name != 'display')
            cpu, records, halted, err = trace_one(
                model, tok, prompt, args.device, max_cycles, mit,
                use_ct, use_cp, seed_memcpy)
            passed = _pass(tier_name, cpu, halted, err, row)
            sink = passing_cycles if passed else failing_cycles
            for r in records:
                for k in sink:
                    if r.get(k) is not None:
                        sink[k].append(r[k])
            all_records.append({'tier': tier_name, 'tag': tag,
                                 'passed': passed, 'ops': len(records),
                                 'records': records})
            print(f'  {"✓" if passed else "✗"} {tier_name:<8s} {tag:<18s} '
                  f'ops={len(records):3d}  '
                  f'mean_pred_norm={np.mean([r["pred_norm"] for r in records]):.2f}',
                  flush=True)

    print('\n=== distribution (passing-task cycles vs failing-task cycles) ===')
    for k in ('top1_sim', 'margin', 'pooled_norm', 'pred_norm', 'bias_dist'):
        qp = _quantiles(passing_cycles[k])
        qf = _quantiles(failing_cycles[k])
        print(f'  {k:<13s}  pass: n={qp["n"]} mean={qp["mean"]} p10={qp["p10"]} p50={qp["p50"]} p90={qp["p90"]}')
        print(f'                fail: n={qf["n"]} mean={qf["mean"]} p10={qf["p10"]} p50={qf["p50"]} p90={qf["p90"]}')

    Path(args.out).write_text(json.dumps({
        'ckpt': args.ckpt, 'device': args.device,
        'passing_distribution': {k: _quantiles(v) for k, v in passing_cycles.items()},
        'failing_distribution': {k: _quantiles(v) for k, v in failing_cycles.items()},
        'all_records': all_records,
    }, indent=2))
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
