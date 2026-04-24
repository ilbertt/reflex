"""Retrain only the target-aware head on top of an existing JEPA
checkpoint, freezing everything else.

This is the Path-B training step from ``HEAD_STUDY.md``. The
``TargetAwareHead`` module defined in ``reflex.model`` adds an explicit
cross-attention pathway from the pooled hidden state to the backbone's
per-token hidden states, gate-initialised to zero so the head starts
as identity. Retraining only the target head (+ optionally the
``head_mlp``, ``embed_head``, and ``instr_table``) teaches the gate
to open and the cross-attention weights to surface the prompt's
target-intent at decoding time.

What stays frozen:
  * backbone (Qwen-Coder-7B, bf16, untouched per README)
  * state encoder
  * seven Flamingo adapters
  * kv_norm

What trains:
  * TargetAwareHead (cross-attn, projections, gate)
  * optionally: head_mlp, embed_head, instr_table (set --train-head)

Corpus:
  Uses the same (family, instr_text, program_bytes) corpus as
  ``reflex.train.main`` — see that module's ``collect_state_sequences``
  for the expected format. The corpus is NOT generated here; run
  ``scripts/generate_programs.py`` once first.

Loss:
  Identical InfoNCE over the codebook as the canonical train; the
  target-head's cross-attention pathway should learn during this
  training to route prompt-target signal into the pooled state so
  that the codebook's near-neighbour ambiguities are resolved.

Typical run on an L40S 48 GB:
  python scripts/retrain_target_head.py \
      --base-ckpt reflex.pt --out reflex_target_head.pt \
      --corpus <corpus.pkl> --steps 3000 --batch 16 --lr 5e-5

Not executed in-session because corpus generation is hours of wall
time; ship the script so the retrain is ready to run when a GPU is
free.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reflex.model import (
    BACKBONE_ID, EMBED_DIM, INJECT_EVERY, MAX_INSTR_TOKENS,
    GroundedReflex, build_backbone, render_prompt,
)

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


def _freeze_everything_except_target_head(model: GroundedReflex,
                                           train_head: bool = False):
    """Return the list of parameters that will receive gradients.

    By default only the TargetAwareHead trains. ``--train-head`` also
    unfreezes ``head_mlp``, ``embed_head``, and the codebook
    ``instr_table`` — useful when the aux-head alone underfits because
    the downstream head was never asked to consume a target signal.
    """
    trainable = []
    for p in model.parameters():
        p.requires_grad_(False)
    if model.target_head is None:
        raise SystemExit(
            'Model was built without use_target_head=True; rebuild via '
            'reflex.demo.load on a ckpt whose config sets '
            'use_target_head=True, or construct GroundedReflex manually.')
    for p in model.target_head.parameters():
        p.requires_grad_(True); trainable.append(p)
    if train_head:
        for name in ('head_mlp', 'embed_head', 'instr_table'):
            mod = getattr(model, name)
            for p in mod.parameters():
                p.requires_grad_(True); trainable.append(p)
    return trainable


def _infonce_step(model, tok, batch, device: str, temp: float):
    """One InfoNCE training step. ``batch`` is a list of
    (prompt_text, state[65], instr_row_index). Returns the loss."""
    prompts = [render_prompt(tok, b[0], use_chat_template=True) for b in batch]
    e = tok(prompts, padding='max_length', truncation=True,
            max_length=MAX_INSTR_TOKENS, return_tensors='pt').to(device)
    states = torch.stack([torch.from_numpy(b[1].astype('int64')) for b in batch]).to(device)
    gt = torch.tensor([b[2] for b in batch], dtype=torch.long, device=device)
    pred = model(e.input_ids, e.attention_mask, states)             # [B, embed_dim]
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(model.instr_table.weight.float(), dim=-1)
    logits = (p @ t.T) / temp
    return F.cross_entropy(logits, gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-ckpt', required=True,
                    help='Existing JEPA ckpt to initialise adapters + codebook from.')
    ap.add_argument('--corpus', required=True,
                    help='Pickle of list[(family, prompt, state[65], gt_instr_idx)] '
                    'pre-flattened from the program cycle pool. See generate.')
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--nce-temp', type=float, default=0.07)
    ap.add_argument('--train-head', action='store_true',
                    help='Also retrain head_mlp + embed_head + instr_table, not '
                    'just the target head.')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    print(f'Loading base ckpt {args.base_ckpt}…', flush=True)
    ckpt = torch.load(args.base_ckpt, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    cfg = {**cfg, 'use_target_head': True}     # flip the flag
    dtype = torch.bfloat16 if args.device.startswith('cuda') else torch.float32
    bb, tok, hidden = build_backbone(cfg['backbone_id'], dtype=dtype)
    bb = bb.to(args.device)
    model = GroundedReflex(
        bb, cfg['hidden'],
        num_instrs=cfg['num_instrs'],
        inject_every=cfg.get('inject_every', INJECT_EVERY),
        adapter_mlp_ratio=cfg.get('adapter_mlp_ratio', 2),
        embed_dim=cfg.get('embed_dim', EMBED_DIM),
        use_target_head=True,
        freeze_backbone=True).to(args.device)
    # Load everything except the new target_head (which is fresh-initialised).
    missing, unexpected = model.load_state_dict(ckpt['state'], strict=False)
    missing = [m for m in missing if 'target_head' not in m]
    if missing:
        raise SystemExit(f'Unexpected missing keys: {missing}')
    print(f'  target_head gate @ init: {float(model.target_head.gate.item()):.4f} '
          f'(should be 0.0 → head is identity)')

    with open(args.corpus, 'rb') as f:
        corpus = pickle.load(f)
    print(f'  corpus: {len(corpus):,} cycles', flush=True)

    trainable = _freeze_everything_except_target_head(model, train_head=args.train_head)
    n_params = sum(p.numel() for p in trainable)
    print(f'  trainable params: {n_params:,}')
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idxs = rng.integers(0, len(corpus), size=args.batch)
        batch = [corpus[i] for i in idxs]
        loss = _infonce_step(model, tok, batch, args.device, args.nce_temp)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            print(f'  step {step:5d}  loss={loss.item():.4f}  '
                  f'gate={float(model.target_head.gate.item()):+.4f}  '
                  f'{elapsed:.0f}s', flush=True)

    # Save with updated config flag.
    torch.save({
        'config': cfg,
        'state': {k: v.detach().cpu() for k, v in model.state_dict().items()
                  if 'backbone' not in k},
    }, args.out)
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
