"""
Train the grounded RV32I control head.

Data: each verified program is re-executed in Unicorn one cycle at a
time, mirroring the inference-time emit-and-write flow — memory is NOT
pre-loaded. At each cycle we record the live state (before the step) and
the ground-truth instruction at PC, then write that instruction and
step. The translation cache is invalidated after every write.

Training: scheduled-sampling ε on the GRU's previous-instruction input,
teacher-forced state trajectory. With ``--no-freeze-backbone`` the LoRA
adapters on the backbone get gradient signal and per-batch re-encoding
is used; otherwise the backbone hidden states are pre-cached once.

Usage:
    uv run train                          # frozen backbone (default)
    uv run train --no-freeze-backbone     # train LoRA on backbone too
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_lora import (
    BACKBONE_ID, CTRL_DIM, FIELD_CLASSES, FIELD_EMBED_DIMS, FIELD_NAMES,
    MAX_INSTR_TOKENS, N_STATE_TOKENS, N_XATTN_LAYERS, PREV_OP_DIM,
    LoraReflex, build_backbone, code_region_halt_fill, extract_state,
    split_fields,
)
from .programs import SRC_OFFSET, generate_tasks
from .riscv import DATA_BASE, HALT_INSTR, PROGRAM_START, Rv32i

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


# ── Dataset ───────────────────────────────────────────────────────────
def collect_state_sequences(tasks, max_cycles: int = 5000):
    """For each (instr_text, program_bytes) step through Unicorn mirroring
    the inference-time emit-and-write flow. Returns a list of
    ``(instr_text, [(state[65], instr_word_at_pc), ...])``."""
    sequences = []
    halt_fill = code_region_halt_fill()
    seed = b''.join(int(i).to_bytes(4, 'little') for i in range(1, 9))
    for instr_text, prog in tasks:
        prog_map = {PROGRAM_START + i: int.from_bytes(prog[i:i+4], 'little')
                    for i in range(0, len(prog), 4)}
        cpu = Rv32i()
        cpu.uc.mem_write(PROGRAM_START, halt_fill)
        cpu.uc.mem_write(DATA_BASE + SRC_OFFSET, seed)

        seq = []
        for _ in range(max_cycles):
            pc = cpu.pc
            if pc not in prog_map:                  # walked off verified trace
                break
            state = extract_state(cpu)              # BEFORE writing — matches inference
            instr_w = prog_map[pc]
            seq.append((state, instr_w))
            if instr_w == HALT_INSTR:
                break
            cpu.uc.mem_write(pc, int(instr_w).to_bytes(4, 'little'))
            cpu.uc.ctl_remove_cache(pc, pc + 4)     # invalidate stale TB
            try:
                cpu.step()
            except Exception:
                break
        sequences.append((instr_text, seq))
    return sequences


# ── Training ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--freeze-backbone', action='store_true', default=False)
    ap.add_argument('--no-freeze-backbone', dest='freeze_backbone',
                    action='store_false')
    ap.add_argument('--ckpt', default='reflex_lora.pt')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    print(f'device: {device}', flush=True)

    backbone, tok, HIDDEN = build_backbone(
        use_lora=not args.freeze_backbone, dtype=torch.float16)
    backbone = backbone.to(device)
    if hasattr(backbone, 'print_trainable_parameters'):
        backbone.print_trainable_parameters()
    print(f'backbone hidden={HIDDEN}', flush=True)

    # ── Data ──
    print('generating tasks...', flush=True)
    tasks = generate_tasks()
    print(f'{len(tasks)} tasks. stepping through Unicorn...', flush=True)
    t0 = time.time()
    seqs = collect_state_sequences(tasks)
    seqs = [(t, s) for t, s in seqs if len(s) > 0]
    print(f'collected state sequences in {time.time()-t0:.1f}s', flush=True)
    lens = [len(s) for _, s in seqs]
    print(f'sample count={len(seqs)}, step-count min/avg/max = '
          f'{min(lens)}/{sum(lens)/len(lens):.1f}/{max(lens)}', flush=True)

    max_steps = max(lens)
    N = len(seqs)

    # Tokenize unique instructions once.
    uniq_instrs = sorted({t for t, _ in seqs})
    enc = tok(uniq_instrs, padding='max_length', truncation=True,
              max_length=MAX_INSTR_TOKENS, return_tensors='pt')
    instr_idx = {s: i for i, s in enumerate(uniq_instrs)}

    STATE = np.zeros((N, max_steps, N_STATE_TOKENS), dtype=np.int64)
    TGT = np.zeros((6, N, max_steps), dtype=np.int64)
    SMASK = np.zeros((N, max_steps), dtype=np.float32)
    INSTR_IDX = np.zeros((N,), dtype=np.int64)
    for i, (txt, seq) in enumerate(seqs):
        INSTR_IDX[i] = instr_idx[txt]
        for t, (st, w) in enumerate(seq):
            STATE[i, t] = st.astype(np.int64)
            for j, f in enumerate(split_fields(int(w))):
                TGT[j, i, t] = f
            SMASK[i, t] = 1.0

    STATE_d = torch.from_numpy(STATE).to(device)
    TGT_d = torch.from_numpy(TGT).to(device)
    SMASK_d = torch.from_numpy(SMASK).to(device)
    INSTR_IDX_d = torch.from_numpy(INSTR_IDX).to(device)
    INSTR_IDS_d = enc.input_ids.to(device)
    INSTR_MASK_d = enc.attention_mask.to(device)

    # ── Model + (optional) pre-cache ──
    model = LoraReflex(backbone, HIDDEN,
                           freeze_backbone=args.freeze_backbone).to(device)
    if args.freeze_backbone:
        print('pre-encoding unique instructions (frozen backbone)...',
              flush=True)
        cache_h = torch.empty(len(uniq_instrs), MAX_INSTR_TOKENS, HIDDEN,
                              dtype=torch.float32, device=device)
        with torch.no_grad():
            for i in range(0, len(uniq_instrs), 16):
                cache_h[i:i+16] = model.encode_instruction(
                    INSTR_IDS_d[i:i+16], INSTR_MASK_d[i:i+16])
        print(f'cached {len(uniq_instrs)} instruction encodings', flush=True)
    else:
        cache_h = None                              # re-encode per step for LoRA

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    print(f'trainable params: {n_tr/1e6:.2f}M', flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    def linear_eps(step, total, start=1.0, end=0.1):
        return start + (end - start) * min(step / total, 1.0)

    def run_batch(idx, eps):
        B = len(idx)
        state = STATE_d[idx]
        smask = SMASK_d[idx]
        tgts = [TGT_d[j, idx] for j in range(6)]
        instr_m = INSTR_MASK_d[INSTR_IDX_d[idx]]
        if cache_h is not None:
            instr_h = cache_h[INSTR_IDX_d[idx]]
        else:
            instr_h = model.encode_instruction(
                INSTR_IDS_d[INSTR_IDX_d[idx]], instr_m)
        batch_max = int(smask.sum(dim=1).max().item())

        h_state = torch.zeros(B, model.ctrl_dim, device=device)
        prev_fields = [torch.zeros(B, dtype=torch.long, device=device)
                       for _ in FIELD_CLASSES]

        total_loss = 0.0
        pfc = [0] * 6; fc = 0; total = 0
        for t in range(batch_max):
            logits, h_state = model.decode_step(
                instr_h, instr_m, state[:, t], prev_fields, h_state)
            m = smask[:, t]
            step_loss = 0
            preds = []; full = None
            for j, lg in enumerate(logits):
                tj = tgts[j][:, t]
                ce = F.cross_entropy(lg, tj, reduction='none') * m
                step_loss = step_loss + ce.sum()
                pr = lg.argmax(-1)
                preds.append(pr)
                correct = ((pr == tj).float() * m)
                pfc[j] += correct.sum().item()
                full = correct if full is None else full * (pr == tj).float()
            fc += (full * m).sum().item()
            total += m.sum().item()
            total_loss = total_loss + step_loss

            use_gt = torch.rand(B, device=device) < eps
            prev_fields = [torch.where(use_gt, tgts[j][:, t], preds[j].detach())
                           for j in range(6)]

        loss = total_loss / max(total * 6, 1)
        return loss, pfc, fc, total

    print(f'training {args.steps} steps batch={args.batch}', flush=True)
    best_inf = 0.0
    t_start = time.time()
    for step in range(args.steps):
        model.train()
        eps = linear_eps(step, args.steps)
        idx = torch.tensor(np.random.choice(N, args.batch, replace=False),
                           device=device)
        opt.zero_grad()
        with torch.autocast('cuda', dtype=torch.float16,
                            enabled=(device == 'cuda')):
            loss, _, _, _ = run_batch(idx, eps)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt); scaler.update(); sched.step()

        if step % 50 == 0:
            model.eval()
            with torch.no_grad(), torch.autocast(
                    'cuda', dtype=torch.float16, enabled=(device == 'cuda')):
                eidx = torch.tensor(
                    np.random.choice(N, min(256, N), replace=False),
                    device=device)
                _, pfc, fc, tot = run_batch(eidx, 0.0)
            pf = [c / max(tot, 1) for c in pfc]
            inf = fc / max(tot, 1)
            print(f'step {step:5d}  eps={eps:.2f}  loss={loss.item():.4f}  '
                  f'inf={inf:.1%}  [' +
                  ' '.join(f"{n}:{a:.2f}" for n, a in zip(FIELD_NAMES, pf)) +
                  f']  {time.time()-t_start:.0f}s', flush=True)
            if inf > best_inf and step > 0:
                best_inf = inf
                torch.save({
                    'state': model.state_dict(),
                    'config': {
                        'backbone_id': BACKBONE_ID,
                        'hidden': HIDDEN,
                        'ctrl_dim': model.ctrl_dim,
                        'n_xattn': N_XATTN_LAYERS,
                        'freeze_backbone': args.freeze_backbone,
                        'max_instr_tokens': MAX_INSTR_TOKENS,
                    },
                }, args.ckpt)

    torch.save({
        'state': model.state_dict(),
        'config': {
            'backbone_id': BACKBONE_ID,
            'hidden': HIDDEN,
            'ctrl_dim': model.ctrl_dim,
            'n_xattn': N_XATTN_LAYERS,
            'freeze_backbone': args.freeze_backbone,
            'max_instr_tokens': MAX_INSTR_TOKENS,
        },
    }, args.ckpt.replace('.pt', '_final.pt'))
    print(f'done. best inf={best_inf:.1%}', flush=True)


if __name__ == '__main__':
    main()
