"""
Train the Flamingo-style grounded RV32I controller.

Data: each verified program is re-executed in Unicorn one cycle at a
time, mirroring the inference-time emit-and-write flow — memory is NOT
pre-loaded. At each cycle we record the live state (before the step)
and the ground-truth instruction at PC, then write that instruction and
step. The translation cache is invalidated after every write.

Training: the previous emitted instruction is already visible to the
model through the memory window in the state vector, so there is no
autoregressive head — we flatten the collected per-cycle (instruction,
state, target) triples into a pool and train with uniform random
mini-batches against the 32 independent instruction-bit heads. The
backbone is frozen; only the state encoder, cross-attn adapters, and
head train. Mixed bf16 autocast, standard AdamW, field-weighted BCE
(rs2 bits × 3 to counter polysemy). Default config tuned to fit
Qwen2.5-Coder-7B-Instruct on A100 80GB at batch=64.

Usage:
    uv run train
    uv run train --backbone-id Qwen/Qwen3-4B --batch 16
"""
import argparse
import hashlib
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import (
    BACKBONE_ID, FIELD_NAMES, N_INSTR_BITS, N_STATE_TOKENS, GroundedReflex,
    build_backbone, code_region_halt_fill, extract_state, render_prompt,
)
from .programs import SRC_OFFSET, load_tasks
from .riscv import DATA_BASE, HALT_INSTR, PROGRAM_START, Rv32i

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


# ── Dataset ───────────────────────────────────────────────────────────
def collect_state_sequences(tasks, max_cycles: int = 5000):
    """For each (family, instr_text, program_bytes) step through Unicorn
    mirroring the inference-time emit-and-write flow. Returns a list of
    ``(family, instr_text, [(state[65], instr_word_at_pc), ...])``."""
    sequences = []
    halt_fill = code_region_halt_fill()
    seed = b''.join(int(i).to_bytes(4, 'little') for i in range(1, 9))
    for family, instr_text, prog in tasks:
        prog_map = {PROGRAM_START + i: int.from_bytes(prog[i:i+4], 'little')
                    for i in range(0, len(prog), 4)}
        cpu = Rv32i()
        cpu.uc.mem_write(PROGRAM_START, halt_fill)
        cpu.uc.mem_write(DATA_BASE + SRC_OFFSET, seed)

        seq = []
        for _ in range(max_cycles):
            pc = cpu.pc
            if pc not in prog_map:
                break
            state = extract_state(cpu)
            instr_w = prog_map[pc]
            seq.append((state, instr_w))
            if instr_w == HALT_INSTR:
                break
            cpu.uc.mem_write(pc, int(instr_w).to_bytes(4, 'little'))
            cpu.uc.ctl_remove_cache(pc, pc + 4)
            try:
                cpu.step()
            except Exception:
                break
        sequences.append((family, instr_text, seq))
    return sequences


# ── Training ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone-id', default=BACKBONE_ID)
    ap.add_argument('--steps', type=int, default=15_000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--ckpt', default='reflex.pt')
    ap.add_argument('--resume', default=None,
                    help='Load adapter/head weights from this checkpoint.')
    ap.add_argument('--init-step', type=int, default=0,
                    help='Advance the cosine LR scheduler this many steps on '
                    'startup so a --resume continues the schedule smoothly '
                    'instead of re-hitting peak LR.')
    ap.add_argument('--probe', default=None,
                    help='Prompt to run end-to-end every eval tick. '
                    'Format: "prompt=value" (checks mem[DATA_BASE]) or '
                    '"prompt@0xADDR=value".')
    ap.add_argument('--inject-every', type=int, default=4,
                    help='Insert a cross-attention adapter every N backbone '
                    'layers. Default 4 → 9 adapters on a 36-layer backbone.')
    ap.add_argument('--adapter-mlp-ratio', type=int, default=4)
    ap.add_argument('--max-instr-tokens', type=int, default=96)
    ap.add_argument('--sample-pool', type=int, default=300_000,
                    help='Subsample the flat cycle pool to this size, '
                    'balanced across families. 0 disables.')
    ap.add_argument('--seq-window', type=int, default=1,
                    help='Latent Recurrence training window. W=1 keeps the '
                    'legacy flat-pool training (prev_hidden=None, 65-token '
                    'K/V). W>1 samples W-cycle windows per batch item and '
                    'unrolls prev_hidden across them so the adapters see '
                    '66-token K/V at train time too, matching inference.')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    print(f'device: {device}  dtype: {dtype}  backbone: {args.backbone_id}',
          flush=True)

    backbone, tok, HIDDEN = build_backbone(args.backbone_id, dtype=dtype)
    backbone = backbone.to(device)
    print(f'backbone hidden={HIDDEN} layers={len(backbone.layers)}', flush=True)

    # ── Data ──
    print('loading tasks from JSON corpus...', flush=True)
    tasks = load_tasks()
    print(f'{len(tasks)} tasks.', flush=True)

    # Deterministic cache of Unicorn-stepped state sequences.
    cache_key = hashlib.sha256(
        b''.join(f'{f}\0{t}\0'.encode() + b for f, t, b in tasks)
    ).hexdigest()[:16]
    cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
    cache_path = os.path.join(cache_dir, f'state_seqs_{cache_key}.pkl')
    if os.path.exists(cache_path):
        t0 = time.time()
        with open(cache_path, 'rb') as fh:
            seqs = pickle.load(fh)
        print(f'loaded cached state sequences ({len(seqs)}) from '
              f'{cache_path} in {time.time()-t0:.1f}s', flush=True)
    else:
        print('stepping through Unicorn...', flush=True)
        t0 = time.time()
        seqs = collect_state_sequences(tasks)
        seqs = [(f, t, s) for f, t, s in seqs if len(s) > 0]
        print(f'collected state sequences in {time.time()-t0:.1f}s', flush=True)
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, 'wb') as fh:
            pickle.dump(seqs, fh)
        print(f'cached to {cache_path}', flush=True)

    lens = [len(s) for _, _, s in seqs]
    print(f'programs={len(seqs)}, step-count min/avg/max = '
          f'{min(lens)}/{sum(lens)/len(lens):.1f}/{max(lens)}', flush=True)

    # Tokenize unique instructions once using the backbone's chat template.
    uniq_instrs = sorted({t for _, t, _ in seqs})
    tok_inputs = [render_prompt(tok, s, use_chat_template=True)
                  for s in uniq_instrs]
    print(f'chat_template sample:\n    {tok_inputs[0][:200]!r}...', flush=True)
    enc = tok(tok_inputs, padding='max_length', truncation=True,
              max_length=args.max_instr_tokens, return_tensors='pt')
    instr_idx = {s: i for i, s in enumerate(uniq_instrs)}

    # Flatten every (state, instr) cycle, tagged with its program family.
    flat_instr, flat_state, flat_word, flat_family = [], [], [], []
    for fam, txt, seq in seqs:
        ii = instr_idx[txt]
        for st, w in seq:
            flat_instr.append(ii)
            flat_state.append(st.astype(np.int64))
            flat_word.append(int(w) & 0xFFFFFFFF)
            flat_family.append(fam)
    N_full = len(flat_instr)
    print(f'flat cycle-pool size = {N_full}', flush=True)

    if args.sample_pool and args.sample_pool < N_full:
        import collections
        by_fam = collections.defaultdict(list)
        for i, f in enumerate(flat_family):
            by_fam[f].append(i)
        per_fam = max(1, args.sample_pool // len(by_fam))
        rng = np.random.default_rng(0)
        keep_idx = []
        for f, idxs in by_fam.items():
            take = min(len(idxs), per_fam)
            keep_idx.extend(rng.choice(idxs, size=take, replace=False).tolist())
        keep_idx = np.array(sorted(keep_idx), dtype=np.int64)
        flat_instr = [flat_instr[i] for i in keep_idx]
        flat_state = [flat_state[i] for i in keep_idx]
        flat_word = [flat_word[i] for i in keep_idx]
        print(f'balanced subsample: {len(by_fam)} families × {per_fam} each '
              f'→ pool size = {len(flat_instr)}', flush=True)
    N = len(flat_instr)

    # 32-bit instruction → LSB-first bit vector.
    words = np.array(flat_word, dtype=np.uint32)
    bits = np.zeros((N, N_INSTR_BITS), dtype=np.float32)
    for i in range(N_INSTR_BITS):
        bits[:, i] = ((words >> i) & 1).astype(np.float32)

    INSTR_IDX_d = torch.from_numpy(np.array(flat_instr, dtype=np.int64)).to(device)
    STATE_d = torch.from_numpy(np.stack(flat_state)).to(device)
    BITS_d = torch.from_numpy(bits).to(device)
    INSTR_IDS_d = enc.input_ids.to(device)
    INSTR_MASK_d = enc.attention_mask.to(device)

    # ── Latent Recurrence window pool ──
    # W-cycle sliding windows over each program. Used when --seq-window > 1
    # so training exercises the 66-token K/V path (prev_hidden threaded
    # across cycles) the same way inference does.
    W = max(1, args.seq_window)
    if W > 1:
        win_instr, win_state, win_word, win_mask, win_family = [], [], [], [], []
        for fam, txt, seq in seqs:
            ii = instr_idx[txt]
            T = len(seq)
            if T == 0:
                continue
            for start in range(T):
                end = min(start + W, T)
                states_w = np.zeros((W, N_STATE_TOKENS), dtype=np.int64)
                words_w = np.zeros(W, dtype=np.uint32)
                mask_w = np.zeros(W, dtype=np.float32)
                for j in range(end - start):
                    states_w[j] = seq[start + j][0].astype(np.int64)
                    words_w[j] = int(seq[start + j][1]) & 0xFFFFFFFF
                    mask_w[j] = 1.0
                win_instr.append(ii)
                win_state.append(states_w)
                win_word.append(words_w)
                win_mask.append(mask_w)
                win_family.append(fam)
        Nw_full = len(win_instr)
        print(f'window-pool size (W={W}) = {Nw_full}', flush=True)

        if args.sample_pool and args.sample_pool < Nw_full:
            import collections as _coll
            by_fam = _coll.defaultdict(list)
            for i, f in enumerate(win_family):
                by_fam[f].append(i)
            per_fam = max(1, args.sample_pool // len(by_fam))
            rng = np.random.default_rng(0)
            keep_idx = []
            for f, idxs in by_fam.items():
                take = min(len(idxs), per_fam)
                keep_idx.extend(rng.choice(idxs, size=take, replace=False).tolist())
            keep_idx = np.array(sorted(keep_idx), dtype=np.int64)
            win_instr = [win_instr[i] for i in keep_idx]
            win_state = [win_state[i] for i in keep_idx]
            win_word = [win_word[i] for i in keep_idx]
            win_mask = [win_mask[i] for i in keep_idx]
            print(f'balanced window subsample → {len(win_instr)}', flush=True)

        Nw = len(win_instr)
        win_bits = np.zeros((Nw, W, N_INSTR_BITS), dtype=np.float32)
        win_words_np = np.stack(win_word)  # [Nw, W]
        for i in range(N_INSTR_BITS):
            win_bits[:, :, i] = ((win_words_np >> i) & 1).astype(np.float32)

        WIN_INSTR_IDX_d = torch.from_numpy(np.array(win_instr, dtype=np.int64)).to(device)
        WIN_STATE_d = torch.from_numpy(np.stack(win_state)).to(device)   # [Nw, W, 65]
        WIN_BITS_d = torch.from_numpy(win_bits).to(device)               # [Nw, W, 32]
        WIN_MASK_d = torch.from_numpy(np.stack(win_mask)).to(device)     # [Nw, W]
    else:
        Nw = 0

    # ── Model ──
    model = GroundedReflex(backbone, HIDDEN,
                           inject_every=args.inject_every,
                           freeze_backbone=True,
                           adapter_mlp_ratio=args.adapter_mlp_ratio).to(device)
    backbone.eval()

    if args.resume:
        _ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(_ck['state'], strict=False)
        del _ck
        import gc; gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        ne_missing = [m for m in missing if not m.startswith('backbone.')]
        print(f'resumed from {args.resume}; new-module missing={len(ne_missing)} '
              f'unexpected={len(unexpected)}', flush=True)

    new_params = [p for p in model.parameters() if p.requires_grad]
    n_new = sum(p.numel() for p in new_params) / 1e6
    print(f'trainable params: {n_new:.2f}M', flush=True)

    opt = torch.optim.AdamW([{'params': new_params, 'lr': args.lr}],
                            weight_decay=0.01)
    # Include prior steps in T_max so the cosine continues smoothly from
    # wherever a previous run stopped instead of restarting from peak LR.
    sched_total = args.steps + args.init_step
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=sched_total, eta_min=1e-6)
    for _ in range(args.init_step):
        sched.step()
    if args.init_step:
        print(f'cosine resumed: step {args.init_step}/{sched_total} → '
              f'lr={sched.get_last_lr()[0]:.2e}', flush=True)

    def trainable_state_dict():
        full = model.state_dict()
        keep = {n for n, p in model.named_parameters() if p.requires_grad}
        for n, _ in model.named_buffers():
            if not n.startswith('backbone.'):
                keep.add(n)
        return {k: v for k, v in full.items()
                if k in keep or not k.startswith('backbone.')}

    FIELD_BIT_SLICES = (
        (0, 7), (7, 12), (12, 15), (15, 20), (20, 25), (25, 32),
    )

    # Field-weighted BCE. rs2 (bits 20-24) is polysemous (register on R-type
    # but imm[4:0] on I/U/J-type) and consistently plateaus at ~0.89 while
    # every other field converges to 1.00. Weighting its 5 bits at 3×
    # roughly doubles rs2's gradient share (15/42 ≈ 36% vs 15.6% uniform),
    # so it stops being drowned out by the already-solved bits.
    bit_weights = torch.ones(N_INSTR_BITS, device=device)
    bit_weights[20:25] = 5.0

    def run_batch(idx):
        B = len(idx)
        state = STATE_d[idx]
        tgts = BITS_d[idx]
        ii = INSTR_IDX_d[idx]
        ids = INSTR_IDS_d[ii]
        mask = INSTR_MASK_d[ii]
        with torch.autocast('cuda', dtype=torch.bfloat16,
                            enabled=(device == 'cuda')):
            # Flat random-batch training: no sequential context is available,
            # so prev_hidden stays None. Latent Recurrence kicks in at eval
            # time where run_grounded threads the pooled hidden state across
            # Unicorn cycles.
            logits, _ = model(ids, mask, state, None)
        loss = F.binary_cross_entropy_with_logits(
            logits.float(), tgts, weight=bit_weights)
        pred_bits = (logits > 0).long()
        bit_ok = (pred_bits == tgts.long())
        pfc = [bit_ok[:, lo:hi].all(dim=1).float().sum().item()
               for lo, hi in FIELD_BIT_SLICES]
        fc = bit_ok.all(dim=1).float().sum().item()
        return loss, pfc, fc, B

    def run_window_batch(idx):
        """Unroll W cycles per sample with prev_hidden threaded through.

        Each batch item is a W-cycle slice from one program. We detach
        prev_hidden between steps (TBPTT=1), matching inference.
        """
        B = len(idx)
        state_w = WIN_STATE_d[idx]     # [B, W, 65]
        tgts_w = WIN_BITS_d[idx]       # [B, W, 32]
        mask_w = WIN_MASK_d[idx]       # [B, W]
        ii = WIN_INSTR_IDX_d[idx]
        ids = INSTR_IDS_d[ii]
        mask = INSTR_MASK_d[ii]
        prev_hidden = None
        loss_sum = torch.zeros((), device=device)
        mask_sum = torch.zeros((), device=device)
        pfc_accum = [0.0] * len(FIELD_BIT_SLICES)
        fc_accum = 0.0
        B_valid = 0.0
        with torch.autocast('cuda', dtype=torch.bfloat16,
                            enabled=(device == 'cuda')):
            for w in range(W):
                logits, prev_hidden = model(ids, mask, state_w[:, w],
                                            prev_hidden)
                step_mask = mask_w[:, w]
                if step_mask.sum() == 0:
                    continue
                bce = F.binary_cross_entropy_with_logits(
                    logits.float(), tgts_w[:, w],
                    weight=bit_weights, reduction='none').mean(dim=-1)  # [B]
                loss_sum = loss_sum + (bce * step_mask).sum()
                mask_sum = mask_sum + step_mask.sum()
                pred_bits = (logits > 0).long()
                bit_ok = (pred_bits == tgts_w[:, w].long())
                for i, (lo, hi) in enumerate(FIELD_BIT_SLICES):
                    pfc_accum[i] += (bit_ok[:, lo:hi].all(dim=1).float()
                                     * step_mask).sum().item()
                fc_accum += (bit_ok.all(dim=1).float() * step_mask).sum().item()
                B_valid += step_mask.sum().item()
        loss = loss_sum / mask_sum.clamp(min=1.0)
        return loss, pfc_accum, fc_accum, int(B_valid)

    # Optional end-to-end probe. Format: "prompt=value" or "prompt@0xADDR=value".
    probe_prompt = probe_expected = probe_addr = None
    if args.probe:
        prompt_part, _, probe_expected = args.probe.rpartition('=')
        probe_expected = int(probe_expected, 0)
        if '@' in prompt_part:
            probe_prompt, _, addr_str = prompt_part.rpartition('@')
            probe_addr = int(addr_str, 0)
        else:
            probe_prompt = prompt_part
            probe_addr = DATA_BASE
        from .demo import run_grounded as _probe_run
        print(f'probe: {probe_prompt!r} expects mem[0x{probe_addr:x}]='
              f'{probe_expected}', flush=True)

    def do_probe():
        model.eval()
        cpu, emitted, halted, err = _probe_run(
            model, tok, probe_prompt, device, max_cycles=200,
            max_instr_tokens=args.max_instr_tokens)
        got = cpu.mem_word(probe_addr)
        mark = '✓' if halted and not err and got == probe_expected else '✗'
        print(f'  probe {mark} ops={len(emitted)} halted={halted} '
              f'mem[0x{probe_addr:x}]={got} (want {probe_expected})', flush=True)

    pool_size = Nw if W > 1 else N
    batch_fn = run_window_batch if W > 1 else run_batch
    mode_tag = f'seq-window (W={W})' if W > 1 else 'flat-pool'
    print(f'training {args.steps} steps batch={args.batch} mode={mode_tag}',
          flush=True)
    best_inf = 0.0
    t_start = time.time()
    for step in range(args.steps):
        model.train()
        idx = torch.tensor(
            np.random.choice(pool_size, args.batch, replace=False),
            device=device)
        opt.zero_grad()
        loss, _, _, _ = batch_fn(idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(new_params, 1.0)
        opt.step(); sched.step()

        # Fast tick: just the training-batch loss (no extra forward needed).
        if step % 50 == 0 and step % 500 != 0:
            print(f'step {step:5d}  loss={loss.item():.4f}  '
                  f'{time.time()-t_start:.0f}s', flush=True)

        # Full eval tick: 512-sample forward for field metrics + probe + ckpt.
        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                eidx = torch.tensor(
                    np.random.choice(pool_size, min(512, pool_size),
                                     replace=False),
                    device=device)
                _, pfc, fc, tot = batch_fn(eidx)
            pf = [c / max(tot, 1) for c in pfc]
            inf = fc / max(tot, 1)
            print(f'step {step:5d}  loss={loss.item():.4f}  inf={inf:.1%}  [' +
                  ' '.join(f"{n}:{a:.2f}" for n, a in zip(FIELD_NAMES, pf)) +
                  f']  {time.time()-t_start:.0f}s', flush=True)
            if probe_prompt is not None:
                do_probe()
            if inf > best_inf and step > 0:
                best_inf = inf
                torch.save({
                    'state': trainable_state_dict(),
                    'config': {
                        'backbone_id': args.backbone_id,
                        'hidden': HIDDEN,
                        'inject_every': args.inject_every,
                        'adapter_mlp_ratio': args.adapter_mlp_ratio,
                        'max_instr_tokens': args.max_instr_tokens,
                        'seq_window': W,
                        'chat_template': True,
                        'context_prefix': False,
                    },
                }, args.ckpt)

    torch.save({
        'state': trainable_state_dict(),
        'config': {
            'backbone_id': args.backbone_id,
            'hidden': HIDDEN,
            'inject_every': args.inject_every,
            'adapter_mlp_ratio': args.adapter_mlp_ratio,
            'max_instr_tokens': args.max_instr_tokens,
            'seq_window': W,
        },
    }, args.ckpt.replace('.pt', '_final.pt'))
    print(f'done. best inf={best_inf:.1%}', flush=True)


if __name__ == '__main__':
    main()
