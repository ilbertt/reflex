"""
Train the Flamingo-style grounded RV32I controller.

Data: each verified program is re-executed in Unicorn one cycle at a
time, mirroring the inference-time emit-and-write flow — memory is NOT
pre-loaded. At each cycle we record the live state (before the step)
and the ground-truth instruction at PC, then write that instruction and
step. The translation cache is invalidated after every write.

Training: the previous emitted instruction is visible through the
memory window in the state vector, so there is no autoregressive head.
We flatten the collected per-cycle triples into a pool, map each
target word to its row in the unique-instruction table, and train the
JEPA head with InfoNCE (cosine-sim logits over every row of the
table). The backbone is frozen; only the state encoder, cross-attn
adapters, head, and instruction-embedding table train.

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
    BACKBONE_ID, EMBED_DIM, GroundedReflex, build_backbone,
    code_region_halt_fill, extract_state, render_prompt,
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
    ap.add_argument('--embed-dim', type=int, default=EMBED_DIM,
                    help='Dimension of the JEPA instruction embedding.')
    ap.add_argument('--nce-temp', type=float, default=0.07,
                    help='InfoNCE temperature (cosine-sim / τ).')
    ap.add_argument('--save-every', type=int, default=0,
                    help='Also save a step-tagged ckpt every N steps '
                    '(0 disables). Independent of the rolling-best save.')
    ap.add_argument('--sample-pool', type=int, default=300_000,
                    help='Subsample the flat cycle pool to this size, '
                    'balanced across families. 0 disables.')
    ap.add_argument('--use-target-head', action='store_true',
                    help='Path B: construct the model with the prompt-'
                    'cross-attention TargetAwareHead. Gate-zero-init — '
                    'at step 0 the head is a no-op identity.')
    ap.add_argument('--freeze-except-target-head', action='store_true',
                    help='Path B retrain: freeze the adapters + '
                    'state_encoder + kv_norm so only the target head '
                    'trains. Implies --use-target-head.')
    ap.add_argument('--also-train-canonical-head', action='store_true',
                    help='With --freeze-except-target-head, also '
                    'unfreeze head_mlp + embed_head + instr_table.')
    args = ap.parse_args()
    if args.freeze_except_target_head:
        args.use_target_head = True

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

    # Build the JEPA instruction table: one row per unique 32-bit word
    # seen anywhere in the flattened cycle pool.
    words = np.array(flat_word, dtype=np.uint32)
    unique_words, target_idx = np.unique(words, return_inverse=True)
    num_instrs = len(unique_words)
    print(f'unique instruction words in training pool: {num_instrs}',
          flush=True)

    INSTR_IDX_d = torch.from_numpy(np.array(flat_instr, dtype=np.int64)).to(device)
    STATE_d = torch.from_numpy(np.stack(flat_state)).to(device)
    TARGET_d = torch.from_numpy(target_idx.astype(np.int64)).to(device)
    INSTR_IDS_d = enc.input_ids.to(device)
    INSTR_MASK_d = enc.attention_mask.to(device)

    # ── Model ──
    model = GroundedReflex(backbone, HIDDEN,
                           num_instrs=num_instrs,
                           inject_every=args.inject_every,
                           freeze_backbone=True,
                           adapter_mlp_ratio=args.adapter_mlp_ratio,
                           embed_dim=args.embed_dim,
                           use_target_head=args.use_target_head).to(device)
    # Seed the decode buffer so nearest-neighbour → real 32-bit word.
    with torch.no_grad():
        model.instr_words.copy_(
            torch.from_numpy(unique_words.astype(np.int64)).to(device))
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

    if args.freeze_except_target_head:
        # Freeze everything the canonical retrain normally leaves
        # trainable (adapters, state_encoder, kv_norm) so only the
        # target head's new cross-attention pathway trains. Optionally
        # also unfreeze head_mlp + embed_head + instr_table when the
        # downstream head needs to learn to consume the new signal.
        if model.target_head is None:
            raise SystemExit(
                'target_head is None; did you build with --use-target-head?')
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.target_head.parameters():
            p.requires_grad_(True)
        if args.also_train_canonical_head:
            for mod_name in ('head_mlp', 'embed_head', 'instr_table'):
                for p in getattr(model, mod_name).parameters():
                    p.requires_grad_(True)
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        trainable_cnt = sum(1 for p in model.parameters() if p.requires_grad)
        print(f'Path-B freeze applied: {trainable_cnt} trainable tensors, '
              f'{frozen} frozen', flush=True)

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

    tau = args.nce_temp

    def run_batch(idx):
        B = len(idx)
        state = STATE_d[idx]
        tgts = TARGET_d[idx]
        ii = INSTR_IDX_d[idx]
        ids = INSTR_IDS_d[ii]
        mask = INSTR_MASK_d[ii]
        with torch.autocast('cuda', dtype=torch.bfloat16,
                            enabled=(device == 'cuda')):
            pred = model(ids, mask, state)
        # InfoNCE over the full instruction codebook. Cosine similarity
        # puts pred and every row on the unit sphere; temperature τ
        # sharpens the contrast.
        sim = model.table_similarity(pred) / tau           # [B, num_instrs]
        loss = F.cross_entropy(sim, tgts)
        pred_idx = sim.argmax(dim=-1)
        top1 = (pred_idx == tgts).float().sum().item()
        top5 = 0.0
        if sim.size(1) >= 5:
            top5 = (sim.topk(5, dim=-1).indices ==
                    tgts.unsqueeze(1)).any(dim=-1).float().sum().item()
        return loss, top1, top5, B

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

    def ckpt_config():
        return {
            'backbone_id': args.backbone_id,
            'hidden': HIDDEN,
            'inject_every': args.inject_every,
            'adapter_mlp_ratio': args.adapter_mlp_ratio,
            'max_instr_tokens': args.max_instr_tokens,
            'embed_dim': args.embed_dim,
            'num_instrs': num_instrs,
            'chat_template': True,
            'context_prefix': False,
            'use_target_head': args.use_target_head,
        }

    print(f'training {args.steps} steps batch={args.batch}  '
          f'table={num_instrs}  embed_dim={args.embed_dim}', flush=True)
    best_top1 = 0.0
    t_start = time.time()
    for step in range(args.steps):
        model.train()
        idx = torch.tensor(np.random.choice(N, args.batch, replace=False),
                           device=device)
        opt.zero_grad()
        loss, _, _, _ = run_batch(idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(new_params, 1.0)
        opt.step(); sched.step()

        if step % 50 == 0 and step % 500 != 0:
            print(f'step {step:5d}  loss={loss.item():.4f}  '
                  f'{time.time()-t_start:.0f}s', flush=True)

        # Full eval tick: 512-sample forward + probe + ckpt.
        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                eidx = torch.tensor(
                    np.random.choice(N, min(512, N), replace=False),
                    device=device)
                _, top1, top5, tot = run_batch(eidx)
            a1 = top1 / max(tot, 1)
            a5 = top5 / max(tot, 1)
            print(f'step {step:5d}  loss={loss.item():.4f}  '
                  f'top1={a1:.1%}  top5={a5:.1%}  '
                  f'{time.time()-t_start:.0f}s', flush=True)
            if probe_prompt is not None:
                do_probe()
            if a1 > best_top1 and step > 0:
                best_top1 = a1
                torch.save({'state': trainable_state_dict(),
                            'config': ckpt_config()}, args.ckpt)

        # Periodic step-tagged save alongside the rolling-best save.
        # Tag by absolute step so resumed runs extend the series instead
        # of overwriting tags from prior runs.
        if args.save_every and step > 0 and step % args.save_every == 0:
            abs_step = step + args.init_step
            tagged = args.ckpt.replace('.pt', f'_step{abs_step}.pt')
            torch.save({'state': trainable_state_dict(),
                        'config': ckpt_config()}, tagged)
            print(f'  [saved {tagged}]', flush=True)

    torch.save({'state': trainable_state_dict(),
                'config': ckpt_config()},
               args.ckpt.replace('.pt', '_final.pt'))
    print(f'done. best top1={best_top1:.1%}', flush=True)


if __name__ == '__main__':
    main()
