"""
Train the Flamingo-style grounded RV32I controller.

Data: each verified program is re-executed in Unicorn one cycle at a
time, mirroring the inference-time emit-and-write flow — memory is NOT
pre-loaded. At each cycle we record the live state (before the step)
and the ground-truth instruction at PC, then write that instruction and
step. The translation cache is invalidated after every write.

Training: because the previous emitted instruction is already visible
to the model through the memory window in the state vector, there is
no autoregressive head to unroll — we flatten the collected per-cycle
(instruction, state, target) triples into a flat pool and train with
uniform random mini-batches. The backbone is fully fine-tuned and the
injected cross-attention adapters are trained from scratch.

Usage:
    uv run train
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import (
    BACKBONE_ID, CONTEXT_PREFIX, FIELD_NAMES, INJECT_EVERY, MAX_INSTR_TOKENS,
    N_INSTR_BITS, N_STATE_TOKENS, GroundedReflex, build_backbone,
    code_region_halt_fill, extract_state, split_fields,
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
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--backbone-lr', type=float, default=2e-5)
    ap.add_argument('--backbone-id', default=BACKBONE_ID)
    ap.add_argument('--freeze-backbone', action='store_true', default=False)
    ap.add_argument('--dtype', default='bf16', choices=('bf16', 'fp16'))
    ap.add_argument('--grad-ckpt', action='store_true', default=False)
    ap.add_argument('--ckpt', default='reflex_grounded.pt')
    ap.add_argument('--resume', default=None,
                    help='Load adapter/head weights from this checkpoint.')
    ap.add_argument('--init-step', type=int, default=0,
                    help='Pre-advance the cosine LR scheduler this many '
                    'steps so the schedule continues smoothly from where a '
                    'prior training run stopped (used with --resume).')
    ap.add_argument('--probe', default=None,
                    help='Prompt to run end-to-end every eval tick '
                    '(format: "prompt=expected_value", e.g. '
                    '"subtract 10 from 25=15"). Logs the emitted answer.')
    ap.add_argument('--context-prefix', action='store_true', default=False,
                    help='Prepend the machine-context prefix to every prompt.')
    ap.add_argument('--max-instr-tokens', type=int, default=MAX_INSTR_TOKENS,
                    help='Tokenizer max_length for the prompt (bump when '
                    'using --context-prefix).')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16}
    dtype = (dtype_map[args.dtype] if device == 'cuda' else torch.float32)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    print(f'device: {device}  dtype: {dtype}  '
          f'backbone: {args.backbone_id}  '
          f'freeze_backbone: {args.freeze_backbone}', flush=True)

    backbone, tok, HIDDEN = build_backbone(args.backbone_id, dtype=dtype)
    backbone = backbone.to(device)
    print(f'backbone hidden={HIDDEN} layers={len(backbone.layers)}', flush=True)

    # ── Data ──
    print('generating tasks...', flush=True)
    tasks = generate_tasks()
    print(f'{len(tasks)} tasks. stepping through Unicorn...', flush=True)
    t0 = time.time()
    seqs = collect_state_sequences(tasks)
    seqs = [(t, s) for t, s in seqs if len(s) > 0]
    print(f'collected state sequences in {time.time()-t0:.1f}s', flush=True)
    lens = [len(s) for _, s in seqs]
    print(f'programs={len(seqs)}, step-count min/avg/max = '
          f'{min(lens)}/{sum(lens)/len(lens):.1f}/{max(lens)}', flush=True)

    # Tokenize unique instructions once.
    uniq_instrs = sorted({t for t, _ in seqs})
    if args.context_prefix:
        tok_inputs = [CONTEXT_PREFIX + s for s in uniq_instrs]
        print(f'context_prefix: ON  max_instr_tokens={args.max_instr_tokens}',
              flush=True)
    else:
        tok_inputs = uniq_instrs
    enc = tok(tok_inputs, padding='max_length', truncation=True,
              max_length=args.max_instr_tokens, return_tensors='pt')
    instr_idx = {s: i for i, s in enumerate(uniq_instrs)}

    # Flatten every (state, instr) cycle across all programs.
    flat_instr = []
    flat_state = []
    flat_word = []
    for txt, seq in seqs:
        ii = instr_idx[txt]
        for st, w in seq:
            flat_instr.append(ii)
            flat_state.append(st.astype(np.int64))
            flat_word.append(int(w) & 0xFFFFFFFF)
    N = len(flat_instr)
    print(f'flat cycle-pool size = {N}', flush=True)

    # Convert each 32-bit instruction to its bit vector [32] (LSB first).
    words = np.array(flat_word, dtype=np.uint32)
    bits = np.zeros((N, N_INSTR_BITS), dtype=np.float32)
    for i in range(N_INSTR_BITS):
        bits[:, i] = ((words >> i) & 1).astype(np.float32)

    INSTR_IDX_d = torch.from_numpy(np.array(flat_instr, dtype=np.int64)).to(device)
    STATE_d = torch.from_numpy(np.stack(flat_state)).to(device)
    BITS_d = torch.from_numpy(bits).to(device)                  # [N, 32]
    WORD_d = torch.from_numpy(words.astype(np.int64)).to(device)  # [N]
    INSTR_IDS_d = enc.input_ids.to(device)
    INSTR_MASK_d = enc.attention_mask.to(device)

    # ── Model ──
    model = GroundedReflex(backbone, HIDDEN,
                           freeze_backbone=args.freeze_backbone).to(device)
    if args.grad_ckpt and hasattr(backbone, 'gradient_checkpointing_enable'):
        backbone.gradient_checkpointing_enable()
        backbone.config.use_cache = False
        print('gradient checkpointing enabled', flush=True)
    if args.freeze_backbone:
        backbone.eval()

    if args.resume:
        # Load to CPU then let load_state_dict copy into the GPU model — keeps
        # the checkpoint dict off GPU so it doesn't double the adapter
        # memory footprint during training start-up.
        _ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(_ck['state'], strict=False)
        del _ck
        import gc; gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        # When resuming with a frozen backbone, all backbone keys are
        # "missing" — we don't care, they come from the pretrained model.
        ne_missing = [m for m in missing if not m.startswith('backbone.')]
        print(f'resumed from {args.resume}; new-module missing={len(ne_missing)} '
              f'unexpected={len(unexpected)}', flush=True)

    bb_params = [p for p in backbone.parameters() if p.requires_grad]
    bb_ids = {id(p) for p in backbone.parameters()}
    new_params = [p for p in model.parameters()
                  if id(p) not in bb_ids and p.requires_grad]
    n_bb_total = sum(p.numel() for p in backbone.parameters()) / 1e6
    n_bb = sum(p.numel() for p in bb_params) / 1e6
    n_new = sum(p.numel() for p in new_params) / 1e6
    print(f'backbone total={n_bb_total:.1f}M  trainable={n_bb:.2f}M  '
          f'new={n_new:.2f}M', flush=True)

    groups = []
    if bb_params:
        groups.append({'params': bb_params, 'lr': args.backbone_lr})
    groups.append({'params': new_params, 'lr': args.lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)

    def trainable_state_dict():
        """State dict containing only parameters with requires_grad — omits
        the frozen backbone so checkpoints stay small."""
        full = model.state_dict()
        keep = {n for n, p in model.named_parameters() if p.requires_grad}
        # Include buffers (LayerNorm running stats etc.) that live outside
        # the backbone — cheap and needed at load time.
        for n, _ in model.named_buffers():
            if not n.startswith('backbone.'):
                keep.add(n)
        return {k: v for k, v in full.items()
                if k in keep or not k.startswith('backbone.')}
    # When resuming, T_max includes the steps that already happened so
    # the cosine continues smoothly instead of restarting from peak LR.
    sched_total = args.steps + args.init_step
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=sched_total, eta_min=1e-6)
    for _ in range(args.init_step):
        sched.step()
    if args.init_step:
        cur_lr = sched.get_last_lr()[0]
        print(f'resumed LR schedule: step {args.init_step}/{sched_total} '
              f'→ lr={cur_lr:.2e}', flush=True)

    # Field bit slices used only for diagnostic per-field accuracy.
    FIELD_BIT_SLICES = (
        (0, 7),    # opcode: bits [0..7)
        (7, 12),   # rd:     bits [7..12)
        (12, 15),  # funct3: bits [12..15)
        (15, 20),  # rs1:    bits [15..20)
        (20, 25),  # rs2:    bits [20..25)
        (25, 32),  # funct7: bits [25..32)
    )

    def run_batch(idx):
        B = len(idx)
        state = STATE_d[idx]
        tgts = BITS_d[idx]                                # [B, 32]
        words = WORD_d[idx]                               # [B]
        ii = INSTR_IDX_d[idx]
        ids = INSTR_IDS_d[ii]
        mask = INSTR_MASK_d[ii]
        logits = model(ids, mask, state)                  # [B, 32]
        loss = F.binary_cross_entropy_with_logits(logits, tgts)

        pred_bits = (logits > 0).long()                   # [B, 32], 0/1
        tgt_bits = tgts.long()
        bit_ok = (pred_bits == tgt_bits)                  # [B, 32]
        pfc = []
        for lo, hi in FIELD_BIT_SLICES:
            pfc.append(bit_ok[:, lo:hi].all(dim=1).float().sum().item())
        fc = bit_ok.all(dim=1).float().sum().item()
        return loss, pfc, fc, B

    # Optional end-to-end probe. Format: "prompt=value" (checks mem[DATA_BASE])
    # or "prompt@0xADDR=value" (checks mem[ADDR]).
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
        print(f'probe: {probe_prompt!r} expects mem[0x{probe_addr:x}]={probe_expected}',
              flush=True)

    def do_probe():
        model.eval()
        cpu, emitted, halted, err = _probe_run(
            model, tok, probe_prompt, device, max_cycles=200,
            context_prefix=args.context_prefix,
            max_instr_tokens=args.max_instr_tokens)
        got = cpu.mem_word(probe_addr)
        mark = '✓' if halted and not err and got == probe_expected else '✗'
        print(f'  probe {mark} ops={len(emitted)} halted={halted} '
              f'mem[0x{probe_addr:x}]={got} (want {probe_expected})', flush=True)

    print(f'training {args.steps} steps batch={args.batch}', flush=True)
    best_inf = 0.0
    t_start = time.time()
    for step in range(args.steps):
        model.train()
        idx = torch.tensor(np.random.choice(N, args.batch, replace=False),
                           device=device)
        opt.zero_grad()
        loss, _, _, _ = run_batch(idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step()

        if step % 50 == 0:
            model.eval()
            with torch.no_grad():
                eidx = torch.tensor(
                    np.random.choice(N, min(512, N), replace=False),
                    device=device)
                _, pfc, fc, tot = run_batch(eidx)
            pf = [c / max(tot, 1) for c in pfc]
            inf = fc / max(tot, 1)
            print(f'step {step + args.init_step:5d}  loss={loss.item():.4f}  inf={inf:.1%}  [' +
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
                        'inject_every': INJECT_EVERY,
                        'max_instr_tokens': args.max_instr_tokens,
                        'context_prefix': args.context_prefix,
                        'dtype': args.dtype,
                        'freeze_backbone': args.freeze_backbone,
                    },
                }, args.ckpt)

    torch.save({
        'state': trainable_state_dict(),
        'config': {
            'backbone_id': args.backbone_id,
            'hidden': HIDDEN,
            'inject_every': INJECT_EVERY,
            'max_instr_tokens': args.max_instr_tokens,
            'context_prefix': args.context_prefix,
            'dtype': args.dtype,
            'freeze_backbone': args.freeze_backbone,
        },
    }, args.ckpt.replace('.pt', '_final.pt'))
    print(f'done. best inf={best_inf:.1%}', flush=True)


if __name__ == '__main__':
    main()
