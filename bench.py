"""Benchmark per-cycle inference latency for the ctx checkpoint."""
import time
import torch
import numpy as np

from reflex.demo import load
from reflex.model import (
    MAX_INSTR_TOKENS, code_region_halt_fill, extract_state, render_prompt,
)
from reflex.programs import SRC_OFFSET
from reflex.riscv import DATA_BASE, PROGRAM_START, Rv32i

ckpt_path = 'reflex.pt'
device = 'cuda'
model, tok, cfg = load(ckpt_path, device)
mt = cfg.get('max_instr_tokens', MAX_INSTR_TOKENS)

prompt = 'say hi'
text = render_prompt(tok, prompt,
                    use_chat_template=cfg.get('chat_template', True),
                    use_context_prefix=cfg.get('context_prefix', False))
e = tok(text, padding='max_length', truncation=True,
        max_length=mt, return_tensors='pt').to(device)
ids, amask = e.input_ids, e.attention_mask
print(f'prompt tokens: {int(amask.sum().item())} / {mt} (padding filled)')

cpu = Rv32i()
cpu.uc.mem_write(PROGRAM_START, code_region_halt_fill())
seed = b''.join(int(i).to_bytes(4, 'little') for i in range(1, 9))
cpu.uc.mem_write(DATA_BASE + SRC_OFFSET, seed)

state = extract_state(cpu)
state_t = torch.from_numpy(state.astype('int64')).unsqueeze(0).to(device)

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = model(ids, amask, state_t)
torch.cuda.synchronize()

# Time 100 full forward passes (state_encoder → backbone-with-injected-adapters → head)
N = 100
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(N):
    state = extract_state(cpu)
    state_t = torch.from_numpy(state.astype('int64')).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(ids, amask, state_t)
torch.cuda.synchronize(); dt = time.perf_counter() - t0
print(f'\nFULL per-cycle forward (state encoder + backbone + injected adapters + head):')
print(f'  {N} cycles in {dt*1000:.1f} ms → {dt*1000/N:.2f} ms/cycle  '
      f'({N/dt:.1f} cycles/sec)')

# Separately time just the state encoder + kv_norm — the stateful part.
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(N):
    state = extract_state(cpu)
    state_t = torch.from_numpy(state.astype('int64')).unsqueeze(0).to(device)
    with torch.no_grad():
        kv = model.kv_norm(model.state_encoder(state_t))
torch.cuda.synchronize(); dt_state = time.perf_counter() - t0
print(f'\nSTATE encoder + kv_norm only (65-token K/V build):')
print(f'  {N} calls in {dt_state*1000:.1f} ms → {dt_state*1000/N:.2f} ms/call')

# Time just the backbone forward (mimics "instruction encoding" cost).
# With hooks disabled (kv=None) the adapters no-op so this is pure backbone.
model._current_kv = None
for handle in model._hook_handles:
    handle.remove()
model._hook_handles = []
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(N):
    with torch.no_grad():
        out = model.backbone(input_ids=ids, attention_mask=amask,
                             use_cache=False, return_dict=True)
torch.cuda.synchronize(); dt_bb = time.perf_counter() - t0
print(f'\nBACKBONE only (no adapters, no state injection) — i.e. the pure '
      'one-time instruction encoding if we were allowed to cache:')
print(f'  {N} calls in {dt_bb*1000:.1f} ms → {dt_bb*1000/N:.2f} ms/call')

print(f'\nSummary:')
print(f'  per-cycle cost:            {dt*1000/N:.2f} ms')
print(f'  pure state encoder cost:   {dt_state*1000/N:.2f} ms')
print(f'  pure backbone forward:     {dt_bb*1000/N:.2f} ms  ({mt} tokens)')
print(f'\n  Note: our Flamingo architecture *cannot* cache the backbone '
      'forward\n  because cross-attention adapters inject live state K/V into '
      'the\n  backbone\'s own layers. Every cycle must re-run the full '
      f'{mt}-token\n  backbone forward. The 3B model is ~98% of per-cycle wall time.')
