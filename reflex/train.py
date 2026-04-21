"""
Training loop for grounded text-to-SQL.

Frozen Qwen2.5-Coder-3B backbone; only the Flamingo cross-attn adapters
(and their K/V-norm) train. Supervision: standard causal-LM loss on the
assistant turn (the gold SQL). Database state is injected through the
adapters; the question goes through the backbone as text.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from .data import BirdDataset, load_bird_split, BirdExample
from .model import GroundedSQL, build_backbone, render_prompt
from .state_encoder import (DEFAULT_MAX_KV_TOKENS, DEFAULT_SAMPLE_ROWS,
                            StateEncoder, render_table, snapshot_sqlite)


@dataclass
class TrainConfig:
    bird_root: str = 'data/bird'
    out_dir: str = 'runs/flamingo_sql'
    backbone: str = 'Qwen/Qwen2.5-Coder-3B-Instruct'
    epochs: int = 2
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_steps: int = 200
    max_question_tokens: int = 256
    max_sql_tokens: int = 256
    max_kv_tokens: int = DEFAULT_MAX_KV_TOKENS
    sample_rows: int = DEFAULT_SAMPLE_ROWS
    seed: int = 0
    log_every: int = 20
    save_every: int = 2000
    adapter_checkpointing: bool = True


# A tiny in-process cache of (db_id → rendered state blocks) so we don't
# re-open the same SQLite file thousands of times during an epoch.
_SCHEMA_CACHE: dict[str, list[str]] = {}


def render_state_blocks(db_path: str, db_id: str,
                        sample_rows: int) -> list[str]:
    if db_id in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[db_id]
    tables = snapshot_sqlite(db_path, sample_rows=sample_rows)
    blocks = [render_table(t) for t in tables]
    _SCHEMA_CACHE[db_id] = blocks
    return blocks


class Collator:
    """Builds batches of (input_ids, attention_mask, labels, kv, kv_mask).

    Target = rendered prompt + gold SQL + EOS. We mask the prompt portion
    out of the loss (set labels=-100) so only the SQL tokens contribute.
    """

    def __init__(self, tokenizer, state_encoder: StateEncoder, cfg: TrainConfig):
        self.tok = tokenizer
        self.state_encoder = state_encoder
        self.cfg = cfg

    def __call__(self, batch: list[BirdExample]):
        prompts, fulls, prompt_lens = [], [], []
        blocks_per_example: list[list[str]] = []
        for ex in batch:
            q = ex.question
            if ex.evidence:
                q = f'{q}\nHint: {ex.evidence}'
            prompt = render_prompt(self.tok, q)
            full = prompt + ex.sql + self.tok.eos_token
            prompts.append(prompt)
            fulls.append(full)
            prompt_ids = self.tok(prompt, add_special_tokens=False,
                                  truncation=True,
                                  max_length=self.cfg.max_question_tokens
                                  )['input_ids']
            prompt_lens.append(len(prompt_ids))
            blocks_per_example.append(
                render_state_blocks(ex.db_path, ex.db_id,
                                    self.cfg.sample_rows))

        max_total = self.cfg.max_question_tokens + self.cfg.max_sql_tokens
        enc = self.tok(fulls, add_special_tokens=False,
                       padding=True, truncation=True,
                       max_length=max_total, return_tensors='pt')
        input_ids = enc['input_ids']
        attention_mask = enc['attention_mask']
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100

        kv, kv_mask = self.state_encoder.encode(blocks_per_example,
                                                device=input_ids.device)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'kv': kv,
            'kv_mask': kv_mask,
        }


def _cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    import math
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * t))


def train(cfg: TrainConfig):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone, tok, hidden = build_backbone(cfg.backbone)
    backbone.to(device)
    model = GroundedSQL(backbone, hidden=hidden, freeze_backbone=True)
    model.adapter_checkpointing = cfg.adapter_checkpointing
    model.to(device)

    input_embed = backbone.get_input_embeddings()
    state_encoder = StateEncoder(tok, input_embed, max_tokens=cfg.max_kv_tokens)

    examples = load_bird_split(cfg.bird_root, 'train')
    random.shuffle(examples)
    ds = BirdDataset(examples)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=0, collate_fn=Collator(tok, state_encoder, cfg))

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f'trainable params: {n_trainable/1e6:.2f}M')
    opt = torch.optim.AdamW(trainable, lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    total_steps = (len(loader) // cfg.grad_accum) * cfg.epochs
    step = 0
    model.train()
    t0 = time.time()
    running_loss = 0.0
    for epoch in range(cfg.epochs):
        for micro_i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(dtype=torch.bfloat16):
                out = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            kv=batch['kv'], kv_mask=batch['kv_mask'],
                            labels=batch['labels'], use_cache=False)
                loss = out.loss / cfg.grad_accum
            loss.backward()
            running_loss += loss.item()

            if (micro_i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                lr_now = _cosine_lr(step, cfg.warmup_steps, total_steps, cfg.lr)
                for g in opt.param_groups:
                    g['lr'] = lr_now
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % cfg.log_every == 0:
                    dt = time.time() - t0
                    avg = running_loss / cfg.log_every
                    running_loss = 0.0
                    print(f'ep{epoch} step{step}/{total_steps} '
                          f'loss={avg:.4f} lr={lr_now:.2e} '
                          f'{dt/cfg.log_every:.2f}s/step')
                    t0 = time.time()

                if cfg.save_every and step % cfg.save_every == 0:
                    _save_adapters(model, out_dir / f'adapters_step{step}.pt')

    _save_adapters(model, out_dir / 'adapters_final.pt')


def _save_adapters(model: GroundedSQL, path: Path):
    state = {
        'adapters': model.adapters.state_dict(),
        'kv_norm': model.kv_norm.state_dict(),
        'inject_indices': model.inject_indices,
    }
    torch.save(state, path)
    print(f'saved adapters → {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bird-root', default='data/bird')
    ap.add_argument('--out-dir', default='runs/flamingo_sql')
    ap.add_argument('--backbone', default='Qwen/Qwen2.5-Coder-3B-Instruct')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--grad-accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    cfg = TrainConfig(
        bird_root=args.bird_root, out_dir=args.out_dir,
        backbone=args.backbone, epochs=args.epochs,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, seed=args.seed,
    )
    train(cfg)


if __name__ == '__main__':
    main()
