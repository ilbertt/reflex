"""
Execution-accuracy evaluation on BIRD-dev.

For each example: generate SQL with the trained cross-attn adapters
(grounded mode) OR with the raw backbone and schema-in-prompt
(baseline mode); execute both the predicted and gold SQL against the
example's SQLite file; compare result sets.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import torch

from .data import load_bird_split
from .model import GroundedSQL, build_backbone, render_prompt
from .state_encoder import (DEFAULT_MAX_KV_TOKENS, DEFAULT_SAMPLE_ROWS,
                            StateEncoder, render_table, snapshot_sqlite)


def run_sql(db_path: str, sql: str, timeout: float = 10.0):
    conn = sqlite3.connect(db_path, timeout=timeout)
    conn.text_factory = lambda b: b.decode('utf-8', errors='replace')
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
    finally:
        conn.close()
    return rows


def results_match(pred_rows, gold_rows) -> bool:
    def norm(rs):
        return sorted(tuple(r) for r in rs)
    try:
        return norm(pred_rows) == norm(gold_rows)
    except TypeError:
        return [list(map(str, r)) for r in pred_rows] == \
               [list(map(str, r)) for r in gold_rows]


def _extract_sql(text: str) -> str:
    """Pull a single SQL statement out of the model output, trimming any
    chatter the backbone occasionally produces despite the system
    prompt."""
    t = text.strip()
    if '```' in t:
        parts = t.split('```')
        for p in parts:
            p = p.strip()
            if p.lower().startswith('sql\n'):
                p = p[4:]
            if p.lower().startswith(('select', 'with', 'insert', 'update',
                                     'delete')):
                t = p
                break
    for stop in ('\n\n', '</s>', '<|im_end|>'):
        if stop in t:
            t = t.split(stop, 1)[0]
    return t.strip().rstrip(';') + ';'


def _build_prompt_baseline(tok, question: str, db_path: str,
                           sample_rows: int) -> str:
    """Baseline: dump the schema as text into the user message, no
    cross-attention."""
    tables = snapshot_sqlite(db_path, sample_rows=sample_rows)
    schema = '\n'.join(render_table(t) for t in tables)
    msg = f'Schema:\n{schema}\n\nQuestion: {question}'
    return render_prompt(tok, msg)


@torch.no_grad()
def generate(model: GroundedSQL, tok, prompt: str,
             kv: torch.Tensor | None, kv_mask: torch.Tensor | None,
             max_new_tokens: int, device) -> str:
    enc = tok(prompt, return_tensors='pt', add_special_tokens=False).to(device)
    model.set_state_kv(kv, kv_mask)
    try:
        out = model.backbone.generate(
            input_ids=enc['input_ids'],
            attention_mask=enc['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    finally:
        model.set_state_kv(None, None)
    gen_ids = out[0, enc['input_ids'].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True)


def evaluate(bird_root: str, adapters_path: str | None, baseline: bool,
             backbone_id: str, max_new_tokens: int = 256,
             limit: int | None = None, log_path: str | None = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone, tok, hidden = build_backbone(backbone_id)
    backbone.to(device)
    model = GroundedSQL(backbone, hidden=hidden, freeze_backbone=True)
    model.to(device)
    model.eval()

    if adapters_path and not baseline:
        state = torch.load(adapters_path, map_location=device)
        model.adapters.load_state_dict(state['adapters'])
        model.kv_norm.load_state_dict(state['kv_norm'])
        print(f'loaded adapters from {adapters_path}')

    input_embed = backbone.get_input_embeddings()
    state_encoder = StateEncoder(tok, input_embed,
                                 max_tokens=DEFAULT_MAX_KV_TOKENS)

    examples = load_bird_split(bird_root, 'dev')
    if limit:
        examples = examples[:limit]

    n = 0
    n_exec_ok = 0
    n_correct = 0
    logs = []
    t0 = time.time()
    for i, ex in enumerate(examples):
        if baseline:
            prompt = _build_prompt_baseline(tok, ex.question, ex.db_path,
                                            DEFAULT_SAMPLE_ROWS)
            kv, kv_mask = None, None
        else:
            q = ex.question
            if ex.evidence:
                q = f'{q}\nHint: {ex.evidence}'
            prompt = render_prompt(tok, q)
            tables = snapshot_sqlite(ex.db_path,
                                     sample_rows=DEFAULT_SAMPLE_ROWS)
            blocks = [render_table(t) for t in tables]
            kv, kv_mask = state_encoder.encode([blocks], device=device)

        raw = generate(model, tok, prompt, kv, kv_mask,
                       max_new_tokens=max_new_tokens, device=device)
        pred_sql = _extract_sql(raw)

        ok = False
        correct = False
        err = ''
        try:
            pred_rows = run_sql(ex.db_path, pred_sql)
            gold_rows = run_sql(ex.db_path, ex.sql)
            ok = True
            correct = results_match(pred_rows, gold_rows)
        except Exception as e:
            err = str(e)[:200]

        n += 1
        n_exec_ok += int(ok)
        n_correct += int(correct)
        logs.append({
            'i': i, 'db_id': ex.db_id, 'question': ex.question,
            'gold': ex.sql, 'pred': pred_sql, 'exec_ok': ok,
            'correct': correct, 'err': err,
        })
        if (i + 1) % 25 == 0:
            dt = time.time() - t0
            print(f'[{i+1}/{len(examples)}] exec_ok={n_exec_ok/n:.3f} '
                  f'acc={n_correct/n:.3f} ({dt:.1f}s)')

    print(f'FINAL: exec_ok={n_exec_ok/n:.4f}  exec_acc={n_correct/n:.4f} '
          f'(n={n})')
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            for row in logs:
                f.write(json.dumps(row) + '\n')
        print(f'wrote per-example log → {log_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bird-root', default='data/bird')
    ap.add_argument('--adapters', default=None,
                    help='path to adapters_*.pt (ignored if --baseline)')
    ap.add_argument('--baseline', action='store_true',
                    help='eval raw backbone with schema-in-prompt')
    ap.add_argument('--backbone', default='Qwen/Qwen2.5-Coder-3B-Instruct')
    ap.add_argument('--max-new-tokens', type=int, default=256)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--log', default=None)
    args = ap.parse_args()
    evaluate(args.bird_root, args.adapters, args.baseline, args.backbone,
             max_new_tokens=args.max_new_tokens, limit=args.limit,
             log_path=args.log)


if __name__ == '__main__':
    main()
