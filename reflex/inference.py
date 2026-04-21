"""
Interactive SQL prompt.

    python -m reflex.inference --db path/to.sqlite --adapters adapters_final.pt

Type a question; see SQL and the executed result. Blank line exits.
"""
from __future__ import annotations

import argparse
import sqlite3

import torch

from .eval import _extract_sql, generate, run_sql
from .model import GroundedSQL, build_backbone, render_prompt
from .state_encoder import (DEFAULT_MAX_KV_TOKENS, DEFAULT_SAMPLE_ROWS,
                            StateEncoder, render_result, render_table,
                            snapshot_sqlite)


def repl(db_path: str, adapters_path: str | None, backbone_id: str,
         feedback: bool):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone, tok, hidden = build_backbone(backbone_id)
    backbone.to(device)
    model = GroundedSQL(backbone, hidden=hidden, freeze_backbone=True)
    model.to(device)
    model.eval()
    if adapters_path:
        state = torch.load(adapters_path, map_location=device)
        model.adapters.load_state_dict(state['adapters'])
        model.kv_norm.load_state_dict(state['kv_norm'])

    input_embed = backbone.get_input_embeddings()
    state_encoder = StateEncoder(tok, input_embed,
                                 max_tokens=DEFAULT_MAX_KV_TOKENS)

    tables = snapshot_sqlite(db_path, sample_rows=DEFAULT_SAMPLE_ROWS)
    base_blocks = [render_table(t) for t in tables]
    prior_blocks: list[str] = []

    print(f'connected to {db_path}; {len(tables)} tables. empty line to quit.')
    while True:
        try:
            q = input('> ').strip()
        except EOFError:
            break
        if not q:
            break
        blocks = base_blocks + prior_blocks
        kv, kv_mask = state_encoder.encode([blocks], device=device)
        prompt = render_prompt(tok, q)
        raw = generate(model, tok, prompt, kv, kv_mask,
                       max_new_tokens=256, device=device)
        sql = _extract_sql(raw)
        print(f'SQL: {sql}')
        try:
            rows = run_sql(db_path, sql)
        except Exception as e:
            print(f'ERR: {e}')
            continue
        for r in rows[:20]:
            print(r)
        if len(rows) > 20:
            print(f'... ({len(rows)} rows total)')
        if feedback and rows:
            conn = sqlite3.connect(db_path)
            try:
                cur = conn.cursor()
                cur.execute(sql)
                cols = [d[0] for d in cur.description] if cur.description else []
            finally:
                conn.close()
            prior_blocks = [render_result(cols, rows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', required=True)
    ap.add_argument('--adapters', default=None)
    ap.add_argument('--backbone', default='Qwen/Qwen2.5-Coder-3B-Instruct')
    ap.add_argument('--feedback', action='store_true',
                    help='feed previous query result back as state for '
                         'the next question (phase-2 multi-step)')
    args = ap.parse_args()
    repl(args.db, args.adapters, args.backbone, feedback=args.feedback)


if __name__ == '__main__':
    main()
