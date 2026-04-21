"""
SQLite database state → cross-attention K/V tokens.

Each table becomes one or more "state tokens". We render each table as a
compact structured string (name, columns with types, sample rows), then
tokenize with the backbone's own tokenizer and embed with the backbone's
input embedding layer. The resulting token sequence is passed to the
adapters as K/V. Query-result rows follow the same path.

Using the backbone's embeddings (instead of a from-scratch encoder)
gives the adapters a K/V space that is already aligned with the
backbone's hidden representations.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn


DEFAULT_SAMPLE_ROWS = 3
DEFAULT_MAX_VALUE_CHARS = 40
DEFAULT_MAX_KV_TOKENS = 1024


@dataclass
class TableSnapshot:
    name: str
    columns: list[tuple[str, str]]   # (col_name, col_type)
    sample_rows: list[tuple]
    primary_key: list[str]
    foreign_keys: list[tuple[str, str, str]]  # (col, ref_table, ref_col)


def _trunc(x, n: int = DEFAULT_MAX_VALUE_CHARS) -> str:
    s = 'NULL' if x is None else str(x)
    if len(s) > n:
        s = s[: n - 1] + '…'
    return s


def snapshot_sqlite(db_path: str,
                    sample_rows: int = DEFAULT_SAMPLE_ROWS
                    ) -> list[TableSnapshot]:
    """Read schema + a few sample rows from a SQLite file."""
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode('utf-8', errors='replace')
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    names = [r[0] for r in cur.fetchall()]
    out: list[TableSnapshot] = []
    for name in names:
        cur.execute(f'PRAGMA table_info("{name}")')
        cols_info = cur.fetchall()       # cid, name, type, notnull, default, pk
        columns = [(r[1], r[2] or '') for r in cols_info]
        pk = [r[1] for r in cols_info if r[5]]
        cur.execute(f'PRAGMA foreign_key_list("{name}")')
        fks = [(r[3], r[2], r[4]) for r in cur.fetchall()]
        try:
            cur.execute(f'SELECT * FROM "{name}" LIMIT {int(sample_rows)}')
            rows = cur.fetchall()
        except sqlite3.DatabaseError:
            rows = []
        out.append(TableSnapshot(name=name, columns=columns, sample_rows=rows,
                                 primary_key=pk, foreign_keys=fks))
    conn.close()
    return out


def render_table(t: TableSnapshot) -> str:
    """Produce the string we'll tokenize for this table."""
    cols = ', '.join(f'{c}:{ty}' for c, ty in t.columns)
    head = f'TABLE {t.name} ({cols})'
    pieces = [head]
    if t.primary_key:
        pieces.append(f'PK({", ".join(t.primary_key)})')
    if t.foreign_keys:
        fks = '; '.join(f'{c}->{rt}.{rc}' for c, rt, rc in t.foreign_keys)
        pieces.append(f'FK({fks})')
    if t.sample_rows:
        col_names = [c for c, _ in t.columns]
        rows = []
        for r in t.sample_rows:
            cells = ', '.join(f'{c}={_trunc(v)}'
                              for c, v in zip(col_names, r))
            rows.append('(' + cells + ')')
        pieces.append('SAMPLES: ' + ' | '.join(rows))
    return '  '.join(pieces)


def render_result(columns: Sequence[str], rows: Sequence[Sequence]) -> str:
    """Render a prior-query result as a single state block."""
    header = 'RESULT cols=(' + ', '.join(columns) + ')'
    body = ' | '.join('(' + ', '.join(_trunc(v) for v in r) + ')'
                      for r in rows[:DEFAULT_SAMPLE_ROWS])
    return header + '  ' + body


class StateEncoder(nn.Module):
    """Turn a list of rendered state strings into K/V tokens for the
    cross-attn adapters.

    Strategy:
      - tokenize each state string with the backbone tokenizer
      - look up each token id in the backbone's input-embedding matrix
      - concatenate, pad to a max length, and return (kv, kv_mask)

    We don't add extra learned projections — the adapters themselves
    project queries and keys inside ``MultiheadAttention``. This keeps
    the encoder parameter-free and its outputs in the backbone's own
    representation space.
    """

    def __init__(self, tokenizer, embed: nn.Embedding,
                 max_tokens: int = DEFAULT_MAX_KV_TOKENS):
        super().__init__()
        self.tok = tokenizer
        self.embed = embed
        self.max_tokens = max_tokens

    @property
    def hidden(self) -> int:
        return self.embed.embedding_dim

    def tokenize_blocks(self, blocks: Iterable[str]) -> list[list[int]]:
        out = []
        for b in blocks:
            ids = self.tok(b, add_special_tokens=False)['input_ids']
            out.append(ids)
        return out

    def encode(self, blocks_per_example: list[list[str]],
               device: torch.device | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """blocks_per_example: outer list is batch, inner list is the
        state blocks (one per table or query result). Returns
        (kv, kv_mask) with shapes [B, T, H] and [B, T]."""
        device = device or self.embed.weight.device
        batch_ids: list[list[int]] = []
        for blocks in blocks_per_example:
            ids: list[int] = []
            for b in blocks:
                ids.extend(self.tok(b, add_special_tokens=False)['input_ids'])
                # soft separator so the attn can tell blocks apart
                ids.append(self.tok.eos_token_id)
            ids = ids[: self.max_tokens]
            batch_ids.append(ids)

        T = max((len(x) for x in batch_ids), default=1)
        T = max(T, 1)
        B = len(batch_ids)
        id_tensor = torch.zeros(B, T, dtype=torch.long, device=device)
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        for i, ids in enumerate(batch_ids):
            if ids:
                id_tensor[i, : len(ids)] = torch.tensor(ids, device=device)
                mask[i, : len(ids)] = True
        with torch.no_grad():
            kv = self.embed(id_tensor)
        return kv, mask


def encode_database(db_path: str, tokenizer, embed: nn.Embedding,
                    sample_rows: int = DEFAULT_SAMPLE_ROWS,
                    max_tokens: int = DEFAULT_MAX_KV_TOKENS,
                    device: torch.device | None = None):
    """Convenience: snapshot a db + encode a single-example batch."""
    tables = snapshot_sqlite(db_path, sample_rows=sample_rows)
    blocks = [render_table(t) for t in tables]
    enc = StateEncoder(tokenizer, embed, max_tokens=max_tokens)
    return enc.encode([blocks], device=device)
