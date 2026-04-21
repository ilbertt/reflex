"""
BIRD dataset loader.

Expects the BIRD release laid out as (after unzip):

    data/bird/
        train/
            train.json
            train_databases/<db_id>/<db_id>.sqlite
        dev/
            dev.json
            dev_databases/<db_id>/<db_id>.sqlite

``download_bird.sh`` fetches and unpacks this layout.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset


@dataclass
class BirdExample:
    question: str
    sql: str
    db_id: str
    db_path: str
    evidence: str = ''


def _find_db(db_root: Path, db_id: str) -> str | None:
    # BIRD: <db_root>/<db_id>/<db_id>.sqlite
    p = db_root / db_id / f'{db_id}.sqlite'
    if p.exists():
        return str(p)
    # fallback: any .sqlite under the db_id dir
    for cand in (db_root / db_id).glob('*.sqlite'):
        return str(cand)
    return None


def load_bird_split(root: str, split: str) -> list[BirdExample]:
    root_p = Path(root)
    if split == 'train':
        json_path = root_p / 'train' / 'train.json'
        db_root = root_p / 'train' / 'train_databases'
    elif split == 'dev':
        json_path = root_p / 'dev' / 'dev.json'
        db_root = root_p / 'dev' / 'dev_databases'
    else:
        raise ValueError(f'Unknown split: {split}')
    with open(json_path, 'r') as f:
        raw = json.load(f)
    out: list[BirdExample] = []
    skipped: set[str] = set()
    malformed: set[str] = set()
    healthy: dict[str, bool] = {}
    for r in raw:
        db_id = r['db_id']
        db_path = _find_db(db_root, db_id)
        if db_path is None:
            skipped.add(db_id)
            continue
        if db_id not in healthy:
            try:
                c = sqlite3.connect(db_path)
                c.execute('SELECT name FROM sqlite_master LIMIT 1').fetchall()
                c.close()
                healthy[db_id] = True
            except sqlite3.DatabaseError:
                healthy[db_id] = False
        if not healthy[db_id]:
            malformed.add(db_id)
            continue
        out.append(BirdExample(
            question=r['question'],
            sql=r.get('SQL') or r.get('query') or '',
            db_id=db_id,
            db_path=db_path,
            evidence=r.get('evidence', ''),
        ))
    if skipped or malformed:
        print(f'[data] skipped {len(raw)-len(out)} examples. '
              f'no_sqlite={sorted(skipped)}  malformed={sorted(malformed)}')
    return out


class BirdDataset(Dataset):
    """Returns raw examples. Collation (tokenization, state-encoding) is
    handled by the collator in ``train.py`` so we can share one tokenizer
    and embed matrix across the batch."""

    def __init__(self, examples: list[BirdExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> BirdExample:
        return self.examples[i]


def iter_splits(root: str) -> Iterator[tuple[str, list[BirdExample]]]:
    for split in ('train', 'dev'):
        try:
            yield split, load_bird_split(root, split)
        except FileNotFoundError:
            continue
