# Flamingo-SQL

Grounded text-to-SQL via gated cross-attention.

Standard text-to-SQL systems dump the schema as text into the prompt.
Flamingo-SQL injects database state — schema, sample rows, and (phase 2)
prior query results — through Flamingo-style gated cross-attention at
every layer of a frozen Qwen2.5-Coder-3B-Instruct backbone. The
question flows through the backbone as text; the database enters through
a separate K/V channel.

Why:

1. Schema doesn't eat the context window.
2. The model sees actual data values, not just column names.
3. After query N runs, its result becomes the K/V state for query N+1 —
   a grounded closed loop, not one-shot text-to-SQL.

Backbone is frozen. Only the Flamingo adapters train (~tens of M params).

## Architecture

```
question text ─► Qwen2.5-Coder-3B (frozen)
                    │
      every 4 layers │
                    ▼
            CrossAttnAdapter(q=hidden, k/v=state_tokens)
                    ▲
       DB snapshot ─┘  (schema + sample rows → tokenizer → embed → K/V)
```

SQL is generated as text using the backbone's own LM head. The adapters
learn to route database understanding into SQL generation.

## Layout

```
reflex/
  model.py          Flamingo gated cross-attention + GroundedSQL wrapper
  state_encoder.py  SQLite snapshot → K/V tokens (via backbone embeddings)
  data.py           BIRD dataset loader
  train.py          training loop (frozen backbone, adapters-only)
  eval.py           BIRD-dev execution accuracy; --baseline flag for
                    schema-in-prompt comparison
  inference.py      interactive REPL
scripts/
  download_bird.sh  fetch and unpack BIRD under data/bird/
```

## Setup

```bash
uv sync
bash scripts/download_bird.sh        # → data/bird/{train,dev}/
```

GPU assumed: AWS A10G 24GB. Backbone runs in bf16.

## Train

```bash
python -m reflex.train \
    --bird-root data/bird \
    --out-dir runs/flamingo_sql \
    --epochs 2 --batch-size 4 --grad-accum 4
```

Adapter checkpoints land at `runs/flamingo_sql/adapters_*.pt`.

## Evaluate

Grounded (trained adapters):

```bash
python -m reflex.eval \
    --bird-root data/bird \
    --adapters runs/flamingo_sql/adapters_final.pt \
    --log runs/flamingo_sql/dev_log.jsonl
```

Baseline (same backbone, schema dumped as text, no adapters):

```bash
python -m reflex.eval --bird-root data/bird --baseline
```

Metric: execution accuracy (predicted SQL produces the same rows as the
gold SQL when executed against the db).

## Interactive

```bash
python -m reflex.inference \
    --db data/bird/dev/dev_databases/<db_id>/<db_id>.sqlite \
    --adapters runs/flamingo_sql/adapters_final.pt \
    --feedback
```

`--feedback` feeds the previous query's result back into the K/V state
for the next question (phase-2 multi-step).

## Notes

- The state encoder reuses the backbone's own tokenizer + input
  embedding matrix. This keeps adapter K/V in the backbone's native
  representation space and adds zero parameters.
- Adapter gates (`attn_gate`, `mlp_gate`) are initialised to zero via a
  `tanh` — the model starts as the frozen backbone and the adapters
  only contribute once they've learned something useful.
- Tune `INJECT_EVERY` in `reflex/model.py` if memory is tight.
