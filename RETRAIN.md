# Retrain procedures for Path B and Path D

The `HEAD_STUDY.md` document scopes why the inference-only ceiling on the released checkpoint sits at 20/23 and what a head-side intervention would cost. This file is the operational recipe for each path — exactly what to run, in what order, and what wall-time to budget. Neither retrain is small enough to execute in a typical interactive session; each expects to be kicked off on an idle GPU and left.

Both paths produce *new* checkpoints with a config flag flipped (`use_target_head` for Path B, `use_progress_state` for Path D). Both remain interchangeable with the canonical `reflex.pt` inference surface — `reflex.demo.load` reads the config flags and builds the appropriate model topology, so the testbed, SPA, CLI, and the five existing decoders all accept the new checkpoint without code changes.

## Prerequisites (shared)

Corpus generation. The canonical `reflex/train.py` expects a flat list of training cycles produced by running verified programs through Unicorn. `scripts/generate_programs.py` produces the 80 k programs; the flattening step lives inside `reflex/train.py` itself (see `collect_state_sequences`). For both paths, the retrain scripts below expect a pickle of `list[(family, prompt, state, gt_instr_index)]` — inline this flattening step with a small harness if the training script's in-memory flattening does not suit.

Wall-time for corpus generation on an RTX 3090 host: ~1–3 h (each program is verified end-to-end in Unicorn, which is CPU work; GPU is unused).

GPU freeing. The testbed SPA (`scripts/jepa_testbed_server.py`) holds the backbone on the GPU while running. Kill it (`tmux kill-session -t spa` or `curl -X POST …`) before starting a retrain.

## Path B — target-aware cross-attention head

### What it trains

Frozen: backbone (Qwen-Coder-7B), seven cross-attn adapters, state encoder, `kv_norm`.

Trainable (the only parameters that receive gradients):
- `TargetAwareHead`: `prompt_norm` (LayerNorm, hidden), `q_proj` (hidden → hidden), `xattn` (MultiheadAttention, hidden, 8 heads), `out_proj` (hidden → hidden), `gate` (scalar).

Optional also trainable, via `--train-head`:
- `head_mlp` (2 × Linear(hidden, hidden) + GELU),
- `embed_head` (Linear(hidden, embed_dim)),
- `instr_table` (Embedding(691, embed_dim)).

The gate is zero-initialised; training opens it. Expect the first ~200 steps to have near-zero loss change as the gate moves off zero, then rapid movement.

### How to run

```bash
# 1. Corpus already exists at programs/*/*.json; train.py loads and
#    caches a flattened cycle pool at .cache/state_seqs_<hash>.pkl on
#    first run, then reuses that cache. No separate corpus-gen step.

# 2. Retrain only the target head on top of the released checkpoint
uv run train \
    --resume reflex.pt \
    --ckpt reflex_target_head.pt \
    --use-target-head \
    --freeze-except-target-head \
    --also-train-canonical-head \
    --steps 3000 --batch 16 --lr 5e-5 --nce-temp 0.07

# 3. Plug into the testbed — the new ckpt's cfg has use_target_head=True
uv run python scripts/jepa_testbed_server.py \
    --ckpt reflex_target_head.pt --device cuda --port 8766
```

The new checkpoint's config will have `use_target_head=True`. Every decoder in `reflex/decoders/` will pick up the new pathway automatically — no code changes required.

### Wall-time on RTX 3090

- Head-only (default): ~2–4 h for 3 000 steps at batch 16. Backbone forward dominates; backward is cheap because only `TargetAwareHead` is trainable.
- Head + canonical head (`--train-head`): ~3–5 h for 3 000 steps. Minor overhead from the additional trainable tensors.

### Expected impact

Three `display`-tier failures (`display OK` already rescued by `rd_consistency`, plus `show 42` and `print hello`) are the primary target. `sum 1..10` — a prompt-interpretation failure — may or may not be rescued; the cross-attention gives the head a second look at the prompt embedding, which could but need not help disambiguate the loop bound.

Measurement: run `pure` (the canonical decoder) against the new checkpoint via the testbed. Target: 22/23 (96%). 23/23 would mean `sum 1..10` was also rescued, which would imply the target head is resolving more than just byte-level ambiguity.

## Path D — state-encoder progress-token enrichment

### What it trains

Frozen: backbone only.

Trainable:
- `state_encoder` (existing 65 role embeddings retrain + 3 new rows for progress tokens; byte value projection reused).
- `kv_norm` (LayerNorm on state KV).
- Seven `CrossAttnAdapter` modules.
- `head_mlp`, `embed_head`, `instr_table`.

This is the canonical retrain surface plus three new role-embedding rows. Effectively a full re-run of `reflex/train.py` with `--use-progress-state` (not yet exposed in CLI; `train.py` does not currently carry a flag for progress-state — see below).

### The `train.py` gap

`reflex/train.py` instantiates `GroundedReflex` without setting `use_progress_state`. To perform a Path-D retrain, add one of:

- A CLI flag `--use-progress-state` in `main()` that is forwarded to `GroundedReflex(..., use_progress_state=True)` and into the ckpt config dict.
- A dedicated `scripts/retrain_full_progress.py` analogous to `retrain_target_head.py`.

The latter keeps `train.py` unchanged for reproducibility of canonical checkpoints. The former is a one-line change if you prefer.

Both approaches must also change the per-cycle `extract_state(cpu)` call in `train.py`'s `collect_state_sequences` to pass `enriched=True`. This produces 68-value state vectors; the corpus pickle consumed by the retrain must match.

### How to run (once the above wiring is in place)

```bash
# 1. Generate corpus with progress-token enrichment
uv run python scripts/generate_programs.py       # programs are identical
# But cycle-collection needs enriched=True; fold into flatten step.

# 2. Full retrain from scratch with the progress state
uv run python scripts/retrain_full_progress.py \
    --base-ckpt reflex.pt \
    --corpus cycles_enriched.pkl \
    --out reflex_progress.pt \
    --steps 15000 --batch 8 --lr 1e-4
```

(Batch 8 because the 3090's 24 GB will not fit batch 16 for a retrainable state encoder + adapters on a frozen 7 B backbone with activations. Gradient checkpointing on the adapters can push it back to 16 if needed.)

### Wall-time on RTX 3090

Full retrain, 15 000 steps, batch 8 (or batch 16 with grad checkpointing), bf16:

- Batch 8, no grad checkpointing: **~25–35 h**
- Batch 16, grad checkpointing on adapters: **~15–20 h**
- Aggressive freeze (only state encoder + head, adapters frozen): **~8–12 h** at the cost of adaptation quality

The larger end is where the comparable A100 80 GB number would have been ~4 h. The 3090 is ~4× slower in sustained bf16 on a 7 B-class backbone, and the VRAM constraint forces smaller batch sizes that need more wall-time steps for equivalent data coverage.

### Expected impact

All three remaining failures could in principle be rescued. State enrichment is the canonical lever: with explicit "bytes written to display" and "data words written" signals, the adapters have what they need to condition on sequence position, which is the missing information. `sum 1..10` is the most uncertain — its failure is a prompt-interpretation issue, not a position-tracking issue. Target: 22/23 at minimum, 23/23 if the progress signals also help the loop-bound ambiguity.

## Picking between B and D

| factor | Path B | Path D |
|---|---|---|
| wall-time on 3090 | ~2–4 h | ~15–35 h |
| code in place today | `retrain_target_head.py` shipped | needs `train.py` flag or dedicated script |
| spirit cost | medium (new cross-attn pathway in head) | **zero** (state-encoder enrichment is the architecture's designed lever) |
| expected lift on display | high | high |
| expected lift on `sum 1..10` | uncertain | plausible |
| reversibility | checkpoint-scoped | checkpoint-scoped |

**Recommendation**: run Path B first. It is the fastest way to confirm that a head-side intervention is effective; if Path B rescues the display tier, the cost-benefit for running Path D (~10× wall-time) depends on whether the remaining `sum 1..10` matters enough to justify the retrain. If Path B only partially rescues (e.g., `show 42` but not `print hello`), Path D is the unambiguous next step.
