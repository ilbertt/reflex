# Reflex

**Thesis:** GUI agents that "think out loud" before clicking are slow because most of the latency is the monologue, not the click. Fine-tuning a vision-language model collapses the reasoning into a reflex — same model, same task, but it skips straight to the action.

**This demo:** Same Qwen2-VL-2B base model, same AGUVIS screenshots and instructions, two inference modes:

- **Reflex** — LoRA fine-tuned to emit the action call directly
- **CoT** — base model prompted to reason step-by-step before emitting the action

We measure inference latency and tokens generated.

## Run it

```bash
# 0. Install deps once (uv reads pyproject.toml)
uv sync

# 1. Prepare data — downloads AGUVIS, saves screenshots + writes train.jsonl
uv run prepare-data --max 1000

# 2. LoRA fine-tune Qwen2-VL-2B on (image, instruction) → action
uv run train

# 3. Benchmark — reflex vs prompted CoT
uv run benchmark
```

## What you'll see

```
  📊 COMPARISON
                              CoT base       Reflex tuned
  ──────────────────────────────────────────────────────
  Avg latency                   2400ms              280ms
  Avg output tokens              142.3               11.8
  Tokens/sec                        58                 42
  ──────────────────────────────────────────────────────
  ⚡ Latency speedup                              8.6x
  📉 Token reduction                              92%
```

(Numbers above are illustrative — actual results depend on your hardware and the size of the run.)

## What's happening

The CoT model generates something like:

```
The task asks me to click on "Submit". Looking at the screen,
I can see a blue button labeled "Submit" in the bottom-right
of the form. Its center is roughly at x=0.84, y=0.92.
Action: click(x=0.84, y=0.92)
```

That's ~50 words of monologue + 1 line of action. The reflex model generates:

```
click(x=0.84, y=0.92)
```

Same action, ~10× fewer tokens, proportionally faster inference. Both runs use the **same model weights** — the only differences are (a) the LoRA adapter and (b) the prompt.

## Files

| File | Purpose |
|------|---------|
| `prepare_data.py` | Streams AGUVIS, saves screenshots, writes `data/train.jsonl` |
| `train.py` | LoRA fine-tunes Qwen2-VL-2B via `mlx-vlm` |
| `benchmark.py` | Runs both modes on the same examples and prints the comparison |

## The point

This is a minimal test of one idea: **most of the latency in a "thinking" GUI agent is the thinking, and the thinking is recoverable as weights.** A reflex model isn't smarter than a CoT model — it's just been trained until the answer is fast enough that explicit reasoning becomes optional. For UI control loops where you want 30+ actions per second, that matters.
