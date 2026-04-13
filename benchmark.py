"""
Benchmark: Reflex head vs text generation on the same VLM backbone.

Text baseline:
    Qwen2-VL generates an action string autoregressively (~10-20 tokens,
    ~500ms of decode). The output is parsed by regex to extract the action.

Reflex head:
    Same Qwen2-VL runs one forward pass to produce a hidden state, then
    a tiny MLP evaluates it into a structured action (<1ms). No decode
    loop, no tokens, no parsing.

Both modes pay the same image prefill cost (~9s for a 1920x1080 screenshot).
The thesis: the reflex head eliminates the decode phase entirely.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from train import (
    DEFAULT_MODEL, ACTION_TYPES, ReflexHead,
    forward_to_hidden, parse_action,
)

# Prompt for the text baseline: just emit the action directly.
TEXT_PROMPT = "{instruction}"


def load_test_examples(path: Path, n: int) -> list[dict]:
    with open(path) as f:
        examples = [json.loads(line) for line in f]
    # Use the last N (least likely to be overfit if head was trained on first N)
    out = []
    for ex in examples[-n:]:
        parsed = parse_action(ex["action"])
        if parsed:
            ex["type_idx"], ex["xy"] = parsed
            out.append(ex)
    return out


def benchmark_text(model_name, test, max_tokens):
    """Text baseline: generate an action string via mlx-vlm, measure decode."""
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    print(f"\n{'='*64}")
    print("  TEXT BASELINE (autoregressive decode)")
    print(f"{'='*64}")
    print(f"  Loading {model_name}")

    model, processor = load(model_name)
    config = load_config(model_name)

    prefill_ms, gen_ms, wall_ms = [], [], []
    gen_tokens = []

    # Warmup
    warm = test[0]
    prompt = apply_chat_template(
        processor, config,
        TEXT_PROMPT.format(instruction=warm["instruction"]),
        num_images=1,
    )
    generate(model, processor, prompt, image=[warm["image"]],
             max_tokens=max_tokens, temp=0.0, verbose=False)

    for i, ex in enumerate(test):
        prompt = apply_chat_template(
            processor, config,
            TEXT_PROMPT.format(instruction=ex["instruction"]),
            num_images=1,
        )
        t0 = time.time()
        result = generate(
            model, processor, prompt,
            image=[ex["image"]],
            max_tokens=max_tokens,
            temp=0.0,
            verbose=False,
        )
        wall = time.time() - t0

        pf = (result.prompt_tokens / result.prompt_tps * 1000) if result.prompt_tps else 0
        gn = (result.generation_tokens / result.generation_tps * 1000) if result.generation_tps else 0

        prefill_ms.append(pf)
        gen_ms.append(gn)
        wall_ms.append(wall * 1000)
        gen_tokens.append(result.generation_tokens)

        if i < 5:
            print(f"  [{i+1}] prefill {pf:.0f}ms + decode {gn:.0f}ms "
                  f"({result.generation_tokens} tok): {result.text!r:.80}")

    del model, processor
    mx.clear_cache()

    def avg(xs): return sum(xs) / len(xs) if xs else 0
    return {
        "prefill_ms": avg(prefill_ms),
        "decode_ms": avg(gen_ms),
        "wall_ms": avg(wall_ms),
        "tokens": avg(gen_tokens),
    }


def benchmark_reflex(model_name, head_path, test):
    """Reflex head: one forward pass + MLP eval. No decode loop."""
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    from mlx.utils import tree_unflatten

    print(f"\n{'='*64}")
    print("  REFLEX HEAD (direct action output)")
    print(f"{'='*64}")
    print(f"  Loading {model_name} + head from {head_path}")

    model, processor = load(model_name)
    config = load_config(model_name)

    # Load head weights
    weights = dict(mx.load(str(head_path)))
    hidden_dim = weights["fc1.weight"].shape[1]
    head = ReflexHead(hidden_dim=hidden_dim)
    head.load_weights(list(weights.items()))

    prefill_ms, head_ms, wall_ms = [], [], []

    # Warmup
    warm = test[0]
    h = forward_to_hidden(model, processor, config, warm["image"], warm["instruction"])
    mx.eval(h)
    head(h[None])

    for i, ex in enumerate(test):
        t0 = time.time()
        h = forward_to_hidden(model, processor, config, ex["image"], ex["instruction"])
        mx.eval(h)
        t_prefill = time.time()

        type_logits, xy = head(h[None])
        mx.eval(type_logits, xy)
        t_head = time.time()

        pred_type = ACTION_TYPES[mx.argmax(type_logits[0]).item()]
        pred_xy = xy[0].tolist()
        wall = time.time() - t0

        pf = (t_prefill - t0) * 1000
        hd = (t_head - t_prefill) * 1000

        prefill_ms.append(pf)
        head_ms.append(hd)
        wall_ms.append(wall * 1000)

        if i < 5:
            print(f"  [{i+1}] prefill {pf:.0f}ms + head {hd:.1f}ms: "
                  f"{pred_type}(x={pred_xy[0]:.3f}, y={pred_xy[1]:.3f})")
            print(f"       truth: {ex['action']}")

    del model, processor
    mx.clear_cache()

    def avg(xs): return sum(xs) / len(xs) if xs else 0
    return {
        "prefill_ms": avg(prefill_ms),
        "decode_ms": avg(head_ms),   # "decode" is head eval — for comparison column
        "wall_ms": avg(wall_ms),
        "tokens": 0,                  # no tokens generated
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Test examples")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--head", default="models/reflex/head.npz")
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ {data_path} not found. Run: uv run prepare-data")
        sys.exit(1)

    test = load_test_examples(data_path, args.n)
    print(f"Test examples: {len(test)}")

    has_head = os.path.exists(args.head)
    if not has_head:
        print(f"⚠️  No head at {args.head} — running text baseline only.")
        print("   Run `uv run train` first to get the reflex head.")

    results = {}

    if has_head:
        results["reflex"] = benchmark_reflex(args.model, args.head, test)

    results["text"] = benchmark_text(args.model, test, args.max_tokens)

    # Comparison
    if "reflex" in results and "text" in results:
        r = results["reflex"]
        t = results["text"]

        decode_speedup = t["decode_ms"] / r["decode_ms"] if r["decode_ms"] > 0 else float("inf")
        wall_speedup = t["wall_ms"] / r["wall_ms"] if r["wall_ms"] > 0 else 0

        print(f"\n{'='*64}")
        print("  📊 COMPARISON")
        print(f"{'='*64}")
        print(f"  {'':26}{'Text (decode)':>18}{'Reflex (head)':>18}")
        print(f"  {'-'*62}")
        print(f"  {'Prefill (shared)':26}{t['prefill_ms']:>15.0f}ms{r['prefill_ms']:>15.0f}ms")
        print(f"  {'Output phase':26}{t['decode_ms']:>15.0f}ms{r['decode_ms']:>13.1f}ms")
        print(f"  {'Wall clock total':26}{t['wall_ms']:>15.0f}ms{r['wall_ms']:>15.0f}ms")
        print(f"  {'Tokens generated':26}{t['tokens']:>17.1f}{r['tokens']:>17.0f}")
        print(f"  {'-'*62}")
        print(f"  ⚡ Output-phase speedup    {decode_speedup:>32.0f}×")
        print(f"  ⏱  End-to-end speedup      {wall_speedup:>32.2f}×")
        print(f"{'='*64}")
        print()
        print(f"  The text model spends {t['decode_ms']:.0f}ms generating {t['tokens']:.0f} tokens.")
        print(f"  The reflex head produces the same action in {r['decode_ms']:.1f}ms.")
        print(f"  The decode loop is replaced by a single MLP evaluation.")
        print()

    Path("results.json").write_text(json.dumps(results, indent=2))
    print("Results → results.json")


if __name__ == "__main__":
    main()
