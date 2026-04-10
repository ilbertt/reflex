"""
Benchmark: Reflex (fine-tuned) vs CoT (prompted base) — same Qwen2-VL model.

Both modes get the same image + same instruction. The difference is what
the model is asked to produce:

  REFLEX (fine-tuned):  emit the action call directly (~5–15 tokens)
  COT    (base prompt): "think step by step, then output the action"
                        (~80–200 tokens of monologue + the action)

We measure wall-clock latency and number of generated tokens. The thesis:
fine-tuning collapses the reasoning into a reflex, so the same model gets
dramatically faster on the same task.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


DEFAULT_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"

# Reflex prompt is just the instruction — that's what the model was tuned on.
REFLEX_PROMPT = "{instruction}"

# CoT prompt asks the unmodified base model to reason aloud first.
COT_PROMPT = (
    "You are a GUI agent. Look at the screenshot.\n\n"
    "Task: {instruction}\n\n"
    "First, think step by step about which on-screen element matches the task "
    "and roughly where it sits. Then on a new line starting with 'Action:', "
    "output exactly one action call (e.g. click(x=0.41, y=0.18))."
)


def load_test_examples(path: Path, n: int) -> list[dict]:
    with open(path) as f:
        examples = [json.loads(line) for line in f]
    return examples[-n:]


def run_one(model, processor, config, image_path, prompt_text, max_tokens, generate, apply_chat_template):
    """Run one inference and return the GenerationResult plus wall time.

    The GenerationResult exposes prompt_tokens / prompt_tps (prefill) and
    generation_tokens / generation_tps (decode) separately, so we can
    measure them as independent quantities instead of conflating them
    into a single wall-clock number.
    """
    formatted = apply_chat_template(processor, config, prompt_text, num_images=1)
    t0 = time.time()
    result = generate(
        model, processor, formatted,
        image=[image_path],
        max_tokens=max_tokens,
        temp=0.0,
        verbose=False,
    )
    wall = time.time() - t0
    return result, wall


def benchmark_mode(label, model_name, adapter_path, prompt_tpl, max_tokens, test, results):
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    import mlx.core as mx

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Loading {model_name}" + (f" + adapter {adapter_path}" if adapter_path else ""))

    if adapter_path:
        model, processor = load(model_name, adapter_path=adapter_path)
    else:
        model, processor = load(model_name)
    config = load_config(model_name)

    prefill_ms, gen_ms, wall_ms = [], [], []
    gen_tokens, prompt_tokens = [], []
    outputs = []

    # Warmup
    warm = test[0]
    run_one(
        model, processor, config, warm["image"],
        prompt_tpl.format(instruction=warm["instruction"]),
        max_tokens, generate, apply_chat_template,
    )

    for i, ex in enumerate(test):
        prompt_text = prompt_tpl.format(instruction=ex["instruction"])
        result, wall = run_one(
            model, processor, config, ex["image"],
            prompt_text, max_tokens, generate, apply_chat_template,
        )

        # Decompose latency into prefill and decode using mlx-vlm's reported tps.
        # prompt_tps and generation_tps are tokens-per-second for each phase.
        pf_ms = (result.prompt_tokens / result.prompt_tps * 1000) if result.prompt_tps else 0.0
        gn_ms = (result.generation_tokens / result.generation_tps * 1000) if result.generation_tps else 0.0

        prefill_ms.append(pf_ms)
        gen_ms.append(gn_ms)
        wall_ms.append(wall * 1000)
        gen_tokens.append(result.generation_tokens)
        prompt_tokens.append(result.prompt_tokens)
        outputs.append(result.text)

        if i < 3:
            preview = result.text.replace("\n", " ")[:90]
            print(f"  [{i+1}] prefill {pf_ms:.0f}ms + gen {gn_ms:.0f}ms ({result.generation_tokens} tok): {preview}")
        elif i == 3:
            print("  ...")

    def avg(xs): return sum(xs) / len(xs)

    results[label] = {
        "prefill_ms": avg(prefill_ms),
        "generation_ms": avg(gen_ms),
        "wall_ms": avg(wall_ms),
        "prompt_tokens": avg(prompt_tokens),
        "generation_tokens": avg(gen_tokens),
        "gen_tps": avg(gen_tokens) / (avg(gen_ms) / 1000) if avg(gen_ms) > 0 else 0,
    }

    print(f"\n  Prefill:    {avg(prefill_ms):.0f}ms  ({avg(prompt_tokens):.0f} image+text tokens)")
    print(f"  Generation: {avg(gen_ms):.0f}ms  ({avg(gen_tokens):.1f} tokens out)")
    print(f"  Wall clock: {avg(wall_ms):.0f}ms")

    del model, processor
    mx.clear_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Test examples")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default="models/reflex")
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--reflex-max-tokens", type=int, default=64)
    parser.add_argument("--cot-max-tokens", type=int, default=256)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ {data_path} not found. Run: uv run prepare-data")
        sys.exit(1)

    test = load_test_examples(data_path, args.n)
    print(f"Test examples: {len(test)}")

    has_adapter = os.path.exists(args.adapter)
    if not has_adapter:
        print(f"⚠️  No adapter at {args.adapter} — running CoT baseline only.")
        print("   Run `uv run train` first to get the reflex model.")

    results: dict = {}

    if has_adapter:
        benchmark_mode(
            "REFLEX (fine-tuned)", args.model, args.adapter,
            REFLEX_PROMPT, args.reflex_max_tokens, test, results,
        )

    benchmark_mode(
        "COT (prompted base)", args.model, None,
        COT_PROMPT, args.cot_max_tokens, test, results,
    )

    if "REFLEX (fine-tuned)" in results and "COT (prompted base)" in results:
        r = results["REFLEX (fine-tuned)"]
        c = results["COT (prompted base)"]

        gen_speedup = c["generation_ms"] / r["generation_ms"] if r["generation_ms"] > 0 else 0
        wall_speedup = c["wall_ms"] / r["wall_ms"] if r["wall_ms"] > 0 else 0
        token_red = 1 - (r["generation_tokens"] / c["generation_tokens"]) if c["generation_tokens"] > 0 else 0

        print(f"\n{'='*64}")
        print("  📊 COMPARISON")
        print(f"{'='*64}")
        print(f"  {'':24}{'CoT base':>18}{'Reflex tuned':>18}")
        print(f"  {'-'*62}")
        print(f"  {'Prefill (image+text)':24}{c['prefill_ms']:>15.0f}ms{r['prefill_ms']:>15.0f}ms")
        print(f"  {'Generation (decode)':24}{c['generation_ms']:>15.0f}ms{r['generation_ms']:>15.0f}ms")
        print(f"  {'Wall clock total':24}{c['wall_ms']:>15.0f}ms{r['wall_ms']:>15.0f}ms")
        print(f"  {'Output tokens':24}{c['generation_tokens']:>17.1f}{r['generation_tokens']:>17.1f}")
        print(f"  {'-'*62}")
        print(f"  ⚡ Generation speedup    {gen_speedup:>34.1f}x   ← thesis claim")
        print(f"  ⏱  Wall-clock speedup    {wall_speedup:>34.1f}x   ← user-perceived")
        print(f"  📉 Token reduction       {token_red:>33.0%}")
        print(f"{'='*64}\n")
        print(f"  Generation alone:  {gen_speedup:.1f}× faster (the part the thesis is about)")
        print(f"  End-to-end:        {wall_speedup:.1f}× faster (image prefill is shared cost)\n")

    Path("results.json").write_text(json.dumps(results, indent=2))
    print("Results → results.json")


if __name__ == "__main__":
    main()
