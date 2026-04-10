"""
Benchmark: Text model vs Token model (kernel)

Feeds the same instructions to both models and measures:
  - Time to generate the output (inference latency)
  - Number of tokens generated
  - Tokens per second

The token model should be faster because it generates fewer tokens
for the same action — 'C0803' (5 chars) vs 'click(x=0.41, y=0.178)' (23 chars).

Usage:
    uv run benchmark                           # run benchmark
    uv run benchmark -- --n 100                # test on 100 examples
    uv run benchmark -- --model Qwen/...       # specify base model
"""

import json
import time
import sys
import os
import argparse
from pathlib import Path
from action_tokens import decode_action_sequence


def load_model_mlx(model_name: str, adapter_path: str = None):
    """Load model with MLX."""
    from mlx_lm import load, generate
    model, tokenizer = load(model_name, adapter_path=adapter_path)
    return model, tokenizer, generate


def load_model_hf(model_name: str, adapter_path: str = None):
    """Load model with HuggingFace."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    def generate_fn(model, tokenizer, prompt="", max_tokens=128, **kwargs):
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens,
                                  do_sample=False, temperature=0.1)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)

    return model, tokenizer, generate_fn


def run_inference(model, tokenizer, generate_fn, prompt: str,
                  max_tokens: int = 128, backend: str = "mlx") -> tuple[str, float, int]:
    """Run inference, return (output, time_seconds, n_tokens)."""
    t0 = time.time()

    if backend == "mlx":
        output = generate_fn(model, tokenizer, prompt=prompt,
                             max_tokens=max_tokens, temp=0.1)
    else:
        output = generate_fn(model, tokenizer, prompt=prompt,
                             max_tokens=max_tokens)

    elapsed = time.time() - t0
    n_tokens = len(tokenizer.encode(output))

    return output.strip(), elapsed, n_tokens


def benchmark(n_examples: int = 50, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Run the benchmark."""

    # Check data exists
    for fmt in ("text", "token"):
        if not os.path.exists(f"data/train_{fmt}.jsonl"):
            print(f"❌ data/train_{fmt}.jsonl not found. Run: uv run prepare-data")
            sys.exit(1)

    # Detect backend
    backend = "mlx"
    try:
        import mlx
    except ImportError:
        backend = "hf"

    # Load test examples (use last N from dataset — not seen during training if split was 90/10)
    text_examples = []
    with open("data/train_text.jsonl") as f:
        for line in f:
            text_examples.append(json.loads(line))
    token_examples = []
    with open("data/train_token.jsonl") as f:
        for line in f:
            token_examples.append(json.loads(line))

    # Use the last N as test set
    test_text = text_examples[-n_examples:]
    test_token = token_examples[-n_examples:]

    # Load both models
    text_adapter = "models/text_model/adapters.npz" if backend == "mlx" else "models/text_model/adapter"
    token_adapter = "models/token_model/adapters.npz" if backend == "mlx" else "models/token_model/adapter"

    has_text_model = os.path.exists(text_adapter)
    has_token_model = os.path.exists(token_adapter)

    if not has_text_model and not has_token_model:
        print("❌ No trained models found. Run: uv run train")
        sys.exit(1)

    loader = load_model_mlx if backend == "mlx" else load_model_hf

    print(f"\n{'='*60}")
    print(f"  ⚡ ACTION KERNEL BENCHMARK")
    print(f"  Base model: {model_name}")
    print(f"  Backend: {backend}")
    print(f"  Test examples: {n_examples}")
    print(f"{'='*60}")

    results = {}

    for fmt, adapter, test_data, has_model in [
        ("text", text_adapter, test_text, has_text_model),
        ("token", token_adapter, test_token, has_token_model),
    ]:
        if not has_model:
            print(f"\n  ⏭ Skipping {fmt} model (not trained)")
            continue

        label = "BASELINE (text)" if fmt == "text" else "KERNEL (tokens)"
        print(f"\n{'─'*60}")
        print(f"  Loading {label}...")
        model, tokenizer, generate_fn = loader(model_name, adapter)

        times = []
        token_counts = []
        output_chars = []

        # Warmup
        prompt = f"Action: {test_data[0]['instruction']}\nOutput:"
        run_inference(model, tokenizer, generate_fn, prompt, backend=backend)

        print(f"  Running {n_examples} inferences...")

        for i, ex in enumerate(test_data):
            prompt = f"Action: {ex['instruction']}\nOutput:"
            output, elapsed, n_tok = run_inference(
                model, tokenizer, generate_fn, prompt, backend=backend
            )
            times.append(elapsed)
            token_counts.append(n_tok)
            output_chars.append(len(output))

            if i < 3:
                print(f"    [{i+1}] {elapsed*1000:.0f}ms, {n_tok} tok: {output[:60]}...")
            elif i == 3:
                print(f"    ...")

        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        avg_chars = sum(output_chars) / len(output_chars)
        p50 = sorted(times)[len(times) // 2]
        p95 = sorted(times)[int(len(times) * 0.95)]

        results[fmt] = {
            "avg_time_ms": avg_time * 1000,
            "p50_ms": p50 * 1000,
            "p95_ms": p95 * 1000,
            "avg_tokens": avg_tokens,
            "avg_chars": avg_chars,
            "tok_per_sec": avg_tokens / avg_time if avg_time > 0 else 0,
        }

        print(f"\n  {label} results:")
        print(f"    Avg latency:     {avg_time*1000:.0f}ms")
        print(f"    P50 latency:     {p50*1000:.0f}ms")
        print(f"    P95 latency:     {p95*1000:.0f}ms")
        print(f"    Avg tokens out:  {avg_tokens:.1f}")
        print(f"    Avg chars out:   {avg_chars:.0f}")
        print(f"    Tokens/sec:      {avg_tokens/avg_time:.0f}")

        # Cleanup to free memory before loading next model
        del model
        if backend == "mlx":
            import mlx.core as mx
            mx.metal.clear_cache()
        else:
            import torch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Comparison ──
    if "text" in results and "token" in results:
        t = results["text"]
        k = results["token"]
        speedup = t["avg_time_ms"] / k["avg_time_ms"] if k["avg_time_ms"] > 0 else 0
        token_reduction = 1 - (k["avg_tokens"] / t["avg_tokens"]) if t["avg_tokens"] > 0 else 0

        print(f"\n{'='*60}")
        print(f"  📊 COMPARISON")
        print(f"{'='*60}")
        print(f"  {'':25} {'Text (baseline)':>18} {'Token (kernel)':>18}")
        print(f"  {'─'*61}")
        print(f"  {'Avg latency':25} {t['avg_time_ms']:>15.0f}ms {k['avg_time_ms']:>15.0f}ms")
        print(f"  {'P50 latency':25} {t['p50_ms']:>15.0f}ms {k['p50_ms']:>15.0f}ms")
        print(f"  {'P95 latency':25} {t['p95_ms']:>15.0f}ms {k['p95_ms']:>15.0f}ms")
        print(f"  {'Avg output tokens':25} {t['avg_tokens']:>17.1f} {k['avg_tokens']:>17.1f}")
        print(f"  {'Avg output chars':25} {t['avg_chars']:>17.0f} {k['avg_chars']:>17.0f}")
        print(f"  {'Tokens/sec':25} {t['tok_per_sec']:>17.0f} {k['tok_per_sec']:>17.0f}")
        print(f"  {'─'*61}")
        print(f"  {'⚡ Latency speedup':25} {speedup:>35.1f}x")
        print(f"  {'📉 Token reduction':25} {token_reduction:>34.0%}")
        print(f"{'='*60}")
        print()
        print(f"  The kernel model generates {token_reduction:.0%} fewer tokens,")
        print(f"  resulting in {speedup:.1f}x faster inference.")
        print(f"  Same actions, less 'talking', more 'doing'.")
        print()

    # Save results
    results_path = Path("results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of test examples")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()
    benchmark(n_examples=args.n, model_name=args.model)


if __name__ == "__main__":
    main()
