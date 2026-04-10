"""
LoRA fine-tune Qwen2-VL-2B on AGUVIS GUI actions via mlx-vlm.

Reads data/train.jsonl produced by prepare_data.py, rewrites it into the
column layout that mlx_vlm.lora expects (image / question / answer), then
invokes `python -m mlx_vlm.lora` as a subprocess.

The fine-tuned model learns to map (screenshot, instruction) → action call
directly. No reasoning, no monologue. That's the "reflex" thesis.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"


def write_mlx_dataset(in_path: Path, out_dir: Path) -> int:
    """Translate our {image, instruction, action} jsonl into mlx_vlm.lora's
    expected {image, question, answer} schema with absolute image paths.

    Sweeps any stale jsonl files first — load_dataset auto-merges every
    json/jsonl in the directory, so a leftover file with a different
    schema will fail the load with a CastError.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jsonl"):
        stale.unlink()

    with open(in_path) as f:
        examples = [json.loads(line) for line in f]

    out_path = out_dir / "train.jsonl"
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps({
                "image": str(Path(ex["image"]).resolve()),
                "question": ex["instruction"],
                "answer": ex["action"],
            }) + "\n")

    return len(examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--output", default="models/reflex")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ {data_path} not found. Run: uv run prepare-data")
        sys.exit(1)

    mlx_data_dir = Path("data/mlx")
    print("📦 Formatting data for mlx_vlm.lora...")
    n = write_mlx_dataset(data_path, mlx_data_dir)
    print(f"   {n} examples → {mlx_data_dir}/train.jsonl")

    # mlx_vlm.lora's --output-path is the full path of the adapter weights
    # *file*, not a directory. The adapter_config.json gets written next to
    # it. We use 'adapters.safetensors' because that's the exact filename
    # mlx_vlm.utils.load → apply_lora_layers looks for at load time.
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = out_dir / "adapters.safetensors"

    cmd = [
        sys.executable, "-m", "mlx_vlm.lora",
        "--model-path", args.model,
        "--dataset", str(mlx_data_dir.resolve()),
        "--split", "train",
        "--output-path", str(adapter_file.resolve()),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.lr),
        "--lora-rank", str(args.lora_rank),
        "--train-on-completions",
    ]

    print(f"\n🚀 Training {args.model}")
    print(f"   Epochs: {args.epochs}  LR: {args.lr}  LoRA rank: {args.lora_rank}")
    print(f"   {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n❌ mlx_vlm.lora exited with {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n✅ Adapter saved to {out_dir}/")
    print("   Next: uv run benchmark")


if __name__ == "__main__":
    main()
