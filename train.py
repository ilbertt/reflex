"""
Train TWO versions of the same model:

1. text_model  — learns to output 'click(x=0.41, y=0.178)' (baseline)
2. token_model — learns to output 'C0803' (kernel)

Same base model, same data, different output format.
The benchmark will show that token_model generates faster
because it has fewer tokens to produce.

Usage:
    uv run train                              # trains both
    uv run train -- --format token            # train kernel only
    uv run train -- --format text             # train baseline only
    uv run train -- --epochs 2 --lr 1e-4      # tweak hyperparams
"""

import json
import argparse
import sys
import os
from pathlib import Path


def train_mlx(data_path: str, output_dir: str, model_name: str,
              epochs: int, lr: float, lora_rank: int):
    """LoRA fine-tune using MLX (Apple Silicon)."""
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.tuner import train as mlx_train
    from mlx_lm.tuner.utils import build_schedule

    print(f"  Loading {model_name}...")
    model, tokenizer = load(model_name)

    # Load our data
    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))

    # Format for mlx-lm: simple prompt/completion pairs
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    split = int(len(examples) * 0.9)
    for name, subset in [("train", examples[:split]), ("valid", examples[split:])]:
        fpath = out_path / f"{name}.jsonl"
        with open(fpath, "w") as f:
            for ex in subset:
                # Simple format: instruction → output
                prompt = f"Action: {ex['instruction']}\nOutput:"
                completion = f" {ex['output']}"
                f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")

    print(f"  Training: {split} examples, Validation: {len(examples) - split}")
    print(f"  Epochs: {epochs}, LR: {lr}, LoRA rank: {lora_rank}")

    # Train
    mlx_train(
        model=model,
        tokenizer=tokenizer,
        args=type('Args', (), {
            'data': str(out_path),
            'train': True,
            'adapter_file': str(out_path / "adapters.npz"),
            'iters': len(examples[:split]) * epochs,
            'batch_size': 1,
            'learning_rate': lr,
            'lora_layers': 8,
            'lora_rank': lora_rank,
            'val_batches': 10,
            'steps_per_report': 50,
            'steps_per_eval': 200,
            'save_every': 500,
            'max_seq_length': 512,
            'grad_checkpoint': False,
            'seed': 42,
        })(),
    )

    print(f"  ✅ Saved adapter to {out_path / 'adapters.npz'}")


def train_hf(data_path: str, output_dir: str, model_name: str,
             epochs: int, lr: float, lora_rank: int):
    """LoRA fine-tune using HuggingFace/PEFT (CUDA or CPU)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))

    class ActionDataset(Dataset):
        def __init__(self, examples, tokenizer, max_len=512):
            self.data = []
            for ex in examples:
                text = f"Action: {ex['instruction']}\nOutput: {ex['output']}"
                enc = tokenizer(text, truncation=True, max_length=max_len,
                                padding="max_length", return_tensors="pt")
                self.data.append({
                    "input_ids": enc["input_ids"].squeeze(),
                    "attention_mask": enc["attention_mask"].squeeze(),
                    "labels": enc["input_ids"].squeeze(),
                })
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i]

    split = int(len(examples) * 0.9)
    train_ds = ActionDataset(examples[:split], tokenizer)
    val_ds = ActionDataset(examples[split:], tokenizer)

    out_path = Path(output_dir)
    training_args = TrainingArguments(
        output_dir=str(out_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        learning_rate=lr,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
    )
    trainer.train()
    model.save_pretrained(str(out_path / "adapter"))
    print(f"  ✅ Saved adapter to {out_path / 'adapter'}")


def train_one(fmt: str, backend: str, model_name: str,
              epochs: int, lr: float, lora_rank: int):
    """Train one format (text or token)."""
    data_path = f"data/train_{fmt}.jsonl"
    output_dir = f"models/{fmt}_model"

    if not os.path.exists(data_path):
        print(f"❌ {data_path} not found. Run: uv run prepare-data")
        sys.exit(1)

    n_examples = sum(1 for _ in open(data_path))
    print(f"\n{'='*50}")
    print(f"  Training {fmt.upper()} model")
    print(f"  Data: {data_path} ({n_examples} examples)")
    print(f"  Output: {output_dir}")
    print(f"{'='*50}\n")

    if backend == "mlx":
        train_mlx(data_path, output_dir, model_name, epochs, lr, lora_rank)
    else:
        train_hf(data_path, output_dir, model_name, epochs, lr, lora_rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["text", "token", "both"], default="both")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base LM (use a small one for fast training)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    args = parser.parse_args()

    # Auto-detect backend
    backend = "mlx"
    try:
        import mlx
    except ImportError:
        backend = "hf"
    print(f"Backend: {backend}")

    if args.format in ("text", "both"):
        train_one("text", backend, args.model, args.epochs, args.lr, args.lora_rank)
    if args.format in ("token", "both"):
        train_one("token", backend, args.model, args.epochs, args.lr, args.lora_rank)

    if args.format == "both":
        print(f"\n{'='*50}")
        print(f"  ✅ Both models trained!")
        print(f"  Next: uv run benchmark")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
