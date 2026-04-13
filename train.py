"""
Train a direct action head on a frozen VLM backbone.

Phase 1 — Cache hidden states:
    Runs each (image, instruction) through the frozen Qwen2-VL backbone
    and saves the last-token hidden vectors to disk. ~15 min, one time.

Phase 2 — Train the action head:
    Loads cached vectors and trains a small MLP to predict
    (action_type, x, y, x2, y2) directly. ~30 sec, repeatable.

The result is a ~5 MB head that replaces the entire autoregressive
decode loop with a single MLP evaluation. No tokens generated.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

DEFAULT_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-bf16"

# ── Action parsing ───────────────────────────────────────────────────────────

ACTION_TYPES = ["click", "double_click", "right_click", "move_mouse", "drag"]
ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_TYPES)}

_COORD_RE = re.compile(
    r"(click|double_click|right_click|move_mouse)"
    r"\(x=([0-9.]+),\s*y=([0-9.]+)\)"
)
_DRAG_RE = re.compile(
    r"drag\(from_coord=\[([0-9.]+),\s*([0-9.]+)\],"
    r"\s*to_coord=\[([0-9.]+),\s*([0-9.]+)\]\)"
)


def parse_action(s: str):
    """AGUVIS action string → (type_idx, [x, y, x2, y2]) or None."""
    s = s.strip()
    m = _COORD_RE.match(s)
    if m:
        return ACTION_TO_IDX[m.group(1)], [float(m.group(2)), float(m.group(3)), 0.0, 0.0]
    m = _DRAG_RE.match(s)
    if m:
        return ACTION_TO_IDX["drag"], [float(g) for g in m.groups()]
    return None


# ── Head definition ──────────────────────────────────────────────────────────

class ReflexHead(nn.Module):
    """Small MLP: frozen hidden state → structured action.

    Outputs:
        type_logits: [B, n_types]
        xy: [B, 4]  — (x, y, x2, y2) in [0,1] via sigmoid
    """
    def __init__(self, hidden_dim: int, n_types: int = 5, inner: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, inner)
        self.fc2 = nn.Linear(inner, inner)
        self.type_head = nn.Linear(inner, n_types)
        self.xy_head = nn.Linear(inner, 4)

    def __call__(self, hidden):
        h = self.norm(hidden)
        h = nn.gelu(self.fc1(h))
        h = nn.gelu(self.fc2(h))
        return self.type_head(h), mx.sigmoid(self.xy_head(h))


# ── Hidden state extraction ─────────────────────────────────────────────────

def forward_to_hidden(model, processor, config, image_path, instruction):
    """Run (image, instruction) through the frozen VLM, return last-token hidden state.

    Uses Qwen2-VL's internal structure:
      1. Processor tokenizes text + processes image → input_ids, pixel_values, grid_thw
      2. model.get_input_embeddings merges vision + text into inputs_embeds
      3. model.language_model.model (the Qwen2 transformer) runs WITHOUT lm_head
      4. We take the last token's hidden state (shape: [hidden_dim])
    """
    from PIL import Image
    from mlx_vlm.prompt_utils import apply_chat_template

    # Format prompt with chat template
    prompt = apply_chat_template(processor, config, instruction, num_images=1)

    # Process image + text through HF processor.
    # Resize to 960x540 to cut image tokens from ~2700 to ~700,
    # reducing prefill from ~9s to ~2s per example. AGUVIS coordinates
    # are normalized [0,1] so the resize doesn't affect targets.
    image = Image.open(image_path).convert("RGB").resize((960, 540))
    inputs = processor(text=[prompt], images=[image], return_tensors="np")

    input_ids = mx.array(inputs["input_ids"])
    pixel_values = mx.array(inputs["pixel_values"])
    image_grid_thw = mx.array(inputs["image_grid_thw"])

    # Merge vision + text embeddings
    emb_output = model.get_input_embeddings(
        input_ids, pixel_values, image_grid_thw=image_grid_thw
    )

    # Position IDs are stored as a side effect of get_input_embeddings
    position_ids = model.language_model._position_ids

    # Forward through transformer backbone — no lm_head, no decode loop
    hidden_states = model.language_model.model(
        input_ids,
        inputs_embeds=emb_output.inputs_embeds,
        cache=None,
        position_ids=position_ids,
    )
    # hidden_states: [1, seq_len, hidden_dim], already post-RMSNorm

    # Concatenate two complementary representations:
    #
    # - last token: carries instruction intent (what to click) via causal
    #   attention over the full sequence, but is dominated by the "assistant
    #   turn starting" template signal — weak alone.
    #
    # - mean pool: carries image context (which screenshot) via averaged
    #   image patch activations, but identical for all instructions on the
    #   same screenshot — also weak alone.
    #
    # Together they give the head both "which screen" and "which element",
    # which is what it needs to predict both action type and coordinates.
    last_token = hidden_states[0, -1, :]        # [hidden_dim]
    mean_pool = hidden_states[0].mean(axis=0)   # [hidden_dim]
    return mx.concatenate([last_token, mean_pool])  # [2 * hidden_dim]


# ── Phase 1: Cache hidden states ────────────────────────────────────────────

def cache_hidden_states(model, processor, config, examples, cache_path):
    """Run the frozen VLM on every example once, save hidden states to disk."""
    all_hidden = []
    all_types = []
    all_xy = []
    all_drag = []

    print(f"  Caching {len(examples)} hidden states...")
    t0 = time.time()

    for i, ex in enumerate(examples):
        h = forward_to_hidden(model, processor, config, ex["image"], ex["instruction"])
        mx.eval(h)  # force computation before moving on
        all_hidden.append(np.array(h.astype(mx.float32)))
        all_types.append(ex["type_idx"])
        all_xy.append(ex["xy"])
        all_drag.append(1.0 if ex["type_idx"] == ACTION_TO_IDX["drag"] else 0.0)

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(examples) - i - 1)
            print(f"  [{i+1}/{len(examples)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    np.savez(
        cache_path,
        hidden=np.stack(all_hidden),
        action_type=np.array(all_types, dtype=np.int32),
        xy=np.array(all_xy, dtype=np.float32),
        drag_mask=np.array(all_drag, dtype=np.float32),
    )
    elapsed = time.time() - t0
    print(f"  ✅ Cached {len(examples)} states in {elapsed:.0f}s → {cache_path}")


# ── Phase 2: Train head ─────────────────────────────────────────────────────

def train_head(cache_path, output_path, hidden_dim, steps, lr, inner_dim):
    """Load cached vectors, train the action head with full-batch gradient descent."""
    data = np.load(cache_path)
    hidden = mx.array(data["hidden"])      # [N, hidden_dim]
    target_type = mx.array(data["action_type"])  # [N]
    target_xy = mx.array(data["xy"])       # [N, 4]
    drag_mask = mx.array(data["drag_mask"])  # [N]
    n = hidden.shape[0]

    head = ReflexHead(hidden_dim=hidden_dim, inner=inner_dim)
    optimizer = optim.Adam(learning_rate=lr)

    def batch_loss(head):
        type_logits, xy = head(hidden)  # [N, 5], [N, 4]
        ce = nn.losses.cross_entropy(type_logits, target_type, reduction="mean")
        mse_primary = ((xy[:, :2] - target_xy[:, :2]) ** 2).mean()
        mse_drag = (
            ((xy[:, 2:] - target_xy[:, 2:]) ** 2) * drag_mask[:, None]
        ).sum() / mx.maximum(drag_mask.sum(), 1.0)
        return ce + mse_primary + mse_drag

    loss_and_grad = nn.value_and_grad(head, batch_loss)

    print(f"\n  Training head on {n} cached vectors, {steps} steps (full-batch)...")
    t0 = time.time()

    for step in range(steps):
        loss, grads = loss_and_grad(head)
        optimizer.update(head, grads)
        mx.eval(head.parameters(), optimizer.state)

        if step % 100 == 0 or step == steps - 1:
            print(f"  Step {step}: loss {loss.item():.4f}")

    elapsed = time.time() - t0
    print(f"  ✅ Trained in {elapsed:.1f}s")

    # Save head weights
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(head.parameters()))
    mx.savez(str(out), **weights)
    print(f"  Saved head → {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")

    return head


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train a reflex action head")
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache", default="models/reflex/hidden_cache.npz")
    parser.add_argument("--output", default="models/reflex/head.npz")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--inner-dim", type=int, default=256)
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit examples for caching (default: all)")
    parser.add_argument("--skip-cache", action="store_true",
                        help="Skip phase 1 if cache already exists")
    args = parser.parse_args()

    # Load and parse training data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ {data_path} not found. Run: uv run prepare-data")
        sys.exit(1)

    examples = []
    skipped = 0
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line)
            parsed = parse_action(ex["action"])
            if parsed:
                ex["type_idx"], ex["xy"] = parsed
                examples.append(ex)
            else:
                skipped += 1

    if args.max_examples:
        examples = examples[:args.max_examples]

    print(f"Loaded {len(examples)} examples ({skipped} skipped — non-coordinate actions)")

    # Phase 1: Cache hidden states
    cache_path = Path(args.cache)
    if args.skip_cache and cache_path.exists():
        print(f"\n⏭  Skipping cache — using existing {cache_path}")
    else:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        print(f"\n📦 Phase 1: Caching hidden states")
        print(f"   Model: {args.model}")
        model, processor = load(args.model)
        config = load_config(args.model)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_hidden_states(model, processor, config, examples, str(cache_path))

        # Free backbone memory before head training
        del model, processor
        mx.clear_cache()

    # Phase 2: Train head
    print(f"\n🧠 Phase 2: Training action head")
    # Determine hidden dim from cache
    data = np.load(str(cache_path))
    hidden_dim = data["hidden"].shape[1]
    data.close()

    head = train_head(
        str(cache_path), args.output,
        hidden_dim=hidden_dim,
        steps=args.steps,
        lr=args.lr,
        inner_dim=args.inner_dim,
    )

    # Quick sanity check: run head on first 5 cached examples
    data = np.load(str(cache_path))
    hidden = mx.array(data["hidden"][:5])
    data.close()

    print(f"\n📋 Sanity check (first 5):")
    for i in range(5):
        type_logits, xy = head(hidden[i][None])
        pred_type = ACTION_TYPES[mx.argmax(type_logits[0]).item()]
        pred_xy = xy[0].tolist()
        truth = examples[i]
        print(f"  [{i+1}] pred: {pred_type}(x={pred_xy[0]:.3f}, y={pred_xy[1]:.3f})")
        print(f"       true: {truth['action']}".rstrip())

    print(f"\nDone. Next: uv run benchmark")


if __name__ == "__main__":
    main()
