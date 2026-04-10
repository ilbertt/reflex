"""
Prepare AGUVIS data for VLM fine-tuning.

Streams smolagents/aguvis-stage-1, saves screenshots to disk,
and writes a single JSONL of {image, instruction, action} examples.

The action format is left exactly as AGUVIS produced it
(e.g. 'click(x=0.41, y=0.178)') — it's already compact, no custom
vocabulary needed.
"""

import argparse
import json
import os
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


# Match any AGUVIS action call. Allows one nesting level of [...] (drag uses it).
ACTION_RE = re.compile(
    r'((?:click|double_click|right_click|move_mouse|drag|type|scroll|press|hotkey)'
    r'\s*\([^()]*(?:\[[^\]]*\][^()]*)*\))'
)


def prepare(max_examples: int, output_dir: str, config: str):
    out = Path(output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Streaming AGUVIS stage-1 ({config})...")
    ds = load_dataset(
        "smolagents/aguvis-stage-1", config, split="train", streaming=True
    )

    examples: list[dict] = []
    rows_seen = 0
    pbar = tqdm(total=max_examples, desc="Examples")

    for row_idx, item in enumerate(ds):
        rows_seen += 1
        images = item.get("images") or []
        texts = item.get("texts") or []
        if not images or not texts:
            continue

        # Save the screenshot once per row, reuse across all its turns
        img_path = img_dir / f"{row_idx:06d}.png"
        if not img_path.exists():
            images[0].save(img_path)

        for turn in texts:
            user = (turn.get("user") or "").strip()
            asst = (turn.get("assistant") or "").strip()
            if not user or not asst:
                continue

            instruction = re.sub(r"<image>|<\|image\|>", "", user).strip()
            actions = ACTION_RE.findall(asst)
            if not instruction or not actions:
                continue

            # Keep raw action format. If multiple actions, join with newline.
            action = actions[0] if len(actions) == 1 else "\n".join(actions)

            examples.append({
                "image": str(img_path),
                "instruction": instruction,
                "action": action,
            })
            pbar.update(1)
            if len(examples) >= max_examples:
                break

        if len(examples) >= max_examples:
            break

    pbar.close()

    out_file = out / "train.jsonl"
    with open(out_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    n_images = sum(1 for _ in img_dir.iterdir())
    print(f"\n✅ Wrote {len(examples)} examples from {rows_seen} rows")
    print(f"   {out_file}")
    print(f"   {n_images} images in {img_dir}/")

    if examples:
        ex = examples[0]
        print("\n📋 Sample:")
        print(f"   image:       {ex['image']}")
        print(f"   instruction: {ex['instruction'][:80]}")
        print(f"   action:      {ex['action']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=1000, help="Max training examples")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument(
        "--config",
        default="seeclick",
        choices=[
            "guienv", "omniact", "ricoig16k", "ricosca",
            "seeclick", "ui_refexp", "webui350k", "widget_captioning",
        ],
        help="AGUVIS stage-1 sub-config (default: seeclick — click-grounding heavy)",
    )
    args = parser.parse_args()
    prepare(max_examples=args.max, output_dir=args.output, config=args.config)

    # The HF streaming dataset spawns a prefetch worker that often gets
    # stuck in a retry loop after we break early, blocking interpreter
    # shutdown. Our work is fully written by this point, so force exit.
    os._exit(0)


if __name__ == "__main__":
    main()
