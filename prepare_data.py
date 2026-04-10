"""
Prepare training data in TWO formats from AGUVIS:

1. TEXT format (baseline): the model outputs natural language like
   'click(x=0.41, y=0.178)' — this is what existing agents do.

2. TOKEN format (kernel): the model outputs compact action tokens like
   'C0803' — this is our "kernel" approach.

Same screenshots, same actions, different output representations.
The benchmark will compare inference speed between the two.
"""

import json
import re
import os
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from action_tokens import encode_action

def extract_actions_from_conversation(conversations: list) -> tuple[str, list[str]]:
    """Extract the instruction and action calls from AGUVIS conversation format."""
    instruction = ""
    actions = []

    for msg in conversations:
        role = msg.get("role", msg.get("from", ""))
        content = msg.get("content", msg.get("value", ""))

        if role in ("user", "human"):
            # Extract the instruction text (strip image placeholders)
            text = re.sub(r'<image>|<\|image\|>', '', content).strip()
            if text:
                instruction = text

        elif role in ("assistant", "gpt"):
            # Extract action calls like click(x=0.41, y=0.178)
            # Handle both raw actions and <code>...</code> wrapped ones
            code_blocks = re.findall(r'<code>(.*?)</code>', content, re.DOTALL)
            if code_blocks:
                for block in code_blocks:
                    calls = re.findall(r'((?:click|type|scroll|press|hotkey)\s*\([^)]+\))', block)
                    actions.extend(calls)
            else:
                calls = re.findall(r'((?:click|type|scroll|press|hotkey)\s*\([^)]+\))', content)
                actions.extend(calls)

    return instruction, actions


def prepare(max_examples: int = 3000, output_dir: str = "data"):
    """Download AGUVIS and create both text and token training sets."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    print("📥 Loading AGUVIS stage-1 from HuggingFace...")
    ds = load_dataset("smolagents/aguvis-stage-1", split="train", streaming=True)

    text_examples = []   # baseline: natural language actions
    token_examples = []  # kernel: action tokens

    count = 0
    skipped = 0

    for item in tqdm(ds, total=max_examples, desc="Processing"):
        convos = item.get("conversations", item.get("messages", []))
        if not convos:
            skipped += 1
            continue

        instruction, actions = extract_actions_from_conversation(convos)
        if not actions or not instruction:
            skipped += 1
            continue

        # Text format: keep original action strings
        text_output = "\n".join(actions)

        # Token format: encode to action tokens
        token_output = " ".join(encode_action(a) for a in actions)

        # Validate token output isn't all fallbacks
        if all(t.startswith("?") for t in token_output.split()):
            skipped += 1
            continue

        text_examples.append({
            "instruction": instruction,
            "output": text_output,
            "n_actions": len(actions),
        })

        token_examples.append({
            "instruction": instruction,
            "output": token_output,
            "n_actions": len(actions),
        })

        count += 1
        if count >= max_examples:
            break

    # Save both formats
    for name, examples in [("text", text_examples), ("token", token_examples)]:
        path = out / f"train_{name}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

    # Show comparison
    print(f"\n{'='*60}")
    print(f"✅ Prepared {count} examples (skipped {skipped})")
    print(f"   Text format:  {out / 'train_text.jsonl'}")
    print(f"   Token format: {out / 'train_token.jsonl'}")
    print(f"{'='*60}")

    # Show a few examples side by side
    print(f"\n📋 Sample comparison:")
    for i in range(min(3, len(text_examples))):
        print(f"\n  Example {i+1}:")
        print(f"  Instruction: {text_examples[i]['instruction'][:80]}...")
        print(f"  TEXT output:  {text_examples[i]['output'][:80]}")
        print(f"  TOKEN output: {token_examples[i]['output'][:80]}")
        text_len = len(text_examples[i]['output'])
        token_len = len(token_examples[i]['output'])
        print(f"  Compression:  {text_len} chars → {token_len} chars ({token_len/text_len:.0%})")

    # Stats
    avg_text = sum(len(e['output']) for e in text_examples) / len(text_examples)
    avg_token = sum(len(e['output']) for e in token_examples) / len(token_examples)
    print(f"\n  Average output length:")
    print(f"    Text:  {avg_text:.0f} chars")
    print(f"    Token: {avg_token:.0f} chars ({avg_token/avg_text:.0%} of text)")
    print(f"    → Fewer tokens to generate = faster inference")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=3000, help="Max training examples")
    parser.add_argument("--output", default="data", help="Output directory")
    args = parser.parse_args()
    prepare(max_examples=args.max, output_dir=args.output)


if __name__ == "__main__":
    main()
