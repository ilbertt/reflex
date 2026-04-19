---
license: mit
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
tags:
  - riscv
  - cross-attention
  - flamingo
  - grounded
  - code-generation
  - rv32i
library_name: transformers
pipeline_tag: text-generation
---

# Reflex-Coder7B-RISCV

**A frozen `Qwen2.5-Coder-7B-Instruct` wired to a RISC-V CPU through Flamingo-style cross-attention. Emits one 32-bit RV32I instruction per cycle, conditioned on live machine state. No text tokens generated at inference.**

This repo contains the **adapter weights only** (~4.2 GB fp32). The frozen backbone is pulled from `Qwen/Qwen2.5-Coder-7B-Instruct` at runtime. Total inference memory: ~18 GB bf16 backbone + 4.2 GB fp32 adapters + activations.

## What it does

Given a natural-language prompt (`"say hi"`, `"multiply 7 and 8"`, `"compute 5 factorial"`), Reflex drives a Unicorn-backed RV32I emulator instruction by instruction. Each cycle:

1. Read live CPU state (32 registers, PC, memory windows around PC and SP).
2. Encode as 65 K/V tokens.
3. Run the frozen backbone forward over the prompt, cross-attn adapters fuse state K/V into hidden states at depths 4, 8, 12, 16, 20, 24.
4. Last-token pool → MLP → 32 bit sigmoid heads → one 32-bit RV32I instruction word.
5. Write the word at PC in Unicorn, step one cycle, loop.

## Base model

[`Qwen/Qwen2.5-Coder-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) — frozen, bf16, untouched.

## Training

- **Corpus**: 80,396 (prompt, program) pairs across 56 RV32I program families (arithmetic, loops, comparisons, memory ops, display writes). Every program verified by running it end-to-end through Unicorn before training.
- **Flattened cycle pool**: ~1.06M `(state, next_instruction)` pairs. Balanced-subsampled to 173k across families per epoch.
- **Objective**: per-bit binary cross-entropy over the 32 instruction bits, with `rs2` bits (positions 20–24) weighted 5× to overcome the register/immediate polysemy ceiling.
- **Optimizer**: standard AdamW, cosine LR schedule `1e-4 → 1e-6` over 20k steps, batch 64.
- **Hardware**: A100 80GB.

## Results (18-task eval + 15-task sweep)

- **13 / 15 on a mixed zero-shot sweep** (see README), including six tasks the model was never trained on: multiply-by-repeated-add, power, abs, min, popcount, say-arbitrary-3-char-strings.
- **popcount(255) = 8 in 199 correct consecutive RISC-V instructions** — an emergent algorithm derived at inference from the frozen backbone's prior on what "popcount" means.
- Full eval script: `uv run eval --checkpoint reflex_coder7b.pt`.

## Usage

```python
import torch
from reflex.demo import load, run_grounded

model, tok, cfg = load("reflex_coder7b.pt", device="cuda")
cpu, emitted, halted, err = run_grounded(
    model, tok, "multiply 7 and 8", device="cuda", max_cycles=200,
)
print(f"halted={halted}  mem[0x5000]={cpu.mem_word(0x5000)}")
# halted=True  mem[0x5000]=56
```

Or, interactively:

```bash
uv run demo --checkpoint reflex_coder7b.pt
```

## Installation

```bash
git clone https://github.com/ilbert/reflex
cd reflex
uv sync
# Download this checkpoint into the repo root:
huggingface-cli download ilbert/reflex-coder7b-riscv reflex_coder7b.pt --local-dir .
```

The first time you run inference, HuggingFace will automatically fetch the frozen `Qwen2.5-Coder-7B-Instruct` backbone (~15 GB).

## Limitations

- **rs2 precision ceiling.** Per-cycle rs2 accuracy ~0.99; long loops (>50 ops) can emit a single-bit-wrong instruction that crashes the program before it stores its result.
- **No domain-knowledge transfer.** Reflex only knows the program-shaped phrasings in its training corpus. Prompts like `"if x5 is fever, display SICK"` fail — the adapters were never taught to route the backbone's semantic knowledge of "fever" through.
- **Display strings degrade past 3 characters.** `say hi`, `say 42`, `say wow` all land cleanly; `say hello` returns `hell`.
- **Some common phrasings are unreliable.** `add 100 and 200 and store the result` can return `100` instead of `300`. `subtract 10 from 25` sometimes returns `35` (semantic confusion on the word "from").
- **RV32I base ISA only** — no M (multiply/divide), no Zbb (count/bitmanip), no F (float). The model synthesizes all "higher" operations from base instructions.

## Files

- `reflex_coder7b.pt` — adapter weights, state encoder, head, and config dict (backbone_id, hidden, inject_every, adapter_mlp_ratio, max_instr_tokens, chat_template, context_prefix).

## Citation

```bibtex
@software{reflex2026,
  title  = {Reflex: wiring a frozen LLM to a CPU through cross-attention},
  author = {<your name>},
  year   = {2026},
  url    = {https://github.com/ilbert/reflex}
}
```

