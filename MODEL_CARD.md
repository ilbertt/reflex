---
license: mit
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
tags:
  - riscv
  - cross-attention
  - flamingo
  - grounded
  - jepa
  - rv32i
  - code-generation
library_name: transformers
pipeline_tag: text-generation
---

# Reflex-Coder7B-JEPA-RISCV

**A frozen `Qwen2.5-Coder-7B-Instruct` wired to a RISC-V CPU through Flamingo-style cross-attention. Emits one 32-bit RV32I instruction per cycle, conditioned on live machine state. The output head is a JEPA-style embedding predictor over a learned 691-row instruction codebook; nearest-neighbour decode gives free error-correction on individual predictions.**

This repo contains the **adapter weights only** (~4.4 GB). The frozen backbone is pulled from `Qwen/Qwen2.5-Coder-7B-Instruct` at runtime. Total inference footprint: ~14 GB bf16 backbone + 4.4 GB adapters + activations.

## What it does

Given a natural-language prompt (`"multiply 7 and 8"`, `"compute 5 factorial"`, `"say hi"`), Reflex drives a Unicorn-backed RV32I emulator instruction by instruction. Each cycle:

1. Read live CPU state (32 registers, PC, memory windows around PC and SP).
2. Encode as 65 K/V tokens.
3. Run the frozen backbone forward over the prompt; cross-attn adapters fuse state K/V into hidden states every 4 layers.
4. Last-token pool → MLP → 256-d embedding.
5. Cosine nearest-neighbour against a 691-row instruction codebook → a real 32-bit RV32I word.
6. Write the word at PC in Unicorn, step one cycle, loop.

## Base model

[`Qwen/Qwen2.5-Coder-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) — frozen, bf16, untouched.

## Training

- **Corpus**: 80,396 `(prompt, program)` pairs across 56 RV32I program families (arithmetic, loops, comparisons, memory ops, display writes). Every program verified by running it end-to-end through Unicorn before training (zero rejects).
- **Flattened cycle pool**: ~1.06 M `(state, next_instruction)` pairs, subsampled to ~173 k balanced across families.
- **Objective**: InfoNCE (temperature τ = 0.07) over the full 691-row instruction codebook. The codebook rows train jointly with the controller.
- **Optimizer**: AdamW (weight decay 0.01), cosine LR `1e-4 → 1e-6` over 15 000 steps, batch 16.
- **Hardware**: single A100 80 GB (~4 h) or L40S 48 GB (batch 32, ~5 h).

## Results (41-task eval)

| section | pass |
|---|---|
| in-distribution (8) | **7 / 8** |
| out-of-distribution (10) | **9 / 10** |
| display strings (4) | 1 / 4 |
| novel zero-shot (9) | **7 / 9** |
| consistency: factorial 5 × 10 | **10 / 10** |
| **total** | **34 / 41 (83 %)** |

Highlights:
- **`popcount(255) = 8` in 199 consecutive correct RISC-V instructions** — emergent bit-counting loop the model was never trained on.
- **Factorial 5 × 10 = 120, deterministic** — every run emits exactly 91 ops and lands on the right answer.
- Zero-shot `multiply 7×8`, `power 2^5`, `min(7,3,9)`, `abs(-5)`, `count up 1..5` all pass.

Per-step top-1 instruction accuracy on 500 random held-out cycles: **96.0 %**. All `BRANCH`, `R-type`, `LOAD`, `STORE`, `JAL`, `JALR` predictions are 100 %. **Every top-1 miss is same-opcode** — never an opcode flip.

## Usage

```python
from reflex.demo import load, run_grounded

model, tok, cfg = load("reflex.pt", device="cuda")
cpu, emitted, halted, err = run_grounded(
    model, tok, "multiply 7 and 8", device="cuda", max_cycles=200,
)
print(f"halted={halted}  mem[0x5000]={cpu.mem_word(0x5000)}")
# halted=True  mem[0x5000]=56
```

Or, interactively:

```bash
uv run demo --ckpt reflex.pt
```

## Installation

```bash
git clone https://github.com/ilbertt/reflex
cd reflex
uv sync
huggingface-cli download ilbert/reflex-coder7b-jepa-riscv reflex.pt --local-dir .
```

On first run, HuggingFace will automatically fetch `Qwen2.5-Coder-7B-Instruct` (~15 GB).

## Limitations

- **Display byte-constants are unreliable.** The model picks ASCII neighbours: `show 42` writes `'·0'` instead of `'42'`; `print hello` writes `'hell·'`. These are same-opcode ±1-immediate misses, not opcode flips.
- **Uncommon-literal arithmetic drifts.** `add 100+200` sometimes halts with 120; `double 100` → 0 in some seeds. Failures concentrate on `ADDI`/`LUI` with rare immediate values.
- **Closed action space.** The codebook has exactly 691 rows — instructions never seen in training have no row and cannot be emitted. Ample for the 56 program families trained on; bounds generalisation to genuinely unseen opcodes.
- **No domain-knowledge transfer.** Prompts like `"x5 is fever, display SICK"` fail. The adapters only route the backbone's prior through for program-shaped prompts seen in training.
- **RV32I base ISA only.** No M, Zbb, F extensions.

## Files

- `reflex.pt` — adapter weights, state encoder, cross-attn adapters, embedding head, 691-row instruction codebook, instruction-word buffer, and config dict (`backbone_id`, `hidden`, `inject_every`, `adapter_mlp_ratio`, `max_instr_tokens`, `embed_dim`, `num_instrs`, `chat_template`, `context_prefix`).
