# Reflex

**Wire a frozen LLM to a CPU through cross-attention. No text generated — one RV32I instruction per cycle, conditioned on live machine state.**

## Architecture

```
                    ┌───────────────────────────────────────┐
                    │  PROMPT     "multiply 7 and 8"        │
                    │  (chat template tokens, seq ≤ 96)     │
                    └──────────────────┬────────────────────┘
                                       ▼
    ┌──────────────────────────────────────────────────────────────┐
    │         Qwen2.5-Coder-7B-Instruct  (FROZEN, bf16)            │
    │                                                              │
    │    ┌────────┐   ┌────────┐   ┌────────┐          ┌────────┐  │
    │    │ layer  │   │ layer  │   │ layer  │   ...    │ layer  │  │
    │    │   0    │→→→│   1    │→→→│   2    │→→→    →→→│   27   │  │
    │    └────────┘   └────────┘   └────────┘          └────────┘  │
    │         ▲ every 4 layers a forward hook injects:     ▲       │
    │         │        h ← h + tanh(α)·XAttn(LN(h), KV)    │       │
    │         │            + tanh(β)·MLP(LN(h))            │       │
    │         │   (7 adapters total, fp32, ~1.1B trained)  │       │
    └─────────┼────────────────────────────────────────────┼───────┘
              │                                            ▼
     ┌────────┴──────────┐                       ┌──────────────────┐
     │   STATE ENCODER   │                       │ last-token pool  │
     │    (trained)      │                       │        ▼         │
     │                   │                       │    2-layer MLP   │
     │  65 tokens × H:   │                       │        ▼         │
     │   x0..x31  (32)   │                       │  32 sigmoid heads│
     │   PC        (1)   │─── K,V ──→ XAttn ────▶│        ▼         │
     │   mem[±pc] (16)   │                       │  32-bit RV32I    │
     │   mem[±sp] (16)   │                       │   instruction    │
     └─────────▲─────────┘                       └─────────┬────────┘
               │                                           ▼
               │                                 write at PC; step
               │                                           │
               └───────────────────────────────────────────┘
                    RV32I Unicorn emulator (state feedback)
```

Each cycle: read the CPU's 32 registers + PC + memory windows, encode as 65 K/V tokens, run the backbone forward over the prompt with those K/V injected at every 4th layer, pool the last token, predict 32 instruction bits, write that word at PC, step Unicorn once. The state has now evolved; loop.

The backbone is **untouched**. Only a 1.1B-parameter stack (seven cross-attn adapters + state encoder + head) is trained. Inference emits no text tokens — the model's output channel is the 32-bit instruction word, executed immediately.

## Results

13 / 15 tasks pass with the released checkpoint. Six of the passes are **zero-shot novel** — the model was never trained on multiply, power, popcount, abs, min, or arbitrary display strings, but it composes them from grounded, cycle-by-cycle emission anyway.

| task | expected | got | ops | ✓/✗ |
|---|---|---|---|---|
| say hi | `hi` | `hi` | 6 | ✓ |
| say no | `no` | `no` | 6 | ✓ |
| say yes | `yes` | `yes` | 8 | ✓ |
| say wow | `wow` | `wow` | 8 | ✓ |
| say 42 | `42` | `42` | 6 | ✓ |
| say ok | `ok` | `ok"` | 8 | ✗ |
| **multiply 7 × 8** | 56 | **56** | 39 | ✓ |
| **multiply 5 × 8** | 40 | **40** | 39 | ✓ |
| **multiply 10 × 12** | 120 | **120** | 55 | ✓ |
| **abs of −5** | 5 | **5** | 6 | ✓ |
| **abs of −10** | 10 | **10** | 6 | ✓ |
| **min of 7, 3, 9** | 3 | **3** | 8 | ✓ |
| **power 2^5** | 32 | **32** | 82 | ✓ |
| **popcount(255)** | 8 | **8** | **199** | ✓ |
| count up 1..5 | `[1,2,3,4,5]` | `[1,2,3,4,0]` | 25 | ✗ |

**`popcount(255) = 8` was emitted in 199 consecutive correct RISC-V instructions** — a bitwise-loop algorithm Reflex was never trained on, derived at inference time from the backbone's prior on what "popcount" means. No text generation, no reasoning trace; just 199 grounded emissions, each conditioned on the post-step state of the previous one.

## How it works

**Architecture**: the backbone is `Qwen/Qwen2.5-Coder-7B-Instruct`, frozen in bf16. Seven [Flamingo-style gated cross-attention](https://arxiv.org/abs/2204.14198) adapters are spliced into the backbone's own transformer layers via forward hooks, one every four layers. Each adapter adds

```
hidden  ←  hidden + tanh(α)·CrossAttn(LN(hidden), state_kv)
        + tanh(β)·MLP(LN(hidden))
```

where `state_kv` is a 65-token K/V built from the live CPU state (32 registers + PC + 16-word window around PC + 16-word window around SP). Both tanh gates start at zero, so at step 0 the adapter is a no-op identity and the pretrained backbone activations are undisturbed.

The state conditions reasoning at every depth of the backbone, not just at the output head — that is what lets the model emit correct code on iteration 199 of a popcount loop based on what bit of the input remains.

**Output head**: last-token pool over the backbone's final hidden state → a small MLP → **32 independent sigmoid heads**, one per bit of the 32-bit RV32I instruction word. Bit heads sidestep the RV32I field polysemy ceiling (`rs2` is a register on R-type, imm[4:0] on I-type) that caps a six-field-categorical head around 85%.

**Grounded execution during training**: every training sample is collected by re-running a verified program through Unicorn one instruction at a time, recording `(state_at_pc, instruction_at_pc)` pairs — the same loop the model runs at inference. No teacher forcing on state; the state trajectory the model trains against is the one it will see in production.

**Training data**: 80,396 (prompt, program) pairs across 56 families (add, sub, mul-by-repeated-add, factorial, fibonacci, countdown, sum, max, min, abs, power, popcount, memcpy, display buffer writes, ...). Each program is verified end-to-end in Unicorn before training (zero rejects). Flattened cycle pool: ~1.06M (state, instruction) pairs, subsampled to 173k balanced across families.

**Loss**: BCE per bit, with the five `rs2` bits (positions 20–24) weighted 5× to overcome their polysemy ceiling.

## How Reflex differs from other "LLM as controller" approaches

| approach | what the LLM outputs | grounding |
|---|---|---|
| RT-2 (Google DeepMind) | text tokens encoding robot actions | decoded by policy head |
| [Neural Computer](https://arxiv.org/abs/2311.01906) | video pixels of a screen | rendered and re-fed next frame |
| **Reflex** | **native RISC-V instruction words** | **executed by Unicorn, state fed back as K/V next cycle** |

Reflex is the thinnest possible bridge: the model's output channel is the CPU's input channel, with no decode step between them. The loop is hardware-native. This is closer to Tesla FSD picking a steering actuator from pixels than GPT writing a Python script.

## Limitations

- **`rs2` bit precision bleeds**. At ~0.99 per-cycle rs2 accuracy, a 6-op program has ~94% chance of being fully correct; a 200-op loop's exit-branch precision is the most fragile point and can derail mid-loop.
- **Programs over ~40 ops can corrupt**. A single bit flip in the middle of a long loop sends PC somewhere invalid and the trailing `sw`/`halt` never get emitted.
- **Basic arithmetic is unreliable for certain phrasings.** `add 100 and 200 and store the result` can halt with just `100` in memory (the ADD instruction's rs1 ends up wrong); `multiply 7 and 8` is 95%+ reliable, `subtract 10 from 25` sometimes outputs `35` (semantic: added instead of subtracted).
- **No domain knowledge transfer.** Given `x5 is body temperature; if fever display SICK`, Reflex emits garbage — it has no idea what "fever" means. The backbone's prior only flows through the cross-attn path for program-shaped training-distribution prompts.
- **Display strings degrade past 3 characters.** `say hi`/`say wow`/`say 42` pass cleanly; `say hello` → `hell`, `say CPU` → `cpe`.

See [the eval artifacts](#reproducing-the-results) for the full failure catalog.

## Run

Requires CUDA (Unicorn is CPU, backbone is GPU). Tested on A100 80GB, CUDA 12.1, PyTorch 2.4.1.

```bash
# 1. Install
uv sync

# 2. Get the checkpoint (adapters only — backbone is downloaded from HF on first run)
# Instructions: see MODEL_CARD.md

# 3. Interactive demo (side-by-side with text-mode baseline)
uv run demo --checkpoint path/to/reflex_coder7b.pt

# 4. Headless eval over the 41-task suite
uv run eval --checkpoint path/to/reflex_coder7b.pt
```

## Reproducing the results

```bash
# Headless eval, writes eval_results.json
uv run eval --checkpoint reflex_coder7b.pt

# Regenerate training corpus (optional)
uv run python scripts/generate_programs.py

# Retrain from scratch (A100 80GB, ~4 hours)
uv run train --steps 15000 --batch 64 \
  --ckpt reflex_coder7b.pt --probe "say hi@0x6000=0x68"
```

## Architectural journey

Each decision in this release was validated by ablation:

- **Flamingo injection** (vs. stacked cross-attn on top of a frozen backbone): deep per-layer injection is what lets the model extrapolate on popcount beyond training templates. Stacked adapters alone can't encode the register-level semantics the backbone lacks.
- **32 bit heads** (vs. 6 field-categorical heads): removes the rs2 polysemy ceiling.
- **Frozen backbone + cross-attn adapters** (vs. LoRA on the backbone): preserves the pretrained code knowledge that gives popcount, min, and multiply their emergent success.
- **Chat template with machine-context system message** (vs. a plain-text prefix): uses the backbone's instruct-tuning rather than bypassing it.
- **Grounded execution during training**: without stepping through Unicorn, the model trains against teacher-forced states that diverge from the distribution it sees at inference.

