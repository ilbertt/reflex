# Reflex

**Wire a frozen LLM to a CPU through cross-attention. No text generated — one RV32I instruction per cycle, conditioned on live machine state. The output head is a JEPA-style embedding predictor that snaps to a learned instruction codebook, so small per-step noise can't flip a bit and crash the program.**

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
    │         │   (7 adapters total, ~1.1B trained params) │       │
    └─────────┼────────────────────────────────────────────┼───────┘
              │                                            ▼
     ┌────────┴──────────┐                       ┌──────────────────┐
     │   STATE ENCODER   │                       │ last-token pool  │
     │    (trained)      │                       │        ▼         │
     │                   │                       │    2-layer MLP   │
     │  65 tokens × H:   │                       │        ▼         │
     │   x0..x31  (32)   │                       │  embedding head  │
     │   PC        (1)   │─── K,V ──→ XAttn ────▶│   (256-d vector) │
     │   mem[±pc] (16)   │                       │        ▼         │
     │   mem[±sp] (16)   │                       │  nearest-neighbour
     └─────────▲─────────┘                       │  over 691-row    │
               │                                 │  instruction     │
               │                                 │  codebook        │
               │                                 │        ▼         │
               │                                 │  32-bit RV32I    │
               │                                 │   instruction    │
               │                                 └─────────┬────────┘
               │                                           ▼
               │                                 write at PC; step
               │                                           │
               └───────────────────────────────────────────┘
                    RV32I Unicorn emulator (state feedback)
```

Each cycle: read the CPU's 32 registers + PC + memory windows, encode as 65 K/V tokens, run the backbone forward over the prompt with those K/V injected at every 4th layer, pool the last token, project to a 256-d embedding, **snap to the nearest row of a learned instruction codebook** (one row per unique 32-bit word seen in training), write that word at PC, step Unicorn once.

The backbone is **untouched**. Only a 1.1B-parameter stack (seven cross-attn adapters + state encoder + embedding head + 691-row codebook) is trained. Inference emits no text tokens — the model's output channel is the 32-bit instruction word, executed immediately.

## Why JEPA instead of per-bit heads

An earlier iteration of Reflex predicted 32 independent sigmoid bit-heads. Even at 99% per-bit accuracy, one wrong bit mid-program flips an ADD into an AND or misroutes a branch and the program derails. A 40-step program had ~67% chance of being bit-perfect; 200-step programs were essentially doomed.

The JEPA codebook head sidesteps this entirely. The output space is not bits; it's a **learned vector per distinct instruction the training set ever contained** (691 rows on our corpus). The model predicts a continuous 256-d vector, the decoder snaps to the nearest row, and the decode is always a *real, seen-in-training* instruction word. Small embedding noise picks a neighbouring instruction rather than a garbage bit-pattern. In practice the neighbours turn out to be semantic neighbours: `addi x5, x0, 8` vs `addi x15, x0, 8` (same compute, different rd), or immediate ±1. **100% of top-1 misses in our diagnostic are same-opcode.**

This is what lets `popcount(255) = 8` complete in **199 consecutive correct RISC-V instructions** at only ~97% per-step top-1 accuracy — without snap-correction that would be 0.6% bit-perfect probability.

## Results

Full 41-task eval on the released checkpoint:

| section | pass | |
|---|---|---|
| in-distribution (8) | **7 / 8** | 88% |
| out-of-distribution (10) | **9 / 10** | 90% |
| display strings (4) | 1 / 4 | 25% |
| novel zero-shot (9) | **7 / 9** | 78% |
| consistency: factorial 5 × 10 | **10 / 10** | 100% |
| **total** | **34 / 41** | **83%** |

**Zero-shot novel wins (model never trained on any of these):** `multiply 3×4` and `7×8` (23 and 39 ops), `power 2^5` (82 ops), `min(7,3,9)`, `abs(-5)`, **`popcount(255) = 8` in 199 consecutive correct instructions**, `count up 1..5`.

**Factorial 5 is perfectly deterministic:** all 10 independent runs emit exactly 91 ops and land on `mem[0x5000] = 120`.

Per-step top-1 instruction accuracy on 500 random held-out cycles: **96.0%**. All `BRANCH`, `R-type`, `LOAD`, `STORE`, `JAL`, `JALR` predictions are 100%; the remaining errors concentrate in `ADDI` (88.8%) and `LUI` (92.3%) — specifically on register-swap (x5 ↔ x15) and ±1-immediate disambiguation.

## How it works

**Backbone**: `Qwen/Qwen2.5-Coder-7B-Instruct`, frozen in bf16. No weights touched.

**Adapters**: seven [Flamingo-style gated cross-attention](https://arxiv.org/abs/2204.14198) blocks spliced into the backbone's own transformer layers via forward hooks, one every four layers. Each adapter adds

```
hidden  ←  hidden + tanh(α)·CrossAttn(LN(hidden), state_kv)
        + tanh(β)·MLP(LN(hidden))
```

where `state_kv` is the 65-token K/V built from the live CPU state. Both tanh gates start at zero, so at step 0 the adapter is a no-op identity and the pretrained backbone activations are undisturbed.

**State encoder**: 32 register tokens + 1 PC token + 16 words around PC + 16 words around SP → 65 tokens, each a sum of a role embedding and a per-byte-quadrupled value embedding. Normed, projected, used as K/V.

**Output head**: last-token pool over the backbone's final hidden state → 2-layer MLP → linear projection to 256-d. At inference, cosine similarity against every row of a `nn.Embedding(num_instrs=691, 256)` codebook, argmax, look up the 32-bit instruction word in a companion int64 buffer.

**Training loss**: InfoNCE. For each ground-truth instruction in the batch, the correct codebook row is the positive; every other row is a negative. Temperature τ = 0.07. The codebook entries train jointly with the adapters and the head.

**Grounded execution during training**: every training sample is collected by re-running a verified program through Unicorn one instruction at a time, recording `(state_at_pc, instruction_at_pc)` pairs — the exact loop the model runs at inference. No teacher forcing on state; the state trajectory the model trains against is the one it sees in production.

**Training corpus**: 80,396 `(prompt, program)` pairs across 56 families (add, sub, mul-by-repeated-add, factorial, fibonacci, countdown, sum, max, min, abs, power, popcount, memcpy, display-buffer writes, …). Every program verified end-to-end in Unicorn before training (zero rejects). Flattened cycle pool: ~1.06M `(state, instruction)` tuples, subsampled to ~173k balanced across families.

Trained 15 000 steps, batch 16, AdamW with cosine LR `1e-4 → 1e-6`, on a single A100 80 GB (~4 h) or L40S 48 GB at batch 32 (~5 h). Full hyperparameters in [`MODEL_CARD.md`](MODEL_CARD.md).

## Limitations

- **Display byte-constants are unreliable.** The failure mode is confusing `addi x15, x0, 0x34` (`'4'`) with `addi x15, x0, 0x32` (`'2'`) — same opcode, same rd, off-by-two immediate. `say hi`, `display OK`, `say wow` all pass; `show 42`, `say 42`, `print hello` fail on one character.
- **Long-integer arithmetic with uncommon literals drifts.** `sum 1..20` computes 210 (correct). `add 100+200` sometimes halts at 120 (the ADDI immediate field decodes to a near-neighbour value). These failures are same-opcode register/immediate swaps, not opcode-flips.
- **No domain-knowledge transfer.** Given `"x5 is body temperature; if fever display SICK"`, Reflex emits garbage. The backbone's prior only flows through the cross-attn path for program-shaped training-distribution prompts.
- **Closed action space.** Any instruction word that never appeared in the training corpus cannot be emitted — the codebook has no row for it. The 691-row vocabulary is ample for the 56 program families we trained on but bounds generalisation to genuinely unseen opcodes.
- **RV32I base ISA only** — no M (multiply/divide), no Zbb (count/bitmanip), no F (float). The model synthesises all "higher" operations (multiply, popcount, etc.) from base instructions.

## Run

Requires CUDA (Unicorn is CPU, backbone is GPU). Tested on A100 80 GB (CUDA 12.1, PyTorch 2.4.1) and L40S 48 GB.

```bash
# 1. Install
uv sync

# 2. Fetch the released checkpoint (adapter weights only, ~4.4 GB)
huggingface-cli download ilbert/reflex-coder7b-jepa-riscv reflex.pt --local-dir .

# 3. Interactive demo (side-by-side with text-mode baseline)
uv run demo --ckpt reflex.pt

# 4. Headless eval over the 41-task suite
uv run eval --ckpt reflex.pt
```

The first run downloads `Qwen/Qwen2.5-Coder-7B-Instruct` (~15 GB) automatically.

## Reproducing the results

```bash
# Generate the training corpus (~80k programs, verified via Unicorn)
uv run python scripts/generate_programs.py

# Train from scratch (A100 80 GB, ~4 h @ batch 16; L40S 48 GB ~5 h @ batch 32)
uv run train --steps 15000 --batch 16 --ckpt reflex.pt \
    --sample-pool 300000

# Full eval
uv run eval --ckpt reflex.pt --out eval_results.json
```

## Ablations

- **Flamingo per-layer injection** (vs. stacked cross-attn on top): deep injection is what lets the model extrapolate on popcount beyond training templates. Stacked adapters alone don't encode the register-level semantics the backbone lacks.
- **JEPA codebook head** (vs. 32 sigmoid bit-heads): snap-to-codebook absorbs per-step noise so 199-op popcount completes cleanly at 97 % per-step top-1. We also tried a richer JEPA encoder over decomposed RV32I fields with an EMA target — it matched eval score (34/41) but had worse same-opcode miss purity (91 % vs 100 %) and didn't justify the added surface.
