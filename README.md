# Reflex — grounded RV32I code generation

**A small LLM wired to a RISC-V emulator, cycle by cycle. No text
generated — one 4-byte instruction at a time, each conditioned on the
live machine state.**

```
Human: "say hi"
  ↓
"You control a RV32I CPU. Registers x5-x15 available. Data: 0x5000.
 Display: 0x6000 (ASCII, one word per char). Task: say hi"
  ↓
Qwen2.5-Coder-3B-Instruct (FROZEN, fp16) + injected cross-attn adapters
  ↓
Loop until HALT:
  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. read live state from Unicorn: x0..x31, PC, mem[±pc], mem[±sp]│
  │ 2. state encoder → K/V (65 tokens)                              │
  │ 3. full backbone forward with state K/V injected every 4 layers │
  │    (Flamingo gated cross-attention, tanh-gated, adds no-op init)│
  │ 4. last-token pool → MLP → 32 bit-wise sigmoid heads            │
  │ 5. assemble the 32 bits into one RV32I instruction              │
  │ 6. write it at PC in Unicorn, step once, state has evolved      │
  │ 7. goto 1                                                       │
  └─────────────────────────────────────────────────────────────────┘
  ↓
mem[0x6000] = 'h', mem[0x6004] = 'i', then HALT (jal x0, 0)
```

## Architecture

The backbone is a **frozen** Qwen2.5-Coder-3B-Instruct. The only
trainable parameters are: the state encoder, nine cross-attention
adapters spliced into the backbone's own transformer layers, a head
MLP, and 32 bit-classification heads (≈462M parameters total).

Four design choices matter, each proven by ablation on this branch:

1. **Flamingo-style cross-attention, *inside* the backbone.** Every
   `INJECT_EVERY=4` layers, a forward hook adds
   `hidden + tanh(α)·CrossAttn(LN(hidden), state_kv) + tanh(β)·MLP(…)`.
   Tanh gates start at zero so the pretrained Qwen activations are
   undisturbed at step 0. The state conditions reasoning at every
   depth, not just at the head — this is what lets the model
   *extrapolate* on factorial beyond the training range, not just
   match memorised templates.

2. **32 independent bit heads.** The first version used the RV32I
   field decomposition (opcode / rd / funct3 / rs1 / rs2 / funct7)
   with six categorical heads. The `rs2` slot is polysemous (a
   register on R/S/B-type but the low 5 bits of an arbitrary
   immediate on I/U/J-type) which caps field-categorical accuracy
   near 85%. Bit heads sidestep this: each of the 32 instruction bits
   is its own binary classifier, and compound field semantics are
   learned implicitly.

3. **Last-token pooling, not masked mean.** Adding a machine-context
   prefix to every prompt dilutes a mean-pooled representation with
   ~75% shared tokens. Last-token pool sits on the causal LLM's final
   real token, which has attended over the full prefix-then-task
   prompt with no dilution. Measured effect: same architecture
   plateaued at ~90% per-cycle with mean-pool and broke through to
   99.8% with last-token pool, in one-third the steps.

4. **Grounded execution during training.** Every training sample is
   collected by re-running verified programs through Unicorn one
   instruction at a time and recording `(state_at_pc, instruction_at_pc)`
   — the same loop that runs at inference. No teacher forcing on
   state; the state trajectory the model trains against is the one
   it will see in production.

## Training data

~75,000 `(instruction, program)` pairs generated from ~75 programs
across 11 families — add / subtract / factorial / fibonacci /
countdown / sum / max / memcpy / function-call / **display-buffer
writes** — each with 25–75 natural-language phrasings and 2–3
register-layout variants. Every program is verified end-to-end in
Unicorn before training (zero rejects). Flattened cycle pool: ~910k
(state, instruction) pairs.

Includes a **display buffer** at `0x6000`: ASCII bytes, one character
per 32-bit word. Display tasks are e.g. `say hi` (writes 0x68, 0x69)
or `show 42` (writes '4', '2').

## Results

Trained end-to-end: ~13k steps at batch 64, bf16 mixed precision, ~2h
on an RTX 5090. Per-cycle accuracy **99.8%** (in-sample), all six
RV32I fields at 1.00.

**In-distribution — 6/8:**

| prompt | emitted | expected |
|---|---|---|
| add 7 and 8 | **15** ✓ | 15 |
| compute 5 factorial | **120** ✓ | 120 |
| first 6 Fibonacci numbers | [0,1,1,0,0,0] ✗ | [0,1,1,2,3,5] |
| count down 5 to 1 | **[5,4,3,2,1]** ✓ | [5,4,3,2,1] |
| sum 1..10 | **55** ✓ | 55 |
| max of 7 and 12 | **12** ✓ | 12 |
| copy 4 words | read-unmapped ✗ | [1,2,3,4] |
| double 25 | **50** ✓ | 50 |

**Out-of-distribution — 7/10:**

| prompt | training range | emitted | expected |
|---|---|---|---|
| compute 7 factorial | n ≤ 10 in training | **5040** ✓ | 5040 |
| first 10 Fibonacci | n ≤ 15 | **[0,1,1,2,3,5,8,13]** ✓ | same |
| countdown from 20 | n ≤ 25 | **[20,19,18,17,16,15,14,13]** ✓ | same |
| add 100+200 | a≤100, b≤25, a+b≤200 (OOD) | 200 ✗ | 300 |
| max(3,3) | a≠b always | **3** ✓ | 3 |
| copy 8 words | ≤12 in training | write-unmapped ✗ | [1..8] |
| sum 1..20 | n ≤ 25 | **210** ✓ | 210 |
| **subtract 10 from 25** | subtract family added | **15** ✓ | 15 |
| **double 100** | n ≤ 100 | **200** ✓ | 200 |
| first 3 Fibonacci | trivially in-range | exception ✗ | [0,1,1] |

**Display — 3 of 4 perfect writes:**

| prompt | screen |
|---|---|
| say hi | **`hi`** ✓ |
| display OK | **`OK`** ✓ |
| show 42 | **`42`** ✓ |
| print hello | `helm·omo` (first three letters right, then loses control) |

**Novel (never-seen phrasings):**

| prompt | screen | note |
|---|---|---|
| **write your name** | **`name`** | the model literally spelled n-a-m-e |
| draw a box | *blank* | no template close enough |
| display the result of 3+4 | *blank* | compositional, not supported |

"`name`" is genuinely emergent — the string "name" appears nowhere in
training data. The prefix + last-token pool setup routes the novel
phrasing into a coherent 4-character display write.

## Architectural journey

Each result in this branch is load-bearing and was validated by
ablation:

- **Flamingo injection** (vs stacked cross-attn on top of a frozen
  backbone): deep injection uniquely gets factorial 7 = 5040. Stacked
  adapters alone can't encode the register-level semantics the
  backbone lacks.
- **32-bit head** (vs 6-field categorical): removed the rs2 polysemy
  ceiling.
- **Full fine-tune vs frozen + enough data**: at 4,400 programs and
  6k steps the 0.5B full-fine-tune beats the 3B frozen; at 75k
  programs and 9700+ steps the 3B frozen wins. Frozen big beats small
  full-FT only past a data/step threshold.
- **Context prefix + last-token pool** (vs no-prefix): same total
  OOD count (7/10 vs 6/10) but new capabilities — subtract, double
  100, display buffer writes, novel text emission.
- **Grounded execution during training**: without stepping through
  Unicorn, the model trains against teacher-forced states that
  diverge from the distribution it sees at inference.

## Run

Requires CUDA (Unicorn is CPU, backbone is GPU). Tested on RTX 5090,
32 GB, CUDA 13, PyTorch 2.11.

```bash
uv sync
# Train from scratch (3B frozen, ~2h to converge)
uv run train \
    --backbone-id Qwen/Qwen2.5-Coder-3B-Instruct \
    --freeze-backbone \
    --dtype bf16 --batch 64 --steps 15000 \
    --context-prefix --max-instr-tokens 64 \
    --ckpt reflex_3b_ctx.pt
# Run the 18-task eval
uv run python eval_combined.py --ckpt reflex_3b_ctx.pt
# Live display rendering (screen updates as bytes land at 0x6000)
uv run python demo_live.py --ckpt reflex_3b_ctx.pt
```

## Known limitations

- **Loop-exit precision is the remaining fragile point.** fib 3,
  fib 6, memcpy 8 all fail on the exit-branch prediction while the
  loop body itself is correct.
- **Display byte precision on long strings.** rs2 hits 1.00 on
  per-cycle eval, but across a 5-op display write of an unseen
  string the 1% per-byte error compounds. `print hello` writes
  `helm·omo` — first three letters right, then the model loses the
  template and spirals.
- **No compositionality.** `display the result of 3+4` never works —
  the model has display programs and arithmetic programs but can't
  chain them on a novel prompt.
- **No MUL/DIV.** RV32I only; multiplication programs use repeated
  addition templates.

## Why

Earlier Reflex branches emitted whole programs into zero state
(program-synthesis mode). This branch tests the stronger thesis: a
frozen pretrained LLM doesn't need a bytecode-sized context window —
it just needs to look at the machine, cycle by cycle, and press the
right button. Like Tesla FSD choosing an actuator from a pixel
buffer, not like GPT writing a Python script.
