# Reflex — grounded RV32I code generation

**A small LLM wired to a RISC-V emulator, cycle by cycle. No text
generated — one 4-byte instruction at a time, each conditioned on the
live machine state.**

```
Human: "compute 5 factorial and store it"
  ↓
Qwen2.5-0.5B + LoRA → hidden states (understanding, cached once)
  ↓
Loop until HALT:
  ┌─────────────────────────────────────────────────────────┐
  │ 1. read live state from Unicorn (x0..x31, PC, mem[±pc], │
  │    mem[±sp])                                            │
  │ 2. state encoder → K/V  │  cached instruction → Q       │
  │ 3. cross-attention + GRU → six-field RV32I logits       │
  │ 4. argmax → one 32-bit instruction                      │
  │ 5. write it at PC in Unicorn, single-step               │
  │ 6. state has evolved — GOTO 1                           │
  └─────────────────────────────────────────────────────────┘
  ↓
Final mem[DATA_BASE] = 120
```

## What it does

Natural-language prompt → a RISC-V program emitted **one opcode per
cycle**, each opcode executed on a real emulator (Unicorn) before the
next is emitted. The model never sees the program as a whole — it sees
what the machine sees, right now.

## Architecture

Same cross-attention core as earlier Reflex branches (instruction Q,
machine state K/V) with three deliberate changes for the grounded mode:

1. **Encode once, decode many.** The backbone runs once per task; the
   hidden states become cached queries. Per-step decoder is a small
   stack of cross-attention blocks — ~10× faster training iteration.
2. **State K/V is live.** 65 state tokens per step: 32 registers, PC,
   16 words around PC, 16 words around SP. Built fresh from Unicorn at
   every cycle.
3. **LoRA on the backbone.** Rank-8 adapters on `q/k/v/o_proj` give the
   0.5B model just enough room to map numeric operands from the prompt
   into the immediate fields. Fully frozen also converges but drops
   operand precision.

Training mirrors inference exactly: memory is **not** pre-loaded;
instead at each cycle we record the state that inference would see
(memory written only as cycles advance), then write the ground-truth
instruction at PC and step.

```
  cached Q (instruction) ────┐
                             ▼
                        [XAttn × 4] ◀── K/V (live state)
                             │
                           [pool]
                             │
                prev_op ──── (⊕) ────▶ [GRU] ──▶ h_state
                                         │
                                         ▼
                                   [MLP + 6 heads]
                                         │
                        (opcode, rd, funct3, rs1, rs2, funct7)
```

Output is a 32-bit instruction assembled from six classification heads
matching RV32I's field decomposition.

## Results

After ~15 min on an RTX 5090 (1500 steps, batch 16, Qwen2.5-0.5B +
LoRA, grounded state): **7 of 8 reference programs correct
end-to-end**, all 8 halt cleanly. The full factorial loop, full
fibonacci sequence, memcpy of four words, and call/return are
byte-perfect.

| prompt | result | expected |
|---|---|---|
| add 7 and 8 | 8 ✗ | 15 |
| compute 5 factorial | **120** ✓ | 120 |
| first 6 Fibonacci numbers | **[0,1,1,2,3,5]** ✓ | [0,1,1,2,3,5] |
| count down 5 to 1 | **[5,4,3,2,1]** ✓ | [5,4,3,2,1] |
| sum 1..10 | **55** ✓ | 55 |
| max of 7 and 12 | **12** ✓ | 12 |
| copy 4 words | **[1,2,3,4]** ✓ | [1,2,3,4] |
| double 25 | **50** ✓ | 50 |

With the backbone frozen (no LoRA) the same architecture gets 1 of 8
semantically correct but still 8 of 8 halting — structural control
comes from the grounded state, operand precision comes from LoRA.

## Run

CUDA required (Unicorn is CPU, backbone is GPU). Tested on an RTX 5090
with 32 GB, CUDA 13, PyTorch 2.11.

```bash
uv sync                              # installs torch/transformers/peft/unicorn
uv run train --no-freeze-backbone    # ~15 min to 100% field accuracy
uv run demo                          # runs the 8 reference prompts
uv run demo --instruction "copy 6 words from source to destination" --verbose
```

For a fresh Vast.ai PyTorch box with a 24 GB+ GPU, `git clone`, the
same commands work unchanged.

## Known limitations

- **Symbolic generalization is weak.** The failing `add 7 and 8`
  emitted only one of the two operands; the same class of mistake
  appears on paraphrases outside the training distribution
  (`subtract 3 from 10` falls back to add, `compute 2 times 5` hits
  the factorial template, etc.). The model learns the *structure* of
  the eight canonical program families and the exact numeric values
  present in their phrasings, but not a general "number in prompt →
  number in immediate" mapping.
- **No MUL/DIV.** RV32I only. Multiplication-like programs
  (`doubles 25`) work via repeated addition templates; anything
  requiring the M extension isn't supported.
- **Training-distribution coverage.** ~4400 verified (prompt,
  program) pairs across eight categories (add / factorial / fibonacci
  / countdown / sum / max / memcpy / call+return) with 2–3 structural
  variants each. Prompts outside these shapes get completed into the
  nearest template rather than cleanly rejected.
- **Teacher-forced state during training.** We never sample from the
  model's own execution trajectory; if a prediction diverges from the
  training state distribution, the next state is off-policy. On-policy
  fine-tuning (run the predicted opcodes through Unicorn during
  training) is the obvious next step.

## Why

Earlier Reflex branches emitted whole programs into zero state
(program-synthesis mode). This branch tests the stronger thesis: the
frozen/LoRA'd LLM doesn't need a bytecode-sized context window — it
just needs to look at the machine, cycle by cycle, and press the right
button. Like Tesla FSD choosing an actuator from a pixel buffer, not
like GPT writing a Python script.
