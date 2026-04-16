# Reflex

**A frozen LLM wired directly to a machine — no text generated.**

A frozen Qwen2.5-Coder-1.5B encodes human instructions. A trained control head
cross-attends to the machine state and emits raw CHIP-8 opcodes. The LLM never
generates a token — it only understands.

```
Human: "draw a smiley"
  ↓
Frozen LLM backbone → hidden states (understanding)
  ↓
Control head: cross-attention + GRU memory
  ↓
Emits: 00E0 603C 6142 62A5 6381 64A5 6599 6642 673C A300 F755 6814 690A A300 D898
  ↓
A smiley face appears on screen
```

## Run

```bash
uv run train    # encodes instructions + trains control head
uv run demo     # runs preset test cases
uv run demo -i  # interactive mode — type any instruction
```

Requires Apple Silicon (MLX).

## Architecture

**Flipped cross-attention**: instruction tokens are queries, machine state is
keys/values. The instruction "looks at" the machine to decide what opcode to
emit next — like how diffusion models use text to query image features.

```
  LLM hidden states ───┐
                       ▼
                  [Cross-attn] ◀── Machine state (K/V)
                       │
                  [Self-attn]
                       │
                   [Pool] ─────────┐
                                   ▼
  Token IDs ──[MLP]────────────── (+)
                                   │
                   prev opcode ────┤
                                   ▼
                       ┌──── [GRU] ◀──── h_state
                       │        │
                       │        └────▶ next h_state
                       ▼
                   [MLP head]
                       │
                       ▼
            (high byte, low byte)
```

Three pathways feed the output:

1. **Cross-attention** reads the machine state. The instruction queries state
   tokens to decide "what kind of opcode comes next given where we are."
2. **Token ID MLP** carries operand-level detail (which digit, which sprite
   name) that mean-pooled LLM hidden states can't separate — "draw digit 1"
   and "draw digit 7" are 0.9943 cosine-similar in the LLM's representation.
3. **GRU** holds autoregressive memory across steps. The previous opcode is
   fed as input, so the head knows exactly what it just emitted and where
   it is in the program.

Trained with **scheduled sampling**: during training, the previous opcode fed
to the GRU is the ground truth with probability ε, else the model's own
argmax (detached). ε decays linearly 1.0 → 0.1 over training, closing the
exposure-bias gap so inference accuracy tracks teacher-forced accuracy.

## What it does

- **Sprites**: smiley, heart, star, circle, box, cross, triangle, diamond,
  letter T, letter H, arrows, zigzag, checkerboard, snake, and more.
- **Digits 0–F** at any position: `draw digit 7 at position 15 10`.
- **Two-digit display**: `draw digits A and B`.
- **Arithmetic**: `3 + 5`, `compute 4 plus 6 and draw result`, `add 3 and 5`.
- **Natural phrasings**: `smiley`, `show me a heart`, `add three and five`.

## Limits

- Generalization is **structural** (LLM maps to the closest known task shape)
  but not **symbolic**. `two plus three` triggers the arithmetic template
  but with wrong operands, because word-form numbers weren't in arithmetic
  training data and the LLM's hidden states don't bridge "two" and "2" in
  arithmetic context.
- Out-of-distribution inputs usually get completed into a plausible-looking
  program rather than cleanly rejected.
- Training covers ~10k instruction variants. Arbitrary CHIP-8 programs
  outside this distribution aren't supported.

## Why this architecture

The project tests a thesis: **LLMs don't need to generate text to control
machines.** Their understanding can drive machine actions through learned
neural pathways — the same way Tesla FSD outputs actuator commands directly
rather than JSON tool calls. The interesting part here isn't that it's a
complete CHIP-8 compiler (it isn't). It's that a general-purpose, frozen LLM
can drive a real VM at the opcode level with zero token generation.
