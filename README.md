# Reflex

**A frozen LLM wired directly to a machine — no text generated.**

A frozen Qwen2.5-Coder-1.5B encodes human instructions. A trained control head
emits raw CHIP-8 opcodes — complete programs with loops, conditionals,
subroutines, and calls. The LLM never generates a token — it only understands.

```
Human: "count from 1 to 9 and display each digit"
  ↓
Frozen LLM backbone → hidden states (understanding)
  ↓
Control head: cross-attention + GRU memory
  ↓
Emits a complete program:
  00E0  MOV V0,0x0A  MOV V1,0x0A  MOV V2,0x01
  FONT V2  DRAW V0,V1,5  ADD V0,0x08  ADD V2,0x01
  SKE V2,0x0A  JUMP 0x208  JUMP 0x216
  ↓
The CHIP-8 emulator runs it. The digits 1..9 appear in a row.
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

The model emits **complete CHIP-8 programs** with real control flow — not
linear opcode sequences. Each generated program is then loaded into the
emulator and runs end-to-end. Categories covered:

- **Loops**: `count from 1 to 9 and display each digit`,
  `count down from 7 to 1`, `blink the screen 5 times`.
- **Conditionals**: `set V0 to 5; if V0 equals 5 draw a 1 else draw a 0`.
- **Arithmetic with display**: `add 7 and 8 and show the result` (uses
  BCD via FX33 + FX65 to draw multi-digit results),
  `compute 3 times 4 and show it` (multiplication via repeated addition).
- **Subroutines**: `draw a star using a subroutine called twice` (real
  2NNN call / 00EE return, sprite stored at 0x300 via FX55).
- **Memory**: `store 42 at address 0x300 and load it into V0`.
- **Timers**: `wait for 30 ticks then draw a 1` (real delay-timer
  busy loop).
- **Random**: `draw a random digit` (CXNN).

Each category includes 2-3 structural variants (forward/backward loops,
different register layouts, subroutine before vs after main body, etc.)
so the model learns the *pattern*, not the exact byte sequence.

## Two modes of cross-attention grounding

This build runs in **program-synthesis mode**: the machine state at
emission time is held at zeros — the program hasn't run yet, there is
nothing to ground against. Cross-attention degenerates to a learned
constant context; the model is doing pure instruction → code translation,
leaning on the GRU + token-ID pathway.

The grounding thesis (instruction queries reading live machine state)
is the focus of the **interactive-execution mode** showcased earlier in
the project (the Pong controller). This mode showcases something
different: direct neural code generation without text.

## Limits

- Generalization is **structural** (LLM maps to the closest known program
  shape) but not **symbolic**. Word-form numbers like `two plus three`
  trigger the arithmetic template but with wrong operands.
- Random programs are not byte-deterministic at execution time; the
  display will show a different digit per run, but the structure is
  fixed.
- Out-of-distribution inputs usually get completed into a plausible-looking
  program rather than cleanly rejected.
- Training covers ~8k instruction variants across the seven categories
  above. Arbitrary CHIP-8 programs outside this distribution aren't
  supported.

## Why this architecture

The project tests a thesis: **LLMs don't need to generate text to control
machines.** Their understanding can drive machine actions through learned
neural pathways — the same way Tesla FSD outputs actuator commands directly
rather than JSON tool calls. The interesting part here isn't that it's a
complete CHIP-8 compiler (it isn't). It's that a general-purpose, frozen LLM
can drive a real VM at the opcode level with zero token generation.
