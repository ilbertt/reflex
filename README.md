# Reflex

**A frozen LLM wired directly to a machine — no text generated.**

A frozen Qwen2.5-Coder-1.5B encodes human instructions. A trained control head uses cross-attention (instruction queries machine state) to emit raw CHIP-8 opcodes. The LLM never generates a token — it only understands.

```
Human: "compute 3 plus 5 and draw result"
  ↓
Frozen LLM backbone → hidden states (understanding)
  ↓
Control head cross-attends to machine state
  ↓
Emits: 00E0 6003 7005 6114 620A F029 D125
  ↓
Digit "8" appears on screen
```

## Run

```bash
uv run train   # ~4 min (encodes instructions + trains control head)
uv run demo    # runs test cases
uv run demo -i # interactive mode
```

Requires Apple Silicon (MLX).

## Architecture

The key insight: **instruction tokens are queries, machine state is keys/values.** The instruction "looks at" the machine to decide what to do — like how diffusion models use text to query image features.

```
Instruction tokens (Q) ──→ [Cross-attention] ←── Machine state (K/V)
                                  ↓
                           [Self-attention]
                                  ↓
                         opcode (high, low)
```
