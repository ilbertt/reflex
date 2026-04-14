# Reflex

**A neural net reads raw machine bytes and emits control signals. No text. No parsing. No human interpretation.**

The model reads `/proc/stat`, `/proc/meminfo`, `/proc/loadavg` as raw bytes — 576 bytes, each value 0-255. It doesn't know what "cpu" means. It learns from the byte patterns alone, like a vision model learns from pixels.

## Architecture

```
Raw bytes from /proc (576 bytes)
         ↓
  [Byte embeddings]  — each byte value (0-255) gets a learned vector
         ↓
  [Transformer]      — attention over the byte sequence
         ↓                learns that 0x63,0x70,0x75 = "cpu"
  [Action heads]     — which process, what priority
         ↓
  Control signal     — renice, kill, adjust scheduling
```

## Run it

```bash
# Requires Docker
uv run reflex
```

The model trains on live machine state (creates CPU load scenarios, reads raw `/proc` bytes, learns what to do), then runs a live control loop — reading raw bytes and emitting process control signals at ~100µs per decision.

## The point

Current AI agents control computers by generating text commands that get parsed and executed. That's like controlling a robot arm by dictating English instructions.

Reflex reads raw machine state and emits raw control signals. No text in the control path. The model is directly wired to the machine — like Tesla FSD is wired to steering and throttle, not to a text interface.
