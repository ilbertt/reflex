# Reflex

**A neural net controls a real Linux VM through direct signals — no text generation.**

Like a pianist's brain controls fingers directly (not by dictating sheet music), this model outputs raw syscall signals that the Linux kernel executes. Zero tokens generated.

## Architecture

```
State (VM memory, files, processes)
  ↓
[Temporal Attention] — sees last 4 states
  ↓
[Instruction Fusion] — "create hello.py"
  ↓
┌─────────────────┬──────────────────────┐
│ Categorical head │  Continuous head     │
│ → syscall number │  → args + buffer     │
│   (openat, write,│    (raw bytes)       │
│    close, ...)   │                      │
└─────────────────┴──────────────────────┘
  ↓
Linux kernel executes the syscall
```

No codebook. No text. The model directly outputs which syscall to call and the raw argument bytes.

## Run it

```bash
# Train (collects syscall traces from a real VM, trains the model)
uv run train

# Demo (model controls VM live)
uv run demo
```

Requires Docker (or OrbStack on macOS).

## What you'll see

The model creates files, directories, and runs Python programs inside a real Linux container — all through raw syscall signals, not text commands.

```
  step 0 → openat     conf=0.98  buf=[hello.py]  (142µs)
  step 1 → write      conf=0.97  buf=[print('.]  (89µs)
  step 2 → close      conf=0.99  buf=[........]  (76µs)

  $ python3 hello.py
    hello world!

  Speedup: 190x faster than text generation
```

## The point

Current AI agents generate text → parse text → execute action. That's like a pianist writing sheet music, handing it to someone, who then plays the notes. The model should play directly.
