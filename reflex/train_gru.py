"""
Train (v2): GRU control head with scheduled sampling.

Fixes the exposure-bias problem of the v1 GRU experiment:
  v1 used pure teacher forcing → GRU hidden state trained on correct
  prev_opcodes → at inference, one wrong prediction diverges the GRU
  state and cascades.

v2 scheduled sampling: at each training step, with probability ε feed
ground-truth prev_opcode (teacher forcing); with probability 1-ε feed
the model's own argmax (detached, no grad). ε decays linearly from
1.0 → 0.1 over training, so the GRU learns to recover from its own
mistakes.

Saves weights to weights_gru.npz (baseline weights.npz is preserved).

Usage:
    uv run train-gru
"""

import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .chip8 import Chip8, PROGRAM_START
from .model import ReflexModelGRU, BACKBONE_DIM, STATE_DIM, MAX_TOKENS, load_backbone, encode_instruction
from .train import generate_tasks, load_or_encode


# ── Data: per-program sequences ────────────────────────────────────────

def collect_sequences(tasks, instr_cache):
    """Collect per-program sequences for autoregressive training."""
    chip = Chip8()
    programs = []

    for instr, program in tasks:
        hidden, tid = instr_cache[instr]
        states, hi_targets, lo_targets = [], [], []

        chip.load_program(program)
        for _ in range(len(program) // 2):
            if chip.pc < PROGRAM_START or chip.pc >= PROGRAM_START + len(program):
                break
            state = chip.get_state()
            hi = int(chip.memory[chip.pc])
            lo = int(chip.memory[chip.pc + 1])
            states.append(state)
            hi_targets.append(hi)
            lo_targets.append(lo)
            chip.step((hi << 8) | lo)

        # STOP token
        states.append(chip.get_state())
        hi_targets.append(0)
        lo_targets.append(0)

        programs.append((hidden, tid, states, hi_targets, lo_targets))

    max_steps = max(len(s) for _, _, s, _, _ in programs)
    max_seq = max(h.shape[0] for h, _, _, _, _ in programs)
    n = len(programs)

    H = np.zeros((n, max_seq, BACKBONE_DIM), dtype=np.float32)
    T = np.zeros((n, MAX_TOKENS), dtype=np.int32)
    S = np.zeros((n, max_steps, STATE_DIM), dtype=np.float32)
    HT = np.zeros((n, max_steps), dtype=np.int32)
    LT = np.zeros((n, max_steps), dtype=np.int32)
    M = np.zeros((n, max_steps), dtype=np.float32)

    for i, (hidden, tid, states, hi_targets, lo_targets) in enumerate(programs):
        H[i, :hidden.shape[0], :] = hidden
        T[i] = tid
        for j in range(len(states)):
            S[i, j] = states[j]
            HT[i, j] = hi_targets[j]
            LT[i, j] = lo_targets[j]
            M[i, j] = 1.0

    total = int(M.sum())
    print(f"  {n} programs, {total} total steps, max {max_steps} steps/program")
    return H, T, S, HT, LT, M, max_steps


# ── Scheduled sampling schedule ────────────────────────────────────────

def linear_epsilon(step, total_steps, start=1.0, end=0.1):
    """Linear decay from start to end."""
    t = min(step / total_steps, 1.0)
    return start + (end - start) * t


# ── Training ───────────────────────────────────────────────────────────

def train(H, T, S, HT, LT, M, max_steps, steps=80000):
    model = ReflexModelGRU()
    scheduler = optim.cosine_decay(3e-4, steps, end=1e-6)
    optimizer = optim.Adam(learning_rate=scheduler)

    Hm = mx.array(H)
    Tm = mx.array(T)
    Sm = mx.array(S)
    HTm = mx.array(HT)
    LTm = mx.array(LT)
    Mm = mx.array(M)
    n = len(H)
    batch_size = min(32, n)
    perfect = 0

    def loss_fn(model, h, s, t, ht, lt, mask, epsilon):
        B = s.shape[0]
        h_state = mx.zeros((B, model.dim))
        prev_hi = mx.zeros((B,), dtype=mx.int32)
        prev_lo = mx.zeros((B,), dtype=mx.int32)
        total_loss = mx.array(0.0)

        for step_t in range(max_steps):
            hi_logits, lo_logits, h_state = model(
                h, s[:, step_t], t, prev_hi, prev_lo, h_state
            )

            # Loss against ground truth (unchanged by sampling)
            loss_hi = nn.losses.cross_entropy(hi_logits, ht[:, step_t]) * mask[:, step_t]
            loss_lo = nn.losses.cross_entropy(lo_logits, lt[:, step_t]) * mask[:, step_t]
            total_loss = total_loss + (loss_hi + loss_lo).sum()

            # Scheduled sampling for next step's prev_opcode
            use_gt = mx.random.uniform(shape=(B,)) < epsilon
            pred_hi = mx.argmax(mx.stop_gradient(hi_logits), axis=-1).astype(mx.int32)
            pred_lo = mx.argmax(mx.stop_gradient(lo_logits), axis=-1).astype(mx.int32)
            prev_hi = mx.where(use_gt, ht[:, step_t], pred_hi)
            prev_lo = mx.where(use_gt, lt[:, step_t], pred_lo)

        return total_loss / mask.sum()

    def eval_accuracy(epsilon):
        """Evaluate with given epsilon (use 0.0 for pure inference, 1.0 for teacher forcing)."""
        chunk = 64
        correct_hi = correct_lo = total = 0
        for i in range(0, n, chunk):
            h = Hm[i:i+chunk]
            s = Sm[i:i+chunk]
            t = Tm[i:i+chunk]
            ht = HTm[i:i+chunk]
            lt = LTm[i:i+chunk]
            mask = Mm[i:i+chunk]
            B = h.shape[0]
            h_state = mx.zeros((B, model.dim))
            prev_hi = mx.zeros((B,), dtype=mx.int32)
            prev_lo = mx.zeros((B,), dtype=mx.int32)
            for step_t in range(max_steps):
                hi_logits, lo_logits, h_state = model(
                    h, s[:, step_t], t, prev_hi, prev_lo, h_state
                )
                m = mask[:, step_t]
                pred_hi = mx.argmax(hi_logits, axis=-1).astype(mx.int32)
                pred_lo = mx.argmax(lo_logits, axis=-1).astype(mx.int32)
                correct_hi += ((pred_hi == ht[:, step_t]) * m).sum().item()
                correct_lo += ((pred_lo == lt[:, step_t]) * m).sum().item()
                total += m.sum().item()

                # Feed predictions forward (inference behavior)
                if epsilon >= 1.0:
                    prev_hi = ht[:, step_t]
                    prev_lo = lt[:, step_t]
                else:
                    use_gt = mx.random.uniform(shape=(B,)) < epsilon
                    prev_hi = mx.where(use_gt, ht[:, step_t], pred_hi)
                    prev_lo = mx.where(use_gt, lt[:, step_t], pred_lo)
        return min(correct_hi / total, correct_lo / total)

    print(f"  Training with scheduled sampling (ε: 1.0 → 0.1)")

    for step in range(steps):
        epsilon = linear_epsilon(step, steps)
        idx = mx.array(np.random.choice(n, batch_size, replace=False))
        loss, grads = nn.value_and_grad(model, loss_fn)(
            model, Hm[idx], Sm[idx], Tm[idx], HTm[idx], LTm[idx], Mm[idx], epsilon)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 250 == 0:
            # Report both: teacher-forced and pure inference accuracy
            tf_acc = eval_accuracy(1.0)
            inf_acc = eval_accuracy(0.0)
            print(f"  step {step:5d}  ε={epsilon:.2f}  loss={loss.item():.4f}  "
                  f"tf_acc={tf_acc:.1%}  inf_acc={inf_acc:.1%}")
            if inf_acc >= 0.9999:
                mx.savez("weights_gru.npz", **dict(tree_flatten(model.parameters())))
                print(f"  Saved weights (inf_acc={inf_acc:.4%})")
            if inf_acc == 1.0:
                perfect += 1
                if perfect >= 2:
                    print(f"  Converged.")
                    return model
            else:
                perfect = 0

    return model


# ── Main ───────────────────────────────────────────────────────────────

B = "\033[1m"
D = "\033[2m"
N = "\033[0m"


def main():
    print(f"{B}Reflex — Training (GRU + scheduled sampling){N}\n")
    print(f"{D}Autoregressive control head with exposure-bias-aware training{N}\n")

    backbone, tokenizer = load_backbone()

    print(f"\n{D}Generating tasks...{N}")
    tasks = generate_tasks()
    print(f"  {len(tasks)} tasks")

    print(f"\n{D}Collecting sequences...{N}")
    t0 = time.time()
    instr_cache = load_or_encode(tasks, backbone, tokenizer)
    print(f"  {len(instr_cache)} unique instructions")
    H, T, S, HT, LT, M, max_steps = collect_sequences(tasks, instr_cache)
    print(f"  Collected in {time.time()-t0:.1f}s")

    print(f"\n{D}Training...{N}")
    t0 = time.time()
    model = train(H, T, S, HT, LT, M, max_steps, steps=80000)
    print(f"  Trained in {time.time()-t0:.1f}s")

    # Validate with actual inference
    print(f"\n{D}Validating inference...{N}")
    from .chip8 import Chip8
    test_instrs = [
        "draw a smiley", "draw a snake", "draw a heart", "draw a star",
        "draw a circle", "draw a box",
        "draw digit 7 at position 15 10", "draw a 7",
        "compute 3 plus 5 and draw result", "3 + 5",
        "smiley", "heart", "show me a star",
    ]
    chip = Chip8()
    passed = 0
    for instr in test_instrs:
        chip.reset()
        h, tid = encode_instruction(instr, backbone, tokenizer)
        mx.eval(h)
        h_state = None
        prev_hi = prev_lo = None
        for _ in range(20):
            state = chip.get_state()
            hi_l, lo_l, h_state = model(
                h, mx.array(state[None]), mx.array(tid[None]),
                prev_hi, prev_lo, h_state
            )
            mx.eval(hi_l, lo_l, h_state)
            hi = int(mx.argmax(hi_l[0]).item())
            lo = int(mx.argmax(lo_l[0]).item())
            opcode = (hi << 8) | lo
            if opcode == 0x0000:
                break
            chip.step(opcode)
            prev_hi = mx.array([hi], dtype=mx.int32)
            prev_lo = mx.array([lo], dtype=mx.int32)
        pixels = int(chip.display.sum())
        ok = "✓" if pixels > 0 else "✗"
        if pixels > 0:
            passed += 1
        print(f"  {ok} {instr} ({pixels} pixels)")

    print(f"\n  Inference: {passed}/{len(test_instrs)} pass")

    mx.savez("weights_gru.npz", **dict(tree_flatten(model.parameters())))
    print(f"  Saved: weights_gru.npz")
    print(f"  Run: uv run demo-gru")


if __name__ == "__main__":
    main()
