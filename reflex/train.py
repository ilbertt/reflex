"""
Train: frozen LLM backbone + autoregressive control head → RV32I programs.

Each program is emitted as a static 32-bit instruction sequence; verified
in unicorn before being added to training data. Trained with scheduled
sampling so pure-inference accuracy tracks teacher-forced accuracy.

Usage:
    uv run train
"""

import hashlib
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .model import (
    BACKBONE_DIM,
    FIELD_CLASSES,
    FIELD_NAMES,
    MAX_KV_LEN,
    MAX_TOKENS,
    ReflexModel,
    encode_instruction,
    load_backbone,
)
from .programs import generate_tasks
from .riscv import decompose


# ── Data collection ───────────────────────────────────────────────────

def load_or_encode(tasks, backbone, tokenizer):
    """Cache backbone encodings to disk."""
    instr_set = sorted(set(i for i, _ in tasks))
    hash_input = f"MAX_TOKENS={MAX_TOKENS}|" + "\n".join(instr_set)
    task_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    cache_file = f"encoding_cache_{task_hash}.npz"

    if os.path.exists(cache_file):
        print(f"  Loading cached encodings from {cache_file}...")
        data = np.load(cache_file, allow_pickle=True)
        return dict(data["cache"].item())

    print("  Encoding instructions through backbone...")
    instr_cache = {}
    for idx, instr in enumerate(instr_set):
        h, tid = encode_instruction(instr, backbone, tokenizer)
        mx.eval(h)
        instr_cache[instr] = (np.array(h[0]), tid)
        if (idx + 1) % 500 == 0:
            print(f"    {idx + 1}/{len(instr_set)} encoded...")

    np.savez(cache_file, cache=instr_cache)
    print(f"  Cached {len(instr_cache)} encodings to {cache_file}")
    return instr_cache


def collect_sequences(tasks, instr_cache):
    """Decompose each program into per-step 6-field targets."""
    programs = []
    for instr, program in tasks:
        hidden, tid = instr_cache[instr]
        # Program bytes are little-endian 32-bit instructions.
        n_ops = len(program) // 4
        field_targets = [[] for _ in FIELD_CLASSES]
        for i in range(n_ops):
            word = int.from_bytes(program[4*i:4*i+4], "little")
            for j, f in enumerate(decompose(word)):
                field_targets[j].append(f)
        # STOP token: all-zero fields.
        for j in range(len(FIELD_CLASSES)):
            field_targets[j].append(0)
        programs.append((hidden, tid, field_targets))

    max_steps = max(len(ft[0]) for _, _, ft in programs)
    max_seq = max(h.shape[0] for h, _, _ in programs)
    n = len(programs)

    H = np.zeros((n, max_seq, BACKBONE_DIM), dtype=np.float32)
    T = np.zeros((n, MAX_TOKENS), dtype=np.int32)
    # Per-field target tensors.
    Targets = [np.zeros((n, max_steps), dtype=np.int32) for _ in FIELD_CLASSES]
    M = np.zeros((n, max_steps), dtype=np.float32)

    for i, (hidden, tid, field_targets) in enumerate(programs):
        H[i, :hidden.shape[0], :] = hidden
        T[i] = tid
        for j, ft in enumerate(field_targets):
            for k, v in enumerate(ft):
                Targets[j][i, k] = v
                M[i, k] = 1.0

    total = int(M.sum())
    print(f"  {n} programs, {total} total steps, max {max_steps} steps/program")
    if max_steps > MAX_KV_LEN:
        raise ValueError(f"max_steps ({max_steps}) exceeds MAX_KV_LEN ({MAX_KV_LEN})")
    return H, T, Targets, M, max_steps


# ── Training ──────────────────────────────────────────────────────────

def linear_epsilon(step, total_steps, start=1.0, end=0.1):
    t = min(step / total_steps, 1.0)
    return start + (end - start) * t


def _update_history(history: mx.array, step_t: int, new_op: mx.array) -> mx.array:
    col_mask = mx.arange(MAX_KV_LEN) == step_t
    new_col = mx.broadcast_to(new_op[:, None], history.shape)
    return mx.where(col_mask[None, :], new_col, history)


def _compose_fields(fields: tuple[mx.array, ...]) -> mx.array:
    """Combine 6 per-field int arrays back into a 32-bit instruction."""
    opc, rd, f3, rs1, rs2, f7 = fields
    return ((f7 & 0x7F) << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) \
           | ((f3 & 0x7) << 12) | ((rd & 0x1F) << 7) | (opc & 0x7F)


def train(H, T, Targets, M, max_steps, steps=15000):
    model = ReflexModel()
    scheduler = optim.cosine_decay(3e-4, steps, end=1e-6)
    optimizer = optim.Adam(learning_rate=scheduler)

    Hm = mx.array(H)
    Tm = mx.array(T)
    TGT = [mx.array(t) for t in Targets]
    Mm = mx.array(M)
    n = len(H)
    batch_size = min(32, n)
    perfect = 0

    def loss_fn(model, h, t, tgts, mask, epsilon):
        B = h.shape[0]
        h_state = mx.zeros((B, model.dim))
        prev_fields = tuple(mx.zeros((B,), dtype=mx.int32) for _ in FIELD_CLASSES)
        history = mx.zeros((B, MAX_KV_LEN), dtype=mx.int32)
        total_loss = mx.array(0.0)

        for step_t in range(max_steps):
            logits, h_state = model(
                h, history, step_t + 1, t, prev_fields, h_state
            )
            step_loss = mx.array(0.0)
            pred_fields = []
            for j, (logit, target) in enumerate(zip(logits, tgts)):
                step_loss = step_loss + (
                    nn.losses.cross_entropy(logit, target[:, step_t]) * mask[:, step_t]
                ).sum()
                pred_fields.append(
                    mx.argmax(mx.stop_gradient(logit), axis=-1).astype(mx.int32)
                )
            total_loss = total_loss + step_loss

            # Scheduled sampling per-field (one ε decision per sample applied
            # to all 6 fields — keeps a sample consistent rather than mixing
            # GT/pred fields within one instruction).
            use_gt = mx.random.uniform(shape=(B,)) < epsilon
            new_fields = []
            for j, pred in enumerate(pred_fields):
                gt = tgts[j][:, step_t]
                new_fields.append(mx.where(use_gt, gt, pred))
            new_op = _compose_fields(tuple(new_fields))
            prev_fields = tuple(new_fields)
            history = _update_history(history, step_t, new_op)

        return total_loss / (mask.sum() * len(FIELD_CLASSES))

    eval_subset_size = min(1024, n)
    np.random.seed(0)
    eval_idx = np.sort(np.random.choice(n, eval_subset_size, replace=False))
    idx_arr = mx.array(eval_idx)
    Hev = Hm[idx_arr]
    Tev = Tm[idx_arr]
    TGTev = [t[idx_arr] for t in TGT]
    Mev = Mm[idx_arr]
    n_ev = eval_subset_size

    def eval_accuracy(epsilon):
        """Per-field accuracy (average) on the eval subset."""
        chunk = 128
        per_field_correct = [0] * len(FIELD_CLASSES)
        full_correct = 0                # all 6 fields right
        total = 0
        for i in range(0, n_ev, chunk):
            h = Hev[i:i+chunk]
            t = Tev[i:i+chunk]
            tgts = [x[i:i+chunk] for x in TGTev]
            mask = Mev[i:i+chunk]
            B = h.shape[0]
            h_state = mx.zeros((B, model.dim))
            prev_fields = tuple(mx.zeros((B,), dtype=mx.int32) for _ in FIELD_CLASSES)
            history = mx.zeros((B, MAX_KV_LEN), dtype=mx.int32)
            for step_t in range(max_steps):
                logits, h_state = model(
                    h, history, step_t + 1, t, prev_fields, h_state
                )
                m = mask[:, step_t]
                preds = [mx.argmax(l, axis=-1).astype(mx.int32) for l in logits]
                step_full_correct = None
                for j, (pred, target) in enumerate(zip(preds, tgts)):
                    correct_j = (pred == target[:, step_t]) * m
                    per_field_correct[j] += correct_j.sum().item()
                    step_full_correct = correct_j if step_full_correct is None \
                                        else step_full_correct * (pred == target[:, step_t])
                full_correct += step_full_correct.sum().item()
                total += m.sum().item()

                if epsilon >= 1.0:
                    new_fields = [target[:, step_t] for target in tgts]
                else:
                    use_gt = mx.random.uniform(shape=(B,)) < epsilon
                    new_fields = [mx.where(use_gt, tgts[j][:, step_t], preds[j])
                                  for j in range(len(FIELD_CLASSES))]
                prev_fields = tuple(new_fields)
                new_op = _compose_fields(prev_fields)
                history = _update_history(history, step_t, new_op)
        per_field_acc = [c / total for c in per_field_correct]
        return min(per_field_acc), full_correct / total, per_field_acc

    print(f"  Training with scheduled sampling (ε: 1.0 → 0.1)")

    for step in range(steps):
        epsilon = linear_epsilon(step, steps)
        idx = mx.array(np.random.choice(n, batch_size, replace=False))
        batch_tgts = [t[idx] for t in TGT]
        loss, grads = nn.value_and_grad(model, loss_fn)(
            model, Hm[idx], Tm[idx], batch_tgts, Mm[idx], epsilon)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 500 == 0:
            _min_tf, full_tf, per_tf = eval_accuracy(1.0)
            _min_if, full_if, per_if = eval_accuracy(0.0)
            per_str = " ".join(f"{n}:{a:.2f}" for n, a in zip(FIELD_NAMES, per_if))
            print(f"  step {step:5d}  ε={epsilon:.2f}  loss={loss.item():.4f}  "
                  f"tf={full_tf:.1%}  inf={full_if:.1%}  [{per_str}]", flush=True)
            if step > 0 and step % 1000 == 0 and full_if > 0.5:
                mx.savez("weights.npz", **dict(tree_flatten(model.parameters())))
                print(f"  Checkpoint saved (inf={full_if:.4%})", flush=True)
            if full_if >= 0.999:
                mx.savez("weights.npz", **dict(tree_flatten(model.parameters())))
                print(f"  Converged at inf={full_if:.4%}", flush=True)
                perfect += 1
                if perfect >= 2:
                    return model
            else:
                perfect = 0

    return model


# ── Main ──────────────────────────────────────────────────────────────

B = "\033[1m"
D = "\033[2m"
N = "\033[0m"


def main():
    print(f"{B}Reflex — Training (RV32I program synthesis){N}\n")
    print(f"{D}Six-field heads (opcode/rd/funct3/rs1/rs2/funct7). "
          f"GRU + scheduled sampling + token-ID pathway.{N}\n")

    backbone, tokenizer = load_backbone()

    print(f"\n{D}Generating tasks...{N}")
    tasks = generate_tasks()

    print(f"\n{D}Collecting sequences...{N}")
    t0 = time.time()
    instr_cache = load_or_encode(tasks, backbone, tokenizer)
    print(f"  {len(instr_cache)} unique instructions")
    H, T, Targets, M, max_steps = collect_sequences(tasks, instr_cache)
    print(f"  Collected in {time.time()-t0:.1f}s")

    print(f"\n{D}Training...{N}")
    t0 = time.time()
    model = train(H, T, Targets, M, max_steps, steps=15000)
    print(f"  Trained in {time.time()-t0:.1f}s")

    mx.savez("weights.npz", **dict(tree_flatten(model.parameters())))
    print(f"  Saved: weights.npz")
    print(f"  Run: uv run demo")


if __name__ == "__main__":
    main()
