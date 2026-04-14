"""
Minimal test: does the copy mechanism generalize to unseen filenames?

Train on: hello.py, main.py, app.py, config.json, README.md
Test on:  foo.py, bar.py, notes.txt, output.csv, run.sh

Same instruction pattern "create X", same syscall (openat).
If the model learns to copy the filename from the instruction,
the unseen filenames should work. If it memorizes, they won't.

Usage:
    uv run python -m vm.test_copy
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .model import (
    INSTR_BYTES_MAX, BUF_MAX, BACKBONE_DIM, INNER_DIM,
    instruction_to_bytes, make_copy_targets,
    load_backbone, encode_instruction_last,
)


# Tiny model: just the copy buffer head, no VM, no state
class TinyCopyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(BACKBONE_DIM, INNER_DIM)
        self.gen_head = nn.Linear(INNER_DIM, BUF_MAX * 256)
        self.copy_head = nn.Linear(INNER_DIM, BUF_MAX * INSTR_BYTES_MAX)
        self.gate_head = nn.Linear(INNER_DIM, BUF_MAX)

    def __call__(self, emb):
        h = nn.relu(self.proj(emb))
        return (
            self.gen_head(h).reshape(-1, BUF_MAX, 256),
            self.copy_head(h).reshape(-1, BUF_MAX, INSTR_BYTES_MAX),
            self.gate_head(h).reshape(-1, BUF_MAX, 1),
        )


def make_data(filenames, backbone, tokenizer):
    """Create training data: 'create X' → openat buffer = X\0"""
    embs, targets, gates, sources, ibs = [], [], [], [], []
    for fname in filenames:
        instr = f"create {fname}"
        emb = encode_instruction_last(instr, backbone, tokenizer)
        mx.eval(emb)
        embs.append(np.array(emb[0]))

        buf = np.zeros(BUF_MAX, dtype=np.int32)
        for i, b in enumerate(fname.encode("utf-8")):
            buf[i] = b
        # null terminator is already 0

        ib = instruction_to_bytes(instr)
        gate, source = make_copy_targets(buf, ib)

        targets.append(buf)
        gates.append(gate)
        sources.append(source)
        ibs.append(ib)

    return (np.stack(embs), np.stack(targets), np.stack(gates),
            np.stack(sources), np.stack(ibs))


def decode(gen_logits, copy_logits, gate, instr_bytes):
    g = np.array(mx.sigmoid(gate[:, 0]))
    gen = np.array(gen_logits)
    cop = np.array(copy_logits)

    result = []
    for i in range(BUF_MAX):
        if g[i] > 0.5:
            src = int(np.argmax(cop[i]))
            result.append(instr_bytes[src])
        else:
            result.append(int(np.argmax(gen[i])))
    return bytes(result).split(b'\x00')[0].decode('utf-8', errors='replace')


def main():
    train_files = ["hello.py", "main.py", "app.py", "config.json", "README.md",
                   "data.csv", "setup.sh", "test.py", "index.html", "server.py"]
    test_files = ["foo.py", "bar.txt", "notes.md", "output.csv", "run.sh",
                  "xyz.py", "report.json", "build.sh", "temp.log", "calc.py"]

    print("Loading backbone...")
    backbone, tokenizer = load_backbone()

    print("Preparing data...")
    E_train, BT_train, CG_train, CS_train, IB_train = make_data(train_files, backbone, tokenizer)
    E_test, BT_test, CG_test, CS_test, IB_test = make_data(test_files, backbone, tokenizer)

    # Show copy analysis
    for i, fname in enumerate(train_files[:3]):
        n_copy = int(CG_train[i].sum())
        print(f"  '{fname}': {n_copy} bytes copied from instruction")

    print(f"\nTrain: {len(train_files)} filenames")
    print(f"Test:  {len(test_files)} filenames (UNSEEN)")

    # Train
    model = TinyCopyHead()
    optimizer = optim.Adam(learning_rate=1e-3)

    Em = mx.array(E_train)
    BTm = mx.array(BT_train)
    CGm = mx.array(CG_train)
    CSm = mx.array(CS_train)

    def loss_fn(model, em, btm, cgm, csm):
        gen, copy, gate = model(em)

        B, L, C = gen.shape
        gen_loss = nn.losses.cross_entropy(
            gen.reshape(B * L, C), btm.reshape(B * L)
        ).mean()

        B2, L2, S = copy.shape
        copy_loss_all = nn.losses.cross_entropy(
            copy.reshape(B2 * L2, S), csm.reshape(B2 * L2)
        ).reshape(B2, L2)
        copy_loss = (copy_loss_all * cgm).sum() / (cgm.sum() + 1e-8)

        gate_pred = mx.sigmoid(gate[:, :, 0])
        n_copy = cgm.sum() + 1e-8
        n_gen = (1 - cgm).sum() + 1e-8
        gate_weight = mx.where(cgm > 0.5, n_gen / n_copy, 1.0)
        gate_bce = -(cgm * mx.log(gate_pred + 1e-8) +
                     (1 - cgm) * mx.log(1 - gate_pred + 1e-8))
        gate_loss = (gate_bce * gate_weight).mean()

        return gen_loss + copy_loss + gate_loss

    print("\nTraining...")
    for step in range(2000):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, Em, BTm, CGm, CSm)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 200 == 0:
            gen, copy, gate = model(Em)
            gate_pred = (mx.sigmoid(gate[:, :, 0]) > 0.5)
            gen_pred = mx.argmax(gen, axis=2)
            copy_idx = mx.argmax(copy, axis=2)
            ib_mx = mx.array(IB_train)
            copy_bytes = mx.take_along_axis(ib_mx, copy_idx.astype(mx.int32), axis=1)
            buf_pred = mx.where(gate_pred, copy_bytes, gen_pred)
            acc = (buf_pred == BTm).mean().item()
            gate_acc = (gate_pred == (CGm > 0.5)).mean().item()
            print(f"  step {step:4d}  loss={loss.item():.4f}  buf={acc:.1%}  gate={gate_acc:.1%}")

    # Test on UNSEEN filenames
    print(f"\n{'='*60}")
    print("TEST: unseen filenames (never in training)")
    print(f"{'='*60}\n")

    E_test_mx = mx.array(E_test)
    gen, copy, gate = model(E_test_mx)

    for i, fname in enumerate(test_files):
        predicted = decode(gen[i], copy[i], gate[i], IB_test[i])
        match = "PASS" if predicted == fname else "FAIL"
        print(f"  [{match}] 'create {fname}' → openat buffer = '{predicted}'")

    # Also test train filenames for comparison
    print(f"\nTrain filenames (sanity check):")
    gen, copy, gate = model(Em)
    for i, fname in enumerate(train_files[:5]):
        predicted = decode(gen[i], copy[i], gate[i], IB_train[i])
        match = "PASS" if predicted == fname else "FAIL"
        print(f"  [{match}] 'create {fname}' → '{predicted}'")


if __name__ == "__main__":
    main()
