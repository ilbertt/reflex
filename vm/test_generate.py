"""
Minimal test: can a small decoder generate file content from instruction embeddings?

Train on: 50 diverse "create X with Y" tasks
Test on:  10 unseen tasks with novel filenames AND content

The decoder sees the backbone embedding of the instruction and must
produce the exact file content bytes autoregressively.

Usage:
    uv run python -m vm.test_generate
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .model import (
    BACKBONE_DIM, BUF_MAX,
    load_backbone, encode_instruction_last,
)

DECODER_DIM = 256
START_TOKEN = 256


class ByteDecoder(nn.Module):
    """Tiny autoregressive byte decoder conditioned on instruction embedding."""
    def __init__(self, n_layers=2, dim=DECODER_DIM):
        super().__init__()
        self.dim = dim
        self.ctx_proj = nn.Linear(BACKBONE_DIM, dim)
        self.byte_emb = nn.Embedding(257, dim)  # 0-255 + start
        self.pos_emb = nn.Embedding(BUF_MAX, dim)
        self.layers = [
            (nn.MultiHeadAttention(dim, 4), nn.RMSNorm(dim),
             nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim), nn.RMSNorm(dim))
            for _ in range(n_layers)
        ]
        self.out_norm = nn.RMSNorm(dim)
        self.out_proj = nn.Linear(dim, 256)

    def __call__(self, ctx, byte_seq):
        """
        ctx: [B, BACKBONE_DIM]
        byte_seq: [B, T] — input bytes (shifted right, starts with START_TOKEN)
        Returns: [B, T, 256] logits
        """
        B, T = byte_seq.shape
        h = self.byte_emb(byte_seq) + self.pos_emb(mx.arange(T))
        h = h + self.ctx_proj(ctx)[:, None, :]

        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        for attn, norm1, ff1, act, ff2, norm2 in self.layers:
            r = norm1(h)
            h = h + attn(r, r, r, mask=mask)
            r = norm2(h)
            h = h + ff2(act(ff1(r)))

        return self.out_proj(self.out_norm(h))


def generate(model, ctx, max_len=BUF_MAX):
    """Autoregressive generation."""
    tokens = [START_TOKEN]
    for _ in range(max_len):
        inp = mx.array([tokens])
        logits = model(ctx, inp)
        mx.eval(logits)
        next_byte = int(mx.argmax(logits[0, -1]).item())
        if next_byte == 0:  # null = stop
            break
        tokens.append(next_byte)
    return bytes(tokens[1:])  # skip start token


def make_tasks():
    """Generate diverse file creation tasks."""
    train = [
        ("create hello.py that prints hello world", b"print('hello world!')\n"),
        ("create greet.py that prints good morning", b"print('good morning!')\n"),
        ("create bye.py that prints goodbye", b"print('goodbye!')\n"),
        ("create count.py that counts to 5", b"for i in range(5):\n    print(i)\n"),
        ("create count10.py that counts to 10", b"for i in range(10):\n    print(i)\n"),
        ("create sum.py that prints sum of 1 to 100", b"print(sum(range(101)))\n"),
        ("create add.py that adds 2 and 3", b"print(2 + 3)\n"),
        ("create mul.py that multiplies 7 and 8", b"print(7 * 8)\n"),
        ("create div.py that divides 10 by 3", b"print(10 / 3)\n"),
        ("create name.py that prints alice", b"print('alice')\n"),
        ("create bob.py that prints bob", b"print('bob')\n"),
        ("create hi.py that prints hi there", b"print('hi there')\n"),
        ("create yes.py that prints yes", b"print('yes')\n"),
        ("create no.py that prints no", b"print('no')\n"),
        ("create pi.py that prints pi", b"import math\nprint(math.pi)\n"),
        ("create rand.py that prints a random number", b"import random\nprint(random.random())\n"),
        ("create time.py that prints the time", b"import time\nprint(time.time())\n"),
        ("create len.py that prints length of hello", b"print(len('hello'))\n"),
        ("create upper.py that prints HELLO", b"print('hello'.upper())\n"),
        ("create rev.py that reverses hello", b"print('hello'[::-1])\n"),
        ("create list.py that prints a list", b"print([1, 2, 3])\n"),
        ("create dict.py that prints a dict", b"print({'a': 1})\n"),
        ("create set.py that prints a set", b"print({1, 2, 3})\n"),
        ("create type.py that prints type of 42", b"print(type(42))\n"),
        ("create abs.py that prints absolute of -5", b"print(abs(-5))\n"),
        ("create max.py that prints max of 3 7 1", b"print(max(3, 7, 1))\n"),
        ("create min.py that prints min of 3 7 1", b"print(min(3, 7, 1))\n"),
        ("create pow.py that prints 2 to the 10", b"print(2 ** 10)\n"),
        ("create hex.py that prints hex of 255", b"print(hex(255))\n"),
        ("create bin.py that prints binary of 42", b"print(bin(42))\n"),
    ]

    test = [
        ("create wave.py that prints wave", b"print('wave')\n"),
        ("create sub.py that subtracts 9 from 20", b"print(20 - 9)\n"),
        ("create count3.py that counts to 3", b"for i in range(3):\n    print(i)\n"),
        ("create night.py that prints good night", b"print('good night!')\n"),
        ("create square.py that prints square of 7", b"print(7 ** 2)\n"),
        ("create lower.py that prints WORLD lowercase", b"print('WORLD'.lower())\n"),
        ("create oct.py that prints octal of 64", b"print(oct(64))\n"),
        ("create chr.py that prints chr of 65", b"print(chr(65))\n"),
        ("create ord.py that prints ord of A", b"print(ord('A'))\n"),
        ("create sorted.py that sorts 3 1 2", b"print(sorted([3, 1, 2]))\n"),
    ]

    return train, test


def main():
    train_tasks, test_tasks = make_tasks()

    print("Loading backbone...")
    backbone, tokenizer = load_backbone()

    print("Encoding instructions...")
    train_embs, train_targets = [], []
    for instr, content in train_tasks:
        emb = encode_instruction_last(instr, backbone, tokenizer)
        mx.eval(emb)
        train_embs.append(np.array(emb[0]))
        buf = np.zeros(BUF_MAX, dtype=np.int32)
        for i, b in enumerate(content[:BUF_MAX]):
            buf[i] = b
        train_targets.append(buf)

    test_embs, test_targets, test_contents = [], [], []
    for instr, content in test_tasks:
        emb = encode_instruction_last(instr, backbone, tokenizer)
        mx.eval(emb)
        test_embs.append(np.array(emb[0]))
        buf = np.zeros(BUF_MAX, dtype=np.int32)
        for i, b in enumerate(content[:BUF_MAX]):
            buf[i] = b
        test_targets.append(buf)
        test_contents.append(content)

    E = mx.array(np.stack(train_embs))
    BT = mx.array(np.stack(train_targets))

    # Teacher forcing inputs: [START, b0, b1, ...]
    n = len(train_tasks)
    buf_in = np.full((n, BUF_MAX), START_TOKEN, dtype=np.int32)
    buf_in[:, 1:] = np.stack(train_targets)[:, :-1]
    BI = mx.array(buf_in)

    print(f"\nTrain: {len(train_tasks)} tasks")
    print(f"Test:  {len(test_tasks)} tasks (UNSEEN instructions + content)\n")

    model = ByteDecoder(n_layers=2, dim=DECODER_DIM)
    optimizer = optim.Adam(learning_rate=1e-3)

    def loss_fn(model, em, bi, btm):
        logits = model(em, bi)  # [B, T, 256]
        B, T, C = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, C), btm.reshape(B * T)
        ).mean()

    print("Training...")
    for step in range(5000):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, E, BI, BT)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 500 == 0:
            logits = model(E, BI)
            preds = mx.argmax(logits, axis=2)
            acc = (preds == BT).mean().item()
            print(f"  step {step:4d}  loss={loss.item():.4f}  byte_acc={acc:.1%}")

    # Test on UNSEEN tasks
    print(f"\n{'='*60}")
    print("TEST: unseen instructions + content")
    print(f"{'='*60}\n")

    E_test = mx.array(np.stack(test_embs))

    n_pass = 0
    for i, (instr, content) in enumerate(test_tasks):
        ctx = E_test[i:i+1]
        generated = generate(model, ctx)
        expected = content.rstrip(b'\n')
        got = generated.rstrip(b'\n')
        match = "PASS" if got == expected else "FAIL"
        if match == "PASS":
            n_pass += 1
        print(f"  [{match}] '{instr}'")
        print(f"         expected: {expected}")
        print(f"              got: {got}")
        print()

    print(f"Result: {n_pass}/{len(test_tasks)} unseen tasks correct")

    # Also show a few train tasks
    print(f"\nTrain tasks (sanity check):")
    for i in range(5):
        instr, content = train_tasks[i]
        ctx = E[i:i+1]
        generated = generate(model, ctx)
        expected = content.rstrip(b'\n')
        got = generated.rstrip(b'\n')
        match = "PASS" if got == expected else "FAIL"
        print(f"  [{match}] '{instr}' → {got}")


if __name__ == "__main__":
    main()
