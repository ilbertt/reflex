"""
Train the reflex action head on top of a frozen LLM backbone.

Usage:
    uv run train
"""

import os
import time
import json
import struct

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .kernel import VM, SYS, BUF_OFFSET, O_CREAT, O_WRONLY, O_TRUNC, pack_syscall, STATE_DIM
from .model import (
    ReflexHead, SYSCALL_TABLE, ARG_DIM, BUF_MAX, BACKBONE_DIM,
    CONTEXT_LEN, load_backbone,
    encode_instruction_full, encode_instruction_last,
    encode_syscall_target, encode_prev_action,
    build_arg_vocab,
)

DIR = os.path.dirname(os.path.abspath(__file__))


# ── Task definitions ───────────────────────────────────────────────────────

def _file_task(instr, filename, content):
    return (instr, [
        (pack_syscall(SYS["openat"], [-100, BUF_OFFSET, O_CREAT|O_WRONLY|O_TRUNC, 0o644],
                      filename.encode() + b"\x00"),
         f"openat({filename})"),
        (pack_syscall(SYS["write"], [3, BUF_OFFSET, len(content)], content.encode()),
         f"write('{content[:20]}...')"),
        (pack_syscall(SYS["close"], [3], b""),
         "close(fd=3)"),
    ])

def _mkdir_task(instr, dirname):
    return (instr, [
        (pack_syscall(SYS["mkdirat"], [-100, BUF_OFFSET, 0o755], dirname.encode() + b"\x00"),
         f"mkdirat({dirname})"),
    ])

def _delete_task(instr, filename):
    return (instr, [
        (pack_syscall(SYS["unlinkat"], [-100, BUF_OFFSET, 0], filename.encode() + b"\x00"),
         f"unlinkat({filename})"),
    ])


def make_tasks():
    T = []
    T.append(_file_task("create hello.py with hello world",
                         "hello.py", "print('hello world!')\n"))
    T.append(_file_task("create main.py with a loop",
                         "main.py", "for i in range(5):\n    print(i)\n"))
    T.append(_file_task("create app.py with pid",
                         "app.py", "import os\nprint('pid:', os.getpid())\n"))
    T.append(_file_task("create config.json",
                         "config.json", '{"port":8080,"debug":true}\n'))
    T.append(_file_task("create README.md",
                         "README.md", "# Project\nControlled by a neural net.\n"))
    T.append(_file_task("create test.py with assertions",
                         "test.py", "assert 1 + 1 == 2\nassert 2 * 3 == 6\nprint('all tests pass')\n"))
    T.append(_file_task("create data.csv",
                         "data.csv", "name,value\nalpha,1\nbeta,2\ngamma,3\n"))
    T.append(_file_task("create setup.sh",
                         "setup.sh", "#!/bin/sh\necho 'setting up...'\n"))
    T.append(_file_task("create sales.csv with sales data",
                         "sales.csv",
                         "month,revenue,costs\nJan,12000,8000\nFeb,15000,9200\nMar,11000,7800\nApr,18000,10500\nMay,22000,13000\nJun,19500,11800\n"))
    T.append(_file_task("create analyze.py to analyze csv",
                         "analyze.py",
                         "import csv\nr=list(csv.DictReader(open('sales.csv')))\n"
                         "rev=sum(int(x['revenue'])for x in r)\n"
                         "cost=sum(int(x['costs'])for x in r)\n"
                         "print(f'Revenue:${rev:,}')\n"
                         "print(f'Costs:  ${cost:,}')\n"
                         "print(f'Profit: ${rev-cost:,}')\n"
                         "print(f'Margin:{(rev-cost)/rev*100:.1f}%')\n"))

    for d in ["src", "lib", "tests", "myapp", "pkg"]:
        T.append(_mkdir_task(f"create directory {d}", d))

    T.append(_file_task("create src/main.py",
                         "src/main.py", "def main():\n    print('running')\nmain()\n"))
    T.append(_file_task("create lib/utils.py",
                         "lib/utils.py", "def add(a, b):\n    return a + b\n"))

    for f in ["hello.py", "main.py", "config.json", "data.csv"]:
        T.append(_delete_task(f"delete {f}", f))

    return T


def make_projects():
    projects = []
    projects.append([
        _file_task("create sales.csv with sales data", "sales.csv",
                   "month,revenue,costs\nJan,12000,8000\nFeb,15000,9200\nMar,11000,7800\nApr,18000,10500\nMay,22000,13000\nJun,19500,11800\n"),
        _file_task("create analyze.py to analyze csv", "analyze.py",
                   "import csv\nr=list(csv.DictReader(open('sales.csv')))\n"
                   "rev=sum(int(x['revenue'])for x in r)\n"
                   "cost=sum(int(x['costs'])for x in r)\n"
                   "print(f'Revenue:${rev:,}')\n"
                   "print(f'Costs:  ${cost:,}')\n"
                   "print(f'Profit: ${rev-cost:,}')\n"
                   "print(f'Margin:{(rev-cost)/rev*100:.1f}%')\n"),
    ])
    projects.append([
        _mkdir_task("create directory src", "src"),
        _file_task("create src/main.py", "src/main.py",
                   "def main():\n    print('running')\nmain()\n"),
        _file_task("create config.json", "config.json",
                   '{"port":8080,"debug":true}\n'),
        _file_task("create README.md", "README.md",
                   "# Project\nControlled by a neural net.\n"),
    ])
    return projects


def extract_arg_vocab(tasks, projects):
    all_vals = {0}
    for _, syscalls in tasks:
        for raw, _ in syscalls:
            all_vals.update(struct.unpack_from("<6q", raw, 4))
    for project in projects:
        for _, syscalls in project:
            for raw, _ in syscalls:
                all_vals.update(struct.unpack_from("<6q", raw, 4))
    return build_arg_vocab(all_vals)


# ── Data collection ────────────────────────────────────────────────────────

def collect(vm, backbone, tokenizer, n_rounds=10):
    tasks = make_tasks()
    projects = make_projects()

    vocab = extract_arg_vocab(tasks, projects)
    print(f"  Arg vocabulary: {len(vocab)} unique values")

    pre_configs = [
        {},
        {"old.txt": "old\n"},
        {"data.csv": "a,b\n1,2\n", "notes.md": "# N\n"},
        {"temp.py": "x=1\n", "log.txt": "started\n"},
        {"README.md": "# Old\n"},
    ]

    print("  Encoding instructions through backbone...")
    all_instrs = set(instr for instr, _ in tasks)
    for project in projects:
        for instr, _ in project:
            all_instrs.add(instr)

    instr_to_emb = {}
    for instr in all_instrs:
        emb = encode_instruction_last(instr, backbone, tokenizer)
        mx.eval(emb)
        instr_to_emb[instr] = np.array(emb[0])
    print(f"  {len(all_instrs)} unique instructions encoded")

    windows, instr_embs, prev_actions = [], [], []
    nr_targets, arg_targets, buf_targets = [], [], []
    descriptions = []

    def record_task(instr, syscalls, state_history, prev_syscall_idx, prev_arg_indices):
        for raw, desc in syscalls:
            state = vm.observe()
            window = np.zeros((CONTEXT_LEN, STATE_DIM), dtype=np.float32)
            hist = state_history + [state]
            for i, s in enumerate(hist[-CONTEXT_LEN:]):
                window[CONTEXT_LEN - len(hist[-CONTEXT_LEN:]) + i] = s

            syscall_idx, arg_indices, buf_target = encode_syscall_target(raw)

            windows.append(window)
            instr_embs.append(instr_to_emb[instr])
            prev_actions.append(encode_prev_action(prev_syscall_idx, prev_arg_indices))
            nr_targets.append(syscall_idx)
            arg_targets.append(arg_indices)
            buf_targets.append(buf_target)
            descriptions.append(desc)

            prev_syscall_idx = syscall_idx
            prev_arg_indices = arg_indices.astype(np.float32)
            vm.execute(raw)
            state_history.append(state)

        return prev_syscall_idx, prev_arg_indices

    for r in range(n_rounds):
        vm.run_command("rm -rf /workspace/* 2>/dev/null")
        pre = pre_configs[r % len(pre_configs)]
        for fname, content in pre.items():
            vm.create_file(fname, content)

        for instr, syscalls in tasks:
            record_task(instr, syscalls, [], -1, np.zeros(ARG_DIM, dtype=np.float32))

        for project in projects:
            vm.run_command("rm -rf /workspace/* 2>/dev/null")
            prev_si, prev_ai = -1, np.zeros(ARG_DIM, dtype=np.float32)
            state_hist = []
            for instr, syscalls in project:
                prev_si, prev_ai = record_task(instr, syscalls, state_hist, prev_si, prev_ai)

        if (r + 1) % 5 == 0:
            print(f"  Round {r + 1}/{n_rounds} done ({len(nr_targets)} traces)")

    W = np.stack(windows)
    E = np.stack(instr_embs)
    PA = np.stack(prev_actions)
    NR = np.array(nr_targets, dtype=np.int32)
    AT = np.stack(arg_targets)
    BT = np.stack(buf_targets)

    return W, E, PA, NR, AT, BT, descriptions, vocab


# ── Training ───────────────────────────────────────────────────────────────

def train(W, E, PA, NR, AT, BT, n_arg_vocab, steps=10000, lr=1e-3):
    head = ReflexHead(n_arg_vocab=n_arg_vocab)
    scheduler = optim.cosine_decay(lr, steps, end=lr * 0.01)
    optimizer = optim.Adam(learning_rate=scheduler)

    Wm, Em, PAm = mx.array(W), mx.array(E), mx.array(PA)
    NRm, ATm, BTm = mx.array(NR), mx.array(AT), mx.array(BT)

    n_samples = W.shape[0]
    batch_size = min(128, n_samples)

    def loss_fn(head, em, wm, pam, nrm, atm, btm):
        syscall_logits, arg_logits, buf_logits = head(em, wm, pam)

        cls_loss = nn.losses.cross_entropy(syscall_logits, nrm).mean()

        B, A, V = arg_logits.shape
        arg_loss = nn.losses.cross_entropy(
            arg_logits.reshape(B * A, V), atm.reshape(B * A)
        ).mean()

        B2, L, C = buf_logits.shape
        buf_loss = nn.losses.cross_entropy(
            buf_logits.reshape(B2 * L, C), btm.reshape(B2 * L)
        ).mean()

        return cls_loss + arg_loss + buf_loss

    perfect_count = 0

    for step in range(steps):
        idx = mx.array(np.random.choice(n_samples, batch_size, replace=False))

        loss, grads = nn.value_and_grad(head, loss_fn)(
            head, Em[idx], Wm[idx], PAm[idx], NRm[idx], ATm[idx], BTm[idx],
        )
        optimizer.update(head, grads)
        mx.eval(head.parameters(), optimizer.state)

        if step % 200 == 0 or step == steps - 1:
            eval_n = min(256, n_samples)
            sl = slice(0, eval_n)
            logits, arg_logits, buf_logits = head(Em[sl], Wm[sl], PAm[sl])

            acc = (mx.argmax(logits, axis=1) == NRm[sl]).mean().item()
            arg_acc = (mx.argmax(arg_logits, axis=2) == ATm[sl]).mean().item()
            buf_acc = (mx.argmax(buf_logits, axis=2) == BTm[sl]).mean().item()

            print(f"  step {step:5d}  loss={loss.item():.4f}  "
                  f"syscall={acc:.1%}  args={arg_acc:.1%}  buf={buf_acc:.1%}")

            if acc == 1.0 and arg_acc == 1.0 and buf_acc == 1.0:
                perfect_count += 1
                if perfect_count >= 3:
                    print(f"  Converged at step {step}.")
                    break
            else:
                perfect_count = 0

    return head


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("Reflex — Training\n")
    print("Frozen backbone: Qwen2.5-Coder-1.5B (full hidden states)")
    print("Trainable: action head + buffer head (cross-attention to instruction)\n")

    backbone, tokenizer = load_backbone()

    print("\nBooting VM...")
    vm = VM()

    print("\nCollecting syscall traces...")
    t0 = time.time()
    W, E, PA, NR, AT, BT, descs, vocab = collect(vm, backbone, tokenizer, n_rounds=30)
    print(f"  {len(NR)} traces, {len(set(descs))} unique syscall patterns")
    print(f"  Collected in {time.time()-t0:.1f}s")

    print(f"\nTraining...")
    t0 = time.time()
    head = train(W, E, PA, NR, AT, BT, n_arg_vocab=len(vocab), steps=10000)
    print(f"  Trained in {time.time()-t0:.1f}s")

    weights_path = os.path.join(DIR, "head_weights.npz")
    vocab_path = os.path.join(DIR, "arg_vocab.json")
    mx.savez(weights_path, **dict(tree_flatten(head.parameters())))
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

    print(f"\nTest predictions (first 20):")
    sl = slice(0, 20)
    logits, arg_logits, buf_logits = head(
        mx.array(E[sl]), mx.array(W[sl]), mx.array(PA[sl])
    )
    pred_nr = mx.argmax(logits, axis=1).tolist()
    pred_buf = np.array(mx.argmax(buf_logits, axis=2))

    for i in range(20):
        true_name = SYSCALL_TABLE[NR[i]]
        pred_name = SYSCALL_TABLE[pred_nr[i]]
        ok = "ok" if pred_nr[i] == NR[i] else "MISS"
        pred_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in pred_buf[i, :24])
        true_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in BT[i, :24])
        print(f"  [{ok:4s}] {true_name:10s}→{pred_name:10s}  "
              f"true=\"{true_str}\"  pred=\"{pred_str}\"")

    vm.stop()
    print(f"\nSaved: {weights_path} ({os.path.getsize(weights_path)//1024} KB)")
    print("Now run: uv run demo")


if __name__ == "__main__":
    main()
