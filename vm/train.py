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
    ReflexHead, SYSCALL_TABLE, ARG_DIM, BUF_MAX, INSTR_BYTES_MAX,
    CONTEXT_LEN, load_backbone, encode_instruction_last,
    encode_syscall_target, encode_prev_action, make_copy_targets,
    instruction_to_bytes, build_arg_vocab,
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
    instr_to_bytes = {}
    for instr in all_instrs:
        emb = encode_instruction_last(instr, backbone, tokenizer)
        mx.eval(emb)
        instr_to_emb[instr] = np.array(emb[0])
        instr_to_bytes[instr] = instruction_to_bytes(instr)
    print(f"  {len(all_instrs)} unique instructions encoded")

    windows, instr_embs, prev_actions = [], [], []
    nr_targets, arg_targets, buf_targets = [], [], []
    copy_gates, copy_sources, instr_bytes_list = [], [], []
    descriptions = []

    def record_task(instr, syscalls, state_history, prev_syscall_idx, prev_arg_indices):
        ib = instr_to_bytes[instr]
        for raw, desc in syscalls:
            state = vm.observe()
            window = np.zeros((CONTEXT_LEN, STATE_DIM), dtype=np.float32)
            hist = state_history + [state]
            for i, s in enumerate(hist[-CONTEXT_LEN:]):
                window[CONTEXT_LEN - len(hist[-CONTEXT_LEN:]) + i] = s

            syscall_idx, arg_indices, buf_target = encode_syscall_target(raw)
            gate, source = make_copy_targets(buf_target, ib)

            windows.append(window)
            instr_embs.append(instr_to_emb[instr])
            instr_bytes_list.append(ib)
            prev_actions.append(encode_prev_action(prev_syscall_idx, prev_arg_indices))
            nr_targets.append(syscall_idx)
            arg_targets.append(arg_indices)
            buf_targets.append(buf_target)
            copy_gates.append(gate)
            copy_sources.append(source)
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

    return (np.stack(windows), np.stack(instr_embs), np.stack(prev_actions),
            np.array(nr_targets, dtype=np.int32), np.stack(arg_targets),
            np.stack(buf_targets), np.stack(copy_gates), np.stack(copy_sources),
            np.stack(instr_bytes_list), descriptions, vocab)


# ── Training ───────────────────────────────────────────────────────────────

def train(W, E, PA, NR, AT, BT, CG, CS, IB, n_arg_vocab, steps=10000, lr=1e-3):
    head = ReflexHead(n_arg_vocab=n_arg_vocab)
    scheduler = optim.cosine_decay(lr, steps, end=lr * 0.01)
    optimizer = optim.Adam(learning_rate=scheduler)

    Wm, Em, PAm = mx.array(W), mx.array(E), mx.array(PA)
    NRm, ATm, BTm = mx.array(NR), mx.array(AT), mx.array(BT)
    CGm, CSm = mx.array(CG), mx.array(CS)
    IBm = mx.array(IB)

    n_samples = W.shape[0]
    batch_size = min(128, n_samples)

    def loss_fn(head, em, wm, pam, nrm, atm, btm, cgm, csm, ibm):
        syscall_logits, arg_logits, gen_logits, copy_logits, gate = head(em, wm, pam)

        cls_loss = nn.losses.cross_entropy(syscall_logits, nrm).mean()
        B, A, V = arg_logits.shape
        arg_loss = nn.losses.cross_entropy(
            arg_logits.reshape(B * A, V), atm.reshape(B * A)
        ).mean()

        # Mixed buffer loss: merge generate + copy into unified byte distribution
        # gen_logits: [B, BUF_MAX, 256]
        # copy_logits: [B, BUF_MAX, INSTR_BYTES_MAX]
        # gate: [B, BUF_MAX, 1]
        #
        # For each position, the probability of byte value v is:
        #   p(v) = (1-g) * softmax(gen)[v] + g * sum_j(softmax(copy)[j] * (instr[j]==v))
        #
        # But that's expensive. Instead, train all three heads directly:
        # 1. gen_logits should predict the right byte (always useful as fallback)
        # 2. copy_logits should point to the right source (when copying)
        # 3. gate should be 1 when copying, 0 when generating

        # Generate loss: on ALL positions (good fallback even for copy positions)
        B2, L, C = gen_logits.shape
        gen_loss = nn.losses.cross_entropy(
            gen_logits.reshape(B2 * L, C), btm.reshape(B2 * L)
        ).mean()

        # Copy loss: only on copy positions
        B3, L2, S = copy_logits.shape
        copy_loss_all = nn.losses.cross_entropy(
            copy_logits.reshape(B3 * L2, S), csm.reshape(B3 * L2)
        ).reshape(B3, L2)
        copy_loss = (copy_loss_all * cgm).sum() / (cgm.sum() + 1e-8)

        # Gate loss: balanced BCE
        gate_pred = mx.sigmoid(gate[:, :, 0])
        gate_bce = -(cgm * mx.log(gate_pred + 1e-8) +
                     (1 - cgm) * mx.log(1 - gate_pred + 1e-8))
        # Upweight copy positions to balance
        n_copy = cgm.sum() + 1e-8
        n_gen = (1 - cgm).sum() + 1e-8
        gate_weight = mx.where(cgm > 0.5, n_gen / n_copy, 1.0)
        gate_loss = (gate_bce * gate_weight).mean()

        return cls_loss + arg_loss + gen_loss + copy_loss + gate_loss

    perfect_count = 0

    for step in range(steps):
        idx = mx.array(np.random.choice(n_samples, batch_size, replace=False))

        loss, grads = nn.value_and_grad(head, loss_fn)(
            head, Em[idx], Wm[idx], PAm[idx], NRm[idx], ATm[idx],
            BTm[idx], CGm[idx], CSm[idx], IBm[idx],
        )
        optimizer.update(head, grads)
        mx.eval(head.parameters(), optimizer.state)

        if step % 200 == 0 or step == steps - 1:
            eval_n = min(256, n_samples)
            sl = slice(0, eval_n)
            s_log, a_log, g_log, c_log, gate = head(Em[sl], Wm[sl], PAm[sl])

            acc = (mx.argmax(s_log, axis=1) == NRm[sl]).mean().item()
            arg_acc = (mx.argmax(a_log, axis=2) == ATm[sl]).mean().item()

            # Decode buffer using gate
            gate_pred = (mx.sigmoid(gate[:, :, 0]) > 0.5)
            gen_pred = mx.argmax(g_log, axis=2)
            copy_pred_idx = mx.argmax(c_log, axis=2)

            # Reconstruct bytes
            ib_eval = mx.array(IB[:eval_n])  # [eval_n, INSTR_BYTES_MAX]
            # For copy positions, look up the byte from instruction
            copy_bytes = mx.take_along_axis(
                ib_eval,
                copy_pred_idx.astype(mx.int32),
                axis=1,
            )
            buf_pred = mx.where(gate_pred, copy_bytes, gen_pred)
            buf_acc = (buf_pred == BTm[sl]).mean().item()

            # Gate accuracy
            gate_acc = ((gate_pred == CGm[sl].astype(mx.bool_))).mean().item()

            print(f"  step {step:5d}  loss={loss.item():.4f}  "
                  f"syscall={acc:.1%}  args={arg_acc:.1%}  "
                  f"buf={buf_acc:.1%}  gate={gate_acc:.1%}")

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
    print("Frozen backbone: Qwen2.5-Coder-1.5B")
    print("Trainable: action head + copy buffer mechanism\n")

    backbone, tokenizer = load_backbone()

    print("\nBooting VM...")
    vm = VM()

    print("\nCollecting syscall traces...")
    t0 = time.time()
    W, E, PA, NR, AT, BT, CG, CS, IB, descs, vocab = collect(
        vm, backbone, tokenizer, n_rounds=30
    )
    print(f"  {len(NR)} traces, {len(set(descs))} unique syscall patterns")
    # Show copy stats
    total_positions = CG.size
    copy_positions = CG.sum()
    print(f"  Copy mechanism: {int(copy_positions)}/{total_positions} positions marked as copy "
          f"({copy_positions/total_positions*100:.1f}%)")
    print(f"  Collected in {time.time()-t0:.1f}s")

    print(f"\nTraining...")
    t0 = time.time()
    head = train(W, E, PA, NR, AT, BT, CG, CS, IB, n_arg_vocab=len(vocab), steps=10000)
    print(f"  Trained in {time.time()-t0:.1f}s")

    weights_path = os.path.join(DIR, "head_weights.npz")
    vocab_path = os.path.join(DIR, "arg_vocab.json")
    mx.savez(weights_path, **dict(tree_flatten(head.parameters())))
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

    # Test predictions
    print(f"\nTest predictions (first 20):")
    sl = slice(0, 20)
    s_log, a_log, g_log, c_log, gate = head(
        mx.array(E[sl]), mx.array(W[sl]), mx.array(PA[sl])
    )
    pred_nr = mx.argmax(s_log, axis=1).tolist()
    gate_pred = (mx.sigmoid(gate[:, :, 0]) > 0.5)
    gen_pred = mx.argmax(g_log, axis=2)
    copy_pred_idx = mx.argmax(c_log, axis=2)
    ib_eval = mx.array(IB[:20])
    copy_bytes = mx.take_along_axis(ib_eval, copy_pred_idx.astype(mx.int32), axis=1)
    buf_pred = np.array(mx.where(gate_pred, copy_bytes, gen_pred))

    for i in range(20):
        true_name = SYSCALL_TABLE[NR[i]]
        pred_name = SYSCALL_TABLE[pred_nr[i]]
        ok = "ok" if pred_nr[i] == NR[i] else "MISS"
        pred_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in buf_pred[i, :24])
        true_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in BT[i, :24])
        n_copy = int(CG[i].sum())
        print(f"  [{ok:4s}] {true_name:10s}→{pred_name:10s}  "
              f"true=\"{true_str}\"  pred=\"{pred_str}\"  (copy:{n_copy})")

    vm.stop()
    print(f"\nSaved: {weights_path} ({os.path.getsize(weights_path)//1024} KB)")
    print("Now run: uv run demo")


if __name__ == "__main__":
    main()
