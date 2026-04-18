"""
RV32I program loader + verifier.

Programs live under the top-level ``programs/`` directory as JSON files
organized by family (arithmetic, loops, display, ...). Each JSON file
contains one family with a list of ``{instruction, instruction_variants,
bytes, expected_result, num_ops}`` entries. ``load_tasks()`` walks the
tree and emits one ``(family, instruction_text, bytes)`` tuple per
(program × variant) pair, matching the legacy in-code generator API.
"""
from __future__ import annotations

import json
from pathlib import Path

from .riscv import DATA_BASE, HALT_INSTR, Rv32i


# Memory layout constants (shared with demo/eval).
DISPLAY_BASE = 0x6000                     # 0x5000 + 0x1000; inside data region
DISPLAY_OFFSET = DISPLAY_BASE - DATA_BASE
SRC_OFFSET = 0x100
DST_OFFSET = 0x200


# Path to the JSON program corpus (top-level sibling of this package).
PROGRAMS_DIR = Path(__file__).resolve().parent.parent / 'programs'


def load_tasks(programs_dir: Path | str | None = None
               ) -> list[tuple[str, str, bytes]]:
    """Walk the JSON corpus and return ``[(family, text, bytes), ...]``.

    One task per (program × natural-language variant); mirrors the old
    in-code generator.
    """
    root = Path(programs_dir) if programs_dir is not None else PROGRAMS_DIR
    tasks: list[tuple[str, str, bytes]] = []
    for jf in sorted(root.rglob('*.json')):
        with open(jf) as fh:
            doc = json.load(fh)
        family = doc.get('family') or jf.stem
        for prog in doc.get('programs', []):
            prog_bytes = bytes(prog['bytes'])
            variants = prog.get('instruction_variants') or [prog['instruction']]
            for text in variants:
                tasks.append((family, text, prog_bytes))
    return tasks


def verify_program(program: bytes, expected: dict, max_cycles: int = 5000,
                   seed: dict | None = None) -> tuple[bool, str]:
    """Run program in unicorn, halt on self-jump (jal x0, 0).

    `expected` keys (all optional):
      - mem_word: (addr, val) — memory[addr:addr+4] as uint32 must equal val
      - reg: (idx, val) — x[idx] must equal val (unsigned)
      - mem_words: (addr, [v0, v1, ...]) — N consecutive words
    `seed`: {addr: [words...]} — pre-fill memory before running.
    """
    cpu = Rv32i()
    cpu.load_program(program)
    if seed:
        for addr, words in seed.items():
            data = b"".join(int(w & 0xFFFFFFFF).to_bytes(4, "little") for w in words)
            cpu.uc.mem_write(addr, data)
    halted = False
    for _ in range(max_cycles):
        try:
            instr = cpu.fetch()
        except Exception as e:
            return False, f"fetch error: {e}"
        if instr == HALT_INSTR:
            halted = True
            break
        if instr == 0x00000000:
            return False, f"hit 0x0 at pc={cpu.pc:#x}"
        try:
            cpu.step()
        except Exception as e:
            return False, f"step error at pc={cpu.pc:#x}: {e}"
    if not halted:
        return False, "did not halt within cycle budget"

    if "mem_word" in expected:
        addr, val = expected["mem_word"]
        got = cpu.mem_word(addr)
        if got != (val & 0xFFFFFFFF):
            return False, f"mem[{addr:#x}]={got} != {val}"
    if "mem_words" in expected:
        addr, vals = expected["mem_words"]
        for i, v in enumerate(vals):
            got = cpu.mem_word(addr + 4 * i)
            if got != (v & 0xFFFFFFFF):
                return False, f"mem[{addr+4*i:#x}]={got} != {v}"
    if "reg" in expected:
        idx, val = expected["reg"]
        got = cpu.reg(idx)
        if got != (val & 0xFFFFFFFF):
            return False, f"x{idx}={got} != {val}"
    return True, "ok"
