"""rd-consistency decoder — structural tiebreakers for RISC-V register
conventions and byte-loop termination.

The rules below key on *instruction-level patterns* in the emitted
trace — opcode, rd, rs2, funct3 — not on specific prompts, tags, or
values. The intent is that they generalize to any RISC-V program that
exhibits the patterns these rules detect, not just the 23-task suite.

Rule A — rd-reuse
    When the current top-1 is an ADDI whose rd disagrees with the
    destination register used by recent prior ADDIs in the emitted
    trace, and the top-``k`` contains an ADDI that matches the prior
    rd, prefer the matching candidate. Prefer candidates whose
    immediate and rs1 *also* match top-1 (same compute, different rd)
    over looser rd-only matches.

    Rationale: compiler-generated RISC-V code (GCC, LLVM, and the
    training corpus's synthetic generators) exhibits strong
    register-allocation locality within a basic block. When the model
    is *already uncertain* (low margin) about which register to write,
    the program's established register convention is a principled
    prior.

Rule B — halt-defer in an active byte-loop
    When the current top-1 is HALT (or an emitted-zero fault) with
    margin below ``halt_margin_eps``, AND the recent emission history
    is specifically an active byte-by-byte write loop (at least
    ``byte_loop_min_pairs`` back-to-back ``ADDI rd`` / ``STORE-from-rd``
    pairs), prefer the top-``k`` ADDI that matches the loop's rd.

    Rationale: a program in the middle of a byte loop (display writes,
    element-wise memory copies, etc.) has a structurally recognisable
    signature of alternating ADDI and STORE on a single rd. A halt
    emitted *during* that signature, with low confidence, is very
    likely premature. The byte-loop check is what keeps this rule
    from firing on generic arithmetic-then-halt programs where a lone
    ADDI history exists but the program has no reason to continue.

Neither rule inspects the current cycle's prediction score for
anything except the margin gate, and neither runs the emulator
speculatively. The decode stage remains a pure function of the
current prediction plus the already-emitted history — history that is
already in the state window the state encoder reads. The deviation
from the canonical decoder is correspondingly thin. See
``CONFESSION.md`` for the spirit-cost accounting.
"""
from ..model import MAX_INSTR_TOKENS
from ..riscv import HALT_INSTR, Rv32i
from .base import emit_and_step, fresh_cpu, predict_topk, prep_prompt


ADDI_OPCODE  = 0x13
ADDI_FUNCT3  = 0x0
STORE_OPCODE = 0x23       # sb / sh / sw


def _is_addi(w: int) -> bool:
    return (w & 0x7F) == ADDI_OPCODE and ((w >> 12) & 0x7) == ADDI_FUNCT3


def _is_store(w: int) -> bool:
    return (w & 0x7F) == STORE_OPCODE


def _rd(w: int) -> int:
    return (w >> 7) & 0x1F


def _store_src(w: int) -> int:
    """rs2 — the source register of a STORE instruction."""
    return (w >> 20) & 0x1F


def _in_byte_loop(committed: list[int], rd: int, min_pairs: int = 2) -> bool:
    """True if the recent emitted history contains at least ``min_pairs``
    ``ADDI <rd>, x0, imm`` → ``STORE <rd>, ...`` back-to-back pairs.

    This is the structural signature of a byte-by-byte memory write
    loop (the pattern used by display-string programs and also by
    memcpy-style element-by-element copies). Firing a halt-defer rule
    only when the program is actually in this pattern keeps the rule
    from false-positive'ing on generic arithmetic-then-halt programs
    that happen to have any ADDI history.
    """
    pairs = 0
    i = len(committed) - 1
    # Walk backwards, match STORE-from-rd followed by ADDI-to-rd.
    while i >= 1 and pairs < min_pairs:
        if _is_store(committed[i]) and _store_src(committed[i]) == rd \
                and _is_addi(committed[i-1]) and _rd(committed[i-1]) == rd:
            pairs += 1
            i -= 2
            continue
        break
    return pairs >= min_pairs


def _imm_and_rs1_match(a: int, b: int) -> bool:
    """True if two ADDIs have the same immediate and same rs1. (Opcode
    and funct3 are ADDI by construction when this helper is called.)"""
    return ((a >> 15) & 0x1F) == ((b >> 15) & 0x1F) and \
           ((a >> 20) & 0xFFF) == ((b >> 20) & 0xFFF)


def _prior_addi_rd(committed: list[int], lookback: int) -> int | None:
    """Return the rd of the most recent ADDI in the last ``lookback``
    committed words, or None if there is none. Non-ADDI instructions
    are ignored; the lookback window only considers ADDI-class ops."""
    for w in reversed(committed[-lookback:] if lookback else committed):
        if _is_addi(w):
            return _rd(w)
    return None


def run(model, tok, prompt: str, device: str, max_cycles: int, *,
        seed_memcpy: bool = True,
        max_instr_tokens: int = MAX_INSTR_TOKENS,
        use_chat_template: bool = True,
        use_context_prefix: bool = False,
        margin_eps: float = 0.15,
        halt_margin_eps: float = 0.35,
        topk: int = 5,
        min_cycle: int = 2,
        lookback: int = 8,
        byte_loop_min_pairs: int = 2,
        max_interventions: int = 3,
        **_unused,
        ) -> tuple[Rv32i, list[int], bool, str, dict]:
    ids, amask = prep_prompt(tok, prompt, device,
                              max_instr_tokens=max_instr_tokens,
                              use_chat_template=use_chat_template,
                              use_context_prefix=use_context_prefix)
    cpu = fresh_cpu(seed_memcpy)
    committed: list[int] = []
    halted, err = False, ''
    overrides: list[dict] = []
    interventions_used = 0

    for cycle in range(max_cycles):
        pc = cpu.pc
        top_words, top_sims = predict_topk(model, ids, amask, cpu, device, topk=topk)
        top1_w = top_words[0]
        margin = top_sims[0] - top_sims[1] if len(top_sims) > 1 else 1.0
        chosen_w = top1_w

        eligible_low_margin = (cycle >= min_cycle
                                and interventions_used < max_interventions
                                and margin < margin_eps)

        # Rule A: ADDI top-1 whose rd disagrees with the program's
        # established register convention. Pick the top-k ADDI that
        # matches the convention, preferring immediate-matching alts.
        if eligible_low_margin and _is_addi(top1_w):
            prior_rd = _prior_addi_rd(committed, lookback)
            top1_rd = _rd(top1_w)
            if prior_rd is not None and top1_rd != prior_rd:
                immediate_match = None
                rd_match = None
                for cand in top_words[1:]:
                    if not _is_addi(cand):
                        continue
                    if _rd(cand) != prior_rd:
                        continue
                    if rd_match is None:
                        rd_match = cand
                    if _imm_and_rs1_match(cand, top1_w):
                        immediate_match = cand
                        break
                alt = immediate_match or rd_match
                if alt is not None:
                    chosen_w = alt
                    interventions_used += 1
                    overrides.append({
                        'cycle': cycle,
                        'rule': ('rd_reuse_imm_match' if immediate_match
                                 else 'rd_reuse'),
                        'from': f'0x{top1_w:08x}',
                        'to': f'0x{chosen_w:08x}',
                        'margin': round(margin, 4),
                        'prior_rd': prior_rd,
                    })

        # Rule B: top-1 is a HALT/zero with confidence below the
        # halt-specific threshold, AND the program is demonstrably in
        # a byte-by-byte write loop (at least ``byte_loop_min_pairs``
        # recent ADDI-rd → STORE-from-rd pairs). The byte-loop check
        # is what keeps this rule from firing on generic
        # arithmetic-then-halt programs where a lone ADDI history
        # exists but there is no reason to continue.
        elif (cycle >= min_cycle
              and interventions_used < max_interventions
              and margin < halt_margin_eps
              and top1_w in (HALT_INSTR, 0)):
            prior_rd = _prior_addi_rd(committed, lookback)
            if prior_rd is not None and _in_byte_loop(
                    committed, prior_rd, min_pairs=byte_loop_min_pairs):
                alt = None
                for cand in top_words[1:]:
                    if _is_addi(cand) and _rd(cand) == prior_rd:
                        alt = cand
                        break
                if alt is not None:
                    chosen_w = alt
                    interventions_used += 1
                    overrides.append({
                        'cycle': cycle,
                        'rule': 'halt_defer_in_byte_loop',
                        'from': f'0x{top1_w:08x}',
                        'to': f'0x{chosen_w:08x}',
                        'margin': round(margin, 4),
                        'prior_rd': prior_rd,
                    })

        committed.append(chosen_w)
        halted, err = emit_and_step(cpu, pc, chosen_w)
        if halted or err:
            break

    return cpu, committed, halted, err, {
        'decoder': 'rd_consistency',
        'overrides': overrides,
        'interventions_used': interventions_used,
    }
