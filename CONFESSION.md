# CONFESSION

This document exists to accompany the `reflex/decoders/` subpackage. Anything in that subpackage other than `pure.py` represents a deliberate departure from the architectural promises made in the repository's `README.md`. Each departure is enumerated here so a reader can evaluate the code honestly, without having to reverse-engineer intent from the diff.

## A note on the practice of confession

In the spring of 1995, a small group of Danish filmmakers circulated a short manifesto in Copenhagen. The document was polemical and intentionally severe: it declared a set of production directives whose purpose was to strip filmmaking back to what the group considered its essential acts — a camera present at an event, a performance occurring in front of it, and the minimum apparatus required to record those two things. The movement's logic was that the craft's prevailing habits — music tracks added after the fact, props chosen for mood, locations dressed to flatter a scene — were accretions: they solved problems the medium did not actually have and introduced problems of their own. The directives were an ascetic experiment in distillation. They asked what was left of cinema when the comforts of cinema were removed.

The manifesto is remembered for its severity, but its most interesting artifact was administrative: every filmmaker who shipped a work under the movement's banner was required to submit, alongside the finished film, a **confession**. The confession enumerated, plainly, each directive the filmmaker had broken. Not as penance; not as a score. As disclosure. The directives could not actually be followed to the letter in every case — artificial light was sometimes the only light, props were sometimes necessary for continuity, music sometimes said what silence could not — and the confession was how the filmmaker acknowledged, for the record, where compromise had been made and why. It was part of the work's honesty. The film existed; the confession made explicit what it had cost to make.

What's worth borrowing from that practice is not the severity — that's a matter of taste — but the idea that **a project with an expressive rhetoric about its own simplicity is obligated to be explicit when its code or its operation stops being simple**. The directives declare the intent; the confession documents the friction. Together they describe the actual object.

## The rhetoric of Reflex

The README of this project is not neutral. It uses specific, repeated phrases to describe the architecture:

> *"The thinnest possible bridge."*
>
> *"The model's output channel is the CPU's input channel, with no decode step between them."*
>
> *"Inference emits no text tokens — the model's output channel is the 32-bit instruction word, executed immediately."*
>
> *"The loop is hardware-native."*
>
> *"No chain-of-thought, no reasoning trace; just grounded emissions."*

These are architectural promises. Taken together they describe a specific object: at each cycle, one forward pass, one 32-bit word, one write, one step. The JEPA codebook introduces a 691-way nearest-neighbour decode, which is itself a decode step, but the README accepts this as within spirit because the decode is (a) deterministic, (b) over the space of real instructions, (c) a constant-time pure function of the prediction. The codebook snap fits inside the phrase *"thinnest possible bridge"* because it does not branch, does not run the model more than once per cycle, and does not consult the emulator.

Any decoder that introduces a branching selection stage, a multi-step lookahead, a tree of cloned CPUs, or a scoring function over simulated trajectories is **not** the thinnest possible bridge. It may be a better bridge by some metrics and a worse one by others; it is not, by construction, the one the README describes.

## Decoders in this package and their confessions

Each alternative decoder is a separate module in `reflex/decoders/`. The canonical decoder is included for uniformity of the call interface and is itself a confession of nothing — its behaviour is identical to `run_grounded`'s historical code path, inlined byte-for-byte in `demo.py`.

### `pure.py` — no confession

The canonical decoder. One forward pass per cycle; argmax over the JEPA codebook; one write at PC; one emulator step; loop until HALT or `max_cycles`. No lookahead, no branching, no state beyond the committed CPU and the prompt. This is the path described by the README and the path used for every numeric claim in the model card.

On the 23-task internal testbed: **19 / 23 passing (83%)**. Deterministic; factorial-5 emits exactly 91 ops on every run.

### `tiebreak.py` — small confession

Two narrow rules, both disabled on early cycles (the first two), both gated on a per-task intervention cap:

- **R1 (halt-guard).** When the codebook's top-1 is `HALT` or `0x00000000` and its cosine-similarity margin to top-2 is below a threshold, prefer top-2. Committing to program termination on weak evidence is a high-cost error; the program can still recover from an incorrect non-halt instruction but cannot recover from a premature halt.

- **R2 (one-step self-consistency).** When the top-1 / top-2 similarity margin is below a threshold, for each of the top-`k` candidates run the model forward *one* more time on a trial CPU that has been stepped under that candidate; pick the candidate whose induced next-cycle top-1 similarity is highest.

**What this breaks.** R1 does not branch: it replaces one deterministic decode step with a different deterministic decode step. R2 does branch — each low-margin cycle now performs `k` additional model forwards and `k` Unicorn steps on cloned CPUs, and the committed instruction is chosen by a function over their outcomes, not by direct argmax. The "one forward per cycle" property no longer holds at low-margin cycles; the per-cycle cost is no longer constant. The loop remains cycle-by-cycle — we do not search across cycles — so the "hardware-native loop" description is bent, not broken.

**What this bought us.** Nothing, measured on the existing checkpoint. `tiebreak` passes 19 / 23 — the same tasks as `pure`. Under aggressive settings it *regresses* to 15 / 23 because early-cycle interventions disrupt register-allocation conventions that the program relies on for its entire trajectory. The module is shipped anyway: the two rules are defensible on their own, the infrastructure is the substrate for the heavier decoders, and having a zero-lift control variant is useful for future checkpoints.

### `rd_consistency.py` — small confession

Two structural rules, both gated by a margin threshold, a minimum-cycle offset, and a per-task intervention cap. Neither consults the model for anything beyond the current cycle's top-``k`` similarities; neither runs the emulator speculatively.

- **Rule A (rd-reuse).** When the current top-1 is an ADDI whose destination register disagrees with the rd used by recent prior ADDIs in the emitted trace, and the top-``k`` contains an ADDI matching the prior rd, prefer the matching candidate (preferring alternates whose immediate and rs1 also match top-1 over looser rd-only matches). Rationale: compiler-generated RISC-V exhibits strong register-allocation locality within a basic block; when the model is already uncertain about which register to write, the program's established convention is a principled prior.

- **Rule B (halt-defer in an active byte-loop).** When the current top-1 is HALT with margin below ``halt_margin_eps`` AND the recent emission history is specifically an active byte-loop (≥ ``byte_loop_min_pairs`` back-to-back ``ADDI rd`` / ``STORE-from-rd`` pairs), prefer the top-``k`` ADDI that matches the loop's rd. The byte-loop check is what keeps this rule from firing on generic arithmetic-then-halt programs where a lone ADDI history exists but the program has no structural reason to continue.

**What this breaks.** The decode is no longer strictly the single-argmax of the canonical decoder at every cycle — at low-margin cycles it becomes a conditional pick over the top-``k``. However: the rules examine only the instruction-level structure of the emitted trace and the current top-``k`` from the codebook; they do not consult the model for further scores, they do not run the emulator speculatively, and they do not maintain any state beyond the already-committed trace. The "one forward per cycle" property is preserved. The deviation from "thinnest possible bridge" is a structural post-filter on the canonical argmax — thinner than either ``tiebreak`` or ``exec_verify``.

**What this bought us.** ``display OK`` is rescued from the baseline failure set: Rule A picks the matching-rd variant at the low-margin cycle-3 ADDI, and Rule B correctly defers a premature halt at cycle 5. Overall pass rate moves from 19/23 to **20/23**. Zero regressions in the remaining three tiers.

**Limits.** ``show 42`` still fails — the failing cycle is the program's *first* ADDI, so no rd-convention exists yet, and no top-``k`` ADDI shares a prior rd. ``print hello`` fires Rule B correctly (the byte-loop is unmistakable) but the top-``k`` at the halt-defer cycle contains multiple candidate bytes (addi x5, 't' / 'o' / 'l' / 'r' — all same opcode and rd, different immediates), and the rule has no principled signal for choosing among them. This is the same ceiling ``exec_verify`` hits: oracle-level validation can rule out *invalid* continuations but cannot pick between *semantically different but equally valid* ones without a target-intent signal. The principled fix is head-side — see ``HEAD_STUDY.md``.

### `exec_verify.py` — substantial confession

At low-margin cycles, fork each of the top-`k` codebook candidates, simulate `lookahead_depth` cycles of the resulting program in a cloned CPU under the canonical argmax policy, and pick the candidate whose forward trajectory scores highest on a small set of execution signals: cleanly halted within the lookahead window (preferred), no invalid memory write or step fault or emitted zero (required), highest cumulative top-1 similarity across the simulated cycles (tiebreaker).

**What this breaks.** At every intervention cycle the inference procedure runs `k × lookahead_depth` additional model forwards and steps. The committed instruction is selected by a scoring function evaluating a tree of possible futures. The decode stage is no longer a pure function of the current prediction; it is a pure function of `k` simulated futures. This is not the thinnest possible bridge; it is a bridge with a lookahead beam built into the decode. The "one forward per cycle" property is violated by construction — at intervention cycles the model is consulted `1 + k × lookahead_depth` times. The "hardware-native loop" property is violated because real CPUs do not speculatively execute candidate instructions in software-managed sandboxes and then pick among them.

**The argument for it anyway.** The Unicorn emulator is already treated as a ground-truth oracle at training time — the README states that every program in the training corpus was verified end-to-end in Unicorn before inclusion ("zero rejects"). Using the same oracle at inference time is not a new primitive; it is the same primitive applied at a different phase of the system's lifecycle. The candidates being consulted are codebook rows — real RV32I instructions — and the simulation is `cpu.step()` itself, not an abstract surrogate. A reader sympathetic to this argument might describe `exec_verify` as *the same reflex loop, applied recursively at uncertainty points*. A reader unsympathetic to it would note that "reflex" is, by name, the opposite of deliberation, and what this decoder performs at uncertainty points is deliberation.

### `beam.py` — full departure

Maintains `beam_width` parallel trajectories through the cycle tree. At each cycle every surviving beam proposes its top-`branching` codebook candidates; the resulting `beam_width × branching` extensions are scored by cumulative log-similarity and pruned back to `beam_width`. Completed (halted) beams are retained and ranked at the end; the winning trajectory's committed words are returned.

**What this breaks.** Nearly everything about the original description. The inference procedure no longer emits one opcode per cycle — it emits `beam_width × branching` opcodes per cycle across a population of CPUs, of which one trajectory is eventually chosen. The loop is no longer over cycles of a single CPU; it is over levels of a search tree. The "output channel is the CPU's input channel" framing survives only if one squints: the output channel is the input channel of **the winning CPU**, chosen after the fact. The decode is no longer any kind of pure function; it is a classical beam search, the decoding technique the README implicitly defines itself against.

**Why it is in the package at all.** Beam search is the obvious upper bound for any re-rank-based approach. Including it lets us measure what is achievable if we abandon the architectural restraint entirely — it is the ceiling analysis for this class of intervention. At `beam_width=1` the decoder degenerates to `pure`, which is a useful sanity check. Using `beam` as the deployment path for this project would, in our reading, require a different `README.md` — one that claimed different things, or claimed less.

## On the decision to break

The project's rhetoric is load-bearing: it is what makes Reflex a particular object rather than a generic RV32I code-generation model. The rhetorical framing — a frozen backbone, a thin bridge, a hardware-native loop, grounded per-cycle emission with no text and no hidden-state thread — is the design, in the sense that a different design would not need those promises to describe itself. A decoder that violates the promises is either describing a different system or asking the reader to tolerate a gap between language and code.

We think there is a position between the two: that the decoders in this subpackage are honest experiments on a real question — *how far can inference-time algorithmic work push fidelity before the system is no longer the one the README describes?* — and that answering that question requires writing the code and measuring. The finding on this checkpoint is that the inference-time algorithmic work available to us does not move the needle: the display-tier failures encode information (program-structural awareness, sequence position) that is not in the state vector, and no amount of self-consistency over model outputs can synthesise information that is not in the inputs. The production-worthy fix is upstream — enrich the state encoding, retrain the head — and when that is done the decoders in this package should be retired to their proper role as experimental tools.

Until then, this confession is the accounting. The canonical decoder is still the canonical decoder. The alternatives are marked as such. A reader who takes the README at its word is free to use only `pure.py` and is owed nothing else.

## Research grounding

The decoders in this package are not ad-hoc. Each approach has a corresponding body of 2024–2026 literature that motivated the specific shape it takes. The relevant threads, and the decision they drove:

- **Execution-grounded verification** (Planning-Guided Transformer Decoding, [arXiv:2601.14525](https://arxiv.org/abs/2601.14525), and the VEG task framing) — for "verifiable execution-grounded" domains with a deterministic oracle, the literature is consistent: use execution feedback, not model self-confidence, as the scoring signal. Reflex has an ideal VEG setup because Unicorn is exactly this oracle. This finding drove `exec_verify.py`'s preference ordering: no-error and halted-cleanly dominate; cumulative confidence is only the tertiary tiebreaker.

- **Lookahead Decoding** ([arXiv:2402.02057](https://arxiv.org/abs/2402.02057)) and the resumable-state-machine framing (OpenEncompass, arXiv:2512.03571) — the emit–step loop is already a state machine; beam search over the tree of states is the natural extension. `beam.py`'s population-of-CPUs structure is that framing applied directly.

- **Nearest Neighbor Speculative Decoding (NEST, [arXiv:2405.19325](https://arxiv.org/html/2405.19325))** — when top-k over a retrieval set are near-ties, the correct move is to treat them as a *retrieval candidate set* and use downstream validation to pick. The JEPA codebook's top-k at low-margin cycles is exactly a retrieval set of codebook neighbours; our validation channel is `cpu.step()`. This directly informs why `exec_verify.py` triggers only on low margin.

- **Confidence-aware fusion** (kNN-TRANX / N2C2 / HyRACC, [arXiv:2503.09218](https://arxiv.org/html/2503.09218)) — fuse the cosine-similarity distribution with an external confidence signal. Our external signal is execution validity; this informs the multi-tier sort key in `_score_trajectory`.

- **Margin over absolute similarity** — the 2026 decoding survey ("Make Every Token Count", 2025, [link](https://www.researchgate.net/profile/Haoran-Wang-96/publication/387703971_Make_Every_Token_Count_A_Systematic_Survey_on_Decoding_Methods_for_Foundation_Models)) finds, across domains, that relative confidence (margin) is a stronger correctness predictor than absolute confidence (top-1 sim). `exec_verify.py` defaults to margin scoring for this reason; the alternative ("sim") is retained as a configurable fallback for ablation.

- **Finite Scalar Quantization** ([openreview FSQ](https://openreview.net/forum?id=fcg9phFVzjd)) — not implemented here because the JEPA head's 691-row codebook shows no collapse (every row used). Mentioned for completeness: if future training hits codebook-capacity limits, FSQ offers an alternative head structure with no learned codebook at all.

### What the latent-recurrence work taught us that survives into JEPA

The `feat/latent-fidelity-testbed` branch is kept in the history because three findings from it remain load-bearing here:

1. **Margin is a real correctness signal.** In the latent-recurrence diagnosis, low margin correlated with failure; in JEPA the same correlation holds (display-tier failures cluster at margin < 0.5 while every passing task has margin ≥ 0.5). What changed is the *actuator*: previously we lacked a mechanism to exploit the signal, now we have three (top-k substitution, one-step self-consistency, execution verification). The decoders in this package are the payoff of that observation.

2. **MultiheadAttention bias terms dominate on near-zero inputs.** The ghost-token finding from the recurrence work — that `k_proj(ε·x) ≈ b_k` regardless of input content — implies a matching caveat for the JEPA embedding head: on low-norm predictions, the head's projection bias contributes a direction-preserving constant. This means `||pred||` is a confidence signal independent of cosine margin. We have not yet used it; documenting it here so a future iteration can.

3. **LayerNorm discards magnitude; only direction survives.** `table_similarity` is cosine, so magnitude is discarded downstream by design. But the *pre-head pooled hidden state* still has magnitude information that the current head throws away. Any head-side fix that wants to use magnitude would need to preserve it before the final normalization.

### A negative result on pred-norm as a second gate

The latent-recurrence work predicted that the embed-head's linear bias would dominate on near-zero inputs, so `||pred||` should carry confidence signal independent of cosine margin. `scripts/jepa_norm_probe.py` measured this across the 23-task suite and found only partial support: the failing-cycle distribution's p90 `pred_norm` (85) is ~2× the passing-cycle p90 (38), but the discriminative mass lives almost entirely in one task — `sum 1..10` has mean `pred_norm` 32.78 while most programs hover at 16–24. The display-tier failures, the ones we were hoping to rescue, look normal on `pred_norm` (16–17). Magnitude signals prompt-misinterpretation ("the model is confidently computing the wrong thing"), not byte-disambiguation. A norm-based gate would help with `sum 1..10` if there were a rescue mechanism that didn't itself need target-intent; there isn't one at inference.

Recorded as a dead-end because it refines the picture: for byte-ambiguity failures the ceiling really is the one argued in `HEAD_STUDY.md`, not a missing easy gate.

### A negative result worth recording

An earlier experiment (`scripts/jepa_lookahead_infer.py`) used the model's own next-cycle margin as the score for each candidate fork. This regressed from 19/23 to 11/23 — decisively worse than baseline. The lesson matches the literature's prediction: the model's self-confidence about the next step is maximized by picking paths where the next step is trivially determined, which is not the same as picking paths that are semantically correct. When the scoring signal is derived from the same model whose uncertainty you are trying to resolve, the circularity is structural. This is the finding that pushed us to build `exec_verify.py` (oracle-backed) rather than a purely model-driven re-rank.
