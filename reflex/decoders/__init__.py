"""Alternative inference-time decoders for the JEPA-head Reflex model.

The canonical inference path is ``reflex.demo.run_grounded`` — pure
argmax over the codebook, one forward per cycle. That path is the
spec-compliant baseline and is NOT modified by anything in this
subpackage.

Decoders in this subpackage deviate from the canonical path in
deliberate, confessed ways — see ``CONFESSION.md`` at the repo root for
the rhetorical/philosophical accounting of each deviation.

Interface
---------
Every decoder exposes::

    def run(model, tok, prompt, device, max_cycles, *,
            seed_memcpy=True, max_instr_tokens=..., use_chat_template=...,
            use_context_prefix=..., **decoder_kwargs)
        -> (cpu, emitted_words, halted, err_str, metadata_dict)

``metadata_dict`` carries per-decoder diagnostic info (overrides taken,
beam history, per-cycle margins, etc.). Callers may ignore it.

Registry
--------
``DECODERS`` maps a short name to the decoder's ``run`` callable.
``get(name)`` resolves a name and raises if unknown.
"""
from .pure import run as _run_pure
from .tiebreak import run as _run_tiebreak
from .exec_verify import run as _run_exec_verify
from .beam import run as _run_beam

DECODERS = {
    'pure':        _run_pure,
    'tiebreak':    _run_tiebreak,
    'exec_verify': _run_exec_verify,
    'beam':        _run_beam,
}


def get(name: str):
    if name not in DECODERS:
        raise ValueError(f'Unknown decoder {name!r}; '
                         f'choose from {sorted(DECODERS)}')
    return DECODERS[name]


__all__ = ['DECODERS', 'get']
