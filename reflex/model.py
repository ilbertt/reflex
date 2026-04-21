"""
Flamingo-SQL — grounded text-to-SQL via gated cross-attention.

Frozen Qwen2.5-Coder-3B-Instruct backbone processes the natural-language
question. Every ``INJECT_EVERY`` transformer layers, a Flamingo-style
gated cross-attention adapter is spliced in via a forward hook: the
backbone's current hidden states (queries) attend over live database
state K/V (schema tokens + sample-row tokens + optional prior-result
tokens). SQL is generated as text using the backbone's LM head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

BACKBONE_ID = 'Qwen/Qwen2.5-Coder-3B-Instruct'
INJECT_EVERY = 4

SYSTEM_MESSAGE = (
    "You are a SQL assistant. Given a natural-language question and a "
    "database schema injected through cross-attention, emit a single "
    "valid SQLite query that answers the question. Output only SQL, no "
    "explanation."
)


def render_prompt(tok, question: str) -> str:
    """Format a question for the backbone using its chat template."""
    msgs = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": question},
    ]
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)


class CrossAttnAdapter(nn.Module):
    """Flamingo-style gated cross-attention + FFN, spliced into a backbone
    layer via a forward hook. Tanh gates are initialised to zero so the
    adapter starts as an identity and the pretrained backbone is not
    disturbed at step 0."""

    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.mlp_gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden: torch.Tensor, kv: torch.Tensor,
                kv_mask: torch.Tensor | None = None) -> torch.Tensor:
        in_dtype = hidden.dtype
        p_dtype = self.norm_q.weight.dtype
        h = hidden.to(p_dtype)
        k = kv.to(p_dtype)
        q = self.norm_q(h)
        key_padding_mask = None
        if kv_mask is not None:
            key_padding_mask = ~kv_mask.bool()
        a, _ = self.attn(q, k, k, need_weights=False,
                         key_padding_mask=key_padding_mask)
        h = h + torch.tanh(self.attn_gate) * a
        m = self.norm_mlp(h)
        out = h + torch.tanh(self.mlp_gate) * self.mlp(m)
        return out.to(in_dtype)


class GroundedSQL(nn.Module):
    """Frozen causal-LM backbone + Flamingo gated cross-attn adapters over
    database-state K/V. Forward returns logits over the backbone's vocab
    so SQL can be generated token-by-token."""

    def __init__(self, backbone, hidden: int,
                 inject_every: int = INJECT_EVERY,
                 freeze_backbone: bool = True,
                 adapter_mlp_ratio: int = 4,
                 n_heads: int = 8):
        super().__init__()
        self.backbone = backbone
        self.hidden = hidden
        self.inject_every = inject_every
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        layers = self._backbone_layers()
        n = len(layers)
        self.inject_indices = list(range(inject_every - 1, n, inject_every))
        self.adapters = nn.ModuleList([
            CrossAttnAdapter(hidden, n_heads=n_heads,
                             mlp_ratio=adapter_mlp_ratio)
            for _ in self.inject_indices
        ])
        self.kv_norm = nn.LayerNorm(hidden)

        self._current_kv: torch.Tensor | None = None
        self._current_kv_mask: torch.Tensor | None = None
        self._hook_handles: list = []
        self._register_hooks()

    def _backbone_layers(self):
        bb = self.backbone
        if hasattr(bb, 'layers'):
            return bb.layers
        if hasattr(bb, 'model') and hasattr(bb.model, 'layers'):
            return bb.model.layers
        if hasattr(bb, 'model') and hasattr(bb.model, 'model') \
                and hasattr(bb.model.model, 'layers'):
            return bb.model.model.layers
        raise RuntimeError('Cannot locate backbone decoder layers')

    adapter_checkpointing: bool = False

    def _register_hooks(self):
        layers = self._backbone_layers()
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        owner = self
        for adapter, idx in zip(self.adapters, self.inject_indices):
            def make_hook(adapter_ref):
                def hook(module, inputs, output):
                    kv = owner._current_kv
                    if kv is None:
                        return output
                    mask = owner._current_kv_mask

                    def _apply(h):
                        if owner.adapter_checkpointing and owner.training:
                            return torch.utils.checkpoint.checkpoint(
                                adapter_ref, h, kv, mask, use_reentrant=False)
                        return adapter_ref(h, kv, mask)

                    if isinstance(output, tuple):
                        return (_apply(output[0]),) + output[1:]
                    return _apply(output)
                return hook
            self._hook_handles.append(
                layers[idx].register_forward_hook(make_hook(adapter)))

    def set_state_kv(self, kv: torch.Tensor | None,
                     kv_mask: torch.Tensor | None = None) -> None:
        """Set the database-state K/V used by every adapter on the next
        forward pass. Pass ``kv=None`` to disable cross-attention."""
        if kv is None:
            self._current_kv = None
        else:
            # kv comes from the backbone's input embedding (often bf16)
            # but kv_norm's params are fp32 by default; cast explicitly
            # so eval-time generate (no autocast) works too.
            kv = kv.to(self.kv_norm.weight.dtype)
            self._current_kv = self.kv_norm(kv)
        self._current_kv_mask = kv_mask

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                kv: torch.Tensor | None = None,
                kv_mask: torch.Tensor | None = None,
                labels: torch.Tensor | None = None,
                use_cache: bool = False):
        self.set_state_kv(kv, kv_mask)
        # NOTE: we deliberately do NOT clear _current_kv after the forward
        # pass. Gradient checkpointing on the backbone recomputes layer
        # forwards during backward, and our hooks must still see the same
        # kv they saw on the original forward. Callers overwrite via
        # set_state_kv on the next forward; ``generate()`` wraps its own
        # set/clear around ``backbone.generate``.
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
            return_dict=True,
        )


def build_backbone(backbone_id: str = BACKBONE_ID,
                   dtype: torch.dtype = torch.bfloat16):
    """Load a HuggingFace causal-LM backbone. Returns (model, tokenizer,
    hidden_size). Unlike the RISC-V version, we load ``AutoModelForCausalLM``
    because SQL is generated as text via the backbone's LM head."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(backbone_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bb = AutoModelForCausalLM.from_pretrained(backbone_id, torch_dtype=dtype)
    hidden = bb.config.hidden_size
    return bb, tok, hidden
