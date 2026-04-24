"""
Reflex — grounded RV32I control head (Flamingo-style fusion, JEPA head).

The backbone processes the instruction tokens; every ``INJECT_EVERY``
transformer layers a Flamingo-style cross-attention adapter fuses live
machine-state K/V (32 regs + pc + 16 words around pc + 16 words around
sp = 65 tokens) into the hidden stream.

Output head is JEPA-style: the pooled hidden state is projected to a
continuous embedding and compared (cosine similarity) against a jointly
trained table of instruction embeddings — one slot per unique 32-bit
word observed in training. At inference the predicted vector snaps to
the nearest table entry, which acts as error correction: small
embedding noise can't flip a bit, it just picks a close-neighbour
instruction.
"""
import numpy as np
import torch
import torch.nn as nn

from .riscv import DATA_BASE, PROGRAM_START, Rv32i

# ── Config ────────────────────────────────────────────────────────────
BACKBONE_ID = 'Qwen/Qwen2.5-Coder-7B-Instruct'
MAX_INSTR_TOKENS = 96
INJECT_EVERY = 4                # cross-attn every N backbone layers → 7 adapters on 28-layer Coder-7B

# Machine-context system message, passed through the backbone's chat
# template as `<|im_start|>system ... <|im_end|><|im_start|>user ...
# <|im_end|><|im_start|>assistant`. Using the template (not a raw text
# prefix) keeps us on the exact format the instruct-tuned backbone saw
# during post-training.
SYSTEM_MESSAGE = (
    "You control a RV32I CPU. "
    "Registers x5-x15 available. "
    "Data: 0x5000. "
    "Display: 0x6000 (ASCII, one word per char)."
)

# Legacy plain-text machine-context prefix — retained so checkpoints
# trained with context_prefix=True (pre-chat-template runs, e.g.
# reflex_3b_ctx.pt) still render the same way at inference.
CONTEXT_PREFIX = (
    "You control a RV32I CPU. "
    "Registers x5-x15 available. "
    "Data: 0x5000. Display: 0x6000 (ASCII, one word per char). "
    "Task: "
)


def render_prompt(tok, task: str, use_chat_template: bool = True,
                  use_context_prefix: bool = False) -> str:
    """Format `task` for the backbone's tokenizer.

    Default (current runs): chat template with SYSTEM_MESSAGE. Legacy
    checkpoints trained with a plain-text prefix pass
    ``use_chat_template=False, use_context_prefix=True``.
    """
    if use_chat_template:
        msgs = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": task},
        ]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    if use_context_prefix:
        return CONTEXT_PREFIX + task
    return task


# JEPA-style output: the model predicts a continuous embedding vector;
# nearest-neighbour over a jointly trained instruction codebook recovers
# the 32-bit word. Small prediction noise in embedding space can't flip
# a bit — it snaps to the closest real instruction, giving free error
# correction that is the reason popcount-255 (199 consecutive opcodes)
# completes cleanly at only ~97% per-step top-1.
EMBED_DIM = 256

# State token layout (65 tokens total)
N_REGS = 32                     # tokens 0..31 = x0..x31
IDX_PC = 32                     # token 32     = pc
IDX_MEM_PC = 33                 # tokens 33..48 = 16 words around pc
IDX_MEM_SP = 49                 # tokens 49..64 = 16 words around sp
N_STATE_TOKENS = 65
MEM_WINDOW_WORDS = 16


# ── Helpers ───────────────────────────────────────────────────────────
def _safe_read_words(cpu: Rv32i, center: int,
                     n_words: int = MEM_WINDOW_WORDS) -> list[int]:
    """Read n_words centered on `center`, zero-filling unmapped bytes."""
    half = (n_words // 2) * 4
    base = (center - half) & ~3
    out = [0] * n_words
    try:
        data = bytes(cpu.uc.mem_read(base, n_words * 4))
        for i in range(n_words):
            out[i] = int.from_bytes(data[i*4:(i+1)*4], 'little')
    except Exception:
        for i in range(n_words):
            try:
                data = bytes(cpu.uc.mem_read(base + i*4, 4))
                out[i] = int.from_bytes(data, 'little')
            except Exception:
                out[i] = 0
    return out


def extract_state(cpu: Rv32i) -> np.ndarray:
    """Returns a [65] uint32 state vector for the given Rv32i instance."""
    regs = [cpu.reg(i) for i in range(N_REGS)]
    pc = cpu.pc
    sp = cpu.reg(2)                          # x2 = sp by ABI
    mem_pc = _safe_read_words(cpu, pc)
    mem_sp = _safe_read_words(cpu, sp)
    return np.array(regs + [pc] + mem_pc + mem_sp, dtype=np.uint32)


# ── Modules ───────────────────────────────────────────────────────────
class StateEncoder(nn.Module):
    """[B, 65] uint32 state → [B, 65, hidden] tokens.

    Each token = role_embedding + value_projection. Value is encoded by
    splitting the 32-bit word into 4 bytes; each byte has its own
    256-vocab embedding, and the four are concatenated + projected.
    """
    def __init__(self, hidden: int, n_tokens: int = N_STATE_TOKENS,
                 byte_dim: int = 32):
        super().__init__()
        self.n_tokens = n_tokens
        self.role_embed = nn.Embedding(n_tokens, hidden)
        self.byte_embeds = nn.ModuleList(
            [nn.Embedding(256, byte_dim) for _ in range(4)])
        self.value_proj = nn.Linear(4 * byte_dim, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, state_vals: torch.Tensor) -> torch.Tensor:
        B, N = state_vals.shape
        v = state_vals.to(torch.long)
        byte_embs = [self.byte_embeds[i](((v >> (8 * i)) & 0xFF))
                     for i in range(4)]
        val_cat = torch.cat(byte_embs, dim=-1)           # [B, N, 4*byte_dim]
        val = self.value_proj(val_cat)                   # [B, N, hidden]
        roles = torch.arange(N, device=state_vals.device)[None].expand(B, N)
        return self.norm(val + self.role_embed(roles))


class TargetAwareHead(nn.Module):
    """Path B (HEAD_STUDY.md) — target-conditioned head.

    The canonical JEPA head compresses the backbone's last-token pooled
    hidden state into a 256-d codebook-neighbour prediction. That pooled
    vector carries the prompt's target-intent only *implicitly* — as
    the backbone's own summary. On low-margin cycles where the top-k
    codebook neighbours are near-ties, implicit target-intent is not
    enough to pick between semantically-different-but-valid candidates
    (e.g., ``addi x15, 'K'`` vs ``addi x15, 'C'`` when the prompt
    target is ``OK``).

    This head adds an *explicit* second pathway: a cross-attention that
    re-reads the backbone's own per-token hidden states, keyed on the
    pooled state. The cross-attended vector is added to the pooled
    feature through a tanh gate initialised to zero, so a checkpoint
    constructed with this head but never retrained produces output
    bit-identical to the canonical head. Retraining the head + the
    cross-attention + the codebook (backbone + adapters stay frozen)
    lets the head route prompt-target signal into the prediction
    exactly where the pooled state alone is insufficient.
    """
    def __init__(self, hidden: int, embed_dim: int, n_heads: int = 8):
        super().__init__()
        self.hidden = hidden
        self.embed_dim = embed_dim
        self.prompt_norm = nn.LayerNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.xattn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.out_proj = nn.Linear(hidden, hidden)
        # tanh-initialised-to-zero gate: at step 0 the cross-attn
        # contribution is exactly zero, so a non-retrained ckpt passes
        # through the canonical head path.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, pooled: torch.Tensor, prompt_hidden: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns a refined pooled vector [B, hidden].

        ``pooled`` is the backbone's last-token pooled hidden state
        ([B, H]). ``prompt_hidden`` is the backbone's full per-token
        hidden-state output ([B, T, H]). ``attention_mask`` is the
        standard HF mask where 1 is a real token, 0 is padding.
        """
        kv = self.prompt_norm(prompt_hidden)                 # [B, T, H]
        q = self.q_proj(pooled).unsqueeze(1)                 # [B, 1, H]
        # key_padding_mask=True on positions to IGNORE.
        kpm = (attention_mask == 0)
        att, _ = self.xattn(q, kv, kv, key_padding_mask=kpm,
                            need_weights=False)
        delta = self.out_proj(att.squeeze(1))                # [B, H]
        return pooled + torch.tanh(self.gate) * delta


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

    def forward(self, hidden: torch.Tensor,
                kv: torch.Tensor) -> torch.Tensor:
        in_dtype = hidden.dtype
        p_dtype = self.norm_q.weight.dtype
        h = hidden.to(p_dtype)
        k = kv.to(p_dtype)
        q = self.norm_q(h)
        a, _ = self.attn(q, k, k, need_weights=False)
        h = h + torch.tanh(self.attn_gate) * a
        m = self.norm_mlp(h)
        out = h + torch.tanh(self.mlp_gate) * self.mlp(m)
        return out.to(in_dtype)


# ── Model ─────────────────────────────────────────────────────────────
class GroundedReflex(nn.Module):
    """End-to-end grounded RV32I controller.

    Forward flow for one cycle:
        1. State encoder turns live [B, 65] state into K/V [B, 65, H].
        2. Backbone runs over instruction tokens; at every
           ``INJECT_EVERY`` layers, a hook applies a
           ``CrossAttnAdapter`` that fuses K/V into the hidden states.
        3. Masked mean-pool over instruction tokens → [B, H].
        4. Small MLP + six linear heads → RV32I field logits.
    """
    def __init__(self, backbone, hidden: int,
                 num_instrs: int,
                 inject_every: int = INJECT_EVERY,
                 freeze_backbone: bool = False,
                 adapter_mlp_ratio: int = 4,
                 embed_dim: int = EMBED_DIM,
                 use_target_head: bool = False):
        super().__init__()
        self.backbone = backbone
        self.hidden = hidden
        self.inject_every = inject_every
        self.freeze_backbone = freeze_backbone
        self.embed_dim = embed_dim
        self.num_instrs = num_instrs
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.state_encoder = StateEncoder(hidden)
        self.kv_norm = nn.LayerNorm(hidden)

        layers = self._backbone_layers()
        n = len(layers)
        self.inject_indices = list(range(inject_every - 1, n, inject_every))
        self.adapters = nn.ModuleList(
            [CrossAttnAdapter(hidden, mlp_ratio=adapter_mlp_ratio)
             for _ in self.inject_indices])

        self.head_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.embed_head = nn.Linear(hidden, embed_dim)

        # Path B: target-conditioned head. Constructed only when
        # requested at build time. Gate is zero-initialised so a
        # freshly-added head on an un-retrained checkpoint has no
        # effect — cosine output remains bit-identical until the head
        # is trained.
        self.use_target_head = bool(use_target_head)
        self.target_head = (
            TargetAwareHead(hidden, embed_dim) if self.use_target_head else None
        )

        # JEPA instruction codebook: one trainable embedding per unique
        # 32-bit word observed in training. ``instr_words`` is the
        # companion int64 buffer so the decoded word travels with the
        # checkpoint and lives on-device for nearest-neighbour lookup.
        self.instr_table = nn.Embedding(num_instrs, embed_dim)
        nn.init.normal_(self.instr_table.weight, std=0.02)
        self.register_buffer('instr_words',
                             torch.zeros(num_instrs, dtype=torch.int64))

        self._current_kv: torch.Tensor | None = None
        self._hook_handles: list = []
        self._register_hooks()

    def _backbone_layers(self):
        bb = self.backbone
        if hasattr(bb, 'layers'):
            return bb.layers
        if hasattr(bb, 'model') and hasattr(bb.model, 'layers'):
            return bb.model.layers
        raise RuntimeError('Cannot locate backbone decoder layers')

    adapter_checkpointing: bool = False      # toggled externally

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
                    def _apply(h):
                        if owner.adapter_checkpointing and owner.training:
                            return torch.utils.checkpoint.checkpoint(
                                adapter_ref, h, kv, use_reentrant=False)
                        return adapter_ref(h, kv)
                    if isinstance(output, tuple):
                        return (_apply(output[0]),) + output[1:]
                    return _apply(output)
                return hook
            self._hook_handles.append(
                layers[idx].register_forward_hook(make_hook(adapter)))

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                state_vals: torch.Tensor) -> torch.Tensor:
        """Returns [B, embed_dim] predicted instruction embedding."""
        kv = self.kv_norm(self.state_encoder(state_vals))    # [B, 65, H]
        self._current_kv = kv
        try:
            out = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=False,
                                return_dict=True, use_cache=False)
        finally:
            self._current_kv = None
        head_dtype = next(self.head_mlp.parameters()).dtype
        h = out.last_hidden_state.to(head_dtype)
        last_idx = attention_mask.sum(dim=1).long() - 1      # [B]
        pooled = h[torch.arange(h.size(0), device=h.device), last_idx]
        if self.target_head is not None:
            # Path B: refine pooled with prompt cross-attention before
            # the head_mlp. Gate-zero-init means this is identity on
            # un-retrained checkpoints.
            pooled = self.target_head(pooled, h, attention_mask)
        feat = self.head_mlp(pooled)
        return self.embed_head(feat)                         # [B, embed_dim]

    def table_similarity(self, pred: torch.Tensor) -> torch.Tensor:
        """Cosine similarity of [B, embed_dim] pred against every row of
        the instruction codebook. Returns [B, num_instrs]."""
        p = torch.nn.functional.normalize(pred.float(), dim=-1)
        t = torch.nn.functional.normalize(self.instr_table.weight.float(),
                                          dim=-1)
        return p @ t.T

    @torch.no_grad()
    def decode_words(self, pred: torch.Tensor) -> torch.Tensor:
        """Nearest-neighbour decode [B, embed_dim] → [B] int64 instruction
        words (values from ``instr_words``)."""
        idx = self.table_similarity(pred).argmax(dim=-1)
        return self.instr_words[idx]


def build_backbone(backbone_id: str = BACKBONE_ID,
                   dtype: torch.dtype = torch.bfloat16):
    """Load a HuggingFace causal-LM backbone (fully unfrozen)."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(backbone_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bb = AutoModel.from_pretrained(backbone_id, torch_dtype=dtype)
    hidden = bb.config.hidden_size
    return bb, tok, hidden


def code_region_halt_fill() -> bytes:
    """Bytes to pre-fill the code region with HALT so Unicorn's pre-decode
    never trips on a zero instruction and forward branches into un-emitted
    space halt cleanly. Call ``cpu.uc.mem_write(PROGRAM_START, ...)`` with
    the return value after constructing a fresh Rv32i."""
    from .riscv import HALT_INSTR
    code_size = DATA_BASE - PROGRAM_START
    return HALT_INSTR.to_bytes(4, 'little') * (code_size // 4)
