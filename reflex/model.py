"""
Reflex — grounded RV32I control head (Flamingo-style fusion).

The backbone is fully unfrozen and processes the instruction tokens.
Every ``INJECT_EVERY`` transformer layers, a cross-attention adapter is
spliced in via a forward hook: the backbone's current hidden states
(queries) attend over live machine-state K/V (32 regs + pc + 16-word
window around pc + 16-word window around sp = 65 tokens). By the time
control reaches the final layer, the instruction representation has
been fused with the machine state at multiple depths.

A small pooled-MLP + six classification heads then emit the RV32I field
decomposition (opcode / rd / funct3 / rs1 / rs2 / funct7).
"""
import numpy as np
import torch
import torch.nn as nn

from .riscv import DATA_BASE, PROGRAM_START, Rv32i

# ── Config ────────────────────────────────────────────────────────────
BACKBONE_ID = 'Qwen/Qwen2.5-0.5B'
MAX_INSTR_TOKENS = 32
INJECT_EVERY = 4                # cross-attn every N backbone layers

# Optional machine-context prefix. When enabled via CLI, this string is
# prepended to every instruction before tokenisation. Tests whether a
# textual prior about the target machine helps the frozen backbone
# route instructions into useful RV32I programs.
CONTEXT_PREFIX = (
    "You control a RV32I CPU. "
    "Registers x5-x15 available. "
    "Data: 0x5000. Display: 0x6000 (ASCII, one word per char). "
    "Task: "
)

# Bit-level output: 32 independent sigmoid heads, one per instruction bit.
# This sidesteps the RV32I field polysemy (e.g. rs2 = register on R-type
# but imm[4:0] on I-type) that bottlenecks a field-decomposition head.
N_INSTR_BITS = 32
FIELD_NAMES = ('opcode', 'rd', 'funct3', 'rs1', 'rs2', 'funct7')
FIELD_CLASSES = (128, 32, 8, 32, 32, 128)

# State token layout (65 tokens total)
N_REGS = 32                     # tokens 0..31 = x0..x31
IDX_PC = 32                     # token 32     = pc
IDX_MEM_PC = 33                 # tokens 33..48 = 16 words around pc
IDX_MEM_SP = 49                 # tokens 49..64 = 16 words around sp
N_STATE_TOKENS = 65
MEM_WINDOW_WORDS = 16


# ── Helpers ───────────────────────────────────────────────────────────
def split_fields(w: int) -> tuple[int, int, int, int, int, int]:
    """Split a 32-bit RV32I instruction into its six field groups."""
    return (w & 0x7F, (w >> 7) & 0x1F, (w >> 12) & 0x7,
            (w >> 15) & 0x1F, (w >> 20) & 0x1F, (w >> 25) & 0x7F)


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
                 inject_every: int = INJECT_EVERY,
                 freeze_backbone: bool = False):
        super().__init__()
        self.backbone = backbone
        self.hidden = hidden
        self.inject_every = inject_every
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.state_encoder = StateEncoder(hidden)
        self.kv_norm = nn.LayerNorm(hidden)

        layers = self._backbone_layers()
        n = len(layers)
        self.inject_indices = list(range(inject_every - 1, n, inject_every))
        self.adapters = nn.ModuleList(
            [CrossAttnAdapter(hidden) for _ in self.inject_indices])

        self.head_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.bit_head = nn.Linear(hidden, N_INSTR_BITS)

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
                    if isinstance(output, tuple):
                        h = adapter_ref(output[0], kv)
                        return (h,) + output[1:]
                    return adapter_ref(output, kv)
                return hook
            self._hook_handles.append(
                layers[idx].register_forward_hook(make_hook(adapter)))

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                state_vals: torch.Tensor) -> torch.Tensor:
        """Returns [B, 32] raw bit logits (sigmoid at inference, BCE at train)."""
        kv = self.kv_norm(self.state_encoder(state_vals))    # [B, 65, H]
        self._current_kv = kv
        try:
            out = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=False,
                                return_dict=True, use_cache=False)
        finally:
            self._current_kv = None
        h = out.last_hidden_state.to(torch.float32)
        # Last-token pool: in a causal LLM the final real token has
        # attended to every preceding token, so it encodes the full
        # prompt (prefix + task) without the dilution that mean-pool
        # suffers when a long shared prefix is present. We locate the
        # final non-pad token via the attention_mask (Qwen uses
        # right-padding by default).
        last_idx = attention_mask.sum(dim=1).long() - 1      # [B]
        pooled = h[torch.arange(h.size(0), device=h.device), last_idx]
        feat = self.head_mlp(pooled)
        return self.bit_head(feat)                           # [B, 32]


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
