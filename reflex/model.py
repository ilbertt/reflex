"""
Reflex — grounded RV32I control head.

A small LLM (Qwen2.5-0.5B) encodes the human instruction once per task;
the cached hidden states become the cross-attention queries. At every
emission step the queries are re-attended against fresh K/V built from
the live Rv32i state (32 registers + PC + a 16-word window around PC +
a 16-word window around SP). A GRU carries autoregressive context
across steps; six classification heads predict the RV32I field
decomposition (opcode / rd / funct3 / rs1 / rs2 / funct7).

The backbone can be frozen or fine-tuned with LoRA. State encoding,
cross-attention decoder, and output heads are always fully trained.
"""
import numpy as np
import torch
import torch.nn as nn

from .riscv import DATA_BASE, PROGRAM_START, Rv32i

# ── Config ────────────────────────────────────────────────────────────
BACKBONE_ID = 'Qwen/Qwen2.5-0.5B'
MAX_INSTR_TOKENS = 32
N_XATTN_LAYERS = 4
CTRL_DIM = 384

# RV32I field heads
N_OPCODE, N_RD, N_FUNCT3, N_RS1, N_RS2, N_FUNCT7 = 128, 32, 8, 32, 32, 128
FIELD_NAMES = ('opcode', 'rd', 'funct3', 'rs1', 'rs2', 'funct7')
FIELD_CLASSES = (N_OPCODE, N_RD, N_FUNCT3, N_RS1, N_RS2, N_FUNCT7)
FIELD_EMBED_DIMS = (32, 16, 16, 16, 16, 32)
PREV_OP_DIM = sum(FIELD_EMBED_DIMS)

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


# ── Model ─────────────────────────────────────────────────────────────
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


class XAttnBlock(nn.Module):
    """Pre-norm cross-attention + feed-forward residual block."""
    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        qh = self.norm_q(q)
        kh = self.norm_kv(kv)
        a, _ = self.attn(qh, kh, kh, need_weights=False)
        q = q + a
        return q + self.mlp(self.norm_mlp(q))


class GroundedReflex(nn.Module):
    """Grounded RV32I control head.

    Forward flow at a single emission step:
        1. Cached instruction hidden [B, T, H] is used as Q.
        2. State encoder turns live [B, 65] state into K/V [B, 65, H].
        3. Stack of cross-attention blocks refines Q using K/V.
        4. Mean-pool Q (masked by instruction attention_mask) → ctx.
        5. GRU(ctx ⊕ previous-instruction embed) → h_state.
        6. Six classification heads → field logits.
    """
    def __init__(self, backbone, hidden: int, ctrl_dim: int = CTRL_DIM,
                 n_xattn: int = N_XATTN_LAYERS,
                 freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        self.hidden = hidden
        self.ctrl_dim = ctrl_dim
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.state_encoder = StateEncoder(hidden)
        self.xattn_blocks = nn.ModuleList(
            [XAttnBlock(hidden) for _ in range(n_xattn)])

        self.pool_proj = nn.Linear(hidden, ctrl_dim)
        self.prev_embeds = nn.ModuleList([
            nn.Embedding(c, d) for c, d in zip(FIELD_CLASSES, FIELD_EMBED_DIMS)
        ])
        self.gru = nn.GRUCell(ctrl_dim + PREV_OP_DIM, ctrl_dim)
        self.mlp = nn.Sequential(
            nn.Linear(ctrl_dim, ctrl_dim * 2), nn.GELU(),
            nn.Linear(ctrl_dim * 2, ctrl_dim),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(ctrl_dim, c) for c in FIELD_CLASSES])

    def encode_instruction(self, input_ids, attention_mask) -> torch.Tensor:
        """Once per task. Returns cached hidden [B, T, H] in fp32."""
        ctx = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with ctx:
            out = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=False, return_dict=True,
                                use_cache=False)
        return out.last_hidden_state.to(torch.float32)

    def decode_step(self, instr_hidden, instr_mask, state_vals,
                    prev_fields, h_state):
        kv = self.state_encoder(state_vals)              # [B, 65, H]
        q = instr_hidden
        for blk in self.xattn_blocks:
            q = blk(q, kv)
        m = instr_mask.unsqueeze(-1).float()
        pooled = (q * m).sum(1) / m.sum(1).clamp_min(1.0)
        ctx = self.pool_proj(pooled)

        prev_parts = [emb(f) for emb, f in zip(self.prev_embeds, prev_fields)]
        prev_emb = torch.cat(prev_parts, dim=-1)
        h_state = self.gru(torch.cat([ctx, prev_emb], dim=-1), h_state)
        out = h_state + self.mlp(h_state)
        logits = [head(out) for head in self.heads]
        return logits, h_state


def build_backbone(backbone_id: str = BACKBONE_ID,
                   use_lora: bool = False,
                   dtype: torch.dtype = torch.float16):
    """Load a HuggingFace causal-LM backbone, optionally wrapped with LoRA."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(backbone_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bb = AutoModel.from_pretrained(backbone_id, torch_dtype=dtype)
    if use_lora:
        from peft import LoraConfig, get_peft_model
        bb = get_peft_model(bb, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias='none',
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']))
    hidden = bb.config.hidden_size if not use_lora else bb.base_model.model.config.hidden_size
    return bb, tok, hidden


def code_region_halt_fill() -> bytes:
    """Bytes to pre-fill the code region with HALT so Unicorn's pre-decode
    never trips on a zero instruction and forward branches into un-emitted
    space halt cleanly. Call ``cpu.uc.mem_write(PROGRAM_START, ...)`` with
    the return value after constructing a fresh Rv32i."""
    from .riscv import HALT_INSTR
    code_size = DATA_BASE - PROGRAM_START
    return HALT_INSTR.to_bytes(4, 'little') * (code_size // 4)
