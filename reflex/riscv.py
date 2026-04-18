"""
RV32I emulator (unicorn-engine backed). Mirrors the Chip8 interface:
reset, load_program, step, halt detection via self-jump (`jal x0, 0`).

The model's "machine": emits 32-bit RV32I instructions, loaded here and
executed instruction-by-instruction.
"""

from unicorn import Uc, UC_ARCH_RISCV, UC_MODE_RISCV32
from unicorn.riscv_const import UC_RISCV_REG_X0, UC_RISCV_REG_PC

MEM_START = 0x1000
MEM_SIZE = 0x10000           # 64KB
PROGRAM_START = MEM_START    # programs load here
DATA_BASE = MEM_START + 0x4000   # data region sits above the program

HALT_INSTR = 0x0000006F      # jal x0, 0 — infinite self-loop


def reg_const(i: int) -> int:
    """unicorn's X0..X31 constants are contiguous."""
    return UC_RISCV_REG_X0 + i


# ── Instruction encoders ─────────────────────────────────────────────
# Each helper returns a 32-bit int. Use `pack(*ops)` to lay them out
# little-endian into the program bytestring.

def _mask(val: int, bits: int) -> int:
    return val & ((1 << bits) - 1)


def r_type(opcode: int, rd: int, funct3: int, rs1: int, rs2: int, funct7: int) -> int:
    return (_mask(funct7, 7) << 25) | (_mask(rs2, 5) << 20) | (_mask(rs1, 5) << 15) \
           | (_mask(funct3, 3) << 12) | (_mask(rd, 5) << 7) | _mask(opcode, 7)


def i_type(opcode: int, rd: int, funct3: int, rs1: int, imm: int) -> int:
    return (_mask(imm, 12) << 20) | (_mask(rs1, 5) << 15) \
           | (_mask(funct3, 3) << 12) | (_mask(rd, 5) << 7) | _mask(opcode, 7)


def s_type(opcode: int, funct3: int, rs1: int, rs2: int, imm: int) -> int:
    imm = _mask(imm, 12)
    imm_hi = (imm >> 5) & 0x7F
    imm_lo = imm & 0x1F
    return (imm_hi << 25) | (_mask(rs2, 5) << 20) | (_mask(rs1, 5) << 15) \
           | (_mask(funct3, 3) << 12) | (imm_lo << 7) | _mask(opcode, 7)


def b_type(opcode: int, funct3: int, rs1: int, rs2: int, imm: int) -> int:
    """Branch offset in bytes; imm must be even (bit 0 dropped). Range: [-4096, 4094]."""
    imm = _mask(imm, 13)                      # 13-bit signed, even
    b12 = (imm >> 12) & 0x1
    b11 = (imm >> 11) & 0x1
    b10_5 = (imm >> 5) & 0x3F
    b4_1 = (imm >> 1) & 0xF
    imm_hi = (b12 << 6) | b10_5               # 7 bits
    imm_lo = (b4_1 << 1) | b11                # 5 bits
    return (imm_hi << 25) | (_mask(rs2, 5) << 20) | (_mask(rs1, 5) << 15) \
           | (_mask(funct3, 3) << 12) | (imm_lo << 7) | _mask(opcode, 7)


def u_type(opcode: int, rd: int, imm: int) -> int:
    """imm is the full 32-bit value; only bits [31:12] are used."""
    return (imm & 0xFFFFF000) | (_mask(rd, 5) << 7) | _mask(opcode, 7)


def j_type(opcode: int, rd: int, imm: int) -> int:
    """Jump offset in bytes; imm must be even. Range: [-1048576, 1048574]."""
    imm = _mask(imm, 21)
    b20 = (imm >> 20) & 0x1
    b19_12 = (imm >> 12) & 0xFF
    b11 = (imm >> 11) & 0x1
    b10_1 = (imm >> 1) & 0x3FF
    imm_field = (b20 << 19) | (b10_1 << 9) | (b11 << 8) | b19_12   # 20 bits
    return (imm_field << 12) | (_mask(rd, 5) << 7) | _mask(opcode, 7)


# Mnemonic-level helpers for template authors.

def addi(rd, rs1, imm): return i_type(0x13, rd, 0b000, rs1, imm)
def slti(rd, rs1, imm): return i_type(0x13, rd, 0b010, rs1, imm)
def andi(rd, rs1, imm): return i_type(0x13, rd, 0b111, rs1, imm)
def ori(rd, rs1, imm):  return i_type(0x13, rd, 0b110, rs1, imm)
def xori(rd, rs1, imm): return i_type(0x13, rd, 0b100, rs1, imm)
def slli(rd, rs1, shamt): return i_type(0x13, rd, 0b001, rs1, shamt & 0x1F)
def srli(rd, rs1, shamt): return i_type(0x13, rd, 0b101, rs1, shamt & 0x1F)

def add(rd, rs1, rs2): return r_type(0x33, rd, 0b000, rs1, rs2, 0b0000000)
def sub(rd, rs1, rs2): return r_type(0x33, rd, 0b000, rs1, rs2, 0b0100000)
def sll(rd, rs1, rs2): return r_type(0x33, rd, 0b001, rs1, rs2, 0b0000000)
def slt(rd, rs1, rs2): return r_type(0x33, rd, 0b010, rs1, rs2, 0b0000000)
def xor_(rd, rs1, rs2): return r_type(0x33, rd, 0b100, rs1, rs2, 0b0000000)
def or_(rd, rs1, rs2):  return r_type(0x33, rd, 0b110, rs1, rs2, 0b0000000)
def and_(rd, rs1, rs2): return r_type(0x33, rd, 0b111, rs1, rs2, 0b0000000)

def lui(rd, imm20): return u_type(0x37, rd, imm20 << 12)
def auipc(rd, imm20): return u_type(0x17, rd, imm20 << 12)

def lw(rd, rs1, imm):  return i_type(0x03, rd, 0b010, rs1, imm)
def lb(rd, rs1, imm):  return i_type(0x03, rd, 0b000, rs1, imm)
def lbu(rd, rs1, imm): return i_type(0x03, rd, 0b100, rs1, imm)
def sw(rs2, rs1, imm): return s_type(0x23, 0b010, rs1, rs2, imm)
def sb(rs2, rs1, imm): return s_type(0x23, 0b000, rs1, rs2, imm)

def beq(rs1, rs2, imm): return b_type(0x63, 0b000, rs1, rs2, imm)
def bne(rs1, rs2, imm): return b_type(0x63, 0b001, rs1, rs2, imm)
def blt(rs1, rs2, imm): return b_type(0x63, 0b100, rs1, rs2, imm)
def bge(rs1, rs2, imm): return b_type(0x63, 0b101, rs1, rs2, imm)

def jal(rd, imm):       return j_type(0x6F, rd, imm)
def jalr(rd, rs1, imm): return i_type(0x67, rd, 0b000, rs1, imm)

def halt() -> int:
    """jal x0, 0 — infinite self-loop, our halt convention."""
    return HALT_INSTR


def pack(*instrs: int) -> bytes:
    """Pack 32-bit instructions into a little-endian bytestring."""
    out = bytearray()
    for i in instrs:
        out += int(i & 0xFFFFFFFF).to_bytes(4, "little")
    return bytes(out)


# ── Field decomposition ──────────────────────────────────────────────
# The six classification heads the model predicts.

def decompose(instr: int) -> tuple[int, int, int, int, int, int]:
    """Split a 32-bit instruction into (opcode, rd, funct3, rs1, rs2, funct7)."""
    return (
        instr & 0x7F,           # opcode [6:0]
        (instr >> 7) & 0x1F,    # rd      [11:7]
        (instr >> 12) & 0x7,    # funct3  [14:12]
        (instr >> 15) & 0x1F,   # rs1     [19:15]
        (instr >> 20) & 0x1F,   # rs2     [24:20]
        (instr >> 25) & 0x7F,   # funct7  [31:25]
    )


def compose(opcode: int, rd: int, funct3: int, rs1: int, rs2: int, funct7: int) -> int:
    return (_mask(funct7, 7) << 25) | (_mask(rs2, 5) << 20) | (_mask(rs1, 5) << 15) \
           | (_mask(funct3, 3) << 12) | (_mask(rd, 5) << 7) | _mask(opcode, 7)


# ── Emulator wrapper ─────────────────────────────────────────────────

class Rv32i:
    def __init__(self):
        self._build()

    def _build(self):
        self.uc = Uc(UC_ARCH_RISCV, UC_MODE_RISCV32)
        self.uc.mem_map(MEM_START, MEM_SIZE)
        self.uc.mem_write(MEM_START, bytes(MEM_SIZE))
        self.uc.reg_write(UC_RISCV_REG_PC, PROGRAM_START)

    def reset(self):
        # Fresh Uc each time — cheaper than trying to zero every register.
        self._build()

    def load_program(self, data: bytes):
        self.reset()
        if len(data) > DATA_BASE - PROGRAM_START:
            raise ValueError(f"program too long ({len(data)} bytes)")
        self.uc.mem_write(PROGRAM_START, data)

    @property
    def pc(self) -> int:
        return self.uc.reg_read(UC_RISCV_REG_PC)

    def reg(self, i: int) -> int:
        """Read x0..x31 as unsigned 32-bit."""
        return self.uc.reg_read(reg_const(i)) & 0xFFFFFFFF

    def reg_s(self, i: int) -> int:
        """Read x0..x31 as signed 32-bit."""
        v = self.reg(i)
        return v - (1 << 32) if v & 0x80000000 else v

    def mem_read(self, addr: int, n: int) -> bytes:
        return bytes(self.uc.mem_read(addr, n))

    def mem_word(self, addr: int) -> int:
        return int.from_bytes(self.mem_read(addr, 4), "little")

    def fetch(self) -> int:
        return self.mem_word(self.pc)

    def step(self) -> None:
        """Execute one instruction from the current pc."""
        pc = self.pc
        # count=1 stops after one instruction regardless of the `until` arg.
        # `until` is still required as a safety bound.
        self.uc.emu_start(pc, pc + 0x10000, count=1)
