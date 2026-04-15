"""
CHIP-8 emulator. 4KB memory, 64x32 display, 16 registers, 35 opcodes.

The model's "machine." It reads raw memory and display bytes,
and emits 2-byte opcodes to control it.
"""

import numpy as np

MEMORY_SIZE = 4096
DISPLAY_W = 64
DISPLAY_H = 32
DISPLAY_SIZE = DISPLAY_W * DISPLAY_H  # 2048 pixels (1-bit each)
PROGRAM_START = 0x200

# Built-in font sprites (0-F), loaded at address 0x000
FONT = [
    0xF0, 0x90, 0x90, 0x90, 0xF0,  # 0
    0x20, 0x60, 0x20, 0x20, 0x70,  # 1
    0xF0, 0x10, 0xF0, 0x80, 0xF0,  # 2
    0xF0, 0x10, 0xF0, 0x10, 0xF0,  # 3
    0x90, 0x90, 0xF0, 0x10, 0x10,  # 4
    0xF0, 0x80, 0xF0, 0x10, 0xF0,  # 5
    0xF0, 0x80, 0xF0, 0x90, 0xF0,  # 6
    0xF0, 0x10, 0x20, 0x40, 0x40,  # 7
    0xF0, 0x90, 0xF0, 0x90, 0xF0,  # 8
    0xF0, 0x90, 0xF0, 0x10, 0xF0,  # 9
    0xF0, 0x90, 0xF0, 0x90, 0x90,  # A
    0xE0, 0x90, 0xE0, 0x90, 0xE0,  # B
    0xF0, 0x80, 0x80, 0x80, 0xF0,  # C
    0xE0, 0x90, 0x90, 0x90, 0xE0,  # D
    0xF0, 0x80, 0xF0, 0x80, 0xF0,  # E
    0xF0, 0x80, 0xF0, 0x80, 0x80,  # F
]


class Chip8:
    def __init__(self):
        self.reset()

    def reset(self):
        self.memory = np.zeros(MEMORY_SIZE, dtype=np.uint8)
        self.memory[:len(FONT)] = FONT
        self.display = np.zeros(DISPLAY_SIZE, dtype=np.uint8)
        self.V = np.zeros(16, dtype=np.uint8)  # registers V0-VF
        self.I = 0       # index register
        self.pc = PROGRAM_START
        self.sp = 0
        self.stack = np.zeros(16, dtype=np.uint16)
        self.prev_opcode = 0
        self.halted = False

    def load_program(self, data: bytes):
        self.reset()
        for i, b in enumerate(data):
            if PROGRAM_START + i < MEMORY_SIZE:
                self.memory[PROGRAM_START + i] = b

    def get_state(self) -> np.ndarray:
        """Machine state: display + registers + prev opcode. No memory window."""
        dim = DISPLAY_SIZE + 16 + 2 + 2  # 2068
        state = np.zeros(dim, dtype=np.float32)
        offset = 0

        state[offset:offset + DISPLAY_SIZE] = self.display.astype(np.float32)
        offset += DISPLAY_SIZE

        state[offset:offset + 16] = self.V / 255.0
        offset += 16

        state[offset] = (self.I & 0xFF) / 255.0
        state[offset + 1] = ((self.I >> 8) & 0xFF) / 255.0
        offset += 2

        state[offset] = ((self.prev_opcode >> 8) & 0xFF) / 255.0
        state[offset + 1] = (self.prev_opcode & 0xFF) / 255.0
        return state

    def get_display(self) -> np.ndarray:
        """Display as flat array of 0/1 pixels."""
        return self.display.astype(np.float32)

    def step(self, opcode: int) -> bool:
        """Execute one opcode. Returns True if display changed."""
        opcode = int(opcode)
        self.prev_opcode = opcode
        display_changed = False
        op = (opcode >> 12) & 0xF
        x = (opcode >> 8) & 0xF
        y = (opcode >> 4) & 0xF
        n = opcode & 0xF
        nn = opcode & 0xFF
        nnn = opcode & 0xFFF

        if opcode == 0x00E0:
            # Clear display
            self.display[:] = 0
            display_changed = True
        elif opcode == 0x00EE:
            # Return from subroutine
            self.sp -= 1
            self.pc = int(self.stack[self.sp])
        elif op == 0x1:
            # Jump to NNN
            self.pc = nnn
            return display_changed
        elif op == 0x2:
            # Call subroutine at NNN
            self.stack[self.sp] = self.pc
            self.sp += 1
            self.pc = nnn
            return display_changed
        elif op == 0x3:
            # Skip if Vx == NN
            if self.V[x] == nn:
                self.pc += 2
        elif op == 0x4:
            # Skip if Vx != NN
            if self.V[x] != nn:
                self.pc += 2
        elif op == 0x5:
            # Skip if Vx == Vy
            if self.V[x] == self.V[y]:
                self.pc += 2
        elif op == 0x6:
            # Set Vx = NN
            self.V[x] = nn
        elif op == 0x7:
            # Add NN to Vx
            self.V[x] = (self.V[x] + nn) & 0xFF
        elif op == 0x8:
            if n == 0x0:
                self.V[x] = self.V[y]
            elif n == 0x1:
                self.V[x] |= self.V[y]
            elif n == 0x2:
                self.V[x] &= self.V[y]
            elif n == 0x3:
                self.V[x] ^= self.V[y]
            elif n == 0x4:
                result = int(self.V[x]) + int(self.V[y])
                self.V[0xF] = 1 if result > 255 else 0
                self.V[x] = result & 0xFF
            elif n == 0x5:
                self.V[0xF] = 1 if self.V[x] >= self.V[y] else 0
                self.V[x] = (int(self.V[x]) - int(self.V[y])) & 0xFF
            elif n == 0x6:
                self.V[0xF] = self.V[x] & 1
                self.V[x] >>= 1
            elif n == 0x7:
                self.V[0xF] = 1 if self.V[y] >= self.V[x] else 0
                self.V[x] = (int(self.V[y]) - int(self.V[x])) & 0xFF
            elif n == 0xE:
                self.V[0xF] = (self.V[x] >> 7) & 1
                self.V[x] = (self.V[x] << 1) & 0xFF
        elif op == 0x9:
            # Skip if Vx != Vy
            if self.V[x] != self.V[y]:
                self.pc += 2
        elif op == 0xA:
            # Set I = NNN
            self.I = nnn
        elif op == 0xB:
            # Jump to NNN + V0
            self.pc = (nnn + self.V[0]) & 0xFFF
            return display_changed
        elif op == 0xC:
            # Vx = random & NN
            self.V[x] = np.random.randint(0, 256) & nn
        elif op == 0xD:
            # Draw sprite at (Vx, Vy), height N
            self.V[0xF] = 0
            for row in range(n):
                if self.I + row >= MEMORY_SIZE:
                    break
                sprite_byte = self.memory[self.I + row]
                for col in range(8):
                    if sprite_byte & (0x80 >> col):
                        px = (int(self.V[x]) + col) % DISPLAY_W
                        py = (int(self.V[y]) + row) % DISPLAY_H
                        idx = py * DISPLAY_W + px
                        if self.display[idx]:
                            self.V[0xF] = 1
                        self.display[idx] ^= 1
            display_changed = True
        elif op == 0xF:
            if nn == 0x07:
                self.V[x] = 0  # no delay timer in this minimal impl
            elif nn == 0x0A:
                self.V[x] = 0  # no key wait
            elif nn == 0x15:
                pass  # set delay timer (ignored)
            elif nn == 0x18:
                pass  # set sound timer (ignored)
            elif nn == 0x1E:
                self.I = (self.I + int(self.V[x])) & 0xFFF
            elif nn == 0x29:
                self.I = int(self.V[x]) * 5  # font sprite address
            elif nn == 0x33:
                val = int(self.V[x])
                self.memory[self.I] = val // 100
                self.memory[self.I + 1] = (val // 10) % 10
                self.memory[self.I + 2] = val % 10
            elif nn == 0x55:
                for i in range(x + 1):
                    self.memory[self.I + i] = self.V[i]
            elif nn == 0x65:
                for i in range(x + 1):
                    self.V[i] = self.memory[self.I + i]

        self.pc += 2
        return display_changed

