"""
Reflex model: reads raw CHIP-8 machine state + goal display, emits opcodes.

The model is a neural CPU: it reads raw machine bytes (display, registers,
memory around PC, previous opcode) and a goal display, then emits the
next 2-byte opcode.
"""

import mlx.nn as nn

from chip8 import DISPLAY_SIZE

PC_WINDOW = 32
STATE_DIM = DISPLAY_SIZE + 16 + 2 + PC_WINDOW + 2  # 2100

# Model input: state + goal display
INPUT_DIM = STATE_DIM + DISPLAY_SIZE  # 4148

N_HIGH = 256
N_LOW = 256


class ReflexModel(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.high_head = nn.Linear(dim, N_HIGH)
        self.low_head = nn.Linear(dim, N_LOW)

    def __call__(self, x):
        h = self.net(x)
        return self.high_head(h), self.low_head(h)
