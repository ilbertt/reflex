"""
Reflex model: reads raw machine state + goal display, emits opcodes.

Input: compact state (display + goal + registers + PC-local memory + prev opcode)
Output: next opcode as (high_byte, low_byte) classification
"""

import mlx.nn as nn

from chip8 import DISPLAY_SIZE

PC_WINDOW = 32
PREV_OPCODE_DIM = 2
COMPACT_DIM = DISPLAY_SIZE + DISPLAY_SIZE + 16 + 2 + PC_WINDOW + PREV_OPCODE_DIM  # 4148

N_HIGH = 256
N_LOW = 256


class ReflexModel(nn.Module):
    """Reads compact machine state + goal, emits the next opcode."""
    def __init__(self, dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(COMPACT_DIM, dim),
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
