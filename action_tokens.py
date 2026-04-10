"""
Action token vocabulary.

Converts GUI actions to/from compact token sequences.
Screen positions are normalized to a 20x20 grid (400 cells).
"""

import re

GRID = 20  # 20x20 grid = 400 position tokens, ~5% screen precision

# ── Encode ───────────────────────────────────────────────────────────────────

def encode_action(action_str: str) -> str:
    """Convert a text action like 'click(x=0.41, y=0.18)' to action tokens.

    Input formats supported (from AGUVIS / Smol2Operator):
        click(x=0.41, y=0.178)
        type(text="hello world")
        scroll(direction="down")
        press(key="enter")
        hotkey("ctrl", "a")
    """
    action_str = action_str.strip()

    # click(x=0.41, y=0.178) or click(0.41, 0.178)
    m = re.match(r'click\s*\(\s*(?:x\s*=\s*)?([0-9.]+)\s*,\s*(?:y\s*=\s*)?([0-9.]+)\s*\)', action_str)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        gx = min(int(x * GRID), GRID - 1)
        gy = min(int(y * GRID), GRID - 1)
        return f"C{gx:02d}{gy:02d}"

    # type(text="...")
    m = re.match(r'type\s*\(\s*(?:text\s*=\s*)?["\'](.+?)["\']\s*\)', action_str)
    if m:
        text = m.group(1)[:50]  # cap length
        return f"T|{text}|"

    # scroll
    m = re.match(r'scroll\s*\(\s*(?:direction\s*=\s*)?["\']?(up|down|left|right)["\']?\s*\)', action_str)
    if m:
        d = m.group(1)[0].upper()  # U/D/L/R
        return f"S{d}"

    # press / hotkey
    m = re.match(r'(?:press|hotkey)\s*\((.+)\)', action_str)
    if m:
        keys = re.findall(r'["\']([^"\']+)["\']', m.group(1))
        key_str = "+".join(k.lower() for k in keys) if keys else m.group(1).strip().strip("'\"").lower()
        return f"K|{key_str}|"

    # Fallback: pass through (shouldn't happen with clean data)
    return f"?|{action_str[:30]}|"


def decode_action(token: str) -> str:
    """Convert an action token back to executable action text."""
    if token.startswith("C") and len(token) == 5:
        gx, gy = int(token[1:3]), int(token[3:5])
        x = (gx + 0.5) / GRID
        y = (gy + 0.5) / GRID
        return f"click(x={x:.3f}, y={y:.3f})"

    if token.startswith("T|") and token.endswith("|"):
        text = token[2:-1]
        return f'type(text="{text}")'

    if token.startswith("S") and len(token) == 2:
        dirs = {"U": "up", "D": "down", "L": "left", "R": "right"}
        return f'scroll(direction="{dirs.get(token[1], "down")}")'

    if token.startswith("K|") and token.endswith("|"):
        key = token[2:-1]
        if "+" in key:
            parts = key.split("+")
            return f'hotkey({", ".join(repr(p) for p in parts)})'
        return f'press(key="{key}")'

    return token


def encode_action_sequence(actions: list[str]) -> str:
    """Encode a list of action strings into a single token sequence."""
    return " ".join(encode_action(a) for a in actions)


def decode_action_sequence(token_str: str) -> list[str]:
    """Decode a token sequence back to action strings."""
    # Split on spaces, but keep T|...|  and K|...| together
    tokens = []
    current = ""
    in_pipe = False
    for ch in token_str:
        if ch == "|":
            current += ch
            if in_pipe:
                tokens.append(current)
                current = ""
                in_pipe = False
            else:
                in_pipe = True
        elif ch == " " and not in_pipe:
            if current:
                tokens.append(current)
            current = ""
        else:
            current += ch
    if current:
        tokens.append(current)

    return [decode_action(t) for t in tokens if t.strip()]


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        'click(x=0.41, y=0.178)',
        'type(text="hello world")',
        'scroll(direction="down")',
        'press(key="enter")',
        'hotkey("ctrl", "a")',
    ]

    print("Round-trip test:")
    for action in tests:
        token = encode_action(action)
        back = decode_action(token)
        print(f"  {action:40s} → {token:15s} → {back}")

    print(f"\nMulti-action sequence:")
    seq = encode_action_sequence(tests)
    print(f"  Encoded: {seq}")
    print(f"  Decoded: {decode_action_sequence(seq)}")
    print(f"  Token count: {len(seq.split())} (vs {sum(len(a) for a in tests)} chars of text)")
