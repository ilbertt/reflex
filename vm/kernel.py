"""
Reflex kernel: controls a Linux VM through a TCP socket.

The model sends syscall signals over TCP to a container running syscall_exec.
syscall_exec receives them, calls syscall(), sends back the result.

This module manages the container and provides:
  - observe(): read VM state as a tensor
  - execute(): send raw syscall bytes to the VM
"""

import socket
import struct
import subprocess
import time
import os
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

# ── Linux syscall numbers (aarch64, Alpine/OrbStack on Apple Silicon) ─────

SYS = {
    "read":      63,
    "write":     64,
    "openat":    56,
    "close":     57,
    "fstat":     80,
    "mkdirat":   34,
    "unlinkat":  35,
    "renameat2": 276,
    "getcwd":    17,
    "getpid":    172,
    "execve":    221,
    "clone":     220,
    "wait4":     260,
    "kill":      129,
    "exit":      93,
}

# Commonly used flags
O_RDONLY = 0
O_WRONLY = 1
O_RDWR = 2
O_CREAT = 0o100
O_TRUNC = 0o1000
O_APPEND = 0o2000

BUF_OFFSET = 0x10000  # buffer pointer base

# ── State encoding ────────────────────────────────────────────────────────

MAX_FILES = 32
FILE_NAME_BYTES = 16
STDOUT_BYTES = 128

STATE_DIM = 3 + 4 + MAX_FILES * (3 + FILE_NAME_BYTES) + STDOUT_BYTES

# TCP port for syscall_exec
SYSCALL_PORT = 7777


def pack_syscall(nr: int, args: list[int], buffer: bytes = b"") -> bytes:
    """Pack a syscall into the binary protocol."""
    buf_len = len(buffer)
    data = struct.pack("<HH", nr, buf_len)
    padded_args = (args + [0] * 6)[:6]
    data += struct.pack("<6q", *padded_args)
    data += buffer[:4096]
    return data


# ── VM Container ──────────────────────────────────────────────────────────

class VM:
    """A real Linux VM controlled through TCP."""

    def __init__(self, name: str = "reflex-vm", build: bool = True):
        self.name = name
        self.sock = None
        self.step_count = 0

        if build:
            self._build()
        self._start()
        self._connect()

    def _build(self):
        """Build the container image."""
        print("  Building VM image...")
        r = subprocess.run(
            ["docker", "build", "-t", "reflex-vm", "-f",
             os.path.join(DIR, "Dockerfile"), DIR],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Docker build failed:\n{r.stderr}")
        print("  Image built.")

    def _start(self):
        """Start the container."""
        subprocess.run(["docker", "rm", "-f", self.name],
                       capture_output=True, timeout=5)

        self.proc = subprocess.Popen(
            ["docker", "run", "--rm", "--name", self.name,
             "--security-opt", "seccomp=unconfined",
             "-p", f"{SYSCALL_PORT}:{SYSCALL_PORT}",
             "reflex-vm"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.5)
        if self.proc.poll() is not None:
            err = self.proc.stderr.read().decode()
            raise RuntimeError(f"Container failed to start: {err}")
        print(f"  VM running: {self.name} (pid={self.proc.pid})")

    def _connect(self):
        """Connect to syscall_exec via TCP."""
        for attempt in range(20):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(5.0)
                self.sock.connect(("127.0.0.1", SYSCALL_PORT))
                return
            except (ConnectionRefusedError, OSError):
                time.sleep(0.2)
        raise RuntimeError("Could not connect to syscall_exec")

    def _recvn(self, n: int) -> bytes:
        """Read exactly n bytes from socket."""
        data = b""
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk
        return data

    def observe(self) -> np.ndarray:
        """Read VM state as a flat tensor via docker exec."""
        state = np.zeros(STATE_DIM, dtype=np.float32)
        offset = 0

        # Loadavg
        r = subprocess.run(["docker", "exec", self.name, "cat", "/proc/loadavg"],
                           capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            parts = r.stdout.split()
            for i in range(min(3, len(parts))):
                try: state[offset + i] = float(parts[i]) / 10.0
                except ValueError: pass
        offset += 3

        # Memory
        r = subprocess.run(["docker", "exec", self.name, "cat", "/proc/meminfo"],
                           capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            for line in r.stdout.split("\n"):
                try:
                    if "MemTotal:" in line:
                        state[offset + 0] = int(line.split()[1]) / 1e6
                    elif "MemFree:" in line:
                        state[offset + 1] = int(line.split()[1]) / 1e6
                    elif "MemAvailable:" in line:
                        state[offset + 2] = int(line.split()[1]) / 1e6
                    elif "Cached:" in line and "Swap" not in line:
                        state[offset + 3] = int(line.split()[1]) / 1e6
                except (ValueError, IndexError): pass
        offset += 4

        # File listing
        r = subprocess.run(
            ["docker", "exec", self.name, "/bin/sh", "-c",
             "cd /workspace && find . -maxdepth 2 -not -path . 2>/dev/null | "
             "while read f; do "
             "  if [ -d \"$f\" ]; then echo \"d 0 $f\"; "
             "  else s=$(stat -c%s \"$f\" 2>/dev/null || echo 0); echo \"f $s $f\"; fi; "
             "done | sort"],
            capture_output=True, text=True, timeout=3)
        files = []
        if r.returncode == 0:
            for line in r.stdout.strip().split("\n"):
                if not line: continue
                parts = line.split(None, 2)
                if len(parts) >= 3:
                    files.append((parts[2].lstrip("./"), parts[0] == "d", int(parts[1])))

        for i in range(MAX_FILES):
            base = offset + i * (3 + FILE_NAME_BYTES)
            if i < len(files):
                name, is_dir, size = files[i]
                state[base + 0] = 1.0
                state[base + 1] = 1.0 if is_dir else 0.0
                state[base + 2] = min(size / 1e6, 1.0)
                for j, b in enumerate(name.encode("utf-8")[:FILE_NAME_BYTES]):
                    state[base + 3 + j] = b / 255.0
        offset += MAX_FILES * (3 + FILE_NAME_BYTES)

        # Stdout from last command
        r = subprocess.run(["docker", "exec", self.name, "cat", "/tmp/last_stdout"],
                           capture_output=True, timeout=3)
        if r.returncode == 0:
            for j, b in enumerate(r.stdout[:STDOUT_BYTES]):
                state[offset + j] = b / 255.0
        offset += STDOUT_BYTES

        return state

    def execute(self, syscall_bytes: bytes) -> tuple[int, int]:
        """Send raw syscall bytes via TCP. Returns (retval, errno)."""
        self.step_count += 1

        header_size = 2 + 2 + 48
        if len(syscall_bytes) < header_size:
            syscall_bytes = syscall_bytes + b'\x00' * (header_size - len(syscall_bytes))

        buf_len = struct.unpack_from("<H", syscall_bytes, 2)[0]
        total = header_size + buf_len
        if len(syscall_bytes) < total:
            syscall_bytes = syscall_bytes + b'\x00' * (total - len(syscall_bytes))

        try:
            self.sock.sendall(syscall_bytes[:total])
            resp = self._recvn(12)
        except (ConnectionError, socket.timeout, OSError):
            self._restart()
            return -1, -1

        retval = struct.unpack("<q", resp[:8])[0]
        err = struct.unpack("<i", resp[8:12])[0]
        return retval, err

    def _restart(self):
        """Restart the container after a failure."""
        print("  [VM] Connection lost, restarting...")
        if self.sock:
            try: self.sock.close()
            except Exception: pass
        try: self.proc.kill()
        except Exception: pass
        subprocess.run(["docker", "rm", "-f", self.name],
                       capture_output=True, timeout=5)
        self._start()
        self._connect()

    def stop(self):
        """Shutdown the VM."""
        if self.sock:
            try: self.sock.close()
            except Exception: pass
        if self.proc and self.proc.poll() is None:
            try: self.proc.kill()
            except Exception: pass
        subprocess.run(["docker", "rm", "-f", self.name],
                       capture_output=True, timeout=5)

    # ── Convenience: execute structured syscalls (for data collection) ───

    AT_FDCWD = -100

    def sys_openat(self, path: str, flags: int = O_RDONLY, mode: int = 0o644) -> int:
        buf = path.encode() + b'\x00'
        data = pack_syscall(SYS["openat"], [self.AT_FDCWD, BUF_OFFSET, flags, mode], buf)
        ret, err = self.execute(data)
        return ret

    def sys_write(self, fd: int, content: bytes) -> int:
        data = pack_syscall(SYS["write"], [fd, BUF_OFFSET, len(content)], content)
        ret, err = self.execute(data)
        return ret

    def sys_close(self, fd: int) -> int:
        data = pack_syscall(SYS["close"], [fd], b"")
        ret, err = self.execute(data)
        return ret

    def sys_mkdirat(self, path: str, mode: int = 0o755) -> int:
        buf = path.encode() + b'\x00'
        data = pack_syscall(SYS["mkdirat"], [self.AT_FDCWD, BUF_OFFSET, mode], buf)
        ret, err = self.execute(data)
        return ret

    def sys_unlinkat(self, path: str, flags: int = 0) -> int:
        buf = path.encode() + b'\x00'
        data = pack_syscall(SYS["unlinkat"], [self.AT_FDCWD, BUF_OFFSET, flags], buf)
        ret, err = self.execute(data)
        return ret

    def sys_getpid(self) -> int:
        data = pack_syscall(SYS["getpid"], [], b"")
        ret, err = self.execute(data)
        return ret

    def create_file(self, path: str, content: str) -> bool:
        fd = self.sys_openat(path, O_CREAT | O_WRONLY | O_TRUNC, 0o644)
        if fd < 0:
            return False
        self.sys_write(fd, content.encode())
        self.sys_close(fd)
        return True

    def run_command(self, cmd: str) -> tuple[int, str]:
        """Run a shell command via docker exec."""
        r = subprocess.run(
            ["docker", "exec", self.name, "/bin/sh", "-c",
             f"cd /workspace && {cmd}"],
            capture_output=True, text=True, timeout=10,
        )
        if r.stdout:
            subprocess.run(
                ["docker", "exec", self.name, "/bin/sh", "-c",
                 f"cat > /tmp/last_stdout << 'HEREDOC'\n{r.stdout[:4096]}\nHEREDOC"],
                capture_output=True, timeout=3,
            )
        return r.returncode, r.stdout

    def render(self) -> str:
        r = subprocess.run(
            ["docker", "exec", self.name, "/bin/sh", "-c",
             "echo '=== FILES ===' && ls -la /workspace/ 2>/dev/null && "
             "echo '=== PROCESSES ===' && ps aux 2>/dev/null && "
             "echo '=== STDOUT ===' && cat /tmp/last_stdout 2>/dev/null"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout
