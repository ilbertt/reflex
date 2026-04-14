"""Tiny HTTP server that simulates variable-latency work."""

import http.server
import time
import random
import threading

# Simulated load — increases when traffic spikes
_load = 0.0
_lock = threading.Lock()


def add_load(delta):
    global _load
    with _lock:
        _load = max(0, min(1.0, _load + delta))


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        # Simulate work — latency increases with load
        base_latency = 0.005  # 5ms base
        load_latency = _load * 0.2  # up to 200ms under full load
        jitter = random.uniform(0, 0.01)
        time.sleep(base_latency + load_latency + jitter)

        # Each request adds a tiny bit of load (simulates resource consumption)
        add_load(0.001)

        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok\n")

    def log_message(self, format, *args):
        pass  # suppress logging


# Natural load decay
def decay_loop():
    while True:
        add_load(-0.01)
        time.sleep(0.1)


if __name__ == "__main__":
    threading.Thread(target=decay_loop, daemon=True).start()
    server = http.server.HTTPServer(("0.0.0.0", 8080), Handler)
    server.serve_forever()
