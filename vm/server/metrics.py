"""Exposes server metrics as a simple JSON endpoint on port 9100."""

import http.server
import json
import os
import time
import threading

_metrics = {
    "request_rate": 0.0,
    "p99_latency": 0.0,
    "error_rate": 0.0,
    "active_connections": 0,
    "worker_count": 2,
}
_lock = threading.Lock()


def update_metrics():
    """Read nginx status and access log to compute live metrics."""
    import urllib.request
    while True:
        try:
            # Read nginx stub_status
            resp = urllib.request.urlopen("http://127.0.0.1:80/nginx_status", timeout=1)
            text = resp.read().decode()
            for line in text.split("\n"):
                if line.startswith("Active connections:"):
                    with _lock:
                        _metrics["active_connections"] = int(line.split(":")[1].strip())

            # Parse recent access log for latency and error stats
            log_path = "/var/log/nginx/access.log"
            if os.path.exists(log_path):
                with open(log_path) as f:
                    lines = f.readlines()
                recent = lines[-100:] if len(lines) > 100 else lines
                latencies = []
                errors = 0
                for line in recent:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            latencies.append(float(parts[0]))
                            status = int(parts[1])
                            if status >= 500:
                                errors += 1
                        except ValueError:
                            pass

                with _lock:
                    if latencies:
                        latencies.sort()
                        _metrics["p99_latency"] = latencies[int(len(latencies) * 0.99)]
                        _metrics["request_rate"] = len(recent) / max(1, latencies[-1] - latencies[0] + 0.1) if len(recent) > 1 else 0
                    _metrics["error_rate"] = errors / max(1, len(recent))

            # Read worker count from nginx config
            try:
                with open("/etc/nginx/nginx.conf") as f:
                    for line in f:
                        if "worker_processes" in line:
                            _metrics["worker_count"] = int(line.split()[1].rstrip(";"))
            except (ValueError, IndexError):
                pass

        except Exception:
            pass

        time.sleep(0.5)


class MetricsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        with _lock:
            data = json.dumps(_metrics)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data.encode())

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    threading.Thread(target=update_metrics, daemon=True).start()
    server = http.server.HTTPServer(("0.0.0.0", 9100), MetricsHandler)
    server.serve_forever()
