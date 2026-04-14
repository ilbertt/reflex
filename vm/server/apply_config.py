"""Apply nginx config changes and reload. Called by the reflex kernel."""

import sys
import subprocess
import re


def apply(worker_processes=None, worker_connections=None, rate_limit=None):
    """Modify nginx config and reload."""
    conf_path = "/etc/nginx/nginx.conf"

    with open(conf_path) as f:
        conf = f.read()

    if worker_processes is not None:
        conf = re.sub(r"worker_processes \d+;", f"worker_processes {worker_processes};", conf)

    if worker_connections is not None:
        conf = re.sub(r"worker_connections \d+;", f"worker_connections {worker_connections};", conf)

    if rate_limit is not None:
        conf = re.sub(r"rate=\d+r/s", f"rate={rate_limit}r/s", conf)

    with open(conf_path, "w") as f:
        f.write(conf)

    subprocess.run(["nginx", "-s", "reload"], capture_output=True)


if __name__ == "__main__":
    # Usage: python3 /apply_config.py workers=4 connections=256 rate=200
    kwargs = {}
    for arg in sys.argv[1:]:
        key, val = arg.split("=")
        if key == "workers":
            kwargs["worker_processes"] = int(val)
        elif key == "connections":
            kwargs["worker_connections"] = int(val)
        elif key == "rate":
            kwargs["rate_limit"] = int(val)
    if kwargs:
        apply(**kwargs)
        print(f"Applied: {kwargs}")
