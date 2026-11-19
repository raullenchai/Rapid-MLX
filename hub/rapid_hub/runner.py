"""Plugin group guardian, including when the Hub parent dies abruptly.

Executed as a script in its own session. The guardian remains alive long enough
to terminate the group after a plugin crash or reparenting, then kills itself
with the rest of the group. It never receives engine credentials.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    parent = os.getppid()
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    process = subprocess.Popen(sys.argv[1:])
    directory = os.environ.get("RAPID_PLUGIN_DATA_DIR")
    if directory:
        path = Path(directory) / "plugin.pid"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(str(process.pid))
    try:
        while process.poll() is None and not stopping and os.getppid() == parent:
            time.sleep(0.1)
    finally:
        os.killpg(os.getpgrp(), signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        # Allow descendant signal handlers to clean up their experiment groups.
        time.sleep(0.2)
        os.killpg(os.getpgrp(), signal.SIGKILL)


if __name__ == "__main__":
    main()
