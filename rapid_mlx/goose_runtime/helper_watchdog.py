#!/usr/bin/env python3
"""Launch a helper and reap its complete observed process tree on Stop or owner death."""

import os
import signal
import subprocess
import sys
import time

if len(sys.argv) > 1 and sys.argv[1] == "--credential-bridge":
    # The inference proxy owns the real engine bearer. Goose only gets a
    # per-launch token accepted here; tools inherit neither credential.
    import hmac
    import http.client
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    listen_port, upstream_port = map(int, sys.argv[2:4])
    scoped = os.environ["WORK_BRIDGE_SCOPED_KEY"]
    upstream = os.environ["WORK_BRIDGE_UPSTREAM_KEY"]
    contract_path = os.environ.get("WORK_BRIDGE_CONTRACT_FILE", "chat-contract.json")

    class Bridge(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            if self.path != "/v1/chat/completions" or not hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + scoped
            ):
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
                if not 0 < length <= 100_000_000:
                    self.send_error(413)
                    return
                body = self.rfile.read(length)
                # Desktop owns these per-turn settings. Goose may append its
                # own tool instructions, but cannot silently replace the user
                # instruction/memory layer or sampling controls.
                with open(contract_path, encoding="utf-8") as contract_file:
                    contract = json.load(contract_file)
                request = json.loads(body)
                messages = request.get("messages", [])
                system = contract.get("system_prompt", "")
                if system:
                    messages.insert(0, {"role": "system", "content": system})
                    request["_rapid_desktop_system"] = True
                request["messages"] = messages
                if not contract.get("use_goose_sampling_defaults", False):
                    request["temperature"] = contract["temperature"]
                    request["top_p"] = contract["top_p"]
                    request["max_tokens"] = contract["max_tokens"]
                    request["repetition_penalty"] = contract["repetition_penalty"]
                    request["frequency_penalty"] = 0.0
                    request["presence_penalty"] = 0.0
                    kwargs = request.get("chat_template_kwargs") or {}
                    if contract["enable_thinking"]:
                        kwargs.pop("enable_thinking", None)
                    else:
                        kwargs["enable_thinking"] = False
                    if kwargs:
                        request["chat_template_kwargs"] = kwargs
                    else:
                        request.pop("chat_template_kwargs", None)
                body = json.dumps(request).encode("utf-8")
                connection = http.client.HTTPConnection(
                    "127.0.0.1", upstream_port, timeout=900
                )
                connection.request(
                    "POST",
                    self.path,
                    body,
                    {
                        "Authorization": "Bearer " + upstream,
                        "Content-Type": "application/json",
                    },
                )
                response = connection.getresponse()
                self.send_response(response.status)
                for name, value in response.getheaders():
                    if name.lower() not in {
                        "connection",
                        "transfer-encoding",
                        "content-length",
                    }:
                        self.send_header(name, value)
                self.end_headers()
                while chunk := response.read1(16384):
                    self.wfile.write(chunk)
                    self.wfile.flush()
                connection.close()
            except (OSError, ValueError):
                self.close_connection = True

    ThreadingHTTPServer(("127.0.0.1", listen_port), Bridge).serve_forever()
    sys.exit(0)

# Goose invokes this executable as its shell. Remove helper credentials before
# the command starts; a login shell would re-import the user's configuration.
if len(sys.argv) > 1 and sys.argv[1] == "-c":
    clean = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "GOOSE_SERVER__SECRET_KEY",
            "RAPID_LOCAL_WORK_KEY",
            "WEB_SEARCH_GATE_TOKEN",
        }
        and not key.startswith("WORK_PROXY_")
    }
    os.execve("/bin/bash", ["/bin/bash", "-c", sys.argv[2]], clean)

owner = int(sys.argv[1])
command = sys.argv[2:]
try:
    os.setsid()
except PermissionError:
    if os.getpgrp() != os.getpid():
        raise
root = os.getpid()
group = os.getpgrp()

if os.fork() != 0:
    os.execv(command[0], command)

known: dict[int, str] = {}
stopping = False


def tree():
    """Add children of known PIDs even after they leave the process group."""
    global known
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,lstart="],
        capture_output=True,
        text=True,
        check=True,
    )
    pairs = {}
    for line in result.stdout.splitlines():
        try:
            pid_text, ppid_text, started = line.split(maxsplit=2)
            pairs[int(pid_text)] = (int(ppid_text), started)
        except ValueError:
            continue
    if not known and root in pairs:
        known[root] = pairs[root][1]
    # A PID can be reused during a long turn. Never signal a process whose
    # start time differs from the one we observed in the helper tree.
    known = {
        pid: stamp
        for pid, stamp in known.items()
        if pid in pairs and pairs[pid][1] == stamp
    }
    changed = True
    while changed:
        changed = False
        for pid, (ppid, stamp) in pairs.items():
            if ppid in known and pid not in known and pid != os.getpid():
                known[pid] = stamp
                changed = True
    return known


def terminate(*_):
    global stopping
    stopping = True


signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)


def token_descendants():
    # A tool can fork and exit between tree snapshots. Its detached child
    # still carries this per-launch marker unless it deliberately clears env.
    token = os.environ.get("WORK_HELPER_OWNERSHIP_TOKEN")
    if not token:
        return set()
    result = subprocess.run(
        ["/bin/ps", "eww", "-axo", "pid=,command="],
        capture_output=True,
        text=True,
        check=True,
    )
    marker = "WORK_HELPER_OWNERSHIP_TOKEN=" + token
    found = set()
    for line in result.stdout.splitlines():
        if marker in line:
            try:
                found.add(int(line.split(maxsplit=1)[0]))
            except (IndexError, ValueError):
                pass
    return found - {os.getpid()}


while not stopping:
    try:
        observed = tree()
        os.kill(owner, 0)
        if root not in observed:
            stopping = True
    except ProcessLookupError:
        stopping = True
    except subprocess.SubprocessError:
        pass
    time.sleep(0.02)

try:
    tree()
except subprocess.SubprocessError:
    pass
for sig in (signal.SIGTERM, signal.SIGKILL):
    try:
        tree()
    except subprocess.SubprocessError:
        break
    try:
        targets = known.keys() | token_descendants()
    except subprocess.SubprocessError:
        targets = known.keys()
    for pid in targets - {os.getpid()}:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
    time.sleep(0.1)
# The original group catches any process created between snapshots.
try:
    os.killpg(group, signal.SIGKILL)
except ProcessLookupError:
    pass
