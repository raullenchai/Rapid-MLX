"""Install a metadata-only ASGI tap in the test server process.

Python loads this through PYTHONPATH. It does not alter the installed package,
request bytes, response bytes, auth, or agent configuration.
"""

import json
import os
import time
from pathlib import Path


class _Tap:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("path") not in {
            "/v1/chat/completions",
            "/v1/messages",
            "/v1/responses",
        }:
            await self.app(scope, receive, send)
            return
        request_chunks = []
        response_tail = bytearray()
        status = None
        started = time.time()

        async def tapped_receive():
            message = await receive()
            if (
                message.get("type") == "http.request"
                and sum(map(len, request_chunks)) < 1_000_000
            ):
                request_chunks.append(message.get("body", b""))
            return message

        async def tapped_send(message):
            nonlocal status
            if message.get("type") == "http.response.start":
                status = message.get("status")
            elif message.get("type") == "http.response.body":
                response_tail.extend(message.get("body", b""))
                if len(response_tail) > 200_000:
                    del response_tail[:-200_000]
            await send(message)

        try:
            await self.app(scope, tapped_receive, tapped_send)
        finally:
            try:
                body = json.loads(b"".join(request_chunks))
                prompt_tokens = None
                for part in bytes(response_tail).splitlines():
                    if part.startswith(b"data: "):
                        part = part[6:]
                    try:
                        item = json.loads(part)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if not isinstance(item, dict):
                        continue
                    usage = (
                        item.get("usage")
                        or item.get("message", {}).get("usage")
                        or item.get("response", {}).get("usage")
                    )
                    if isinstance(usage, dict):
                        prompt_tokens = usage.get(
                            "prompt_tokens", usage.get("input_tokens", prompt_tokens)
                        )
                log = os.environ.get("AGENT_E2E_REQUEST_LOG")
                if log:
                    agent_file = os.environ.get("AGENT_E2E_CURRENT_AGENT")
                    agent = Path(agent_file).read_text().strip() if agent_file else None
                    event = {
                        "time": started,
                        "agent": agent,
                        "path": scope["path"],
                        "model": body.get("model"),
                        "status": status,
                        "prompt_tokens": prompt_tokens,
                    }
                    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                    try:
                        os.write(fd, (json.dumps(event) + "\n").encode())
                    finally:
                        os.close(fd)
            except (OSError, ValueError, TypeError, AttributeError):
                pass  # Observation must never break inference.


try:
    from rapid_mlx.middleware import request_logging as _logging

    _install = _logging.install_request_logging_middleware

    def _instrumented_install(app):
        _install(app)
        app.add_middleware(_Tap)

    _logging.install_request_logging_middleware = _instrumented_install
except ImportError:
    pass
