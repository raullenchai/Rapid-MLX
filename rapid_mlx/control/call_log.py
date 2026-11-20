# SPDX-License-Identifier: Apache-2.0
import time


class CallLog:
    def __init__(self, app, calls):
        self.app, self.calls = app, calls

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if (
            scope["type"] != "http"
            or not path.startswith("/v1/")
            or path == "/v1/models"
        ):
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        label = headers.get(
            b"x-rapid-client",
            headers.get(b"user-agent", b"").split(b" ")[0].split(b"/")[0],
        )
        text = label.decode("utf-8", "replace")
        label = "".join(c for c in text if c.isprintable())[:40]
        started, stamp = time.monotonic(), time.time()
        status: int | None = None

        async def observe(message):
            nonlocal status
            if message["type"] == "http.response.start" and status is None:
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, observe)
        finally:
            self.calls.append(
                dict(
                    ts=stamp,
                    method=scope["method"],
                    path=path,
                    status=status if status is not None else 500,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    client=label,
                )
            )
