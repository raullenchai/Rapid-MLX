"""Request-local queue priority for text/VLM batched inference."""

from contextvars import ContextVar

from starlette.types import ASGIApp, Receive, Scope, Send

request_priority: ContextVar[str] = ContextVar(
    "request_priority", default="interactive"
)


class PriorityMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        value = next(
            (v for k, v in scope["headers"] if k.lower() == b"x-rapid-priority"), b""
        )
        priority = "background" if value.lower() == b"background" else "interactive"
        token = request_priority.set(priority)
        try:
            await self.app(scope, receive, send)
        finally:
            request_priority.reset(token)
