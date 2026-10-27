"""Pin localhost connections to loopback addresses without relying on DNS."""

import httpx


class LoopbackTransport(httpx.HTTPTransport):
    def handle_request(self, request):
        if request.url.host != "localhost":
            return super().handle_request(request)
        original = request.url
        try:
            for host in ("127.0.0.1", "::1"):
                request.url = original.copy_with(host=host)
                try:
                    return super().handle_request(request)
                except httpx.ConnectError:
                    if host == "::1":
                        raise
        finally:
            request.url = original


class AsyncLoopbackTransport(httpx.AsyncHTTPTransport):
    async def handle_async_request(self, request):
        if request.url.host != "localhost":
            return await super().handle_async_request(request)
        original = request.url
        try:
            for host in ("127.0.0.1", "::1"):
                request.url = original.copy_with(host=host)
                try:
                    return await super().handle_async_request(request)
                except httpx.ConnectError:
                    if host == "::1":
                        raise
        finally:
            request.url = original
