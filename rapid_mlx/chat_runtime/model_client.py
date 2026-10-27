"""Lifespan-owned, loopback-only inference transport."""

from urllib.parse import urlsplit, urlunsplit

import httpx
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from .loopback_transport import AsyncLoopbackTransport


def inference_url(value: str) -> str:
    url = urlsplit(value)
    if (
        url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "::1", "localhost"}
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
        or url.path.rstrip("/") not in {"", "/v1"}
        or not url.port
    ):
        raise ValueError("inference_url must be a loopback HTTP server URL with a port")
    # Preserve the hostname; the transport pins localhost to loopback addresses.
    host = "[::1]" if url.hostname == "::1" else url.hostname
    return urlunsplit(("http", f"{host}:{url.port}", "/v1", "", ""))


class ModelClients:
    def __init__(self, *, http_client: httpx.AsyncClient | None = None):
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(180, connect=5),
            trust_env=False,
            transport=AsyncLoopbackTransport(trust_env=False),
            follow_redirects=False,
        )
        self._clients: dict[tuple[str, str], AsyncOpenAI] = {}
        self._models: dict[tuple[str, str, str], OpenAIChatModel] = {}

    def model(self, endpoint: str, model: str, credential: str | None = None):
        endpoint = inference_url(endpoint)
        key = (endpoint, credential or "rapid-local")
        if key not in self._clients:
            self._clients[key] = AsyncOpenAI(
                base_url=endpoint,
                api_key=key[1],
                max_retries=0,
                http_client=self._http,
            )
        model_key = (*key, model)
        if model_key not in self._models:
            self._models[model_key] = OpenAIChatModel(
                model,
                provider=OpenAIProvider(openai_client=self._clients[key]),
                profile=OpenAIModelProfile(
                    openai_chat_thinking_field="reasoning_content",
                    openai_chat_send_back_thinking_parts="field",
                    openai_supports_strict_tool_definition=False,
                ),
            )
        return self._models[model_key]

    async def close(self):
        for client in self._clients.values():
            await client.close()
        await self._http.aclose()
        self._clients.clear()
        self._models.clear()
