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

    async def vision_capability(self, endpoint, model, credential=None):
        """Return True/False only for well-formed selected-model capabilities."""
        try:
            response = await self._http.get(
                inference_url(endpoint) + "/models",
                headers={"Authorization": "Bearer " + (credential or "rapid-local")},
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()["data"]
            entry = next(item for item in data if item["id"] == model)
            capabilities = entry["capabilities"]
            if not isinstance(capabilities, list) or not all(
                isinstance(c, str) for c in capabilities
            ):
                return None
            return "vision" in capabilities
        except (httpx.HTTPError, ValueError, KeyError, TypeError, StopIteration):
            return None

    async def folder_results(self, endpoint, scope, query, credential=None):
        """Read only the explicit local index; never open source files or redirect."""
        import json

        from .folder_chat import UNAVAILABLE, FolderChatError

        base = inference_url(endpoint) + "/local-search"
        headers = {"Authorization": "Bearer " + (credential or "rapid-local")}

        async def fetch(method, route, payload=None):
            try:
                async with self._http.stream(
                    method, base + route, headers=headers, json=payload, timeout=15
                ) as response:
                    if response.status_code in {401, 403}:
                        raise FolderChatError(
                            "local_search_unauthorized",
                            "Folder search needs authorization. Reconnect to the local server.",
                        )
                    response.raise_for_status()
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        content.extend(chunk)
                        if len(content) > 1024 * 1024:
                            raise ValueError("local search response too large")
                value = json.loads(content)
                if not isinstance(value, dict):
                    raise ValueError("invalid local search response")
                return value
            except FolderChatError:
                raise
            except (httpx.HTTPError, ValueError, TypeError):
                raise FolderChatError(UNAVAILABLE.code, UNAVAILABLE.message) from None

        status = await fetch("GET", "/status")
        if status.get("enabled") is not True:
            raise FolderChatError(
                "local_search_disabled",
                "Enable Local Search and choose a registered folder.",
            )
        available = status.get("folders")
        if not isinstance(available, list):
            raise FolderChatError(UNAVAILABLE.code, UNAVAILABLE.message)
        folders = []
        for identifier in scope.folder_ids:
            folder = next(
                (
                    folder
                    for folder in available
                    if isinstance(folder, dict) and folder.get("id") == identifier
                ),
                None,
            )
            if folder is None or folder.get("state") in {"missing", "error"}:
                raise FolderChatError(
                    "folder_not_found",
                    "The selected folder is unavailable. Reconnect it or choose another folder.",
                )
            folders.append(folder)
        result = await fetch(
            "POST",
            "/search",
            {
                "query": query,
                "limit": scope.limit,
                "folder_ids": scope.folder_ids,
                "kinds": ["text"],
                "include_text": True,
            },
        )
        if not isinstance(result.get("results"), list) or result.get("mode") not in {
            "keyword",
            "hybrid",
        }:
            raise FolderChatError(UNAVAILABLE.code, UNAVAILABLE.message)
        for hit in result["results"]:
            if not isinstance(hit, dict):
                raise FolderChatError(UNAVAILABLE.code, UNAVAILABLE.message)
            if (
                hit.get("kind", "text") == "text"
                and set(hit.get("matched", ())) & {"keyword", "semantic"}
                and (not isinstance(hit.get("text"), str) or not hit["text"].strip())
            ):
                raise FolderChatError(
                    "local_search_full_text_unsupported",
                    "The local server did not return the matched passages. Update it before using folder chat.",
                )
        return result, folders

    async def close(self):
        for client in self._clients.values():
            await client.close()
        await self._http.aclose()
        self._clients.clear()
        self._models.clear()
