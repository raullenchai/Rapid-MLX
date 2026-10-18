"""Small stdio MCP server giving Goose one citable web search tool."""

from __future__ import annotations

import ipaddress
import json
import os
import sys
import urllib.request
from urllib.parse import urlsplit


def _post(url: str, payload: dict, headers: dict[str, str] | None = None) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Keenable-Title": "Rapid-MLX",
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("Search provider returned an invalid result")
    return result


def _get(url: str, headers: dict[str, str] | None = None) -> dict:
    with urllib.request.urlopen(
        urllib.request.Request(
            url, headers={"X-Keenable-Title": "Rapid-MLX", **(headers or {})}
        ),
        timeout=25,
    ) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("Page provider returned an invalid result")
    return result


def _search(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("Search query is empty")
    parallel = os.environ.get("PARALLEL_API_KEY", "")
    tavily = os.environ.get("TAVILY_API_KEY", "")
    brave = os.environ.get("BRAVE_API_KEY", "")
    if parallel:
        data = _post(
            "https://api.parallel.ai/v1/search",
            {
                "objective": query,
                "search_queries": [query],
                "mode": "advanced",
                "advanced_settings": {
                    "max_results": 5,
                    "excerpt_settings": {"max_chars_per_result": 240},
                },
            },
            {"x-api-key": parallel},
        )
        results = [
            (
                item.get("title", ""),
                item.get("url", ""),
                " ".join(item.get("excerpts") or []),
            )
            for item in data.get("results", [])
        ]
    elif tavily:
        data = _post(
            "https://api.tavily.com/search",
            {
                "query": query,
                "max_results": 5,
                "search_depth": "basic",
                "api_key": tavily,
            },
        )
        results = [
            (item.get("title", ""), item.get("url", ""), item.get("content", ""))
            for item in data.get("results", [])
        ]
    elif brave:
        from urllib.parse import urlencode

        data = _get(
            "https://api.search.brave.com/res/v1/web/search?"
            + urlencode({"q": query, "count": 5, "safesearch": "moderate"}),
            {"X-Subscription-Token": brave},
        )
        results = [
            (item.get("title", ""), item.get("url", ""), item.get("description", ""))
            for item in (data.get("web") or {}).get("results", [])
        ]
    else:
        data = _post(
            "https://api.keenable.ai/mcp",
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "search_web_pages",
                    "arguments": {"query": query, "snippet_max_length": 240},
                },
            },
            {"Accept": "application/json, text/event-stream"},
        )
        if data.get("error") or (data.get("result") or {}).get("isError"):
            raise ValueError("Keenable search failed")
        content = (data.get("result") or {}).get("content") or []
        blocks = "\n".join(str(item.get("text", "")) for item in content)
        return (
            "\n\n".join(blocks.split("\n---\n")[:5])[:4096]
            or "No search results found."
        )
    lines = []
    for index, (title, url, excerpt) in enumerate(results[:5], 1):
        if isinstance(url, str) and url.startswith(("https://", "http://")):
            lines.append(f"[{index}] {title}\n{url}\n{str(excerpt)[:240]}")
    return "\n\n".join(lines)[:4096] or "No search results found."


def _fetch_page(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if (
        parsed.scheme not in {"http", "https"}
        or not host
        or host.lower().rstrip(".") == "localhost"
    ):
        raise ValueError("Expected a public HTTP(S) page URL")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError("Expected a public page URL")
    from urllib.parse import urlencode

    data = _get("https://api.keenable.ai/v1/fetch/public?" + urlencode({"url": url}))
    title = str(data.get("title") or "Web page")
    content = str(data.get("content") or data.get("markdown") or "")[:12000]
    return f"[1] {title}\n{url}\n{content}"


def _call_gate(name: str, arguments: dict) -> str:
    port = int(os.environ["WEB_SEARCH_GATE_PORT"])
    token = os.environ["WEB_SEARCH_GATE_TOKEN"]
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/web-search",
        data=json.dumps({"name": name, **arguments}).encode(),
        headers={"Content-Type": "application/json", "X-Search-Key": token},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return str(json.load(response)["result"])


TOOLS = [
    {
        "name": "web_search",
        "description": "Search the live web for current facts, weather, news, and prices. Returns source URLs to cite in the answer.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "fetch_page",
        "description": "Read a public web page from its URL as text for more detail after web_search.",
        "inputSchema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
]


def _handle(message: dict) -> dict | None:
    method = message.get("method")
    if "id" not in message:
        return None
    result: dict[str, object]
    if method == "initialize":
        result = {
            "protocolVersion": "2025-03-26",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "rapid-web-search", "version": "1.0"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "ping":
        result = {}
    elif method == "tools/call":
        params = message.get("params") or {}
        arguments = params.get("arguments") or {}
        try:
            if params.get("name") == "web_search":
                content = _call_gate(
                    "web_search", {"query": str(arguments.get("query", ""))}
                )
            elif params.get("name") == "fetch_page":
                content = _call_gate(
                    "fetch_page", {"url": str(arguments.get("url", ""))}
                )
            else:
                raise ValueError("Unknown tool")
            result = {"content": [{"type": "text", "text": content}]}
        except (OSError, ValueError, TypeError) as exc:
            result = {"content": [{"type": "text", "text": str(exc)}], "isError": True}
    else:
        return {
            "jsonrpc": "2.0",
            "id": message["id"],
            "error": {"code": -32601, "message": "Method not found"},
        }
    return {"jsonrpc": "2.0", "id": message["id"], "result": result}


def main() -> None:
    for line in sys.stdin:
        try:
            response = _handle(json.loads(line))
            if response is not None:
                print(json.dumps(response, ensure_ascii=False), flush=True)
        except (ValueError, TypeError, KeyError):
            continue


if __name__ == "__main__":
    main()
