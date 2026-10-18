"""The Goose search extension provides citable results without browser tools."""

from rapid_mlx.goose_runtime import web_search_mcp


def test_keyless_search_returns_source_url(monkeypatch):
    calls = []

    def fake_post(url, payload, headers=None):
        calls.append((url, payload, headers))
        return {
            "results": [
                {
                    "title": "Weather service",
                    "url": "https://forecast.weather.gov/test",
                    "snippet": "Sunny",
                }
            ]
        }

    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(web_search_mcp, "_post", fake_post)
    result = web_search_mcp._search("current weather")
    assert calls[0][0] == "https://api.keenable.ai/v1/search/public"
    assert calls[0][1]["query"] == "current weather"
    assert "https://forecast.weather.gov/test" in result
    assert "Sunny" in result


def test_mcp_advertises_search_and_dispatches_through_gate(monkeypatch):
    seen = []
    monkeypatch.setattr(
        web_search_mcp,
        "_call_gate",
        lambda name, args: (
            seen.append((name, args)) or "[1] source\nhttps://example.org"
        ),
    )
    tools = web_search_mcp._handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert [tool["name"] for tool in tools["result"]["tools"]] == [
        "web_search",
        "fetch_page",
    ]
    call = web_search_mcp._handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "web_search", "arguments": {"query": "news"}},
        }
    )
    assert seen == [("web_search", {"query": "news"})]
    assert "https://example.org" in call["result"]["content"][0]["text"]
