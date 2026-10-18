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
                    "snippet": "Sunny " * 500,
                }
            ]
        }

    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.setattr(web_search_mcp, "_post", fake_post)
    result = web_search_mcp._search("current weather")
    assert calls[0][0] == "https://api.keenable.ai/v1/search/public"
    assert calls[0][1]["query"] == "current weather"
    assert "https://forecast.weather.gov/test" in result
    assert "Sunny" in result
    assert len(result) < 350


def test_brave_key_uses_api_without_exposing_key(monkeypatch):
    seen = []
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("BRAVE_API_KEY", "test-secret")
    monkeypatch.setattr(
        web_search_mcp,
        "_get",
        lambda url, headers=None: (
            seen.append((url, headers))
            or {
                "web": {
                    "results": [
                        {
                            "title": "Story",
                            "url": "https://example.org",
                            "description": "News",
                        }
                    ]
                }
            }
        ),
    )
    result = web_search_mcp._search("latest story")
    assert "api.search.brave.com" in seen[0][0]
    assert seen[0][1] == {"X-Subscription-Token": "test-secret"}
    assert "https://example.org" in result
    assert "test-secret" not in result


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
