"""The Goose search extension provides citable results without browser tools."""

from rapid_mlx.goose_runtime import web_search_mcp


def test_keyless_search_returns_source_url(monkeypatch):
    calls = []

    def fake_post(url, payload, headers=None):
        calls.append((url, payload, headers))
        return {
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": "Title: Weather service\nURL: https://forecast.weather.gov/test\nSnippets:\nSunny",
                    }
                ]
            }
        }

    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.setattr(web_search_mcp, "_post", fake_post)
    result = web_search_mcp._search("current weather")
    assert calls[0][0] == "https://api.keenable.ai/mcp"
    assert calls[0][1]["params"]["arguments"]["query"] == "current weather"
    assert calls[0][1]["params"]["arguments"]["snippet_max_length"] == 240
    assert "https://forecast.weather.gov/test" in result
    assert "Sunny" in result


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


def test_tavily_key_stays_in_gate_request(monkeypatch):
    calls = []
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "test-secret")
    monkeypatch.setattr(
        web_search_mcp,
        "_post",
        lambda url, payload, headers=None: (
            calls.append((url, payload, headers))
            or {
                "results": [
                    {"title": "Story", "url": "https://example.org", "content": "News"}
                ]
            }
        ),
    )
    result = web_search_mcp._search("latest story")
    assert calls[0][0] == "https://api.tavily.com/search"
    assert calls[0][1]["api_key"] == "test-secret"
    assert calls[0][1]["search_depth"] == "basic"
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
