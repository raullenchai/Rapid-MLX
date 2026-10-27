"""Provider contracts, bounds and cancellation with no real network access."""

import asyncio
import json
from dataclasses import asdict
from datetime import date

import httpx
import pytest

from rapid_mlx.chat_runtime.tools import web
from rapid_mlx.chat_runtime.tools.web import WebTools

CITY = {
    "name": "San Francisco",
    "latitude": 37.77,
    "longitude": -122.42,
    "country": "United States",
    "country_code": "US",
    "admin1": "California",
    "population": 873965,
    "timezone": "America/Los_Angeles",
}
FORECAST = {
    "timezone": "America/Los_Angeles",
    "current": {
        "time": "2026-10-26T09:00",
        "temperature_2m": 18.2,
        "relative_humidity_2m": 70,
        "wind_speed_10m": 8.4,
        "weather_code": 2,
    },
    "daily": {
        "time": ["2026-10-26", "2026-10-27"],
        "weather_code": [2, 61],
        "temperature_2m_min": [11, 10],
        "temperature_2m_max": [19, 18],
        "precipitation_probability_max": [10, None],
        "precipitation_sum": [0],
    },
}


def client(handler, environ=None):
    return WebTools(httpx.MockTransport(handler), {} if environ is None else environ)


def keenable_response():
    return httpx.Response(
        200,
        json={
            "results": [
                {
                    "title": "Live story",
                    "url": "https://example.com/live",
                    "snippet": "A current news snippet.",
                }
            ]
        },
    )


@pytest.mark.parametrize(
    "units,temperature,wind,precip",
    [("metric", "celsius", "kmh", "mm"), ("imperial", "fahrenheit", "mph", "inch")],
)
async def test_weather_units_qualifiers_dates_and_clamped_days(
    units, temperature, wind, precip
):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.host == "geocoding-api.open-meteo.com":
            assert request.url.params["name"] == "San Francisco"
            return httpx.Response(200, json={"results": [CITY]})
        params = request.url.params
        assert params["temperature_unit"] == temperature
        assert params["wind_speed_unit"] == wind
        assert params["precipitation_unit"] == precip
        assert params["forecast_days"] == "7"
        assert params["timezone"] == "auto"
        return httpx.Response(200, json=FORECAST)

    result = await client(handler).weather(
        "SF, CA", country="US", days=100, units=units
    )
    assert result.error is None
    assert "San Francisco, California, United States" in result.content
    assert "2026-10-26 Mon (today)" in result.content
    assert "2026-10-27 Tue (tomorrow)" in result.content
    assert "n/a" in result.content
    assert len(requests) == 2


@pytest.mark.parametrize(
    "candidates,location,kwargs,code",
    [
        ([], "Missing City", {}, "not_found"),
        (
            [
                {**CITY, "name": "Springfield", "population": 115000},
                {**CITY, "name": "Springfield", "population": 75000},
            ],
            "Springfield",
            {},
            "ambiguous_location",
        ),
        ([CITY], "San Francisco, France", {}, "qualifier_mismatch"),
        ([CITY], "San Francisco", {"country": "California"}, "qualifier_mismatch"),
        ([CITY], "San Francisco", {"admin1": "US"}, "qualifier_mismatch"),
        ([CITY], "California", {}, "needs_clarification"),
        ([{**CITY, "name": "Xián", "population": 50}], "Xian", {}, "not_found"),
    ],
)
async def test_weather_resolution_failures_do_not_fetch_forecast(
    candidates, location, kwargs, code
):
    seen = []

    def handler(request):
        seen.append(request.url.host)
        return httpx.Response(200, json={"results": candidates})

    result = await client(handler).weather(location, **kwargs)
    assert result.error.code == code
    assert seen == ["geocoding-api.open-meteo.com"]


async def test_major_city_wins_and_broken_optional_daily_keeps_current():
    def handler(request):
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(
                200, json={"results": [{**CITY, "population": 20}, CITY]}
            )
        return httpx.Response(200, json={**FORECAST, "daily": {"time": "broken"}})

    result = await client(handler).weather("San Francisco")
    assert result.error is None
    assert "18.2°C" in result.content
    assert "Daily forecast" not in result.content


async def test_default_keyless_provider_request_and_shaping():
    def handler(request):
        assert request.url == "https://api.keenable.ai/v1/search/public"
        payload = json.loads(request.content)
        assert payload == {
            "query": f"current news as of {date.today().isoformat()}",
            "max_results": 5,
        }
        return keenable_response()

    result = await client(handler).search(" current news ")
    assert result.error is None
    assert len(result.sources) == 1
    assert "Published" not in result.content
    assert "Acquired" not in result.content


@pytest.mark.parametrize("failure", [429, 503, "timeout", "network"])
async def test_transient_failover_order(failure):
    seen = []

    def handler(request):
        seen.append(request.url.host)
        if request.url.host != "api.keenable.ai":
            if failure == "timeout":
                raise httpx.ReadTimeout("secret request details")
            if failure == "network":
                raise httpx.ConnectError("secret request details")
            return httpx.Response(failure)
        return keenable_response()

    result = await client(
        handler, {"RAPID_MLX_BRAVE_API_KEY": "b", "RAPID_MLX_TAVILY_API_KEY": "t"}
    ).search("news")
    assert result.error is None
    assert seen == ["api.tavily.com", "api.search.brave.com", "api.keenable.ai"]


@pytest.mark.parametrize("provider", ["brave", "tavily"])
async def test_keyed_provider_transport_and_absent_key_fallback(provider):
    env = {
        "RAPID_MLX_WEB_SEARCH_PROVIDER": provider,
        f"RAPID_MLX_{provider.upper()}_API_KEY": "  test-secret  ",
    }

    def handler(request):
        if provider == "brave":
            assert request.headers["X-Subscription-Token"] == "test-secret"
            assert request.url.params["count"] == "5"
            return httpx.Response(
                200,
                json={
                    "web": {
                        "results": [
                            {
                                "title": "Story",
                                "url": "https://example.com",
                                "description": "News",
                            }
                        ]
                    }
                },
            )
        payload = json.loads(request.content)
        assert payload["api_key"] == "test-secret"
        assert payload["search_depth"] == "basic"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Story", "url": "https://example.com", "content": "News"}
                ]
            },
        )

    result = await client(handler, env).search("news")
    assert result.error is None and result.sources[0].snippet == "News"
    assert "test-secret" not in str(asdict(result))

    def keyless(request):
        assert request.url.host == "api.keenable.ai"
        return keenable_response()

    result = await client(keyless, {"RAPID_MLX_WEB_SEARCH_PROVIDER": provider}).search(
        "news"
    )
    assert "No " + provider + " API key" in result.content


async def test_auto_select_tavily_before_brave_and_explicit_selection_wins():
    seen = []

    def handler(request):
        seen.append(request.url.host)
        return httpx.Response(200, json={"results": [], "web": {"results": []}})

    env = {"RAPID_MLX_TAVILY_API_KEY": "tavily", "RAPID_MLX_BRAVE_API_KEY": "brave"}
    await client(handler, env).search("news")
    await client(handler, {**env, "RAPID_MLX_WEB_SEARCH_PROVIDER": "brave"}).search(
        "news"
    )
    assert seen == ["api.tavily.com", "api.search.brave.com"]


@pytest.mark.parametrize(
    "status,code",
    [
        (401, "access_denied"),
        (403, "access_denied"),
    ],
)
async def test_keyed_errors_are_typed_not_silently_fallback_or_leaked(status, code):
    seen = []

    def handler(request):
        seen.append(request.url.host)
        return httpx.Response(status, text="test-secret private response")

    result = await client(handler, {"RAPID_MLX_BRAVE_API_KEY": "test-secret"}).search(
        "news"
    )
    assert result.error.code == code
    assert seen == ["api.search.brave.com"]
    assert "test-secret" not in str(asdict(result))


@pytest.mark.parametrize("direct", [False, True], ids=["provider", "direct-page"])
@pytest.mark.parametrize("declared", [True, False], ids=["declared", "streamed"])
async def test_response_byte_cap_and_cleanup(public_dns, direct, declared):
    closed = asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"x" * (web.RESPONSE_CAP // 2)
            yield b"x" * (web.RESPONSE_CAP // 2 + 1)

        async def aclose(self):
            closed.set()

    def handler(request):
        if direct and request.url.host == "api.keenable.ai":
            return httpx.Response(503)
        return httpx.Response(
            200,
            stream=Stream(),
            headers={
                "Content-Type": "text/plain",
                **({"Content-Length": str(web.RESPONSE_CAP + 1)} if declared else {}),
            },
        )

    tool = client(handler)
    result = (
        await tool.fetch("https://example.com") if direct else await tool.search("news")
    )
    assert result.error.code == "response_too_large"
    assert closed.is_set()


async def test_search_output_cap_safe_schemes_dedup_and_snippet_cap():
    rows = [
        {
            "title": "Title" * 100,
            "url": f"https://example.com/{i}",
            "content": "snippet " * 1000,
        }
        for i in range(20)
    ]
    rows[:0] = [{"url": "javascript:alert(1)"}, {"url": "file:///private/file"}]
    result = await client(
        lambda _: httpx.Response(200, json={"results": rows}),
        {"RAPID_MLX_TAVILY_API_KEY": "key"},
    ).search("news")
    assert result.error is None
    assert len(result.sources) <= 5
    assert len(result.content) <= web.OUTPUT_CAP
    assert all(len(row.snippet) <= 240 for row in result.sources)
    assert all(row.url.startswith("https://") for row in result.sources)


@pytest.mark.parametrize(
    "location", ["http://127.0.0.1/", "file:///etc/passwd", "https://evil.example/"]
)
async def test_redirects_are_not_followed(location):
    seen = []

    def handler(request):
        seen.append(request.url.host)
        return httpx.Response(302, headers={"Location": location})

    result = await client(handler).weather("SF")
    assert result.error.code == "provider_error"
    assert seen == ["geocoding-api.open-meteo.com"]


async def test_explicit_endpoint_allowlist():
    tools = client(lambda _: pytest.fail("must not make a request"))
    async with tools._client() as http:
        with pytest.raises(web.ProviderError, match="not allowed"):
            await tools._request(http, "GET", "https://127.0.0.1/")


async def test_hard_deadline_stops_stream_and_closes_transport(monkeypatch):
    monkeypatch.setattr(web, "TIMEOUT", 0.03)
    closed = asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                await asyncio.sleep(0.005)
                yield b" "

        async def aclose(self):
            closed.set()

    result = await client(lambda _: httpx.Response(200, stream=Stream())).weather("SF")
    assert result.error.code == "timeout"
    assert closed.is_set()


@pytest.mark.parametrize("direct", [False, True], ids=["provider", "direct-page"])
async def test_cancel_aborts_inflight_http_and_does_not_fallback(public_dns, direct):
    started, closed = asyncio.Event(), asyncio.Event()
    seen = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b""

        async def aclose(self):
            closed.set()

    def handler(request):
        seen.append(request.url.host)
        if direct and request.url.host == "api.keenable.ai":
            return httpx.Response(503)
        return httpx.Response(
            200, headers={"Content-Type": "text/plain"}, stream=Stream()
        )

    tool = client(handler)
    task = asyncio.create_task(
        tool.fetch("https://example.com") if direct else tool.search("news")
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    assert seen == ["api.keenable.ai", *(["93.184.216.34"] if direct else [])]


async def test_timeout_exception_maps_without_details():
    def handler(_):
        raise httpx.ReadTimeout("secret request body")

    result = await client(handler).weather("SF")
    assert result.error.code == "timeout"
    assert "secret" not in result.content


async def test_invalid_key_never_reaches_transport():
    result = await client(
        lambda _: pytest.fail("no HTTP"),
        {"RAPID_MLX_BRAVE_API_KEY": "secret\r\ninjected"},
    ).search("news")
    assert result.error.code == "configuration"
    assert "secret" not in result.content


def test_relative_week_query_has_concrete_dates():
    assert web.prepared_query("news last week", date(2026, 10, 26)).endswith(
        "(date range: 2026-10-19 through 2026-10-25)"
    )
    assert (
        web.prepared_query("news last week in July", date(2026, 10, 26))
        == "news last week in July"
    )


@pytest.mark.parametrize(
    "payload", [{"web": None}, {"web": []}, {"web": {"results": "wrong"}}, ["wrong"]]
)
async def test_malformed_brave_response_is_typed(payload):
    result = await client(
        lambda _: httpx.Response(200, json=payload), {"RAPID_MLX_BRAVE_API_KEY": "key"}
    ).search("news")
    assert result.error.code == "invalid_response"


async def test_administrative_geocoding_hit_is_not_a_city_forecast():
    hit = {**CITY, "name": "California", "feature_code": "ADM1"}
    result = await client(
        lambda _: httpx.Response(200, json={"results": [hit]})
    ).weather("California")
    assert result.error.code == "needs_clarification"


async def test_provider_cannot_bypass_response_cap_with_compression():
    import gzip

    result = await client(
        lambda _: httpx.Response(
            200,
            headers={"Content-Encoding": "gzip"},
            content=gzip.compress(b"x" * (web.RESPONSE_CAP + 1)),
        )
    ).weather("SF")
    assert result.error.code == "invalid_response"


async def test_unicode_search_result_budget_is_bytes():
    rows = [
        {
            "title": "新聞" * 120,
            "url": f"https://example.com/{i}",
            "content": "新聞" * 120,
        }
        for i in range(5)
    ]
    result = await client(
        lambda _: httpx.Response(200, json={"results": rows}),
        {"RAPID_MLX_TAVILY_API_KEY": "key"},
    ).search("新聞" * 1000)
    assert result.error is None
    assert len(result.content.encode("utf-8")) <= web.OUTPUT_CAP
    assert 0 < len(result.sources) < 5


async def test_zero_keyless_results_do_not_trigger_fallback():
    seen = []

    def handler(request):
        seen.append(request.url.host)
        return httpx.Response(200, json={"results": []})

    result = await client(handler).search("nonexistent")
    assert result.error is None
    assert seen == ["api.keenable.ai"]
    assert "No results found" in result.content


async def test_separator_only_location_is_typed_invalid_arguments():
    def handler(request):
        raise AssertionError("Invalid input must not use the network")

    result = await client(handler).weather(",,,")
    assert result.error.code == "invalid_arguments"


@pytest.mark.parametrize(
    "location,language", [("西安", "zh"), ("서울", "ko"), ("おおさか", "ja")]
)
async def test_geocoding_uses_query_script(location, language):
    def handler(request):
        if request.url.host == "geocoding-api.open-meteo.com":
            assert request.url.params["language"] == language
            return httpx.Response(200, json={"results": [{**CITY, "name": location}]})
        return httpx.Response(200, json=FORECAST)

    result = await client(handler).weather(location)
    assert result.error is None
    assert location in result.content


async def test_keenable_keyed_and_public_fallback_never_leak_key():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        assert request.headers["X-Keenable-Title"] == "rapid-mlx"
        if request.url.path == "/v1/search":
            assert request.headers["Authorization"] == "Bearer secret"
            return httpx.Response(429)
        assert "Authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "results": [
                    {"url": "https://example.com", "title": "T", "description": "D"}
                ]
            },
        )

    result = await client(handler, {"KEENABLE_API_KEY": "secret"}).search("news")
    assert result.error is None
    assert result.sources[0].snippet == "D"
    assert seen == ["/v1/search", "/v1/search/public"]
    assert "secret" not in result.content


async def test_explicit_keyless_keenable_precedes_other_keys():
    def handler(request):
        assert request.url.path == "/v1/search/public"
        return keenable_response()

    result = await client(
        handler,
        {"RAPID_MLX_WEB_SEARCH_PROVIDER": "keenable", "RAPID_MLX_BRAVE_API_KEY": "b"},
    ).search("news")
    assert result.error is None


@pytest.fixture
async def public_dns(monkeypatch):
    loop = asyncio.get_running_loop()
    calls = []

    async def lookup(host, port, **kwargs):
        calls.append(host)
        return [(2, 1, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(loop, "getaddrinfo", lookup)
    return calls


@pytest.mark.parametrize("key", ["", "secret"])
async def test_fetch_keenable_contract_and_cap(public_dns, key):
    from rapid_mlx.chat_runtime.tools.page import PAGE_CAP

    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/v1/fetch" + ("" if key else "/public")
        assert request.url.params["url"] == "https://example.com/story"
        assert request.headers["X-Keenable-Title"] == "rapid-mlx"
        assert request.headers.get("Authorization") == (
            "Bearer secret" if key else None
        )
        return httpx.Response(
            200,
            json={
                "url": "https://example.com/story",
                "title": "Story",
                "content": "content " * 4000,
            },
        )

    result = await client(handler, {"KEENABLE_API_KEY": key}).fetch(
        "https://example.com/story"
    )
    assert result.error is None
    assert result.sources[0].url == "https://example.com/story"
    assert len(result.content) == PAGE_CAP
    assert result.content.endswith("[Page truncated]")


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://user:secret@example.com/",
        "https://example.com:0",
        "https://example.com:65536",
        "https://example.com:bad",
        "https://[fe80::1%en0]/",
        "http://example.com/\nprivate",
    ],
)
async def test_fetch_invalid_urls_before_network(url):
    result = await client(lambda _: pytest.fail("no network")).fetch(url)
    assert result.error.code in {"invalid_arguments", "unsafe_url"}


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "100.64.0.1",
        "192.0.0.9",
        "192.168.1.1",
        "::1",
        "::ffff:127.0.0.1",
        "64:ff9b::7f00:1",
        "64:ff9b:1:7f00:1::",
        "fe80::1",
        "fec0::1",
        "ff00::1",
        "224.1.1.1",
        "0.0.0.0",
    ],
)
async def test_fetch_rejects_any_private_dns_answer(monkeypatch, ip):
    async def lookup(host, port, **kwargs):
        return [(2, 1, 6, "", ("93.184.216.34", port)), (2, 1, 6, "", (ip, port))]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", lookup)
    result = await client(lambda _: pytest.fail("no network")).fetch(
        "https://example.com"
    )
    assert result.error.code == "unsafe_url"


async def test_fetch_direct_pins_ip_host_and_tls_and_extracts_main(public_dns):
    seen = []

    def handler(request):
        seen.append(request.url.host)
        if request.url.host == "api.keenable.ai":
            return httpx.Response(503)
        assert request.url.host == "93.184.216.34"
        assert request.headers["Host"] == "example.com:8443"
        assert request.extensions["sni_hostname"] == "example.com"
        assert "Authorization" not in request.headers
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html"},
            text='<html><head><title>Story</title></head><body><nav>Menu</nav><main><h1>News</h1><p>Hello <a href="/story">world</a></p><script>secret</script></main><footer>Footer</footer></body></html>',
        )

    result = await client(handler, {"KEENABLE_API_KEY": "key"}).fetch(
        "https://example.com:8443/news"
    )
    assert result.error is None
    assert result.sources[0].title == "Story"
    assert (
        "# News" in result.content
        and "[world](https://example.com:8443/story)" in result.content
    )
    assert all(word not in result.content for word in ("Menu", "secret", "Footer"))
    assert seen == ["api.keenable.ai", "93.184.216.34"]


async def test_fetch_redirect_revalidates_dns_and_never_contacts_private(monkeypatch):
    seen = []

    async def lookup(host, port, **kwargs):
        return [
            (
                2,
                1,
                6,
                "",
                ("127.0.0.1" if host == "private.example" else "93.184.216.34", port),
            )
        ]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", lookup)

    def handler(request):
        seen.append(request.url.host)
        if request.url.host == "api.keenable.ai":
            return httpx.Response(503)
        return httpx.Response(
            302, headers={"Location": "http://private.example/metadata"}
        )

    result = await client(handler).fetch("https://example.com")
    assert result.error.code == "unsafe_url"
    assert seen == ["api.keenable.ai", "93.184.216.34"]


async def test_fetch_redirect_same_host_revalidates_and_pins_new_address(monkeypatch):
    count = 0

    async def lookup(host, port, **kwargs):
        nonlocal count
        count += 1
        return [(2, 1, 6, "", ("93.184.216.34" if count < 3 else "1.1.1.1", port))]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", lookup)

    def handler(request):
        if request.url.host == "api.keenable.ai":
            return httpx.Response(503)
        if request.url.path == "/":
            assert request.url.host == "93.184.216.34"
            return httpx.Response(302, headers={"Location": "/story"})
        assert request.url.host == "1.1.1.1"
        return httpx.Response(200, headers={"Content-Type": "text/plain"}, text="News")

    result = await client(handler).fetch("https://example.com")
    assert result.error is None
    assert result.sources[0].url == "https://example.com/story"
    assert count == 3


@pytest.mark.parametrize(
    "headers,code",
    [
        ({"Content-Type": "image/png"}, "unsupported_content"),
        (
            {"Content-Type": "text/plain", "Content-Length": str(web.RESPONSE_CAP + 1)},
            "response_too_large",
        ),
        (
            {"Content-Type": "text/plain", "Content-Encoding": "unsupported"},
            "invalid_response",
        ),
    ],
)
async def test_fetch_direct_response_guards(public_dns, headers, code):
    def handler(request):
        if request.url.host == "api.keenable.ai":
            return httpx.Response(503)
        return httpx.Response(200, headers=headers, text="body")

    result = await client(handler).fetch("https://example.com")
    assert result.error.code == code


async def test_fetch_redirect_limit(public_dns):
    def handler(request):
        if request.url.host == "api.keenable.ai":
            return httpx.Response(503)
        return httpx.Response(302, headers={"Location": "/again"})

    result = await client(handler).fetch("https://example.com")
    assert result.error.code == "redirect_limit"


async def test_fetch_provider_markdown_main_heading_before_cap(public_dns):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "url": "https://example.com/news",
                "title": "News",
                "content": "Navigation link\n" * 2000
                + "# News\nTop story\nArticle contents",
            },
        )

    result = await client(handler).fetch("https://example.com/news")
    assert result.error is None
    assert result.content == "# News\nTop story\nArticle contents"


async def test_search_timeout_budget_leaves_time_for_fallback(monkeypatch):
    monkeypatch.setattr(web, "TIMEOUT", 0.09)
    seen = []

    async def handler(request):
        seen.append(request.url.host)
        if request.url.host != "api.keenable.ai":
            await asyncio.sleep(0.5)
        return keenable_response()

    result = await client(
        handler, {"RAPID_MLX_TAVILY_API_KEY": "t", "RAPID_MLX_BRAVE_API_KEY": "b"}
    ).search("news")
    assert result.error is None
    assert seen == ["api.tavily.com", "api.search.brave.com", "api.keenable.ai"]


@pytest.mark.parametrize(
    "location",
    ["https://example.com:bad/", "https://example.com:65536/", "file:///etc/passwd"],
)
async def test_fetch_redirect_invalid_urls_are_typed(public_dns, location):
    def handler(request):
        if request.url.host == "api.keenable.ai":
            return httpx.Response(503)
        return httpx.Response(302, headers={"Location": location})

    result = await client(handler).fetch("https://example.com")
    assert result.error.code == (
        "network_error" if ":bad/" in location else "unsafe_url"
    )


async def test_fetch_empty_direct_page_is_typed(public_dns):
    def handler(request):
        if request.url.host == "api.keenable.ai":
            return httpx.Response(200, json={"content": ""})
        return httpx.Response(
            200, headers={"Content-Type": "text/html"}, text="<script>ignored</script>"
        )

    result = await client(handler).fetch("https://example.com")
    assert result.error.code == "empty_page"
    assert not result.sources


async def test_fetch_provider_timeout_leaves_direct_fallback_budget(
    public_dns, monkeypatch
):
    monkeypatch.setattr(web, "FETCH_TIMEOUT", 0.08)
    seen = []

    async def handler(request):
        seen.append(request.url.host)
        if request.url.host == "api.keenable.ai":
            await asyncio.sleep(1)
        return httpx.Response(
            200, headers={"Content-Type": "text/plain"}, text="Direct content"
        )

    result = await client(handler).fetch("https://example.com")
    assert result.error is None
    assert result.content == "Direct content"
    assert seen == ["api.keenable.ai", "93.184.216.34"]


async def test_fetch_total_deadline_closes_direct_stream(public_dns, monkeypatch):
    monkeypatch.setattr(web, "FETCH_TIMEOUT", 0.04)
    closed = asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.Event().wait()
            yield b""

        async def aclose(self):
            closed.set()

    def handler(request):
        if request.url.host == "api.keenable.ai":
            return httpx.Response(503)
        return httpx.Response(
            200, headers={"Content-Type": "text/plain"}, stream=Stream()
        )

    result = await client(handler).fetch("https://example.com")
    assert result.error.code == "timeout"
    assert closed.is_set()


@pytest.mark.parametrize(
    "provider, payload, expected",
    [
        (
            "keenable",
            {"published_date": "2026-10-13", "updated_at": "2026-10-14"},
            "Published: 2026-10-13; Updated: 2026-10-14",
        ),
        (
            "brave",
            {"page_age": "2026-10-13T08:00:00", "age": "2 weeks ago"},
            "Published: 2026-10-13T08:00:00",
        ),
        ("tavily", {"publishedDate": "2026-10-13"}, "Published: 2026-10-13"),
        ("keenable", {}, "Date: not provided"),
    ],
)
async def test_search_preserves_provider_dates_without_inventing_recency(
    provider, payload, expected
):
    def handler(request):
        row = {
            "title": "Announcement",
            "url": "https://example.com/2026/10/26/story",
            "snippet": "Published today",
            "description": "Published today",
            "content": "Published today",
            **payload,
        }
        return httpx.Response(
            200,
            json=(
                {"web": {"results": [row]}}
                if provider == "brave"
                else {"results": [row]}
            ),
        )

    result = await client(
        handler,
        {
            "RAPID_MLX_WEB_SEARCH_PROVIDER": provider,
            (
                "KEENABLE_API_KEY"
                if provider == "keenable"
                else f"RAPID_MLX_{provider.upper()}_API_KEY"
            ): "key",
        },
    ).search("latest announcements")
    assert result.error is None
    assert result.sources[0].date_label == expected
    assert expected in result.content


def test_search_snippet_budget_keeps_text_instead_of_image_and_link_urls():
    snippet = "![](" + "https://example.com/" + "image" * 100 + ") "
    snippet += (
        "[Driver Name](https://example.com/profile) won on 2026-10-25. "
        + "detail " * 100
    )
    result = web._snippet(snippet)
    assert result.startswith("Driver Name won on 2026-10-25.")
    assert len(result) == web.SNIPPET_CAP
    assert "https:" not in result


def test_direct_html_table_preserves_headers_cells_and_image_alt():
    from rapid_mlx.chat_runtime.tools.page import _PageParser, page_result

    parser = _PageParser("https://example.com/results")
    parser.feed(
        '<main><h1>Results</h1><table><tr><th>Event</th><th>Date</th><th>Winner</th><th>Time</th></tr><tr><td><a href="/race">Latest race</a></td><td>25 Oct</td><td><img alt="Driver Name" src="/image.webp"></td><td>1:30:00</td></tr></table></main>'
    )
    result = page_result(parser.url, "Results", parser.text())
    assert (
        "| Event | Date | Winner | Time |\n| --- | --- | --- | --- |" in result.content
    )
    assert "| Latest race | 25 Oct | Driver Name | 1:30:00 |" in result.content
    assert "image.webp" not in result.content


def test_long_tables_keep_header_and_tail_without_clipping_columns():
    from rapid_mlx.chat_runtime.tools.page import _bounded_content, page_result

    content = "# Results\n| Event | Date | Winner |\n| --- | --- | --- |\n"
    content += "\n".join(f"| Race {i} | Day {i} | Winner {i} |" for i in range(100))
    result = page_result("https://example.com/results", "Results", content)
    assert "[Table rows omitted; header and final rows retained]" in result.content
    assert "| Race 99 | Day 99 | Winner 99 |" in result.content
    assert result.content.count("| Event | Date | Winner |") == 2
    bounded = _bounded_content(result.content, 250)
    assert bounded.endswith("[Page truncated]")
    assert all(
        line.endswith("|") for line in bounded.splitlines() if line.startswith("|")
    )


def test_fetch_compact_links_survive_content_cap_without_images():
    from rapid_mlx.chat_runtime.tools.page import PAGE_CAP, page_result

    content = (
        "# News\n[![Image 1](https://example.com/huge-image.webp) First story](https://example.com/story)\n"
        + "Details " * 2000
    )
    result = page_result("https://example.com/news", "News", content)
    assert result.content.startswith(
        "# News\n[ First story](https://example.com/story)"
    )
    assert "huge-image" not in result.content
    assert "[Page truncated]" in result.content
    assert (
        "Page links (fetch the article for story details):\n- [First story](https://example.com/story)"
        in result.content
    )
    assert len(result.content) <= PAGE_CAP


def test_markdown_table_images_keep_meaningful_alt_without_inference():
    from rapid_mlx.chat_runtime.tools.page import page_result

    content = "| Event | Winner |\n| --- | --- |\n| First | ![Driver Name](https://example.com/image.webp) |\n| Second | ![](https://example.com/driver-alias.webp) |"
    result = page_result("https://example.com/results", "Results", content)
    assert "| First | Driver Name |" in result.content
    assert "| Second |  |" in result.content
    assert "driver-alias" not in result.content


def test_utf8_runtime_page_cap_keeps_whole_rows_and_followup_links():
    from rapid_mlx.chat_runtime.tools.page import truncate_page

    footer = "\n\nPage links (fetch the article for story details):\n- [Full article](https://example.com/article)"
    content = (
        "# 結果\n| 項目 | 値 |\n| --- | --- |\n"
        + "| 出来事 | 重要な詳細 |\n" * 80
        + footer
    )
    bounded = truncate_page(content, 900)
    assert len(bounded.encode()) <= 900
    assert bounded.endswith(footer)
    assert "[Truncated to remaining context.]" in bounded
    assert all(
        line.endswith("|") for line in bounded.splitlines() if line.startswith("|")
    )
    assert truncate_page(content, 10) == ""


def test_nested_media_links_expose_article_not_video():
    from rapid_mlx.chat_runtime.tools.page import page_result

    result = page_result(
        "https://example.com/news",
        "News",
        "# News\n[![Image 3](/photo.webp)[Video 9](/clip.mp4) Story details](/article)",
    )
    assert result.content.endswith("- [Story details](https://example.com/article)")
    assert "mp4" not in result.content and "photo.webp" not in result.content


def test_fetch_keeps_top_story_url_beside_headline_but_tables_compact():
    from rapid_mlx.chat_runtime.tools.page import page_result

    result = page_result(
        "https://example.com/news",
        "News",
        "# News\n[First story](/article)\n"
        + "Other text. " * 200
        + "\n| Race | Winner |\n| --- | --- |\n| [Recent race](/results) | [Driver Name](/driver) |",
    )
    assert result.content.startswith(
        "# News\n[First story](https://example.com/article)"
    )
    assert "| Recent race | Driver Name |" in result.content
    table_lines = [line for line in result.content.splitlines() if line.startswith("|")]
    assert not any("https:" in line or "](" in line for line in table_lines)
    assert "- [Recent race](https://example.com/results)" in result.content


@pytest.mark.parametrize(
    "query",
    [
        "latest industry news",
        "most recent election results October 2026",
        "current software release 2026",
        "today science headlines Oct 2026",
        "traffic now",
        "latest product status 2026-10",
    ],
)
def test_current_searches_receive_day_anchor(query):
    assert (
        web.anchor_current_query(query, date(2026, 10, 26))
        == query + " as of 2026-10-26"
    )


@pytest.mark.parametrize(
    "query",
    [
        "latest release 2026-10-25",
        "latest release October 25 2026",
        "latest release 25 Oct 2026",
        "current policy in 2025",
        "latest forecast for 2027",
        "latest release in September 2026",
        "most recent results 2026-09",
        "latest release yesterday",
        "latest release last week",
        "latest release on Friday",
        "latest release 10/25",
        "latest release on the 25th",
        "quantum mechanics basics",
        "history of the October revolution",
    ],
)
def test_search_anchor_preserves_explicit_date_scopes_and_noncurrent_queries(query):
    assert web.anchor_current_query(query, date(2026, 10, 26)) == query


def test_search_anchor_is_bounded_and_idempotent():
    query = "latest developments " + "technology " * 100
    anchored = web.anchor_current_query(query, date(2026, 10, 26))
    assert len(anchored) <= 2048 and anchored.endswith("as of 2026-10-26")
    assert web.anchor_current_query(anchored, date(2026, 10, 27)) == anchored
    long_query = "latest " + "x" * 2040
    assert len(long_query) == 2047
    assert web.anchor_current_query(long_query, date(2026, 10, 26)) == long_query
    assert web.prepared_query("latest research", date(2026, 10, 26)).endswith(
        "as of 2026-10-26"
    )


def test_today_results_retain_current_unknown_and_newly_updated_evidence():
    from rapid_mlx.chat_runtime.tools.web import SearchResult

    results = [
        SearchResult("Old", "https://example.com/old", "Old news", "2026-10-23"),
        SearchResult("Current", "https://example.com/current", "News", "2026-10-26"),
        SearchResult("Unknown", "https://example.com/unknown", "News"),
        SearchResult(
            "Updated",
            "https://example.com/updated",
            "News",
            "2026-10-23",
            "2026-10-26T09:00:00Z",
        ),
        SearchResult(
            "Bad update",
            "https://example.com/bad",
            "News",
            "October 23, 2026",
            "unknown",
        ),
    ]
    for query in (
        "news today",
        "news today as of 2026-10-26",
        "latest news today 2026-10-26",
        "news today October 26 2026",
    ):
        kept, filtered = web._today_results(results, query, date(2026, 10, 26))
        assert filtered and [row.title for row in kept] == [
            "Current",
            "Unknown",
            "Updated",
        ]
    for query in (
        "news today in 2025",
        "news today October 23 2026",
        "news today in September",
        "latest news",
        "history",
    ):
        kept, filtered = web._today_results(results, query, date(2026, 10, 26))
        assert not filtered and kept == results


async def test_today_search_discloses_when_only_old_results_remain():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Old news",
                        "url": "https://example.com/old",
                        "snippet": "Old",
                        "published_date": "2000-01-01",
                    }
                ]
            },
        )

    result = await client(handler).search("news today")
    assert result.error is None and result.sources == ()
    assert "No results dated today; older results omitted." in result.content
