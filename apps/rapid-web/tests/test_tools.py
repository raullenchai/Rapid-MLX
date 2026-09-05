# SPDX-License-Identifier: Apache-2.0
"""Tool execution: the dispatch gate, the SSRF guard, and each tool's parsing."""

from __future__ import annotations

import ipaddress

import httpx
import pytest

from rmlx_web import tools


class TestNormalizeArguments:
    def test_empty_string_is_an_empty_object(self):
        assert tools.normalize_arguments("", tools.WEATHER_DEFINITION) == {}

    def test_drops_keys_outside_the_schema(self):
        parsed = tools.normalize_arguments(
            '{"location": "Paris", "sudo": true}', tools.WEATHER_DEFINITION
        )
        assert parsed == {"location": "Paris"}

    def test_rejects_a_non_object(self):
        assert tools.normalize_arguments("[1, 2]", tools.WEATHER_DEFINITION) is None

    def test_rejects_malformed_json(self):
        assert (
            tools.normalize_arguments('{"location":', tools.WEATHER_DEFINITION) is None
        )


class TestDispatchGate:
    """Omitting a tool from the request body does not stop a malformed model
    emitting a call for it, so this gate is the load-bearing one."""

    @pytest.mark.asyncio
    async def test_refuses_a_tool_that_was_not_advertised(self):
        result = await tools.run_tool(
            httpx.AsyncClient(),
            name="browse",
            arguments="{}",
            advertised={"weather"},
        )
        assert result.is_error
        assert "isn't available" in result.content

    @pytest.mark.asyncio
    async def test_names_the_alternatives_for_an_unknown_tool(self):
        result = await tools.run_tool(
            httpx.AsyncClient(),
            name="rm_rf",
            arguments="{}",
            advertised={"weather"},
        )
        assert result.is_error
        assert "unknown tool 'rm_rf'" in result.content
        assert "weather" in result.content

    @pytest.mark.asyncio
    async def test_refuses_arguments_that_are_not_an_object(self):
        result = await tools.run_tool(
            httpx.AsyncClient(),
            name="weather",
            arguments="not json",
            advertised={"weather"},
        )
        assert result.is_error
        assert "JSON object" in result.content


class TestSSRFGuard:
    @pytest.mark.parametrize(
        "address",
        [
            "127.0.0.1",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            # Cloud metadata — the single most valuable SSRF target.
            "169.254.169.254",
            # CGNAT, which `is_private` alone does not cover.
            "100.64.0.1",
            "0.0.0.0",
            "224.0.0.1",
        ],
    )
    def test_blocks_private_v4(self, address):
        assert tools.is_blocked_address(ipaddress.ip_address(address))

    @pytest.mark.parametrize("address", ["8.8.8.8", "93.184.216.34", "1.1.1.1"])
    def test_allows_public_v4(self, address):
        assert not tools.is_blocked_address(ipaddress.ip_address(address))

    @pytest.mark.parametrize(
        "address", ["::1", "fc00::1", "fd00::1", "fe80::1", "ff02::1"]
    )
    def test_blocks_private_v6(self, address):
        assert tools.is_blocked_address(ipaddress.ip_address(address))

    def test_allows_public_v6(self):
        assert not tools.is_blocked_address(ipaddress.ip_address("2606:4700::1111"))

    def test_blocks_a_v4_mapped_loopback(self):
        # Would otherwise skip the v4 checks entirely by riding inside a v6 word.
        assert tools.is_blocked_address(ipaddress.ip_address("::ffff:127.0.0.1"))

    def test_blocks_a_nat64_encoded_loopback(self):
        assert tools.is_blocked_address(ipaddress.ip_address("64:ff9b::7f00:1"))

    def test_allows_a_nat64_encoded_public_address(self):
        assert not tools.is_blocked_address(ipaddress.ip_address("64:ff9b::808:808"))

    @pytest.mark.asyncio
    async def test_rejects_a_loopback_literal_without_touching_dns(self):
        with pytest.raises(tools.ToolError, match="private/loopback"):
            await tools.validate_url("http://127.0.0.1:8080/admin")

    @pytest.mark.asyncio
    async def test_rejects_a_bracketed_v6_loopback(self):
        with pytest.raises(tools.ToolError, match="private/loopback"):
            await tools.validate_url("http://[::1]/")

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://x/", "ftp://x/"])
    @pytest.mark.asyncio
    async def test_rejects_a_scheme_outside_the_allowlist(self, url):
        with pytest.raises(tools.ToolError, match="not allowed"):
            await tools.validate_url(url)

    @pytest.mark.asyncio
    async def test_rejects_a_url_with_no_host(self):
        with pytest.raises(tools.ToolError, match="no host"):
            await tools.validate_url("http:///path")

    @pytest.mark.asyncio
    async def test_rejects_when_any_dns_answer_is_private(self, monkeypatch):
        async def mixed_answers(host):
            return [
                ipaddress.ip_address("93.184.216.34"),
                ipaddress.ip_address("127.0.0.1"),
            ]

        monkeypatch.setattr(tools, "resolve_host", mixed_answers)
        with pytest.raises(tools.ToolError, match="private/loopback"):
            await tools.validate_url("https://mixed.example/")

    @pytest.mark.asyncio
    async def test_fetch_pins_ip_but_preserves_host_and_tls_name(self, monkeypatch):
        captured = {}

        class Response:
            status_code = 200
            headers = {"content-type": "text/plain"}

            async def aiter_bytes(self):
                yield b"pinned"

        class Stream:
            async def __aenter__(self):
                return Response()

            async def __aexit__(self, *args):
                return None

        def fake_stream(self, method, url, **kwargs):
            captured.update(
                method=method,
                url=str(url),
                headers=kwargs["headers"],
                extensions=kwargs["extensions"],
                trust_env=self._trust_env,
            )
            return Stream()

        monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
        body, content_type = await tools._fetch_from_address(
            "https://rebind.example:8443/path?q=1",
            ipaddress.ip_address("93.184.216.34"),
        )

        assert body == b"pinned"
        assert content_type == "text/plain"
        assert captured["url"] == "https://93.184.216.34:8443/path?q=1"
        assert captured["headers"]["Host"] == "rebind.example:8443"
        assert captured["extensions"] == {"sni_hostname": "rebind.example"}
        assert captured["trust_env"] is False

    @pytest.mark.asyncio
    async def test_public_first_private_second_rebinding_never_reresolves(
        self, monkeypatch
    ):
        resolutions = 0
        connected = []

        async def rebinding_answers(host):
            nonlocal resolutions
            resolutions += 1
            if resolutions == 1:
                return [ipaddress.ip_address("93.184.216.34")]
            return [ipaddress.ip_address("127.0.0.1")]

        async def fake_fetch(url, address):
            connected.append(address)
            return b"safe", "text/plain"

        monkeypatch.setattr(tools, "resolve_host", rebinding_answers)
        monkeypatch.setattr(tools, "_fetch_from_address", fake_fetch)

        async with httpx.AsyncClient() as client:
            result = await tools.run_browse(
                client,
                {"url": "https://rebind.example/page"},
                {"https://rebind.example:443"},
            )

        assert not result.is_error
        assert resolutions == 1
        assert connected == [ipaddress.ip_address("93.184.216.34")]

    @pytest.mark.asyncio
    async def test_redirect_hop_is_resolved_and_checked_again(self, monkeypatch):
        resolutions = 0
        connected = []

        async def rebinding_answers(host):
            nonlocal resolutions
            resolutions += 1
            if resolutions == 1:
                return [ipaddress.ip_address("93.184.216.34")]
            return [ipaddress.ip_address("127.0.0.1")]

        async def redirecting_fetch(url, address):
            connected.append(address)
            return b"", "/second-hop"

        monkeypatch.setattr(tools, "resolve_host", rebinding_answers)
        monkeypatch.setattr(tools, "_fetch_from_address", redirecting_fetch)

        with pytest.raises(tools.ToolError, match="private/loopback"):
            async with httpx.AsyncClient() as client:
                await tools.run_browse(
                    client,
                    {"url": "https://redirect.example/first-hop"},
                    {"https://redirect.example:443"},
                )

        assert resolutions == 2
        assert connected == [ipaddress.ip_address("93.184.216.34")]


class TestOrigin:
    def test_makes_the_default_port_explicit(self):
        assert tools.origin_of("http://a.example/x") == "http://a.example:80"
        assert tools.origin_of("http://a.example:80/y") == "http://a.example:80"
        assert tools.origin_of("https://a.example/x") == "https://a.example:443"

    def test_a_path_change_is_the_same_origin(self):
        assert tools.origin_of("https://a.example/one") == tools.origin_of(
            "https://a.example/two"
        )

    def test_a_host_change_is_not(self):
        assert tools.origin_of("https://a.example/") != tools.origin_of(
            "https://b.example/"
        )


class TestBrowseApproval:
    @pytest.mark.asyncio
    async def test_refuses_a_url_whose_origin_was_not_approved(self):
        result = await tools.run_tool(
            httpx.AsyncClient(),
            name="browse",
            arguments='{"url": "https://example.com/a"}',
            advertised={"browse"},
            approved_origins=set(),
        )
        assert result.is_error
        assert "not approved" in result.content


class TestGeocodingSelection:
    def test_picks_the_dominant_city(self):
        hit = tools.select_geocoding_hit(
            "Tokyo",
            [],
            [
                {"name": "Tokyo", "population": 8_336_599, "country": "Japan"},
                {"name": "Tokyo", "population": 500, "country": "United States"},
            ],
        )
        assert hit is not None
        assert hit["country"] == "Japan"

    def test_reports_ambiguity_rather_than_guessing(self):
        # Without a decisive margin the name is genuinely ambiguous, and
        # falling through to API order would answer for the wrong place.
        assert (
            tools.select_geocoding_hit(
                "Springfield",
                [],
                [
                    {"name": "Springfield", "population": 120_000},
                    {"name": "Springfield", "population": 117_000},
                ],
            )
            is None
        )

    def test_a_qualifier_settles_an_ambiguous_name(self):
        hit = tools.select_geocoding_hit(
            "Springfield",
            ["Illinois"],
            [
                {"name": "Springfield", "population": 120_000, "admin1": "Missouri"},
                {"name": "Springfield", "population": 117_000, "admin1": "Illinois"},
            ],
        )
        assert hit is not None
        assert hit["admin1"] == "Illinois"

    def test_a_fold_only_match_needs_a_population_floor(self):
        # "Xian" folds onto a tiny Spanish hamlet; answering with its weather
        # would be worse than saying the name was ambiguous.
        assert (
            tools.select_geocoding_hit(
                "Xian", [], [{"name": "Xián", "population": 300}]
            )
            is None
        )

    def test_a_fold_only_match_resolves_for_a_major_city(self):
        hit = tools.select_geocoding_hit(
            "Medellin", [], [{"name": "Medellín", "population": 2_000_000}]
        )
        assert hit is not None

    def test_no_candidates_is_no_hit(self):
        assert tools.select_geocoding_hit("Nowhere", [], []) is None


class TestWeatherCodes:
    def test_labels_a_known_code(self):
        assert tools.weather_code_label(0) == "Clear sky"
        assert tools.weather_code_label(95) == "Thunderstorm"

    def test_falls_back_to_the_number(self):
        assert tools.weather_code_label(1234) == "Code 1234"


class TestSearchParsing:
    def test_unwraps_a_duckduckgo_redirect(self):
        assert (
            tools.ddg_redirect_extract("/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=x")
            == "https://example.com/a"
        )

    def test_refuses_a_smuggled_javascript_url(self):
        # DDG's HTML surface has been used to smuggle these into result lists,
        # and the model would surface one for the user to click.
        assert tools.ddg_redirect_extract("/l/?uddg=javascript%3Aalert(1)") is None

    @pytest.mark.parametrize(
        "raw", ["javascript:alert(1)", "data:text/html,x", "file:///x"]
    )
    def test_rejects_unsafe_schemes(self, raw):
        assert not tools.is_safe_http_url(raw)

    def test_parses_a_results_page(self):
        page = """
        <div class="links_main links_deep result__body">
          <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fa">A &amp; B</a>
          <a class="result__snippet">Some <b>text</b>.</a>
        </div>
        """
        results = tools.parse_ddg_html(page, cap=6)
        assert results == [
            {"title": "A & B", "url": "https://example.com/a", "snippet": "Some text."}
        ]

    def test_honours_the_cap(self):
        block = (
            '<div class="result__body">'
            '<a class="result__a" href="https://example.com/{i}">T{i}</a>'
            '<a class="result__snippet">S</a></div>'
        )
        page = "".join(block.format(i=i) for i in range(20))
        assert len(tools.parse_ddg_html(page, cap=6)) == 6

    def test_a_results_page_is_never_a_throttle(self):
        # The signature is "202 with a NON-results body", not 202 alone.
        page = '<div class="links_main result__body">hits</div>'
        assert not tools.is_ddg_throttled(202, page)

    @pytest.mark.parametrize("status", [202, 403, 429])
    def test_a_bodiless_response_is_a_throttle(self, status):
        assert tools.is_ddg_throttled(status, "<html>nothing here</html>")

    def test_detects_the_anti_bot_modal_on_a_200(self):
        assert tools.is_ddg_throttled(200, '<div class="anomaly-modal__title">x</div>')

    def test_truncates_past_the_output_budget(self):
        results = [
            {"title": "T", "url": "https://example.com", "snippet": "x" * 500}
            for _ in range(50)
        ]
        assert len(tools.format_search_output("q", results)) <= (
            tools.WEB_SEARCH_TOTAL_CHARS + len("\n…(truncated)")
        )

    def test_caps_a_long_snippet(self):
        output = tools.format_search_output(
            "q", [{"title": "T", "url": "https://example.com", "snippet": "x" * 900}]
        )
        assert "…" in output
        assert "x" * 900 not in output

    def test_says_so_when_there_are_no_results(self):
        assert "no results found" in tools.format_search_output("q", [])


class TestHtmlToText:
    def test_pulls_out_the_title(self):
        title, _ = tools.html_to_text("<html><title>Hello</title><body>x</body></html>")
        assert title == "Hello"

    def test_drops_script_and_style_bodies(self):
        _, text = tools.html_to_text(
            "<p>keep</p><script>secret()</script><style>.a{}</style>"
        )
        assert "keep" in text
        assert "secret" not in text
        assert ".a{}" not in text

    def test_decodes_entities(self):
        _, text = tools.html_to_text("<p>A &amp; B</p>")
        assert "A & B" in text


class TestDefinitions:
    def test_every_definition_is_wire_shaped(self):
        for definition in tools.DEFINITIONS:
            assert definition["type"] == "function"
            function = definition["function"]
            assert function["name"] and function["description"]
            assert function["parameters"]["type"] == "object"

    def test_only_browse_requires_approval(self):
        # weather and web_search read public data with no destination the
        # model chose; browse fetches a URL the model picked.
        assert set(tools.APPROVAL_REQUIRED) == {"browse"}

    def test_filters_to_the_enabled_subset(self):
        names = [d["function"]["name"] for d in tools.definitions_for({"weather"})]
        assert names == ["weather"]

    def test_no_filter_advertises_everything(self):
        assert len(tools.definitions_for()) == len(tools.DEFINITIONS)
