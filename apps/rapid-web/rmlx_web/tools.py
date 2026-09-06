# SPDX-License-Identifier: Apache-2.0
"""Tools the model can call, executed here rather than in the browser.

The page owns the tool loop — it is the thing streaming the answer — but the
tools themselves run on the Mac, because a browser cannot fetch an arbitrary
origin: every provider here would need CORS headers it does not send.

The browse transport resolves and validates every destination, then connects
only to the selected IP while preserving the original Host and TLS identity.
That coupling is the SSRF boundary; validation without transport pinning would
leave a DNS-rebinding time-of-check/time-of-use gap.

Two enforcement points, not one. The page filters disabled tools out of the
request body, and :func:`run_tool` refuses anything not in the list it was
told was advertised. Omitting a tool from the body does not stop a malformed
model emitting a call for it, so the second gate is the load-bearing one.
"""

from __future__ import annotations

import asyncio
import html
import ipaddress
import json
import re
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse, urlsplit, urlunsplit

import httpx

# Tools whose result the model reads directly, so every cap here is a context
# budget as much as a transfer one.
WEB_SEARCH_RESULT_CAP = 6
WEB_SEARCH_SNIPPET_CHARS = 240
WEB_SEARCH_TOTAL_CHARS = 4096

BROWSE_CHAR_BUDGET = 15_000
BROWSE_MAX_BYTES = 2 * 1024 * 1024
BROWSE_TIMEOUT = 12.0
BROWSE_MAX_REDIRECTS = 5

RESOLVE_TIMEOUT = 8.0

_BROWSE_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) RapidMLX/1.0 Safari/605.1.15"
)
_SEARCH_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)


class ToolError(RuntimeError):
    """A tool could not run. The message goes back to the model verbatim."""


@dataclass
class ToolResult:
    """One tool's answer, as the ``role: "tool"`` message the page appends."""

    content: str
    is_error: bool = False
    # Set only by ``browse`` when a redirect left every approved origin. The
    # page re-prompts and calls again with the origin added, rather than the
    # server holding a request open waiting for a human.
    needs_approval: dict | None = None

    def to_dict(self) -> dict:
        body: dict[str, Any] = {"content": self.content, "is_error": self.is_error}
        if self.needs_approval is not None:
            body["needs_approval"] = self.needs_approval
        return body


# ----------------------------------------------------------------- definitions

WEATHER_DEFINITION = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": (
            "Get current weather or current temperature for a city or place. Use "
            "this tool—not web_search—for current conditions. Pass the location "
            "from the user's request, preserving country or state/province "
            "qualifiers. This tool does not provide future forecasts, and "
            "ambiguous places are not guessed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "City or place name in the user's language, optionally "
                        "followed by region/country, e.g. '西安', 'Springfield, "
                        "Illinois', or 'Paris, France'."
                    ),
                },
                "country": {
                    "type": "string",
                    "description": (
                        "Optional country name or two-letter country code used "
                        "to disambiguate the place."
                    ),
                },
                "admin1": {
                    "type": "string",
                    "description": (
                        "Optional state, province, or first-level administrative "
                        "region used to disambiguate the place."
                    ),
                },
                "units": {
                    "type": "string",
                    "description": (
                        "Either 'metric' (Celsius, km/h) or 'imperial' "
                        "(Fahrenheit, mph). Defaults to metric."
                    ),
                    "enum": ["metric", "imperial"],
                },
            },
            "required": ["location"],
        },
    },
}

WEB_SEARCH_DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web and get the top results (title + URL + snippet). Use "
            "this for current events, recent news, or facts that may have changed "
            "since training. Do not use it for current weather when the weather "
            "tool is available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query in natural language.",
                }
            },
            "required": ["query"],
        },
    },
}

BROWSE_DEFINITION = {
    "type": "function",
    "function": {
        "name": "browse",
        "description": (
            "Fetch a web page (http/https) and return its readable content as "
            "text. Use this to read articles, documentation, or any URL the user "
            "shares or you find via web_search. Long pages are paginated: the "
            "result includes 'next_offset' and 'has_more' — call browse again "
            "with that 'offset' to read the next part. Fetching requires user "
            "approval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "The absolute http(s) URL to fetch, e.g. "
                        "'https://example.com/article'."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "Character offset for pagination. Omit (or 0) for the "
                        "start of the page; pass the 'next_offset' from a "
                        "previous browse call to continue reading the same URL."
                    ),
                },
            },
            "required": ["url"],
        },
    },
}

DEFINITIONS = [WEB_SEARCH_DEFINITION, BROWSE_DEFINITION, WEATHER_DEFINITION]

# Which tools need the user to approve each call. Only ``browse`` does: the
# MODEL picks the URL, so an unapproved fetch is an exfiltration primitive no
# SSRF check can close — the destination host is public by construction.
APPROVAL_REQUIRED = frozenset({"browse"})

TOOL_NAMES = frozenset(d["function"]["name"] for d in DEFINITIONS)


def definitions_for(enabled: set[str] | None = None) -> list[dict]:
    """The advertised list, filtered to what the caller enabled."""
    if enabled is None:
        return list(DEFINITIONS)
    return [d for d in DEFINITIONS if d["function"]["name"] in enabled]


# ------------------------------------------------------------------- dispatch


def normalize_arguments(raw: str, definition: dict) -> dict | None:
    """Parse the model's argument string and drop keys outside the schema.

    Generic envelope handling only — nested semantics belong to each tool's
    own parsing. ``None`` means the model did not produce a JSON object.
    """
    text = (raw or "").strip()
    try:
        parsed = json.loads(text or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    properties = definition["function"]["parameters"].get("properties")
    if isinstance(properties, dict):
        parsed = {k: v for k, v in parsed.items() if k in properties}
    return parsed


def refusal_message(name: str, advertised: set[str]) -> str:
    if name in TOOL_NAMES:
        return (
            f"tool '{name}' isn't available in this conversation — answer "
            "directly, or ask the user to enable it in Settings."
        )
    listed = ", ".join(sorted(advertised))
    suffix = f" — available: {listed}" if listed else ""
    return f"unknown tool '{name}'{suffix}. Answer directly instead."


async def run_tool(
    client: httpx.AsyncClient,
    *,
    name: str,
    arguments: str,
    advertised: set[str],
    approved_origins: set[str] | None = None,
) -> ToolResult:
    """Execute one call, having first checked it was actually advertised."""
    if name not in advertised:
        return ToolResult(refusal_message(name, advertised), is_error=True)

    definition = next(d for d in DEFINITIONS if d["function"]["name"] == name)
    args = normalize_arguments(arguments, definition)
    if args is None:
        return ToolResult(
            f"tool '{name}' error: arguments must be a JSON object matching the "
            "advertised schema",
            is_error=True,
        )

    try:
        if name == "weather":
            return await run_weather(client, args)
        if name == "web_search":
            return await run_web_search(client, args)
        return await run_browse(client, args, approved_origins or set())
    except ToolError as exc:
        return ToolResult(f"{name} error: {exc}", is_error=True)
    except httpx.HTTPError as exc:
        return ToolResult(f"{name} error: {exc}", is_error=True)


# -------------------------------------------------------------------- weather

_WMO_LABELS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Freezing drizzle",
    57: "Freezing drizzle",
    61: "Light rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Freezing rain",
    67: "Freezing rain",
    71: "Light snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Light rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Snow showers",
    86: "Snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with hail",
}


def weather_code_label(code: int) -> str:
    return _WMO_LABELS.get(code, f"Code {code}")


def _match_key(value: str) -> str:
    import unicodedata

    folded = unicodedata.normalize("NFKD", value)
    stripped = "".join(c for c in folded if not unicodedata.combining(c))
    return "".join(c for c in stripped if c.isalnum()).lower()


def geocoding_language(location: str) -> str | None:
    """Which language to ask the geocoder for.

    Open-Meteo answers in English by default, so a query for 北京 comes back
    as "Beijing" and then matches nothing — the place resolves only if the
    reply is in the same script the user typed.
    """
    for character in location:
        code = ord(character)
        if 0xAC00 <= code <= 0xD7AF:
            return "ko"
        if 0x3040 <= code <= 0x30FF:
            return "ja"
        if 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF:
            return "zh"
    return None


def select_geocoding_hit(
    location: str, qualifiers: list[str], candidates: list[dict]
) -> dict | None:
    """Pick one place, or none. Ambiguity is reported, never guessed.

    Same-named candidates are ranked by population and the leader must beat
    the runner-up 5x, otherwise the name is genuinely ambiguous (the US
    Springfields) and the caller asks the user to qualify it.
    """
    requested = _match_key(location)
    if not requested:
        return None

    wanted = []
    for qualifier in qualifiers:
        key = _match_key(qualifier)
        if key:
            wanted.append(key)

    matches = []
    for candidate in candidates:
        if _match_key(str(candidate.get("name", ""))) != requested:
            continue
        fields = {
            _match_key(str(candidate.get(field) or ""))
            for field in ("admin1", "admin2", "country", "country_code")
        }
        fields.discard("")
        # A qualifier counts when it matches a field either way round: the
        # model writes "北京市" where the geocoder says "北京", and "Illinois"
        # where it says "Illinois, US".
        if all(
            any(key == field or key in field or field in key for field in fields)
            for key in wanted
        ):
            matches.append(candidate)

    # A qualifier the geocoder does not carry must not turn a resolvable
    # place into "ambiguous" — fall back to the bare name.
    if not matches and wanted:
        matches = [
            candidate
            for candidate in candidates
            if _match_key(str(candidate.get("name", ""))) == requested
        ]
    if not matches:
        return None

    ranked = sorted(matches, key=lambda c: c.get("population") or 0, reverse=True)
    first = ranked[0]
    leader = first.get("population") or 0
    if len(ranked) > 1:
        runner_up = ranked[1].get("population") or 0
        if leader < max(1, runner_up) * 5:
            return None

    # An explicit qualifier or an exact spelling settles it. A match that only
    # holds after folding accents needs a major-city floor, so a bare "Xian"
    # does not resolve to a Spanish hamlet.
    if wanted or str(first.get("name", "")).strip().lower() == location.strip().lower():
        return first
    return first if leader >= 100_000 else None


async def run_weather(client: httpx.AsyncClient, args: dict) -> ToolResult:
    location = str(args.get("location") or "").strip()
    if not location:
        raise ToolError("empty location")
    imperial = str(args.get("units") or "").lower() == "imperial"

    parts = [p.strip() for p in location.split(",") if p.strip()]
    name = parts[0] if parts else location
    qualifiers = parts[1:] + [
        str(args[key]).strip()
        for key in ("admin1", "country")
        if isinstance(args.get(key), str) and str(args[key]).strip()
    ]

    params: dict[str, Any] = {"name": name, "count": 10, "format": "json"}
    language = geocoding_language(name)
    if language is not None:
        params["language"] = language

    geo = await client.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params=params,
        timeout=8.0,
    )
    if geo.status_code >= 300:
        raise ToolError(f"geocoding HTTP {geo.status_code}")
    candidates = (geo.json() or {}).get("results") or []
    hit = select_geocoding_hit(name, qualifiers, candidates)
    if hit is None:
        return ToolResult(
            f'weather: could not uniquely identify "{location}". Add a country '
            "or state/province and try again."
        )

    forecast = await client.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": hit["latitude"],
            "longitude": hit["longitude"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "temperature_unit": "fahrenheit" if imperial else "celsius",
            "wind_speed_unit": "mph" if imperial else "kmh",
        },
        timeout=10.0,
    )
    if forecast.status_code >= 300:
        raise ToolError(f"weather HTTP {forecast.status_code}")
    current = (forecast.json() or {}).get("current")
    if not isinstance(current, dict):
        raise ToolError("unrecognised forecast payload")

    label = ", ".join(
        dict.fromkeys(
            part
            for part in (hit.get("name"), hit.get("admin1"), hit.get("country"))
            if isinstance(part, str) and part.strip()
        )
    )
    temp_unit = "°F" if imperial else "°C"
    wind_unit = "mph" if imperial else "km/h"
    lines = [
        f"Current weather for {label or name}:",
        f"  Conditions: {weather_code_label(int(current.get('weather_code', -1)))}",
        f"  Temperature: {float(current.get('temperature_2m', 0)):.1f}{temp_unit}",
        f"  Humidity: {int(current.get('relative_humidity_2m', 0))}%",
        f"  Wind: {float(current.get('wind_speed_10m', 0)):.1f} {wind_unit}",
        f"Data: Open-Meteo (timezone: {hit.get('timezone') or 'n/a'})",
    ]
    return ToolResult("\n".join(lines))


# ----------------------------------------------------------------- web_search

_RESULT_BODY_CLASS = re.compile(r'class="[^"]*\bresult__body\b[^"]*"')
_DDG_ANCHOR = re.compile(
    r'<a[^>]+class="[^"]*\bresult__a\b[^"]*"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_DDG_SNIPPET = re.compile(
    r'<a[^>]+class="[^"]*\bresult__snippet\b[^"]*"[^>]*>(.*?)</a>', re.DOTALL
)
_TAG = re.compile(r"<[^>]*>")


def is_safe_http_url(raw: str) -> bool:
    """DuckDuckGo has been used to smuggle ``javascript:`` into result lists."""
    try:
        return urlparse(raw).scheme.lower() in ("http", "https")
    except ValueError:
        return False


def ddg_redirect_extract(href: str) -> str | None:
    """DDG wraps every href as ``/l/?uddg=<encoded>``; recover the real URL."""
    query = parse_qs(urlsplit(href).query)
    target = (query.get("uddg") or [None])[0]
    if target and is_safe_http_url(target):
        return target
    return None


def _strip_tags(fragment: str) -> str:
    return html.unescape(_TAG.sub("", fragment)).strip()


def parse_ddg_html(page: str, cap: int) -> list[dict]:
    results: list[dict] = []
    snippets = _DDG_SNIPPET.findall(page)
    for index, (href, title) in enumerate(_DDG_ANCHOR.findall(page)):
        if len(results) >= cap:
            break
        url = ddg_redirect_extract(href) or href
        if not is_safe_http_url(url):
            continue
        results.append(
            {
                "title": _strip_tags(title),
                "url": url,
                "snippet": _strip_tags(snippets[index])
                if index < len(snippets)
                else "",
            }
        )
    return results


def is_ddg_throttled(status_code: int, page: str) -> bool:
    """A real results page is never a throttle, whatever the status line says.

    Measured behaviour: the first request of a session is 200 with result
    blocks and every one after it is 202 with a page carrying none. So the
    signature is "202 with a non-results body", not 202 alone.
    """
    if _RESULT_BODY_CLASS.search(page):
        return False
    if status_code in (202, 403, 429):
        return True
    return "anomaly-modal" in page or "cc=botnet" in page


def format_search_output(query: str, results: list[dict]) -> str:
    if not results:
        return f'web_search: no results found for "{query}"'
    bullets = []
    for index, result in enumerate(results, start=1):
        snippet = result["snippet"]
        if len(snippet) > WEB_SEARCH_SNIPPET_CHARS:
            snippet = snippet[:WEB_SEARCH_SNIPPET_CHARS] + "…"
        title = result["title"] or result["url"]
        bullets.append(f"{index}. {title}\n   {result['url']}\n   {snippet}")
    content = f'Web search: "{query}" — {len(results)} results\n\n' + "\n\n".join(
        bullets
    )
    if len(content) > WEB_SEARCH_TOTAL_CHARS:
        content = content[:WEB_SEARCH_TOTAL_CHARS] + "\n…(truncated)"
    return content


async def run_web_search(client: httpx.AsyncClient, args: dict) -> ToolResult:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ToolError("empty query")

    response = await client.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        # Without a UA header DDG redirects to a landing page.
        headers={"User-Agent": _SEARCH_USER_AGENT},
        timeout=15.0,
        follow_redirects=True,
    )
    page = response.text
    if is_ddg_throttled(response.status_code, page):
        # Written as facts about the backend, not as instructions: the older
        # wording left enough room for a small model to conclude it has no web
        # access at all and say so, contradicting the tool it just called.
        return ToolResult(
            "web_search error: the DuckDuckGo backend rate-limited this Mac, so "
            "this query returned no results. The web_search tool is enabled and "
            "working — DuckDuckGo throttles its free endpoint per IP after a few "
            "searches, and usually recovers after a few minutes.",
            is_error=True,
        )
    if response.status_code >= 300:
        raise ToolError(f"DuckDuckGo returned HTTP {response.status_code}")

    return ToolResult(
        format_search_output(query, parse_ddg_html(page, WEB_SEARCH_RESULT_CAP))
    )


# --------------------------------------------------------------------- browse
#
# `browse` turns a model-supplied URL into a GET whose body is fed back to the
# model. Unguarded that is a pivot into the user's private network:
# 169.254.169.254 (cloud metadata), 127.0.0.1 (local admin panels),
# 192.168.x (LAN devices). Every host the fetch would contact — the initial
# URL and every redirect hop — is resolved and range-checked.
#
# The guarantee is bounded. DNS rebinding between our resolution and httpx's
# connect, and a system HTTP proxy resolving on a path we never see, both stay
# open at this layer. They are not left bare: the user approves the exact host
# first, and the action is a read-only GET.

ALLOWED_SCHEMES = frozenset({"http", "https"})

_NAT64_WKP = bytes([0x00, 0x64, 0xFF, 0x9B, 0, 0, 0, 0, 0, 0, 0, 0])
_NAT64_LOCAL = bytes([0x00, 0x64, 0xFF, 0x9B, 0x00, 0x01])


def _blocked_v4(packed: bytes) -> bool:
    if len(packed) != 4:
        return True
    a, b = packed[0], packed[1]
    # 100.64.0.0/10 (CGNAT) is not covered by `is_private` on every version.
    if a == 100 and (b & 0xC0) == 0x40:
        return True
    address = ipaddress.IPv4Address(packed)
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def is_blocked_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True for any loopback / private / link-local / reserved destination."""
    if isinstance(address, ipaddress.IPv4Address):
        return _blocked_v4(address.packed)

    packed = address.packed
    # A v4 address carried inside a v6 word would otherwise skip the v4 checks.
    if address.ipv4_mapped is not None:
        return _blocked_v4(address.ipv4_mapped.packed)
    if packed[:12] == _NAT64_WKP:
        return _blocked_v4(packed[12:16])
    if packed[:6] == _NAT64_LOCAL:
        return _blocked_v4(bytes([packed[6], packed[7], packed[9], packed[10]]))
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def parse_ip_literal(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    bare = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    try:
        return ipaddress.ip_address(bare)
    except ValueError:
        return None


async def resolve_host(
    host: str,
) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Every A/AAAA address for ``host``. Bounded: a stalled resolver would
    otherwise pin the tool call for the OS timeout (~30 s)."""
    loop = asyncio.get_running_loop()
    try:
        infos = await asyncio.wait_for(
            loop.getaddrinfo(host, None, type=socket.SOCK_STREAM),
            timeout=RESOLVE_TIMEOUT,
        )
    except (socket.gaierror, asyncio.TimeoutError):
        raise ToolError(f"could not resolve host '{host}'") from None

    out = []
    seen = set()
    for info in infos:
        try:
            address = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if address not in seen:
            seen.add(address)
            out.append(address)
    return out


async def validate_url(
    url: str,
) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Reject before a socket opens and return the only allowed destinations."""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ToolError(f"scheme '{scheme}' is not allowed (only http/https)")
    host = parsed.hostname
    if not host:
        raise ToolError("URL has no host")

    literal = parse_ip_literal(host)
    if literal is not None:
        if is_blocked_address(literal):
            raise ToolError(
                f"host '{host}' is a private/loopback address ({literal}) and "
                "cannot be browsed"
            )
        return [literal]

    addresses = await resolve_host(host)
    if not addresses:
        raise ToolError(f"could not resolve host '{host}'")
    for address in addresses:
        if is_blocked_address(address):
            raise ToolError(
                f"host '{host}' resolves to a private/loopback address "
                f"({address}) and cannot be browsed"
            )
    return addresses


def origin_of(url: str) -> str:
    """scheme://host:port, with the default port made explicit so a path-only
    redirect counts as same-origin and a host change does not."""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    port = parsed.port or (443 if scheme == "https" else 80)
    return f"{scheme}://{host}:{port}"


_SCRIPT_STYLE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE
)
_BLOCK_END = re.compile(r"</(p|div|h[1-6]|li|tr|section|article|br)\s*>", re.IGNORECASE)
_BLANK_LINES = re.compile(r"\n{3,}")


def html_to_text(page: str) -> tuple[str | None, str]:
    """Title plus a readable text rendering. Not a parser — the result is
    context for a model, and a dependency-free strip is enough for that."""
    match = re.search(r"<title[^>]*>(.*?)</title>", page, re.DOTALL | re.IGNORECASE)
    title = _strip_tags(match.group(1)) if match else None

    body = _SCRIPT_STYLE.sub(" ", page)
    body = _BLOCK_END.sub("\n", body)
    body = _TAG.sub(" ", body)
    body = html.unescape(body)
    lines = [re.sub(r"[ \t\u00a0]+", " ", line).strip() for line in body.split("\n")]
    return title, _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def _authority_for(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    host = host.encode("idna").decode("ascii")
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError as exc:
        raise ToolError("URL has an invalid port") from exc
    return f"{host}:{port}" if port is not None else host


def _pinned_url(
    url: str, address: ipaddress.IPv4Address | ipaddress.IPv6Address
) -> str:
    parsed = urlsplit(url)
    host = (
        f"[{address}]" if isinstance(address, ipaddress.IPv6Address) else str(address)
    )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ToolError("URL has an invalid port") from exc
    authority = f"{host}:{port}" if port is not None else host
    return urlunsplit((parsed.scheme, authority, parsed.path or "/", parsed.query, ""))


async def _fetch_from_address(
    url: str, address: ipaddress.IPv4Address | ipaddress.IPv6Address
) -> tuple[bytes, str]:
    """Fetch one hop without resolving the original hostname again."""
    parsed = urlsplit(url)
    original_host = parsed.hostname or ""
    extensions = (
        {"sni_hostname": original_host.encode("idna").decode("ascii")}
        if parsed.scheme.lower() == "https"
        else None
    )
    # A fresh client prevents a connection pooled by pinned IP from being
    # reused for a different Host/SNI. trust_env=False also prevents a proxy
    # from resolving and connecting to the original hostname outside this gate.
    async with (
        httpx.AsyncClient(trust_env=False, follow_redirects=False) as pinned,
        pinned.stream(
            "GET",
            _pinned_url(url, address),
            headers={
                "Host": _authority_for(url),
                "User-Agent": _BROWSE_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
                "Accept-Encoding": "identity",
            },
            timeout=BROWSE_TIMEOUT,
            extensions=extensions,
            # Followed by hand so every hop is SSRF-checked before it connects.
            follow_redirects=False,
        ) as response,
    ):
        if 300 <= response.status_code < 400:
            return b"", response.headers.get("location", "")
        if response.status_code >= 400:
            raise ToolError(
                f"HTTP {response.status_code} from {urlparse(url).hostname}"
            )
        chunks = bytearray()
        async for chunk in response.aiter_bytes():
            chunks.extend(chunk)
            if len(chunks) > BROWSE_MAX_BYTES:
                raise ToolError(
                    f"page exceeded {BROWSE_MAX_BYTES // (1024 * 1024)} MB cap"
                )
        return bytes(chunks), response.headers.get("content-type", "")


async def _fetch_capped(
    url: str,
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address],
) -> tuple[bytes, str]:
    """Try only validated addresses, bounded by one deadline for the hop."""
    last_error: httpx.TransportError | None = None

    async def attempt_all() -> tuple[bytes, str]:
        nonlocal last_error
        for address in addresses:
            try:
                return await _fetch_from_address(url, address)
            except httpx.TransportError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise ToolError("no validated destination address")

    try:
        return await asyncio.wait_for(attempt_all(), timeout=BROWSE_TIMEOUT)
    except asyncio.TimeoutError:
        raise ToolError(f"request to {urlparse(url).hostname} timed out") from None


async def run_browse(
    client: httpx.AsyncClient, args: dict, approved_origins: set[str]
) -> ToolResult:
    url = str(args.get("url") or "").strip()
    if not url:
        raise ToolError("empty url")
    try:
        offset = max(0, int(args.get("offset") or 0))
    except (TypeError, ValueError):
        offset = 0

    if origin_of(url) not in approved_origins:
        raise ToolError(f"browsing {urlparse(url).hostname or url} was not approved")

    current = url
    content_type = ""
    raw = b""
    for _ in range(BROWSE_MAX_REDIRECTS + 1):
        addresses = await validate_url(current)
        raw, header = await _fetch_capped(current, addresses)
        if raw:
            content_type = header
            break
        if not header:
            raise ToolError("redirect with no destination")
        nxt = str(httpx.URL(current).join(header))
        # A server must not be able to bounce the fetch to a host the user
        # never saw, so a cross-origin hop goes back for its own approval.
        if origin_of(nxt) not in approved_origins:
            return ToolResult(
                f"browse: the redirect to {urlparse(nxt).hostname} needs approval",
                needs_approval={"url": nxt, "host": urlparse(nxt).hostname or nxt},
            )
        current = nxt
    else:
        raise ToolError(f"too many redirects (> {BROWSE_MAX_REDIRECTS})")

    mime = content_type.split(";")[0].strip().lower()
    text = raw.decode("utf-8", errors="replace")
    if "html" in mime or "xml" in mime:
        title, rendered = html_to_text(text)
    elif not mime or mime.startswith("text/") or "json" in mime:
        title, rendered = None, text
    else:
        title, rendered = None, f"[browse: content type '{mime}' is not text]"

    total = len(rendered)
    start = min(offset, total)
    end = min(start + BROWSE_CHAR_BUDGET, total)
    payload: dict[str, Any] = {
        "url": current,
        "content": rendered[start:end],
        "offset": start,
        "total_chars": total,
        "has_more": end < total,
    }
    if title:
        payload["title"] = title
    if current != url:
        payload["final_url"] = current
    if end < total:
        payload["next_offset"] = end
        payload["note"] = (
            f"Showing characters {start}–{end} of {total}. Call browse again "
            f"with offset={end} to continue."
        )
    return ToolResult(json.dumps(payload, sort_keys=True))
