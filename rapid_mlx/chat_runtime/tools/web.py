"""Bounded read-only live information clients used by the chat owner.

Provider calls use fixed HTTPS endpoints; explicit page fetches additionally
validate DNS and pin direct connections to public IP addresses. Errors deliberately exclude response bodies,
request headers, and exception strings, which may contain provider credentials.
"""

import asyncio
import json
import math
import os
import re
import unicodedata
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from html import unescape
from urllib.parse import urlsplit

import httpx

TIMEOUT = 10.0
FETCH_TIMEOUT = 20.0
RESPONSE_CAP = 1_048_576
RESULT_CAP = 5
SNIPPET_CAP = 240
OUTPUT_CAP = 4096
HOSTS = frozenset(
    {
        "geocoding-api.open-meteo.com",
        "api.open-meteo.com",
        "api.keenable.ai",
        "api.search.brave.com",
        "api.tavily.com",
    }
)


@dataclass(frozen=True)
class ToolError:
    code: str
    message: str


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    published: str = ""
    updated: str = ""

    @property
    def date_label(self) -> str:
        dates = []
        if self.published:
            dates.append(f"Published: {self.published}")
        if self.updated:
            dates.append(f"Updated: {self.updated}")
        return "; ".join(dates) or "Date: not provided"


@dataclass(frozen=True)
class WebResult:
    content: str
    sources: tuple[SearchResult, ...] = ()
    error: ToolError | None = None


class ProviderError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False):
        self.retryable = retryable or code == "rate_limited"
        self.error = ToolError(code, message)
        super().__init__(message)


async def response_body(response, *, page=False) -> bytes:
    """Apply the same compression and byte limits to provider and direct reads."""
    subject = "page" if page else "provider"
    size_subject = "page" if page else "provider response"
    if response.headers.get("content-encoding", "identity").lower() != "identity":
        raise ProviderError(
            "invalid_response",
            f"The {subject} ignored the uncompressed response requirement.",
        )
    too_large = ProviderError(
        "response_too_large", f"The {size_subject} exceeded the size limit."
    )
    length = response.headers.get("content-length", "")
    if length.isdecimal() and int(length) > RESPONSE_CAP:
        raise too_large
    data = bytearray()
    async for chunk in response.aiter_bytes():
        if len(data) + len(chunk) > RESPONSE_CAP:
            raise too_large
        data.extend(chunk)
    return bytes(data)


def image_labels(value):
    """Keep meaningful Markdown image labels without spending evidence on URLs."""
    return re.sub(
        r"!\[([^\]]*)\]\([^)]*\)",
        lambda match: (
            "" if re.fullmatch(r"(?:Image(?: \d+)?)?", match[1], re.I) else match[1]
        ),
        value,
    )


def _clean(value: str) -> str:
    return " ".join(
        "".join(
            c for c in unescape(value) if not unicodedata.category(c).startswith("C")
        ).split()
    )


def _snippet(value: str) -> str:
    # Image URLs and inline link destinations consumed the entire F1 snippet
    # budget. Preserve their visible labels instead of splitting a long URL.
    value = image_labels(value)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    return _clean(re.sub(r"<[^>]*>", "", value))[:SNIPPET_CAP]


def _safe_url(value: str) -> bool:
    if any(ord(c) < 33 or ord(c) == 127 for c in value):
        return False
    try:
        parsed = urlsplit(value)
        return bool(
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and len(value) <= 2048
        )
    except ValueError:
        return False


def _json(data: bytes) -> dict:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return value


def _date_metadata(row: dict, *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return _clean(value)[:80]
    return ""


def _rows(value: object, snippet: str) -> list[SearchResult]:
    if not isinstance(value, list):
        raise ValueError("expected results")
    results = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("expected result object")
        url = row.get("url", "")
        title = row.get("title") or ""
        body = row.get(snippet) or ""
        if not all(isinstance(item, str) for item in (url, title, body)):
            raise ValueError("expected result text")
        if _safe_url(url):
            # Keep provider dates as evidence; never infer publication from URL
            # paths or retrieval time. Relative provider ages retain their label.

            results.append(
                SearchResult(
                    title,
                    url,
                    body,
                    _date_metadata(
                        row,
                        "published_date",
                        "publishedDate",
                        "published_at",
                        "datePublished",
                        "published",
                        "date",
                        "page_age",
                        "age",
                    ),
                    _date_metadata(
                        row,
                        "updated_date",
                        "updatedDate",
                        "updated_at",
                        "dateModified",
                        "modified",
                        "updated",
                    ),
                )
            )
    return results[:RESULT_CAP]


def _check_key(key: str) -> None:
    if any(ord(c) < 32 or ord(c) > 126 for c in key):
        raise ProviderError(
            "configuration", "The search API key contains invalid characters."
        )


def _keenable_headers(key: str) -> dict[str, str]:
    return {
        "X-Keenable-Title": "rapid-mlx",
        **({"Authorization": f"Bearer {key}"} if key else {}),
    }


def _can_anchor_current_query(query: str, today: date) -> bool:
    if not re.search(r"\b(?:latest|most recent|today|current|now)\b", query, re.I):
        return False
    # Keep user/model date scopes, including relative days and calendar periods.
    if re.search(
        r"\b(?:yesterday|tomorrow|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"
        r"|\b(?:last|next|previous)\s+(?:day|week|month|year)\b"
        r"|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"
        r"|\b\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?\b"
        r"|\b\d{1,2}(?:st|nd|rd|th)\b",
        query,
        re.I,
    ):
        return False
    years = re.findall(r"\b(?:19|20)\d{2}\b", query)
    if any(int(year) != today.year for year in years):
        return False
    months = (
        "jan(?:uary)?",
        "feb(?:ruary)?",
        "mar(?:ch)?",
        "apr(?:il)?",
        "may",
        "jun(?:e)?",
        "jul(?:y)?",
        "aug(?:ust)?",
        "sep(?:t(?:ember)?)?",
        "oct(?:ober)?",
        "nov(?:ember)?",
        "dec(?:ember)?",
    )
    for number, month in enumerate(months, 1):
        if not re.search(rf"\b{month}\b", query, re.I):
            continue
        if number != today.month or re.search(
            rf"\b{month}\.?\s+\d{{1,2}}\b|\b\d{{1,2}}\s+{month}\b", query, re.I
        ):
            return False
    for year, month in re.findall(r"\b(\d{4})[-/](\d{1,2})\b", query):
        if (int(year), int(month)) != (today.year, today.month):
            return False
    return True


def anchor_current_query(query: str, today: date | None = None) -> str:
    """Give explicitly current searches a day without rewriting dated research."""
    today = today or date.today()
    query = query.strip()
    if not _can_anchor_current_query(query, today):
        return query
    suffix = f" as of {today.isoformat()}"
    return query + suffix if len(query) + len(suffix) <= 2048 else query


def _provider_day(value: str) -> date | None:
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        pass
    for pattern in (
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
        "%d %B %Y",
        "%d %b %Y",
    ):
        try:
            return datetime.strptime(value.strip(), pattern).date()
        except ValueError:
            pass
    return None


def _today_results(
    results: list[SearchResult], query: str, today: date | None = None
) -> tuple[list[SearchResult], bool]:
    """Drop known-old publication evidence only for unambiguous today searches."""
    today = today or date.today()
    if not re.search(r"\btoday\b", query, re.I):
        return results, False
    # The auto-added/current explicit day agrees with today; all other date
    # scopes still pass through the same conservative historical-scope checks.
    scope = query.replace(today.isoformat(), "")
    month = rf"(?:{today:%B}|{today:%b})"
    scope = re.sub(
        rf"\b{month}\.?\s+0?{today.day}\b|\b0?{today.day}\s+{month}\b",
        "",
        scope,
        flags=re.I,
    )
    scope = re.sub(rf"\b0?{today.month}/0?{today.day}\b", "", scope)
    if not _can_anchor_current_query(scope, today):
        return results, False
    kept = []
    for row in results:
        observed = _provider_day(row.updated) or _provider_day(row.published)
        if observed is None or observed >= today:
            kept.append(row)
    return kept, len(kept) != len(results)


def prepared_query(query: str, today: date | None = None) -> str:
    today = today or date.today()
    # An explicitly anchored historical week must not be shifted to this year.
    if re.search(r"\blast week\s+(?:in|of)\b", query, re.I):
        return query
    if re.search(r"\blast week\b|(?<!上)(?:上周|上一周)(?!末)", query, re.I):
        start = today - timedelta(days=today.weekday() + 7)
        return f"{query} (date range: {start} through {start + timedelta(days=6)})"
    return anchor_current_query(query, today)


class WebTools:
    def __init__(
        self,
        transport: httpx.AsyncBaseTransport | None = None,
        environ: Mapping[str, str] | None = None,
    ):
        self.transport = transport
        self.environ = os.environ if environ is None else environ

    async def _request(
        self, client: httpx.AsyncClient, method: str, url: str, **kwargs
    ) -> bytes:
        target = httpx.URL(url)
        if (
            target.scheme != "https"
            or target.host not in HOSTS
            or target.port not in {None, 443}
        ):
            raise ProviderError(
                "unsafe_endpoint", "The provider endpoint is not allowed."
            )
        # No redirects, including to another provider: credentials cannot cross hosts.
        async with client.stream(method, url, **kwargs) as response:
            status = response.status_code
            if status in {401, 403}:
                raise ProviderError(
                    "access_denied",
                    "The search provider rejected access; check its API key or try later.",
                )
            if status in {402, 429}:
                raise ProviderError(
                    "rate_limited",
                    "The provider's rate or credit limit was reached; try later.",
                )
            if not 200 <= status < 300:
                raise ProviderError(
                    "provider_error",
                    f"The provider returned HTTP {status}.",
                    retryable=status >= 500,
                )
            return await response_body(response)

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=self.transport,
            timeout=TIMEOUT,
            follow_redirects=False,
            trust_env=False,
            headers={
                "User-Agent": "Mozilla/5.0 Rapid-MLX/1.0",
                "Accept-Encoding": "identity",
                "Accept": "application/json",
            },
        )

    async def _guard(
        self,
        name: str,
        operation: Callable[[httpx.AsyncClient], Awaitable[WebResult]],
        timeout: float | None = None,
    ) -> WebResult:
        try:
            # Total wall-clock budget covers geocoding plus forecast, or all fallback hops.
            async with (
                asyncio.timeout(TIMEOUT if timeout is None else timeout),  # type: ignore[attr-defined]  # Harness requires Python 3.11; base mypy target is 3.10.
                self._client() as client,
            ):
                return await operation(client)
        except (TimeoutError, httpx.TimeoutException):
            error = ToolError(
                "timeout", "The live information request timed out; try again later."
            )
        except ProviderError as exc:
            error = exc.error
        except httpx.HTTPError:
            error = ToolError(
                "network_error", "The live information provider could not be reached."
            )
        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            UnicodeError,
            OverflowError,
        ):
            error = ToolError(
                "invalid_response",
                "The live information provider returned an invalid response.",
            )
        # CancelledError is intentionally not caught: closing the context cancels HTTP.
        return WebResult(f"{name} error [{error.code}]: {error.message}", error=error)

    async def search(self, query: str) -> WebResult:
        query = query.strip()
        if not query or len(query) > 2048:
            return WebResult(
                "web_search error: provide a query of 1–2048 characters.",
                error=ToolError("invalid_arguments", "Invalid search query."),
            )
        return await self._guard(
            "web_search", lambda client: self._search(client, prepared_query(query))
        )

    async def _search(self, client: httpx.AsyncClient, query: str) -> WebResult:
        keys = {
            "keenable": self.environ.get("KEENABLE_API_KEY", "").strip(),
            "tavily": self.environ.get("RAPID_MLX_TAVILY_API_KEY", "").strip(),
            "brave": self.environ.get("RAPID_MLX_BRAVE_API_KEY", "").strip(),
        }
        explicit = self.environ.get("RAPID_MLX_WEB_SEARCH_PROVIDER", "").strip().lower()
        if explicit and explicit not in keys:
            raise ProviderError(
                "configuration", "Unsupported RAPID_MLX_WEB_SEARCH_PROVIDER."
            )
        candidates = []
        note = ""
        if explicit:
            if explicit == "keenable" or keys[explicit]:
                candidates.append((explicit, keys[explicit]))
            else:
                note = (
                    f"No {explicit} API key is set; using the next available provider."
                )
        candidates.extend((name, key) for name, key in keys.items() if key)
        candidates.append(("keenable", ""))
        candidates = list(dict.fromkeys(candidates))
        for index, (provider, key) in enumerate(candidates):
            _check_key(key)
            try:
                # Each attempt gets a share of the total deadline, leaving time
                # for the remaining providers even when a connection stalls.
                async with asyncio.timeout(TIMEOUT / len(candidates)):  # type: ignore[attr-defined]  # Python 3.11-only harness.
                    results = await self._search_provider(client, provider, key, query)
                break
            except ProviderError as exc:
                if not exc.retryable or index == len(candidates) - 1:
                    raise
            except (httpx.TransportError, TimeoutError):
                if index == len(candidates) - 1:
                    raise
        results, filtered_today = _today_results(results, query)
        content = f'Web search via {provider}: "{_clean(query)[:256]}"\n'
        if note:
            content += note.strip() + "\n"
        sources: list[SearchResult] = []
        seen = set()
        for row in results[:RESULT_CAP]:
            if row.url in seen:
                continue
            seen.add(row.url)
            source = SearchResult(
                _clean(row.title)[:240] or row.url,
                row.url,
                _snippet(row.snippet),
                row.published,
                row.updated,
            )
            block = f"\n{len(sources) + 1}. {source.title}\n   {source.url}\n   {source.date_label}\n   {source.snippet}\n"
            if len((content + block).encode("utf-8")) > OUTPUT_CAP:
                break
            sources.append(source)
            content += block
        if not sources:
            content += (
                "No results dated today; older results omitted."
                if filtered_today
                else "No results found."
            )
        return WebResult(content, tuple(sources))

    async def _search_provider(self, client, provider, key, query):
        if provider == "brave":
            data = await self._request(
                client,
                "GET",
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": RESULT_CAP},
                headers={"X-Subscription-Token": key},
            )
            return _rows(_json(data).get("web", {}).get("results", []), "description")
        if provider == "tavily":
            data = await self._request(
                client,
                "POST",
                "https://api.tavily.com/search",
                json={
                    "api_key": key,
                    "query": query,
                    "max_results": RESULT_CAP,
                    "search_depth": "basic",
                },
            )
            return _rows(_json(data).get("results"), "content")
        data = await self._request(
            client,
            "POST",
            "https://api.keenable.ai/v1/search" + ("" if key else "/public"),
            headers=_keenable_headers(key),
            json={"query": query, "max_results": RESULT_CAP},
        )
        payload = _json(data).get("results")
        if isinstance(payload, list):
            payload = [
                (
                    {
                        **row,
                        "snippet": row.get("snippet") or row.get("description") or "",
                    }
                    if isinstance(row, dict)
                    else row
                )
                for row in payload
            ]
        return _rows(payload, "snippet")

    async def fetch(self, url: str) -> WebResult:
        from .page import validate_url

        if not _safe_url(url):
            return WebResult(
                "web_fetch error: provide a public HTTP or HTTPS URL.",
                error=ToolError("invalid_arguments", "Invalid page URL."),
            )
        try:
            validate_url(url)
        except ProviderError as exc:
            return WebResult(
                f"web_fetch error [{exc.error.code}]: {exc.error.message}",
                error=exc.error,
            )
        return await self._guard(
            "web_fetch", lambda client: self._fetch(client, url), timeout=FETCH_TIMEOUT
        )

    async def _fetch(self, client, url):
        from .page import direct_fetch, page_result, resolve_public

        # Reject private targets before sending even the URL to a remote service.
        await resolve_public(url)
        key = self.environ.get("KEENABLE_API_KEY", "").strip()
        _check_key(key)
        try:
            async with asyncio.timeout(FETCH_TIMEOUT / 2):
                data = _json(
                    await self._request(
                        client,
                        "GET",
                        "https://api.keenable.ai/v1/fetch" + ("" if key else "/public"),
                        params={"url": url},
                        headers=_keenable_headers(key),
                    )
                )
                final_url = data.get("url") or url
                title, content = data.get("title") or "", data.get("content")
                if (
                    not all(
                        isinstance(value, str) for value in (final_url, title, content)
                    )
                    or not content.strip()
                ):
                    raise ValueError("invalid page")
                await resolve_public(final_url)
                # Provider Markdown may retain site navigation. Like the direct
                # HTML main-region extractor, prefer an explicit document H1.
                heading = re.search(r"(?m)^# [^\n]+", content)
                if heading:
                    content = content[heading.start() :]
                return page_result(final_url, title, content)
        except (ProviderError, httpx.HTTPError, TimeoutError, ValueError, TypeError):
            return await direct_fetch(url, self.transport)

    async def weather(
        self,
        location: str,
        country: str | None = None,
        admin1: str | None = None,
        days: int = 3,
        units: str = "metric",
    ) -> WebResult:
        if (
            not location.strip(" ,，")
            or len(location) > 256
            or units not in {"metric", "imperial"}
            or any(value and len(value) > 256 for value in (country, admin1))
        ):
            return WebResult(
                "weather error: provide a city and metric or imperial units.",
                error=ToolError("invalid_arguments", "Invalid weather arguments."),
            )
        return await self._guard(
            "weather",
            lambda client: self._weather(
                client, location, country, admin1, max(1, min(7, days)), units
            ),
        )

    async def _weather(self, client, location, country, admin1, days, units):
        parts = [part.strip() for part in location.split(",") if part.strip()]
        name = parts[0]
        # Common conversational abbreviation; preserve all explicit qualifiers.
        if name.casefold() == "sf":
            name = "San Francisco"
        params = {"name": name, "count": 5, "format": "json"}
        # Ask for the same script as the query, as the Desktop geocoder does.
        for character in name:
            codepoint = ord(character)
            if 0xAC00 <= codepoint <= 0xD7AF:
                params["language"] = "ko"
                break
            if 0x3040 <= codepoint <= 0x30FF:
                params["language"] = "ja"
                break
            if 0x3400 <= codepoint <= 0x4DBF or 0x4E00 <= codepoint <= 0x9FFF:
                params["language"] = "zh"
                break
        data = _json(
            await self._request(
                client,
                "GET",
                "https://geocoding-api.open-meteo.com/v1/search",
                params=params,
            )
        )
        hit = _select_place(name, parts[1:], country, admin1, data.get("results", []))
        lat, lon = float(hit["latitude"]), float(hit["longitude"])
        if (
            not math.isfinite(lat)
            or not math.isfinite(lon)
            or not -90 <= lat <= 90
            or not -180 <= lon <= 180
        ):
            raise ValueError("invalid coordinates")
        imperial = units == "imperial"
        forecast = _json(
            await self._request(
                client,
                "GET",
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum",
                    "timezone": "auto",
                    "forecast_days": days,
                    "precipitation_unit": "inch" if imperial else "mm",
                    "temperature_unit": "fahrenheit" if imperial else "celsius",
                    "wind_speed_unit": "mph" if imperial else "kmh",
                },
            )
        )
        return WebResult(_forecast_report(hit, forecast, imperial))


def _match(value: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("expected place name")
    return "".join(
        c for c in unicodedata.normalize("NFKD", value.casefold()) if c.isalnum()
    )


_COUNTRIES = {
    "usa": "us",
    "unitedstatesofamerica": "us",
    "unitedstates": "us",
    "greatbritain": "gb",
    "britain": "gb",
    "uae": "ae",
    "uk": "gb",
    "unitedkingdom": "gb",
    "china": "cn",
    "中国": "cn",
    "中國": "cn",
}
# Abbreviations only match a subdivision inside its own country.
_SUBDIVISIONS = {
    "au": {
        "act": "Australian Capital Territory",
        "nsw": "New South Wales",
        "nt": "Northern Territory",
        "qld": "Queensland",
        "sa": "South Australia",
        "tas": "Tasmania",
        "vic": "Victoria",
        "wa": "Western Australia",
    },
    "us": dict(
        item.split(":")
        for item in [
            "al:Alabama",
            "ak:Alaska",
            "az:Arizona",
            "ar:Arkansas",
            "ca:California",
            "co:Colorado",
            "ct:Connecticut",
            "de:Delaware",
            "dc:District of Columbia",
            "fl:Florida",
            "ga:Georgia",
            "hi:Hawaii",
            "id:Idaho",
            "il:Illinois",
            "in:Indiana",
            "ia:Iowa",
            "ks:Kansas",
            "ky:Kentucky",
            "la:Louisiana",
            "me:Maine",
            "md:Maryland",
            "ma:Massachusetts",
            "mi:Michigan",
            "mn:Minnesota",
            "ms:Mississippi",
            "mo:Missouri",
            "mt:Montana",
            "ne:Nebraska",
            "nv:Nevada",
            "nh:New Hampshire",
            "nj:New Jersey",
            "nm:New Mexico",
            "ny:New York",
            "nc:North Carolina",
            "nd:North Dakota",
            "oh:Ohio",
            "ok:Oklahoma",
            "or:Oregon",
            "pa:Pennsylvania",
            "ri:Rhode Island",
            "sc:South Carolina",
            "sd:South Dakota",
            "tn:Tennessee",
            "tx:Texas",
            "ut:Utah",
            "vt:Vermont",
            "va:Virginia",
            "wa:Washington",
            "wv:West Virginia",
            "wi:Wisconsin",
            "wy:Wyoming",
        ]
    ),
    "ca": dict(
        item.split(":")
        for item in [
            "ab:Alberta",
            "bc:British Columbia",
            "mb:Manitoba",
            "nb:New Brunswick",
            "nl:Newfoundland and Labrador",
            "ns:Nova Scotia",
            "nt:Northwest Territories",
            "nu:Nunavut",
            "on:Ontario",
            "pe:Prince Edward Island",
            "qc:Quebec",
            "sk:Saskatchewan",
            "yt:Yukon",
        ]
    ),
}


def _admin_key(value):
    return _match(re.sub(r"\s+(province|state|prefecture)$", "", value, flags=re.I))


def _select_place(name, qualifiers, country, admin1, candidates):
    if not isinstance(candidates, list) or any(
        not isinstance(c, dict) for c in candidates
    ):
        raise ValueError("invalid geocoding response")
    same_name = [
        c
        for c in candidates
        if _match(c.get("name", "")) == _match(name)
        and not (c.get("feature_code") or "").startswith(("ADM", "PCL"))
    ]

    def country_matches(value, c):
        key = _match(value)
        key = _COUNTRIES.get(key, key)
        return key in {_match(c.get("country", "")), _match(c.get("country_code", ""))}

    def admin_matches(value, c):
        key = _match(value)
        code = _match(c.get("country_code", ""))
        alias = _SUBDIVISIONS.get(code, {}).get(key, value)
        return _admin_key(alias) == _admin_key(c.get("admin1", ""))

    matches = [
        c
        for c in same_name
        if all(country_matches(q, c) or admin_matches(q, c) for q in qualifiers)
        and (not country or country_matches(country, c))
        and (not admin1 or admin_matches(admin1, c))
    ]
    if not matches:
        if same_name and (qualifiers or country or admin1):
            raise ProviderError(
                "qualifier_mismatch",
                "Places with that name exist, but none match the country or state/province. Preserve the user's qualifiers and ask them to clarify.",
            )
        if any(_admin_key(name) == _admin_key(c.get("admin1", "")) for c in candidates):
            raise ProviderError(
                "needs_clarification",
                "This is a region, not a single place. Ask which city the user means; do not retry the region or search the web for its weather.",
            )
        raise ProviderError(
            "not_found",
            "No matching city was found. Ask the user for the city and country or state/province.",
        )
    matches.sort(key=lambda c: c.get("population") or 0, reverse=True)
    first = matches[0]
    if len(matches) > 1 and (first.get("population") or 0) < 5 * max(
        1, matches[1].get("population") or 0
    ):
        raise ProviderError(
            "ambiguous_location",
            "The place name is ambiguous. Ask for a country or state/province.",
        )
    if (
        not (qualifiers or country or admin1)
        and first["name"].casefold() != name.casefold()
        and (first.get("population") or 0) < 100_000
    ):
        raise ProviderError(
            "not_found",
            "No sufficiently reliable city match was found. Ask for the city and country or state/province.",
        )
    return first


_CODES = {
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


def _number(value):
    if value is None:
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite weather value")
    return f"{number:.1f}"


def _forecast_report(hit, forecast, imperial):
    current = forecast["current"]
    temp, wind, precip = ("°F", "mph", "in") if imperial else ("°C", "km/h", "mm")
    place = ", ".join(
        dict.fromkeys(
            _clean(hit.get(k) or "")
            for k in ("name", "admin1", "country")
            if hit.get(k)
        )
    )
    zone = _clean(forecast.get("timezone") or hit.get("timezone") or "n/a")
    lines = [
        f"Current weather for {place}:",
        f"  Conditions: {_CODES.get(current['weather_code'], 'Unknown')}",
        f"  Temperature: {_number(current['temperature_2m'])}{temp}",
        f"  Humidity: {_number(current['relative_humidity_2m'])}%",
        f"  Wind: {_number(current['wind_speed_10m'])} {wind}",
    ]
    if current.get("time"):
        lines.append(f"  Observed: {_clean(current['time'])} ({zone})")
    daily = forecast.get("daily")
    # Optional malformed daily data does not discard valid current conditions.
    if isinstance(daily, dict):
        required = [
            daily.get(k)
            for k in (
                "time",
                "weather_code",
                "temperature_2m_min",
                "temperature_2m_max",
            )
        ]
        if all(isinstance(v, list) for v in required):
            days = []
            for index, (day, code, low, high) in enumerate(zip(*required)):
                if index == 7:
                    break
                try:
                    weekday = date.fromisoformat(day).strftime("%a")
                    relative = (
                        " (today)"
                        if index == 0
                        else " (tomorrow)"
                        if index == 1
                        else ""
                    )
                    probability = daily.get("precipitation_probability_max", [])
                    rain = daily.get("precipitation_sum", [])
                    probability = (
                        probability[index]
                        if isinstance(probability, list) and index < len(probability)
                        else None
                    )
                    rain = (
                        rain[index]
                        if isinstance(rain, list) and index < len(rain)
                        else None
                    )
                    days.append(
                        f"  {day} {weekday}{relative}: {_CODES.get(code, 'n/a')}, {_number(low)}–{_number(high)}{temp}, precip {_number(probability)}%, {_number(rain)} {precip}"
                    )
                except (ValueError, TypeError):
                    continue
            if days:
                lines.append(f"Daily forecast (local dates, timezone {zone}):")
                lines.extend(days)
    lines.append(f"Data: Open-Meteo (timezone: {zone}) https://open-meteo.com/")
    return "\n".join(lines)[:OUTPUT_CAP]
