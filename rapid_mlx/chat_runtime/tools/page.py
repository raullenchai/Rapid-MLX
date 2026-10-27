"""Public-page fetching with per-hop DNS validation and IP-pinned HTTPS."""

import asyncio
import ipaddress
import re
import socket
import unicodedata
from html.parser import HTMLParser
from urllib.parse import urljoin

import httpx

from .web import (
    TIMEOUT,
    ProviderError,
    SearchResult,
    WebResult,
    _safe_url,
    image_labels,
    response_body,
)

PAGE_CAP = 8_000
TABLE_ROW_CAP = 32
LINK_CAP = 6
LINK_CHARS = 1600
TRUNCATION = "\n[Page truncated]"
CONTENT_TYPES = {
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "text/markdown",
    "application/json",
}


def validate_url(url: str) -> httpx.URL:
    if not isinstance(url, str) or not _safe_url(url):
        raise ProviderError(
            "unsafe_url", "Only public HTTP and HTTPS page URLs are allowed."
        )
    try:
        target = httpx.URL(url)
    except httpx.InvalidURL:
        raise ProviderError("unsafe_url", "The page URL is invalid.") from None
    if (
        not target.host
        or "%" in target.host
        or (target.port is not None and not 1 <= target.port <= 65535)
    ):
        raise ProviderError("unsafe_url", "The page URL has an invalid host or port.")
    return target


def public_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped:
            return public_ip(str(ip.ipv4_mapped))
        if ip in ipaddress.ip_network("64:ff9b::/96"):
            return public_ip(str(ipaddress.IPv4Address(ip.packed[-4:])))
        if ip in ipaddress.ip_network("64:ff9b:1::/48"):
            packed = ip.packed
            return public_ip(
                str(ipaddress.IPv4Address(bytes([packed[i] for i in (6, 7, 9, 10)])))
            )
        if ip in ipaddress.ip_network("fe00::/8"):
            return False
    elif ip in ipaddress.ip_network("192.0.0.0/24"):
        return False
    return ip.is_global and not ip.is_multicast


async def resolve_public(url: str) -> tuple[httpx.URL, str]:
    target = validate_url(url)
    try:
        async with asyncio.timeout(min(TIMEOUT, 8.0)):  # type: ignore[attr-defined]  # Python 3.11-only harness.
            addresses = await asyncio.get_running_loop().getaddrinfo(
                target.host,
                target.port or (443 if target.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
    except OSError:
        raise ProviderError(
            "unresolvable", "The page hostname could not be resolved."
        ) from None
    ips = list(dict.fromkeys(str(row[4][0]) for row in addresses))
    if not ips or any(not public_ip(ip) for ip in ips):
        raise ProviderError(
            "unsafe_url", "The page hostname must resolve only to public addresses."
        )
    return target, ips[0]


class _PageParser(HTMLParser):
    """Extract readable blocks, dropping executable/hidden markup and chrome."""

    def __init__(self, url: str):
        super().__init__(convert_charrefs=True)
        self.url = url
        self.parts: list[str] = []
        self.title: list[str] = []
        self.in_title = False
        self.hidden: list[str] = []
        self.links: list[str] = []
        self.regions: dict[str, tuple[int, int | None]] = {}
        self.depths: dict[str, int] = {}
        self.tables: list[dict] = []

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.in_title = True
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.append(tag)
            return
        if tag in {
            "script",
            "style",
            "noscript",
            "svg",
            "head",
            "template",
            "iframe",
            "nav",
            "footer",
        }:
            self.hidden.append(tag)
            return
        if tag == "img":
            alt = dict(attrs).get("alt") or ""
            if alt and not re.fullmatch(r"Image(?: \d+)?", alt, re.I):
                self.parts.append(alt)
        if tag == "table":
            self.tables.append(
                {"start": len(self.parts), "rows": [], "cells": [], "cell": None}
            )
        elif self.tables and tag == "tr":
            self.tables[-1]["cells"] = []
        elif self.tables and tag in {"th", "td"}:
            table = self.tables[-1]
            table["cell"] = len(self.parts)
            raw_span = dict(attrs).get("colspan", "1")
            table["span"] = min(64, int(raw_span)) if str(raw_span).isdigit() else 1
        if tag in {"article", "main", "body"}:
            if tag not in self.regions:
                self.regions[tag] = (len(self.parts), None)
                self.depths[tag] = 1
            elif self.depths.get(tag, 0):
                self.depths[tag] += 1
        if tag in {
            "p",
            "div",
            "section",
            "article",
            "main",
            "br",
            "li",
            "tr",
            "blockquote",
        }:
            self.parts.append("\n")
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n" + "#" * int(tag[1]) + " ")
        if tag == "li":
            self.parts.append("- ")
        if tag == "a":
            raw = dict(attrs).get("href") or ""
            destination = urljoin(self.url, raw)
            destination = destination if raw and _safe_url(destination) else ""
            self.links.append(destination)
            if destination:
                self.parts.append("[")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
            return
        if self.tables:
            table = self.tables[-1]
            if tag in {"th", "td"} and table["cell"] is not None:
                cell = "".join(self.parts[table["cell"] :])
                cell = " ".join(cell.split()).replace("|", "\\|")
                table["cells"].extend([cell] + [""] * (max(1, table["span"]) - 1))
                table["cell"] = None
            elif tag == "tr" and table["cells"]:
                table["rows"].append(table["cells"])
                table["cells"] = []
            elif tag == "table":
                self.tables.pop()
                rows = table["rows"]
                if rows:
                    width = max(map(len, rows))
                    lines = [
                        "| " + " | ".join(row + [""] * (width - len(row))) + " |"
                        for row in rows
                    ]
                    lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
                    self.parts[table["start"] :] = ["\n" + "\n".join(lines) + "\n"]
        if tag == "a" and self.links:
            destination = self.links.pop()
            if destination:
                self.parts.append(f"]({destination})")
        if tag in {
            "p",
            "div",
            "section",
            "article",
            "main",
            "li",
            "tr",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }:
            self.parts.append("\n")
        if self.depths.get(tag, 0):
            self.depths[tag] -= 1
            if not self.depths[tag]:
                self.regions[tag] = (self.regions[tag][0], len(self.parts))

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)
        if not self.hidden and not self.in_title:
            self.parts.append(re.sub(r"\s+", " ", data))

    def text(self):
        parts = self.parts
        for tag in ("article", "main", "body"):
            if tag in self.regions:
                start, end = self.regions[tag]
                if "".join(parts[start:end]).strip():
                    parts = parts[start:end]
                    break
        return "\n".join(line.strip() for line in "".join(parts).splitlines()).strip()


def _compact_tables(content: str) -> str:
    """Bound markdown tables by whole rows, retaining headers and both ends."""
    lines = content.splitlines()
    output = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            output.append(lines[index])
            index += 1
            continue
        end = index
        while end < len(lines) and lines[end].lstrip().startswith("|"):
            end += 1
        rows = lines[index:end]
        if len(rows) > TABLE_ROW_CAP + 2:
            output.extend(rows[: 2 + TABLE_ROW_CAP // 2])
            output.append("[Table rows omitted; header and final rows retained]")
            # Repeat the header so the tail remains a valid, interpretable table.
            output.extend(rows[:2])
            output.extend(rows[-TABLE_ROW_CAP // 2 :])
        else:
            output.extend(rows)
        index = end
    return "\n".join(output)


def _bounded_content(content: str, cap: int) -> str:
    if len(content) <= cap:
        return content
    prefix = content[: cap - len(TRUNCATION)]
    # A partial table row can silently associate the wrong value with a
    # column. Drop the last row completely rather than clipping its columns.
    last_newline = prefix.rfind("\n")
    if prefix[last_newline + 1 :].lstrip().startswith("|"):
        prefix = prefix[: max(0, last_newline)]
    return prefix + TRUNCATION


def truncate_page(
    content: str, byte_budget: int, marker: str = "\n[Truncated to remaining context.]"
) -> str:
    """Fit UTF-8 evidence without cutting table cells or losing follow-up links."""
    if len(content.encode()) <= byte_budget:
        return content
    footer_start = content.rfind(
        "\n\nPage links (fetch the article for story details):\n"
    )
    footer = content[footer_start:] if footer_start >= 0 else ""
    body = content[:footer_start] if footer else content
    # Leave enough main text to identify what was actually retrieved. A footer
    # that alone consumes a tiny remaining budget must not masquerade as a page.
    if len((footer + marker).encode()) + 128 > byte_budget:
        footer = ""
    available = max(0, byte_budget - len((footer + marker).encode()))
    prefix = body.encode()[:available].decode("utf-8", errors="ignore")
    last_newline = prefix.rfind("\n")
    if prefix[last_newline + 1 :].lstrip().startswith("|"):
        prefix = prefix[: max(0, last_newline)]
    return prefix + marker + footer if prefix.strip() else ""


def page_result(url: str, title: str, content: str) -> WebResult:
    title = " ".join(title.split())[:240] or url
    content = "".join(
        c for c in content if c in "\n\t" or not unicodedata.category(c).startswith("C")
    )
    if not content.strip():
        raise ProviderError("empty_page", "The page did not contain readable text.")
    # Provider Markdown contains image URLs hundreds of characters long, often
    # inside story links or table cells. They convey no readable evidence.
    content = image_labels(content)
    # Some extractors nest video links inside story links. Remove generic
    # media markers before parsing the enclosing article destination.
    content = re.sub(r"\[(?:Video|Image) \d+\]\([^)]*\)", "", content)
    links: list[str] = []
    seen = {url}

    def link_text(match):
        label, destination = match.groups()
        destination = urljoin(url, destination)
        if _safe_url(destination) and destination not in seen and label.strip():
            seen.add(destination)
            line = f"- [{' '.join(label.split())[:160]}]({destination})"
            if (
                len(links) < LINK_CAP
                and sum(len(item) + 1 for item in links) + len(line) <= LINK_CHARS
            ):
                links.append(line)
                # Keep a story's actionable destination beside its headline;
                # moving every URL to a distant footer encouraged index-only
                # summaries. Tables retain compact labels and full columns.
                line_start = match.string.rfind("\n", 0, match.start()) + 1
                if (
                    not match.string[line_start : match.start()]
                    .lstrip()
                    .startswith("|")
                ):
                    return f"[{label}]({destination})"
        return label

    content = re.sub(
        r"\[([^\[\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", link_text, content
    )
    content = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", content).strip()
    content = _compact_tables(content)
    footer = (
        "\n\nPage links (fetch the article for story details):\n" + "\n".join(links)
        if links
        else ""
    )
    content = _bounded_content(content, PAGE_CAP - len(footer)) + footer
    return WebResult(content, (SearchResult(title, url, ""),))


async def direct_fetch(url: str, transport=None) -> WebResult:
    for hop in range(6):
        target, address = await resolve_public(url)
        # A numeric URL prevents a second hostname lookup. Host and TLS identity
        # stay tied to the original hostname; httpcore verifies that TLS name.
        pinned = target.copy_with(host=address, fragment=None)
        async with (
            httpx.AsyncClient(
                transport=transport,
                timeout=TIMEOUT,
                follow_redirects=False,
                trust_env=False,
                headers={"Accept-Encoding": "identity", "User-Agent": "Rapid-MLX/1.0"},
            ) as client,
            client.stream(
                "GET",
                pinned,
                headers={
                    "Host": target.netloc.decode("ascii"),
                    "Accept": ", ".join(sorted(CONTENT_TYPES)),
                },
                extensions={"sni_hostname": target.host},
            ) as response,
        ):
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location or hop == 5:
                    raise ProviderError(
                        "redirect_limit",
                        "The page redirected too many times or omitted its destination.",
                    )
                url = urljoin(str(target), location)
                continue
            if not 200 <= response.status_code < 300:
                raise ProviderError(
                    "fetch_error", f"The page returned HTTP {response.status_code}."
                )
            content_type = (
                response.headers.get("content-type", "")
                .split(";", 1)[0]
                .strip()
                .lower()
            )
            if content_type not in CONTENT_TYPES:
                raise ProviderError(
                    "unsupported_content",
                    "The page is not a supported text document.",
                )
            data = await response_body(response, page=True)
            try:
                text = data.decode(response.encoding or "utf-8", errors="replace")
            except LookupError:
                text = data.decode("utf-8", errors="replace")
            title = ""
            if content_type in {"text/html", "application/xhtml+xml"}:
                parser = _PageParser(str(target))
                parser.feed(text)
                parser.close()
                title, text = "".join(parser.title), parser.text()
            return page_result(str(target), title, text)
    raise AssertionError("unreachable")  # pragma: no cover - loop always exits
