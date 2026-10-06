"""Admission-only connection reuse; every authorization GET still hits GitHub."""

from __future__ import annotations

import http.client
import json
import re
from urllib.parse import parse_qs, urlencode, urlsplit

from scripts import queue_tree_evidence as evidence


class PersistentGitHubClient(evidence.GitHubClient):
    """One TLS connection per CLI, no response cache, redirects or retry fallback."""

    def __init__(self, repo: str, token: str) -> None:
        super().__init__(repo)
        if not token or any(ord(c) < 33 or ord(c) > 126 for c in token):
            raise evidence.EvidenceError("missing or invalid admission API token")
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "rapid-mlx-candidate-admission",
        }
        self._connection = http.client.HTTPSConnection("api.github.com", timeout=12)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._connection.close()

    def json(self, endpoint: str, *fields: str, paginate: bool = False):
        prefix = f"repos/{self.repo}/"
        if (
            not endpoint.startswith(prefix)
            or any(c in endpoint for c in ("?", "#", "\\"))
            or any(part in (".", "..") for part in endpoint.split("/"))
        ):
            raise evidence.EvidenceError("admission API path is outside own repository")
        params = []
        for field in fields:
            key, separator, value = field.partition("=")
            if not separator or not key or key == "page":
                raise evidence.EvidenceError("invalid admission API query field")
            params.append((key, value))
        page, pages = 1, []
        while True:
            query = urlencode(params + ([("page", str(page))] if page > 1 else []))
            path = "/" + endpoint + ("?" + query if query else "")
            try:
                self._connection.request("GET", path, headers=self._headers)
                response = self._connection.getresponse()
                raw = response.read(16_000_001)
                if len(raw) > 16_000_000:
                    raise evidence.EvidenceError("admission API response exceeds bound")
                if response.status != 200:
                    raise evidence.EvidenceError(
                        f"admission API returned HTTP {response.status}"
                    )
                value = json.loads(raw)
                link = response.getheader("Link", "")
            except evidence.EvidenceError:
                self._connection.close()
                raise
            except (OSError, http.client.HTTPException, ValueError) as exc:
                self._connection.close()
                raise evidence.EvidenceError("admission API transport failed") from exc
            if not paginate:
                return value
            pages.append(value)
            following = re.findall(r'<([^>]+)>;\s*rel="next"', link)
            if not following:
                return pages
            if len(following) != 1:
                raise evidence.EvidenceError("ambiguous admission API next page")
            try:
                target = urlsplit(following[0])
            except ValueError as exc:
                raise evidence.EvidenceError("invalid admission API next page") from exc
            # GitHub may use its numeric /repositories/id alias in Link headers.
            # Only extract the page; always request the original own-repo path.
            suffix = endpoint[len(prefix) :]
            expected = target.path == "/" + endpoint or re.fullmatch(
                r"/repositories/[1-9][0-9]*/" + re.escape(suffix), target.path
            )
            following_page = parse_qs(target.query).get("page", [])
            if (
                target.scheme != "https"
                or target.netloc != "api.github.com"
                or target.fragment
                or not expected
                or len(following_page) != 1
                or not following_page[0].isascii()
                or not following_page[0].isdigit()
                or not page < int(following_page[0]) <= 100
            ):
                raise evidence.EvidenceError("untrusted admission API pagination")
            page = int(following_page[0])
