# SPDX-License-Identifier: Apache-2.0
"""Convert browser Hugging Face links to the existing repository-ID interface."""

import os
import re
from collections.abc import Callable
from urllib.parse import unquote, urlsplit

_HOSTS = frozenset({"huggingface.co", "www.huggingface.co", "hf.co", "hf-mirror.com"})
_NON_MODEL_ROUTES = {
    "datasets": "dataset",
    "spaces": "Space",
    "collections": "collection",
    "models": "model listing",
}


def normalize_model_reference(
    name: str, *, notice: Callable[[str], None] | None = None
) -> str:
    """Normalize HF URLs only; preserve aliases, repo IDs and local paths.

    Browser revisions/subpaths do not select an artifact: downloads retain
    their existing default-branch/catalog-pin behavior. Notices are opt-in so
    programmatic resolution stays quiet; the CLI reports them on stderr.
    """
    if os.path.exists(name):
        return name
    candidate = name.strip()
    wrappers = {"'": "'", '"': '"', "<": ">"}
    while len(candidate) >= 2 and wrappers.get(candidate[0]) == candidate[-1]:
        candidate = candidate[1:-1].strip()
    if os.path.exists(candidate):
        return name
    # Do not parse unrelated strings as URLs (even malformed URL-like paths
    # and aliases must keep their existing resolver behavior).
    if not re.match(
        r"^(?:https?://)?(?:huggingface\.co|www\.huggingface\.co|hf\.co|hf-mirror\.com)(?:[/?#]|$)",
        candidate,
        re.IGNORECASE,
    ):
        return name
    parsed = urlsplit(candidate if "://" in candidate else "https://" + candidate)
    if parsed.netloc.lower() not in _HOSTS or parsed.scheme.lower() not in {
        "http",
        "https",
    }:  # pragma: no cover - defence in depth; the regex above pins host/scheme
        return name

    parts = parsed.path.strip("/").split("/")
    if kind := _NON_MODEL_ROUTES.get(parts[0].lower()):
        raise ValueError(
            f"That's a Hugging Face {kind} link, not a model. "
            "Paste a model page URL (https://huggingface.co/org/name)."
        )
    if len(parts) < 2 or not all(
        re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", part) for part in parts[:2]
    ):
        raise ValueError(
            "Paste a Hugging Face model page URL: https://huggingface.co/org/name."
        )
    repo = "/".join(parts[:2])
    if notice is not None:
        notice(f"Using {repo}")
        if (
            len(parts) >= 4
            and parts[2] in {"tree", "blob", "resolve"}
            and unquote(parts[3]) != "main"
        ):
            notice(
                "Note: URL revision is ignored; using the default branch or catalog pin."
            )
    return repo
