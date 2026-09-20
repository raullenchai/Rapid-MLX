# SPDX-License-Identifier: Apache-2.0
"""``telemetry_model_id`` — the ONLY model identity telemetry may report.

The field telemetry used to send was ``request.model``: whatever string the
HTTP client typed. On a server started with ``--served-model-name
acme-internal-support-bot`` — or one serving a private fine-tune by its
``org/name`` — that string went on the wire verbatim. This module replaces
it with an identity that is safe by construction:

1. **Catalog alias** — the model is in the shipped ``aliases.json`` (matched
   by alias spelling, or by ``hf_path`` case-insensitively). Public by
   definition: we publish the catalog.
2. **``org/name``** — only when we have *proof* the repo is public, i.e. a
   Hugging Face Hub call for it succeeded while **no token was in use**
   (see :func:`note_hub_fetch`). Non-catalog public models people actually
   run are the demand signal telemetry exists to find.
3. **``"<local>"``** — a local checkout / path. Never the path itself.
4. **``"<custom>"``** — everything else: gated repos, token-authenticated
   fetches, not-found repos, user aliases, anything we cannot prove public,
   and anything that fails the pattern/length cap.

``--served-model-name`` can never influence the result because the id is
computed from the *resolved checkpoint* at load time (``ModelEntry.model_path``),
not from the name the server advertises or the client requests.

**Limit, stated honestly.** Proof of anonymity is recorded when we touch the
Hub, so a model that was already in the HF cache before this shipped (or was
copied in out of band) has no proof and reports ``"<custom>"`` forever unless
some later Hub call for it succeeds anonymously. That is the conservative
direction: we under-report demand rather than over-report identity. We do
NOT infer "public" from "no token is set right now" — the weights on disk may
have been pulled months ago by an authenticated ``huggingface-cli download``.
"""

from __future__ import annotations

import os
import re
import threading

#: Model is local, or we cannot prove anything about it.
LOCAL = "<local>"
CUSTOM = "<custom>"

_SENTINELS: frozenset[str] = frozenset({LOCAL, CUSTOM})

# Defence in depth on top of the rules above: even a value that passed them
# must look like a catalog alias or a repo id, and be short. A catalog alias
# that somehow grew a query string, or a 4 KB repo name, collapses to
# ``<custom>`` rather than riding out on a payload.
_MAX_ID_LEN = 96
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}(?:/[A-Za-z0-9._-]{1,64})?$")

# Repo-id shape, i.e. the ``org/name`` branch of the pattern above.
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}/[A-Za-z0-9._-]{1,64}$")

# Env vars huggingface_hub reads for an API token. Checked directly (rather
# than only through ``get_token()``) so a missing/newer huggingface_hub does
# not silently turn an authenticated environment into a "proven public" one.
_HF_TOKEN_ENV_VARS: tuple[str, ...] = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HF_API_TOKEN",
)

_PUBLIC_MARKER_NAME = "public-anon-fetch"

_proven_lock = threading.Lock()
_proven_public: set[str] = set()


# ------------------------------------------------------------------- tokens


def hf_auth_in_use() -> bool:
    """Whether a Hugging Face token is available to this process.

    Fail-closed: if we cannot tell (no huggingface_hub, unreadable token
    file), we answer ``True`` — "assume authenticated" — because the only
    thing this answer gates is permission to report a repo id.
    """
    for var in _HF_TOKEN_ENV_VARS:
        if (os.environ.get(var) or "").strip():
            return True
    try:
        from huggingface_hub import get_token
    except Exception:
        return True
    try:
        return bool(get_token())
    except Exception:
        return True


# -------------------------------------------------------------------- proof


def _marker_path(repo_id: str) -> str | None:
    try:
        from rapid_mlx._download_gate import rapid_cache_marker_path

        return rapid_cache_marker_path(repo_id, _PUBLIC_MARKER_NAME)
    except Exception:
        return None


def note_hub_fetch(repo_id: str) -> None:
    """Record that a Hub call for ``repo_id`` just succeeded.

    Call this ONLY after a real Hub round trip returned successfully. It
    decides for itself whether that round trip was anonymous; if a token was
    available, nothing is recorded (we cannot tell a public repo from one
    our token opened).

    The proof is persisted as a marker file inside the repo's own HF cache
    directory, because the load that proves a model public is usually the
    cold pull, while the telemetry that wants to name it comes from every
    later warm start. The marker dies with the cached repo.

    Never raises: this sits on the download path.
    """
    try:
        if not isinstance(repo_id, str) or not _HF_REPO_RE.match(repo_id):
            return
        if hf_auth_in_use():
            return
        with _proven_lock:
            if repo_id in _proven_public:
                return
            _proven_public.add(repo_id)
        path = _marker_path(repo_id)
        if path is None:
            return
        if os.path.exists(path):
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Content-free marker; its existence is the whole signal. Written
        # non-atomically on purpose — an empty/partial file means exactly
        # what a complete one means.
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return


def is_proven_public(repo_id: str) -> bool:
    """Whether some anonymous Hub fetch of ``repo_id`` has ever succeeded."""
    try:
        with _proven_lock:
            if repo_id in _proven_public:
                return True
        path = _marker_path(repo_id)
        if path is None:
            return False
        return os.path.exists(path)
    except Exception:
        return False


def _reset_for_tests() -> None:
    """Drop the in-process proof cache. Tests-only seam."""
    with _proven_lock:
        _proven_public.clear()


# ----------------------------------------------------------------- the rule


def _looks_local(ref: str) -> bool:
    from rapid_mlx.telemetry.redact import normalize_model_path

    if normalize_model_path(ref) == LOCAL:
        return True
    try:
        return os.path.exists(ref)
    except (OSError, ValueError):
        return True


def _capped(value: str) -> str:
    """Pattern + length cap — the last gate before a value can be reported."""
    if len(value) > _MAX_ID_LEN or not _MODEL_ID_RE.match(value):
        return CUSTOM
    return value


def telemetry_model_id(model_ref: object) -> str:
    """Map a model reference to its privacy-safe telemetry identity.

    ``model_ref`` should be the RESOLVED checkpoint (``ModelEntry.model_path``
    / the output of ``resolve_model``), never a served name and never a
    client-supplied ``request.model``. Passing one of those is not unsafe —
    the rules below reject anything unproven — it just yields ``<custom>``.

    Total function: never raises, always returns one of a catalog alias, a
    proven-public ``org/name``, ``"<local>"`` or ``"<custom>"``.
    """
    try:
        if not isinstance(model_ref, str):
            return CUSTOM
        ref = model_ref.strip()
        if not ref:
            return CUSTOM
        if ref in _SENTINELS:
            return ref
        if _looks_local(ref):
            return LOCAL
        from rapid_mlx.model_aliases import catalog_alias_for

        alias = catalog_alias_for(ref)
        if alias is not None:
            return _capped(alias)
        if _HF_REPO_RE.match(ref) and is_proven_public(ref):
            return _capped(ref)
        return CUSTOM
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return CUSTOM


def served_model_id(requested: object = None) -> str:
    """The telemetry id of the model that actually served ``requested``.

    Route call sites pass the client's ``request.model`` purely as a routing
    key: it selects which loaded model answered, exactly as
    ``ModelRegistry.get_engine`` does, and the id returned is the one
    computed from that entry's resolved checkpoint at LOAD time. The
    caller's string never reaches a payload.
    """
    try:
        from rapid_mlx.config.server_config import get_config

        cfg = get_config()
    except Exception:
        return CUSTOM
    try:
        registry = getattr(cfg, "model_registry", None)
        if registry:
            key = requested if isinstance(requested, str) and requested else None
            try:
                entry = registry.get_entry(key)
            except Exception:
                entry = None
            if entry is not None:
                stored = getattr(entry, "telemetry_model_id", None)
                if isinstance(stored, str) and stored:
                    # Already the product of the full rule, applied at load
                    # time. Re-running it here would only re-stat the disk
                    # on every request; the cap is kept as the cheap part.
                    return stored if stored in _SENTINELS else _capped(stored)
                return telemetry_model_id(getattr(entry, "model_path", None))
        # Single-model server with no registry: ``model_path`` is the
        # resolved checkpoint, ``model_name`` may be a served name — only
        # the former may be consulted.
        return telemetry_model_id(getattr(cfg, "model_path", None))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return CUSTOM
