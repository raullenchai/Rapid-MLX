# SPDX-License-Identifier: Apache-2.0
"""KV cache export/import HTTP API — stub for issue #476.

This module defines the wire surface (request/response shapes, auth,
path-whitelist, manifest validation) that ``moc375``'s follow-up PR will
fill in with the engine-level save/load body. Stub behavior:

* ``POST /v1/cache/export`` — validates request, resolves destination
  under the sandbox, then returns **501 Not Implemented** with a link
  to the tracking issue. Engine integration is the follow-up's job.
* ``POST /v1/cache/import`` — validates request, resolves source under
  the sandbox, reads + checks the manifest against caller expectations,
  then returns **501 Not Implemented**. Mismatches surface as 409 so
  the wire-level contract is exercised today.
* ``GET /v1/cache/info`` — fully implemented: reads the manifest at a
  whitelisted path and returns it. Lets a peer instance (or oai-mlx) GC
  / inspect an export root without round-tripping a full import.

Auth follows ``vllm_mlx.routes.health``'s ``router``: the bearer key is
enforced when ``--api-key`` is set, no new header is invented.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..cache.protocol import (
    PROTOCOL_VERSION,
    InvalidExportPathError,
    MalformedManifestError,
    ManifestMismatchError,
    ManifestNotFoundError,
    read_manifest,
    resolve_cache_dir,
)
from ..middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

_ISSUE_URL = "https://github.com/raullenchai/Rapid-MLX/issues/476"
_NOT_IMPLEMENTED_MSG = (
    "engine integration pending; the wire contract (auth, path-whitelist, "
    f"manifest schema v{PROTOCOL_VERSION}) is live — see {_ISSUE_URL}"
)


router = APIRouter(
    prefix="/v1/cache",
    tags=["cache"],
    dependencies=[Depends(verify_api_key)],
)


class ExportRequest(BaseModel):
    """Request body for ``POST /v1/cache/export``."""

    destination: str | None = Field(
        default=None,
        description=(
            "Path under RAPID_MLX_CACHE_EXPORT_DIR (default "
            "~/.cache/rapid-mlx/cache_exports/). May be relative (resolved "
            "against the sandbox root) or absolute (must resolve inside "
            "the sandbox). Omit to use the sandbox root itself."
        ),
    )
    max_bytes: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional cap on the exported blob size. Implementation may "
            "stop adding entries once the total exceeds this; the engine "
            "integration follow-up defines the precise eviction order."
        ),
    )


class ImportRequest(BaseModel):
    """Request body for ``POST /v1/cache/import``."""

    source: str = Field(
        ...,
        description=(
            "Path to an export root containing manifest.json + index.json. "
            "Resolved under the export sandbox (see ExportRequest.destination)."
        ),
    )
    expected_protocol_version: str = Field(
        default=PROTOCOL_VERSION,
        description=(
            "Manifest protocol version the caller expects. Mismatch → 409. "
            f"Current: {PROTOCOL_VERSION!r}."
        ),
    )
    expected_model_id: str | None = Field(
        default=None,
        description=(
            "If set, manifest.model_id must match exactly. Mismatch → 409. "
            "Omit to skip the model-identity check (importer accepts any "
            "model — only use when you're sure the engine matches)."
        ),
    )
    merge_strategy: Literal["replace", "merge"] = Field(
        default="merge",
        description=(
            "'merge' keeps existing entries and adds new ones (token-tuple "
            "key collisions resolved by the engine). 'replace' clears the "
            "in-memory cache before loading. Implementation lands with the "
            "follow-up PR."
        ),
    )


def _resolve_or_400(caller_path: str | None) -> Path:
    """Wrap ``resolve_cache_dir`` so path violations surface as 403."""
    try:
        return resolve_cache_dir(caller_path)
    except InvalidExportPathError as exc:
        # 403 (not 400) because the request is well-formed JSON — what's
        # rejected is the *authorization* to write/read outside the sandbox.
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _read_manifest_or_http(root: Path):
    """Wrap ``read_manifest`` so missing/malformed surface as 404/400.

    Without this, a peer-written corrupt ``manifest.json`` would escape
    as a JSONDecodeError → FastAPI 500, hiding a caller-controlled bug
    inside an opaque server error. Mapping the three failure modes
    distinctly is what makes the contract usable from a client.

    Response details are caller-oriented — the fully resolved local
    filesystem path stays in the server log only, not in the HTTP body
    where a bearer-token holder could harvest the export-root layout.
    """
    try:
        return read_manifest(root)
    except ManifestNotFoundError as exc:
        logger.info("cache: manifest not found at %s", root)
        raise HTTPException(
            status_code=404,
            detail="no manifest.json at the requested cache path",
        ) from exc
    except MalformedManifestError as exc:
        # ``str(exc)`` is already path-free (see protocol.read_manifest).
        # It carries the structural reason — "not valid JSON: ...",
        # "must decode to a JSON object, got list", "manifest field
        # 'entries': expected int, got str" — which the client needs to
        # fix its own payload. The resolved path only lands in server logs.
        logger.warning("cache: malformed manifest at %s: %s", root, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/export", status_code=501)
async def export_cache(req: ExportRequest):
    """Export the engine's KV prefix cache to disk under the sandbox root.

    **Status: stub (issue #476).** This handler validates the request,
    resolves the destination against the path whitelist, and would call
    ``EngineCore.save_cache_to_disk`` — except the engine integration is
    the follow-up PR's responsibility. Returns 501 once all wire-level
    checks pass.
    """
    destination = _resolve_or_400(req.destination)
    logger.info(
        "cache/export: validated destination=%s max_bytes=%s — %s",
        destination,
        req.max_bytes,
        _NOT_IMPLEMENTED_MSG,
    )
    raise HTTPException(
        status_code=501,
        detail={
            "message": _NOT_IMPLEMENTED_MSG,
            "issue": _ISSUE_URL,
            "validated": {
                "destination": str(destination),
                "max_bytes": req.max_bytes,
                "protocol_version": PROTOCOL_VERSION,
            },
        },
    )


@router.post("/import", status_code=501)
async def import_cache(req: ImportRequest):
    """Import a peer instance's export into the local engine.

    **Status: stub (issue #476).** Validates the request, resolves the
    source under the sandbox, reads the manifest, and rejects on
    protocol-version or model-id mismatch before reaching the engine
    layer. The actual ``EngineCore.load_cache_from_disk`` call lands in
    the follow-up.
    """
    source = _resolve_or_400(req.source)
    manifest = _read_manifest_or_http(source)

    if manifest.protocol_version != req.expected_protocol_version:
        raise HTTPException(
            status_code=409,
            detail=str(
                ManifestMismatchError(
                    "protocol_version",
                    req.expected_protocol_version,
                    manifest.protocol_version,
                )
            ),
        )

    if req.expected_model_id is not None and manifest.model_id != req.expected_model_id:
        raise HTTPException(
            status_code=409,
            detail=str(
                ManifestMismatchError(
                    "model_id", req.expected_model_id, manifest.model_id
                )
            ),
        )

    logger.info(
        "cache/import: validated source=%s manifest=%s merge=%s — %s",
        source,
        manifest.model_id,
        req.merge_strategy,
        _NOT_IMPLEMENTED_MSG,
    )
    raise HTTPException(
        status_code=501,
        detail={
            "message": _NOT_IMPLEMENTED_MSG,
            "issue": _ISSUE_URL,
            "validated": {
                "source": str(source),
                "merge_strategy": req.merge_strategy,
                "manifest": manifest.to_dict(),
            },
        },
    )


@router.get("/info")
async def cache_info(path: str | None = None):
    """Read the manifest at a whitelisted export root.

    Returns the manifest dict so callers (peer instances, oai-mlx, ops
    tooling) can GC / route / version-gate without paying a full import.
    Path resolution follows the same sandbox rules as export/import.
    """
    root = _resolve_or_400(path)
    manifest = _read_manifest_or_http(root)

    return {
        "path": str(root),
        "protocol_version": PROTOCOL_VERSION,
        "manifest": manifest.to_dict(),
    }
