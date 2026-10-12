# SPDX-License-Identifier: Apache-2.0
"""Rerank endpoint (Cohere / Jina compatible)."""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from ..api.models import (
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankResultDocument,
    RerankUsage,
)
from ..config import get_config
from ..middleware.auth import check_rate_limit, verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


def _invalid_request(status_code: int, message: str, code: str, param: str | None):
    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": code,
                "param": param,
            }
        },
    )


@router.post(
    "/v1/rerank",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    response_model_exclude_none=True,
)
async def create_rerank(request: RerankRequest, raw_request: Request) -> RerankResponse:
    """Score documents against a query and return them by descending relevance."""
    from rapid_mlx.telemetry import inference as _telemetry_inference
    from rapid_mlx.telemetry.model_id import telemetry_model_id

    from ..rerank import (
        RerankDocumentTooLongError,
        RerankInputTooLongError,
        RerankInstructionNotSupportedError,
    )

    cfg = get_config()
    caller_agent, caller_client = _telemetry_inference.request_caller_headers(
        raw_request
    )
    # Bridge: fall back to server globals if config not yet synced.
    if cfg.rerank_engine is None or cfg.rerank_model_locked is None:
        from .. import server

        cfg.rerank_engine = cfg.rerank_engine or server._rerank_engine
        cfg.rerank_model_locked = cfg.rerank_model_locked or server._rerank_model_locked

    # A missing --rerank-model is a deployment gap, not a bad payload: 503 so
    # clients back off instead of treating the request as permanently invalid.
    if cfg.rerank_model_locked is None or cfg.rerank_engine is None:
        _telemetry_inference.emit_capability_rejected(
            "rerank_unavailable",
            caller_agent=caller_agent,
            caller_client=caller_client,
        )
        raise _invalid_request(
            503,
            "No rerank model loaded. Restart the server with "
            "--rerank-model <hf-id> to enable /v1/rerank.",
            "no_rerank_model",
            None,
        )

    from ..service.helpers import _resolve_request_alias_or_default

    model_name = _resolve_request_alias_or_default(
        request.model, cfg.rerank_model_locked
    )
    if model_name is None:
        _telemetry_inference.emit_capability_rejected(
            "rerank_unavailable",
            model_type="rerank",
            model=telemetry_model_id(cfg.rerank_model_locked),
            caller_agent=caller_agent,
            caller_client=caller_client,
        )
        raise _invalid_request(
            400,
            f"Rerank model '{request.model}' is not available. This server was "
            f"started with --rerank-model {cfg.rerank_model_locked}; only "
            f"'{cfg.rerank_model_locked}' can be used for reranking.",
            "model_not_found",
            "model",
        )

    documents = [doc if isinstance(doc, str) else doc.text for doc in request.documents]
    rerank_engine = cfg.rerank_engine
    limit = cfg.rerank_max_documents
    if limit > 0 and len(documents) > limit:
        if rerank_engine.overflow_policy == "error":
            raise _invalid_request(
                400,
                f"Too many documents: {len(documents)} exceeds this server's limit "
                f"of {limit} (--rerank-max-documents). Split the request into "
                "smaller batches.",
                "too_many_documents",
                "documents",
            )
        logger.warning(
            "Rerank request has %d documents; scoring only the first %d "
            "(--rerank-max-documents). Set --rerank-overflow-policy error to "
            "reject such requests instead.",
            len(documents),
            limit,
        )
        documents = documents[:limit]
    start_time = time.perf_counter()
    try:
        scored = rerank_engine.score(
            request.query, documents, instruction=request.instruction
        )
    except RerankInstructionNotSupportedError as exc:
        raise _invalid_request(
            400,
            f"{exc} Omit the 'instruction' field for this model.",
            "invalid_request",
            "instruction",
        ) from exc
    except RerankInputTooLongError as exc:
        raise _invalid_request(
            400,
            f"The query needs {exc.observed_tokens} tokens, which leaves no room "
            f"for a document within the rerank model's {exc.allowed_tokens}-token "
            "input window. Shorten the query or instruction.",
            "input_too_long",
            "query",
        ) from exc
    except RerankDocumentTooLongError as exc:
        raise _invalid_request(
            400,
            f"Document {exc.index} needs {exc.observed_tokens} tokens together "
            f"with the query, which exceeds the rerank model's "
            f"{exc.allowed_tokens}-token input window. Shorten the document, or "
            "restart the server with --rerank-overflow-policy truncate.",
            "input_too_long",
            "documents",
        ) from exc
    except Exception as exc:
        logger.error("Rerank failed: %s", exc)
        _telemetry_inference.emit_completed_request(
            model=telemetry_model_id(cfg.rerank_model_locked),
            endpoint="/v1/rerank",
            caller_agent=caller_agent,
            caller_client=caller_client,
            result="failed",
            error_class=_telemetry_inference.classify_inference_failure(exc),
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    logger.info(
        "Rerank: %d documents, %d tokens in %.2fs",
        len(documents),
        scored.prompt_tokens,
        time.perf_counter() - start_time,
    )

    ranked = sorted(range(len(documents)), key=lambda i: scored.scores[i], reverse=True)
    if request.top_n:
        ranked = ranked[: request.top_n]
    results = [
        RerankResult(
            index=i,
            relevance_score=scored.scores[i],
            document=(
                RerankResultDocument(text=documents[i])
                if request.return_documents
                else None
            ),
        )
        for i in ranked
    ]
    _telemetry_inference.emit_completed_request(
        model=telemetry_model_id(cfg.rerank_model_locked),
        endpoint="/v1/rerank",
        caller_agent=caller_agent,
        caller_client=caller_client,
        result="ok",
    )
    return RerankResponse(
        model=model_name,
        results=results,
        usage=RerankUsage(
            prompt_tokens=scored.prompt_tokens, total_tokens=scored.prompt_tokens
        ),
    )
