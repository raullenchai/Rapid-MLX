# SPDX-License-Identifier: Apache-2.0
"""Embeddings endpoint."""

import base64
import logging
import struct
import time

from fastapi import APIRouter, Depends, HTTPException

from ..api.models import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from ..config import get_config
from ..middleware.auth import check_rate_limit, verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/v1/embeddings",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Create embeddings for the given input text(s)."""
    from ..server import load_embedding_model

    cfg = get_config()
    # Bridge: fall back to server globals if config not yet synced
    if cfg.embedding_engine is None:
        from ..server import _embedding_engine

        cfg.embedding_engine = _embedding_engine
    if cfg.embedding_model_locked is None:
        from ..server import _embedding_model_locked

        if _embedding_model_locked is not None:
            cfg.embedding_model_locked = _embedding_model_locked

    try:
        model_name = request.model

        if (
            cfg.embedding_model_locked is not None
            and model_name != cfg.embedding_model_locked
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Embedding model '{model_name}' is not available. "
                    f"This server was started with --embedding-model {cfg.embedding_model_locked}. "
                    f"Only '{cfg.embedding_model_locked}' can be used for embeddings. "
                    f"Restart the server with a different --embedding-model to use '{model_name}'."
                ),
            )

        load_embedding_model(model_name, lock=False, reuse_existing=True)

        texts = request.input if isinstance(request.input, list) else [request.input]

        if not texts:
            raise HTTPException(status_code=400, detail="Input must not be empty")

        if request.dimensions is not None and request.dimensions < 1:
            raise HTTPException(
                status_code=400,
                detail="dimensions must be a positive integer",
            )

        start_time = time.perf_counter()
        prompt_tokens = cfg.embedding_engine.count_tokens(texts)
        embeddings = cfg.embedding_engine.embed(texts)
        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Embeddings: {len(texts)} inputs, {prompt_tokens} tokens in {elapsed:.2f}s"
        )

        # Optional truncation (OpenAI MRL semantics). Sliced post-embed
        # because mlx-embeddings doesn't expose a native dim parameter.
        if request.dimensions is not None:
            embeddings = [list(vec)[: request.dimensions] for vec in embeddings]

        # encoding_format=base64 per OpenAI spec — float32 little-endian
        # bytes, base64-encoded as ASCII. Saves ~2-4× bandwidth on large
        # batches and is the default for several OpenAI client SDKs.
        if request.encoding_format == "base64":
            encoded: list[list[float] | str] = []
            for vec in embeddings:
                vec_list = list(vec)
                packed = struct.pack(f"<{len(vec_list)}f", *vec_list)
                encoded.append(base64.b64encode(packed).decode("ascii"))
            data = [
                EmbeddingData(index=i, embedding=enc) for i, enc in enumerate(encoded)
            ]
        else:
            data = [
                EmbeddingData(index=i, embedding=list(vec))
                for i, vec in enumerate(embeddings)
            ]

        return EmbeddingResponse(
            data=data,
            model=model_name,
            usage=EmbeddingUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
            ),
        )

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="mlx-embeddings not installed. Install with: pip install 'rapid-mlx[embeddings]'",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
