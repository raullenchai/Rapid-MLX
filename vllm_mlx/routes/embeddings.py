# SPDX-License-Identifier: Apache-2.0
"""Embeddings endpoint."""

import base64
import logging
import math
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

        # OpenAI spec supports 4 input shapes (see EmbeddingRequest):
        #   str                — single text
        #   list[str]          — batch of texts
        #   list[int]          — single pre-tokenized
        #   list[list[int]]    — batch of pre-tokenized
        # The int forms must NOT go through the str path: ``str(101)``
        # is a different embedding from token id 101.
        raw_input = request.input
        token_batches: list[list[int]] | None = None
        texts: list[str] | None = None
        if isinstance(raw_input, str):
            texts = [raw_input]
        elif isinstance(raw_input, list) and not raw_input:
            raise HTTPException(status_code=400, detail="Input must not be empty")
        elif isinstance(raw_input, list) and all(isinstance(x, str) for x in raw_input):
            texts = raw_input
        elif isinstance(raw_input, list) and all(isinstance(x, int) for x in raw_input):
            token_batches = [list(raw_input)]
        elif isinstance(raw_input, list) and all(
            isinstance(x, list) and all(isinstance(t, int) for t in x)
            for x in raw_input
        ):
            token_batches = [list(x) for x in raw_input]
        else:
            raise HTTPException(
                status_code=400,
                detail=("input must be str, list[str], list[int], or list[list[int]]"),
            )

        # Reject empty token sequences. ``[[]]`` would produce a
        # zero-width tensor; ``[[1, 2], []]`` produces a row whose
        # attention mask is all zeros (mlx-embeddings would either
        # NaN or return a meaningless zero vector depending on the
        # pooling head). Better to 400 with a clear message than
        # ship garbage embeddings to a vector store.
        if token_batches is not None and any(len(b) == 0 for b in token_batches):
            raise HTTPException(
                status_code=400,
                detail="input must not contain empty token sequences",
            )

        if request.dimensions is not None and request.dimensions < 1:
            raise HTTPException(
                status_code=400,
                detail="dimensions must be a positive integer",
            )

        start_time = time.perf_counter()
        if token_batches is not None:
            # count_tokens for pre-tokenized: trust the caller's count
            # (capped at 512 same as embed_tokens does).
            prompt_tokens = sum(min(len(b), 512) for b in token_batches)
            embeddings = cfg.embedding_engine.embed_tokens(token_batches)
            n_inputs = len(token_batches)
        else:
            prompt_tokens = cfg.embedding_engine.count_tokens(texts)
            embeddings = cfg.embedding_engine.embed(texts)
            n_inputs = len(texts)
        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Embeddings: {n_inputs} inputs, {prompt_tokens} tokens in {elapsed:.2f}s"
        )

        # Optional truncation (OpenAI MRL semantics). Sliced post-embed
        # because mlx-embeddings doesn't expose a native dim parameter,
        # then L2-renormalized so the resulting vector is still a valid
        # unit-norm embedding for cosine similarity (matches OpenAI
        # cookbook recommendation for text-embedding-3-large).
        if request.dimensions is not None:
            full_dim = len(list(embeddings[0]))
            if request.dimensions > full_dim:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"dimensions={request.dimensions} exceeds the "
                        f"model's native embedding size of {full_dim}"
                    ),
                )

            truncated: list[list[float]] = []
            for vec in embeddings:
                sliced = list(vec)[: request.dimensions]
                norm = math.sqrt(sum(x * x for x in sliced))
                if norm > 0:
                    sliced = [x / norm for x in sliced]
                truncated.append(sliced)
            embeddings = truncated

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
