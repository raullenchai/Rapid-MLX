# SPDX-License-Identifier: Apache-2.0
"""Model identifiers reported by retrieval-only health probes."""


def retrieval_model_ids(cfg) -> list[str] | None:
    """Return served retrieval IDs, or None for other server modes."""
    if cfg.model_name is not None or not cfg.retrieval_only:
        return None

    from .models import _locked_embedding_id, _locked_rerank_id

    return [
        model_id
        for model_id in (_locked_embedding_id(), _locked_rerank_id())
        if model_id
    ]
