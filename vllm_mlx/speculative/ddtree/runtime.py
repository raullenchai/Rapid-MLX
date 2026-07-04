# SPDX-License-Identifier: Apache-2.0
"""DDTree runtime boundary.

The first DDTree integration deliberately treats ``dtree_mlx`` as an
optional external runtime. That keeps the rapid-mlx MVP small and lets us
validate correctness/performance before deciding whether to vendor or
reimplement the tree verifier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .eligibility import have_runtime

logger = logging.getLogger(__name__)


@dataclass
class DDTreeRuntime:
    generator: Any
    main_model_repo: str
    drafter_repo: str
    speculative_tokens: int
    tree_budget: int


def load_runtime(
    *,
    main_model_repo: str,
    drafter_repo: str,
    speculative_tokens: int,
    tree_budget: int,
) -> DDTreeRuntime:
    if not have_runtime():
        raise RuntimeError(
            "DDTree runtime not available — install the experimental runtime with: "
            "pip install 'dtree-mlx @ git+https://github.com/DrHB/dtree-mlx.git'"
        )

    from dtree_mlx.api import DFlashGenerator

    logger.info(
        "Loading DDTree runtime: target=%s drafter=%s spec=%d tree_budget=%d",
        main_model_repo,
        drafter_repo,
        speculative_tokens,
        tree_budget,
    )
    generator = DFlashGenerator(
        target_model=main_model_repo,
        draft_model=drafter_repo,
        draft_attention_mask="auto",
    )
    return DDTreeRuntime(
        generator=generator,
        main_model_repo=main_model_repo,
        drafter_repo=drafter_repo,
        speculative_tokens=speculative_tokens,
        tree_budget=tree_budget,
    )
