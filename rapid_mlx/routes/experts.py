# SPDX-License-Identifier: Apache-2.0
"""``GET /experts`` for the live MoE panel.

Unauthenticated, like ``/metrics``. Absent unless the server was started
with ``--expert-stats`` or ``RAPID_MLX_EXPERT_STATS=1``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..expert_stats import enabled, snapshot

router = APIRouter()


@router.get("/experts")
def experts():
    if not enabled():
        raise HTTPException(status_code=404, detail="expert stats are off")
    return snapshot()
