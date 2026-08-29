# SPDX-License-Identifier: Apache-2.0
"""Dependency-free engine error types shared across runtime boundaries.

Keep these exceptions outside MLX-owning modules so API and service code can
translate engine failures without importing the Apple-only scheduler runtime.
"""


class BackpressureError(Exception):
    """Raised when admission control rejects a new request.

    Route handlers convert this to HTTP 503 with a Retry-After header.  It is
    distinct from ``ValueError`` so narrow batch-error handlers do not swallow
    admission failures.
    """
