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


class PagedCacheUnsupportedLayoutError(Exception):
    """Raised at startup when ``--use-paged-cache`` cannot serve the model.

    The paged prefix cache only supports cache layouts its block serializer
    can losslessly slice and reconstruct (plain full-attention KV layers).
    The check is structural — it inspects the loaded model's prompt-cache
    factory output, so it can only run after weights/model construction —
    and it fails closed: an explicit ``--use-paged-cache`` must abort before
    the server starts serving requests rather than degrade into a
    healthy-looking mode that provides zero reuse.

    ``incompatible_layers`` carries the offending cache class names (empty
    when the layout could not be determined at all).
    """

    def __init__(self, message: str, *, incompatible_layers: tuple[str, ...] = ()):
        super().__init__(message)
        self.incompatible_layers = incompatible_layers
