# SPDX-License-Identifier: Apache-2.0
"""Coordinate process-wide MLX cache limits during bounded operations."""

import threading
from contextlib import contextmanager

_lock = threading.Lock()
_bound_depth = 0
_bound: int | None = None
_requested: int | None = None
_entry_limit: int | None = None
_applied_bound: int | None = None


def request_cache_limit(limit: int) -> int:
    """Set the requested limit, respecting an active temporary bound."""
    global _requested
    import mlx.core as mx

    with _lock:
        if _bound is None:
            return mx.set_cache_limit(limit)
        previous = _requested if _requested is not None else _entry_limit
        _requested = limit
        mx.set_cache_limit(min(limit, _bound))
        return previous if previous is not None else limit


@contextmanager
def bounded_cache_limit(bound: int):
    """Temporarily clamp the cache limit across overlapping contexts."""
    global _bound_depth, _bound, _requested, _entry_limit, _applied_bound

    with _lock:
        if _bound_depth == 0:
            try:
                import mlx.core as mx

                previous = mx.set_cache_limit(0)
                applied = min(previous, bound)
                try:
                    mx.set_cache_limit(applied)
                except Exception:
                    try:
                        mx.set_cache_limit(previous)
                    except Exception:
                        pass
                else:
                    _entry_limit = previous
                    _bound = bound
                    _applied_bound = applied
            except Exception:
                pass
        _bound_depth += 1
    try:
        yield
    finally:
        with _lock:
            _bound_depth -= 1
            if _bound_depth == 0:
                try:
                    if _entry_limit is not None:
                        import mlx.core as mx

                        if _requested is not None:
                            mx.set_cache_limit(_requested)
                        else:
                            current = mx.set_cache_limit(_entry_limit)
                            if current != _applied_bound:
                                mx.set_cache_limit(current)
                except Exception:
                    pass
                finally:
                    _bound = _requested = _entry_limit = _applied_bound = None
