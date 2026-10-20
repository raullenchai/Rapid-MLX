"""Switchable compilation of stateless decode operations.

Array parameters (including weights) are explicit inputs. Shape specialization
is deliberate: decode and each prefill chunk size have separate traces. No
position, cache offset, or mutable module attribute is captured by a trace.
"""

from functools import wraps
from typing import Callable, ParamSpec, TypeVar

import mlx.core as mx

ENABLED = True

P = ParamSpec("P")
R = TypeVar("R")


def fuse(function: Callable[P, R]) -> Callable[P, R]:
    """Keep the eager reference available through ``__wrapped__`` for A/B tests."""
    compiled = mx.compile(function)

    @wraps(function)
    def dispatch(*args: P.args, **kwargs: P.kwargs) -> R:
        return (compiled if ENABLED else function)(*args, **kwargs)

    return dispatch
