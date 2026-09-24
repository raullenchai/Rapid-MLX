"""Vendored speculative-decode core (upstream ``mlx_vlm.speculative`` @ 0.7.2).

VENDOR-DEVIATION(subset-exports): the upstream init also re-exports
``load_drafter`` from ``.drafters``; the drafter registry and concrete
drafters land in a later step-3 slice, so this init exports only the
ddtree surface until then.
"""

from .ddtree import DDTreeNode, build_ddtree

__all__ = [
    "DDTreeNode",
    "build_ddtree",
]
