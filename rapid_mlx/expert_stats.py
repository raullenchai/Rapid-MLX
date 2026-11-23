# SPDX-License-Identifier: Apache-2.0
"""Per-layer MoE routing for ``GET /experts``.

Off unless the server is started with ``--expert-stats`` or
``RAPID_MLX_EXPERT_STATS=1``. The wrappers sit on the calls that already
receive the chosen expert ids. A second projection of the same choice
(gate and down) is suppressed. Inkling's loop calls the same layer again,
so each pass is recorded.
"""

from __future__ import annotations

import contextvars
import inspect
import logging
import os
import re
import threading
from collections import deque
from typing import Any

log = logging.getLogger(__name__)

WINDOW = 256
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_ENV = "RAPID_MLX_EXPERT_STATS"
_ON = {"1", "true", "yes", "on"}

_enabled = False
_patched: set[type] = set()
_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "rapid_mlx_expert_stats_depth", default=0
)
_lock = threading.Lock()
_n_expert = 0
_n_expert_used = 0
_layers: dict[int, dict[str, Any]] = {}


def enabled() -> bool:
    return _enabled


def env_enabled() -> bool:
    return os.environ.get(_ENV, "").strip().lower() in _ON


def reset() -> None:
    """Clear recorded routings and the enable flag. Used by tests."""
    global _enabled, _n_expert, _n_expert_used
    with _lock:
        _enabled = False
        _n_expert = 0
        _n_expert_used = 0
        _layers.clear()


def activate() -> None:
    """Turn recording on and wrap the known expert-id receivers once.

    Wrapping is deferred until :func:`bind`, which runs after the other
    MoE patches have replaced ``__call__``. An earlier wrap would be
    bypassed by the fused gate/up path and would also change the
    signature those patches require.
    """
    global _enabled
    _enabled = True
    for cls in _receiver_types():
        _patch(cls)


def enable_from(flag: bool = False) -> bool:
    """Enable when the CLI flag or the environment asks for it.

    This only arms the recorder. The wrappers are installed by
    :func:`bind` once the loaded modules exist.
    """
    global _enabled
    if flag or env_enabled():
        _enabled = True
        return True
    return False


def bind(model: Any) -> int:
    """Stamp each receiver with the layer number in its module name.

    Call after the model and its MoE patches exist. Returns how many
    modules were stamped. A disabled server stamps nothing.
    """
    if not (enabled() or env_enabled()):
        return 0
    if not _patched:
        activate()
    _set_counts(_counts_from(model))
    stamped = 0
    named = getattr(model, "named_modules", None)
    if named is None:
        return 0
    for name, module in named():
        match = _LAYER.search(name)
        if match is None or type(module) not in _patched:
            continue
        module._expert_stats_layer = int(match.group(1))
        stamped += 1
    return stamped


def note(layer: int, indices: Any) -> None:
    """Append one routing per token position."""
    rows = _rows(indices)
    if not rows:
        return
    with _lock:
        global _n_expert, _n_expert_used
        width = max(len(row) for row in rows)
        if width > _n_expert_used:
            _n_expert_used = width
        high = max(expert for row in rows for expert in row)
        if high + 1 > _n_expert:
            _grow(high + 1)
        state = _layers.get(layer)
        if state is None:
            state = {
                "tokens": deque(maxlen=WINDOW),
                "recent": [0] * _n_expert,
                "n_tokens": 0,
            }
            _layers[layer] = state
        for row in rows:
            if len(state["tokens"]) == WINDOW:
                old = state["tokens"][0]
                for expert in old:
                    if expert < len(state["recent"]):
                        state["recent"][expert] -= 1
            state["tokens"].append(row)
            state["n_tokens"] += 1
            for expert in row:
                if 0 <= expert < len(state["recent"]):
                    state["recent"][expert] += 1


def snapshot() -> dict[str, Any]:
    """JSON body ``parse_experts`` already accepts."""
    with _lock:
        layers = []
        n_tokens = 0
        for il in sorted(_layers):
            state = _layers[il]
            n_tokens = max(n_tokens, state["n_tokens"])
            recent = list(state["recent"])
            if len(recent) < _n_expert:
                recent.extend([0] * (_n_expert - len(recent)))
            layers.append(
                {
                    "il": il,
                    "n_tokens": state["n_tokens"],
                    "tokens": [list(row) for row in state["tokens"]],
                    "recent": recent,
                }
            )
        return {
            "n_expert": _n_expert,
            "n_expert_used": _n_expert_used,
            "n_tokens": n_tokens,
            "window": WINDOW,
            "layers": layers,
        }


def _patch(cls: type) -> None:
    if cls in _patched:
        return
    original = cls.__call__
    if getattr(original, "_expert_stats_wrapped", False):
        _patched.add(cls)
        return

    def wrapped(self, x, indices=None, *args, **kwargs):
        token = None
        if (
            indices is not None
            and enabled()
            and _depth.get() == 0
            and isinstance(getattr(self, "_expert_stats_layer", None), int)
        ):
            try:
                note(self._expert_stats_layer, indices)
            except Exception:
                log.debug("expert routing note failed", exc_info=True)
            token = _depth.set(1)
        try:
            return original(self, x, indices, *args, **kwargs)
        finally:
            if token is not None:
                _depth.reset(token)

    try:
        wrapped.__signature__ = inspect.signature(original)
    except (TypeError, ValueError):
        pass
    wrapped._expert_stats_wrapped = True
    cls.__call__ = wrapped
    _patched.add(cls)


def _receiver_types() -> list[type]:
    found: list[type] = []
    try:
        from mlx_lm.models.switch_layers import SwitchGLU, SwitchMLP

        found.extend([SwitchGLU, SwitchMLP])
    except Exception:
        log.debug("mlx-lm switch layers unavailable", exc_info=True)
    try:
        from rapid_mlx.models.inkling2.layers import Experts
        from rapid_mlx.models.inkling2.switch_layers import (
            QuantizedSwitchLinear,
            SwitchLinear,
        )

        found.extend([Experts, SwitchLinear, QuantizedSwitchLinear])
    except Exception:
        log.debug("inkling switch layers unavailable", exc_info=True)
    return found


def _rows(indices: Any) -> list[list[int]]:
    data = indices.tolist() if hasattr(indices, "tolist") else indices
    if not isinstance(data, list) or not data:
        return []
    rows: list[list[int]] = []

    def walk(node: Any) -> None:
        if isinstance(node, list) and node and isinstance(node[0], list):
            for child in node:
                walk(child)
        elif isinstance(node, list):
            rows.append([int(expert) for expert in node])

    walk(data)
    return rows


def _set_counts(counts: tuple[int, int]) -> None:
    global _n_expert, _n_expert_used
    n_expert, n_used = counts
    with _lock:
        if n_expert > _n_expert:
            _grow(n_expert)
        if n_used > _n_expert_used:
            _n_expert_used = n_used


def _grow(n_expert: int) -> None:
    global _n_expert
    extra = n_expert - _n_expert
    if extra <= 0:
        return
    for state in _layers.values():
        state["recent"].extend([0] * extra)
    _n_expert = n_expert


def _counts_from(model: Any) -> tuple[int, int]:
    objects: list[Any] = [model]
    args = getattr(model, "args", None)
    objects.append(args)
    if args is not None:
        objects.append(getattr(args, "text", None))
        objects.append(getattr(args, "text_config", None))
    inner = getattr(model, "model", None)
    if inner is not None:
        objects.append(getattr(inner, "args", None))
    n_expert = 0
    n_used = 0
    for obj in objects:
        found = _pair(obj)
        if found is None:
            continue
        n_expert = max(n_expert, found[0])
        n_used = max(n_used, found[1])
    return n_expert, n_used


def _pair(obj: Any) -> tuple[int, int] | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        n = _first_int(
            obj,
            "n_routed_experts",
            "num_local_experts",
            "num_experts",
            "moe_num_experts",
        )
        k = _first_int(obj, "num_experts_per_tok", "num_experts_per_token", "top_k")
    else:
        n = _first_attr(
            obj,
            "n_routed_experts",
            "num_local_experts",
            "num_experts",
            "moe_num_experts",
        )
        k = _first_attr(obj, "num_experts_per_tok", "num_experts_per_token", "top_k")
    if n is None and k is None:
        return None
    return (n or 0, k or 0)


def _first_int(obj: dict, *names: str) -> int | None:
    for name in names:
        value = obj.get(name)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _first_attr(obj: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None
