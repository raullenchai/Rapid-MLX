# SPDX-License-Identifier: Apache-2.0
"""Per-layer MoE routing for ``GET /experts``.

Off unless the server is started with ``--expert-stats`` or
``RAPID_MLX_EXPERT_STATS=1``. The wrappers sit on the calls that already
receive the chosen expert ids. A second projection of the same choice
(gate and down) is suppressed. Inkling's loop calls the same layer again,
so each pass is recorded.

The recorder is owned by the most recently bound model. Binding clears
prior history; unloading that model clears it again. Concurrent resident
models therefore do not share one mixed window.

Each bound model owns a fresh ownership stamp. A record is committed only
while it still carries the current owner's stamp — checked at commit time,
under the lock, so a recording made before or during a rebind never leaks
into the new owner's window.

Activation happens at server start (``enable_from`` + ``bind``); teardown
happens at server shutdown (``disable``), which clears history, restores
the patched receivers, and puts the compiled-decode opt-out switch back the
way it was. The unauthenticated probe-surface body carries aggregate counts
and a sanitized model identifier only.
"""

from __future__ import annotations

import contextvars
import inspect
import logging
import os
import re
import sys
import threading
from collections import deque
from typing import Any

log = logging.getLogger(__name__)

WINDOW = 256
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_ENV = "RAPID_MLX_EXPERT_STATS"
_ON = {"1", "true", "yes", "on"}
# Single opt-out switch for the compiled decode path. While the recorder
# is armed the fast compiled path would skip the Python receivers, so
# arming forces this to "0"; teardown restores the original value.
_FAST_DECODE_ENV = "RAPID_MLX_INKLING2_FAST_DECODE"
_fast_decode_original: str | None = None
_fast_decode_armed = False
# A hub repo id (owner/name) passes through intact; anything else is treated
# as path-like and reduced to its final component before it reaches the wire.
_REPO_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+")

_enabled = False
_patched: set[type] = set()
_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "rapid_mlx_expert_stats_depth", default=0
)
_lock = threading.Lock()
_n_expert = 0
_n_expert_used = 0
_layers: dict[int, dict[str, Any]] = {}
_owner_id: int | None = None
_owner_token: int | None = None
_binding_token: int = 0
_originals: dict[type, Any] = {}
_model_name: str | None = None


def enabled() -> bool:
    return _enabled


def env_enabled() -> bool:
    return os.environ.get(_ENV, "").strip().lower() in _ON


def _set_fast_decode_off() -> None:
    """Arm the eager decode path (see ``_FAST_DECODE_ENV``). Idempotent."""
    global _fast_decode_original, _fast_decode_armed
    if _fast_decode_armed:
        return
    _fast_decode_original = os.environ.get(_FAST_DECODE_ENV)
    os.environ[_FAST_DECODE_ENV] = "0"
    _fast_decode_armed = True


def _restore_fast_decode() -> None:
    """Restore the original decode switch at teardown. Idempotent."""
    global _fast_decode_original, _fast_decode_armed
    if not _fast_decode_armed:
        return
    if _fast_decode_original is None:
        os.environ.pop(_FAST_DECODE_ENV, None)
    else:
        os.environ[_FAST_DECODE_ENV] = _fast_decode_original
    _fast_decode_original = None
    _fast_decode_armed = False


def reset() -> None:
    """Clear recorded routings, the enable flag and the decode switch. Used by tests."""
    global _enabled, _n_expert, _n_expert_used, _owner_id, _owner_token, _model_name
    with _lock:
        _enabled = False
        _n_expert = 0
        _n_expert_used = 0
        _owner_id = None
        _owner_token = None
        _model_name = None
        _layers.clear()
    _restore_fast_decode()


def clear_history() -> None:
    """Drop recorded routings while leaving the feature armed."""
    global _n_expert, _n_expert_used, _owner_id, _owner_token, _model_name
    with _lock:
        _n_expert = 0
        _n_expert_used = 0
        _owner_id = None
        _owner_token = None
        _model_name = None
        _layers.clear()


def activate() -> None:
    """Turn recording on and wrap the known expert-id receivers.

    Wrapping is deferred until :func:`bind`, which runs after the other
    MoE patches have replaced ``__call__``. An earlier wrap would be
    bypassed by the fused gate/up path and would also change the
    signature those patches require. Bind re-checks each class on every
    call so a later fusion that replaced our wrapper is restored.
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
    global _enabled, _owner_token
    if flag or env_enabled():
        _enabled = True
        _owner_token = None
        _set_fast_decode_off()
        return True
    return False


def bind(model: Any, model_name: str | None = None) -> int:
    """Stamp receivers on ``model`` and make it the active recorder owner.

    Call after the model and its MoE patches exist. Clears any previous
    model's history so resident swaps do not mix windows. Returns how
    many modules were stamped. A disabled server stamps nothing.
    """
    if not (enabled() or env_enabled()):
        return 0
    clear_history()
    global _enabled, _binding_token, _owner_token
    _enabled = True
    _binding_token += 1
    stamp_token = _binding_token
    _owner_token = stamp_token
    # Fusion can replace a wrapped __call__. Rewrap every class we already
    # know about, then discover stock receivers when their modules are
    # already imported (so unit tests with dummy classes stay mlx-free).
    for cls in list(_patched):
        _patch(cls)
    if _should_discover_receivers():
        for cls in _receiver_types():
            _patch(cls)
    _set_counts(_counts_from(model))
    _drop_compiled_decode(model)
    stamped = 0
    named = getattr(model, "named_modules", None)
    if named is None:
        return 0
    for name, module in named():
        match = _LAYER.search(name)
        if match is None or type(module) not in _patched:
            continue
        module._expert_stats_layer = int(match.group(1))
        module._expert_stats_token = stamp_token
        stamped += 1
    global _owner_id, _model_name
    with _lock:
        _owner_id = id(model)
        _model_name = model_name or _name_of(model)
    return stamped


def unbind(model: Any | None = None) -> None:
    """Retire history when the owning model is unloaded.

    The dead owner's stamps are dropped too, so even if the model object
    outlives the engine that owned it, its receivers can never record
    into the new owner's window.
    """
    with _lock:
        if model is not None and _owner_id is not None and id(model) != _owner_id:
            return
    clear_history()
    named = getattr(model, "named_modules", None) if model is not None else None
    if named is not None:
        for _, module in named():
            for attr in ("_expert_stats_layer", "_expert_stats_token"):
                try:
                    delattr(module, attr)
                except AttributeError:
                    pass


def disable() -> None:
    """Turn the recorder off and restore the patched receivers.

    Called at server shutdown — the server lifecycle owns the activation
    (``enable_from``) and this teardown. After this runs, the flag is off,
    history is cleared, the owner stamp is invalidated, and every class we
    wrapped is restored to the callable it was wrapped over (only when the
    current ``__call__`` still carries our marker, so a fusion or another
    patch that replaced our wrapper is never clobbered). The classes are
    forgotten so a fresh ``bind`` re-wraps cleanly.
    """
    global _enabled
    with _lock:
        _enabled = False
    clear_history()
    _restore_fast_decode()
    for cls in list(_patched):
        current = cls.__call__
        if getattr(current, "_expert_stats_wrapped", False):
            cls.__call__ = _originals[cls]
        _patched.discard(cls)


def note(layer: int, indices: Any, token: int | None = None) -> None:
    """Append one routing per token position.

    ``token`` is the ownership stamp taken from the receiving module. When
    given, the record commits only if it still matches the current owner —
    checked at commit time, under the lock, so a recording in flight during
    a rebind is rejected even after the rows were built. A direct caller
    that does not stamp a token (tests) is unrestricted.
    """
    rows = _rows(indices)
    if not rows:
        return
    with _lock:
        if token is not None and token != _owner_token:
            return
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


def _sanitize_model_name(value: Any) -> str | None:
    """Keep the identifier, never a filesystem path.

    A served name and a hub repo id pass through intact. Anything path-like
    (absolute/relative paths, home dirs, drive letters) is reduced to its
    final component, so the unauthenticated probe surface never leaks where
    the model lives on disk.
    """
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    if _REPO_ID.fullmatch(value):
        return value
    if "/" in value or "\\" in value:
        parts = [part for part in value.replace("\\", "/").split("/") if part]
        return parts[-1] if parts else None
    return value


def snapshot() -> dict[str, Any]:
    """JSON body for ``GET /experts``: aggregate counts and dimensions only.

    This route sits on the unauthenticated probe surface, so the body carries
    per-layer aggregate hit counts, global dimensions and a sanitized model
    identifier — never ordered per-token routings or a filesystem path.
    """
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
                    "recent": recent,
                }
            )
        body: dict[str, Any] = {
            "n_expert": _n_expert,
            "n_expert_used": _n_expert_used,
            "n_tokens": n_tokens,
            "window": WINDOW,
            "layers": layers,
        }
        model = _sanitize_model_name(_model_name)
        if model:
            body["model"] = model
        return body


def _patch(cls: type) -> None:
    original = cls.__call__
    if getattr(original, "_expert_stats_wrapped", False):
        _patched.add(cls)
        return

    def wrapped(self, x, indices=None, *args, **kwargs):
        depth_token = None
        if (
            indices is not None
            and enabled()
            and _depth.get() == 0
            and isinstance(getattr(self, "_expert_stats_layer", None), int)
        ):
            try:
                note(
                    self._expert_stats_layer,
                    indices,
                    getattr(self, "_expert_stats_token", None),
                )
            except Exception:
                log.debug("expert routing note failed", exc_info=True)
            depth_token = _depth.set(1)
        try:
            return original(self, x, indices, *args, **kwargs)
        finally:
            if depth_token is not None:
                _depth.reset(depth_token)

    try:
        wrapped.__signature__ = inspect.signature(original)
    except (TypeError, ValueError):
        pass
    wrapped._expert_stats_wrapped = True
    _originals[cls] = original
    cls.__call__ = wrapped
    _patched.add(cls)


def _should_discover_receivers() -> bool:
    if not _patched:
        return True
    return any(
        name in sys.modules
        for name in (
            "mlx_lm.models.switch_layers",
            "mlx_vlm.models.switch_layers",
            "rapid_mlx.models.inkling2.layers",
            "rapid_mlx.models.inkling2.switch_layers",
        )
    )


def _receiver_types() -> list[type]:
    found: list[type] = []
    try:
        from mlx_lm.models.switch_layers import SwitchGLU, SwitchMLP

        found.extend([SwitchGLU, SwitchMLP])
    except Exception:
        log.debug("mlx-lm switch layers unavailable", exc_info=True)
    # mlx-vlm keeps a private copy. Importing it here would defeat the
    # optional vision dependency, so enroll it only once the package is
    # already resident (true for any loaded VLM), matching moe_fusion.
    vlm_switch = sys.modules.get("mlx_vlm.models.switch_layers")
    if vlm_switch is not None:
        for name in ("SwitchGLU", "SwitchMLP"):
            cls = getattr(vlm_switch, name, None)
            if isinstance(cls, type) and cls not in found:
                found.append(cls)
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


def _drop_compiled_decode(model: Any) -> None:
    """Drop Inkling fast-decode caches so stats see eager routing again."""
    named = getattr(model, "named_modules", None)
    if not callable(named):
        return
    for _, module in named():
        stale = [
            attr
            for attr in vars(module)
            if isinstance(attr, str) and attr.startswith("_block_compiled_")
        ]
        for attr in stale:
            try:
                delattr(module, attr)
            except Exception:
                log.debug("failed to drop compiled decode cache %s", attr, exc_info=True)


def _name_of(model: Any) -> str | None:
    for attr in ("model_id", "name", "model_type"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    args = getattr(model, "args", None)
    for obj in (args, getattr(args, "text", None) if args is not None else None):
        value = getattr(obj, "model_type", None) if obj is not None else None
        if isinstance(value, str) and value:
            return value
    return None


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
    language = getattr(model, "language_model", None)
    if language is not None:
        objects.append(getattr(language, "args", None))
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
