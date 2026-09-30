# SPDX-License-Identifier: MIT
#
# Row-invariant lane matmul: the design and the original Metal kernels are
# TensorFold's (github.com/ashhart/TensorFold, MIT, (c) 2026 TensorFold
# contributors); this implementation is adapted from mlx2
# (github.com/pierre427/mlx2, src/mlx2/runtime/lane/). See LICENSE-LANE-MATMUL.

"""Route a loaded model's linear projections through the lane matmul.

Covered projections use the lane arithmetic for calls of ``min_rows`` to
``max_rows`` rows and stock MLX otherwise (chunked prefill above, and in
crossover mode one-token decode and short verifies below).  The installed
arithmetic is a numerical law distinct from stock MLX; callers bind
``receipt["law_id"]`` into any cache identity that stores state computed
under it.  Unsupported projections keep stock kernels and are reported.

Lane state lives on each module (outside its parameter tree), never in
process-wide maps keyed by ``id(module)``: CPython reuses a freed module's id
at once, and a dropped model's prepared weights would then reach the next
model's modules.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

import mlx.core as mx
from mlx import nn

from . import simd
from .matmul import (
    MAX_ROWS,
    UNQUANTIZED_BITS,
    LaneUnsupportedError,
    LaneWeights,
    available,
    backend,
    lane_matmul,
    prepare,
    split_k,
)

# One numerical law per backend: the M5 tensor-unit kernels and the M1-M4
# simdgroup kernels compute different bits for the same row.
LAW_IDS = {"mpp": "lane-matmul-v1", "simd": "lane-simd-v1"}
DEFAULT_MAX_ROWS = 32

# Sibling projections that read the same input, by attribute name under one
# parent module (attention q/k/v, MLP gate/up, Qwen3.5 GatedDeltaNet in_proj_*).
DEFAULT_GROUPS = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
    ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
)

# Host-side call counters for covered projections.
ROW_BUCKETS = ((1, 3), (4, 7), (8, 15), (16, 32), (33, MAX_ROWS))
STATS: Counter = Counter()


def _prepared(module):
    return module.__dict__.get("_lane_prepared")


def _group(module):
    return module.__dict__.get("_lane_group")


@dataclass
class _Group:
    """Same-format siblings stacked along N; one launch computes all of them."""

    lw: LaneWeights
    stack: dict  # the stacked MLX arrays the members' weights are views of
    call: dict  # the members' stock quantization arguments (bits, group size, mode)
    size: int  # members; a shared result is dropped once all of them took it
    last: list | None = None  # [x, stacked output, members served] of a lane launch
    # Below the crossover the members' stock calls can run as one stacked
    # stock launch when a probe proved it bitwise equal (see _probe_stock_stack).
    stock_stacked: bool = False
    stock_last: list | None = None


def _shared(group: _Group, attr: str, x, compute):
    """The group's stacked result for input ``x``: computed by the first
    member that sees ``x``, released once every member has taken its columns
    (so no layer keeps its last input and output alive between steps)."""
    held = getattr(group, attr)
    launched = held is None or held[0] is not x
    if launched:
        held = [x, compute(), 0]
        setattr(group, attr, held)
    held[2] += 1
    if held[2] >= group.size:
        setattr(group, attr, None)
    return held[1], launched


# The device can run a lane backend (fixed at install; checked once, not per call).
_LIVE = [False]


def _bucket(rows: int) -> str:
    # Lane calls take 1..MAX_ROWS rows, which the buckets cover exactly.
    return next(f"{low}-{high}" for low, high in ROW_BUCKETS if low <= rows <= high)


def stats() -> dict:
    """Copy of the call counters."""
    return dict(STATS)


def _rows(x) -> int:
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    return rows


def law_id(min_rows: int, backend_name: str | None = None) -> str:
    """Numerical-law identity for a given crossover (1 = exact) and backend."""
    base = LAW_IDS[backend_name or backend() or "mpp"]
    return base if min_rows == 1 else f"{base}+stock-below-{min_rows}"


class _LaneMixin:
    _lane_min_rows = 1
    _lane_max_rows = MAX_ROWS

    def __call__(self, x):
        rows = _rows(x)
        if rows < self._lane_min_rows:
            # The hot path of one-token decode: stock arithmetic, as cheap as
            # possible (a stacked stock launch where proven bitwise equal).
            group = self.__dict__.get("_lane_group")
            if group is not None and group.stock_stacked:
                return _stock_stacked(self, group, x)
            return self._lane_stock_call(x)
        lw = _prepared(self)
        if lw is None or not _LIVE[0]:
            STATS["stock_disabled"] += 1
        elif rows > self._lane_max_rows:
            STATS["stock_above_max_rows"] += 1
        else:
            group = _group(self)
            try:
                if group is None:
                    y = lane_matmul(x, lw)
                else:
                    # The first sibling to see this input computes the whole
                    # group; the others take their columns of the same result.
                    start, stop = self.__dict__["_lane_columns"]
                    out, launched = _shared(
                        group, "last", x, lambda: lane_matmul(x, group.lw)
                    )
                    STATS["group_launches" if launched else "group_reuses"] += 1
                    y = out[..., start:stop]
                    if lw.bias is not None:
                        y = y + lw.bias
                STATS["lane_calls"] += 1
                STATS[f"rows_{_bucket(rows)}"] += 1
                return y
            except LaneUnsupportedError:
                STATS["stock_unsupported"] += 1
        return self._lane_stock_call(x)


def _stock_matmul(group: _Group, x):
    stack, call = group.stack, group.call
    if "scales" not in stack:
        return x @ stack["weight"].T
    return mx.quantized_matmul(
        x,
        stack["weight"],
        stack["scales"],
        stack["biases"],
        transpose=True,
        group_size=call["group_size"],
        bits=call["bits"],
        mode=call["mode"],
    )


def _stock_stacked(module, group: _Group, x):
    """This member's columns of one stock launch over the group's stack."""
    out, _ = _shared(group, "stock_last", x, lambda: _stock_matmul(group, x))
    start, stop = module.__dict__["_lane_columns"]
    y = out[..., start:stop]
    bias = module.get("bias")
    return y if bias is None else y + bias


def _probe_stock_stack(
    members, group: _Group, rows_below: int, seen: dict[tuple, bool]
) -> bool:
    """Stacked stock launch bitwise equal to the members' own stock calls?

    Checked on this GPU for every row count that takes the stock path below
    the crossover, once per member-shape tuple (kernel choice depends only on
    shapes and dtypes).  MLX's small-M quantized kernels compute each output
    column independently, which this confirms rather than assumes.
    """
    stack = group.stack
    key = (
        tuple(tuple(m["weight"].shape) for m in members),
        tuple(sorted(group.call.items())),
        str(stack["weight"].dtype),
        str(stack["scales"].dtype if "scales" in stack else None),
        rows_below,
    )
    if key in seen:
        return bool(seen[key])
    k = group.lw.k
    same = True
    # Both activation dtypes a model may run: MLX specializes kernels by dtype.
    for dtype in (mx.bfloat16, mx.float16):
        for rows in range(1, rows_below + 1):
            x = (mx.random.normal((rows, k), key=mx.random.key(rows)) * 0.5).astype(
                dtype
            )
            apart = mx.concatenate([m._lane_stock_call(x) for m in members], axis=-1)
            together = _stock_matmul(group, x)
            if not bool(mx.array_equal(apart, together).item()):
                same = False
                break
        if not same:
            break
    seen[key] = same
    return same


class LaneQuantizedLinear(_LaneMixin, nn.QuantizedLinear):
    def _lane_stock_call(self, x):
        return nn.QuantizedLinear.__call__(self, x)


class LaneLinear(_LaneMixin, nn.Linear):
    def _lane_stock_call(self, x):
        return nn.Linear.__call__(self, x)


_SWAP = {nn.QuantizedLinear: LaneQuantizedLinear, nn.Linear: LaneLinear}
_RESTORE = {new: old for old, new in _SWAP.items()}


def _format_key(lw: LaneWeights) -> tuple:
    return (lw.backend, lw.bits, lw.group_size, lw.k, lw.weight.dtype, lw.scales_dtype)


def _stack(members) -> _Group:
    """Stack same-format siblings; each module's arrays become views of the stack."""
    first = _prepared(members[0])
    quantized = first.bits != UNQUANTIZED_BITS
    names = ("weight", "scales", "biases") if quantized else ("weight",)
    stacked = {
        name: mx.concatenate([m[name] for m in members], axis=0) for name in names
    }
    mx.eval(*stacked.values())
    start = 0
    for m in members:
        stop = start + int(m["weight"].shape[0])
        for name in names:
            setattr(m, name, stacked[name][start:stop])  # zero-copy row views
        object.__setattr__(m, "_lane_columns", (start, stop))
        start = stop
    mx.eval([m[name] for m in members for name in names])
    n = start
    if first.backend == "simd":
        # The simd kernels read MLX's scale/bias layout: the stack is enough.
        lw = LaneWeights(
            first.bits,
            first.group_size,
            n,
            first.k,
            simd.splits(n, first.k),
            stacked["weight"],
            None,
            None,
            "simd",
            stacked["scales"],
            stacked["biases"],
        )
    else:
        pairs = (
            mx.contiguous(mx.stack([stacked["scales"].T, stacked["biases"].T], axis=-1))
            if quantized
            else None
        )
        lw = LaneWeights(
            first.bits,
            first.group_size,
            n,
            first.k,
            split_k(n, first.k, first.group_size, first.bits),
            stacked["weight"],
            pairs,
            None,
        )
    for m in members:
        object.__setattr__(
            m, "_lane_prepared", prepare(m, first.backend)
        )  # individual views
    call = (
        {
            "bits": int(members[0].bits),
            "group_size": int(members[0].group_size),
            "mode": getattr(members[0], "mode", "affine"),
        }
        if quantized
        else {}
    )
    return _Group(lw, stacked, call, len(members))


def _dissolve(model) -> None:
    for _name, module in model.named_modules():
        for name in ("_lane_group", "_lane_columns"):
            module.__dict__.pop(name, None)


def _group_siblings(model, groups) -> Counter:
    formed: Counter = Counter()
    for _name, parent in model.named_modules():
        for names in groups:
            present = [getattr(parent, n, None) for n in names]
            present = [
                m
                for m in present
                if m is not None and _prepared(m) is not None and _group(m) is None
            ]
            by_format: dict[tuple, list] = {}
            for m in present:
                by_format.setdefault(_format_key(_prepared(m)), []).append(m)
            for members in by_format.values():
                if len(members) < 2:
                    continue
                group = _stack(members)
                for m in members:
                    object.__setattr__(m, "_lane_group", group)
                formed[f"{group.lw.format}x{len(members)}"] += 1
    return formed


def _enable_stock_stacks(model) -> dict:
    """Probe each group's stacked stock launch for the rows below its crossover."""
    members: dict[int, list] = {}
    groups: dict[int, _Group] = {}
    for _name, module in model.named_modules():
        group = _group(module)
        if group is not None:
            groups[id(group)] = group
            members.setdefault(id(group), []).append(module)
    seen: dict = {}
    proven = unproven = 0
    for gid, group in groups.items():
        ordered = sorted(members[gid], key=lambda m: m.__dict__["_lane_columns"][0])
        rows_below = min(int(m._lane_min_rows) for m in ordered) - 1
        if rows_below < 1:
            continue
        group.stock_stacked = _probe_stock_stack(ordered, group, rows_below, seen)
        proven += group.stock_stacked
        unproven += not group.stock_stacked
    return {"groups": proven, "unproven": unproven} if proven or unproven else {}


def _check_simd_twins(model) -> dict:
    """Run the simd twin check once per weight shape a call can launch.

    A shape whose scalar twin differs from the matrix kernel on this GPU is
    rerouted so every row count keeps one arithmetic (see ``simd.check``).
    """
    seen: dict = {}
    for _name, module in model.named_modules():
        group = _group(module)
        for lw in (_prepared(module), group.lw if group is not None else None):
            if lw is None or lw.backend != "simd":
                continue
            key = (lw.n, lw.k, lw.group_size, lw.bits, str(lw.scales_dtype))
            if key not in seen:
                simd.check(lw.weight, lw.scales, lw.biases, lw.group_size, lw.bits)
                # Read the route back: a shape rerouted by an earlier install
                # stays rerouted and must be reported (and in the law) again.
                seen[key] = simd.rerouted(lw.n, lw.k, lw.group_size, lw.bits)
    if not seen:
        return {}
    by_kind = {
        kind: sorted(
            f"{n}x{k}q{bits}g{gs}"
            for (n, k, gs, bits, _d), how in seen.items()
            if how == kind
        )
        for kind in ("affine", "mma")
    }
    return {
        "shapes": len(seen),
        "rerouted": sum(1 for how in seen.values() if how),
        **{kind: shapes for kind, shapes in by_kind.items() if shapes},
    }


def install(
    model,
    *,
    min_rows_by_format: dict[str, int],
    max_rows: int = DEFAULT_MAX_ROWS,
    groups=DEFAULT_GROUPS,
) -> dict:
    """Swap every supported projection to its lane class; returns a receipt.

    ``min_rows_by_format`` maps a format class (``q2``..``q8``, ``bf16``,
    ``fp16``) to the smallest row count that takes the lane arithmetic; a
    format absent from it stays on stock.  Same-format siblings in ``groups``
    are stacked into one launch.  Idempotent; a repeat call under another
    backend starts over.
    """
    if not 1 <= max_rows <= MAX_ROWS:
        raise ValueError(f"need 1 <= max_rows <= {MAX_ROWS}")
    current = backend() or "mpp"
    _LIVE[0] = available()
    if any(
        lw is not None and lw.backend != current
        for lw in (_prepared(m) for _n, m in model.named_modules())
    ):
        # Installed under the other backend: its prepared weights and groups
        # would keep running that law under this receipt.  Start over.
        uninstall(model)
    covered: Counter = Counter()
    refused: Counter = Counter()
    for _name, module in model.named_modules():
        kind = type(module)
        if kind not in _SWAP and kind not in _RESTORE:
            continue
        module_format = format_class(module)
        rows = min_rows_by_format.get(module_format) if module_format is not None else None
        if rows is None or rows > max_rows:
            if kind in _RESTORE:
                # Covered by an earlier install, not by this one: back to stock
                # so the receipt describes what runs.
                _restore(module)
            refused["no threshold for this format"] += 1
            continue
        if kind in _SWAP:
            try:
                lw = prepare(module)
            except LaneUnsupportedError as exc:
                refused[str(exc)] += 1
                continue
            object.__setattr__(module, "_lane_prepared", lw)
            module.__class__ = _SWAP[kind]
        object.__setattr__(module, "_lane_min_rows", int(rows))
        object.__setattr__(module, "_lane_max_rows", int(max_rows))
        covered[_prepared(module).format] += 1
    _dissolve(model)
    formed = _group_siblings(model, groups) if groups else Counter()
    twins = _check_simd_twins(model)
    stacked_stock = _enable_stock_stacks(model)
    spec = ",".join(f"{k}:{v}" for k, v in sorted(min_rows_by_format.items()))
    law = f"{LAW_IDS[current]}+stock-below[{spec}]"
    if max_rows != DEFAULT_MAX_ROWS:
        law += f"+rows-le-{max_rows}"
    if twins.get("affine"):
        # A 5/6/8-bit shape whose twins differ here runs affine_rows: other bits.
        shapes = ",".join(twins["affine"])
        law += f"+simd-affine[{hashlib.sha256(shapes.encode()).hexdigest()[:12]}]"
    if formed:
        law += "+grouped"
    receipt = {
        "law_id": law,
        "backend": current,
        "min_rows": dict(min_rows_by_format),
        "max_rows": max_rows,
        "covered": dict(covered),
        "groups": dict(formed),
        "refused": dict(refused),
    }
    if twins:
        receipt["simd_twins"] = twins
    if stacked_stock:
        receipt["stock_stacked"] = stacked_stock
    return receipt


def _restore(module) -> None:
    module.__class__ = _RESTORE[type(module)]
    for name in (
        "_lane_prepared",
        "_lane_group",
        "_lane_columns",
        "_lane_min_rows",
        "_lane_max_rows",
    ):
        module.__dict__.pop(name, None)


def uninstall(model) -> int:
    """Restore stock classes; returns how many projections were restored."""
    restored = 0
    for _name, module in model.named_modules():
        if type(module) in _RESTORE:
            _restore(module)
            restored += 1
    return restored


def format_class(module) -> str | None:
    """Format class of one projection, or None when it is not a linear layer."""
    if isinstance(module, nn.QuantizedLinear):
        return f"q{int(module.bits)}"
    if isinstance(module, nn.Linear):
        return {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(module["weight"].dtype)
    return None
