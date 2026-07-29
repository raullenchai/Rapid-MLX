"""LoRA merge-at-load for the MLX SA3 DiT.

Adds LoRA inference support to the MLX path, which the `sa3_mlx.py` CLI otherwise
lacks. The LoRA delta is **merged into the DiT weight dict at load time**, before
the model is built — no runtime parametrization and no extra per-step forward
cost. A strength of 0 is a bit-exact bypass.

Trust boundary: only `.safetensors` adapters are accepted. The legacy pickle
`.ckpt`/`.pt`/`.bin` path — which `torch.load` would execute arbitrary code from —
is refused outright; this module never calls `torch.load`.

Two on-disk conventions are supported:

  * **SA3-native** (`scripts/train_lora.py` output): tensor keys
    ``<layer>.parametrizations.weight.0.{lora_A,lora_B,M_xs,magnitude,
    magnitude_r,magnitude_c}`` with the adapter config
    (``adapter_type``/``rank``/``alpha``/``include``/``exclude``) JSON-encoded in
    the safetensors **metadata** under ``"lora_config"``. Covers all nine adapter
    types (lora, dora-rows/cols, bora, and the four -xs variants). Underfit
    (github.com/dada-bots/underfit) checkpoints are this convention with
    full-wrapper layer names (``model.…`` / ``conditioners.…``) — handled by
    ``_layer_to_npz_key``.
  * **PEFT** (huggingface `peft`): keys ``base_model.model.<layer>.lora_{A,B}.weight``
    with ``r``/``lora_alpha`` in a sibling ``adapter_config.json``. Standard LoRA,
    plus DoRA when ``use_dora`` is set.

The per-adapter-type math mirrors ``LoRAParametrization.*_forward`` in
``stable_audio_3/models/lora/model.py`` (and the accumulate-deltas-against-the-
original-weight semantics of ``merge_loras_into_base_model``), computed in
float32 and cast back to the DiT dtype. `-xs` adapters do not store their frozen
SVD bases, so they are recomputed from the base weight here, matching the
reference (`torch.linalg.svd` + a deterministic sign convention).
"""

from __future__ import annotations

import json
import os

import mlx.core as mx
import numpy as np

# Pickle-backed extensions we refuse to load (the trust boundary).
_PICKLE_EXTS = (".ckpt", ".pt", ".pth", ".bin")

# Adapter param names per type (mirrors utils._get_adapter_param_names).
_PARAMS_FOR = {
    "lora": ("lora_A", "lora_B"),
    "dora-rows": ("lora_A", "lora_B", "magnitude"),
    "dora-cols": ("lora_A", "lora_B", "magnitude"),
    "bora": ("lora_A", "lora_B", "magnitude_r", "magnitude_c"),
    "lora-xs": ("M_xs",),
    "dora-rows-xs": ("M_xs", "magnitude"),
    "dora-cols-xs": ("M_xs", "magnitude"),
    "bora-xs": ("M_xs", "magnitude_r", "magnitude_c"),
}


class LoraError(Exception):
    """An adapter could not be loaded or applied."""


# ── safetensors reading (no torch, no safetensors pkg — MLX reads it) ──────────

def _np(arr) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32)


def _load_safetensors(path: str):
    """Return ``(tensors: dict[str, np.ndarray], metadata: dict)``. Refuses pickle."""
    lower = path.lower()
    if lower.endswith(_PICKLE_EXTS):
        raise LoraError(
            f"refusing to load pickle-format adapter {os.path.basename(path)!r} — "
            f"only .safetensors adapters are accepted (a .ckpt/.pt is unpickled by "
            f"torch.load and can execute arbitrary code)"
        )
    if not lower.endswith(".safetensors"):
        raise LoraError(f"not a .safetensors adapter: {path!r}")
    arrs, meta = mx.load(path, return_metadata=True)
    return {k: _np(v) for k, v in arrs.items()}, (meta or {})


# ── SVD bases for -xs adapters (recomputed; mirrors model.py) ──────────────────

def _canonicalize_svd_signs(U: np.ndarray, Vh: np.ndarray):
    """Deterministic sign convention: largest-magnitude element of each U column
    is positive (mirrors model._canonicalize_svd_signs)."""
    max_abs_idx = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs_idx, np.arange(U.shape[1])])
    signs[signs == 0] = 1.0
    return U * signs[None, :], Vh * signs[:, None]


def _svd_bases(W0: np.ndarray, rank: int):
    """Return ``(U[:, :rank], V[:, :rank])`` from the SVD of ``W0`` (fan_out, fan_in),
    with V such that ``U @ diag(S) @ V.T`` reconstructs W0 (mirrors model.py)."""
    U_full, _S, Vh_full = np.linalg.svd(W0, full_matrices=False)
    U_full, Vh_full = _canonicalize_svd_signs(U_full, Vh_full)
    U = U_full[:, :rank]
    V = Vh_full[:rank, :].T
    return U, V


# ── per-type merge math (numpy, float32) ───────────────────────────────────────

def _merged_weight(W0: np.ndarray, p: dict, adapter_type: str, scaling: float) -> np.ndarray:
    """Return the LoRA-merged weight for one layer at full strength (lora_strength=1).

    ``W0`` is (fan_out, fan_in) float32; ``p`` holds the adapter tensors for this
    layer. Mirrors the matching ``*_forward`` in model.py.
    """
    if adapter_type == "lora":
        delta = p["lora_B"] @ p["lora_A"]
        return W0 + scaling * delta

    if adapter_type in ("dora-rows", "dora-cols"):
        norm_dim = 1 if adapter_type == "dora-rows" else 0
        delta = p["lora_B"] @ p["lora_A"]
        V = W0 + scaling * delta
        V_hat = V / (np.linalg.norm(V, axis=norm_dim, keepdims=True) + 1e-12)
        mag = _mag_2d(p["magnitude"], norm_dim)
        return V_hat * mag

    if adapter_type == "bora":
        delta = p["lora_B"] @ p["lora_A"]
        V = W0 + scaling * delta
        V_r = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        inter = p["magnitude_r"].reshape(-1, 1) * V_r
        H_c = inter / (np.linalg.norm(inter, axis=0, keepdims=True) + 1e-12)
        return H_c * p["magnitude_c"].reshape(1, -1)

    if adapter_type.endswith("-xs"):
        rank = p["M_xs"].shape[0]
        U, V = _svd_bases(W0, rank)
        delta = U @ p["M_xs"] @ V.T
        Vfull = W0 + scaling * delta
        if adapter_type == "lora-xs":
            return Vfull
        if adapter_type in ("dora-rows-xs", "dora-cols-xs"):
            norm_dim = 1 if adapter_type == "dora-rows-xs" else 0
            V_hat = Vfull / (np.linalg.norm(Vfull, axis=norm_dim, keepdims=True) + 1e-12)
            mag = _mag_2d(p["magnitude"], norm_dim)
            return V_hat * mag
        if adapter_type == "bora-xs":
            V_r = Vfull / (np.linalg.norm(Vfull, axis=1, keepdims=True) + 1e-12)
            inter = p["magnitude_r"].reshape(-1, 1) * V_r
            H_c = inter / (np.linalg.norm(inter, axis=0, keepdims=True) + 1e-12)
            return H_c * p["magnitude_c"].reshape(1, -1)

    raise LoraError(f"unknown adapter_type {adapter_type!r}")


def _check_shapes(layer: str, W0: np.ndarray, p: dict, adapter_type: str) -> None:
    """Fail with a clear message when the adapter doesn't fit the base weight —
    almost always because the adapter was trained for a different base than
    ``--dit`` (e.g. a medium adapter on sm-music). Without this the mismatch
    surfaces as a raw numpy broadcasting error deep in the merge."""
    fan_out, fan_in = W0.shape
    if adapter_type.endswith("-xs"):
        rank = p["M_xs"].shape[0]
        if rank > min(fan_out, fan_in):
            raise LoraError(
                f"{layer}: LoRA-XS rank {rank} exceeds base min-dim "
                f"{min(fan_out, fan_in)} for weight {W0.shape} — wrong base model?"
            )
        return
    b_out, b_rank = p["lora_B"].shape
    a_rank, a_in = p["lora_A"].shape
    if b_out != fan_out or a_in != fan_in or a_rank != b_rank:
        raise LoraError(
            f"{layer}: adapter lora_B{p['lora_B'].shape}·lora_A{p['lora_A'].shape} "
            f"does not fit base weight {W0.shape} — wrong base model?"
        )


def _mag_2d(mag: np.ndarray, norm_dim: int) -> np.ndarray:
    """Reshape a (possibly 2D) magnitude vector to broadcast against the weight on
    ``norm_dim`` (mirrors `magnitude.unsqueeze(norm_dim)` after a squeeze).
    ``atleast_1d`` guards the degenerate (1, 1) case where squeeze yields a
    0-d array (no real DiT layer has a single output, but keep it total)."""
    mag = np.atleast_1d(np.squeeze(mag))
    return mag.reshape(-1, 1) if norm_dim == 1 else mag.reshape(1, -1)


# ── checkpoint parsing → normalized per-layer adapter ──────────────────────────

# Underfit (github.com/dada-bots/underfit) saves adapters with full-wrapper
# layer names: DiT layers as ``model.transformer...`` and the seconds
# conditioner Linear as ``conditioners.seconds_total.embedder.embedding.1``.
# The MLX npz bakes that conditioner Linear as ``cond.seconds_total_weight``.
_COND_SECONDS_LAYER = "conditioners.seconds_total.embedder.embedding.1"


def _layer_to_npz_key(layer: str) -> str:
    """Map a checkpoint layer name to its DiT npz weight key. Strips the
    full-model ``model.`` prefix (underfit / torch-wrapper checkpoints; bare-DiT
    names pass through), maps the seconds-conditioner Linear onto its baked npz
    key, and renames ``to_local_embed.{0,2}`` → ``to_local_embed.seq.{0,2}``
    (dit_mlx.py); every other Linear/Conv1d name passes through unchanged."""
    if layer == _COND_SECONDS_LAYER:
        return "cond.seconds_total_weight"
    if layer.startswith("model."):
        layer = layer[len("model."):]
    layer = layer.replace(".to_local_embed.0", ".to_local_embed.seq.0")
    layer = layer.replace(".to_local_embed.2", ".to_local_embed.seq.2")
    return f"{layer}.weight"


def _resolve_path(path: str) -> str:
    """Accept a .safetensors file or a PEFT adapter directory (resolve to the
    adapter_model.safetensors inside it)."""
    if os.path.isdir(path):
        cand = os.path.join(path, "adapter_model.safetensors")
        if os.path.isfile(cand):
            return cand
        hits = [f for f in os.listdir(path) if f.lower().endswith(".safetensors")]
        if len(hits) == 1:
            return os.path.join(path, hits[0])
        raise LoraError(
            f"{path!r}: expected one .safetensors adapter in the directory, found {hits}"
        )
    return path


def _parse_adapter(path: str):
    """Load one adapter and return ``(adapter_type, scaling, layers)`` where
    ``layers`` maps a checkpoint layer name → its param dict (numpy float32)."""
    tensors, meta = _load_safetensors(path)

    native_marker = ".parametrizations.weight.0."
    is_native = any(native_marker in k for k in tensors)

    if is_native:
        cfg = json.loads(meta.get("lora_config", "{}")) if meta else {}
        layers = _group_native(tensors)
        rank = int(cfg.get("rank") or _infer_rank(layers))
        alpha = float(cfg.get("alpha", rank))
        adapter_type = _resolve_native_type(cfg.get("adapter_type", "lora"))
        scaling = alpha / rank
        return adapter_type, scaling, layers

    # PEFT — config lives in a sibling adapter_config.json
    peft_marker = ".lora_A.weight"
    if any(k.endswith(peft_marker) for k in tensors):
        cfg = _read_peft_config(path)
        rank = int(cfg["r"])
        alpha = float(cfg.get("lora_alpha", rank))
        use_dora = bool(cfg.get("use_dora", False))
        use_rslora = bool(cfg.get("use_rslora", False))
        adapter_type = "dora-rows" if use_dora else "lora"
        scaling = alpha / (np.sqrt(rank) if use_rslora else rank)
        layers = _group_peft(tensors)
        return adapter_type, scaling, layers

    raise LoraError(
        f"{os.path.basename(path)!r}: not a recognised LoRA (no SA3-native "
        f"parametrization keys and no PEFT lora_A/lora_B keys)"
    )


def _group_native(tensors: dict) -> dict:
    marker = ".parametrizations.weight.0."
    layers: dict[str, dict] = {}
    for k, v in tensors.items():
        if marker not in k:
            continue
        layer, _, param = k.partition(marker)
        layers.setdefault(layer, {})[param] = v
    return layers


def _group_peft(tensors: dict) -> dict:
    prefix = "base_model.model."
    layers: dict[str, dict] = {}
    for k, v in tensors.items():
        name = k[len(prefix):] if k.startswith(prefix) else k
        for suffix, param in ((".lora_A.weight", "lora_A"),
                              (".lora_B.weight", "lora_B"),
                              (".lora_magnitude_vector.weight", "magnitude")):
            if name.endswith(suffix):
                layers.setdefault(name[: -len(suffix)], {})[param] = v
                break
    return layers


def _read_peft_config(path: str) -> dict:
    base = os.path.dirname(path)
    cfg_path = os.path.join(base, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        raise LoraError(
            f"PEFT adapter at {path!r} is missing its adapter_config.json sibling"
        )
    with open(cfg_path) as fh:
        return json.load(fh)


def _infer_rank(layers: dict) -> int:
    for params in layers.values():
        if "lora_A" in params:
            return params["lora_A"].shape[0]
        if "M_xs" in params:
            return params["M_xs"].shape[0]
    raise LoraError("cannot infer LoRA rank (no lora_A / M_xs tensors)")


def _resolve_native_type(adapter_type: str) -> str:
    """Legacy 'dora' → 'dora-rows' (the paper-correct default; mirrors
    utils.resolve_adapter_type, minus the 2D-magnitude shape sniff we don't need
    because saved magnitudes are 1D)."""
    return "dora-rows" if adapter_type == "dora" else adapter_type


# ── public entry point ─────────────────────────────────────────────────────────

def merge_loras_into_weights(weights: dict, lora_paths, strength: float = 1.0,
                             log=lambda _m: None) -> dict:
    """Merge one or more LoRA adapters into ``weights`` in place.

    ``weights`` is the DiT weight dict as loaded from the npz (str → mx.array).
    ``strength`` is the application weight applied to every adapter's delta (the
    `--lora-strength` knob; matches ``application_weight`` in
    ``merge_loras_into_base_model``). Deltas are accumulated against the original
    weight, then applied once, so stacking is order-independent for linear LoRA.

    Returns a stats dict ``{"merged": int, "skipped": list[str], "adapters": int}``.
    """
    if not lora_paths:
        return {"merged": 0, "skipped": [], "adapters": 0}

    parsed = []
    for raw in lora_paths:
        path = _resolve_path(raw)
        adapter_type, scaling, layers = _parse_adapter(path)
        parsed.append((path, adapter_type, scaling, layers))
        log(f"lora: {os.path.basename(path)} — {adapter_type}, "
            f"scaling={scaling:.3f}, {len(layers)} target layers")

    # Accumulate deltas per npz key against the *original* weight. Each entry is
    # [summed_delta, restore] — the layout restorer is the same across repeats.
    accum: dict[str, list] = {}
    skipped: list[str] = []
    for path, adapter_type, scaling, layers in parsed:
        need = _PARAMS_FOR.get(adapter_type, ())
        for layer, params in layers.items():
            key = _layer_to_npz_key(layer)
            if key not in weights:
                skipped.append(layer)
                continue
            missing = [n for n in need if n not in params]
            if missing:
                raise LoraError(f"{layer}: adapter is {adapter_type} but missing {missing}")
            W0, restore = _weight_as_2d(weights[key])
            _check_shapes(layer, W0, params, adapter_type)
            merged = _merged_weight(W0, params, adapter_type, scaling)
            delta = strength * (merged - W0)
            if key in accum:
                accum[key][0] += delta
            else:
                accum[key] = [delta, restore]

    for key, (delta, restore) in accum.items():
        W0, _ = _weight_as_2d(weights[key])
        weights[key] = mx.array(restore(W0 + delta))

    if skipped:
        log(f"lora: skipped {len(skipped)} layer(s) not in this DiT "
            f"(e.g. {skipped[0]})")
    if not accum:
        log("lora: WARNING — merged 0 layers; the adapter targets nothing in this "
            "DiT (wrong base model, or unsupported target modules)")
    return {"merged": len(accum), "skipped": skipped, "adapters": len(parsed)}


def _weight_as_2d(arr):
    """Return ``(W2d, restore)`` where ``W2d`` is the PyTorch-layout 2D weight
    (fan_out, fan_in) as numpy float32, and ``restore(W2d)`` rebuilds the MLX
    layout. Linear weights are already 2D == PyTorch layout; Conv1d weights are
    stored MLX-style (out, k, in) and round-trip through PyTorch (out, in, k)."""
    np_arr = _np(arr)
    if np_arr.ndim == 2:
        return np_arr, lambda w: w.astype(np.float32)
    if np_arr.ndim == 3:
        out, k, cin = np_arr.shape
        w2d = np_arr.transpose(0, 2, 1).reshape(out, cin * k)  # (out, in*k), PyTorch order

        def restore(w):
            return w.reshape(out, cin, k).transpose(0, 2, 1).astype(np.float32)

        return w2d, restore
    raise LoraError(f"unexpected weight rank {np_arr.ndim} for a LoRA target")


# ── per-step gating (--lora … steps=A-B) ────────────────────────────────────────
#
# Every adapter type's per-layer contribution can be written in one closed form
#
#     strength·(merged − W0) = strength·(α βᵀ − 1) ⊙ W0  +  B′ @ A′
#
# with α (row) / β (col) scale vectors (absent → 1) and rank-r factors B′/A′
# computed ONCE at load, while the pristine W0 is still in the weight dict.
# That lets the sampler move between step-sets IN PLACE, with no stored W0:
#
#     W(S)  = W0 ⊙ M_S + LR_S      M_S = 1 + Σ_{i∈S} strength_i (α_i β_iᵀ − 1)
#     W_new = (W_cur − LR_S) · (M_S′ / M_S) + LR_S′
#
# computed in float32 and cast back to the model dtype. One transition costs
# ~26 ms (small DiT) / ~80 ms (medium) — a quarter of one sampling step — and
# only fires when the active set changes (≤ steps−1 times per generation).
# Every transition routes through base, so fp16 round-trip drift is bounded;
# in the one-shot CLI it is negligible. A long-lived host that cycles
# partial-interval adapters across MANY generations on one cached model should
# reload weights occasionally (drift grows ~1e-5 rel per boundary, decelerating).

def parse_lora_spec(tokens, default_strength: float = 1.0) -> dict:
    """Parse one ``--lora`` group: ``PATH [strength=S] [steps=RANGE]``.

    RANGE grammar (1-based, inclusive): ``3`` (just step 3), ``2-8``, ``2-``
    (2..last), ``-4`` (1..4). Returns ``{"path", "strength", "steps"}`` where
    ``steps`` is None (all steps) or a raw ``(lo|None, hi|None)`` pair resolved
    against the actual ``--steps`` by :func:`resolve_steps`.
    """
    if not tokens:
        raise LoraError("--lora needs an adapter path")
    spec = {"path": tokens[0], "strength": default_strength, "steps": None}
    for tok in tokens[1:]:
        key, eq, val = tok.partition("=")
        if not eq or key not in ("strength", "steps"):
            raise LoraError(
                f"--lora: unexpected token {tok!r} — one adapter per --lora flag; "
                f"options after the path are strength=S and steps=RANGE "
                f"(e.g. --lora a.safetensors strength=0.8 steps=2-8)"
            )
        if key == "strength":
            try:
                spec["strength"] = float(val)
            except ValueError:
                raise LoraError(f"--lora: bad strength {val!r}") from None
        else:
            spec["steps"] = _parse_step_range(val)
    return spec


def _parse_step_range(s: str):
    def _int(part, what):
        try:
            v = int(part)
        except ValueError:
            raise LoraError(f"--lora steps={s!r}: {what} is not an integer") from None
        if v < 1:
            raise LoraError(f"--lora steps={s!r}: steps are 1-based")
        return v

    if "-" in s:
        lo_s, _, hi_s = s.partition("-")
        lo = _int(lo_s, "min step") if lo_s else None
        hi = _int(hi_s, "max step") if hi_s else None
        if lo is None and hi is None:
            raise LoraError(f"--lora steps={s!r}: empty range")
    else:
        lo = hi = _int(s, "step")
    if lo is not None and hi is not None and lo > hi:
        raise LoraError(f"--lora steps={s!r}: min > max")
    return (lo, hi)


def resolve_steps(raw, num_steps: int):
    """Raw 1-based ``(lo|None, hi|None)`` → 0-based inclusive ``(lo0, hi0)``
    clamped to the schedule. None (no steps= given) covers every step. Returns
    None when the range misses the schedule entirely."""
    if raw is None:
        return (0, num_steps - 1)
    lo, hi = raw
    lo0 = (lo or 1) - 1
    hi0 = min((hi or num_steps) - 1, num_steps - 1)
    if lo0 > hi0 or lo0 >= num_steps:
        return None
    return (lo0, hi0)


def _delta_form(W0: np.ndarray, p: dict, adapter_type: str, scaling: float,
                strength: float):
    """Closed-form ``(row, col, Bp, Ap)`` float32 for one layer such that
    ``strength·(merged − W0) == strength·(row·colᵀ − 1)⊙W0 + Bp @ Ap``
    (row/col None when the type has no magnitude on that axis). Algebraically
    identical to :func:`_merged_weight`, just refactored around W0."""
    if adapter_type.endswith("-xs"):
        rank = p["M_xs"].shape[0]
        U, V = _svd_bases(W0, rank)
        B, A = U @ p["M_xs"], V.T
    else:
        B, A = p["lora_B"], p["lora_A"]
    if adapter_type in ("lora", "lora-xs"):
        return None, None, (strength * scaling) * B, A
    V_ = W0 + scaling * (B @ A)
    if adapter_type in ("dora-rows", "dora-rows-xs"):
        c = np.atleast_1d(np.squeeze(p["magnitude"])) / (np.linalg.norm(V_, axis=1) + 1e-12)
        return (c.astype(np.float32), None,
                (strength * scaling) * (c[:, None] * B), A)
    if adapter_type in ("dora-cols", "dora-cols-xs"):
        c = np.atleast_1d(np.squeeze(p["magnitude"])) / (np.linalg.norm(V_, axis=0) + 1e-12)
        return (None, c.astype(np.float32),
                (strength * scaling) * B, A * c[None, :])
    if adapter_type in ("bora", "bora-xs"):
        alpha = p["magnitude_r"] / (np.linalg.norm(V_, axis=1) + 1e-12)
        inter = alpha[:, None] * V_
        beta = p["magnitude_c"] / (np.linalg.norm(inter, axis=0) + 1e-12)
        return (alpha.astype(np.float32), beta.astype(np.float32),
                (strength * scaling) * (alpha[:, None] * B), A * beta[None, :])
    raise LoraError(f"unknown adapter_type {adapter_type!r}")


def _as2d_mx(w: mx.array):
    """MLX-side analogue of _weight_as_2d: (W2d, shape3|None)."""
    if w.ndim == 2:
        return w, None
    if w.ndim == 3:
        out, k, cin = w.shape
        return w.transpose(0, 2, 1).reshape(out, cin * k), (out, k, cin)
    raise LoraError(f"unexpected weight rank {w.ndim} for a LoRA target")


def _from2d_mx(w2: mx.array, shape3):
    if shape3 is None:
        return w2
    out, k, cin = shape3
    return w2.reshape(out, cin, k).transpose(0, 2, 1)


def _walk_module(root, path: str):
    """Resolve 'transformer.layers.3.ff.ff.2' to the live module — integer
    segments index plain python list containers, names use getattr."""
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


_EVAL_CHUNK = 16  # layers per mx.eval during a transition (26/80 ms per boundary)


class LoraStepPlan:
    """Per-step LoRA gating state for partial-interval adapters.

    Built by :func:`prepare_loras` while the pristine weight dict is available;
    ``attach(model)`` resolves the live modules after construction; ``sync(i)``
    (wired as the sampler's ``before_step``) retargets weights in place whenever
    the active adapter set changes between steps.
    """

    def __init__(self, num_steps: int, log=lambda _m: None):
        self.num_steps = num_steps
        self.log = log
        self.adapters: list[dict] = []   # {"name","strength","lo","hi"}
        # npz key → {"parts": {ai: (row|None, col|None, Bp, Ap) mx fp32}}
        self.layers: dict[str, dict] = {}
        self.step_sets: tuple = ()
        self.mods: dict[str, object] = {}
        self.current: frozenset = frozenset()

    # ── build-time ──────────────────────────────────────────────────────────

    def add_adapter(self, name: str, strength: float, lo: int, hi: int) -> int:
        self.adapters.append({"name": name, "strength": strength, "lo": lo, "hi": hi})
        return len(self.adapters) - 1

    def add_layer_part(self, key: str, ai: int, row, col, Bp, Ap) -> None:
        parts = self.layers.setdefault(key, {"parts": {}})["parts"]
        parts[ai] = (
            None if row is None else mx.array(row),
            None if col is None else mx.array(col),
            mx.array(Bp), mx.array(Ap),
        )

    def finalize(self, weights: dict) -> None:
        """Compute per-step active sets, validate masks, and apply the step-1
        state to the weight dict (still fp32-safe: results are written back
        float32; the loader casts to the model dtype)."""
        self.step_sets = tuple(
            frozenset(ai for ai, a in enumerate(self.adapters)
                      if a["lo"] <= i <= a["hi"])
            for i in range(self.num_steps)
        )
        self._validate_masks()
        n_bound = sum(1 for i in range(1, self.num_steps)
                      if self.step_sets[i] != self.step_sets[i - 1])
        init = self.step_sets[0]
        if init:
            self._transition_dict(weights, frozenset(), init)
        self.current = init
        for ai, a in enumerate(self.adapters):
            self.log(f"lora: {a['name']} gated to steps {a['lo'] + 1}-{a['hi'] + 1} "
                     f"(strength {a['strength']:g})")
        self.log(f"lora: step plan over {len(self.layers)} layer(s), "
                 f"{n_bound} in-sampler boundary transition(s)")

    def _validate_masks(self) -> None:
        """Every set that appears must have an invertible mask (|M| bounded away
        from 0) — checked once in numpy at build time so sync() needs no guard."""
        for S in set(self.step_sets):
            for key, info in self.layers.items():
                m = None
                for ai in S:
                    part = info["parts"].get(ai)
                    if part is None or (part[0] is None and part[1] is None):
                        continue
                    row, col, _, _ = part
                    rv = np.array(row) if row is not None else np.ones(1, np.float32)
                    cv = np.array(col) if col is not None else np.ones(1, np.float32)
                    term = self.adapters[ai]["strength"] * (rv[:, None] * cv[None, :] - 1.0)
                    m = term if m is None else m + term
                if m is not None and np.abs(1.0 + m).min() < 1e-3:
                    raise LoraError(
                        f"{key}: near-singular magnitude mask for step set "
                        f"{sorted(S)} — this strength/adapter combination cannot "
                        f"be gated in place (try full-interval merge)")

    # ── transition math ─────────────────────────────────────────────────────

    def _lowrank(self, parts: dict, active: list):
        facs = [(parts[ai][2], parts[ai][3]) for ai in active]
        if not facs:
            return None
        if len(facs) == 1:
            return facs[0][0] @ facs[0][1]
        B = mx.concatenate([f[0] for f in facs], axis=1)
        A = mx.concatenate([f[1] for f in facs], axis=0)
        return B @ A

    def _mask(self, parts: dict, active: list):
        m = None
        for ai in active:
            row, col, _, _ = parts[ai]
            if row is None and col is None:
                continue
            rv = row[:, None] if row is not None else 1.0
            cv = col[None, :] if col is not None else 1.0
            term = self.adapters[ai]["strength"] * (rv * cv - 1.0)
            m = term if m is None else m + term
        return None if m is None else 1.0 + m

    def _retarget(self, w: mx.array, parts: dict, sub: list, add: list,
                  out_dtype) -> mx.array:
        w2, shape3 = _as2d_mx(w)
        x = w2.astype(mx.float32)
        lr = self._lowrank(parts, sub)
        if lr is not None:
            x = x - lr
        m_s, m_t = self._mask(parts, sub), self._mask(parts, add)
        if m_s is not None or m_t is not None:
            x = x * ((m_t if m_t is not None else 1.0) /
                     (m_s if m_s is not None else 1.0))
        lr = self._lowrank(parts, add)
        if lr is not None:
            x = x + lr
        if out_dtype is not None:
            x = x.astype(out_dtype)
        return _from2d_mx(x, shape3)

    def _transition_dict(self, weights: dict, S: frozenset, T: frozenset) -> None:
        pending = []
        for key, info in self.layers.items():
            sub = sorted(ai for ai in S if ai in info["parts"])
            add = sorted(ai for ai in T if ai in info["parts"])
            if sub == add:
                continue
            weights[key] = self._retarget(weights[key], info["parts"], sub, add,
                                          out_dtype=None)
            pending.append(weights[key])
            if len(pending) >= _EVAL_CHUNK:
                mx.eval(pending)
                pending = []
        if pending:
            mx.eval(pending)

    def _transition_model(self, S: frozenset, T: frozenset) -> None:
        pending = []
        for key, info in self.layers.items():
            mod = self.mods[key]
            sub = sorted(ai for ai in S if ai in info["parts"])
            add = sorted(ai for ai in T if ai in info["parts"])
            if sub == add:
                continue
            mod.weight = self._retarget(mod.weight, info["parts"], sub, add,
                                        out_dtype=mod.weight.dtype)
            pending.append(mod.weight)
            if len(pending) >= _EVAL_CHUNK:
                mx.eval(pending)
                pending = []
        if pending:
            mx.eval(pending)

    # ── run-time ────────────────────────────────────────────────────────────

    def attach(self, model) -> None:
        """Resolve live modules for every managed layer (direct attribute
        rebinding is a plain python write in mlx.nn.Module — no copy, no GPU
        work; nothing in this pipeline is mx.compile'd)."""
        for key in self.layers:
            path = key[: -len(".weight")] if key.endswith(".weight") else key
            try:
                self.mods[key] = _walk_module(model, path)
            except (AttributeError, IndexError, TypeError) as e:
                raise LoraError(f"lora step plan: cannot resolve module {path!r} "
                                f"in the DiT: {e}") from e

    def sync(self, i: int) -> None:
        """Sampler ``before_step`` hook: make the weights match step i's set."""
        tgt = self.step_sets[i] if i < self.num_steps else frozenset()
        if tgt == self.current:
            return
        self._transition_model(self.current, tgt)
        self.current = tgt

    def clear(self) -> None:
        """Restore the attached model to its base weights (exact-inverse
        transition to the empty set). Used by long-lived hosts (gradio) before
        applying a different adapter configuration to a cached model."""
        if self.current:
            self._transition_model(self.current, frozenset())
            self.current = frozenset()


def prepare_loras(weights: dict, specs: list, num_steps: int,
                  log=lambda _m: None, gate_all: bool = False):
    """Entry point for the step-gated path. ``specs`` come from
    :func:`parse_lora_spec`. Full-interval adapters are merged permanently via
    :func:`merge_loras_into_weights` (bit-identical to the plain --lora path);
    partial-interval ones become a :class:`LoraStepPlan` whose step-1 state is
    applied to ``weights`` here. Returns the plan, or None when nothing is
    step-gated. The seconds-conditioner delta (underfit adapters) cannot be
    step-gated — conditioning is computed once per generation — so it is left
    to :func:`apply_conditioner_lora` and excluded from the plan.

    gate_all=True routes full-interval adapters through the plan too (they just
    never transition mid-run). Long-lived hosts use this so plan.clear() can
    restore the base weights for a cached model — permanent merges can't be
    undone in place."""
    full, partial = [], []
    for s in specs:
        rs = resolve_steps(s["steps"], num_steps)
        if rs is None:
            log(f"lora: {os.path.basename(s['path'])} steps="
                f"{s['steps']} misses the {num_steps}-step schedule — inactive")
            continue
        if rs == (0, num_steps - 1) and not gate_all:
            full.append(s)
        else:
            partial.append((s, rs))

    for s in full:
        stats = merge_loras_into_weights(weights, [s["path"]],
                                         strength=s["strength"], log=log)
        log(f"lora: merged {stats['merged']} layer(s) (all steps)")

    if not partial:
        return None

    plan = LoraStepPlan(num_steps, log=log)
    for s, (lo, hi) in partial:
        path = _resolve_path(s["path"])
        adapter_type, scaling, layers = _parse_adapter(path)
        ai = plan.add_adapter(os.path.basename(path), s["strength"], lo, hi)
        skipped = []
        matched = has_cond = 0
        for layer, params in layers.items():
            key = _layer_to_npz_key(layer)
            if key == "cond.seconds_total_weight":
                has_cond = 1
                log(f"lora: {os.path.basename(path)} conditioner delta applies "
                    f"to the whole generation (conditioning runs once, before "
                    f"sampling — steps= does not gate it)")
                continue
            if key not in weights:
                skipped.append(layer)
                continue
            need = _PARAMS_FOR.get(adapter_type, ())
            missing = [n for n in need if n not in params]
            if missing:
                raise LoraError(f"{layer}: adapter is {adapter_type} but missing {missing}")
            W0, _ = _weight_as_2d(weights[key])
            _check_shapes(layer, W0, params, adapter_type)
            row, col, Bp, Ap = _delta_form(W0, params, adapter_type, scaling,
                                           s["strength"])
            plan.add_layer_part(key, ai, row, col, Bp, Ap)
            matched += 1
        if not matched and not has_cond:
            raise LoraError(
                f"{os.path.basename(path)}: none of this adapter's "
                f"{len(layers)} layer(s) exist in the target DiT — it was "
                f"trained for a different base model")
        if skipped:
            log(f"lora: skipped {len(skipped)} layer(s) not in this DiT "
                f"(e.g. {skipped[0]})")
    if not plan.layers:
        log("lora: WARNING — no step-gated layers matched this DiT")
        return None
    plan.finalize(weights)
    return plan


def apply_conditioner_lora(W: mx.array, specs: list, log=lambda _m: None):
    """Apply the seconds-conditioner delta (underfit adapters carry one) to the
    LIVE conditioner weight — the DiT weight-dict copy of ``cond.*`` is inert
    because the conditioner is loaded straight from the npz file. Interval-
    agnostic: conditioning runs once per generation. Returns ``(W_new, n)``
    with n = number of adapters that contributed."""
    W0 = np.array(W.astype(mx.float32), dtype=np.float32)
    acc = None
    n = 0
    for s in specs:
        path = _resolve_path(s["path"])
        adapter_type, scaling, layers = _parse_adapter(path)
        params = layers.get(_COND_SECONDS_LAYER)
        if params is None:
            continue
        _check_shapes(_COND_SECONDS_LAYER, W0, params, adapter_type)
        delta = s["strength"] * (_merged_weight(W0, params, adapter_type, scaling) - W0)
        acc = delta if acc is None else acc + delta
        n += 1
    if acc is None:
        return W, 0
    return mx.array(W0 + acc), n
