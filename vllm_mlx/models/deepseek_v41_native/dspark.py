"""Product-owned DeepSeek V4.1 DSpark draft runtime.

This module consumes checkpoint data only. It never imports or executes Python
from a model directory. The supported contract is the released three-stage MTP
sidecar with greedy speculative decoding; sampling is deliberately excluded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx

from .fakequant import fake_quant_fp8_ue8m0

_REQUIRED_CONFIG = {
    "dspark_block_size",
    "dspark_n_routed_experts",
    "dspark_noise_token_id",
    "dspark_num_experts_per_tok",
    "dspark_target_layer_ids",
    "hc_eps",
    "hc_mult",
    "hc_sinkhorn_iters",
    "head_dim",
    "hidden_size",
    "norm_topk_prob",
    "num_attention_heads",
    "o_groups",
    "qk_rope_head_dim",
    "rms_norm_eps",
    "rope_theta",
    "routed_scaling_factor",
    "sliding_window",
    "swiglu_limit",
}

_RELEASE_ARCHITECTURE = {
    "dspark_block_size": 5,
    "dspark_n_routed_experts": 128,
    "dspark_num_experts_per_tok": 3,
    "hc_mult": 4,
    "head_dim": 512,
    "hidden_size": 5120,
    "num_attention_heads": 64,
    "o_groups": 8,
    "qk_rope_head_dim": 64,
}


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _safe_shard(root: Path, name: object) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("DSpark shard names must be non-empty basenames")
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.is_symlink():
        # Standard Hub snapshots are symlink forests into the same repo's
        # content-addressed blobs directory. Accept only that exact shape;
        # arbitrary local sidecar symlinks remain rejected so a crafted index
        # cannot escape its artifact boundary.
        resolved = path.resolve(strict=True)
        blobs = (root.parents[1] / "blobs").resolve()
        try:
            resolved.relative_to(blobs)
        except ValueError as exc:
            raise ValueError(
                f"DSpark shard symlink leaves its Hub repository: {name}"
            ) from exc
        if not resolved.is_file():  # pragma: no cover - guarded filesystem race
            raise ValueError(f"DSpark shard symlink is not a file: {name}")
    return path


def _validate_config(config: dict) -> None:
    missing = sorted(_REQUIRED_CONFIG - config.keys())
    if missing:
        raise ValueError(f"DSpark config is missing: {', '.join(missing)}")
    positive = (
        "dspark_block_size",
        "dspark_n_routed_experts",
        "dspark_num_experts_per_tok",
        "hc_mult",
        "hc_sinkhorn_iters",
        "head_dim",
        "hidden_size",
        "num_attention_heads",
        "o_groups",
        "qk_rope_head_dim",
        "sliding_window",
    )
    for key in positive:
        if isinstance(config[key], bool) or not isinstance(config[key], int):
            raise ValueError(f"DSpark config {key} must be an integer")
        if config[key] <= 0:
            raise ValueError(f"DSpark config {key} must be positive")
    if config["dspark_num_experts_per_tok"] > config["dspark_n_routed_experts"]:
        raise ValueError("DSpark top-k exceeds routed expert count")
    if config["qk_rope_head_dim"] > config["head_dim"]:
        raise ValueError("DSpark RoPE width exceeds attention head width")
    if config.get("model_type") not in (
        "deepseek_v4",
        "deepseek_v41",
        "deepseek_v41_text",
    ):
        raise ValueError("DSpark sidecar model_type is not DeepSeek V4.1")
    for key, expected in _RELEASE_ARCHITECTURE.items():
        if config[key] != expected:
            raise ValueError(
                f"unsupported DSpark architecture: {key}={config[key]!r}, "
                f"expected {expected}"
            )
    for key in ("hc_eps", "rms_norm_eps", "rope_theta", "routed_scaling_factor"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"DSpark config {key} must be numeric")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"DSpark config {key} must be finite and positive")
    targets = config["dspark_target_layer_ids"]
    if not isinstance(targets, (list, tuple)) or len(targets) != 3:
        raise ValueError("DSpark requires exactly three target hidden layers")


def _required_tensors(config: dict) -> set[str]:
    required = {"mtp.0.main_norm.weight"}
    required.update(
        f"mtp.0.main_proj.{suffix}" for suffix in ("weight", "scales", "biases")
    )
    for stage in range(3):
        base = f"mtp.{stage}"
        required.update(
            f"{base}.{name}"
            for name in (
                "attn.attn_sink",
                "attn.kv_norm.weight",
                "attn.q_norm.weight",
                "attn_norm.weight",
                "ffn.gate.bias",
                "ffn.gate.weight",
                "ffn_norm.weight",
                "hc_attn_base",
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_ffn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
            )
        )
        for projection in (
            "attn.wkv",
            "attn.wo_a",
            "attn.wo_b",
            "attn.wq_a",
            "attn.wq_b",
        ):
            required.update(
                f"{base}.{projection}.{suffix}"
                for suffix in ("weight", "scales", "biases")
            )
        for projection in ("w1", "w2", "w3"):
            required.update(
                f"{base}.ffn.shared_experts.{projection}.{suffix}"
                for suffix in ("weight", "scales", "biases")
            )
            for expert in range(config["dspark_n_routed_experts"]):
                required.update(
                    f"{base}.ffn.experts.{expert}.{projection}.{suffix}"
                    for suffix in ("weight", "scales", "biases")
                )
    final = "mtp.2"
    required.add(final + ".norm.weight")
    for projection in ("confidence_head.proj", "markov_head.embed", "markov_head.head"):
        required.update(
            f"{final}.{projection}.{suffix}"
            for suffix in ("weight", "scales", "biases")
        )
    return required


class DSparkWeights:
    """Strict loader for MTP tensors plus target-owned shared I/O weights."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        if not self.path.is_dir():
            raise NotADirectoryError(self.path)
        raw = _read_object(self.path / "config.json")
        nested = raw.get("text_config")
        self.config = {**raw, **nested} if isinstance(nested, dict) else dict(raw)
        _validate_config(self.config)
        index = _read_object(self.path / "model.safetensors.index.json")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model index must contain a non-empty weight_map")
        mtp_map = {}
        for key, shard in weight_map.items():
            if not isinstance(key, str):
                raise ValueError("model index tensor names must be strings")
            if key.startswith("mtp."):
                if not isinstance(shard, str):
                    raise ValueError("model index shard names must be strings")
                mtp_map[key] = shard
        if not mtp_map:
            raise ValueError("model index contains no MTP tensors")
        self._arrays: dict[str, mx.array] = {}
        self.mtp_file_bytes = 0
        for shard_name in sorted(set(mtp_map.values())):
            shard = _safe_shard(self.path, shard_name)
            self.mtp_file_bytes += shard.stat().st_size
            loaded = mx.load(str(shard))
            if not isinstance(loaded, dict):
                raise ValueError(f"DSpark shard is not a tensor map: {shard.name}")
            for key, value in loaded.items():
                if key.startswith("mtp."):
                    if key in self._arrays:
                        raise ValueError(f"duplicate DSpark tensor: {key}")
                    if value.dtype not in (mx.bfloat16, mx.float32, mx.uint32):
                        raise ValueError(f"unsupported DSpark tensor dtype: {key}")
                    if not value.shape or any(int(size) <= 0 for size in value.shape):
                        raise ValueError(f"invalid DSpark tensor shape: {key}")
                    self._arrays[key] = value
        if set(self._arrays) != set(mtp_map):
            missing = sorted(set(mtp_map) - set(self._arrays))
            extra = sorted(set(self._arrays) - set(mtp_map))
            raise ValueError(
                f"DSpark index/shard mismatch (missing={missing[:1]}, extra={extra[:1]})"
            )
        missing_contract = sorted(_required_tensors(self.config) - set(self._arrays))
        if missing_contract:
            raise ValueError(
                f"DSpark tensor contract is missing: {missing_contract[0]}"
            )
        self.entries = {
            key: {"shape": tuple(value.shape)} for key, value in self._arrays.items()
        }
        self.resident: dict[str, mx.array] = {}
        self.resident_bytes = 0
        self.q = raw.get("quantization") or raw.get("quantization_config", {})
        if not isinstance(self.q, dict):
            raise ValueError("DSpark quantization config must be an object")

    def attach_target(self, model) -> None:
        """Share target embedding/head arrays without loading their sidecar shards."""
        for base, module in (("embed", model.embed), ("head", model.head)):
            attached = 0
            for suffix in ("weight", "scales", "biases", "bias"):
                value = getattr(module, suffix, None)
                if value is not None:
                    key = f"{base}.{suffix}"
                    self._arrays[key] = value
                    self.entries[key] = {"shape": tuple(value.shape)}
                    self.resident[key] = value
                    attached += 1
            if not attached:
                raise ValueError(f"target model has no shareable {base} weights")

    def pin_mtp(self, reserve_gb: float = 8.0) -> int:
        import psutil

        keys = sorted(key for key in self._arrays if key.startswith("mtp."))
        total = sum(int(self._arrays[key].nbytes) for key in keys)
        reserve = int(reserve_gb * 1e9)
        device_free = (
            int(mx.device_info()["max_recommended_working_set_size"])
            - mx.get_active_memory()
        )
        safe = min(int(psutil.virtual_memory().available), device_free) - reserve
        if total > safe:
            raise MemoryError(
                f"MTP needs {total / 1e9:.2f} GB but only {safe / 1e9:.2f} GB "
                f"remains after the {reserve_gb:.1f} GB safety reserve"
            )
        values = [self._arrays[key] for key in keys]
        mx.eval(*values)
        for key, value in zip(keys, values):
            if key not in self.resident:
                self.resident[key] = value
                self.resident_bytes += value.nbytes
        mx.clear_cache()
        return total

    def read(self, key: str, rows=None, _pin: bool = False):
        del _pin
        try:
            value = self._arrays[key]
        except KeyError as exc:
            raise KeyError(f"missing DSpark tensor: {key}") from exc
        return value if rows is None else value[mx.array(rows, dtype=mx.int32)]

    def release_tensor(self, key: str) -> None:
        """Drop a tensor after a product-owned packed replacement is evaluated."""
        value = self.resident.pop(key, None)
        if value is not None:
            self.resident_bytes -= value.nbytes
        self._arrays.pop(key, None)
        self.entries.pop(key, None)

    def quant_config(self, base: str) -> dict:
        modules = self.q.get("modules", {})
        specific = modules.get(base) if isinstance(modules, dict) else None
        source = specific if isinstance(specific, dict) else self.q
        try:
            return {
                "bits": int(source["bits"]),
                "group_size": int(source["group_size"]),
                "mode": source.get("mode", self.q.get("mode", "affine")),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"missing quantization contract for {base}") from exc

    def linear(self, base: str, x):
        if base + ".scales" in self._arrays:
            bias = (
                self.read(base + ".biases")
                if base + ".biases" in self._arrays
                else None
            )
            y = mx.quantized_matmul(
                x,
                self.read(base + ".weight"),
                self.read(base + ".scales"),
                bias,
                transpose=True,
                **self.quant_config(base),
            )
        else:
            weight = self.read(base + ".weight")
            y = x.astype(weight.dtype) @ weight.T
        if base + ".bias" in self._arrays:
            y = y + self.read(base + ".bias")
        return y.astype(x.dtype)

    def embedding(self, base: str, ids):
        ids = mx.array(ids, dtype=mx.int32)
        weight = self.read(base + ".weight", ids)
        if base + ".scales" in self._arrays:
            scales = self.read(base + ".scales", ids)
            biases = (
                self.read(base + ".biases", ids)
                if base + ".biases" in self._arrays
                else None
            )
            weight = mx.dequantize(weight, scales, biases, **self.quant_config(base))
        return weight

    def grouped(self, base: str, x, groups: int):
        rows = self.entries[base + ".weight"]["shape"][0] // groups
        output = []
        for group in range(groups):
            ids = mx.arange(group * rows, (group + 1) * rows, dtype=mx.int32)
            weight = self.read(base + ".weight", ids)
            if base + ".scales" in self._arrays:
                output.append(
                    mx.quantized_matmul(
                        x[group : group + 1],
                        weight,
                        self.read(base + ".scales", ids),
                        self.read(base + ".biases", ids),
                        transpose=True,
                        **self.quant_config(base),
                    )
                )
            else:
                output.append(x[group : group + 1] @ weight.T)
        return mx.concatenate(output, axis=-1)


# Keep the benchmark/runtime adapter contract intentionally tiny: a provider
# exposes ``Weights`` and ``DSpark``. The descriptive class name remains useful
# to callers that want to construct the sidecar loader directly.
Weights = DSparkWeights


def _rms_impl(x, weight, eps: float):
    value = x.astype(mx.float32)
    return (
        value * mx.rsqrt(mx.mean(value * value, axis=-1, keepdims=True) + eps) * weight
    ).astype(x.dtype)


def _hc_mixes_impl(x, weight, scale, base, hc, iterations, eps, norm_eps):
    flat = x.reshape(*x.shape[:-2], -1).astype(mx.float32)
    mixed = (flat @ weight.T) * mx.rsqrt(
        mx.mean(flat * flat, axis=-1, keepdims=True) + norm_eps
    )
    pre = mx.sigmoid(mixed[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * mx.sigmoid(mixed[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    combine = (mixed[..., 2 * hc :] * scale[2] + base[2 * hc :]).reshape(
        *flat.shape[:-1], hc, hc
    )
    combine = mx.softmax(combine, axis=-1) + eps
    combine = combine / (mx.sum(combine, axis=-2, keepdims=True) + eps)
    for _ in range(iterations - 1):
        combine = combine / (mx.sum(combine, axis=-1, keepdims=True) + eps)
        combine = combine / (mx.sum(combine, axis=-2, keepdims=True) + eps)
    return pre, post, combine


_compiled_rms = mx.compile(_rms_impl)
_compiled_hc_mixes = mx.compile(_hc_mixes_impl)


def _hc_pre(x, pre):
    return mx.sum(x.astype(mx.float32) * pre[..., None], axis=-2).astype(x.dtype)


def _hc_post(x, residual, post, combine):
    carried = mx.einsum("...ij,...id->...jd", combine, residual.astype(mx.float32))
    return (post[..., None] * x[..., None, :] + carried).astype(x.dtype)


def _rotary(x, position, dim: int, base: float, inverse: bool = False):
    frequency = 1 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    angle = frequency * position * (-1 if inverse else 1)
    tail = x[..., -dim:].astype(mx.float32).reshape(*x.shape[:-1], dim // 2, 2)
    real, imag = tail[..., 0], tail[..., 1]
    rotated = mx.stack(
        (
            real * mx.cos(angle) - imag * mx.sin(angle),
            real * mx.sin(angle) + imag * mx.cos(angle),
        ),
        axis=-1,
    )
    return mx.concatenate(
        (x[..., :-dim], rotated.reshape(*x.shape[:-1], dim).astype(x.dtype)), axis=-1
    )


def quantize_cache(x, bits: int, group: int, scale_format: str = "e8m0"):
    if (bits, group, scale_format) != (8, 32, "e8m0"):
        raise ValueError("DSpark supports only FP8/e8m0 cache blocks of 32")
    return fake_quant_fp8_ue8m0(x, block=group)


def draft_attention(query, key_value, sink):
    scores = (
        mx.einsum("thd,kd->thk", query.astype(mx.float32), key_value.astype(mx.float32))
        * query.shape[-1] ** -0.5
    )
    sinks = mx.broadcast_to(sink[None, :, None], (*scores.shape[:-1], 1))
    probabilities = mx.softmax(mx.concatenate((scores, sinks), axis=-1), axis=-1)
    return mx.einsum(
        "thk,kd->thd", probabilities[..., :-1], key_value.astype(mx.float32)
    ).astype(query.dtype)


class _DraftAdapter:
    def __init__(self, weights: DSparkWeights, config: dict):
        self.w, self.c = weights, config

    def norm(self, name, x):
        return _compiled_rms(x, self.w.read(name + ".weight"), self.c["rms_norm_eps"])

    def rotate(self, x, position, ratio, inverse=False):
        if ratio:
            raise ValueError("DSpark draft attention does not use compressed RoPE")
        return _rotary(
            x, position, self.c["qk_rope_head_dim"], self.c["rope_theta"], inverse
        )

    def mixes(self, base, kind, x):
        return _compiled_hc_mixes(
            x,
            self.w.read(f"{base}.hc_{kind}_fn"),
            self.w.read(f"{base}.hc_{kind}_scale"),
            self.w.read(f"{base}.hc_{kind}_base"),
            self.c["hc_mult"],
            self.c["hc_sinkhorn_iters"],
            self.c["hc_eps"],
            self.c["rms_norm_eps"],
        )

    def expert(self, base, x, routing=None):
        gate = self.w.linear(base + ".w1", x).astype(mx.float32)
        up = self.w.linear(base + ".w3", x).astype(mx.float32)
        limit = self.c["swiglu_limit"]
        if limit > 0:
            gate, up = mx.minimum(gate, limit), mx.clip(up, -limit, limit)
        hidden = gate * mx.sigmoid(gate) * up
        if routing is not None:
            hidden = hidden * routing
        return self.w.linear(base + ".w2", hidden.astype(x.dtype))

    def moe(self, base, x):
        logits = (
            x.astype(mx.float32)
            @ self.w.read(base + ".gate.weight").astype(mx.float32).T
        )
        scores = mx.sqrt(mx.logaddexp(logits, mx.zeros_like(logits)))
        topk = self.c["dspark_num_experts_per_tok"]
        picks = mx.argsort(scores + self.w.read(base + ".gate.bias"), axis=-1)[
            ..., -topk:
        ]
        selected = mx.take_along_axis(scores, picks, axis=-1)
        if self.c["norm_topk_prob"] and topk > 1:
            selected = selected / (mx.sum(selected, axis=-1, keepdims=True) + 1e-20)
        selected = selected * self.c["routed_scaling_factor"]
        mx.eval(picks, selected)
        result = mx.zeros_like(x).astype(mx.float32)
        for index, weight in zip(picks.tolist()[0], selected.tolist()[0]):
            result = result + self.expert(f"{base}.experts.{index}", x, weight).astype(
                mx.float32
            )
        return (
            result + self.expert(base + ".shared_experts", x).astype(mx.float32)
        ).astype(x.dtype)


class DSpark:
    """Greedy draft state driven only by contiguous verified target states."""

    def __init__(self, target, pin_weights: bool = True):
        self.w, self.c = target.w, target.c
        self.layers = sorted(
            {int(key.split(".")[1]) for key in self.w.entries if key.startswith("mtp.")}
        )
        if not self.layers or self.layers != list(range(len(self.layers))):
            raise ValueError("missing contiguous DSpark MTP stages")
        self.block_size = self.c["dspark_block_size"]
        self.adapter = _DraftAdapter(self.w, self.c)
        self.position = -1
        self.windows: dict[int, list[Any]] = {layer: [] for layer in self.layers}
        if pin_weights:
            self.w.pin_mtp()

    def observe(self, main_hidden, position: int):
        if position != self.position + 1:
            raise ValueError("DSpark observations must be contiguous target positions")
        main = self.adapter.norm(
            "mtp.0.main_norm", self.w.linear("mtp.0.main_proj", main_hidden)
        )
        values = []
        for layer in self.layers:
            base = f"mtp.{layer}.attn"
            key_value = self.adapter.norm(
                base + ".kv_norm", self.w.linear(base + ".wkv", main)
            )
            key_value = quantize_cache(
                self.adapter.rotate(key_value, position, 0), 8, 32
            )
            self.windows[layer] = (self.windows[layer] + [key_value])[
                -self.c["sliding_window"] :
            ]
            values.append(key_value)
        mx.eval(*values)
        self.position = position

    def attention(self, base, x, stage: int):
        count = x.shape[0]
        query = self.adapter.norm(base + ".q_norm", self.w.linear(base + ".wq_a", x))
        query = self.w.linear(base + ".wq_b", query).reshape(
            count, self.c["num_attention_heads"], self.c["head_dim"]
        )
        key_value = self.adapter.norm(
            base + ".kv_norm", self.w.linear(base + ".wkv", x)
        )
        query = mx.stack(
            [
                self.adapter.rotate(query[i], self.position + 1 + i, 0)
                for i in range(count)
            ]
        )
        key_value = mx.concatenate(
            [
                quantize_cache(
                    self.adapter.rotate(key_value[i : i + 1], self.position + 1 + i, 0),
                    8,
                    32,
                )
                for i in range(count)
            ]
        )
        keys = mx.concatenate([*self.windows[stage], key_value])
        output = draft_attention(query, keys, self.w.read(base + ".attn_sink"))
        output = mx.stack(
            [
                self.adapter.rotate(output[i], self.position + 1 + i, 0, inverse=True)
                for i in range(count)
            ]
        )
        projected = mx.concatenate(
            [
                self.w.grouped(
                    base + ".wo_a",
                    output[i].reshape(self.c["o_groups"], -1),
                    self.c["o_groups"],
                )
                for i in range(count)
            ]
        )
        return self.w.linear(base + ".wo_b", projected)

    def propose(self, next_token: int):
        if self.position < 1:
            raise ValueError("seed at least two verified positions before drafting")
        ids = [next_token] + [self.c["dspark_noise_token_id"]] * (self.block_size - 1)
        hidden = mx.repeat(
            self.w.embedding("embed", ids)[:, None, :], self.c["hc_mult"], axis=1
        )
        pre = mx.broadcast_to(
            mx.array([[1.0] + [0.0] * (self.c["hc_mult"] - 1)]),
            (self.block_size, self.c["hc_mult"]),
        )
        for stage in self.layers:
            base = f"mtp.{stage}"
            attn_pre, post, combine = self.adapter.mixes(base, "attn", hidden)
            attention = self.attention(
                base + ".attn",
                self.adapter.norm(base + ".attn_norm", _hc_pre(hidden, pre)),
                stage,
            )
            hidden = _hc_post(attention, hidden, post, combine)
            pre, post, combine = self.adapter.mixes(base, "ffn", hidden)
            x = self.adapter.norm(base + ".ffn_norm", _hc_pre(hidden, attn_pre))
            ffn = mx.concatenate(
                [
                    self.adapter.moe(base + ".ffn", x[i : i + 1])
                    for i in range(self.block_size)
                ]
            )
            hidden = _hc_post(ffn, hidden, post, combine)
        base = f"mtp.{self.layers[-1]}"
        collapsed = _hc_pre(hidden, pre)
        logits = self.w.linear(
            "head", self.adapter.norm(base + ".norm", collapsed).astype(mx.float32)
        )
        output, biased_logits, markov_embeddings = [next_token], [], []
        for index in range(self.block_size):
            markov = self.w.embedding(base + ".markov_head.embed", [output[-1]])
            biased = logits[index : index + 1] + self.w.linear(
                base + ".markov_head.head", markov.astype(mx.float32)
            )
            output.append(int(mx.argmax(biased)))
            biased_logits.append(biased)
            markov_embeddings.append(markov)
        logits = mx.concatenate(biased_logits)
        confidence = self.w.linear(
            base + ".confidence_head.proj",
            mx.concatenate(
                (collapsed, mx.concatenate(markov_embeddings)), axis=-1
            ).astype(mx.float32),
        )
        mx.eval(logits, confidence)
        if not bool(mx.all(mx.isfinite(logits))) or not bool(
            mx.all(mx.isfinite(confidence))
        ):
            raise FloatingPointError("non-finite DSpark output")
        return output, confidence


def verify_greedy(proposals, logits, advance, observe, eos, remaining):
    """Serial oracle: rejected proposals never advance target state."""
    output, stats = [], {"accepted_draft_tokens": 0, "rejected_blocks": 0}
    for index, proposal in enumerate(proposals):
        if len(output) >= remaining:
            break
        expected = int(mx.argmax(logits))
        matched = proposal == expected
        output.append(expected)
        if matched and index > 0:
            stats["accepted_draft_tokens"] += 1
        if not matched:
            stats["rejected_blocks"] += 1
        if expected == eos or len(output) == remaining:
            break
        logits, hidden = advance(expected)
        observe(hidden)
        if not matched:
            break
    return output, logits, stats
