"""Qualified single-stream DeepSeek V4.1 Flash + DSpark K4 runtime."""

from __future__ import annotations

import contextlib
import weakref
from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import Any

import mlx.core as mx
from fastapi import HTTPException
from transformers import PreTrainedTokenizerFast

from vllm_mlx import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402
from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer  # noqa: E402

from .affine_route_qmv import affine2_route_down_qmv
from .dspark import DSpark, DSparkWeights, draft_attention, quantize_cache
from .load import load, supports_engram_ssd_offload


@dataclass(frozen=True)
class GenerationChunk:
    text: str
    token: int
    prompt_tokens: int
    generation_tokens: int


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    generation_tokens: int


@dataclass
class DSparkRuntime:
    drafter: DSpark
    drafter_repo: str
    target_revision: str
    drafter_revision: str
    kind: str = "dspark"
    algorithm: str = "dspark-k4"

    def reset(self) -> None:
        self.drafter.position = -1
        self.drafter.windows = {stage: [] for stage in self.drafter.layers}


class ExactDirectDownSwitchGLU(SwitchGLU):
    """Stock gate/up numerics plus the qualified exact 2-bit down QMV."""

    def __call__(self, x, indices):
        if int(x.shape[0]) * int(indices.shape[-1]) > 36:
            return super().__call__(x, indices)
        expanded = mx.expand_dims(x, (-2, -3))
        up = self.up_proj(expanded, indices).squeeze(-2)
        gate = self.gate_proj(expanded, indices).squeeze(-2)
        activated = self.activation(up, gate)
        tokens, topk, width = map(int, activated.shape)
        down = affine2_route_down_qmv(
            self.down_proj,
            activated.reshape(tokens * topk, width),
            indices.reshape(tokens * topk, 1),
        )
        return down.reshape(tokens, topk, -1)


def install_target_qmv(model: Any) -> int:
    replaced = 0
    for layer in model.layers:
        experts = layer.ffn.experts
        if not isinstance(experts, SwitchGLU):
            raise TypeError(f"unsupported expert module: {type(experts).__name__}")
        experts.__class__ = ExactDirectDownSwitchGLU
        replaced += 1
    return replaced


def _install_vectorized_mtp_attention(draft: DSpark) -> None:
    def rotate_positions(x, positions, config, inverse=False):
        rope_dim = config["qk_rope_head_dim"]
        frequencies = 1 / (
            config["rope_theta"]
            ** (mx.arange(0, rope_dim, 2, dtype=mx.float32) / rope_dim)
        )
        angles = positions[:, None] * frequencies[None, :]
        if inverse:
            angles = -angles
        tail = x[..., -rope_dim:].astype(mx.float32)
        tail = tail.reshape(*tail.shape[:-1], rope_dim // 2, 2)
        real, imag = tail[..., 0], tail[..., 1]
        expand = (slice(None),) + (None,) * (real.ndim - 2) + (slice(None),)
        cosine = mx.cos(angles)[expand]
        sine = mx.sin(angles)[expand]
        rotated = mx.stack(
            (real * cosine - imag * sine, real * sine + imag * cosine), axis=-1
        ).reshape(*x.shape[:-1], rope_dim)
        return mx.concatenate((x[..., :-rope_dim], rotated.astype(x.dtype)), axis=-1)

    def vectorized_attention(self, base, x, stage):
        adapter = self.adapter
        config = self.c
        count = x.shape[0]
        positions = mx.arange(count, dtype=mx.float32) + self.position + 1
        query_a = adapter.norm(base + ".q_norm", adapter.w.linear(base + ".wq_a", x))
        query = adapter.w.linear(base + ".wq_b", query_a).reshape(
            count, config["num_attention_heads"], config["head_dim"]
        )
        key_value = adapter.norm(base + ".kv_norm", adapter.w.linear(base + ".wkv", x))
        query = rotate_positions(query, positions, config)
        key_value = quantize_cache(
            rotate_positions(key_value, positions, config), 8, 32
        )
        keys = mx.concatenate([*self.windows[stage], key_value])
        output = draft_attention(query, keys, adapter.w.read(base + ".attn_sink"))
        output = rotate_positions(output, positions, config, inverse=True)
        output = output.reshape(count, config["o_groups"], -1)
        shape = adapter.w.entries[base + ".wo_a.weight"]["shape"]
        rows = shape[0] // config["o_groups"]
        grouped = mx.quantized_matmul(
            output.swapaxes(0, 1),
            adapter.w.read(base + ".wo_a.weight").reshape(config["o_groups"], rows, -1),
            adapter.w.read(base + ".wo_a.scales").reshape(config["o_groups"], rows, -1),
            adapter.w.read(base + ".wo_a.biases").reshape(config["o_groups"], rows, -1),
            transpose=True,
            **adapter.w.quant_config(base + ".wo_a"),
        )
        return adapter.w.linear(
            base + ".wo_b", grouped.swapaxes(0, 1).reshape(count, -1)
        )

    setattr(draft, "attention", MethodType(vectorized_attention, draft))


def _install_packed_mtp_moe(draft: DSpark) -> int:
    weights, expert_count = draft.w, draft.c["dspark_n_routed_experts"]
    packed: dict[str, dict[str, dict[str, Any]]] = {}
    packed_bytes = 0
    bases = sorted(
        {
            key.split(".experts.", 1)[0]
            for key in weights.entries
            if key.startswith("mtp.") and ".ffn.experts." in key
        }
    )
    for base in bases:
        projections = {}
        for projection in ("w1", "w3", "w2"):
            values: dict[str, Any] = {}
            for suffix in ("weight", "scales", "biases"):
                keys = [
                    f"{base}.experts.{expert}.{projection}.{suffix}"
                    for expert in range(expert_count)
                ]
                value = mx.stack([weights.read(key) for key in keys])
                mx.eval(value)
                values[suffix] = value
                packed_bytes += value.nbytes
                for key in keys:
                    weights.release_tensor(key)
            values["config"] = weights.quant_config(f"{base}.experts.0.{projection}")
            projections[projection] = values
        packed[base] = projections
        mx.clear_cache()

    def packed_moe(self, base, x):
        config = self.c
        logits = (
            x.astype(mx.float32)
            @ self.w.read(base + ".gate.weight").astype(mx.float32).T
        )
        scores = mx.sqrt(mx.logaddexp(logits, mx.zeros_like(logits)))
        topk = config["dspark_num_experts_per_tok"]
        picks = mx.argsort(scores + self.w.read(base + ".gate.bias"), axis=-1)[
            ..., -topk:
        ]
        selected = mx.take_along_axis(scores, picks, axis=-1)
        if config["norm_topk_prob"] and topk > 1:
            selected = selected / (mx.sum(selected, axis=-1, keepdims=True) + 1e-20)
        selected = selected * config["routed_scaling_factor"]
        expanded = mx.expand_dims(x, (-2, -3))

        def project(name, values):
            projection = packed[base][name]
            return mx.gather_qmm(
                values,
                projection["weight"],
                projection["scales"],
                projection["biases"],
                rhs_indices=picks,
                transpose=True,
                sorted_indices=False,
                **projection["config"],
            )

        gate = project("w1", expanded).astype(mx.float32)
        up = project("w3", expanded).astype(mx.float32)
        limit = config["swiglu_limit"]
        if limit > 0:
            gate, up = mx.minimum(gate, limit), mx.clip(up, -limit, limit)
        hidden = (gate * mx.sigmoid(gate) * up * selected[..., None, None]).astype(
            x.dtype
        )
        routed = mx.sum(project("w2", hidden).squeeze(-2).astype(mx.float32), axis=-2)
        shared = self.expert(base + ".shared_experts", x).astype(mx.float32)
        return (routed + shared).astype(x.dtype)

    setattr(
        draft.adapter,
        "moe",
        MethodType(packed_moe, weakref.proxy(draft.adapter)),
    )
    setattr(draft.adapter, "_packed_mtp_moe", packed)
    return packed_bytes


def load_product_runtime(
    target_path: str,
    mtp_path: str,
    *,
    target_revision: str,
    mtp_revision: str,
    mtp_identity: str | None = None,
):
    model, _args = load(
        target_path,
        lazy=False,
        engram_ssd_offload=supports_engram_ssd_offload(target_path),
    )
    model.eval_interval = 40
    if install_target_qmv(model) != len(model.layers):
        raise RuntimeError("failed to install target affine-2bit QMV on every layer")
    weights = DSparkWeights(mtp_path)
    weights.attach_target(model)
    weights.pin_mtp()
    target = SimpleNamespace(w=weights, c=weights.config, execution_mode="compiled")
    with contextlib.redirect_stdout(__import__("sys").stderr):
        draft = DSpark(target, pin_weights=False)
    _install_vectorized_mtp_attention(draft)
    _install_packed_mtp_moe(draft)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        target_path, local_files_only=True
    )
    runtime = DSparkRuntime(
        drafter=draft,
        drafter_repo=mtp_identity or mtp_path,
        target_revision=target_revision,
        drafter_revision=mtp_revision,
    )
    return model, tokenizer, runtime


def render_prompt(processor, _model, request, *, enable_thinking=False) -> str:
    if request.tools:
        raise HTTPException(
            status_code=400,
            detail="tool calling is not qualified for DeepSeek V4.1 Flash",
        )
    bos = processor.bos_token or "<｜begin▁of▁sentence｜>"
    eos = processor.eos_token or "<｜end▁of▁sentence｜>"
    pieces = [bos]
    for message in request.messages:
        content = message.content
        if not isinstance(content, str):
            raise HTTPException(
                status_code=400,
                detail="DeepSeek V4.1 Flash currently accepts text content only",
            )
        if message.role == "system":
            pieces.append(content)
        elif message.role == "user":
            pieces.extend(("<｜User｜>", content))
        elif message.role == "assistant":
            pieces.extend(("<｜Assistant｜>", content, eos))
        else:
            raise HTTPException(
                status_code=400, detail=f"unsupported message role: {message.role}"
            )
    pieces.append("<｜Assistant｜>")
    if not enable_thinking:
        pieces.append("</think>")
    prompt = "".join(pieces)
    if len(processor.encode(prompt, add_special_tokens=False)) > 8192:
        raise HTTPException(
            status_code=400,
            detail=(
                "DeepSeek V4.1 Flash currently supports at most 8192 input "
                "tokens in the qualified product lane"
            ),
        )
    return prompt


def validate_request(request) -> None:
    """Reject accepted wire fields that the narrow greedy lane cannot honor."""
    unsupported = [
        name
        for name in (
            "stop",
            "top_k",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "top_logprobs",
            "logit_bias",
            "video_fps",
            "video_max_frames",
            "reasoning_max_tokens",
            "reasoning_effort",
            "seed",
        )
        if getattr(request, name, None) is not None
    ]
    template_kwargs = getattr(request, "chat_template_kwargs", None) or {}
    if set(template_kwargs) - {"enable_thinking"}:
        unsupported.append("chat_template_kwargs")
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=(
                "DeepSeek V4.1 DSpark K4 does not support request field(s): "
                + ", ".join(unsupported)
            ),
        )


def generation_kwargs(
    *, max_tokens: int, temperature: float, top_p: float
) -> dict[str, Any]:
    if not 1 <= max_tokens <= 4096:
        raise HTTPException(
            status_code=400,
            detail=(
                "DeepSeek V4.1 DSpark K4 currently supports 1 <= max_tokens <= 4096"
            ),
        )
    if temperature != 0.0:
        raise HTTPException(
            status_code=400,
            detail="DeepSeek V4.1 DSpark K4 currently supports greedy decoding only",
        )
    if top_p != 1.0:
        raise HTTPException(
            status_code=400,
            detail="DeepSeek V4.1 DSpark K4 currently requires top_p=1",
        )
    return {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}


def _match(candidate, target_logits, eos_id, seed_already_emitted):
    committed = [] if seed_already_emitted else [candidate[0]]
    accepted = 0
    for index in range(1, len(candidate)):
        expected = int(mx.argmax(target_logits[:, index - 1]))
        committed.append(expected)
        if candidate[index] != expected:
            return committed, index, expected == eos_id, accepted
        accepted += 1
        if expected == eos_id:
            return committed, None, True, accepted
    return committed, None, False, accepted


def stream_generate(
    model, processor, prompt: str, *, runtime: DSparkRuntime, max_tokens: int, **_kwargs
):
    input_ids = processor.encode(prompt, add_special_tokens=False)
    if len(input_ids) < 2:
        raise ValueError("prompt must contain at least two tokens")
    cache = model.make_cache(max_seq_len=len(input_ids) + max_tokens + 8)
    runtime.reset()
    logits = hidden = None
    for start in range(0, len(input_ids), 512):
        logits, hidden = model(
            mx.array([input_ids[start : start + 512]]),
            cache,
            last_logit_only=False,
            return_dspark_hidden=True,
        )
        mx.eval(logits, hidden)
        for index in range(hidden.shape[1]):
            runtime.drafter.observe(hidden[:, index], start + index)
    assert logits is not None
    logits = logits[:, -1]
    eos_id = processor.eos_token_id
    detokenizer = NaiveStreamingDetokenizer(processor)
    emitted_text = ""
    generated = 0
    seed_already_emitted = False
    while generated < max_tokens:
        seed = int(mx.argmax(logits))
        if not seed_already_emitted and seed == eos_id:
            tokens = [seed]
            hit_eos = True
        else:
            proposals, _confidence = runtime.drafter.propose(seed)
            candidate = proposals[
                : min(5, max_tokens - generated + int(seed_already_emitted))
            ]
            base_offset = cache.offset
            target_logits, target_hidden = model(
                mx.array([candidate]),
                cache,
                last_logit_only=False,
                return_dspark_hidden=True,
                enable_rollback=True,
            )
            mx.eval(target_logits, target_hidden)
            tokens, mismatch, hit_eos, _accepted = _match(
                candidate, target_logits, eos_id, seed_already_emitted
            )
            if mismatch is None:
                for index in range(len(candidate)):
                    runtime.drafter.observe(
                        target_hidden[:, index], base_offset + index
                    )
                logits = target_logits[:, -1]
                seed_already_emitted = False
            else:
                cache.rollback(base_offset + mismatch)
                for index in range(mismatch):
                    runtime.drafter.observe(
                        target_hidden[:, index], base_offset + index
                    )
                logits = target_logits[:, mismatch - 1]
                seed_already_emitted = True
        for token in tokens:
            generated += 1
            # EOS is control state, not user-visible text. The release
            # tokenizer otherwise decodes it to a literal marker.
            if token != eos_id:
                detokenizer.add_token(token)
            current = detokenizer.text
            ending = token == eos_id or generated >= max_tokens
            if ending:
                detokenizer.finalize()
                current = detokenizer.text
            text = (
                current[len(emitted_text) :] if current.startswith(emitted_text) else ""
            )
            if text:
                emitted_text = current
            yield GenerationChunk(text, token, len(input_ids), generated)
            if ending:
                return


def generate(model, processor, prompt: str, *, runtime: DSparkRuntime, **kwargs):
    chunks = list(stream_generate(model, processor, prompt, runtime=runtime, **kwargs))
    return GenerationResult(
        text="".join(chunk.text for chunk in chunks),
        prompt_tokens=chunks[-1].prompt_tokens
        if chunks
        else len(processor.encode(prompt)),
        generation_tokens=chunks[-1].generation_tokens if chunks else 0,
    )
