# SPDX-License-Identifier: Apache-2.0
"""Decision model backends for the System One service."""

from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol, cast

from .schema import Question, answer_from_probabilities, clm_pairs, to_text


class DecisionBackend(Protocol):
    default_model: str

    def answer(
        self, state: Any, questions: dict[str, Question], model: str, temperature: float
    ) -> dict: ...
    def rank(
        self,
        context: Any,
        question: str | None,
        answers: list[str],
        model: str,
        temperature: float,
    ) -> list[dict]: ...
    def models(self) -> list[dict]: ...


class LayaBackend:
    def __init__(
        self,
        model: str,
        *,
        device: str = "gpu",
        dtype: str = "float16",
        batch_size: int = 16,
    ):
        try:
            from laya_mlx import load
        except ImportError as exc:
            raise RuntimeError(
                "Laya requires Python 3.11+ and the System One extra. "
                "Install with: pip install 'rapid-mlx[system-one]'"
            ) from exc
        self._agent = load(model, device=device, dtype=dtype, batch_size=batch_size)
        self.default_model = model
        self._lock = threading.Lock()

    @staticmethod
    def _wire_questions(questions: dict[str, Question]) -> dict[str, dict]:
        return {
            key: value.model_dump(exclude_none=True) for key, value in questions.items()
        }

    def answer(
        self, state: Any, questions: dict[str, Question], model: str, temperature: float
    ) -> dict:
        if model not in (self.default_model, "laya-rl-agent"):
            raise KeyError(
                f"unknown model {model!r}; available: {[self.default_model]}"
            )
        if temperature != 1.0:
            raise ValueError(
                "the Laya backend uses checkpoint calibration and requires temperature=1"
            )
        with self._lock:
            result = cast(
                dict, self._agent.system_one(state, self._wire_questions(questions))
            )
        result["model"] = self.default_model
        result.setdefault("usage", {})["billing_units"] = len(questions)
        return result

    def rank(
        self,
        context: Any,
        question: str | None,
        answers: list[str],
        model: str,
        temperature: float,
    ) -> list[dict]:
        request = Question(
            type="choice",
            instructions=question or "Choose the best answer.",
            criteria={str(i): answer for i, answer in enumerate(answers)},
        )
        result = self.answer(context, {"rank": request}, model, temperature)["answers"][
            "rank"
        ]
        ordered = sorted(result["probabilities"].items(), key=lambda item: -item[1])
        return [
            {"rank": rank + 1, "candidate": answers[int(index)], "prob": probability}
            for rank, (index, probability) in enumerate(ordered)
        ]

    def models(self) -> list[dict]:
        return [
            {
                "name": self.default_model,
                "backend": "laya-mlx",
                "description": "Laya typed decision model on native MLX",
            }
        ]


class _ProjectionHead:
    def __init__(self, config: dict[str, Any]):
        import mlx.nn as nn

        activation = config.get("activation", "gelu")
        if activation not in {"gelu", "relu", "silu"}:
            raise ValueError(f"unsupported CLM head activation {activation!r}")
        hidden = int(config.get("hidden_size", 4096))
        width = int(config["width"])
        depth = int(config["depth"])
        projection_dim = int(config.get("projection_dim", 512))
        if depth < 2:
            raise ValueError("CLM head depth must be at least 2")

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(hidden, width)
                self.hidden = [
                    nn.Linear(width, width) for _ in range(max(0, depth - 2))
                ]
                self.norms = (
                    [nn.LayerNorm(width) for _ in self.hidden]
                    if config.get("layernorm", False)
                    else []
                )
                self.out = nn.Linear(width, projection_dim)

            def __call__(self, value):
                import mlx.core as mx

                activation = {
                    "gelu": nn.gelu,
                    "relu": nn.relu,
                    "silu": nn.silu,
                }[config.get("activation", "gelu")]
                value = activation(self.inp(value))
                for index, layer in enumerate(self.hidden):
                    projected = layer(value)
                    if self.norms:
                        projected = self.norms[index](projected)
                    projected = activation(projected)
                    value = (
                        value + projected
                        if config.get("residual", False)
                        else projected
                    )
                value = self.out(value).astype(mx.float32)
                return value / mx.maximum(
                    mx.linalg.norm(value, axis=-1, keepdims=True), 1e-12
                )

        self.module = Head()

    def __call__(self, value):
        return self.module(value)

    def load_weights(self, weights: dict[str, Any], prefix: str) -> None:
        pairs = []
        for name, value in weights.items():
            if name.startswith(prefix + "."):
                local = name[len(prefix) + 1 :]
                if (
                    local.startswith("hidden.")
                    or local.startswith("norms.")
                    or local.startswith("inp.")
                    or local.startswith("out.")
                ):
                    pairs.append((local, value))
        expected = 4 + 2 * len(self.module.hidden) + (2 * len(self.module.norms))
        if len(pairs) != expected:
            raise ValueError(
                f"CLM {prefix} has {len(pairs)} tensors; expected {expected}"
            )
        self.module.load_weights(pairs, strict=True)
        self.module.eval()


class CLMBackend:
    """Native MLX CLM-8B encoder and projection heads.

    The official checkpoint is a PyTorch pickle. Convert it once with
    ``rapid-mlx-convert-clm-head``; serving never imports PyTorch.
    """

    def __init__(
        self,
        encoder: str,
        head: str,
        *,
        model_name: str | None = None,
        device: str = "gpu",
        cache_entries: int = 20_000,
        max_tokens: int = 2048,
        max_work_tokens: int = 32_768,
    ):
        from rapid_mlx.system_one.convert_clm import _artifact_lock

        if device not in {"gpu", "cpu"}:
            raise ValueError("CLM device must be 'gpu' or 'cpu'")

        head_path = Path(head).expanduser()
        if head_path.suffix == ".pt":
            raise ValueError(
                "CLM .pt checkpoints must be converted before serving: "
                f"rapid-mlx-convert-clm-head {head_path} OUTPUT_DIR"
            )
        file_form = head_path.suffix.lower() == ".safetensors"
        artifact_root = head_path.parent if file_form else head_path
        if head_path.is_file() and not file_form:
            raise ValueError("CLM head weights must be a .safetensors file")
        with _artifact_lock(artifact_root, exclusive=False):
            config_path = (
                head_path.with_name("config.json")
                if file_form
                else head_path / "config.json"
            )
            weights_path = head_path if file_form else head_path / "model.safetensors"
            if not config_path.is_file() or not weights_path.is_file():
                raise ValueError(
                    "CLM head must contain config.json and model.safetensors"
                )
            self.config = json.loads(config_path.read_text(encoding="utf-8"))
            import mlx.core as mx

            # System One runs as a dedicated process, so select the MLX device
            # before loading either the head or the Qwen3 encoder. Keep this
            # after format/existence checks so controlled artifact errors remain
            # available in no-MLX environments.
            mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
            self.device = device
            self._state_head = _ProjectionHead(self.config)
            self._action_head = _ProjectionHead(self.config)
            weights = mx.load(str(weights_path))
            self._state_head.load_weights(weights, "state_head")
            self._action_head.load_weights(weights, "action_head")
            mx.eval(weights)
        from rapid_mlx.utils.tokenizer import load_model_with_fallback

        self._model, self._tokenizer = load_model_with_fallback(encoder)
        inner = getattr(self._model, "model", None)
        if inner is None or not callable(inner):
            raise ValueError(
                "CLM encoder must expose its pre-lm-head transformer as model.model"
            )
        hidden_size = int(getattr(getattr(self._model, "args", None), "hidden_size", 0))
        model_type = getattr(getattr(self._model, "args", None), "model_type", None)
        if model_type != "qwen3":
            raise ValueError(
                f"CLM-v0.1 heads require a Qwen3 encoder, got {model_type!r}"
            )
        quantization = getattr(getattr(self._model, "args", None), "quantization", None)
        try:
            import mlx.nn as nn

            quantized_layers = any(
                isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding))
                for _, module in inner.named_modules()
            )
        except AttributeError:
            quantized_layers = False
        if quantization or quantized_layers:
            raise ValueError(
                "CLM-v0.1 probability parity requires the BF16 Qwen3-8B "
                "encoder; quantized encoders are not yet qualified"
            )
        expected = int(self.config.get("hidden_size", 4096))
        if hidden_size != expected:
            raise ValueError(
                f"CLM head expects hidden size {expected}, encoder exposes {hidden_size}"
            )
        self._encoder = inner
        self.encoder_name = encoder
        self.default_model = model_name or self.config.get("model_name", "clm-latest")
        logit_scale = float(self.config["logit_scale"])
        if not math.isfinite(logit_scale):
            raise ValueError("CLM logit_scale must be finite")
        # Match upstream HeadPair: exp(logit_scale).clamp(max=100).
        self._scale = math.exp(min(logit_scale, math.log(100.0)))
        self._max_tokens = max_tokens
        self._max_work_tokens = max_work_tokens
        self._max_text_bytes = max(1024, max_tokens * 16)
        self._cache_entries = max(0, cache_entries)
        self._cache: OrderedDict[tuple[str, tuple[int, ...]], Any] = OrderedDict()
        self._lock = threading.Lock()

    def _token_ids(self, text: str) -> list[int]:
        if (
            len(text) > self._max_text_bytes
            or len(text.encode("utf-8")) > self._max_text_bytes
        ):
            raise ValueError(
                f"CLM input text exceeds {self._max_text_bytes} UTF-8 bytes before tokenization"
            )
        tokenizer = getattr(self._tokenizer, "_tokenizer", self._tokenizer)
        ids = cast(list[int], tokenizer.encode(text, add_special_tokens=True))
        if not ids:
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is None:
                raise ValueError(
                    "CLM encoder tokenizer produced no tokens and has no eos token"
                )
            ids = [eos]
        # vLLM's upstream `truncate_prompt_tokens` keeps the final N prompt
        # tokens. CLM uses last-token pooling, so mirror that left truncation.
        return ids[-self._max_tokens :]

    def _project(self, kind: str, texts: list[str], token_rows: list[list[int]]):
        import mlx.core as mx

        head = self._state_head if kind == "state" else self._action_head
        output = []
        input_tokens = 0
        for text, token_ids in zip(texts, token_rows, strict=True):
            key = (kind, tuple(token_ids))
            cached = self._cache.get(key)
            if cached is None:
                input_tokens += len(token_ids)
                ids = mx.array([token_ids])
                hidden = self._encoder(ids)[:, -1, :]
                cached = head(hidden)[0]
                mx.eval(cached)
                if self._cache_entries:
                    self._cache[key] = cached
                    self._cache.move_to_end(key)
                    while len(self._cache) > self._cache_entries:
                        self._cache.popitem(last=False)
            else:
                self._cache.move_to_end(key)
            output.append(cached)
        return mx.stack(output), input_tokens

    def answer(
        self, state: Any, questions: dict[str, Question], model: str, temperature: float
    ) -> dict:
        import mlx.core as mx

        if model != self.default_model:
            raise KeyError(
                f"unknown model {model!r}; available: {[self.default_model]}"
            )
        # Bound shared state before clm_pairs duplicates it for each question.
        state_text = to_text(state).strip()
        if (
            len(state_text) > self._max_text_bytes
            or len(state_text.encode("utf-8")) > self._max_text_bytes
        ):
            raise ValueError(
                f"CLM state exceeds {self._max_text_bytes} UTF-8 bytes before tokenization"
            )
        pairs = clm_pairs(state_text, questions)
        states = [item[0] for item in pairs.values()]
        candidates = [candidate for item in pairs.values() for candidate in item[2]]
        state_token_rows = [self._token_ids(text) for text in states]
        action_token_rows = [self._token_ids(text) for text in candidates]
        requested_tokens = sum(len(row) for row in state_token_rows + action_token_rows)
        if requested_tokens > self._max_work_tokens:
            raise ValueError(
                "request needs "
                f"{requested_tokens} encoder tokens; limit is {self._max_work_tokens}"
            )
        with self._lock:
            state_vectors, state_tokens = self._project(
                "state", states, state_token_rows
            )
            action_vectors, action_tokens = self._project(
                "action", candidates, action_token_rows
            )
            answers_out = {}
            offset = 0
            for row, (question_id, (_, keys, option_texts)) in enumerate(pairs.items()):
                count = len(option_texts)
                logits = (self._scale / temperature) * (
                    action_vectors[offset : offset + count] @ state_vectors[row]
                )
                probabilities = mx.softmax(logits).tolist()
                answers_out[question_id] = answer_from_probabilities(
                    questions[question_id], keys, probabilities
                )
                offset += count
        return {
            "model": model,
            "answers": answers_out,
            "usage": {
                "billing_units": len(questions),
                # Match upstream CLM: input_tokens measures encoder work and
                # therefore falls on cache hits. Keep submitted work explicit.
                "input_tokens": state_tokens + action_tokens,
                "requested_tokens": requested_tokens,
                "output_tokens": 0,
            },
        }

    def rank(
        self,
        context: Any,
        question: str | None,
        answers: list[str],
        model: str,
        temperature: float,
    ) -> list[dict]:
        q = Question(
            type="choice",
            instructions=question or "",
            criteria={str(i): answer for i, answer in enumerate(answers)},
        )
        result = self.answer(context, {"rank": q}, model, temperature)["answers"][
            "rank"
        ]
        ordered = sorted(result["probabilities"].items(), key=lambda item: -item[1])
        return [
            {"rank": rank + 1, "candidate": answers[int(index)], "prob": probability}
            for rank, (index, probability) in enumerate(ordered)
        ]

    def models(self) -> list[dict]:
        return [
            {
                "name": self.default_model,
                "backend": "clm-mlx",
                "encoder": self.encoder_name,
                "description": "CLM contrastive state/action model on native MLX",
            }
        ]
