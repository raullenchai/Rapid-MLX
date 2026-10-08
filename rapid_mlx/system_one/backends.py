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


class ClefBackend:
    """Cloudflare's joint-schema decision model, using its audited release code.

    Clef scores all questions in one backbone pass.  In particular it cannot be
    implemented by asking a generative Qwen model for JSON: that would discard
    the trained joint head and its calibrated option probabilities.
    """

    _MODELS = {
        "clef": ("Cloudflare/clef", "2f3de3dd85f379784083b0814d997ab627200f0c"),
        "clef-flash": (
            "Cloudflare/clef-flash",
            "17f0b0ad64efb65d273590632833508766b2aae6",
        ),
    }

    def __init__(self, model: str, *, device: str = "gpu") -> None:
        import importlib.metadata

        from packaging.specifiers import SpecifierSet

        selected = model.rsplit("/", 1)[-1].lower()
        if selected not in self._MODELS or model not in {
            selected,
            self._MODELS[selected][0],
        }:
            raise ValueError(f"unknown Clef model {model!r}; choose clef or clef-flash")
        if device not in {"gpu", "cpu"}:
            raise ValueError("Clef device must be 'gpu' or 'cpu'")
        for package, constraint in (
            ("torch", ">=2.11.0"),
            ("transformers", ">=5.10.2,!=5.13.0,<5.16"),
        ):
            try:
                installed = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    "Clef requires the optional runtime: pip install 'rapid-mlx[clef]'"
                ) from exc
            if installed not in SpecifierSet(constraint):
                raise RuntimeError(
                    f"Clef requires {package}{constraint}, found {installed}; "
                    "install 'rapid-mlx[clef]'"
                )

        import torch
        from huggingface_hub import snapshot_download

        if device == "gpu" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "Clef GPU mode requires Apple Metal/MPS; use --device cpu"
            )
        from rapid_mlx.clef.vendor.joint_schema_model import load_release_model

        self.default_model = selected
        self.repo_id, revision = self._MODELS[selected]
        path = snapshot_download(self.repo_id, revision=revision)
        self._model, self._processor = load_release_model(
            path, device="mps" if device == "gpu" else "cpu"
        )
        self._lock = threading.Lock()

    def answer(
        self, state: Any, questions: dict[str, Question], model: str, temperature: float
    ) -> dict:
        return self.answer_media(state, questions, model, temperature, None, None)

    def answer_media(
        self,
        state: Any,
        questions: dict[str, Question],
        model: str,
        temperature: float,
        images: list[str] | None,
        videos: list[list[str]] | None,
    ) -> dict:
        if model not in (self.default_model, self.repo_id):
            raise KeyError(f"unknown model {model!r}; available: {self.default_model}")
        if temperature != 1.0:
            raise ValueError(
                "Clef uses checkpoint calibration and requires temperature=1"
            )

        from rapid_mlx.clef.media import decode_media
        from rapid_mlx.clef.vendor.joint_schema_model import systemone

        request = {
            "model": self.default_model,
            "state": state,
            "questions": {
                key: value.model_dump(exclude_none=True)
                for key, value in questions.items()
            },
        }
        with self._lock:
            # Decode only after acquiring the model lock. Otherwise every
            # queued request can hold an expanded RGB copy while it waits.
            if images or videos:
                decoded_images, decoded_videos = decode_media(images, videos)
                if decoded_images:
                    request["images"] = decoded_images
                if decoded_videos:
                    request["videos"] = decoded_videos
            result = systemone(self._model, self._processor, request)
        result["usage"]["billing_units"] = len(questions)
        return dict(result)

    def rank(
        self,
        context: Any,
        question: str | None,
        answers: list[str],
        model: str,
        temperature: float,
    ) -> list[dict]:
        if model not in (self.default_model, self.repo_id):
            raise KeyError(f"unknown model {model!r}; available: {self.default_model}")
        if temperature != 1.0:
            raise ValueError(
                "Clef uses checkpoint calibration and requires temperature=1"
            )
        request = Question(
            type="choice",
            instructions=question or "Choose the best answer.",
            criteria={str(index): value for index, value in enumerate(answers)},
        )
        # The official SystemOne response rounds option probabilities to four
        # decimals. Rank from the raw head scores so near-ties keep their true
        # order instead of falling back to candidate insertion order.
        import torch

        from rapid_mlx.clef.vendor.joint_schema_model import (
            collate_records,
            encode_record,
        )

        encoded = encode_record(
            self._processor.tokenizer,
            {
                "model": self.default_model,
                "state": context,
                "questions": {"rank": request.model_dump(exclude_none=True)},
            },
            processor=self._processor,
        )
        device = next(self._model.parameters()).device
        with self._lock, torch.inference_mode():
            logits = self._model(
                collate_records(
                    [encoded], self._processor.tokenizer.pad_token_id, device
                )
            )[0][0]
            scores = logits.float().softmax(-1).tolist()
        probabilities = dict(zip(encoded.questions[0].option_ids, scores))
        ordered = sorted(probabilities.items(), key=lambda item: -item[1])
        return [
            {"rank": rank + 1, "candidate": answers[int(index)], "prob": probability}
            for rank, (index, probability) in enumerate(ordered)
        ]

    def models(self) -> list[dict]:
        return [
            {
                "name": self.default_model,
                "backend": "clef-torch-mps",
                "hf_id": self.repo_id,
                "description": "Cloudflare Clef typed decisions with joint schema head",
            }
        ]


class ClefMLXBackend:
    """Cloudflare Clef typed decisions on native MLX, without the Torch runtime.

    Serves the prepared, quantized MLX conversions of the Clef release. The
    joint schema head and prompt are the release's own, so answers keep the
    Clef wire shape; only the arithmetic runs on MLX.
    """

    _MODELS = {
        "clef-mlx": (
            "nativ-community/clef-MLX-MXFP4",
            "b94c97b0d0b80fdda6d7d36c745d289890d1c4e1",
        ),
        "clef-flash-mlx": (
            "nativ-community/clef-flash-MLX-MXFP4",
            "63a0d4df0213be9843968151609e05c2f6683870",
        ),
    }

    def __init__(self, model: str, *, device: str = "gpu") -> None:
        import importlib.util

        if device not in {"gpu", "cpu"}:
            raise ValueError("Clef device must be 'gpu' or 'cpu'")
        local = Path(model).expanduser()
        repos = {repo.lower(): name for name, (repo, _) in self._MODELS.items()}
        selected = repos.get(model.lower(), model.lower())
        # A published name always means the pinned release, even when a
        # directory of the same name sits in the working directory.
        named = selected in self._MODELS
        if not named and not local.is_dir():
            raise ValueError(
                f"unknown native Clef model {model!r}; choose "
                f"{', '.join(self._MODELS)} or a local checkpoint directory"
            )
        if importlib.util.find_spec("mlx_vlm") is None:
            raise RuntimeError(
                "native Clef requires the vision runtime: "
                "pip install 'rapid-mlx[vision]'"
            )
        if named:
            from rapid_mlx._mirror import pinned_snapshot_download

            self.default_model = selected
            self.repo_id, revision = self._MODELS[selected]
            path = pinned_snapshot_download(self.repo_id, revision)
        else:
            # A local directory serves other prepared conversions (8-bit, NVFP4).
            path = str(local)
            self.default_model = local.name
            self.repo_id = str(local)

        import mlx.core as mx

        from .clef_mlx import ClefScorer, load_clef

        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        self._scorer = ClefScorer(*load_clef(path))
        self._lock = threading.Lock()

    def _check(self, model: str, temperature: float) -> None:
        if model not in (self.default_model, self.repo_id):
            raise KeyError(f"unknown model {model!r}; available: {self.default_model}")
        if temperature != 1.0:
            raise ValueError(
                "Clef uses checkpoint calibration and requires temperature=1"
            )

    def answer(
        self, state: Any, questions: dict[str, Question], model: str, temperature: float
    ) -> dict:
        return self.answer_media(state, questions, model, temperature, None, None)

    def answer_media(
        self,
        state: Any,
        questions: dict[str, Question],
        model: str,
        temperature: float,
        images: list[str] | None,
        videos: list[list[str]] | None,
    ) -> dict:
        self._check(model, temperature)

        from rapid_mlx.clef.media import decode_media

        from .clef_mlx import clef_answer

        specs = {
            key: value.model_dump(exclude_none=True) for key, value in questions.items()
        }
        with self._lock:
            # Decode only after acquiring the model lock. Otherwise every
            # queued request can hold an expanded RGB copy while it waits.
            decoded_images, decoded_videos = (
                decode_media(images, videos) if images or videos else (None, None)
            )
            scored, tokens = self._scorer.score(
                state, specs, decoded_images, decoded_videos
            )
        return {
            "model": self.default_model,
            "answers": {
                key: clef_answer(specs[key], probabilities)
                for key, probabilities in scored.items()
            },
            "usage": {
                "billing_units": len(questions),
                "input_tokens": tokens,
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
        self._check(model, temperature)
        request = {
            "type": "choice",
            "instructions": question or "Choose the best answer.",
            "criteria": {str(index): value for index, value in enumerate(answers)},
        }
        # Rank from the unrounded head probabilities so near-ties keep their
        # true order instead of falling back to candidate insertion order.
        with self._lock:
            scored, _ = self._scorer.score(context, {"rank": request})
        ordered = sorted(scored["rank"].items(), key=lambda item: -item[1])
        return [
            {"rank": rank + 1, "candidate": answers[int(index)], "prob": probability}
            for rank, (index, probability) in enumerate(ordered)
        ]

    def models(self) -> list[dict]:
        return [
            {
                "name": self.default_model,
                "backend": "clef-mlx",
                "hf_id": self.repo_id,
                "description": "Cloudflare Clef typed decisions on native MLX",
            }
        ]


class DeciderBackend:
    """Decider typed decisions on the native MLX Qwen3.5 text backbone.

    Decider reads the state and one question, then scores the answer labels at
    the last position with the checkpoint's own per-type calibration. It does
    not generate text, so it cannot be served as a chat model.
    """

    _MODELS = {
        "decider-2b": (
            "nativ-community/decider-2b",
            "acbae4ecce4dbcc0aea8c5a501c9f54008458a70",
        ),
    }

    def __init__(self, model: str, *, device: str = "gpu") -> None:
        if device not in {"gpu", "cpu"}:
            raise ValueError("Decider device must be 'gpu' or 'cpu'")
        local = Path(model).expanduser()
        selected = model.rsplit("/", 1)[-1].lower()
        if local.is_dir():
            # A local directory serves converted or fine-tuned checkpoints.
            path = str(local)
            self.default_model = local.name
            self.repo_id = str(local)
        elif selected in self._MODELS and model.lower() in {
            selected,
            self._MODELS[selected][0].lower(),
        }:
            # Match the CLI, which routes model names case-insensitively.
            from rapid_mlx._mirror import pinned_snapshot_download

            self.default_model = selected
            self.repo_id, revision = self._MODELS[selected]
            path = pinned_snapshot_download(self.repo_id, revision)
        else:
            raise ValueError(
                f"unknown Decider model {model!r}; choose "
                f"{', '.join(self._MODELS)} or a local checkpoint directory"
            )

        import mlx.core as mx

        from .decider import DeciderScorer, load_decider

        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        text_model, tokenizer, settings = load_decider(path)
        self._scorer = DeciderScorer(text_model, tokenizer, settings)
        self._lock = threading.Lock()

    def _check(self, model: str, temperature: float) -> None:
        if model not in (self.default_model, self.repo_id):
            raise KeyError(f"unknown model {model!r}; available: {self.default_model}")
        if temperature != 1.0:
            raise ValueError(
                "Decider uses checkpoint calibration and requires temperature=1"
            )

    def _score(self, state: Any, questions: dict[str, Question]):
        with self._lock:
            return self._scorer.score(state, questions)

    def answer(
        self, state: Any, questions: dict[str, Question], model: str, temperature: float
    ) -> dict:
        self._check(model, temperature)
        scored, tokens = self._score(state, questions)
        answers_out = {}
        for question_id, (keys, probabilities) in scored.items():
            answer = answer_from_probabilities(
                questions[question_id], keys, probabilities
            )
            # Decider publishes the winning probability as its confidence.
            if "confidence" in answer:
                answer["confidence"] = max(probabilities)
            answers_out[question_id] = answer
        return {
            "model": self.default_model,
            "answers": answers_out,
            "usage": {
                "billing_units": len(questions),
                "input_tokens": tokens,
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
        self._check(model, temperature)
        request = Question(
            type="choice",
            instructions=question or "Choose the best answer.",
            criteria={str(index): value for index, value in enumerate(answers)},
        )
        scored, _ = self._score(context, {"rank": request})
        keys, probabilities = scored["rank"]
        ordered = sorted(zip(keys, probabilities), key=lambda item: -item[1])
        return [
            {"rank": rank + 1, "candidate": answers[int(index)], "prob": probability}
            for rank, (index, probability) in enumerate(ordered)
        ]

    def models(self) -> list[dict]:
        return [
            {
                "name": self.default_model,
                "backend": "decider-mlx",
                "hf_id": self.repo_id,
                "description": "Decider typed decisions on native MLX",
            }
        ]


class OpenJevBackend:
    """OpenJev typed decisions on the native MLX Qwen3.8 text model.

    OpenJev reads the option letters at the first output position of a chat
    prompt and applies the release calibration. The published MLX conversion
    has no vision tower, so this backend takes text and JSON state only.

    The weights are licensed CC BY-NC 4.0: non-commercial use only.
    """

    LICENSE_NOTICE = (
        "OpenJev weights are licensed CC BY-NC 4.0 (non-commercial use only); "
        "commercial use needs a licence from the OpenJev authors."
    )
    _MODELS = {
        "openjev": (
            "openjev/openjev-MLX",
            "a9dcc20aa827a6c7eae478f6ebb3b255bb135451",
        ),
    }

    def __init__(self, model: str, *, device: str = "gpu") -> None:
        if device not in {"gpu", "cpu"}:
            raise ValueError("OpenJev device must be 'gpu' or 'cpu'")
        local = Path(model).expanduser()
        repos = {repo.lower(): name for name, (repo, _) in self._MODELS.items()}
        selected = repos.get(model.lower(), model.lower())
        if local.is_dir():
            # A local directory serves other conversions, such as a 4-bit build.
            path = str(local)
            self.default_model = local.name
            self.repo_id = str(local)
        elif selected in self._MODELS:
            from rapid_mlx._mirror import pinned_snapshot_download

            self.default_model = selected
            self.repo_id, revision = self._MODELS[selected]
            path = pinned_snapshot_download(self.repo_id, revision)
        else:
            raise ValueError(
                f"unknown OpenJev model {model!r}; choose "
                f"{', '.join(self._MODELS)} or a local checkpoint directory"
            )

        import mlx.core as mx

        from .openjev import OpenJevScorer, load_openjev

        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        self._scorer = OpenJevScorer(*load_openjev(path))
        self._lock = threading.Lock()

    def _check(self, model: str, temperature: float) -> None:
        if model not in (self.default_model, self.repo_id):
            raise KeyError(f"unknown model {model!r}; available: {self.default_model}")
        if temperature != 1.0:
            raise ValueError(
                "OpenJev uses release calibration and requires temperature=1"
            )

    def _score(self, state: Any, questions: dict[str, Question]):
        with self._lock:
            return self._scorer.score(state, questions)

    def answer(
        self, state: Any, questions: dict[str, Question], model: str, temperature: float
    ) -> dict:
        from .openjev import openjev_answer

        self._check(model, temperature)
        scored, tokens = self._score(state, questions)
        return {
            "model": self.default_model,
            "answers": {
                question_id: openjev_answer(
                    questions[question_id], options, probabilities
                )
                for question_id, (options, probabilities) in scored.items()
            },
            "usage": {
                "billing_units": len(questions),
                "input_tokens": tokens,
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
        self._check(model, temperature)
        request = Question(
            type="choice",
            instructions=question or "Choose the best answer.",
            criteria={str(index): value for index, value in enumerate(answers)},
        )
        scored, _ = self._score(context, {"rank": request})
        options, probabilities = scored["rank"]
        ordered = sorted(zip(options, probabilities), key=lambda item: -item[1])
        return [
            {"rank": rank + 1, "candidate": answers[int(key)], "prob": probability}
            for rank, ((key, _), probability) in enumerate(ordered)
        ]

    def models(self) -> list[dict]:
        return [
            {
                "name": self.default_model,
                "backend": "openjev-mlx",
                "hf_id": self.repo_id,
                "license": "CC-BY-NC-4.0",
                "description": "OpenJev typed decisions on native MLX (text only)",
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
        for _text, token_ids in zip(texts, token_rows, strict=True):
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
