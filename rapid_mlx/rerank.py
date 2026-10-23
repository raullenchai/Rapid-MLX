# SPDX-License-Identifier: Apache-2.0
"""Cross-encoder reranking for the ``/v1/rerank`` endpoint.

Each supported model family has a profile in :data:`FAMILIES` that pins the
input format it was trained on. Generative rerankers (Qwen3-Reranker, BGE
reranker v2 Gemma, mxbai-rerank-v2) run on ``mlx-lm``, a base dependency: the
pair is rendered into the family's prompt and the relevance score comes from
the answer-token logits at the next position. XLM-RoBERTa cross-encoders such
as bge-reranker-v2-m3 score the pair with their classification head and run on
the encoder from the ``[embeddings]`` extra. jina-reranker-v3 is listwise: it
reads the query and many documents in one prompt and scores each document by
the cosine similarity of projected hidden states at marker tokens.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# ChatML turns that mxbai2 and jina3 write out by hand upstream.
_CHATML_PREFIX = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"
_CHATML_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
_CROSS_ENCODER_FILES = ["*.json", "model*.safetensors", "sentencepiece.bpe.model"]
# Splits the rendered template around the document so its tokens can be
# truncated without re-tokenizing the query part for every document.
_DOCUMENT_MARK = "\x00rapid-mlx-rerank-document\x00"
_JINA3_FILES = ["*.json", "model*.safetensors", "*.jinja", "*.txt"]
_MAX_BATCH_SEQUENCES = 16
_MAX_BATCH_TOKENS = 16384
_NO_THINKING = "<think>\n\n</think>\n\n"
# Default system message of Qwen2's chat template, which mxbai2 keeps.
_QWEN2_DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)
# Families whose mlx-lm ``Model.__call__`` is exactly ``head(model(x))``, so the
# head can run on the last position only instead of the full sequence.
_SPLIT_HEAD_MODEL_TYPES = frozenset({"gemma", "llama", "qwen2", "qwen3"})


class UnsupportedRerankModelError(ValueError):
    """The checkpoint is not a reranker of a supported family."""


@dataclass(frozen=True, kw_only=True)
class RerankFamily:
    """Input format, scoring and upstream limits of one reranker family."""

    family_id: str
    label: str
    example: str

    model_type: str
    architecture_suffix: str
    # Matched against the model id and the config's ``_name_or_path``. Empty
    # for families recognized by ``architecture`` alone.
    name_markers: tuple[str, ...] = ()

    accepts_instruction: bool
    # The request-specific task; the request's ``instruction`` or
    # ``--rerank-instruction`` replaces it. None leaves it out of the prompt.
    default_instruction: str | None = None
    accepts_prompt: bool
    # The family's judging prompt; ``--rerank-prompt`` replaces it.
    default_prompt: str | None = None

    # Text the family puts in front of each document.
    document_prefix: str = ""

    # "generative", "cross-encoder" or "listwise".
    scoring: str
    # Two tokens score ``sigmoid(first - second)``, one scores ``sigmoid(first)``.
    answer_tokens: tuple[str, ...] = ()
    # Listwise: tokens whose hidden states embed each document and the query.
    marker_tokens: tuple[str, str] | None = None

    # Upstream default pair length; ``max_length="auto"`` never exceeds it. Not
    # read from config.json: ``max_position_embeddings`` is the base model's
    # limit (e.g. 131072 for jina3), not the length the reranker was trained on.
    auto_max_length: int
    max_query_tokens: int | None = None
    max_block_documents: int | None = None


FAMILIES: dict[str, RerankFamily] = {
    family.family_id: family
    for family in (
        RerankFamily(
            family_id="bge2",
            label="BGE reranker v2 Gemma",
            example="BAAI/bge-reranker-v2-gemma",
            model_type="gemma",
            architecture_suffix="ForCausalLM",
            name_markers=("bge-reranker-v2-gemma",),
            accepts_instruction=True,
            default_instruction=(
                "Given a query A and a passage B, determine whether the passage "
                "contains an answer to the query by providing a prediction of "
                "either 'Yes' or 'No'."
            ),
            accepts_prompt=False,
            document_prefix="B: ",
            scoring="generative",
            answer_tokens=("Yes",),
            auto_max_length=1024,
        ),
        RerankFamily(
            family_id="jina3",
            label="jina-reranker-v3",
            example="mlx-community/jina-reranker-v3-4bit-mxfp4",
            model_type="qwen3",
            architecture_suffix="JinaForRanking",
            accepts_instruction=True,
            accepts_prompt=True,
            default_prompt=(
                "You are a search relevance expert who can determine a ranking "
                "of the passages based on how relevant they are to the query. If "
                "the query is a question, how relevant a passage is depends on "
                "how well it answers the question. If not, try to analyze the "
                "intent of the query and assess how well each passage satisfies "
                "the intent. If an instruction is provided, you should follow "
                "the instruction when determining the ranking."
            ),
            scoring="listwise",
            marker_tokens=("<|embed_token|>", "<|rerank_token|>"),
            # Per document; the query has its own max_query_tokens.
            auto_max_length=2048,
            max_query_tokens=512,
            max_block_documents=125,
        ),
        RerankFamily(
            family_id="mxbai2",
            label="mxbai-rerank-v2",
            example="mlx-community/mxbai-rerank-large-v2",
            model_type="qwen2",
            architecture_suffix="ForCausalLM",
            name_markers=("mxbai-rerank",),
            accepts_instruction=True,
            accepts_prompt=True,
            default_prompt=(
                "You are a search relevance expert who evaluates how well "
                "documents match search queries. For each query-document pair, "
                "carefully analyze the semantic relationship between them, then "
                "provide your binary relevance judgment (0 for not relevant, 1 "
                "for relevant).\nRelevance:"
            ),
            document_prefix="document: ",
            scoring="generative",
            answer_tokens=("1", "0"),
            auto_max_length=8192,
        ),
        RerankFamily(
            family_id="qwen3",
            label="Qwen3-Reranker",
            example="mlx-community/Qwen3-Reranker-0.6B-mxfp8",
            model_type="qwen3",
            architecture_suffix="ForCausalLM",
            name_markers=("qwen3-reranker",),
            accepts_instruction=True,
            default_instruction=(
                "Given a web search query, retrieve relevant passages that "
                "answer the query"
            ),
            accepts_prompt=True,
            default_prompt=(
                "Judge whether the Document meets the requirements based on the "
                "Query and the Instruct provided. Note that the answer can only "
                'be "yes" or "no".'
            ),
            document_prefix=" ",
            scoring="generative",
            answer_tokens=("yes", "no"),
            auto_max_length=8192,
        ),
        RerankFamily(
            family_id="roberta",
            label="XLM-RoBERTa cross-encoder",
            example="BAAI/bge-reranker-v2-m3",
            model_type="xlm-roberta",
            architecture_suffix="XLMRobertaForSequenceClassification",
            accepts_instruction=False,
            accepts_prompt=False,
            scoring="cross-encoder",
            auto_max_length=512,
        ),
    )
}


def detect_family(
    config: dict, model_name: str, family_id: str | None = None
) -> RerankFamily:
    """Return the family of ``model_name`` or raise ``UnsupportedRerankModelError``.

    ``family_id`` (``--rerank-model-family``) skips the name match for models
    whose name does not reveal their family; the architecture must still fit.
    """
    model_type = config.get("model_type")
    architectures = [str(arch) for arch in config.get("architectures") or []]
    names = f"{model_name} {config.get('_name_or_path', '')}".lower().replace("_", "-")
    candidates = [FAMILIES[family_id]] if family_id else FAMILIES.values()
    for family in candidates:
        if model_type != family.model_type:
            continue
        if not any(arch.endswith(family.architecture_suffix) for arch in architectures):
            continue
        if (
            not family_id
            and family.name_markers
            and not any(m in names for m in family.name_markers)
        ):
            continue
        return family
    found = f"model_type={model_type}, architectures={architectures or 'unknown'}"
    if family_id:
        family = FAMILIES[family_id]
        raise UnsupportedRerankModelError(
            f"'{model_name}' ({found}) does not fit --rerank-model-family "
            f"{family_id}, which needs model_type={family.model_type} and an "
            f"architecture ending in {family.architecture_suffix}"
        )
    supported = ", ".join(
        f"{f.family_id} ({f.label}, e.g. {f.example})" for f in FAMILIES.values()
    )
    raise UnsupportedRerankModelError(
        f"'{model_name}' ({found}) is not a recognized reranker. Supported "
        f"families: {supported}. If the model belongs to one of them under "
        "another name, pass --rerank-model-family."
    )


OVERFLOW_POLICIES = ("truncate", "error")


def normalize_max_length(max_length: int | str) -> int | str:
    """Return ``"auto"`` or a positive ``int`` for a ``--rerank-max-length`` value."""
    if isinstance(max_length, str):
        if max_length.strip().lower() == "auto":
            return "auto"
        try:
            max_length = int(max_length)
        except ValueError:
            raise ValueError(
                f"must be 'auto' or a positive integer, got {max_length!r}"
            ) from None
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise ValueError(f"must be 'auto' or a positive integer, got {max_length!r}")
    if max_length < 1:
        raise ValueError(f"must be >= 1, got {max_length}")
    return max_length


class RerankDocumentTooLongError(Exception):
    """A query/document pair exceeds the input window under the ``error`` policy."""

    def __init__(self, *, index: int, observed_tokens: int, allowed_tokens: int):
        self.index = index
        self.observed_tokens = observed_tokens
        self.allowed_tokens = allowed_tokens
        super().__init__(
            f"document {index} needs {observed_tokens} tokens with the query, "
            f"which exceeds the {allowed_tokens}-token rerank input window"
        )


class RerankInputTooLongError(Exception):
    """The query alone exceeds the reranker's input window."""

    def __init__(self, *, observed_tokens: int, allowed_tokens: int):
        self.observed_tokens = observed_tokens
        self.allowed_tokens = allowed_tokens
        super().__init__(
            f"query and instruction need {observed_tokens} tokens, which leaves "
            f"no room for a document within the {allowed_tokens}-token "
            "rerank input window"
        )


class RerankInstructionNotSupportedError(ValueError):
    """The request sets ``instruction`` but the model family takes none."""


@dataclass(frozen=True)
class RerankScores:
    scores: list[float]
    prompt_tokens: int


def _read_model_config(model_name: str) -> dict[str, Any]:
    local = Path(os.path.expanduser(model_name))
    if local.is_dir():
        config_path = local / "config.json"
    else:
        from . import model_downloads

        config_path = Path(model_downloads.hf_hub_download(model_name, "config.json"))
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    return config


class _ClassificationHead(nn.Module):
    """``XLMRobertaClassificationHead``: dense + tanh + projection on ``<s>``."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, 1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_proj(mx.tanh(self.dense(x)))


class _CrossEncoder(nn.Module):
    """``XLMRobertaForSequenceClassification`` with a single relevance logit."""

    def __init__(self, encoder: nn.Module, hidden_size: int):
        super().__init__()
        self.roberta = encoder
        self.classifier = _ClassificationHead(hidden_size)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        hidden = self.roberta(input_ids, attention_mask=attention_mask)
        return self.classifier(hidden.last_hidden_state[:, 0])[:, 0]


def _load_cross_encoder(model_name: str, config: dict):
    import importlib.util

    if importlib.util.find_spec("mlx_embeddings") is None:
        from .runtime.optional_runtime import optional_extra_install_hint

        raise UnsupportedRerankModelError(
            f"'{model_name}' is an XLM-RoBERTa cross-encoder, which needs the "
            "[embeddings] extra. " + optional_extra_install_hint("embeddings")
        )
    if config.get("num_labels", len(config.get("id2label") or {0: None})) != 1:
        raise UnsupportedRerankModelError(
            f"'{model_name}' is a sequence classifier with more than one label, "
            "not a relevance cross-encoder"
        )
    from mlx_embeddings.models import xlm_roberta
    from transformers import AutoTokenizer

    from . import model_downloads

    path = Path(os.path.expanduser(model_name))
    if not path.is_dir():
        path = Path(
            model_downloads.snapshot_download(
                model_name, allow_patterns=_CROSS_ENCODER_FILES
            )
        )
    args = xlm_roberta.ModelArgs.from_dict({**config, "add_pooling_layer": False})
    model = _CrossEncoder(xlm_roberta.Model(args), args.hidden_size)
    weights: dict[str, mx.array] = {}
    for file in sorted(path.glob("model*.safetensors")):
        weights.update(mx.load(str(file)))
    weights = {k: v for k, v in weights.items() if not k.endswith("position_ids")}
    try:
        model.load_weights(list(weights.items()), strict=True)
    except ValueError as exc:
        raise UnsupportedRerankModelError(
            f"'{model_name}' does not have the XLMRobertaForSequenceClassification "
            f"weight layout: {exc}"
        ) from exc
    model.eval()
    mx.eval(model.parameters())
    from .utils.model_file_guard import remote_code_enabled

    return model, AutoTokenizer.from_pretrained(
        str(path), trust_remote_code=remote_code_enabled()
    )


def _jina_model_classes(config: dict):
    from mlx_lm.models import qwen3

    class JinaForRanking(qwen3.Model):
        """Qwen3 backbone plus the projector that maps marker states to 512 dims."""

        def __init__(self, args):
            super().__init__(args)
            hidden = args.hidden_size
            self.projector = nn.Sequential(
                nn.Linear(hidden, hidden // 2, bias=False),
                nn.ReLU(),
                nn.Linear(hidden // 2, 512, bias=False),
            )

        def sanitize(self, weights):
            # Transformers names the projector layers ``projector.0`` and
            # ``projector.2``; MLX conversions use ``projector.layers.N``.
            weights = {
                re.sub(r"^projector\.(\d+)\.", r"projector.layers.\1.", k): v
                for k, v in weights.items()
            }
            return super().sanitize(weights)

    return JinaForRanking, qwen3.ModelArgs


def _load_jina(model_name: str):
    from mlx_lm.utils import load_model, load_tokenizer

    from . import model_downloads

    path = Path(os.path.expanduser(model_name))
    if not path.is_dir():
        path = Path(
            model_downloads.snapshot_download(model_name, allow_patterns=_JINA3_FILES)
        )
    from .model_load_errors import load_model_checked

    try:
        model, _config = load_model_checked(
            load_model, path, get_model_classes=_jina_model_classes
        )
    except ValueError as exc:
        raise UnsupportedRerankModelError(
            f"'{model_name}' does not have the JinaForRanking weight layout: {exc}"
        ) from exc
    from .utils.tokenizer import apply_remote_code_policy

    tokenizer_config, _ = apply_remote_code_policy(None)
    return model, load_tokenizer(path, tokenizer_config=tokenizer_config)


class RerankEngine:
    """Scores ``(query, document)`` pairs with a reranker of a supported family."""

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int | str = "auto",
        overflow_policy: str = "truncate",
        instruction: str | None = None,
        prompt: str | None = None,
        model_family: str | None = None,
    ):
        if overflow_policy not in OVERFLOW_POLICIES:
            raise ValueError(
                f"overflow_policy must be one of {OVERFLOW_POLICIES}, "
                f"got {overflow_policy!r}"
            )
        if model_family is not None and model_family not in FAMILIES:
            raise ValueError(
                f"model_family must be one of {tuple(FAMILIES)}, got {model_family!r}"
            )
        self.model_name = model_name
        self.overflow_policy = overflow_policy
        self._family_setting = model_family
        self._instruction_setting = instruction or None
        self.instruction: str | None = None
        self._prompt_setting = prompt or None
        self.prompt: str | None = None
        self.family: RerankFamily | None = None
        self._max_length_setting = normalize_max_length(max_length)
        # One of several model and tokenizer classes, depending on the family.
        self._model: Any = None
        self._tokenizer: Any = None
        self._answer_ids: list[int] = []
        self._marker_ids: tuple[int, int] = (0, 0)
        self._context_length = 0
        self._pad_id = 0
        self.max_length = 0
        self.num_truncations = 0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def accepts_instruction(self) -> bool:
        return self.family is not None and self.family.accepts_instruction

    @property
    def _loaded_family(self) -> RerankFamily:
        family = self.family
        assert family is not None, "rerank model is not loaded"
        return family

    def load(self) -> None:
        from . import model_downloads

        model_downloads.require_local(self.model_name)
        config = _read_model_config(self.model_name)
        family = detect_family(config, self.model_name, self._family_setting)
        if self._instruction_setting and not family.accepts_instruction:
            raise UnsupportedRerankModelError(
                f"'{self.model_name}' ({family.label}) takes no instruction; "
                "remove --rerank-instruction"
            )
        if self._prompt_setting and not family.accepts_prompt:
            raise UnsupportedRerankModelError(
                f"'{self.model_name}' ({family.label}) takes no prompt; "
                "remove --rerank-prompt"
            )

        logger.info("Loading rerank model: %s (%s)", self.model_name, family.label)
        start = time.perf_counter()
        # Untyped config.json value, validated by _resolve_max_length.
        model_max: Any
        if family.scoring == "generative":
            from rapid_mlx.utils.tokenizer import load_model_with_fallback as load_lm

            model, tokenizer = load_lm(self.model_name)
            model_max = config.get("max_position_embeddings")
        elif family.scoring == "listwise":
            model, tokenizer = _load_jina(self.model_name)
            model_max = config.get("max_position_embeddings")
        else:
            model, tokenizer = _load_cross_encoder(self.model_name, config)
            # XLM-RoBERTa offsets positions past the padding index.
            model_max = config.get("max_position_embeddings", 0) - (
                config.get("pad_token_id", 1) + 1
            )
        if family.family_id == "qwen3" and not getattr(
            tokenizer, "chat_template", None
        ):
            raise UnsupportedRerankModelError(
                f"'{self.model_name}' has no chat template to render rerank prompts"
            )
        answer_ids = []
        for token in family.answer_tokens:
            ids = list(tokenizer.encode(token, add_special_tokens=False))
            if len(ids) != 1:
                raise UnsupportedRerankModelError(
                    f"'{self.model_name}' has no single '{token}' answer token"
                )
            answer_ids.append(int(ids[0]))
        if family.scoring == "listwise":
            assert family.marker_tokens is not None
            markers = []
            for token in family.marker_tokens:
                ids = list(tokenizer.encode(token, add_special_tokens=False))
                if len(ids) != 1:
                    raise UnsupportedRerankModelError(
                        f"'{self.model_name}' has no single '{token}' marker token"
                    )
                markers.append(int(ids[0]))
            self._marker_ids = (markers[0], markers[1])
            self._context_length = int(model_max)
        self._model, self._tokenizer = model, tokenizer
        self.family = family
        self.instruction = self._instruction_setting or family.default_instruction
        self.prompt = self._prompt_setting or family.default_prompt
        self._answer_ids = answer_ids
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None)
        self._pad_id = int(pad_id) if isinstance(pad_id, int) else 0
        self.max_length = self._resolve_max_length(model_max)
        logger.info(
            "Rerank model loaded in %.2fs: %s (max %d tokens per pair)",
            time.perf_counter() - start,
            self.model_name,
            self.max_length,
        )

    def _resolve_max_length(self, declared: object) -> int:
        model_max = (
            declared
            if isinstance(declared, int)
            and not isinstance(declared, bool)
            and declared > 0
            else None
        )
        setting = self._max_length_setting
        # normalize_max_length returns "auto" as its only string value.
        if isinstance(setting, str):
            auto = self._loaded_family.auto_max_length
            return min(model_max or auto, auto)
        if model_max is not None and setting > model_max:
            logger.warning(
                "Requested --rerank-max-length=%d exceeds the model's declared "
                "maximum of %d; clamping to %d.",
                setting,
                model_max,
                model_max,
            )
            return model_max
        return setting

    def _prompt_parts(
        self, query: str, instruction: str | None
    ) -> tuple[list[int], list[int]]:
        """Token ids before and after the document for this family's format."""
        tokenizer = self._tokenizer
        family_id = self._loaded_family.family_id

        def encode(text: str) -> list[int]:
            return list(tokenizer.encode(text, add_special_tokens=False))

        if family_id == "bge2":
            # bge2 has a default instruction, so this is never None.
            assert instruction is not None
            head = [tokenizer.bos_token_id] + encode(f"A: {query}") + encode("\n")
            return head, encode("\n") + encode(instruction)
        if family_id == "mxbai2":
            query_prompt = f"query: {query}"
            if instruction:
                query_prompt = f"instruction: {instruction}\n{query_prompt}"
            prefix = _CHATML_PREFIX.format(system=_QWEN2_DEFAULT_SYSTEM_PROMPT)
            head = encode(prefix) + encode(query_prompt) + encode("\n")
            # mxbai2 was trained with its judging prompt after the document.
            assert self.prompt is not None
            tail = encode(self.prompt) + encode(_CHATML_SUFFIX)
            return head, encode("\n") + tail
        if family_id == "roberta":
            # ``<s> query </s></s> document </s>``, as the pair tokenizer builds it.
            sep = tokenizer.sep_token_id
            return [tokenizer.cls_token_id] + encode(query) + [sep, sep], [sep]
        messages = [
            {"role": "system", "content": self.prompt},
            {
                "role": "user",
                "content": (
                    f"<Instruct>: {instruction}\n<Query>: {query}\n"
                    f"<Document>:{_DOCUMENT_MARK}"
                ),
            },
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        head, tail = rendered.rsplit(_DOCUMENT_MARK, 1)
        return encode(head), encode(tail)

    def _build_sequences(
        self, query: str, documents: list[str], instruction: str | None
    ) -> list[list[int]]:
        head, tail = self._prompt_parts(query, instruction)
        budget = self.max_length - len(head) - len(tail)
        if budget < 1:
            raise RerankInputTooLongError(
                observed_tokens=len(head) + len(tail),
                allowed_tokens=self.max_length,
            )
        prefix = self._loaded_family.document_prefix
        sequences = []
        truncated = 0
        for index, document in enumerate(documents):
            doc_ids = list(
                self._tokenizer.encode(prefix + document, add_special_tokens=False)
            )
            if len(doc_ids) > budget:
                if self.overflow_policy == "error":
                    raise RerankDocumentTooLongError(
                        index=index,
                        observed_tokens=len(head) + len(doc_ids) + len(tail),
                        allowed_tokens=self.max_length,
                    )
                doc_ids = doc_ids[:budget]
                truncated += 1
            sequences.append(head + doc_ids + tail)
        if truncated:
            self.num_truncations += truncated
            logger.warning(
                "Rerank truncation: %d of %d document(s) exceeded the %d-token "
                "pair limit and were truncated. Set --rerank-overflow-policy "
                "error to reject them instead.",
                truncated,
                len(documents),
                self.max_length,
            )
        return sequences

    def _answer_logits(self, batch: mx.array, last: mx.array) -> mx.array:
        rows = mx.arange(batch.shape[0])
        model = self._model
        inner = getattr(model, "model", None)
        split_head = getattr(model, "model_type", None) in _SPLIT_HEAD_MODEL_TYPES
        if split_head and inner is not None:
            hidden = inner(batch)[rows, last]
            head = getattr(model, "lm_head", None)
            logits = (
                head(hidden)
                if head is not None
                else inner.embed_tokens.as_linear(hidden)
            )
        else:
            logits = model(batch)[rows, last]
        return mx.take(logits, mx.array(self._answer_ids), axis=-1)

    def _score_batch(self, sequences: list[list[int]]) -> list[float]:
        width = max(len(ids) for ids in sequences)
        # Right padding is exact under the causal mask: real positions never
        # attend to the trailing pad tokens. The bidirectional cross-encoder
        # masks them explicitly.
        padded = mx.array(
            [ids + [self._pad_id] * (width - len(ids)) for ids in sequences]
        )
        if self._loaded_family.scoring == "cross-encoder":
            mask = mx.array(
                [[1] * len(ids) + [0] * (width - len(ids)) for ids in sequences]
            )
            scores = mx.sigmoid(self._model(padded, mask).astype(mx.float32))
        else:
            last = mx.array([len(ids) - 1 for ids in sequences])
            logits = self._answer_logits(padded, last).astype(mx.float32)
            if logits.shape[-1] == 2:
                scores = mx.sigmoid(logits[:, 0] - logits[:, 1])
            else:
                scores = mx.sigmoid(logits[:, 0])
        mx.eval(scores)
        values: list[float] = scores.tolist()
        return values

    def _truncate_listwise(
        self, text: str, limit: int, index: int | None
    ) -> tuple[str, int, bool]:
        ids = list(self._tokenizer.encode(text, add_special_tokens=False))
        if len(ids) <= limit:
            return text, len(ids), False
        if self.overflow_policy == "error":
            if index is None:
                raise RerankInputTooLongError(
                    observed_tokens=len(ids), allowed_tokens=limit
                )
            raise RerankDocumentTooLongError(
                index=index, observed_tokens=len(ids), allowed_tokens=limit
            )
        return self._tokenizer.decode(ids[:limit]), limit, True

    def _listwise_markers(self) -> tuple[str, str]:
        marker_tokens = self._loaded_family.marker_tokens
        assert marker_tokens is not None
        return marker_tokens

    def _listwise_clean(self, text: str) -> str:
        doc_token, query_token = self._listwise_markers()
        return text.replace(doc_token, "").replace(query_token, "")

    def _listwise_passage(self, document: str, index: int = 0) -> str:
        doc_token, _ = self._listwise_markers()
        return (
            f'<passage id="{index}">\n{self._listwise_clean(document)}'
            f"{doc_token}\n</passage>"
        )

    def _listwise_prompt(
        self, query: str, documents: list[str], instruction: str | None
    ) -> str:
        _, query_token = self._listwise_markers()

        clean = self._listwise_clean
        query = clean(query)
        prompt = (
            f"I will provide you with {len(documents)} passages, each indicated by "
            "a numerical identifier. Rank the passages based on their relevance "
            f"to query: {query}\n"
        )
        if instruction:
            prompt += f"<instruct>\n{clean(instruction)}\n</instruct>\n"
        prompt += "\n".join(
            self._listwise_passage(doc, i) for i, doc in enumerate(documents)
        )
        prompt += f"\n<query>\n{query}{query_token}\n</query>"
        # Marker tokens must only appear where the scorer expects them.
        prefix = _CHATML_PREFIX.format(system=clean(self.prompt or ""))
        return prefix + prompt + _CHATML_SUFFIX + _NO_THINKING

    def _embed_listwise_block(
        self, query: str, documents: list[str], instruction: str | None
    ) -> tuple[mx.array, mx.array, int]:
        """Projected query and document marker states for one prompt."""
        ids = list(
            self._tokenizer.encode(
                self._listwise_prompt(query, documents, instruction),
                add_special_tokens=False,
            )
        )
        doc_id, query_id = self._marker_ids
        doc_positions = [i for i, token in enumerate(ids) if token == doc_id]
        query_positions = [i for i, token in enumerate(ids) if token == query_id]
        hidden = self._model.model(mx.array(ids)[None])[0]
        projector = self._model.projector
        doc_embeds = projector(hidden[mx.array(doc_positions)]).astype(mx.float32)
        query_embed = projector(hidden[mx.array(query_positions[-1:])])
        return query_embed[0].astype(mx.float32), doc_embeds, len(ids)

    def _score_listwise(
        self, query: str, documents: list[str], instruction: str | None
    ) -> RerankScores:
        """jina-reranker-v3 ``rerank()``: blocks of documents, then a rescore
        against the block-weighted query state."""
        family = self._loaded_family
        assert family.max_query_tokens is not None
        assert family.max_block_documents is not None
        max_block_documents = family.max_block_documents
        query, query_length, _ = self._truncate_listwise(
            query, family.max_query_tokens, None
        )
        truncated_indices: set[int] = set()
        docs, lengths = [], []
        for index, document in enumerate(documents):
            text, length, cut = self._truncate_listwise(
                document, self.max_length, index
            )
            lengths.append(length)
            if cut:
                truncated_indices.add(index)
            docs.append(text)

        def prompt_length(block_docs: list[str]) -> int:
            return len(
                self._tokenizer.encode(
                    self._listwise_prompt(query, block_docs, instruction),
                    add_special_tokens=False,
                )
            )

        # Account for the complete rendered prompt. In particular, the ChatML
        # turns and passage wrappers are not represented in the raw document
        # token counts used by ``_truncate_listwise``.
        empty_length = prompt_length([""])
        if empty_length > self._context_length:
            raise RerankInputTooLongError(
                observed_tokens=empty_length, allowed_tokens=self._context_length
            )

        def exact_singleton(index: int, observed: int) -> int:
            document = docs[index]
            if observed <= self._context_length:
                return observed
            if self.overflow_policy == "error":
                raise RerankDocumentTooLongError(
                    index=index,
                    observed_tokens=observed,
                    allowed_tokens=self._context_length,
                )
            ids = list(self._tokenizer.encode(document, add_special_tokens=False))
            low, high = 0, len(ids)
            while low < high:
                midpoint = (low + high + 1) // 2
                if (
                    prompt_length([self._tokenizer.decode(ids[:midpoint])])
                    <= self._context_length
                ):
                    low = midpoint
                else:
                    high = midpoint - 1
            docs[index] = self._tokenizer.decode(ids[:low])
            lengths[index] = low
            truncated_indices.add(index)
            return exact_singleton(index, prompt_length([docs[index]]))

        empty_passage_length = len(
            self._tokenizer.encode(self._listwise_passage(""), add_special_tokens=False)
        )
        singleton_lengths = []
        for index, document in enumerate(docs):
            observed = (
                empty_length
                - empty_passage_length
                + len(
                    self._tokenizer.encode(
                        self._listwise_passage(document), add_special_tokens=False
                    )
                )
            )
            # Tokenization across wrapper boundaries need not be additive.
            if observed >= self._context_length - 16:
                observed = exact_singleton(index, prompt_length([document]))
            singleton_lengths.append(observed)

        two_empty_length = prompt_length(["", ""])
        # A second empty passage measures its wrapper, joining newline, and id.
        # Extra room per passage conservatively covers wider header/id digits;
        # the exact validation below handles tokenizer non-additivity.
        per_doc_slack = max(0, two_empty_length - empty_length) + len(
            str(max_block_documents)
        )
        fixed_slack = len(str(max_block_documents))

        packed_blocks: list[list[int]] = []
        block: list[int] = []
        estimated_length = empty_length + fixed_slack
        # Upstream ``rerank()`` also closes a block once the raw token budget
        # left for documents drops to one maximum-length document; keep that
        # rule so block boundaries (and the block-weighted query) match it.
        upstream_budget = self._context_length - 2 * query_length
        capacity = upstream_budget
        for index in range(len(docs)):
            candidate_length = (
                estimated_length
                + singleton_lengths[index]
                - empty_length
                + per_doc_slack
            )
            if block and (
                len(block) >= max_block_documents
                or candidate_length > self._context_length
            ):
                packed_blocks.append(block)
                block = [index]
                capacity = upstream_budget
                estimated_length = (
                    empty_length
                    + fixed_slack
                    + singleton_lengths[index]
                    - empty_length
                    + per_doc_slack
                )
            else:
                block.append(index)
                estimated_length = candidate_length
            capacity -= lengths[index]
            if len(block) >= max_block_documents or capacity <= self.max_length:
                packed_blocks.append(block)
                block = []
                capacity = upstream_budget
                estimated_length = empty_length + fixed_slack
        if block:
            packed_blocks.append(block)

        def validated_blocks(block: list[int]) -> list[list[int]]:
            observed = prompt_length([docs[i] for i in block])
            if observed <= self._context_length:
                return [block]
            if len(block) == 1:
                exact_singleton(block[0], observed)
                return [block]
            # Bisect non-additive blocks; singletons use exact overflow handling.
            midpoint = len(block) // 2
            return validated_blocks(block[:midpoint]) + validated_blocks(
                block[midpoint:]
            )

        blocks = [
            verified for block in packed_blocks for verified in validated_blocks(block)
        ]

        truncated = len(truncated_indices)
        if truncated:
            self.num_truncations += truncated
            logger.warning(
                "Rerank truncation: %d of %d document(s) exceeded a token limit "
                "and were truncated. Set --rerank-overflow-policy error to "
                "reject them instead.",
                truncated,
                len(documents),
            )

        def cosine(query_embed: mx.array, doc_embeds: mx.array) -> mx.array:
            norms = mx.linalg.norm(doc_embeds, axis=-1) * mx.linalg.norm(query_embed)
            # A zero projected vector has no direction; score it as orthogonal
            # (cosine 0) instead of dividing by zero.
            return (doc_embeds @ query_embed) / mx.maximum(norms, 1e-12)

        query_embeds, doc_embeds, weights = [], [], []
        prompt_tokens = 0
        for block in blocks:
            query_embed, block_docs, tokens = self._embed_listwise_block(
                query, [docs[i] for i in block], instruction
            )
            query_embeds.append(query_embed)
            doc_embeds.append(block_docs)
            weights.append(mx.max((1.0 + cosine(query_embed, block_docs)) / 2.0))
            prompt_tokens += tokens
        weight = mx.stack(weights)
        # All-zero weights (every block maximally dissimilar) fall back to an
        # unweighted mean instead of dividing by zero.
        weight = mx.where(weight.sum() > 0, weight, mx.ones_like(weight))
        query_embed = (mx.stack(query_embeds) * weight[:, None]).sum(0) / weight.sum()
        # Cosine similarity is in [-1, 1]; map it to the API's [0, 1] range.
        scores = (1.0 + cosine(query_embed, mx.concatenate(doc_embeds))) / 2.0
        # Clamp float rounding so scores never leave the documented range.
        scores = mx.clip(scores, 0.0, 1.0)
        mx.eval(scores)
        return RerankScores(scores=scores.tolist(), prompt_tokens=prompt_tokens)

    def score(
        self,
        query: str,
        documents: list[str],
        *,
        instruction: str | None = None,
    ) -> RerankScores:
        """Return one relevance score in ``[0, 1]`` per document, in input order."""
        if not self.is_loaded:
            self.load()
        if instruction and not self.accepts_instruction:
            raise RerankInstructionNotSupportedError(
                f"{self._loaded_family.label} rerankers take no instruction."
            )
        if not documents:
            return RerankScores(scores=[], prompt_tokens=0)
        if self._loaded_family.scoring == "listwise":
            try:
                return self._score_listwise(
                    query, documents, instruction or self.instruction
                )
            finally:
                mx.clear_cache()

        sequences = self._build_sequences(
            query, documents, instruction or self.instruction
        )
        scores = [0.0] * len(sequences)
        # Length-sorted micro-batches keep padding small.
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        batch: list[int] = []
        try:
            for index in order:
                width = len(sequences[index])
                if batch and (
                    len(batch) >= _MAX_BATCH_SEQUENCES
                    or width * (len(batch) + 1) > _MAX_BATCH_TOKENS
                ):
                    for i, value in zip(
                        batch, self._score_batch([sequences[i] for i in batch])
                    ):
                        scores[i] = value
                    batch = []
                batch.append(index)
            for i, value in zip(
                batch, self._score_batch([sequences[i] for i in batch])
            ):
                scores[i] = value
        finally:
            # Padded batch shapes vary per request; release the pooled buffers.
            mx.clear_cache()
        return RerankScores(
            scores=scores, prompt_tokens=sum(len(ids) for ids in sequences)
        )
