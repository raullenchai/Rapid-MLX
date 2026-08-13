# SPDX-License-Identifier: Apache-2.0
"""
Embedding engine using mlx-embeddings.

Provides lazy-loaded model management and batch embedding generation
for the OpenAI-compatible /v1/embeddings endpoint.
"""

import logging
import sys
import time

import mlx.core as mx

logger = logging.getLogger(__name__)

# Canonical install-hint copy. Shared between the CLI startup probe
# (H-08) and the ``/v1/embeddings`` route guard (H-09) so the user sees
# the same actionable line no matter which surface tripped the guard.
EMBEDDINGS_EXTRA_INSTALL_HINT = "Install with: pip install 'rapid-mlx[embeddings]'"

# HuggingFace stamps ``tokenizer.model_max_length`` with a huge sentinel
# (``VERY_LARGE_INTEGER`` ≈ 1e30) when the tokenizer config declares no real
# limit. Any "maximum" at or above this threshold is treated as *unset*
# rather than as a genuine multi-quadrillion-token context (issue #1381).
_MODEL_MAX_SENTINEL_THRESHOLD = 1_000_000
# Conservative fallback when neither the model config nor the tokenizer
# declares a usable maximum. Matches the historical hardcoded ceiling, so a
# model with no discoverable limit keeps its prior behavior — except the
# truncation is now surfaced (warning + metric) instead of silent.
_FALLBACK_MAX_LENGTH = 512

# Operator-facing overflow policies for inputs longer than the effective
# max input length.
OVERFLOW_POLICIES = ("truncate", "error")


class EmbeddingInputTooLongError(Exception):
    """Raised under the ``error`` overflow policy when an embedding input
    exceeds the effective max input length.

    The ``/v1/embeddings`` route maps this to a structured 4xx carrying the
    observed and allowed token counts so the caller can react precisely
    instead of silently indexing a truncated vector (issue #1381).
    """

    def __init__(self, *, observed_tokens: int, allowed_tokens: int, index: int):
        self.observed_tokens = observed_tokens
        self.allowed_tokens = allowed_tokens
        self.index = index
        super().__init__(
            f"input {index} has {observed_tokens} tokens, which exceeds the "
            f"configured embedding max input length of {allowed_tokens}"
        )


def mlx_embeddings_available() -> bool:
    """Probe whether ``mlx_embeddings`` is importable.

    Uses :func:`importlib.util.find_spec` so we only answer "no" for
    the specific case the install hint is meant to address — the
    top-level ``mlx_embeddings`` package isn't installed. A broken
    transitive dependency raising ``ImportError`` deep inside the
    package surfaces as the real exception (not masked behind the
    "install the extra" hint), making misdiagnosis less likely
    (raised in Codex review on PR #800).

    Lazy resolution — keeps the base install (without the
    ``[embeddings]`` extra) free of ``mlx_embeddings`` at module
    top-level. Callers decide what to do when ``False``:

    * CLI startup (:mod:`vllm_mlx.cli`, :mod:`vllm_mlx.server`) calls
      :func:`require_mlx_embeddings_or_exit` when ``--embedding-model``
      is passed so the user gets a clear install hint on stderr and
      ``sys.exit(2)`` — H-08 fix.
    * The ``/v1/embeddings`` route (:mod:`vllm_mlx.routes.embeddings`)
      raises a 400 with the same hint when no embedding model is
      configured — H-09 fix.
    """
    import importlib.util

    return importlib.util.find_spec("mlx_embeddings") is not None


def require_mlx_embeddings_or_exit() -> None:
    """CLI-side guard: bail out cleanly when ``--embedding-model`` is
    passed but the ``[embeddings]`` extra isn't installed.

    H-08: previously the server crashed deep inside
    :meth:`EmbeddingEngine.load` with a raw ``ModuleNotFoundError``
    traceback because the help text advertises ``--embedding-model``
    while ``mlx_embeddings`` lives behind the ``[embeddings]`` extra.
    Probe at flag-parse time and exit ``2`` (the conventional argparse
    usage-error code) with an actionable hint to stderr.
    """
    if mlx_embeddings_available():
        return
    print(
        "error: --embedding-model requires the [embeddings] extra. "
        + EMBEDDINGS_EXTRA_INSTALL_HINT,
        file=sys.stderr,
    )
    sys.exit(2)


def normalize_max_length_setting(max_length: int | str) -> int | str:
    """Validate an operator ``--embedding-max-length`` value.

    Accepts the literal ``"auto"`` (case-insensitive) or a positive
    integer (int, or its string form). Returns ``"auto"`` or an ``int``.
    Raises ``ValueError`` otherwise.
    """
    if isinstance(max_length, bool):
        raise ValueError(
            "embedding max_length must be 'auto' or a positive integer, "
            f"got {max_length!r}"
        )
    if isinstance(max_length, str):
        if max_length.strip().lower() == "auto":
            return "auto"
        try:
            max_length = int(max_length)
        except ValueError:
            raise ValueError(
                "embedding max_length must be 'auto' or a positive integer, "
                f"got {max_length!r}"
            ) from None
    if not isinstance(max_length, int):
        raise ValueError(
            "embedding max_length must be 'auto' or a positive integer, "
            f"got {max_length!r}"
        )
    if max_length < 1:
        raise ValueError(f"embedding max_length must be >= 1, got {max_length}")
    return max_length


class EmbeddingEngine:
    """
    Wrapper around mlx-embeddings for text embedding generation.

    Supports lazy model loading and batch embedding with proper
    tokenization and pooling. The effective max input length is resolved
    against the loaded model (issue #1381): inputs longer than the limit
    are never truncated silently — under the ``truncate`` policy they emit
    a warning and bump a metric, and under ``error`` they raise
    :class:`EmbeddingInputTooLongError`.
    """

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int | str = "auto",
        overflow_policy: str = "truncate",
    ):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        if overflow_policy not in OVERFLOW_POLICIES:
            raise ValueError(
                f"overflow_policy must be one of {OVERFLOW_POLICIES}, "
                f"got {overflow_policy!r}"
            )
        self._max_length_setting = normalize_max_length_setting(max_length)
        self.overflow_policy = overflow_policy
        # Resolved against the loaded model in ``load()``.
        self.effective_max_length: int | None = None
        # Observable, non-silent signal (issue #1381): number of inputs
        # whose tail was discarded under the ``truncate`` policy. Exposed
        # on ``/metrics`` as ``rapid_mlx_embedding_truncations_total``.
        self.num_truncations = 0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the embedding model and tokenizer."""
        from mlx_embeddings import load

        logger.info(f"Loading embedding model: {self.model_name}")
        start = time.perf_counter()
        self._model, self._tokenizer = load(self.model_name)
        elapsed = time.perf_counter() - start
        logger.info(f"Embedding model loaded in {elapsed:.2f}s: {self.model_name}")
        self._resolve_effective_max_length()

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load()

    def _discover_model_max_length(self) -> int | None:
        """Best-effort model-declared max input length, or ``None``.

        Prefers the model's ``config.max_position_embeddings`` (the
        authoritative positional-embedding ceiling), then a couple of
        common aliases, then ``tokenizer.model_max_length`` — guarding
        HuggingFace's large "unset" sentinel so it reads as unknown rather
        than a real limit.
        """
        model_cfg = getattr(self._model, "config", None)
        if model_cfg is None:
            model_cfg = getattr(self._model, "args", None)
        if model_cfg is not None:
            for attr in ("max_position_embeddings", "max_seq_len", "n_positions"):
                val = getattr(model_cfg, attr, None)
                if (
                    isinstance(val, int)
                    and not isinstance(val, bool)
                    and 0 < val < _MODEL_MAX_SENTINEL_THRESHOLD
                ):
                    return val
        inner_tok = getattr(self._tokenizer, "_tokenizer", self._tokenizer)
        tok_max = getattr(inner_tok, "model_max_length", None)
        if (
            isinstance(tok_max, int)
            and not isinstance(tok_max, bool)
            and 0 < tok_max < _MODEL_MAX_SENTINEL_THRESHOLD
        ):
            return tok_max
        return None

    def _resolve_effective_max_length(self) -> None:
        """Resolve ``effective_max_length`` from the operator setting and
        the loaded model's declared maximum."""
        model_max = self._discover_model_max_length()
        setting = self._max_length_setting
        if setting == "auto":
            self.effective_max_length = model_max or _FALLBACK_MAX_LENGTH
            source = "model" if model_max else "fallback"
        else:
            requested = int(setting)
            if model_max is not None and requested > model_max:
                logger.warning(
                    "Requested --embedding-max-length=%d exceeds the model's "
                    "declared maximum of %d; clamping to %d.",
                    requested,
                    model_max,
                    model_max,
                )
                self.effective_max_length = model_max
            else:
                self.effective_max_length = requested
            source = "operator"
        logger.info(
            "Embedding effective max input length: %d tokens "
            "(source=%s, overflow_policy=%s)",
            self.effective_max_length,
            source,
            self.overflow_policy,
        )

    def _enforce_overflow(self, lengths: list[int]) -> None:
        """Apply the overflow policy to per-input true token ``lengths``.

        Under ``error`` raise :class:`EmbeddingInputTooLongError` on the first
        over-limit input. Under ``truncate`` emit a warning and bump the
        ``num_truncations`` metric so the discarded tail is never silent
        (issue #1381). No-op when nothing exceeds the limit.
        """
        limit = self.effective_max_length
        if not limit:
            return
        over = [(i, n) for i, n in enumerate(lengths) if n > limit]
        if not over:
            return
        if self.overflow_policy == "error":
            idx, observed = over[0]
            raise EmbeddingInputTooLongError(
                observed_tokens=observed, allowed_tokens=limit, index=idx
            )
        self.num_truncations += len(over)
        worst = max(n for _, n in over)
        logger.warning(
            "Embedding truncation: %d of %d input(s) exceeded the %d-token "
            "limit (largest=%d tokens); the tail was discarded. Raise "
            "--embedding-max-length or set --embedding-overflow-policy error "
            "to reject over-limit inputs instead.",
            len(over),
            len(lengths),
            limit,
            worst,
        )

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """
        Generate embeddings for one or more texts.

        Args:
            texts: A single string or list of strings.

        Returns:
            List of embedding vectors (one per input text).

        Raises:
            EmbeddingInputTooLongError: under the ``error`` overflow policy when
                an input exceeds ``effective_max_length``.
        """
        self._ensure_loaded()

        if isinstance(texts, str):
            texts = [texts]

        # Tokenize directly instead of using mlx_embeddings.generate(),
        # which has compatibility issues with newer tokenizers (e.g.
        # GemmaTokenizer lacks batch_encode_plus, and the model's __call__
        # expects positional `inputs` not `input_ids` as a kwarg).
        inner_tok = getattr(self._tokenizer, "_tokenizer", self._tokenizer)

        # Measure true (un-truncated) lengths first so overflow is
        # observable and the resolved effective limit — not a hardcoded
        # 512 — governs truncation (issue #1381).
        measured = inner_tok(texts, truncation=False, padding=False)
        lengths = [len(ids) for ids in measured["input_ids"]]
        self._enforce_overflow(lengths)

        # Let the tokenizer perform the (special-token-aware) truncation and
        # padding for the actual forward pass.
        encoded = inner_tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.effective_max_length,
            return_tensors="np",
        )

        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])

        output = self._model(input_ids, attention_mask=attention_mask)

        # text_embeds shape: (batch_size, embedding_dim)
        embeds: mx.array = output.text_embeds

        # Convert to Python lists for JSON serialization
        result = embeds.tolist()

        # Release the Metal buffers this pass allocated (issue #1380). MLX
        # keeps freed buffers in its allocator pool keyed by size, and the
        # ``padding=True`` above makes the batch's sequence length vary from
        # request to request — so nearly every batch asks for a size the pool
        # has never seen and cannot reuse. Without this the pool only grows
        # (measured: ~70 MB retained per input text, 2.3 GB → 24 GB over 320
        # texts). Mirrors every LLM path in this engine, which already clear
        # the cache after a forward.
        mx.clear_cache()

        return result

    def embed_tokens(self, token_batches: list[list[int]]) -> list[list[float]]:
        """Embed pre-tokenized inputs (OpenAI spec input formats 3 and 4).

        Skips the tokenizer entirely — the caller has already produced
        token IDs (typically from a shared HF tokenizer in a retrieval
        pipeline). We still need to right-pad to a uniform length to
        form a batch tensor and build the matching attention mask.

        Args:
            token_batches: List of pre-tokenized inputs. Each inner
                list is a sequence of token IDs.

        Returns:
            List of embedding vectors (one per input).

        Raises:
            EmbeddingInputTooLongError: under the ``error`` overflow policy when
                an input exceeds ``effective_max_length``.
        """
        self._ensure_loaded()

        if not token_batches:
            return []

        lengths = [len(ids) for ids in token_batches]
        self._enforce_overflow(lengths)

        # Pad each sequence to the longest in the batch, capped at the
        # resolved effective limit (issue #1381) — the same ceiling the str
        # path uses — so client-controlled ``input`` cannot allocate
        # unbounded memory.
        limit = self.effective_max_length or _FALLBACK_MAX_LENGTH
        max_len = min(max(lengths), limit)
        pad_id = (
            getattr(self._tokenizer, "pad_token_id", None)
            or getattr(
                getattr(self._tokenizer, "_tokenizer", self._tokenizer),
                "pad_token_id",
                None,
            )
            or 0
        )
        padded = []
        masks = []
        for ids in token_batches:
            ids = list(ids)[:max_len]
            n = len(ids)
            pad = max_len - n
            padded.append(ids + [pad_id] * pad)
            masks.append([1] * n + [0] * pad)

        input_ids = mx.array(padded)
        attention_mask = mx.array(masks)

        output = self._model(input_ids, attention_mask=attention_mask)
        embeds: mx.array = output.text_embeds
        result = embeds.tolist()
        # Same allocator-pool release as embed() (issue #1380): drop the
        # per-batch Metal buffers now that the vectors are Python lists.
        mx.clear_cache()
        return result

    def count_tokens(self, texts: str | list[str]) -> int:
        """Token count for usage reporting.

        Capped at ``effective_max_length`` per input so the reported usage
        matches the tokens actually embedded rather than over-reporting the
        pre-truncation length (issue #1381).
        """
        self._ensure_loaded()

        if isinstance(texts, str):
            texts = [texts]

        limit = self.effective_max_length
        total = 0
        for text in texts:
            try:
                tokens = self._tokenizer.encode(text)
                if isinstance(tokens, list) or hasattr(tokens, "__len__"):
                    n = len(tokens)
                else:
                    n = tokens.size
            except Exception:
                # Fallback: rough estimate of ~4 chars per token
                n = max(1, len(text) // 4)
            total += min(n, limit) if limit else n
        return total
