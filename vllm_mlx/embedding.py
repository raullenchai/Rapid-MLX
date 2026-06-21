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


def mlx_embeddings_available() -> bool:
    """Probe whether ``mlx_embeddings`` is importable.

    Uses :func:`importlib.util.find_spec` so we only answer "no" for
    the specific case the install hint is meant to address — the
    top-level ``mlx_embeddings`` package isn't installed. A broken
    transitive dependency raising ``ImportError`` deep inside the
    package surfaces as the real exception (not masked behind the
    "install the extra" hint), making misdiagnosis less likely
    (codex review nit on PR #800).

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


class EmbeddingEngine:
    """
    Wrapper around mlx-embeddings for text embedding generation.

    Supports lazy model loading and batch embedding with proper
    tokenization and pooling.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

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

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load()

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """
        Generate embeddings for one or more texts.

        Args:
            texts: A single string or list of strings.

        Returns:
            List of embedding vectors (one per input text).
        """
        self._ensure_loaded()

        if isinstance(texts, str):
            texts = [texts]

        # Tokenize directly instead of using mlx_embeddings.generate(),
        # which has compatibility issues with newer tokenizers (e.g.
        # GemmaTokenizer lacks batch_encode_plus, and the model's __call__
        # expects positional `inputs` not `input_ids` as a kwarg).
        inner_tok = getattr(self._tokenizer, "_tokenizer", self._tokenizer)
        encoded = inner_tok(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])

        output = self._model(input_ids, attention_mask=attention_mask)

        # text_embeds shape: (batch_size, embedding_dim)
        embeds: mx.array = output.text_embeds

        # Convert to Python lists for JSON serialization
        return embeds.tolist()

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
        """
        self._ensure_loaded()

        if not token_batches:
            return []

        # Pad each sequence to the longest in the batch, capped at the
        # same 512 ceiling as the str path so client-controlled
        # ``input`` cannot allocate unbounded memory.
        max_len = min(max(len(ids) for ids in token_batches), 512)
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
        return embeds.tolist()

    def count_tokens(self, texts: str | list[str]) -> int:
        """Approximate token count for usage reporting."""
        self._ensure_loaded()

        if isinstance(texts, str):
            texts = [texts]

        total = 0
        for text in texts:
            try:
                tokens = self._tokenizer.encode(text)
                if isinstance(tokens, list) or hasattr(tokens, "__len__"):
                    total += len(tokens)
                else:
                    total += tokens.size
            except Exception:
                # Fallback: rough estimate of ~4 chars per token
                total += max(1, len(text) // 4)
        return total
