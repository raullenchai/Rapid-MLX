# SPDX-License-Identifier: Apache-2.0
"""
Base engine interface for vllm-mlx inference.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationOutput:
    """
    Output from generation.

    Compatible with both simple and batched engines.
    """

    text: str
    tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = "stop"
    # For streaming
    new_text: str = ""
    finished: bool = True
    # Per-token logprobs (mx.array of shape [vocab_size] for current token)
    logprobs: Any = None
    # Semantic channel: "content", "reasoning", "tool_call", or None
    channel: str | None = None
    # NOTE: keep the following fields LAST, in the order they were added.
    # ``raw_text`` and ``reasoning_text`` were added after v0.6.65 and
    # inserting them in the middle silently rebound positional
    # constructor args for downstream callers (text, tokens, ...) — see
    # codex round-1 review of the v0.6.66 release. New optional fields go
    # at the end of this dataclass to preserve positional compatibility,
    # and existing trailing fields stay pinned in their original order
    # (enforced by ``tests/test_server_utils.py::
    # TestGenerationOutputFieldOrder``).
    # Pre-cleaning model output, preserved so the route's reasoning parser
    # can see harmony channel markers that ``clean_output_text`` strips out
    # of ``text``. Without this, ``HarmonyReasoningParser.extract_reasoning``
    # on the non-stream + no-tool path runs on already-cleaned text and
    # returns ``(None, None)`` — leaking the analysis channel into
    # ``content`` and emitting empty ``reasoning_content`` to clients.
    raw_text: str = ""
    # Token-level reasoning extraction, populated by the engine via
    # ``OutputRouter.feed_sequence`` for tokenizers it supports
    # (Harmony / Gemma 4 / Qwen3 / DeepSeek R1 — see
    # ``output_router.from_tokenizer``). AUTHORITATIVE source of
    # reasoning_content for non-streaming responses: it tracks channel
    # state at the token level instead of regex-parsing the decoded text
    # after the fact, so truncated outputs (``finish_reason=length``,
    # no ``<|end|>`` terminator) still produce correct
    # ``reasoning_content`` without leaking the analysis body into
    # ``content``. Empty string means the engine didn't populate it
    # (no router, or router failed) — routes fall back to the
    # text-based ``ReasoningParser`` in that case. Issue #442.
    reasoning_text: str = ""
    # Pre-parsed structured tool calls surfaced by routers that already
    # speak the model's native tool-call protocol natively (currently
    # ``HarmonyStreamingRouter`` via ``openai-harmony.StreamableParser``).
    # Each entry is ``{"name": str, "arguments": str}`` where
    # ``arguments`` is the JSON string the model produced (verbatim body
    # bytes — no escaping, no normalisation).
    #
    # When present, the route layer SKIPS text-based tool-call
    # extraction (``_parse_tool_calls_with_parser``) and uses these
    # entries directly. This bypasses the wire-text round-trip that
    # previously corrupted tool calls whose JSON arguments happened to
    # contain harmony sentinel substrings (e.g. ``{"text":"<|call|>"}``)
    # — see PR #515 codex round-12/14 BLOCKING. ``None`` means the
    # router did not surface structured calls; the route falls back to
    # the legacy regex-based parser path.
    tool_calls: list[dict] | None = None
    # Number of input prompt tokens served from the prefix cache
    # (``Request.cached_tokens`` from the scheduler). Surfaced through
    # ``Usage.prompt_tokens_details.cached_tokens`` on the OpenAI
    # response and ``cache_read_input_tokens`` on the Anthropic adapter
    # so cost-tracking clients can attribute prefix-cache hits without
    # tokenizer-side estimation. 0 when the engine doesn't run through
    # the prefix-cache path (guided generation, dflash speculative
    # server) — semantically "no cache hits", not "unknown".
    cached_tokens: int = 0


class BaseEngine(ABC):
    """
    Abstract base class for inference engines.

    BatchedEngine implements this interface.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

    @property
    @abstractmethod
    def is_mllm(self) -> bool:
        """Check if this is a multimodal model."""
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        pass

    @property
    def preserve_native_tool_format(self) -> bool:
        """
        Whether to preserve native tool message format.

        When True, role="tool" messages and tool_calls fields are preserved
        instead of being converted to text. Set by server based on tool parser.
        """
        return getattr(self, "_preserve_native_tool_format", False)

    @preserve_native_tool_format.setter
    def preserve_native_tool_format(self, value: bool) -> None:
        self._preserve_native_tool_format = value

    def generate_warmup(self) -> None:  # noqa: B027 — intentional no-op default
        """Run a minimal generation to compile Metal shaders.

        This prevents the first real request from hanging for minutes
        while shaders compile on-demand.

        The default is a no-op; BatchedEngine overrides this.
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate a complete response (non-streaming).

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with complete text
        """
        pass

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream generation token by token.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        pass

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Chat completion (non-streaming).

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with assistant response
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream chat completion token by token.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics. Override in subclasses."""
        return {}

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics. Override in subclasses."""
        return None

    async def abort_request(self, request_id: str) -> bool:
        """Abort an active or queued request when the engine supports it."""
        return False

    # ------------------------------------------------------------------
    # Route-layer contract
    #
    # The OpenAI / Anthropic routes call these directly on the engine —
    # they're declared here so a missing implementation fails at
    # instantiation (ABC enforcement) instead of silently degrading at
    # request time under a ``hasattr`` guard or broad ``try/except``.
    #
    # Bug history this contract closes: #500 (``hasattr(engine,
    # "build_prompt")`` silently disabled cloud routing for ~6 weeks
    # after #155 deleted SimpleEngine which hosted the method) and the
    # v0.6.70 hotfix (``engine.model.estimate_new_tokens`` AttributeError
    # was swallowed by the cloud branch's broad try/except → silent
    # fallback). Both regressions surfaced only via Gate 6 (real-server
    # live repro); none of the unit/integration suites caught them
    # because every test mocked the engine with a MagicMock that
    # auto-satisfies any attribute access.
    # ------------------------------------------------------------------

    @abstractmethod
    def build_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        """Render the chat prompt for ``messages`` + ``tools`` without
        starting generation.

        Called by ``routes/chat.py`` for cloud-routing token estimation
        and for eager streaming chat-template validation (so template
        errors surface as HTTP 400 instead of mid-stream failures).
        """

    @abstractmethod
    def estimate_new_tokens(self, prompt: str) -> tuple[int, int]:
        """Return ``(total_tokens, new_tokens)`` for ``prompt``.

        Called by ``routes/chat.py`` cloud routing to decide whether the
        request crosses ``--cloud-threshold`` and should be offloaded.
        ``new_tokens`` is the count that would need fresh prefill — i.e.
        total minus the prefix already warm in cache. A conservative
        ``(total, total)`` is acceptable; correctness only requires that
        the threshold semantics hold.
        """

    @property
    def supports_guided_generation(self) -> bool:
        """Whether the engine can constrain output to a JSON schema.

        Default ``False``; override to return ``True`` only when
        ``generate_with_schema`` is also implemented (the route checks
        this flag before calling). Allows engines without ``outlines`` /
        guided decoding to participate in the contract without
        implementing the optional schema path.
        """
        return False

    async def generate_with_schema(
        self,
        messages: list[dict[str, Any]],
        json_schema: dict[str, Any],
        **kwargs,
    ) -> "GenerationOutput":
        """Generate output constrained to ``json_schema``.

        Default raises ``NotImplementedError``. The route only calls this
        when ``supports_guided_generation`` is ``True``, so engines that
        leave that flag at the default ``False`` need not override.
        """
        raise NotImplementedError(
            "generate_with_schema is not implemented for this engine. "
            "Override supports_guided_generation to advertise capability."
        )
