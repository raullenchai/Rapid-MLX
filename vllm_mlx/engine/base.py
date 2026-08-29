# SPDX-License-Identifier: Apache-2.0
"""
Base engine interface for rapid-mlx inference.
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
    # H-03: when a user-supplied ``stop`` string fired (vs an EOS token
    # or ``max_tokens`` cap), the scheduler/engine records the matched
    # string here so route adapters can surface the precise reason.
    # The Anthropic ``/v1/messages`` adapter maps this onto
    # ``stop_reason="stop_sequence"`` + ``stop_sequence: <str>`` per the
    # public spec; OpenAI ``/v1/completions`` and ``/v1/chat/
    # completions`` keep ``finish_reason="stop"`` (a single bucket for
    # both EOS and stop-string per OpenAI's wire spec) so the field is
    # harmless to ignore on the OpenAI surface. ``None`` means "no
    # user stop matched" — ``finish_reason`` was set by EOS, the
    # length cap, or never fired. Appended LAST per the field-order
    # note above to preserve positional constructor arg indices for
    # pre-existing fields.
    matched_stop: str | None = None


def _callable_accepts_kwarg(func: Any, name: str, inspect_mod: Any) -> bool:
    """True iff ``func`` can be called with keyword ``name``.

    #1100 codex round 10 (#5): used to pick the call shape for a possibly-legacy
    override BEFORE invoking it, so a ``TypeError`` from inside the method body
    is never mistaken for a signature mismatch (which would re-invoke and
    duplicate partial mutations). Accepts the kwarg when it is a named parameter
    OR the signature has ``**kwargs``. Falls back to ``True`` (assume the modern
    signature) if the signature can't be introspected — a genuine legacy
    one-arg override still raises ``TypeError`` in that rare case, but no
    double-invocation path exists to mask it.
    """
    try:
        sig = inspect_mod.signature(func)
    except (ValueError, TypeError):  # pragma: no cover — builtins / C funcs
        return True
    params = sig.parameters
    if name in params:
        return True
    return any(p.kind == inspect_mod.Parameter.VAR_KEYWORD for p in params.values())


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

    @property
    def supports_completion_logprobs(self) -> bool:
        """Whether legacy completions can extract per-token logprobs.

        The `/v1/completions` logprobs path consumes streaming
        generation chunks plus the tokenizer to map token ids back to
        strings. Expose that as an engine capability so routes do not
        probe for optional methods with `hasattr(engine, ...)`.
        """
        stream_generate = getattr(self, "stream_generate", None)
        return getattr(self, "tokenizer", None) is not None and callable(
            stream_generate
        )

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

    def save_cache_with_outcome(self, cache_dir: str, should_abort=None):
        """Save the prefix cache and return a ``SaveOutcome`` (#1100 codex
        round 4 #2).

        Declared here (not behind a ``hasattr`` guard in the route — the #500
        silent-skip shape) so the cache route can call it directly. Real
        engines override to compute the outcome IN the step-thread task
        alongside the save (closing the cross-path race where a cache-global
        outcome field is clobbered between op and read).

        #1100 codex round 7 (#2): the default must NOT map every ``False`` from
        ``save_cache_to_disk`` to ``"empty"`` — that method also returns
        ``False`` for a NON-empty cache that failed to commit any entry, so a
        subclass overriding only ``save_cache_to_disk`` (not this method) would
        report a FAILED export as a successful empty snapshot (the export route
        then publishes an empty manifest instead of 500ing). Disambiguate via
        authoritative cache state: ``True`` → ``"committed"``; ``False`` with a
        cache that still reports entries → ``"failed"``; ``False`` with an empty
        / absent cache → ``"empty"``.
        """
        from ..cache.protocol import SaveOutcome

        saved = self.save_cache_to_disk(cache_dir, should_abort=should_abort)
        if saved:
            return SaveOutcome(outcome="committed")
        # Not committed — distinguish a genuine empty no-op from a failed save
        # by asking the cache how many entries it holds.
        #
        # #1100 codex round 7 (#2) → round 10 (#4): FAIL CLOSED when the entry
        # count is UNAVAILABLE. ``get_cache_stats`` returns ``None`` on the
        # BaseEngine default (and can be malformed on an odd subclass); the
        # round-7 version treated that as ``entry_count = 0`` → ``"empty"``,
        # which reports a FAILED non-empty save as a successful empty export (the
        # route then publishes an empty manifest instead of 500ing). ``"empty"``
        # must be reserved for an EXPLICIT authoritative zero; anything we can't
        # read authoritatively is ``"failed"`` — the safe direction (a false
        # "failed" only 500s a genuinely-empty export; a false "empty" ships a
        # lie).
        try:
            stats = self.get_cache_stats()
        except Exception:  # pragma: no cover — defensive against odd stats shapes
            stats = None
        if not isinstance(stats, dict) or "entry_count" not in stats:
            # No authoritative count → cannot prove emptiness → fail closed.
            return SaveOutcome(outcome="failed")
        try:
            entry_count = int(stats.get("entry_count") or 0)
        except (TypeError, ValueError):  # non-numeric entry_count → unauthoritative
            return SaveOutcome(outcome="failed")
        return SaveOutcome(outcome="failed" if entry_count > 0 else "empty")

    def load_cache_with_result(
        self, cache_dir: str, replace: bool = False, protected_import: bool = True
    ):
        """Load the prefix cache and return a ``LoadResult`` (#1100 codex
        round 4 #2).

        Same rationale as ``save_cache_with_outcome``. The default delegates
        to ``load_cache_from_disk`` and reports 0 loaded bytes (the count is
        authoritative; bytes are best-effort for engines that don't override).

        #1100 codex round 9 (#4) → round 10 (#5): ``replace`` was added to
        ``load_cache_from_disk`` for #476. A pre-existing engine overriding only
        the OLD one-arg ``load_cache_from_disk(self, cache_dir)`` would
        ``TypeError`` on the ``replace=`` keyword. Round 9 caught that TypeError
        and retried — but a ``TypeError`` raised from INSIDE the method body
        (not a signature mismatch) would then be re-invoked, DUPLICATING partial
        mutations and masking the real fault. Round 10: decide signature
        compatibility by INTROSPECTION (``inspect.signature``) BEFORE the call,
        so we never re-invoke on a body error. If the callee accepts ``replace``
        (or ``**kwargs``), pass it; otherwise call one-arg and — if the caller
        asked for a replace the callee can't honor — fail loudly rather than
        silently merge.
        """
        import inspect

        from ..cache.protocol import LoadResult

        # #1111 codex r4: forward each optional kwarg INDEPENDENTLY — never gate
        # one on another's acceptance. ``replace`` (default False) and
        # ``protected_import`` (default True) were both added over time; a legacy
        # override may accept neither, one, or both. For EACH kwarg:
        #  * callee accepts it            → forward the caller's value.
        #  * callee lacks it, caller left it at DEFAULT → drop silently (the
        #    callee's own default matches the contract, nothing is lost).
        #  * callee lacks it, caller passed a NON-DEFAULT → fail loudly, because
        #    silently dropping it would degrade behavior the caller explicitly
        #    requested (a replace silently downgraded to merge leaves stale
        #    entries; a protected_import=False silently upgraded to protected
        #    re-opens the restart-cycle growth bug).
        kwargs: dict[str, object] = {}
        for name, value, default in (
            ("replace", replace, False),
            ("protected_import", protected_import, True),
        ):
            if _callable_accepts_kwarg(self.load_cache_from_disk, name, inspect):
                kwargs[name] = value
            elif value != default:
                raise TypeError(
                    f"{type(self).__name__}.load_cache_from_disk does not "
                    f"support {name}={value!r} (legacy signature); cannot honor "
                    "the caller's non-default request"
                )
        entries = self.load_cache_from_disk(cache_dir, **kwargs)
        return LoadResult(entries=entries, bytes_loaded=0)

    def save_cache_to_disk(self, cache_dir: str, should_abort=None) -> bool:
        """Persist the prefix cache. Override in subclasses that have one."""
        return False

    def load_cache_from_disk(
        self, cache_dir: str, replace: bool = False, protected_import: bool = True
    ) -> int:
        """Hydrate the prefix cache. Override in subclasses that have one."""
        return 0

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
    # Declaring route-facing methods here makes missing implementations fail
    # at instantiation instead of silently degrading behind ``hasattr`` guards.
    # ------------------------------------------------------------------

    @abstractmethod
    def build_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        enable_thinking: bool | None = None,
        add_generation_prompt: bool = True,
        chat_template_kwargs: dict | None = None,
    ) -> str:
        """Render the chat prompt for ``messages`` + ``tools`` without
        starting generation.

        ``add_generation_prompt`` (default True) toggles the assistant
        generation prefix so the reasoning-budget seed probe can diff two
        renders to isolate the template-added prefix.

        Called by ``routes/chat.py`` for eager streaming chat-template
        validation so template errors surface as HTTP 400 instead of
        mid-stream failures.
        """

    @property
    def supports_guided_generation(self) -> bool:
        """Whether the engine can constrain output to a JSON schema.

        Default ``False``; override to return ``True`` only when
        ``generate_with_schema`` is also implemented (the route checks
        this flag before calling). Allows engines without ``llguidance`` /
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
