"""Diffusion lane — discrete text-diffusion inference engine.

Wraps mlx-vlm 0.6.3's ``stream_diffusion_generate`` so DiffusionGemma
(and any future block-diffusion text model in the same family) can ride
the same ``BaseEngine`` contract as ``BatchedEngine`` does for AR LLMs.

Why a separate engine — not a path inside ``BatchedEngine``
-----------------------------------------------------------
DiffusionGemma denoises a fixed-size canvas (default 256 tokens) for K
steps and emits the whole block at once, then slides the window. This
is incompatible with the auto-regressive scheduler in three ways:

  * No per-token logits stream — emission is block-granular.
  * No KV cache mutation per token — the canvas is overwritten in
    place.
  * Spec-decode + DFlash are silently meaningless (no draft tokens to
    verify when the whole block lands at once).

So we route at the ``modality`` boundary in ``server.load_model``: a
``modality="text-diffusion"`` alias instantiates ``DiffusionEngine``
here instead of ``BatchedEngine``. Everything downstream of the
``_engine`` slot in ``server.py`` is blind to the difference because
``DiffusionEngine`` implements the same ``BaseEngine`` interface.

Dependency
----------
mlx-vlm >= 0.6.3, which contains Blaizzy/mlx-vlm#1347 (Gemma 4 DLM
model files) and #1348 (long-context prefill fix). Verified locally
2026-06-10. The pyproject pin floor is bumped to ``>=0.6.3`` so a
fresh ``pip install rapid-mlx`` lands a build that has the
``mlx_vlm.models.diffusion_gemma`` package on disk.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ..engine.base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)


# Bumped when the wire format we expose to ``routes/chat.py`` changes
# (e.g. block-vs-token deltas, finish_reason semantics).
DIFFUSION_LANE_VERSION = "0.1-wired"

# Sentinel pushed onto the streaming queue to signal end of generation.
# Plain string to keep the queue homogeneous-ish; the consumer checks
# ``is`` identity, so the value is irrelevant.
_STREAM_DONE = object()


def _normalize_stops(value: Any) -> list[str]:
    """Accept the OpenAI ``stop`` shape: ``None``, a string, or a list
    of strings. Return a non-empty-string list; empty input → ``[]``.

    Empty strings would match everywhere — silently dropped.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [s for s in value if isinstance(s, str) and s]
    return []


def _earliest_stop_index(text: str, stops: list[str]) -> int:
    """Return the earliest index at which ANY stop sequence begins in
    ``text``, or -1 if none match. O(len(text) * len(stops)) which is
    fine for the small input shapes we see (chunk lengths < 4 KB,
    stop lists <= 4 entries).
    """
    best = -1
    for s in stops:
        idx = text.find(s)
        if idx != -1 and (best == -1 or idx < best):
            best = idx
    return best


@dataclass(frozen=True)
class DiffusionGenerationConfig:
    """Sampling / decoding knobs for the diffusion lane.

    Holds the subset of mlx-vlm's diffusion-generator parameters that
    the engine forwards to ``stream_diffusion_generate``. The route
    layer translates the OpenAI schema → this dataclass at the
    dispatch boundary so the engine never sees ``ChatCompletionRequest``
    directly.

    API surface (v0): ``temperature`` is the only knob threaded from
    /v1/* requests today. ``diffusion_steps``, ``diffusion_sampler``,
    and ``prefill_step_size`` are NOT declared on the OpenAI
    request models so they cannot be overridden per-request via
    /v1/chat/completions or /v1/completions — Pydantic silently drops
    extra fields. mlx-vlm's own defaults (entropy-bound sampler,
    48 denoise steps for DiffusionGemma) are used instead. The
    kwargs are still honoured for direct programmatic callers
    (``engine.stream_chat(..., diffusion_steps=24)``) and for the
    operator-tuned ``prefill_step_size`` which flows from
    SchedulerConfig at engine construction. A future PR can declare
    them on the request models if user-facing tuning is needed
    (codex round 10 [P2]).
    """

    # Per-block denoising steps. ``None`` → use the model's own
    # generation_config default (mlx-vlm: 48 for DiffusionGemma).
    diffusion_steps: int | None = None
    # Temperature applied at the per-token argmax inside the denoiser.
    # 0.0 = greedy; matches the AR lane's convention.
    temperature: float = 0.0
    # Sampler family. Currently mlx-vlm 0.6.3 only ships
    # ``entropy-bound``; ``confidence-threshold`` exists in code but
    # the only canvas_length-driven config DiffusionGemma uses points
    # at the entropy-bound sampler. Surface the knob anyway so callers
    # can switch when mlx-vlm extends it.
    diffusion_sampler: str = "entropy-bound"
    # Long-context chunked-prefill size (mlx-vlm
    # ``prefill_step_size``). ``None`` → run the prefill as one
    # monolithic forward pass (mlx-vlm default). Set on long-context
    # workloads to bound peak Metal allocation per step — without it
    # the diffusion lane OOMs on 30k+ prompts (codex round 5 [P2]).
    prefill_step_size: int | None = None


class DiffusionEngine(BaseEngine):
    """``BaseEngine`` adapter over mlx-vlm's diffusion-text generator.

    Single-batch only — mlx-vlm's diffusion code path raises on
    ``input_ids.shape[0] > 1``. The route layer rejects ``n > 1`` and
    structured-output flags before the engine sees them.

    Threading model: ``stream_diffusion_generate`` is a synchronous
    generator that occupies the GPU for a non-trivial slice (block of
    256 tokens × K denoising steps). We push it onto a worker thread
    and drain into an ``asyncio.Queue`` so the event loop stays
    responsive. One thread per active request — DiffusionGemma is
    batch-1 only, so concurrent requests sit in admission queue at
    the route layer, not here.
    """

    # Engine-level capability bit consulted by ``routes/chat.py``'s
    # ``_engine_supports_channel_routed_tool_calls`` probe. The
    # DiffusionGemma generator emits a free-form denoised canvas with
    # no tool-call channel — even though its tokenizer would trip the
    # Gemma 4 channel-routed allowlist, the actual engine path never
    # runs OutputRouter and always emits ``channel="content"``. Without
    # this bit, ``tool_choice="required"`` would silently slip past
    # the streaming-required gate and finish with plain text instead
    # of 422'ing upfront (codex round 9 [P2]).
    supports_tool_calls: bool = False

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 4096,
        scheduler_config: Any = None,
    ) -> None:
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._scheduler_config = scheduler_config
        self._model: Any = None
        self._processor: Any = None
        self._loaded = False
        self._load_error: BaseException | None = None
        # Admission control mirrors BatchedEngine.check_admission —
        # reservations counter under a lock, BackpressureError raised
        # when the configured ``max_concurrent_requests`` is reached.
        # The diffusion lane is batch-1 only at the GPU layer, but we
        # still let operators tune the cap because queued requests
        # waiting on ``_generation_lock`` are valid load to admit (the
        # asyncio lock serializes cooperatively without burning the
        # event loop). codex round 2 [P2]: routes/chat.py's
        # ``_check_admission_or_503`` was silently no-op'ing for this
        # lane because the methods did not exist; concurrent local
        # requests piled up behind the generation lock instead of
        # returning the documented 503/Retry-After at the cap.
        self._admission_lock = threading.Lock()
        self._admission_reservations = 0
        # ``_worker_stuck`` is flipped when ``_stream_prompt_raw``'s
        # ``done_event.wait`` ceiling fires — i.e. the worker did not
        # observe cancel within the 30 s drain window. Once set, every
        # subsequent ``check_admission`` raises so the operator's
        # health-check + restart catches the wedge instead of routing
        # new requests onto an engine whose GPU is still consuming the
        # abandoned job (codex round 7 [P2]).
        self._worker_stuck: bool = False
        # Per-engine lock — DiffusionGemma is single-batch only, so we
        # serialize the generator at the engine level rather than rely
        # on the route admission queue. Two concurrent
        # ``stream_diffusion_generate`` calls against the same model
        # corrupt each other's canvas state.
        #
        # ``asyncio.Lock`` (NOT ``threading.Lock``): ``stream_chat`` is
        # an async generator that ``await``s between block-complete
        # chunks. A ``threading.Lock`` acquired on the event-loop
        # thread by a second concurrent request would block that
        # thread until the first request released — but the first
        # request can't release because the event loop it needs to
        # advance is the very thread now blocked on ``acquire()``.
        # That is a textbook async deadlock. The asyncio lock yields
        # the loop instead, so the first request can finish draining
        # its queue and release.
        self._generation_lock = asyncio.Lock()

        # Persistent GPU worker thread: owns model loading AND every
        # GPU op. Required because mlx's per-stream binding is
        # thread-local — weights loaded on thread A can't be
        # mx.eval'd from thread B without crashing with
        # ``RuntimeError: There is no Stream(gpu, 0) in current
        # thread``. mlx-vlm's own server uses the same pattern (one
        # ``ResponseGenerator._thread`` that loads + runs in one
        # place; see mlx_vlm/server/generation.py:894).
        #
        # Job protocol: callers push a ``(prompt, max_tokens, cfg,
        # out_queue)`` tuple onto ``_jobs``; the worker either
        # streams ``GenerationOutput`` chunks back via ``out_queue``
        # (terminated by ``_STREAM_DONE``) or pushes the exception
        # on failure. Push ``None`` to request shutdown.
        self._jobs: queue.Queue[Any] = queue.Queue()
        self._ready = threading.Event()
        self._stop = False
        # codex round 11 [P2]: the worker is NOT started in __init__
        # any more. Plain construction must not kick off an
        # mlx-vlm load — that breaks contract tests that instantiate
        # the engine without ever calling start(), and would crash
        # under CI environments without usable Metal. The worker is
        # started on the first call to start() / _load_blocking() /
        # check_admission, gated by _start_worker_once.
        self._worker: threading.Thread | None = None
        self._worker_start_lock = threading.Lock()
        # The worker installs this each time it pulls a job from
        # the queue and clears it in the matching ``finally``. ``stop()``
        # reads it to signal cancellation on an in-flight job — without
        # this, ``stop()`` pushes a sentinel that sits behind the active
        # generator until ``max_tokens`` finishes (codex pr_validate r5
        # BLOCKING). Plain attribute (not a property): the worker thread
        # and the asyncio thread both touch it under the discipline
        # ``worker writes None→event→None``, ``stop`` reads-and-uses; a
        # racy concurrent read returning a stale event is harmless (we
        # just signal an already-finished cancel_event).
        self._active_cancel: threading.Event | None = None

    # ------------------------------------------------------------------
    # BaseEngine — required properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_mllm(self) -> bool:
        # DiffusionGemma technically inherits the Gemma 4 multimodal
        # processor, but v0 of this engine routes text-only — vision
        # inputs raise at chat() time. Reporting ``False`` keeps the
        # route's text-only paths (cloud routing, build_prompt) wired.
        return False

    @property
    def tokenizer(self) -> Any:
        self._ensure_loaded()
        return self._processor.tokenizer

    # ------------------------------------------------------------------
    # BaseEngine — lifecycle
    # ------------------------------------------------------------------

    def _start_worker_once(self) -> None:
        """Spin up the worker thread on demand. Idempotent under
        concurrent callers (the lock guards the start invariant)."""
        with self._worker_start_lock:
            if self._worker is not None:
                return
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="rapid-mlx-diffusion-worker",
                daemon=True,
            )
            self._worker.start()

    async def start(self) -> None:
        # Block the asyncio loop while the worker initialises the
        # model on its own thread. Load takes ~2-3 s on M3 Ultra, well
        # under the lifespan startup budget.
        self._start_worker_once()
        await asyncio.to_thread(self._wait_until_ready)

    async def stop(self) -> None:
        self._stop = True
        # codex pr_validate r5 BLOCKING: a sentinel alone does NOT
        # unblock an in-flight ``_run_generator`` — it sits behind
        # the active job in the queue. Signal the live cancel event
        # FIRST so the worker breaks out of mlx-vlm's
        # ``stream_diffusion_generate`` at the next per-chunk
        # cancel-check, then push the sentinel for the parked-on-
        # queue.get() case. The handle is published by the worker's
        # job-pull block; a None read here just means no job is
        # active right now (also fine).
        active = self._active_cancel
        if active is not None:
            active.set()
        self._jobs.put(None)
        # mlx-vlm loads weights into mx.array buffers backed by the
        # MTL allocator. Clearing references is enough for the next
        # serve cycle to repopulate — there is no explicit ``unload``
        # in mlx-vlm 0.6.3. We MUST wait for the worker to actually
        # exit before nulling ``_model`` / ``_processor`` — clearing
        # while the worker is still inside an mx.eval can crash the
        # GPU op mid-iteration (codex pr_validate r5 BLOCKING). The
        # 30s ceiling matches ``_stream_prompt_raw``'s drain budget
        # so a wedged worker doesn't block lifespan shutdown forever;
        # if it expires, leave the model refs intact so GC reclaims
        # them after the (orphaned) worker eventually returns.
        if self._worker is not None:
            await asyncio.to_thread(self._worker.join, 30.0)
            if self._worker.is_alive():
                logger.warning(
                    "DiffusionEngine.stop(): worker did not exit "
                    "within 30s; leaving model refs to GC after "
                    "worker drain to avoid clearing under live "
                    "mx.eval."
                )
                return
        self._model = None
        self._processor = None
        self._loaded = False

    # ------------------------------------------------------------------
    # BaseEngine — admission control
    # ------------------------------------------------------------------

    def check_admission(self) -> None:
        """Atomic admission gate; reserves a slot on success or raises
        ``BackpressureError`` at the cap. Mirrors
        ``BatchedEngine.check_admission`` so ``routes/chat.py``'s
        ``_check_admission_or_503`` returns a clean 503 + Retry-After
        for the diffusion lane instead of silently no-op'ing (codex
        round 2 [P2]).

        Cap source: the ``SchedulerConfig.max_concurrent_requests``
        passed at engine construction (server.load_model wires it
        through). When no config is provided (test stubs,
        programmatic callers), the dataclass default applies.
        """
        from ..scheduler import BackpressureError, SchedulerConfig

        # Stuck-worker short-circuit — once the drain ceiling has been
        # hit, refuse new work until the engine is restarted. Routing
        # requests onto a wedged engine would let them queue behind
        # GPU work that the lock was meant to exclude (codex round 7
        # [P2]).
        if self._worker_stuck:
            raise BackpressureError(
                "DiffusionEngine worker did not drain a cancelled job "
                "within the 30 s ceiling — engine marked unhealthy. "
                "Restart the server to recover."
            )
        sc = self._scheduler_config
        if sc is None:
            sc = SchedulerConfig()
        cap = getattr(sc, "max_concurrent_requests", None)
        if cap is None or cap <= 0:
            return
        with self._admission_lock:
            if self._admission_reservations >= cap:
                raise BackpressureError(
                    f"max_concurrent_requests={cap} reached "
                    f"(currently {self._admission_reservations} in-flight)"
                )
            self._admission_reservations += 1

    def release_admission_reservation(self) -> None:
        """Release a slot reserved by ``check_admission``. Idempotent
        below zero so a stray double release does not corrupt the
        cap accounting (matches BatchedEngine behavior)."""
        with self._admission_lock:
            if self._admission_reservations > 0:
                self._admission_reservations -= 1

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            if self._load_error is not None:
                raise self._load_error
            raise RuntimeError("DiffusionEngine not loaded — call start() first")

    def _wait_until_ready(self, timeout: float | None = None) -> None:
        """Block until the worker thread reports model-load done.

        Surfaces the load exception if one occurred so the calling
        ``await start()`` raises with the original cause.
        """
        if not self._ready.wait(timeout):
            raise RuntimeError("Timed out waiting for DiffusionEngine model load")
        if self._load_error is not None:
            raise self._load_error

    def _load_blocking(self) -> None:
        """Public-named helper retained from the skeleton PR. Callers
        in ``server.py`` invoke this synchronously at startup; the
        persistent worker is started here (the constructor no longer
        does that — codex round 11 [P2]) and we then wait for its
        ready signal."""
        self._start_worker_once()
        self._wait_until_ready()

    def _worker_loop(self) -> None:
        """GPU worker — owns model load AND every diffusion call.

        Step 1: load model. Once loaded, set ``_ready`` so
        ``_wait_until_ready`` returns.
        Step 2: pump jobs until ``_stop`` flips or sentinel arrives.

        Every failure path BEFORE ``self._ready.set()`` MUST surface
        through ``self._load_error`` and then ``self._ready.set()`` —
        otherwise ``_load_blocking()`` / ``start()`` waits forever
        (codex round 5 [P2]). The original code only caught
        ``ImportError`` on the upstream module imports; a Metal
        runtime error during ``import mlx.core`` or a non-Import
        exception from ``mlx_vlm.generate.diffusion`` would silently
        kill the worker before any startup signal.
        """
        try:
            import mlx.core as mx
            from mlx_vlm.generate.diffusion import (
                diffusion_generation_family,
            )
            from mlx_vlm.utils import load
        except BaseException as e:  # noqa: BLE001 — propagate to caller
            self._load_error = RuntimeError(
                "DiffusionEngine failed to import its mlx / mlx-vlm "
                "dependencies. Install or upgrade: "
                "`pip install -U 'mlx-vlm>=0.6.3'`. "
                f"Underlying error: {e}"
            )
            self._ready.set()
            return

        try:
            logger.info(f"Loading DiffusionEngine model: {self._model_name}")
            model, processor = load(self._model_name)
            family = diffusion_generation_family(model)
            if family != "block":
                raise RuntimeError(
                    f"{self._model_name!r} is not a block-diffusion model "
                    f"(diffusion_generation_family returned {family!r}). "
                    "DiffusionEngine only supports DiffusionGemma-family "
                    "block-canvas checkpoints."
                )
            self._model = model
            self._processor = processor
            self._loaded = True
        except BaseException as e:  # noqa: BLE001 — propagate to caller
            self._load_error = e
            self._ready.set()
            return

        # Pre-bind the GPU stream on THIS thread so the diffusion
        # generator's internal ``mx.eval`` calls have a valid default
        # to dispatch to. Once set here, it persists for the lifetime
        # of the worker — every job below inherits the same binding.
        # Any failure here also has to flip ``_ready`` so the lifespan
        # startup doesn't deadlock on a partially-loaded worker.
        try:
            worker_stream = mx.default_stream(mx.default_device())
            mx.set_default_stream(worker_stream)
        except BaseException as e:  # noqa: BLE001 — propagate to caller
            self._load_error = e
            self._ready.set()
            return
        self._ready.set()

        # Job loop.
        while not self._stop:
            job = self._jobs.get()
            if job is None:
                return
            prompt, max_tokens, cfg, out_q, cancel_event, done_event = job
            # Publish the in-flight cancel handle BEFORE we start
            # consuming GPU so ``stop()`` (called from the lifespan
            # shutdown coroutine) can signal cancellation immediately
            # instead of pushing a sentinel that sits behind a
            # multi-block diffusion run (codex pr_validate r5 BLOCKING).
            self._active_cancel = cancel_event
            try:
                # Fast-skip jobs that were cancelled BEFORE we picked
                # them up. Without this, a request whose coroutine was
                # cancelled while still queued behind a slower job
                # would still cost a full prefill + first-block of GPU
                # the moment we got around to it (codex round 6 [P2]).
                if cancel_event.is_set():
                    continue
                self._run_generator(prompt, max_tokens, cfg, out_q, cancel_event)
            except BaseException as e:  # noqa: BLE001 — surface to caller
                out_q.put(e)
            finally:
                self._active_cancel = None
                out_q.put(_STREAM_DONE)
                # ``done_event`` lets the request-side coroutine know
                # the worker has fully released this job's resources
                # so it can release the engine-level generation lock.
                # codex round 6 [P2]: without this, releasing the lock
                # on the consumer's exit (before the worker observed
                # cancel_event mid-block) head-of-line-blocked the
                # next queued request behind abandoned GPU work.
                done_event.set()

    # ------------------------------------------------------------------
    # BaseEngine — prompt / token helpers used by the route layer
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        self._ensure_loaded()
        if tools:
            # DiffusionGemma's generator emits a free-form denoised
            # canvas — no function-call grammar, no tool-name decoding
            # path. Early versions raised here, but Big-AGI (and other
            # OpenAI-compatible frontends) attach their built-in tools
            # to every chat request even when the user didn't intend a
            # tool invocation, which turned the very first message into
            # an opaque 500. The OpenAI contract treats a model that
            # never emits tool calls as still serviceable for plain
            # chat, so we follow that shape: drop the tools list and
            # log a warning so the operator can see it happen.
            #
            # ``tool_choice="required"`` is a stricter contract — the
            # caller is asserting "you MUST emit a tool call" and
            # cannot be satisfied. routes/chat.py post-parses the
            # generated text and 422s when ``required`` was set and no
            # tool_calls came back, so the contract violation surfaces
            # to the client without us needing to special-case it
            # here.
            logger.warning(
                "DiffusionEngine dropped %d tool(s) — model has no "
                "function-call grammar; chat continues without them.",
                len(tools),
            )
        # ``apply_chat_template`` returns either a string or a list of
        # token IDs depending on tokenize=. We want the rendered text
        # for build_prompt's contract; the tokenization happens inside
        # stream_chat right before we hand off to mlx-vlm.
        return self._processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def estimate_new_tokens(self, prompt: str) -> tuple[int, int]:
        self._ensure_loaded()
        ids = self._processor.tokenizer.encode(prompt)
        n = len(ids)
        return (n, n)

    # ------------------------------------------------------------------
    # BaseEngine — chat / generate
    # ------------------------------------------------------------------

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
        # Buffer the stream into one output. The diffusion lane has no
        # cheaper non-stream path inside mlx-vlm — the same generator
        # underlies both surfaces.
        text_parts: list[str] = []
        last: GenerationOutput | None = None
        async for chunk in self.stream_chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            images=images,
            videos=videos,
            **kwargs,
        ):
            text_parts.append(chunk.new_text)
            last = chunk
        if last is None:
            return GenerationOutput(text="", finish_reason="stop")
        return GenerationOutput(
            text="".join(text_parts),
            tokens=last.tokens,
            prompt_tokens=last.prompt_tokens,
            completion_tokens=last.completion_tokens,
            finish_reason=last.finish_reason or "stop",
            finished=True,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,  # noqa: ARG002 — diffusion lane ignores it
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        self._ensure_loaded()
        # ``tools`` is silently dropped in ``build_prompt`` with a
        # warning log — see the matching block there for the rationale.
        # We forward ``tools`` so the warning actually fires (codex
        # pr_validate r5 NIT — previously ``build_prompt(messages)``
        # passed an empty tools arg, so direct engine callers got
        # neither the drop nor the visible warning).
        if images or videos:
            raise RuntimeError(
                "DiffusionEngine v0 is text-only. Vision inputs "
                "(images/videos) will be wired in a follow-up; for now "
                "drop them from the request."
            )
        prompt = self.build_prompt(messages, tools=tools)
        async for chunk in self._stream_prompt_raw(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        ):
            yield chunk

    async def _stream_prompt_raw(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Shared queue / cancel / stop-sequence plumbing for chat and
        completions. Caller is responsible for any chat-template wrap
        BEFORE invoking this helper — ``prompt`` is fed verbatim to
        mlx-vlm's tokenizer (codex round 5 [P2]).
        """
        # Pull the operator-configured prefill chunk size from the
        # scheduler config so long-context requests honor it. mlx-vlm
        # only enables its chunked-prefill path when the kwarg is
        # non-None.
        _sc = self._scheduler_config
        _prefill_step_size = getattr(_sc, "prefill_step_size", None) if _sc else None
        cfg = DiffusionGenerationConfig(
            diffusion_steps=kwargs.get("diffusion_steps"),
            temperature=temperature,
            diffusion_sampler=kwargs.get("diffusion_sampler", "entropy-bound"),
            prefill_step_size=_prefill_step_size,
        )
        loop = asyncio.get_running_loop()
        # Cancellation handle — set by the stream_chat finally clause
        # so the persistent worker stops reading mlx-vlm's generator
        # when the caller disconnects or we truncate on an early stop.
        # Without this, the worker keeps generating up to ``max_tokens``
        # AFTER stream_chat has returned, monopolizing the single GPU
        # worker thread until the next queued request can land (codex
        # round 3 [P2]).
        cancel_event = threading.Event()
        # ``done_event`` is paired with ``cancel_event`` for the
        # life of a single job: the worker thread sets it after
        # ``_run_generator`` returns (or is fast-skipped). The
        # request-side finally awaits it before releasing the engine
        # lock, so a queued sibling request can't acquire the lock
        # while the worker is still burning GPU on this job (codex
        # round 6 [P2]).
        done_event = threading.Event()
        # The engine-level generation lock serializes concurrent
        # requests (DiffusionGemma is batch-1 only). codex round 4
        # [P2]: per-request resources (queues + pump thread) are now
        # set up INSIDE the lock — if a queued request gets cancelled
        # while waiting on the lock, no pump thread was ever started,
        # so the cancelled coroutine cannot leak a daemon thread
        # blocked on ``thread_q.get()`` forever.
        async with self._generation_lock:
            # Post-lock unhealthy gate — a request that passed
            # admission BEFORE a sibling tripped the drain timeout
            # would otherwise queue work to a wedged worker once it
            # acquired the lock. Re-check here so it errors out
            # cleanly instead (codex round 8 [P2]).
            #
            # NOTE on streaming-503 contract (codex round 9 [P2]):
            # The route's ``stream_chat_completion`` yields the
            # ``role`` SSE chunk BEFORE entering ``engine.stream_chat``.
            # So for the in-flight race (request waiting on the lock
            # when stuck flips), this raise lands inside the streaming
            # generator and the route surfaces it as an SSE error on
            # HTTP 200 — not a clean 503. The PRIMARY contract this
            # gate enforces is "do not enqueue work to a wedged
            # worker"; the streaming-protocol equivalent of the 503
            # is the SSE error chunk, and clients should treat it as
            # equivalent. ``check_admission`` continues to deliver the
            # clean 503 for the NORMAL admission path.
            if self._worker_stuck:
                from ..scheduler import BackpressureError

                raise BackpressureError(
                    "DiffusionEngine worker is unhealthy — restart the "
                    "server to recover."
                )
            # Two queues bridge the persistent worker thread to the
            # asyncio loop. ``thread_q`` is owned by the worker (sync
            # ``queue.Queue.put`` is safe from any thread); ``aio_q``
            # is owned by the loop; the pump thread relays items via
            # ``call_soon_threadsafe``.
            thread_q: queue.Queue[Any] = queue.Queue()
            aio_q: asyncio.Queue[Any] = asyncio.Queue()

            def pump() -> None:
                while True:
                    item = thread_q.get()
                    loop.call_soon_threadsafe(aio_q.put_nowait, item)
                    if item is _STREAM_DONE:
                        return

            pump_thread = threading.Thread(
                target=pump,
                name="rapid-mlx-diffusion-pump",
                daemon=True,
            )
            pump_thread.start()
            self._jobs.put(
                (prompt, max_tokens, cfg, thread_q, cancel_event, done_event)
            )
            # Caller-supplied stop sequences (OpenAI /v1/completions
            # ``stop`` knob — single string or list). mlx-vlm's
            # ``stream_diffusion_generate`` does not honor stop
            # strings natively, so we post-process the block-emitted
            # text. The hold-back contract is:
            #
            #   * Keep ``tail_len = max(stop_len) - 1`` characters
            #     buffered. Anything in the buffer might still grow
            #     into a stop match on the next chunk, so it cannot
            #     yet be safely emitted to the client.
            #   * Single-character stops degenerate ``tail_len`` to
            #     0; in that case no lookback is needed (the match
            #     lands fully inside the current chunk every time),
            #     so we skip buffering and emit the chunk live. codex
            #     round 3 [P2]: without this special-case, common
            #     stops like ``"\n"`` or ``"}"`` buffered every block
            #     until the terminal chunk arrived, dragging streaming
            #     TTFT off a cliff.
            #   * On a stop match, yield ``combined[:cut]`` with
            #     finish_reason="stop" and stop reading further.
            #   * On a terminal chunk (finish_reason from the
            #     generator) with no stop match, flush the buffer.
            #
            # codex round 2 [P2]: the previous version updated ``tail``
            # but still ``yield``ed the full chunk, leaking the leading
            # bytes of a boundary-straddling stop sequence to the
            # client. The hold-back is the only correct fix.
            stop_list = _normalize_stops(kwargs.get("stop"))
            tail_len = (max(len(s) for s in stop_list) - 1) if stop_list else 0
            tail = ""
            try:
                while True:
                    item = await aio_q.get()
                    if item is _STREAM_DONE:
                        return
                    if isinstance(item, BaseException):
                        raise item
                    # No stop list → fast path, unchanged.
                    if not stop_list:
                        yield item
                        continue
                    combined = tail + item.new_text
                    cut = _earliest_stop_index(combined, stop_list)
                    if cut >= 0:
                        # Stop match. Truncate the buffer + this chunk
                        # at ``cut`` and terminate the stream cleanly.
                        truncated = combined[:cut]
                        # Signal the worker to drop the rest of the
                        # generator so the GPU isn't burning cycles
                        # past the truncation point.
                        cancel_event.set()
                        yield GenerationOutput(
                            text=truncated,
                            new_text=truncated,
                            tokens=item.tokens,
                            prompt_tokens=item.prompt_tokens,
                            completion_tokens=item.completion_tokens,
                            finish_reason="stop",
                            finished=True,
                        )
                        return
                    is_terminal = item.finish_reason is not None
                    if is_terminal:
                        # Last chunk — no future text can complete a
                        # stop, so the buffered tail is safe to flush.
                        yield GenerationOutput(
                            text=combined,
                            new_text=combined,
                            tokens=item.tokens,
                            prompt_tokens=item.prompt_tokens,
                            completion_tokens=item.completion_tokens,
                            finish_reason=item.finish_reason,
                            finished=item.finished,
                        )
                        tail = ""
                        continue
                    # Intermediate chunk — emit the safe prefix and
                    # buffer the lookback. ``tail_len == 0`` (single-
                    # character stops) needs the special-case below
                    # because Python's ``s[:-0]`` evaluates to ``""``.
                    if tail_len == 0:
                        safe = combined
                        tail = ""
                    elif len(combined) > tail_len:
                        safe = combined[:-tail_len]
                        tail = combined[-tail_len:]
                    else:
                        safe = ""
                        tail = combined
                    if safe:
                        yield GenerationOutput(
                            text=safe,
                            new_text=safe,
                            tokens=item.tokens,
                            prompt_tokens=item.prompt_tokens,
                            completion_tokens=item.completion_tokens,
                            finish_reason=None,
                            finished=False,
                        )
            finally:
                # Always cancel the worker job — covers the early-stop
                # path, caller disconnect (asyncio CancelledError), and
                # exceptions raised mid-stream. Idempotent under repeat
                # ``set()`` calls.
                cancel_event.set()
                # Wait for the worker to fully release this job before
                # we drop the engine lock, else a queued sibling
                # request acquires the lock while the worker is still
                # burning GPU on this job — head-of-line blocking
                # (codex round 6 [P2]). The per-step cancel check
                # inside ``_run_generator`` fires every diffusion
                # block (~50-200 ms), so this normally waits one
                # block at most. The 30 s ceiling is a defence-in-
                # depth so a stuck worker can never wedge the lock
                # forever. If the ceiling DOES fire, we mark the
                # engine as ``_worker_stuck`` so subsequent
                # ``check_admission`` calls fail fast and the operator
                # learns the engine is unhealthy — otherwise we'd
                # silently release the lock onto a worker that's
                # still burning GPU on the abandoned job (codex round
                # 7 [P2]).
                drained = await asyncio.to_thread(done_event.wait, 30.0)
                if not drained:
                    self._worker_stuck = True
                    logger.error(
                        "DiffusionEngine worker did not drain cancelled job "
                        "within 30 s — engine marked unhealthy. Further "
                        "admissions will fail until restart."
                    )
                # Unconditional pump terminator. The worker's own
                # _STREAM_DONE arrives eventually, but if cancellation
                # happens before the worker has produced any output
                # (e.g. mid-disconnect after the lock acquired), the
                # pump would otherwise block on ``thread_q.get()``
                # forever. Pushing our own sentinel guarantees pump
                # observes one even when the worker is slow / never
                # ran.
                thread_q.put(_STREAM_DONE)
                pump_thread.join(timeout=2.0)
                while not aio_q.empty():
                    try:
                        aio_q.get_nowait()
                    except asyncio.QueueEmpty:
                        break

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,  # noqa: ARG002
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        # Buffered raw-prompt completion (/v1/completions non-stream).
        # See ``stream_generate`` for why we bypass the chat template.
        text_parts: list[str] = []
        last: GenerationOutput | None = None
        async for chunk in self.stream_generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **kwargs,
        ):
            text_parts.append(chunk.new_text)
            last = chunk
        if last is None:
            return GenerationOutput(text="", finish_reason="stop")
        return GenerationOutput(
            text="".join(text_parts),
            tokens=last.tokens,
            prompt_tokens=last.prompt_tokens,
            completion_tokens=last.completion_tokens,
            finish_reason=last.finish_reason or "stop",
            finished=True,
        )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,  # noqa: ARG002
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        # /v1/completions sends RAW prompts — applying the chat
        # template here would prepend ``<start_of_turn>user`` etc.
        # and the client asking to continue "Once upon" would get a
        # response to a chat message rather than a continuation
        # (codex round 5 [P2]). Tokenize directly and call the
        # internal raw-prompt path.
        self._ensure_loaded()
        async for chunk in self._stream_prompt_raw(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **kwargs,
        ):
            yield chunk

    # ------------------------------------------------------------------
    # Internal — sync generator pump
    # ------------------------------------------------------------------

    def _run_generator(
        self,
        prompt: str,
        max_tokens: int,
        cfg: DiffusionGenerationConfig,
        out_q: queue.Queue,
        cancel_event: threading.Event,
    ) -> None:
        """Run mlx-vlm's diffusion generator on the current thread and
        push collapsed-per-block ``GenerationOutput`` instances onto
        ``out_q``. Mirrors mlx-vlm's own ``_diffusion_block_chunks``
        helper in ``server/generation.py:750-784`` — one SSE-friendly
        chunk per finished block.
        """
        import mlx.core as mx  # noqa: F401 — kept for symmetry with mlx_vlm
        from mlx_vlm.generate.diffusion import stream_diffusion_generate

        # NOTE: this method ALWAYS runs on the persistent worker
        # thread (see ``_worker_loop``), so the model weights, the
        # tokenizer-managed kv_cache, and the default GPU stream are
        # all bound to the same thread. We intentionally do NOT wrap
        # in mlx-vlm's ``wired_limit(model, [generation_stream])`` —
        # on M3 Ultra with the 26B-A4B-it-4bit checkpoint, that wrap
        # forces a single Metal command buffer past the per-buffer
        # IOGPU timeout (~5 s for the cold-shader first denoising
        # step) and crashes with ``[METAL] Command buffer execution
        # failed: Caused GPU Timeout Error``. Direct mlx-vlm probe
        # runs cleanly without wired_limit (0.9 s for 32 tokens).
        # wired_limit is only a perf hint (asks the OS to keep model
        # pages wired in physical RAM); dropping it costs at most an
        # extra page fault on the very first request.

        # Cancel-check #1 — covers the race where the request was
        # cancelled between the worker-loop fast-skip gate
        # (``_worker_loop`` line ~394) and us getting here. Without
        # this, we'd still tokenize + materialize input_ids + dispatch
        # the first diffusion block before the per-iteration check at
        # the bottom kicks in (codex round 7 [P2]).
        if cancel_event.is_set():
            return

        tokenizer = self._processor.tokenizer
        eos_id = getattr(self._model.config, "eos_token_id", None)
        if eos_id is not None and hasattr(tokenizer, "stopping_criteria"):
            tokenizer.stopping_criteria.reset(eos_id)

        # mlx-vlm expects ``input_ids`` as an mx.array of shape [1, N].
        ids = tokenizer.encode(prompt)
        input_ids = mx.array(ids)[None]

        skip_ids: set[int] = set()
        special = getattr(tokenizer, "all_special_ids", None) or []
        for sid in special:
            skip_ids.add(int(sid))

        kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "skip_special_token_ids": skip_ids,
            "temperature": float(cfg.temperature),
            "diffusion_sampler": cfg.diffusion_sampler,
        }
        if cfg.diffusion_steps is not None:
            kwargs["max_denoising_steps"] = int(cfg.diffusion_steps)
        if cfg.prefill_step_size is not None:
            # mlx-vlm only enables its chunked-prefill path when this
            # kwarg is non-None — forward only when the operator opted
            # in via --prefill-step-size / SchedulerConfig (codex r5).
            kwargs["prefill_step_size"] = int(cfg.prefill_step_size)

        block_parts: list[str] = []
        last_prompt_tokens = 0
        last_completion_tokens = 0
        last_token: int = 0

        # Cancel-check #2 — last opportunity before the first
        # ``next()`` on stream_diffusion_generate triggers the
        # expensive prefill. Race window is small (tokenize +
        # input_ids construction) but real (codex round 7 [P2]).
        if cancel_event.is_set():
            return

        for result in stream_diffusion_generate(
            self._model,
            self._processor,
            tokenizer,
            input_ids,
            None,  # pixel_values — text-only path
            None,  # attention_mask — auto from input_ids
            **kwargs,
        ):
            # Cancellation point — stream_chat sets this on an early
            # stop-sequence match or on disconnect so the worker
            # doesn't keep burning GPU up to ``max_tokens`` after the
            # caller has stopped consuming output. codex round 3 [P2]:
            # without this, a stop in the first block left the worker
            # generating hundreds of tokens before it could pick up
            # the next queued request.
            if cancel_event.is_set():
                break
            if getattr(result, "is_draft", False):
                # Mid-canvas denoising preview; ignore for SSE.
                continue

            if getattr(result, "prompt_tokens", 0):
                last_prompt_tokens = result.prompt_tokens
            if getattr(result, "generation_tokens", 0):
                last_completion_tokens = result.generation_tokens
            # codex pr_validate r5 NIT: ``... or last_token`` silently
            # swallows token id 0 (Gemma's <pad>, plus countless other
            # tokenizers' sentinel tokens) — keep the previous token id
            # only when the result truly omits the field.
            _tok = getattr(result, "token", None)
            if _tok is not None:
                last_token = int(_tok)

            text_piece = result.text or ""
            if text_piece:
                block_parts.append(text_piece)

            block_complete = bool(getattr(result, "diffusion_block_complete", False))
            finish_reason = getattr(result, "finish_reason", None)

            if block_complete or finish_reason:
                joined = "".join(block_parts)
                block_parts.clear()
                if joined or finish_reason:
                    out_q.put(
                        GenerationOutput(
                            text="",
                            new_text=joined,
                            tokens=[last_token],
                            prompt_tokens=last_prompt_tokens,
                            completion_tokens=last_completion_tokens,
                            finish_reason=(finish_reason if finish_reason else None),
                            finished=bool(finish_reason),
                            channel="content",
                        )
                    )
                if finish_reason:
                    return

        # Generator exited without an explicit finish_reason — treat
        # as a hard stop. We ALWAYS emit a finish chunk here even when
        # ``block_parts`` is empty (output ended exactly on a block
        # boundary). Otherwise the routes finish the stream with only
        # ``[DONE]`` and the client gets no terminal finish_reason /
        # usage; the stop-sequence holdback in stream_chat would also
        # never see a terminal item to flush its buffered tail (codex
        # round 4 [P2]). Skip when the worker was cancelled so we
        # don't push a stale terminal chunk into a queue the
        # stream_chat finally is already draining.
        if not cancel_event.is_set():
            out_q.put(
                GenerationOutput(
                    text="",
                    new_text="".join(block_parts),
                    tokens=[last_token],
                    prompt_tokens=last_prompt_tokens,
                    completion_tokens=last_completion_tokens,
                    finish_reason="stop",
                    finished=True,
                    channel="content",
                )
            )


# ------------------------------------------------------------------
# Backward-compat shim — PR #551 (skeleton) introduced ``DiffusionRunner``
# and ``load_runner``. Keep them as thin aliases so the existing test
# imports and any downstream draft branches keep working.
# ------------------------------------------------------------------

DiffusionRunner = DiffusionEngine
"""Alias retained from the skeleton PR; new code should use
``DiffusionEngine`` directly so the BaseEngine inheritance is
explicit at the call site."""


def load_runner(hf_path: str) -> DiffusionEngine:
    """Construct and load a ``DiffusionEngine`` for ``hf_path``.

    Synchronous — calls into mlx-vlm's blocking loader. Async callers
    should ``await asyncio.to_thread(load_runner, hf_path)`` instead
    of calling this directly from the event loop.
    """
    engine = DiffusionEngine(model_name=hf_path)
    engine._load_blocking()  # noqa: SLF001 — module-internal helper
    return engine


__all__ = [
    "DIFFUSION_LANE_VERSION",
    "DiffusionEngine",
    "DiffusionGenerationConfig",
    "DiffusionRunner",
    "load_runner",
]
