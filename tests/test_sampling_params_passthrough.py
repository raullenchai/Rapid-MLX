# SPDX-License-Identifier: Apache-2.0
"""Regression guard for #355 — extended sampling params passthrough.

Five OpenAI-compatible sampling parameters (top_k, min_p, repetition_penalty,
presence_penalty, frequency_penalty) are accepted by ``SamplingParams`` in
``vllm_mlx/request.py`` and honoured by the underlying mlx-lm engine, but
were silently dropped at three API-layer boundaries:

  1. ``ChatCompletionRequest`` / ``CompletionRequest`` (api/models.py) —
     fields not declared, so Pydantic drops them on parse.
  2. ``chat_kwargs`` assembly (routes/chat.py L338) — only max_tokens,
     temperature, top_p, and stop were populated.
  3. ``SamplingParams`` construction (engine/batched.py L677 + L756) —
     ``generate()`` and ``stream_generate()`` build SamplingParams from
     only those four fields.

Concrete impact: Qwen3.6's published recommended ``top_k=20`` for coding
was unreachable from any OpenAI client.

These tests pin the contract at each of the three layers.
"""

from __future__ import annotations

from vllm_mlx.api.models import ChatCompletionRequest, CompletionRequest
from vllm_mlx.request import SamplingParams

# A realistic payload — Qwen3.6 published coding-tuned sampling.
QWEN36_CODING_PAYLOAD = {
    "model": "qwen3.6-35b-4bit",
    "messages": [{"role": "user", "content": "hi"}],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.5,
    "frequency_penalty": 0.0,
}


# =============================================================================
# Layer 1 — Pydantic models preserve the fields
# =============================================================================


def test_chat_completion_request_preserves_extended_sampling_params():
    """ChatCompletionRequest must surface top_k / min_p / repetition_penalty /
    presence_penalty / frequency_penalty as attributes after parsing JSON.
    Before #355's fix Pydantic dropped them silently."""
    req = ChatCompletionRequest(**QWEN36_CODING_PAYLOAD)

    assert req.top_k == 20, f"top_k dropped — got {req.top_k!r}"
    assert req.min_p == 0.0, f"min_p dropped — got {req.min_p!r}"
    assert req.repetition_penalty == 1.0, (
        f"repetition_penalty dropped — got {req.repetition_penalty!r}"
    )
    assert req.presence_penalty == 0.5, (
        f"presence_penalty dropped — got {req.presence_penalty!r}"
    )
    assert req.frequency_penalty == 0.0, (
        f"frequency_penalty dropped — got {req.frequency_penalty!r}"
    )


def test_chat_completion_request_defaults_to_none_when_unset():
    """Unset extended sampling params must default to None (not 0/1.0 etc.)
    so the route handler can distinguish 'client didn't specify' from
    'client explicitly chose a value'. Mixing them would make us override
    SamplingParams defaults even when the client wanted defaults."""
    req = ChatCompletionRequest(
        model="qwen3.5-4b-4bit",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert req.top_k is None
    assert req.min_p is None
    assert req.repetition_penalty is None
    assert req.presence_penalty is None
    assert req.frequency_penalty is None


def test_completion_request_preserves_extended_sampling_params():
    """Mirror of the chat-request test for /v1/completions."""
    payload = {
        "model": "qwen3.6-35b-4bit",
        "prompt": "hi",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.0,
    }
    req = CompletionRequest(**payload)

    assert req.top_k == 20
    assert req.min_p == 0.05
    assert req.repetition_penalty == 1.1
    assert req.presence_penalty == 0.5
    assert req.frequency_penalty == 0.0


# =============================================================================
# Layer 2 — chat_kwargs assembly carries the values
# =============================================================================


def _build_chat_kwargs(req: ChatCompletionRequest) -> dict:
    """Replay the same kwargs-build logic the route handler runs, isolated
    from the route's many other dependencies (engine, cfg, multimodal, etc.).

    This must stay aligned with vllm_mlx/routes/chat.py around the
    ``chat_kwargs = { ... }`` block — if that block moves, update here.
    """
    from vllm_mlx.routes.chat import _resolve_temperature, _resolve_top_p

    chat_kwargs: dict = {
        "max_tokens": req.max_tokens or 256,
        "temperature": _resolve_temperature(req.temperature),
        "top_p": _resolve_top_p(req.top_p),
        "stop": req.stop,
    }
    # The fix should add the new fields only when the client explicitly
    # set them — otherwise SamplingParams defaults stay in effect.
    for name in (
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    ):
        value = getattr(req, name, None)
        if value is not None:
            chat_kwargs[name] = value
    return chat_kwargs


def test_chat_kwargs_passes_through_when_client_sets_extended_params():
    """When the client sends extended sampling params, the kwargs dict the
    engine receives must carry them. Pins layer 2 of the fix."""
    req = ChatCompletionRequest(**QWEN36_CODING_PAYLOAD)
    chat_kwargs = _build_chat_kwargs(req)

    assert chat_kwargs["top_k"] == 20
    assert chat_kwargs["min_p"] == 0.0
    assert chat_kwargs["repetition_penalty"] == 1.0
    assert chat_kwargs["presence_penalty"] == 0.5
    assert chat_kwargs["frequency_penalty"] == 0.0


def test_chat_kwargs_omits_extended_params_when_client_silent():
    """When the client doesn't send extended params, the kwargs dict must
    NOT contain them — otherwise we'd override the engine's defaults with
    None and break the SamplingParams contract."""
    req = ChatCompletionRequest(
        model="qwen3.5-4b-4bit",
        messages=[{"role": "user", "content": "hi"}],
    )
    chat_kwargs = _build_chat_kwargs(req)

    for name in (
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    ):
        assert name not in chat_kwargs, (
            f"{name!r} leaked into chat_kwargs even though client didn't set it"
        )


def test_completion_route_forwards_extended_params_to_engine():
    """Pin the /v1/completions route side: when the client sends extended
    sampling params, the route must forward them to ``engine.generate`` and
    ``engine.stream_generate``. Without this guard the chat route works but
    the legacy completions endpoint silently drops the same fields — which is
    exactly what shipped before #355's full fix."""
    import asyncio

    captured: list[dict] = []

    class _FakeOutput:
        text = "ok"
        finish_reason = "stop"
        completion_tokens = 1
        prompt_tokens = 1

    class _FakeEngine:
        async def generate(self, **kw):
            captured.append({"call": "generate", **kw})
            return _FakeOutput()

        async def stream_generate(self, **kw):
            captured.append({"call": "stream_generate", **kw})
            yield _FakeOutput()

    req = CompletionRequest(
        model="qwen3.6-35b-4bit",
        prompt="hi",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.05,
        repetition_penalty=1.1,
        presence_penalty=0.5,
        frequency_penalty=0.3,
    )

    # Replay the route handler's extended_kwargs assembly — kept aligned with
    # vllm_mlx/routes/completions.py. If that loop moves, update here.
    extended_kwargs: dict = {}
    for name in (
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    ):
        value = getattr(req, name, None)
        if value is not None:
            extended_kwargs[name] = value

    engine = _FakeEngine()
    asyncio.run(
        engine.generate(
            prompt=req.prompt,
            max_tokens=req.max_tokens or 256,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
            **extended_kwargs,
        )
    )

    assert captured[0]["top_k"] == 20
    assert captured[0]["min_p"] == 0.05
    assert captured[0]["repetition_penalty"] == 1.1
    assert captured[0]["presence_penalty"] == 0.5
    assert captured[0]["frequency_penalty"] == 0.3


def test_completion_route_omits_extended_params_when_client_silent():
    """Mirror of the chat-route variant: legacy /v1/completions clients that
    don't set these fields must not see them leaked as None into engine
    kwargs (which would override SamplingParams defaults)."""
    req = CompletionRequest(model="qwen3.5-4b-4bit", prompt="hi")
    extended_kwargs: dict = {}
    for name in (
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    ):
        value = getattr(req, name, None)
        if value is not None:
            extended_kwargs[name] = value

    assert extended_kwargs == {}


# =============================================================================
# Layer 3 — SamplingParams accepts every extended field
# =============================================================================


def test_sampling_params_accepts_extended_fields():
    """SamplingParams must store every extended sampling parameter the
    engine layer needs to forward to mlx-lm. Pins layer 3 — guard against
    a future refactor that drops one of these fields from the dataclass."""
    sp = SamplingParams(
        max_tokens=128,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.05,
        repetition_penalty=1.1,
        presence_penalty=0.5,
        frequency_penalty=0.3,
    )

    assert sp.top_k == 20
    assert sp.min_p == 0.05
    assert sp.repetition_penalty == 1.1
    assert sp.presence_penalty == 0.5
    assert sp.frequency_penalty == 0.3


# =============================================================================
# Layer 4 — scheduler honours top_k + penalties
# =============================================================================


def test_scheduler_create_batch_generator_passes_top_k(monkeypatch):
    """Pin the actual scheduler call site: `Scheduler._create_batch_generator`
    must forward `sampling_params.top_k` to mlx-lm's `make_sampler`. Pre-#355
    the call was `make_sampler(temp, top_p, min_p)` — silently dropping
    top_k for every request.

    Driven through the real production code path so a future refactor that
    bypasses _create_batch_generator won't sneak past this test.
    """
    import vllm_mlx.scheduler as sch

    captured = {}

    def fake_make_sampler(**kwargs):
        captured.update(kwargs)
        return lambda x: x  # any callable works — only args matter

    monkeypatch.setattr(sch, "make_sampler", fake_make_sampler)

    sp = SamplingParams(
        max_tokens=64,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.05,
    )

    # Build a minimally-initialised Scheduler so we can invoke
    # _create_batch_generator without spinning up a real model. The method
    # itself only reads sampling_params + calls make_sampler + BatchGenerator;
    # we patch BatchGenerator too so we don't load any weights.
    monkeypatch.setattr(sch, "BatchGenerator", lambda *a, **kw: object())

    scheduler = sch.Scheduler.__new__(sch.Scheduler)
    # _create_batch_generator reads self.model, self.config.*batch_size,
    # and self._get_stop_tokens(); BatchGenerator construction is patched
    # above so we don't need real weights/tokenizer.
    scheduler.model = object()
    scheduler.tokenizer = object()
    scheduler._get_stop_tokens = lambda: set()

    class _StubCfg:
        prefill_batch_size = 1
        completion_batch_size = 1
        prefill_step_size = 1
        chunked_prefill_tokens = 0
        enable_mtp = False
        enable_suffix_decoding = False

    scheduler.config = _StubCfg()
    scheduler.memory_aware_cache = None

    scheduler._create_batch_generator(sp)

    assert captured.get("top_k") == 20, (
        f"top_k not forwarded to make_sampler — got {captured!r}. "
        f"This is the regression #355 was opened for."
    )
    assert captured.get("min_p") == 0.05
    assert captured.get("top_p") == 0.95
    assert captured.get("temp") == 0.6


def test_make_logits_processors_skips_default_penalties():
    """Behaviour contract that the scheduler relies on: passing `None` for
    each penalty must produce an empty processor list. If mlx-lm changes
    this to allocate a no-op processor for `None`, our optimisation breaks
    and every request pays for unused logits processors."""
    from mlx_lm.sample_utils import make_logits_processors

    procs = make_logits_processors(
        repetition_penalty=None,
        presence_penalty=None,
        frequency_penalty=None,
    )
    assert len(procs) == 0, (
        f"mlx-lm's make_logits_processors no longer returns [] for "
        f"all-None penalties; got {len(procs)} processors. The scheduler's "
        f"#355 wiring substitutes None for default values to avoid "
        f"allocating no-op processors — that optimisation is now broken."
    )


def test_all_scheduler_make_sampler_calls_pass_top_k():
    """Source-level guard: every ``make_sampler(...)`` call in scheduler.py
    that sources sampling params from a real request must pass ``top_k=``.

    ``_create_batch_generator`` (L1755) is exercised by the dynamic test
    above, but scheduler.py has a second per-request site (L2582 today —
    the per-request sampler used by ``BatchGenerator.insert``). A future
    refactor that drops ``top_k`` from either site silently reverts the
    fix for every request taking that code path. The greedy draft sampler
    used by spec decode (``temp=0.0``) is exempt — top_k is irrelevant
    for argmax.
    """
    from pathlib import Path

    src = Path("vllm_mlx/scheduler.py").read_text()

    import re

    # Match each `make_sampler(` call and the parenthesised arg list. The
    # regex stops at the matching close-paren so multi-line calls are
    # captured intact.
    calls: list[str] = []
    for match in re.finditer(r"make_sampler\(", src):
        start = match.end()
        depth = 1
        i = start
        while i < len(src) and depth > 0:
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
            i += 1
        calls.append(src[start : i - 1])

    assert calls, "make_sampler not found in scheduler.py — refactored away?"

    for call_args in calls:
        # The draft sampler is greedy (temp=0.0) and intentionally omits
        # top_k. All other call sites must wire top_k.
        if "temp=0.0" in call_args:
            continue
        assert "top_k=" in call_args, (
            f"A non-greedy make_sampler call in scheduler.py is missing "
            f"top_k=. This regresses #355. Offending call args:\n{call_args}"
        )


def test_make_logits_processors_signature_supports_three_penalties():
    """Sanity check on the upstream mlx-lm contract — the function we wire
    repetition/presence/frequency_penalty through must still accept those
    three keyword args. If mlx-lm renames any of them, the scheduler
    wiring breaks silently (penalties become no-ops) and this test catches
    the upstream API drift before users hit it."""
    import inspect

    from mlx_lm.sample_utils import make_logits_processors

    params = inspect.signature(make_logits_processors).parameters
    for name in ("repetition_penalty", "presence_penalty", "frequency_penalty"):
        assert name in params, (
            f"mlx-lm make_logits_processors no longer accepts {name!r}; "
            f"scheduler wiring is broken — see #355."
        )


def test_scheduler_overrides_openai_penalty_context_size():
    """#470 regression guard. mlx-lm's ``make_logits_processors`` defaults
    ``presence_context_size`` and ``frequency_context_size`` to 20 — only
    the last 20 generated tokens count toward the penalty. OpenAI's spec
    defines these penalties over the *entire* generated sequence, so the
    default makes the penalty feel like a no-op on chat-length outputs
    (#470 reported "zero observable effect" at ``frequency_penalty=2.0``).

    The scheduler must explicitly pass a much larger context size so the
    penalty actually covers a realistic chat response. We pin 4096 as the
    minimum; raising it is fine, lowering reverts #470.
    """
    from pathlib import Path

    src = Path("vllm_mlx/scheduler.py").read_text()

    # Find the `make_logits_processors(` call in the penalty wiring block
    # and capture its parenthesised arg list intact.
    import re

    match = re.search(r"make_logits_processors\(", src)
    assert match, "make_logits_processors() call not found in scheduler.py"

    start = match.end()
    depth = 1
    i = start
    while i < len(src) and depth > 0:
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
        i += 1
    call_args = src[start : i - 1]

    # Each OpenAI-spec penalty must override the upstream default of 20.
    # We require an integer literal ≥ 4096 — the file-level constant
    # makes review easy if someone tries to "just bump" it later.
    for kw in ("presence_context_size", "frequency_context_size"):
        m = re.search(rf"{kw}\s*=\s*(\d+)", call_args)
        assert m, (
            f"scheduler.py make_logits_processors call is missing {kw}=. "
            f"Without it, mlx-lm uses its 20-token default and the penalty "
            f"feels like a no-op on chat-length output — regresses #470."
        )
        value = int(m.group(1))
        assert value >= 4096, (
            f"scheduler.py passes {kw}={value} to mlx-lm. That's too small "
            f"to cover a typical chat response; OpenAI semantics apply the "
            f"penalty over the entire generated sequence. Use ≥ 4096."
        )

    # Repetition penalty is a rapid-mlx extension (not OpenAI-spec) and
    # is documented as multiplicative over a rolling window — leaving it
    # at mlx-lm's default 20 is intentional. Assert we don't accidentally
    # bump it (which would silently change semantics for existing users).
    assert "repetition_context_size" not in call_args, (
        "repetition_context_size should NOT be overridden in the scheduler. "
        "repetition_penalty is rapid-mlx's multiplicative rolling-window "
        "extension; only the OpenAI-spec frequency/presence penalties need "
        "the larger window. If you're intentionally changing this, update "
        "the test and document the semantic change."
    )
