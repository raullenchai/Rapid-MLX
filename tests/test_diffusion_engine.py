# SPDX-License-Identifier: Apache-2.0
"""Behaviour tests for ``DiffusionEngine`` — the BaseEngine wrapper
over mlx-vlm 0.6.3's diffusion generator.

We mock mlx-vlm at the import surface (``mlx_vlm.utils.load``,
``mlx_vlm.generate.diffusion.stream_diffusion_generate``, etc.) so the
tests run without weights and without touching the GPU. The mock
shape mirrors the actual upstream contract documented in
``vllm_mlx/runtime/diffusion_lane.py`` so any drift between the
expected and real surface is loud at unit-test time.
"""

from __future__ import annotations

import sys
import threading
import types
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

# ----------------------------------------------------------------------
# Helpers — minimal mlx-vlm surface mock
# ----------------------------------------------------------------------


@dataclass
class FakeGenerationResult:
    """Mirror of mlx_vlm.generate.common.GenerationResult — only the
    fields DiffusionEngine reads. Keeps the test free of any actual
    mlx-vlm import."""

    text: str = ""
    token: int = 0
    prompt_tokens: int = 0
    generation_tokens: int = 0
    finish_reason: str | None = None
    is_draft: bool = False
    diffusion_block_complete: bool = False


class FakeTokenizer:
    """The bits of mlx-vlm's TokenizerWrapper that DiffusionEngine
    touches: ``apply_chat_template``, ``encode``, ``all_special_ids``,
    plus a no-op ``stopping_criteria.reset``."""

    all_special_ids = [0, 1, 2]

    class _StoppingCriteria:
        def reset(self, *_args: Any) -> None:
            pass

    stopping_criteria = _StoppingCriteria()

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        # Concatenate the user turns; good enough for a deterministic
        # prompt fingerprint inside the test.
        rendered = "\n".join(m.get("content", "") for m in messages)
        if add_generation_prompt:
            rendered += "\n<start_of_turn>model\n"
        return rendered

    def encode(self, text: str) -> list[int]:
        # Map characters to incrementing IDs; deterministic and
        # length-correlated so estimate_new_tokens can be exercised.
        return [ord(c) % 256 for c in text]


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()


class FakeModelConfig:
    eos_token_id = 7
    canvas_length = 256


class FakeModel:
    config = FakeModelConfig()


def _install_mlx_vlm_mock(
    monkeypatch: pytest.MonkeyPatch,
    *,
    family: str = "block",
    stream_yields: list[FakeGenerationResult] | None = None,
) -> None:
    """Wire stub modules into ``sys.modules`` so the real mlx-vlm
    imports inside ``diffusion_lane.py`` resolve to our fakes. We
    install everything DiffusionEngine touches; anything else will
    raise AttributeError at import time, which is the loud-failure
    behaviour we want."""

    # The real ``mlx`` package is a hard dependency and already
    # installed; we use ``mx.array`` from it directly. Only mock the
    # mlx-vlm-side modules so this test can run without the
    # 14 GB DiffusionGemma checkpoint on disk.

    # mlx_vlm.utils.load
    mlx_vlm_pkg = sys.modules.get("mlx_vlm") or types.ModuleType("mlx_vlm")
    mlx_vlm_utils = types.ModuleType("mlx_vlm.utils")

    def _load(hf_path: str) -> tuple[FakeModel, FakeProcessor]:
        return FakeModel(), FakeProcessor()

    mlx_vlm_utils.load = _load  # type: ignore[attr-defined]

    # mlx_vlm.generate.diffusion
    mlx_vlm_generate = types.ModuleType("mlx_vlm.generate")
    mlx_vlm_diffusion = types.ModuleType("mlx_vlm.generate.diffusion")

    def _family(_model: Any) -> str:
        return family

    captured_calls: dict[str, Any] = {}

    def _stream(
        model: Any,
        processor: Any,
        tokenizer: Any,
        input_ids: Any,
        pixel_values: Any,
        attention_mask: Any,
        **kwargs: Any,
    ) -> Iterator[FakeGenerationResult]:
        captured_calls["last"] = {
            "input_ids": input_ids,
            "kwargs": kwargs,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
        }
        yield from (stream_yields or [])

    mlx_vlm_diffusion.diffusion_generation_family = _family  # type: ignore[attr-defined]
    mlx_vlm_diffusion.stream_diffusion_generate = _stream  # type: ignore[attr-defined]
    mlx_vlm_diffusion.__captured__ = captured_calls  # type: ignore[attr-defined]

    # mlx_vlm.generate.common
    mlx_vlm_common = types.ModuleType("mlx_vlm.generate.common")

    class _WiredLimit:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def __enter__(self) -> _WiredLimit:
            return self

        def __exit__(self, *_a: Any) -> None:
            pass

    mlx_vlm_common.wired_limit = _WiredLimit  # type: ignore[attr-defined]
    mlx_vlm_common.generation_stream = object()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm_pkg)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", mlx_vlm_utils)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", mlx_vlm_generate)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate.diffusion", mlx_vlm_diffusion)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate.common", mlx_vlm_common)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class TestLoadAndIntrospection:
    def test_load_succeeds_for_block_family(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mlx_vlm_mock(monkeypatch)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        assert engine.model_name == "x/y"
        assert engine.is_mllm is False
        assert engine.tokenizer is not None

    def test_load_rejects_non_block_family(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mlx_vlm_mock(monkeypatch, family="masked")
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        with pytest.raises(RuntimeError, match="not a block-diffusion model"):
            engine._load_blocking()


class TestPromptAndTokenAccounting:
    def test_build_prompt_renders_via_chat_template(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mlx_vlm_mock(monkeypatch)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        rendered = engine.build_prompt([{"role": "user", "content": "Hello there"}])
        assert "Hello there" in rendered
        assert rendered.endswith("model\n")

    def test_build_prompt_silently_drops_tools_with_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # OpenAI-compatible frontends (Big-AGI, BCG, etc.) attach a
        # built-in tools list to every chat request even when the user
        # has not invoked a tool. We dropped the hard-reject so the
        # very first chat doesn't 500; the warning lets the operator
        # observe the drop in serve logs.
        _install_mlx_vlm_mock(monkeypatch)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        with caplog.at_level("WARNING", logger="vllm_mlx.runtime.diffusion_lane"):
            rendered = engine.build_prompt(
                [{"role": "user", "content": "Hello there"}],
                tools=[{"name": "foo"}, {"name": "bar"}, {"name": "baz"}],
            )
        # Prompt still rendered cleanly — chat surface keeps working.
        assert "Hello there" in rendered
        assert rendered.endswith("model\n")
        # Exactly one warning, with the tool count and a clear message.
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1, [r.message for r in warnings]
        assert "dropped 3 tool" in warnings[0].getMessage()

    def test_build_prompt_no_warning_when_tools_absent(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # tools=None and tools=[] are the bare-chat case — no warning
        # should fire (otherwise every plain message spams serve logs).
        _install_mlx_vlm_mock(monkeypatch)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        with caplog.at_level("WARNING", logger="vllm_mlx.runtime.diffusion_lane"):
            engine.build_prompt([{"role": "user", "content": "hi"}])
            engine.build_prompt([{"role": "user", "content": "hi"}], tools=None)
            engine.build_prompt([{"role": "user", "content": "hi"}], tools=[])
        assert [r for r in caplog.records if r.levelname == "WARNING"] == []

    def test_estimate_new_tokens_returns_conservative_pair(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mlx_vlm_mock(monkeypatch)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        total, new = engine.estimate_new_tokens("hello")
        assert total == new == 5


class TestStreamChatBlockCollapse:
    @pytest.mark.asyncio
    async def test_yields_one_chunk_per_block_complete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two finished blocks then a finish_reason — DiffusionEngine
        # should emit three GenerationOutput chunks (block1, block2,
        # terminal flush).
        yields = [
            # canvas 0: token yields → block complete
            FakeGenerationResult(text="Once "),
            FakeGenerationResult(text="upon "),
            FakeGenerationResult(text="a time.", diffusion_block_complete=True),
            # canvas 1: token yields → block complete
            FakeGenerationResult(text="There "),
            FakeGenerationResult(text="was a", diffusion_block_complete=True),
            # final stop marker
            FakeGenerationResult(
                text=" cat.",
                finish_reason="stop",
                prompt_tokens=4,
                generation_tokens=7,
            ),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()

        collected = []
        async for out in engine.stream_chat(
            [{"role": "user", "content": "tell story"}],
            max_tokens=64,
        ):
            collected.append(out)

        assert [c.new_text for c in collected] == [
            "Once upon a time.",
            "There was a",
            " cat.",
        ]
        # Only the final chunk carries finish_reason; the route uses
        # this to emit the terminal SSE event.
        assert collected[-1].finish_reason == "stop"
        assert collected[-1].finished is True
        assert collected[0].finish_reason is None
        # Token accounting flows through on the terminal chunk.
        assert collected[-1].prompt_tokens == 4
        assert collected[-1].completion_tokens == 7

    @pytest.mark.asyncio
    async def test_drafts_are_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Drafts (mid-canvas previews) must not reach the SSE stream
        # — they would flicker through the chat UI as in-progress
        # garbage. Only block_complete and finish_reason emit.
        yields = [
            FakeGenerationResult(text="[Mask][Mask]", is_draft=True),
            FakeGenerationResult(text="[Mask]Hi", is_draft=True),
            FakeGenerationResult(text="Hi there", diffusion_block_complete=True),
            FakeGenerationResult(text="", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        collected = []
        async for out in engine.stream_chat(
            [{"role": "user", "content": "hi"}], max_tokens=16
        ):
            collected.append(out)
        # Block 1 only; terminal finish carries no text payload but
        # still emits because the finish_reason needs to land.
        assert [c.new_text for c in collected] == ["Hi there", ""]
        assert collected[-1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_chat_with_tools_completes_normally(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Direct stream_chat invocation with a tools payload must not
        # crash — drops them and streams text. The warning is logged
        # by build_prompt, which routes/chat.py:691 calls upfront for
        # every request; stream_chat itself stays silent to avoid
        # double-warning when the route layer is in front. This test
        # pins "stream survives tools" — see
        # test_route_layer_warning_fires_exactly_once for the
        # warning-side contract.
        yields = [
            FakeGenerationResult(text="Hello ", diffusion_block_complete=True),
            FakeGenerationResult(text="world.", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        collected = []
        async for out in engine.stream_chat(
            [{"role": "user", "content": "hi"}],
            tools=[{"name": "web_search"}],
            max_tokens=16,
        ):
            collected.append(out)
        # Stream completes normally with the model's text output.
        assert [c.new_text for c in collected] == ["Hello ", "world."]
        assert collected[-1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_route_layer_warning_fires_exactly_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Mimics the routes/chat.py contract: build_prompt is called
        # once with the full tools list (line 691 in chat.py — the
        # eager template validation), then stream_chat runs the
        # generation. The single drop-warning must fire on the
        # build_prompt call; stream_chat re-uses build_prompt(messages)
        # internally with tools=None so we do NOT double-log.
        yields = [
            FakeGenerationResult(text="ok.", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        with caplog.at_level("WARNING", logger="vllm_mlx.runtime.diffusion_lane"):
            engine.build_prompt(
                [{"role": "user", "content": "hi"}],
                tools=[{"name": "web_search"}, {"name": "weather"}],
            )
            collected = []
            async for out in engine.stream_chat(
                [{"role": "user", "content": "hi"}],
                tools=[{"name": "web_search"}, {"name": "weather"}],
                max_tokens=16,
            ):
                collected.append(out)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        assert "dropped 2 tool" in warnings[0].getMessage()
        # And the stream still produced its output.
        assert collected[-1].new_text == "ok."

    @pytest.mark.asyncio
    async def test_stream_chat_rejects_vision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_mlx_vlm_mock(monkeypatch)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        with pytest.raises(RuntimeError, match="text-only"):
            async for _ in engine.stream_chat(
                [{"role": "user", "content": "hi"}],
                images=["/tmp/x.png"],
            ):
                pass

    @pytest.mark.asyncio
    async def test_chat_buffers_into_single_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        yields = [
            FakeGenerationResult(text="part 1 ", diffusion_block_complete=True),
            FakeGenerationResult(text="part 2", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        out = await engine.chat([{"role": "user", "content": "hi"}], max_tokens=32)
        assert out.text == "part 1 part 2"
        assert out.finish_reason == "stop"
        assert out.finished is True

    @pytest.mark.asyncio
    async def test_kwargs_forwarded_to_mlx_vlm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The diffusion-specific knobs (diffusion_steps, sampler) must
        # land in the stream_diffusion_generate call; without this
        # pin a future refactor could silently drop them.
        yields = [FakeGenerationResult(text="ok", finish_reason="stop")]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        async for _ in engine.stream_chat(
            [{"role": "user", "content": "hi"}],
            max_tokens=128,
            temperature=0.4,
            diffusion_steps=24,
            diffusion_sampler="entropy-bound",
        ):
            pass

        captured = sys.modules["mlx_vlm.generate.diffusion"].__captured__["last"]  # type: ignore[attr-defined]
        kwargs = captured["kwargs"]
        assert kwargs["max_tokens"] == 128
        assert kwargs["temperature"] == 0.4
        assert kwargs["max_denoising_steps"] == 24
        assert kwargs["diffusion_sampler"] == "entropy-bound"
        # Special-token skip set is forwarded; the FakeTokenizer
        # advertises three special IDs.
        assert {0, 1, 2} == kwargs["skip_special_token_ids"]


class TestStopSequenceHandling:
    """Codex round 1 [P2]: ``stop`` was previously dropped on the
    floor. The chat surface now post-processes block-chunk text and
    truncates at the first stop match across the lookback window so
    boundary-straddling matches are caught too."""

    @pytest.mark.asyncio
    async def test_stop_truncates_within_a_single_chunk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The visible joined output must be "Hello, " — anything past
        # the stop is excluded. Per-chunk shape varies with the
        # lookback buffer but the JOINED contract is what callers see.
        yields = [
            FakeGenerationResult(text="Hello, ", diffusion_block_complete=True),
            FakeGenerationResult(
                text="world! And more.", diffusion_block_complete=True
            ),
            FakeGenerationResult(text="", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        collected = []
        async for out in engine.stream_chat(
            [{"role": "user", "content": "hi"}],
            max_tokens=64,
            stop=["world"],
        ):
            collected.append(out)
        joined = "".join(c.new_text for c in collected)
        assert joined == "Hello, "
        assert "world" not in joined
        assert collected[-1].finish_reason == "stop"
        assert collected[-1].finished is True

    @pytest.mark.asyncio
    async def test_stop_straddling_block_boundary_no_leak(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # codex round 2 [P2]: a stop string straddling two block
        # boundaries must not leak its leading bytes to the client.
        # Stop ``</end>`` is split across chunks "Answer: 42</" and
        # "end> trailing.". The joined visible output must end at the
        # boundary BEFORE ``</`` started — the previous version
        # emitted the full first chunk before the second arrived and
        # leaked ``</`` to the client.
        yields = [
            FakeGenerationResult(text="Answer: 42</", diffusion_block_complete=True),
            FakeGenerationResult(text="end> trailing.", diffusion_block_complete=True),
            FakeGenerationResult(text="", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        collected = []
        async for out in engine.stream_chat(
            [{"role": "user", "content": "x"}],
            max_tokens=64,
            stop=["</end>"],
        ):
            collected.append(out)
        joined = "".join(c.new_text for c in collected)
        assert joined == "Answer: 42"
        assert "</" not in joined  # no leak of the stop's leading bytes
        assert "</end>" not in joined
        assert collected[-1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stop_accepts_string_list_and_picks_earliest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # OpenAI ``stop`` may be a string or list. Two stops, both
        # present in the chunk: the one with the LOWER INDEX in the
        # text wins (matches OpenAI behavior — order in the input
        # list is irrelevant; earliest match in the model output is
        # what truncates).
        yields = [
            FakeGenerationResult(
                text="hello [A] middle [B] tail", diffusion_block_complete=True
            ),
            FakeGenerationResult(text="", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        collected = []
        async for out in engine.stream_chat(
            [{"role": "user", "content": "x"}],
            max_tokens=64,
            # [B] appears LATER in the text but is FIRST in the stop
            # list — order in the list must not change the outcome.
            stop=["[B]", "[A]"],
        ):
            collected.append(out)
        assert collected[0].new_text == "hello "
        assert collected[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stop_none_means_passthrough(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Bare-chat case: stop=None or stop=[] → no post-processing.
        yields = [
            FakeGenerationResult(text="part1 ", diffusion_block_complete=True),
            FakeGenerationResult(text="part2", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        collected = []
        async for out in engine.stream_chat(
            [{"role": "user", "content": "x"}], max_tokens=64, stop=None
        ):
            collected.append(out)
        assert [c.new_text for c in collected] == ["part1 ", "part2"]
        assert collected[-1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_single_char_stop_streams_without_buffering(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # codex round 3 [P2]: for one-character stops like ``"\n"`` or
        # ``"}"``, ``tail_len`` is 0 — Python's ``s[:-0]`` is ``""``,
        # so the previous code buffered every chunk and TTFT collapsed
        # until the terminal chunk arrived. The special-case must
        # stream each chunk live and still truncate cleanly when the
        # stop character appears.
        yields = [
            FakeGenerationResult(text="line1", diffusion_block_complete=True),
            FakeGenerationResult(text="\nafter newline", diffusion_block_complete=True),
            FakeGenerationResult(text="", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        collected = []
        async for out in engine.stream_chat(
            [{"role": "user", "content": "x"}],
            max_tokens=64,
            stop=["\n"],
        ):
            collected.append(out)
        joined = "".join(c.new_text for c in collected)
        assert joined == "line1"
        # First chunk streamed live (NOT buffered) — exactly the
        # "no-buffering" property the round-3 fix is pinning.
        assert collected[0].new_text == "line1"
        assert collected[0].finish_reason is None
        assert collected[-1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_early_stop_cancels_worker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # codex round 3 [P2]: when stream_chat returns early on a stop
        # match, the persistent worker must observe the cancel signal
        # and stop reading mlx-vlm's generator. Without this it would
        # keep generating up to ``max_tokens`` and monopolize the
        # single GPU worker thread until the next request can land.
        # We model this with an infinite mlx-vlm stream and assert the
        # consumption count is bounded.
        consumed = {"n": 0}

        def infinite_yields() -> Iterator[FakeGenerationResult]:
            while True:
                consumed["n"] += 1
                yield FakeGenerationResult(
                    text=f"chunk{consumed['n']} STOP rest",
                    diffusion_block_complete=True,
                )

        # Install a custom mock that uses the live counter instead of
        # the pre-baked list ``_install_mlx_vlm_mock`` expects.
        _install_mlx_vlm_mock(monkeypatch)

        def _stream_infinite(*_a: Any, **_k: Any) -> Iterator[FakeGenerationResult]:
            yield from infinite_yields()

        sys.modules["mlx_vlm.generate.diffusion"].stream_diffusion_generate = (  # type: ignore[attr-defined]
            _stream_infinite
        )

        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        collected = []
        async for out in engine.stream_chat(
            [{"role": "user", "content": "x"}],
            max_tokens=10_000,
            stop=["STOP"],
        ):
            collected.append(out)
        # Stop landed in the very first chunk so output ends cleanly.
        joined = "".join(c.new_text for c in collected)
        assert joined == "chunk1 "
        assert collected[-1].finish_reason == "stop"
        # Worker MUST have observed cancellation and stopped iterating.
        # The mock is pure-Python so it spins fast; what matters is
        # the worker is no longer ADVANCING ``consumed`` after we
        # wait for it to settle — i.e., cancellation actually fired,
        # not "the loop runs forever and we just measure a snapshot."
        import time as _time

        _time.sleep(0.3)
        n_settled = consumed["n"]
        _time.sleep(0.5)
        assert consumed["n"] == n_settled, (
            f"Worker still iterating after cancel: {n_settled} → {consumed['n']}"
        )


class TestTerminationEdgeCases:
    """Codex round 4 [P2] x 2: pump-thread leak on lock-cancel, and a
    missing finish chunk when the diffusion generator ends exactly on
    a block boundary (block_parts already cleared)."""

    @pytest.mark.asyncio
    async def test_generator_exit_at_block_boundary_still_emits_finish(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Generator yields one block-complete chunk and then exhausts
        # WITHOUT an explicit finish_reason. Without the round-4 fix
        # the worker's tail-flush guard ``if block_parts`` skipped
        # emitting a terminal chunk (block_parts had been cleared on
        # the previous block_complete), and stream_chat closed with
        # no finish_reason — routes shipped only [DONE] and clients
        # got no usage / terminal marker.
        yields = [
            FakeGenerationResult(text="all done.", diffusion_block_complete=True),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        collected = []
        async for out in engine.stream_chat(
            [{"role": "user", "content": "hi"}], max_tokens=16
        ):
            collected.append(out)
        # At least one chunk must carry finish_reason="stop" and
        # finished=True so the route layer can emit the terminal SSE
        # event with usage.
        assert any(c.finish_reason == "stop" and c.finished for c in collected)
        # Visible text is unchanged.
        joined = "".join(c.new_text for c in collected)
        assert joined == "all done."

    @pytest.mark.asyncio
    async def test_pump_thread_does_not_leak_on_lock_cancel(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # codex round 4 [P2]: if a queued request is cancelled while
        # waiting on _generation_lock, the pump thread must NOT have
        # been started — otherwise it would block on thread_q.get()
        # forever (no job ever runs to push _STREAM_DONE). We model
        # this by holding the lock with a long-running request, then
        # cancelling a second request while it's queued.
        import asyncio as _aio

        yields = [
            FakeGenerationResult(text="slow", diffusion_block_complete=True),
            FakeGenerationResult(text="", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()

        # Manually acquire the engine lock so the next request is
        # queued, then cancel it before releasing.
        await engine._generation_lock.acquire()

        baseline_threads = {
            t.name for t in threading.enumerate() if "diffusion-pump" in t.name
        }

        async def queued_request() -> None:
            async for _ in engine.stream_chat(
                [{"role": "user", "content": "go"}], max_tokens=16
            ):
                pass

        task = _aio.create_task(queued_request())
        # Give the task a moment to enter stream_chat and reach the
        # lock-acquire await.
        await _aio.sleep(0.1)
        task.cancel()
        try:
            await task
        except _aio.CancelledError:
            pass
        engine._generation_lock.release()

        # No new pump threads should be alive — the cancelled
        # request must not have started one.
        import time as _time

        _time.sleep(0.2)
        alive_pumps = {
            t.name
            for t in threading.enumerate()
            if "diffusion-pump" in t.name and t.is_alive()
        }
        new_pumps = alive_pumps - baseline_threads
        assert not new_pumps, f"Leaked pump threads: {new_pumps}"


class TestConcurrentRequests:
    """Codex round 1 [P1]: a sync ``threading.Lock`` held across
    ``await aio_q.get()`` deadlocked the event loop when a second
    request arrived. Switching to ``asyncio.Lock`` lets the loop
    advance the first request to completion while the second waits."""

    @pytest.mark.asyncio
    async def test_two_concurrent_requests_serialize_without_deadlock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Both requests produce a single block + stop. The second one
        # must wait for the first to release the lock — total wall is
        # 2x the per-request cost, but neither should hang.
        yields = [
            FakeGenerationResult(text="A1", diffusion_block_complete=True),
            FakeGenerationResult(text="", finish_reason="stop"),
            FakeGenerationResult(text="B1", diffusion_block_complete=True),
            FakeGenerationResult(text="", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()

        async def drain() -> list[str]:
            chunks: list[str] = []
            async for out in engine.stream_chat(
                [{"role": "user", "content": "go"}], max_tokens=16
            ):
                chunks.append(out.new_text)
            return chunks

        import asyncio as _aio

        results = await _aio.wait_for(
            _aio.gather(drain(), drain()),
            timeout=10.0,
        )
        # Both requests completed. Mock yields four items total, so
        # the two requests together drain them in submission order.
        assert all(len(r) == 2 for r in results), results

    def test_generation_lock_is_asyncio_lock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Source-level pin: the lock type matters for correctness
        # under the async generator model. A regression to threading.
        # Lock would silently reintroduce the deadlock.
        import asyncio as _aio

        _install_mlx_vlm_mock(monkeypatch)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        assert isinstance(engine._generation_lock, _aio.Lock)


class TestAdmissionControl:
    """Codex round 2 [P2]: routes/chat.py's ``_check_admission_or_503``
    was silently no-op'ing for the diffusion lane because the engine
    did not implement ``check_admission`` / ``release_admission_
    reservation``. Concurrent local requests piled up behind
    ``_generation_lock`` instead of returning the documented 503 +
    Retry-After at the configured cap."""

    def test_check_admission_no_op_without_scheduler_config_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Default SchedulerConfig has a max_concurrent_requests
        # default; under it, check_admission should reserve a slot
        # and NOT raise.
        _install_mlx_vlm_mock(monkeypatch)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        engine.check_admission()  # No raise.
        assert engine._admission_reservations == 1
        engine.release_admission_reservation()
        assert engine._admission_reservations == 0

    def test_check_admission_raises_at_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # With cap=2, the 3rd reservation must raise BackpressureError.
        # The route layer catches that and emits 503 + Retry-After.
        _install_mlx_vlm_mock(monkeypatch)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine
        from vllm_mlx.scheduler import BackpressureError, SchedulerConfig

        engine = DiffusionEngine(
            model_name="x/y",
            scheduler_config=SchedulerConfig(max_concurrent_requests=2),
        )
        engine._load_blocking()
        engine.check_admission()
        engine.check_admission()
        with pytest.raises(BackpressureError, match="max_concurrent_requests=2"):
            engine.check_admission()
        # Releasing one lets the next call succeed.
        engine.release_admission_reservation()
        engine.check_admission()  # No raise.

    def test_release_idempotent_below_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A stray double-release must not corrupt the counter into
        # negative territory (would silently raise the cap forever).
        _install_mlx_vlm_mock(monkeypatch)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine.release_admission_reservation()  # extra release at 0
        engine.release_admission_reservation()
        assert engine._admission_reservations == 0


class TestAliasIntegration:
    def test_diffusion_gemma_alias_resolves_to_text_diffusion_modality(
        self,
    ) -> None:
        # End-to-end pin on the actual aliases.json entry — if a
        # future edit drops the modality field, this test catches it
        # before the server boot does.
        from vllm_mlx.model_aliases import resolve_profile

        profile = resolve_profile("diffusion-gemma-26b")
        assert profile is not None
        assert profile.modality == "text-diffusion"
        assert profile.supports_spec_decode is False
        assert profile.supports_dflash is False
        assert profile.hf_path == "mlx-community/diffusiongemma-26B-A4B-it-4bit"
