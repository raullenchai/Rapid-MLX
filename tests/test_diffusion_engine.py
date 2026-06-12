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

    last_tools: list[dict] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        tools: list[dict] | None = None,
    ) -> str:
        # Concatenate the user turns; good enough for a deterministic
        # prompt fingerprint inside the test.
        rendered = "\n".join(m.get("content", "") for m in messages)
        if tools:
            rendered = (
                "TOOLS=" + ",".join(t.get("name", "?") for t in tools) + "\n" + rendered
            )
        FakeTokenizer.last_tools = tools
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

    def test_build_prompt_forwards_tools_when_alias_declares_parser(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When the alias declares a supported tool parser (gemma4),
        # ``tools`` are forwarded to ``apply_chat_template`` so the
        # template renders the function declarations into the prompt.
        # routes/chat.py then runs the gemma4 text parser against the
        # canvas output to recover structured ``tool_calls``.
        _install_mlx_vlm_mock(monkeypatch)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="diffusion-gemma-26b-4bit")
        engine._load_blocking()
        FakeTokenizer.last_tools = None
        rendered = engine.build_prompt(
            [{"role": "user", "content": "Hello there"}],
            tools=[{"name": "foo"}, {"name": "bar"}, {"name": "baz"}],
        )
        assert "Hello there" in rendered
        assert rendered.endswith("model\n")
        # ``tools`` reached the underlying chat template (FakeTokenizer
        # records the last list it saw) and the rendered prompt carries
        # the function names — both halves of the contract.
        assert FakeTokenizer.last_tools is not None
        assert [t["name"] for t in FakeTokenizer.last_tools] == ["foo", "bar", "baz"]
        assert "TOOLS=foo,bar,baz" in rendered

    def test_build_prompt_drops_tools_when_alias_has_no_parser(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # codex r1 BLOCKING #1 negative control: a bare HF path with
        # no matching alias (and therefore no resolved tool parser)
        # MUST NOT forward ``tools`` to ``apply_chat_template``.
        # Tokenizers whose template doesn't accept the kwarg would
        # otherwise raise ``TypeError`` and turn a frontend's
        # incidentally-attached tools list into a 500.
        _install_mlx_vlm_mock(monkeypatch)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="some/random-hf-path")
        engine._load_blocking()
        assert engine.supports_tool_calls is False
        FakeTokenizer.last_tools = None
        rendered = engine.build_prompt(
            [{"role": "user", "content": "Hello there"}],
            tools=[{"name": "foo"}],
        )
        assert "Hello there" in rendered
        # Tools never reached the chat template.
        assert FakeTokenizer.last_tools is None
        assert "TOOLS=" not in rendered

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
    async def test_stream_chat_forwards_tools_to_template(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Direct stream_chat invocation with a tools payload must
        # (a) not crash, (b) forward ``tools`` through build_prompt
        # to ``apply_chat_template`` so the model sees the function
        # declarations, and (c) emit no "dropped" warning (the
        # v0.7.1 fallback that silently swallowed tools is gone).
        # routes/chat.py recovers structured ``tool_calls`` via the
        # alias's ``tool_call_parser`` text parser; the engine just
        # has to surface the rendered prompt and the canvas tokens.
        yields = [
            FakeGenerationResult(text="Hello ", diffusion_block_complete=True),
            FakeGenerationResult(text="world.", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        # diffusion-gemma-26b-4bit alias declares ``tool_call_parser="gemma4"``
        # → engine.supports_tool_calls is True → tools are forwarded.
        engine = DiffusionEngine(model_name="diffusion-gemma-26b-4bit")
        engine._load_blocking()
        FakeTokenizer.last_tools = None
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
        # ``tools`` reached the underlying chat template via
        # build_prompt → apply_chat_template — the contract for
        # callers that bypass routes/chat.py.
        assert FakeTokenizer.last_tools is not None
        assert [t["name"] for t in FakeTokenizer.last_tools] == ["web_search"]

    @pytest.mark.asyncio
    async def test_route_layer_forwards_tools_on_every_pass(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Mimics the routes/chat.py contract: build_prompt is called
        # once eagerly with the full tools list to validate the
        # template, then stream_chat runs the generation (which
        # internally re-renders via build_prompt). Both passes MUST
        # forward ``tools`` to ``apply_chat_template`` so the model
        # sees the function declarations on both renders.
        yields = [
            FakeGenerationResult(text="ok.", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        # Tool-enabled alias.
        engine = DiffusionEngine(model_name="diffusion-gemma-26b-4bit")
        engine._load_blocking()
        FakeTokenizer.last_tools = None
        rendered_eager = engine.build_prompt(
            [{"role": "user", "content": "hi"}],
            tools=[{"name": "web_search"}, {"name": "weather"}],
        )
        # Eager pass forwarded the tools to the template.
        assert FakeTokenizer.last_tools is not None
        assert [t["name"] for t in FakeTokenizer.last_tools] == [
            "web_search",
            "weather",
        ]
        assert "TOOLS=web_search,weather" in rendered_eager
        # Now reset and run stream_chat — it should also forward.
        FakeTokenizer.last_tools = None
        collected = []
        async for out in engine.stream_chat(
            [{"role": "user", "content": "hi"}],
            tools=[{"name": "web_search"}, {"name": "weather"}],
            max_tokens=16,
        ):
            collected.append(out)
        # stream_chat's internal build_prompt forwarded tools too.
        assert FakeTokenizer.last_tools is not None
        assert [t["name"] for t in FakeTokenizer.last_tools] == [
            "web_search",
            "weather",
        ]
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


class TestRawCompletionPath:
    """Codex round 5 [P2]: /v1/completions sends RAW prompts. The
    ``stream_generate`` / ``generate`` path must feed those bytes
    verbatim to the tokenizer; wrapping them in the Gemma chat
    template would prepend ``<start_of_turn>user`` and answer a
    fictitious chat turn instead of continuing the raw text."""

    @pytest.mark.asyncio
    async def test_stream_generate_does_not_apply_chat_template(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The fake tokenizer's apply_chat_template appends
        # "\n<start_of_turn>model\n" — if stream_generate accidentally
        # wraps the raw prompt, the encoded token sequence will be
        # ~21 bytes longer than the prompt itself. We pin the exact
        # length to catch any future regression.
        yields = [FakeGenerationResult(text="rest", finish_reason="stop")]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        raw = "Once upon"
        async for _ in engine.stream_generate(raw, max_tokens=8):
            pass
        captured = sys.modules["mlx_vlm.generate.diffusion"].__captured__["last"]  # type: ignore[attr-defined]
        ids = captured["input_ids"]
        # input_ids shape is [1, N] — N must equal len(raw), not the
        # chat-template-wrapped length.
        # FakeTokenizer.encode is ``ord(c) % 256`` — encoding the raw
        # prompt yields exactly len(raw) tokens. If the chat template
        # had been wrapped, the shape would be len(raw) + len(
        # "\n<start_of_turn>model\n") = 9 + 21 = 30, NOT 9. Shape is
        # safe to query off the worker thread's stream (it's metadata,
        # not a materialized read), whereas ``.tolist()`` would need
        # the GPU stream binding we deliberately don't share.
        assert ids.shape == (1, len(raw))

    @pytest.mark.asyncio
    async def test_generate_buffers_completions_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Non-stream completion path must collapse stream chunks into
        # one GenerationOutput AND still bypass the chat template.
        yields = [
            FakeGenerationResult(text=" of ", diffusion_block_complete=True),
            FakeGenerationResult(text="time.", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        out = await engine.generate("the rest", max_tokens=16)
        assert out.text == " of time."
        assert out.finish_reason == "stop"
        assert out.finished is True
        captured = sys.modules["mlx_vlm.generate.diffusion"].__captured__["last"]  # type: ignore[attr-defined]
        ids = captured["input_ids"]
        assert ids.shape == (1, len("the rest"))

    @pytest.mark.asyncio
    async def test_stream_generate_honors_stop_sequences(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stop handling on the raw-prompt path must work the same as
        # the chat path — the shared helper is the only correct way
        # to guarantee that. Without delegation, ``stop`` would silently
        # no-op on /v1/completions.
        yields = [
            FakeGenerationResult(text="abc STOP tail", diffusion_block_complete=True),
            FakeGenerationResult(text="more", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        collected = []
        async for chunk in engine.stream_generate(
            "prefix", max_tokens=64, stop=["STOP"]
        ):
            collected.append(chunk)
        # First emitted chunk truncates at STOP and ends the stream.
        assert collected[0].new_text == "abc "
        assert collected[0].finish_reason == "stop"
        assert collected[0].finished is True
        assert len(collected) == 1

    @pytest.mark.asyncio
    async def test_stream_generate_accepts_single_string_stop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # codex pr_validate r7 BLOCKING #2: ``stop`` previously had
        # type ``list[str] | None`` and a static type-checker (or
        # strict-validation wrapper) would have rejected the OpenAI
        # single-string shape. ``_normalize_stops`` already handled
        # it internally, so the runtime worked — but the type
        # boundary lied. This test pins that the string form ALSO
        # truncates correctly end-to-end so a future re-tightening
        # of the signature trips here.
        yields = [
            FakeGenerationResult(text="abc STOP tail", diffusion_block_complete=True),
            FakeGenerationResult(text="more", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        collected = []
        async for chunk in engine.stream_generate(
            "prefix",
            max_tokens=64,
            stop="STOP",  # <-- the regression point (string, not list)
        ):
            collected.append(chunk)
        assert collected[0].new_text == "abc "
        assert collected[0].finish_reason == "stop"
        assert collected[0].finished is True
        assert len(collected) == 1

    @pytest.mark.asyncio
    async def test_generate_accepts_single_string_stop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Mirror of the stream_generate string-stop test for the
        # buffered ``generate`` path. ``generate`` delegates to
        # ``stream_generate`` so the type tightening cascades — but
        # an end-to-end pin guarantees the buffered surface honours
        # the OpenAI single-string shape too.
        yields = [
            FakeGenerationResult(text="abc STOP tail", diffusion_block_complete=True),
            FakeGenerationResult(text="more", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        out = await engine.generate("prefix", max_tokens=64, stop="STOP")
        assert out.text == "abc "
        assert out.finish_reason == "stop"
        assert out.finished is True


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
        # Codex round 1 [P1] + pr_validate r12 BLOCKING #2: this test
        # originally only asserted both requests returned two chunks,
        # which would stay green even if serialization were removed
        # entirely (the fake stream is replayed independently per
        # call). The strengthened version uses an entry/exit-counted
        # fake generator and asserts the in-flight count never exceeds
        # 1 — proving the engine-level _generation_lock actually
        # serialized the two concurrent calls.
        import asyncio as _aio
        import threading as _threading
        import time as _time

        # Track concurrent in-flight generator entries.
        in_flight = 0
        max_in_flight = 0
        in_flight_lock = _threading.Lock()

        def _counted_stream(*_a: Any, **_k: Any) -> Iterator[FakeGenerationResult]:
            nonlocal in_flight, max_in_flight
            with in_flight_lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            try:
                # Hold each generator alive long enough that a missing
                # lock would let the second one overlap.
                _time.sleep(0.05)
                yield FakeGenerationResult(text="x", diffusion_block_complete=True)
                _time.sleep(0.05)
                yield FakeGenerationResult(text="", finish_reason="stop")
            finally:
                with in_flight_lock:
                    in_flight -= 1

        _install_mlx_vlm_mock(monkeypatch, stream_yields=[])
        diffusion_mod = sys.modules["mlx_vlm.generate.diffusion"]
        diffusion_mod.stream_diffusion_generate = _counted_stream  # type: ignore[attr-defined]
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

        results = await _aio.wait_for(
            _aio.gather(drain(), drain()),
            timeout=10.0,
        )
        # Both requests completed without deadlock.
        assert all(len(r) == 2 for r in results), results
        # The strict serialization invariant: at no point did the
        # engine have two concurrent diffusion generators running.
        # If serialization regresses, max_in_flight would hit 2.
        assert max_in_flight == 1, (
            f"engine ran {max_in_flight} concurrent generators; "
            "_generation_lock failed to serialize"
        )

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

    def test_run_generator_cancel_check_at_top_skips_tokenization(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Codex round 7 [P2]: even after the worker-loop fast-skip,
        # cancel may flip BEFORE ``_run_generator`` tokenizes. Without
        # the top-of-function cancel-check, we'd materialize input_ids
        # + dispatch prefill before the per-iteration check fires.
        # We verify by setting cancel before invoking _run_generator
        # directly and asserting the tokenizer.encode was never called.
        import queue as _queue
        import threading as _threading

        encode_calls: list[str] = []
        invoked: list[bool] = []
        yields = [FakeGenerationResult(text="x", finish_reason="stop")]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        diffusion_mod = sys.modules["mlx_vlm.generate.diffusion"]

        def _tracker(*a: Any, **k: Any) -> Iterator[FakeGenerationResult]:
            invoked.append(True)
            yield from yields

        diffusion_mod.stream_diffusion_generate = _tracker  # type: ignore[attr-defined]
        from vllm_mlx.runtime.diffusion_lane import (
            DiffusionEngine,
            DiffusionGenerationConfig,
        )

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        # Wrap tokenizer.encode so we can detect whether it was hit.
        real_encode = engine._processor.tokenizer.encode

        def _tracking_encode(text: str) -> list[int]:
            encode_calls.append(text)
            return real_encode(text)

        engine._processor.tokenizer.encode = _tracking_encode  # type: ignore[method-assign]

        cancel_event = _threading.Event()
        cancel_event.set()
        thread_q: _queue.Queue[Any] = _queue.Queue()
        engine._run_generator(
            "prompt",
            16,
            DiffusionGenerationConfig(),
            thread_q,
            cancel_event,
        )
        # Top-of-function cancel-check must short-circuit before
        # encode runs.
        assert encode_calls == [], "Tokenizer hit despite pre-cancel"
        assert invoked == [], "Generator dispatched despite pre-cancel"

    def test_init_does_not_start_worker_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Codex round 11 [P2]: plain construction must NOT start the
        # worker thread, otherwise a contract test that instantiates
        # the engine with a bogus model would race against the
        # background loader's import + load. Lazy start is gated on
        # the first explicit call to start() / _load_blocking().
        # We verify with a deliberately bad model_name so that if
        # the worker DID start, _load_error would surface; instead
        # construction must complete cleanly and the worker stays
        # None.
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        # No mlx-vlm mock — we want to prove that init doesn't
        # trigger the worker (which would import mlx_vlm + load).
        engine = DiffusionEngine(model_name="mlx-community/whatever-bogus")
        assert engine._worker is None, "Worker started in __init__"
        assert engine._load_error is None, "Load attempted in __init__ (load_error set)"

    def test_supports_tool_calls_is_instance_level_gated_on_alias_parser(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # codex r1 BLOCKING #2: ``supports_tool_calls`` MUST be an
        # instance attribute gated on the resolved alias profile, not
        # a class-wide True. Without a known tool parser, the engine
        # cannot surface ``tool_calls`` no matter what the request
        # asked for — flipping the class default to True would let
        # ``tool_choice="required"`` slip past the route's
        # ``_engine_opts_out_of_tools`` gate and run a full canvas
        # generation that always ends in a plain-text fallback.
        _install_mlx_vlm_mock(monkeypatch)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        # Class default is conservative — bare HF paths with no alias
        # entry land here.
        assert DiffusionEngine.supports_tool_calls is False

        # Bare HF path that no alias references → still False (no
        # profile to consult).
        bare = DiffusionEngine(model_name="some/random-hf-path")
        assert bare.supports_tool_calls is False

        # Alias whose profile sets ``tool_call_parser="gemma4"`` opts
        # the instance into True. The diffusion-gemma-26b-4bit alias is
        # configured this way in aliases.json.
        gemma = DiffusionEngine(model_name="diffusion-gemma-26b-4bit")
        assert gemma.supports_tool_calls is True

    def test_build_skip_special_token_ids_carves_out_gemma4_wire_markers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # codex r1 BLOCKING #3: the skip_ids carve-out is the load-
        # bearing piece of the tool-call wire — if a gemma4-aliased
        # engine ever stops dropping the five marker ids from
        # ``skip_special_token_ids``, mlx-vlm's detokenizer strips
        # the call invocations and the parser sees plain prose.
        # Stand up a tokenizer that (a) returns five distinct marker
        # ids from ``all_special_ids`` and (b) encodes each wire
        # marker as a single token; assert the helper drops exactly
        # those five ids and leaves the other special ids intact.
        _install_mlx_vlm_mock(monkeypatch)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        # Five marker ids the helper must remove + three unrelated
        # special ids it must preserve.
        marker_ids = {
            46: "<|tool>",
            47: "<tool|>",
            48: "<|tool_call>",
            49: "<tool_call|>",
            52: '<|"|>',
        }
        unrelated_special_ids = {0, 1, 2}

        class _ToolAwareTokenizer:
            all_special_ids = list(unrelated_special_ids | set(marker_ids))

            def encode(
                self,
                text: str,
                add_special_tokens: bool = True,
            ) -> list[int]:
                # Look the text up in the marker table; tokens not in
                # the table get hashed to a non-special id so the
                # helper's len(token_ids)==1 guard still applies.
                for sid, marker in marker_ids.items():
                    if text == marker:
                        return [sid]
                return [hash(text) % 1000 + 1000]

        gemma = DiffusionEngine(model_name="diffusion-gemma-26b-4bit")
        assert gemma.supports_tool_calls is True

        # codex r2 BLOCKING #1: helper takes ``has_tools=True``
        # (request actually attached a tools array) — the carve-out
        # only fires under that flag. ``has_tools=False`` keeps the
        # markers in the skip set; see the dedicated test below.
        skip_ids = gemma._build_skip_special_token_ids(
            _ToolAwareTokenizer(),
            has_tools=True,
        )
        # The unrelated special ids stay in the skip set (still
        # filtered from detokenize output).
        assert unrelated_special_ids.issubset(skip_ids)
        # All five wire marker ids are dropped (so detokenizer
        # surfaces them and the parser can extract tool_calls).
        for sid in marker_ids:
            assert sid not in skip_ids, (
                f"marker id {sid} ({marker_ids[sid]!r}) was not carved out"
            )

    def test_build_skip_special_token_ids_keeps_markers_without_request_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # codex r2 BLOCKING #1: even on a gemma4-aliased engine, plain
        # non-tool chat requests MUST NOT carve the markers out of the
        # skip set. ``routes/chat.py`` only invokes the gemma4 text
        # parser when ``request.tools`` is set (line ~595), so a model
        # that spontaneously emits a ``<|tool_call>`` token in a plain
        # chat response would otherwise leak the raw wire characters
        # into the client's content stream without any parser to
        # interpret them.
        _install_mlx_vlm_mock(monkeypatch)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        marker_ids = {
            46: "<|tool>",
            47: "<tool|>",
            48: "<|tool_call>",
            49: "<tool_call|>",
            52: '<|"|>',
        }
        unrelated_special_ids = {0, 1, 2}

        class _ToolAwareTokenizer:
            all_special_ids = list(unrelated_special_ids | set(marker_ids))

            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                for sid, marker in marker_ids.items():
                    if text == marker:
                        return [sid]
                return [hash(text) % 1000 + 1000]

        gemma = DiffusionEngine(model_name="diffusion-gemma-26b-4bit")
        assert gemma.supports_tool_calls is True

        # ``has_tools`` defaults to False — non-tool requests take
        # this branch. Markers stay in the skip set.
        skip_ids = gemma._build_skip_special_token_ids(_ToolAwareTokenizer())
        assert set(marker_ids).issubset(skip_ids)
        assert unrelated_special_ids.issubset(skip_ids)
        # Explicit False (request-side gate) is the same as default.
        skip_ids_explicit = gemma._build_skip_special_token_ids(
            _ToolAwareTokenizer(),
            has_tools=False,
        )
        assert skip_ids == skip_ids_explicit

    @pytest.mark.asyncio
    async def test_stream_chat_suppresses_carveout_when_is_streaming_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pr_validate r8 BLOCKING #2 — the SSE path forwards each
        chunk as ``delta.content`` without running the tool parser,
        so leaving the gemma4 wire markers in (carve-out active)
        would surface raw ``<|tool_call>`` text to the client.
        Stream callers pass ``is_streaming=True`` to disable the
        carve-out; non-stream buffering callers (default
        ``is_streaming=False``) keep markers so the post-canvas
        parser can extract structured ``tool_calls``.
        """
        _install_mlx_vlm_mock(monkeypatch)
        from vllm_mlx.runtime.diffusion_lane import (
            DiffusionEngine,
            DiffusionGenerationConfig,
        )

        gemma = DiffusionEngine(model_name="diffusion-gemma-26b-4bit")
        assert gemma.supports_tool_calls is True

        captured: list[DiffusionGenerationConfig] = []

        async def _capture_raw(*args, **kwargs):  # noqa: ARG001
            # Stash the cfg the engine built so we can assert on its
            # ``has_tools`` flag — this is the bit that gates the
            # downstream ``_build_skip_special_token_ids`` carve-out.
            captured.append(
                DiffusionGenerationConfig(
                    diffusion_steps=None,
                    temperature=0.0,
                    has_tools=kwargs.get("has_tools", False),
                )
            )
            if False:  # never yields — we only need the cfg snapshot
                yield None

        monkeypatch.setattr(gemma, "_stream_prompt_raw", _capture_raw)
        # Also stub build_prompt so we don't need a real tokenizer
        # roundtrip; the cfg is what we care about.
        monkeypatch.setattr(gemma, "build_prompt", lambda *a, **kw: "prompt")
        # Bypass the load-state guard — the mock doesn't run a real
        # start() and we're only testing the cfg construction path.
        monkeypatch.setattr(gemma, "_ensure_loaded", lambda: None)

        # Stream path — has_tools must be False even though tools are
        # attached and the engine supports them.
        async for _ in gemma.stream_chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            is_streaming=True,
        ):
            pass
        assert captured, "engine never built a cfg"
        assert captured[-1].has_tools is False, (
            "is_streaming=True must suppress the marker carve-out — "
            "leaving has_tools True would leak raw <|tool_call> text "
            "into SSE delta.content"
        )

        # Non-stream buffering path — has_tools must be True so the
        # post-canvas parser receives the markers.
        captured.clear()
        async for _ in gemma.stream_chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            is_streaming=False,
        ):
            pass
        assert captured, "engine never built a cfg on non-stream path"
        assert captured[-1].has_tools is True

    def test_build_skip_special_token_ids_keeps_markers_without_parser(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Negative control for codex r1 BLOCKING #3: a bare HF path
        # whose alias has no tool parser MUST keep the markers in
        # the skip set — there's no downstream parser to receive
        # them, and surfacing raw ``<|tool_call>`` tokens to plain
        # chat clients would corrupt the conversation transcript.
        #
        # codex r2 BLOCKING #4 hardening: the tokenizer mirrors the
        # positive case's marker mapping (encode resolves each wire
        # marker string back to its single-token id) so this
        # assertion would actually catch a regression where the parser
        # gate was accidentally removed. Without the mapping, the
        # helper's encode-then-check-single-id guard would never fire
        # and a broken implementation could pass anyway.
        _install_mlx_vlm_mock(monkeypatch)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        marker_ids = {
            46: "<|tool>",
            47: "<tool|>",
            48: "<|tool_call>",
            49: "<tool_call|>",
            52: '<|"|>',
        }
        unrelated_special_ids = {0, 1, 2}

        class _BareTokenizer:
            all_special_ids = list(unrelated_special_ids | set(marker_ids))

            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                # Mirror the positive case: marker strings resolve to
                # their single-token id; everything else hashes to a
                # non-special id.
                for sid, marker in marker_ids.items():
                    if text == marker:
                        return [sid]
                return [hash(text) % 1000 + 1000]

        # Bare HF path → no profile → supports_tool_calls False →
        # helper does NOT carve out anything regardless of tokenizer
        # cooperativeness.
        bare = DiffusionEngine(model_name="some/no-alias-hf-path")
        assert bare.supports_tool_calls is False

        # Strong assertion: even passing ``has_tools=True`` (simulating
        # an upstream that forgot to gate on the engine's
        # ``supports_tool_calls`` flag) must NOT carve markers out of
        # a parser-less profile. The bare HF path has no parser, so
        # there's no downstream consumer of the raw markers; surfacing
        # them would only damage the conversation transcript.
        skip_ids = bare._build_skip_special_token_ids(
            _BareTokenizer(),
            has_tools=True,
        )
        # All special ids (including the marker ids) stay skipped —
        # confirmed even when the tokenizer encode roundtrip would
        # have let the carve-out fire under a broken implementation.
        assert set(marker_ids).issubset(skip_ids)
        assert unrelated_special_ids.issubset(skip_ids)

    def test_engine_opts_out_blocks_tool_choice_required_even_with_parser(
        self,
    ) -> None:
        # Codex round 10 [P2] + pr_validate r11 BLOCKING #2: even with
        # a global --tool-call-parser configured, the route's
        # streaming-required gate must reject tool_choice="required"
        # + stream=true on an engine that has opted out of tool calls.
        # The previous version of this test only asserted a local
        # getattr expression — pr_validate r11 flagged that it would
        # stay green if the route gate were silently deleted. This
        # version fires a real HTTP request through the chat router
        # so the gate is exercised end-to-end.
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_mlx.config import reset_config
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.routes.chat import router as chat_router

        class _DiffusionEngineStub:
            supports_tool_calls = False
            preserve_native_tool_format = False
            is_mllm = False
            supports_guided_generation = False
            tokenizer = None

            def build_prompt(self, messages, tools=None, enable_thinking=None):
                return "PROMPT"

            async def chat(self, messages, **kwargs):
                return GenerationOutput(
                    text="should-not-be-reached",
                    raw_text="",
                    prompt_tokens=1,
                    completion_tokens=1,
                    finished=True,
                    finish_reason="stop",
                )

            async def stream_chat(self, messages, **kwargs):
                yield GenerationOutput(
                    text="should-not-be-reached",
                    new_text="should-not-be-reached",
                    finished=True,
                    finish_reason="stop",
                )

        cfg = reset_config()
        cfg.engine = _DiffusionEngineStub()
        cfg.model_name = "diffusion-gemma-26b-4bit"
        cfg.model_registry = None
        cfg.no_thinking = True
        # Critical: parser IS configured. Without the
        # ``supports_tool_calls=False`` veto, the gate would let the
        # request through because ``cfg.tool_call_parser`` is truthy.
        cfg.tool_call_parser = "hermes"

        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "diffusion-gemma-26b-4bit",
                "stream": True,
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": "required",
                "max_tokens": 32,
            },
        )
        assert resp.status_code == 422, resp.text
        body = resp.text.lower()
        assert "tool" in body and "required" in body

    def test_engine_opts_out_blocks_named_function_tool_choice(
        self,
    ) -> None:
        # codex pr_validate r8 NIT #2: the opted-out engine veto
        # previously only fired for ``tool_choice="required"``,
        # leaving the named-function shape
        # (``{"type":"function","function":{"name":"foo"}}``)
        # unprotected. That form is ALSO a forced contract — the
        # caller demands a specific tool be called — and an
        # engine that has opted out cannot satisfy it either.
        # Without this gate, named tool_choice on a diffusion
        # engine would run a full generation, return plain text,
        # and surface as a confusing post-parse 422.
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_mlx.config import reset_config
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.routes.chat import router as chat_router

        class _DiffusionEngineStub:
            supports_tool_calls = False
            preserve_native_tool_format = False
            is_mllm = False
            supports_guided_generation = False
            tokenizer = None

            def build_prompt(self, messages, tools=None, enable_thinking=None):
                return "PROMPT"

            async def chat(self, messages, **kwargs):
                raise RuntimeError(
                    "engine.chat() executed despite supports_tool_calls="
                    "False and a forced named tool_choice; named-tool veto "
                    "regressed"
                )

            async def stream_chat(self, messages, **kwargs):
                yield GenerationOutput(text="x", finished=True)

        cfg = reset_config()
        cfg.engine = _DiffusionEngineStub()
        cfg.model_name = "diffusion-gemma-26b-4bit"
        cfg.model_registry = None
        cfg.no_thinking = True
        cfg.tool_call_parser = "hermes"

        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "diffusion-gemma-26b-4bit",
                "stream": False,
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
                "max_tokens": 32,
            },
        )
        assert resp.status_code == 422, resp.text
        body = resp.text.lower()
        assert "tool_choice" in body and "forces" in body

    def test_engine_opts_out_blocks_legacy_function_literal_tool_choice(
        self,
    ) -> None:
        # codex pr_validate r9 NIT #1: the pre-fix predicate matched
        # ``tc == "required"`` and the dict-shape named-function
        # form but skipped the LEGACY bare-string ``"function"``
        # literal that some pre-2024 OpenAI SDKs emit to mean "force
        # any function call". An opted-out engine couldn't satisfy
        # it either, but the upfront veto missed it.
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_mlx.config import reset_config
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.routes.chat import router as chat_router

        class _DiffusionEngineStub:
            supports_tool_calls = False
            preserve_native_tool_format = False
            is_mllm = False
            supports_guided_generation = False
            tokenizer = None

            def build_prompt(self, messages, tools=None, enable_thinking=None):
                return "PROMPT"

            async def chat(self, messages, **kwargs):
                raise RuntimeError(
                    "engine.chat() executed despite supports_tool_calls="
                    'False and a legacy tool_choice="function" literal; '
                    "legacy-literal veto regressed"
                )

            async def stream_chat(self, messages, **kwargs):
                yield GenerationOutput(text="x", finished=True)

        cfg = reset_config()
        cfg.engine = _DiffusionEngineStub()
        cfg.model_name = "diffusion-gemma-26b-4bit"
        cfg.model_registry = None
        cfg.no_thinking = True
        cfg.tool_call_parser = "hermes"

        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "diffusion-gemma-26b-4bit",
                "stream": False,
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": "function",  # <-- the legacy literal
                "max_tokens": 32,
            },
        )
        assert resp.status_code == 422, resp.text
        body = resp.text.lower()
        assert "tool_choice" in body and "forces" in body

    def test_engine_opts_out_blocks_tool_choice_required_non_stream_too(
        self,
    ) -> None:
        # codex pr_validate r6 BLOCKING #1: the previous engine-level
        # veto was nested inside the ``request.stream`` branch, so
        # ``tool_choice="required"`` non-stream requests still ran a
        # full diffusion generation and only failed in the
        # post-parse 422 gate at line ~1101. This test pins the
        # upfront rejection for the non-stream flow — the request
        # MUST 422 BEFORE the engine.chat() call would have been made
        # (verified by the ``raise RuntimeError`` stub: if the chat
        # method ever executed, the test would explode).
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_mlx.config import reset_config
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.routes.chat import router as chat_router

        class _DiffusionEngineStub:
            supports_tool_calls = False
            preserve_native_tool_format = False
            is_mllm = False
            supports_guided_generation = False
            tokenizer = None

            def build_prompt(self, messages, tools=None, enable_thinking=None):
                return "PROMPT"

            async def chat(self, messages, **kwargs):
                # Reaching here means the upfront veto did NOT fire —
                # the request would have consumed GPU before failing.
                raise RuntimeError(
                    "engine.chat() executed despite "
                    "supports_tool_calls=False and tool_choice=required; "
                    "the non-stream veto regressed"
                )

            async def stream_chat(self, messages, **kwargs):
                yield GenerationOutput(text="should-not-be-reached", finished=True)

        cfg = reset_config()
        cfg.engine = _DiffusionEngineStub()
        cfg.model_name = "diffusion-gemma-26b-4bit"
        cfg.model_registry = None
        cfg.no_thinking = True
        cfg.tool_call_parser = "hermes"

        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "diffusion-gemma-26b-4bit",
                "stream": False,  # <-- the regression point
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": "required",
                "max_tokens": 32,
            },
        )
        assert resp.status_code == 422, resp.text
        body = resp.text.lower()
        # The new error mentions the opt-out reason, not the streaming
        # parser language.
        assert "opted out" in body or "supports_tool_calls" in body, body

    def test_route_probe_rejects_engine_when_supports_tool_calls_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end pin: the probe function in routes/chat.py must
        # short-circuit to False for any engine whose
        # supports_tool_calls attribute is False, regardless of
        # tokenizer shape.
        _install_mlx_vlm_mock(monkeypatch)
        from vllm_mlx.routes.chat import (
            _engine_supports_channel_routed_tool_calls,
        )
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        assert _engine_supports_channel_routed_tool_calls(engine) is False

    @pytest.mark.asyncio
    async def test_post_lock_stuck_check_rejects_in_flight_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Codex round 8 [P2] + pr_validate r12 BLOCKING #1: a request
        # that passed admission BEFORE the engine was marked stuck
        # (e.g. another request tripped the 30 s drain timeout while
        # this one was waiting on the lock) must NOT enqueue work to
        # the wedged worker.
        #
        # The previous version of this test flipped ``_worker_stuck``
        # BEFORE calling stream_chat, so a pre-lock admission check
        # would satisfy it. The strengthened version mirrors the
        # actual production race:
        #   1. Test pre-acquires the engine's _generation_lock.
        #   2. The request enters stream_chat and BLOCKS on the lock.
        #   3. While it's blocked, we flip ``_worker_stuck``.
        #   4. We release the lock; the request acquires it and the
        #      post-lock gate MUST raise.
        # This way a regression that moved the check pre-lock would
        # break: at the moment we entered stream_chat, the flag was
        # still False.
        import asyncio as _aio

        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine
        from vllm_mlx.scheduler import BackpressureError

        _install_mlx_vlm_mock(monkeypatch)
        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()

        # Hold the lock so the request blocks waiting for it.
        await engine._generation_lock.acquire()
        assert engine._worker_stuck is False, (
            "_worker_stuck must be False at the moment the request "
            "enters stream_chat — otherwise this test degenerates "
            "into the pre-lock check it's trying to disprove"
        )

        captured: dict[str, BaseException] = {}

        async def queued_request() -> None:
            try:
                async for _ in engine.stream_chat(
                    [{"role": "user", "content": "go"}], max_tokens=16
                ):
                    pass
            except BaseException as e:  # noqa: BLE001 — capture for assertion
                captured["err"] = e

        task = _aio.create_task(queued_request())
        # Give the task time to enter stream_chat and reach the
        # ``async with self._generation_lock:`` await.
        await _aio.sleep(0.05)
        assert not task.done(), "request should be blocked on the lock"

        # Now flip the stuck flag — proving the check fires AFTER
        # the lock was acquired, not as a pre-lock gate.
        engine._worker_stuck = True

        # Release the lock; the request acquires it, post-lock gate
        # fires, BackpressureError raised.
        engine._generation_lock.release()
        await _aio.wait_for(task, timeout=2.0)

        err = captured.get("err")
        assert err is not None, "request completed without raising"
        assert isinstance(err, BackpressureError), (
            f"expected BackpressureError, got {type(err).__name__}: {err}"
        )
        assert "unhealthy" in str(err).lower()

    @pytest.mark.asyncio
    async def test_worker_stuck_marks_admission_unhealthy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Codex round 7 [P2]: when the done_event drain ceiling fires,
        # the engine must refuse subsequent admissions — otherwise the
        # next request rides onto a worker still burning GPU on the
        # abandoned job. We simulate by setting _worker_stuck directly
        # and verifying check_admission raises immediately.
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine
        from vllm_mlx.scheduler import BackpressureError, SchedulerConfig

        _install_mlx_vlm_mock(monkeypatch)
        engine = DiffusionEngine(
            model_name="x/y",
            scheduler_config=SchedulerConfig(max_concurrent_requests=2),
        )
        engine._load_blocking()
        # Healthy path — capacity available.
        engine.check_admission()
        engine.release_admission_reservation()
        # Flip stuck and verify check_admission refuses even at zero
        # in-flight reservations.
        engine._worker_stuck = True
        with pytest.raises(BackpressureError, match="marked unhealthy"):
            engine.check_admission()

    @pytest.mark.asyncio
    async def test_worker_fast_skips_pre_cancelled_jobs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Codex round 6 [P2]: jobs cancelled BEFORE the worker picked
        # them up (e.g. caller disconnected while still queued behind
        # a slower request) must not run a single diffusion step.
        # We verify by pre-cancelling a job's cancel_event and
        # confirming stream_diffusion_generate was NEVER called.
        import queue as _queue
        import threading as _threading

        invoked: list[bool] = []
        yields = [FakeGenerationResult(text="should not see", finish_reason="stop")]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        diffusion_mod = sys.modules["mlx_vlm.generate.diffusion"]
        real_stream = diffusion_mod.stream_diffusion_generate

        def _stream_tracker(*a: Any, **k: Any) -> Iterator[FakeGenerationResult]:
            invoked.append(True)
            return real_stream(*a, **k)

        diffusion_mod.stream_diffusion_generate = _stream_tracker  # type: ignore[attr-defined]
        from vllm_mlx.runtime.diffusion_lane import (
            DiffusionEngine,
            DiffusionGenerationConfig,
        )

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()

        # Build a job tuple directly and put it on the engine's queue
        # with cancel_event PRE-set. The worker should pull it, see
        # cancel set, and fast-skip without ever invoking the generator.
        thread_q: _queue.Queue[Any] = _queue.Queue()
        cancel_event = _threading.Event()
        cancel_event.set()  # PRE-cancelled.
        done_event = _threading.Event()
        engine._jobs.put(
            (
                "prompt",
                16,
                DiffusionGenerationConfig(),
                thread_q,
                cancel_event,
                done_event,
            )
        )
        # Wait for the worker to handle the job.
        assert done_event.wait(timeout=2.0), "Worker never finished pre-cancelled job"
        # stream_diffusion_generate must NOT have been called.
        assert invoked == [], "Worker ran generator despite pre-cancel"

    @pytest.mark.asyncio
    async def test_lock_held_until_worker_finishes_cancelled_job(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Codex round 6 [P2] + pr_validate r11 BLOCKING #3: if
        # stream_chat releases the lock on its own consumer exit
        # (early stop / disconnect) before the worker has actually
        # observed cancel_event, a queued sibling acquires the lock
        # while the worker is still burning GPU on the cancelled job
        # — head-of-line blocking. The done_event contract pins this:
        # the worker's job-finally MUST set it AFTER ``_run_generator``
        # returns, and stream_chat's finally MUST await it BEFORE the
        # ``async with`` exits.
        #
        # The previous version of this test only asserted "ticks
        # fired and req1 released" — that would stay green even if
        # the regression came back. This version records the actual
        # ordering of:
        #   t_req1_done   — when req1's worker_loop set its done_event
        #   t_req2_acquire — when req2 actually entered its
        #                    _run_generator (proves it acquired the
        #                    lock AND the worker picked its job)
        # and asserts t_req2_acquire >= t_req1_done so the regression
        # is genuinely pinned.
        import asyncio as _aio
        import time as _time

        def slow_yields() -> Iterator[FakeGenerationResult]:
            for text in ("first ", "second", " third"):
                _time.sleep(0.05)
                yield FakeGenerationResult(text=text, diffusion_block_complete=True)
            yield FakeGenerationResult(text="", finish_reason="stop")

        _install_mlx_vlm_mock(monkeypatch, stream_yields=[])
        diffusion_mod = sys.modules["mlx_vlm.generate.diffusion"]

        def _slow_stream(*_a: Any, **_k: Any) -> Iterator[FakeGenerationResult]:
            return slow_yields()

        diffusion_mod.stream_diffusion_generate = _slow_stream  # type: ignore[attr-defined]
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()

        # Patch the worker's job loop to capture done_event timestamps
        # AND the moment each job actually entered _run_generator
        # (i.e. the worker picked it up after the lock was released).
        prompt_to_done_ts: dict[str, float] = {}
        prompt_to_run_ts: dict[str, float] = {}
        real_run_generator = engine._run_generator

        def _instrumented_run_generator(
            prompt: str,
            max_tokens: int,
            cfg: Any,
            out_q: Any,
            cancel_event: threading.Event,
        ) -> None:
            prompt_to_run_ts[prompt] = _time.monotonic()
            real_run_generator(prompt, max_tokens, cfg, out_q, cancel_event)

        engine._run_generator = _instrumented_run_generator  # type: ignore[method-assign]

        # Wrap done_event.set so we can record req1's worker-release
        # timestamp directly from the source of truth.
        import queue as _queue

        real_jobs_put = engine._jobs.put

        def _instrumented_put(job: Any) -> None:
            if isinstance(job, tuple) and len(job) == 6:
                prompt, max_tokens, cfg, thread_q, cancel_event, done_event = job
                real_set = done_event.set

                def _wrapped_set() -> None:
                    prompt_to_done_ts[prompt] = _time.monotonic()
                    real_set()

                done_event.set = _wrapped_set  # type: ignore[method-assign]
            real_jobs_put(job)

        engine._jobs.put = _instrumented_put  # type: ignore[method-assign]
        # ``put`` is normally engaged via Queue.put; the patched bound
        # ref is what callers will hit (Queue methods are bound
        # attributes on the instance after ``__init__``).
        assert isinstance(engine._jobs, _queue.Queue)

        # We use distinct prompts so each shows up uniquely in the
        # instrument dicts.
        req1_prompt_marker = "REQ1_GO"
        req2_prompt_marker = "REQ2_NEXT"

        async def req1() -> None:
            async for _ in engine.stream_chat(
                [{"role": "user", "content": req1_prompt_marker}],
                max_tokens=64,
                stop=["first"],
            ):
                pass

        async def req2() -> None:
            async for _ in engine.stream_chat(
                [{"role": "user", "content": req2_prompt_marker}],
                max_tokens=16,
            ):
                pass

        t1 = _aio.create_task(req1())
        # Give req1 a moment to enter stream_chat and acquire the
        # generation lock before req2 starts queueing for it.
        await _aio.sleep(0.02)
        t2 = _aio.create_task(req2())
        await _aio.wait_for(_aio.gather(t1, t2), timeout=15.0)

        # Map back from FakeTokenizer.apply_chat_template's rendered
        # prompt to find the matching dict key. The fake template
        # joins messages with \n and appends "<start_of_turn>model\n",
        # so the marker substring identifies the job.
        def _find(prompt_marker: str, store: dict[str, float]) -> float:
            for prompt, ts in store.items():
                if prompt_marker in prompt:
                    return ts
            raise AssertionError(
                f"no timestamp recorded for marker {prompt_marker!r} in {store}"
            )

        t_req1_done = _find(req1_prompt_marker, prompt_to_done_ts)
        t_req2_acquire = _find(req2_prompt_marker, prompt_to_run_ts)
        # The whole point: req2 cannot have started its worker
        # iteration BEFORE req1's done_event was set, because the
        # lock-release happens AFTER awaiting done_event. If a
        # regression brings back early lock-release, this strict
        # ordering breaks.
        assert t_req2_acquire >= t_req1_done, (
            f"req2 picked up by worker at t={t_req2_acquire:.4f}, "
            f"BEFORE req1's done_event fired at t={t_req1_done:.4f} — "
            "head-of-line block regression"
        )


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


class TestMlxVlmImportContract:
    """pr_validate codex r13 BLOCKING: every test above this point
    replaces ``mlx_vlm.generate.diffusion`` with a synthetic module
    via monkeypatch, so those tests would silently still pass if a
    future mlx-vlm release renamed or removed the symbols the
    runtime imports. These tests deliberately do NOT install the
    mock — they bind against the installed mlx-vlm package's real
    surface, so any upstream rename trips the test suite at the
    next ``pip install -e .`` cycle.

    These tests skip cleanly if mlx-vlm is not installed (CI lanes
    without Metal). With mlx-vlm == 0.6.3 they MUST find the
    runtime-imported symbols at the documented paths.
    """

    def test_load_symbol_exists_in_installed_mlx_vlm(self) -> None:
        pytest.importorskip("mlx_vlm")
        from mlx_vlm.utils import load

        # Callable check is enough — signature varies across upstream
        # versions but rapid-mlx only invokes with the HF-path arg.
        assert callable(load)

    def test_diffusion_generation_family_exists_in_installed_mlx_vlm(self) -> None:
        pytest.importorskip("mlx_vlm")
        from mlx_vlm.generate.diffusion import diffusion_generation_family

        assert callable(diffusion_generation_family)

    def test_stream_diffusion_generate_exists_in_installed_mlx_vlm(self) -> None:
        pytest.importorskip("mlx_vlm")
        from mlx_vlm.generate.diffusion import stream_diffusion_generate

        # Generator factory — callable, not the iterator type.
        assert callable(stream_diffusion_generate)

    def test_runtime_imports_match_installed_surface(self) -> None:
        # Pin the exact import paths that diffusion_lane.py uses at
        # request time. A future mlx-vlm release that moves these
        # symbols (e.g. into a different submodule) would break the
        # production path; this test would break first.
        pytest.importorskip("mlx_vlm")
        import importlib

        # Match diffusion_lane.py:_worker_loop imports verbatim.
        importlib.import_module("mlx_vlm.utils")
        importlib.import_module("mlx_vlm.generate.diffusion")

        # And the per-request imports inside _run_generator.
        gen_diff = importlib.import_module("mlx_vlm.generate.diffusion")
        for symbol in ("diffusion_generation_family", "stream_diffusion_generate"):
            assert hasattr(gen_diff, symbol), (
                f"mlx_vlm.generate.diffusion.{symbol} missing — "
                "diffusion_lane.py would fail at request time"
            )

    def test_stopping_criteria_reset_accepts_scalar_eos_id(self) -> None:
        # codex pr_validate r7 BLOCKING #3 (FALSE positive): codex
        # flagged ``tokenizer.stopping_criteria.reset(eos_id)`` as
        # passing a scalar where mlx-vlm allegedly wanted a list.
        # The mlx-vlm contract (utils.py:1921 in 0.6.3 install) is:
        #
        #     def reset(self, eos_token_ids: List[int] = None):
        #         ...
        #         if isinstance(eos_token_ids, int):
        #             eos_token_ids = [eos_token_ids]
        #
        # i.e. the upstream method explicitly normalises scalar →
        # list, and mlx-vlm's own server (server/generation.py:1412,
        # :1461; generate/dispatch.py:1332) calls it with a scalar.
        # We follow that pattern exactly. This test pins the
        # upstream contract so a future mlx-vlm release that drops
        # the scalar normalisation would trip here BEFORE the
        # production path crashes on a real DiffusionGemma request.
        pytest.importorskip("mlx_vlm")
        from mlx_vlm.utils import StoppingCriteria

        # Build a fake tokenizer just enough for StoppingCriteria's
        # init contract. We're not exercising encoding — only the
        # scalar-tolerance of reset().
        class _StubTok:
            eos_token_ids = [3]

            def encode(self, text: str, add_special_tokens: bool = False):
                return [0]

        sc = StoppingCriteria(eos_token_ids=[7], tokenizer=_StubTok())
        # The exact call shape diffusion_lane.py:1007 uses.
        sc.reset(42)
        # After the scalar → list normalisation, the new list MUST
        # contain only the value we passed.
        assert sc.eos_token_ids == [42], (
            f"mlx-vlm StoppingCriteria.reset(scalar) no longer "
            f"normalises to a single-item list; eos_token_ids="
            f"{sc.eos_token_ids}. diffusion_lane.py:1007 needs to "
            "update to match the new contract."
        )


class TestAliasIntegration:
    def test_diffusion_gemma_alias_resolves_to_text_diffusion_modality(
        self,
    ) -> None:
        # End-to-end pin on the actual aliases.json entry — if a
        # future edit drops the modality field, this test catches it
        # before the server boot does.
        from vllm_mlx.model_aliases import resolve_profile

        profile = resolve_profile("diffusion-gemma-26b-4bit")
        assert profile is not None
        assert profile.modality == "text-diffusion"
        assert profile.supports_spec_decode is False
        assert profile.supports_dflash is False
        assert profile.hf_path == "mlx-community/diffusiongemma-26B-A4B-it-4bit"
        assert profile.tool_call_parser == "gemma4"


class TestStopRace:
    """pr_validate r5 BLOCKING: ``stop()`` was push-sentinel-then-
    join(5s), which left an in-flight ``_run_generator`` running
    (its cancel_event was never set) AND then cleared ``_model`` /
    ``_processor`` even when the worker was still alive — recipe
    for an mx.eval crash mid-iteration during lifespan shutdown.

    The fix tracks the worker's currently-installed cancel event
    in ``_active_cancel`` and signals it from ``stop()`` BEFORE
    pushing the sentinel; the model-state clear is gated on the
    worker actually exiting (join timeout returns false → leave
    refs intact so GC reclaims them later).
    """

    @pytest.mark.asyncio
    async def test_stop_signals_active_cancel_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Long-running stream that only ends when cancel_event is
        # set. If stop() forgets to signal the active job, the
        # worker is still inside stream_diffusion_generate and the
        # test will time out at gather().
        import asyncio as _aio
        import threading as _threading
        import time as _time

        active_cancel: dict[str, threading.Event | None] = {"e": None}
        observed_engine: dict[str, Any] = {}

        def _long_stream(*_a: Any, **_k: Any) -> Iterator[FakeGenerationResult]:
            # Capture the engine's active cancel handle the moment
            # the worker reaches into stream_diffusion_generate.
            eng = observed_engine.get("eng")
            assert eng is not None
            active_cancel["e"] = eng._active_cancel
            for _ in range(1000):
                if eng._active_cancel is not None and eng._active_cancel.is_set():
                    return
                _time.sleep(0.01)
                yield FakeGenerationResult(text="tick ", diffusion_block_complete=True)
            yield FakeGenerationResult(text="", finish_reason="stop")

        _install_mlx_vlm_mock(monkeypatch, stream_yields=[])
        diff_mod = sys.modules["mlx_vlm.generate.diffusion"]
        diff_mod.stream_diffusion_generate = _long_stream  # type: ignore[attr-defined]
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        observed_engine["eng"] = engine

        # Kick off a long stream; let it actually enter the generator
        # before we call stop() so _active_cancel is populated.
        ready = _threading.Event()

        async def consumer() -> None:
            async for out in engine.stream_chat(
                [{"role": "user", "content": "loop forever"}],
                max_tokens=10000,
            ):
                if out.new_text and not ready.is_set():
                    ready.set()

        consume_task = _aio.create_task(consumer())
        # Wait until the worker is inside the generator (we've seen
        # at least one streamed chunk).
        for _ in range(200):
            if ready.is_set() and engine._active_cancel is not None:
                break
            await _aio.sleep(0.01)
        assert engine._active_cancel is not None, (
            "stop()-race fix relies on the worker installing "
            "_active_cancel BEFORE entering stream_diffusion_generate; "
            "the worker never published it within 2 s"
        )
        captured_cancel = engine._active_cancel

        # stop() must signal the captured cancel event so the
        # in-flight generator exits at the next per-chunk check.
        await _aio.wait_for(engine.stop(), timeout=10.0)
        assert captured_cancel.is_set(), (
            "stop() did not signal the active job's cancel_event; "
            "in-flight diffusion would have kept burning GPU until "
            "max_tokens"
        )

        # Drain the consumer; cancellation should land normally.
        try:
            await _aio.wait_for(consume_task, timeout=5.0)
        except (_aio.CancelledError, Exception):
            pass

    @pytest.mark.asyncio
    async def test_stop_waits_for_worker_before_clearing_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``stop()`` must NOT null _model / _processor while the
        # worker thread is mid-eval. Pre-fix code joined for 5 s
        # then cleared unconditionally; post-fix code joins 30 s
        # and skips the clear if the worker is still alive. On
        # clean shutdown the worker bookkeeping is also reset so
        # a subsequent ``_load_blocking`` can restart cleanly
        # (codex pr_validate r6 NIT).
        _install_mlx_vlm_mock(monkeypatch, stream_yields=[])
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        assert engine._loaded is True
        original_model = engine._model
        original_processor = engine._processor
        assert original_model is not None
        assert original_processor is not None
        worker_before = engine._worker
        assert worker_before is not None

        # Worker is parked on queue.get(); stop() pushes sentinel
        # and the worker exits its loop. After join returns, the
        # model refs MUST be cleared (no in-flight job to protect)
        # AND the worker bookkeeping MUST be reset so a subsequent
        # restart spawns a fresh worker.
        await engine.stop()
        assert engine._loaded is False
        assert engine._model is None
        assert engine._processor is None
        assert engine._worker is None, (
            "stop() must reset _worker to None on clean shutdown so "
            "_start_worker_once is willing to spawn a fresh worker; "
            "codex pr_validate r6 NIT"
        )
        assert engine._stop is False, "stop() must clear _stop for restart"
        assert engine._active_cancel is None

    @pytest.mark.asyncio
    async def test_engine_can_restart_after_clean_stop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # codex pr_validate r6 NIT: after a clean ``stop()``,
        # ``_load_blocking()`` MUST be able to spin up a fresh
        # worker. Pre-fix code left ``_worker`` non-None so
        # ``_start_worker_once`` no-op'd and the engine remained
        # permanently un-loaded after a lifecycle restart.
        _install_mlx_vlm_mock(monkeypatch, stream_yields=[])
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        worker_first = engine._worker
        assert worker_first is not None
        assert engine._loaded is True

        await engine.stop()
        assert engine._loaded is False
        assert engine._worker is None

        # Second load must succeed and produce a NEW worker thread —
        # not the dead one from before.
        engine._load_blocking()
        assert engine._loaded is True
        assert engine._worker is not None
        assert engine._worker is not worker_first, (
            "expected a fresh worker thread after restart; got the "
            "same (dead) instance — _start_worker_once likely no-op'd"
        )
        # Sanity: the fresh worker is actually alive.
        assert engine._worker.is_alive() is True

        # Cleanup so the test doesn't leak a daemon thread.
        await engine.stop()

    @pytest.mark.asyncio
    async def test_stop_defers_model_clear_when_worker_wedged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If the worker can't be unwedged within the 30 s join
        # ceiling, stop() MUST leave model refs intact so the
        # worker doesn't crash inside mx.eval on a None model.
        # We simulate the wedge by monkey-patching worker.join to
        # return immediately AND is_alive to return True forever.
        import asyncio as _aio

        _install_mlx_vlm_mock(monkeypatch, stream_yields=[])
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        original_model = engine._model
        original_processor = engine._processor

        class _WedgedThread:
            def __init__(self, real: threading.Thread) -> None:
                self._real = real

            def join(self, timeout: float | None = None) -> None:
                # No-op — pretend join expired without thread exit.
                return None

            def is_alive(self) -> bool:
                return True

        engine._worker = _WedgedThread(engine._worker)  # type: ignore[assignment]

        await _aio.wait_for(engine.stop(), timeout=5.0)
        # Wedge branch: model + processor remain referenced so the
        # orphaned worker can finish its mx.eval without exploding.
        assert engine._model is original_model
        assert engine._processor is original_processor

    @pytest.mark.asyncio
    async def test_stop_drains_queue_so_restart_does_not_pick_stale_sentinel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # codex pr_validate r8 BLOCKING #2: ``stop()`` always pushes
        # a ``None`` sentinel. If a previous ``stop()`` had pushed
        # but the worker exited on its ``while not self._stop``
        # check WITHOUT consuming the sentinel, the stale ``None``
        # sits in ``_jobs``. The next ``_load_blocking()`` would
        # spawn a fresh worker that immediately pulls the stale
        # sentinel and returns at ``if job is None: return`` — the
        # engine reports ``_loaded = True`` but the worker is dead.
        # Pre-fix code did not drain the queue.
        import time as _time

        _install_mlx_vlm_mock(monkeypatch, stream_yields=[])
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        # Simulate "stale sentinel from a previous shutdown": push
        # a ``None`` directly while the worker is parked, then call
        # stop(). The first None unblocks the worker (clean exit);
        # the second None (pushed by stop() itself) would be the
        # stale one — stop() MUST drain it.
        engine._jobs.put(None)
        await engine.stop()
        # The queue MUST be empty so restart can't pick anything up.
        assert engine._jobs.empty(), (
            f"stop() left {engine._jobs.qsize()} stale item(s) in _jobs; "
            "next restart's worker would consume one and exit immediately"
        )

        # Restart and prove the new worker survives past its first
        # _jobs.get() — i.e. it's BLOCKED on the empty queue, not
        # dead from a stale None.
        engine._load_blocking()
        assert engine._loaded is True
        new_worker = engine._worker
        assert new_worker is not None and new_worker.is_alive() is True
        # Give it a moment so a "dead on first iteration" worker
        # has time to actually die before we re-check.
        _time.sleep(0.1)
        assert new_worker.is_alive() is True, (
            "fresh worker died on first iteration — a stale sentinel "
            "must have leaked through stop()'s queue drain"
        )

        # Cleanup.
        await engine.stop()

    @pytest.mark.asyncio
    async def test_stop_resets_poison_state_for_restart(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # codex pr_validate r8 NIT #1: ``stop()`` previously left
        # ``_load_error``, ``_worker_stuck``, and admission
        # reservations intact, so a poisoned engine stayed poisoned
        # across the restart — admission would 503 forever despite
        # a healthy fresh worker, and a cached ``_load_error`` from
        # a transient mlx-vlm import failure would be re-raised by
        # ``_ensure_loaded`` even after a successful reload.
        _install_mlx_vlm_mock(monkeypatch, stream_yields=[])
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()

        # Inject poison state.
        engine._load_error = RuntimeError("simulated stale failure")
        engine._worker_stuck = True
        with engine._admission_lock:
            engine._admission_reservations = 7

        await engine.stop()
        # All poison flags MUST be cleared on the clean-stop path.
        assert engine._load_error is None
        assert engine._worker_stuck is False
        assert engine._admission_reservations == 0


class TestMaxTokensClamp:
    """codex pr_validate r8 BLOCKING #1: ``DiffusionEngine``'s
    constructor accepted a ``max_tokens`` server cap (default 32768)
    but never consulted it on the request path. Per-request
    ``max_tokens`` went straight to mlx-vlm with no upper bound, so
    a misbehaving client could request 1 M tokens and burn GPU
    time the operator never authorised. Fix: clamp in
    ``_stream_prompt_raw`` against ``self._max_tokens`` before
    enqueuing the job.
    """

    @pytest.mark.asyncio
    async def test_request_max_tokens_clamped_against_constructor_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Engine cap = 64. Request = 10000. The job submitted to
        # the worker MUST carry the clamped value (64), not the
        # request's 10000.
        captured_max_tokens: list[int] = []

        def _stream(*_a: Any, **kwargs: Any) -> Iterator[FakeGenerationResult]:
            captured_max_tokens.append(kwargs.get("max_tokens"))
            yield FakeGenerationResult(text="ok", finish_reason="stop")

        _install_mlx_vlm_mock(monkeypatch, stream_yields=[])
        diff_mod = sys.modules["mlx_vlm.generate.diffusion"]
        diff_mod.stream_diffusion_generate = _stream  # type: ignore[attr-defined]
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y", max_tokens=64)
        engine._load_blocking()
        async for _ in engine.stream_chat(
            [{"role": "user", "content": "hi"}], max_tokens=10000
        ):
            pass
        assert captured_max_tokens, "stream_diffusion_generate never called"
        assert captured_max_tokens[0] == 64, (
            f"expected clamped max_tokens=64 (engine cap); got "
            f"{captured_max_tokens[0]} — the request value leaked "
            "past the server-side cap"
        )

    @pytest.mark.asyncio
    async def test_request_max_tokens_below_cap_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When the request is below the cap, the engine MUST forward
        # the request value verbatim — clamping with min() must not
        # also lower legitimate-sized requests.
        captured_max_tokens: list[int] = []

        def _stream(*_a: Any, **kwargs: Any) -> Iterator[FakeGenerationResult]:
            captured_max_tokens.append(kwargs.get("max_tokens"))
            yield FakeGenerationResult(text="ok", finish_reason="stop")

        _install_mlx_vlm_mock(monkeypatch, stream_yields=[])
        diff_mod = sys.modules["mlx_vlm.generate.diffusion"]
        diff_mod.stream_diffusion_generate = _stream  # type: ignore[attr-defined]
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y", max_tokens=4096)
        engine._load_blocking()
        async for _ in engine.stream_chat(
            [{"role": "user", "content": "hi"}], max_tokens=256
        ):
            pass
        assert captured_max_tokens[0] == 256


class TestTokenIdZeroNotSwallowed:
    """pr_validate r5 NIT: ``last_token = int(getattr(result, "token",
    last_token) or last_token)`` treated token id ``0`` as missing
    and reused the previous one. Gemma's <pad> sits at id 0 and many
    tokenizers reserve ids 0-2 for special tokens; silently
    discarding them shipped wrong ``tokens=[...]`` in the
    GenerationOutput.
    """

    @pytest.mark.asyncio
    async def test_token_id_zero_is_recorded_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stream a sequence where the FIRST block sets a non-zero
        # previous token, then the second block sends ``token=0``.
        # The pre-fix code (``getattr(...) or last_token``) would
        # have reused 42 instead of recording 0 — pr_validate r6
        # codex BLOCKING #2: the earlier version of this test only
        # had a token=0 chunk and ``last_token`` was initialised to
        # 0, so the broken code would have stayed green. By priming
        # the previous token with 42 first, a regression to the
        # truthy-fallback pattern would now report tokens=[42] on
        # the second block and fail the assertion.
        yields = [
            FakeGenerationResult(
                text="first",
                token=42,  # <-- primes last_token to non-zero
                diffusion_block_complete=True,
            ),
            FakeGenerationResult(
                text="zero",
                token=0,  # <-- the regression point
                diffusion_block_complete=True,
            ),
            FakeGenerationResult(text="", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        outs: list[Any] = []
        async for out in engine.stream_chat(
            [{"role": "user", "content": "hi"}], max_tokens=16
        ):
            outs.append(out)
        non_finish = [o for o in outs if not o.finished]
        assert len(non_finish) == 2, outs
        # First block records 42 (sanity check on the priming step).
        assert non_finish[0].tokens == [42], (
            f"first block: expected tokens=[42]; got {non_finish[0].tokens}"
        )
        # Second block MUST record 0 verbatim — if the
        # ``or last_token`` regression returned, this would be [42].
        assert non_finish[1].tokens == [0], (
            f"second block: expected tokens=[0] (verbatim from "
            f"result.token=0); got tokens={non_finish[1].tokens} — "
            "the ``or last_token`` truthy-fallback regression would "
            "have leaked the previous token through here"
        )


class TestR10Regressions:
    """codex pr_validate r10 BLOCKING fixes.

    BLOCKING #1: dead-worker reset in ``_start_worker_once``.
    BLOCKING #2: skip ``cancel_event.set()`` + use 2 s ``done_event``
                 budget on clean ``_STREAM_DONE`` path.
    BLOCKING #3: pump-thread setup failure must drain the pump via
                 a ``_STREAM_DONE`` sentinel + join, so a daemon
                 thread does not leak parked on ``thread_q.get()``.
    """

    @pytest.mark.asyncio
    async def test_start_worker_once_resets_state_when_worker_dead(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # codex pr_validate r10 BLOCKING #1: if the first worker died
        # during load (mlx-vlm import failure, Metal unavailable,
        # block-family mismatch — paths where ``_worker_loop`` sets
        # ``_load_error`` and returns without ever entering the job
        # loop), the non-None ``_worker`` reference prevented any
        # subsequent ``start()`` / ``_load_blocking()`` from spawning
        # a fresh worker, leaving the engine permanently stuck on the
        # original load error. Verify the dead-worker branch detects
        # this state and resets the bookkeeping (worker / ready /
        # load_error / loaded / stop flags).
        _install_mlx_vlm_mock(monkeypatch, stream_yields=[])
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")

        # Inject a "worker that died during load": run a no-op thread
        # to completion, then plant it on ``_worker`` together with a
        # stale ``_load_error`` to simulate the failed-load state.
        dead = threading.Thread(target=lambda: None, daemon=True)
        dead.start()
        dead.join(timeout=1.0)
        assert dead.is_alive() is False
        engine._worker = dead
        engine._load_error = RuntimeError("simulated stale load failure")
        engine._loaded = False
        engine._ready.set()  # poison: a stale ready event from prior load

        # Pre-fix: ``_start_worker_once`` would see ``_worker is not
        # None`` and refuse to spawn. Post-fix: dead worker is
        # detected, all reset state cleared, fresh worker spawned.
        engine._start_worker_once()

        assert engine._worker is not dead, (
            "expected fresh worker after dead-worker reset; got the "
            "same dead instance — BLOCKING #1 fix regressed"
        )
        assert engine._worker is not None
        assert engine._worker.is_alive() is True, (
            "fresh worker is not running — _start_worker_once spawned "
            "but the thread terminated immediately"
        )
        assert engine._stop is False

        # Wait for the FRESH worker's load cycle to complete, then
        # verify the engine is healthy — load-error cleared and
        # ``_loaded`` flipped to True (proves the reset actually
        # unblocked a real reload, not just spawned a dead worker).
        import asyncio as _aio

        await _aio.to_thread(engine._wait_until_ready)
        assert engine._loaded is True, (
            "fresh worker did not flip _loaded to True — the dead-"
            "worker reset cleared bookkeeping but the new load did "
            "not actually run"
        )
        assert engine._load_error is None, (
            "dead-worker reset must clear stale _load_error so a "
            "successful re-load isn't drowned out by the cached error"
        )

        # Clean up so the daemon thread doesn't outlive the test.
        await engine.stop()

    @pytest.mark.asyncio
    async def test_clean_stream_does_not_burn_30s_done_event_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # codex pr_validate r10 BLOCKING #2: previously the finally
        # block in ``_stream_prompt_raw`` ALWAYS invoked
        # ``cancel_event.set()`` followed by ``done_event.wait(30.0)``
        # — both wasteful on the clean ``_STREAM_DONE`` path because
        # the worker had already returned. Worse, on a genuinely slow
        # OS-scheduling moment the 30 s ceiling could hang the
        # response. Pin two invariants:
        #   * clean stream end-to-end < 5 s on the fake stream (any
        #     value near 30 s means the wait-budget split regressed),
        #   * engine is NOT marked unhealthy after a clean stream
        #     (a fall-through to the cancellation drain path would
        #     flip ``_worker_stuck`` to True).
        yields = [
            FakeGenerationResult(text="hi", token=1, diffusion_block_complete=True),
            FakeGenerationResult(text="", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()

        import time as _time

        start = _time.monotonic()
        async for _ in engine.stream_chat(
            [{"role": "user", "content": "hi"}], max_tokens=16
        ):
            pass
        elapsed = _time.monotonic() - start

        assert elapsed < 5.0, (
            f"clean stream took {elapsed:.2f}s — expected <5s. The "
            "30s ``done_event.wait`` budget was meant only for the "
            "cancellation path; running it on every clean stream "
            "means BLOCKING #2's ``stream_done_observed`` guard "
            "regressed."
        )
        assert engine._worker_stuck is False, (
            "engine was poisoned by ``_worker_stuck = True`` on the "
            "clean ``_STREAM_DONE`` path — BLOCKING #2 fix regressed; "
            "only the cancellation drain timeout should mark unhealthy"
        )
        await engine.stop()

    @pytest.mark.asyncio
    async def test_pump_thread_drained_when_jobs_put_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # codex pr_validate r10 BLOCKING #3: if ``self._jobs.put``
        # raises AFTER ``pump_thread.start()`` succeeded, the pump
        # thread is left blocked on ``thread_q.get()`` forever — a
        # daemon-thread leak. The fix wraps both calls in a
        # try/except that pushes ``_STREAM_DONE`` on the pump's queue
        # so it observes the sentinel and exits cleanly.
        _install_mlx_vlm_mock(monkeypatch, stream_yields=[])
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()

        # Capture every pump thread that gets started so we can
        # verify it eventually exits.
        captured_pumps: list[threading.Thread] = []
        original_start = threading.Thread.start

        def _capture_start(self_t: threading.Thread) -> None:
            if self_t.name == "rapid-mlx-diffusion-pump":
                captured_pumps.append(self_t)
            original_start(self_t)

        monkeypatch.setattr(threading.Thread, "start", _capture_start)

        # Force ``_jobs.put`` to raise on the FIRST stream attempt
        # (the worker is already loaded so its queue.put during
        # startup is not affected). Fall through to the real put on
        # subsequent calls so cleanup ``stop()`` can still enqueue
        # its sentinel.
        original_put = engine._jobs.put
        put_call_count = [0]

        def _failing_put(*a: Any, **k: Any) -> None:
            put_call_count[0] += 1
            if put_call_count[0] == 1:
                raise RuntimeError("simulated _jobs.put failure")
            return original_put(*a, **k)

        monkeypatch.setattr(engine._jobs, "put", _failing_put)

        # The failure must propagate to the caller AS-IS (no silent
        # swallow), and the pump thread must drain on the way out.
        with pytest.raises(RuntimeError, match="simulated _jobs.put failure"):
            async for _ in engine.stream_chat(
                [{"role": "user", "content": "hi"}], max_tokens=16
            ):
                pass

        assert len(captured_pumps) == 1, (
            "expected exactly one pump thread to start during the "
            f"failed stream; saw {len(captured_pumps)}"
        )
        captured_pumps[0].join(timeout=5.0)
        assert captured_pumps[0].is_alive() is False, (
            "pump thread leaked after ``_jobs.put`` raised — the "
            "setup-failure except block must push ``_STREAM_DONE`` "
            "on ``thread_q`` so the pump exits its ``get()`` loop. "
            "BLOCKING #3 fix regressed."
        )

        await engine.stop()


class TestChannelHeaderStrip:
    """DiffusionGemma chat template wraps responses in
    ``<|channel>NAME\\n…<channel|>`` blocks. The angle-bracket tokens
    are special and get stripped by mlx-vlm's detokenizer (via
    ``skip_special_token_ids = all_special_ids`` — the standard
    construction we share with ``mlx_vlm/server/generation.py``), but
    the literal channel NAME (``thought`` / ``final``) and its newline
    leak through as the first visible text.

    These tests pin the strip behaviour on the first emitted block per
    request, plus the helper-function contract.
    """

    def test_helper_strips_thought_prefix(self) -> None:
        from vllm_mlx.runtime.diffusion_lane import _strip_leading_channel_header

        assert _strip_leading_channel_header("thought\nHello world") == "Hello world"

    def test_helper_strips_final_prefix(self) -> None:
        from vllm_mlx.runtime.diffusion_lane import _strip_leading_channel_header

        assert (
            _strip_leading_channel_header("final\nThe answer is 42.")
            == "The answer is 42."
        )

    def test_helper_no_op_on_plain_text(self) -> None:
        from vllm_mlx.runtime.diffusion_lane import _strip_leading_channel_header

        assert _strip_leading_channel_header("Hello, world!\n") == "Hello, world!\n"

    def test_helper_does_not_strip_mid_text(self) -> None:
        # ``thought\n`` mid-string is legitimate prose, not a leaked
        # channel header — the strip must only fire at position 0.
        from vllm_mlx.runtime.diffusion_lane import _strip_leading_channel_header

        text = "Here is my thought\nLet's continue."
        assert _strip_leading_channel_header(text) == text

    def test_helper_does_not_strip_without_newline(self) -> None:
        # The channel-header pattern requires a trailing ``\n``. A bare
        # ``thought`` at the start of legitimate content (e.g. an essay
        # titled ``thought experiments``) must pass through unchanged.
        from vllm_mlx.runtime.diffusion_lane import _strip_leading_channel_header

        text = "thought experiments are fun."
        assert _strip_leading_channel_header(text) == text

    @pytest.mark.asyncio
    async def test_engine_strips_thought_channel_header_on_first_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate DiffusionGemma's leak shape: first block emits
        # ``thought\n<actual content>`` because the tokenizer stripped
        # ``<|channel>`` (id 100) and ``<channel|>`` (id 101) but left
        # the literal channel name behind. The engine must remove the
        # ``thought\n`` prefix from the first delivered block.
        yields = [
            FakeGenerationResult(
                text="thought\n**Reasoning:**\n",
                token=1,
                diffusion_block_complete=True,
            ),
            FakeGenerationResult(
                text="They both weigh exactly 1 kg.",
                token=2,
                diffusion_block_complete=True,
            ),
            FakeGenerationResult(text="", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        outs: list[Any] = []
        async for out in engine.stream_chat(
            [{"role": "user", "content": "Show your reasoning"}],
            max_tokens=128,
        ):
            outs.append(out)
        # First emitted block had its ``thought\n`` prefix stripped.
        first_text = outs[0].new_text
        assert not first_text.startswith("thought\n"), (
            f"first block still leaks ``thought\\n`` prefix: {first_text!r}"
        )
        assert first_text.startswith("**Reasoning:**"), (
            f"strip removed too much; expected ``**Reasoning:**`` prefix, "
            f"got {first_text!r}"
        )
        # Subsequent block is untouched — sanity-check second block
        # text was delivered intact even though it follows a stripped one.
        assert any("They both weigh exactly 1 kg." in o.new_text for o in outs)

    @pytest.mark.asyncio
    async def test_engine_strips_final_channel_header_on_first_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Same path, ``final`` channel name (the model's other declared
        # channel per its tokenizer_config.json x-regex template).
        yields = [
            FakeGenerationResult(
                text="final\nThe capital of France is Paris.",
                token=1,
                diffusion_block_complete=True,
            ),
            FakeGenerationResult(text="", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        outs: list[Any] = []
        async for out in engine.stream_chat(
            [{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=64,
        ):
            outs.append(out)
        first_text = outs[0].new_text
        assert first_text.startswith("The capital of France is Paris."), (
            f"``final\\n`` prefix not stripped: {first_text!r}"
        )

    @pytest.mark.asyncio
    async def test_engine_does_not_strip_thought_from_second_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Mid-stream ``thought\n`` is legitimate text in many genres
        # (essays, code comments, lyrics). After the first block is
        # delivered, the strip MUST NOT fire on later blocks.
        yields = [
            FakeGenerationResult(
                text="Here is some prose. ",
                token=1,
                diffusion_block_complete=True,
            ),
            FakeGenerationResult(
                text="thought\nThis is a continuation, not a channel header.",
                token=2,
                diffusion_block_complete=True,
            ),
            FakeGenerationResult(text="", finish_reason="stop"),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        outs: list[Any] = []
        async for out in engine.stream_chat(
            [{"role": "user", "content": "tell me prose"}], max_tokens=64
        ):
            outs.append(out)
        # The second block's literal ``thought\n`` must be preserved
        # because the leading-strip already fired on block 1.
        non_finish = [o for o in outs if not o.finished and o.new_text]
        assert len(non_finish) >= 2
        assert "thought\nThis is a continuation" in non_finish[1].new_text, (
            f"second block was wrongly stripped: {non_finish[1].new_text!r}"
        )

    @pytest.mark.asyncio
    async def test_engine_strips_channel_header_on_short_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Edge case: entire response fits in one block that exits via
        # the trailing-finish path (no ``diffusion_block_complete`` ever
        # set; the worker's generator just returns). The strip must
        # still fire there.
        yields = [
            FakeGenerationResult(text="thought\nYes.", token=1),
        ]
        _install_mlx_vlm_mock(monkeypatch, stream_yields=yields)
        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="x/y")
        engine._load_blocking()
        outs: list[Any] = []
        async for out in engine.stream_chat(
            [{"role": "user", "content": "answer yes or no"}], max_tokens=16
        ):
            outs.append(out)
        # The trailing-finish path emits the buffered text as a single
        # terminal chunk; verify the strip applied there too.
        combined = "".join(o.new_text for o in outs)
        assert "thought\n" not in combined, (
            f"channel header not stripped on trailing-finish path: {combined!r}"
        )
        assert "Yes." in combined
