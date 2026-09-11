# SPDX-License-Identifier: Apache-2.0
"""Model-free contracts for the optional optimized DFlash runtime adapter."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from vllm_mlx.speculative.dflash import upstream_runtime as upstream
from vllm_mlx.speculative.dflash.generation import generate, stream_generate
from vllm_mlx.speculative.dflash.runtime import DFlashRuntime


def test_optimized_runtime_requires_published_dflash2_capability(monkeypatch) -> None:
    monkeypatch.setattr(upstream, "version", lambda _name: "0.1.8")
    monkeypatch.delitem(sys.modules, "dflash_mlx.draft.dflash2", raising=False)

    with pytest.raises(RuntimeError, match="does not contain.*DFlash2"):
        upstream.require_dflash_mlx_runtime()

    fake_dflash2 = types.ModuleType("dflash_mlx.draft.dflash2")
    fake_dflash2.DFlash2DraftModel = object
    fake_dflash2.normalize_dflash2_config = lambda config: config
    monkeypatch.setitem(sys.modules, "dflash_mlx.draft.dflash2", fake_dflash2)
    assert upstream.require_dflash_mlx_runtime() == "0.1.8"


def test_optimized_runtime_requires_immutable_remote_revision() -> None:
    with pytest.raises(RuntimeError, match="immutable revision pin"):
        upstream._immutable_snapshot("not-a-local-target", None, role="target")


def test_generation_dispatch_uses_explicit_optimized_backend() -> None:
    calls: list[tuple[str, dict]] = []

    class Backend:
        def stream_generate(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return iter(("chunk",))

        def generate(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return "result"

    runtime = DFlashRuntime(
        drafter=object(),
        kind="dflash",
        drafter_repo="draft",
        algorithm="dflash2",
        backend="dflash-mlx",
        backend_state=Backend(),
    )

    assert list(
        stream_generate(runtime, object(), object(), "prompt", max_tokens=8)
    ) == ["chunk"]
    assert generate(runtime, object(), object(), "prompt", max_tokens=8) == "result"
    assert calls == [
        ("prompt", {"max_tokens": 8}),
        ("prompt", {"max_tokens": 8}),
    ]


def test_greedy_adapter_preserves_counts_and_does_not_decode_eos(monkeypatch) -> None:
    @dataclass
    class PrefillCompleteEvent:
        prompt_token_count: int

    @dataclass
    class TokenEvent:
        token_id: int
        generated_tokens: int

    @dataclass
    class SummaryEvent:
        prompt_token_count: int
        generation_tokens: int

    known_types = (PrefillCompleteEvent, TokenEvent, SummaryEvent)
    fake_events = types.ModuleType("dflash_mlx.engine.events")
    fake_events.PrefillCompleteEvent = PrefillCompleteEvent
    fake_events.TokenEvent = TokenEvent
    fake_events.SummaryEvent = SummaryEvent
    fake_events.is_engine_event = lambda event: isinstance(event, known_types)

    receipt: dict[str, object] = {}

    def _stream_dflash_generate(**kwargs):
        receipt.update(kwargs)
        yield PrefillCompleteEvent(prompt_token_count=3)
        yield TokenEvent(token_id=7, generated_tokens=1)
        yield TokenEvent(token_id=99, generated_tokens=2)
        yield SummaryEvent(prompt_token_count=3, generation_tokens=2)

    fake_runtime = types.ModuleType("dflash_mlx.runtime")
    fake_runtime.get_stop_token_ids = lambda _tokenizer: [99]
    fake_runtime.stream_dflash_generate = _stream_dflash_generate
    monkeypatch.setitem(sys.modules, "dflash_mlx.engine.events", fake_events)
    monkeypatch.setitem(sys.modules, "dflash_mlx.runtime", fake_runtime)

    class Detokenizer:
        last_segment = ""

        def reset(self):
            self.last_segment = ""

        def add_token(self, token):
            self.last_segment = f"<{token}>"

        def finalize(self):
            self.last_segment = ""

    tokenizer = SimpleNamespace(detokenizer=Detokenizer())
    bundle = SimpleNamespace(
        target_model=object(),
        tokenizer=tokenizer,
        draft_model=object(),
        draft_backend=object(),
        target_ops=object(),
    )
    runtime = upstream.UpstreamDFlashRuntime(
        bundle=bundle, runtime_context=object(), version="0.1.10"
    )

    chunks = list(runtime.stream_generate("prompt", max_tokens=8))

    assert [(chunk.text, chunk.token) for chunk in chunks] == [
        ("<7>", 7),
        ("", 99),
    ]
    assert chunks[-1].prompt_tokens == 3
    assert chunks[-1].generation_tokens == 2
    assert receipt["stop_token_ids"] == [99]
    assert receipt["quantize_kv_cache"] is False
    assert runtime.last_summary == SummaryEvent(
        prompt_token_count=3, generation_tokens=2
    )


def test_greedy_adapter_rejects_unknown_engine_event(monkeypatch) -> None:
    class PrefillCompleteEvent:
        pass

    class TokenEvent:
        pass

    class SummaryEvent:
        pass

    fake_events = types.ModuleType("dflash_mlx.engine.events")
    fake_events.PrefillCompleteEvent = PrefillCompleteEvent
    fake_events.TokenEvent = TokenEvent
    fake_events.SummaryEvent = SummaryEvent
    fake_events.is_engine_event = lambda _event: False
    fake_runtime = types.ModuleType("dflash_mlx.runtime")
    fake_runtime.get_stop_token_ids = lambda _tokenizer: []
    fake_runtime.stream_dflash_generate = lambda **_kwargs: iter((object(),))
    monkeypatch.setitem(sys.modules, "dflash_mlx.engine.events", fake_events)
    monkeypatch.setitem(sys.modules, "dflash_mlx.runtime", fake_runtime)

    tokenizer = SimpleNamespace(
        detokenizer=SimpleNamespace(reset=lambda: None, finalize=lambda: None)
    )
    bundle = SimpleNamespace(
        target_model=object(),
        tokenizer=tokenizer,
        draft_model=object(),
        draft_backend=object(),
        target_ops=object(),
    )
    runtime = upstream.UpstreamDFlashRuntime(
        bundle=bundle, runtime_context=object(), version="0.1.10"
    )

    with pytest.raises(TypeError, match="Unsupported DFlash engine event"):
        list(runtime.stream_generate("prompt", max_tokens=8))


def test_sampled_request_uses_target_only_generation(monkeypatch) -> None:
    calls: dict[str, object] = {}
    expected_chunk = SimpleNamespace(
        text="sampled", token=4, prompt_tokens=2, generation_tokens=1
    )
    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_sampling = types.ModuleType("mlx_lm.sample_utils")

    def _make_sampler(**kwargs):
        calls["sampler_kwargs"] = kwargs
        return "sampler"

    def _stream_generate(model, tokenizer, prompt, **kwargs):
        calls["generate"] = (model, tokenizer, prompt, kwargs)
        yield expected_chunk

    fake_mlx_lm.stream_generate = _stream_generate
    fake_sampling.make_sampler = _make_sampler
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sampling)

    bundle = SimpleNamespace(
        target_model=object(),
        tokenizer=object(),
        draft_model=object(),
    )
    runtime = upstream.UpstreamDFlashRuntime(
        bundle=bundle, runtime_context=object(), version="0.1.10"
    )

    assert list(
        runtime.stream_generate("prompt", max_tokens=8, temperature=0.7, top_p=0.9)
    ) == [expected_chunk]
    assert calls["sampler_kwargs"] == {"temp": 0.7, "top_p": 0.9}
    assert calls["generate"] == (
        bundle.target_model,
        bundle.tokenizer,
        "prompt",
        {"max_tokens": 8, "sampler": "sampler"},
    )


def test_load_runtime_disables_unqualified_custom_qmm(monkeypatch) -> None:
    @dataclass(frozen=True)
    class VerifyConfig:
        mode: str
        enable_qmm: bool

    @dataclass(frozen=True)
    class RuntimeContext:
        verify: object

    receipt: dict[str, object] = {}
    bundle = SimpleNamespace(
        target_model=object(),
        tokenizer=object(),
        draft_model=object(),
    )
    fake_runtime = types.ModuleType("dflash_mlx.runtime")
    fake_runtime.VerifyConfig = VerifyConfig
    fake_bundle = types.ModuleType("dflash_mlx.runtime.bundle")
    fake_context = types.ModuleType("dflash_mlx.runtime.context")

    def _load_runtime_bundle(**kwargs):
        receipt["load"] = kwargs
        return bundle

    def _build_offline_runtime_context(**kwargs):
        receipt["context"] = kwargs
        return RuntimeContext(verify=VerifyConfig(mode="adaptive", enable_qmm=True))

    fake_bundle.load_runtime_bundle = _load_runtime_bundle
    fake_context.build_offline_runtime_context = _build_offline_runtime_context
    monkeypatch.setitem(sys.modules, "dflash_mlx.runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "dflash_mlx.runtime.bundle", fake_bundle)
    monkeypatch.setitem(sys.modules, "dflash_mlx.runtime.context", fake_context)
    monkeypatch.setattr(upstream, "require_dflash_mlx_runtime", lambda: "0.1.10")
    monkeypatch.setattr(
        upstream,
        "_immutable_snapshot",
        lambda repo, _revision, *, role: f"/cache/{role}/{repo.rsplit('/', 1)[-1]}",
    )

    loaded = upstream.load_upstream_runtime(
        main_model_repo="rapid-mlx/target",
        main_model_revision="a" * 40,
        drafter_repo="z-lab/draft",
        drafter_revision="b" * 40,
    )

    assert loaded.bundle is bundle
    assert loaded.runtime_context.verify == VerifyConfig(
        mode="adaptive", enable_qmm=False
    )
    assert receipt["context"] == {
        "verify_mode": "adaptive",
        "copyspec_mode": "off",
        "quantize_kv_cache": False,
    }
    assert receipt["load"] == {
        "model_ref": "/cache/target/target",
        "draft_ref": "/cache/drafter/draft",
        "draft_quant": "w4:gs64",
        "verify_config": VerifyConfig(mode="adaptive", enable_qmm=False),
        "quantize_kv_cache": False,
    }


def test_runtime_backend_contract_is_explicit() -> None:
    runtime = DFlashRuntime(
        drafter=object(),
        kind="dflash",
        drafter_repo="draft",
        algorithm="dflash2",
        backend="dflash-mlx",
        backend_state=SimpleNamespace(),
    )

    assert runtime.backend == "dflash-mlx"
