# SPDX-License-Identifier: Apache-2.0
"""Routing tests for ``runtime/diffusion_lane.py`` — backend selection
between in-house rapid loop and upstream mlx-vlm.

Backend choice is driven entirely by ``profile.diffusion_backend``
(AliasProfile is the SSOT for every per-model routing decision —
``test_no_out_of_band_routing.py`` forbids env-var routing). Default is
``"rapid"``; ``"mlx-vlm"`` falls through to upstream verbatim.

(``cfg.temperature`` is NOT a gate — both rapid and mlx-vlm honor the
same temperature semantics; rapid extends greedy → categorical via
``_sample_canvas``.)

We spy on which generator was invoked by patching both
``mlx_vlm.generate.diffusion.stream_diffusion_generate`` AND
``vllm_mlx.runtime.diffusion_loop.rapid_stream_diffusion_generate``
to set sentinels. The actual model never has to run.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class _FakeResult:
    text: str = ""
    token: int = 0
    prompt_tokens: int = 0
    generation_tokens: int = 0
    finish_reason: str | None = None
    is_draft: bool = False
    diffusion_block_complete: bool = False


class _FakeTokenizer:
    all_special_ids = [0, 1, 2]

    class _SC:
        def reset(self, *_a: Any) -> None:
            pass

    stopping_criteria = _SC()

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        rendered = "\n".join(m.get("content", "") for m in messages)
        if add_generation_prompt:
            rendered += "\n<model>\n"
        return rendered

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 256 for c in text]


class _FakeProc:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()


class _FakeModelConfig:
    eos_token_id = 7
    canvas_length = 256


class _FakeModel:
    config = _FakeModelConfig()


def _install_backend_spies(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    """Wire fakes for both backends. Return a dict whose keys flip True
    when the corresponding generator is invoked. This is the
    spy mechanism — neither generator actually does any math; they
    both yield two ``_FakeResult`` items so the consumer at
    ``diffusion_lane.py:1229`` runs to completion."""

    called: dict[str, bool] = {"rapid": False, "mlx_vlm": False}
    captured_kwargs: dict[str, dict[str, Any]] = {}

    # ----------- mlx-vlm side ----------------------------------------
    mlx_vlm = sys.modules.get("mlx_vlm") or types.ModuleType("mlx_vlm")
    mlx_vlm_utils = types.ModuleType("mlx_vlm.utils")
    mlx_vlm_utils.load = lambda _: (_FakeModel(), _FakeProc())  # type: ignore[attr-defined]
    mlx_vlm_generate = types.ModuleType("mlx_vlm.generate")
    mlx_vlm_diffusion = types.ModuleType("mlx_vlm.generate.diffusion")
    mlx_vlm_diffusion.diffusion_generation_family = lambda _: "block"  # type: ignore[attr-defined]

    def _stream_upstream(
        model, processor, tokenizer, input_ids, pixel_values, attention_mask, **kw
    ):
        called["mlx_vlm"] = True
        captured_kwargs["mlx_vlm"] = kw
        yield _FakeResult(text="UP", diffusion_block_complete=True)
        yield _FakeResult(text="", finish_reason="stop", generation_tokens=1)

    mlx_vlm_diffusion.stream_diffusion_generate = _stream_upstream  # type: ignore[attr-defined]

    mlx_vlm_common = types.ModuleType("mlx_vlm.generate.common")

    class _WL:
        def __init__(self, *_a: Any, **_k: Any) -> None: ...
        def __enter__(self) -> _WL:
            return self

        def __exit__(self, *_a: Any) -> None: ...

    mlx_vlm_common.wired_limit = _WL  # type: ignore[attr-defined]
    mlx_vlm_common.generation_stream = object()  # type: ignore[attr-defined]

    for n, mod in [
        ("mlx_vlm", mlx_vlm),
        ("mlx_vlm.utils", mlx_vlm_utils),
        ("mlx_vlm.generate", mlx_vlm_generate),
        ("mlx_vlm.generate.diffusion", mlx_vlm_diffusion),
        ("mlx_vlm.generate.common", mlx_vlm_common),
    ]:
        monkeypatch.setitem(sys.modules, n, mod)

    # ----------- Rapid loop side --------------------------------------
    # The lane does ``from .diffusion_loop import rapid_stream_diffusion_generate``
    # INSIDE _run_mlxvlm_generator (lazy import). Patch the attribute on
    # the already-imported module so the lazy import sees our spy.
    import vllm_mlx.runtime.diffusion_loop as loop_mod

    def _stream_rapid(
        model, processor, tokenizer, input_ids, pixel_values, attention_mask, **kw
    ):
        called["rapid"] = True
        captured_kwargs["rapid"] = kw
        yield _FakeResult(text="RA", diffusion_block_complete=True)
        yield _FakeResult(text="", finish_reason="stop", generation_tokens=1)

    monkeypatch.setattr(loop_mod, "rapid_stream_diffusion_generate", _stream_rapid)

    # Stash captured_kwargs on the result so tests can introspect
    # what the lane forwarded to the chosen backend.
    called["_kwargs"] = captured_kwargs  # type: ignore[assignment]
    return called


@pytest.mark.asyncio
async def test_default_alias_uses_rapid_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default profile (no JSON overrides) → diffusion_backend='rapid'
    → rapid path. This is the production path most users hit."""
    spy = _install_backend_spies(monkeypatch)

    from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

    engine = DiffusionEngine(model_name="x/y")
    engine._load_blocking()
    assert engine._profile.diffusion_backend == "rapid"

    async for _ in engine.stream_chat(
        [{"role": "user", "content": "hi"}], max_tokens=8
    ):
        pass

    assert spy["rapid"] is True, "rapid backend should have fired"
    assert spy["mlx_vlm"] is False, "mlx-vlm should NOT have been called"
    # Profile knobs threaded through. Default profile has
    # ``diffusion_fixed_steps=None`` → ``fixed_steps`` is OMITTED
    # entirely (lane only forwards when non-None) so the rapid loop's
    # own ``fixed_steps=None`` default enables adaptive stop.
    # ``sc_every`` is always forwarded.
    rapid_kw = spy["_kwargs"]["rapid"]  # type: ignore[index]
    assert "fixed_steps" not in rapid_kw, (
        "default profile uses adaptive stop; fixed_steps must NOT be "
        f"forwarded, got {rapid_kw.get('fixed_steps')!r}"
    )
    assert rapid_kw["sc_every"] == 1


@pytest.mark.asyncio
async def test_mlx_vlm_backend_when_profile_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting ``"diffusion_backend": "mlx-vlm"`` on an alias falls
    through to upstream verbatim. Emergency rollback path: edit
    aliases.json + restart."""
    spy = _install_backend_spies(monkeypatch)

    # Hand-build a profile with diffusion_backend="mlx-vlm" and bind
    # it to the engine post-init (resolve_profile would only return
    # "rapid" for any real alias today).
    from vllm_mlx.model_aliases import AliasProfile
    from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

    engine = DiffusionEngine(model_name="x/y")
    engine._profile = AliasProfile(
        hf_path="x/y",
        modality="text-diffusion",
        supports_spec_decode=False,
        diffusion_backend="mlx-vlm",
    )
    engine._load_blocking()

    async for _ in engine.stream_chat(
        [{"role": "user", "content": "hi"}], max_tokens=8
    ):
        pass

    assert spy["mlx_vlm"] is True
    assert spy["rapid"] is False
    # Rapid-only kwargs must NOT leak into the mlx-vlm call (would
    # raise TypeError upstream — defense in depth).
    mlx_kw = spy["_kwargs"]["mlx_vlm"]  # type: ignore[index]
    assert "fixed_steps" not in mlx_kw
    assert "sc_every" not in mlx_kw


@pytest.mark.asyncio
async def test_temperature_does_not_gate_rapid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``temperature>0`` must still take the rapid path. The lane
    used to gate on ``temperature==0`` (rapid was greedy-only); with
    ``_sample_canvas`` honoring temperature, that gate is gone — the
    default ``_FALLBACK_TEMPERATURE=0.7`` flows through cleanly."""
    spy = _install_backend_spies(monkeypatch)
    monkeypatch.delenv("RAPID_MLX_DIFFUSION_BACKEND", raising=False)

    from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

    engine = DiffusionEngine(model_name="x/y")
    engine._load_blocking()
    async for _ in engine.stream_chat(
        [{"role": "user", "content": "hi"}], max_tokens=8, temperature=0.7
    ):
        pass

    assert spy["rapid"] is True
    rapid_kw = spy["_kwargs"]["rapid"]  # type: ignore[index]
    assert rapid_kw["temperature"] == pytest.approx(0.7)


# =============================================================================
# Profile resolution — alias name vs HF path vs unknown
# =============================================================================


def test_profile_resolves_for_known_alias_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_backend_spies(monkeypatch)
    from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

    engine = DiffusionEngine(model_name="diffusion-gemma-26b")
    assert engine._profile.modality == "text-diffusion"
    assert engine._profile.diffusion_backend == "rapid"
    assert engine._profile.diffusion_fixed_steps is None  # adaptive stop
    assert engine._profile.diffusion_sc_every == 1


def test_profile_falls_back_for_unknown_hf_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bare HF paths that no alias references get a throwaway profile
    with safe diffusion defaults so the engine still has knobs to
    pass into the rapid loop."""
    _install_backend_spies(monkeypatch)
    from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

    engine = DiffusionEngine(model_name="some-org/no-such-alias")
    assert engine._profile is not None
    assert engine._profile.modality == "text-diffusion"
    assert engine._profile.diffusion_backend == "rapid"  # default
    assert engine._profile.diffusion_fixed_steps is None  # adaptive stop
    assert engine._profile.diffusion_sc_every == 1  # default


@pytest.mark.asyncio
async def test_alias_with_explicit_fixed_steps_forwards_int(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operators can opt into a fixed step budget; when set, the lane
    MUST forward ``fixed_steps=<int>`` so the rapid loop honors it and
    disables adaptive stop."""
    spy = _install_backend_spies(monkeypatch)
    from vllm_mlx.model_aliases import AliasProfile
    from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

    engine = DiffusionEngine(model_name="x/y")
    engine._profile = AliasProfile(
        hf_path="x/y",
        modality="text-diffusion",
        supports_spec_decode=False,
        diffusion_fixed_steps=8,
    )
    engine._load_blocking()
    async for _ in engine.stream_chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=8,
    ):
        pass
    rapid_kw = spy["_kwargs"]["rapid"]  # type: ignore[index]
    assert rapid_kw["fixed_steps"] == 8
