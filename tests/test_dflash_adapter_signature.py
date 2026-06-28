# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the DFlash driver vs mlx-vlm 0.6.3 (#343).

The previous adapter in ``vllm_mlx/spec_decode/dflash/drafter.py`` called
``DFlashDraftModel.draft_block(prefix_tokens, current_position)`` — a
signature that mlx-vlm 0.6.3 does NOT expose. The real signature is
``draft_block(last_bonus, hidden, cache, block_size, sampler,
token_dtype)`` (the drafter is hidden-state-conditioned). That mismatch
made the bench script non-functional and would have failed at the first
real call into the drafter.

These tests pin the assumptions the new :class:`MlxVlmDFlashDriver`
makes about mlx-vlm so a future bump that drifts the signature (or
renames a callback) trips a clear assertion here instead of producing
silently-wrong bench numbers.

Coverage:

* :func:`test_mlx_vlm_draft_block_signature_matches_0_6_3` —
  ``inspect.signature(DFlashDraftModel.draft_block)`` exposes the
  expected parameter names AND the ``token_dtype`` default. If
  mlx-vlm renames any of these, this test pins the failure to the
  upstream contract change.
* :func:`test_dflash_runtime_kind_attribute` — the rapid-mlx wrapper
  reads ``runtime.kind`` to populate the ``draft_kind`` kwarg into
  ``stream_generate``. Pin that attribute name.
* :func:`test_driver_generate_invokes_stream_generate_with_drafter` —
  with the heavy dependencies stubbed, exercise
  :meth:`MlxVlmDFlashDriver.generate` and assert the kwargs forwarded
  to ``mlx_vlm.stream_generate`` include both ``draft_model`` (the
  drafter object) and ``draft_kind`` (the runtime kind).
* :func:`test_driver_accept_stats_reads_drafter_lists` — the wrapper
  uses ``drafter.accept_lens`` and ``drafter.draft_lens`` to compute
  accept rate. Pin those attribute names and the summary math.
* :func:`test_driver_load_is_idempotent` — :meth:`load` must be safe
  to call repeatedly without re-loading.
"""

from __future__ import annotations

import inspect
import sys
import types
from typing import Any

import pytest

# ``mlx.core`` is an optional dep (in ``[dflash]`` extras). The
# signature test below imports it lazily after ``importorskip`` so a
# collect on a bare-Python env doesn't crash with ImportError before
# pytest can skip.

# ---------------------------------------------------------------------------
# Upstream-contract pins — fail loudly when mlx-vlm drifts the signature.
# ---------------------------------------------------------------------------


def test_mlx_vlm_draft_block_signature_matches_0_6_3() -> None:
    """Pin the mlx-vlm 0.6.3 ``draft_block`` parameter contract.

    The rapid-mlx DFlash driver assumes mlx-vlm's drafter is called
    with these positional args by mlx-vlm's own ``_dflash_rounds``
    loop. We don't call it directly anymore — but if the upstream
    drifts the signature, ``_dflash_rounds`` itself stops working,
    and we want a CLEAR diagnostic ("mlx-vlm signature changed,
    update the rapid-mlx wrapper") instead of a flaky GPU crash.
    """
    pytest.importorskip("mlx_vlm")
    mx = pytest.importorskip("mlx.core")
    from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import DFlashDraftModel

    sig = inspect.signature(DFlashDraftModel.draft_block)
    params = list(sig.parameters)
    assert params == [
        "self",
        "last_bonus",
        "hidden",
        "cache",
        "block_size",
        "sampler",
        "token_dtype",
    ], (
        "mlx-vlm 0.6.3 DFlashDraftModel.draft_block parameter list "
        f"changed: got {params}. Update MlxVlmDFlashDriver in "
        "vllm_mlx/spec_decode/dflash/drafter.py to match — see #343."
    )
    # token_dtype has a default; the other slots are positional-required.
    assert sig.parameters["token_dtype"].default is mx.int32


def test_dflash_runtime_kind_attribute_exists() -> None:
    """Pin ``DFlashRuntime.kind`` — the wrapper reads it for stream_generate."""
    from vllm_mlx.speculative.dflash.runtime import DFlashRuntime

    # Construct a stub runtime (load_runtime hits mlx-vlm, but the
    # dataclass itself is plain).
    rt = DFlashRuntime(drafter=object(), kind="dflash", drafter_repo="z-lab/test")
    assert rt.kind == "dflash"
    assert rt.drafter_repo == "z-lab/test"


# ---------------------------------------------------------------------------
# Driver-level tests — mock the mlx-vlm surface to isolate wiring.
# ---------------------------------------------------------------------------


class _FakeChunk:
    """Minimal stand-in for mlx-vlm's ``GenerationResult`` chunk."""

    def __init__(self, text: str, token: int, generation_tokens: int) -> None:
        self.text = text
        self.token = token
        self.generation_tokens = generation_tokens
        self.prompt_tokens = 7  # arbitrary; bench doesn't read this for tok/s


class _FakeDrafter:
    """Stand-in for mlx-vlm's ``DFlashDraftModel`` for wiring tests.

    Owns the ``accept_lens`` / ``draft_lens`` lists the driver reads
    after generation. We don't simulate the per-block diffusion math —
    the driver delegates that to the stubbed ``stream_generate``.
    """

    def __init__(self) -> None:
        self.accept_lens: list[int] = []
        self.draft_lens: list[int] = []

    def populate(self, accept_lens: list[int], draft_lens: list[int]) -> None:
        self.accept_lens = list(accept_lens)
        self.draft_lens = list(draft_lens)


class _FakeRuntime:
    """Minimal stand-in for :class:`DFlashRuntime`."""

    def __init__(self, drafter: _FakeDrafter, kind: str = "dflash") -> None:
        self.drafter = drafter
        self.kind = kind
        self.drafter_repo = "z-lab/fake-drafter"
        self.reset_calls = 0

    def reset_accept_lens(self) -> None:
        self.reset_calls += 1
        if isinstance(self.drafter.accept_lens, list):
            self.drafter.accept_lens.clear()


@pytest.fixture
def stub_mlx_vlm(monkeypatch: pytest.MonkeyPatch):
    """Stub ``mlx_vlm.load`` + ``stream_generate`` + ``load_runtime``.

    NOTE: this fixture uses ``importorskip("mlx_vlm")`` to skip the
    driver wiring tests cleanly when ``rapid-mlx`` is installed
    WITHOUT the ``[dflash]`` extra (mlx-vlm is an optional dep). The
    ``test_mlx_vlm_draft_block_signature_matches_0_6_3`` test above
    already gates on ``importorskip`` for the same reason; mirroring
    here keeps the surface consistent.

    Patches:
      * ``mlx_vlm.load`` — returns (model, processor) tuples
      * ``mlx_vlm.stream_generate`` — records call args, yields fake chunks
      * ``vllm_mlx.speculative.dflash.load_runtime`` — returns a fake runtime

    Returns a dict with ``calls`` (list of dict, one per stream_generate
    invocation) and ``runtime`` (the fake DFlashRuntime instance), so
    individual tests can swap in different drafter state.
    """
    mlx_vlm = pytest.importorskip("mlx_vlm")

    calls: list[dict[str, Any]] = []
    drafter = _FakeDrafter()
    runtime = _FakeRuntime(drafter)

    def fake_load(repo: str):
        return (f"<target-{repo}>", f"<processor-{repo}>")

    def fake_stream_generate(model, processor, prompt, **kwargs):
        calls.append(
            {
                "model": model,
                "processor": processor,
                "prompt": prompt,
                "kwargs": dict(kwargs),
            }
        )
        # Emit 3 fake chunks; the bench reads .generation_tokens
        # so we yield a strictly-increasing counter.
        yield _FakeChunk("hello", token=1, generation_tokens=1)
        yield _FakeChunk(" world", token=2, generation_tokens=2)
        yield _FakeChunk("", token=3, generation_tokens=3)

    monkeypatch.setattr(mlx_vlm, "load", fake_load, raising=True)
    monkeypatch.setattr(mlx_vlm, "stream_generate", fake_stream_generate, raising=True)

    # Patch load_runtime in the speculative bridge module.
    import vllm_mlx.speculative.dflash as bridge

    monkeypatch.setattr(bridge, "load_runtime", lambda repo: runtime, raising=True)
    # Also patch the symbol re-export so ``from ... import load_runtime``
    # inside the driver picks up the fake one.
    monkeypatch.setattr(
        sys.modules["vllm_mlx.speculative.dflash"],
        "load_runtime",
        lambda repo: runtime,
        raising=True,
    )

    return {"calls": calls, "runtime": runtime, "drafter": drafter}


def test_driver_generate_invokes_stream_generate_with_drafter(stub_mlx_vlm) -> None:
    """The driver must forward ``draft_model`` and ``draft_kind`` to mlx-vlm.

    Pinning this catches a regression where the wrapper forgets to set
    the drafter (resulting in a silent fall-back to plain AR decode —
    the exact failure mode that masked the broken 0.9 bench).
    """
    from vllm_mlx.spec_decode.dflash.drafter import MlxVlmDFlashDriver

    driver = MlxVlmDFlashDriver(
        target_repo="mlx-community/Qwen3.5-9B-4bit",
        drafter_repo="z-lab/Qwen3.5-9B-DFlash",
    )
    driver.load()
    chunks = list(driver.generate("Hello, world.", max_tokens=8, temperature=0.0))

    assert len(chunks) == 3
    assert len(stub_mlx_vlm["calls"]) == 1
    call = stub_mlx_vlm["calls"][0]
    assert call["prompt"] == "Hello, world."
    kwargs = call["kwargs"]
    # Critical contract: drafter + kind both threaded through.
    assert kwargs["draft_model"] is stub_mlx_vlm["runtime"].drafter
    assert kwargs["draft_kind"] == "dflash"
    assert kwargs["max_tokens"] == 8
    assert kwargs["temperature"] == 0.0
    # block_size left to mlx-vlm default when not set.
    assert "draft_block_size" not in kwargs


def test_driver_forwards_block_size_override(stub_mlx_vlm) -> None:
    """When ``block_size`` is set on the driver, forward it to mlx-vlm."""
    from vllm_mlx.spec_decode.dflash.drafter import MlxVlmDFlashDriver

    driver = MlxVlmDFlashDriver(
        target_repo="mlx-community/Qwen3.5-9B-4bit",
        drafter_repo="z-lab/Qwen3.5-9B-DFlash",
        block_size=8,
    )
    driver.load()
    list(driver.generate("Hello.", max_tokens=4, temperature=0.0))
    assert stub_mlx_vlm["calls"][0]["kwargs"]["draft_block_size"] == 8


def test_driver_accept_stats_reads_drafter_lists(stub_mlx_vlm) -> None:
    """Pin the ``accept_lens`` / ``draft_lens`` attribute names + the math."""
    from vllm_mlx.spec_decode.dflash.drafter import MlxVlmDFlashDriver

    driver = MlxVlmDFlashDriver(
        target_repo="mlx-community/Qwen3.5-9B-4bit",
        drafter_repo="z-lab/Qwen3.5-9B-DFlash",
    )
    driver.load()

    # Pre-populate so the bench sees a real number — generate() will
    # call reset_accept_lens() which clears accept_lens; we re-populate
    # AFTER iteration to simulate what _dflash_rounds does.
    list(driver.generate("Hi.", max_tokens=4, temperature=0.0))
    stub_mlx_vlm["drafter"].populate(
        accept_lens=[3, 2, 4, 0],
        draft_lens=[7, 7, 7, 7],
    )

    stats = driver.accept_stats()
    assert stats["attempts"] == 4
    assert stats["accepted_tokens"] == 9
    assert stats["drafted_tokens"] == 28
    assert stats["accept_rate"] == pytest.approx(9 / 28)
    assert stats["mean_accepted_per_attempt"] == pytest.approx(9 / 4)
    assert stats["accept_lens"] == [3, 2, 4, 0]
    assert stats["draft_lens"] == [7, 7, 7, 7]


def test_driver_accept_stats_handles_empty_runs(stub_mlx_vlm) -> None:
    """Zero-attempt runs must not divide by zero."""
    from vllm_mlx.spec_decode.dflash.drafter import MlxVlmDFlashDriver

    driver = MlxVlmDFlashDriver(
        target_repo="mlx-community/Qwen3.5-9B-4bit",
        drafter_repo="z-lab/Qwen3.5-9B-DFlash",
    )
    driver.load()
    stats = driver.accept_stats()
    assert stats["attempts"] == 0
    assert stats["accepted_tokens"] == 0
    assert stats["accept_rate"] == 0.0
    assert stats["mean_accepted_per_attempt"] == 0.0


def test_driver_load_is_idempotent(stub_mlx_vlm) -> None:
    """A second ``load()`` must not re-invoke the loaders."""
    from vllm_mlx.spec_decode.dflash.drafter import MlxVlmDFlashDriver

    # Patch load to count invocations.
    load_calls = {"n": 0}
    import mlx_vlm

    original = mlx_vlm.load

    def counting_load(repo):
        load_calls["n"] += 1
        return original(repo)

    mlx_vlm.load = counting_load
    try:
        driver = MlxVlmDFlashDriver(
            target_repo="mlx-community/Qwen3.5-9B-4bit",
            drafter_repo="z-lab/Qwen3.5-9B-DFlash",
        )
        driver.load()
        assert load_calls["n"] == 1
        driver.load()  # idempotent
        assert load_calls["n"] == 1
    finally:
        mlx_vlm.load = original


def test_driver_generate_without_load_raises() -> None:
    """``generate`` before ``load`` should error clearly, not crash inside mlx-vlm."""
    from vllm_mlx.spec_decode.dflash.drafter import MlxVlmDFlashDriver

    driver = MlxVlmDFlashDriver(
        target_repo="mlx-community/Qwen3.5-9B-4bit",
        drafter_repo="z-lab/Qwen3.5-9B-DFlash",
    )
    with pytest.raises(RuntimeError, match="load"):
        next(driver.generate("Hi.", max_tokens=4))


def test_driver_rejects_empty_repos() -> None:
    """Empty target / drafter args should fail at construction, not load."""
    from vllm_mlx.spec_decode.dflash.drafter import MlxVlmDFlashDriver

    with pytest.raises(ValueError, match="target_repo"):
        MlxVlmDFlashDriver(target_repo="", drafter_repo="z-lab/Qwen3.5-9B-DFlash")
    with pytest.raises(ValueError, match="drafter_repo"):
        MlxVlmDFlashDriver(target_repo="mlx-community/Qwen3.5-9B-4bit", drafter_repo="")


def test_driver_adopt_accepts_preloaded_objects(stub_mlx_vlm) -> None:
    """``adopt()`` lets a caller inject already-loaded objects.

    Bench scripts pay one mlx-vlm.load() for both baseline and DFlash
    conditions and re-use the resulting model. Without ``adopt()`` they
    would either have to load twice (28+ GB for Qwen3.5-27B-8bit) or
    poke at the driver's private attrs.
    """
    from vllm_mlx.spec_decode.dflash.drafter import MlxVlmDFlashDriver

    runtime = stub_mlx_vlm["runtime"]
    driver = MlxVlmDFlashDriver(
        target_repo="mlx-community/Qwen3.5-9B-4bit",
        drafter_repo="z-lab/Qwen3.5-9B-DFlash",
    )
    assert not driver.loaded
    driver.adopt(
        target="<pre-loaded-target>", processor="<pre-loaded-proc>", runtime=runtime
    )
    assert driver.loaded
    assert driver.target == "<pre-loaded-target>"
    assert driver.processor == "<pre-loaded-proc>"
    assert driver.runtime is runtime

    # generate() works as it does after load().
    list(driver.generate("Hi.", max_tokens=4, temperature=0.0))
    assert stub_mlx_vlm["calls"][0]["model"] == "<pre-loaded-target>"
    assert stub_mlx_vlm["calls"][0]["processor"] == "<pre-loaded-proc>"


def test_driver_adopt_rejects_double_call(stub_mlx_vlm) -> None:
    """Calling ``adopt()`` after ``load()`` (or twice) raises."""
    from vllm_mlx.spec_decode.dflash.drafter import MlxVlmDFlashDriver

    runtime = stub_mlx_vlm["runtime"]
    driver = MlxVlmDFlashDriver(
        target_repo="mlx-community/Qwen3.5-9B-4bit",
        drafter_repo="z-lab/Qwen3.5-9B-DFlash",
    )
    driver.load()
    with pytest.raises(ValueError, match="already loaded"):
        driver.adopt(target="x", processor="y", runtime=runtime)


def test_driver_adopt_rejects_none(stub_mlx_vlm) -> None:
    """``adopt(target=None)`` etc. fail fast — easier than debugging a NoneType error mid-decode."""
    from vllm_mlx.spec_decode.dflash.drafter import MlxVlmDFlashDriver

    driver = MlxVlmDFlashDriver(
        target_repo="mlx-community/Qwen3.5-9B-4bit",
        drafter_repo="z-lab/Qwen3.5-9B-DFlash",
    )
    runtime = stub_mlx_vlm["runtime"]
    with pytest.raises(ValueError, match="non-None"):
        driver.adopt(target=None, processor="p", runtime=runtime)
    with pytest.raises(ValueError, match="non-None"):
        driver.adopt(target="t", processor=None, runtime=runtime)
    with pytest.raises(ValueError, match="non-None"):
        driver.adopt(target="t", processor="p", runtime=None)


def test_driver_rejects_nonpositive_block_size() -> None:
    """``block_size <= 0`` is a programming error; surface it early."""
    from vllm_mlx.spec_decode.dflash.drafter import MlxVlmDFlashDriver

    with pytest.raises(ValueError, match="block_size"):
        MlxVlmDFlashDriver(
            target_repo="mlx-community/Qwen3.5-9B-4bit",
            drafter_repo="z-lab/Qwen3.5-9B-DFlash",
            block_size=0,
        )
    with pytest.raises(ValueError, match="block_size"):
        MlxVlmDFlashDriver(
            target_repo="mlx-community/Qwen3.5-9B-4bit",
            drafter_repo="z-lab/Qwen3.5-9B-DFlash",
            block_size=-1,
        )


# ---------------------------------------------------------------------------
# Symbol-export pin — make sure nothing renames the public surface.
# ---------------------------------------------------------------------------


def test_drafter_module_exports() -> None:
    """``vllm_mlx.spec_decode.dflash.drafter.__all__`` is the stable surface."""
    import vllm_mlx.spec_decode.dflash.drafter as mod

    assert set(mod.__all__) == {
        "BlockDiffusionDrafter",
        "StubBlockDiffusionDrafter",
        "MlxVlmDFlashDriver",
    }


def test_old_class_name_removed() -> None:
    """``MlxVlmBlockDiffusionDrafter`` was the broken pre-#343 wrapper.

    Removed in #343 — keep this test green to catch accidental
    resurrection of the dead per-block adapter API.
    """
    import vllm_mlx.spec_decode.dflash.drafter as mod

    assert not hasattr(mod, "MlxVlmBlockDiffusionDrafter"), (
        "MlxVlmBlockDiffusionDrafter was the broken pre-#343 per-block "
        "adapter (signature mismatch with mlx-vlm 0.6.3). Use "
        "MlxVlmDFlashDriver instead."
    )


# Silence the unused-import linter — ``types`` is reserved for the
# fixture's monkeypatching infrastructure.
_ = types
