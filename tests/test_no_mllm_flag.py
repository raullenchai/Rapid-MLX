# SPDX-License-Identifier: Apache-2.0
"""
Tests for the --no-mllm / --text-only escape hatch (#393).

Some HuggingFace model repos ship a `config.json` that declares
multimodal capabilities (e.g. `vision_config` block) but the actual
safetensors only contain text-model weights — a partial quant, a
text-only fork, or a checkpoint that was uploaded before vision shards
were finalized. Auto-detection (`is_mllm_model`) correctly identifies
the config as multimodal-capable, but the load path then crashes inside
mlx_vlm with `ValueError: Missing N parameters: vision_tower.*`.

`--no-mllm` (alias `--text-only`) is the user-facing escape hatch:
force the text path even when auto-detection would route to MLLM.

These tests verify:
1. BatchedEngine respects force_text=True (skips is_mllm_model probe).
2. force_text and force_mllm are not both honored — server.load_model
   raises ValueError if both are passed.
3. The friendly-error wrapper in MLLMModel.load() catches the
   missing-vision-tensor ValueError and re-raises as RuntimeError that
   mentions `--no-mllm`.
"""

from __future__ import annotations

import pytest


def test_force_text_overrides_auto_detection(monkeypatch):
    """When force_text=True, BatchedEngine._is_mllm is False even if
    is_mllm_model would return True. Verifies the probe is short-
    circuited (not just overridden later) by checking it isn't called."""
    from vllm_mlx.engine import batched as batched_mod

    probe_calls = []

    def _fake_is_mllm_model(name):
        probe_calls.append(name)
        return True  # would normally route to MLLM

    monkeypatch.setattr(batched_mod, "is_mllm_model", _fake_is_mllm_model)

    engine = batched_mod.BatchedEngine(
        model_name="fake/model-name",
        force_text=True,
    )

    assert engine._is_mllm is False, (
        "force_text=True must override auto-detection to False"
    )
    assert probe_calls == [], (
        "force_text=True should short-circuit the probe entirely; "
        f"is_mllm_model was called for: {probe_calls}"
    )


def test_force_mllm_still_works_when_force_text_is_false():
    """Regression: adding force_text must not break force_mllm."""
    from vllm_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine(
        model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
        force_mllm=True,
        force_text=False,
    )
    assert engine._is_mllm is True


def test_force_text_is_keyword_only_in_load_model():
    """Regression: ``force_text`` must remain keyword-only so existing
    positional callers (e.g. ``load_model(name, None, 1, 32768, False,
    0.5)`` setting ``gpu_memory_utilization=0.5``) don't suddenly
    pass that float as a truthy ``force_text``. Codex R2 caught this
    on the original PR — the original placement after ``force_mllm``
    shifted every subsequent positional arg by one slot."""
    import inspect

    from vllm_mlx.server import load_model

    sig = inspect.signature(load_model)
    assert sig.parameters["force_text"].kind == inspect.Parameter.KEYWORD_ONLY, (
        "force_text must be KEYWORD_ONLY to preserve positional-arg "
        "compatibility for downstream callers — see codex R2 on PR #407."
    )

    from vllm_mlx.engine.batched import BatchedEngine

    sig = inspect.signature(BatchedEngine.__init__)
    assert sig.parameters["force_text"].kind == inspect.Parameter.KEYWORD_ONLY, (
        "BatchedEngine.__init__ force_text must be KEYWORD_ONLY too."
    )


def test_force_text_and_force_mllm_mutually_exclusive_in_load_model():
    """server.load_model raises ValueError if both flags are True. This
    is the second line of defense — CLI already rejects this via
    sys.exit(2), but load_model is also a public entry point so guard
    here too."""
    from vllm_mlx.server import load_model

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_model(
            "fake/model",
            force_mllm=True,
            force_text=True,
        )


def test_friendly_error_on_missing_vision_tensors(monkeypatch):
    """MLLMModel.load() must translate mlx_vlm's
    `ValueError: Missing N parameters: vision_tower.*` into a RuntimeError
    that mentions --no-mllm, so users find the escape hatch without
    grepping the source. Verifies the wrapper fires only on the
    vision-shaped missing-parameter signature."""
    import importlib
    import sys

    # mlx_vlm may not be installed (vision extra is opt-in). The wrapper
    # logic lives in MLLMModel.load, which doesn't need mlx_vlm to be
    # importable for the catch path. But we DO need mlx_vlm to satisfy
    # the `_require_mlx_vlm()` precondition. Skip cleanly if absent.
    try:
        importlib.import_module("mlx_vlm")
    except ImportError:
        pytest.skip("mlx_vlm not installed (vision extra)")

    from vllm_mlx.models import mllm as mllm_mod

    # Inject a fake mlx_vlm.load that raises the M5-style missing-tensor
    # ValueError. We poke sys.modules so the `from mlx_vlm import load`
    # inside MLLMModel.load() picks up our fake.
    real_mlx_vlm = sys.modules["mlx_vlm"]

    class _FakeMlxVlm:
        @staticmethod
        def load(_name):
            raise ValueError(
                "Missing 60 parameters: \n"
                "vision_tower.blocks.27.attn.proj.bias,\n"
                "vision_tower.blocks.27.attn.proj.weight,\n"
                "vision_tower.blocks.27.attn.qkv.bias."
            )

    class _FakeMlxVlmUtils:
        @staticmethod
        def load_config(_name):
            return {}

    monkeypatch.setitem(sys.modules, "mlx_vlm", _FakeMlxVlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", _FakeMlxVlmUtils)

    # Avoid the global instance count guard
    inst = mllm_mod.MLXMultimodalLM(model_name="fake/incomplete-vlm")

    try:
        with pytest.raises(RuntimeError) as excinfo:
            inst.load()

        msg = str(excinfo.value)
        assert "--no-mllm" in msg, (
            f"Friendly error must mention --no-mllm; got: {msg!r}"
        )
        assert "#393" in msg, "Friendly error must reference #393 for searchability"
        assert "60 vision tensors missing" in msg, (
            "Friendly error must surface the count from the underlying error"
        )
    finally:
        # Restore original mlx_vlm so subsequent tests aren't poisoned.
        sys.modules["mlx_vlm"] = real_mlx_vlm


def test_auto_routing_flags_have_force_on_and_force_off_pair():
    """SOP gate (#393 lesson): every binary auto-routing decision must
    expose BOTH a force-on and a force-off CLI flag.

    The pattern that bit us with #393 — ``--mllm`` (force on) shipped
    without a paired ``--no-mllm`` (force off) — applies to every
    auto-detection path. False positives are inevitable (incomplete
    quants, custom forks, hardware-shaped edge cases); when we hit one
    the user needs an escape hatch *immediately*, not in a follow-up
    release that ships ~2 weeks later.

    Registry below is the source of truth: every routing flag we expose
    must appear with both directions. Adding a new auto-detection
    *requires* adding both flags and registering them here. If someone
    in the future removes one direction of the pair, CI fails with a
    pointer to this principle.

    Past incidents this rule would have caught:
      - #393: ``--mllm`` had no inverse → Tylast had to wait for a
        patch release instead of overriding from his launchd plist.
      - #404 (related, hardware-side): no user override for MLX stream
        capability, only an internal probe. The bug went undetected on
        every chip family we don't own.
    """
    import importlib.resources
    import pathlib

    pkg_root = pathlib.Path(
        str(importlib.resources.files("vllm_mlx").joinpath(""))
    ).resolve()
    cli_source = (pkg_root / "cli.py").read_text()
    server_source = (pkg_root / "server.py").read_text()

    # Every entry: (force-on flag, force-off flag, what it routes).
    # When you add a new auto-detection, add the pair here too.
    AUTO_ROUTING_FLAG_PAIRS = [
        ("--mllm", "--no-mllm", "MLLM vs text-only routing (#393)"),
    ]

    missing = []
    for force_on, force_off, desc in AUTO_ROUTING_FLAG_PAIRS:
        on_quoted = f'"{force_on}"'
        off_quoted = f'"{force_off}"'
        if on_quoted not in cli_source and on_quoted not in server_source:
            missing.append(f"force-on flag {force_on} missing ({desc})")
        if off_quoted not in cli_source and off_quoted not in server_source:
            missing.append(
                f"force-off flag {force_off} missing ({desc}) — "
                "every binary auto-routing decision needs an escape hatch "
                "in BOTH directions; see #393 for the past incident this "
                "rule encodes."
            )
    assert not missing, "\n".join(missing)


def test_friendly_error_does_not_swallow_unrelated_valueerror(monkeypatch):
    """An unrelated ValueError (e.g. config parsing) must NOT trigger
    the friendly-error path — it should propagate as-is so genuine bugs
    surface and don't get misattributed to vision-tower issues."""
    import importlib
    import sys

    try:
        importlib.import_module("mlx_vlm")
    except ImportError:
        pytest.skip("mlx_vlm not installed (vision extra)")

    from vllm_mlx.models import mllm as mllm_mod

    real_mlx_vlm = sys.modules["mlx_vlm"]

    class _FakeMlxVlm:
        @staticmethod
        def load(_name):
            raise ValueError("config.json has an invalid model_type field")

    class _FakeMlxVlmUtils:
        @staticmethod
        def load_config(_name):
            return {}

    monkeypatch.setitem(sys.modules, "mlx_vlm", _FakeMlxVlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", _FakeMlxVlmUtils)

    inst = mllm_mod.MLXMultimodalLM(model_name="fake/bad-config")

    try:
        with pytest.raises(ValueError, match="invalid model_type"):
            inst.load()
    finally:
        sys.modules["mlx_vlm"] = real_mlx_vlm
