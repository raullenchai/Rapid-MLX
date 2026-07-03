# SPDX-License-Identifier: Apache-2.0
"""PR-A of 0.9.13 stack: CLI wiring for ``--spec-decode mtp --mtp-sidecar``.

Coverage for the four surfaces PR-A ships:

1. ``detect_mtp_eligibility(..., has_external_sidecar=True)`` — Gemma 4
   unified base checkpoint (no baked-in MTP head) is promoted from NONE
   to CHAIN when the CLI has resolved a sidecar path. Qwen3.5 / Qwen3.6
   eligibility is unaffected (their MTP head is baked into the target,
   ``--mtp-sidecar`` is a no-op for those).

2. ``vllm_mlx.cli`` argparse — ``--mtp-sidecar PATH`` is present in the
   serve subcommand's ``--help`` and parses without a value error.

3. ``SchedulerConfig.mtp_sidecar`` — round-trips as expected; default
   is ``None`` so pre-0.9.13 callers keep the old behaviour.

4. Engine dispatch call site — the batched engine's ``_start_llm``
   routes through ``dispatch_mtp_inject`` with the sidecar path when
   ``--spec-decode mtp`` + ``--mtp-sidecar`` are set. Verified via
   a monkeypatched dispatch that captures the call args (no real model
   load).

Deliberately out of scope (deferred to PR-B / PR-C):

* Auto-K controller wiring
* Batched residual+bonus verify
* EOS holdout
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 1. detect_mtp_eligibility(has_external_sidecar=...) contract
# ---------------------------------------------------------------------------


def test_detect_sidecar_promotes_gemma4_unified_with_missing_mtp_layers():
    """Base Gemma 4 unified checkpoint (no MTP head) + sidecar → CHAIN.

    The stock ``mlx-community/gemma-4-12B-it-4bit/config.json`` reports
    ``model_type: gemma4_unified`` with no ``mtp_num_hidden_layers``
    key. Without ``--mtp-sidecar``, detection collapses to NONE
    (previous CHECK). With ``--mtp-sidecar``, the ~4-layer assistant
    drafter comes from an external repo so absence of the field is
    expected — detection MUST return CHAIN so ``--spec-decode mtp``
    boots.
    """
    from vllm_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "gemma4_unified"}  # no mtp_num_hidden_layers
    # Pre-PR-A shape: NONE.
    assert detect_mtp_eligibility(config) is MTPEligibility.NONE
    # PR-A shape: sidecar promotes to CHAIN.
    assert (
        detect_mtp_eligibility(config, has_external_sidecar=True)
        is MTPEligibility.CHAIN
    )


def test_detect_sidecar_promotes_gemma4_unified_with_zero_mtp_layers():
    """Explicit ``mtp_num_hidden_layers: 0`` + sidecar → CHAIN too.

    Same shape as the base 12B checkpoint after someone hand-edited
    the config to stamp a zero on it. Sidecar-mode must still allow
    through — the assistant weights come from the external path.
    """
    from vllm_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "gemma4_unified", "mtp_num_hidden_layers": 0}
    assert detect_mtp_eligibility(config) is MTPEligibility.NONE
    assert (
        detect_mtp_eligibility(config, has_external_sidecar=True)
        is MTPEligibility.CHAIN
    )


def test_detect_sidecar_no_effect_on_qwen3_5_missing_mtp():
    """Sidecar mode is scoped to Gemma 4 unified — Qwen3.5 stays NONE.

    Qwen3.5 / Qwen3.6 MTP is baked into the TARGET checkpoint via
    mlx-lm PR #990's sanitize() path. An operator who passes
    ``--mtp-sidecar`` against a Qwen3.5 config with no MTP head still
    needs to re-convert from HF; sidecar mode MUST NOT flip that to
    CHAIN because the assistant-drafter path in ``gemma4_inject.py``
    doesn't know how to graft onto a Qwen3.5 target.
    """
    from vllm_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "qwen3_5", "mtp_num_hidden_layers": 0}
    assert (
        detect_mtp_eligibility(config, has_external_sidecar=True)
        is MTPEligibility.NONE
    )
    # Same for Qwen3.5 MoE.
    config_moe = {"model_type": "qwen3_5_moe", "mtp_num_hidden_layers": 0}
    assert (
        detect_mtp_eligibility(config_moe, has_external_sidecar=True)
        is MTPEligibility.NONE
    )


def test_detect_sidecar_no_effect_on_gemma4_multimodal():
    """Multimodal ``gemma4`` (Gemma4ForConditionalGeneration) — sidecar
    does NOT promote to CHAIN.

    ``gemma4_unified`` is the ONLY lineage on the sidecar-allowlist for
    PR-A because that's the only one with a verified external assistant
    drafter today (``google/gemma-4-*-it-assistant``). Multimodal
    ``gemma4`` (26B-A4B / e2b / e4b) stays NONE regardless of the
    sidecar flag — a future release can add it once the multimodal
    drafter lineage lands.
    """
    from vllm_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "gemma4", "mtp_num_hidden_layers": 0}
    assert (
        detect_mtp_eligibility(config, has_external_sidecar=True)
        is MTPEligibility.NONE
    )


def test_detect_sidecar_leaves_qwen3_5_with_mtp_layers_untouched():
    """Qwen3.5 with mtp_num_hidden_layers >= 1 still returns CHAIN
    regardless of the sidecar flag. Sidecar flag is additive — it
    NEVER downgrades an already-eligible model.
    """
    from vllm_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "qwen3_5", "mtp_num_hidden_layers": 1}
    assert (
        detect_mtp_eligibility(config, has_external_sidecar=False)
        is MTPEligibility.CHAIN
    )
    assert (
        detect_mtp_eligibility(config, has_external_sidecar=True)
        is MTPEligibility.CHAIN
    )


def test_detect_sidecar_default_argument_matches_pre_0913_behaviour():
    """The ``has_external_sidecar`` kwarg defaults to False, preserving
    the pre-0.9.13 rejection contract for every non-CLI caller.

    Regression guard against a future refactor that flips the default to
    True — bench scripts, unit tests, and the CLI eligibility gate all
    rely on the None-argument case being identical to the old ``NONE``
    shape when MTP layers are missing.
    """
    from vllm_mlx.spec_decode.mtp import (
        MTPEligibility,
        detect_mtp_eligibility,
    )

    config = {"model_type": "gemma4_unified", "mtp_num_hidden_layers": 0}
    # No kwarg → old behaviour.
    assert detect_mtp_eligibility(config) is MTPEligibility.NONE
    # Explicit False → same as no kwarg.
    assert (
        detect_mtp_eligibility(config, has_external_sidecar=False)
        is MTPEligibility.NONE
    )


# ---------------------------------------------------------------------------
# 2. CLI argparse for --mtp-sidecar
# ---------------------------------------------------------------------------


def _serve_help_stdout() -> str:
    """Run ``python -m vllm_mlx.cli serve --help`` and return stdout.

    Mirrors ``tests/test_dflash_spec_decode.py::_serve_help_stdout`` —
    same pattern lets us pin the flag without importing the giant CLI
    argparse module in-process (which would drag in torch/mlx-vlm).
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "vllm_mlx.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_cli_serve_help_advertises_mtp_sidecar():
    """``--mtp-sidecar`` appears in ``serve --help`` output.

    Codex round-N regression guard: a prior refactor moved the flag out
    of the serve parser and into a separate ``mtp`` subcommand, silently
    breaking the dogfood invocation ``rapid-mlx serve <model>
    --spec-decode mtp --mtp-sidecar <path>``. Pin the surface here so
    the same regression can't ship again without breaking this test.
    """
    text = _serve_help_stdout()
    assert "--mtp-sidecar" in text, (
        "--mtp-sidecar flag missing from `rapid-mlx serve --help`. "
        "PR-A of 0.9.13 stack ships this flag — check "
        "vllm_mlx/cli.py::serve_parser."
    )


# ---------------------------------------------------------------------------
# 3. SchedulerConfig.mtp_sidecar field
# ---------------------------------------------------------------------------


def test_scheduler_config_mtp_sidecar_default_none():
    """Default matches the argparse default so pre-0.9.13 callers who
    construct ``SchedulerConfig()`` positionally / with defaults keep
    the old (Qwen3.5-only) MTP behaviour."""
    from vllm_mlx.scheduler import SchedulerConfig

    cfg = SchedulerConfig()
    assert cfg.mtp_sidecar is None


def test_scheduler_config_mtp_sidecar_round_trip():
    """Value passed at construction time is retained verbatim.

    Accepts str; ``None`` is the "no sidecar" sentinel.
    """
    from vllm_mlx.scheduler import SchedulerConfig

    cfg = SchedulerConfig(
        spec_decode="mtp",
        mtp_sidecar="google/gemma-4-12B-it-assistant",
    )
    assert cfg.spec_decode == "mtp"
    assert cfg.mtp_sidecar == "google/gemma-4-12B-it-assistant"


def test_scheduler_config_mtp_sidecar_local_path_round_trip():
    """Accepts a local safetensors directory path too.

    Resolution (HF repo id vs local dir) is deferred to the family
    injector — ``SchedulerConfig`` stores the string as-is.
    """
    from vllm_mlx.scheduler import SchedulerConfig

    cfg = SchedulerConfig(
        spec_decode="mtp",
        mtp_sidecar="/tmp/gemma-4-12B-it-assistant",
    )
    assert cfg.mtp_sidecar == "/tmp/gemma-4-12B-it-assistant"


# ---------------------------------------------------------------------------
# 4. Engine dispatch call site — dispatch_mtp_inject sees the sidecar path
# ---------------------------------------------------------------------------


def test_run_dispatch_mtp_inject_forwards_sidecar_path(monkeypatch):
    """``_run_dispatch_mtp_inject`` forwards ``mtp_sidecar`` verbatim
    to ``dispatch_mtp_inject`` after resolving ``model_type`` from HF
    config.

    Uses a monkeypatched dispatch so no real model / weight load runs.
    The captured call args pin the wiring contract:

    * ``model`` is the loaded model object (any duck type).
    * ``model_type`` is the string returned by ``_resolve_hf_model_type``.
    * ``mtp_sidecar`` is passed through as-is.
    """
    from vllm_mlx.engine import batched as _batched

    sentinel_model = object()
    captured: dict = {}

    def _fake_dispatch_mtp_inject(model, model_type, *, mtp_sidecar=None, **kwargs):
        captured["model"] = model
        captured["model_type"] = model_type
        captured["mtp_sidecar"] = mtp_sidecar
        return True

    # ``_run_dispatch_mtp_inject`` imports ``dispatch_mtp_inject`` from
    # ``vllm_mlx.spec_decode.mtp`` (the ``__init__`` re-export). Patch
    # THAT symbol so the internal import inside the function picks up
    # the fake.
    import vllm_mlx.spec_decode.mtp as _mtp

    monkeypatch.setattr(_mtp, "dispatch_mtp_inject", _fake_dispatch_mtp_inject)
    # Force ``_resolve_hf_model_type`` to a deterministic value — we
    # don't want this test to depend on what's cached in the local HF
    # cache (which varies between contributors).
    monkeypatch.setattr(
        _batched,
        "_resolve_hf_model_type",
        lambda name: "gemma4_unified",
    )

    ok = _batched._run_dispatch_mtp_inject(
        sentinel_model,
        "mlx-community/gemma-4-12B-it-4bit",
        "google/gemma-4-12B-it-assistant",
    )
    assert ok is True
    assert captured["model"] is sentinel_model
    assert captured["model_type"] == "gemma4_unified"
    assert captured["mtp_sidecar"] == "google/gemma-4-12B-it-assistant"


def test_run_dispatch_mtp_inject_returns_false_on_unresolvable_model_type(monkeypatch):
    """When ``_resolve_hf_model_type`` returns ``None`` (offline / gated
    repo / hand-rolled path), the dispatch step is skipped cleanly and
    the caller sees ``False`` — engine boot must not abort here.
    """
    from vllm_mlx.engine import batched as _batched
    import vllm_mlx.spec_decode.mtp as _mtp

    called = {"n": 0}

    def _fake_dispatch_mtp_inject(*args, **kwargs):
        called["n"] += 1
        return True

    monkeypatch.setattr(_mtp, "dispatch_mtp_inject", _fake_dispatch_mtp_inject)
    monkeypatch.setattr(_batched, "_resolve_hf_model_type", lambda name: None)

    ok = _batched._run_dispatch_mtp_inject(
        object(),
        "some/unresolvable-repo",
        "google/gemma-4-12B-it-assistant",
    )
    assert ok is False
    assert called["n"] == 0, (
        "dispatch_mtp_inject must NOT be called when model_type is "
        "unresolvable — the caller has no way to pick the family "
        "injector."
    )


def test_run_dispatch_mtp_inject_propagates_none_sidecar(monkeypatch):
    """``mtp_sidecar=None`` (i.e. Qwen3.5 native MTP path — no external
    sidecar) is forwarded through as-is. The family injector
    (``qwen3_5_inject``) then follows its own default (no random init;
    the baked-in MTP head on the target checkpoint is used).
    """
    from vllm_mlx.engine import batched as _batched
    import vllm_mlx.spec_decode.mtp as _mtp

    captured: dict = {}

    def _fake_dispatch_mtp_inject(model, model_type, *, mtp_sidecar=None, **kwargs):
        captured["mtp_sidecar"] = mtp_sidecar
        return True

    monkeypatch.setattr(_mtp, "dispatch_mtp_inject", _fake_dispatch_mtp_inject)
    monkeypatch.setattr(_batched, "_resolve_hf_model_type", lambda name: "qwen3_5")

    ok = _batched._run_dispatch_mtp_inject(
        object(),
        "mlx-community/Qwen3.5-4B-4bit",
        None,
    )
    assert ok is True
    assert captured["mtp_sidecar"] is None
