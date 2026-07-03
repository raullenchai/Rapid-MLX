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
        detect_mtp_eligibility(config, has_external_sidecar=True) is MTPEligibility.NONE
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
        detect_mtp_eligibility(config, has_external_sidecar=True) is MTPEligibility.NONE
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


def test_scheduler_config_mtp_model_type_default_none():
    """Codex round-E blocker #2 regression guard: the new
    ``mtp_model_type`` field defaults to ``None`` so bench-harness /
    direct-SchedulerConfig callers keep the pre-round-E lenient
    behaviour in ``_start_llm``.
    """
    from vllm_mlx.scheduler import SchedulerConfig

    cfg = SchedulerConfig()
    assert cfg.mtp_model_type is None


def test_scheduler_config_mtp_model_type_round_trip():
    """Value passed at construction time is retained verbatim.

    The CLI resolves ``config.json::model_type`` on the asyncio
    thread and threads it through SchedulerConfig so the engine's
    model-load-executor dispatch step does NOT re-read config.json
    (codex round-E fix for the "silent no-op" regression).
    """
    from vllm_mlx.scheduler import SchedulerConfig

    cfg = SchedulerConfig(
        spec_decode="mtp",
        mtp_model_type="gemma4_unified",
    )
    assert cfg.mtp_model_type == "gemma4_unified"


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

    result = _batched._run_dispatch_mtp_inject(
        sentinel_model,
        "mlx-community/gemma-4-12B-it-4bit",
        "google/gemma-4-12B-it-assistant",
    )
    assert result == _batched._DISPATCH_ATTACHED
    assert captured["model"] is sentinel_model
    assert captured["model_type"] == "gemma4_unified"
    assert captured["mtp_sidecar"] == "google/gemma-4-12B-it-assistant"


def test_run_dispatch_mtp_inject_returns_unresolved_when_model_type_missing(
    monkeypatch,
):
    """Codex round-D blocker #1 regression guard: ``_run_dispatch_mtp_inject``
    returns the ``_DISPATCH_UNRESOLVED`` sentinel (NOT ``_DISPATCH_REJECTED``)
    when ``_resolve_hf_model_type`` fails.

    This is the fine-grained routing distinction: ``_DISPATCH_UNRESOLVED``
    means the executor-thread config lookup couldn't find ``config.json``
    (offline HF cache, race with the CLI's asyncio-thread read, hand-
    rolled local path), which is a SOFT-fail; ``_start_llm`` continues
    on plain autoregressive decode. ``_DISPATCH_REJECTED`` — a distinct
    return — is the HARD-fail path.
    """
    import vllm_mlx.spec_decode.mtp as _mtp
    from vllm_mlx.engine import batched as _batched

    called = {"n": 0}

    def _fake_dispatch_mtp_inject(*args, **kwargs):
        called["n"] += 1
        return True

    monkeypatch.setattr(_mtp, "dispatch_mtp_inject", _fake_dispatch_mtp_inject)
    monkeypatch.setattr(_batched, "_resolve_hf_model_type", lambda name: None)

    result = _batched._run_dispatch_mtp_inject(
        object(),
        "some/unresolvable-repo",
        "google/gemma-4-12B-it-assistant",
    )
    assert result == _batched._DISPATCH_UNRESOLVED
    assert result != _batched._DISPATCH_REJECTED, (
        "codex round-D blocker #1 regression: resolution failure MUST NOT "
        "collapse into _DISPATCH_REJECTED — _start_llm hard-raises on "
        "_DISPATCH_REJECTED and that would break offline environments the "
        "CLI already accepted the flag on."
    )
    assert called["n"] == 0, (
        "dispatch_mtp_inject must NOT be called when model_type is "
        "unresolvable — the caller has no way to pick the family "
        "injector."
    )


def test_run_dispatch_mtp_inject_returns_rejected_when_injector_refuses(monkeypatch):
    """Codex round-D blocker #1 regression guard: when the family
    injector is CALLED and returns ``False``, we surface
    ``_DISPATCH_REJECTED`` — the HARD-fail sentinel that
    ``_start_llm`` translates to ``RuntimeError``.

    This is the operator-facing misconfiguration path (bad sidecar,
    wrong assistant model_type, etc.) that MUST not silently fall
    back to plain decode.
    """
    import vllm_mlx.spec_decode.mtp as _mtp
    from vllm_mlx.engine import batched as _batched

    def _fake_dispatch_mtp_inject(*args, **kwargs):
        return False  # family injector rejected

    monkeypatch.setattr(_mtp, "dispatch_mtp_inject", _fake_dispatch_mtp_inject)
    monkeypatch.setattr(
        _batched, "_resolve_hf_model_type", lambda name: "gemma4_unified"
    )

    result = _batched._run_dispatch_mtp_inject(
        object(),
        "mlx-community/gemma-4-12B-it-4bit",
        "/nonexistent/sidecar/path",
    )
    assert result == _batched._DISPATCH_REJECTED, (
        "family-injector rejection MUST surface as _DISPATCH_REJECTED so "
        "_start_llm can raise RuntimeError — silent no-op is unacceptable "
        "for an explicit --spec-decode mtp flag."
    )


def test_run_dispatch_mtp_inject_returns_no_inject_for_unregistered_model_type(
    monkeypatch,
):
    """Codex round-D blocker #1 regression guard: when ``model_type``
    resolves but is not in the dispatch table (plumbing skew between
    the CLI gate and the dispatcher registry), return
    ``_DISPATCH_NO_INJECT`` — a distinct SOFT-fail sentinel that
    ``_start_llm`` treats identically to ``_DISPATCH_UNRESOLVED``.

    Also verifies we do NOT call ``dispatch_mtp_inject`` under this
    path: the module-level helper would just return False (via its
    own "unknown model_type" branch) and we'd lose the distinction
    from a family-injector-refused case.
    """
    import vllm_mlx.spec_decode.mtp as _mtp
    from vllm_mlx.engine import batched as _batched

    called = {"n": 0}

    def _fake_dispatch_mtp_inject(*args, **kwargs):
        called["n"] += 1
        return True

    monkeypatch.setattr(_mtp, "dispatch_mtp_inject", _fake_dispatch_mtp_inject)
    monkeypatch.setattr(
        _batched,
        "_resolve_hf_model_type",
        lambda name: "llama",  # not registered
    )

    result = _batched._run_dispatch_mtp_inject(
        object(),
        "meta-llama/Llama-3.1-8B",
        None,
    )
    assert result == _batched._DISPATCH_NO_INJECT
    assert called["n"] == 0, (
        "dispatch_mtp_inject must NOT be called for an unregistered "
        "model_type — the caller pre-filters via the dispatch table so "
        "we can distinguish this soft-skip from a family-injector "
        "rejection."
    )


def test_run_dispatch_mtp_inject_prefers_cli_provided_model_type(monkeypatch):
    """Codex round-E blocker #2 regression guard: when the caller
    passes ``preferred_model_type``, the dispatch step MUST use it
    verbatim and MUST NOT fall back to reading ``config.json`` on the
    executor thread.

    This is the CLI's escape hatch out of the offline-HF-cache race:
    the CLI has already vetted the model_type on the asyncio thread,
    so re-reading on the executor is both wasteful and racy.
    """
    import vllm_mlx.spec_decode.mtp as _mtp
    from vllm_mlx.engine import batched as _batched

    captured: dict = {}
    resolve_calls = {"n": 0}

    def _fake_dispatch_mtp_inject(model, model_type, *, mtp_sidecar=None, **kwargs):
        captured["model_type"] = model_type
        return True

    def _fake_resolve(*args, **kwargs):
        resolve_calls["n"] += 1
        return "SHOULD_NOT_BE_USED"

    monkeypatch.setattr(_mtp, "dispatch_mtp_inject", _fake_dispatch_mtp_inject)
    monkeypatch.setattr(_batched, "_resolve_hf_model_type", _fake_resolve)

    result = _batched._run_dispatch_mtp_inject(
        object(),
        "mlx-community/gemma-4-12B-it-4bit",
        None,
        preferred_model_type="gemma4_unified",
    )
    assert result == _batched._DISPATCH_ATTACHED
    assert captured["model_type"] == "gemma4_unified"
    assert resolve_calls["n"] == 0, (
        "codex round-E blocker #2 regression: dispatch step re-read "
        "config.json on the executor even though the CLI already "
        "vetted the model_type. This reintroduces the offline-cache "
        "race the round-E fix eliminated."
    )


def test_run_dispatch_mtp_inject_falls_back_when_no_preferred_model_type(monkeypatch):
    """When ``preferred_model_type`` is None (bench-harness path where
    no CLI vetted the config), the dispatch step falls back to
    reading ``config.json`` on the executor thread. This preserves
    pre-round-E behaviour for direct callers.
    """
    import vllm_mlx.spec_decode.mtp as _mtp
    from vllm_mlx.engine import batched as _batched

    captured: dict = {}

    def _fake_dispatch_mtp_inject(model, model_type, *, mtp_sidecar=None, **kwargs):
        captured["model_type"] = model_type
        return True

    monkeypatch.setattr(_mtp, "dispatch_mtp_inject", _fake_dispatch_mtp_inject)
    monkeypatch.setattr(_batched, "_resolve_hf_model_type", lambda name: "qwen3_5")

    result = _batched._run_dispatch_mtp_inject(
        object(),
        "mlx-community/Qwen3.5-4B-4bit",
        None,
        # explicitly None — should fall back to _resolve_hf_model_type
        preferred_model_type=None,
    )
    assert result == _batched._DISPATCH_ATTACHED
    assert captured["model_type"] == "qwen3_5"


def test_run_dispatch_mtp_inject_propagates_none_sidecar(monkeypatch):
    """``mtp_sidecar=None`` (i.e. Qwen3.5 native MTP path — no external
    sidecar) is forwarded through as-is. The family injector
    (``qwen3_5_inject``) then follows its own default (no random init;
    the baked-in MTP head on the target checkpoint is used).
    """
    import vllm_mlx.spec_decode.mtp as _mtp
    from vllm_mlx.engine import batched as _batched

    captured: dict = {}

    def _fake_dispatch_mtp_inject(model, model_type, *, mtp_sidecar=None, **kwargs):
        captured["mtp_sidecar"] = mtp_sidecar
        return True

    monkeypatch.setattr(_mtp, "dispatch_mtp_inject", _fake_dispatch_mtp_inject)
    monkeypatch.setattr(_batched, "_resolve_hf_model_type", lambda name: "qwen3_5")

    result = _batched._run_dispatch_mtp_inject(
        object(),
        "mlx-community/Qwen3.5-4B-4bit",
        None,
    )
    assert result == _batched._DISPATCH_ATTACHED
    assert captured["mtp_sidecar"] is None


# ---------------------------------------------------------------------------
# 4b. Boot-time contract — codex round-D NIT #4: verify that _start_llm
#     interprets the four dispatch return codes correctly.
# ---------------------------------------------------------------------------


def _drive_start_llm_dispatch_gate(dispatch_result, cli_vetted_model_type=None):
    """Exercise the production ``_decide_mtp_dispatch_action`` helper
    that ``_start_llm`` calls after the executor-side dispatch
    completes.

    Codex round-F NIT: earlier revisions of this test suite
    reimplemented the predicate inline, so the tests could pass
    while the production ``_start_llm`` branch silently drifted.
    Fix: import the real production helper and let the tests
    exercise it directly. Now any predicate change in the boot path
    is automatically covered by every test below.

    Returns ``"continued"`` when the helper says the boot should
    proceed on plain autoregressive decode, and raises
    ``RuntimeError`` with the helper's message on the hard-fail
    path — matching what ``_start_llm`` actually does.
    """
    from vllm_mlx.engine import batched as _batched

    action, err_msg = _batched._decide_mtp_dispatch_action(
        dispatch_result,
        cli_vetted_model_type=cli_vetted_model_type,
    )
    if action == "raise":
        raise RuntimeError(err_msg)
    if action == "attached":
        return "attached"
    return "continued"


def test_decide_mtp_dispatch_action_returns_attached_for_attached_result():
    """Codex round-F NIT regression guard: pin the happy-path return
    of the production predicate helper."""
    from vllm_mlx.engine import batched as _batched

    action, msg = _batched._decide_mtp_dispatch_action(
        _batched._DISPATCH_ATTACHED, cli_vetted_model_type=None
    )
    assert action == "attached"
    assert msg is None


def test_decide_mtp_dispatch_action_carries_cli_vetted_model_type_into_error():
    """The hard-fail message includes the CLI-vetted model_type so
    the operator sees exactly which model_type the CLI accepted vs.
    what the dispatcher failed to attach. Pin this in the helper
    directly so a docstring-only refactor can't drop it.
    """
    from vllm_mlx.engine import batched as _batched

    action, msg = _batched._decide_mtp_dispatch_action(
        _batched._DISPATCH_UNRESOLVED,
        cli_vetted_model_type="gemma4_unified",
    )
    assert action == "raise"
    assert msg is not None and "gemma4_unified" in msg


def test_start_llm_raises_runtime_error_on_dispatch_rejected():
    """Codex round-D NIT #4 regression guard: ``_start_llm`` MUST raise
    a startup ``RuntimeError`` when dispatch returns
    ``_DISPATCH_REJECTED`` — the operator's explicit ``--spec-decode
    mtp`` flag was accepted by the CLI and rejected by the family
    injector; silent no-op boot is not an acceptable outcome. The
    hard-fail fires regardless of whether the CLI vetted the
    model_type (round-E) — an active injector rejection is always a
    hard-fail.
    """
    from vllm_mlx.engine import batched as _batched

    for cli_vetted in (None, "gemma4_unified"):
        try:
            _drive_start_llm_dispatch_gate(
                _batched._DISPATCH_REJECTED, cli_vetted_model_type=cli_vetted
            )
        except RuntimeError as e:
            assert "rejected" in str(e).lower()
            continue
        raise AssertionError(
            "codex round-D NIT #4 regression: _start_llm did NOT raise "
            "RuntimeError on _DISPATCH_REJECTED "
            f"(cli_vetted_model_type={cli_vetted!r}) — operator would "
            "boot with MTP silently disabled."
        )


def test_start_llm_continues_on_dispatch_unresolved_when_not_cli_vetted():
    """Codex round-D BLOCKING #1 regression guard (bench-harness path).

    When ``SchedulerConfig.mtp_model_type`` is None — the bench /
    direct-SchedulerConfig caller shape — ``_DISPATCH_UNRESOLVED``
    (executor-thread config lookup missed) MUST fall through to plain
    autoregressive decode. Bench scripts already know the target is
    Qwen3.5 / Gemma 4; they don't want a boot abort on a transient
    HF cache race.

    This preserves the round-D fix for callers that don't set
    ``mtp_model_type``.
    """
    from vllm_mlx.engine import batched as _batched

    result = _drive_start_llm_dispatch_gate(
        _batched._DISPATCH_UNRESOLVED, cli_vetted_model_type=None
    )
    assert result == "continued", (
        "codex round-D BLOCKING #1 regression: _DISPATCH_UNRESOLVED "
        "must NOT abort boot for a caller without mtp_model_type "
        "(bench harness shape)."
    )


def test_start_llm_raises_on_dispatch_unresolved_when_cli_vetted():
    """Codex round-E BLOCKING #2 regression guard.

    When the CLI has populated ``mtp_model_type`` (production
    ``rapid-mlx serve --spec-decode mtp`` path), an executor-thread
    ``_DISPATCH_UNRESOLVED`` return can only be a plumbing bug (the
    executor doesn't even use the fallback config lookup because the
    CLI-vetted value takes precedence). Hard-fail so the operator
    doesn't boot with MTP silently disabled.

    This is the specific behaviour codex round-E BLOCKING #2
    demanded: "unresolved / no-inject cases for explicit MTP" must
    NOT silently continue.
    """
    from vllm_mlx.engine import batched as _batched

    try:
        _drive_start_llm_dispatch_gate(
            _batched._DISPATCH_UNRESOLVED, cli_vetted_model_type="gemma4_unified"
        )
    except RuntimeError as e:
        assert "cli vetted" in str(e).lower() or "vetted model_type" in str(e).lower()
        return
    raise AssertionError(
        "codex round-E BLOCKING #2 regression: _start_llm did NOT raise "
        "RuntimeError on _DISPATCH_UNRESOLVED even though the CLI "
        "vetted model_type. Operator's explicit --spec-decode mtp "
        "would silently boot without MTP."
    )


def test_start_llm_continues_on_dispatch_no_inject_when_not_cli_vetted():
    """Codex round-D + round-E companion: ``_DISPATCH_NO_INJECT``
    without a CLI-vetted model_type is a bench-harness "unknown
    lineage" path. Continue on plain decode; the scheduler's install
    gate also skips.
    """
    from vllm_mlx.engine import batched as _batched

    result = _drive_start_llm_dispatch_gate(
        _batched._DISPATCH_NO_INJECT, cli_vetted_model_type=None
    )
    assert result == "continued"


def test_start_llm_raises_on_dispatch_no_inject_when_cli_vetted():
    """Codex round-E BLOCKING #2 companion: when the CLI vetted
    the model_type, ``_DISPATCH_NO_INJECT`` means the eligibility
    gate and the dispatch table are out of sync — a code bug, not
    an environment issue. Hard-fail so the operator doesn't boot
    with MTP silently disabled.
    """
    from vllm_mlx.engine import batched as _batched

    try:
        _drive_start_llm_dispatch_gate(
            _batched._DISPATCH_NO_INJECT, cli_vetted_model_type="qwen3_5"
        )
    except RuntimeError as e:
        assert "cli vetted" in str(e).lower() or "vetted model_type" in str(e).lower()
        return
    raise AssertionError(
        "codex round-E BLOCKING #2 regression: _start_llm did NOT raise "
        "RuntimeError on _DISPATCH_NO_INJECT even though the CLI "
        "vetted model_type. This is a plumbing skew that operator-"
        "explicit --spec-decode mtp should NOT silently absorb."
    )


class _SyncExecutor:
    """Executor stub that runs submitted callables inline.

    Mirrors just enough of ``concurrent.futures.Executor`` for
    :func:`_apply_mtp_dispatch` to work: ``submit(fn, *args, **kw)``
    returns a completed ``Future`` whose ``.result(timeout=...)``
    yields the return value. Used to exercise the production
    dispatch helper without spinning up a real thread pool.
    """

    def submit(self, fn, /, *args, **kwargs):
        import concurrent.futures as _cf

        f: _cf.Future = _cf.Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except BaseException as e:  # noqa: BLE001
            f.set_exception(e)
        return f


class _TimeoutExecutor:
    """Executor stub whose ``submit(...).result(timeout=T)`` always
    raises ``concurrent.futures.TimeoutError``.

    Used to drive the codex round-G BLOCKING #3 timeout branch in
    :func:`_apply_mtp_dispatch` without a real ``time.sleep``.
    """

    def submit(self, fn, /, *args, **kwargs):
        import concurrent.futures as _cf

        class _NeverFuture:
            @staticmethod
            def result(timeout=None):
                raise _cf.TimeoutError("simulated dispatch hang")

            @staticmethod
            def cancel():
                return True

        return _NeverFuture()


def test_apply_mtp_dispatch_returns_attached_on_happy_path(monkeypatch):
    """Codex round-G NIT #4 regression guard: exercise the production
    :func:`_apply_mtp_dispatch` helper — the exact entry point
    ``_start_llm`` calls — with a fake dispatch that returns
    ``_DISPATCH_ATTACHED``.

    Replaces the earlier ``inspect.getsource()`` string check which
    could pass while runtime behavior drifted.
    """
    from vllm_mlx.engine import batched as _batched
    from vllm_mlx.scheduler import SchedulerConfig

    monkeypatch.setattr(
        _batched,
        "_run_dispatch_mtp_inject",
        lambda *a, **kw: _batched._DISPATCH_ATTACHED,
    )
    sc = SchedulerConfig(spec_decode="mtp", mtp_model_type="gemma4_unified")
    result = _batched._apply_mtp_dispatch(
        model=object(),
        model_name="mlx-community/gemma-4-12B-it-4bit",
        scheduler_config=sc,
        executor=_SyncExecutor(),
    )
    assert result == _batched._DISPATCH_ATTACHED


def test_apply_mtp_dispatch_raises_on_rejected(monkeypatch):
    """Codex round-G NIT #4: end-to-end runtime coverage of the
    hard-fail branch — not a source-string check.

    Behavior: when dispatch returns ``_DISPATCH_REJECTED``,
    :func:`_apply_mtp_dispatch` raises ``RuntimeError`` regardless of
    whether the CLI vetted the model_type.
    """
    from vllm_mlx.engine import batched as _batched
    from vllm_mlx.scheduler import SchedulerConfig

    monkeypatch.setattr(
        _batched,
        "_run_dispatch_mtp_inject",
        lambda *a, **kw: _batched._DISPATCH_REJECTED,
    )
    sc = SchedulerConfig(
        spec_decode="mtp",
        mtp_sidecar="/nonexistent/sidecar",
    )
    try:
        _batched._apply_mtp_dispatch(
            model=object(),
            model_name="mlx-community/gemma-4-12B-it-4bit",
            scheduler_config=sc,
            executor=_SyncExecutor(),
        )
    except RuntimeError as e:
        assert "rejected" in str(e).lower()
        return
    raise AssertionError(
        "codex round-G NIT #4 regression: _apply_mtp_dispatch did NOT "
        "raise RuntimeError on _DISPATCH_REJECTED — the production "
        "hard-fail branch is not being exercised."
    )


def test_apply_mtp_dispatch_raises_when_cli_vetted_and_unresolved(monkeypatch):
    """Codex round-G NIT #4 + round-E cross-check: when the CLI
    vetted the model_type but the executor-side dispatch returns
    ``_DISPATCH_UNRESOLVED``, ``_apply_mtp_dispatch`` must raise —
    this is the exact "silent no-op" regression codex round-E
    demanded be closed.
    """
    from vllm_mlx.engine import batched as _batched
    from vllm_mlx.scheduler import SchedulerConfig

    monkeypatch.setattr(
        _batched,
        "_run_dispatch_mtp_inject",
        lambda *a, **kw: _batched._DISPATCH_UNRESOLVED,
    )
    sc = SchedulerConfig(spec_decode="mtp", mtp_model_type="gemma4_unified")
    try:
        _batched._apply_mtp_dispatch(
            model=object(),
            model_name="mlx-community/gemma-4-12B-it-4bit",
            scheduler_config=sc,
            executor=_SyncExecutor(),
        )
    except RuntimeError as e:
        assert "gemma4_unified" in str(e)
        return
    raise AssertionError(
        "codex round-G NIT #4 regression: _apply_mtp_dispatch did NOT "
        "raise RuntimeError on CLI-vetted _DISPATCH_UNRESOLVED — "
        "operator would boot with MTP silently disabled."
    )


def test_apply_mtp_dispatch_soft_skips_when_not_cli_vetted(monkeypatch):
    """Codex round-G NIT #4 + round-D cross-check: bench-harness path
    (no ``mtp_model_type`` on SchedulerConfig) preserves the round-D
    lenient behaviour — ``_DISPATCH_UNRESOLVED`` continues on plain
    decode instead of aborting boot.
    """
    from vllm_mlx.engine import batched as _batched
    from vllm_mlx.scheduler import SchedulerConfig

    monkeypatch.setattr(
        _batched,
        "_run_dispatch_mtp_inject",
        lambda *a, **kw: _batched._DISPATCH_UNRESOLVED,
    )
    sc = SchedulerConfig(spec_decode="mtp")  # no mtp_model_type — bench shape
    result = _batched._apply_mtp_dispatch(
        model=object(),
        model_name="mlx-community/gemma-4-12B-it-4bit",
        scheduler_config=sc,
        executor=_SyncExecutor(),
    )
    assert result == _batched._DISPATCH_UNRESOLVED


def test_apply_mtp_dispatch_raises_runtime_error_on_timeout(monkeypatch):
    """Codex round-G BLOCKING #3 regression guard.

    A stuck sidecar download / HF hang would previously block server
    startup indefinitely (no timeout on ``future.result()``). Fix:
    ``_apply_mtp_dispatch`` wraps the executor call with a bounded
    timeout and converts a ``TimeoutError`` into a ``RuntimeError``
    with an operator-facing message.
    """
    from vllm_mlx.engine import batched as _batched
    from vllm_mlx.scheduler import SchedulerConfig

    monkeypatch.setenv("RAPID_MLX_MTP_DISPATCH_TIMEOUT_SEC", "1.0")
    sc = SchedulerConfig(spec_decode="mtp", mtp_model_type="gemma4_unified")
    try:
        _batched._apply_mtp_dispatch(
            model=object(),
            model_name="mlx-community/gemma-4-12B-it-4bit",
            scheduler_config=sc,
            executor=_TimeoutExecutor(),
        )
    except RuntimeError as e:
        assert "timed out" in str(e).lower()
        assert "1s" in str(e).replace(" ", "") or "1.0" in str(e) or "1s" in str(e)
        return
    raise AssertionError(
        "codex round-G BLOCKING #3 regression: _apply_mtp_dispatch did "
        "NOT convert a TimeoutError into a startup RuntimeError. A "
        "stuck HF/DNS load would hang `rapid-mlx serve` indefinitely."
    )


def test_get_mtp_dispatch_timeout_sec_default():
    """The dispatch timeout defaults to 600s when the env var is
    unset — long enough for slow 4-16GB assistant downloads on a
    typical residential connection.
    """
    import os

    from vllm_mlx.engine import batched as _batched

    if "RAPID_MLX_MTP_DISPATCH_TIMEOUT_SEC" in os.environ:
        # Test environment cleanup — leave the default codepath
        # unaffected by an operator override in CI.
        del os.environ["RAPID_MLX_MTP_DISPATCH_TIMEOUT_SEC"]
    assert _batched._get_mtp_dispatch_timeout_sec() == 600.0


def test_get_mtp_dispatch_timeout_sec_zero_disables(monkeypatch):
    """An explicit ``0`` in the env var disables the timeout — for
    corp networks where the bounded-wait would false-positive.
    """
    from vllm_mlx.engine import batched as _batched

    monkeypatch.setenv("RAPID_MLX_MTP_DISPATCH_TIMEOUT_SEC", "0")
    assert _batched._get_mtp_dispatch_timeout_sec() is None


def test_get_mtp_dispatch_timeout_sec_malformed_falls_back_to_default(monkeypatch):
    """Bad env var values fall back to the default with a warning
    instead of crashing engine boot.
    """
    from vllm_mlx.engine import batched as _batched

    monkeypatch.setenv("RAPID_MLX_MTP_DISPATCH_TIMEOUT_SEC", "not-a-number")
    assert _batched._get_mtp_dispatch_timeout_sec() == 600.0


def test_start_llm_calls_apply_mtp_dispatch():
    """Codex round-G NIT #4 regression guard: verify that
    ``BatchedEngine._start_llm`` invokes the extracted
    :func:`_apply_mtp_dispatch` helper.

    This complements the runtime-behavior tests above by pinning
    the wiring itself — if a future refactor inlines the dispatch
    call back into ``_start_llm``, this guard forces the change to
    also update the ``_apply_mtp_dispatch`` tests. Without it, a
    silent inline would leave the helper's tests running against a
    dead codepath.

    Uses ``inspect.getsource`` as a lightweight wiring check — the
    behavioral tests above are the actual correctness gate; this
    just prevents drift where the helper survives but stops being
    called.
    """
    import inspect

    from vllm_mlx.engine import batched as _batched

    src = inspect.getsource(_batched.BatchedEngine._start_llm)
    assert "_apply_mtp_dispatch" in src, (
        "codex round-G NIT #4: _start_llm no longer calls "
        "_apply_mtp_dispatch. Either the helper was renamed / inlined "
        "and the behavioral tests above are now covering dead code, or "
        "the boot path was refactored. Update the tests OR restore the "
        "helper invocation."
    )


# ---------------------------------------------------------------------------
# 5. _install_mtp_vendored gate closures (codex round-A findings)
# ---------------------------------------------------------------------------


class _StubBatchGen:
    """Minimum shape of ``BatchGenerator._generation_batch`` needed to
    exercise ``_install_mtp_vendored``'s gate matrix without loading a
    real Qwen3.5 / Gemma 4 checkpoint.

    Codex round-B blocker #3: earlier revision's ``_step`` was a no-op
    stub. That papered over any bug where the wrapper leaked state
    through to the fallthrough — the test wouldn't have caught a
    double-append or missed-sample because the stub didn't model
    mlx-lm's real ``GenerationBatch._step`` bookkeeping.

    This shape now mirrors the pieces of mlx-lm's real ``_step`` the
    wrapper interacts with (see
    ``mlx_lm.generate.GenerationBatch._step`` — cached at
    verification time):

    * Reads ``_next_tokens`` (previously-primed token per uid) and
      appends each element to ``tokens[e]``.
    * Advances ``_next_tokens`` by one — the stub picks the sampled
      value from ``_orig_next_sample`` so tests can inspect what the
      fallthrough emitted.
    * Returns the tokens list + logprobs list, matching the real
      shape ``(List[int], List[mx.array])``.

    The forward pass / model / sampler / cache pieces are elided —
    that's not what these tests validate.
    """

    def __init__(self):
        import mlx.core as mx

        self.uids: list[int] = []
        self.tokens: list[list[int]] = [[]]
        self.logits_processors: list = []
        self.prompt_cache: list = []
        self.max_tokens: list[int] = [4096]
        self._next_tokens = None
        self._next_logprobs: list = []
        self.orig_step_calls = 0
        # What ``_step`` will stash into ``_next_tokens`` after each
        # call — the "next sampled token." Tests can override.
        self._orig_next_sample = mx.array([999], dtype=mx.uint32)
        self._orig_next_logprob = mx.array([0.0])

    def _step(self):
        """Model-side ``mlx_lm.generate.GenerationBatch._step`` mimic.

        Follows the real shape closely enough that any wrapper bug
        involving ``_next_tokens`` reuse or ``tokens`` double-book
        would surface in the observable state.
        """
        import mlx.core as mx

        self.orig_step_calls += 1
        # Real _step reads _next_tokens as the current input, appends
        # each element to tokens[e], samples the next token, and
        # returns the current inputs.
        current = self._next_tokens
        if current is None:
            return [], []
        current_list = [int(current[i].item()) for i in range(current.shape[0])]
        for e, ct in enumerate(current_list):
            self.tokens[e].append(ct)
        # Advance _next_tokens for the next call (matches real
        # _step semantics — asynchronously computed next sample).
        self._next_tokens = self._orig_next_sample
        self._next_logprobs = [self._orig_next_logprob]
        _ = mx.eval  # noqa: F841 — imported to keep parity with real path
        return current_list, self._next_logprobs


class _StubModel:
    """Duck-type ``model`` with the three attributes
    ``_install_mtp_vendored``'s outer gate checks."""

    mtp_forward = object()
    make_mtp_cache = object()
    mtp = object()


def _make_batch_gen_with_gb():
    """Return a ``batch_gen`` shell exposing ``_generation_batch`` so
    the install path binds cleanly."""
    from types import SimpleNamespace

    gb = _StubBatchGen()
    return SimpleNamespace(_generation_batch=gb), gb


def test_install_mtp_vendored_gate_fails_closed_on_missing_request_metadata(
    monkeypatch,
):
    """Codex round-A blocker #1 regression guard.

    Prior revision returned ``True`` from ``_is_greedy_for_uid`` when
    ``requests`` / ``uid_to_request_id`` were unresolvable — that
    silently applied greedy sampling to any request whose bookkeeping
    had just been evicted. The fix flips the default to ``False`` so
    the caller falls through to ``_orig_step()`` (which reads the real
    sampler).

    We can't easily exercise the closure directly (it's local to
    ``_install_mtp_vendored``). But we CAN observe the outer contract:
    when ``requests=None`` and there's a single-uid batch, the patched
    ``_step`` MUST fall through to ``_orig_step()`` — not enter the
    MTP construction path — because the gate now returns False.
    """
    from vllm_mlx.scheduler import _install_mtp_vendored

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [42]  # single uid — passes the B==1 gate

    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests=None,
        uid_to_request_id=None,
    )
    assert ok is True

    # Fire the patched _step. With requests=None, _is_greedy_for_uid
    # must return False → fallthrough to _orig_step. Pre-fix the gate
    # returned True and we would have entered the mtp_generate_step
    # construction path.
    gb._step()
    stats = batch_gen._mtp_vendored_stats
    assert stats["fallthrough_steps"] >= 1
    assert stats["ft_non_greedy"] >= 1, (
        "codex round-A blocker #1 regression: gate did not fall closed "
        "when request bookkeeping is unresolvable"
    )
    assert gb.orig_step_calls == 1


def test_install_mtp_vendored_cleans_up_state_on_fallthrough_batch_size(monkeypatch):
    """Codex round-A blocker #3 regression guard.

    A uid that ran MTP for a while then transitions to a B>1 batch
    (or non-greedy, or logits-processors) fell through to
    ``_orig_step()`` — but the per-uid MTP state stayed live. On the
    next single-uid greedy step for the same uid, the stale generator
    would resume with a ``prompt_cache`` view one position stale
    behind the actual live cache, silently emitting wrong tokens.
    Fix: cleanup on every fallthrough branch.

    Codex round-B blocker: the earlier revision of this test never
    reached the successful first-call path, so ``_state`` was empty
    and the cleanup call at the B>1 branch was a no-op. The test
    would have passed even with the cleanup removed. Fix: monkeypatch
    ``mtp_generate_step`` to a fake iterator so the first call
    populates ``_state[uid]``, drive one warm decode call, then
    trigger the fallthrough. Prove the generator was ``.close()``-d
    (side-effect observable) AND that a subsequent single-uid call
    re-enters the FIRST-call path (i.e. constructs a fresh generator).
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from vllm_mlx.scheduler import _install_mtp_vendored
    from vllm_mlx.spec_decode.mtp import generator as _gen_mod

    fake_gen_calls = {"constructed": 0, "closed": 0}

    class _FakeGen:
        def __init__(self):
            fake_gen_calls["constructed"] += 1
            self._n = 0

        def __iter__(self):
            return self

        def __next__(self):
            # Emit a bogus token each call — the test never checks its
            # value, only that we get through iteration.
            self._n += 1
            return (self._n + 1000, mx.array([0.0]), False)

        def close(self):
            fake_gen_calls["closed"] += 1

    def _fake_mtp_generate_step(*args, **kwargs):
        return _FakeGen()

    monkeypatch.setattr(_gen_mod, "mtp_generate_step", _fake_mtp_generate_step)

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [7]
    request_stub = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-7": request_stub},
        uid_to_request_id={7: "req-7"},
    )
    assert ok is True

    gb._next_tokens = mx.array([500], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # First call — construct the fake generator and populate _state[7].
    gb._step()
    assert fake_gen_calls["constructed"] == 1
    assert fake_gen_calls["closed"] == 0

    # Second call in the SAME warm state — draining the queue.
    gb._step()
    assert fake_gen_calls["closed"] == 0

    # Now transition to B=2. The B>1 fallthrough branch must call
    # _cleanup_uid on the stale uid, which closes the fake generator.
    gb.uids = [1, 2]
    gb._step()
    stats = batch_gen._mtp_vendored_stats
    assert stats["ft_batch_size"] >= 1
    assert fake_gen_calls["closed"] >= 1, (
        "codex round-A blocker #3 regression: B>1 fallthrough did not "
        "clean up the stale per-uid MTP state"
    )

    # Back to a single-uid batch — must re-enter FIRST call path
    # (proving state was cleared). If cleanup was missed the resume
    # path would reuse the same fake generator instance.
    gb.uids = [7]
    gb._next_tokens = mx.array([501], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]
    gb._step()
    assert fake_gen_calls["constructed"] == 2, (
        "cleanup contract broken: single-uid step after fallthrough "
        "did not re-enter the first-call path (would silently resume "
        "a stale generator)."
    )


def test_install_mtp_vendored_first_call_construction_failure_does_not_double_book(
    monkeypatch,
):
    """Codex round-A blocker #2 regression guard.

    Prior revision appended the first token to ``gb.tokens[0]`` BEFORE
    constructing the generator. When ``mtp_generate_step(...)`` raised
    (missing dep, weight-shape mismatch, etc.), the fallthrough path
    then called ``_orig_step()`` which appends the SAME token again,
    double-booking bookkeeping and duplicating the token in the emitted
    stream.

    Fix: construct the generator first, only mutate ``gb.tokens`` on
    success. On construction failure the fallthrough path calls
    ``_orig_step`` on a clean ``tokens`` list.

    Implementation note: ``mtp_generate_step`` is imported lazily
    inside ``_install_mtp_vendored`` via a ``from … import …`` and is
    then captured by the closure that patches ``_step``. Any patch has
    to be installed on the source module BEFORE the install call runs
    so the from-import picks up the fake; a post-install monkeypatch
    would target the module attribute but not the closure's local
    binding.
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from vllm_mlx.scheduler import _install_mtp_vendored
    from vllm_mlx.spec_decode.mtp import generator as _gen_mod

    def _raising_generator(*args, **kwargs):
        raise RuntimeError("simulated generator construction failure")

    monkeypatch.setattr(_gen_mod, "mtp_generate_step", _raising_generator)

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [99]

    # Provide a sampling_params.temperature=0.0 stub so the greedy
    # gate passes (we want to reach the first-call construction path).
    request_stub = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-99": request_stub},
        uid_to_request_id={99: "req-99"},
    )
    assert ok is True

    # Simulate mlx-lm's original _step having primed the first token
    # into ``_next_tokens`` — a 1-D mx.array of length 1 with a real
    # int payload. The realistic stub (_StubBatchGen._step) mirrors
    # mlx-lm's real _step in ``gb.tokens[0].append(int(inputs[0]))``,
    # so the exact double-book bug the codex round-A fix addressed
    # would manifest as a length-2 tokens list with 12345 repeated.
    gb._next_tokens = mx.array([12345], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    gb._step()

    # Fallthrough happened → _orig_step ran exactly once, which does
    # ONE ``tokens[0].append(first_tok)`` per mlx-lm's real shape.
    # Under the round-A pre-fix, our wrapper would ALSO have appended
    # first_tok before construction — leaving gb.tokens[0] == [first,
    # first]. Codex round-B blocker #3: this assertion now runs
    # against the mlx-lm-shaped stub, so it can actually observe the
    # double-book.
    assert gb.orig_step_calls == 1
    assert gb.tokens[0] == [12345], (
        f"codex round-A blocker #2 regression: gb.tokens[0] = "
        f"{gb.tokens[0]!r} (expected [12345] — one append from "
        "_orig_step, none from our wrapper's pre-construction append)."
    )
    stats = batch_gen._mtp_vendored_stats
    assert stats["fallthrough_steps"] >= 1


def test_install_mtp_vendored_first_call_failure_disables_subsequent_calls(monkeypatch):
    """Codex round-D blocker #2 regression guard.

    Under a deterministic first-call construction failure (bad sidecar,
    weight-shape mismatch, etc.), the wrapper's original
    ``state is None`` branch would re-run the failing ``try/except``
    every step — one construction attempt per token, effectively DoSing
    the request while never getting any MTP benefit.

    Fix: track ``_disabled_uids`` and short-circuit to ``_orig_step``
    once construction has failed for a given uid. This test drives
    two ``_step()`` calls under a deterministically-failing generator
    constructor and asserts:

    * The first call attempts construction (raises internally → falls
      through to ``_orig_step``).
    * The second call does NOT re-attempt construction — the
      ``mtp_generate_step`` monkeypatch's counter stays at 1.
    * Both calls advance ``_orig_step`` correctly (no double-book).
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from vllm_mlx.scheduler import _install_mtp_vendored
    from vllm_mlx.spec_decode.mtp import generator as _gen_mod

    construction_attempts = {"n": 0}

    def _raising_generator(*args, **kwargs):
        construction_attempts["n"] += 1
        raise RuntimeError("simulated persistent construction failure")

    monkeypatch.setattr(_gen_mod, "mtp_generate_step", _raising_generator)

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [77]
    request_stub = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-77": request_stub},
        uid_to_request_id={77: "req-77"},
    )
    assert ok is True

    gb._next_tokens = mx.array([500], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]
    gb._orig_next_sample = mx.array([501], dtype=mx.uint32)

    # First call — construction is attempted, fails, fall through.
    gb._step()
    assert construction_attempts["n"] == 1
    stats = batch_gen._mtp_vendored_stats
    assert stats["fallthrough_steps"] >= 1

    # Second call — must short-circuit via the disabled-uid path.
    # No new construction attempt.
    gb._orig_next_sample = mx.array([502], dtype=mx.uint32)
    gb._step()
    assert construction_attempts["n"] == 1, (
        "codex round-D blocker #2 regression: wrapper retried "
        f"construction after a first-call failure "
        f"(attempts={construction_attempts['n']!r}). It must mark the "
        "uid as disabled and delegate directly to _orig_step for the "
        "rest of the request."
    )
    stats = batch_gen._mtp_vendored_stats
    assert stats.get("ft_disabled", 0) >= 1, (
        "codex round-D blocker #2 regression: the second _step call did "
        "not hit the disabled-uid short-circuit — check the "
        "_disabled_uids gate ordering vs. _is_greedy_for_uid."
    )
    # And _orig_step ran twice — once per _step() call.
    assert gb.orig_step_calls == 2


def test_install_mtp_vendored_disabled_uid_cleared_on_uid_reuse(monkeypatch):
    """Codex round-E blocker #1 regression guard.

    mlx-lm reuses uid ints once a request completes. The round-D
    ``_disabled_uids`` fix keyed disable state by uid alone; that
    let a bad-sidecar disable from request N silently apply to
    request N+1, N+2, ... if they happened to draw the same uid,
    permanently disabling MTP after a single bad request.

    Fix: store the request_id at disable time. When the same uid
    shows up with a DIFFERENT request_id, the disable is stale —
    clear it and re-enter the normal MTP path.

    This test:
      1. Drives request A (uid=42, req-A) through a first-call
         construction failure — uid=42 lands in _disabled_uids.
      2. Simulates uid=42 being reused for request B (req-B) with
         a working generator constructor.
      3. Verifies that the wrapper does NOT stay in the disabled
         short-circuit — it re-enters the FIRST-call path and
         successfully seeds a fresh generator for request B.
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from vllm_mlx.scheduler import _install_mtp_vendored
    from vllm_mlx.spec_decode.mtp import generator as _gen_mod

    class _RecoveringCtor:
        """First construction raises; subsequent calls yield a fake
        generator. Simulates "request A had a bad sidecar path,
        request B was retargeted at a working path."
        """

        def __init__(self):
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated request-A sidecar failure")
            return _FakeGen()

    class _FakeGen:
        def __init__(self):
            self._n = 0

        def __iter__(self):
            return self

        def __next__(self):
            self._n += 1
            return (5000 + self._n, mx.array([0.0]), False)

        def close(self):
            pass

    ctor = _RecoveringCtor()
    monkeypatch.setattr(_gen_mod, "mtp_generate_step", ctor)

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [42]
    uid_to_request_id: dict[int, str] = {42: "req-A"}
    requests: dict = {
        "req-A": SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0)),
    }
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests=requests,
        uid_to_request_id=uid_to_request_id,
    )
    assert ok is True

    gb._next_tokens = mx.array([1], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # Request A step 1 — construction fails, uid=42 goes into _disabled_uids
    # keyed by req-A.
    gb._step()
    assert ctor.calls == 1
    stats = batch_gen._mtp_vendored_stats
    assert stats["fallthrough_steps"] >= 1

    # Request A step 2 — still req-A, so the disabled short-circuit
    # fires; ctor is NOT called again.
    gb._orig_next_sample = mx.array([2], dtype=mx.uint32)
    gb._step()
    assert ctor.calls == 1
    assert stats.get("ft_disabled", 0) >= 1

    # Now simulate request A completing and uid=42 being reused for
    # request B. mlx-lm would update uid_to_request_id to the new
    # request's ID.
    uid_to_request_id[42] = "req-B"
    requests["req-B"] = SimpleNamespace(
        sampling_params=SimpleNamespace(temperature=0.0)
    )
    gb._next_tokens = mx.array([100], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # Request B step 1 — request_id changed, disabled state MUST be
    # cleared and the wrapper MUST re-enter the FIRST-call path.
    gb._step()
    assert ctor.calls == 2, (
        "codex round-E blocker #1 regression: uid=42 was reused for "
        f"a new request (req-B), but the wrapper stayed in the "
        "disabled short-circuit and did not attempt fresh MTP "
        f"construction (ctor.calls={ctor.calls!r}). This lets one "
        "bad-sidecar disable permanently downgrade every subsequent "
        "request that draws the same uid."
    )


def test_install_mtp_vendored_cleanup_does_not_clear_disabled_uids(monkeypatch):
    """Codex round-G BLOCKING #1 regression guard.

    Earlier revision's ``_cleanup_uid`` unconditionally popped
    ``_disabled_uids[uid]``, which meant any fallthrough branch (B>1
    transition, non-greedy switch, logits-processors override) that
    called ``_cleanup_uid`` would silently un-disable a uid — the
    next single-uid greedy call would then retry MTP construction
    and hit the same broken path all over again, one construction
    attempt per token.

    Fix: ``_cleanup_uid`` no longer touches ``_disabled_uids``.
    The disable state is a per-REQUEST marker cleared only by
    (a) uid reuse detection with a new request_id, or (b) explicit
    delete in the reuse-gate branch. State (the generator + queue)
    is still cleaned by ``_cleanup_uid`` — that's per-generator
    lifecycle, not per-request.

    This test:
      1. Drives a first-call construction failure → uid=99 lands
         in ``_disabled_uids`` keyed by req-A.
      2. Triggers a B>1 fallthrough (which calls ``_cleanup_uid``
         for stale uids in ``_state``).
      3. Returns to B=1 single-uid and drives another step.
      4. Asserts that MTP construction is NOT retried — the
         disable marker survived the cleanup.
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from vllm_mlx.scheduler import _install_mtp_vendored
    from vllm_mlx.spec_decode.mtp import generator as _gen_mod

    construction_attempts = {"n": 0}

    def _raising_ctor(*args, **kwargs):
        construction_attempts["n"] += 1
        raise RuntimeError("simulated persistent construction failure")

    monkeypatch.setattr(_gen_mod, "mtp_generate_step", _raising_ctor)

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [99]
    request_stub = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-99": request_stub},
        uid_to_request_id={99: "req-99"},
    )
    assert ok is True

    gb._next_tokens = mx.array([100], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # Step 1 — construction fails, uid=99 disabled.
    gb._step()
    assert construction_attempts["n"] == 1

    # Force a B>1 fallthrough — this calls _cleanup_uid for every
    # uid in _state. Under the round-G BLOCKING #1 pre-fix this
    # would ALSO have popped _disabled_uids[99].
    gb.uids = [99, 100]
    gb._step()
    stats = batch_gen._mtp_vendored_stats
    assert stats.get("ft_batch_size", 0) >= 1

    # Return to B=1 same uid; if _cleanup_uid cleared the disable
    # (pre-fix), the wrapper would retry construction here. Post-
    # fix, the disable marker is intact and we short-circuit.
    gb.uids = [99]
    gb._next_tokens = mx.array([200], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]
    gb._step()
    assert construction_attempts["n"] == 1, (
        "codex round-G BLOCKING #1 regression: _cleanup_uid cleared "
        "_disabled_uids on a B>1 fallthrough. Next single-uid step "
        "retried MTP construction "
        f"(attempts={construction_attempts['n']!r})."
    )


def test_install_mtp_vendored_stop_iteration_disables_uid_before_raise(monkeypatch):
    """Codex round-G BLOCKING #2 regression guard (StopIteration branch).

    On ``StopIteration`` mid-stream, the wrapper must:
    (a) record the current request_id in ``_disabled_uids`` so any
        retry short-circuits to plain decode; and
    (b) raise ``RuntimeError`` so mlx-lm surfaces the failure.

    Earlier revision called ``_cleanup_uid`` which cleared the
    disable, meaning a retry on the same uid+request_id would re-
    enter FIRST-call construction and hit the same bug.
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from vllm_mlx.scheduler import _install_mtp_vendored
    from vllm_mlx.spec_decode.mtp import generator as _gen_mod

    class _EmptyGen:
        """Yields nothing — first next() call raises StopIteration."""

        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self):
            pass

    monkeypatch.setattr(_gen_mod, "mtp_generate_step", lambda *a, **kw: _EmptyGen())

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [88]
    request_stub = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-88": request_stub},
        uid_to_request_id={88: "req-88"},
    )
    assert ok is True

    gb._next_tokens = mx.array([777], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # First call — construct + emit first_gen_tok = 777, populates
    # _state[88].
    gb._step()

    # Second call — draining the queue is empty, pulls from _EmptyGen
    # which raises StopIteration. Wrapper must record 88 in
    # _disabled_uids (with req-88 as the marker) before raising.
    try:
        gb._step()
    except RuntimeError as e:
        assert (
            "generator exhausted" in str(e).lower()
            or "stopiteration" in str(e).lower()
            or "before mlx-lm hit" in str(e).lower()
        )
        # Simulate a retry: if mlx-lm re-enters _mtp_step with the
        # same uid+request_id, the disable marker MUST fire and
        # short-circuit to _orig_step (not re-enter construction).
        # This can happen if the caller uses the exception as
        # "back off then retry" rather than propagating.
        gb._next_tokens = mx.array([500], dtype=mx.uint32)
        gb._next_logprobs = [mx.array([0.0])]
        gb.uids = [88]
        pre_retry_orig_step_calls = gb.orig_step_calls
        gb._step()
        # The wrapper hit the disable short-circuit and called
        # _orig_step. NOT a fresh construction attempt.
        assert gb.orig_step_calls == pre_retry_orig_step_calls + 1, (
            "codex round-G BLOCKING #2 regression: retry on the same "
            "uid+request_id after a StopIteration failure did NOT hit "
            "the disable short-circuit."
        )
        stats = batch_gen._mtp_vendored_stats
        assert stats.get("ft_disabled", 0) >= 1
        return
    raise AssertionError(
        "codex round-G BLOCKING #2 regression: wrapper did NOT raise "
        "RuntimeError on internal generator StopIteration."
    )


def test_install_mtp_vendored_mid_stream_generator_failure_raises(monkeypatch):
    """Codex round-D blocker #3 regression guard.

    Mid-stream failure of the internal ``mtp_generate_step`` generator
    cannot fall back to plain ``_orig_step`` because the wrapper never
    updates ``gb._next_tokens`` — it still holds ``first_gen_tok`` from
    the priming ``_step``. A silent fallback would emit
    ``first_gen_tok`` AGAIN, corrupting the output stream.

    Fix: re-raise as ``RuntimeError`` so mlx-lm surfaces the failure
    to the caller cleanly.

    This test constructs a generator that yields once (the first
    subsequent-call sample) and then raises on the second ``next()``,
    then asserts the wrapper propagates the failure instead of
    delegating to ``_orig_step``.
    """
    from types import SimpleNamespace

    import mlx.core as mx

    from vllm_mlx.scheduler import _install_mtp_vendored
    from vllm_mlx.spec_decode.mtp import generator as _gen_mod

    class _MidStreamFailingGen:
        def __init__(self):
            self._n = 0

        def __iter__(self):
            return self

        def __next__(self):
            self._n += 1
            if self._n <= 1:
                return (2001, mx.array([0.0]), False)
            raise RuntimeError("simulated mid-stream generator failure")

        def close(self):
            pass

    def _mid_stream_failing_ctor(*args, **kwargs):
        return _MidStreamFailingGen()

    monkeypatch.setattr(_gen_mod, "mtp_generate_step", _mid_stream_failing_ctor)

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [55]
    request_stub = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    ok = _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={"req-55": request_stub},
        uid_to_request_id={55: "req-55"},
    )
    assert ok is True

    gb._next_tokens = mx.array([1000], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]

    # First call — construct, emit first_gen_tok = 1000.
    gb._step()

    # Second call — pulls from generator, yields 2001.
    gb._step()

    # Third call — generator raises. MUST propagate as RuntimeError
    # rather than falling back to _orig_step (which would emit 1000
    # again and duplicate the token stream).
    orig_step_calls_before = gb.orig_step_calls
    try:
        gb._step()
    except RuntimeError as e:
        assert "mid-stream" in str(e).lower() or "generator raised" in str(e).lower()
        # _orig_step must NOT have been called on the failure branch.
        assert gb.orig_step_calls == orig_step_calls_before, (
            "codex round-D blocker #3 regression: wrapper delegated to "
            "_orig_step on mid-stream generator failure, which duplicates "
            f"first_gen_tok in the output stream "
            f"(orig_step_calls: {orig_step_calls_before} -> "
            f"{gb.orig_step_calls})."
        )
        stats = batch_gen._mtp_vendored_stats
        assert stats.get("gen_raised", 0) >= 1
        return
    raise AssertionError(
        "codex round-D blocker #3 regression: wrapper did NOT raise on "
        "mid-stream generator failure. Falling back to _orig_step here "
        "would emit first_gen_tok twice (duplicated) because _next_"
        "tokens is stale relative to what the vendored path already "
        "emitted."
    )
