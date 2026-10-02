# SPDX-License-Identifier: Apache-2.0
"""0.10.16 dogfood P1 (④) — a base-wheel serve of a HYBRID-backbone VLM
alias must boot text-only WITHOUT demanding the ``[vision]`` extra.

Context
-------
#1178 added ``mllm_backbone_is_hybrid`` + ``resolve_serving_lane``: a
multimodal alias whose LANGUAGE backbone is hybrid/linear-attention
(Qwen3.6 GatedDeltaNet — ``mlx-community/Qwen3.6-27B-4bit``) auto-downgrades
to the text-only mlx-lm lane at load time and never touches mlx-vlm.

But the CLI boot guard (``serve_command``) hard-required the ``[vision]``
extra whenever ``is_mllm_model(args.model)`` was True — which fires for a
hybrid VLM checkpoint (its config declares ``vision_config``) BEFORE the
auto-downgrade decision. Result: a base-wheel user was pushed into a ~1 GB
``[vision]`` install for a model that then serves text-only.

The fix makes the guard consult the SAME resolved-lane signal the engine
uses (``resolve_serving_lane``) via the ``_serve_will_run_on_mllm_lane``
helper: require ``[vision]`` ONLY when the model will actually run on the
MLLM lane. A genuine VLM (non-hybrid backbone, e.g. qwen3-vl) still requires
it; ``--mllm`` / ``--no-mllm`` are honoured.

These tests pin:
  * the helper decision for hybrid / genuine / forced / text-only / non-VLM,
  * the end-to-end guard behaviour with mlx-vlm mocked ABSENT (base wheel):
    hybrid VLM boots past the guard, genuine VLM still exits rc=2.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _args(model: str = "some/model", *, mllm: bool = False, no_mllm: bool = False):
    return SimpleNamespace(model=model, mllm=mllm, no_mllm=no_mllm)


# ---------------------------------------------------------------------------
# Unit: the decision helper ``_serve_will_run_on_mllm_lane``.
# ---------------------------------------------------------------------------


def _patch_probes(
    monkeypatch,
    *,
    is_mllm: bool,
    hybrid: bool,
    hybrid_runtime_supported: bool = False,
):
    """Stub the two offline probes ``resolve_serving_lane`` consults so the
    lane decision is exercised without a materialized checkpoint config."""
    from rapid_mlx.api import utils as api_utils
    from rapid_mlx.models import mllm as mllm_mod

    monkeypatch.setattr(api_utils, "is_mllm_model", lambda name: is_mllm)
    monkeypatch.setattr(
        api_utils,
        "mllm_backbone_cache_mode",
        lambda name: "arrays" if hybrid else None,
    )
    monkeypatch.setattr(
        api_utils,
        "mllm_hybrid_runtime_supported",
        lambda: hybrid_runtime_supported,
    )
    monkeypatch.setattr(
        mllm_mod,
        "vision_runtime_status",
        lambda: (mllm_mod.VisionRuntimeStatus.OK, None),
    )


def test_helper_hybrid_vlm_does_not_run_on_mllm_lane(monkeypatch):
    """A multimodal alias with a hybrid backbone auto-downgrades to text —
    the helper reports it will NOT run on the MLLM lane, so the guard skips
    the ``[vision]`` requirement (the base-wheel fix)."""
    from rapid_mlx import cli

    _patch_probes(monkeypatch, is_mllm=True, hybrid=True)
    assert cli._serve_will_run_on_mllm_lane(_args()) is False


def test_helper_genuine_vlm_runs_on_mllm_lane(monkeypatch):
    """A genuine VLM (non-hybrid backbone) stays on the MLLM lane, so the
    helper reports True and the guard STILL requires ``[vision]``."""
    from rapid_mlx import cli

    _patch_probes(monkeypatch, is_mllm=True, hybrid=False)
    assert cli._serve_will_run_on_mllm_lane(_args()) is True


def test_helper_supported_hybrid_vlm_runs_on_mllm_lane(monkeypatch):
    from rapid_mlx import cli

    _patch_probes(
        monkeypatch,
        is_mllm=True,
        hybrid=True,
        hybrid_runtime_supported=True,
    )
    assert cli._serve_will_run_on_mllm_lane(_args()) is True


def test_helper_force_mllm_on_hybrid_still_requires_vision(monkeypatch):
    """Explicit ``--mllm`` wins: even a hybrid backbone is reported as the
    MLLM lane so the operator gets the flag they asked for (and its
    ``[vision]`` requirement)."""
    from rapid_mlx import cli

    _patch_probes(monkeypatch, is_mllm=True, hybrid=True)
    assert cli._serve_will_run_on_mllm_lane(_args(mllm=True)) is True


def test_helper_no_mllm_never_runs_on_mllm_lane(monkeypatch):
    """``--no-mllm`` forces the text lane regardless of the checkpoint — the
    guard must never require ``[vision]`` for it."""
    from rapid_mlx import cli

    _patch_probes(monkeypatch, is_mllm=True, hybrid=False)
    assert cli._serve_will_run_on_mllm_lane(_args(no_mllm=True)) is False


def test_helper_non_vlm_does_not_run_on_mllm_lane(monkeypatch):
    """A plain text model is never on the MLLM lane."""
    from rapid_mlx import cli

    _patch_probes(monkeypatch, is_mllm=False, hybrid=False)
    assert cli._serve_will_run_on_mllm_lane(_args()) is False


# ---------------------------------------------------------------------------
# Integration: drive the real ``serve_command`` boot guard with mlx-vlm
# mocked ABSENT (the fresh ``pip install rapid-mlx`` state, no [vision]).
# ---------------------------------------------------------------------------


class _ReachedPastVisionGuardError(Exception):
    """Sentinel raised by the stubbed audio probe that immediately follows
    the vision guard — proves the vision guard did NOT sys.exit(2)."""


def _mock_mllm_absent(monkeypatch):
    from rapid_mlx.models import mllm as mllm_mod

    monkeypatch.setattr(
        mllm_mod,
        "vision_runtime_status",
        lambda: (mllm_mod.VisionRuntimeStatus.ABSENT, "mlx_vlm"),
    )


def _stub_post_guard_sentinel(monkeypatch):
    """Make the audio boot guard (the very next step after the vision guard)
    raise a sentinel so ``serve_command`` stops right there — we only care
    whether the vision guard let us through."""
    import rapid_mlx.audio.probe as audio_probe

    def _raise(_name):
        raise _ReachedPastVisionGuardError()

    monkeypatch.setattr(audio_probe, "is_audio_model_alias", _raise)


def test_serve_guard_hybrid_vlm_boots_without_vision_extra(monkeypatch, capsys):
    """Base wheel (mlx-vlm ABSENT): a hybrid-backbone VLM must pass the boot
    guard WITHOUT the ``[vision]``-required ``sys.exit(2)`` — it will
    auto-downgrade to the text lane."""
    from rapid_mlx import cli

    _patch_probes(monkeypatch, is_mllm=True, hybrid=True)
    _mock_mllm_absent(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar",
        lambda: False,
    )
    _stub_post_guard_sentinel(monkeypatch)

    args = _args("qwen3.5-4b-4bit")
    # Reaching the sentinel means the vision guard did NOT exit — the model
    # is allowed to boot text-only from the base wheel.
    with pytest.raises(_ReachedPastVisionGuardError):
        cli.serve_command(args)

    err = capsys.readouterr().err
    assert err.count("warning: vision runtime absent") == 1
    assert "rapid-mlx[vision]==" in err


def test_serve_guard_text_incapable_vlm_still_requires_vision_extra(
    monkeypatch, capsys
):
    """Base wheel (mlx-vlm ABSENT): a VLM whose backbone has NO text-lane
    loader must STILL fail fast with the ``[vision]``-required guard — the
    degrade covers only checkpoints whose language backbone the text lane
    loads. Bonsai 2's ``prism_hadamard_qwen35`` pack has its only loader on
    the MLLM lane, so it keeps the loud guard (and its ``--no-mllm`` hint is
    guarded separately by the text-lane pack rejection)."""
    from pathlib import Path

    from rapid_mlx import cli
    from rapid_mlx.api import utils as api_utils

    _patch_probes(monkeypatch, is_mllm=True, hybrid=False)
    _mock_mllm_absent(monkeypatch)
    # Feed the degrade probe the Bonsai 2 pack config: its top-level
    # ``model_type`` has no ``mlx_lm.models`` module and no vendored loader.
    monkeypatch.setattr(
        api_utils,
        "read_model_metadata",
        lambda _name: SimpleNamespace(
            config={
                "model_type": "prism_hadamard_qwen35",
                "architectures": ["PrismHadamardQwen35ForConditionalGeneration"],
                "vision_config": {},
            },
            snapshot_dir=Path("/snap"),
        ),
    )
    # If the guard wrongly let this through, the sentinel would surface
    # instead of SystemExit — making the test fail loudly rather than pass.
    _stub_post_guard_sentinel(monkeypatch)

    args = _args("bonsai2-27b-2bit")
    with pytest.raises(SystemExit) as exc_info:
        cli.serve_command(args)
    assert exc_info.value.code == 2

    err = capsys.readouterr().err
    assert "bonsai2-27b-2bit" in err
    assert "[vision]" in err
    # The guard message also surfaces --no-mllm as the text-only escape hatch.
    assert "--no-mllm" in err


def test_serve_guard_no_mllm_skips_vision_extra(monkeypatch):
    """``--no-mllm`` on a genuine VLM must bypass the vision guard entirely
    even on a base wheel (the pre-existing escape hatch, preserved)."""
    from rapid_mlx import cli

    _patch_probes(monkeypatch, is_mllm=True, hybrid=False)
    _mock_mllm_absent(monkeypatch)
    _stub_post_guard_sentinel(monkeypatch)

    args = _args("mlx-community/Qwen3-VL-2B-Instruct-4bit", no_mllm=True)
    with pytest.raises(_ReachedPastVisionGuardError):
        cli.serve_command(args)


def test_absent_runtime_degrade_rejects_image_with_capability_event(
    monkeypatch, capsys
):
    from rapid_mlx import cli, server
    from rapid_mlx.api.utils import validate_content_blocks_for_capabilities
    from rapid_mlx.model_aliases import resolve_profile
    from rapid_mlx.models import mllm as mllm_mod

    profile = resolve_profile("qwen3.5-4b-4bit")
    assert profile is not None and profile.vision_min_memory_gb is not None
    _patch_probes(monkeypatch, is_mllm=True, hybrid=True)
    monkeypatch.setattr("rapid_mlx.api.utils.physical_ram_gb", lambda: 64.0)
    monkeypatch.setattr(
        mllm_mod,
        "vision_runtime_status",
        lambda: (mllm_mod.VisionRuntimeStatus.ABSENT, "mlx_vlm"),
    )
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar",
        lambda: False,
    )
    monkeypatch.setattr(server, "_preflight_vision_runtime", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_ensure_routing_config", lambda _model: None)
    monkeypatch.setattr(
        "rapid_mlx.utils.tokenizer._resolve_subfolder_checkpoint",
        lambda model: model,
    )
    monkeypatch.setattr(
        "rapid_mlx.model_metadata.read_model_metadata", lambda _model: None
    )
    events = []
    monkeypatch.setattr(
        "rapid_mlx.telemetry.inference.emit_capability_rejected",
        lambda capability, **props: events.append((capability, props)),
    )

    assert cli._warn_vision_text_only_degrade(profile) is True
    serving_checkpoint = server._resolve_serving_checkpoint("qwen3.5-4b-4bit")
    assert serving_checkpoint.lane_reason == "vision_runtime_absent"
    assert serving_checkpoint.auto_text_fallback is True
    with pytest.raises(Exception) as caught:
        validate_content_blocks_for_capabilities(
            [{"content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
            model_name="qwen3.5-4b-4bit",
            allow_image=serving_checkpoint.is_mllm,
            allow_video=serving_checkpoint.is_mllm,
        )

    assert getattr(caught.value, "code", None) == "image_input_unsupported"
    assert events[0][0] == "image_input_unsupported"
    assert capsys.readouterr().err.count("warning: vision runtime absent") == 1


@pytest.mark.parametrize(
    "alias",
    ["mistral-24b-4bit", "devstral-24b-4bit", "devstral-v2-24b-4bit"],
)
def test_absent_runtime_warning_uses_resolved_lane_without_memory_floor(
    alias, monkeypatch, capsys
):
    from rapid_mlx import cli
    from rapid_mlx.model_aliases import resolve_profile

    profile = resolve_profile(alias)
    assert profile is not None and profile.vision_min_memory_gb is None
    _patch_probes(monkeypatch, is_mllm=True, hybrid=True)
    _mock_mllm_absent(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar",
        lambda: False,
    )

    assert cli._warn_vision_text_only_degrade(profile, args=_args(alias)) is True
    assert capsys.readouterr().err.count("warning: vision runtime absent") == 1


def test_broken_runtime_and_forced_vision_never_degrade(monkeypatch, capsys):
    from rapid_mlx import cli
    from rapid_mlx.model_aliases import resolve_profile
    from rapid_mlx.models import mllm as mllm_mod

    profile = resolve_profile("qwen3.5-4b-4bit")
    assert profile is not None
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar",
        lambda: False,
    )
    monkeypatch.setattr(
        mllm_mod,
        "vision_runtime_status",
        lambda: (mllm_mod.VisionRuntimeStatus.BROKEN, "PIL"),
    )
    assert cli._warn_vision_text_only_degrade(profile, args=_args()) is False

    monkeypatch.setattr(
        mllm_mod,
        "vision_runtime_status",
        lambda: (mllm_mod.VisionRuntimeStatus.ABSENT, "mlx_vlm"),
    )
    assert cli._warn_vision_text_only_degrade(profile, args=_args(mllm=True)) is False
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# Real-probe coverage: drive the ACTUAL resolve_serving_lane / is_mllm_model /
# mllm_backbone_is_hybrid chain (no mocks) against genuinely-cached
# checkpoint configs. Skips cleanly on a host that hasn't materialized the
# config (CI), so the mock-driven tests above remain the always-on gate while
# these prove the production probe path when the checkpoint is present.
# ---------------------------------------------------------------------------


def _cached_or_skip(hf_path: str):
    """Skip unless ``hf_path``'s config is materialized in the local cache —
    the probes read it offline; without it there is nothing real to test."""
    from rapid_mlx.model_metadata import read_model_metadata

    if read_model_metadata(hf_path) is None:
        pytest.skip(f"{hf_path} config not materialized in local cache")


def test_real_probe_cached_hybrid_vlm_downgrades_to_text(monkeypatch):
    """REAL probe path (no mocks): the actually-cached hybrid-backbone
    Qwen3.6 checkpoint (config declares ``vision_config`` but ``layer_types``
    is linear-attention) must resolve to the TEXT lane, so the guard skips
    ``[vision]``. This is the exact production scenario the fix targets —
    exercised end-to-end through ``resolve_serving_lane`` rather than a
    hard-coded ``hybrid=True``."""
    from rapid_mlx import cli
    from rapid_mlx.api.utils import is_mllm_model, mllm_backbone_is_hybrid

    hf_path = "mlx-community/Qwen3.6-27B-4bit"
    _cached_or_skip(hf_path)

    # Precondition: the real probes see a multimodal config with a hybrid
    # backbone (else this would pass vacuously if the checkpoint changed).
    assert is_mllm_model(hf_path) is True
    assert mllm_backbone_is_hybrid(hf_path) is True

    # The guard consults the real resolve_serving_lane → text lane → no
    # [vision] requirement, even though the raw classification is "VLM".
    assert cli._serve_will_run_on_mllm_lane(_args(hf_path)) is False

    # And end-to-end: with mlx-vlm ABSENT the boot guard lets it through.
    _mock_mllm_absent(monkeypatch)
    _stub_post_guard_sentinel(monkeypatch)
    with pytest.raises(_ReachedPastVisionGuardError):
        cli.serve_command(_args(hf_path))


def test_real_probe_cached_genuine_vlm_keeps_classification_and_degrades(
    monkeypatch,
):
    """REAL probe path (no lane mocks): an actually-cached genuine multimodal
    checkpoint (Gemma-4, non-hybrid backbone) keeps its MLLM classification —
    and with the vision runtime ABSENT the base wheel degrades it to the text
    lane (same lane ``--no-mllm`` selects) instead of demanding ``[vision]``."""
    from rapid_mlx import cli
    from rapid_mlx.api.utils import is_mllm_model, mllm_backbone_is_hybrid

    hf_path = "mlx-community/gemma-4-12B-it-4bit"
    _cached_or_skip(hf_path)

    # Precondition: the real probes see a genuine multimodal checkpoint with a
    # standard-attention backbone (else this would pass vacuously).
    assert is_mllm_model(hf_path) is True
    assert mllm_backbone_is_hybrid(hf_path) is False

    # With the runtime ABSENT the degrade applies: the guard does not demand
    # [vision] for a text-capable backbone.
    _mock_mllm_absent(monkeypatch)
    _stub_post_guard_sentinel(monkeypatch)
    with pytest.raises(_ReachedPastVisionGuardError):
        cli.serve_command(_args(hf_path))


# ---------------------------------------------------------------------------
# Explicit contract: the SAFE default for an uncached / unclassifiable
# checkpoint. When no config is materialized, the hybrid probe answers "not
# hybrid" by design, so a VLM-named checkpoint keeps the [vision]-required
# guard rather than being silently let through onto a lane that might crash.
# This is a DELIBERATE, task-scoped default (the guard runs before download
# to fail fast) — pinned here so it can't silently drift, and so the guard's
# message is verified to point the user at --no-mllm for a text-capable
# backbone. (A genuinely-hybrid uncached checkpoint whose NAME does not match
# a VLM pattern — e.g. qwen3.6-27b-4bit — is classified text and boots
# without [vision] anyway; see the real-probe test above for the cached form.)
# ---------------------------------------------------------------------------


def test_uncached_vlm_named_checkpoint_keeps_safe_vision_default(monkeypatch, capsys):
    """No cached config + a VLM-pattern name → the REAL is_mllm_model matches
    on the name, the REAL hybrid probe can't prove hybrid (no config), so the
    guard keeps the safe ``[vision]``-required default and fails fast with a
    message that also names ``--no-mllm``."""
    from rapid_mlx import cli
    from rapid_mlx.api.utils import is_mllm_model, mllm_backbone_is_hybrid
    from rapid_mlx.model_metadata import read_model_metadata

    # A name that (a) is NOT a resolvable/cached repo and (b) trips the VLM
    # name pattern, so the real probes are exercised without any mock.
    fake = "nonexistent-org/Made-Up-VL-Hybrid-Model-4bit"
    assert read_model_metadata(fake) is None, "test name must be uncached"
    assert is_mllm_model(fake) is True, "VL-pattern name must classify as VLM"
    assert mllm_backbone_is_hybrid(fake) is False, "no config → not provably hybrid"

    # Guard decision is real (not mocked): safe default → MLLM lane → require.
    assert cli._serve_will_run_on_mllm_lane(_args(fake)) is True

    # End-to-end on a base wheel: fail fast rc=2 with the actionable message.
    _mock_mllm_absent(monkeypatch)
    _stub_post_guard_sentinel(monkeypatch)
    with pytest.raises(SystemExit) as exc_info:
        cli.serve_command(_args(fake))
    assert exc_info.value.code == 2

    err = capsys.readouterr().err
    assert "[vision]" in err
    # The safe-default message still surfaces the text-only escape hatch.
    assert "--no-mllm" in err
