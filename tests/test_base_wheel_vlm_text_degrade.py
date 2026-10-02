# SPDX-License-Identifier: Apache-2.0
"""Base-wheel text-only degrade for standard-attention VLM checkpoints.

Companion to ``test_basewheel_hybrid_vlm_serve_1017`` (which covers the
#3831 hybrid/arrays-cache degrade). The #3831 contract only auto-downgraded
aliases whose LANGUAGE backbone is hybrid/linear-attention; a curated VLM
alias whose checkpoint has a text-capable standard-attention backbone
(Gemma 4, Qwen3-VL, Gemma 3(n), …) still exited with
``RAPID-MLX-STARTUP-FAILURE: runtime_extra_missing extra=vision`` on a
clean ``pip install rapid-mlx`` even though ``--no-mllm`` serves the same
checkpoint from the base wheel.

These tests pin the extended contract:

* vision runtime ABSENT + a text-lane-loadable backbone (mlx-lm ships the
  arch module, or the vendored Gemma 4 family loader covers it) → the serve
  proceeds on the text lane with one warning naming the lost image/video
  input and the install-method-aware repair command;
* image/video requests get the existing capability-rejected handling;
* a backbone whose ONLY loader lives on the MLLM lane (Bonsai 2's
  ``prism_hadamard_qwen35``), a BROKEN runtime, and an explicit ``--mllm``
  request keep the loud ``[vision]``-required failure;
* the Desktop sidecar keeps its silent-degrade contract (no stderr copy).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _args(
    model: str = "gemma-4-26b-4bit", *, mllm: bool = False, no_mllm: bool = False
):
    return SimpleNamespace(model=model, mllm=mllm, no_mllm=no_mllm)


GEMMA4_VLM_CONFIG = {
    "model_type": "gemma4",
    "architectures": ["Gemma4ForConditionalGeneration"],
    "vision_config": {},
}
PRISM_PACK_CONFIG = {
    "model_type": "prism_hadamard_qwen35",
    "architectures": ["PrismHadamardQwen35ForConditionalGeneration"],
    "vision_config": {},
}
UNKNOWN_VLM_CONFIG = {
    "model_type": "brand_new_vlm_arch",
    "architectures": ["BrandNewVLMForConditionalGeneration"],
    "vision_config": {},
}
TEXT_CONFIG = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
}


def _meta(config):
    return SimpleNamespace(config=config, snapshot_dir=Path("/snap"))


def _patch_lane_probes(monkeypatch, *, is_mllm: bool, cache_mode=None):
    """Stub the offline lane probes ``resolve_serving_lane_decision`` reads.

    ``is_mllm`` simulates the checkpoint-weight evidence the engine classifies
    after the pull; ``cache_mode`` is the backbone cache contract (``None`` =
    ordinary attention, i.e. NOT the #3831 arrays path).
    """
    from rapid_mlx.api import utils as api_utils

    monkeypatch.setattr(api_utils, "is_mllm_model", lambda name: is_mllm)
    monkeypatch.setattr(api_utils, "mllm_backbone_cache_mode", lambda name: cache_mode)


def _mock_vision_absent(monkeypatch) -> None:
    from rapid_mlx.models import mllm as mllm_mod

    monkeypatch.setattr(
        mllm_mod,
        "vision_runtime_status",
        lambda: (mllm_mod.VisionRuntimeStatus.ABSENT, "mlx_vlm"),
    )


def _mock_vision_broken(monkeypatch) -> None:
    from rapid_mlx.models import mllm as mllm_mod

    monkeypatch.setattr(
        mllm_mod,
        "vision_runtime_status",
        lambda: (mllm_mod.VisionRuntimeStatus.BROKEN, "PIL"),
    )


def _patch_degrade_config(monkeypatch, config) -> None:
    """Feed the text-degrade probe a materialized checkpoint config."""
    from rapid_mlx.api import utils as api_utils

    monkeypatch.setattr(api_utils, "read_model_metadata", lambda _name: _meta(config))


def _allow_desktop_warning(monkeypatch) -> None:
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar",
        lambda: False,
    )


class _ReachedPastVisionGuardError(Exception):
    """Sentinel proving the vision guard did NOT sys.exit(2)."""


def _stub_post_guard_sentinel(monkeypatch) -> None:
    import rapid_mlx.audio.probe as audio_probe

    def _raise(_name):
        raise _ReachedPastVisionGuardError()

    monkeypatch.setattr(audio_probe, "is_audio_model_alias", _raise)


def _stub_boot_prefetch(monkeypatch) -> list[str]:
    """Replace the boot guard's config prefetch with a recording no-op.

    Unit tests must never open a Hub window; the prefetch contract itself is
    pinned by the dedicated tests below. Returns the recorded repos.
    """
    from rapid_mlx.api import utils as api_utils

    fetched: list[str] = []
    monkeypatch.setattr(api_utils, "_DEGRADE_CONFIG_PREFETCHED", set())
    monkeypatch.setattr(
        api_utils,
        "_prefetch_config_for_degrade_probe",
        lambda repo: fetched.append(repo),
    )
    return fetched


# ---------------------------------------------------------------------------
# Base install (vision runtime ABSENT): Gemma 4 serves text-only.
# ---------------------------------------------------------------------------


def test_base_install_gemma4_boots_text_only_with_warning(monkeypatch, capsys):
    """A cached Gemma 4 checkpoint on a base wheel degrades to the text lane:
    the boot guard does not exit, and one warning names what is lost plus the
    install-method-aware repair command."""
    from rapid_mlx import cli

    _patch_lane_probes(monkeypatch, is_mllm=True, cache_mode=None)
    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(monkeypatch, GEMMA4_VLM_CONFIG)
    _allow_desktop_warning(monkeypatch)
    _stub_post_guard_sentinel(monkeypatch)
    fetched = _stub_boot_prefetch(monkeypatch)

    args = _args("gemma-4-26b-4bit")
    with pytest.raises(_ReachedPastVisionGuardError):
        cli.serve_command(args)

    # The boot guard materialized the config exactly once, for the profile's
    # hf_path, before the (cache-only) degrade probes ran.
    assert fetched == ["mlx-community/gemma-4-26b-a4b-it-4bit"]

    err = capsys.readouterr().err
    assert err.count("warning: vision runtime absent") == 1
    assert "image and video input" in err
    assert "rapid-mlx[vision]==" in err


def _cold_cache_with_boot_prefetch(monkeypatch, config) -> dict:
    """Cold cache whose ONLY way to materialize metadata is the boot guard's
    ``_prefetch_config_for_degrade_probe``.

    The lane probes stay real (``is_mllm_model``, the cache-mode probe, the
    degrade predicate): every ``read_model_metadata`` answers ``None`` until
    the boot prefetch "downloads" ``config.json`` for a repo, after which that
    repo — and only that repo — reads back ``config``. The legacy pre-weights
    prefetch is recorded separately so a test can prove it never ran.
    """
    from rapid_mlx import cli, model_metadata
    from rapid_mlx.api import utils as api_utils

    state: dict = {"materialized": set(), "fetched": [], "legacy_fetched": []}

    def _read(name):
        if name in state["materialized"]:
            # A config-only Hub snapshot: no weight index, not a local dir.
            return SimpleNamespace(
                config=config, snapshot_dir=Path("/snap"), is_local=False
            )
        return None

    def _boot_prefetch(repo):
        state["fetched"].append(repo)
        state["materialized"].add(repo)

    monkeypatch.setattr(model_metadata, "read_model_metadata", _read)
    monkeypatch.setattr(api_utils, "read_model_metadata", _read)
    monkeypatch.setattr(api_utils, "_DEGRADE_CONFIG_PREFETCHED", set())
    monkeypatch.setattr(api_utils, "_prefetch_config_for_degrade_probe", _boot_prefetch)
    monkeypatch.setattr(
        cli, "_prefetch_config_for_lane_guard", state["legacy_fetched"].append
    )
    return state


def _pretend_mlx_lm_ships(monkeypatch, *module_types: str) -> None:
    """Make ``mlx_lm.models.<type>`` resolvable for the listed arches only, so
    the allow-list probe answers the same with and without mlx-lm installed
    (GitHub Linux CI has no MLX)."""
    import importlib.util

    real_find_spec = importlib.util.find_spec
    wanted = {f"mlx_lm.models.{t}" for t in module_types}

    def _find_spec(name, *a, **kw):
        if name in wanted:
            return object()
        if name.startswith("mlx_lm.models."):
            return None
        return real_find_spec(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", _find_spec)


def test_base_install_gemma4_fresh_cache_degrades(monkeypatch, capsys):
    """Nothing cached yet, catalog alias: ``serve_command``'s boot prefetch
    materializes config.json for the profile's repo exactly once, the guard
    sees a text-lane-loadable backbone and lets the serve through, and the
    degrade warning prints once."""
    from rapid_mlx import cli

    state = _cold_cache_with_boot_prefetch(monkeypatch, GEMMA4_VLM_CONFIG)
    _mock_vision_absent(monkeypatch)
    _allow_desktop_warning(monkeypatch)
    _stub_post_guard_sentinel(monkeypatch)

    with pytest.raises(_ReachedPastVisionGuardError):
        cli.serve_command(_args("gemma-4-26b-4bit"))

    assert state["fetched"] == ["mlx-community/gemma-4-26b-a4b-it-4bit"]
    assert state["legacy_fetched"] == []
    err = capsys.readouterr().err
    assert err.count("warning: vision runtime absent") == 1
    assert "image and video input" in err
    assert "[vision]" in err  # the repair command, not a guard failure


DIRECT_QWEN3_VL_REPO = "example-org/Qwen3-VL-8B-Instruct-abliterated-4bit"
QWEN3_VL_CONFIG = {
    "model_type": "qwen3_vl",
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "vision_config": {},
}


def test_base_install_direct_repo_fresh_cache_degrades(monkeypatch, capsys):
    """Nothing cached yet, direct Hugging Face repo id (no alias profile): the
    boot prefetch targets ``args.model`` itself, once, and the guard consults
    that same repo — an allow-listed backbone degrades instead of failing on
    the name-pattern VLM verdict."""
    from rapid_mlx import cli
    from rapid_mlx.model_aliases import resolve_profile

    assert resolve_profile(DIRECT_QWEN3_VL_REPO) is None
    state = _cold_cache_with_boot_prefetch(monkeypatch, QWEN3_VL_CONFIG)
    _pretend_mlx_lm_ships(monkeypatch, "qwen3_vl")
    _mock_vision_absent(monkeypatch)
    _allow_desktop_warning(monkeypatch)
    _stub_post_guard_sentinel(monkeypatch)

    with pytest.raises(_ReachedPastVisionGuardError):
        cli.serve_command(_args(DIRECT_QWEN3_VL_REPO))

    assert state["fetched"] == [DIRECT_QWEN3_VL_REPO]
    err = capsys.readouterr().err
    assert err.count("warning: vision runtime absent") == 1
    assert "image and video input" in err


def test_base_install_direct_repo_without_prefetch_keeps_guard(monkeypatch, capsys):
    """Control for the test above: the same cold direct repo WITHOUT the boot
    prefetch has no config to classify, so the (cache-only) probe fails closed
    and the guard exits 2 — proving the prefetch is what unlocks the degrade."""
    from rapid_mlx import cli

    state = _cold_cache_with_boot_prefetch(monkeypatch, QWEN3_VL_CONFIG)
    from rapid_mlx.api import utils as api_utils

    monkeypatch.setattr(
        api_utils, "_prefetch_config_for_degrade_probe", state["fetched"].append
    )
    _pretend_mlx_lm_ships(monkeypatch, "qwen3_vl")
    _mock_vision_absent(monkeypatch)
    _allow_desktop_warning(monkeypatch)
    _stub_post_guard_sentinel(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        cli.serve_command(_args(DIRECT_QWEN3_VL_REPO))
    assert exc_info.value.code == 2
    assert state["fetched"] == [DIRECT_QWEN3_VL_REPO]
    err = capsys.readouterr().err
    assert "[vision]" in err
    assert "warning: vision runtime absent" not in err


def test_base_install_direct_repo_unlisted_arch_keeps_guard(monkeypatch, capsys):
    """A direct repo whose backbone is NOT on the reviewed allow-list keeps
    the [vision] guard after the prefetch — even though a module of that name
    is importable — and prints no degrade warning."""
    from rapid_mlx import cli

    repo = "example-org/BrandNew-VL-7B-4bit"
    state = _cold_cache_with_boot_prefetch(monkeypatch, UNKNOWN_VLM_CONFIG)
    _pretend_mlx_lm_ships(monkeypatch, "brand_new_vlm_arch")
    _mock_vision_absent(monkeypatch)
    _allow_desktop_warning(monkeypatch)
    _stub_post_guard_sentinel(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        cli.serve_command(_args(repo))
    assert exc_info.value.code == 2
    assert state["fetched"] == [repo]
    err = capsys.readouterr().err
    assert "[vision]" in err
    assert "warning: vision runtime absent" not in err


def test_direct_repo_explicit_flags_skip_prefetch_and_warning(monkeypatch, capsys):
    """``--mllm`` / ``--no-mllm`` / a spec decoder on a direct repo id are not
    degrade candidates: no boot prefetch and no degrade warning."""
    from rapid_mlx import cli

    state = _cold_cache_with_boot_prefetch(monkeypatch, QWEN3_VL_CONFIG)
    _pretend_mlx_lm_ships(monkeypatch, "qwen3_vl")
    _mock_vision_absent(monkeypatch)
    _allow_desktop_warning(monkeypatch)

    for flags in (
        {"mllm": True},
        {"no_mllm": True},
        {"spec_decode": "dflash"},
        {"enable_mtp": True},
        {"force_spec_decode": True},
    ):
        args = SimpleNamespace(
            **{"model": DIRECT_QWEN3_VL_REPO, "mllm": False, "no_mllm": False} | flags
        )
        assert cli._warn_vision_text_only_degrade(None, args=args) is False
    assert cli._warn_vision_text_only_degrade(None) is False
    assert state["fetched"] == []
    assert capsys.readouterr().err == ""


def test_direct_repo_broken_runtime_never_warns(monkeypatch, capsys):
    """Only an ABSENT runtime may degrade a direct repo id: a BROKEN install
    prints no degrade warning (the guard owns the repair message)."""
    from rapid_mlx import cli

    state = _cold_cache_with_boot_prefetch(monkeypatch, QWEN3_VL_CONFIG)
    state["materialized"].add(DIRECT_QWEN3_VL_REPO)
    _pretend_mlx_lm_ships(monkeypatch, "qwen3_vl")
    _mock_vision_broken(monkeypatch)
    _allow_desktop_warning(monkeypatch)

    args = _args(DIRECT_QWEN3_VL_REPO)
    assert cli._warn_vision_text_only_degrade(None, args=args) is False
    assert capsys.readouterr().err == ""
    assert cli._serve_will_run_on_mllm_lane(args) is True


def test_degraded_lane_contract_is_vision_runtime_absent(monkeypatch):
    """The engine-side resolver returns the SAME lane ``--no-mllm`` selects:
    text lane, reason ``vision_runtime_absent``, auto-downgrade flagged so
    diagnostics and the ready banner name the cause."""
    from rapid_mlx.api.utils import resolve_serving_lane_decision

    _patch_lane_probes(monkeypatch, is_mllm=True, cache_mode=None)
    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(monkeypatch, GEMMA4_VLM_CONFIG)

    decision = resolve_serving_lane_decision("gemma-4-26b-4bit")
    assert decision.is_mllm is False
    assert decision.reason == "vision_runtime_absent"
    assert decision.auto_text_fallback is True


def test_degraded_gemma4_rejects_image_and_video_with_capability_event(
    monkeypatch, capsys
):
    """With the degraded lane contract, image/video requests are rejected by
    the existing capability boundary and emit the telemetry event."""
    from rapid_mlx import cli, model_aliases, server
    from rapid_mlx.api.utils import validate_content_blocks_for_capabilities
    from rapid_mlx.models import mllm as mllm_mod

    _patch_lane_probes(monkeypatch, is_mllm=True, cache_mode=None)
    monkeypatch.setattr(
        mllm_mod,
        "vision_runtime_status",
        lambda: (mllm_mod.VisionRuntimeStatus.ABSENT, "mlx_vlm"),
    )
    _patch_degrade_config(monkeypatch, GEMMA4_VLM_CONFIG)
    _allow_desktop_warning(monkeypatch)
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

    assert (
        cli._warn_vision_text_only_degrade(
            model_aliases.resolve_profile("gemma-4-26b-4bit"), args=_args()
        )
        is True
    )
    serving_checkpoint = server._resolve_serving_checkpoint("gemma-4-26b-4bit")
    assert serving_checkpoint.is_mllm is False
    assert serving_checkpoint.lane_reason == "vision_runtime_absent"
    assert serving_checkpoint.auto_text_fallback is True

    with pytest.raises(Exception) as image_caught:
        validate_content_blocks_for_capabilities(
            [{"content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
            model_name="gemma-4-26b-4bit",
            allow_image=serving_checkpoint.is_mllm,
            allow_video=serving_checkpoint.is_mllm,
        )
    assert getattr(image_caught.value, "code", None) == "image_input_unsupported"

    with pytest.raises(ValueError):
        validate_content_blocks_for_capabilities(
            [{"content": [{"type": "video_url", "video_url": {"url": "x"}}]}],
            model_name="gemma-4-26b-4bit",
            allow_image=serving_checkpoint.is_mllm,
            allow_video=serving_checkpoint.is_mllm,
        )
    assert {capability for capability, _props in events} == {
        "image_input_unsupported",
        "video_input_unsupported",
    }
    assert capsys.readouterr().err.count("warning: vision runtime absent") == 1


# ---------------------------------------------------------------------------
# Bonsai 2 (prism_hadamard_qwen35): no text backbone → unchanged failure.
# ---------------------------------------------------------------------------


def test_bonsai2_pack_still_requires_vision_extra(monkeypatch, capsys):
    """A pack whose only loader lives on the MLLM lane (Bonsai 2's
    ``prism_hadamard_qwen35``) is NOT text-capable: the degrade must not fire
    and the boot guard still exits 2 with the [vision] hint."""
    from rapid_mlx import cli
    from rapid_mlx.model_aliases import resolve_profile

    _patch_lane_probes(monkeypatch, is_mllm=True, cache_mode=None)
    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(monkeypatch, PRISM_PACK_CONFIG)
    _allow_desktop_warning(monkeypatch)
    _stub_post_guard_sentinel(monkeypatch)
    fetched = _stub_boot_prefetch(monkeypatch)

    args = _args("bonsai2-27b-2bit")
    assert cli._serve_will_run_on_mllm_lane(args) is True
    profile = resolve_profile("bonsai2-27b-2bit")
    assert profile is not None
    assert cli._warn_vision_text_only_degrade(profile, args=args) is False

    with pytest.raises(SystemExit) as exc_info:
        cli.serve_command(args)
    assert exc_info.value.code == 2
    assert "[vision]" in capsys.readouterr().err
    # The boot prefetch targets the profile's repo even when the probe then
    # fails closed (the pack is not text-capable) — once, and nothing more.
    assert fetched == ["prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"]


def test_degrade_probe_rejects_mllm_only_pack(monkeypatch):
    """The probe itself fails closed on the MLLM-lane-only pack, whatever the
    alias profile claims."""
    from rapid_mlx.api.utils import checkpoint_serves_text_without_vision

    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(monkeypatch, PRISM_PACK_CONFIG)
    assert checkpoint_serves_text_without_vision("bonsai2-27b-2bit") is False


# ---------------------------------------------------------------------------
# BROKEN runtime and explicit vision requests keep the loud guard.
# ---------------------------------------------------------------------------


def test_broken_runtime_never_degrades(monkeypatch, capsys):
    """Only an ABSENT runtime may degrade. A BROKEN install keeps the repair
    guard and prints no degrade warning."""
    from rapid_mlx import cli
    from rapid_mlx.model_aliases import resolve_profile

    _patch_lane_probes(monkeypatch, is_mllm=True, cache_mode=None)
    _mock_vision_broken(monkeypatch)
    _patch_degrade_config(monkeypatch, GEMMA4_VLM_CONFIG)
    _allow_desktop_warning(monkeypatch)

    profile = resolve_profile("gemma-4-26b-4bit")
    assert profile is not None
    assert cli._warn_vision_text_only_degrade(profile, args=_args()) is False
    assert capsys.readouterr().err == ""
    assert cli._serve_will_run_on_mllm_lane(_args()) is True


def test_explicit_mllm_on_gemma4_still_requires_vision_extra(monkeypatch, capsys):
    """``--mllm`` is a deliberate demand for the vision lane: no degrade, no
    warning, and the guard still exits 2 on a base wheel."""
    from rapid_mlx import cli

    _patch_lane_probes(monkeypatch, is_mllm=True, cache_mode=None)
    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(monkeypatch, GEMMA4_VLM_CONFIG)
    _allow_desktop_warning(monkeypatch)
    _stub_post_guard_sentinel(monkeypatch)
    fetched = _stub_boot_prefetch(monkeypatch)

    args = _args("gemma-4-26b-4bit", mllm=True)
    assert cli._serve_will_run_on_mllm_lane(args) is True
    with pytest.raises(SystemExit) as exc_info:
        cli.serve_command(args)
    assert exc_info.value.code == 2
    assert capsys.readouterr().err.count("warning: vision runtime absent") == 0
    # An explicit vision demand is never a degrade candidate: no prefetch.
    assert fetched == []


def test_desktop_sidecar_degrades_silently(monkeypatch, capsys):
    """Desktop owns its copy surface: the degrade still applies but the CLI
    warning stays suppressed (the #3831 sidecar contract, unchanged)."""
    from rapid_mlx import cli
    from rapid_mlx.model_aliases import resolve_profile

    _patch_lane_probes(monkeypatch, is_mllm=True, cache_mode=None)
    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(monkeypatch, GEMMA4_VLM_CONFIG)
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar",
        lambda: True,
    )

    profile = resolve_profile("gemma-4-26b-4bit")
    assert profile is not None
    assert cli._warn_vision_text_only_degrade(profile, args=_args()) is False
    assert capsys.readouterr().err == ""
    # The lane contract still degrades — Desktop renders its own copy.
    assert cli._serve_will_run_on_mllm_lane(_args()) is False


# ---------------------------------------------------------------------------
# Speculative decode / MTP requests never consult the degrade. The resolver
# routes those requests to the text lane on its own (the decoder is only
# honoured there), so both the [vision] guard and the degrade warning must
# answer from that routing — checkpoint_serves_text_without_vision is never
# applied to them. Pinned for the guard (fresh AND weight-evidenced caches)
# and the warning path; a control proves the degrade still applies without.
# ---------------------------------------------------------------------------


def _spec_decode_args(**flags):
    base = {"model": "gemma-4-26b-4bit", "mllm": False, "no_mllm": False}
    base.update(flags)
    return SimpleNamespace(**base)


def _forbid_degrade_consult(monkeypatch) -> None:
    """Make the degrade predicate scream if anyone consults it."""
    from rapid_mlx.api import utils as api_utils

    def _forbidden(_name):
        raise AssertionError(
            "checkpoint_serves_text_without_vision must not be consulted for "
            "speculative-decode / MTP requests"
        )

    monkeypatch.setattr(api_utils, "checkpoint_serves_text_without_vision", _forbidden)


def test_spec_decode_requests_skip_the_degrade_in_the_guard(monkeypatch):
    """Guard, fresh install: spec-decode / MTP requests short-circuit BEFORE
    the degrade probe — the predicate is never consulted and the lane follows
    the resolver's own text-lane routing for the decoder."""
    from rapid_mlx import cli
    from rapid_mlx.api import utils as api_utils

    _mock_vision_absent(monkeypatch)
    # Fresh install: no weight evidence (the resolver's spec branch then
    # answers text_checkpoint on its own).
    monkeypatch.setattr(api_utils, "read_model_metadata", lambda _name: None)
    monkeypatch.setattr(cli, "_prefetch_config_for_lane_guard", lambda _p: None)
    _forbid_degrade_consult(monkeypatch)

    for flags in (
        {"spec_decode": "dflash"},
        {"spec_decode": "mtp"},
        {"enable_mtp": True},
        {"force_spec_decode": True},
    ):
        assert cli._serve_will_run_on_mllm_lane(_spec_decode_args(**flags)) is False


def test_spec_decode_requests_skip_the_degrade_with_weight_evidence(
    monkeypatch,
):
    """Guard, cached VLM: even with MLLM weight evidence (the resolver's own
    ``text_lane_speculative_decode`` routing) a spec-decode request never
    reaches the degrade probe."""
    from rapid_mlx import cli

    _mock_vision_absent(monkeypatch)
    _patch_lane_probes(monkeypatch, is_mllm=True, cache_mode=None)
    _patch_degrade_config(monkeypatch, GEMMA4_VLM_CONFIG)
    _forbid_degrade_consult(monkeypatch)

    for flags in (
        {"spec_decode": "dflash"},
        {"enable_mtp": True},
        {"force_spec_decode": True},
    ):
        assert cli._serve_will_run_on_mllm_lane(_spec_decode_args(**flags)) is False


def test_spec_decode_requests_never_trigger_the_degrade_warning(monkeypatch):
    """Warning path: a spec-decode / MTP request gets no degrade warning and
    never consults the predicate."""
    from rapid_mlx import cli
    from rapid_mlx.model_aliases import resolve_profile

    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(monkeypatch, GEMMA4_VLM_CONFIG)
    _forbid_degrade_consult(monkeypatch)
    profile = resolve_profile("gemma-4-26b-4bit")
    assert profile is not None

    for flags in (
        {"spec_decode": "dflash"},
        {"spec_decode": "mtp"},
        {"enable_mtp": True},
        {"force_spec_decode": True},
    ):
        args = _spec_decode_args(**flags)
        assert cli._alias_text_degrades_without_vision(profile, args=args) is False
        assert cli._warn_vision_text_only_degrade(profile, args=args) is False


def test_plain_serve_still_consults_the_degrade(monkeypatch):
    """Control for the spec-decode short-circuit: without a requested decoder
    the guard DOES consult the predicate — a False verdict keeps the loud
    [vision] guard (the probe's fail-closed default)."""
    from rapid_mlx import cli

    _mock_vision_absent(monkeypatch)
    _patch_lane_probes(monkeypatch, is_mllm=True, cache_mode=None)
    _patch_degrade_config(monkeypatch, UNKNOWN_VLM_CONFIG)

    args = _spec_decode_args()
    assert cli._serve_will_run_on_mllm_lane(args) is True


# ---------------------------------------------------------------------------
# Eligibility probe unit contracts (real mlx_lm import probes, no mocks).
# ---------------------------------------------------------------------------


def test_text_lane_backbone_probe_requires_reviewed_arch_and_module():
    """The probe needs a REVIEWED multimodal-layout arch AND its mlx-lm
    module: vendored Gemma 4 family members answer True everywhere (pure
    data, no mlx import); allow-listed arches follow the installed mlx-lm
    (False on the no-MLX CI lane, True once installed); and an arch whose
    module merely exists — ``llama`` is importable wherever mlx-lm is — is
    never evidence, so it stays False even on a full install."""
    import importlib.util

    from rapid_mlx.api.utils import _text_lane_loads_model_type
    from rapid_mlx.models.text_lane_arches import MLX_LM_MM_BACKBONE_ARCHES

    # Vendored family: environment-independent.
    assert _text_lane_loads_model_type("gemma4") is True
    assert _text_lane_loads_model_type("gemma4_unified") is True
    assert _text_lane_loads_model_type("gemma4_assistant") is True
    # Allow-listed arches OUTSIDE the vendored family follow the installed
    # mlx-lm: the probe must agree with a direct spec probe (both False on
    # the no-MLX CI lane, both True once mlx-lm ships the arch).
    for arch in ("qwen3_vl", "qwen3_5", "mistral3"):
        assert arch in MLX_LM_MM_BACKBONE_ARCHES
        try:
            installed = importlib.util.find_spec(f"mlx_lm.models.{arch}") is not None
        except ImportError:
            # No mlx-lm on this environment (no-MLX CI lane).
            installed = False
        assert _text_lane_loads_model_type(arch) is installed
    # An existing-but-unreviewed module is not proof: even with mlx-lm
    # installed, plain-text arches keep the guard (fail closed).
    assert _text_lane_loads_model_type("llama") is False
    assert _text_lane_loads_model_type("qwen3") is False
    assert _text_lane_loads_model_type("prism_hadamard_qwen35") is False
    assert _text_lane_loads_model_type("brand_new_vlm_arch") is False
    # A malformed arch name fails closed through the probe's error guard.
    assert _text_lane_loads_model_type("qwen3.") is False


def test_multimodal_config_with_plain_text_arch_keeps_the_guard(monkeypatch):
    """Regression (pr_validate codex, round 4): a MULTIMODAL checkpoint whose
    top-level ``model_type`` is a plain text arch (``llama`` — its mlx-lm
    module exists but cannot consume the vision layout) must NOT degrade:
    the [vision] guard stays and the checkpoint is never routed to a lane
    that would crash at load with a worse error."""
    from rapid_mlx.api.utils import checkpoint_serves_text_without_vision

    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(
        monkeypatch,
        {
            "model_type": "llama",
            "architectures": ["LlamaForConditionalGeneration"],
            "vision_config": {},
        },
    )
    assert checkpoint_serves_text_without_vision("gemma-4-26b-4bit") is False


def test_listed_arch_with_module_degrades(monkeypatch):
    """The allow-list's positive path: a multimodal config whose arch is
    reviewed (``qwen3_vl``) degrades exactly when the mlx-lm module is
    installed — asserted against a direct find_spec so the test holds on the
    no-MLX CI lane (guard keeps) AND a full install (degrade applies)."""
    import importlib.util

    from rapid_mlx.api.utils import checkpoint_serves_text_without_vision

    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(
        monkeypatch,
        {
            "model_type": "qwen3_vl",
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "vision_config": {},
        },
    )
    try:
        installed = importlib.util.find_spec("mlx_lm.models.qwen3_vl") is not None
    except ImportError:
        installed = False
    assert checkpoint_serves_text_without_vision("qwen3-vl-8b-4bit") is installed


def test_degrade_probe_is_cache_only_and_cold_cache_fails_closed(monkeypatch):
    """The probe NEVER fetches: a cold cache fails closed to the safe
    [vision]-required default — materializing the config is the boot guard's
    job (:func:`_prefetch_config_for_degrade_probe`), not the probe's."""
    import huggingface_hub

    from rapid_mlx.api import utils as api_utils
    from rapid_mlx.api.utils import checkpoint_serves_text_without_vision

    _mock_vision_absent(monkeypatch)
    monkeypatch.setattr(api_utils, "read_model_metadata", lambda _name: None)

    def _no_network(*a, **k):
        raise AssertionError("the degrade probe must stay cache-only")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _no_network)
    assert checkpoint_serves_text_without_vision("gemma-4-26b-4bit") is False


def test_resolver_never_touches_the_network(monkeypatch):
    """``resolve_serving_lane_decision``'s offline contract holds on a cold
    cache: the degrade probe answers from the cache alone (fail closed), with
    zero Hub calls — even on the ABSENT-runtime path."""
    import huggingface_hub

    from rapid_mlx.api import utils as api_utils
    from rapid_mlx.api.utils import resolve_serving_lane_decision

    _patch_lane_probes(monkeypatch, is_mllm=True, cache_mode=None)
    _mock_vision_absent(monkeypatch)
    monkeypatch.setattr(api_utils, "read_model_metadata", lambda _name: None)

    def _no_network(*a, **k):
        raise AssertionError("the lane resolver must stay cache-only")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _no_network)
    decision = resolve_serving_lane_decision("gemma-4-26b-4bit")
    # No config is no evidence: the safe MLLM-lane default stands.
    assert decision.is_mllm is True
    assert decision.auto_text_fallback is False


def test_degrade_probe_prefetch_bounded_once_and_exact_repo(monkeypatch):
    """The boot guard's prefetch: targets the exact repo's ``config.json``
    only, under the shared Hub deadline, at most once per process — and Hub
    failures are swallowed (best-effort, never fatal)."""
    import huggingface_hub

    from rapid_mlx import _download_gate, model_metadata
    from rapid_mlx.api import utils as api_utils

    monkeypatch.setattr(api_utils, "_DEGRADE_CONFIG_PREFETCHED", set())
    # The hermetic suite pins HF_HUB_OFFLINE=1; the prefetch's offline
    # short-circuit is pinned separately (see the skips test below).
    monkeypatch.setattr(model_metadata, "hub_offline_mode_active", lambda: False)

    deadlines = []
    real_call_with_deadline = _download_gate.call_with_deadline

    def _recording_deadline(fn, timeout, /, *args, **kwargs):
        deadlines.append(timeout)
        return real_call_with_deadline(fn, timeout, *args, **kwargs)

    monkeypatch.setattr(_download_gate, "call_with_deadline", _recording_deadline)

    calls = []

    def _record(repo, filename, **kwargs):
        calls.append((repo, filename))
        return "cfg"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _record)

    api_utils._prefetch_config_for_degrade_probe("org/checkpoint")
    # A repeated probe within the same process never refetches.
    api_utils._prefetch_config_for_degrade_probe("org/checkpoint")
    assert calls == [("org/checkpoint", "config.json")]
    assert deadlines == [_download_gate._HF_RESOLVE_TIMEOUT_SECONDS]

    def _boom(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom)
    monkeypatch.setattr(api_utils, "_DEGRADE_CONFIG_PREFETCHED", set())
    api_utils._prefetch_config_for_degrade_probe("org/unreachable")  # must not raise


def test_degrade_probe_prefetch_skips_local_dir_and_offline(monkeypatch, tmp_path):
    """A real local directory is served in place and offline mode is honoured:
    neither ever touches the Hub."""
    import huggingface_hub

    from rapid_mlx import model_metadata
    from rapid_mlx.api import utils as api_utils

    calls = []
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **k: calls.append(a)
    )

    local_dir = tmp_path / "snapshot"
    local_dir.mkdir()
    api_utils._prefetch_config_for_degrade_probe(str(local_dir))
    assert calls == []

    monkeypatch.setattr(model_metadata, "hub_offline_mode_active", lambda: True)
    api_utils._prefetch_config_for_degrade_probe("org/checkpoint")
    assert calls == []


def test_degrade_probe_rejects_config_without_model_type(monkeypatch):
    """A vision config with no top-level ``model_type`` gives the probe no
    dispatch key: fail closed."""
    from rapid_mlx.api.utils import checkpoint_serves_text_without_vision

    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(
        monkeypatch,
        {"architectures": ["SomeForConditionalGeneration"], "vision_config": {}},
    )
    assert checkpoint_serves_text_without_vision("gemma-4-26b-4bit") is False


def test_degrade_probe_fails_closed_without_config(monkeypatch):
    """No config (offline / unreachable Hub) is no evidence: keep the safe
    [vision]-required default."""
    from rapid_mlx.api import utils as api_utils
    from rapid_mlx.api.utils import checkpoint_serves_text_without_vision

    _mock_vision_absent(monkeypatch)
    monkeypatch.setattr(api_utils, "read_model_metadata", lambda _name: None)
    assert checkpoint_serves_text_without_vision("gemma-4-26b-4bit") is False


def test_degrade_probe_rejects_text_config(monkeypatch):
    """A config that declares no vision tower never needed the vision lane."""
    from rapid_mlx.api.utils import checkpoint_serves_text_without_vision

    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(monkeypatch, TEXT_CONFIG)
    assert checkpoint_serves_text_without_vision("gemma-4-26b-4bit") is False


def test_degrade_probe_rejects_unknown_arch(monkeypatch):
    """An arch with no text-lane loader fails closed instead of crashing deep
    in mlx-lm after a silent degrade."""
    from rapid_mlx.api.utils import checkpoint_serves_text_without_vision

    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(monkeypatch, UNKNOWN_VLM_CONFIG)
    assert checkpoint_serves_text_without_vision("gemma-4-26b-4bit") is False


def test_degrade_probe_requires_absent_runtime(monkeypatch):
    """A working (OK) vision runtime serves the vision lane — the probe must
    not degrade anything."""
    from rapid_mlx.api.utils import checkpoint_serves_text_without_vision
    from rapid_mlx.models import mllm as mllm_mod

    monkeypatch.setattr(
        mllm_mod,
        "vision_runtime_status",
        lambda: (mllm_mod.VisionRuntimeStatus.OK, None),
    )
    _patch_degrade_config(monkeypatch, GEMMA4_VLM_CONFIG)
    assert checkpoint_serves_text_without_vision("gemma-4-26b-4bit") is False


def test_degrade_probe_keys_on_top_level_model_type(monkeypatch):
    """The top-level ``model_type`` is the loader dispatch key: a nested
    ``text_config`` label cannot smuggle an unloadable pack into a degrade."""
    from rapid_mlx.api.utils import checkpoint_serves_text_without_vision

    _mock_vision_absent(monkeypatch)
    _patch_degrade_config(
        monkeypatch,
        {
            "model_type": "prism_hadamard_qwen35",
            "base_model_type": "qwen3_5",
            "text_config": {"model_type": "qwen3_5"},
            "vision_config": {},
        },
    )
    assert checkpoint_serves_text_without_vision("bonsai2-27b-2bit") is False
