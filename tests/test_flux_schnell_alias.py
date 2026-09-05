# SPDX-License-Identifier: Apache-2.0
"""Public catalog contract for the FLUX.1 schnell image alias."""

from types import SimpleNamespace

from vllm_mlx._download_gate import IMAGE_MODEL_REVISIONS
from vllm_mlx.image.engine import ImageGenerationEngine
from vllm_mlx.model_aliases import list_profiles, resolve_profile
from vllm_mlx.model_sizes import size_bytes
from vllm_mlx.runtime.resident_models import estimate_model_bytes

REPO = "mflux-community/flux-1-schnell-mflux-q4"
REVISION = "bcdbe817ad51175959b2e691e64eca626db30558"


def test_flux_schnell_alias_is_generation_only_mflux_model():
    profile = resolve_profile("flux-schnell")

    assert profile is not None
    assert profile.hf_path == REPO
    assert profile.modality == "image-gen"
    assert profile.min_memory_gb == 16

    engine = ImageGenerationEngine(profile.hf_path)
    assert engine.family == "flux-schnell"
    assert engine.supports_generation is True
    assert engine.supports_editing is False
    assert engine.default_steps == 4
    assert engine._prequantized is True
    assert engine._quantize is None


def test_flux_schnell_weights_are_revision_pinned():
    assert list_profiles()["flux-schnell"].hf_path == REPO
    assert IMAGE_MODEL_REVISIONS[REPO] == REVISION
    assert size_bytes(REPO) == 9_613_040_056
    assert estimate_model_bytes("flux-schnell") == int(9.5 * 1024**3)


def test_flux_schnell_prefetch_automatically_uses_the_pinned_revision(monkeypatch):
    """A cold serve must not fetch mutable HEAD before the pinned checkpoint."""
    from huggingface_hub import __dict__ as hub

    from vllm_mlx import cli

    monkeypatch.setattr(cli, "_cache_runnability", lambda _repo: False)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *_a, **_kw: None)

    mirror_revisions = []

    def _mirror(_repo, _on_pull_start=None, **kwargs):
        mirror_revisions.append(kwargs.get("revision"))
        return False

    monkeypatch.setattr(cli, "_try_mirror_prefetch", _mirror)
    monkeypatch.setitem(
        hub,
        "model_info",
        lambda *_a, **kwargs: SimpleNamespace(sha=kwargs.get("revision"), siblings=[]),
    )
    downloaded_revisions = []
    monkeypatch.setitem(
        hub,
        "snapshot_download",
        lambda *_a, **kwargs: downloaded_revisions.append(kwargs.get("revision")),
    )
    monkeypatch.setattr(
        "vllm_mlx._download_gate.pin_main_ref",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            AssertionError("an explicit commit must not rewrite refs/main")
        ),
    )

    cli._ensure_model_downloaded(REPO)

    assert mirror_revisions == [REVISION]
    assert downloaded_revisions == [REVISION]


def test_flux_schnell_cachedness_rejects_a_non_pinned_snapshot(monkeypatch):
    """Generic cache probes cannot let a different complete commit win."""
    from vllm_mlx import _download_gate as gate
    from vllm_mlx import cli

    monkeypatch.setattr(gate, "_snapshot_is_complete_mflux_model", lambda _r: False)
    monkeypatch.setattr(
        gate,
        "is_repo_cached",
        lambda _r: (_ for _ in ()).throw(
            AssertionError("pinned image aliases must not use generic cache probes")
        ),
    )

    assert cli._cache_runnability(REPO) is False
