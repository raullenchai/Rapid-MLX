"""Image pulls price and fetch the runtime layout, excluding duplicate weights."""

from fnmatch import fnmatchcase
from types import SimpleNamespace

import pytest

from rapid_mlx import cli
from rapid_mlx.model_aliases import resolve_hub_model, subfolder_allow_patterns


@pytest.fixture
def klein_info():
    files = {
        "model.safetensors": 11_100_000_000,
        "ae.safetensors": 340_000_000,
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": 9_910_000_000,
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": 1_190_000_000,
        "text_encoder/model-00001-of-00002.safetensors": 4_970_000_000,
        "text_encoder/model-00002-of-00002.safetensors": 3_910_000_000,
        "vae/diffusion_pytorch_model.safetensors": 340_000_000,
        "variants/fp8r/model.safetensors": 6_270_000_000,
        "variants/nvfp4/model.safetensors": 5_280_000_000,
        "launch-gallery.jpg": 100_000,
    }
    for name in (
        "config.json",
        "LICENSE.md",
        "README.md",
        "transformer/config.json",
        "text_encoder/config.json",
        "vae/config.json",
        "text_encoder/tokenizer.json",
        "text_encoder/merges.txt",
        "text_encoder/chat_template.jinja",
        "variants/fp8r/config.json",
        "variants/nvfp4/manifest.json",
    ):
        files[name] = 1000
    return SimpleNamespace(
        sha="test-pin",
        siblings=[SimpleNamespace(rfilename=n, size=s) for n, s in files.items()],
    )


@pytest.mark.parametrize("alias", ["flux3-klein", "flux3-klein-base"])
def test_klein_missing_bytes_match_selected_download(monkeypatch, klein_info, alias):
    repo = resolve_hub_model(alias)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.mflux_missing_weights", lambda *a: None
    )
    monkeypatch.setattr("rapid_mlx._mirror._packaged_digest_info", lambda *a: None)
    monkeypatch.setattr("huggingface_hub.model_info", lambda *a, **kw: klein_info)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **kw: None)
    size = cli._missing_download_bytes(alias)
    assert 20.3e9 < size < 20.4e9
    patterns = subfolder_allow_patterns(alias)
    assert patterns == subfolder_allow_patterns(repo)
    selected = [
        s
        for s in klein_info.siblings
        if any(fnmatchcase(s.rfilename, p) for p in patterns)
    ]
    assert size == sum(s.size for s in selected)
    assert not any(s.rfilename.startswith("variants/") for s in selected)
    assert not any(
        s.rfilename in {"model.safetensors", "ae.safetensors", "launch-gallery.jpg"}
        for s in selected
    )
    # A 37.8 GB disk must admit the selected payload, including headroom.
    monkeypatch.setattr(
        "rapid_mlx._mirror._cache_disk_space", lambda: ("cache", int(37.8e9))
    )
    assert cli._download_disk_space_status(size)[0] is True


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("alias", ["flux3-klein", "flux3-klein-base"])
def test_pull_uses_klein_selection_for_transfer_and_verification(
    tmp_path, monkeypatch, alias, mirror
):
    from rapid_mlx.model_aliases import resolve_revision

    repo = resolve_hub_model(alias)
    pin = resolve_revision(alias)
    patterns = subfolder_allow_patterns(alias)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(
        "rapid_mlx._version_check.print_staleness_warning_if_any", lambda: None
    )
    monkeypatch.setattr(cli, "_print_pull_summary", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_sync_pulled_variant_marker", lambda *a: None)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    seen = []

    def transfer(*a, **kw):
        assert kw["allow_patterns"] == patterns
        assert kw["revision"] == pin
        seen.append("mirror")
        return mirror

    def snapshot(*a, **kw):
        assert kw == {"revision": pin, "allow_patterns": patterns}
        seen.append("hf")
        return str(tmp_path / "snapshots" / pin)

    def verify(*a, **kw):
        assert kw["allow_patterns"] == patterns
        assert kw["revision"] == pin
        seen.append("verify")
        return False

    monkeypatch.setattr("rapid_mlx._mirror.download_with_mirror_fallback", transfer)
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    monkeypatch.setattr(cli, "_verify_pulled_snapshot", verify)
    cli._pull_repository(
        SimpleNamespace(model=repo, bits=None, format=None),
        revision_override=pin,
        emit_lifecycle_event=False,
    )
    assert seen == (["mirror", "verify"] if mirror else ["mirror", "hf", "verify"])


@pytest.mark.parametrize("alias", ["flux3-klein", "flux3-klein-base"])
def test_klein_packaged_inventory_and_live_size_agree(monkeypatch, alias):
    from rapid_mlx import _download_gate as gate
    from rapid_mlx import _mirror
    from rapid_mlx.model_aliases import resolve_revision
    from rapid_mlx.model_sizes import size_bytes

    repo = resolve_hub_model(alias)
    patterns = subfolder_allow_patterns(alias)
    info = _mirror._packaged_digest_info(repo, resolve_revision(alias), patterns)
    assert info is not None  # Retain offline metadata and integrity verification.
    monkeypatch.setattr(
        "huggingface_hub.model_info", lambda *a, **k: pytest.fail("network")
    )
    monkeypatch.setattr(
        gate, "_model_info_with_timeout", lambda *a: pytest.fail("network")
    )
    monkeypatch.setattr(gate, "mflux_missing_weights", lambda *a: None)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    selected = [
        s for s in info.siblings if any(fnmatchcase(s.rfilename, p) for p in patterns)
    ]
    missing = cli._missing_download_bytes(alias)
    assert missing == sum(s.size for s in selected)
    assert missing == pytest.approx(size_bytes(repo), rel=0.002)
    assert gate.estimate_repo_size_bytes(alias) == pytest.approx(missing, abs=100_000)


@pytest.mark.parametrize("alias", ["flux3-klein", "flux3-klein-base"])
def test_klein_runtime_cold_download_uses_same_selection(monkeypatch, alias):
    from rapid_mlx.image.engine import ImageGenerationEngine
    from rapid_mlx.model_aliases import resolve_revision

    repo = resolve_hub_model(alias)
    engine = ImageGenerationEngine(repo)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.mflux_local_snapshot", lambda *a: None
    )
    monkeypatch.setattr(engine, "_verify_weights_complete", lambda: None)

    def download(model, revision, **kw):
        assert model == repo
        assert revision == resolve_revision(alias)
        assert kw["allow_patterns"] == subfolder_allow_patterns(alias)
        return "/snapshot"

    monkeypatch.setattr("rapid_mlx._mirror.pinned_snapshot_download", download)
    assert engine._resolve_model_path_for_mflux() == "/snapshot"


@pytest.mark.parametrize("alias", ["flux3-klein", "flux3-klein-base"])
def test_klein_live_estimate_filters_hub_siblings(monkeypatch, klein_info, alias):
    from rapid_mlx import _download_gate as gate

    monkeypatch.setattr("rapid_mlx._mirror._packaged_digest_info", lambda *a: None)
    monkeypatch.setattr(gate, "_model_info_with_timeout", lambda *a: klein_info)
    assert 20.3e9 < gate.estimate_repo_size_bytes(alias) < 20.4e9


def test_partial_digest_inventory_cannot_prove_whole_repository(tmp_path, monkeypatch):
    import json

    from rapid_mlx import _mirror

    monkeypatch.setattr(_mirror, "__file__", str(tmp_path / "_mirror.py"))
    (tmp_path / "model_digests.json").write_text(
        json.dumps(
            {
                "aliases": {
                    "partial": {
                        "repo": "owner/model",
                        "revision": "pin",
                        "allow_patterns": ["transformer/*"],
                        "files": [
                            {
                                "path": "transformer/model.safetensors",
                                "size": 1,
                                "sha256": "a" * 64,
                            }
                        ],
                    }
                },
            }
        )
    )
    assert _mirror._packaged_digest_info("owner/model", "pin") is None
    assert _mirror._packaged_digest_info("owner/model", "pin", ["vae/*"]) is None
