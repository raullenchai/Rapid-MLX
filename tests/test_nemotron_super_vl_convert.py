"""Published guidance and checkpoint path confinement for Super VL conversion."""

import json
from contextlib import nullcontext

import pytest

pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

from scripts import nemotron_super_vl_convert as convert  # noqa: E402


@pytest.mark.parametrize("bits,size", [(3, 42_764_296_516), (4, 69_540_654_086)])
def test_published_model_card_guidance(bits, size):
    card = convert.model_card(bits, size)
    assert f"Artifact size: {size / 1e9:.3f} GB." in card
    assert "Requires Rapid-MLX 0.16.2 or newer." in card
    assert "Bolt only" not in card
    assert "Memory qualification pending" not in card
    if bits == 3:
        assert (
            "Targets **64 GB** unified\n"
            "  memory for short contexts. Measured MLX allocator peak: **45.903 GB**\n"
            "  for 1,024 prompt tokens plus 64 generated tokens, with prefill chunks of 256.\n"
            "  Measurements use a 256 GiB M3 Ultra; a physical 64 GB Mac has not been tested.\n"
            "  Start at 1K context and leave memory for macOS and other applications."
        ) in card
    else:
        assert (
            "Recommend **128 GB+** unified memory (measured peak 72.7 GB at 1K context "
            "on M3 Ultra; 96 GB Macs may need a raised GPU wired-memory limit and are "
            "not yet verified).\n"
            "  Allow headroom beyond the weights for prefill, caches, macOS and applications."
        ) in card


def checkpoint(source, name):
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "nemotron_h", "layers_block_type": ["moe"]})
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"backbone.weight": name}})
    )


@pytest.mark.parametrize(
    "kind", ["absolute", "absolute_inside", "traversal", "symlink", "directory_symlink"]
)
def test_converter_rejects_source_escape_before_loading(tmp_path, monkeypatch, kind):
    source = tmp_path / "source"
    outside = tmp_path / "source-sibling"
    outside.mkdir()
    weight = outside / "model.safetensors"
    weight.write_bytes(b"must not be read")
    name = {
        "absolute": str(weight),
        "absolute_inside": str(source / "model.safetensors"),
        "traversal": "../source-sibling/model.safetensors",
        "symlink": "model.safetensors",
        "directory_symlink": "linked/model.safetensors",
    }[kind]
    checkpoint(source, name)
    if kind == "absolute_inside":
        (source / "model.safetensors").write_bytes(b"checkpoint")
    elif kind == "symlink":
        (source / name).symlink_to(weight)
    elif kind == "directory_symlink":
        (source / "linked").symlink_to(outside, target_is_directory=True)

    def forbidden(*args, **kwargs):
        pytest.fail("Unsafe checkpoint must be rejected before model loading")

    monkeypatch.setattr(convert, "load_model_checked", forbidden)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="shard path escapes"):
        convert.main(["--source", str(source), "--output", str(output), "--bits", "3"])
    assert not output.exists()
    assert weight.read_bytes() == b"must not be read"


@pytest.mark.parametrize("kind", ["symlink", "traversal"])
def test_converter_rejects_temporary_destination_escape(tmp_path, monkeypatch, kind):
    source = tmp_path / "source"
    name = (
        "nested/model.safetensors"
        if kind == "symlink"
        else "../source/model.safetensors"
    )
    checkpoint(source, name)
    (source / "nested").mkdir()
    (source / name).write_bytes(b"checkpoint")
    view = tmp_path / "view"
    view.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if kind == "symlink":
        (view / "nested").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(
        convert.tempfile, "TemporaryDirectory", lambda **kw: nullcontext(view)
    )
    monkeypatch.setattr(convert.mx, "set_default_device", lambda device: None)

    def forbidden(*args, **kwargs):
        pytest.fail("Unsafe destination must be rejected before model loading")

    monkeypatch.setattr(convert, "load_model_checked", forbidden)
    with pytest.raises(ValueError, match="shard path escapes"):
        convert.main(
            ["--source", str(source), "--output", str(tmp_path / "out"), "--bits", "4"]
        )
    assert list(outside.iterdir()) == []


def test_shard_paths_allow_contained_names_and_symlinks(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    nested = root / "nested"
    nested.mkdir()
    weight = nested / "model.safetensors"
    weight.write_bytes(b"checkpoint")
    (root / "linked.safetensors").symlink_to(weight)
    for name in ("model.safetensors", "nested/model.safetensors", "linked.safetensors"):
        assert convert.shard_path(root, name) == root / name
