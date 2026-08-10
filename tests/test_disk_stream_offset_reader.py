# SPDX-License-Identifier: Apache-2.0
"""``vllm_mlx.offset_reader`` — byte-offset / direct-name expert-bundle fetch.

Tickets: ``.scratch/rapid-mlx-disk-stream/issues/01-registry-offset-reader-lfm25.md``
(``"stacked"`` layout, LFM2.5) and
``.scratch/rapid-mlx-disk-stream/issues/05-qwen2-moe-shared-expert-arch.md``
(``"direct"`` layout, qwen2_moe).

Two tiers:

1. Synthetic-fixture tests (this file's majority) — build a tiny local
   safetensors file with known values via ``mx.save_safetensors`` and
   assert ``fetch_expert_bundle`` reads it correctly for both layouts. No
   checkpoint download, no real-model load; these run in the default
   ``pytest tests/`` suite with no ``--run-slow``.
2. Two ``@pytest.mark.slow`` bit-exact tests against the real, already
   locally-cached LFM2.5 and Qwen1.5-MoE checkpoints (each skips cleanly
   if its checkpoint isn't present — never downloads, never required for
   the rest of the suite to pass).

Ticket 01 shipped a ``test_direct_layout_not_implemented`` test asserting
``"direct"`` raised ``NotImplementedError`` — by that ticket's own design
("out of scope for this ticket / LFM2.5 ... a later ticket", see
``registry.py``'s ``ExpertTensorTemplate`` docstring at the time). Ticket
05 implements that layout, so that test is replaced below by
``test_unknown_layout_raises_not_implemented`` (pinning the NotImplementedError
contract for a genuinely unrecognized third layout value) plus real
``"direct"``-layout coverage.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import pytest

from vllm_mlx.offset_reader import fetch_expert_bundle
from vllm_mlx.registry import (
    ExpertTensorTemplate,
    StreamingAdapter,
    get_adapter,
)

NUM_EXPERTS = 4
OUT_DIM = 6
IN_DIM = 8

# Mirrors LFM2.5's real on-disk dtypes: weight is the packed-quant integer
# tensor (uint32 here as a stand-in — the byte-slicing math doesn't care
# about the *meaning* of the dtype, only its size), scales/biases are
# bfloat16 (exercises the mx.view bit-reinterpret path).
_DTYPES = {"weight": mx.uint32, "scales": mx.bfloat16, "biases": mx.bfloat16}


def _stacked_fixture_adapter() -> StreamingAdapter:
    return StreamingAdapter(
        model_type="_test_stacked",
        moe_block_module="not.imported.in.this.test",
        moe_block_class_name="Unused",
        tensor_template=ExpertTensorTemplate(
            layout="stacked",
            name_template="layer{layer}.{proj}.{component}",
        ),
        num_experts=NUM_EXPERTS,
    )


@pytest.fixture
def stacked_checkpoint(tmp_path: Path) -> tuple[Path, dict]:
    """Write a small synthetic safetensors file with one MoE layer's
    gate/up/down x weight/scales/biases tensors, each stacked on axis 0
    across ``NUM_EXPERTS`` experts with distinct, checkable values per
    expert (``expert_id`` broadcast into every element). Returns the file
    path plus the source per-(proj, component) full stacked arrays so
    tests can slice the same way ``fetch_expert_bundle`` is expected to.
    """
    tensors: dict[str, mx.array] = {}
    sources: dict[str, mx.array] = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for component in ("weight", "scales", "biases"):
            dtype = _DTYPES[component]
            # Each expert's slice is filled with a distinct constant
            # (100*expert + a per-tensor offset) so a wrong offset/expert
            # index shows up as a value mismatch, not just a shape match.
            offset = hash((proj, component)) % 7
            per_expert = [
                mx.full((OUT_DIM, IN_DIM), 100 * e + offset, dtype=dtype)
                for e in range(NUM_EXPERTS)
            ]
            stacked = mx.stack(per_expert, axis=0)
            name = f"layer3.{proj}.{component}"
            tensors[name] = stacked
            sources[f"{proj}.{component}"] = stacked

    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), tensors)
    return path, sources


def test_fetch_expert_bundle_returns_nine_tensors(stacked_checkpoint):
    path, _sources = stacked_checkpoint
    adapter = _stacked_fixture_adapter()

    bundle = fetch_expert_bundle(adapter, path, layer_idx=3, expert_id=1)

    assert set(bundle.keys()) == {"gate_proj", "up_proj", "down_proj"}
    for proj in bundle:
        assert set(bundle[proj].keys()) == {"weight", "scales", "biases"}


def test_fetch_expert_bundle_slices_correct_expert_and_matches_dtype_shape(
    stacked_checkpoint,
):
    path, sources = stacked_checkpoint
    adapter = _stacked_fixture_adapter()

    for expert_id in range(NUM_EXPERTS):
        bundle = fetch_expert_bundle(adapter, path, layer_idx=3, expert_id=expert_id)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for component in ("weight", "scales", "biases"):
                expected = sources[f"{proj}.{component}"][expert_id]
                got = bundle[proj][component]
                assert got.shape == expected.shape
                assert got.dtype == expected.dtype
                assert mx.array_equal(got, expected), (
                    f"expert={expert_id} proj={proj} component={component} "
                    "mismatch — offset math sliced the wrong expert or "
                    "byte range"
                )


def test_fetch_expert_bundle_different_experts_are_distinct(stacked_checkpoint):
    """Sanity check that the fixture (and the slicer) actually vary by
    expert — guards against a fixture/test bug where every expert
    accidentally got the same values and the equality checks above would
    pass trivially regardless of whether ``expert_id`` was even used.
    """
    path, _sources = stacked_checkpoint
    adapter = _stacked_fixture_adapter()

    bundle_0 = fetch_expert_bundle(adapter, path, layer_idx=3, expert_id=0)
    bundle_2 = fetch_expert_bundle(adapter, path, layer_idx=3, expert_id=2)
    assert not mx.array_equal(
        bundle_0["gate_proj"]["weight"], bundle_2["gate_proj"]["weight"]
    )


def test_fetch_expert_bundle_stacked_layout_resolves_checkpoint_directory(
    stacked_checkpoint,
):
    """Real CLI bug repro: ``_resolve_model_path`` (``vllm_mlx/utils/
    tokenizer.py``) hands ``disk_stream_patch.install`` a checkpoint
    *directory* (the HF-cache snapshot dir), not a bare file — the
    ``"direct"`` layout already handled this via ``_resolve_shard_path``,
    but the ``"stacked"`` layout (LFM2.5) passed ``path`` straight to
    ``_fetch_tensor_slice``, which does ``open(path, "rb")`` and raised
    ``IsADirectoryError`` when ``path`` was a directory containing
    ``model.safetensors`` rather than the file itself.
    """
    path, sources = stacked_checkpoint
    checkpoint_dir = (
        path.parent
    )  # stacked_checkpoint already writes model.safetensors here
    assert path.name == "model.safetensors"
    adapter = _stacked_fixture_adapter()

    bundle = fetch_expert_bundle(adapter, checkpoint_dir, layer_idx=3, expert_id=2)

    for proj in ("gate_proj", "up_proj", "down_proj"):
        for component in ("weight", "scales", "biases"):
            expected = sources[f"{proj}.{component}"][2]
            assert mx.array_equal(bundle[proj][component], expected)


def test_fetch_expert_bundle_stacked_layout_rejects_negative_expert_id(
    stacked_checkpoint,
):
    """A negative ``expert_id`` must raise a clear ``ValueError`` naming the
    offending id and ``num_experts`` -- not silently wrap around (Python's
    negative-indexing-like arithmetic) and return a *different* expert's
    bytes with no signal anything went wrong.
    """
    path, _sources = stacked_checkpoint
    adapter = _stacked_fixture_adapter()

    with pytest.raises(ValueError) as exc_info:
        fetch_expert_bundle(adapter, path, layer_idx=3, expert_id=-1)
    message = str(exc_info.value)
    assert "-1" in message
    assert str(NUM_EXPERTS) in message


def test_fetch_expert_bundle_stacked_layout_rejects_expert_id_at_or_above_num_experts(
    stacked_checkpoint,
):
    """An out-of-range (too-high) ``expert_id`` must raise a clear
    ``ValueError`` rather than reading whatever bytes happen to follow the
    stacked tensor in the file -- which may or may not hit EOF depending on
    file layout, so it must not be allowed to fail only "by accident".
    """
    path, _sources = stacked_checkpoint
    adapter = _stacked_fixture_adapter()

    with pytest.raises(ValueError) as exc_info:
        fetch_expert_bundle(adapter, path, layer_idx=3, expert_id=NUM_EXPERTS)
    message = str(exc_info.value)
    assert str(NUM_EXPERTS) in message


def test_unknown_layout_raises_not_implemented(stacked_checkpoint):
    """A genuinely unrecognized ``layout`` value (neither ``"stacked"`` nor
    ``"direct"``) must still fail loudly rather than silently mis-reading
    bytes — this is what ticket 01's original ``"direct"``-layout test
    pinned before ticket 05 implemented that layout for real (see the
    module docstring).
    """
    path, _sources = stacked_checkpoint
    adapter = StreamingAdapter(
        model_type="_test_bogus_layout",
        moe_block_module="not.imported.in.this.test",
        moe_block_class_name="Unused",
        tensor_template=ExpertTensorTemplate(
            layout="bogus",
            name_template="layer{layer}.{proj}.{component}",
        ),
        num_experts=NUM_EXPERTS,
    )

    with pytest.raises(NotImplementedError):
        fetch_expert_bundle(adapter, path, layer_idx=3, expert_id=0)


# ---------------------------------------------------------------------
# "direct" layout (qwen2_moe-style, per-expert-named-tensor) — ticket 05.
# ---------------------------------------------------------------------


def _direct_fixture_adapter() -> StreamingAdapter:
    return StreamingAdapter(
        model_type="_test_direct",
        moe_block_module="not.imported.in.this.test",
        moe_block_class_name="Unused",
        tensor_template=ExpertTensorTemplate(
            layout="direct",
            name_template="model.layers.{layer}.mlp.experts.{expert}.{proj}.{component}",
        ),
        num_experts=NUM_EXPERTS,
    )


@pytest.fixture
def direct_checkpoint(tmp_path: Path) -> tuple[Path, dict]:
    """Write a small synthetic safetensors file with one MoE layer's
    gate/up/down x weight/scales/biases tensors, each expert its own
    directly-named tensor (no axis-0 stacking — the qwen2_moe layout).
    Returns the file path plus the source per-(expert, proj, component)
    arrays so tests can compare against exactly what
    ``fetch_expert_bundle`` is expected to return.
    """
    tensors: dict[str, mx.array] = {}
    sources: dict[str, mx.array] = {}
    for expert_id in range(NUM_EXPERTS):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for component in ("weight", "scales", "biases"):
                dtype = _DTYPES[component]
                offset = hash((proj, component)) % 7
                arr = mx.full((OUT_DIM, IN_DIM), 100 * expert_id + offset, dtype=dtype)
                name = f"model.layers.3.mlp.experts.{expert_id}.{proj}.{component}"
                tensors[name] = arr
                sources[f"{expert_id}.{proj}.{component}"] = arr

    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), tensors)
    return path, sources


def test_fetch_expert_bundle_direct_layout_returns_nine_tensors(direct_checkpoint):
    path, _sources = direct_checkpoint
    adapter = _direct_fixture_adapter()

    bundle = fetch_expert_bundle(adapter, path, layer_idx=3, expert_id=1)

    assert set(bundle.keys()) == {"gate_proj", "up_proj", "down_proj"}
    for proj in bundle:
        assert set(bundle[proj].keys()) == {"weight", "scales", "biases"}


def test_fetch_expert_bundle_direct_layout_matches_named_tensor(direct_checkpoint):
    path, sources = direct_checkpoint
    adapter = _direct_fixture_adapter()

    for expert_id in range(NUM_EXPERTS):
        bundle = fetch_expert_bundle(adapter, path, layer_idx=3, expert_id=expert_id)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for component in ("weight", "scales", "biases"):
                expected = sources[f"{expert_id}.{proj}.{component}"]
                got = bundle[proj][component]
                assert got.shape == expected.shape
                assert got.dtype == expected.dtype
                assert mx.array_equal(got, expected), (
                    f"expert={expert_id} proj={proj} component={component} "
                    "mismatch — direct-layout name lookup fetched the "
                    "wrong tensor"
                )


def test_fetch_expert_bundle_direct_layout_resolves_shard_via_index_json(tmp_path):
    """Ticket 05's real checkpoint (Qwen1.5-MoE) is sharded across 2 files.
    ``path`` passed as a checkpoint *directory* must resolve each tensor's
    shard via ``model.safetensors.index.json``'s ``weight_map`` — this is
    what makes ``fetch_expert_bundle`` work against the shape
    ``disk_stream_patch.install`` actually receives from the real CLI flow
    (``_resolve_model_path`` returns a directory, not a file).
    """
    adapter = _direct_fixture_adapter()
    name = "model.layers.3.mlp.experts.2.gate_proj.weight"
    other_name = "model.layers.3.mlp.experts.2.up_proj.weight"

    shard_a = tmp_path / "model-00001-of-00002.safetensors"
    shard_b = tmp_path / "model-00002-of-00002.safetensors"
    expected = mx.full((OUT_DIM, IN_DIM), 42, dtype=mx.uint32)
    mx.save_safetensors(str(shard_a), {name: expected})
    mx.save_safetensors(
        str(shard_b), {other_name: mx.full((OUT_DIM, IN_DIM), 7, dtype=mx.uint32)}
    )
    index = {
        "weight_map": {
            name: shard_a.name,
            other_name: shard_b.name,
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    from vllm_mlx.offset_reader import _fetch_tensor_slice

    got = _fetch_tensor_slice(shard_a, name)  # sanity: value really is in shard_a
    assert mx.array_equal(got, expected)

    from vllm_mlx.offset_reader import _resolve_shard_path

    assert _resolve_shard_path(tmp_path, name) == shard_a
    assert _resolve_shard_path(tmp_path, other_name) == shard_b


def test_resolve_shard_path_rejects_absolute_shard_name_outside_checkpoint_dir(
    tmp_path,
):
    """A malformed/adversarial ``index.json`` whose ``weight_map`` maps a
    tensor name to an *absolute* path must not let ``_resolve_shard_path``
    escape the checkpoint directory. ``pathlib``'s ``/`` operator discards
    the left operand entirely when the right operand is absolute
    (``Path("/ckpt") / "/etc/passwd" == Path("/etc/passwd")``), so without
    a containment check this would resolve straight to the outside file.
    """
    from vllm_mlx.offset_reader import _resolve_shard_path

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    secret = outside_dir / "secret.safetensors"
    secret.write_bytes(b"not a real safetensors file")

    tensor_name = "model.layers.3.mlp.experts.2.gate_proj.weight"
    index = {"weight_map": {tensor_name: str(secret)}}
    (checkpoint_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    with pytest.raises(ValueError, match=tensor_name):
        _resolve_shard_path(checkpoint_dir, tensor_name)


def test_resolve_shard_path_rejects_dotdot_relative_shard_name(tmp_path):
    """Same containment gap, via a ``../``-prefixed relative ``shard_name``
    instead of an absolute one — the other half of the path-traversal
    surface flagged in the security review.
    """
    from vllm_mlx.offset_reader import _resolve_shard_path

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    secret = outside_dir / "secret.safetensors"
    secret.write_bytes(b"not a real safetensors file")

    tensor_name = "model.layers.3.mlp.experts.2.gate_proj.weight"
    index = {"weight_map": {tensor_name: "../outside/secret.safetensors"}}
    (checkpoint_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    with pytest.raises(ValueError, match=tensor_name):
        _resolve_shard_path(checkpoint_dir, tensor_name)


def test_resolve_shard_path_rejects_basename_symlink_escape(tmp_path):
    """A lexical basename is still unsafe when the file itself is a symlink."""
    from vllm_mlx.offset_reader import _resolve_shard_path

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    secret = tmp_path / "secret.safetensors"
    secret.write_bytes(b"outside")
    (checkpoint_dir / "model-00001.safetensors").symlink_to(secret)

    tensor_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    index = {"weight_map": {tensor_name: "model-00001.safetensors"}}
    (checkpoint_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    with pytest.raises(ValueError, match="outside checkpoint"):
        _resolve_shard_path(checkpoint_dir, tensor_name)


# ---------------------------------------------------------------------
# Defensive/error-path coverage for `_fetch_tensor_slice` /
# `_resolve_shard_path` -- malformed/inconsistent headers and sharded
# lookup misses. `mx.save_safetensors` always produces a self-consistent
# header, so these hand-craft the raw safetensors bytes (8-byte
# little-endian header length + JSON header + data) to exercise headers
# that don't agree with their own declared shape/dtype/data_offsets.
# ---------------------------------------------------------------------


def _write_raw_safetensors_file(path: Path, header: dict, data: bytes) -> None:
    """Write a safetensors file from an exact, hand-crafted header + data
    section, bypassing ``mx.save_safetensors``'s always-consistent output --
    lets tests construct the malformed headers the defensive ``ValueError``
    branches in ``_fetch_tensor_slice`` are meant to catch.
    """
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(data)


def test_fetch_tensor_slice_raises_on_uneven_stacking(tmp_path):
    """A stacked-layout tensor whose byte range doesn't evenly divide by its
    own declared ``shape[0]`` (num_experts) must raise a clear ``ValueError``
    naming the tensor, not silently mis-slice.
    """
    from vllm_mlx.offset_reader import _fetch_tensor_slice

    path = tmp_path / "model.safetensors"
    # shape (3, 2, 2) F32 = 48 bytes total, but data_offsets claims only 47
    # bytes -- 47 is not evenly divisible by num_experts=3.
    header = {
        "layer0.tensor": {"dtype": "F32", "shape": [3, 2, 2], "data_offsets": [0, 47]}
    }
    _write_raw_safetensors_file(path, header, data=b"\x00" * 48)

    with pytest.raises(ValueError, match="not evenly stacked"):
        _fetch_tensor_slice(path, "layer0.tensor", expert_id=0)


def test_fetch_tensor_slice_raises_on_byte_range_size_mismatch(tmp_path):
    """A tensor whose declared byte range doesn't match its own declared
    shape x dtype size (a corrupted/inconsistent header) must raise a clear
    ``ValueError`` rather than silently reshaping the wrong amount of data.
    """
    from vllm_mlx.offset_reader import _fetch_tensor_slice

    path = tmp_path / "model.safetensors"
    # shape (2, 2) F32 expects 16 bytes; data_offsets claims only 12.
    header = {
        "layer0.tensor": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 12]}
    }
    _write_raw_safetensors_file(path, header, data=b"\x00" * 16)

    with pytest.raises(ValueError, match="byte-range size"):
        _fetch_tensor_slice(path, "layer0.tensor")


def test_fetch_tensor_slice_raises_on_short_read(tmp_path):
    """If the file is truncated so fewer bytes exist than the header
    promises, the resulting short read must raise a clear ``ValueError``
    instead of silently returning a too-small/garbage array.
    """
    from vllm_mlx.offset_reader import _fetch_tensor_slice

    path = tmp_path / "model.safetensors"
    # shape (2, 2) F32 = 16 bytes, header says data_offsets=[0, 16], but the
    # file only actually has 8 bytes of data after the header.
    header = {
        "layer0.tensor": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}
    }
    _write_raw_safetensors_file(path, header, data=b"\x00" * 8)

    with pytest.raises(ValueError, match="short read"):
        _fetch_tensor_slice(path, "layer0.tensor")


def test_resolve_shard_path_raises_keyerror_on_missing_tensor(tmp_path):
    """A tensor name absent from ``index.json``'s ``weight_map`` entirely
    (as opposed to a hit) must raise a clear ``KeyError`` naming the tensor,
    not silently return a bogus/None-derived path.
    """
    from vllm_mlx.offset_reader import _resolve_shard_path

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    index = {"weight_map": {"some.other.tensor": "model-00001-of-00001.safetensors"}}
    (checkpoint_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    missing_name = "model.layers.3.mlp.experts.2.gate_proj.weight"
    with pytest.raises(KeyError) as exc_info:
        _resolve_shard_path(checkpoint_dir, missing_name)
    assert missing_name in str(exc_info.value)


# ---------------------------------------------------------------------
# Regression coverage: header/index.json metadata must be parsed once per
# `fetch_expert_bundle` call (9 tensors sharing one shard's header, or one
# checkpoint's index.json), not once per tensor.
# ---------------------------------------------------------------------


def test_fetch_expert_bundle_parses_header_once_per_shard_not_per_tensor(
    stacked_checkpoint, monkeypatch
):
    """A single-shard 9-tensor bundle fetch must parse that shard's header
    exactly once, not once per tensor (9x).
    """
    import vllm_mlx.offset_reader as offset_reader_module

    path, _sources = stacked_checkpoint
    adapter = _stacked_fixture_adapter()

    call_count = 0
    real_read_header = offset_reader_module._read_header

    def counting_read_header(p):
        nonlocal call_count
        call_count += 1
        return real_read_header(p)

    monkeypatch.setattr(offset_reader_module, "_read_header", counting_read_header)

    fetch_expert_bundle(adapter, path, layer_idx=3, expert_id=1)

    assert call_count == 1, (
        f"expected 1 header parse for a single-shard bundle, got {call_count}"
    )


def test_fetch_expert_bundle_parses_index_json_once_per_call(tmp_path, monkeypatch):
    """A sharded 'direct'-layout checkpoint's ``index.json`` must be parsed
    once per ``fetch_expert_bundle`` call, not once per tensor lookup (9x).
    """
    import vllm_mlx.offset_reader as offset_reader_module

    adapter = _direct_fixture_adapter()
    tensors = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for component in ("weight", "scales", "biases"):
            dtype = _DTYPES[component]
            name = f"model.layers.3.mlp.experts.1.{proj}.{component}"
            tensors[name] = mx.full((OUT_DIM, IN_DIM), 1, dtype=dtype)

    shard = tmp_path / "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(shard), tensors)
    index = {"weight_map": {name: shard.name for name in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    call_count = 0
    real_load_weight_map = offset_reader_module._load_weight_map

    def counting_load_weight_map(p):
        nonlocal call_count
        call_count += 1
        return real_load_weight_map(p)

    monkeypatch.setattr(
        offset_reader_module, "_load_weight_map", counting_load_weight_map
    )

    fetch_expert_bundle(adapter, tmp_path, layer_idx=3, expert_id=1)

    assert call_count == 1, f"expected 1 index.json parse per call, got {call_count}"


# ---------------------------------------------------------------------
# Bit-exact check against the real, already-locally-cached LFM2.5
# checkpoint. Never downloads; skips cleanly (does not fail the rest of
# the suite) if the checkpoint isn't present locally.
# ---------------------------------------------------------------------


def _local_lfm25_checkpoint() -> Path | None:
    snapshots = (
        Path.home()
        / ".cache/huggingface/hub/models--mlx-community--LFM2.5-8B-A1B-MLX-4bit"
        / "snapshots"
    )
    if not snapshots.is_dir():
        return None
    for snapshot_dir in snapshots.iterdir():
        candidate = snapshot_dir / "model.safetensors"
        if candidate.is_file():
            return candidate
    return None


@pytest.mark.slow
def test_fetch_expert_bundle_bit_exact_against_loaded_model():
    """``fetch_expert_bundle`` on LFM2.5's real registry adapter returns
    exactly the same 9 tensors as the same (layer, expert) slice read off
    a fully in-RAM-loaded model — the guarantee the whole feature depends
    on. Requires ``--run-slow`` (real model load) and the checkpoint
    already present in the local HF cache (reused from the
    ``.scratch/moe-disk-stream`` spike, never re-downloaded here).
    """
    checkpoint_path = _local_lfm25_checkpoint()
    if checkpoint_path is None:
        pytest.skip(
            "LFM2.5-8B-A1B-MLX-4bit not found in local HF cache — "
            "bit-exact check needs the spike's already-downloaded checkpoint"
        )

    from mlx_lm import load

    adapter = get_adapter("lfm2_moe")
    assert adapter is not None

    model, _tokenizer = load(str(checkpoint_path.parent))

    layer_idx, expert_id = 2, 5  # same pair the spike's self-check used
    bundle = fetch_expert_bundle(adapter, checkpoint_path, layer_idx, expert_id)

    switch_mlp = model.layers[layer_idx].feed_forward.switch_mlp
    for proj in ("gate_proj", "up_proj", "down_proj"):
        ref_layer = getattr(switch_mlp, proj)
        for component in ("weight", "scales", "biases"):
            reference = getattr(ref_layer, component)[expert_id]
            fetched = bundle[proj][component]
            assert fetched.shape == reference.shape
            assert fetched.dtype == reference.dtype
            assert mx.array_equal(fetched, reference), (
                f"layer={layer_idx} expert={expert_id} proj={proj} "
                f"component={component}: disk-streamed bundle does not "
                "bit-exact match the fully-loaded model's in-RAM tensor"
            )


# ---------------------------------------------------------------------
# Bit-exact check against the real, already-locally-cached qwen2_moe
# checkpoint (Qwen1.5-MoE-A2.7B-Chat-4bit, "direct" layout, sharded across
# 2 safetensors files). Never downloads; skips cleanly if not present.
# ---------------------------------------------------------------------


def _local_qwen2_moe_checkpoint_dir() -> Path | None:
    snapshots = (
        Path.home()
        / ".cache/huggingface/hub/models--mlx-community--Qwen1.5-MoE-A2.7B-Chat-4bit"
        / "snapshots"
    )
    if not snapshots.is_dir():
        return None
    for snapshot_dir in snapshots.iterdir():
        if (snapshot_dir / "model.safetensors.index.json").is_file():
            return snapshot_dir
    return None


@pytest.mark.slow
def test_fetch_expert_bundle_qwen2_moe_bit_exact_against_loaded_model():
    """``fetch_expert_bundle`` on qwen2_moe's real registry adapter (
    ``"direct"`` layout, sharded checkpoint, ``path`` given as the
    checkpoint *directory* — the shape the real CLI flow actually passes)
    returns exactly the same 9 tensors as the same (layer, expert) slice
    read off a fully in-RAM-loaded model. Requires ``--run-slow`` and the
    checkpoint already present in the local HF cache (reused from the
    ``.scratch/moe-disk-stream`` spike, never re-downloaded here).
    """
    checkpoint_dir = _local_qwen2_moe_checkpoint_dir()
    if checkpoint_dir is None:
        pytest.skip(
            "Qwen1.5-MoE-A2.7B-Chat-4bit not found in local HF cache — "
            "bit-exact check needs the spike's already-downloaded checkpoint"
        )

    from mlx_lm import load

    adapter = get_adapter("qwen2_moe")
    assert adapter is not None

    model, _tokenizer = load(str(checkpoint_dir))

    layer_idx, expert_id = 2, 5  # same pair the spike's ticket 09 self-check used
    bundle = fetch_expert_bundle(adapter, checkpoint_dir, layer_idx, expert_id)

    switch_mlp = model.layers[layer_idx].mlp.switch_mlp
    for proj in ("gate_proj", "up_proj", "down_proj"):
        ref_layer = getattr(switch_mlp, proj)
        for component in ("weight", "scales", "biases"):
            reference = getattr(ref_layer, component)[expert_id]
            fetched = bundle[proj][component]
            assert fetched.shape == reference.shape
            assert fetched.dtype == reference.dtype
            assert mx.array_equal(fetched, reference), (
                f"layer={layer_idx} expert={expert_id} proj={proj} "
                f"component={component}: disk-streamed bundle does not "
                "bit-exact match the fully-loaded model's in-RAM tensor"
            )
