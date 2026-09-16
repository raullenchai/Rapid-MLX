# SPDX-License-Identifier: Apache-2.0
"""Contract pins for the ``qwen-image`` image-gen alias.

The mflux backend already understands the ``qwen-image`` family
(``rapid_mlx/image/engine.py``); this alias is what makes
``rapid-mlx serve qwen-image`` resolve to it without the caller having
to type the raw Hugging Face path.

Repo choice, and why it is NOT ``filipstrand/Qwen-Image-mflux-6bit``
(the community's most-downloaded Qwen-Image mflux conversion): that repo
quantizes the text encoder to 6-bit (packed ``embed_tokens.weight`` shape
``[152064, 672]`` instead of the full-precision ``[152064, 3584]``), but
the mflux version this project pins (0.18.1) hardcodes
``skip_quantization=True`` for the Qwen text encoder ("quantization causes
significant semantic degradation" per mflux's own weight definition) and
never dequantizes it on load. The result is silent runtime corruption, not
a load-time error: the model builds, ``generate()`` runs, and crashes deep
in the text encoder's RMSNorm with a shape mismatch
(``[broadcast_shapes] Shapes (3584) and (1,50,672)``) — reproduced live
against this project's pinned mflux before this alias was pointed
elsewhere. ``mflux-community/qwen-image-mflux-q6`` (uploaded 2026-08-16,
commit ``c628fe4392d963557c3013c2709e6d3b67bca79d`` at the time of
pinning) keeps the text encoder full-precision (verified: zero
``scale``/``bias`` keys, correct ``[152064, 3584]`` embed shape) and was
confirmed end-to-end: both a plain text-to-image prompt and a
text-rendering prompt ("RAPID MLX" on a sign) produced correct images
with this project's pinned mflux.

Revision pinning: ``aliases.json`` has no revision-pin field (adding one
across every alias/modality would be a system-wide schema change out of
scope here), so the commit above is enforced separately — a code-only
pin table, ``_download_gate.IMAGE_MODEL_REVISIONS``, mirroring
``video/wan.py``'s ``WAN_REVISIONS``. Without it, this alias would only
ever resolve whatever ``refs/main`` currently points to (see
``_resolved_snapshot_sha``), so an upstream force-push or account
compromise on ``mflux-community``'s repo — plausible given the
provenance caveat below — would silently change the weights a fresh
pull fetches next. A cached snapshot that resolves to any other commit
is treated as unvouched-for, exactly like an incomplete download; a cold
pull fetches the pinned commit explicitly rather than whatever ``main``
is at pull time.

Provenance: the ``mflux-community`` HF account has published ~50 repos
as a systematic multi-family, multi-bitwidth conversion pipeline (krea-2,
z-image-turbo, flux2-klein-9b, each at q3-q8/bf16) — a pattern, not a
one-off upload — though it carries no README/license file and is not a
Hugging Face-verified org; this is a real supply-chain tradeoff against
filipstrand (the mflux maintainer personally) that a future maintainer
should weigh if a more authoritative source appears.

Guarded going forward, not just for this one repo:
``ImageGenerationEngine._verify_text_encoder_not_quantized`` is a
structural preflight that reads ONLY the text encoder's safetensors
header (no tensor data, no network beyond what is already cached) and
refuses to build the model if ``embed_tokens`` is packed/quantized — so a
FUTURE re-pin to a different repo with this same defect fails loudly
before generation starts, not three prompts later.

These tests guard the things a data-only alias can still get wrong:

  1. It routes to the mflux image lane (``modality == "image-gen"``),
     not the default text lane.
  2. The pinned repo is a *pre-quantized* mflux checkpoint, so the
     loader takes the ``model_path`` branch instead of pulling the
     ~57 GB canonical ``Qwen/Qwen-Image`` weights and quantizing on
     load (which would also error against an already-quantized repo).
  3. The pinned repo is specifically NOT one with a quantized text
     encoder — see above. A future re-pin (e.g. chasing a smaller
     download) must not silently reintroduce the corruption, whether by
     shape (structurally guarded now) or by name (guarded by the
     regression pin below).
"""

from __future__ import annotations

import json
import struct

import pytest

from rapid_mlx.image.engine import (
    ImageGenerationEngine,
    ImageRuntimeError,
    _detect_family,
    _looks_like_prequantized,
)
from rapid_mlx.model_aliases import resolve_profile
from rapid_mlx.runtime.resident_models import estimate_model_bytes

_GIB = 1024**3

# The specific repo this alias must NOT point at: its text encoder is
# quantized in a way the pinned mflux version cannot load correctly (see
# module docstring). Regression pin for "someone re-pins to a smaller
# download and silently reintroduces broken generation".
_KNOWN_INCOMPATIBLE_REPO = "filipstrand/Qwen-Image-mflux-6bit"


def test_qwen_image_alias_routes_to_image_lane() -> None:
    profile = resolve_profile("qwen-image")
    assert profile is not None
    assert profile.modality == "image-gen"
    assert profile.hf_path == "mflux-community/qwen-image-mflux-q6"


def test_qwen_image_alias_does_not_point_at_the_known_broken_repo() -> None:
    profile = resolve_profile("qwen-image")
    assert profile.hf_path != _KNOWN_INCOMPATIBLE_REPO


def test_qwen_image_alias_declares_a_memory_floor() -> None:
    # Full-precision text encoder + 6-bit transformer measured ~55.7 GiB
    # peak RSS during a real generation (`/usr/bin/time -l`) at 1024x1024 —
    # the API/GUI default resolution, not the smaller size this was first
    # measured at. The floor gates it off Macs that cannot hold that
    # resident.
    profile = resolve_profile("qwen-image")
    assert getattr(profile, "min_memory_gb", None) == 64


def test_qwen_image_repo_is_detected_as_qwen_image_family() -> None:
    hf_path = resolve_profile("qwen-image").hf_path
    assert _detect_family(hf_path) == "qwen-image"


def test_qwen_image_repo_is_prequantized_so_no_full_weight_pull() -> None:
    hf_path = resolve_profile("qwen-image").hf_path
    assert _looks_like_prequantized(hf_path) is True


def test_qwen_image_resident_estimate_is_not_the_bare_default() -> None:
    # The alias carries no digit params, so without a known-image entry the
    # fallback would charge the 4 GB default and mis-admit a ~29 GB model.
    # Both the alias and the pinned repo id must resolve to the real charge.
    for name in ("qwen-image", resolve_profile("qwen-image").hf_path):
        assert estimate_model_bytes(name) == int(55.7 * _GIB)


# MARK: - Revision pinning for a cold (not-yet-cached) pull
#
# ``mflux_local_snapshot`` (exercised in test_download_gate.py) already
# refuses to vouch for a cached snapshot that doesn't match the registered
# pin. These tests cover the other half: when there is nothing cached to
# vouch for at all, ``_model_path_for_mflux`` must fetch the pinned commit
# itself rather than let mflux resolve and download whatever ``main``
# currently is.


def test_model_path_for_mflux_forces_pinned_revision_on_cold_pull(
    monkeypatch,
) -> None:
    from rapid_mlx._download_gate import IMAGE_MODEL_REVISIONS

    repo = "mflux-community/qwen-image-mflux-q6"
    pinned_sha = IMAGE_MODEL_REVISIONS[repo]
    engine = ImageGenerationEngine(repo)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.mflux_local_snapshot", lambda _repo: None
    )
    calls = []

    def _fake_snapshot_download(repo_id, revision=None):
        calls.append((repo_id, revision))
        return "/fake/snapshot/dir"

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot_download)

    assert engine._model_path_for_mflux() == "/fake/snapshot/dir"
    assert calls == [(repo, pinned_sha)]


def _seed_pinned_download(cache_root, repo_id: str, sha: str, *, omit_shard=False):
    """A cache layout shaped like what ``snapshot_download(repo,
    revision=<commit sha>)`` actually leaves behind for a pinned repo:
    ``snapshots/<sha>`` populated, deliberately with NO ``refs/main`` (an
    explicit-commit download doesn't move it — see
    ``test_mflux_local_snapshot_accepts_pinned_revision_without_refs_main``
    in test_download_gate.py) — so ``_verify_weights_complete()``, called
    right after the fake download, gets a real verdict from the same code
    path a genuine cold pull would.
    """
    repo_root = cache_root / f"models--{repo_id.replace('/', '--')}"
    snap = repo_root / "snapshots" / sha
    (snap / "tokenizer").mkdir(parents=True)
    (snap / "tokenizer" / "tokenizer.json").write_text("{}")
    for component in ("transformer", "vae"):
        component_dir = snap / component
        component_dir.mkdir()
        (component_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "0.safetensors"}})
        )
        (component_dir / "0.safetensors").write_bytes(b"weights")
    text_encoder_dir = snap / "text_encoder"
    text_encoder_dir.mkdir()
    (text_encoder_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"encoder.embed_tokens.weight": "0.safetensors"}})
    )
    if not omit_shard:
        _write_safetensors_header(
            text_encoder_dir / "0.safetensors",
            {"encoder.embed_tokens.weight": {"shape": [152064, 3584], "dtype": "F16"}},
        )
    return str(snap)


def test_model_path_for_mflux_verifies_a_freshly_pinned_download(
    tmp_path, monkeypatch
) -> None:
    # Codex review (PR #2157): a cold pinned pull bypassed
    # `_verify_weights_complete()` entirely — it ran (as a no-op, nothing
    # was cached yet) *before* `_model_path_for_mflux()` did the actual
    # download, so the freshly downloaded snapshot was handed straight to
    # mflux unverified. This is the passing half: a complete, unquantized
    # download must resolve normally, verification included.
    from rapid_mlx._download_gate import IMAGE_MODEL_REVISIONS

    repo = "mflux-community/qwen-image-mflux-q6"
    pinned_sha = IMAGE_MODEL_REVISIONS[repo]
    cache_root = tmp_path / "hf-cache"
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo_id, revision=None: _seed_pinned_download(
            cache_root, repo_id, revision
        ),
    )

    engine = ImageGenerationEngine(repo)
    path = engine._model_path_for_mflux()
    assert path is not None
    assert path.endswith(pinned_sha)


def test_model_path_for_mflux_rejects_a_freshly_pinned_incomplete_download(
    tmp_path, monkeypatch
) -> None:
    # The failing half: a download that lands incomplete (interrupted,
    # disk full, whatever) must be refused before it reaches mflux, exactly
    # like an incomplete WARM cache already is via `_verify_weights_complete`.
    repo = "mflux-community/qwen-image-mflux-q6"
    cache_root = tmp_path / "hf-cache"
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo_id, revision=None: _seed_pinned_download(
            cache_root, repo_id, revision, omit_shard=True
        ),
    )

    engine = ImageGenerationEngine(repo)
    with pytest.raises(ImageRuntimeError, match="partially downloaded"):
        engine._model_path_for_mflux()


def test_model_path_for_mflux_falls_back_to_bare_repo_when_unpinned(
    monkeypatch,
) -> None:
    # A prequantized repo with no entry in IMAGE_MODEL_REVISIONS keeps
    # today's behavior: hand mflux the bare repo id and let it resolve +
    # download normally, rather than raising or refusing.
    engine = ImageGenerationEngine("filipstrand/Z-Image-Turbo-mflux-4bit")
    assert engine._prequantized is True
    monkeypatch.setattr(
        "rapid_mlx._download_gate.mflux_local_snapshot", lambda _repo: None
    )

    def _unexpected_download(*_args, **_kwargs):
        raise AssertionError("must not download an unpinned repo itself")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _unexpected_download)

    assert engine._model_path_for_mflux() == "filipstrand/Z-Image-Turbo-mflux-4bit"


# MARK: - Structural guard against a quantized text encoder
#
# The regression these tests actually need to pin is not "this one repo name
# is banned" (trivially defeated by a re-pin to a different broken repo) but
# "a Qwen-Image checkpoint whose text encoder is packed/quantized is refused
# BEFORE generation starts", using nothing but a synthetic safetensors header
# — no network, no multi-gigabyte download, no real mflux model. See
# ``ImageGenerationEngine._verify_text_encoder_not_quantized``.


def _write_safetensors_header(path, header: dict) -> None:
    """Write a syntactically valid ``.safetensors`` file carrying only
    ``header`` — enough for header-only readers; the tensor data section is
    empty because nothing under test reads past the header."""
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)


def _make_snapshot(
    tmp_path,
    embed_header_entry: dict | None,
    *,
    weight_map_extra: dict | None = None,
    write_shard: bool = True,
):
    """A synthetic ``text_encoder/`` layout: an index naming
    ``encoder.embed_tokens.weight``'s shard, plus (usually) that shard's
    header. ``weight_map_extra`` adds sibling entries to the INDEX (real
    quantization scale/bias keys land there, possibly in a different shard
    than the weight itself — see the check's docstring); ``write_shard=False``
    exercises the index naming a shard that never materializes.
    """
    snapshot = tmp_path / "snapshot"
    text_encoder = snapshot / "text_encoder"
    text_encoder.mkdir(parents=True)
    shard_name = "0.safetensors"
    weight_map = {"encoder.embed_tokens.weight": shard_name}
    if weight_map_extra:
        weight_map.update(weight_map_extra)
    (text_encoder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    if write_shard:
        header = {}
        if embed_header_entry is not None:
            header["encoder.embed_tokens.weight"] = embed_header_entry
        _write_safetensors_header(text_encoder / shard_name, header)
    return snapshot


def _engine_for_family_check(monkeypatch, snapshot) -> ImageGenerationEngine:
    # ``ImageGenerationEngine.__init__`` does real work (family detection,
    # prequantized sniffing) that only needs a plausible qwen-image repo
    # name — no engine method under test touches the network or mflux itself.
    engine = ImageGenerationEngine("mflux-community/qwen-image-mflux-q6")
    # ``_verify_text_encoder_not_quantized`` imports ``mflux_local_snapshot``
    # from ``_download_gate`` inside the method body, so patching the
    # source module's attribute is what the late-bound import picks up.
    monkeypatch.setattr(
        "rapid_mlx._download_gate.mflux_local_snapshot",
        lambda _repo: str(snapshot),
    )
    return engine


def test_full_precision_text_encoder_passes(tmp_path, monkeypatch) -> None:
    snapshot = _make_snapshot(tmp_path, {"shape": [152064, 3584], "dtype": "F16"})
    engine = _engine_for_family_check(monkeypatch, snapshot)
    engine._verify_text_encoder_not_quantized()  # must not raise


def test_packed_shape_is_refused(tmp_path, monkeypatch) -> None:
    # The exact failure mode reproduced against filipstrand's repo: a packed
    # last dimension instead of the real hidden size.
    snapshot = _make_snapshot(tmp_path, {"shape": [152064, 672], "dtype": "U32"})
    engine = _engine_for_family_check(monkeypatch, snapshot)
    with pytest.raises(ImageRuntimeError, match="quantized text encoder"):
        engine._verify_text_encoder_not_quantized()


def test_scales_sibling_is_refused_even_with_a_plausible_shape(
    tmp_path, monkeypatch
) -> None:
    # Belt-and-suspenders: a quantization scheme that happens to keep the
    # embedding's own shape at [vocab, hidden] is still caught by the
    # scales/biases siblings every quantized linear/embedding carries.
    #
    # Key convention verified against the real filipstrand shard this check
    # exists to catch: MLX names scales/biases as siblings of the MODULE
    # prefix (``encoder.embed_tokens.scales``), not nested under
    # ``.weight`` (NOT ``encoder.embed_tokens.weight.scales``, which
    # convention never produces).
    snapshot = _make_snapshot(
        tmp_path,
        {"shape": [152064, 3584], "dtype": "F16"},
        weight_map_extra={"encoder.embed_tokens.scales": "0.safetensors"},
    )
    engine = _engine_for_family_check(monkeypatch, snapshot)
    with pytest.raises(ImageRuntimeError, match="quantized text encoder"):
        engine._verify_text_encoder_not_quantized()


def test_biases_sibling_alone_is_also_refused(tmp_path, monkeypatch) -> None:
    # scales and biases are independent siblings in the real convention —
    # pin biases separately so a fix that only checks one doesn't regress.
    snapshot = _make_snapshot(
        tmp_path,
        {"shape": [152064, 3584], "dtype": "F16"},
        weight_map_extra={"encoder.embed_tokens.biases": "0.safetensors"},
    )
    engine = _engine_for_family_check(monkeypatch, snapshot)
    with pytest.raises(ImageRuntimeError, match="quantized text encoder"):
        engine._verify_text_encoder_not_quantized()


def test_scales_sibling_in_a_different_shard_is_still_refused(
    tmp_path, monkeypatch
) -> None:
    # The sibling check reads the INDEX's full weight_map, not one shard's
    # header — a quantized embedding whose scales/biases were split into a
    # separate shard from the weight itself must not slip past.
    snapshot = _make_snapshot(
        tmp_path,
        {"shape": [152064, 3584], "dtype": "F16"},
        weight_map_extra={"encoder.embed_tokens.scales": "1.safetensors"},
    )
    engine = _engine_for_family_check(monkeypatch, snapshot)
    with pytest.raises(ImageRuntimeError, match="quantized text encoder"):
        engine._verify_text_encoder_not_quantized()


def test_shard_named_by_index_but_missing_on_disk_fails_closed(
    tmp_path, monkeypatch
) -> None:
    # The index makes an explicit claim about where this tensor lives; if
    # that claim doesn't resolve to a real file, that is itself suspicious
    # and must refuse rather than silently skip the check.
    snapshot = _make_snapshot(tmp_path, None, write_shard=False)
    engine = _engine_for_family_check(monkeypatch, snapshot)
    with pytest.raises(ImageRuntimeError, match="quantized text encoder"):
        engine._verify_text_encoder_not_quantized()


def test_shard_present_but_key_absent_fails_closed(tmp_path, monkeypatch) -> None:
    # The index names a shard for embed_tokens.weight, but that shard's own
    # header doesn't actually contain it — inconsistent with what the index
    # promised, so this must not be read as "nothing to check".
    snapshot = _make_snapshot(tmp_path, embed_header_entry=None)
    engine = _engine_for_family_check(monkeypatch, snapshot)
    with pytest.raises(ImageRuntimeError, match="quantized text encoder"):
        engine._verify_text_encoder_not_quantized()


def test_oversized_header_length_is_rejected_without_reading_it(tmp_path) -> None:
    # A corrupt or hostile shard could claim a multi-gigabyte header to force
    # a multi-gigabyte read before JSON parsing ever runs. The length prefix
    # is checked against an absolute cap BEFORE the file is read further.
    shard = tmp_path / "0.safetensors"
    oversized = ImageGenerationEngine._SAFETENSORS_HEADER_MAX_BYTES + 1
    shard.write_bytes(struct.pack("<Q", oversized))
    assert ImageGenerationEngine._read_safetensors_header(str(shard)) is None


def test_header_length_past_the_8_byte_prefix_is_rejected(tmp_path) -> None:
    # header_len is measured from AFTER the 8-byte length prefix, so the
    # budget is file_size - 8, not the raw file size. A file truncated by up
    # to 8 bytes (only trailing JSON padding lost) must still be rejected
    # rather than appear to "fit".
    shard = tmp_path / "0.safetensors"
    payload = json.dumps({"encoder.embed_tokens.weight": {"shape": [1, 1]}}).encode()
    shard.write_bytes(struct.pack("<Q", len(payload)) + payload)
    shard.write_bytes(shard.read_bytes()[:-1])  # drop the last byte
    assert ImageGenerationEngine._read_safetensors_header(str(shard)) is None


def test_malformed_shapes_are_rejected_despite_a_matching_last_dimension(
    tmp_path, monkeypatch
) -> None:
    # A bare `shape[-1] == 3584` check would accept nonsense like a rank-1
    # shape, a zero dimension, or a non-integer entry as long as the last
    # element happened to equal 3584. A genuine embedding shape is rank-2
    # with two positive integers.
    for index, bad_shape in enumerate(([3584], [0, 3584], ["x", 3584], [True, 3584])):
        snapshot = _make_snapshot(
            tmp_path / str(index), {"shape": bad_shape, "dtype": "F16"}
        )
        engine = _engine_for_family_check(monkeypatch, snapshot)
        with pytest.raises(ImageRuntimeError, match="quantized text encoder"):
            engine._verify_text_encoder_not_quantized()


def test_shard_reached_via_symlink_is_accepted(tmp_path, monkeypatch) -> None:
    # A Hugging Face hub cache's snapshot files ARE symlinks into a sibling
    # blobs/ directory — the real shape this check runs against in
    # production. A resolve()-then-compare-parents guard walks straight out
    # of text_encoder_dir through that symlink and false-positives on every
    # real cached checkpoint (caught live against mflux-community's repo);
    # the string-only guard must accept this shape.
    snapshot = tmp_path / "snapshot"
    text_encoder = snapshot / "text_encoder"
    text_encoder.mkdir(parents=True)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    real_file = blobs / "deadbeef"
    _write_safetensors_header(
        real_file,
        {"encoder.embed_tokens.weight": {"shape": [152064, 3584], "dtype": "F16"}},
    )
    (text_encoder / "0.safetensors").symlink_to(real_file)
    (text_encoder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"encoder.embed_tokens.weight": "0.safetensors"}})
    )
    engine = _engine_for_family_check(monkeypatch, snapshot)
    engine._verify_text_encoder_not_quantized()  # must not raise


def test_shard_path_traversal_is_rejected(tmp_path, monkeypatch) -> None:
    # A shard name escaping text_encoder/ (path traversal, or an absolute
    # path elsewhere on disk) must not be trusted even though it would
    # resolve to a real, well-formed file. The escape target below is a
    # REAL, VALID shard — if the traversal guard were ever removed, this
    # would resolve and pass the shape check, so the test would go red
    # instead of staying accidentally green (the bug caught in review: an
    # earlier version put the escape target where "../outside.safetensors"
    # does NOT actually resolve, so the test passed only because the file
    # was missing, whether or not the guard did anything).
    snapshot = _make_snapshot(
        tmp_path, {"shape": [152064, 3584], "dtype": "F16"}
    )  # baseline valid shard, then override the index to point outside
    # "../outside.safetensors" from text_encoder/ resolves to snapshot/ —
    # one level out, not two — so the escape target has to live there.
    outside = snapshot / "outside.safetensors"
    _write_safetensors_header(
        outside,
        {"encoder.embed_tokens.weight": {"shape": [152064, 3584], "dtype": "F16"}},
    )
    (snapshot / "text_encoder" / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"encoder.embed_tokens.weight": "../outside.safetensors"}}
        )
    )
    engine = _engine_for_family_check(monkeypatch, snapshot)
    with pytest.raises(ImageRuntimeError, match="quantized text encoder"):
        engine._verify_text_encoder_not_quantized()


def test_non_qwen_family_is_not_checked(tmp_path, monkeypatch) -> None:
    # The 3584 hidden-size expectation is Qwen2-specific; z-image/flux2-klein
    # text encoders have entirely different architectures and must not be
    # held to it.
    engine = ImageGenerationEngine("filipstrand/Z-Image-Turbo-mflux-4bit")
    assert engine.family != "qwen-image"
    engine._verify_text_encoder_not_quantized()  # no-op, must not raise


def test_missing_index_does_not_raise(tmp_path, monkeypatch) -> None:
    # No index (or an unreadable one) is outside what this check can vouch
    # for — ``_verify_weights_complete``'s completeness check is what must
    # catch that; this guard degrades to a no-op rather than a false claim.
    empty_snapshot = tmp_path / "empty"
    empty_snapshot.mkdir()
    engine = _engine_for_family_check(monkeypatch, empty_snapshot)
    engine._verify_text_encoder_not_quantized()  # no-op, must not raise


def test_index_present_but_key_wholly_absent_fails_closed(
    tmp_path, monkeypatch
) -> None:
    # A readable, non-empty index that simply never mentions
    # `encoder.embed_tokens.weight` (as opposed to `_make_snapshot`'s shard
    # naming the key but the shard's own header omitting it, covered by
    # `test_shard_present_but_key_absent_fails_closed`) must not be read as
    # "nothing to check" — codex review: silently passing here let a
    # differently packaged or renamed quantized encoder bypass the guard.
    snapshot = tmp_path / "snapshot"
    text_encoder = snapshot / "text_encoder"
    text_encoder.mkdir(parents=True)
    (text_encoder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"some.other.tensor": "0.safetensors"}})
    )
    engine = _engine_for_family_check(monkeypatch, snapshot)
    with pytest.raises(ImageRuntimeError, match="does not expose the expected"):
        engine._verify_text_encoder_not_quantized()


def test_no_verdict_completeness_skips_the_text_encoder_check(
    tmp_path, monkeypatch
) -> None:
    # ``mflux_missing_weights`` returning ``None`` means "no verdict" — not
    # a registered alias, or nothing cached — and ``not missing`` is True
    # for BOTH ``[]`` and ``None``, so an earlier version of
    # ``_verify_weights_complete`` ran the structural text-encoder check on
    # a `None` verdict too. That is wrong even when the check WOULD have
    # found something to object to: no completeness verdict means no
    # standing to inspect internals yet. A quantized-shaped text encoder
    # sitting behind a ``None`` verdict must NOT raise.
    snapshot = _make_snapshot(tmp_path, {"shape": [152064, 672], "dtype": "U32"})
    engine = ImageGenerationEngine("mflux-community/qwen-image-mflux-q6")
    monkeypatch.setattr(
        "rapid_mlx._download_gate.mflux_missing_weights", lambda _repo: None
    )
    monkeypatch.setattr(
        "rapid_mlx._download_gate.mflux_local_snapshot", lambda _repo: str(snapshot)
    )
    engine._verify_weights_complete()  # no-op, must not raise
