from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import benchmark_qwen38_mllm_eager_dispatch as bench


def _exact_fixture(tmp_path: Path):
    repo_id = "acme/qwen"
    revision = "a" * 40
    repo = tmp_path / "models--acme--qwen"
    snapshot = repo / "snapshots" / revision
    blobs = repo / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    config = {
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "full_attention_interval": 4,
            "layer_types": [
                "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
                for index in range(64)
            ],
        },
        "quantization": {"bits": 4, "group_size": 64},
    }
    shard_names = ("01.safetensors", "02.safetensors", "03.safetensors")
    blob_ids = ("1" * 64, "2" * 64, "3" * 64)
    index = {"weight_map": {f"weight.{i}": name for i, name in enumerate(shard_names)}}
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    for name, blob_id in zip(shard_names, blob_ids):
        (blobs / blob_id).write_bytes(b"weights")
        (snapshot / name).symlink_to(Path("../../blobs") / blob_id)
    identity = {
        "repo_id": repo_id,
        "revision": revision,
        "subfolder": None,
        "config_sha256": bench._canonical_json_sha(config),
        "index_sha256": bench._canonical_json_sha(index),
        "shards": list(shard_names),
        "file_identities": [
            (name, f"hf_blob:{blob_id}") for name, blob_id in zip(shard_names, blob_ids)
        ],
    }
    expected = bench.ExactArtifact(
        repo_id=repo_id,
        revision=revision,
        verification_id="hf-snapshot-sha256:" + bench._canonical_json_sha(identity),
        config_sha256=identity["config_sha256"],
        index_sha256=identity["index_sha256"],
        shard_blobs=tuple(zip(shard_names, blob_ids)),
    )
    return snapshot, expected


def test_exact_artifact_accepts_canonical_verified_snapshot(tmp_path):
    snapshot, expected = _exact_fixture(tmp_path)
    result = bench._inspect_exact_artifact(snapshot, expected)
    assert result["verified"] is True
    assert result["verification_id"] == expected.verification_id
    assert "local_snapshot" not in result


@pytest.mark.parametrize("mutation", ["repo", "revision", "config", "regular_shard"])
def test_exact_artifact_rejects_identity_drift(tmp_path, mutation):
    snapshot, expected = _exact_fixture(tmp_path)
    if mutation == "repo":
        expected = bench.ExactArtifact(
            "wrong/repo",
            expected.revision,
            expected.verification_id,
            expected.config_sha256,
            expected.index_sha256,
            expected.shard_blobs,
        )
    elif mutation == "revision":
        expected = bench.ExactArtifact(
            expected.repo_id,
            "b" * 40,
            expected.verification_id,
            expected.config_sha256,
            expected.index_sha256,
            expected.shard_blobs,
        )
    elif mutation == "config":
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
    else:
        shard = snapshot / expected.shard_blobs[0][0]
        shard.unlink()
        shard.write_bytes(b"not canonical")
    with pytest.raises(ValueError):
        bench._inspect_exact_artifact(snapshot, expected)


def _fake_loaded(
    *,
    layers=64,
    linear=48,
    hidden=5120,
    interval=4,
    model_type="qwen3_5_text",
    head_dim=256,
    linear_conv_kernel_dim=4,
):
    decoder = type(bench.EXPECTED_DECODER_CLASS, (), {})
    decoder.__module__ = bench.EXPECTED_DECODER_MODULE

    def call(
        self,
        x,
        mask=None,
        cache=None,
        position_ids=None,
        position_embeddings=None,
    ):
        return x

    decoder.__call__ = call
    layer_objects = []
    for index in range(layers):
        layer = decoder()
        layer.is_linear = index < linear
        layer_objects.append(layer)
    model = SimpleNamespace(
        layers=layer_objects,
        args=SimpleNamespace(
            hidden_size=hidden,
            num_hidden_layers=layers,
            full_attention_interval=interval,
            model_type=model_type,
            head_dim=head_dim,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
        ),
    )
    return model, decoder


def test_loaded_model_qualification_is_exact():
    model, decoder = _fake_loaded()
    result = bench._qualify_loaded_model(model, decoder)
    assert result["qualified"]
    assert result["linear_layers"] == 48


@pytest.mark.parametrize(
    "kwargs",
    [
        {"layers": 63},
        {"linear": 47},
        {"hidden": 4096},
        {"interval": 8},
        {"model_type": "qwen3_5"},
        {"head_dim": 128},
        {"linear_conv_kernel_dim": 3},
    ],
)
def test_loaded_model_qualification_rejects_wrong_geometry(kwargs):
    model, decoder = _fake_loaded(**kwargs)
    with pytest.raises(RuntimeError):
        bench._qualify_loaded_model(model, decoder)


def test_loaded_model_qualification_rejects_wrong_class_origin():
    model, decoder = _fake_loaded()
    decoder.__module__ = "other.module"
    with pytest.raises(RuntimeError, match="exact mlx-vlm"):
        bench._qualify_loaded_model(model, decoder)


def test_benchmark_runtime_decoder_abi_when_available():
    language = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    import inspect

    assert tuple(
        inspect.signature(language.Qwen3_5DecoderLayer.__call__).parameters
    ) == (
        "self",
        "x",
        "mask",
        "cache",
        "position_ids",
        "position_embeddings",
    )


class _Array:
    def __init__(self, shape):
        self.shape = shape


class _MX:
    def __init__(self):
        self.calls = []

    def async_eval(self, value):
        self.calls.append(value)


def test_patch_is_candidate_shape_and_instance_guarded_and_reversible():
    model, decoder = _fake_loaded()
    mx = _MX()
    original = decoder.__call__
    patch = bench.EagerLayerPatch(decoder, model.layers, mx)
    patch.install()
    assert decoder.__call__ is original
    exact = _Array((1, 1, 5120))
    assert model.layers[0](exact) is exact
    patch.set_candidate(True)
    assert decoder.__call__ is patch.wrapped
    model.layers[0](exact)
    model.layers[0](_Array((1, 2, 5120)))
    decoder()(exact)
    assert patch.hits == 1
    assert patch.layer_hits == [1] + [0] * 63
    assert mx.calls == [exact]
    patch.set_candidate(False)
    assert decoder.__call__ is original
    patch.close()
    assert decoder.__call__ is original


def test_patch_rejects_double_install():
    model, decoder = _fake_loaded()
    patch = bench.EagerLayerPatch(decoder, model.layers, _MX())
    patch.install()
    with pytest.raises(RuntimeError, match="already installed"):
        patch.install()
    patch.close()


@pytest.mark.parametrize(
    ("tokens", "completed", "seen", "expected"),
    [([3], 3, 2, [3]), ([1, 2, 3], 3, 2, [3]), ([], 2, 2, [])],
)
def test_stream_token_delta_accepts_delta_and_cumulative(
    tokens, completed, seen, expected
):
    output = SimpleNamespace(tokens=tokens, completion_tokens=completed)
    assert bench._delta_token_ids(output, seen) == expected


def test_stream_token_delta_fails_closed_when_ambiguous():
    output = SimpleNamespace(tokens=[1, 2], completion_tokens=5)
    with pytest.raises(RuntimeError, match="neither delta nor cumulative"):
        bench._delta_token_ids(output, 2)


def _media_sample(text="cat", token_hash="tokens", eager_hits=128):
    return {
        "text": text,
        "text_sha256": "text-hash",
        "token_sha256": token_hash,
        "singleton_batch_delta": 1,
        "eager_hits": eager_hits,
        "eager_layer_hits": [eager_hits // 64] * 64,
    }


def test_media_sequence_requires_exact_recovery_and_candidate_engagement():
    before = _media_sample()
    middle = _media_sample(text="answer")
    after = _media_sample()
    assert bench._media_sequence_pass(before, middle, after, "cat")
    after["token_sha256"] = "drift"
    assert not bench._media_sequence_pass(before, middle, after, "cat")
    after = _media_sample(eager_hits=0)
    assert not bench._media_sequence_pass(before, middle, after, "cat")


def _sample(tps, ttft=1.0, token_hash="same", active=100, peak=200, hits=0):
    return {
        "decode_tps": tps,
        "ttft_s": ttft,
        "completion_tokens": bench.EXPECTED_TOKENS,
        "token_sha256": token_hash,
        "singleton_batch_delta": 1,
        "eager_hits": hits,
        "eager_layer_hits": [hits // 64] * 64,
        "memory": {"active_bytes": active, "peak_bytes": peak},
    }


def _passing_receipt():
    return {
        "pairs": [
            {
                "baseline": [
                    _sample(100 + index / 10),
                    _sample(100 + index / 10),
                ],
                "candidate": [
                    _sample(
                        (100 + index / 10) * 1.04,
                        ttft=1.05,
                        active=100 + bench.MIB,
                        peak=200 + bench.MIB,
                        hits=64 * bench.EXPECTED_TOKENS,
                    ),
                    _sample(
                        (100 + index / 10) * 1.04,
                        ttft=1.05,
                        active=100 + bench.MIB,
                        peak=200 + bench.MIB,
                        hits=64 * bench.EXPECTED_TOKENS,
                    ),
                ],
            }
            for index in range(bench.EXPECTED_PAIRS)
        ],
        "source": {"dirty": False, "source_tree_match": True},
        "identity": {"same_model": True, "same_executor": True},
        "configuration": {
            "prefix_cache": False,
            "temperature": 0.0,
            "thinking": False,
            "measured_ignore_eos": True,
        },
        "artifact": {"verified": True},
        "media_recovery": {"pass": True},
        "errors": [],
    }


def test_all_gates_pass_on_qualified_receipt():
    result = bench.evaluate_gates(_passing_receipt())
    assert result["pass"] is True
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    ("gate", "mutate"),
    [
        ("six_complete_prompt_strata", lambda r: r["pairs"].pop()),
        (
            "median_decode_speedup_gte_1_03",
            lambda r: [
                sample.update(decode_tps=p["baseline"][0]["decode_tps"] * 1.02)
                for p in r["pairs"]
                for sample in p["candidate"]
            ],
        ),
        (
            "five_of_six_strata_positive",
            lambda r: [
                sample.update(decode_tps=90)
                for i in range(2)
                for sample in r["pairs"][i]["candidate"]
            ],
        ),
        (
            "exact_token_ids_all_runs_per_stratum",
            lambda r: r["pairs"][0]["candidate"][0].update(token_sha256="drift"),
        ),
        (
            "all_twenty_four_samples_are_256_tokens",
            lambda r: r["pairs"][0]["candidate"][0].update(completion_tokens=255),
        ),
        (
            "paired_ratio_cv_lte_0_05",
            lambda r: r["pairs"][0]["candidate"][0].update(decode_tps=200),
        ),
        (
            "median_ttft_ratio_lte_1_10",
            lambda r: [
                sample.update(ttft_s=1.11)
                for p in r["pairs"]
                for sample in p["candidate"]
            ],
        ),
        (
            "active_delta_lte_64_mib",
            lambda r: r["pairs"][0]["candidate"][0]["memory"].update(
                active_bytes=100 + 65 * bench.MIB
            ),
        ),
        (
            "isolated_peak_delta_lte_64_mib",
            lambda r: r["pairs"][0]["candidate"][0]["memory"].update(
                peak_bytes=200 + 65 * bench.MIB
            ),
        ),
        (
            "singleton_fastpath_engaged",
            lambda r: r["pairs"][0]["candidate"][0].update(singleton_batch_delta=0),
        ),
        (
            "exact_candidate_hits_per_layer_and_zero_baseline",
            lambda r: r["pairs"][0]["candidate"][0]["eager_layer_hits"].__setitem__(
                0, 255
            ),
        ),
        ("same_model_and_executor", lambda r: r["identity"].update(same_model=False)),
        (
            "prefix_cache_disabled",
            lambda r: r["configuration"].update(prefix_cache=True),
        ),
        (
            "measured_ignore_eos_enabled",
            lambda r: r["configuration"].update(measured_ignore_eos=False),
        ),
        ("artifact_exact_b0_verified", lambda r: r["artifact"].update(verified=False)),
        ("source_clean", lambda r: r["source"].update(dirty=True)),
        (
            "source_tree_match",
            lambda r: r["source"].update(source_tree_match=False),
        ),
        (
            "vlm_media_text_media_recovery",
            lambda r: r["media_recovery"].update(pass_=False),
        ),
        ("no_errors", lambda r: r["errors"].append("boom")),
    ],
)
def test_gate_fails_closed(gate, mutate):
    receipt = _passing_receipt()
    mutate(receipt)
    if gate == "vlm_media_text_media_recovery":
        receipt["media_recovery"]["pass"] = False
    result = bench.evaluate_gates(receipt)
    assert result["pass"] is False
    assert result["checks"][gate] is False


@pytest.mark.parametrize("image_expect", ["", "   ", "\n\t"])
def test_blank_image_expect_is_rejected(tmp_path, image_expect):
    model = tmp_path / "model"
    model.mkdir()
    image = tmp_path / "image.png"
    image.write_bytes(b"png")
    args = argparse.Namespace(
        model=model,
        image_path=image,
        image_expect=image_expect,
        max_memory_delta_mib=64,
    )
    with pytest.raises(SystemExit):
        bench._validate_args(bench.build_parser(), args)


def test_data_url_records_digest_not_path(tmp_path):
    image = tmp_path / "secret-name.png"
    image.write_bytes(b"image")
    url, identity = bench._data_url(image)
    assert url.startswith("data:image/png;base64,")
    assert identity["sha256"]
    assert "path" not in identity
