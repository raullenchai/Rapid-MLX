from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_mlx import cli
from vllm_mlx.models.deepseek_v41_native import artifacts


def _write_file(path: Path, size: int, byte: bytes = b"x") -> str:
    path.write_bytes(byte * size)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _product_serve_args(*extra: str):
    captured = []
    with (
        patch.object(
            sys,
            "argv",
            [
                "rapid-mlx",
                "serve",
                "deepseek-v41-flash-reap-2bit",
                *extra,
            ],
        ),
        patch.object(cli, "serve_command", side_effect=captured.append),
    ):
        cli.main()
    return captured[0]


def test_artifact_contract_import_does_not_load_the_model_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import vllm_mlx.models.deepseek_v41_native.artifacts; "
                "assert 'vllm_mlx.models.deepseek_v41_native.model' "
                "not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.requires_mlx
def test_package_lazy_exports_resolve_public_model_types():
    from vllm_mlx.models import deepseek_v41_native

    assert deepseek_v41_native.ModelArgs.__name__ == "ModelArgs"
    assert deepseek_v41_native.Model.__name__ == "Model"
    with pytest.raises(AttributeError):
        deepseek_v41_native.__getattr__("not_a_public_export")


def test_verify_mtp_snapshot_accepts_declared_regular_files(monkeypatch, tmp_path):
    declared = []
    for name in ("config.json", "model.safetensors.index.json", "model.safetensors"):
        digest = _write_file(tmp_path / name, 3)
        declared.append(artifacts.ArtifactFile(name, 3, digest))
    monkeypatch.setattr(artifacts, "MTP_FILES", tuple(declared))

    assert artifacts.verify_mtp_snapshot(tmp_path) == tmp_path


def test_verify_mtp_snapshot_rejects_changed_bytes(monkeypatch, tmp_path):
    path = tmp_path / "model.safetensors"
    _write_file(path, 3)
    monkeypatch.setattr(
        artifacts,
        "MTP_FILES",
        (artifacts.ArtifactFile(path.name, 3, "0" * 64),),
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        artifacts.verify_mtp_snapshot(tmp_path)


def test_verify_mtp_snapshot_rejects_missing_and_wrong_sized_files(
    monkeypatch, tmp_path
):
    expected = artifacts.ArtifactFile("model.safetensors", 3, "0" * 64)
    monkeypatch.setattr(artifacts, "MTP_FILES", (expected,))
    with pytest.raises(FileNotFoundError, match="missing pinned"):
        artifacts.verify_mtp_snapshot(tmp_path)

    (tmp_path / expected.name).write_bytes(b"xx")
    with pytest.raises(ValueError, match="size mismatch"):
        artifacts.verify_mtp_snapshot(tmp_path)


def test_verify_mtp_snapshot_trusts_same_repo_content_address(monkeypatch, tmp_path):
    repo = tmp_path / "models--owner--repo"
    snapshot = repo / "snapshots" / ("a" * 40)
    blobs = repo / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    content = b"weights"
    digest = hashlib.sha256(content).hexdigest()
    blob = blobs / digest
    blob.write_bytes(content)
    os.symlink(Path("../../blobs") / digest, snapshot / "model.safetensors")
    monkeypatch.setattr(
        artifacts,
        "MTP_FILES",
        (artifacts.ArtifactFile("model.safetensors", len(content), digest),),
    )
    monkeypatch.setattr(
        artifacts,
        "_sha256",
        lambda _path: (_ for _ in ()).throw(AssertionError("CAS blob was rehashed")),
    )

    assert artifacts.verify_mtp_snapshot(snapshot) == snapshot


def test_verify_mtp_snapshot_hashes_symlink_outside_repo_blobs(monkeypatch, tmp_path):
    repo = tmp_path / "models--owner--repo"
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside.safetensors"
    digest = _write_file(outside, 3)
    os.symlink(outside, snapshot / "model.safetensors")
    monkeypatch.setattr(
        artifacts,
        "MTP_FILES",
        (artifacts.ArtifactFile("model.safetensors", 3, digest),),
    )

    assert artifacts.verify_mtp_snapshot(snapshot) == snapshot


@pytest.mark.requires_mlx
def test_safe_shard_accepts_only_same_repo_hub_blob(tmp_path):
    from vllm_mlx.models.deepseek_v41_native.dspark import _safe_shard

    repo = tmp_path / "models--owner--repo"
    snapshot = repo / "snapshots" / ("a" * 40)
    blobs = repo / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    blob = blobs / ("b" * 64)
    blob.write_bytes(b"weights")
    os.symlink(Path("../../blobs") / blob.name, snapshot / "model.safetensors")

    assert _safe_shard(snapshot, "model.safetensors").resolve() == blob

    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"weights")
    os.symlink(outside, snapshot / "outside.safetensors")
    with pytest.raises(ValueError, match="leaves its Hub repository"):
        _safe_shard(snapshot, "outside.safetensors")


def test_hard_memory_floor_refuses_before_load(monkeypatch):
    profile = SimpleNamespace(min_memory_gb=224, enforce_min_memory=True)
    monkeypatch.setattr("vllm_mlx.model_aliases.resolve_profile", lambda _: profile)
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: SimpleNamespace(total=192 * 1024**3)
    )

    with pytest.raises(SystemExit) as exc:
        cli._check_alias_min_memory("deepseek-v41-flash-reap-2bit")
    assert exc.value.code == 1


def test_programmatic_server_memory_gate_is_fail_closed(monkeypatch):
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: SimpleNamespace(total=192 * 1024**3)
    )
    with pytest.raises(RuntimeError, match="requires at least 224"):
        artifacts.require_product_memory()


def test_programmatic_server_memory_gate_returns_detected_capacity(monkeypatch):
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: SimpleNamespace(total=256 * 1024**3)
    )
    assert artifacts.require_product_memory() == 256.0


def test_artifact_download_helpers_pin_revisions_and_subset(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *args, **kwargs: calls.append((args, kwargs)) or str(tmp_path),
    )
    monkeypatch.setattr(artifacts, "verify_mtp_snapshot", lambda path: Path(path))

    assert artifacts.download_mtp_snapshot() == tmp_path
    assert artifacts.download_target_snapshot() == tmp_path
    assert calls == [
        (
            (artifacts.MTP_REPO,),
            {
                "revision": artifacts.MTP_REVISION,
                "allow_patterns": list(artifacts.MTP_ALLOW_PATTERNS),
            },
        ),
        ((artifacts.TARGET_REPO,), {"revision": artifacts.TARGET_REVISION}),
    ]


def test_implicit_download_checks_hard_memory_first(monkeypatch):
    checked = []
    monkeypatch.setattr(
        cli, "_check_alias_min_memory", lambda name: checked.append(name)
    )
    monkeypatch.setattr(
        cli,
        "_cache_runnability",
        lambda _name: (_ for _ in ()).throw(AssertionError("cache probe ran first")),
    )

    with pytest.raises(AssertionError, match="cache probe ran first"):
        cli._ensure_model_downloaded(artifacts.TARGET_REPO)
    assert checked == [artifacts.TARGET_REPO]


def test_implicit_download_tolerates_profile_resolution_failure(monkeypatch):
    import vllm_mlx.model_aliases as aliases

    monkeypatch.setattr(cli.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(
        aliases,
        "resolve_profile",
        lambda _name: (_ for _ in ()).throw(RuntimeError("bad catalog")),
    )
    monkeypatch.setattr(
        cli,
        "_cache_runnability",
        lambda _name: (_ for _ in ()).throw(AssertionError("continued")),
    )
    with pytest.raises(AssertionError, match="continued"):
        cli._ensure_model_downloaded("owner/model")


def test_product_alias_pulls_pinned_target_and_mixed_sidecar(monkeypatch):
    calls = []
    disk_checks = []
    monkeypatch.setattr(
        cli, "_pull_repository", lambda args, **kw: calls.append((args.model, kw))
    )
    monkeypatch.setattr("vllm_mlx.audio.registry.runtime_assets_for", lambda _: ())
    monkeypatch.setattr(
        "vllm_mlx.audio.registry.runtime_requirements_for", lambda _: ()
    )
    monkeypatch.setattr(
        "vllm_mlx._download_gate.image_runtime_assets_for", lambda _: ()
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda *a, **k: "/snapshot"
    )
    monkeypatch.setattr(artifacts, "verify_mtp_snapshot", lambda path: Path(path))
    monkeypatch.setattr(cli, "_emit_pull_activation", lambda: None)
    monkeypatch.setattr(
        cli,
        "_check_disk_space",
        lambda model, **kwargs: disk_checks.append((model, kwargs)),
    )
    args = SimpleNamespace(
        model=artifacts.TARGET_REPO,
        _original_alias="deepseek-v41-flash-reap-2bit",
        bits=None,
        format=None,
    )

    cli.pull_command(args)

    assert artifacts.MTP_REPO == ("rapid-mlx/DeepSeek-V4.1-Flash-DSpark-4d2e-MLX")
    assert artifacts.MTP_REVISION == "9530d6d2bf59e0d05177bd538095d5704ded1488"
    assert [file.name for file in artifacts.MTP_FILES] == [
        "config.json",
        "dspark-mixed-stage-0.safetensors",
        "dspark-mixed-stage-1.safetensors",
        "dspark-mixed-stage-2.safetensors",
        "model.safetensors.index.json",
        "rapid-dspark-manifest.json",
    ]
    assert calls[0] == (
        artifacts.TARGET_REPO,
        {"revision_override": artifacts.TARGET_REVISION},
    )
    assert calls[1] == (
        artifacts.MTP_REPO,
        {
            "allow_patterns_override": list(artifacts.MTP_ALLOW_PATTERNS),
            "revision_override": artifacts.MTP_REVISION,
        },
    )
    assert disk_checks == [
        (
            artifacts.TARGET_REPO,
            {"revision_override": artifacts.TARGET_REVISION},
        ),
        (
            artifacts.MTP_REPO,
            {
                "revision_override": artifacts.MTP_REVISION,
                "allow_patterns": list(artifacts.MTP_ALLOW_PATTERNS),
            },
        ),
    ]


def test_product_alias_contract_is_ultra_only_and_greedy():
    from vllm_mlx.model_aliases import resolve_profile

    profile = resolve_profile("deepseek-v41-flash-reap-2bit")
    assert profile is not None
    assert profile.hf_path == artifacts.TARGET_REPO
    assert profile.min_memory_gb == 224
    assert profile.enforce_min_memory is True
    assert profile.default_max_tokens == 4096
    assert profile.experimental is True
    assert profile.tool_call_parser is None
    assert profile.is_text_only is True
    assert profile.mtp_default_enabled is False


def test_product_info_reports_dedicated_runtime_instead_of_generic_guess(capsys):
    cli.info_command(
        SimpleNamespace(
            model=artifacts.TARGET_REPO,
            _original_alias="deepseek-v41-flash-reap-2bit",
        )
    )
    output = capsys.readouterr().out
    assert "DeepSeek V4.1 MoE; native MLX runtime" in output
    assert "target-authoritative DSpark K4; greedy" in output
    assert "hard refusal below 224 GiB" in output
    assert "no MTP/drafter trained" not in output
    assert "DFlash eligibility" not in output


def test_product_profile_default_caps_plain_serve_but_preserves_explicit_value():
    from vllm_mlx.model_aliases import resolve_profile

    profile = resolve_profile("deepseek-v41-flash-reap-2bit")
    assert cli._resolve_v41_serve_max_tokens(None, profile) == (4096, False)
    assert cli._resolve_v41_serve_max_tokens(1024, profile) == (1024, True)
    assert cli._resolve_v41_serve_max_tokens(None, None) == (32768, False)


@pytest.mark.parametrize(
    "args",
    [
        SimpleNamespace(_speculative_config=object(), no_spec_decode=False),
        SimpleNamespace(_speculative_config=None, no_spec_decode=True),
    ],
)
def test_product_runtime_rejects_conflicting_spec_controls(args):
    with pytest.raises(SystemExit) as exc:
        cli._validate_v41_product_spec_flags(args, owns_runtime=True)
    assert exc.value.code == 2


class _Tokenizer:
    bos_token = "<bos>"
    eos_token = "<eos>"
    eos_token_id = 9
    clean_up_tokenization_spaces = False

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [10, 11] if text else []

    def decode(self, tokens):
        return "".join(
            "<eos>" if token == self.eos_token_id else chr(96 + token)
            for token in tokens
        )


@pytest.mark.requires_mlx
def test_prompt_renderer_pins_release_framing_and_limits():
    from vllm_mlx.models.deepseek_v41_native.serving import render_prompt

    request = SimpleNamespace(
        tools=None,
        messages=[
            SimpleNamespace(role="system", content="safe"),
            SimpleNamespace(role="user", content="hello"),
        ],
    )
    assert render_prompt(_Tokenizer(), None, request) == (
        "<bos>safe<｜User｜>hello<｜Assistant｜></think>"
    )
    assert render_prompt(_Tokenizer(), None, request, enable_thinking=True).endswith(
        "<｜Assistant｜>"
    )


@pytest.mark.requires_mlx
def test_generation_kwargs_rejects_unqualified_sampling_and_long_output():
    from fastapi import HTTPException

    from vllm_mlx.models.deepseek_v41_native.serving import generation_kwargs

    with pytest.raises(HTTPException, match="greedy"):
        generation_kwargs(max_tokens=16, temperature=0.1, top_p=1.0)
    with pytest.raises(HTTPException, match="top_p=1"):
        generation_kwargs(max_tokens=16, temperature=0.0, top_p=0.9)
    with pytest.raises(HTTPException, match="4096"):
        generation_kwargs(max_tokens=4097, temperature=0.0, top_p=1.0)
    with pytest.raises(HTTPException, match="1 <= max_tokens"):
        generation_kwargs(max_tokens=0, temperature=0.0, top_p=1.0)
    assert generation_kwargs(max_tokens=16, temperature=0.0, top_p=1.0) == {
        "max_tokens": 16,
        "temperature": 0.0,
        "top_p": 1.0,
    }


@pytest.mark.requires_mlx
def test_prompt_renderer_rejects_unqualified_protocol_shapes():
    from fastapi import HTTPException

    from vllm_mlx.models.deepseek_v41_native.serving import render_prompt

    with pytest.raises(HTTPException, match="tool calling"):
        render_prompt(
            _Tokenizer(),
            None,
            SimpleNamespace(tools=[object()], messages=[]),
        )
    with pytest.raises(HTTPException, match="text content only"):
        render_prompt(
            _Tokenizer(),
            None,
            SimpleNamespace(
                tools=None, messages=[SimpleNamespace(role="user", content=[])]
            ),
        )
    with pytest.raises(HTTPException, match="unsupported message role"):
        render_prompt(
            _Tokenizer(),
            None,
            SimpleNamespace(
                tools=None, messages=[SimpleNamespace(role="tool", content="x")]
            ),
        )

    class LongTokenizer(_Tokenizer):
        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return [1] * 8193

    with pytest.raises(HTTPException, match="8192"):
        render_prompt(
            LongTokenizer(),
            None,
            SimpleNamespace(
                tools=None, messages=[SimpleNamespace(role="assistant", content="x")]
            ),
        )


@pytest.mark.requires_mlx
def test_product_request_rejects_sampling_fields_it_cannot_honor():
    from fastapi import HTTPException

    from vllm_mlx.models.deepseek_v41_native.serving import validate_request

    request = SimpleNamespace(
        stop=["done"],
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        presence_penalty=None,
        frequency_penalty=None,
        top_logprobs=None,
        logit_bias=None,
        video_fps=None,
        video_max_frames=None,
        reasoning_max_tokens=None,
        reasoning_effort=None,
        seed=7,
        chat_template_kwargs={"custom": True},
    )
    with pytest.raises(HTTPException, match="stop, seed, chat_template_kwargs"):
        validate_request(request)


@pytest.mark.requires_mlx
def test_k4_stream_commits_target_authoritative_batch():
    import mlx.core as mx

    from vllm_mlx.models.deepseek_v41_native.serving import (
        DSparkRuntime,
        stream_generate,
    )

    class Cache:
        offset = 0

        def rollback(self, offset):
            self.offset = offset

    class Model:
        def make_cache(self, max_seq_len):
            assert max_seq_len == 15
            return Cache()

        def __call__(self, ids, cache, **kwargs):
            width = int(ids.shape[1])
            cache.offset += width
            logits = mx.full((1, width, 10), -1.0)
            if kwargs.get("enable_rollback"):
                for row, token in enumerate((3, 4, 5, 6, 7)[:width]):
                    logits[:, row, token] = 1.0
            else:
                logits[:, -1, 2] = 1.0
            hidden = mx.zeros((1, width, 1))
            return logits, hidden

    class Draft:
        layers = [0]
        position = -1
        windows = {0: []}

        def observe(self, _hidden, position):
            self.position = position

        def propose(self, seed):
            assert seed == 2
            return [2, 3, 4, 5, 6], mx.ones((5,))

    runtime = DSparkRuntime(Draft(), "mtp", "t" * 40, "m" * 40)
    chunks = list(
        stream_generate(Model(), _Tokenizer(), "prompt", runtime=runtime, max_tokens=5)
    )
    assert [chunk.token for chunk in chunks] == [2, 3, 4, 5, 6]
    assert chunks[-1].generation_tokens == 5


@pytest.mark.requires_mlx
def test_stream_does_not_emit_literal_eos_marker():
    import mlx.core as mx

    from vllm_mlx.models.deepseek_v41_native.serving import (
        DSparkRuntime,
        stream_generate,
    )

    class Cache:
        offset = 0

    class Model:
        def make_cache(self, max_seq_len):
            del max_seq_len
            return Cache()

        def __call__(self, ids, cache, **kwargs):
            cache.offset += int(ids.shape[1])
            logits = mx.full((1, int(ids.shape[1]), 10), -1.0)
            logits[:, -1, 9] = 1.0
            return logits, mx.zeros((1, int(ids.shape[1]), 1))

    class Draft:
        layers = [0]
        position = -1
        windows = {0: []}

        def observe(self, _hidden, position):
            self.position = position

    runtime = DSparkRuntime(Draft(), "mtp", "t" * 40, "m" * 40)
    chunks = list(
        stream_generate(Model(), _Tokenizer(), "prompt", runtime=runtime, max_tokens=2)
    )
    assert [chunk.token for chunk in chunks] == [9]
    assert "<eos>" not in "".join(chunk.text for chunk in chunks)


@pytest.mark.requires_mlx
def test_target_qmv_install_and_small_route_execution():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU

    from vllm_mlx.models.deepseek_v41_native.serving import (
        ExactDirectDownSwitchGLU,
        install_target_qmv,
    )

    experts = SwitchGLU(64, 64, 8, bias=False)
    nn.quantize(experts, group_size=64, bits=2)
    experts.down_proj.scales = experts.down_proj.scales.astype(mx.bfloat16)
    experts.down_proj.biases = experts.down_proj.biases.astype(mx.bfloat16)
    model = SimpleNamespace(
        layers=[SimpleNamespace(ffn=SimpleNamespace(experts=experts))]
    )
    assert install_target_qmv(model) == 1
    assert isinstance(experts, ExactDirectDownSwitchGLU)

    small = experts(mx.zeros((1, 64), mx.float32), mx.zeros((1, 1), mx.uint32))
    large = experts(mx.zeros((37, 64), mx.float32), mx.zeros((37, 1), mx.uint32))
    mx.eval(small, large)
    assert small.shape == (1, 1, 64)
    assert large.shape == (37, 1, 64)

    bad = SimpleNamespace(
        layers=[SimpleNamespace(ffn=SimpleNamespace(experts=object()))]
    )
    with pytest.raises(TypeError, match="unsupported expert module"):
        install_target_qmv(bad)


@pytest.mark.requires_mlx
def test_vectorized_mtp_attention_executes_installed_path(monkeypatch):
    import mlx.core as mx

    from vllm_mlx.models.deepseek_v41_native import serving

    config = {
        "qk_rope_head_dim": 2,
        "rope_theta": 10_000,
        "num_attention_heads": 2,
        "head_dim": 4,
        "o_groups": 2,
    }

    class Weights:
        entries = {"mtp.0.attn.wo_a.weight": {"shape": (4, 1)}}

        def linear(self, base, value):
            count = int(value.shape[0])
            if base.endswith("wq_b"):
                return mx.zeros((count, 8), mx.float32)
            if base.endswith("wkv"):
                return mx.zeros((count, 4), mx.float32)
            if base.endswith("wo_b"):
                return mx.zeros((count, 4), mx.float32)
            return value

        def read(self, key):
            if key.endswith("attn_sink"):
                return mx.zeros((2,), mx.float32)
            return mx.zeros((4, 1), mx.uint32)

        def quant_config(self, _base):
            return {"group_size": 32, "bits": 2}

    class Adapter:
        w = Weights()

        @staticmethod
        def norm(_base, value):
            return value

    draft = SimpleNamespace(
        adapter=Adapter(),
        c=config,
        position=0,
        windows={0: [mx.zeros((1, 4), mx.float32)]},
    )
    monkeypatch.setattr(serving, "quantize_cache", lambda value, *_args: value)
    monkeypatch.setattr(serving, "draft_attention", lambda query, _keys, _sink: query)
    monkeypatch.setattr(
        mx,
        "quantized_matmul",
        lambda value, *_args, **_kwargs: mx.zeros(
            (int(value.shape[0]), int(value.shape[1]), 2), mx.float32
        ),
    )

    serving._install_vectorized_mtp_attention(draft)
    result = draft.attention("mtp.0.attn", mx.zeros((1, 4), mx.float32), 0)
    mx.eval(result)
    assert result.shape == (1, 4)


@pytest.mark.requires_mlx
def test_packed_mtp_moe_releases_sources_and_executes(monkeypatch):
    import mlx.core as mx

    from vllm_mlx.models.deepseek_v41_native import serving

    base = "mtp.0.ffn"

    class Weights:
        def __init__(self):
            self.entries = {}
            self.values = {
                base + ".gate.weight": mx.zeros((2, 4), mx.float32),
                base + ".gate.bias": mx.zeros((2,), mx.float32),
            }
            self.released = []
            for expert in range(2):
                for projection in ("w1", "w3", "w2"):
                    for suffix, value in (
                        ("weight", mx.zeros((4, 1), mx.uint32)),
                        ("scales", mx.ones((4, 1), mx.float32)),
                        ("biases", mx.zeros((4, 1), mx.float32)),
                    ):
                        key = f"{base}.experts.{expert}.{projection}.{suffix}"
                        self.entries[key] = {"shape": value.shape}
                        self.values[key] = value

        def read(self, key):
            return self.values[key]

        @staticmethod
        def quant_config(_base):
            return {"group_size": 32, "bits": 2}

        def release_tensor(self, key):
            self.released.append(key)

    class Adapter:
        def __init__(self, weights, config):
            self.w = weights
            self.c = config

        @staticmethod
        def expert(_base, value):
            return mx.zeros_like(value)

    weights = Weights()
    adapter_config = {
        "dspark_num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "swiglu_limit": 1.0,
    }
    draft = SimpleNamespace(
        w=weights,
        c={"dspark_n_routed_experts": 2, **adapter_config},
        adapter=Adapter(weights, adapter_config),
    )

    def fake_gather(value, weight, _scales, _biases, *, rhs_indices, **_kwargs):
        return mx.zeros(
            (
                int(value.shape[0]),
                int(rhs_indices.shape[-1]),
                1,
                int(weight.shape[1]),
            ),
            value.dtype,
        )

    monkeypatch.setattr(mx, "gather_qmm", fake_gather)
    packed_bytes = serving._install_packed_mtp_moe(draft)
    result = draft.adapter.moe(base, mx.ones((1, 4), mx.float32))
    mx.eval(result)
    assert packed_bytes > 0
    assert len(weights.released) == 18
    assert result.shape == (1, 4)


@pytest.mark.requires_mlx
def test_load_product_runtime_composes_owned_components(monkeypatch):
    from vllm_mlx.models.deepseek_v41_native import serving

    model = SimpleNamespace(
        layers=[object()], eval_interval=0, embed=object(), head=object()
    )
    weights = SimpleNamespace(
        config={"dspark_block_size": 5},
        attach_target=lambda _model: None,
        pin_mtp=lambda: None,
    )
    draft = SimpleNamespace()
    calls = []
    monkeypatch.setattr(serving, "load", lambda *_a, **_k: (model, object()))
    monkeypatch.setattr(serving, "install_target_qmv", lambda _model: 1)
    monkeypatch.setattr(serving, "DSparkWeights", lambda _path: weights)
    monkeypatch.setattr(serving, "DSpark", lambda _target, pin_weights: draft)
    monkeypatch.setattr(
        serving, "_install_vectorized_mtp_attention", lambda value: calls.append(value)
    )
    monkeypatch.setattr(
        serving, "_install_packed_mtp_moe", lambda value: calls.append(value)
    )
    monkeypatch.setattr(
        serving.PreTrainedTokenizerFast,
        "from_pretrained",
        lambda *_a, **_k: _Tokenizer(),
    )

    loaded, tokenizer, runtime = serving.load_product_runtime(
        "/target",
        "/mtp",
        target_revision="target-rev",
        mtp_revision="mtp-rev",
        mtp_identity="owner/head",
    )
    assert loaded is model
    assert isinstance(tokenizer, _Tokenizer)
    assert runtime.drafter is draft
    assert runtime.drafter_repo == "owner/head"
    assert model.eval_interval == 40
    assert calls == [draft, draft]


@pytest.mark.requires_mlx
def test_load_product_runtime_rejects_partial_target_qmv_install(monkeypatch):
    from vllm_mlx.models.deepseek_v41_native import serving

    model = SimpleNamespace(layers=[object()], eval_interval=0)
    monkeypatch.setattr(serving, "load", lambda *_a, **_k: (model, object()))
    monkeypatch.setattr(serving, "install_target_qmv", lambda _model: 0)
    with pytest.raises(RuntimeError, match="every layer"):
        serving.load_product_runtime(
            "/target",
            "/mtp",
            target_revision="target-rev",
            mtp_revision="mtp-rev",
        )


@pytest.mark.requires_mlx
def test_match_and_generate_cover_rejection_eos_and_empty_paths(monkeypatch):
    import mlx.core as mx

    from vllm_mlx.models.deepseek_v41_native import serving

    logits = mx.full((1, 2, 10), -1.0)
    logits[:, 0, 4] = 1
    logits[:, 1, 9] = 1
    assert serving._match([2, 3], logits[:, :1], 9, False) == (
        [2, 4],
        1,
        False,
        0,
    )
    assert serving._match([2, 4, 9], logits, 9, True) == (
        [4, 9],
        None,
        True,
        2,
    )
    monkeypatch.setattr(serving, "stream_generate", lambda *_a, **_k: iter(()))
    result = serving.generate(
        object(), _Tokenizer(), "prompt", runtime=object(), max_tokens=1
    )
    assert result.text == ""
    assert result.prompt_tokens == 2
    assert result.generation_tokens == 0


@pytest.mark.requires_mlx
def test_stream_rejects_short_prompt_and_rolls_back_rejected_suffix():
    import mlx.core as mx

    from vllm_mlx.models.deepseek_v41_native.serving import (
        DSparkRuntime,
        stream_generate,
    )

    class Processor(_Tokenizer):
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [10] if text == "short" else [10, 11]

    class Cache:
        offset = 0

        def __init__(self):
            self.rollbacks = []

        def rollback(self, offset):
            self.rollbacks.append(offset)
            self.offset = offset

    cache = Cache()

    class Model:
        def make_cache(self, max_seq_len):
            del max_seq_len
            return cache

        def __call__(self, ids, active_cache, **kwargs):
            width = int(ids.shape[1])
            active_cache.offset += width
            logits = mx.full((1, width, 10), -1.0)
            logits[:, -1, 2] = 1.0
            if kwargs.get("enable_rollback"):
                logits[:, 0, 4] = 2.0
            return logits, mx.zeros((1, width, 1))

    class Draft:
        layers = [0]
        position = -1
        windows = {0: []}

        def observe(self, _hidden, position):
            self.position = position

        @staticmethod
        def propose(seed):
            return [seed, 3, 5, 6, 7], mx.ones((5,))

    runtime = DSparkRuntime(Draft(), "mtp", "t", "m")
    with pytest.raises(ValueError, match="at least two"):
        list(
            stream_generate(
                Model(), Processor(), "short", runtime=runtime, max_tokens=2
            )
        )

    chunks = list(
        stream_generate(Model(), Processor(), "prompt", runtime=runtime, max_tokens=2)
    )
    assert [chunk.token for chunk in chunks] == [2, 4]
    assert cache.rollbacks == [3]


@pytest.mark.requires_mlx
def test_product_server_reuses_guarded_serial_boundary(monkeypatch):
    from vllm_mlx.models.deepseek_v41_native import server
    from vllm_mlx.models.deepseek_v41_native.serving import (
        generation_kwargs,
        validate_request,
    )
    from vllm_mlx.speculative.dflash import server as serial_server

    calls = {}

    class ImmediateExecutor:
        def submit(self, fn):
            return SimpleNamespace(result=lambda: fn())

    runtime = SimpleNamespace(
        algorithm="dspark-k4",
        drafter_repo=artifacts.MTP_REPO,
        target_revision=artifacts.TARGET_REVISION,
        drafter_revision=artifacts.MTP_REVISION,
    )
    monkeypatch.setattr(server, "download_target_snapshot", lambda: Path("/target"))
    monkeypatch.setattr(server, "download_mtp_snapshot", lambda: Path("/mtp"))
    monkeypatch.setattr(server, "require_product_memory", lambda: 256.0)

    def _load_product_runtime(*args, **kwargs):
        calls["load"] = (args, kwargs)
        return object(), _Tokenizer(), runtime

    monkeypatch.setattr(server, "load_product_runtime", _load_product_runtime)
    monkeypatch.setattr(serial_server, "_dflash_executor", ImmediateExecutor())
    monkeypatch.setattr(
        serial_server,
        "_build_app",
        lambda **kwargs: calls.setdefault("app", kwargs) or object(),
    )
    monkeypatch.setattr(
        "uvicorn.run", lambda app, **kwargs: calls.update(uvicorn=(app, kwargs))
    )

    server.run_server(
        host="127.0.0.1",
        port=8000,
        served_model_name="deepseek-v41-flash-reap-2bit",
        default_max_tokens=4096,
        cors_origins=[],
        uvicorn_log_level="warning",
    )

    assert calls["app"]["backend_name"] == "DeepSeek V4.1 DSpark K4"
    assert calls["app"]["tool_call_parser"] is None
    assert calls["app"]["generation_kwargs_fn"] is generation_kwargs
    assert calls["app"]["validate_request_fn"] is validate_request
    assert calls["load"][1]["mtp_identity"] == artifacts.MTP_REPO
    assert calls["uvicorn"][1]["port"] == 8000


@pytest.mark.requires_mlx
def test_shared_serial_app_invokes_product_callbacks_and_generators():
    from fastapi.testclient import TestClient

    from vllm_mlx.models.deepseek_v41_native.serving import (
        GenerationChunk,
        GenerationResult,
    )
    from vllm_mlx.speculative.dflash import server as serial_server

    runtime = SimpleNamespace(
        algorithm="dspark-k4",
        drafter_repo="owner/head",
        target_revision="target-rev",
        drafter_revision="head-rev",
        drafter=object(),
        kind="dspark",
    )
    validated = []
    generated_kwargs = []

    def generation_kwargs_fn(**kwargs):
        generated_kwargs.append(kwargs)
        return kwargs

    def generate_fn(_model, _processor, _prompt, **kwargs):
        generated_kwargs.append(kwargs)
        return GenerationResult("ok", prompt_tokens=2, generation_tokens=1)

    def stream_generate_fn(_model, _processor, _prompt, **kwargs):
        generated_kwargs.append(kwargs)
        yield GenerationChunk("ok", token=2, prompt_tokens=2, generation_tokens=1)

    app = serial_server._build_app(
        model=object(),
        processor=_Tokenizer(),
        runtime=runtime,
        served_model_name="deepseek-v41-flash-reap-2bit",
        default_max_tokens=4,
        cors_origins=[],
        default_timeout=0,
        render_prompt_fn=lambda *_a, **_k: "prompt",
        generate_fn=generate_fn,
        stream_generate_fn=stream_generate_fn,
        generation_kwargs_fn=generation_kwargs_fn,
        validate_request_fn=validated.append,
        backend_name="DeepSeek V4.1 DSpark K4",
    )
    payload = {
        "model": "deepseek-v41-flash-reap-2bit",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1,
        "temperature": 0,
        "top_p": 1,
    }
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)
        streamed = client.post("/v1/chat/completions", json={**payload, "stream": True})
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert streamed.status_code == 200
    assert "data: [DONE]" in streamed.text
    assert len(validated) == 2
    assert generated_kwargs[0] == {
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
    }


@pytest.mark.requires_mlx
def test_shared_serial_app_builds_legacy_draft_kwargs_when_no_callback():
    from fastapi.testclient import TestClient

    from vllm_mlx.models.deepseek_v41_native.serving import GenerationResult
    from vllm_mlx.speculative.dflash import server as serial_server

    captured = []
    runtime = SimpleNamespace(
        algorithm="dspark-k4",
        drafter_repo="owner/head",
        target_revision="target-rev",
        drafter_revision="head-rev",
        drafter=object(),
        kind="dspark",
    )

    def generate_fn(_model, _processor, _prompt, **kwargs):
        captured.append(kwargs)
        return GenerationResult("ok", prompt_tokens=2, generation_tokens=1)

    app = serial_server._build_app(
        model=object(),
        processor=_Tokenizer(),
        runtime=runtime,
        served_model_name="model",
        default_max_tokens=4,
        cors_origins=[],
        default_timeout=0,
        render_prompt_fn=lambda *_a, **_k: "prompt",
        generate_fn=generate_fn,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
                "temperature": 0,
                "top_p": 1,
            },
        )
    assert response.status_code == 200
    assert captured[0]["draft_model"] is runtime.drafter
    assert captured[0]["draft_kind"] == "dspark"


@pytest.mark.requires_mlx
def test_shared_stream_reports_deadline_when_client_stalls(monkeypatch):
    import asyncio

    from vllm_mlx.api.models import ChatCompletionRequest
    from vllm_mlx.models.deepseek_v41_native.serving import GenerationChunk
    from vllm_mlx.speculative.dflash import server as serial_server

    monkeypatch.setattr(serial_server, "_STREAM_QUEUE_MAXSIZE", 1)

    def stream_generate_fn(*_args, **_kwargs):
        for index in range(20):
            yield GenerationChunk(
                "x", token=index, prompt_tokens=2, generation_tokens=index + 1
            )

    request = ChatCompletionRequest(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        max_tokens=20,
    )

    async def collect_after_stall():
        stream = serial_server._stream_completion(
            prompt="prompt",
            request=request,
            served_model_name="model",
            gen_kwargs={"max_tokens": 20},
            model=object(),
            processor=_Tokenizer(),
            timeout=0.02,
            timeout_label=0.02,
            stream_generate_fn=stream_generate_fn,
            backend_name="DeepSeek V4.1 DSpark K4",
        )
        frames = [await anext(stream)]
        await asyncio.sleep(0.05)
        async for frame in stream:
            frames.append(frame)
        return b"".join(frames)

    output = asyncio.run(collect_after_stall())
    assert b"stream timed out after" in output
    assert b"data: [DONE]" in output


@pytest.mark.requires_mlx
def test_serve_command_runs_complete_product_owned_dispatch(monkeypatch, capsys):
    from vllm_mlx import _version_check
    from vllm_mlx import server as server_module
    from vllm_mlx.models.deepseek_v41_native import server as product_server

    args = _product_serve_args()
    calls = {"memory": [], "disk": [], "downloads": [], "run": []}
    monkeypatch.setattr(_version_check, "prompt_upgrade_if_available", lambda: False)
    monkeypatch.setattr(
        _version_check, "print_staleness_warning_if_any", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        cli, "_check_alias_min_memory", lambda name: calls["memory"].append(name)
    )
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *_a, **_k: None)
    monkeypatch.setattr(
        cli,
        "_check_disk_space",
        lambda model, **kwargs: calls["disk"].append((model, kwargs)),
    )
    monkeypatch.setattr(
        artifacts,
        "download_target_snapshot",
        lambda: calls["downloads"].append("target"),
    )
    monkeypatch.setattr(
        artifacts,
        "download_mtp_snapshot",
        lambda: calls["downloads"].append("mtp"),
    )
    monkeypatch.setattr(server_module, "configure_logging", lambda _level: "warning")
    monkeypatch.setattr(
        product_server, "run_server", lambda **kw: calls["run"].append(kw)
    )

    cli.serve_command(args)

    assert calls["memory"] == ["deepseek-v41-flash-reap-2bit"]
    assert calls["downloads"] == ["target", "mtp"]
    assert calls["disk"][0][1]["revision_override"] == artifacts.TARGET_REVISION
    assert calls["disk"][1][1]["revision_override"] == artifacts.MTP_REVISION
    assert calls["run"][0]["default_max_tokens"] == 4096
    assert calls["run"][0]["served_model_name"] == "deepseek-v41-flash-reap-2bit"
    assert "dspark-k4: experimental single-user" in capsys.readouterr().out


@pytest.mark.requires_mlx
@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--max-tokens", "4097"), "1 <= --max-tokens <= 4096"),
        (("--mcp-config", "/tmp/mcp.json"), "MCP is not supported"),
    ],
)
def test_serve_command_rejects_product_limits(monkeypatch, capsys, extra, message):
    from vllm_mlx import _version_check

    args = _product_serve_args(*extra)
    monkeypatch.setattr(_version_check, "prompt_upgrade_if_available", lambda: False)
    monkeypatch.setattr(
        _version_check, "print_staleness_warning_if_any", lambda **_kwargs: None
    )
    monkeypatch.setattr(cli, "_check_alias_min_memory", lambda _name: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *_a, **_k: None)
    monkeypatch.setattr(artifacts, "download_target_snapshot", lambda: None)
    monkeypatch.setattr(artifacts, "download_mtp_snapshot", lambda: None)

    with pytest.raises(SystemExit) as exc:
        cli.serve_command(args)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
