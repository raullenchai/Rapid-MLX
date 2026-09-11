from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_mlx import cli
from vllm_mlx.models.deepseek_v41_native import artifacts


def _write_file(path: Path, size: int, byte: bytes = b"x") -> str:
    path.write_bytes(byte * size)
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    monkeypatch.setitem(
        sys.modules,
        "vllm_mlx.optimizations",
        SimpleNamespace(get_system_memory_gb=lambda: 192),
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


def test_product_alias_pulls_pinned_target_and_three_shard_sidecar(monkeypatch):
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
