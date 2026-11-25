# SPDX-License-Identifier: Apache-2.0
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from huggingface_hub import constants

from rapid_mlx import cli
from rapid_mlx import finetune as ft
from rapid_mlx.lora_adapter import validate_adapter
from rapid_mlx.telemetry.registry import validate

ALIAS = "qwen3-0.6b-4bit"
REPO = "mlx-community/Qwen3-0.6B-4bit"


@pytest.mark.parametrize(
    ("code", "output", "status", "exit_code"),
    [
        (0, "Iter 1", "ok", 0),
        (1, "RuntimeError: Metal outofmemory", "oom", 1),
        (-9, "terminated", "err", 1),
        (130, "interrupted", "abort", 130),
        (None, "", "abort", 130),
    ],
)
def test_cli_finetune(tmp_path, monkeypatch, capsys, code, output, status, exit_code):
    datasets = tmp_path / "training data.jsonl"
    datasets.write_text("\n".join(json.dumps({"text": str(i)}) for i in range(20)))
    hub = tmp_path / "hub"
    snapshot = hub / "models--mlx-community--Qwen3-0.6B-4bit/snapshots/pin"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").touch()
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(hub))
    monkeypatch.setattr(
        ft, "resolve_offline_cached_snapshot", lambda _: snapshot if code == 0 else None
    )
    download = MagicMock(return_value=str(snapshot))
    monkeypatch.setattr(ft.model_downloads, "snapshot_download", download)
    monkeypatch.setattr(ft, "version", lambda _: "0.31.3")
    metadata = SimpleNamespace(
        snapshot_dir=snapshot, config={"quantization": {"bits": 4}}
    )
    monkeypatch.setattr(ft, "read_model_metadata", lambda _: metadata)
    out = tmp_path / "adapter path"
    resume = tmp_path / "previous.safetensors"
    resume.touch()
    command = "lora" if code == 0 else "finetune"
    argv = ["rapid-mlx", command, ALIAS, "--data", str(datasets), f"--out={out}"]
    monkeypatch.setattr(sys, "argv", argv + [f"--resume={resume}", "--iters=10"])
    monkeypatch.setattr(
        ft.psutil, "virtual_memory", lambda: SimpleNamespace(available=100)
    )
    server = SimpleNamespace(info={"cmdline": ["rapid-mlx", "serve", REPO]})
    monkeypatch.setattr(ft.psutil, "process_iter", lambda _: [server])
    track = MagicMock()
    monkeypatch.setattr(ft, "track", track)
    child = MagicMock()
    child.__enter__.return_value = child
    child.stdout = [output + "\n"]
    child.wait.return_value = child.poll.return_value = code
    if code is None:
        child.wait.side_effect = KeyboardInterrupt

    def fake_popen(argv, **kwargs):
        assert argv[0] == sys.executable
        assert argv[1:7] == [
            "-u",
            "-m",
            "mlx_lm",
            "lora",
            "--train",
            "--fine-tune-type=lora",
        ]
        assert {f"--model={snapshot}", f"--resume-adapter-file={resume}"} <= set(argv)
        defaults = "--iters=10 --batch-size=4 --num-layers=16 --learning-rate=1e-05 --max-seq-length=2048"
        assert set(defaults.split()) <= set(argv)
        assert kwargs["start_new_session"]
        data = Path(
            next(item.split("=", 1)[1] for item in argv if item.startswith("--data="))
        )
        splits = [
            (data / f"{name}.jsonl").read_text().splitlines()
            for name in ("train", "valid")
        ]
        assert list(map(len, splits)) == [16, 4]
        assert len(set(splits[0] + splits[1])) == 20
        child.data = data
        (out / "adapter_config.json").write_text(json.dumps({"model": str(snapshot)}))
        (out / "adapters.safetensors").touch()
        return child

    monkeypatch.setattr(ft.subprocess, "Popen", fake_popen)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == exit_code
    assert not child.data.exists()
    assert child.kill.called == (code is None)
    assert download.called == (code != 0)
    track.assert_called_once()
    event, props = track.call_args.args
    assert event == "finetune_completed" and validate(event, props) == props
    assert list(props.values()) == [status, "<1m", "<100", True]
    stdout, stderr = capsys.readouterr()
    assert output in stdout and "Estimated unified memory:" in stdout
    assert (
        "Warning: low available memory." in stderr
        and "stop the local Rapid-MLX server" in stderr
    )
    sidecar = out / ft.ADAPTER_METADATA
    assert sidecar.exists() == (code == 0)
    if code == 0:
        assert json.loads(sidecar.read_text())["quantization_bits"] == 4
        serve = shlex.join(["rapid-mlx", "serve", ALIAS, "--adapter", str(out)])
        assert serve in stdout
        assert validate_adapter(str(out), ALIAS).path == str(out)
        sidecar.write_text('{"base_model":"other/model"}')
        with pytest.raises(ValueError, match="base mismatch.*other/model"):
            validate_adapter(str(out), ALIAS)
        sidecar.write_text('{"base_model":""}')
        with pytest.raises(ValueError, match="invalid rapid_mlx_adapter.json"):
            validate_adapter(str(out), ALIAS)


@pytest.mark.parametrize(
    "row",
    [
        b"{}",
        b"[]",
        b"{",
        b'{"text":3}',
        b"\xff",
        b'{"messages":[]}',
        b'{"prompt":"a","completion":"b"}',
    ],
)
def test_bad_row(tmp_path, row):
    data = tmp_path / "data.jsonl"
    data.write_bytes(b'{"text":"first"}\n' + row)
    with pytest.raises(ValueError, match="data.jsonl:2:"):
        ft._read_rows(data)


def test_directory_data(tmp_path):
    (tmp_path / "train.jsonl").write_text(
        '{"messages":[{"role":"user","content":"hey"}]}\n'
    )
    (tmp_path / "valid.jsonl").write_text('{"prompt":"a","completion":"b"}\n')
    (tmp_path / "test.jsonl").write_text('{"text":"a"}\n')
    assert ft._prepare_data(tmp_path, "", 1) == (tmp_path, 1, 1)
    with pytest.raises(ValueError, match="Each split needs"):
        ft._prepare_data(tmp_path, "", 2)
    (tmp_path / "train.jsonl").write_text("")
    with pytest.raises(ValueError, match="at least one JSONL row"):
        ft._prepare_data(tmp_path, "", 1)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (["--iters", "0"], "Use positive"),
        (["--learning-rate", "nan"], "positive, finite"),
        (["--resume", "missing"], "existing LoRA weights"),
        ([], "--data must be"),
    ],
)
def test_preflight_error(tmp_path, monkeypatch, capsys, overrides, error):
    args = cli.build_parser().parse_args(
        ["finetune", REPO, "--data", str(tmp_path / "bad"), *overrides]
    )
    track = MagicMock()
    monkeypatch.setattr(ft, "track", track)
    assert ft.finetune_command(args) == 1
    assert error in capsys.readouterr().err
    track.assert_called_once()
    event, props = track.call_args.args
    assert validate(event, props) == props
    assert list(props.values()) == ["err", "<1m", "0", False]


def test_unknown_alias_completes(monkeypatch, capsys):
    track = MagicMock()
    monkeypatch.setattr(ft, "track", track)
    args = cli.build_parser().parse_args(
        ["lora", "not-a-real-alias", "--data", "data.jsonl"]
    )
    assert ft.finetune_command(args) == 1
    assert "Unknown model alias" in capsys.readouterr().err
    assert track.call_args.args[1]["cmd_status"] == "err"


def test_probes_are_advisory(tmp_path, monkeypatch, capsys):
    denied = MagicMock(side_effect=OSError)
    monkeypatch.setattr(ft, "read_model_metadata", denied)
    monkeypatch.setattr(ft.psutil, "process_iter", denied)
    ft._print_plan(SimpleNamespace(model=REPO, iters=1), tmp_path, 1, 1)
    assert "Estimated unified memory: unavailable." in capsys.readouterr().out


def test_output_collision(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ft, "_print_plan", MagicMock())
    monkeypatch.setattr(ft.model_downloads, "disabled", lambda: False)
    for split in ("train", "valid"):
        (tmp_path / f"{split}.jsonl").write_text('{"text":"example"}')
    assert ft._training_model(str(tmp_path), REPO) == str(tmp_path)
    out = tmp_path / "existing"
    out.mkdir()
    args = cli.build_parser().parse_args(
        [
            "finetune",
            "test-org/test-repo",
            f"--data={tmp_path}",
            f"--out={out}",
            "--batch-size=1",
        ]
    )
    assert ft.finetune_command(args) == 1
    assert "use a new --out directory" in capsys.readouterr().err
