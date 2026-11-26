# SPDX-License-Identifier: Apache-2.0
import json
import select
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from huggingface_hub import constants
from safetensors.numpy import save_file

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
        (0, "Invalid text_config", "ok", 0),
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
    if output == "Invalid text_config":
        metadata.config["text_config"] = "malformed"
    monkeypatch.setattr(ft, "read_model_metadata", lambda _: metadata)
    out = tmp_path / "adapter path"
    resume = tmp_path / "previous.safetensors"
    resume.touch()
    if output == "Invalid text_config":
        (tmp_path / ft.ADAPTER_METADATA).write_text(
            json.dumps({"base_model": ALIAS, "num_layers": 16})
        )
    elif code == 0:
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"model": str(snapshot), "num_layers": 16})
        )
        (tmp_path / ft.ADAPTER_METADATA).write_text(json.dumps({"base_model": ALIAS}))
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
    killpg = MagicMock(side_effect=ProcessLookupError)
    monkeypatch.setattr("os.killpg", killpg)
    if code is None:
        child.wait.side_effect = [KeyboardInterrupt, -signal.SIGTERM]

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
        (out / "adapter_config.json").write_text(
            json.dumps({"model": str(snapshot), "num_layers": 16})
        )
        save_file(
            {"lora_a": np.ones((1, 1), dtype=np.float32)}, out / "adapters.safetensors"
        )
        return child

    monkeypatch.setattr(ft.subprocess, "Popen", fake_popen)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == exit_code
    assert not child.data.exists()
    if exit_code == 130:
        killpg.assert_called_once_with(child.pid, signal.SIGTERM)
        assert child.wait.call_count == 2
    else:
        killpg.assert_not_called()
    assert download.called == (code != 0)
    track.assert_called_once()
    event, props = track.call_args.args
    assert event == "finetune_completed" and validate(event, props) == props
    assert list(props.values()) == [status, "<1m", "<100", False]
    stdout, stderr = capsys.readouterr()
    assert output in stdout and "Estimated unified memory:" in stdout
    assert "The tuner prints measured 'Peak mem' during training." in stdout
    assert "Default --iters=" not in stdout
    assert "stop the local Rapid-MLX server" in stderr
    if output == "Invalid text_config":
        assert "Estimated unified memory: unavailable." in stdout
    else:
        assert "Warning: low available memory." in stderr
    if exit_code == 130:
        assert "Fine-tuning aborted." in stderr
    if exit_code:
        checkpoint = out / "adapters.safetensors"
        assert checkpoint.is_file()
        assert (
            shlex.join(["--resume", str(checkpoint), "--out", f"{out}-retry"]) in stderr
        )
    sidecar = out / ft.ADAPTER_METADATA
    assert sidecar.exists() == (code == 0)
    if code == 0:
        metadata = json.loads(sidecar.read_text())
        assert metadata["quantization_bits"] == 4
        assert metadata["num_layers"] == 16
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


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"tool_calls": "invalid"}, "tool_calls"),
        ({"content": "hello", "tool_calls": "invalid"}, "tool_calls"),
        ({"tool_calls": []}, "tool_calls"),
        ({"tool_calls": [1]}, "tool_calls"),
        ({"tool_calls": [{"function": {"name": "lookup"}}]}, "arguments"),
        ({"tool_calls": [{"function": {"name": "", "arguments": {}}}]}, "name"),
        (
            {"tool_calls": [{"function": {"name": "lookup", "arguments": []}}]},
            "arguments",
        ),
        (
            {"tool_calls": [{"function": {"name": "lookup", "arguments": "bad"}}]},
            "arguments",
        ),
        ({"content": ["invalid"]}, "content"),
        ({"content": [{"type": "text", "text": 3}]}, "content"),
    ],
)
def test_bad_chat_message(tmp_path, fields, error):
    data = tmp_path / "data.jsonl"
    good = {"messages": [{"role": "user", "content": "hello"}]}
    bad = {"messages": [{"role": "assistant", **fields}]}
    data.write_text(json.dumps(good) + "\n" + json.dumps(bad))
    with pytest.raises(ValueError, match=f"data.jsonl:2:.*{error}"):
        ft._read_rows(data)


def test_structured_chat_message(tmp_path):
    data = tmp_path / "data.jsonl"
    row = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "lookup", "arguments": {"id": 1}}},
                    {"function": {"name": "lookup", "arguments": '{"id":2}'}},
                    {"name": "lookup", "arguments": {}},
                ],
            },
        ]
    }
    data.write_text(json.dumps(row))
    assert ft._read_rows(data) == [data.read_bytes()]


def test_directory_data(tmp_path):
    (tmp_path / "train.jsonl").write_text(
        '{"messages":[{"role":"user","content":"hey"}]}\n'
    )
    (tmp_path / "valid.jsonl").write_text('{"prompt":"a","completion":"b"}\n')
    (tmp_path / "test.jsonl").write_text('{"text":"a"}\n')
    assert ft._prepare_data(tmp_path, "", 1) == (tmp_path, 1, 1, True)
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


@pytest.mark.parametrize(
    ("model", "config", "reason"),
    [
        (
            "qwen3.5-4b-4bit",
            {"model_type": "qwen3_5", "vision_config": {}},
            "vision/MLLM or audio",
        ),
        (REPO, {"num_nextn_predict_layers": 1}, "MTP checkpoint heads"),
        (REPO, {"quantization_config": {"quant_method": "fp8"}}, "fp8 repack"),
    ],
)
def test_incompatible_checkpoint_rejected_before_training(
    tmp_path, monkeypatch, capsys, model, config, reason
):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    config_download = MagicMock(return_value=str(config_path))
    monkeypatch.setattr(ft.model_downloads, "hf_hub_download", config_download)
    prepare = MagicMock()
    load = MagicMock()
    launch = MagicMock()
    monkeypatch.setattr(ft, "_prepare_data", prepare)
    monkeypatch.setattr(ft, "_training_model", load)
    monkeypatch.setattr(ft.subprocess, "Popen", launch)
    track = MagicMock()
    monkeypatch.setattr(ft, "track", track)
    out = tmp_path / "out"
    args = cli.build_parser().parse_args(
        ["finetune", model, "--data", "unused.jsonl", "--out", str(out)]
    )
    assert ft.finetune_command(args) == 1
    stdout, stderr = capsys.readouterr()
    assert model in stderr and reason in stderr and "Choose a text-only model" in stderr
    assert "Estimated unified memory" not in stdout
    config_download.assert_called_once_with(
        args.model, "config.json", local_files_only=True
    )
    prepare.assert_not_called()
    load.assert_not_called()
    launch.assert_not_called()
    assert not out.exists()
    assert track.call_args.args[1]["cmd_status"] == "err"


@pytest.mark.parametrize(
    ("rows", "batch", "iters"), [(1, 1, 3), (53, 4, 39), (2000, 4, 1000)]
)
def test_default_iterations_from_full_training_batches(
    tmp_path, monkeypatch, capsys, rows, batch, iters
):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text('{"text":"training"}\n' * rows)
    (data / "valid.jsonl").write_text('{"text":"validation"}\n' * batch)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}')
    out = tmp_path / "out"
    args = cli.build_parser().parse_args(
        [
            "finetune",
            str(model),
            f"--data={data}",
            f"--out={out}",
            f"--batch-size={batch}",
        ]
    )
    launch = MagicMock()
    monkeypatch.setattr(ft, "_train", launch)
    monkeypatch.setattr(ft, "version", lambda _: "0.31.3")
    track = MagicMock()
    monkeypatch.setattr(ft, "track", track)
    assert ft.finetune_command(args) == 0
    launch.assert_called_once_with(args, str(model), data, out)
    event, props = track.call_args.args
    assert validate(event, props) == props
    assert props["has_valid_split"] is True
    assert args.iters == iters
    assert f"Default --iters={iters} targets 3 epochs" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("losses", "hint"),
    [
        ([], False),
        (["3.040"], False),
        (["3.040", "2.550"], False),
        (["3.040", "3.040"], True),
        (["3.040", "2.550", "3.050"], True),
        (["3.040", "nan"], True),
        (["3.040", "invalid"], False),
    ],
)
def test_validation_loss_hint(monkeypatch, capsys, losses, hint):
    args = cli.build_parser().parse_args(
        ["finetune", REPO, "--data=data", "--iters=10"]
    )
    child = MagicMock()
    child.__enter__.return_value = child
    child.wait.return_value = 0
    child.stdout = [
        f"Iter {i}: Val loss {loss}, Val took 1.000s\n"
        for i, loss in enumerate(losses, 1)
    ]
    child.stdout.append("Iter 10: Train loss 2.000, Peak mem 3.300 GB\n")
    monkeypatch.setattr(ft.subprocess, "Popen", lambda *args, **kwargs: child)
    ft._train(args, REPO, "data", "out")
    stdout = capsys.readouterr().out
    assert stdout.startswith("".join(child.stdout))
    assert ("Try more training data or a different --learning-rate" in stdout) == hint


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


@pytest.mark.parametrize(
    ("hidden", "vocab", "weights", "batch", "estimate"),
    [
        (1024, 151936, 335450584, 1, "3.8 GB"),  # Qwen3-0.6B
        (2048, 65536, 658540250, 1, "4.3 GB"),  # LFM2.5-1.2B
        (4096, 151936, 4607835174, 1, "10.9 GB"),  # Qwen3-8B
        (3072, 128256, 1807496278, 2, "10.8 GB"),  # Llama-3.2-3B
    ],
)
def test_memory_estimate_calibration(
    tmp_path, monkeypatch, capsys, hidden, vocab, weights, batch, estimate
):
    metadata = SimpleNamespace(
        snapshot_dir=None, config={"hidden_size": hidden, "vocab_size": vocab}
    )
    monkeypatch.setattr(ft, "read_model_metadata", lambda _: metadata)
    monkeypatch.setattr(ft, "size_bytes", lambda _: weights)
    monkeypatch.setattr(ft.psutil, "process_iter", lambda _: [])
    args = SimpleNamespace(
        model=REPO, iters=1, num_layers=16, batch_size=batch, max_seq_length=1024
    )
    ft._print_plan(args, tmp_path, 2, 2)
    assert (
        f"Estimated unified memory: approximately {estimate}" in capsys.readouterr().out
    )


def test_shutdown_kills_descendant_after_leader_exits(monkeypatch):
    script = """\
import os, signal
if os.fork() == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print("ready", flush=True)
while True:
    signal.pause()
"""
    monkeypatch.setattr(ft, "_STOP_GRACE_SECONDS", 0.15)
    with subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, start_new_session=True
    ) as child:
        try:
            assert select.select([child.stdout], [], [], 5)[0]
            assert child.stdout.readline() == b"ready\n"
            ft._stop_training(child)
            assert child.returncode == -signal.SIGTERM
            # The descendant inherits stdout. EOF confirms it also exited.
            assert child.communicate(timeout=2)[0] == b""
        finally:
            if child.returncode is None or not child.stdout.closed:
                try:
                    ft.os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_second_interrupt_skips_shutdown_grace(monkeypatch):
    child = MagicMock()
    killpg = MagicMock()
    monkeypatch.setattr(ft.os, "killpg", killpg)
    monkeypatch.setattr(ft.time, "sleep", MagicMock(side_effect=KeyboardInterrupt))
    ft._stop_training(child)
    assert [call.args for call in killpg.call_args_list] == [
        (child.pid, signal.SIGTERM),
        (child.pid, 0),
        (child.pid, signal.SIGKILL),
    ]
    child.wait.assert_called_once()


@pytest.mark.parametrize(
    ("filename", "config", "error"),
    [
        (
            "adapter_config.json",
            {"model": "other/model", "num_layers": 4},
            "base mismatch",
        ),
        (ft.ADAPTER_METADATA, {"base_model": REPO, "num_layers": 8}, "--num-layers=4"),
        (ft.ADAPTER_METADATA, {"base_model": REPO}, "missing num_layers"),
        ("adapter_config.json", [], "invalid adapter_config.json"),
        ("adapter_config.json", {"model": REPO, "num_layers": "4"}, "--num-layers=4"),
        (
            "adapter_config.json",
            {"model": REPO, "num_layers": 4, "fine_tune_type": "dora"},
            "must declare a LoRA adapter",
        ),
    ],
)
def test_resume_rejected_before_launch(
    tmp_path, monkeypatch, capsys, filename, config, error
):
    for split in ("train", "valid"):
        (tmp_path / f"{split}.jsonl").write_text('{"text":"example"}')
    resume = tmp_path / "previous.safetensors"
    resume.touch()
    (tmp_path / filename).write_text(json.dumps(config))
    launch = MagicMock()
    monkeypatch.setattr(ft.subprocess, "Popen", launch)
    monkeypatch.setattr(ft, "track", MagicMock())
    args = cli.build_parser().parse_args(
        [
            "finetune",
            ALIAS,
            "--data",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--resume",
            str(resume),
            "--num-layers",
            "4",
            "--batch-size",
            "1",
        ]
    )
    assert ft.finetune_command(args) == 1
    stderr = capsys.readouterr().err
    assert "--resume" in stderr and error in stderr
    launch.assert_not_called()


def test_output_collision(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ft, "_print_plan", MagicMock())
    monkeypatch.setattr(ft.model_downloads, "disabled", lambda: False)
    for split in ("train", "valid"):
        (tmp_path / f"{split}.jsonl").write_text('{"text":"example"}')
    assert ft._training_model(str(tmp_path), REPO) == str(tmp_path)
    out = tmp_path / "existing"
    out.mkdir()
    owned = out / "user-file"
    owned.write_text("keep me")
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
    assert owned.read_text() == "keep me"


@pytest.mark.parametrize(
    ("failure", "checkpoint", "exit_code"),
    [
        ("train", False, 1),
        ("interrupt", False, 130),
        ("metadata", False, 1),
        ("metadata", True, 1),
    ],
)
def test_failed_output_can_be_retried(
    tmp_path, monkeypatch, capsys, failure, checkpoint, exit_code
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}')
    data = tmp_path / "data.jsonl"
    data.write_text('{"text":"example"}\n' * 2)
    out = tmp_path / "adapter path"
    argv = ["finetune", str(model), f"--data={data}", f"--out={out}", "--batch-size=1"]
    saved = out / "0000100_adapters.safetensors"

    def train(args, model, data, out):
        (out / "adapter_config.json").write_text(
            json.dumps({"model": model, "num_layers": args.num_layers})
        )
        (out / "adapters.safetensors").write_bytes(b"truncated")
        if checkpoint:
            save_file({"lora_a": np.ones((1, 1), dtype=np.float32)}, saved)
        if failure == "interrupt":
            raise KeyboardInterrupt
        if failure == "train":
            raise ValueError("tuner failed")

    monkeypatch.setattr(ft, "_train", train)
    monkeypatch.setattr(ft, "version", lambda _: "0.31.3")
    monkeypatch.setattr(ft, "track", MagicMock())
    write_metadata = ft.atomic_write_json
    if failure == "metadata":
        monkeypatch.setattr(
            ft, "atomic_write_json", MagicMock(side_effect=OSError("metadata failed"))
        )
    parser = cli.build_parser()
    assert ft.finetune_command(parser.parse_args(argv)) == exit_code
    stderr = capsys.readouterr().err
    assert out.exists() == checkpoint
    if checkpoint:
        assert saved.is_file()
        retry_out = f"{out}-retry"
        assert shlex.join(["--resume", str(saved), "--out", retry_out]) in stderr
        argv = [*argv, f"--out={retry_out}", f"--resume={saved}"]
    monkeypatch.setattr(ft, "_train", MagicMock())
    monkeypatch.setattr(ft, "atomic_write_json", write_metadata)
    assert ft.finetune_command(parser.parse_args(argv)) == 0
