# SPDX-License-Identifier: Apache-2.0
"""Run mlx-lm LoRA fine-tuning without loading MLX into the CLI process."""

import json
import math
import random
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import psutil
from huggingface_hub.utils import validate_repo_id

from . import __version__, model_downloads
from ._download_gate import _format_size
from .launch._common import atomic_write_json
from .local_model_path import raise_if_missing_local_model, resolved_model_ref
from .lora_adapter import ADAPTER_METADATA
from .model_aliases import resolve_model, resolve_revision, resolve_subfolder
from .model_metadata import read_model_metadata, resolve_offline_cached_snapshot
from .model_reference import normalize_model_reference
from .model_sizes import size_bytes
from .request import ENGINE_ABORT_CODE_INSUFFICIENT_MEMORY, classify_engine_abort
from .serve_process import serve_subcommand_index
from .telemetry.track import track

_MEMORY_TIP = "Try smaller --batch-size, --num-layers or --max-seq-length, or stop the local server."
_TUNER_FLAGS = ("iters", "batch_size", "num_layers", "learning_rate", "max_seq_length")
# Match the mlx-lm downloader when materializing a pinned checkpoint.
_MODEL_FILES = [
    "*.json",
    "model*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
]


def _row_format(row):
    if isinstance(row, dict):
        if "prompt" in row and "completion" in row:
            if all(isinstance(row[key], str) for key in ("prompt", "completion")):
                return "prompt/completion"
        elif "messages" in row:
            messages = row["messages"]
            if (
                isinstance(messages, list)
                and messages
                and all(
                    isinstance(message, dict)
                    and isinstance(message.get("role"), str)
                    and (
                        isinstance(message.get("content"), (str, list))
                        or message.get("tool_calls")
                    )
                    for message in messages
                )
            ):
                return "messages"
        elif isinstance(row.get("text"), str) and row["text"]:
            return "text"
    raise ValueError("expected a JSON object with messages, prompt/completion, or text")


def _read_rows(path):
    rows, expected = [], None
    with path.open("rb") as file:
        for number, line in enumerate(file, 1):
            try:
                # mlx-lm chooses a dataset class from the first row of each split.
                kind = _row_format(json.loads(line.decode("utf-8")))
                if expected is not None and kind != expected:
                    raise ValueError(f"expected {expected} data to match line 1")
                expected = kind
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            rows.append(line)
    if not rows:
        raise ValueError(f"{path}: expected at least one JSONL row")
    return rows


def _prepare_data(path, scratch, batch_size):
    if path.is_dir():
        train, valid = (
            _read_rows(path / name) for name in ("train.jsonl", "valid.jsonl")
        )
        if (path / "test.jsonl").exists():
            _read_rows(path / "test.jsonl")  # mlx-lm opens it even without --test.
    elif path.suffix == ".jsonl":
        rows = _read_rows(path)
        random.Random(0).shuffle(rows)
        valid_count = max(batch_size, (len(rows) + 9) // 10)
        valid, train = rows[:valid_count], rows[valid_count:]
        path = Path(scratch)
        for name, split in (("train", train), ("valid", valid)):
            (path / f"{name}.jsonl").write_bytes(
                b"\n".join(row.rstrip(b"\r\n") for row in split) + b"\n"
            )
    else:
        raise ValueError(
            "--data must be a .jsonl file or a directory with train.jsonl and valid.jsonl"
        )
    if min(len(train), len(valid)) < batch_size:
        raise ValueError(
            f"Each split needs at least --batch-size={batch_size} rows "
            f"(train={len(train)}, valid={len(valid)}). Add data or lower --batch-size."
        )
    return path, len(train), len(valid)


def _print_plan(args, out, train, valid):
    print(
        f"Fine-tuning {args.model} for {args.iters} iterations ({train} train, {valid} valid)."
    )
    print(f"Output: {out}", flush=True)
    try:
        metadata = read_model_metadata(args.model)
        config = (metadata.config or {}) if metadata else {}
        config = config.get("text_config") or config
        weights = size_bytes(args.model)
        if metadata and metadata.snapshot_dir:
            shards = metadata.snapshot_dir.glob("model*.safetensors")
            weights = sum(shard.stat().st_size for shard in shards) or weights
        # Approximate saved activations across layers. Actual peaks vary by model.
        hidden = config.get("hidden_size") or 4096
        layers = max(args.num_layers, 0) or config.get("num_hidden_layers") or 32
        activations = 16 * hidden * layers * args.batch_size * args.max_seq_length
        estimate = (weights or 0) + activations
        suffix = " plus unknown base weights" if weights is None else ""
        available = psutil.virtual_memory().available
        print(
            f"Estimated unified memory: {_format_size(estimate)}{suffix} "
            f"({_format_size(available)} available).",
            flush=True,
        )
        if estimate > available:
            print(f"Warning: low available memory. {_MEMORY_TIP}", file=sys.stderr)
    except (OSError, TypeError, ValueError, psutil.Error):
        print("Estimated unified memory: unavailable.", flush=True)
    try:
        running = any(
            serve_subcommand_index(p.info.get("cmdline") or []) is not None
            for p in psutil.process_iter(["cmdline"])
        )
    except (OSError, psutil.Error):
        running = False
    if running:
        print(
            "Warning: stop the local Rapid-MLX server to free memory.", file=sys.stderr
        )


def _training_model(model, reference):
    if Path(model).exists():
        return str(Path(model).absolute())
    subfolder = resolve_subfolder(reference)
    pinned = resolve_revision(reference)
    if not (pinned or subfolder or model_downloads.disabled()):
        return model
    if cached := resolve_offline_cached_snapshot(reference):
        return str(cached)
    prefix = f"{subfolder}/" if subfolder else ""
    snapshot = model_downloads.snapshot_download(
        reference if pinned else model,
        allow_patterns=[prefix + pattern for pattern in _MODEL_FILES],
    )
    return str(Path(snapshot) / (subfolder or ""))


def _train(args, model, data, out):
    command = [
        sys.executable,
        "-u",
        "-m",
        "mlx_lm",
        "lora",
        "--train",
        "--fine-tune-type=lora",
        f"--model={model}",
        f"--data={data}",
        f"--adapter-path={out}",
    ]
    command.extend(
        f"--{key.replace('_', '-')}={getattr(args, key)}" for key in _TUNER_FLAGS
    )
    if args.resume:
        command.append(
            f"--resume-adapter-file={Path(args.resume).expanduser().absolute()}"
        )
    oom = False
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
        errors="replace",
    ) as child:
        try:
            assert child.stdout is not None
            for line in child.stdout:
                print(line, end="", flush=True)
                oom |= line.strip().startswith("MemoryError") or (
                    classify_engine_abort(line) == ENGINE_ABORT_CODE_INSUFFICIENT_MEMORY
                )
            code = child.wait()
        finally:
            if child.poll() is None:
                child.kill()
    if code in (-signal.SIGINT, 130):
        raise KeyboardInterrupt
    if code and oom:
        raise MemoryError
    if code:
        raise ValueError(
            f"mlx-lm exited with status {code}. Check the tuner output above and retry."
        )


def finetune_command(args):
    started = time.monotonic()
    train = valid = 0
    status = "err"
    try:
        if (
            min(args.iters, args.batch_size, args.max_seq_length) < 1
            or args.num_layers < -1
        ):
            raise ValueError(
                "Use positive --iters, --batch-size and --max-seq-length, and --num-layers >= -1."
            )
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError("Use a positive, finite --learning-rate.")
        if args.resume and not Path(args.resume).expanduser().is_file():
            raise ValueError("--resume must point to an existing LoRA weights file.")
        reference = resolved_model_ref(normalize_model_reference(args.model))
        raise_if_missing_local_model(reference)
        args.model = resolve_model(reference)
        local = Path(args.model).exists()
        if not local:
            validate_repo_id(args.model)
            if "/" not in args.model:
                raise ValueError(
                    f"Unknown model alias {args.model!r}. Run rapid-mlx models to list aliases."
                )
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        default = f"~/.rapid-mlx/adapters/{Path(reference).name}-{stamp}"
        out = Path(args.out or default).expanduser().absolute()
        with tempfile.TemporaryDirectory(prefix="rapid-mlx-finetune-") as scratch:
            data, train, valid = _prepare_data(
                Path(args.data).expanduser(), scratch, args.batch_size
            )
            _print_plan(args, out, train, valid)
            model = _training_model(args.model, reference)
            out.mkdir(parents=True, exist_ok=False)
            _train(args, model, data, out)
            metadata = read_model_metadata(model)
            config = (metadata.config or {}) if metadata else {}
            quant = (
                config.get("quantization") or config.get("quantization_config") or {}
            )
            atomic_write_json(
                out / ADAPTER_METADATA,
                {
                    "base_model": model if local else args.model,
                    "quantization_bits": quant.get("bits"),
                    "adapter_type": "lora",
                    "rapid_mlx_version": __version__,
                    "mlx_lm_version": version("mlx-lm"),
                },
            )
        serve_model = model if local and reference == args.model else reference
        print(shlex.join(["rapid-mlx", "serve", serve_model, "--adapter", str(out)]))
        status = "ok"
        return 0
    except KeyboardInterrupt:
        status = "abort"
        print(
            "Fine-tuning aborted. Use --resume with saved weights if available.",
            file=sys.stderr,
        )
        return 130
    except MemoryError:
        status = "oom"
        print(f"Fine-tuning ran out of memory. {_MEMORY_TIP}", file=sys.stderr)
        return 1
    except FileExistsError:
        print("Fine-tuning error: use a new --out directory.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Fine-tuning error: {exc}", file=sys.stderr)
        return 1
    finally:
        elapsed = time.monotonic() - started
        rows = train + valid
        duration = "<1m" if elapsed < 60 else "1-5m" if elapsed < 300 else "5m+"
        rows_bucket = "0" if not rows else "<100" if rows < 100 else "100-999"
        track(
            "finetune_completed",
            {
                "cmd_status": status,
                "duration_bucket": duration,
                "rows_bucket": "1000+" if rows >= 1000 else rows_bucket,
                "has_valid_split": bool(valid),
            },
        )
