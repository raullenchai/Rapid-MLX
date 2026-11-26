# SPDX-License-Identifier: Apache-2.0
"""Run mlx-lm LoRA fine-tuning without loading MLX into the CLI process."""

import json
import math
import os
import random
import re
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import psutil
from huggingface_hub.utils import validate_repo_id
from safetensors import SafetensorError, safe_open

from . import __version__, model_downloads
from ._download_gate import _format_size
from .launch._common import atomic_write_json
from .local_model_path import raise_if_missing_local_model, resolved_model_ref
from .lora_adapter import (
    ADAPTER_METADATA,
    _base_identity,
    _read_checkpoint_config,
    _validate_checkpoint_config,
    validate_adapter,
    validate_adapter_runtime,
)
from .model_aliases import resolve_model, resolve_revision, resolve_subfolder
from .model_metadata import read_model_metadata, resolve_offline_cached_snapshot
from .model_reference import normalize_model_reference
from .model_sizes import size_bytes
from .request import ENGINE_ABORT_CODE_INSUFFICIENT_MEMORY, classify_engine_abort
from .serve_process import serve_subcommand_index
from .telemetry.track import track

_MEMORY_TIP = "Try smaller --batch-size, --num-layers or --max-seq-length, or stop the local server."
_TUNER_FLAGS = ("iters", "batch_size", "num_layers", "learning_rate", "max_seq_length")
_VAL_LOSS = re.compile(r"^Iter \d+: Val loss (-?(?:\d+(?:\.\d+)?|inf|nan))(?=,|\s|$)")
_DEFAULT_EPOCHS = 3
_MAX_DEFAULT_ITERS = 1000
_STOP_GRACE_SECONDS = 2.0
_NUMBERED_ADAPTER = re.compile(r"\d{7,}_adapters\.safetensors")
_LORA_TENSOR = re.compile(r"(?:^|\.)(lora_a|lora_b)(?:\.|$)")
# Match the mlx-lm downloader when materializing a checkpoint.
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


def _validate_message(message):
    if not isinstance(message, dict) or not isinstance(message.get("role"), str):
        raise ValueError("each message must be an object with a string role")
    content = message.get("content")
    if isinstance(content, list) and not all(
        isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
        for part in content
    ):
        raise ValueError("message content list must contain text objects")
    if "tool_calls" in message:
        calls = message["tool_calls"]
        if not isinstance(calls, list) or not calls:
            raise ValueError("message tool_calls must be a non-empty list of objects")
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError(
                    "message tool_calls must be a non-empty list of objects"
                )
            function = call.get("function", call)
            name = function.get("name") if isinstance(function, dict) else None
            if not isinstance(name, str) or not name.strip():
                raise ValueError("each tool call must have a non-empty function name")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = None
            if not isinstance(arguments, dict):
                raise ValueError(
                    "tool call arguments must be an object or a JSON object string"
                )
        if content is None:
            return
    if not isinstance(content, (str, list)):
        raise ValueError("message content must be a string or a list of text objects")


def _row_format(row):
    if isinstance(row, dict):
        if "prompt" in row and "completion" in row:
            if all(isinstance(row[key], str) for key in ("prompt", "completion")):
                return "prompt/completion"
        elif "messages" in row:
            messages = row["messages"]
            if isinstance(messages, list) and messages:
                for message in messages:
                    _validate_message(message)
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
    has_valid_split = path.is_dir()
    if has_valid_split:
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
    return path, len(train), len(valid), has_valid_split


def _print_plan(args, out, train, valid):
    print(
        f"Fine-tuning {args.model} for {args.iters} iterations ({train} train, {valid} valid)."
    )
    print(f"Output: {out}", flush=True)
    try:
        metadata = read_model_metadata(args.model)
        config = (metadata.config or {}) if metadata else {}
        if (text_config := config.get("text_config")) is not None:
            config = text_config
        weights = size_bytes(args.model)
        if metadata and metadata.snapshot_dir:
            shards = metadata.snapshot_dir.glob("model*.safetensors")
            weights = sum(shard.stat().st_size for shard in shards) or weights
        # Text-model calibration against measured mlx-lm LoRA peaks. Allow for
        # backward intermediates in the activation budget and extra workspace.
        hidden = config.get("hidden_size") or 4096
        layers = max(args.num_layers, 0) or config.get("num_hidden_layers") or 32
        tokens = args.batch_size * args.max_seq_length
        activations = 56 * hidden * layers * tokens
        # Allow one float32 buffer each for the full-vocabulary logits and their
        # gradient during cross-entropy training.
        logits_and_grad = 8 * (config.get("vocab_size") or 0) * tokens
        workspace = 1_250_000_000
        estimate = (weights or 0) + activations + logits_and_grad + workspace
        suffix = " plus unknown base weights" if weights is None else ""
        available = psutil.virtual_memory().available
        print(
            f"Estimated unified memory: approximately {_format_size(estimate)}{suffix} "
            f"({_format_size(available)} available).",
            flush=True,
        )
        if estimate > available:
            print(f"Warning: low available memory. {_MEMORY_TIP}", file=sys.stderr)
    except Exception:
        print("Estimated unified memory: unavailable.", flush=True)
    print("The tuner prints measured 'Peak mem' during training.", flush=True)
    try:
        running = any(
            serve_subcommand_index(p.info.get("cmdline") or []) is not None
            for p in psutil.process_iter(["cmdline"])
        )
    except Exception:
        running = False
    if running:
        print(
            "Warning: stop the local Rapid-MLX server to free memory.", file=sys.stderr
        )


def _training_model(model, reference):
    if Path(model).exists():
        return str(Path(model).absolute())
    repo = reference if resolve_revision(reference) else model
    # The cache resolver already returns the checkpoint directory.
    if cached := resolve_offline_cached_snapshot(repo):
        return str(cached)
    subfolder = resolve_subfolder(reference)
    prefix = f"{subfolder}/" if subfolder else ""
    snapshot = model_downloads.snapshot_download(
        repo, allow_patterns=[prefix + pattern for pattern in _MODEL_FILES]
    )
    return str(Path(snapshot) / (subfolder or ""))


def _check_checkpoint(model, reference):
    try:
        if not Path(model).is_dir():
            repo = reference if resolve_revision(reference) else model
            if cached := resolve_offline_cached_snapshot(repo):
                model = str(cached)
        config = _read_checkpoint_config(model, reference)
        if not config:
            raise ValueError(f"missing or invalid config.json for {model}")
        _validate_checkpoint_config(config)
    except Exception as exc:
        raise ValueError(
            f"Cannot fine-tune {reference!r} for serving with LoRA. {exc}. "
            "Choose a text-only model that supports LoRA adapters."
        ) from exc


def _validate_weights(path, rank=None):
    try:
        with safe_open(str(path), framework="np") as weights:
            groups = {}
            keys = weights.keys()
            for key in keys:
                match = _LORA_TENSOR.search(key)
                shape = weights.get_slice(key).get_shape()
                if not match or len(shape) not in (2, 3) or not all(shape):
                    raise ValueError(f"{path} contains an invalid LoRA tensor: {key}")
                # Expert LoRA adds an expert axis and transposes the dense layout.
                rank_axis = -2 if (match[1] == "lora_a") == (len(shape) == 3) else -1
                if rank is not None and shape[rank_axis] != rank:
                    raise ValueError(
                        f"{path} tensor {key} does not match LoRA rank {rank}"
                    )
                start, end = match.span(1)
                groups.setdefault((key[:start], key[end:]), set()).add(match[1])
            if not groups or any(
                pair != {"lora_a", "lora_b"} for pair in groups.values()
            ):
                raise ValueError(f"{path} must contain non-empty LoRA tensor pairs")
    except (OSError, SafetensorError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def _validate_output(out, model, num_layers):
    try:
        validate_adapter(str(out), model)
        config = json.loads((out / "adapter_config.json").read_text(encoding="utf-8"))
        if (
            type(config.get("num_layers")) is not int
            or config["num_layers"] != num_layers
        ):
            raise ValueError("adapter_config.json must declare the trained num_layers")
        params = config.get("lora_parameters")
        if (
            not isinstance(params, dict)
            or type(params.get("rank")) is not int
            or params["rank"] < 1
            or any(
                type(params.get(key)) not in (int, float)
                or not math.isfinite(params[key])
                for key in ("dropout", "scale")
            )
            or not 0 <= params["dropout"] < 1
            or (
                (keys := params.get("keys")) is not None
                and (
                    not isinstance(keys, list)
                    or not all(isinstance(key, str) for key in keys)
                )
            )
        ):
            raise ValueError("adapter_config.json is missing usable lora_parameters")
        _validate_weights(out / "adapters.safetensors", params["rank"])
    except (OSError, ValueError) as exc:
        raise ValueError(f"mlx-lm did not produce a usable adapter: {exc}") from exc


def _validate_resume(weights, model, num_layers):
    if not weights.is_file():
        raise ValueError("--resume must point to an existing LoRA weights file.")
    found_metadata = layers_verified = False
    for filename, base_key, type_key in (
        ("adapter_config.json", "model", "fine_tune_type"),
        (ADAPTER_METADATA, "base_model", "adapter_type"),
    ):
        path = weights.parent / filename
        if not path.exists():
            continue
        found_metadata = True
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
            base = config.get(base_key) if isinstance(config, dict) else None
            if not isinstance(base, str) or not base.strip():
                raise ValueError(f"{base_key} must be a nonempty string")
        except (OSError, ValueError) as exc:
            raise ValueError(f"--resume invalid {filename}: {exc}") from exc
        saved_path = Path(base).expanduser()
        selected_path = Path(model).expanduser()
        if _base_identity(base) != _base_identity(model) or (
            saved_path.is_dir()
            and selected_path.is_dir()
            and saved_path.resolve() != selected_path.resolve()
        ):
            raise ValueError(
                f"--resume base mismatch in {filename}: trained on {base!r}, selected {model!r}"
            )
        if config.get(type_key, "lora") != "lora":
            raise ValueError(f"--resume {filename} must declare a LoRA adapter")
        if "num_layers" in config:
            saved = config["num_layers"]
            if type(saved) is not int or saved != num_layers:
                raise ValueError(
                    f"--resume layer mismatch in {filename}: saved num_layers={saved!r}, "
                    f"requested --num-layers={num_layers}"
                )
            layers_verified = True
    if not found_metadata:
        raise ValueError(
            f"--resume cannot verify the base and layers without {ADAPTER_METADATA} "
            "or adapter_config.json beside the weights"
        )
    if not layers_verified:
        raise ValueError("--resume adapter metadata is missing num_layers")
    try:
        _validate_weights(weights)
    except ValueError as exc:
        raise ValueError(f"--resume invalid LoRA weights: {exc}") from exc


def _stop_training(child):
    try:
        try:
            os.killpg(child.pid, signal.SIGTERM)
            deadline = time.monotonic() + _STOP_GRACE_SECONDS
            # The group can outlive its leader, so keep checking after reaping it.
            while time.monotonic() < deadline:
                child.poll()
                os.killpg(child.pid, 0)
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass  # A second Ctrl-C skips the rest of the grace period.
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # The whole group already exited.
    finally:
        child.wait()


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
    first_val_loss = last_val_loss = None
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
                if match := _VAL_LOSS.match(line):
                    loss = float(match[1])
                    if first_val_loss is None:
                        first_val_loss = loss
                    else:
                        last_val_loss = loss
                oom |= line.strip().startswith("MemoryError") or (
                    classify_engine_abort(line) == ENGINE_ABORT_CODE_INSUFFICIENT_MEMORY
                )
            code = child.wait()
            if code in (-signal.SIGINT, 130):
                raise KeyboardInterrupt
        except BaseException:
            _stop_training(child)
            raise
    if code and oom:
        raise MemoryError
    if code:
        raise ValueError(
            f"mlx-lm exited with status {code}. Check the tuner output above and retry."
        )
    if (
        first_val_loss is not None
        and last_val_loss is not None
        and not last_val_loss < first_val_loss
    ):
        print(
            "Validation loss did not improve. Try more training data or a different --learning-rate.",
            flush=True,
        )


def _recover_output(out, model, num_layers, sidecar_stat):
    try:
        numbered = sorted(
            (path for path in out.iterdir() if _NUMBERED_ADAPTER.fullmatch(path.name)),
            key=lambda path: int(path.name.split("_", 1)[0]),
            reverse=True,
        )
        checkpoints = (out / "adapters.safetensors", *numbered)
        for checkpoint in checkpoints:
            try:
                _validate_resume(checkpoint, model, num_layers)
            except ValueError:
                continue
            retry = shlex.join(["--resume", str(checkpoint), "--out", f"{out}-retry"])
            print(
                f"Saved adapter weights are available. Retry with the same model, data "
                f"and training options, replacing --resume and --out with {retry}. "
                "Choose a fresh --out path if that directory exists.",
                file=sys.stderr,
            )
            return
        sidecar = out / ADAPTER_METADATA
        artifacts = (out / "adapter_config.json", *checkpoints, sidecar)
        for path in artifacts:
            try:
                file_stat = path.lstat()
                if not stat.S_ISREG(file_stat.st_mode):
                    continue
                if path == sidecar and (
                    sidecar_stat is None
                    or not os.path.samestat(file_stat, sidecar_stat)
                ):
                    continue
                path.unlink()
            except OSError:
                # Report any files that could not be removed along with user files.
                continue
        try:
            out.rmdir()
        except OSError:
            kept = ", ".join(repr(path.name) for path in sorted(out.iterdir()))
            if not kept:
                raise
            print(
                f"Kept {kept} in {out}. Retry with a fresh --out directory.",
                file=sys.stderr,
            )
    except FileNotFoundError:
        pass
    except OSError:
        print(
            f"Could not clean up {out}. Retry with a fresh --out directory.",
            file=sys.stderr,
        )


def finetune_command(args):
    started = time.monotonic()
    train = valid = 0
    has_valid_split = False
    created_out = None
    sidecar_stat = None
    status = "err"
    try:
        if (
            min(args.batch_size, args.max_seq_length) < 1
            or (args.iters is not None and args.iters < 1)
            or args.num_layers < -1
        ):
            raise ValueError(
                "Use positive --iters, --batch-size and --max-seq-length, and --num-layers >= -1."
            )
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError("Use a positive, finite --learning-rate.")
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
        try:
            validate_adapter_runtime(args.model)
        except ValueError as exc:
            raise ValueError(
                f"Cannot fine-tune {reference!r} for serving with LoRA. {exc}. "
                "Choose a text-only model that supports LoRA adapters."
            ) from exc
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        default = f"~/.rapid-mlx/adapters/{Path(reference).name}-{stamp}"
        out = Path(args.out or default).expanduser().absolute()
        if os.path.lexists(out):
            raise ValueError("use a new --out directory.")
        if args.resume:
            _validate_resume(
                Path(args.resume).expanduser(), args.model, args.num_layers
            )
        _check_checkpoint(args.model, args.model if local else reference)
        with tempfile.TemporaryDirectory(prefix="rapid-mlx-finetune-") as scratch:
            data, train, valid, has_valid_split = _prepare_data(
                Path(args.data).expanduser(), scratch, args.batch_size
            )
            if args.iters is None:
                # mlx-lm trains on full batches and drops the remainder each epoch.
                args.iters = min(
                    _MAX_DEFAULT_ITERS, _DEFAULT_EPOCHS * (train // args.batch_size)
                )
                print(
                    f"Default --iters={args.iters} targets {_DEFAULT_EPOCHS} epochs over full batches "
                    f"with a cap of {_MAX_DEFAULT_ITERS} iterations."
                )
            _print_plan(args, out, train, valid)
            model = _training_model(args.model, reference)
            if model != args.model:
                _check_checkpoint(model, reference)
                if args.resume:
                    _validate_resume(
                        Path(args.resume).expanduser(), model, args.num_layers
                    )
            serve_model = (
                reference if not local and resolve_revision(reference) else model
            )
            try:
                out.mkdir(mode=0o700, parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise ValueError("use a new --out directory.") from exc
            created_out = out
            _train(args, model, data, out)
            _validate_output(out, serve_model, args.num_layers)
            metadata = read_model_metadata(model)
            config = (metadata.config or {}) if metadata else {}
            quant = (
                config.get("quantization") or config.get("quantization_config") or {}
            )
            with tempfile.TemporaryDirectory(dir=out) as staging:
                sidecar = Path(staging) / ADAPTER_METADATA
                atomic_write_json(
                    sidecar,
                    {
                        "base_model": serve_model,
                        "quantization_bits": quant.get("bits"),
                        "adapter_type": "lora",
                        "num_layers": args.num_layers,
                        "rapid_mlx_version": __version__,
                        "mlx_lm_version": version("mlx-lm"),
                    },
                )
                # Record the inode before linking so cleanup also covers an interrupt
                # immediately after publication, while preserving a colliding file.
                sidecar_stat = sidecar.stat()
                os.link(sidecar, out / ADAPTER_METADATA)
        print(shlex.join(["rapid-mlx", "serve", serve_model, "--adapter", str(out)]))
        status = "ok"
        return 0
    except KeyboardInterrupt:
        status = "abort"
        print("Fine-tuning aborted.", file=sys.stderr)
        return 130
    except MemoryError:
        status = "oom"
        print(f"Fine-tuning ran out of memory. {_MEMORY_TIP}", file=sys.stderr)
        return 1
    except OSError as exc:
        reason = exc.strerror or "system operation failed"
        print(f"Fine-tuning error: {reason}.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Fine-tuning error: {exc}", file=sys.stderr)
        return 1
    finally:
        if created_out is not None and status != "ok":
            _recover_output(created_out, model, args.num_layers, sidecar_stat)
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
                "has_valid_split": has_valid_split,
            },
        )
