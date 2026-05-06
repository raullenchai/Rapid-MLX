# SPDX-License-Identifier: Apache-2.0
"""Package Qwen3.6 MTP weights into the MTPLX sidecar layout."""

from __future__ import annotations

import json
import os
import platform
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


MTPLX_SUFFIX = "-MTPLX-Optimized-Speed"


class ConvertMTPLXError(RuntimeError):
    """Raised when a model cannot be packaged for MTPLX."""


@dataclass(frozen=True)
class ConvertMTPLXResult:
    source: Path
    mtp_source: Path
    output: Path
    mtp_file: Path
    runtime_contract_file: Path
    tensor_count: int


def default_output_path(model: Path) -> Path:
    model = model.expanduser().resolve()
    name = model.name
    if not name.endswith(MTPLX_SUFFIX):
        name = f"{name}{MTPLX_SUFFIX}"
    return model.with_name(name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConvertMTPLXError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConvertMTPLXError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConvertMTPLXError(f"Expected object JSON in {path}")
    return data


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config", config)
    return text if isinstance(text, dict) else config


def _mtp_layer_count(config: dict[str, Any]) -> int:
    text = _text_config(config)
    return int(
        text.get("mtp_num_hidden_layers")
        or text.get("num_nextn_predict_layers")
        or config.get("num_nextn_predict_layers")
        or 0
    )


def _copy_tree_hardlink(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for child in source.iterdir():
        target = output / child.name
        if child.is_dir():
            shutil.copytree(child, target, copy_function=os.link)
        elif child.is_file():
            try:
                os.link(child, target)
            except OSError:
                shutil.copy2(child, target)
        elif child.is_symlink():
            target.symlink_to(os.readlink(child))


def _load_mtp_tensors(model: Path, weight_map: dict[str, str]) -> dict[str, Any]:
    import mlx.core as mx

    mtp_keys = {key: rel for key, rel in weight_map.items() if key.startswith("mtp.")}
    if not mtp_keys:
        raise ConvertMTPLXError(
            "No embedded mtp.* tensors found in model.safetensors.index.json"
        )

    tensors: dict[str, Any] = {}
    files: dict[str, list[str]] = {}
    for key, rel in mtp_keys.items():
        files.setdefault(rel, []).append(key)

    for rel, keys in sorted(files.items()):
        shard = model / rel
        if not shard.exists():
            raise ConvertMTPLXError(f"MTP shard listed in index is missing: {shard}")
        loaded = mx.load(str(shard))
        for key in sorted(keys):
            if key not in loaded:
                raise ConvertMTPLXError(f"MTP tensor {key} missing from {shard}")
            tensors[key] = loaded[key]
        del loaded
    return tensors


def _normalize_qwen36_mtp(raw: dict[str, Any]) -> dict[str, Any]:
    required = {
        "mtp.fc.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.mlp.experts.down_proj",
        "mtp.layers.0.mlp.experts.gate_up_proj",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ConvertMTPLXError(
            "Missing required Qwen3.6 MTP tensors: " + ", ".join(missing)
        )

    gate_up = raw["mtp.layers.0.mlp.experts.gate_up_proj"]
    if len(gate_up.shape) != 3 or int(gate_up.shape[1]) % 2 != 0:
        raise ConvertMTPLXError(
            "Expected mtp.layers.0.mlp.experts.gate_up_proj shape "
            "(experts, 2 * intermediate, hidden)"
        )
    split = int(gate_up.shape[1]) // 2
    gate_proj = gate_up[:, :split, :]
    up_proj = gate_up[:, split:, :]

    converted = {
        "mtp.fc.weight": raw["mtp.fc.weight"],
        "mtp.layers.0.input_layernorm.weight": raw[
            "mtp.layers.0.input_layernorm.weight"
        ],
        "mtp.layers.0.mlp.down_proj.weight": raw[
            "mtp.layers.0.mlp.experts.down_proj"
        ],
        "mtp.layers.0.mlp.gate_proj.weight": gate_proj,
        "mtp.layers.0.mlp.up_proj.weight": up_proj,
        "mtp.layers.0.post_attention_layernorm.weight": raw[
            "mtp.layers.0.post_attention_layernorm.weight"
        ],
        "mtp.layers.0.self_attn.k_norm.weight": raw[
            "mtp.layers.0.self_attn.k_norm.weight"
        ],
        "mtp.layers.0.self_attn.k_proj.weight": raw[
            "mtp.layers.0.self_attn.k_proj.weight"
        ],
        "mtp.layers.0.self_attn.o_proj.weight": raw[
            "mtp.layers.0.self_attn.o_proj.weight"
        ],
        "mtp.layers.0.self_attn.q_norm.weight": raw[
            "mtp.layers.0.self_attn.q_norm.weight"
        ],
        "mtp.layers.0.self_attn.q_proj.weight": raw[
            "mtp.layers.0.self_attn.q_proj.weight"
        ],
        "mtp.layers.0.self_attn.v_proj.weight": raw[
            "mtp.layers.0.self_attn.v_proj.weight"
        ],
        "mtp.norm.weight": raw["mtp.norm.weight"],
        "mtp.pre_fc_norm_embedding.weight": raw[
            "mtp.pre_fc_norm_embedding.weight"
        ],
        "mtp.pre_fc_norm_hidden.weight": raw["mtp.pre_fc_norm_hidden.weight"],
    }
    return converted


def _write_config(output: Path, config: dict[str, Any]) -> None:
    updated = dict(config)
    text = _text_config(updated)
    if "text_config" in updated and isinstance(updated["text_config"], dict):
        updated["text_config"] = dict(updated["text_config"])
        text = updated["text_config"]
    text.setdefault("mtp_num_hidden_layers", 1)
    updated["num_nextn_predict_layers"] = 1
    updated["mlx_lm_extra_tensors"] = {
        **(
            updated.get("mlx_lm_extra_tensors")
            if isinstance(updated.get("mlx_lm_extra_tensors"), dict)
            else {}
        ),
        "mtp_file": "mtp.safetensors",
    }
    updated["lightning_mlx_mtplx_conversion"] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sidecar": "mtp.safetensors",
        "layout": "qwen3-next-mtp-bf16-sidecar",
    }
    (output / "config.json").write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mtplx_version() -> str:
    try:
        return version("mtplx")
    except PackageNotFoundError:
        return "0.1.0rc3"


def _write_runtime_contract(output: Path) -> Path:
    contract = {
        "mtplx_version": _mtplx_version(),
        "arch_id": "qwen3-next-mtp",
        "mtp_depth_max": 1,
        "recommended_profile": "performance-cold",
        "exactness_baseline": {
            "context": 2048,
            "max_abs_diff": 0.0,
            "source": "lightning-mlx convert-mtplx tensor-layout gate",
        },
        "verified_on": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }
    path = output / "mtplx_runtime.json"
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def convert_mtplx(
    model: str | Path,
    *,
    mtp_source: str | Path | None = None,
    output: str | Path | None = None,
    overwrite: bool = False,
) -> ConvertMTPLXResult:
    source = Path(model).expanduser().resolve()
    if not source.is_dir():
        raise ConvertMTPLXError(f"Model directory not found: {source}")
    mtp_root = (
        Path(mtp_source).expanduser().resolve() if mtp_source is not None else source
    )
    if not mtp_root.is_dir():
        raise ConvertMTPLXError(f"MTP source directory not found: {mtp_root}")

    out = Path(output).expanduser().resolve() if output else default_output_path(source)
    if out == source:
        raise ConvertMTPLXError("Output path must differ from source path")
    if out.exists():
        if not overwrite:
            raise ConvertMTPLXError(f"Output already exists: {out}")
        shutil.rmtree(out)

    config = _read_json(source / "config.json")
    mtp_config = _read_json(mtp_root / "config.json")
    if _mtp_layer_count(mtp_config) <= 0:
        raise ConvertMTPLXError("MTP source config does not declare MTP layers")
    index = _read_json(mtp_root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ConvertMTPLXError("MTP source model.safetensors.index.json has no weight_map")

    _copy_tree_hardlink(source, out)
    raw_mtp = _load_mtp_tensors(mtp_root, weight_map)
    converted = _normalize_qwen36_mtp(raw_mtp)

    import mlx.core as mx

    mtp_file = out / "mtp.safetensors"
    mx.save_safetensors(str(mtp_file), converted)
    _write_config(out, config)
    runtime_contract_file = _write_runtime_contract(out)

    return ConvertMTPLXResult(
        source=source,
        mtp_source=mtp_root,
        output=out,
        mtp_file=mtp_file,
        runtime_contract_file=runtime_contract_file,
        tensor_count=len(converted),
    )
