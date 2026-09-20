"""Central framework for extracting native MTP tensors into a standalone drafter.

One ``MTPSplitter`` base owns the shared mechanics (shard discovery, selective
load, config assembly, tokenizer copy). A family customizes by subclassing and
overriding the small hooks that vary: ``select_keys`` (which tensors are MTP),
``rename`` / ``on_mlx_source``, ``sanitize_ctx``, ``postprocess``, and
``quantization``. Register each splitter by its base ``model_type`` in
``MTP_SPLITTERS`` (lazy import paths) so ``convert`` and ``split_mtp`` can
dispatch on a source checkpoint.
"""

import glob
import importlib
import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mlx.core as mx
from safetensors import safe_open

from ...fp8 import transform_fp8_weights

# Documented pinned redirects: quant_utils/utils live at the mlx_vlm root
# and are vendored by later slices (quant_utils exists in this package;
# utils is step-3e scope).
from mlx_vlm.utils import get_model_path

from ...quant_utils import get_quantization_params


def _safetensor_files(model_path: Path) -> List[Path]:
    return [
        Path(path)
        for path in glob.glob(str(model_path / "*.safetensors"))
        if not path.endswith("consolidated.safetensors")
    ]


def _weight_map(model_path: Path) -> Dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    with open(index_path) as f:
        index = json.load(f)
    # Rapid upstream-bugfix (documented deviation): pinned 0.7.1 assumes
    # both the index document and ``weight_map`` are objects; a malformed
    # index crashed with ``AttributeError`` instead of a clear error.
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        raise ValueError(
            f"malformed safetensors index {index_path.name}: "
            "weight_map must be an object"
        )
    for filename in weight_map.values():
        if not isinstance(filename, str):
            raise ValueError(
                f"malformed safetensors index {index_path.name}: "
                f"non-string filename entry {filename!r}"
            )
    return weight_map


def _is_mlx_safetensors(file: Path) -> bool:
    with safe_open(file, framework="mlx") as f:
        return (f.metadata() or {}).get("format") == "mlx"


class MTPSplitter:
    # --- declarative per-family config (override in subclass) ---
    output_model_type: str = ""
    draft_model_cls = None  # class with a ``sanitize(self, weights)`` staticlike method
    require_text_config: bool = (
        True  # True: use text_config only; False: fall back to root
    )
    tie_word_embeddings_default: bool = False
    depth_field: str = "num_nextn_predict_layers"
    block_size_extra: int = 1  # block_size default = depth + block_size_extra
    supports_mlx_source: bool = False  # re-split of an already-MLX drafter
    tokenizer_files: Tuple[str, ...] = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )

    # --- hooks (override the small differences) ---
    def read_text_config(self, source_config: dict) -> dict:
        text_config = dict(
            source_config.get("text_config")
            or ({} if self.require_text_config else source_config)
        )
        if self.require_text_config and not text_config:
            raise ValueError("source config does not contain a text_config.")
        return text_config

    def select_keys(self, key: str, text_config: dict) -> bool:
        raise NotImplementedError

    def load_shard(self, file: Path, keys: List[str]) -> Dict[str, mx.array]:
        try:
            with safe_open(file, framework="mlx") as f:
                return {key: mx.array(f.get_tensor(key)) for key in keys}
        except (AttributeError, RuntimeError, TypeError):
            shard = mx.load(str(file))
            return {key: shard[key] for key in keys}

    def rename(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        return tensors

    def on_mlx_source(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        return tensors

    def sanitize_ctx(self, text_config: dict):
        return None

    def run_sanitize(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        if self.draft_model_cls is None:
            return tensors
        return self.draft_model_cls.sanitize(self.sanitize_ctx(text_config), tensors)

    def postprocess(self, tensors: Dict[str, mx.array], text_config: dict) -> None:
        pass

    def quantization(
        self,
        tensors: Dict[str, mx.array],
        source_config: dict,
        text_config: dict,
        quant_opts: dict,
    ) -> Optional[dict]:
        # (a) source already carries quantized MTP tensors -> record their config
        existing = self.quantization_from_source(tensors, source_config)
        if existing is not None:
            return existing
        # (b) fp drafter + a quantization was requested (e.g. convert --mtp --bits)
        q_bits = quant_opts.get("q_bits")
        q_mode = quant_opts.get("q_mode")
        if q_bits is None and q_mode is None:
            return None
        return self._quantize(
            tensors,
            quant_opts.get("q_group_size"),
            q_bits,
            q_mode or "affine",
        )

    def quantization_from_source(
        self, tensors: Dict[str, mx.array], source_config: dict
    ) -> Optional[dict]:
        return None

    def should_quantize_key(self, key: str) -> bool:
        # skip the router gate (kept full precision for routing stability); norms
        # and the fp32 correction bias fall out via the ndim/divisibility checks
        return key.endswith(".weight") and not key.endswith("mlp.gate.weight")

    def _quantize(
        self,
        weights: Dict[str, mx.array],
        group_size: Optional[int],
        bits: Optional[int],
        mode: str,
    ) -> Optional[dict]:
        quantization = get_quantization_params(group_size, bits, mode)
        group_size = quantization["group_size"]
        quantized_any = False
        for key in list(weights):
            if not self.should_quantize_key(key):
                continue
            weight = weights[key]
            if weight.ndim < 2 or weight.shape[-1] % group_size != 0:
                continue
            quantized = mx.quantize(weight, **quantization)
            weights[key] = quantized[0]
            weights[key[: -len(".weight")] + ".scales"] = quantized[1]
            if len(quantized) == 3:
                weights[key[: -len(".weight")] + ".biases"] = quantized[2]
            quantized_any = True
        if not quantized_any:
            return None
        return quantization

    def depth(self, text_config: dict) -> int:
        return int(text_config.get(self.depth_field, 1) or 1)

    def extra_config(self, text_config: dict) -> dict:
        return {}

    # --- shared orchestration (do not override) ---
    def iter_selected(
        self, source_path: Path, text_config: dict
    ) -> Iterable[Tuple[Path, List[str]]]:
        weight_map = _weight_map(source_path)
        if weight_map:
            by_file: Dict[str, List[str]] = {}
            for key, filename in weight_map.items():
                if self.select_keys(key, text_config):
                    by_file.setdefault(filename, []).append(key)
            if by_file:
                # Rapid upstream-bugfix (documented deviation): shard
                # filenames come from an untrusted safetensors index.
                # Absolute paths and '..' traversal are rejected lexically;
                # symlinks are followed but the resolved target must stay
                # inside the model directory or the repository's own HF
                # blob cache (snapshot shards symlink into ../blobs).
                resolved_source = source_path.resolve()
                allowed_roots = [resolved_source]
                blobs_root = resolved_source.parent.parent / "blobs"
                if resolved_source.parent.name == "snapshots" and blobs_root.is_dir():
                    allowed_roots.append(blobs_root.resolve())
                for filename, keys in by_file.items():
                    shard = Path(filename)
                    if shard.is_absolute() or ".." in shard.parts:
                        raise ValueError(
                            "safetensors index entry escapes the model "
                            f"directory: {filename!r}"
                        )
                    resolved_shard = (source_path / shard).resolve()
                    if not any(
                        resolved_shard.is_relative_to(root) for root in allowed_roots
                    ):
                        raise ValueError(
                            "safetensors index entry escapes the model "
                            f"directory: {filename!r}"
                        )
                    yield resolved_shard, keys
                return

        for file in _safetensor_files(source_path):
            with safe_open(file, framework="mlx") as f:
                keys = [key for key in f.keys() if self.select_keys(key, text_config)]
            if keys:
                yield file, keys

    def transform(
        self, tensors: Dict[str, mx.array], text_config: dict, source_is_mlx: bool
    ) -> Dict[str, mx.array]:
        tensors = self.rename(tensors, text_config)
        if source_is_mlx and self.supports_mlx_source:
            return self.on_mlx_source(tensors, text_config)
        tensors = self.run_sanitize(tensors, text_config)
        self.postprocess(tensors, text_config)
        return tensors

    def split(
        self,
        source: str,
        output: str,
        *,
        revision: Optional[str] = None,
        block_size: Optional[int] = None,
        force_download: bool = False,
        **quant_opts,
    ) -> Path:
        source_path = get_model_path(
            source, revision=revision, force_download=force_download
        )
        output_path = Path(output)

        with open(source_path / "config.json") as f:
            source_config = json.load(f)
        text_config = self.read_text_config(source_config)

        # Rapid upstream-bugfix (documented deviation): validate every
        # configuration argument BEFORE creating or writing the output —
        # rejected input must not leave a partially generated directory.
        # Minimum supported block size is 2: with 1 the MTP drafting loops
        # feed an empty token list into ``mx.concatenate`` and crash.
        resolved_block_size = (
            self.depth(text_config) + self.block_size_extra
            if block_size is None
            else int(block_size)
        )
        if resolved_block_size < 2:
            raise ValueError(f"block_size must be >= 2, got {block_size!r}")
        if output_path.resolve() == source_path.resolve():
            raise ValueError("output must differ from the source checkpoint")

        # Rapid upstream-bugfix (documented deviation): pinned 0.7.1 writes
        # directly into the destination, so a pre-existing directory keeps
        # stale tokenizer files and a failure after the weight save leaves
        # new weights paired with an old config.json. Build the complete
        # checkpoint in a unique sibling staging directory (concurrent
        # splits must not share one), swap it in only after every save and
        # copy succeeds, and keep the old destination as a backup until the
        # staged checkpoint is installed.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output_path.name}.mtp-split-",
                dir=str(output_path.parent),
            )
        )
        try:
            selected: Dict[str, mx.array] = {}
            source_is_mlx = False
            for file, keys in self.iter_selected(source_path, text_config):
                if self.supports_mlx_source:
                    source_is_mlx = source_is_mlx or _is_mlx_safetensors(file)
                selected.update(self.load_shard(file, keys))
            if not selected:
                raise ValueError(f"No MTP tensors found in {source_path}.")

            q_bits = quant_opts.get("q_bits")
            q_mode = quant_opts.get("q_mode")
            quantize = q_bits is not None or q_mode is not None
            fp8_target_quantization = None
            if quantize:
                fp8_target_quantization = get_quantization_params(
                    quant_opts.get("q_group_size"), q_bits, q_mode or "affine"
                )
            selected, transformed_quantization = transform_fp8_weights(
                selected,
                source_config,
                target_quantization=fp8_target_quantization,
            )
            if transformed_quantization is not None:
                source_config = dict(source_config)
                source_config["quantization"] = transformed_quantization
                source_config["quantization_config"] = transformed_quantization
            weights = self.transform(selected, text_config, source_is_mlx)
            quantization = self.quantization(
                weights, source_config, text_config, quant_opts
            )

            mx.eval(list(weights.values()))
            mx.save_safetensors(
                str(staging / "model.safetensors"),
                weights,
                metadata={"format": "mlx"},
            )

            draft_config = {
                "model_type": self.output_model_type,
                "text_config": text_config,
                "block_size": resolved_block_size,
                "tie_word_embeddings": bool(
                    text_config.get("tie_word_embeddings", self.tie_word_embeddings_default)
                ),
            }
            draft_config.update(self.extra_config(text_config))
            if quantization is not None:
                draft_config["quantization"] = quantization
                draft_config["quantization_config"] = quantization

            with open(staging / "config.json", "w") as f:
                json.dump(dict(sorted(draft_config.items())), f, indent=2)

            # Rapid upstream-bugfix (documented deviation): tokenizer
            # sidecars are copied through symlinks, so an untrusted
            # checkpoint could copy an arbitrary readable host file into
            # the generated output; resolve each sidecar and require it
            # to stay inside the checkpoint directory or the
            # repository's own HF blob cache.
            resolved_source = source_path.resolve()
            allowed_roots = [resolved_source]
            blobs_root = resolved_source.parent.parent / "blobs"
            if resolved_source.parent.name == "snapshots" and blobs_root.is_dir():
                allowed_roots.append(blobs_root.resolve())
            for name in self.tokenizer_files:
                src = source_path / name
                if not src.exists():
                    continue
                resolved = src.resolve()
                if not any(resolved.is_relative_to(root) for root in allowed_roots):
                    raise ValueError(
                        f"tokenizer sidecar escapes the checkpoint "
                        f"directory: {name!r}"
                    )
                shutil.copy(resolved, staging / name)

            # Install under a per-destination advisory lock: concurrent
            # splits' destination moves must not interleave. The old
            # destination moves into a unique, exclusively-created backup
            # owned by this invocation; the staged checkpoint replaces it
            # and the backup is restored if the install rename fails —
            # the destination is never destroyed before its replacement
            # exists.
            import fcntl
            import stat as stat_module

            lock_path = output_path.parent / f".{output_path.name}.mtp-split-lock"
            # O_NOFOLLOW + regular-file/owner checks: the predictable lock
            # path must not become a symlink-following write primitive for
            # anyone who can write the output directory.
            lock_fd = os.open(
                lock_path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600
            )
            try:
                lock_stat = os.fstat(lock_fd)
                if not stat_module.S_ISREG(lock_stat.st_mode):
                    raise RuntimeError(
                        f"split lock {lock_path} is not a regular file"
                    )
                if lock_stat.st_uid != os.getuid():
                    raise RuntimeError(
                        f"split lock {lock_path} is not owned by the current user"
                    )
                lock_handle = os.fdopen(lock_fd, "w")
            except (OSError, RuntimeError):
                os.close(lock_fd)
                raise
            try:
                fcntl.flock(lock_handle, fcntl.LOCK_EX)
                self._install_staged(output_path, staging)
            finally:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)
                lock_handle.close()
            return output_path
        finally:
            if staging.is_dir() and not staging.is_symlink():
                shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def _install_staged(output_path: Path, staging: Path) -> None:
        backup = None
        if output_path.exists() or output_path.is_symlink():
            # A unique, nonexistent backup path: mkdtemp pre-creates a
            # directory, which os.replace refuses to overwrite with a
            # symlinked destination (IsADirectoryError).
            backup = output_path.parent / (
                f".{output_path.name}.mtp-split-bak-{uuid.uuid4().hex}"
            )
            os.replace(output_path, backup)
        try:
            os.replace(staging, output_path)
        except OSError:
            # is_symlink() covers broken symlinks, which exists() misses —
            # a moved-aside broken destination must still be restored.
            if backup is not None and (backup.exists() or backup.is_symlink()):
                os.replace(backup, output_path)
            raise
        if backup is not None:
            try:
                if backup.is_dir() and not backup.is_symlink():
                    shutil.rmtree(backup)
                else:
                    backup.unlink()
            except OSError:
                # The new destination is safely installed; surface the
                # retained duplicate instead of deleting silently.
                logging.getLogger(__name__).warning(
                    "failed to remove split backup %s; remove it manually",
                    backup,
                )


# base model_type -> "module_path:ClassName" (lazy so importing this module is cheap)
MTP_SPLITTERS: Dict[str, str] = {
    "qwen3_5": "rapid_mlx.models.mlx_vlm_vendored.speculative.drafters.qwen3_5_mtp.split:Qwen3_5MTPSplitter",
    "qwen3_5_moe": "rapid_mlx.models.mlx_vlm_vendored.speculative.drafters.qwen3_5_mtp.split:Qwen3_5MTPSplitter",
    "qwen3_next": "rapid_mlx.models.mlx_vlm_vendored.speculative.drafters.qwen3_5_mtp.split:Qwen3NextMTPSplitter",
    "qwen4_exp": "mlx_vlm.speculative.drafters.qwen4_exp_mtp.split:Qwen4ExpMTPSplitter",
    "qwen4_exp_text": "mlx_vlm.speculative.drafters.qwen4_exp_mtp.split:Qwen4ExpMTPSplitter",
    "deepseek_v4": "mlx_vlm.speculative.drafters.deepseek_v4_mtp.split:DeepseekV4MTPSplitter",
    "hy_v4": "mlx_vlm.speculative.drafters.hy_v4_mtp.split:HyV4MTPSplitter",
    "glm4_moe_lite": "mlx_vlm.speculative.drafters.glm4_moe_lite_mtp.split:Glm4MoeLiteMTPSplitter",
    "glm5_next": "rapid_mlx.models.mlx_vlm_vendored.speculative.drafters.glm5_next_mtp.split:Glm5NextMTPSplitter",
    "glm5_next_text": "rapid_mlx.models.mlx_vlm_vendored.speculative.drafters.glm5_next_mtp.split:Glm5NextMTPSplitter",
    "glm_moe_dsa": "mlx_vlm.speculative.drafters.glm_moe_dsa_mtp.split:GlmMoeDsaMTPSplitter",
    "inkling_mm_model": "mlx_vlm.speculative.drafters.inkling_mtp.split:InklingMTPSplitter",
}


def get_mtp_splitter(base_model_type: str) -> Optional[MTPSplitter]:
    target = MTP_SPLITTERS.get(base_model_type)
    if target is None:
        return None
    module_path, class_name = target.split(":")
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls()


def detect_mtp_splitter(model_path: Path) -> Optional[MTPSplitter]:
    """Return the splitter for a source checkpoint, or None.

    Chooses by base ``model_type`` in config.json, then confirms MTP tensors are
    actually present (config flags alone are unreliable -- some models declare
    MTP but ship no tensors, others ship tensors with no flag).
    """
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        source_config = json.load(f)
    text_config = source_config.get("text_config") or source_config
    model_types = {
        text_config.get("model_type"),
        source_config.get("model_type"),
    }
    if "deepseek_v4" in model_types and (
        text_config.get("dspark_target_layer_ids")
        or source_config.get("dspark_target_layer_ids")
    ):
        # Documented pinned redirect: the deepseek_v4_dspark family is
        # outside the served set (not vendored); detection must resolve
        # the pinned splitter module.
        from mlx_vlm.speculative.drafters.deepseek_v4_dspark.split import (
            DeepseekV4DsparkSplitter,
        )

        splitter = DeepseekV4DsparkSplitter()
        tc = splitter.read_text_config(source_config)
        for _ in splitter.iter_selected(model_path, tc):
            return splitter

    # Some checkpoints name the inner text stack separately from the
    # architecture (Apodex 1.1 uses text_config "qwen3_5_moe_text" under a
    # root "qwen3_5_moe"), so fall back to the root type before giving up.
    splitter = None
    for base_model_type in (
        text_config.get("model_type"),
        source_config.get("model_type"),
    ):
        if not base_model_type:
            continue
        splitter = get_mtp_splitter(base_model_type)
        if splitter is not None:
            break
    if splitter is None:
        return None
    tc = splitter.read_text_config(source_config)
    for _ in splitter.iter_selected(model_path, tc):
        return splitter
    return None
