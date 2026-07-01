#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert the Mia-AiLab Gemmable-4 MTP GGUF to a Rapid-MLX safetensors sidecar.

Sidecar source
--------------

* Repo: ``Mia-AiLab/Gemmable-4-12B-MTP-GGUF`` (~98k HF downloads at
  release-time)
* File (this converter): ``gemmable-4-12b-fp16-mtp.gguf`` (~0.80 GB,
  fp16 / BF16 weights, F32 for RMSNorms + RoPE freqs + layer scalars).
* Excluded on purpose: the K-quant variants (``Q4_K_M-mtp`` /
  ``Q6_K-mtp``) — memory ``project_glm52_abandoned`` (2026-06-30)
  established that K-quant requant paths on Gemma-family models fail
  coherence in Rapid-MLX. We stick to the bf16-linear file.
* Also usable but not the default: ``gemmable-4-12b-Q8_0-mtp.gguf``
  (~0.43 GB, straight linear q8). Pass ``--gguf-filename
  gemmable-4-12b-Q8_0-mtp.gguf`` to convert it instead. Q8_0 is a
  safe path (no per-super-block scales like K-quant); it stays in the
  supported set.

Architecture surprise (documented in the PR body)
-------------------------------------------------

The GGUF header reports ``general.architecture = 'gemma4-assistant'``,
not the "MTP head" architecture the Qwen3.5 sidecar uses. This is
llama.cpp's ``draft-mtp`` shape:

* Its own transformer stack — 4 decoder blocks at hidden_size=1024
  (Gemma 4 12B base is hidden_size=3840).
* Cross-model projections — ``nextn.pre_projection`` reduces
  ``concat(base_hidden, base_embed)`` at 7680 dims down to 1024;
  ``nextn.post_projection`` projects the assistant's 1024 output back
  up to base 3840.
* Missing K/V — the assistant's attention has only Q + O projections;
  it borrows K/V from the corresponding base-model layer
  (``shared_kv_layers = 4`` in the GGUF metadata).

This converter therefore emits the tensors under a namespace that
signals the architecture:

* ``mtp.embed_tokens.weight``, ``mtp.norm.weight`` — assistant's own
  embed + final norm.
* ``mtp.pre_projection.weight``, ``mtp.post_projection.weight`` — the
  ``nextn.*`` projection layers, renamed for consistency with the
  Rapid-MLX "everything under ``mtp.*``" convention.
* ``mtp.layers.0..3.*`` — assistant decoder layers, with GGUF's
  ``blk.N.*`` names rewritten to the mlx-lm ``gemma4_text``
  DecoderLayer surface (``self_attn.q_proj`` etc.).

Downstream: :func:`vllm_mlx.spec_decode.mtp.gemma4_inject.inject_mtp_support`
DOES NOT yet consume this sidecar — its ``_looks_like_assistant_sidecar``
guard REFUSES to inject when it fingerprints the pre/post projection
keys, and logs a specific "AssistantModel code path lands in follow-up
PR" message. The sidecar is produced here as the deliverable that
follow-up PR will consume.

Config
------

Mia-AiLab has NO ``config.json`` in the repo — the GGUF header carries
all the metadata. We stitch a config.json from:

1. The parent MLX checkpoint's config (``mlx-community/gemma-4-12b-it-4bit``
   or the operator-supplied ``--parent-repo``).
2. GGUF header fields (``gemma4-assistant.block_count``,
   ``embedding_length``, ``embedding_length_out``,
   ``nextn_predict_layers``, etc.).
3. ``mtp_num_hidden_layers = 1`` layered on top for detect-path parity
   with Qwen3.5.

Idempotency
-----------

* GGUF download uses ``hf_hub_download`` which caches by content hash
  — re-runs skip the download.
* Output safetensors + config.json are written atomically via ``.tmp``
  suffix then renamed, so a partial run leaves no half-file on disk.

Usage
-----

::

    python scripts/convert_gemma4_mtp_gguf.py \\
        --out-dir /path/to/staging/mlx-community-gemma-4-12b-mtp-fp16

By default, output goes under ``$HOME/rapid-mlx-staging/
gemma-4-12b-mtp-fp16``. Q4 quantization of the layer bodies is NOT
supported here — the AssistantModel class doesn't exist yet, so the
``nn.quantize`` walk has nothing to walk. Track the q4 conversion
work item on the follow-up "AssistantModel MTP" PR.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger("convert_gemma4_mtp_gguf")


DEFAULT_REPO = "Mia-AiLab/Gemmable-4-12B-MTP-GGUF"
DEFAULT_FILENAME = "gemmable-4-12b-fp16-mtp.gguf"
DEFAULT_PARENT_REPO = "mlx-community/gemma-4-12b-it-4bit"
DEFAULT_OUT_DIR = "~/rapid-mlx-staging/gemma-4-12b-mtp-fp16"


# ---------------------------------------------------------------------------
# GGUF tensor decode
# ---------------------------------------------------------------------------


def _decode_tensor(t):
    """Convert one ``gguf.ReaderTensor`` to an ``mx.array``.

    Handles F32 (0), F16 (1), BF16 (30) — matching what the Mia-AiLab
    fp16-mtp GGUF ships. K-quant variants (Q4_K_M / Q6_K, tensor_type
    12+) are intentionally NOT supported here — the operator explicitly
    scoped this to fp16 / bf16 per ``project_glm52_abandoned``. If a
    caller points ``--gguf-filename`` at a Q8_0 file (linear q8, safe),
    tensor_type 8 is also handled.

    The GGUF shape convention has the rightmost dim as the fastest-
    changing (``ne[0]``). Our raw view is ``shape[::-1]`` — for a
    Linear.weight of shape ``(1024, 4096)`` in GGUF, the raw uint16
    view is ``(4096, 1024)`` = PyTorch ``(out=4096, in=1024)``. No
    transpose needed.
    """
    import mlx.core as mx
    import numpy as np

    shape_pt = tuple(int(x) for x in t.shape[::-1])
    raw = np.ascontiguousarray(t.data)
    ttype = int(t.tensor_type)
    if ttype == 0:  # F32
        arr = raw.view(np.float32).reshape(shape_pt)
        return mx.array(arr)
    if ttype == 1:  # F16
        arr = raw.view(np.float16).reshape(shape_pt)
        return mx.array(arr)
    if ttype == 30:  # BF16 — no numpy dtype; shift into fp32 for mlx.array.
        u16 = raw.view(np.uint16).reshape(shape_pt)
        fp32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return mx.array(fp32).astype(mx.bfloat16)
    if ttype == 8:  # Q8_0 — linear 8-bit, safe to decode via gguf helpers.
        # gguf's helper interprets Q8_0 blocks. Fall back to a straight
        # dequantize via the gguf.dequantize module.
        try:
            from gguf import dequantize as _dq

            arr = _dq.dequantize(raw, t.tensor_type).reshape(shape_pt)
            return mx.array(arr)
        except Exception as exc:  # pragma: no cover — best-effort Q8_0 path
            raise NotImplementedError(
                f"Q8_0 decode failed for tensor {t.name!r}: {exc}. Use the "
                "fp16-mtp variant instead."
            ) from exc

    raise NotImplementedError(
        f"Unsupported GGUF tensor_type={ttype} for {t.name!r}. This converter "
        f"only handles F32/F16/BF16 and Q8_0 — K-quant variants are excluded "
        "per project_glm52_abandoned."
    )


# ---------------------------------------------------------------------------
# Tensor-name mapping (GGUF → Rapid-MLX / mlx-lm gemma4_text)
# ---------------------------------------------------------------------------


class _UnmappedTensorError(Exception):
    """Raised when a GGUF tensor name is neither mapped nor on the drop-list.

    Codex round-4 blocking fix: silently returning ``None`` for
    unrecognized names meant a GGUF schema drift (new field added by
    an upstream converter, typo in a block name) would produce a
    sidecar missing required weights while the converter exits with
    success. This exception surfaces at the caller, which then aborts
    the run with a non-zero exit — the operator sees the missing
    mapping explicitly.
    """


# Names we intentionally drop from the sidecar (MLX computes them
# dynamically from ``rope_theta`` — no on-disk state needed).
_INTENTIONAL_DROP: frozenset[str] = frozenset({"rope_freqs.weight"})


def _map_tensor_name(name: str) -> str | None:
    """Rewrite a GGUF tensor name to the Rapid-MLX MTP sidecar name.

    Returns:
        * The remapped name for known tensors.
        * ``None`` ONLY when the name is on the explicit
          :data:`_INTENTIONAL_DROP` list.

    Raises:
        _UnmappedTensorError: for any tensor that is neither in the map
            nor on the drop-list. The caller aborts on this exception —
            silent drops of unknown tensors are a schema-drift trap
            that the earlier version of this converter was exposed to.
    """
    if name in _INTENTIONAL_DROP:
        return None

    if name == "token_embd.weight":
        return "mtp.embed_tokens.weight"
    if name == "output_norm.weight":
        return "mtp.norm.weight"

    if name == "nextn.pre_projection.weight":
        return "mtp.pre_projection.weight"
    if name == "nextn.post_projection.weight":
        return "mtp.post_projection.weight"

    # blk.N.* → mtp.layers.N.*
    if name.startswith("blk."):
        _, _, tail = name.partition(".")  # "blk"
        blk_idx, _, sub = tail.partition(".")  # "0", "attn_q.weight"
        block_prefix = f"mtp.layers.{blk_idx}"
        # Per-block name map.
        _blk_map = {
            "attn_norm.weight": f"{block_prefix}.input_layernorm.weight",
            "post_attention_norm.weight": (
                f"{block_prefix}.post_attention_layernorm.weight"
            ),
            "ffn_norm.weight": f"{block_prefix}.pre_feedforward_layernorm.weight",
            "post_ffw_norm.weight": (
                f"{block_prefix}.post_feedforward_layernorm.weight"
            ),
            "attn_q.weight": f"{block_prefix}.self_attn.q_proj.weight",
            "attn_q_norm.weight": f"{block_prefix}.self_attn.q_norm.weight",
            "attn_output.weight": f"{block_prefix}.self_attn.o_proj.weight",
            "ffn_gate.weight": f"{block_prefix}.mlp.gate_proj.weight",
            "ffn_up.weight": f"{block_prefix}.mlp.up_proj.weight",
            "ffn_down.weight": f"{block_prefix}.mlp.down_proj.weight",
            "layer_output_scale.weight": f"{block_prefix}.layer_scalar",
        }
        mapped = _blk_map.get(sub)
        if mapped is None:
            raise _UnmappedTensorError(
                f"blk.* tensor {name!r} has no mapping. This is either a "
                "GGUF schema drift (upstream added a new field) or a typo "
                "in the converter — investigate before proceeding to avoid "
                "shipping a partial sidecar."
            )
        return mapped

    raise _UnmappedTensorError(
        f"top-level tensor {name!r} has no mapping. Update _map_tensor_name "
        "or _INTENTIONAL_DROP explicitly rather than silently dropping."
    )


# ---------------------------------------------------------------------------
# GGUF metadata → config.json
# ---------------------------------------------------------------------------


def _read_field_scalar(reader, key: str):
    """Read a scalar / string metadata field. Returns None if absent."""
    import gguf

    f = reader.fields.get(key)
    if f is None:
        return None
    # String field
    if f.types and f.types[0] == gguf.GGUFValueType.STRING:
        if len(f.parts) >= 2:
            return bytes(f.parts[-1]).decode("utf-8", errors="replace")
        return None
    # Numeric scalar
    if f.parts:
        v = f.parts[-1]
        if v.size == 1:
            return v[0].item() if hasattr(v[0], "item") else v[0]
        return v.tolist()
    return None


def _build_sidecar_config(reader, parent_config: dict, num_mtp_layers: int) -> dict:
    """Merge GGUF metadata with the parent Gemma 4 config.

    The parent config is the base Gemma-4 checkpoint's config.json
    (its ``text_config`` block is what the inject reads). We layer
    ``mtp_num_hidden_layers`` on top plus the assistant-architecture
    hints from the GGUF header.
    """
    cfg = dict(parent_config)

    # Ensure ``model_type`` is preserved from the parent (this file
    # ships to the parent's alias — the sidecar itself doesn't have a
    # standalone model_type in mlx-lm sense).
    cfg.setdefault("model_type", parent_config.get("model_type", "gemma4_unified"))

    # Layer in mtp_num_hidden_layers at the top level AND under
    # text_config (both surfaces the inject checks).
    cfg["mtp_num_hidden_layers"] = int(num_mtp_layers)
    text_cfg = cfg.get("text_config")
    if isinstance(text_cfg, dict):
        text_cfg = dict(text_cfg)
        text_cfg["mtp_num_hidden_layers"] = int(num_mtp_layers)
        cfg["text_config"] = text_cfg

    # Assistant-model metadata block — signals the follow-up code path
    # will consume this sidecar. Keys mirror the GGUF header naming.
    cfg["mtp_assistant"] = {
        "architecture": "gemma4-assistant",
        "block_count": _read_field_scalar(reader, "gemma4-assistant.block_count"),
        "hidden_size": _read_field_scalar(reader, "gemma4-assistant.embedding_length"),
        "hidden_size_out": _read_field_scalar(
            reader, "gemma4-assistant.embedding_length_out"
        ),
        "num_attention_heads": _read_field_scalar(
            reader, "gemma4-assistant.attention.head_count"
        ),
        "num_key_value_heads": _read_field_scalar(
            reader, "gemma4-assistant.attention.head_count_kv"
        ),
        "head_dim": _read_field_scalar(reader, "gemma4-assistant.attention.key_length"),
        "head_dim_swa": _read_field_scalar(
            reader, "gemma4-assistant.attention.key_length_swa"
        ),
        "shared_kv_layers": _read_field_scalar(
            reader, "gemma4-assistant.attention.shared_kv_layers"
        ),
        "sliding_window": _read_field_scalar(
            reader, "gemma4-assistant.attention.sliding_window"
        ),
        "sliding_window_pattern": _read_field_scalar(
            reader, "gemma4-assistant.attention.sliding_window_pattern"
        ),
        "rms_norm_eps": _read_field_scalar(
            reader, "gemma4-assistant.attention.layer_norm_rms_epsilon"
        ),
        "feed_forward_length": _read_field_scalar(
            reader, "gemma4-assistant.feed_forward_length"
        ),
        "nextn_predict_layers": _read_field_scalar(
            reader, "gemma4-assistant.nextn_predict_layers"
        ),
        "rope_theta_global": _read_field_scalar(
            reader, "gemma4-assistant.rope.freq_base"
        ),
        "rope_theta_swa": _read_field_scalar(
            reader, "gemma4-assistant.rope.freq_base_swa"
        ),
        "source_repo": DEFAULT_REPO,
        "source_file": _read_field_scalar(reader, "general.name") or "unknown",
        # Bookkeeping: mark this as an assistant sidecar so the inject's
        # `_looks_like_assistant_sidecar` guard has TWO signals (tensor
        # keys AND config marker).
        "_conversion_notes": (
            "Converted from Mia-AiLab GGUF via "
            "scripts/convert_gemma4_mtp_gguf.py. The tensor namespace uses "
            "the 'mtp.*' prefix but the underlying architecture is llama.cpp "
            "draft-mtp (assistant model with pre/post projections), not the "
            "Qwen3.5-style MTP head. The Rapid-MLX inject at "
            "vllm_mlx.spec_decode.mtp.gemma4_inject deliberately refuses to "
            "consume this file until the AssistantModel code path lands "
            "(follow-up PR after PR-3)."
        ),
    }
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_parent_config(repo: str) -> dict:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo_id=repo, filename="config.json")
    with open(p) as f:
        return json.load(f)


def _atomic_write_bytes(dst: Path, contents: bytes) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_bytes(contents)
    os.replace(tmp, dst)


def _atomic_move(tmp: Path, dst: Path) -> None:
    os.replace(tmp, dst)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo", default=DEFAULT_REPO, help="Source HF repo id")
    p.add_argument(
        "--gguf-filename",
        default=DEFAULT_FILENAME,
        help="GGUF filename in the source repo (default: fp16-mtp)",
    )
    p.add_argument(
        "--parent-repo",
        default=DEFAULT_PARENT_REPO,
        help="Parent Gemma 4 MLX checkpoint whose config.json to stitch",
    )
    p.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Local staging path for the MLX sidecar. Do NOT point this at "
        "an HF cache location — the operator uploads AFTER manual review.",
    )
    p.add_argument(
        "--mtp-num-hidden-layers",
        type=int,
        default=1,
        help="Value to stitch into config.json (matches Qwen3.5 sidecar convention)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output directory (default: refuse if non-empty).",
    )
    args = p.parse_args(argv)

    out_dir = Path(os.path.expanduser(args.out_dir)).resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        logger.error("Output dir %s is non-empty. Pass --force to overwrite.", out_dir)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    # Codex round-4 nit: --force previously overwrote only the two
    # target filenames but left any stale sibling files behind. Clear
    # KNOWN generated artifacts explicitly (not the whole dir — that
    # would nuke unrelated operator files if they pointed --out-dir at
    # a shared location). The safetensors + config.json + the .tmp
    # scratch file are the full artifact set this converter emits.
    if args.force:
        for known_artifact in (
            "model-mtp.safetensors",
            "model-mtp.tmp.safetensors",
            "config.json",
            # ``_atomic_write_bytes`` writes to ``<dst>.tmp`` before
            # renaming; include the config.json scratch file too
            # (codex round-5 NIT: was previously left behind).
            "config.json.tmp",
        ):
            with contextlib.suppress(FileNotFoundError):
                (out_dir / known_artifact).unlink()

    # --- Download GGUF ---
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:  # pragma: no cover
        logger.error("huggingface_hub not installed; cannot download GGUF.")
        return 3
    logger.info(
        "Downloading %s from %s (cached if present)...", args.gguf_filename, args.repo
    )
    gguf_path = hf_hub_download(repo_id=args.repo, filename=args.gguf_filename)
    logger.info("GGUF cached at: %s", gguf_path)

    # --- Read tensors ---
    try:
        import gguf as _gguf
    except ImportError:  # pragma: no cover
        logger.error("gguf package not installed. `pip install gguf`.")
        return 4
    reader = _gguf.GGUFReader(gguf_path)
    logger.info(
        "GGUF: %d tensors, %d metadata fields", len(reader.tensors), len(reader.fields)
    )

    architecture = _read_field_scalar(reader, "general.architecture")
    if architecture != "gemma4-assistant":
        logger.warning(
            "GGUF general.architecture=%r; expected 'gemma4-assistant'. "
            "Continuing anyway — the sidecar-writer emits by tensor-name "
            "mapping and doesn't hard-fail on architecture mismatch, but "
            "the resulting sidecar may not match any known inject path.",
            architecture,
        )

    mtp_weights: dict[str, object] = {}
    dropped: list[str] = []
    for t in reader.tensors:
        try:
            mapped = _map_tensor_name(t.name)
        except _UnmappedTensorError as exc:
            logger.error(
                "Refusing to write a partial sidecar: %s",
                exc,
            )
            return 6
        if mapped is None:
            dropped.append(t.name)
            continue
        with contextlib.suppress(Exception):
            logger.debug("  %s -> %s (dtype=%d)", t.name, mapped, int(t.tensor_type))
        arr = _decode_tensor(t)
        mtp_weights[mapped] = arr

    logger.info(
        "Mapped %d tensor(s); dropped %d (%s)",
        len(mtp_weights),
        len(dropped),
        ", ".join(dropped) if dropped else "none",
    )

    # --- Load parent config + stitch ---
    logger.info("Fetching parent config.json from %s...", args.parent_repo)
    try:
        parent_cfg = _load_parent_config(args.parent_repo)
    except Exception as exc:
        logger.warning(
            "Could not fetch parent config from %s: %s. Using a minimal "
            "gemma4_unified stub.",
            args.parent_repo,
            exc,
        )
        parent_cfg = {
            "model_type": "gemma4_unified",
            "text_config": {"model_type": "gemma4_unified_text"},
        }
    sidecar_cfg = _build_sidecar_config(reader, parent_cfg, args.mtp_num_hidden_layers)

    # --- Write safetensors atomically ---
    import mlx.core as mx

    st_dst = out_dir / "model-mtp.safetensors"
    # ``mx.save_safetensors`` auto-appends ``.safetensors`` when the
    # given path doesn't already end in that extension. Sidestep by
    # writing directly to a ``.tmp.safetensors`` filename, then
    # renaming to the final destination in an atomic step.
    st_tmp_write = out_dir / "model-mtp.tmp.safetensors"
    logger.info("Writing %d tensors to %s ...", len(mtp_weights), st_dst)
    # Codex round-3 nit: guard against a failed os.replace leaving the
    # ``.tmp.safetensors`` file behind. try/finally always unlinks the
    # temp path (a no-op after successful rename).
    try:
        mx.save_safetensors(str(st_tmp_write), mtp_weights)
        _atomic_move(st_tmp_write, st_dst)
    finally:
        with contextlib.suppress(FileNotFoundError):
            st_tmp_write.unlink()

    # --- Write config.json atomically ---
    cfg_dst = out_dir / "config.json"
    _atomic_write_bytes(cfg_dst, json.dumps(sidecar_cfg, indent=2).encode() + b"\n")
    logger.info("Wrote %s", cfg_dst)

    # --- Verify by re-loading ---
    # Codex round-5 NIT: check shape + dtype per tensor, not just the
    # key set. A corrupt shape / dtype mapping would otherwise pass
    # verification and produce a sidecar the follow-up consumer
    # cannot load.
    try:
        reloaded = mx.load(str(st_dst))
        missing_after_load = set(mtp_weights.keys()) - set(reloaded.keys())
        extra_after_load = set(reloaded.keys()) - set(mtp_weights.keys())
        assert not missing_after_load and not extra_after_load, (
            f"Re-load key set mismatch: missing={sorted(missing_after_load)[:5]}, "
            f"extra={sorted(extra_after_load)[:5]}"
        )
        for k, original in mtp_weights.items():
            reloaded_arr = reloaded[k]
            if tuple(reloaded_arr.shape) != tuple(original.shape):
                raise AssertionError(
                    f"{k}: shape drift on re-load "
                    f"(wrote {tuple(original.shape)}, read {tuple(reloaded_arr.shape)})"
                )
            if reloaded_arr.dtype != original.dtype:
                raise AssertionError(
                    f"{k}: dtype drift on re-load "
                    f"(wrote {original.dtype}, read {reloaded_arr.dtype})"
                )
        logger.info(
            "Verify: re-load matches (%d keys, shapes + dtypes checked). "
            "Sidecar is loadable via mx.load.",
            len(reloaded),
        )
    except Exception as exc:
        logger.error("Verify failed — re-load raised %s. Sidecar may be corrupt.", exc)
        return 5

    logger.info(
        "\n=== Conversion OK ===\n"
        "Sidecar dir:  %s\n"
        "  model-mtp.safetensors: %d MB\n"
        "  config.json:           %d bytes\n"
        "  tensor keys:           %d\n"
        "Next: this file is REFUSED by the current gemma4_inject due to the "
        "gemma4-assistant architecture guard. The follow-up 'AssistantModel' "
        "PR will wire the consumer.\n",
        out_dir,
        st_dst.stat().st_size >> 20,
        cfg_dst.stat().st_size,
        len(mtp_weights),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Silence unused-import warnings when this module is imported for its
# helpers rather than executed.
_ = shutil  # for future --clean re-runs.
