#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract + quantize the HY3 (Tencent Hunyuan 3) native MTP sidecar.

Provenance / regeneration tool for ``mlx-community/Hy3-preview-MTP-4bit``.

The 4-bit MLX checkpoint ``mlx-community/Hy3-preview-4bit`` STRIPS the native
MTP head at convert time. Upstream ``tencent/Hy3-preview`` keeps it at
``model.layers.<N>.*`` where ``N == num_hidden_layers`` (a DeepSeek-V3-style
single next-n predict layer, K=1). This script finds which shard(s) hold that
layer via the safetensors index, downloads ONLY those, extracts the layer-N
tensors, remaps them to the ``_HY3MTPModule`` param tree (mirroring
``hy_v3.Model.sanitize``'s expert stacking + router-bias rename), quantizes to
match the base checkpoint (4-bit gs64 affine for Linear; 8-bit gs64 for
``mlp.router.gate``; FP for all norms + ``expert_bias``), and writes
``model-mtp.safetensors``.

Concat order (``eh_proj(cat([enorm(embed), hnorm(prev_hidden)]))`` — embedding
first) confirmed from vLLM ``deepseek_mtp.py`` and SGLang ``hunyuan`` nextn.
HY3 uses standard ``nn.RMSNorm`` (no +1 norm shift). Idiom reference:
``scripts/extract_mtp_weights.py`` + ``scripts/add_mtp_weights.py``.

Usage:
    python scripts/extract_hy3_mtp.py \
        --base-repo mlx-community/Hy3-preview-4bit \
        --upstream-repo tencent/Hy3-preview \
        --out ./hy3-mtp-sidecar/model-mtp.safetensors

Provenance: `--base-rev` / `--upstream-rev` pin immutable commit SHAs so
config, index, and shards all resolve from one reproducible snapshot. The
downloaded shards stay in the shared HF cache; reclaim that space with
`huggingface-cli delete-cache` (never unlink the shared blobs by hand).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# NOTE ON WHERE THE SIDECAR IS WRITTEN. Write ``model-mtp.safetensors`` to a
# DEDICATED directory, never into the base checkpoint snapshot dir: ``mlx_lm.load``
# globs ``model*.safetensors`` in the model dir and would try to load the
# sidecar's 44 keys into the base HYV3 model under strict=True (which fails). The
# runtime inject (:mod:`rapid_mlx.spec_decode.mtp.hy3_inject`) loads the sidecar
# separately, so it must live in its own repo / directory.
DEFAULT_BASE_REPO = "mlx-community/Hy3-preview-4bit"
DEFAULT_UPSTREAM_REPO = "tencent/Hy3-preview"
DEFAULT_OUT = "./hy3-mtp-sidecar/model-mtp.safetensors"

# Router-gate override bits: HY3 quantizes ``mlp.router.gate`` at 8-bit while
# the rest of the layer uses the base's default bits (both group_size from the
# base). Read the actual bit-widths from the base ``quantization`` block so a
# differently-quantized base produces a matching sidecar (codex BLOCKING #4).
ROUTER_GATE_BITS = 8


def _resolve_base_quant(cfg: dict) -> dict:
    """Read {bits, group_size} from the base checkpoint's ``quantization`` block.

    Rejects a base with no quantization metadata rather than silently emitting
    a 4-bit sidecar for it (codex BLOCKING #4).
    """
    q = cfg.get("quantization")
    if not isinstance(q, dict) or "bits" not in q or "group_size" not in q:
        raise ValueError(
            "base config.json has no top-level quantization {bits, group_size}; "
            "cannot derive matching sidecar quant. Pass a quantized MLX base "
            "(e.g. mlx-community/Hy3-preview-4bit)."
        )
    return {"bits": int(q["bits"]), "group_size": int(q["group_size"])}


def _base_config(base_repo: str, revision: str | None) -> dict:
    """Fetch (and cache) the base checkpoint's ``config.json`` at ``revision``."""
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(base_repo, "config.json", revision=revision)
    return json.loads(Path(cfg_path).read_text())


def _resolve_mtp_layer(cfg: dict, base_repo: str) -> int:
    """The native MTP head lives at ``model.layers.<num_hidden_layers>.*``."""
    n = int(cfg["num_hidden_layers"])
    logger.info(
        "Base %s: num_hidden_layers=%d -> MTP head at layer %d", base_repo, n, n
    )
    return n


def _resolve_num_experts(cfg: dict) -> int:
    for key in ("num_experts", "n_routed_experts", "moe_num_experts"):
        if cfg.get(key):
            return int(cfg[key])
    raise KeyError("could not find expert count in base config.json")


def _find_shards_for_layer(
    upstream_repo: str, layer: int, revision: str | None
) -> list[str]:
    """Return the shard filenames that hold ``model.layers.<layer>.*``."""
    from huggingface_hub import hf_hub_download

    idx_path = hf_hub_download(
        upstream_repo, "model.safetensors.index.json", revision=revision
    )
    weight_map = json.loads(Path(idx_path).read_text())["weight_map"]
    prefix = f"model.layers.{layer}."
    shards = sorted({fn for k, fn in weight_map.items() if k.startswith(prefix)})
    if not shards:
        raise RuntimeError(f"no shards hold {prefix}* in {upstream_repo} index")
    logger.info("Layer %d spans %d shard(s): %s", layer, len(shards), shards)
    return shards


def _quantize(w: mx.array, bits: int, gs: int):
    q_w, q_s, q_b = mx.quantize(w, group_size=gs, bits=bits)
    mx.eval(q_w, q_s, q_b)
    return q_w, q_s, q_b


def _resolve_commit_sha(repo: str, revision: str | None) -> str:
    """Pin ``revision`` (or HEAD when ``None``) to one immutable commit SHA.

    Resolving up front — before any file download — guarantees the config,
    index, and every shard come from the *same* commit, so a repo update mid-
    extraction can never mix files from different revisions into a published
    sidecar (codex R4 BLOCKING). A 40-char hex string is already immutable and
    returned as-is; anything else (a branch/tag, or ``None``=HEAD) is resolved
    via the Hub API.
    """
    if (
        revision
        and len(revision) == 40
        and all(c in "0123456789abcdef" for c in revision.lower())
    ):
        return revision
    from huggingface_hub import HfApi

    sha = HfApi().repo_info(repo, revision=revision).sha
    if not sha:
        raise RuntimeError(
            f"could not resolve a commit SHA for {repo}@{revision or 'HEAD'}"
        )
    logger.info("Resolved %s@%s -> %s", repo, revision or "HEAD", sha)
    return sha


def main() -> int:
    import argparse

    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--base-repo",
        default=DEFAULT_BASE_REPO,
        help="4-bit MLX base checkpoint (defines quant + layer count).",
    )
    ap.add_argument(
        "--upstream-repo",
        default=DEFAULT_UPSTREAM_REPO,
        help="Full-precision checkpoint that still carries the MTP head.",
    )
    ap.add_argument(
        "--out", default=DEFAULT_OUT, help="Output path for model-mtp.safetensors."
    )
    ap.add_argument(
        "--base-rev",
        default=None,
        help="Immutable commit SHA for --base-repo (recommended for "
        "reproducible provenance; defaults to repo HEAD).",
    )
    ap.add_argument(
        "--upstream-rev",
        default=None,
        help="Immutable commit SHA for --upstream-repo (defaults to repo HEAD).",
    )
    args = ap.parse_args()

    # Pin both repos to one immutable commit SHA BEFORE any download so config,
    # index, and shards all come from a single reproducible snapshot (codex R4
    # BLOCKING). --base-rev / --upstream-rev may be a SHA (used as-is), a
    # branch/tag, or omitted (=HEAD, resolved now).
    base_rev = _resolve_commit_sha(args.base_repo, args.base_rev)
    upstream_rev = _resolve_commit_sha(args.upstream_repo, args.upstream_rev)

    base_cfg = _base_config(args.base_repo, base_rev)
    mtp_layer = _resolve_mtp_layer(base_cfg, args.base_repo)
    num_experts = _resolve_num_experts(base_cfg)
    base_quant = _resolve_base_quant(base_cfg)
    q_bits, q_gs = base_quant["bits"], base_quant["group_size"]
    logger.info(
        "Base quant: %d-bit gs=%d (router.gate %d-bit)", q_bits, q_gs, ROUTER_GATE_BITS
    )
    prefix = f"model.layers.{mtp_layer}."
    shards = _find_shards_for_layer(args.upstream_repo, mtp_layer, upstream_rev)

    # --- 1. Download only the shard(s) holding the MTP layer. ---
    raw_paths: list[Path] = []
    all_weights: dict[str, mx.array] = {}
    for shard in shards:
        logger.info("Downloading %s ...", shard)
        p = Path(hf_hub_download(args.upstream_repo, shard, revision=upstream_rev))
        raw_paths.append(p)
        w = mx.load(str(p))
        for k, v in w.items():
            if k.startswith(prefix):
                all_weights[k] = v
        del w

    logger.info("Extracted %d layer-%d tensors", len(all_weights), mtp_layer)

    # --- 2. Remap to the _HY3MTPModule param tree. ---
    # Wrapper-level: enorm/hnorm/eh_proj/norm.
    # Inner DecoderLayer: layers.0.{input_layernorm,post_attention_layernorm,
    #   self_attn.*, mlp.router.gate, mlp.router.expert_bias, mlp.switch_mlp.*,
    #   mlp.shared_mlp.*}.
    remapped: dict[str, mx.array] = {}

    def put(key: str, val: mx.array) -> None:
        remapped[key] = val

    # Wrapper norms + projection + final norm.
    put("enorm.weight", all_weights.pop(prefix + "enorm.weight"))
    put("hnorm.weight", all_weights.pop(prefix + "hnorm.weight"))
    put("eh_proj.weight", all_weights.pop(prefix + "eh_proj.weight"))
    put("norm.weight", all_weights.pop(prefix + "final_layernorm.weight"))

    lp = "layers.0."
    # Layernorms + attention (incl q_norm/k_norm).
    put(
        lp + "input_layernorm.weight",
        all_weights.pop(prefix + "input_layernorm.weight"),
    )
    put(
        lp + "post_attention_layernorm.weight",
        all_weights.pop(prefix + "post_attention_layernorm.weight"),
    )
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        put(
            lp + f"self_attn.{name}.weight",
            all_weights.pop(prefix + f"self_attn.{name}.weight"),
        )
    for name in ("q_norm", "k_norm"):
        put(
            lp + f"self_attn.{name}.weight",
            all_weights.pop(prefix + f"self_attn.{name}.weight"),
        )

    # Router gate + expert bias (bias renamed to router.expert_bias to match
    # hy_v3.MoEGate, which holds ``expert_bias`` under the ``router`` module).
    put(
        lp + "mlp.router.gate.weight",
        all_weights.pop(prefix + "mlp.router.gate.weight"),
    )
    put(
        lp + "mlp.router.expert_bias",
        all_weights.pop(prefix + "mlp.expert_bias"),
    )

    # Shared expert MLP.
    for proj in ("gate_proj", "down_proj", "up_proj"):
        put(
            lp + f"mlp.shared_mlp.{proj}.weight",
            all_weights.pop(prefix + f"mlp.shared_mlp.{proj}.weight"),
        )

    # Stack the 192 experts per projection -> switch_mlp (mirrors
    # hy_v3.Model.sanitize expert-stacking with mx.stack).
    for proj in ("gate_proj", "down_proj", "up_proj"):
        expert_keys = [
            prefix + f"mlp.experts.{e}.{proj}.weight" for e in range(num_experts)
        ]
        missing = [k for k in expert_keys if k not in all_weights]
        if missing:
            logger.error(
                "Missing %d expert tensors for %s (first: %s)",
                len(missing),
                proj,
                missing[0],
            )
            return 1
        stacked = mx.stack([all_weights.pop(k) for k in expert_keys])
        mx.eval(stacked)
        put(lp + f"mlp.switch_mlp.{proj}.weight", stacked)
        logger.info(
            "Stacked %d experts for %s -> %s", num_experts, proj, tuple(stacked.shape)
        )

    if all_weights:
        # Schema-drift guard (codex NIT): unconsumed layer tensors mean the
        # upstream head grew params this extractor does not map, so a warning-
        # only path could publish a silently-incomplete sidecar. Fail hard
        # unless the operator explicitly acknowledges via
        # HY3_MTP_ALLOW_UNCONSUMED=1 (e.g. deliberately dropping a known-unused
        # tensor during a controlled re-extract).
        leftover = sorted(all_weights)
        if os.environ.get("HY3_MTP_ALLOW_UNCONSUMED") == "1":
            logger.warning(
                "UNCONSUMED layer-%d tensors ignored via HY3_MTP_ALLOW_UNCONSUMED "
                "(count=%d, first 8): %s",
                mtp_layer,
                len(leftover),
                leftover[:8],
            )
        else:
            logger.error(
                "UNCONSUMED layer-%d tensors (count=%d, first 8): %s. The upstream "
                "MTP head carries params this extractor does not map — refusing to "
                "publish a silently-incomplete sidecar. Update the remap, or set "
                "HY3_MTP_ALLOW_UNCONSUMED=1 to override deliberately.",
                mtp_layer,
                len(leftover),
                leftover[:8],
            )
            return 1

    # --- 3. Quantize to match the base checkpoint. ---
    # Norms (1-D) + expert_bias stay FP. eh_proj / attn proj / switch_mlp /
    # shared_mlp go 4-bit. router.gate goes 8-bit.
    def _is_norm(k: str) -> bool:
        return k.endswith("norm.weight") or k.endswith("layernorm.weight")

    quantized: dict[str, mx.array] = {}
    for k, v in remapped.items():
        if _is_norm(k) or k.endswith("expert_bias"):
            quantized[k] = v  # FP — never quantize 1-D tensors / norms.
            continue
        if not k.endswith(".weight"):
            quantized[k] = v
            continue
        bits = ROUTER_GATE_BITS if k.endswith("mlp.router.gate.weight") else q_bits
        q_w, q_s, q_b = _quantize(v, bits, q_gs)
        quantized[k] = q_w
        quantized[k.replace(".weight", ".scales")] = q_s
        quantized[k.replace(".weight", ".biases")] = q_b

    # --- 4. Save into a DEDICATED sidecar dir (see the module note above). ---
    # Atomic write (codex NIT): serialize to a temp sibling then os.replace onto
    # the destination, so an interrupt / disk-full mid-write can't truncate or
    # destroy a previously-valid sidecar at ``out``.
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    # mx.save_safetensors SILENTLY no-ops unless the path ends in .safetensors,
    # so the temp sibling must keep that suffix (…​.tmp.safetensors), not a bare
    # .tmp. A per-process pid tag (codex NIT) keeps concurrent extractions
    # targeting the same output from clobbering each other's temp file.
    # os.replace within the same dir is atomic.
    tmp_out = out.with_name(f"{out.stem}.tmp.{os.getpid()}{out.suffix}")
    logger.info("Saving %d tensors to %s (atomic via %s)", len(quantized), out, tmp_out)
    mx.save_safetensors(str(tmp_out), quantized)
    if not tmp_out.exists():
        logger.error(
            "mx.save_safetensors wrote nothing to %s (path must end in "
            ".safetensors); aborting before replace.",
            tmp_out,
        )
        return 1
    os.replace(tmp_out, out)

    # --- 5. Spot-checks. ---
    total_bytes = sum(v.nbytes for v in quantized.values())
    logger.info(
        "Sidecar size on disk: %.1f MB (%d tensors)", total_bytes / 1e6, len(quantized)
    )
    smw = quantized.get(lp + "mlp.switch_mlp.gate_proj.weight")
    smw_s = quantized.get(lp + "mlp.switch_mlp.gate_proj.scales")
    smw_b = quantized.get(lp + "mlp.switch_mlp.gate_proj.biases")
    logger.info(
        "spot-check switch_mlp.gate_proj: weight.shape=%s (lead dim=%s, want 192), "
        "has scales=%s biases=%s",
        tuple(smw.shape) if smw is not None else None,
        smw.shape[0] if smw is not None else None,
        smw_s is not None,
        smw_b is not None,
    )
    rg = quantized.get(lp + "mlp.router.gate.weight")
    rg_s = quantized.get(lp + "mlp.router.gate.scales")
    logger.info(
        "spot-check router.gate: weight.shape=%s scales.shape=%s (8-bit)",
        tuple(rg.shape) if rg is not None else None,
        tuple(rg_s.shape) if rg_s is not None else None,
    )

    # --- 6. Disk hygiene hint (codex BLOCKING #5). ---
    # NEVER unlink the content-addressed HF cache blobs directly: they are
    # shared by ref-count across snapshots, and removing one leaves dangling
    # symlinks in every OTHER snapshot that points at it. The downloaded
    # shards are large; point the operator at the safe, ref-count-aware
    # deletion path instead of corrupting the cache here.
    if raw_paths:
        logger.info(
            "Downloaded %d upstream shard(s) into the shared HF cache. To "
            "reclaim that space safely (ref-count aware), run:\n"
            "    huggingface-cli delete-cache\n"
            "and select the %s revision. (This script does not unlink shared "
            "cache blobs — doing so would dangle other snapshots' symlinks.)",
            len(raw_paths),
            args.upstream_repo,
        )

    logger.info("Done. Sidecar ready at %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
