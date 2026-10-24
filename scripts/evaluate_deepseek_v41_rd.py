#!/usr/bin/env python3
"""Cache teacher-forced logits, screen pruning masks, and compare full-vocabulary KL.

Collection uses the existing native mmap/Engram SSD path. Mask screening leaves
expert weights unchanged and does NOT qualify mixed-bit quantization quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.deepseek_v41_public_ruler import (
    PUBLIC_RULERS,
    compact_arrays,
    compact_hashes,
    compact_metrics,
    read_slice_manifest,
    retain_ruler_manifest,
    slice_tokens,
    validate_ruler_manifest,
    verify_input_hashes,
)
from scripts.deepseek_v41_rd_common import (
    bank_plan,
    checkpoint_hashes,
    file_hash,
)


def token_metrics(reference, candidate):
    """Stable KL(reference || candidate), in nats, and top-1 disagreements."""
    r, c = np.asarray(reference, np.float64), np.asarray(candidate, np.float64)
    if (
        r.ndim != 2
        or 0 in r.shape
        or r.shape != c.shape
        or not np.isfinite(r).all()
        or not np.isfinite(c).all()
    ):
        raise ValueError("logits must be finite matching [tokens, vocabulary] arrays")
    r = r - r.max(-1, keepdims=True)
    c = c - c.max(-1, keepdims=True)
    r = r - np.log(np.exp(r).sum(-1, keepdims=True))
    c = c - np.log(np.exp(c).sum(-1, keepdims=True))
    kl = np.maximum(0.0, (np.exp(r) * (r - c)).sum(-1))
    return kl, np.argmax(reference, -1) != np.argmax(candidate, -1)


def validate_allocation(allocation, layers, experts, topk):
    if set(allocation["layers"]) != {str(layer_id) for layer_id in range(layers)}:
        raise ValueError("allocation layer coverage mismatch")
    for entry in allocation["layers"].values():
        ids = entry["kept_expert_ids"]
        if (
            len(ids) < topk
            or ids != sorted(set(ids))
            or any(type(e) is not int or not 0 <= e < experts for e in ids)
        ):
            raise ValueError("invalid kept expert IDs")
        if "bit_widths" in entry:
            bits = entry["bit_widths"]
            if len(bits) != experts or ids != bank_plan(bits, topk)["kept_expert_ids"]:
                raise ValueError("allocation bit widths and keep IDs disagree")
        elif set(entry.get("projection_bits", {})) != {
            "gate_proj",
            "up_proj",
            "down_proj",
        }:
            raise ValueError("allocation needs layer projection bits")


def install_native_masks(model, allocation, masses):
    import mlx.core as mx

    from rapid_mlx.models.deepseek_v41_native.moe import Gate

    class EvaluationGate(Gate):
        def __call__(self, x):
            weights, indices = super().__call__(x)
            retained = mx.take(self._rd_mask, indices)
            mass = mx.sum(weights * retained, axis=-1) / mx.maximum(
                mx.sum(weights, axis=-1), 1e-30
            )
            mx.eval(mass)
            masses[self._rd_layer].extend(np.array(mass).reshape(-1).tolist())
            original = self.bias
            try:
                self.bias = mx.where(self._rd_mask, original, -mx.inf)
                return super().__call__(x)
            finally:
                self.bias = original

    validate_allocation(
        allocation,
        len(model.layers),
        model.args.n_routed_experts,
        model.args.n_activated_experts,
    )
    for layer_id, layer in enumerate(model.layers):
        gate = layer.ffn.gate
        mask = np.zeros(model.args.n_routed_experts, bool)
        mask[allocation["layers"][str(layer_id)]["kept_expert_ids"]] = True
        gate._rd_mask = mx.array(mask)
        gate._rd_layer = layer_id
        gate.__class__ = EvaluationGate


def validate_checkpoint_identity(identity, index_json=None):
    if (
        not isinstance(identity, dict)
        or not {"config.json", "model.safetensors.index.json"} <= identity.keys()
        or any(
            not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None
            for v in identity.values()
        )
    ):
        raise ValueError(
            "invalid immutable checkpoint identity; regenerate legacy caches"
        )
    if (
        not isinstance(index_json, str)
        or hashlib.sha256(index_json.encode()).hexdigest()
        != identity["model.safetensors.index.json"]
    ):
        raise ValueError("checkpoint index document hash mismatch")
    index = json.loads(index_json)
    mapping = index.get("weight_map") if isinstance(index, dict) else None
    if (
        not isinstance(mapping, dict)
        or not mapping
        or any(
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            for path in mapping.values()
        )
    ):
        raise ValueError("invalid checkpoint weight map")
    if set(identity) != {
        "config.json",
        "model.safetensors.index.json",
        *mapping.values(),
    }:
        raise ValueError("checkpoint identity must cover every indexed weight shard")
    return identity


def source_identity_for_checkpoint(checkpoint, provenance_path=None, index_json=None):
    """A conversion record binds the exact source and candidate payload sets.

    The record is an operator-supplied conversion artifact, not proof of model
    quality. Its checkpoint_sha256 must match the files actually evaluated.
    """
    validate_checkpoint_identity(checkpoint, index_json)
    if provenance_path is None:
        return checkpoint, None, index_json
    raw = provenance_path.read_bytes()
    provenance = json.loads(raw)
    if (
        provenance.get("checkpoint_sha256") != checkpoint
        or provenance.get("checkpoint_index_json") != index_json
    ):
        raise ValueError("conversion provenance candidate checkpoint mismatch")
    return (
        validate_checkpoint_identity(
            provenance.get("source_checkpoint_sha256"),
            provenance.get("source_index_json"),
        ),
        hashlib.sha256(raw).hexdigest(),
        provenance["source_index_json"],
    )


def collect(a):
    import mlx.core as mx
    from tokenizers import Tokenizer

    from rapid_mlx.models.deepseek_v41_native.load import load

    if a.output.exists():
        raise FileExistsError("cache destination must not exist")
    if a.tokens < 1 or getattr(a, "prefill_chunk", 1) < 1:
        raise ValueError("tokens must be positive")
    ruler_digest = validate_ruler_manifest(a.slices)
    manifest_raw, manifest, slices = read_slice_manifest(a.slices)
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in slices):
        raise ValueError("slice names must be safe filenames")
    input_hashes = {
        a.slices: hashlib.sha256(manifest_raw).hexdigest(),
        a.model / "tokenizer.json": file_hash(a.model / "tokenizer.json"),
    }
    tokenizer = Tokenizer.from_file(str(a.model / "tokenizer.json"))
    expected_tokenizer = manifest.get("tokenizer_sha256")
    if (
        expected_tokenizer
        and expected_tokenizer != input_hashes[a.model / "tokenizer.json"]
    ):
        raise ValueError("slice manifest tokenizer mismatch")
    input_hashes.update({p: file_hash(p) for paths in slices.values() for p in paths})
    reference = getattr(a, "reference", None)
    rm = json.loads((reference / "metadata.json").read_text()) if reference else None
    if rm and (
        not rm.get("compact_k")
        or rm["mode"] != "unpruned_reference"
        or rm["tokenizer_sha256"] != input_hashes[a.model / "tokenizer.json"]
        or rm["model_revision"] != a.model_revision
        or set(rm["slices"]) != set(slices)
        or rm.get("ruler") != manifest.get("ruler")
    ):
        raise ValueError("incompatible compact reference")
    # Explicit lazy avoids full materialization and automatic page-cache prewarm.
    source_identity = checkpoint_hashes(a.model)
    checkpoint_index = (
        (a.model / "model.safetensors.index.json").read_bytes().decode("utf-8")
    )
    origin_identity, conversion_digest, origin_index = source_identity_for_checkpoint(
        source_identity, getattr(a, "conversion_provenance", None), checkpoint_index
    )
    if (
        rm
        and validate_checkpoint_identity(
            rm.get("source_checkpoint_sha256"), rm.get("source_index_json")
        )
        != origin_identity
    ):
        raise ValueError("reference source checkpoint identity mismatch")
    model, config = load(str(a.model), lazy=True, engram_ssd_offload=True)
    if rm and rm["vocab_size"] != config.vocab_size:
        raise ValueError("reference vocabulary mismatch")
    allocation_bytes = a.allocation.read_bytes() if a.allocation else None
    allocation = json.loads(allocation_bytes) if allocation_bytes is not None else None
    masses = {layer_id: [] for layer_id in range(config.n_layers)}
    if getattr(a, "serving_qmv", False):
        from rapid_mlx.models.deepseek_v41_native.serving import install_target_qmv

        install_target_qmv(model)
    if allocation is not None:
        install_native_masks(model, allocation, masses)
    a.output.mkdir(parents=True)
    (a.output / "slices-manifest.json").write_bytes(manifest_raw)
    retain_ruler_manifest(a.slices, a.output, ruler_digest)
    if allocation_bytes is not None:
        (a.output / "allocation.json").write_bytes(allocation_bytes)
    if conversion_digest is not None:
        record = a.conversion_provenance.read_bytes()
        if hashlib.sha256(record).hexdigest() != conversion_digest:
            raise ValueError("conversion provenance changed during collection")
        (a.output / "conversion-provenance.json").write_bytes(record)
    checkpoint_config = json.loads((a.model / "config.json").read_text())
    packed = checkpoint_config.get("rapid_quantization", {}).get("allocation")
    metadata = {
        "schema_version": 1,
        "num_layers": config.n_layers,
        "slices_manifest_sha256": input_hashes[a.slices],
        "n_routed_experts": config.n_routed_experts,
        "routing_topk": config.n_activated_experts,
        "serving_qmv": getattr(a, "serving_qmv", False),
        "model_revision": a.model_revision,
        "checkpoint_sha256": source_identity,
        "checkpoint_index_json": checkpoint_index,
        "source_index_json": origin_index,
        "source_checkpoint_sha256": origin_identity,
        "conversion_provenance_sha256": conversion_digest,
        "checkpoint_config_sha256": source_identity["config.json"],
        "tokenizer_sha256": input_hashes[a.model / "tokenizer.json"],
        "mode": "pruning_mask_only"
        if allocation is not None
        else "packed_candidate"
        if packed and not packed.get("reference")
        else "unpruned_reference",
        "allocation_sha256": hashlib.sha256(allocation_bytes).hexdigest()
        if allocation_bytes is not None
        else None,
        "slices": {},
        "ruler": manifest.get("ruler"),
        "ruler_manifest_sha256": ruler_digest,
        "reference_metadata_sha256": file_hash(reference / "metadata.json")
        if reference
        else None,
        "prefill_chunk": getattr(a, "prefill_chunk", 1),
    }
    for name, paths in slices.items():
        ids = slice_tokens(paths, tokenizer, a.tokens)
        if rm and (
            compact_hashes(reference, name) != rm["slices"][name]["compact_sha256"]
            or rm["slices"][name]["corpus_sha256"] != [input_hashes[p] for p in paths]
            or file_hash(reference / f"{name}-tokens.npy")
            != rm["slices"][name].get("tokens_sha256")
            or not np.array_equal(
                ids, np.load(reference / f"{name}-tokens.npy", allow_pickle=False)
            )
        ):
            raise ValueError("reference tokens/corpus/hash mismatch")
        np.save(a.output / f"{name}-tokens.npy", ids)
        cache = model.make_cache(max_seq_len=a.tokens + 1)
        scores = np.empty((a.tokens, 2), np.float64) if rm else None
        ref_arrays = compact_arrays(reference, name) if rm else None
        logits = (
            None
            if rm
            else np.lib.format.open_memmap(
                a.output / f"{name}-logits.npy",
                mode="w+",
                dtype=np.float32,
                shape=(a.tokens, config.vocab_size),
            )
        )
        for layer_id in masses:
            masses[layer_id].clear()
        chunk = getattr(a, "prefill_chunk", 1)
        for pos in range(0, len(ids), chunk):
            end = min(pos + chunk, len(ids))
            value = model(mx.array(ids[None, pos:end]), cache)
            mx.eval(value)
            host = np.array(value).reshape(end - pos, -1)
            if rm:
                kl, flip = compact_metrics(
                    {f: v[pos:end] for f, v in ref_arrays.items()}, host
                )
                scores[pos:end, 0], scores[pos:end, 1] = kl, flip
            else:
                logits[pos:end] = host
            if end % 16 == 0 or end == len(ids):
                if logits is not None:
                    logits.flush()
                print(
                    json.dumps(
                        {"slice": name, "tokens": end, "mode": metadata["mode"]}
                    ),
                    flush=True,
                )
        if logits is not None:
            logits.flush()
        if rm:
            np.save(a.output / f"{name}-scores.npy", scores)
        del logits, cache
        if allocation is not None:
            mass = np.array([masses[layer_id] for layer_id in range(config.n_layers)]).T
        else:
            mass = np.ones((a.tokens, config.n_layers), np.float32)
        np.save(a.output / f"{name}-mass.npy", mass)
        metadata["slices"][name] = {
            "tokens": a.tokens,
            "mass_sha256": file_hash(a.output / f"{name}-mass.npy"),
            "tokens_sha256": file_hash(a.output / f"{name}-tokens.npy"),
            "corpus_sha256": [input_hashes[p] for p in paths],
            **(
                {"scores_sha256": file_hash(a.output / f"{name}-scores.npy")}
                if rm
                else {"logits_sha256": file_hash(a.output / f"{name}-logits.npy")}
            ),
        }
    verify_input_hashes(input_hashes)
    if checkpoint_hashes(a.model) != source_identity:
        raise ValueError("checkpoint changed during collection")
    (a.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def summarize_metrics(kl, flips):
    return dict(
        kl_nats=float(np.mean(kl)),
        p99_kl_nats=float(np.quantile(kl, 0.99)),
        top1_flip_rate=float(np.mean(flips)),
        top1_agreement=float(1 - np.mean(flips)),
    )


def compare(reference: Path, candidate: Path, subset=False):
    rm = json.loads((reference / "metadata.json").read_text())
    cm = json.loads((candidate / "metadata.json").read_text())
    if rm["mode"] != "unpruned_reference":
        raise ValueError("reference must be an unpruned cache")
    if (
        rm["model_revision"] != cm["model_revision"]
        or rm["tokenizer_sha256"] != cm["tokenizer_sha256"]
        or not set(cm["slices"]).issubset(rm["slices"])
        or (not subset and set(rm["slices"]) != set(cm["slices"]))
        or not cm["slices"]
    ):
        raise ValueError("cache provenance/slice mismatch")
    for root, meta in ((reference, rm), (candidate, cm)):
        checkpoint = validate_checkpoint_identity(
            meta.get("checkpoint_sha256"), meta.get("checkpoint_index_json")
        )
        source = validate_checkpoint_identity(
            meta.get("source_checkpoint_sha256"), meta.get("source_index_json")
        )
        digest = meta.get("conversion_provenance_sha256")
        if checkpoint != source or digest is not None:
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError("cache lacks conversion provenance digest")
            try:
                recorded_source, recorded_digest, _ = source_identity_for_checkpoint(
                    checkpoint,
                    root / "conversion-provenance.json",
                    meta["checkpoint_index_json"],
                )
            except OSError as exc:
                raise ValueError("cache conversion provenance record missing") from exc
            if recorded_digest != digest or recorded_source != source:
                raise ValueError("cache conversion provenance record mismatch")
    if rm["source_checkpoint_sha256"] != cm["source_checkpoint_sha256"]:
        raise ValueError("cache source checkpoint identity mismatch")
    if rm.get("ruler") in PUBLIC_RULERS or cm.get("ruler") in PUBLIC_RULERS:
        for root, meta in ((reference, rm), (candidate, cm)):
            digest = meta.get("ruler_manifest_sha256")
            if (
                not digest
                or validate_ruler_manifest(root / "ruler" / "manifest.json") != digest
            ):
                raise ValueError("cache public ruler manifest mismatch")
            saved = json.loads((root / "ruler" / "manifest.json").read_bytes())
            if saved["ruler"] != meta.get("ruler"):
                raise ValueError("cache public ruler label mismatch")
        if set(rm["slices"]) != set(PUBLIC_RULERS[rm["ruler"]]):
            raise ValueError("reference public ruler slice coverage mismatch")
        if rm["ruler_manifest_sha256"] != cm["ruler_manifest_sha256"]:
            raise ValueError("cache public ruler identity mismatch")
    layers = cm.get("num_layers")
    if (
        type(layers) is not int
        or layers < 1
        or type(rm.get("num_layers")) is not int
        or rm["num_layers"] != layers
    ):
        raise ValueError("cache layer counts mismatch; regenerate legacy caches")
    if cm["mode"] == "pruning_mask_only":
        try:
            allocation_bytes = (candidate / "allocation.json").read_bytes()
        except OSError as exc:
            raise ValueError("cache pruning allocation missing") from exc
        if hashlib.sha256(allocation_bytes).hexdigest() != cm.get("allocation_sha256"):
            raise ValueError("cache pruning allocation hash mismatch")
        experts, topk = cm.get("n_routed_experts"), cm.get("routing_topk")
        if (
            type(experts) is not int
            or type(topk) is not int
            or not 1 <= topk <= experts
        ):
            raise ValueError("invalid cache routing dimensions")
        validate_allocation(json.loads(allocation_bytes), layers, experts, topk)
    results, all_kl, all_flips = {}, [], []
    for name in rm["slices"]:
        if name not in cm["slices"]:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("unsafe slice name")
        if rm["slices"][name]["corpus_sha256"] != cm["slices"][name]["corpus_sha256"]:
            raise ValueError("cache corpus mismatch")
        compact = bool(rm.get("compact_k"))
        online = bool(cm.get("reference_metadata_sha256"))
        if online and (
            not compact
            or cm["reference_metadata_sha256"] != file_hash(reference / "metadata.json")
        ):
            raise ValueError("online scores belong to a different reference")
        for root, meta in ((reference, rm), (candidate, cm)):
            entry = meta["slices"][name]
            if file_hash(root / f"{name}-tokens.npy") != entry.get("tokens_sha256"):
                raise ValueError("cache token IDs hash mismatch")
            if "compact_sha256" in entry:
                if compact_hashes(root, name) != entry["compact_sha256"]:
                    raise ValueError("cache compact hash mismatch")
            else:
                kind = "scores" if "scores_sha256" in entry else "logits"
                if file_hash(root / f"{name}-{kind}.npy") != entry[f"{kind}_sha256"]:
                    raise ValueError("cache logits/scores hash mismatch")
        ids = np.load(reference / f"{name}-tokens.npy", allow_pickle=False)
        if not np.array_equal(
            ids, np.load(candidate / f"{name}-tokens.npy", allow_pickle=False)
        ):
            raise ValueError("teacher-forced token IDs differ")
        if rm.get("ruler") in PUBLIC_RULERS and not np.array_equal(
            ids, np.load(reference / "ruler" / f"{name}.npy", allow_pickle=False)
        ):
            raise ValueError("cache tokens differ from authoritative public ruler")
        n = len(ids)
        if (
            not n
            or rm["slices"][name].get("tokens", n) != n
            or cm["slices"][name].get("tokens", n) != n
        ):
            raise ValueError("cache token counts mismatch")
        if online:
            scores = np.load(candidate / f"{name}-scores.npy", allow_pickle=False)
            if (
                scores.shape != (n, 2)
                or not np.isfinite(scores).all()
                or np.any(scores[:, 0] < 0)
                or not np.isin(scores[:, 1], [0, 1]).all()
            ):
                raise ValueError("invalid online scores")
            kls, flip_values = scores[:, 0], scores[:, 1]
        else:
            r = (
                compact_arrays(reference, name)
                if compact
                else np.load(
                    reference / f"{name}-logits.npy", mmap_mode="r", allow_pickle=False
                )
            )
            c = np.load(
                candidate / f"{name}-logits.npy", mmap_mode="r", allow_pickle=False
            )
            if (
                c.ndim != 2
                or c.shape[0] != n
                or (compact and c.shape[1] != rm["vocab_size"])
                or (not compact and r.shape != c.shape)
            ):
                raise ValueError("cache logits shapes differ")
            kls, flip_values = np.empty(n), np.empty(n)
            for start in range(0, n, 8):
                end = start + 8
                rr = (
                    {f: v[start:end] for f, v in r.items()} if compact else r[start:end]
                )
                fn = compact_metrics if compact else token_metrics
                kls[start:end], flip_values[start:end] = fn(rr, c[start:end])
        all_kl.append(kls)
        all_flips.append(flip_values)
        mass_path = candidate / f"{name}-mass.npy"
        if file_hash(mass_path) != cm["slices"][name].get("mass_sha256"):
            raise ValueError("cache routing mass hash mismatch")
        mass = np.load(mass_path, allow_pickle=False)
        if (
            mass.shape != (n, layers)
            or not np.isfinite(mass).all()
            or np.any((mass < 0) | (mass > 1 + 1e-6))
        ):
            raise ValueError("invalid routing mass cache")
        results[name] = {
            "tokens": n,
            **summarize_metrics(kls, flip_values),
            "mean_retained_routing_mass": float(mass.mean()),
        }
    ruler = rm.get("ruler")
    if ruler != cm.get("ruler"):
        raise ValueError("ruler label mismatch")
    summary = summarize_metrics(np.concatenate(all_kl), np.concatenate(all_flips))
    return {
        "mode": cm["mode"],
        "kl_estimator": "topk_plus_tail_bucket_lower_bound"
        if rm.get("compact_k")
        else "full_vocabulary",
        **summary,
        **(
            {
                "ruler": ruler,
                "rulers": {
                    ruler: {
                        "tokens": sum(v["tokens"] for v in results.values()),
                        **summary,
                    }
                },
            }
            if ruler
            else {}
        ),
        "slices": results,
        **({"compared_slices": list(results)} if subset else {}),
        "worst_domain_kl_nats": max(v["kl_nats"] for v in results.values()),
        "mean_domain_kl_nats": float(np.mean([v["kl_nats"] for v in results.values()])),
    }


def select_knee(points, threshold=0.02):
    """Largest rise in worst-domain KL/GB; select one larger tested size."""
    ordered = sorted(points, key=lambda p: p["budget_gb"], reverse=True)
    if (
        len(ordered) < 3
        or not np.isfinite(threshold)
        or threshold < 0
        or any(
            not np.isfinite(p[k]) or p[k] < 0
            for p in ordered
            for k in ("budget_gb", "worst_domain_kl_nats")
        )
        or len({p["budget_gb"] for p in ordered}) != len(ordered)
    ):
        raise ValueError("need >=3 distinct budgets and a nonnegative fixed threshold")
    slopes = [
        (b["worst_domain_kl_nats"] - a["worst_domain_kl_nats"])
        / (a["budget_gb"] - b["budget_gb"])
        for a, b in zip(ordered, ordered[1:])
    ]
    if max(slopes) <= 0:
        raise ValueError("no KL knee detected; extend sweep")
    edge = int(np.argmax(slopes))
    # The knee is the smaller endpoint of this edge; safe side is its predecessor.
    eligible = [
        p for p in ordered[: edge + 1] if p["worst_domain_kl_nats"] <= threshold
    ]
    if not eligible:
        raise ValueError(
            "no safe-side candidate passes the fixed worst-domain KL threshold"
        )
    return eligible[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("collect")
    cache.add_argument(
        "--model",
        type=Path,
        required=True,
        help="converted native checkpoint, never raw FP4 release",
    )
    cache.add_argument("--model-revision", required=True)
    cache.add_argument("--slices", type=Path, required=True)
    cache.add_argument("--tokens", type=int, default=128)
    cache.add_argument("--prefill-chunk", type=int, default=1)
    cache.add_argument(
        "--reference",
        type=Path,
        help="compact reference: score online without saving full candidate logits",
    )
    cache.add_argument("--allocation", type=Path)
    cache.add_argument(
        "--conversion-provenance",
        type=Path,
        help="JSON conversion record with source/candidate hash maps and exact source_index_json/checkpoint_index_json documents; required when source and candidate weights differ",
    )
    cache.add_argument(
        "--serving-qmv",
        action="store_true",
        help="exercise guarded serving down-QMV where eligible",
    )
    cache.add_argument("--output", type=Path, required=True)
    metrics = sub.add_parser("compare")
    metrics.add_argument("--reference", type=Path, required=True)
    metrics.add_argument("--candidate", type=Path, required=True)
    metrics.add_argument("--output", type=Path, required=True)
    metrics.add_argument("--subset", action="store_true")
    select = sub.add_parser("select")
    select.add_argument(
        "--points",
        type=Path,
        required=True,
        help="JSON list: budget_gb, worst_domain_kl_nats",
    )
    select.add_argument("--worst-domain-kl-threshold", type=float, default=0.02)
    a = p.parse_args()
    if a.command == "collect":
        collect(a)
    elif a.command == "compare":
        result = compare(a.reference, a.candidate, subset=a.subset)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    else:
        print(
            json.dumps(
                select_knee(
                    json.loads(a.points.read_text()), a.worst_domain_kl_threshold
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
