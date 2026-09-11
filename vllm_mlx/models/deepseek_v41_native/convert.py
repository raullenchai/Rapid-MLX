"""Convert the raw DeepSeek-V4.1-Flash release to an MLX build.

The hub release already uses inference-style names (``layers.N.attn.wq_a.weight``,
``ffn``, ``.scale`` — no ``model.`` prefix, no ``self_attn``/``mlp``/
``weight_scale_inv``), so sanitizing is mostly dequantization plus expert
stacking:

* fp8 ``weight``+``scale`` pairs (32x32 ue8m0 blocks) -> bf16;
* fp4 routed experts (packed nibbles, per-32 ue8m0) -> bf16, stacked per layer
  into SwitchGLU's ``experts.{gate,up,down}_proj.weight`` (w1/w3/w2);
* ``engram.embed`` stays in its native fp8+scale form (dequantizing ~197B
  parameters to bf16 would double 40% of the model); both tensors pass through
  as uint8;
* ``hc_*``, ``attn_sink``, ``gate.bias``, ``gate.bias_vl`` -> fp32;
* ``mtp.*`` (the DSpark/MTP draft stack) is dropped — it hangs off the
  reference's separate ``forward_spec`` path;
* the vision tower (``vision.*``, ``aligner.*``, ``image_start/end/newline``)
  is carried through **unchanged** under its original names. The text-only
  runtime does not load it, but the tensors stay in the build (they are bf16
  and small, ~0.8 GB) so nothing silently vanishes; ``load.py`` accounts for
  them explicitly.

Quantization (optional): MLX affine quantization applied per module, experts at
``expert_bits`` and everything else at ``bits``. Rapid-MLX represents ``wo_a``
as a group-aware ``MultiLinear``, so it can remain quantized without reshaping
packed uint32 storage.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, cast

import mlx.core as mx

from .dequant import dequant_fp4, dequant_fp8, is_fp4_expert

VISION_PREFIXES = ("vision.", "aligner.", "image_start", "image_end", "image_newline")

# modules kept at bf16 even when quantizing (beyond shape-ineligible ones):
#   engram.embed  — stays in native fp8+ue8m0, dequantized per lookup
#   compressor.*  — tiny, and the ratio>1 projections run in fp32
#   indexer.wk / weights_proj — tiny, and they drive discrete top-k selection
_NEVER_QUANT = re.compile(
    r"(\.engram\.embed\.|\.compressor\.|\.indexer\.wk\.|\.weights_proj\.)"
)


def is_quant_target(name: str, in_dim: int, group_size: int) -> bool:
    if not name.endswith(".weight"):
        return False
    if _NEVER_QUANT.search(name):
        return False
    if in_dim % group_size:
        return False
    tail = name.rsplit(".", 2)
    quantizable = (
        "wq_a",
        "wq_b",
        "wkv",
        "wo_a",
        "wo_b",
        "w1",
        "w2",
        "w3",
        "gate_proj",
        "up_proj",
        "down_proj",
        "embed",
        "head",
    )
    return len(tail) >= 2 and tail[-2] in quantizable


def bits_for(name: str, bits: int, expert_bits: int | None) -> int:
    if expert_bits and is_fp4_expert(name):
        return expert_bits
    return bits


def _shard_iter(src: str):
    index_path = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(index_path):
        wmap = json.load(open(index_path))["weight_map"]
        return wmap
    wmap = {}
    for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        for k in mx.load(f):
            wmap[k] = os.path.basename(f)
    return wmap


def _to_u8(a: mx.array) -> mx.array:
    """Raw release bytes (fp8 values / ue8m0 scales) as uint8, however stored."""
    if a.dtype in (mx.uint8, mx.int8):
        return a.view(mx.uint8)
    return a.view(mx.uint8)


def sanitize_group(
    names: list[str], tensors: dict[str, mx.array], dtype=mx.bfloat16
) -> dict[str, mx.array]:
    """Sanitize one coherent group of raw tensors (a layer, or the top level).

    Returns final-name -> array. ``mtp.*`` must be filtered by the caller;
    vision tensors pass through unchanged.
    """
    out: dict[str, mx.array] = {}
    consumed = set()
    expert_parts: dict[str, dict[int, mx.array]] = {}

    for name in names:
        if name in consumed:
            continue
        if name.startswith(VISION_PREFIXES):
            out[name] = tensors[name]
            continue
        if name.endswith(".scale"):
            continue  # handled with its weight
        t = tensors[name]

        if ".engram.embed." in name:  # stays native fp8 + ue8m0
            out[name] = _to_u8(t)
            engram_scale_name = name[: -len(".weight")] + ".scale"
            out[engram_scale_name] = _to_u8(tensors[engram_scale_name])
            consumed.add(engram_scale_name)
            continue

        scale_name = (
            name[: -len(".weight")] + ".scale" if name.endswith(".weight") else None
        )
        if scale_name is not None and scale_name in tensors:
            consumed.add(scale_name)
            s = _to_u8(tensors[scale_name])
            if is_fp4_expert(name):
                w = dequant_fp4(t, s, dtype)
            else:
                w = dequant_fp8(_to_u8(t), s, dtype)
        else:
            w = t

        m = re.match(r"(layers\.\d+\.ffn\.experts)\.(\d+)\.(w[123])\.weight$", name)
        if m:
            prefix, idx, wn = m.group(1), int(m.group(2)), m.group(3)
            proj = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}[wn]
            expert_parts.setdefault(f"{prefix}.{proj}.weight", {})[idx] = w
            continue

        if re.search(r"(hc_\w+|attn_sink|gate\.bias(_vl)?)$", name):
            w = w.astype(mx.float32)
        out[name] = w

    for final, parts in expert_parts.items():
        n = len(parts)
        assert sorted(parts) == list(range(n)), f"missing experts for {final}"
        out[final] = mx.stack([parts[i] for i in range(n)])

    orphans = [
        n
        for n in names
        if n.endswith(".scale")
        and n not in consumed
        and not n.startswith(VISION_PREFIXES)
    ]
    if orphans:
        raise ValueError(
            f"{len(orphans)} scale tensors without a weight, "
            f"e.g. {orphans[:3]} — layout change?"
        )
    return out


def quantize_tree(
    weights: dict[str, mx.array], bits: int, expert_bits: int | None, group_size: int
) -> tuple[dict[str, mx.array], dict[str, dict]]:
    """Replace eligible ``.weight`` tensors with MLX affine (weight, scales,
    biases) triples, matching what ``nn.quantize`` produces at load time.
    Also returns the per-module map {module path: {group_size, bits}} that
    config.json carries for the loader to replay."""
    out = {}
    modules: dict[str, dict] = {}
    for name, w in weights.items():
        if name.startswith(VISION_PREFIXES) or not is_quant_target(
            name, w.shape[-1], group_size
        ):
            out[name] = w
            continue
        b = bits_for(name, bits, expert_bits)
        wq, scales, biases = mx.quantize(w, group_size=group_size, bits=b)
        base = name[: -len(".weight")]
        out[f"{base}.weight"] = wq
        out[f"{base}.scales"] = scales
        out[f"{base}.biases"] = biases
        modules[base] = {"group_size": group_size, "bits": b}
    return out, modules


def quantize_engram_table(
    weight_u8: mx.array, scale_u8: mx.array, bits: int, group_size: int
) -> dict[str, mx.array]:
    """Native fp8+ue8m0 table (already in memory) -> MLX affine triple, in row
    chunks. NOTE: evaluating any chunk of a lazily-loaded mx tensor
    materializes the WHOLE load node, so for the 98 GB release tables use
    :func:`quantize_engram_table_from_file`, which slices rows from disk.
    This in-memory variant serves tests and the ladder's gathered rows."""
    from .dequant import dequant_fp8_rows

    rows = weight_u8.shape[0]
    step = 1 << 22
    cw, cs, cb = [], [], []
    for i in range(0, rows, step):
        deq = dequant_fp8_rows(weight_u8[i : i + step], scale_u8[i : i + step])
        w, s, b = mx.quantize(deq.astype(mx.bfloat16), group_size=group_size, bits=bits)
        mx.eval(w, s, b)
        cw.append(w)
        cs.append(s)
        cb.append(b)
    return {
        "weight": mx.concatenate(cw),
        "scales": mx.concatenate(cs),
        "biases": mx.concatenate(cb),
    }


def quantize_engram_table_from_file(
    src_dir: str,
    wmap: dict,
    wname: str,
    bits: int,
    group_size: int,
    step: int = 1 << 22,
) -> dict[str, mx.array]:
    """Stream a release engram table off disk in row chunks (safetensors row
    slicing — never materializes the 98 GB tensor) into an MLX affine triple.
    Chunk arithmetic is identical to :func:`quantize_engram_table` (affine
    quantization is row-local)."""
    import torch
    from safetensors import safe_open

    from .dequant import dequant_fp8_rows

    sname = wname[: -len(".weight")] + ".scale"
    fw = safe_open(os.path.join(src_dir, wmap[wname]), framework="pt")
    fs = (
        fw
        if wmap[sname] == wmap[wname]
        else safe_open(os.path.join(src_dir, wmap[sname]), framework="pt")
    )
    slw, sls = fw.get_slice(wname), fs.get_slice(sname)
    rows = slw.get_shape()[0]
    cw, cs, cb = [], [], []
    for i in range(0, rows, step):
        w_u8 = mx.array(slw[i : i + step].view(torch.uint8).numpy())
        s_u8 = mx.array(sls[i : i + step].view(torch.uint8).numpy())
        deq = dequant_fp8_rows(w_u8, s_u8)
        w, s, b = mx.quantize(deq.astype(mx.bfloat16), group_size=group_size, bits=bits)
        mx.eval(w, s, b)
        cw.append(w)
        cs.append(s)
        cb.append(b)
        del w_u8, s_u8, deq
        mx.clear_cache()
    return {
        "weight": mx.concatenate(cw),
        "scales": mx.concatenate(cs),
        "biases": mx.concatenate(cb),
    }


def _group_tag(g: str) -> str:
    if g.startswith("layers."):
        return f"layers-{int(g.split('.')[1]):02d}"
    return g


def _stream_raw_copy(
    src_dir: str,
    wmap: dict,
    names: list[str],
    out_path: str,
    chunk_bytes: int = 1 << 28,
):
    """Byte-stream tensors from source shards into one new safetensors file,
    preserving their stored dtype tags, without materializing them (the native
    engram tables are ~101 GB each). All passthrough dtypes are 1 byte/elt."""
    import struct

    from safetensors import safe_open

    handles: dict[str, Any] = {}
    entries = []
    off = 0
    for n in names:
        f = os.path.join(src_dir, wmap[n])
        if f not in handles:
            handles[f] = safe_open(f, framework="pt")
        sl = handles[f].get_slice(n)
        shape, dt = sl.get_shape(), sl.get_dtype()
        nbytes = 1
        for s in shape:
            nbytes *= s
        entries.append((n, dt, shape, off, off + nbytes, sl))
        off += nbytes
    header = {
        n: {"dtype": dt, "shape": list(shape), "data_offsets": [a, b]}
        for n, dt, shape, a, b, _ in entries
    }
    hjson = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(hjson)) % 8
    hjson += b" " * pad
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(hjson)))
        out.write(hjson)
        for n, dt, shape, a, b, sl in entries:
            row_bytes = max(1, (b - a) // max(shape[0], 1))
            step = max(1, chunk_bytes // row_bytes)
            for i in range(0, shape[0], step):
                t = sl[i : i + step]
                out.write(t.view(__import__("torch").uint8).numpy().tobytes())
    os.replace(tmp, out_path)


def _read_st_header(path: str) -> dict:
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return cast(dict[Any, Any], json.loads(f.read(n)))


def convert(
    src: str,
    dst: str,
    bits: int | None = None,
    expert_bits: int | None = None,
    engram_bits: int | None = None,
    group_size: int = 64,
    dtype=mx.bfloat16,
    resume: bool = False,
) -> dict:
    """Stream the release group by group (top level, each layer, vision) into
    an MLX build at ``dst`` — one output shard per group, written to a temp
    name and renamed on completion, so ``resume=True`` skips every finished
    group at a clean layer boundary.

    ``engram_bits`` re-quantizes the two engram tables from native fp8+ue8m0
    (203 GB total) to MLX affine, streamed in row chunks (never materialized
    whole); ``None`` keeps them native, byte-streamed into their own shard.

    ``config.json`` gets a ``quantization`` block with defaults plus an
    explicit per-module map keyed by runtime-internal module paths.
    Returns {kept, dropped_mtp, vision_passthrough}.
    """
    os.makedirs(dst, exist_ok=True)
    wmap = _shard_iter(src)

    def owner(name: str) -> str:
        m = re.match(r"(layers\.\d+)\.", name)
        if m:
            return m.group(1)
        if name.startswith("mtp."):
            return "mtp"
        if name.startswith(VISION_PREFIXES):
            return "vision"
        return "top"

    groups: dict[str, list[str]] = {}
    for name in wmap:
        groups.setdefault(owner(name), []).append(name)
    accounting = {
        "kept": 0,
        "dropped_mtp": len(groups.pop("mtp", [])),
        "vision_passthrough": 0,
    }

    shard_cache: dict[str, dict] = {}

    def fetch(names):
        res = {}
        for n in names:
            f = wmap[n]
            if f not in shard_cache:
                shard_cache.clear()
                shard_cache[f] = mx.load(os.path.join(src, f))
            res[n] = shard_cache[f][n]
        return res

    order = ["top"] + sorted(
        (g for g in groups if g.startswith("layers.")),
        key=lambda s: int(s.split(".")[1]),
    )
    if "vision" in groups:
        order.append("vision")

    index: dict[str, str] = {}
    module_map: dict[str, dict] = {}

    for g in order:
        tag = _group_tag(g)
        names = sorted(groups[g])
        out_files = {"main": os.path.join(dst, f"model-{tag}.safetensors")}
        engram_pair = [n for n in names if ".engram.embed." in n]
        if engram_pair:
            out_files["engram"] = os.path.join(dst, f"model-{tag}-engram.safetensors")
        manifest_path = os.path.join(dst, f"model-{tag}.manifest.json")

        if (
            resume
            and os.path.exists(manifest_path)
            and all(os.path.exists(p) for p in out_files.values())
        ):
            man = json.load(open(manifest_path))
        else:
            work = [n for n in names if n not in engram_pair]
            sane = sanitize_group(work, fetch(work), dtype)
            grp_modules: dict[str, dict] = {}
            if bits and g != "vision":
                sane, grp_modules = quantize_tree(sane, bits, expert_bits, group_size)
            for v in sane.values():
                mx.eval(v)
            tmp = out_files["main"] + ".tmp.safetensors"
            mx.save_safetensors(tmp, sane)
            os.replace(tmp, out_files["main"])
            engram_names = []
            if engram_pair and engram_bits:
                wname = next(n for n in engram_pair if n.endswith(".weight"))
                base = wname[: -len(".weight")]
                triple = quantize_engram_table_from_file(
                    src, wmap, wname, engram_bits, group_size
                )
                shard = {f"{base}.{suf}": v for suf, v in triple.items()}
                tmp = out_files["engram"] + ".tmp.safetensors"
                mx.save_safetensors(tmp, shard)
                os.replace(tmp, out_files["engram"])
                engram_names = sorted(shard)
                del triple, shard
                mx.clear_cache()
            elif engram_pair:
                _stream_raw_copy(src, wmap, engram_pair, out_files["engram"])
                engram_names = engram_pair
            man = {
                "files": {
                    os.path.basename(out_files["main"]): sorted(sane),
                    **(
                        {os.path.basename(out_files["engram"]): engram_names}
                        if engram_pair
                        else {}
                    ),
                },
                "modules": grp_modules,
            }
            json.dump(man, open(manifest_path, "w"))
            del sane
            mx.clear_cache()

        for fname, tnames in man["files"].items():
            for t in tnames:
                index[t] = fname
            accounting["kept"] += len(tnames)
            if g == "vision":
                accounting["vision_passthrough"] += len(tnames)
        module_map.update(man["modules"])

    json.dump(
        {"metadata": {}, "weight_map": index},
        open(os.path.join(dst, "model.safetensors.index.json"), "w"),
    )

    cfg_path = os.path.join(src, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    out_cfg = dict(cfg)
    out_cfg["model_type"] = "deepseek_v41"
    if bits or engram_bits:
        out_cfg["quantization"] = {
            "group_size": group_size,
            "bits": bits,
            "expert_bits": expert_bits,
            "engram_bits": engram_bits,
            "modules": module_map,
        }
    json.dump(out_cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)

    for f in ("tokenizer.json", "tokenizer_config.json"):
        p = os.path.join(src, f)
        if os.path.exists(p):
            import shutil

            shutil.copyfile(p, os.path.join(dst, f))
    return accounting
