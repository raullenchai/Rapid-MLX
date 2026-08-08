# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx convert`` — GGUF → MLX model conversion pipeline.

Turns a GGUF-only checkpoint (the dominant distribution format for small
models in the llama.cpp ecosystem) into a directory ``rapid-mlx serve`` /
``mlx_lm.load`` can consume directly:

    GGUF input ──▶ per-tensor streaming dequant to bf16/fp16/fp32
                ──▶ HF-style directory assembly (config.json + tokenizer +
                    sharded safetensors + optional index.json)
                ──▶ [optional] re-quantize to MLX N-bit via ``mlx_lm.convert``
                ──▶ [optional] perplexity quality report (full-precision
                    intermediate vs. re-quantized product)

Design decisions worth knowing about:

* **Lazy imports.** ``gguf``, ``mlx``, ``mlx_lm``, ``transformers`` and
  ``huggingface_hub`` are imported inside the functions that need them, never
  at module top level. ``rapid-mlx`` CLI startup cost must stay flat — the
  ``convert`` subcommand module itself is only imported by ``cli.main()``
  when dispatching this command (same pattern as ``jlens``/``doctor``).

* **Streaming write-through.** Tensors are dequantized one at a time off the
  GGUFReader's memmap and accumulated into ~4 GiB safetensors shards that are
  flushed to disk as they fill. Peak memory is one shard plus one tensor, so
  an 18 GB machine can convert 30B+ models. Shard *assignment* is planned in
  a cheap metadata-only first pass so multi-shard runs still get the standard
  ``model-0000N-of-0000M.safetensors`` naming (with ``M`` known up front).

* **Config/tokenizer acquisition, in reliability order.** (1) If the GGUF
  carries ``general.base_model.*`` provenance (fine-tunes of a hub model
  almost always do), pull the base repo's config.json/tokenizer files — they
  are the exact artifacts the model was trained against. (2) transformers'
  native GGUF reader (only usable when torch happens to be installed — it is
  gated behind ``is_torch_available()`` upstream). (3) Deterministic rebuild
  from the GGUF metadata kv pairs, driven by transformers' own
  ``GGUF_CONFIG_MAPPING`` / ``convert_gguf_tokenizer`` tables (torch-free).
  ``torch_dtype`` in config.json is always overridden to the *actual* dtype
  we wrote, regardless of where the config came from — ``mlx_lm.convert``
  keys its dtype cast off that field.

* **Weight naming.** ggml tensor names (``blk.0.attn_q.weight``) are mapped
  to the HF convention mlx-lm expects (``model.layers.0.self_attn.q_proj.weight``)
  by *inverting* ``gguf.tensor_mapping.get_tensor_name_map`` and picking the
  ``model.*``-prefixed candidate (the llama-hf convention every mlx-lm dense
  model uses). Tensors with no HF-convention counterpart are skipped with a
  recorded warning rather than aborting the conversion — a model that loads
  with three exotic tensors missing beats no model at all, and the skip list
  is printed so the user can judge. Precomputed RoPE frequency buffers
  (``rope.freqs``) are always skipped: mlx-lm computes ``inv_freq`` at model
  init and its loaders explicitly discard such keys.

* **No torch, ever.** Dequantization is pure numpy (``gguf.quants``), dtype
  casting and safetensors writing go through ``mlx.core`` (numpy has no
  bf16), and re-quantization is delegated to ``mlx_lm.convert`` (which loads
  lazily, so the re-quant pass is also memory-safe).
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Target size for one safetensors shard. ~4 GiB mirrors the HF/MLX
# convention (mlx-lm's own saver shards at 5 GB; HF at 4-5 GB) and bounds
# peak RAM during the streaming write to one shard plus one tensor.
_SHARD_TARGET_BYTES = 4 * 1024**3

_DEFAULT_GROUP_SIZE = 64
_QUANT_BITS = (2, 3, 4, 6, 8)
_DTYPES = ("bfloat16", "float16", "float32")
_DTYPE_ITEMSIZE = {"bfloat16": 2, "float16": 2, "float32": 4}

# ggml tensor *types* whose contents are precomputed runtime caches rather
# than learned weights. mlx-lm recomputes RoPE inv_freq at init and its
# per-model ``sanitize()`` drops these keys anyway, so shipping them would
# just bloat the artifact. Resolved by name in ``_stream_weights`` (the enum
# import stays lazy with the rest of gguf).
_SKIP_GGML_TENSOR_NAMES = frozenset(
    {
        "rope_freqs.weight",  # MODEL_TENSOR.ROPE_FREQS (top-level)
    }
)
_SKIP_GGML_TENSOR_PATTERNS = (
    re.compile(r"^blk\.\d+\.attn_rot_embd\.weight$"),  # per-layer inv_freq
)

# HF ``architectures`` class for the model_type values we can emit. mlx-lm
# resolves models by ``model_type`` alone, so a missing entry here only
# matters to transformers-side consumers; omitted rather than guessed.
_ARCHITECTURES = {
    "llama": "LlamaForCausalLM",
    "mistral": "MistralForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
    "qwen2_moe": "Qwen2MoeForCausalLM",
    "qwen3_moe": "Qwen3MoeForCausalLM",
    "gemma2": "Gemma2ForCausalLM",
    "gemma3_text": "Gemma3ForCausalLM",
    "phi3": "Phi3ForCausalLM",
    "starcoder2": "Starcoder2ForCausalLM",
}

# Keys into transformers' ``GGUF_TO_FAST_CONVERTERS`` dispatch for the
# model_types we emit. Llama's converter covers every BPE/SentencePiece
# llama-family variant, which is why mistral maps to it.
_TOKENIZER_CONVERTER_KEY = {
    "llama": "llama",
    "mistral": "llama",
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "qwen2_moe": "qwen2_moe",
    "qwen3_moe": "qwen3_moe",
    "gemma2": "gemma2",
    "gemma3_text": "gemma3_text",
    "gemma4_text": "gemma4_text",
    "phi3": "phi3",
    "starcoder2": "starcoder2",
}

# MoE archs whose HF convention puts the router/experts under ``mlp.*``
# (qwen2moe/qwen3moe) rather than ``block_sparse_moe.*`` (mixtral). Drives
# candidate preference in ``_build_name_map``.
_MLP_STYLE_MOE_GGUF_ARCHES = frozenset({"qwen2moe", "qwen3moe", "olmoe"})

# Bundled evaluation corpus for --report. Public domain (Jane Austen,
# "Pride and Prejudice", 1813). ~1.8 KB of natural English prose — long
# enough for a stable perplexity signal on small models, short enough that
# the report adds seconds, not minutes, to a conversion.
_REPORT_TEXT = """\
It is a truth universally acknowledged, that a single man in possession of a
good fortune, must be in want of a wife.

However little known the feelings or views of such a man may be on his first
entering a neighbourhood, this truth is so well fixed in the minds of the
surrounding families, that he is considered the rightful property of some one
or other of their daughters.

"My dear Mr. Bennet," said his lady to him one day, "have you heard that
Netherfield Park is let at last?"

Mr. Bennet replied that he had not.

"But it is," returned she; "for Mrs. Long has just been here, and she told me
all about it."

Mr. Bennet made no answer.

"Do you not want to know who has taken it?" cried his wife impatiently.

"You want to tell me, and I have no objection to hearing it."

This was invitation enough.

"Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a
young man of large fortune from the north of England; that he came down on
Monday in a chaise and four to see the place, and was so much delighted with
it, that he agreed with Mr. Morris immediately; that he is to take possession
before Michaelmas, and some of his servants are to be in the house by the end
of next week."

"What is his name?"

"Bingley."

"Is he married or single?"

"Oh! Single, my dear, to be sure! A single man of large fortune; four or five
thousand a year. What a fine thing for our girls!"

"How so? How can it affect them?"

"My dear Mr. Bennet," replied his wife, "how can you be so tiresome! You must
know that I am thinking of his marrying one of them."

"Is that his design in settling here?"

"Design! Nonsense, how can you talk so! But it is very likely that he may fall
in love with one of them, and therefore you must visit him as soon as he comes."

"I see no occasion for that. You and the girls may go, or you may send them by
themselves, which perhaps will be still better, for as you are as handsome as
any of them, Mr. Bingley may like you the best of the party."

"My dear, you flatter me. I certainly have had my share of beauty, but I do not
pretend to be anything extraordinary now. When a woman has five grown-up
daughters, she ought to give over thinking of her own beauty."
"""

# Report token budget: rows × sequence length. 4×256 = 1024 tokens keeps the
# two forward passes in the single-digit seconds for ≤4B models while giving
# perplexity enough samples to be meaningful. The sequence length is clamped
# to the model's own context window when that is smaller (tiny test models).
_REPORT_ROWS = 4
_REPORT_SEQ_LEN = 256


class ConvertError(Exception):
    """User-facing conversion failure (bad input, unsupported arch, …).

    The CLI catches this and prints a one-line ``Error: …``; anything else
    propagates as a bug with a stack trace.
    """


@dataclass
class ConvertResult:
    """Outcome of a completed conversion (also powers the CLI summary)."""

    out_dir: Path
    model_type: str
    config_source: str
    tokenizer_source: str
    tensors_written: int
    skipped: list[tuple[str, str]] = field(default_factory=list)
    size_bytes: int = 0
    bits: int | None = None
    report: dict[str, float] | None = None
    seconds: float = 0.0


# ---------------------------------------------------------------------------
# GGUF metadata helpers
# ---------------------------------------------------------------------------


def _field(reader: Any, name: str) -> Any:
    """Read one GGUF metadata value, or ``None`` when absent.

    ``ReaderField.contents()`` decodes scalars to Python values and arrays to
    lists, which is exactly the shape config building wants.
    """
    f = reader.get_field(name)
    if f is None:
        return None
    return f.contents()


def _gguf_arch(reader: Any) -> str:
    arch = _field(reader, "general.architecture")
    if not arch:
        raise ConvertError(
            "GGUF metadata has no 'general.architecture' — not a model file?"
        )
    return str(arch)


def _kv_prefix_and_model_type(reader: Any) -> tuple[str, str]:
    """Return ``(gguf_kv_prefix, hf_model_type)`` for the file.

    Mirrors the two arch adjustments transformers applies in
    ``load_gguf_checkpoint``: llama.cpp stores Mistral models under the llama
    architecture (disambiguated by ``general.name``), and gemma3/gemma4 GGUF
    archs map to the ``*_text`` HF model types when (as always for a pure
    GGUF) only the text stack is present.
    """
    arch = _gguf_arch(reader)
    name = (_field(reader, "general.name") or "").lower()
    mapping_arch = arch
    if "llama" in arch and "mistral" in name:
        mapping_arch = "mistral"
    model_type = {
        "gemma3": "gemma3_text",
        "gemma4": "gemma4_text",
    }.get(mapping_arch, mapping_arch)
    return arch, model_type


# GGML quantization types in the 1-bit family. Dequantizing these to bf16 is
# already a big information loss, so *re*-quantizing the result with --bits
# stacks quantization error on quantization error — see the warning in
# ``convert_gguf`` for the measured impact.
_ONE_BIT_QTYPE_NAMES = frozenset({"IQ1_S", "IQ1_M", "TQ1_0", "TQ2_0"})


def _one_bit_source_quant(reader: Any) -> str | None:
    """Return the 1-bit family quant name if the source is one, else ``None``.

    Primary signal is ``general.file_type`` (the file-wide quant stamp);
    mixed-quant files without a matching file_type fall back to a tensor-type
    majority vote.
    """
    from gguf import GGMLQuantizationType

    ft = _field(reader, "general.file_type")
    if ft is not None:
        try:
            qname = GGMLQuantizationType(int(ft)).name
        except ValueError:
            qname = None
        if qname in _ONE_BIT_QTYPE_NAMES:
            return qname
    counts: dict[str, int] = {}
    for t in reader.tensors:
        counts[t.tensor_type.name] = counts.get(t.tensor_type.name, 0) + 1
    for name in _ONE_BIT_QTYPE_NAMES:
        if counts.get(name, 0) > len(reader.tensors) / 2:
            return name
    return None


# ---------------------------------------------------------------------------
# Source resolution: local path / org/repo / org/repo:file.gguf
# ---------------------------------------------------------------------------


def _resolve_source(source: str) -> Path:
    """Resolve a ``convert`` source argument to a local ``.gguf`` path.

    Accepts a local file path, an HF repo id (``org/repo`` — downloads only
    ``*.gguf`` files via ``snapshot_download(allow_patterns=...)`` and
    requires the repo to contain exactly one), or the explicit
    ``org/repo:filename.gguf`` form for multi-quant repos.
    """
    p = Path(source).expanduser()
    if p.is_file():
        if p.suffix.lower() != ".gguf":
            raise ConvertError(f"'{p}' is not a .gguf file.")
        return p

    repo_id, sep, fname = source.partition(":")
    if "/" not in repo_id or repo_id.startswith("/") or repo_id.startswith("."):
        raise ConvertError(
            f"'{source}' is neither an existing local .gguf file nor a "
            "HuggingFace repo id (expected 'org/repo' or 'org/repo:file.gguf')."
        )

    from huggingface_hub import snapshot_download

    if sep:
        # Explicit filename — fetch just that file.
        try:
            d = snapshot_download(repo_id, allow_patterns=[fname])
        except Exception as e:
            raise ConvertError(f"Failed to download '{fname}' from '{repo_id}': {e}")
        matches = sorted(Path(d).rglob("*.gguf"))
        if not matches:
            raise ConvertError(f"'{fname}' not found in '{repo_id}'.")
        return matches[0]

    try:
        d = snapshot_download(repo_id, allow_patterns=["*.gguf"])
    except Exception as e:
        raise ConvertError(f"Failed to download GGUF files from '{repo_id}': {e}")
    ggufs = sorted(Path(d).rglob("*.gguf"))
    if not ggufs:
        raise ConvertError(f"No .gguf files found in '{repo_id}'.")
    if len(ggufs) > 1:
        listing = "\n".join(f"    {g.name}" for g in ggufs)
        raise ConvertError(
            f"'{repo_id}' contains {len(ggufs)} GGUF files — pick one with "
            f"'{repo_id}:<filename>':\n{listing}"
        )
    return ggufs[0]


# ---------------------------------------------------------------------------
# config.json construction
# ---------------------------------------------------------------------------


def _build_config_from_gguf(reader: Any, tie_word_embeddings: bool) -> dict[str, Any]:
    """Rebuild an HF config dict from GGUF metadata kv pairs.

    The field mapping comes from transformers' own ``GGUF_CONFIG_MAPPING``
    (lazily imported — torch-free, and maintained upstream as new archs
    land), so our manual path tracks the same conventions transformers uses.
    """
    from transformers.integrations.ggml import (
        GGUF_CONFIG_DEFAULTS_MAPPING,
        GGUF_CONFIG_MAPPING,
    )

    kv_prefix, model_type = _kv_prefix_and_model_type(reader)
    arch_map = GGUF_CONFIG_MAPPING.get(model_type) or GGUF_CONFIG_MAPPING.get(kv_prefix)
    if arch_map is None:
        supported = ", ".join(sorted(k for k in GGUF_CONFIG_MAPPING if k != "general"))
        raise ConvertError(
            f"Unsupported GGUF architecture '{kv_prefix}'. "
            f"Known architectures: {supported}."
        )

    cfg: dict[str, Any] = {"model_type": model_type}
    for ggml_key, hf_key in arch_map.items():
        if hf_key is None:
            continue
        v = _field(reader, f"{kv_prefix}.{ggml_key}")
        if v is not None:
            cfg[hf_key] = v

    # GQA fallback: files without an explicit kv-head count use MHA.
    if "num_key_value_heads" not in cfg and "num_attention_heads" in cfg:
        cfg["num_key_value_heads"] = cfg["num_attention_heads"]

    # head_dim fallback. Upstream's qwen3/qwen2 tables drop
    # rope.dimension_count entirely, but mlx-lm's qwen3 ModelArgs *requires*
    # head_dim — and qwen3 GGUFs always carry attention.key_length. Order:
    # explicit key_length kv, then hidden_size // heads (the HF default).
    if "head_dim" not in cfg:
        key_length = _field(reader, f"{kv_prefix}.attention.key_length")
        if key_length is not None:
            cfg["head_dim"] = key_length
        elif "hidden_size" in cfg and "num_attention_heads" in cfg:
            cfg["head_dim"] = cfg["hidden_size"] // cfg["num_attention_heads"]

    # vocab_size fallback: count the embedded token list.
    if "vocab_size" not in cfg:
        tokens = _field(reader, "tokenizer.ggml.tokens")
        if tokens is not None:
            cfg["vocab_size"] = len(tokens)

    # RoPE scaling. Three cases, in order:
    #  1. Explicit rope.scaling.type in the GGUF → translate. For "llama3"
    #     the low/high freq factors are NOT stored in GGUF kv (llama.cpp
    #     hardcodes them), so spell out the family constants — mlx-lm's
    #     Llama3RoPE requires them.
    #  2. Llama-3.x fingerprint: GGUFs written by older converters omit
    #     rope.scaling.* entirely, but arch=llama + theta=500000 is unique
    #     to the Llama-3.1/3.2 family (llama.cpp keys llama3 scaling off
    #     the same signature). Without this block mlx-lm builds vanilla
    #     RoPE and the model visibly degenerates (verified on the real
    #     Llama-3.2-1B Q8_0 GGUF from hugging-quants).
    #  3. Otherwise no rope_scaling key at all.
    rope_type = _field(reader, f"{kv_prefix}.rope.scaling.type")
    rope_freq_base = _field(reader, f"{kv_prefix}.rope.freq_base")
    if rope_type == "llama3":
        factor = _field(reader, f"{kv_prefix}.rope.scaling.factor") or 32.0
        orig_ctx = _field(reader, f"{kv_prefix}.rope.scaling.orig_ctx_len") or 8192
        cfg["rope_scaling"] = {
            "rope_type": "llama3",
            "factor": float(factor),
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": int(orig_ctx),
        }
    elif rope_type:
        rope_scaling: dict[str, Any] = {"type": str(rope_type)}
        passthrough = {
            "factor": "factor",
            "attn_factor": "attn_factor",
            "beta_fast": "beta_fast",
            "beta_slow": "beta_slow",
            "orig_ctx_len": "original_max_position_embeddings",
        }
        for ggml_key, hf_key in passthrough.items():
            v = _field(reader, f"{kv_prefix}.rope.scaling.{ggml_key}")
            if v is not None:
                rope_scaling[hf_key] = v
        cfg["rope_scaling"] = rope_scaling
    elif (
        kv_prefix == "llama"
        and rope_freq_base is not None
        and float(rope_freq_base) == 500000.0
    ):
        cfg["rope_scaling"] = {
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }

    # Upstream per-arch defaults (e.g. qwen3_moe norm_topk_prob=True).
    for k, v in GGUF_CONFIG_DEFAULTS_MAPPING.get(model_type, {}).items():
        cfg.setdefault(k, v)

    cfg["tie_word_embeddings"] = tie_word_embeddings
    if cls := _ARCHITECTURES.get(model_type):
        cfg["architectures"] = [cls]
    return cfg


def _build_config_transformers(gguf_path: Path) -> dict[str, Any]:
    """Strategy ②: transformers' native GGUF config reader.

    Only reachable when torch is installed (upstream gates GGUF loading on
    ``is_torch_available()``); callers treat any exception as "fall through
    to the manual rebuild".
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(gguf_path.parent), gguf_file=gguf_path.name)
    return cfg.to_dict()


# ---------------------------------------------------------------------------
# Tokenizer construction
# ---------------------------------------------------------------------------


def _write_tokenizer_from_gguf(reader: Any, model_type: str, out_dir: Path) -> None:
    """Strategy ③: rebuild the tokenizer from embedded ``tokenizer.ggml.*``.

    Uses transformers' torch-free ``convert_gguf_tokenizer`` on a dict built
    straight from the GGUF fields, then saves a standard fast-tokenizer
    directory (tokenizer.json + tokenizer_config.json).
    """
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.ggml import (
        GGUF_TOKENIZER_MAPPING,
        convert_gguf_tokenizer,
    )

    tokenizer_dict: dict[str, Any] = {}
    for ggml_key, hf_key in GGUF_TOKENIZER_MAPPING["tokenizer"].items():
        v = _field(reader, f"tokenizer.{ggml_key}")
        if v is not None:
            tokenizer_dict[hf_key] = v
    if "tokens" not in tokenizer_dict:
        raise ConvertError(
            "GGUF has no embedded tokenizer (tokenizer.ggml.tokens missing) "
            "and no usable base-model fallback was found."
        )

    converter_key = _TOKENIZER_CONVERTER_KEY.get(model_type, "llama")
    fast_tok, extra_kwargs = convert_gguf_tokenizer(converter_key, tokenizer_dict)

    # llama-bpe (Llama-3 family) pre-tokenizer. transformers' GGUF converter
    # falls back to stock ByteLevel(use_regex=True) — the GPT-2 split
    # pattern — which segments punctuation/newline runs differently from
    # Llama-3's training-time tokenizer (verified: 601 vs 549 tokens on the
    # same sample, ~20% perplexity inflation on converted weights). The GGUF
    # records the family in tokenizer.ggml.pre; install the canonical
    # llama3 Split+ByteLevel sequence (as shipped in the official
    # Llama-3.x tokenizer.json) when we see it.
    if _field(reader, "tokenizer.ggml.pre") == "llama-bpe":
        from tokenizers import Regex, pre_tokenizers

        fast_tok.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(
                        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+"
                        r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+"
                        r"|\s+(?!\S)|\s+"
                    ),
                    behavior="isolated",
                    invert=False,
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=False, trim_offsets=True, use_regex=False
                ),
            ]
        )

    # BOS/EOS post-processing. transformers' GGUF converter leaves an EMPTY
    # TemplateProcessing on the BPE — it never prepends BOS, which silently
    # degrades models for which BOS is mandatory (verified: converted
    # Llama-3.2-1B looped degenerately until this was added; Qwen3 does not
    # use BOS and is unaffected). The GGUF tokenizer.ggml.add_bos_token /
    # add_eos_token flags are authoritative; when absent, mirror llama.cpp's
    # default (llama/gemma families add BOS; qwen2-style BPE does not).
    add_bos = _field(reader, "tokenizer.ggml.add_bos_token")
    if add_bos is None:
        pre = _field(reader, "tokenizer.ggml.pre") or ""
        add_bos = pre != "qwen2"
    add_eos = _field(reader, "tokenizer.ggml.add_eos_token") or False
    tokens_list = tokenizer_dict.get("tokens") or []
    specials: list[tuple[str, int]] = []
    if add_bos and (bos_id := tokenizer_dict.get("bos_token_id")) is not None:
        specials.append((tokens_list[int(bos_id)], int(bos_id)))
    if add_eos and (eos_id := tokenizer_dict.get("eos_token_id")) is not None:
        specials.append((tokens_list[int(eos_id)], int(eos_id)))
    if specials:
        from tokenizers import processors

        prefix = specials[0][0] if add_bos else None
        suffix = specials[-1][0] if add_eos else None
        single = " ".join(p for p in (prefix, "$A", suffix) if p)
        pair = " ".join(p for p in (prefix, "$A", "$B", suffix) if p)
        fast_tok.post_processor = processors.TemplateProcessing(
            single=single, pair=pair, special_tokens=specials
        )

    tok = PreTrainedTokenizerFast(tokenizer_object=fast_tok, **extra_kwargs)

    # transformers ≤5.12's GGUFLlamaConverter mis-assigns bos/eos (it reads
    # eos from bos_token_id, then swaps the two kwargs — see
    # transformers/integrations/ggml.py GGUFLlamaConverter). The GGUF's own
    # tokenizer.ggml.*_token_id fields are authoritative, so re-apply them
    # explicitly after construction; harmless when the converter got it right.
    for id_key, attr in (
        ("bos_token_id", "bos_token"),
        ("eos_token_id", "eos_token"),
        ("unk_token_id", "unk_token"),
        ("pad_token_id", "pad_token"),
    ):
        idx = tokenizer_dict.get(id_key)
        if idx is not None and 0 <= int(idx) < len(tokens_list):
            setattr(tok, attr, tokens_list[int(idx)])

    # Carry the chat template when the source embeds one (instruct tunes).
    chat_template = _field(reader, "tokenizer.chat_template")
    if isinstance(chat_template, str) and chat_template.strip():
        tok.chat_template = chat_template

    tok.save_pretrained(str(out_dir))


def _write_tokenizer_transformers(gguf_path: Path, out_dir: Path) -> None:
    """Strategy ②: transformers' native GGUF tokenizer reader (needs torch)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(gguf_path.parent), gguf_file=gguf_path.name)
    tok.save_pretrained(str(out_dir))


# ---------------------------------------------------------------------------
# Base-model (strategy ①) acquisition
# ---------------------------------------------------------------------------

_BASE_AUX_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
]


def _base_model_repo(reader: Any) -> str | None:
    """Extract an HF repo id from ``general.base_model.*`` provenance."""
    count = _field(reader, "general.base_model.count") or 0
    for i in range(max(int(count), 1)):
        repo_url = _field(reader, f"general.base_model.{i}.repo_url")
        if isinstance(repo_url, str):
            m = re.match(r"https?://huggingface\.co/([^/]+/[^/?#]+)", repo_url)
            if m:
                return m.group(1)
        org = _field(reader, f"general.base_model.{i}.organization")
        name = _field(reader, f"general.base_model.{i}.name")
        if org and name:
            candidate = f"{org}/{name}"
            if re.fullmatch(r"[\w.-]+/[\w.-]+", candidate):
                return candidate
    return None


def _base_repo_from_hub_tags(repo_id: str) -> str | None:
    """Extract a base-model repo from an HF repo's ``base_model:*`` tags.

    Fine-tune authors increasingly record provenance in hub tags rather than
    GGUF kv metadata (verified: neither Qwen's nor yuxinlu1's gemma4 GGUFs
    carry ``general.base_model.*``, but both repos tag the base). The
    ``base_model:quantized:`` and ``base_model:adapter:`` namespaces point at
    derived artifacts, not the architecture source — skip them.
    """
    from huggingface_hub import HfApi

    try:
        tags = HfApi().model_info(repo_id).tags or []
    except Exception:
        return None
    for tag in tags:
        if tag.startswith(("base_model:quantized:", "base_model:adapter:")):
            continue
        if tag.startswith("base_model:"):
            candidate = tag.split(":", 1)[1]
            if re.fullmatch(r"[\w.-]+/[\w.-]+", candidate):
                return candidate
    return None


def _flatten_unified_text_config(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten unified (multimodal) configs that nest text params.

    google/gemma-4-* publishes top ``model_type=gemma4`` with the
    language-model fields (``num_key_value_heads`` scalar pair,
    ``layer_types``, ``num_kv_shared_layers``, …) inside ``text_config``.
    A text-only artifact needs them at top level so downstream consumers —
    this converter's own planning pass and mlx-lm's ``gemma4_text``
    ModelArgs — can read them without unified-config awareness.
    """
    text_cfg = config.get("text_config")
    if not isinstance(text_cfg, dict):
        return config
    flat = {k: v for k, v in config.items() if k != "text_config"}
    flat.update(text_cfg)
    return flat


def _fetch_base_aux(repo_id: str, out_dir: Path) -> dict[str, Any] | None:
    """Download config/tokenizer artifacts from the base model repo.

    Returns the parsed base ``config.json`` on success, ``None`` when the
    repo is unreachable or lacks the minimum artifact set (callers fall
    through to the next strategy).
    """
    from huggingface_hub import snapshot_download

    try:
        d = snapshot_download(repo_id, allow_patterns=_BASE_AUX_PATTERNS)
    except Exception:
        return None
    src = Path(d)
    has_config = (src / "config.json").is_file()
    has_tokenizer = (src / "tokenizer.json").is_file() or (
        (src / "tokenizer.model").is_file()
        or ((src / "vocab.json").is_file() and (src / "merges.txt").is_file())
    )
    if not has_tokenizer:
        return None
    for pattern in _BASE_AUX_PATTERNS:
        f = src / pattern
        if f.is_file():
            shutil.copy2(f, out_dir / pattern)
    if not has_config:
        return {}
    try:
        with open(out_dir / "config.json") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Weight streaming
# ---------------------------------------------------------------------------


def _build_name_map(reader: Any) -> tuple[dict[str, str], str]:
    """Invert gguf's tensor-name map to ggml → HF for this file's arch.

    ``get_tensor_name_map`` is built for the HF → GGUF direction: its
    ``mapping`` dict keys are *both* ggml names (mapping to themselves) and
    every known HF convention for each tensor type. We group the HF
    candidates per ggml name and pick the convention mlx-lm expects:
    ``model.*``-prefixed (llama-hf style), with ``mlp.*`` preferred for the
    MoE archs that use it, and bare ``lm_head`` for the output projection.

    Returns ``(ggml_base_name → hf_base_name, human-readable arch)``.
    """
    from gguf import MODEL_ARCH_NAMES, get_tensor_name_map

    kv_prefix = _gguf_arch(reader)
    arch = next((a for a, n in MODEL_ARCH_NAMES.items() if n == kv_prefix), None)
    if arch is None:
        raise ConvertError(
            f"gguf {kv_prefix!r} architecture has no tensor name mapping "
            "(unknown to gguf-py's MODEL_ARCH_NAMES)."
        )
    n_blocks = int(_field(reader, f"{kv_prefix}.block_count") or 0)
    name_map = get_tensor_name_map(arch, n_blocks)

    candidates: dict[str, list[str]] = {}
    for key, (_ttype, ggml_name) in name_map.mapping.items():
        if key == ggml_name:
            continue  # self-mapping entry, not an HF candidate
        candidates.setdefault(ggml_name, []).append(key)

    prefer_mlp = kv_prefix in _MLP_STYLE_MOE_GGUF_ARCHES

    def pick(cands: list[str], ggml_base: str) -> str | None:
        # The output projection must be bare ``lm_head`` (what mlx-lm
        # instantiates for untied models). gguf's candidate table also
        # carries legacy names like ``model.transformer.ff_out`` that the
        # generic "model.* first" rule below would wrongly prefer — the
        # Dolphin3.0 llama GGUF (untied, vocab 128258) hit exactly this.
        # Only exercised by untied models: tied ones have no output.weight.
        if ggml_base == "output" and "lm_head" in cands:
            return "lm_head"
        # Attention q/k-norm: mlx-lm (and HF qwen3/cohere) call these
        # ``q_norm``/``k_norm``, but gguf's candidate table lists the
        # persimmon ``*_layernorm`` variant first, so the generic
        # "first model.*" rule below would pick a name mlx-lm's qwen3
        # rejects at load time. Prefer the exact-leaf match when present.
        leaf = ggml_base.rsplit(".", 1)[-1]  # blk.N.attn_q_norm → attn_q_norm
        if leaf in ("attn_q_norm", "attn_k_norm"):
            wanted = leaf.removeprefix("attn_")
            exact = [c for c in cands if c.endswith(f".{wanted}")]
            if exact:
                return exact[0]
        if prefer_mlp:
            mlp = [c for c in cands if ".mlp." in c]
            if mlp:
                return mlp[0]
        model_pref = [c for c in cands if c.startswith("model.")]
        if model_pref:
            return model_pref[0]
        if "lm_head" in cands:
            return "lm_head"
        return None

    mapping: dict[str, str] = {}
    for ggml_base, cands in candidates.items():
        if hf_base := pick(cands, ggml_base):
            mapping[ggml_base] = hf_base
    return mapping, kv_prefix


def _numpy_shape(gguf_tensor: Any) -> tuple[int, ...]:
    """Logical numpy shape of a GGUF tensor (ggml stores dims reversed)."""
    return tuple(int(d) for d in gguf_tensor.shape[::-1])


def _dequantize_tensor(gguf_tensor: Any) -> Any:
    """Dequantize one reader tensor to a float32 numpy array (logical shape)."""
    import numpy as np
    from gguf.quants import dequantize

    arr = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
    shape = _numpy_shape(gguf_tensor)
    if tuple(arr.shape) != shape:
        arr = np.reshape(arr, shape)
    return arr


def _reverse_permute_qk(weights: Any, n_head: int, n_kv_heads: int) -> Any:
    """Undo llama.cpp's q/k rope-layout permutation.

    llama.cpp's HF→GGUF conversion permutes attention q/k weights into its
    interleaved-RoPE layout (``convert_hf_to_gguf.py`` ``permute()``), so a
    straight GGUF→HF read must apply the inverse — verbatim from
    transformers' ``LlamaTensorProcessor._reverse_permute_weights``. Without
    this the converted model "almost works": it produces topical words and
    immediately degenerates into loops (observed on real Llama-3.2-1B: ppl
    624 vs 7.3 for the official MLX weights; +2.65% requant delta hid the
    damage because both report sides shared the same broken weights).
    """
    if n_head != n_kv_heads:
        n_head = n_kv_heads
    dim = weights.shape[0] // n_head // 2
    w = weights.reshape(n_head, dim, 2, *weights.shape[1:])
    return w.swapaxes(2, 1).reshape(weights.shape)


def _postprocess_tensor(
    arr: Any, hf_name: str, model_type: str, n_head: int, n_kv_heads: int
) -> Any:
    """Per-arch value adjustments, mirroring transformers' TENSOR_PROCESSORS.

    * llama: reverse-permute ``self_attn.{q,k}_proj`` (see above).
    * gemma2/3 (text): ggml stores RMSNorm weights as ``1 + w`` (gemma's
      kernel folds the +1 into the scale), HF/mlx-lm store raw ``w``.
      **gemma4 is NOT adjusted**: llama.cpp stopped folding +1 there —
      measured on yuxinlu1/gemma-4-12B-coder Q8_0 vs google/gemma-4-12B-it,
      ``layers.0.input_layernorm.weight`` means match to 0.0000 (GGUF holds
      raw ``w``). Subtracting 1 anyway zeroed near-zero entries and left
      the model outputting ``<unused>`` gibberish (ppl 3.7e19).
    """
    if model_type == "llama":
        if "self_attn.q_proj." in hf_name:
            return _reverse_permute_qk(arr, n_head, n_head)
        if "self_attn.k_proj." in hf_name:
            return _reverse_permute_qk(arr, n_head, n_kv_heads)
    elif model_type in ("gemma2", "gemma3_text"):
        if hf_name.endswith("norm.weight"):
            return arr - 1
    return arr


def _scalar_int(value: Any, default: int) -> int:
    """Coerce a config value to int; some archs store per-layer lists.

    gemma4-class GGUFs record ``num_key_value_heads`` as a per-layer list
    (sliding=8 / full=1 for the 12B). The head counts here only feed the
    llama-family q/k reverse-permutation, which is gated off for every
    other arch — collapsing a list to its max is safe for that sole use.
    """
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return int(max(value)) if value else default
    return int(value)


def _adjust_hf_name(hf_name: str, model_type: str) -> str:
    """Per-arch HF-name corrections that gguf's mapping gets wrong for mlx-lm.

    gemma4: transformers' convention (what gguf's table yields) writes the
    MLP as ``feed_forward.*`` and the per-layer scalar as
    ``layer_scalar.weight``, but mlx-lm's ``gemma4_text`` instantiates
    ``mlp.*`` and stores the scalar *bare* — official
    mlx-community/gemma-4-12B-it: ``layers.N.mlp.up_proj`` and
    ``layers.N.layer_scalar`` (no ``.weight``). Loading otherwise rejects
    both as "parameters not in model".
    """
    if model_type == "gemma4_text":
        hf_name = hf_name.replace(".feed_forward.", ".mlp.")
        if hf_name.endswith(".layer_scalar.weight"):
            hf_name = hf_name[: -len(".weight")]
    return hf_name


def _stream_weights(
    reader: Any,
    hf_dir: Path,
    dtype: str,
    config: dict[str, Any],
) -> tuple[int, list[tuple[str, str]], int]:
    """Dequantize + rename + shard-write every mappable tensor.

    Two passes over the tensor list: a metadata-only planning pass (name
    mapping, skip classification, shard assignment) and a streaming write
    pass. Planning first is what lets multi-shard outputs use the canonical
    ``-of-0000M`` naming without holding more than one shard in RAM.

    Returns ``(tensors_written, skipped[(ggml_name, reason)], bytes_written)``.
    """
    import mlx.core as mx
    from tqdm import tqdm

    mapping, _ = _build_name_map(reader)
    mx_dtype = getattr(mx, dtype)
    itemsize = _DTYPE_ITEMSIZE[dtype]
    model_type = str(config.get("model_type", ""))
    n_head = _scalar_int(config.get("num_attention_heads"), 1)
    n_kv_heads = _scalar_int(config.get("num_key_value_heads"), n_head)

    # ---- planning pass -------------------------------------------------
    plan: list[tuple[str, int, int]] = []  # (hf_name, tensor_idx, out_nbytes)
    skipped: list[tuple[str, str]] = []
    seen_hf: set[str] = set()
    for idx, t in enumerate(reader.tensors):
        base, _, suffix = t.name.rpartition(".")
        if not base or suffix not in ("weight", "bias"):
            base, suffix = t.name, "weight"
        if t.name in _SKIP_GGML_TENSOR_NAMES or any(
            p.match(t.name) for p in _SKIP_GGML_TENSOR_PATTERNS
        ):
            skipped.append((t.name, "runtime-computed buffer (RoPE inv_freq)"))
            continue
        hf_base = mapping.get(base)
        if hf_base is None:
            skipped.append((t.name, "no HF-convention name known for this tensor"))
            continue
        hf_name = _adjust_hf_name(f"{hf_base}.{suffix}", model_type)
        if hf_name in seen_hf:
            skipped.append((t.name, f"duplicate mapping to {hf_name}"))
            continue
        seen_hf.add(hf_name)
        plan.append((hf_name, idx, int(t.n_elements) * itemsize))

    if not plan:
        raise ConvertError(
            "No convertible tensors found — the architecture's tensors all "
            "failed name mapping. Is this a supported text model GGUF?"
        )

    # Greedy in-order shard assignment: start a new shard when the current
    # one would overflow the target. Tensors are never split, so a single
    # tensor larger than the target gets a shard of its own.
    assignments: list[int] = []
    shard_sizes: list[int] = [0]
    for _hf_name, _idx, nbytes in plan:
        if shard_sizes[-1] > 0 and shard_sizes[-1] + nbytes > _SHARD_TARGET_BYTES:
            shard_sizes.append(0)
        assignments.append(len(shard_sizes) - 1)
        shard_sizes[-1] += nbytes
    n_shards = len(shard_sizes)

    def shard_name(i: int) -> str:
        if n_shards == 1:
            return "model.safetensors"
        return f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"

    # ---- streaming write pass ------------------------------------------
    weight_map: dict[str, str] = {}
    current: dict[str, Any] = {}
    current_shard = -1
    total_bytes = 0
    written = 0

    def flush() -> None:
        nonlocal current
        if not current:
            return
        fname = shard_name(current_shard)
        mx.save_safetensors(str(hf_dir / fname), current, metadata={"format": "mlx"})
        print(f"  wrote {fname} ({len(current)} tensors)")
        current = {}

    bar = tqdm(plan, desc="  dequantizing", unit="tensor", leave=False)
    for (hf_name, tensor_idx, _nbytes), shard_idx in zip(bar, assignments):
        if shard_idx != current_shard:
            flush()
            current_shard = shard_idx
        t = reader.get_tensor(tensor_idx)
        arr = _dequantize_tensor(t)
        arr = _postprocess_tensor(arr, hf_name, model_type, n_head, n_kv_heads)
        current[hf_name] = mx.array(arr).astype(mx_dtype)
        weight_map[hf_name] = shard_name(shard_idx)
        total_bytes += _nbytes
        written += 1
    flush()
    bar.close()

    if n_shards > 1:
        index = {
            "metadata": {"total_size": total_bytes},
            "weight_map": weight_map,
        }
        with open(hf_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)

    return written, skipped, total_bytes


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------


def _eval_ppl_for_dir(model_dir: Path, tokens: Any, seq_len: int) -> float:
    """Load a model directory and evaluate perplexity on the given token ids."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.perplexity import eval_ppl

    model, _tok = load(str(model_dir))
    n_rows = tokens.size // seq_len
    data = tokens[: n_rows * seq_len].reshape(n_rows, seq_len)
    ppl, _se = eval_ppl(model, data, batch_size=min(_REPORT_ROWS, n_rows))
    del model
    try:
        mx.clear_cache()
    except Exception:
        pass
    return ppl


def _run_report(
    reference_dir: Path,
    candidate_dir: Path,
    candidate_label: str,
) -> dict[str, float]:
    """Compare perplexity: full-precision reference vs. quantized candidate.

    Tokenizes the bundled public-domain sample once (with the reference
    model's tokenizer — both dirs share tokenizer files) and runs
    ``mlx_lm.perplexity.eval_ppl`` on each model in turn.
    """
    import mlx.core as mx
    from mlx_lm import load

    # Tokenize with the reference tokenizer, then free it — model loads
    # below are the memory-heavy part and run one at a time.
    _m, tokenizer = load(str(reference_dir), lazy=True)
    del _m
    ids = tokenizer.encode(_REPORT_TEXT)

    # Clamp the sequence length to the reference model's context window so
    # tiny-context models (and tiny test fixtures) still produce a row.
    max_ctx = 1 << 30
    try:
        with open(reference_dir / "config.json") as f:
            max_ctx = int(json.load(f).get("max_position_embeddings", max_ctx))
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    seq_len = max(8, min(_REPORT_SEQ_LEN, max_ctx, len(ids)))
    n_rows = max(1, min(_REPORT_ROWS, len(ids) // seq_len))
    tokens = mx.array(ids[: n_rows * seq_len])

    ppl_ref = _eval_ppl_for_dir(reference_dir, tokens, seq_len)
    ppl_cand = _eval_ppl_for_dir(candidate_dir, tokens, seq_len)

    delta_pct = (ppl_cand - ppl_ref) / ppl_ref * 100.0
    return {
        "tokens": int(tokens.size),
        "ppl_reference": ppl_ref,
        "ppl_candidate": ppl_cand,
        "delta_pct": delta_pct,
        "candidate_label": candidate_label,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _default_out_dir(gguf_path: Path, bits: int | None) -> Path:
    stem = gguf_path.stem
    suffix = f"-{bits}bit" if bits is not None else ""
    return Path.cwd() / f"{stem}-mlx{suffix}"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def convert_gguf(
    source: str,
    out_dir: str | Path | None = None,
    bits: int | None = None,
    group_size: int = _DEFAULT_GROUP_SIZE,
    dtype: str = "bfloat16",
    report: bool | None = None,
) -> ConvertResult:
    """Convert a GGUF model to an MLX-servable directory.

    ``bits=None`` produces a full-precision (``dtype``) MLX directory;
    ``bits=N`` additionally re-quantizes through ``mlx_lm.convert``.
    ``report=None`` enables the perplexity report exactly when ``bits`` is
    given (comparing the full-precision intermediate against the quantized
    product); a report without ``--bits`` is meaningless and skipped.
    """
    t0 = time.monotonic()
    if dtype not in _DTYPES:
        raise ConvertError(
            f"--dtype must be one of {', '.join(_DTYPES)} (got {dtype!r})"
        )
    if bits is not None and bits not in _QUANT_BITS:
        raise ConvertError(f"--bits must be one of {_QUANT_BITS} (got {bits})")

    gguf_path = _resolve_source(source)
    out = Path(out_dir) if out_dir is not None else _default_out_dir(gguf_path, bits)
    if out.exists() and any(out.iterdir()):
        raise ConvertError(f"Output directory '{out}' already exists and is not empty.")
    if bits is None:
        out.mkdir(parents=True, exist_ok=True)
    else:
        # mlx_lm.convert refuses to write into an existing directory, so it
        # must create `out` itself; we only need the parent for staging.
        out.parent.mkdir(parents=True, exist_ok=True)

    import gguf

    reader = gguf.GGUFReader(str(gguf_path))
    kv_prefix, model_type = _kv_prefix_and_model_type(reader)
    tie_word_embeddings = not any(t.name == "output.weight" for t in reader.tensors)
    n_tensors = len(reader.tensors)
    print(f"  Source : {gguf_path}")
    print(f"  Arch   : {kv_prefix} (model_type={model_type}), {n_tensors} tensors")

    # When re-quantizing, the HF full-precision directory is an intermediate
    # staged in a temp dir (kept alive for the optional report, then
    # removed); otherwise it *is* the output.
    staging_ctx = (
        tempfile.TemporaryDirectory(prefix=".rapid-mlx-convert-", dir=out.parent)
        if bits is not None
        else None
    )
    hf_dir = Path(staging_ctx.name) if staging_ctx is not None else out

    try:
        # ---- config.json -------------------------------------------------
        # Strategy order: base-model repo → transformers native → manual
        # rebuild. Each strategy is validated before winning; the manual
        # path is deterministic and always available.
        config: dict[str, Any] | None = None
        config_source = ""
        base_cfg: dict[str, Any] | None = None
        base_repo = _base_model_repo(reader)
        if base_repo is None:
            # GGUF kv carries no provenance (the common case for community
            # quants) — fall back to the hub repo's base_model:* tags. Any
            # non-local source form counts: ``org/repo`` and
            # ``org/repo:file.gguf`` alike (the earlier endswith(".gguf")
            # check wrongly excluded the latter — the gemma4 sample hit it).
            source_str = str(source)
            if not Path(source_str).is_file():
                hub_candidate = source_str.split(":", 1)[0]
                if "/" in hub_candidate:
                    base_repo = _base_repo_from_hub_tags(hub_candidate)
        if base_repo is not None:
            base_cfg = _fetch_base_aux(base_repo, hf_dir)
        if base_cfg:
            n_layers_gguf = _field(reader, f"{kv_prefix}.block_count")
            if n_layers_gguf is None or base_cfg.get("num_hidden_layers") in (
                None,
                n_layers_gguf,
            ):
                config = _flatten_unified_text_config(base_cfg)
                config["model_type"] = model_type
                config_source = f"base model {base_repo}"
        if config is None:
            try:
                config = _build_config_transformers(gguf_path)
                config_source = "transformers native GGUF reader"
            except Exception:
                config = None
        if config is None:
            config = _build_config_from_gguf(reader, tie_word_embeddings)
            config_source = "GGUF metadata"
        # The written dtype is authoritative regardless of config origin —
        # mlx_lm.convert keys its dtype cast off config["torch_dtype"].
        config["torch_dtype"] = dtype
        with open(hf_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")

        # ---- tokenizer ---------------------------------------------------
        tokenizer_source = ""
        has_tok = (hf_dir / "tokenizer.json").is_file() or (
            hf_dir / "tokenizer.model"
        ).is_file()
        if has_tok:
            tokenizer_source = f"base model {base_repo}"
        else:
            try:
                _write_tokenizer_transformers(gguf_path, hf_dir)
                tokenizer_source = "transformers native GGUF reader"
            except Exception:
                _write_tokenizer_from_gguf(reader, model_type, hf_dir)
                tokenizer_source = "GGUF metadata"

        # generation_config.json: only written when the source provided one
        # (base-model fetch) — otherwise emit a minimal bos/eos stub from the
        # GGUF tokenizer fields so mlx-lm's loader has its eos_token_id.
        if not (hf_dir / "generation_config.json").is_file():
            gen_cfg: dict[str, Any] = {}
            for ggml_key, hf_key in (
                ("bos_token_id", "bos_token_id"),
                ("eos_token_id", "eos_token_id"),
                ("padding_token_id", "pad_token_id"),
            ):
                v = _field(reader, f"tokenizer.ggml.{ggml_key}")
                if v is not None:
                    gen_cfg[hf_key] = v
            if gen_cfg:
                with open(hf_dir / "generation_config.json", "w") as f:
                    json.dump(gen_cfg, f, indent=2)
                    f.write("\n")

        # ---- weights -----------------------------------------------------
        written, skipped, _total = _stream_weights(reader, hf_dir, dtype, config)
        print(f"  Config : {config_source}")
        print(f"  Tokens : {tokenizer_source}")
        print(f"  Weights: {written} tensors → {dtype}")
        for name, reason in skipped:
            print(f"  skipped: {name} ({reason})", file=sys.stderr)

        # ---- optional re-quantization ------------------------------------
        if bits is not None:
            import mlx_lm

            # 1-bit sources (IQ1_*/TQ*_0): dequantization to bf16 already
            # cost most of the signal; squeezing the remainder through a
            # second quantizer compounds the error. Measured on
            # Qwen3-0.6B IQ1_M: bf16 ppl 747 → 3-bit ppl 19927 (26× worse).
            # Warn loudly but don't block — the user may have a reason.
            one_bit = _one_bit_source_quant(reader)
            if one_bit is not None:
                print(
                    f"  WARNING: source is {one_bit} (1-bit quantization). "
                    f"Re-quantizing to {bits}-bit stacks quantization error "
                    "on quantization error\n"
                    "  (measured: Qwen3-0.6B IQ1_M bf16 ppl 747 vs 3-bit "
                    "ppl 19927).\n"
                    "  Recommendation: rerun without --bits for a plain "
                    f"--dtype {dtype} artifact instead.",
                    file=sys.stderr,
                )

            print(f"  Re-quantizing to {bits}-bit (group_size={group_size}) …")
            mlx_lm.convert(
                str(hf_dir),
                str(out),
                quantize=True,
                q_bits=bits,
                q_group_size=group_size,
            )

        # ---- optional quality report -------------------------------------
        want_report = report if report is not None else bits is not None
        report_data = None
        if want_report and bits is None:
            print("  (report skipped: --report only applies together with --bits)")
        elif want_report:
            print("  Quality report (perplexity on bundled sample) …")
            report_data = _run_report(hf_dir, out, f"{bits}-bit")
            print(
                f"    full-precision ({dtype}): {report_data['ppl_reference']:.3f}\n"
                f"    {bits}-bit quantized    : {report_data['ppl_candidate']:.3f} "
                f"({report_data['delta_pct']:+.2f}%)"
            )

        return ConvertResult(
            out_dir=out,
            model_type=model_type,
            config_source=config_source,
            tokenizer_source=tokenizer_source,
            tensors_written=written,
            skipped=skipped,
            size_bytes=_dir_size(out),
            bits=bits,
            report=report_data,
            seconds=time.monotonic() - t0,
        )
    finally:
        if staging_ctx is not None:
            staging_ctx.cleanup()


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def _format_bytes(n: int) -> str:
    """1-decimal IEC rendering, mirroring ``vllm_mlx.cli._format_bytes``."""
    if n <= 0:
        return "0 B"
    for unit, exp in (("TiB", 4), ("GiB", 3), ("MiB", 2), ("KiB", 1)):
        if n >= 1024**exp:
            return f"{n / 1024**exp:.1f} {unit}"
    return f"{n} B"


def convert_command(args: Any) -> None:
    """``rapid-mlx convert`` entry point (dispatched from ``cli.main``)."""
    try:
        result = convert_gguf(
            args.source,
            out_dir=getattr(args, "output", None),
            bits=getattr(args, "bits", None),
            group_size=getattr(args, "group_size", _DEFAULT_GROUP_SIZE),
            dtype=getattr(args, "dtype", "bfloat16"),
            report=getattr(args, "report", None),
        )
    except ConvertError as e:
        print(f"\n  Error: {e}")
        sys.exit(1)

    print("\n  Conversion complete.")
    print(f"    Output : {result.out_dir}  ({_format_bytes(result.size_bytes)})")
    print(
        f"    Tensors: {result.tensors_written} written, {len(result.skipped)} skipped"
    )
    if result.bits is not None:
        print(f"    Quant  : {result.bits}-bit (MLX affine)")
    print(f"    Time   : {result.seconds:.1f}s")
    print("\n  Next steps:")
    print(f"    rapid-mlx serve {result.out_dir}")
    print(f"    rapid-mlx chat {result.out_dir}")
