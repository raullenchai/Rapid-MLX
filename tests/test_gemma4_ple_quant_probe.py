# SPDX-License-Identifier: Apache-2.0
"""Guard: the Gemma 4 Per-Layer-Embedding (PLE) table is never low-bit quantized.

Gemma 4 (Gemma-3n lineage) e2b/e4b "altup" variants carry a Per-Layer-
Embedding table — ``embed_tokens_per_layer``, a large ``nn.Embedding``
(``vocab_size_per_layer_input x num_hidden_layers*hidden_size_per_layer_input``)
whose rows feed every decoder layer's per-layer input gate
(``vllm_mlx/models/gemma4_vendored/language.py`` ``get_per_layer_inputs`` /
``project_per_layer_inputs``). Early 4-bit conversions quantized this table
at the model default (4-bit) and the model emitted garbage — the PLE rows
are too information-dense to survive 4-bit affine quantization.

mlx-community's shipped checkpoints never 4-bit this table:

* the e2b/e4b **8-bit** builds keep ``embed_tokens_per_layer`` at 8-bit
  (verified 2026-07-14 against ``gemma-4-e2b-it-8bit``: the table carries
  ``.scales``/``.biases`` companions and the config quant block is 8-bit), and
* the dense large sizes (12B / 26B-A4B / 31B) set
  ``hidden_size_per_layer_input == 0`` and have no PLE table at all
  (verified: ``gemma-4-26b-a4b-it-4bit`` has zero ``embed_tokens_per_layer``
  keys).

The invariant that guards against the garbage-output regression is enforced
in ``LanguageModel.quant_predicate`` (the hook ``mlx_vlm.convert`` calls
per-module at conversion time, ``mlx_vlm/convert.py`` ``base_quant_predicate``):
the PLE table must be pinned to a safe (>=8-bit) width, NOT the model default
that a 4-bit convert would otherwise apply. ``nn.Embedding`` DOES expose
``to_quantized`` (asserted below), so without the explicit pin the predicate's
fall-through ``return True`` WOULD 4-bit-quantize it — this test is
load-bearing, not a no-op.

These are static/structural probes: they walk ``quant_predicate`` over a
freshly-built (tiny, no-weights) vendored ``LanguageModel`` and assert the
predicate's decision for the PLE table. No checkpoint download, no model
weights, no inference.
"""

from __future__ import annotations

import mlx.nn as nn
import pytest

from vllm_mlx.models.gemma4_vendored.config import TextConfig
from vllm_mlx.models.gemma4_vendored.language import LanguageModel

# Minimum bit-width the PLE table may be quantized to. 4-bit corrupts it
# (the historical garbage-output failure); 8-bit is what mlx-community ships.
MIN_SAFE_PLE_BITS = 8

# PLE-bearing size shapes (Gemma-3n lineage e-series). ``num_kv_shared_layers``
# is orthogonal to PLE; kept small here so builds stay cheap. The dense large
# sizes (12B/26B/31B) have hidden_size_per_layer_input == 0 -> no PLE table,
# covered separately by ``test_dense_variant_has_no_ple_table``.
PLE_SIZES = [
    ("E2B-like", 4),
    ("E4B-like", 6),
]


def _build_ple_text_config(num_hidden_layers: int) -> TextConfig:
    """A tiny vendored ``TextConfig`` that still builds a PLE table.

    ``hidden_size_per_layer_input > 0`` is what turns the PLE table on
    (see ``Gemma4TextModel.__init__``). Every other dim is shrunk so
    ``LanguageModel(tc)`` allocates a trivially-small embedding instead of
    the real 262144-row table.
    """
    return TextConfig.from_dict(
        {
            "num_hidden_layers": num_hidden_layers,
            "num_kv_shared_layers": 0,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "vocab_size": 64,
            "vocab_size_per_layer_input": 64,
            "hidden_size_per_layer_input": 8,  # > 0 => PLE table exists
            "use_double_wide_mlp": False,
        }
    )


def _ple_module_paths(lm: LanguageModel) -> list[tuple[str, nn.Module]]:
    """All ``embed_tokens_per_layer`` (PLE) modules and their dotted paths."""
    return [
        (path, mod)
        for path, mod in lm.named_modules()
        if "embed_tokens_per_layer" in path
    ]


def _resolved_bits(predicate_result) -> int | None:
    """Bit-width a ``quant_predicate`` result would quantize a module to.

    * ``False``            -> not quantized at all (``None``)
    * ``dict`` w/ ``bits`` -> that explicit bit-width
    * ``True``             -> the model *default* bits. At conversion time
      ``mlx_vlm.convert`` / ``nn.quantize`` apply the ``--bits`` the caller
      chose, so a 4-bit convert makes this 4. We map ``True`` -> 4 to model
      exactly the dangerous "inherits the low-bit default" case this guard
      exists to reject.
    """
    if predicate_result is False:
        return None
    if isinstance(predicate_result, dict):
        return int(predicate_result["bits"])
    if predicate_result is True:
        return 4  # worst-case default for a 4-bit conversion
    raise AssertionError(f"unexpected predicate result: {predicate_result!r}")


def test_nn_embedding_is_quantizable_so_guard_is_load_bearing():
    """Sanity: ``nn.Embedding`` exposes ``to_quantized``.

    If it did not, ``quant_predicate``'s ``hasattr(m, 'to_quantized')`` gate
    would already exclude the PLE table and this whole guard would be a
    no-op. It IS quantizable, so the explicit pin below is doing real work.
    """
    emb = nn.Embedding(64, 32)
    assert hasattr(emb, "to_quantized")


@pytest.mark.parametrize("label,num_layers", PLE_SIZES)
def test_ple_table_not_low_bit_quantized(label, num_layers):
    """The PLE table's ``quant_predicate`` decision must be >= 8-bit.

    This is the actual anti-regression invariant: a future change to
    ``quant_predicate`` (or a blanket ``return True``) that lets the PLE
    table inherit a 4-bit conversion default fails HERE, before it can ship
    a garbage-output checkpoint.
    """
    tc = _build_ple_text_config(num_layers)
    lm = LanguageModel(tc)
    predicate = lm.quant_predicate

    ple_modules = _ple_module_paths(lm)
    # The table must actually exist for this shape, else the test is vacuous.
    assert ple_modules, f"{label}: expected an embed_tokens_per_layer module"

    for path, mod in ple_modules:
        assert hasattr(mod, "to_quantized"), (
            f"{label}: {path} unexpectedly not quantizable"
        )
        result = predicate(path, mod)
        bits = _resolved_bits(result)
        # ``None`` == excluded from quantization == kept full precision (fp),
        # which is strictly safer than any quantization. Only a *low-bit*
        # quantization corrupts the PLE table, so fp is accepted here.
        if bits is None:
            continue
        assert bits >= MIN_SAFE_PLE_BITS, (
            f"{label}: PLE table {path} would be {bits}-bit quantized "
            f"(predicate result {result!r}). 4-bit PLE produces garbage output; "
            f"pin it to >= {MIN_SAFE_PLE_BITS}-bit in "
            f"LanguageModel.quant_predicate."
        )


@pytest.mark.parametrize("label,num_layers", PLE_SIZES)
def test_regular_linear_still_takes_default_bits(label, num_layers):
    """Guard is scoped: ordinary decoder Linears are NOT force-pinned.

    The fix must only protect the PLE table. A regular MLP/attention Linear
    must still return ``True`` (take the conversion's default bits) so 4-bit
    models stay 4-bit everywhere they should. This keeps the guard from
    silently ballooning model size.
    """
    tc = _build_ple_text_config(num_layers)
    lm = LanguageModel(tc)
    predicate = lm.quant_predicate

    checked = 0
    for path, mod in lm.named_modules():
        if not hasattr(mod, "to_quantized"):
            continue
        # Only ordinary attention/mlp Linears — skip PLE, router, and the
        # bare-fp altup projection which have deliberate special handling.
        if any(s in path for s in ("embed_tokens_per_layer", "router", "per_layer")):
            continue
        if path.endswith(("o_proj", "q_proj", "k_proj", "v_proj")) or path.endswith(
            ("gate_proj", "up_proj", "down_proj")
        ):
            assert predicate(path, mod) is True, (
                f"{label}: {path} should take default bits (True), "
                f"got {predicate(path, mod)!r}"
            )
            checked += 1
    assert checked > 0, f"{label}: no ordinary Linear modules were checked"


def test_dense_variant_has_no_ple_table():
    """Negative control: dense sizes (12B/26B/31B) build no PLE table.

    They set ``hidden_size_per_layer_input == 0``; the concern simply does
    not apply. Documents *why* the guard targets only the e-series and
    protects against a future config change that resurrects the table
    without protection.
    """
    tc = TextConfig.from_dict(
        {
            "num_hidden_layers": 4,
            "num_kv_shared_layers": 0,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "vocab_size": 64,
            "vocab_size_per_layer_input": 64,
            "hidden_size_per_layer_input": 0,  # dense => no PLE
            "use_double_wide_mlp": False,
        }
    )
    lm = LanguageModel(tc)
    assert lm.model.embed_tokens_per_layer is None
    assert not _ple_module_paths(lm)
