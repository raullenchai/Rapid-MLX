"""Probes for the vendored speculative core (step 3b).

Mechanical guarantee: every function/class body in the vendored
``speculative/`` coordinator modules, ``models/base.py``,
``models/linear.py``, ``fp8.py``, and ``quant_utils.py`` is
byte-identical to the pinned upstream ``mlx-vlm==0.7.1`` source. The
only permitted differences are the documented module-level import
redirects (see the package inventory), which never enter a function's
``getsource``, plus two inventoried function-level lazy-import redirects
(``native_batch_linear``'s verifier fallback and ``dequantize_model``'s
mla/switch_layers resolution — both pinned upstream until step 3c) and
one set of documented bugfix hunks (``build_ddtree``'s ``ValueError``
validation; ``_dflash_rounds_batch``/``_mtp_rounds_batch``'s unfinished-row
budget). The walker compares function/class name sets in both directions.
Two exemption mechanisms exist and must not be confused: ``documented``
filters strict-compare divergences for REAL permitted behavioral hunks;
``normalized`` entries compare on behavior only (comments, blanks, and
import statements stripped from both sides) and their divergences are
emitted with a marker the documented filter cannot match — they always
fail.

Behavioral guarantee: the vendored coordinator binds the vendored cache
and model foundations; the two deliberately-pinned dependencies (the
quantized verifier and the eagle3 backend) resolve upstream and are
pure-array functions, so the cross-namespace calls are identity-safe.
"""

import inspect

import pytest

import rapid_mlx.models.mlx_vlm_vendored.cache as vendored_cache
import rapid_mlx.models.mlx_vlm_vendored.fp8 as vendored_fp8
import rapid_mlx.models.mlx_vlm_vendored.models.base as vendored_base
import rapid_mlx.models.mlx_vlm_vendored.models.linear as vendored_linear
import rapid_mlx.models.mlx_vlm_vendored.quant_utils as vendored_quant_utils
import rapid_mlx.models.mlx_vlm_vendored.speculative as vendored_speculative
import rapid_mlx.models.mlx_vlm_vendored.speculative.cache_state as vs_cache_state
import rapid_mlx.models.mlx_vlm_vendored.speculative.common as vs_common
import rapid_mlx.models.mlx_vlm_vendored.speculative.ddtree as vs_ddtree
import rapid_mlx.models.mlx_vlm_vendored.speculative.dflash as vs_dflash
import rapid_mlx.models.mlx_vlm_vendored.speculative.mtp as vs_mtp
import rapid_mlx.models.mlx_vlm_vendored.speculative.utils as vs_utils

pytest.importorskip("mlx_vlm")


def _code_lines(src):
    """Drop comments and blank lines; canonicalize the permitted redirect
    import statements (pinned ``mlx_vlm.*`` / relative targets) to their
    imported SYMBOL names only — the module-path difference is ignored but
    the symbols are still compared, so swapping a redirect to a different
    source (or different symbols) diverges. Any non-redirect import is
    retained verbatim."""
    lines = []
    in_redirect_import = False
    redirect_names = []

    def _flush_redirect_names():
        if redirect_names:
            lines.append("import " + ", ".join(sorted(redirect_names)))
            redirect_names.clear()

    for line in src.splitlines():
        stripped = line.strip()
        if in_redirect_import:
            if not stripped or stripped.startswith("#"):
                continue
            if ")" in stripped:
                in_redirect_import = False
                name = stripped.rsplit(")", 1)[0].strip().rstrip(",").strip()
                if name:
                    redirect_names.append(name)
                _flush_redirect_names()
            else:
                redirect_names.append(stripped.rstrip(",").strip())
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(("from ", "import ")):
            is_redirect = stripped.startswith(
                ("from mlx_vlm", "from .", "import mlx_vlm", "import .")
            )
            if not is_redirect:
                lines.append(line)
                continue
            if stripped.endswith("("):
                in_redirect_import = True
                continue
            names_part = stripped.split(" import ", 1)
            if len(names_part) == 2:
                redirect_names.extend(
                    n.strip() for n in names_part[1].split(",") if n.strip()
                )
            _flush_redirect_names()
            continue
        lines.append(line)
    return lines


def _body_divergences(vendored_module, upstream_module, normalized=()):
    diverged = []
    vendored_defs = {
        name: obj
        for name, obj in vars(vendored_module).items()
        if not name.startswith("__")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
    }
    upstream_defs = {
        name: obj
        for name, obj in vars(upstream_module).items()
        if not name.startswith("__")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
    }
    for name in sorted(set(vendored_defs) - set(upstream_defs)):
        diverged.append(f"{name}: vendored-only")
    for name in sorted(set(upstream_defs) - set(vendored_defs)):
        diverged.append(f"{name}: missing in vendored")
    for name in sorted(set(vendored_defs) & set(upstream_defs)):
        obj = vendored_defs[name]
        upstream_obj = upstream_defs[name]
        if type(obj) is not type(upstream_obj):
            diverged.append(f"{name}: kind mismatch")
            continue
        try:
            vendored_src = inspect.getsource(obj)
            upstream_src = inspect.getsource(upstream_obj)
        except (OSError, TypeError):
            continue
        if name in normalized:
            # Normalized entries compare on behavior only; any remaining
            # divergence is fatal and MUST NOT be filterable by the
            # documented set (emitted with a marker the bare-name filter
            # cannot match).
            if _code_lines(vendored_src) != _code_lines(upstream_src):
                diverged.append(f"{name}: normalized-body-divergence")
        elif vendored_src != upstream_src:
            diverged.append(name)
    return diverged


def test_vendored_speculative_bodies_match_upstream():
    from mlx_vlm.speculative import (
        cache_state as up_cache_state,
    )
    from mlx_vlm.speculative import (
        common as up_common,
    )
    from mlx_vlm.speculative import (
        ddtree as up_ddtree,
    )
    from mlx_vlm.speculative import (
        dflash as up_dflash,
    )
    from mlx_vlm.speculative import (
        mtp as up_mtp,
    )
    from mlx_vlm.speculative import (
        utils as up_utils,
    )

    # ``BatchRotatingKVCache`` is defined in the vendored cache.py, whose
    # merge() carries the documented 2a upstream-bugfix hunk (see the
    # cache.py inventory entry); its source therefore differs from pinned
    # upstream by exactly that hunk.
    for vendored, upstream, documented in (
        (vs_cache_state, up_cache_state, {"BatchRotatingKVCache"}),
        (vs_common, up_common, set()),
        # ``build_ddtree`` carries the documented assert→ValueError hunk —
        # a REAL permitted behavioral difference, hence documented-filtered.
        (vs_ddtree, up_ddtree, {"build_ddtree"}),
        (vs_dflash, up_dflash, {"_dflash_rounds_batch"}),
        (vs_mtp, up_mtp, {"_mtp_rounds_batch"}),
        # ``_dflash_rounds_batch``/``_mtp_rounds_batch`` also appear in
        # utils' namespace via their dflash/mtp imports; the hunks live in
        # their own modules (documented above and in the inventory).
        (vs_utils, up_utils, {"_dflash_rounds_batch", "_mtp_rounds_batch"}),
    ):
        normalized = {"native_batch_linear"} if vendored is vs_mtp else set()
        # ``native_batch_linear`` (imported from vendored models.linear)
        # compares on behavior only; its divergences are never filtered.
        divergences = _body_divergences(vendored, upstream, normalized=normalized)
        divergences = [d for d in divergences if d not in documented]
        assert divergences == []


def test_vendored_foundations_bodies_match_upstream():
    from mlx_vlm import fp8 as up_fp8
    from mlx_vlm import quant_utils as up_quant_utils
    from mlx_vlm.models import base as up_base
    from mlx_vlm.models import linear as up_linear

    # Documented function-level lazy-import redirects (see the package
    # inventory): linear's verifier fallback and quant_utils' mla /
    # switch_layers resolution stay pinned until the 3c slices. Those two
    # functions compare on behavior only (comments, blanks, and the
    # redirected import statements are stripped from both sides) and their
    # divergences are NEVER filtered — any behavioral edit fails.
    for vendored, upstream, documented, normalized in (
        (vendored_base, up_base, set(), set()),
        (vendored_linear, up_linear, set(), {"native_batch_linear"}),
        (vendored_fp8, up_fp8, set(), set()),
        (
            vendored_quant_utils,
            up_quant_utils,
            set(),
            {"dequantize_model"},
        ),
    ):
        divergences = _body_divergences(vendored, upstream, normalized=normalized)
        divergences = [d for d in divergences if d not in documented]
        assert divergences == []


def test_speculative_core_binds_vendored_foundations():
    assert vs_mtp.cache is vendored_cache
    assert vs_cache_state.BatchRotatingKVCache is (vendored_cache.BatchRotatingKVCache)
    assert vs_cache_state.RotatingKVCache is vendored_cache.RotatingKVCache
    assert vs_common.LanguageModelOutput is vendored_base.LanguageModelOutput
    assert vs_utils._dflash_rounds.__module__.endswith("vendored.speculative.dflash")
    assert vs_utils.get_speculative_rounds_batch("mtp").__module__.endswith(
        "vendored.speculative.mtp"
    )


def test_pinned_redirects_resolve_upstream():
    # The quantized verifier and the eagle3 backend stay pinned; their
    # round/decode helpers are pure array functions.
    assert vs_mtp.decode_quantized_argmax.__module__ == (
        "mlx_vlm.models.quantized_verifier"
    )
    assert vs_utils._eagle3_rounds.__module__ == "mlx_vlm.speculative.eagle3"
    assert (
        vs_utils.get_speculative_rounds_batch("eagle3").__module__
        == "mlx_vlm.speculative.eagle3"
    )
    with pytest.raises(ValueError, match="Unknown draft_kind"):
        vs_utils.get_speculative_rounds_batch("nope")


def test_speculative_shim_exports_ddtree_only():
    assert vendored_speculative.DDTreeNode is vs_ddtree.DDTreeNode
    assert vendored_speculative.build_ddtree is vs_ddtree.build_ddtree
    # load_drafter arrives with the drafter registry slice (step 3c).
    assert not hasattr(vendored_speculative, "load_drafter")


def test_native_batch_linear_foundation_path_runs():
    # r1 fix: the lazy verifier import inside native_batch_linear must
    # resolve (redirected to pinned upstream until 3c) and the quantized
    # fallback must execute for B>1 batched hidden states (the MTP
    # projection path at mtp.py calls exactly this).
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    module = nn.Linear(64, 32, bias=False)
    nn.quantize(module, group_size=32, bits=4)
    x = mx.random.normal((2, 3, 64))
    out = vendored_linear.native_batch_linear(module, x)
    assert out.shape == (2, 3, 32)


def test_dequantize_model_foundation_path_runs():
    # r1 fix: dequantize_model's lazy mla/switch_layers imports execute at
    # function entry — before type dispatch — so even a plain
    # nn.QuantizedLinear model needs them to resolve (redirected to pinned
    # upstream until the model-module slices).
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(64, 32, bias=False)

    tiny = Tiny()
    nn.quantize(tiny, group_size=32, bits=4)
    assert isinstance(tiny.proj, nn.QuantizedLinear)
    dequantized = vendored_quant_utils.dequantize_model(tiny)
    assert type(dequantized.proj) is nn.Linear
    probe = mx.random.normal((2, 64))
    assert dequantized.proj(probe).shape == (2, 32)


def test_build_ddtree_validates_without_assert():
    # r2 fix: pinned upstream validates with ``assert``, which disappears
    # under ``python -O`` and would silently process invalid ranks or
    # multi-row logits as row zero; the vendored hunk raises ValueError.
    mx = pytest.importorskip("mlx.core")
    with pytest.raises(ValueError, match="single-row"):
        vs_ddtree.build_ddtree(mx.zeros((2, 4, 8)), budget=4)
    with pytest.raises(ValueError, match="single-row"):
        vs_ddtree.build_ddtree(mx.zeros((4, 8)), budget=4)


def test_code_lines_canonicalizes_redirect_imports():
    # r5 fix: the normalizer must keep imported SYMBOL names comparable —
    # only the module-path difference between the pinned redirect and the
    # upstream relative import is ignored. A symbol swap still diverges.
    vendored = (
        "def f():\n"
        "    # VENDOR-DEVIATION(redirect): pinned until 3c.\n"
        "    from mlx_vlm.models.quantized_verifier import (\n"
        "        exact_quantized_linear,\n"
        "        singleton_quantized_linear,\n"
        "    )\n"
        "    return exact_quantized_linear\n"
    )
    upstream = (
        "def f():\n"
        "    from .quantized_verifier import (\n"
        "        exact_quantized_linear,\n"
        "        singleton_quantized_linear,\n"
        "    )\n"
        "    return exact_quantized_linear\n"
    )
    assert _code_lines(vendored) == _code_lines(upstream)
    swapped = vendored.replace("singleton_quantized_linear", "other_helper")
    assert _code_lines(swapped) != _code_lines(upstream)
    # A non-redirect import is retained verbatim and diverges.
    foreign = vendored.replace(
        "    from mlx_vlm.models.quantized_verifier import (",
        "    from some_other_package import (",
    )
    assert _code_lines(foreign) != _code_lines(upstream)
