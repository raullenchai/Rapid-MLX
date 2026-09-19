"""Probes for the vendored speculative core (step 3b).

Mechanical guarantee: every function/class body in the vendored
``speculative/`` coordinator modules, ``models/base.py``,
``models/linear.py``, ``fp8.py``, and ``quant_utils.py`` is
byte-identical to the pinned upstream ``mlx-vlm==0.7.1`` source. The
only permitted differences are the documented module-level import
redirects (see the package inventory), which never enter a function's
``getsource``.

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


def _body_divergences(vendored_module, upstream_module):
    diverged = []
    for name, obj in vars(vendored_module).items():
        if name.startswith("__"):
            continue
        upstream_obj = getattr(upstream_module, name, None)
        if upstream_obj is None:
            continue
        if (inspect.isfunction(obj) or inspect.isclass(obj)) and type(obj) is not type(
            upstream_obj
        ):
            diverged.append(f"{name}: kind mismatch")
            continue
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        try:
            vendored_src = inspect.getsource(obj)
            upstream_src = inspect.getsource(upstream_obj)
        except (OSError, TypeError):
            continue
        if vendored_src != upstream_src:
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
        (vs_ddtree, up_ddtree, set()),
        (vs_dflash, up_dflash, set()),
        (vs_mtp, up_mtp, set()),
        (vs_utils, up_utils, set()),
    ):
        divergences = _body_divergences(vendored, upstream)
        divergences = [d for d in divergences if d not in documented]
        assert divergences == []


def test_vendored_foundations_bodies_match_upstream():
    from mlx_vlm import fp8 as up_fp8
    from mlx_vlm import quant_utils as up_quant_utils
    from mlx_vlm.models import base as up_base
    from mlx_vlm.models import linear as up_linear

    for vendored, upstream in (
        (vendored_base, up_base),
        (vendored_linear, up_linear),
        (vendored_fp8, up_fp8),
        (vendored_quant_utils, up_quant_utils),
    ):
        assert _body_divergences(vendored, upstream) == []


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
