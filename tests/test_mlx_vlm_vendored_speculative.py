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
The vendored processor installer also differs in one security hunk: a matching
remote ``model_type`` is intercepted only after explicit
``trust_remote_code=True`` consent.
Two exemption mechanisms exist and must not be confused: ``documented``
filters strict-compare divergences for REAL permitted behavioral hunks;
``normalized`` entries compare on behavior only (comments, blanks, and
import statements stripped from both sides) and their divergences are
emitted with a marker the documented filter cannot match — they always
fail.

Behavioral guarantee: the vendored coordinator binds the vendored cache
and model foundations while recognizing cache trees returned by still-pinned
model implementations in the upstream namespace. The two deliberately-pinned
dependencies (the quantized verifier and the eagle3 backend) resolve upstream;
their permitted cross-namespace calls are identity-safe.
"""

import inspect
from types import SimpleNamespace

import pytest

# This file is part of the explicit Apple-Silicon lane, but the ordinary Linux
# shard discovers every test module before marker deselection.  Skip before any
# vendored import can transitively import ``mlx.core``.
pytest.importorskip("mlx")
pytest.importorskip("mlx_vlm")
pytestmark = pytest.mark.requires_mlx

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

# Documented behavioral hunks, specified EXACTLY: applying each
# (vendored → upstream) replacement to the vendored body must reproduce
# the pinned upstream body byte-for-byte; any other edit inside the
# function diverges. Each hunk is inventoried in the package
# ``__init__.py`` and behavior-tested in this module.
_HUNK_SPECS = {
    "_RotatingCacheTransaction": (
        (
            "        # VENDOR-DEVIATION(dual-namespace): transactions may wrap caches\n"
            "        # returned by either the vendored fallback or a pinned model.\n"
            "        if len(set(lengths)) > 1 and not isinstance(\n"
            "            self.cache, (BatchRotatingKVCache, UpstreamBatchRotatingKVCache)\n"
            "        ):\n",
            "        if len(set(lengths)) > 1 and not isinstance(self.cache, BatchRotatingKVCache):\n",
        ),
    ),
    "iter_leaf_caches": (
        (
            "        # VENDOR-DEVIATION(dual-namespace): pinned model-owned cache trees use\n"
            "        # the upstream container while fallback trees use the vendored one.\n"
            "        if isinstance(cache, (CacheList, UpstreamCacheList)):\n",
            "        if isinstance(cache, CacheList):\n",
        ),
    ),
    "start_speculative_cache": (
        (
            "            # VENDOR-DEVIATION(dual-namespace): a pinned model's make_cache()\n"
            "            # returns upstream rotating caches, which need the same replay\n"
            "            # transaction as vendored fallbacks after a partial acceptance.\n"
            "            if isinstance(\n"
            "                cache,\n"
            "                (\n"
            "                    RotatingKVCache,\n"
            "                    BatchRotatingKVCache,\n"
            "                    UpstreamRotatingKVCache,\n"
            "                    UpstreamBatchRotatingKVCache,\n"
            "                ),\n"
            "            ):\n",
            "            if isinstance(cache, (RotatingKVCache, BatchRotatingKVCache)):\n",
        ),
    ),
    "rollback_speculative_cache": (
        (
            "    # VENDOR-DEVIATION(dual-namespace): still-upstream target hooks can start\n"
            "    # and return their coordinator's transaction around model-owned caches.\n"
            "    if isinstance(\n"
            "        transaction,\n"
            "        (SpeculativeCacheTransaction, UpstreamSpeculativeCacheTransaction),\n"
            "    ):\n",
            "    if isinstance(transaction, SpeculativeCacheTransaction):\n",
        ),
    ),
    "abort_speculative_round": (
        (
            "    # VENDOR-DEVIATION(dual-namespace): finalize transactions returned by\n"
            "    # either vendored fallback verification or a still-upstream target hook.\n"
            "    if isinstance(\n"
            "        state,\n"
            "        (SpeculativeCacheTransaction, UpstreamSpeculativeCacheTransaction),\n"
            "    ):\n",
            "    if isinstance(state, SpeculativeCacheTransaction):\n",
        ),
    ),
    "commit_speculative_round": (
        (
            "    # VENDOR-DEVIATION(dual-namespace): Qwen4/Qwen3.5-style upstream hooks\n"
            "    # return their own transaction class. Commit it directly instead of\n"
            "    # falling through to an optional legacy model rollback method.\n"
            "    if isinstance(\n"
            "        state,\n"
            "        (SpeculativeCacheTransaction, UpstreamSpeculativeCacheTransaction),\n"
            "    ):\n",
            "    if isinstance(state, SpeculativeCacheTransaction):\n",
        ),
    ),
    "_mtp_shared_kv_from_prompt_cache": (
        (
            "            # VENDOR-DEVIATION(dual-namespace): preserve temporal ordering for\n"
            "            # rotating caches produced by either cache namespace.\n"
            "            isinstance(\n"
            "                layer_cache,\n"
            "                (cache.RotatingKVCache, upstream_cache.RotatingKVCache),\n"
            "            )\n"
            "            and not isinstance(\n"
            "                layer_cache,\n"
            "                (\n"
            "                    cache.BufferedRotatingKVCache,\n"
            "                    upstream_cache.BufferedRotatingKVCache,\n"
            "                ),\n"
            "            )\n",
            "            isinstance(layer_cache, cache.RotatingKVCache)\n"
            "            and not isinstance(layer_cache, cache.BufferedRotatingKVCache)\n",
        ),
    ),
    "_buffer_mtp_target_cache": (
        (
            "        # VENDOR-DEVIATION(dual-namespace): recurse through both model-owned\n"
            "        # upstream trees and vendored fallback trees.\n"
            "        if isinstance(entry, (cache.CacheList, upstream_cache.CacheList)):\n",
            "        if isinstance(entry, cache.CacheList):\n",
        ),
        (
            "        if isinstance(\n"
            "            entry,\n"
            "            (\n"
            "                cache.BufferedRotatingKVCache,\n"
            "                upstream_cache.BufferedRotatingKVCache,\n"
            "            ),\n"
            "        ):\n",
            "        if isinstance(entry, cache.BufferedRotatingKVCache):\n",
        ),
        (
            "            isinstance(\n"
            "                entry,\n"
            "                (cache.RotatingKVCache, upstream_cache.RotatingKVCache),\n"
            "            )\n"
            '            and getattr(entry, "keep", 0) == 0\n',
            '            isinstance(entry, cache.RotatingKVCache) and getattr(entry, "keep", 0) == 0\n',
        ),
        (
            "            # Keep the replacement in the producer's namespace; downstream\n"
            "            # model code can use exact-type dispatch for its cache classes.\n"
            "            namespace = (\n"
            "                upstream_cache\n"
            "                if isinstance(entry, upstream_cache.RotatingKVCache)\n"
            "                else cache\n"
            "            )\n"
            "            return namespace.BufferedRotatingKVCache.from_cache(\n",
            "            return cache.BufferedRotatingKVCache.from_cache(\n",
        ),
    ),
    "build_ddtree": (
        (
            "    # VENDOR-DEVIATION(bugfix): pinned upstream validates with ``assert``,\n"
            "    # which disappears under ``python -O`` and would silently process\n"
            "    # invalid ranks/multi-row logits as row zero; raise explicitly instead.\n"
            "    if drafter_logits.ndim != 3 or drafter_logits.shape[0] != 1:\n"
            "        raise ValueError(\n"
            '            "drafter_logits must be a single-row [1, L, V] tensor, got "\n'
            '            f"shape {tuple(drafter_logits.shape)}"\n'
            "        )\n",
            "    assert drafter_logits.ndim == 3 and drafter_logits.shape[0] == 1\n",
        ),
    ),
    "install_auto_processor_patch": (
        (
            "            # VENDOR-DEVIATION(security): discovering a matching remote model\n"
            "            # type is not consent to execute repository code. Only intercept\n"
            "            # after the caller explicitly opts in.\n"
            "            if (\n"
            "                model_type in target_model_types\n"
            '                and kwargs.get("trust_remote_code") is True\n'
            "            ):\n",
            "            if model_type in target_model_types:\n"
            '                kwargs.setdefault("trust_remote_code", True)\n',
        ),
    ),
    "_mtp_verify_without_logits": (
        (
            '    layers = getattr(getattr(lm, "model", None), "layers", [])\n'
            "    if len(prompt_cache) == len(layers):\n"
            "        # VENDOR-DEVIATION(bugfix): the hook-less fallback must participate in\n"
            "        # the same cache transaction as every other speculative verifier.\n"
            "        transaction = start_speculative_cache(prompt_cache, verify_input.shape[1])\n"
            "        try:\n"
            "            hidden = lm.model(\n"
            "                verify_input,\n"
            "                cache=prompt_cache,\n"
            "                skip_final_norm=True,\n"
            "            )\n"
            "            shared_kv_states = _mtp_shared_kv_from_prompt_cache(lm, prompt_cache)\n"
            "            if shared_kv_states:\n"
            "                return _MTPVerifyResult(\n"
            "                    hidden=hidden,\n"
            "                    shared_kv_states=shared_kv_states,\n"
            "                    rollback_state=transaction,\n"
            "                )\n"
            "        except BaseException:\n"
            "            transaction.abort()\n"
            "            raise\n"
            "        # The sink retry must not append the same verifier block a second time.\n"
            "        transaction.abort()\n"
            "\n"
            "    shared_kv_sink: dict = {}\n"
            "    transaction = start_speculative_cache(prompt_cache, verify_input.shape[1])\n"
            "    try:\n"
            "        hidden = lm.model(\n"
            "            verify_input,\n"
            "            cache=prompt_cache,\n"
            "            shared_kv_sink=shared_kv_sink,\n"
            "            skip_final_norm=True,\n"
            "        )\n"
            "    except BaseException:\n"
            "        transaction.abort()\n"
            "        raise\n"
            "    if not shared_kv_sink:\n"
            "        transaction.abort()\n"
            "        return None\n"
            "    return _MTPVerifyResult(\n"
            "        hidden=hidden,\n"
            "        shared_kv_states=shared_kv_sink,\n"
            "        rollback_state=transaction,\n"
            "    )\n",
            '    layers = getattr(getattr(lm, "model", None), "layers", [])\n'
            "    if len(prompt_cache) == len(layers):\n"
            "        hidden = lm.model(\n"
            "            verify_input,\n"
            "            cache=prompt_cache,\n"
            "            skip_final_norm=True,\n"
            "        )\n"
            "        shared_kv_states = _mtp_shared_kv_from_prompt_cache(lm, prompt_cache)\n"
            "        if shared_kv_states:\n"
            "            return _MTPVerifyResult(hidden=hidden, shared_kv_states=shared_kv_states)\n"
            "\n"
            "    shared_kv_sink: dict = {}\n"
            "    hidden = lm.model(\n"
            "        verify_input,\n"
            "        cache=prompt_cache,\n"
            "        shared_kv_sink=shared_kv_sink,\n"
            "        skip_final_norm=True,\n"
            "    )\n"
            "    if not shared_kv_sink:\n"
            "        return None\n"
            "    return _MTPVerifyResult(hidden=hidden, shared_kv_states=shared_kv_sink)\n",
        ),
    ),
    "_speculative_walk_batch_uniform_acceptance": (
        (
            '    """Clamp a batch to the earliest rejection with verifier-token fallback."""\n'
            "    # VENDOR-DEVIATION(bugfix): pinned upstream mins over every row, so a\n"
            "    # retained finished row (zero budget under the non-filterable-cache\n"
            "    # fallback) would clamp the whole batch to zero acceptance and collapse\n"
            "    # throughput to bonus-only decoding. Budget from rows that still have\n"
            "    # tokens to spend; default to 0 when none do.\n"
            "    positive = [a for a, budget in zip(accepted_list, budgets) if budget > 0]\n"
            "    accepted = min(positive) if positive else 0\n",
            '    """Clamp a batch to the earliest rejection with verifier-token fallback."""\n'
            "    accepted = min(accepted_list)\n",
        ),
    ),
    "_dflash_rounds": (
        (
            "    use_model_initial_block_size: bool = True,\n"
            "    greedy_sampling: bool = True,\n"
            "    row_id: int = 0,\n",
            "    use_model_initial_block_size: bool = True,\n"
            "    greedy_sampling: bool = True,\n",
        ),
        (
            "        draft_sampler = (\n"
            "            _PositionedDraftSampler(\n"
            "                sampler,\n"
            "                # VENDOR-DEVIATION(bugfix): pinned upstream hard-codes row 0,\n"
            "                # so a nonzero server row ID reads the wrong per-request\n"
            "                # sampling stream (mirrors mtp's row_id threading).\n"
            "                row_ids=[row_id],\n"
            "                positions=[emitted],\n"
            "            )\n",
            "        draft_sampler = (\n"
            "            _PositionedDraftSampler(\n"
            "                sampler,\n"
            "                row_ids=[0],\n"
            "                positions=[emitted],\n"
            "            )\n",
        ),
        (
            "                    row_ids=[row_id],\n",
            "                    row_ids=[0],\n",
        ),
    ),
    "_dflash_rounds_batch": (
        (
            "        # VENDOR-DEVIATION(bugfix): mirror of the mtp budget fix — with the\n"
            "        # non-filterable-cache fallback, a retained finished row's\n"
            "        # ``remaining == 1`` would force ``bs <= 1`` and terminate the\n"
            "        # whole batched loop. Budget from unfinished rows only.\n"
            "        remaining = [\n"
            "            max(1, max_tokens - emitted[active_idx[j]] + 1)\n"
            "            for j in range(len(active_idx))\n"
            "            if not finished[active_idx[j]]\n"
            "        ]\n"
            "        if not remaining:\n"
            "            break\n"
            "        bs = _dflash_next_block_size(\n",
            "        remaining = [\n"
            "            max(1, max_tokens - emitted[active_idx[j]] + 1)\n"
            "            for j in range(len(active_idx))\n"
            "        ]\n"
            "        bs = _dflash_next_block_size(\n",
        ),
        (
            "                # VENDOR-DEVIATION(bugfix): pinned upstream takes the min\n"
            "                # over every row, so a retained finished row (empty token\n"
            "                # budget under the non-filterable-cache fallback) yields\n"
            "                # -1, empties every unfinished row's output, and can stall\n"
            "                # the loop with no progress. Clamp over rows that still\n"
            "                # have a positive budget and floor the acceptance at 0.\n"
            "                positive = [\n"
            "                    len(nt) - 1\n"
            "                    for nt, j in zip(new_tokens_list, range(n_active))\n"
            "                    if budgets[j] > 0\n"
            "                ]\n"
            "                uniform = max(0, min(positive)) if positive else 0\n",
            "                uniform = min(len(nt) - 1 for nt in new_tokens_list)\n",
        ),
        (
            "            # VENDOR-DEVIATION(bugfix): pinned upstream filters only the\n"
            "            # caches exposing ``filter()`` but unconditionally shrinks\n"
            "            # ``active_idx`` — with a mixed cache list the non-filterable\n"
            "            # leaves keep the old batch dimension while the verifier sees\n"
            "            # the reduced batch. Compact only when EVERY cache is\n"
            "            # filterable; otherwise keep all rows active (finished rows\n"
            "            # emit nothing until the round ends).\n"
            '            if all(hasattr(c, "filter") for c in prompt_cache):\n'
            "                keep_mx = mx.array(keep_slots, dtype=mx.int32)\n"
            "                for c in prompt_cache:\n"
            "                    c.filter(keep_mx)\n"
            "                # Update active index mapping\n"
            "                active_idx = [active_idx[j] for j in keep_slots]\n",
            "            # Filter target caches (BatchKVCache supports this)\n"
            "            keep_mx = mx.array(keep_slots, dtype=mx.int32)\n"
            "            for c in prompt_cache:\n"
            '                if hasattr(c, "filter"):\n'
            "                    c.filter(keep_mx)\n"
            "            # Update active index mapping\n"
            "            active_idx = [active_idx[j] for j in keep_slots]\n",
        ),
    ),
    "run_speculative_server_rounds": (
        (
            "                greedy_sampling=greedy_sampling,\n"
            "                # VENDOR-DEVIATION(bugfix): thread the server's per-request\n"
            "                # row ID into the singleton sampling identity (upstream\n"
            "                # hard-codes row 0).\n"
            "                row_id=(row_ids[0] if row_ids else 0),\n"
            "            ):\n",
            "                greedy_sampling=greedy_sampling,\n            ):\n",
        ),
    ),
    "_mtp_rounds_batch": (
        (
            "        # VENDOR-DEVIATION(bugfix): pinned upstream budgets the block size\n"
            "        # from every active row; with the non-filterable-cache fallback\n"
            "        # (finished rows retained), a finished row's ``remaining == 1``\n"
            "        # would force ``bs <= 1`` and terminate the whole batched loop\n"
            "        # while other rows still have tokens to generate. Budget from\n"
            "        # unfinished rows only; an empty budget ends the loop.\n"
            "        remaining = [\n"
            "            max(1, max_tokens - emitted[active_idx[j]] + 1)\n"
            "            for j in range(len(active_idx))\n"
            "            if not finished[active_idx[j]]\n"
            "        ]\n"
            "        if not remaining:\n"
            "            break\n"
            "        bs = _mtp_next_block_size(\n",
            "        remaining = [\n"
            "            max(1, max_tokens - emitted[active_idx[j]] + 1)\n"
            "            for j in range(len(active_idx))\n"
            "        ]\n"
            "        bs = _mtp_next_block_size(\n",
        ),
        (
            '        cache_filterable = all(hasattr(c, "filter") for c in prompt_cache)\n'
            "        # VENDOR-DEVIATION(bugfix): pinned upstream compacts whenever the\n"
            "        # target caches are filterable but only shrinks the drafter when it\n"
            "        # happens to expose ``filter_batch`` — a drafter without it keeps\n"
            "        # its per-row state at the old batch shape while the caches, hidden\n"
            "        # states, and shared KV shrink, misaligning every subsequent draft\n"
            "        # step. Compact only when the caches AND the drafter can be shrunk\n"
            "        # together; otherwise ride the unfinished-row budget fallback\n"
            "        # (finished rows stay active and simply stop emitting).\n"
            '        drafter_filterable = callable(getattr(draft_model, "filter_batch", None))\n'
            "        if all(finished[active_idx[j]] for j in range(n_active)):\n"
            "            break\n"
            "        if cache_filterable and drafter_filterable:\n",
            '        cache_filterable = all(hasattr(c, "filter") for c in prompt_cache)\n'
            "        if all(finished[active_idx[j]] for j in range(n_active)):\n"
            "            break\n"
            "        if cache_filterable:\n",
        ),
        (
            "                draft_model.filter_batch(keep_mx)\n",
            '                filter_drafter = getattr(draft_model, "filter_batch", None)\n'
            "                if callable(filter_drafter):\n"
            "                    filter_drafter(keep_mx)\n",
        ),
    ),
}


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


def _body_divergences(vendored_module, upstream_module, normalized=(), hunk_specs=None):
    # Only symbols DEFINED in the walked module are compared: imported
    # symbols carry their defining module's hunks into every importer's
    # namespace, so each hunk is owned (and documented) exactly once — by
    # the module that defines it.
    diverged = []
    vendored_defs = {
        name: obj
        for name, obj in vars(vendored_module).items()
        if not name.startswith("__")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and getattr(obj, "__module__", None) == vendored_module.__name__
    }
    upstream_defs = {
        name: obj
        for name, obj in vars(upstream_module).items()
        if not name.startswith("__")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and getattr(obj, "__module__", None) == upstream_module.__name__
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
        elif hunk_specs and name in hunk_specs:
            # Documented behavioral hunks are specified EXACTLY: applying
            # each (vendored → upstream) replacement must reproduce the
            # pinned upstream body byte-for-byte. Any other edit inside
            # the function diverges.
            ok = True
            for vendored_snippet, upstream_snippet in hunk_specs[name]:
                if vendored_src.count(vendored_snippet) != 1:
                    diverged.append(f"{name}: documented hunk not found")
                    ok = False
                    break
                vendored_src = vendored_src.replace(
                    vendored_snippet, upstream_snippet, 1
                )
            if ok and vendored_src != upstream_src:
                diverged.append(f"{name}: hunk-normalized divergence")
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

    # Only symbols DEFINED in each walked module are compared (imported
    # symbols are owned by their defining module's entry — e.g.
    # ``BatchRotatingKVCache``'s 2a merge bugfix is covered by the cache
    # suite that walks ``cache.py`` itself).
    # Documented behavioral hunks are specified exactly in ``_HUNK_SPECS``
    # (vendored → upstream replacement must reproduce the pinned body);
    # ``normalized`` entries compare on behavior only (comments, blanks,
    # and the redirected import statements are stripped from both sides)
    # and their divergences are NEVER filtered — any behavioral edit fails.
    for vendored, upstream in (
        (vs_cache_state, up_cache_state),
        (vs_common, up_common),
        (vs_ddtree, up_ddtree),
        (vs_dflash, up_dflash),
        (vs_mtp, up_mtp),
        (vs_utils, up_utils),
    ):
        divergences = _body_divergences(vendored, upstream, hunk_specs=_HUNK_SPECS)
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
        divergences = _body_divergences(
            vendored,
            upstream,
            hunk_specs=_HUNK_SPECS,
            normalized=normalized,
        )
        divergences = [d for d in divergences if d not in documented]
        assert divergences == []


def test_speculative_core_binds_vendored_foundations():
    from mlx_vlm.models import cache as upstream_cache
    from mlx_vlm.speculative import cache_state as upstream_cache_state

    assert vs_mtp.cache is vendored_cache
    assert vs_mtp.upstream_cache is upstream_cache
    assert vs_cache_state.BatchRotatingKVCache is (vendored_cache.BatchRotatingKVCache)
    assert vs_cache_state.RotatingKVCache is vendored_cache.RotatingKVCache
    assert vs_cache_state.UpstreamBatchRotatingKVCache is (
        upstream_cache.BatchRotatingKVCache
    )
    assert vs_cache_state.UpstreamRotatingKVCache is upstream_cache.RotatingKVCache
    assert vs_cache_state.UpstreamSpeculativeCacheTransaction is (
        upstream_cache_state.SpeculativeCacheTransaction
    )
    assert vs_common.LanguageModelOutput is vendored_base.LanguageModelOutput
    assert vs_utils._dflash_rounds.__module__.endswith("vendored.speculative.dflash")
    assert vs_utils.get_speculative_rounds_batch("mtp").__module__.endswith(
        "vendored.speculative.mtp"
    )


def test_speculative_core_preserves_upstream_model_cache_namespace():
    """Pinned model caches must get the same rotating replay and buffering
    behavior as vendored fallback caches, without changing their namespace."""
    import mlx.core as mx
    from mlx_vlm.models import cache as upstream_cache

    rotating = upstream_cache.RotatingKVCache(max_size=4)
    initial = mx.array([[[[0.0], [1.0], [2.0], [3.0]]]])
    rotating.update_and_fetch(initial, initial)
    tree = upstream_cache.CacheList(rotating)

    assert list(vs_cache_state.iter_leaf_caches([tree])) == [rotating]
    transaction = vs_cache_state.start_speculative_cache([tree], length=3)
    assert id(rotating) in transaction._rotating

    speculative = mx.array([[[[4.0], [5.0], [6.0]]]])
    rotating.update_and_fetch(speculative, speculative)
    transaction.commit([1])

    # Only the first verifier token is retained. Restoring and replaying is
    # essential here: a simple cursor trim cannot recover the ring entries
    # overwritten by the rejected tokens.
    assert rotating.offset == 5
    ordered = rotating._temporal_order(rotating.keys)
    assert ordered.reshape(-1).tolist() == [1.0, 2.0, 3.0, 4.0]

    lm = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(layer_type="attention")])
    )
    shared = vs_mtp._mtp_shared_kv_from_prompt_cache(lm, [rotating])
    assert shared["attention"][0].reshape(-1).tolist() == [1.0, 2.0, 3.0, 4.0]

    prompt_cache = [upstream_cache.CacheList(upstream_cache.RotatingKVCache(8))]
    draft_model = SimpleNamespace(config=SimpleNamespace(block_size=4))
    vs_mtp._buffer_mtp_target_cache(prompt_cache, draft_model, draft_block_size=4)
    buffered = prompt_cache[0].caches[0]
    assert type(buffered) is upstream_cache.BufferedRotatingKVCache

    batch = upstream_cache.BatchRotatingKVCache(max_size=8, left_padding=[0, 0])
    batch_transaction = vs_cache_state._RotatingCacheTransaction(batch)
    batch_transaction.validate([1, 2])
    batch_transaction.abort()


def test_speculative_core_finalizes_upstream_model_transactions():
    """Model hooks still imported from mlx-vlm return their namespace's
    transaction object; vendored ownership must commit or abort it directly."""
    from mlx_vlm.speculative import cache_state as upstream_cache_state

    class NoLegacyRollback:
        def rollback_speculative_cache(self, *_args):
            raise AssertionError("upstream transaction used legacy rollback")

    model = NoLegacyRollback()

    partial = upstream_cache_state.SpeculativeCacheTransaction([], {}, [], length=3)
    vs_cache_state.commit_speculative_round(
        model, [], partial, accepted=0, block_size=3
    )
    assert partial.active is False

    complete = upstream_cache_state.SpeculativeCacheTransaction([], {}, [], length=3)
    vs_cache_state.commit_speculative_round(
        model, [], complete, accepted=2, block_size=3
    )
    assert complete.active is False

    aborted = upstream_cache_state.SpeculativeCacheTransaction([], {}, [], length=3)
    vs_cache_state.abort_speculative_round(aborted)
    assert aborted.active is False


def test_vendored_auto_processor_patch_requires_explicit_remote_code_opt_in(
    monkeypatch, tmp_path
):
    from transformers import AutoProcessor

    (tmp_path / "config.json").write_text('{"model_type":"glm5_next"}')
    calls = []

    def previous(cls, path, **kwargs):
        calls.append(("previous", path, kwargs))
        return "previous"

    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("custom", path, kwargs))
            return "custom"

    monkeypatch.setattr(AutoProcessor, "from_pretrained", classmethod(previous))
    vendored_base.install_auto_processor_patch("glm5_next", Processor)

    assert AutoProcessor.from_pretrained(tmp_path) == "previous"
    assert (
        AutoProcessor.from_pretrained(tmp_path, trust_remote_code=False) == "previous"
    )
    assert AutoProcessor.from_pretrained(tmp_path, trust_remote_code=True) == "custom"
    assert [kind for kind, _, _ in calls] == ["previous", "previous", "custom"]


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


def test_uniform_acceptance_clamps_over_positive_budgets():
    # r6 fix: pinned upstream mins the acceptance over every row, so a
    # retained finished row (zero budget under the non-filterable-cache
    # fallback) would collapse the whole batch to zero acceptance (dflash's
    # min(len(nt)-1) variant even stalled the loop). The vendored hunks
    # clamp over positive-budget rows and floor at 0.
    mx = pytest.importorskip("mlx.core")
    import rapid_mlx.models.mlx_vlm_vendored.speculative.common as vs_common_mod

    draft = mx.array([[7, 8, 9], [7, 8, 5]])
    target = mx.array([[7, 8, 9], [7, 8, 9]])
    # Row 0 is a retained finished row (zero budget) whose walk accepted 0;
    # row 1 still has tokens to spend and accepted 2. Upstream mins over ALL
    # rows → accepted=0 → row 1 collapses to bonus-only decoding; the
    # vendored hunk clamps over positive-budget rows and keeps row 1's
    # acceptance.
    out_accepted, out_tokens = vs_common_mod._speculative_walk_batch_uniform_acceptance(
        draft, target, [0, 2], [0, 4]
    )
    assert out_accepted == [2, 2]
    assert out_tokens[0] == []
    assert out_tokens[1] == [7, 8, 9]
    # All-zero budgets floor the clamp at 0 instead of crashing.
    out_accepted, out_tokens = vs_common_mod._speculative_walk_batch_uniform_acceptance(
        draft, target, [0, 2], [0, 0]
    )
    assert out_accepted == [0, 0]
    assert out_tokens == [[], []]


def test_hookless_mtp_verify_aborts_before_sink_retry(monkeypatch):
    """The hook-less verifier retries with a shared-KV sink only after
    rolling back its first forward, and returns the second transaction to the
    speculative-round owner."""
    mx = pytest.importorskip("mlx.core")

    class _Cache:
        def __init__(self):
            self.offset = 0

        def update_and_fetch(self, keys, values):
            self.offset += keys.shape[2]
            return keys, values

        def trim(self, count):
            self.offset -= count

    class _Model:
        layers = [SimpleNamespace(layer_type="attention")]

        def __init__(self):
            self.calls = 0

        def __call__(
            self,
            inputs,
            *,
            cache,
            shared_kv_sink=None,
            skip_final_norm=False,
        ):
            self.calls += 1
            width = inputs.shape[1]
            kv = mx.zeros((1, 1, width, 2))
            cache[0].update_and_fetch(kv, kv)
            if shared_kv_sink is not None:
                shared_kv_sink["attention"] = (kv, kv)
            return mx.zeros((1, width, 4))

    model = _Model()
    lm = SimpleNamespace(model=model)
    prompt_cache = [_Cache()]
    width = vs_common.DECODE_BLOCK_SIZE + 1
    result = vs_mtp._mtp_verify_without_logits(
        lm,
        mx.zeros((1, width), dtype=mx.int32),
        prompt_cache,
    )

    assert result is not None
    assert model.calls == 2
    assert prompt_cache[0].offset == width
    assert result.rollback_state.active is True
    result.abort()
    assert prompt_cache[0].offset == 0

    def fail_shared_kv(*_args):
        raise RuntimeError("bad shared KV")

    monkeypatch.setattr(vs_mtp, "_mtp_shared_kv_from_prompt_cache", fail_shared_kv)
    with pytest.raises(RuntimeError, match="bad shared KV"):
        vs_mtp._mtp_verify_without_logits(
            lm,
            mx.zeros((1, width), dtype=mx.int32),
            prompt_cache,
        )
    assert model.calls == 3
    assert prompt_cache[0].offset == 0


def test_server_singleton_dflash_threads_nonzero_row_identity(monkeypatch):
    """The server coordinator owns stable request row IDs. The separate
    ``run_speculative_rounds`` helper serves standalone generation, whose sole
    request intentionally owns row zero."""
    mx = pytest.importorskip("mlx.core")
    captured = {}

    def fake_rounds(*_args, **kwargs):
        captured["row_id"] = kwargs.get("row_id")
        yield 9, None

    monkeypatch.setattr(vs_utils, "_dflash_rounds", fake_rounds)
    output = list(
        vs_utils.run_speculative_server_rounds(
            SimpleNamespace(),
            SimpleNamespace(requires_greedy_sampling=False),
            [],
            mx.zeros((1, 1, 1)),
            draft_kind="dflash",
            first_bonus=mx.array([7]),
            max_tokens=2,
            sampler=lambda logits: logits,
            row_ids=[41],
        )
    )

    assert output == [([9], None)]
    assert captured["row_id"] == 41
