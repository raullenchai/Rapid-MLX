# SPDX-License-Identifier: Apache-2.0
"""Lossless contract integration test for MTP spec decode (R15-P1 #302).

The lossless contract says: for the SAME prompt + seed at temp=0,
``--spec-decode mtp`` must emit byte-identical decoded tokens to
``--spec-decode none``. Accept rate is a *speedup* signal; lossless-
ness is a *correctness* contract that holds at every accept rate
between 0% (every draft rejected) and 100% (every draft accepted).

The standard way to test this is to load a real Qwen3.5 / 3.6
checkpoint with MTP weights and run both paths back-to-back. That
requires:

* A 4-50 GB model download (Qwen3.5-9B-w4 is ~5 GB).
* GPU time on M-series silicon.
* The Stage B PonyExl3 Viterbi conversion to finish freeing the GPU
  (PID 56486 at vendoring time — see PR body for the deferred-bench
  note).

None of those is acceptable for a unit-test-tier integration test.
Instead, we use a deterministic mocked Qwen3.5-shaped model that:

* Returns scripted backbone logits (so we control verify/accept).
* Returns scripted MTP draft logits.
* Exposes the same four contract surfaces
  (``return_hidden``, ``n_confirmed``, ``mtp_forward``,
  ``make_mtp_cache``) the real ``inject_mtp_support`` adds.

We then run TWO full sequences through ``mtp_generate_step``:

1. **All-accept scenario** — the MTP head always proposes exactly
   what the backbone would have decoded next. Tokens emitted should
   match a synthetic ``--spec-decode none`` reference sequence
   computed by stepping the same mocked model forward one token at
   a time.
2. **Adversarial-reject scenario** — the MTP head proposes a token
   the backbone always rejects (random sentinel). The rejection
   path emits the verify_pred token instead, which is exactly the
   token the standard ``generate_step`` would have decoded. So the
   emitted sequence STILL matches the reference, just slower.

A passing test pins the contract: at temp=0, BOTH accept and reject
branches emit the same tokens the non-spec-decode path would have
emitted. Tests do NOT pin the per-step latency.

Why this is a meaningful lossless test
--------------------------------------

The two scripts above are the only two arithmetic paths through
``mtp_generate_step`` at temp=0:

* On accept: yield ``draft_tok`` (which equals ``verify_pred`` by
  the accept condition ``verify_pred.item() == draft_tok_id``).
* On reject: yield ``verify_pred`` directly.

Both paths therefore emit ``verify_pred`` — which is exactly what
``generate_step`` emits (it argmax's the same backbone logits).

If the lossless contract ever breaks at temp=0, it breaks here:

* If the accept comparison were wrong (e.g. ``!=`` instead of
  ``==``), the accept-path token would not match.
* If the verify_pred indexing were off-by-one
  (``hidden[:, 1, :]`` vs ``hidden[:, 0, :]``), the verify_pred
  computed from logits[:, 0, :] (which we EXPLICITLY script) would
  drift from the standard generate_step's argmax.
* If the rollback path failed to restore state between rejections,
  the next backbone call's logits would change shape and the
  scripted token would not match.

The bench script (``bench/bench_spec_decode_mtp.py``) covers the
end-to-end correctness check against a real Qwen3.5 checkpoint when
the GPU is free — see PR body for the follow-up plan.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")


# Re-import the mocked model from the unit-test file so the contract
# only has ONE definition that needs to be maintained as the
# generator evolves.
from tests.test_mtp_spec_decode import _MockedQwen35Model  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_mtp_module_state():
    """Mirror the autouse teardown installed in ``test_mtp_spec_decode.py``
    so this file's tests are also robust to sweep-ordering state leak from
    the MTP module-level singletons (``cache_patch._patched`` install gate
    + ``accept_counter._global_counter`` monotonic singleton). See the
    fixture in ``test_mtp_spec_decode.py`` for the full rationale."""
    from vllm_mlx.spec_decode.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )
    from vllm_mlx.spec_decode.mtp.cache_patch import _unpatch_for_tests

    _unpatch_for_tests()
    reset_global_counter_for_tests()
    yield
    _unpatch_for_tests()
    reset_global_counter_for_tests()


def _generate_step_none_path(
    model: _MockedQwen35Model,
    prompt: mx.array,
    max_tokens: int,
) -> list[int]:
    """Run a synthetic ``--spec-decode none`` step over the same mocked model.

    The ``none`` path is just argmax over backbone logits at the last
    position, then feeds the token forward. We don't bring in the
    full ``generate_step`` machinery — it requires a real cache list
    matching ``model.layers`` and would couple this test to mlx-lm
    internals. The minimal simulation here is enough to establish
    the reference token sequence the lossless contract must match.

    Args:
        model: A fresh ``_MockedQwen35Model`` instance. Its scripted
            backbone outputs determine what each forward returns.
        prompt: Length-1 prompt — multi-token prompts are not
            supported by this minimal simulator (prefill handling is
            the generator's job).
        max_tokens: Number of decode tokens to produce.

    Returns:
        The emitted token sequence as a Python list.
    """
    assert prompt.size == 1, "minimal reference path supports length-1 prompts"
    emitted: list[int] = []
    y = prompt
    for _ in range(max_tokens):
        logits = model(y[None], cache=None, return_hidden=False, n_confirmed=0)
        tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        emitted.append(tok)
        y = mx.array([tok], mx.uint32)
    return emitted


def _spec_decode_mtp_path(
    backbone_script: list[int],
    mtp_script: list[int],
    prompt: mx.array,
    max_tokens: int,
) -> list[int]:
    """Run ``mtp_generate_step`` over a fresh mocked model."""
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    model = _MockedQwen35Model(backbone_script, mtp_script)
    counter = MTPAcceptCounter()
    return [
        tok
        for tok, _lp, _fd in mtp_generate_step(
            prompt,
            model,
            max_tokens=max_tokens,
            accept_counter=counter,
        )
    ]


# ---------------------------------------------------------------------------
# All-accept lossless test
# ---------------------------------------------------------------------------


def test_lossless_temp0_all_accept_matches_none_reference():
    """When every MTP draft matches the verify_pred, the emitted
    sequence must byte-match the non-spec-decode reference.

    Construction:

    * Backbone script for ``--spec-decode mtp``: alternating
      (verify_pred, bonus) pairs after the primary, where each
      verify_pred matches the previous draft AND the bonus is what
      the next draft will be (chained accept).
    * MTP script: the same sequence as the bonus tokens, so every
      draft is accepted.

    For the ``--spec-decode none`` reference, the backbone is queried
    one token at a time and yields the same sequence as the union of
    (primary, draft_i, bonus_i) above (because the bonus IS what the
    backbone would emit next).
    """
    # MTP path: primary=7, then 3 verify/bonus pairs.
    #   Primary (S=1): 7
    #   Verify1 (S=2, pred=11, bonus=13): accept draft1=11; ntoks=3
    #   Verify2 (S=2, pred=15, bonus=17): accept draft2=15; ntoks=5
    #   Verify3 (S=2, pred=19, bonus=21): accept draft3=19; ntoks=7
    #
    # The bonus token from step i becomes the verify token's "context"
    # for step i+1 in the generator. The drafts (11, 15, 19) match
    # the verify preds, so every accept fires.
    backbone_mtp = [
        7,  # cold-start primary
        11,
        13,  # verify1
        15,
        17,  # verify2
        19,
        21,  # verify3
    ]
    # MTP head proposes the same sequence the backbone would emit
    # next. Cache-commit MTP calls consume 2 slots (a sentinel + the
    # next draft); the cold-start MTP call consumes 1 slot.
    mtp_drafts = [
        11,  # cold-start draft1
        0,
        15,  # cache_commit after accept1: sentinel + draft2
        0,
        19,  # cache_commit after accept2: sentinel + draft3
    ]
    mtp_tokens = _spec_decode_mtp_path(
        backbone_mtp,
        mtp_drafts,
        prompt=mx.array([1], mx.uint32),
        max_tokens=7,
    )

    # Reference path. The "none" path just walks the backbone one
    # token at a time. We script its backbone to emit the SAME
    # sequence (7, 11, 13, 15, 17, 19, 21) so an apples-to-apples
    # comparison is possible.
    none_model = _MockedQwen35Model([7, 11, 13, 15, 17, 19, 21], [])
    none_tokens = _generate_step_none_path(
        none_model, prompt=mx.array([1], mx.uint32), max_tokens=7
    )

    assert mtp_tokens == none_tokens, (
        "All-accept MTP path must emit byte-identical tokens to the "
        f"--spec-decode none reference. mtp={mtp_tokens} none={none_tokens}"
    )


# ---------------------------------------------------------------------------
# All-reject lossless test
# ---------------------------------------------------------------------------


def test_lossless_temp0_all_reject_matches_none_reference():
    """Adversarial: every MTP draft mismatches the verify_pred. The
    reject path should emit the verify_pred itself, so the sequence
    still matches the non-spec-decode reference (just slower).
    """
    # Backbone for MTP path. Primary=7. Then every verify backbone
    # call returns (verify_pred, bonus). The MTP head proposes a
    # sentinel that never matches, so every verify is a reject.
    # The reject path emits verify_pred and re-runs cold-start MTP
    # for the next draft (without a cache_commit, so S=1).
    backbone_mtp = [
        7,  # cold-start primary
        11,
        99,  # verify1: pred=11 ≠ draft1, bonus=99 (unused on reject)
        13,
        99,  # verify2: pred=13 ≠ draft2, bonus=99
        15,
        99,  # verify3: pred=15 ≠ draft3, bonus=99
        17,  # final cold-start backbone after the last reject
    ]
    # MTP proposes a sentinel that won't match any verify_pred.
    mtp_drafts = [
        9001,  # cold-start draft1 — won't match 11
        9002,  # cold-start draft2 (post-reject1) — won't match 13
        9003,  # cold-start draft3 (post-reject2) — won't match 15
        9004,  # cold-start draft4 (post-reject3) — used after final emit
    ]
    mtp_tokens = _spec_decode_mtp_path(
        backbone_mtp,
        mtp_drafts,
        prompt=mx.array([1], mx.uint32),
        max_tokens=4,
    )

    # Reference: backbone emits primary then the three verify_preds
    # (each reject yields the verify_pred). Sequence: [7, 11, 13, 15].
    none_model = _MockedQwen35Model([7, 11, 13, 15], [])
    none_tokens = _generate_step_none_path(
        none_model, prompt=mx.array([1], mx.uint32), max_tokens=4
    )

    assert mtp_tokens == none_tokens, (
        "All-reject MTP path must still emit the verify_pred and "
        "therefore match the --spec-decode none reference. "
        f"mtp={mtp_tokens} none={none_tokens}"
    )


# ---------------------------------------------------------------------------
# Sanity: the unit-test runner picks up this file
# ---------------------------------------------------------------------------


def test_lossless_test_module_smoke():
    """The lossless contract test module must be discoverable.

    Empty assertion — the pytest collection itself is the check. If
    the import / fixture path breaks, this whole module fails to
    collect, which is the signal we want to see in CI.
    """
    assert True
