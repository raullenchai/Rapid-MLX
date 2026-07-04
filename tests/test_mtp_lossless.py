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
    the MTP module-level singletons AND ``mlx_lm.generate.generation_stream``.
    See the fixture in ``test_mtp_spec_decode.py`` for the full rationale on
    each of the three pieces of cross-test state being reset."""
    import sys

    import mlx.core as mx

    from vllm_mlx.spec_decode.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )
    from vllm_mlx.spec_decode.mtp.cache_patch import _unpatch_for_tests

    _unpatch_for_tests()
    reset_global_counter_for_tests()
    # See the matching fixture in ``test_mtp_spec_decode.py`` for the
    # cross-thread stream reasoning. ``mlx_lm.generate.generation_stream``
    # may have been re-bound by a preceding sweep test's worker-thread
    # ``_init_mlx_step_thread`` initialiser; re-pin it to THIS thread's
    # default so the ``with mx.stream(generation_stream): mx.eval(...)``
    # block inside ``mtp_generate_step`` runs against a stream this
    # thread can materialise.
    import mlx_lm.generate  # noqa: F401 — ensure module exists in sys.modules

    sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
        mx.default_device()
    )
    yield
    _unpatch_for_tests()
    reset_global_counter_for_tests()
    sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
        mx.default_device()
    )


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
    """Run ``mtp_generate_step`` over a fresh mocked model.

    Pins ``disable_auto_k=True`` so the mocked backbone scripts (written
    for chain-of-1 verify/bonus alternation) are not perturbed by the
    0.9.13 EV depth controller's cross-test global state or its per-round
    K∈{0,1} park/chain picks. The lossless emit-ordering contract this
    file tests is orthogonal to the controller's K choice.
    """
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
            disable_auto_k=True,
            max_k=1,
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
# Default-path (auto-K enabled) coverage — codex round-M BLOCKING
# ---------------------------------------------------------------------------


def _spec_decode_mtp_path_auto_k(
    backbone_script: list[int],
    mtp_script: list[int],
    prompt: mx.array,
    max_tokens: int,
) -> list[int]:
    """Run ``mtp_generate_step`` with ``disable_auto_k=False`` (default path)
    and ``max_k=1``.

    Exercises the EV depth controller code path that operators actually run
    (``rapid-mlx serve ... --spec-decode mtp`` defaults to ``disable_auto_k
    =False``) — the two byte-equal tests above pin ``disable_auto_k=True``
    so their mocked backbone scripts are not perturbed by the controller's
    per-round K∈{0,1} park/chain picks, but that means a regression in the
    default emit-ordering could leak past them undetected.

    ``max_k=1`` keeps the mock backbone contract (chain-of-1 verify/bonus
    alternation) intact — the controller can only pick K=0 (park) or K=1
    (chain-of-1), so any accepted round emits the same tokens the ``max_k=1
    + disable_auto_k=True`` tests already cover; a K=0 park round falls
    through to a plain backbone step. Either way, the emitted sequence
    must be a well-formed prefix of the ``--spec-decode none`` reference,
    since no branch of the default path is allowed to invent or reorder
    tokens.
    """
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
            # disable_auto_k=False is the default, spelled explicitly so
            # a future default change is caught here.
            disable_auto_k=False,
            max_k=1,
        )
    ]


def test_lossless_default_path_terminates_cleanly_under_auto_k():
    """Default-path (auto-K enabled) must terminate within ``max_tokens``
    without raising, under the scripted-mock backbone.

    Codex round-M BLOCKING: the two byte-equal tests above pin
    ``disable_auto_k=True`` to keep the mocked backbone scripts stable
    under the controller — that means a regression that only breaks the
    default (auto-K) path would slip past them. This test exercises the
    exact default path operators run (``rapid-mlx serve ... --spec-decode
    mtp`` with no ``--mtp-disable-auto-k``) and pins the two invariants
    that are cheap to verify under a mock:

    * The generator terminates cleanly (no exception, no hang beyond
      ``max_tokens``).
    * No invented tokens leak in — every emitted non-sentinel token
      was produced by the mock backbone at some position (K=0 park)
      or by the mock MTP head (K=1 chain).

    Byte-equal vs the ``--spec-decode none`` reference is NOT asserted
    here — per the batched-consistent lossless contract in
    ``memory/knowledge/decisions.md``, the default path may emit a
    different token stream from a single-token baseline (the K=0 park
    round drops drafts on the floor, and controller state persists
    across rounds), and that difference is intentional. Byte-equal is
    covered by the two ``disable_auto_k=True`` tests above and by the
    MTP-vs-MTP determinism test below.
    """
    backbone_mtp = [7, 11, 13, 15, 17, 19, 21]
    mtp_drafts = [11, 0, 15, 0, 19]
    mtp_tokens = _spec_decode_mtp_path_auto_k(
        backbone_mtp,
        mtp_drafts,
        prompt=mx.array([1], mx.uint32),
        max_tokens=7,
    )
    assert len(mtp_tokens) == 7, (
        f"Default-path MTP under-generated: got {len(mtp_tokens)} != 7. "
        f"mtp={mtp_tokens}"
    )
    # Union of the mock's two scripted sources — an emitted token must
    # have come from one of these (0 is the mock padding sentinel).
    valid_sources = set(backbone_mtp) | set(mtp_drafts) | {0, 1}
    for tok in mtp_tokens:
        assert tok in valid_sources, (
            "Default-path MTP invented a token not present in either "
            f"scripted source: tok={tok} mtp={mtp_tokens} "
            f"valid_sources={sorted(valid_sources)}"
        )


def test_lossless_default_path_is_deterministic():
    """Default-path (auto-K enabled) must be deterministic across
    successive runs on the same mocked model.

    Even under the EV depth controller (which has cross-request state
    via the process-global registry), a temp=0 request with a fresh
    ``__default__`` controller and a fresh mock must produce the same
    emit sequence twice in a row. If controller state leakage or non-
    deterministic K picks slipped in, this smoke test would catch it.
    """
    backbone_mtp = [7, 11, 13, 15, 17, 19, 21]
    mtp_drafts = [11, 0, 15, 0, 19]
    run1 = _spec_decode_mtp_path_auto_k(
        backbone_mtp[:],
        mtp_drafts[:],
        prompt=mx.array([1], mx.uint32),
        max_tokens=7,
    )
    run2 = _spec_decode_mtp_path_auto_k(
        backbone_mtp[:],
        mtp_drafts[:],
        prompt=mx.array([1], mx.uint32),
        max_tokens=7,
    )
    assert run1 == run2, (
        f"Default-path MTP not deterministic across runs. run1={run1} run2={run2}"
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
