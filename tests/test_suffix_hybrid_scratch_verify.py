# SPDX-License-Identifier: Apache-2.0
"""Hybrid scratch-verify battery for the opt-in suffix_hybrid path (#2561 B2.1).

The B1 slice (#3090) gave Qwen4ExpStateCache / QSAIndexCache a ``cache_rollback``
contract. Empirically that contract cannot support a multi-token rejection on the
real hybrid capture: ``record_slot_snapshots(finalize=True)`` publishes *K*
boundaries for a K-token verify (not K+1), so ``can_trim(1)`` is ``False`` once a
K-token verify has run, and ``Qwen4ExpStateCache.trim``/``trim_all`` cannot roll a
rejected tail back. The hybrid path therefore mirrors SGLang NGRAMWorker + the
scheduler's DSpark path: verify against a copy of the committed cache (scratch),
then on accept replay ONLY the accepted prefix onto the live cache (commit-only),
dropping the rejected tail by never advancing the live cache past it. This is
step-equivalent and lossless.

These tests exercise the real tiny Qwen4-exp hybrid model (GDN recurrent + PLE +
QSA attention) so the state-equality claims are non-tautological.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from dataclasses import asdict  # noqa: E402

import numpy as np  # noqa: E402
from mlx_lm.models.cache import CacheList  # noqa: E402

from tests.test_qwen4_exp_vendored import _ple_args  # noqa: E402
from vllm_mlx.models.qwen4_exp import Model, ModelArgs  # noqa: E402
from vllm_mlx.models.qwen4_exp_cache import (  # noqa: E402
    Qwen4ExpStateCache,
)
from vllm_mlx.scheduler import (  # noqa: E402
    SchedulerConfig,
    _commit_scratch_accepted,
    _verify_scratch,
)


def _offsets(cache):
    return [
        (
            None
            if not isinstance(lc, CacheList)
            else (int(lc[0].offset), int(lc[1].offset))
        )
        for lc in cache
    ]


def _assert_state_equal(a, b):
    """Recurrent tensors + attention/QSA offsets + raw ring exact equality."""
    assert len(a) == len(b)
    for la, lb in zip(a, b):
        if isinstance(la, Qwen4ExpStateCache):
            assert isinstance(lb, Qwen4ExpStateCache)
            assert len(la.cache) == len(lb.cache)
            for x, y in zip(la.cache, lb.cache):
                if x is None and y is None:
                    continue
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        elif isinstance(la, CacheList):
            assert isinstance(lb, CacheList)
            assert la[0].offset == lb[0].offset
            assert la[1].offset == lb[1].offset
            qa, qb = la[1], lb[1]
            assert qa._offsets == qb._offsets
            assert qa._compressed_counts == qb._compressed_counts
            ra, rb = qa.raw_ring, qb.raw_ring
            np.testing.assert_array_equal(np.asarray(ra), np.asarray(rb))
        else:  # pragma: no cover - unexpected layer type
            pytest.fail(f"unexpected cache layer: {type(la)!r}")


def _model_and_prompt():
    args = _ple_args()
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(args)))
    cache = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3]]), cache=cache))
    return model, cache


class TestScratchVerifyCore:
    """Direct (unit-level) tests of the scratch verify helper."""

    def test_accept_all_commits_all(self):
        import copy

        model, cache = _model_and_prompt()
        # Build the draft FROM the model's actual stepwise greedy predictions,
        # so both draft tokens are guaranteed to match the verify and accept
        # UNCONDITIONALLY (no ``if``-gated assertion). Probe on a THROWAWAY
        # copy so the live ``cache`` is not advanced by the probe.
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))

        def _next_step(tok):
            logits = model(mx.array([[tok]]), cache=probe)
            mx.eval(logits)
            return int(mx.argmax(logits[:, -1], axis=-1).item())

        x = 4
        d0 = _next_step(x)  # greedy next after [1,2,3,4]
        d1 = _next_step(d0)  # greedy next after [1,2,3,4,d0]

        draft = [d0, d1]
        verify_input = mx.array([[x, d0, d1]])
        commit_head = copy.deepcopy(cache)
        result = _verify_scratch(model, cache, verify_input, draft, commit_head)
        # Both draft tokens are the model's greedy stepwise predictions, so
        # the chunked verify MUST accept both — assert it unconditionally.
        assert result["n_accepted"] == 2
        # Commit accepted (whatever it is) and compare to the stepwise gold
        # of THAT same accepted prefix: gold = prompt + committed [x, d0].
        na = result["n_accepted"]
        _commit_scratch_accepted(model, cache, verify_input, na)
        g2 = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=g2))
        for t in ([x], [d0], [d1])[: na + 1]:
            mx.eval(model(mx.array([t]), cache=g2))
        _assert_state_equal(cache, g2)

    def test_partial_reject_commits_only_accepted(self):
        import copy

        model, _cache = _model_and_prompt()
        # Build a draft with token 0 accepted and token 1 rejected
        # UNCONDITIONALLY, from the model's ACTUAL greedy predictions (not a
        # guess): d0 = greedy next after X; d1 = (greedy next after d0) + 1,
        # which is guaranteed to mismatch the position-1 pred.
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))

        def _next_step(tok):
            logits = model(mx.array([[tok]]), cache=probe)
            mx.eval(logits)
            return int(mx.argmax(logits[:, -1], axis=-1).item())

        x = 4
        d0 = _next_step(x)  # greedy next after [1,2,3,4] -> accepted at pos 0
        g1 = _next_step(d0)  # greedy next after [1,2,3,4,d0]
        # A guarantee-rejected token: the model's greedy choice plus one,
        # wrapped within the vocabulary so ``g1 == last_vocab_id`` cannot
        # produce an out-of-range token that breaks the embedding lookup.
        d1 = (g1 + 1) % 32  # guaranteed rejected at position 1

        live = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=live))
        commit_head = copy.deepcopy(live)
        verify_input = mx.array([[x, d0, d1]])
        result = _verify_scratch(model, live, verify_input, [d0, d1], commit_head)
        assert result["n_accepted"] == 1
        _commit_scratch_accepted(model, live, verify_input, 1)
        # gold: prompt + committed [4, d0]
        g2 = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=g2))
        mx.eval(model(mx.array([[4]]), cache=g2))
        mx.eval(model(mx.array([[d0]]), cache=g2))
        _assert_state_equal(live, g2)

    def test_reject_first_commits_nothing(self):
        import copy

        model, _cache = _model_and_prompt()
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))
        vlog = model(mx.array([[4, 5, 6]]), cache=probe)
        mx.eval(vlog)
        preds = mx.argmax(vlog, axis=-1).tolist()[0]
        d0 = (preds[0] + 1) % 32  # guaranteed rejected at position 0
        d1 = preds[1]

        before = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=before))
        commit_head = copy.deepcopy(before)
        verify_input = mx.array([[4, d0, d1]])
        result = _verify_scratch(model, before, verify_input, [d0, d1], commit_head)
        assert result["n_accepted"] == 0
        # No commit → cache bit-identical to a fresh unadvanced prompt cache.
        fresh = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=fresh))
        _assert_state_equal(before, fresh)


class TestHybridInstallGate:
    """suffix_hybrid must open the hybrid install gate, default stays closed."""

    def _make_fake_bg(self):
        class _GB:
            pass

        gb = _GB()
        gb._step = lambda: ([], [])
        gb.next = lambda: []

        class _BG:
            def __init__(self):
                self.removed = []

            def remove(self, uids, return_prompt_caches=False):
                self.removed.append(list(uids))
                return {}

        bg = _BG()
        bg._generation_batch = gb
        return bg, gb

    def test_hybrid_profile_without_optin_still_skips(self):
        from unittest.mock import MagicMock

        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.scheduler import _install_suffix_decoding

        bg, gb = self._make_fake_bg()
        orig_step = gb._step
        profile = ModelConfig(is_hybrid=True, supports_spec_decode=False)
        assert not _install_suffix_decoding(
            bg,
            model=MagicMock(),
            profile=profile,
            max_draft=8,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
        )
        assert gb._step is orig_step  # skipped install leaves the step untouched

    def test_hybrid_profile_with_optin_installs(self):
        from unittest.mock import MagicMock

        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.scheduler import _install_suffix_decoding

        bg, gb = self._make_fake_bg()
        orig_step = gb._step
        profile = ModelConfig(is_hybrid=True, supports_spec_decode=False)
        assert _install_suffix_decoding(
            bg,
            model=MagicMock(),
            profile=profile,
            max_draft=8,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
            suffix_hybrid=True,
        )
        assert gb._step is not orig_step
        assert hasattr(bg, "_suffix_stats")
        assert bg._suffix_stats["suffix_hybrid"] is True
        assert bg._suffix_stats["suffix_min_match_len"] == 24

    def test_hybrid_drafter_reaches_floor_but_pure_attention_keeps_cap(
        self, monkeypatch
    ):
        """Final design (rounds 5-7): on the HYBRID path the drafter cap is
        raised to at least the 24-token match floor so the opt-in feature is
        not a silent no-op; on PURE-ATTENTION (where --suffix-hybrid is a
        no-op flag) the configured ``max_draft`` is preserved untouched —
        the hybrid cap raise must NOT leak into pure-attention models."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from vllm_mlx import scheduler
        from vllm_mlx.model_auto_config import ModelConfig

        def _install_opts(max_draft, is_hybrid):
            bg, gb = self._make_fake_bg()
            # Pure-attention installs need supports_spec_decode=True (the
            # hybrid opt-in path bypasses that gate); hybrid is gated only on
            # is_hybrid + suffix_hybrid.
            profile = ModelConfig(
                is_hybrid=is_hybrid, supports_spec_decode=not is_hybrid
            )
            assert scheduler._install_suffix_decoding(
                bg,
                model=MagicMock(),
                profile=profile,
                max_draft=max_draft,
                max_suffix_len=4,
                min_confidence=0.3,
                requests={},
                uid_to_request_id={},
                suffix_hybrid=True,
                suffix_min_match_len=24,
            )
            gb._next_tokens = mx.array([1], dtype=mx.int32)
            gb._num_tokens = [0]
            gb.max_tokens = [100]
            gb.uids = [1]
            gb.tokens = [[]]
            gb.state_machines = [SimpleNamespace(match=lambda s, _t: (s, None, None))]
            gb._matcher_states = [None]
            gb.logits_processors = []
            gb.model = MagicMock()
            gb._orig_step = lambda: ([1], [])
            gb._step()  # lazy-init constructs the drafter this step
            return gb._suffix_drafters[1]

        # The installer HONORS ``max_draft`` on every path (CLI reject-projects
        # a below-floor hybrid combo before it reaches here). The hybrid width
        # is seeded at min(floor, max_draft); the drafter's per-step width
        # equals that seed.
        #
        # (a) Hybrid, configured cap AT the floor: width = the 24-token floor.
        drafter24 = _install_opts(24, is_hybrid=True)
        assert drafter24.max_draft_tokens == 24
        # (b) Hybrid, cap ABOVE the floor: width still seeds at the floor
        # (adaptively grows toward the cap only on full-acceptance).
        drafter32 = _install_opts(32, is_hybrid=True)
        assert drafter32.max_draft_tokens == 24
        # (c) Hybrid, BELOW-floor cap (programmatic; CLI rejects this): width
        # degrades to the cap — no silent raise above the documented limit.
        drafter8 = _install_opts(8, is_hybrid=True)
        assert drafter8.max_draft_tokens == 8
        # (d) PURE-ATTENTION, same flag: width stays at the normal adaptive
        # start (2) — the hybrid floor must NOT leak into a no-op-flag model.
        drafter_pa8 = _install_opts(8, is_hybrid=False)
        assert drafter_pa8.max_draft_tokens == 2
        # (e) Pure-attention, explicit higher cap: also not pushed to the floor.
        drafter_pa32 = _install_opts(32, is_hybrid=False)
        assert drafter_pa32.max_draft_tokens == 2

    def test_installed_reject_first_commits_primary(self, monkeypatch):
        """Installed-path reject-first verify: the live cache must equal
        prompt + X after ``gb._step()``.

        A reject-first draft (n_accepted == 0) still emits X (the greedy
        primary). The hybrid commit-only path must therefore commit the
        [X] prefix even when no draft token is accepted — otherwise the live
        cache is left BEFORE X and the next decode step reads stale state.
        This exercises ``_hybrid_scratch_verify`` + ``_suffix_step`` (not the
        helper directly), so it guards the production zero-accept path.
        """
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from vllm_mlx import scheduler
        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.speculative import suffix_decoding

        model, _c = _model_and_prompt()
        # Probe the greedy next token on a throwaway copy (X = 7, the token
        # we're about to feed as the primary). Build a draft whose token 0 is
        # guaranteed rejected.
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))
        l0 = model(mx.array([[7]]), cache=probe)
        mx.eval(l0)
        d0 = (int(mx.argmax(l0[:, -1], axis=-1).item()) + 1) % 32  # rejected
        d1 = (d0 + 1) % 32
        draft = [d0, d1]

        class Drafter:
            max_draft_tokens = 2

            def __init__(self, **_kw):
                pass

            def add_prompt_tokens(self, _t):
                pass

            def add_generated_token(self, _t):
                pass

            def record_acceptance(self, _c):
                pass

            def get_draft(self):
                return list(draft)

        cache = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=cache))
        gb = SimpleNamespace(
            _next_tokens=mx.array([7], dtype=mx.int32),
            _next_logprobs=[mx.zeros((64,))],
            uids=[1],
            tokens=[[]],
            logits_processors=[],
            prompt_cache=cache,
            _num_tokens=[0],
            max_tokens=[10],
            state_machines=[SimpleNamespace(match=lambda s, _t: (s, None, None))],
            _matcher_states=[None],
            extract_cache=lambda _r: [],
            model=model,
        )
        gb.Response = SimpleNamespace
        gb._step = lambda: ([7], [])
        gb.next = lambda: []
        bg = MagicMock()
        bg._generation_batch = gb
        bg.remove = lambda *a, **k: {}
        profile = ModelConfig(is_hybrid=True, supports_spec_decode=False)
        monkeypatch.setattr(suffix_decoding, "SuffixDecodingDrafter", Drafter)
        scheduler._install_suffix_decoding(
            bg,
            model=model,
            profile=profile,
            max_draft=2,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
            suffix_hybrid=True,
            suffix_min_match_len=2,
        )
        result = gb._step()
        assert result[0] == [7]
        # Live cache (now ``gb.prompt_cache`` — the guard-on commit path SWAPS
        # in the probe head by rebinding, leaving the original pristine list as
        # the retained replay snapshot) must equal prompt + [7] (X committed
        # despite reject).
        gold = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=gold))
        mx.eval(model(mx.array([[7]]), cache=gold))
        _assert_state_equal(gb.prompt_cache, gold)
        # Finding (round-8 #2): reject-first (n_accepted == 0) has NO synthetic
        # emits, so no pristine replay head should be retained — the live cache
        # already holds exactly what will be surfaced.
        assert 1 not in gb._suffix_hybrid_replay

    def test_pure_attention_optin_ignored(self):
        """suffix_hybrid on a pure-attention model is a no-op (not a raised error)."""
        from unittest.mock import MagicMock

        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.scheduler import _install_suffix_decoding

        bg, gb = self._make_fake_bg()
        profile = ModelConfig()  # supports_spec_decode=True, not hybrid
        assert _install_suffix_decoding(
            bg,
            model=MagicMock(),
            profile=profile,
            max_draft=8,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
            suffix_hybrid=True,
        )
        assert bg._suffix_stats["suffix_hybrid"] is False


class TestMinMatchFloor:
    """Drafts shorter than suffix_min_match_len fall through, no verify."""

    def _install(self, suffix_min_match_len=24, monkeypatch=None, **kw):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import mlx.core as mx

        from vllm_mlx import scheduler
        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.speculative import suffix_decoding

        class Drafter:
            def __init__(self, **_kw):
                self.max_draft_tokens = 8
                self.draft = kw.get("draft", [5, 6])

            def add_prompt_tokens(self, _t):
                pass

            def add_generated_token(self, _t):
                pass

            def get_draft(self):
                return list(self.draft)

            def record_acceptance(self, _c):
                pass

        class Cache:
            def can_advance(self, _n):
                return True

        if monkeypatch:
            monkeypatch.setattr(suffix_decoding, "SuffixDecodingDrafter", Drafter)
        bg = MagicMock()
        gb = SimpleNamespace(
            _next_tokens=mx.array([1], dtype=mx.int32),
            _next_logprobs=[mx.zeros((5,))],
            uids=[1],
            tokens=[[]],
            logits_processors=[],
            prompt_cache=[Cache()],
            _num_tokens=[0],
            max_tokens=[10],
            state_machines=[SimpleNamespace(match=lambda s, _t: (s, None, None))],
            _matcher_states=[None],
            extract_cache=lambda _r: [],
        )
        gbResponse = SimpleNamespace
        gb.Response = gbResponse
        gb._step = lambda: ([1], [])
        gb.next = lambda: []
        bg._generation_batch = gb
        bg.remove = lambda *a, **k: {}
        profile = ModelConfig(is_hybrid=True, supports_spec_decode=False)
        scheduler._install_suffix_decoding(
            bg,
            model=MagicMock(),
            profile=profile,
            max_draft=8,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
            suffix_hybrid=True,
            suffix_min_match_len=suffix_min_match_len,
        )
        return bg, gb

    def test_short_draft_falls_through_without_verify(self, monkeypatch):
        _bg, gb = self._install(suffix_min_match_len=24, monkeypatch=monkeypatch)
        # nb: drafter returns [5,6] (len 2 < 24) -> falls through, no verify.
        gb._suffix_uid_state = {
            1: {"cooldown": 0, "level": 0, "zeros": 0, "k": 2, "pending": []}
        }
        # Patch a model that would fail if called (verify must NOT run).
        calls = {"n": 0}

        def boom(*a, **k):
            calls["n"] += 1
            raise AssertionError("verify should not run")

        gb.model = boom
        gb._orig_step = lambda: ([1], [])
        # reset the stats/uid state wiring by reaching into installed closure
        result = gb._step()
        assert result[0] == [1]
        assert calls["n"] == 0
        assert _bg._suffix_stats["ft_short_match"] == 1

    def test_long_enough_draft_runs_verify(self, monkeypatch):
        _bg, gb = self._install(suffix_min_match_len=2, monkeypatch=monkeypatch)
        gb._suffix_uid_state = {
            1: {"cooldown": 0, "level": 0, "zeros": 0, "k": 2, "pending": []}
        }

        class Cache2:
            def can_advance(self, _n):
                return True

        # A real-ish model returning greedy-matching logits.
        import mlx.core as mx

        calls = {"n": 0}

        def fake_model(tokens, cache=None, **kw):
            calls["n"] += 1
            L = tokens.shape[1]
            logits = mx.full((1, L, 5), -10.0)
            # pred at each position = the draft token at that position (so
            # everything accepted)
            for i in range(L):
                tok = int(tokens[0, i].item())
                logits[0, i, (tok + 1) % 4] = 10.0
            return logits

        gb.model = fake_model
        gb._orig_step = lambda: ([1], [])
        result = gb._step()
        assert result[0] == [1]  # primary returned
        assert calls["n"] >= 1
        assert _bg._suffix_stats["verify_steps"] >= 1


class TestTokenIdentityAndStateEquality:
    """Real-model, multi-step: token-identity vs baseline + state equality.

    The tiny Qwen4-exp hybrid is step-exact (chunked == stepwise for argmax),
    so a scratch-verify decode that commits exactly the greedy-accepted prefix
    must be token-identical to baseline greedy decode, and the cache state
    must match at every step.
    """

    def _stepwise_next(self, model, cache, inp_tok):
        logits = model(mx.array([[inp_tok]]), cache=cache)
        mx.eval(logits)
        return int(mx.argmax(logits[:, -1], axis=-1).item())

    def test_identity_over_repeated_drafts(self):
        import copy

        model, _ = _model_and_prompt()
        # Prompt-only live cache (X=7 NOT committed yet).
        cache = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=cache))
        # Probe the greedy next tokens on a THROWAWAY copy so the live cache
        # is not advanced by the probe (production feeds X during verify).
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))
        d0 = self._stepwise_next(model, probe, 7)
        d1 = self._stepwise_next(model, probe, d0)
        verify_input = mx.array([[7, d0, d1]])
        draft = [d0, d1]
        commit_head = copy.deepcopy(cache)
        res = _verify_scratch(model, cache, verify_input, draft, commit_head)
        na = res["n_accepted"]
        # d0,d1 ARE the greedy stepwise next tokens, so accept-all (2).
        _commit_scratch_accepted(model, cache, verify_input, na)
        assert na == 2
        # Committed state == baseline greedy commit of [7, d0, d1].
        gold = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=gold))
        for t in ([7], [d0], [d1]):
            mx.eval(model(mx.array([t]), cache=gold))
        _assert_state_equal(cache, gold)

    def test_full_reject_leaves_committed_state_intact(self):
        import copy

        model, _ = _model_and_prompt()
        cache = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=cache))
        # A draft whose token 0 is guaranteed rejected (probe on a copy).
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))
        d0 = (self._stepwise_next(model, probe, 7) + 1) % 32
        verify_input = mx.array([[7, d0, (d0 + 1) % 32]])
        draft = [d0, (d0 + 1) % 32]
        commit_head = copy.deepcopy(cache)
        res = _verify_scratch(model, cache, verify_input, draft, commit_head)
        assert res["n_accepted"] == 0
        _commit_scratch_accepted(model, cache, verify_input, 0)
        # reject-first still advances X=7 (the primary is always emitted):
        # committed prefix = [7]. State must equal prompt + [7].
        gold = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=gold))
        mx.eval(model(mx.array([[7]]), cache=gold))
        _assert_state_equal(cache, gold)

    def test_prefix_cache_reuse_after_commit(self):
        """A committed scratch prefix is reusable as a prefix-cache source."""
        import copy

        model, _ = _model_and_prompt()
        cache = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=cache))
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))
        d0 = self._stepwise_next(model, probe, 7)
        verify_input = mx.array([[7, d0, (d0 + 1) % 32]])
        draft = [d0, (d0 + 1) % 32]
        commit_head = copy.deepcopy(cache)
        res = _verify_scratch(model, cache, verify_input, draft, commit_head)
        na = res["n_accepted"]
        _commit_scratch_accepted(model, cache, verify_input, na)
        # The committed prefix cache must remain directly decodable (i.e. it
        # is a valid prefix-cache source that the next step can continue from).
        next_tok = model(mx.array([[8]]), cache=cache)
        mx.eval(next_tok, [c.state for c in cache])
        assert next_tok.shape[-1] > 0


class TestBitExactnessGuard:
    """The opt-in hybrid drift guard refuses (falls through) on any chunked
    verify that disagrees with stepwise greedy, without committing."""

    def test_guard_does_not_misfire_on_step_exact_model(self, monkeypatch):
        # On the tiny (step-exact) model, enabling the guard must not change
        # the accepted commit — the drift probe matches the chunked preds.

        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from vllm_mlx import scheduler
        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.speculative import suffix_decoding

        model, _ = _model_and_prompt()
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))
        # greedy next two tokens (accept-all draft)
        l0 = model(mx.array([[7]]), cache=probe)
        mx.eval(l0)
        d0 = int(mx.argmax(l0[:, -1], axis=-1).item())
        l1 = model(mx.array([[d0]]), cache=probe)
        mx.eval(l1)
        d1 = int(mx.argmax(l1[:, -1], axis=-1).item())
        greedy_draft = [d0, d1]

        drafter = [greedy_draft]

        class MockDrafter:
            max_draft_tokens = 2

            def __init__(self, **_kw):
                pass

            def add_prompt_tokens(self, _t):
                pass

            def add_generated_token(self, _t):
                pass

            def record_acceptance(self, _c):
                pass

            def get_draft(self):
                return list(drafter[0])

        cache = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=cache))
        gb = SimpleNamespace(
            _next_tokens=mx.array([7], dtype=mx.int32),
            _next_logprobs=[mx.zeros((64,))],
            uids=[1],
            tokens=[[]],
            logits_processors=[],
            prompt_cache=cache,
            _num_tokens=[0],
            max_tokens=[10],
            state_machines=[SimpleNamespace(match=lambda s, _t: (s, None, None))],
            _matcher_states=[None],
            extract_cache=lambda _r: [],
            model=model,
        )
        gb.Response = SimpleNamespace
        gb._step = lambda: ([7], [])
        gb.next = lambda: []
        bg = MagicMock()
        bg._generation_batch = gb
        bg.remove = lambda *a, **k: {}
        profile = ModelConfig(is_hybrid=True, supports_spec_decode=False)
        monkeypatch.setattr(suffix_decoding, "SuffixDecodingDrafter", MockDrafter)
        scheduler._install_suffix_decoding(
            bg,
            model=model,
            profile=profile,
            max_draft=2,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
            suffix_hybrid=True,
            suffix_min_match_len=2,
            suffix_hybrid_bit_exact=True,
        )
        # The step-exact model's chunked verify == stepwise, so the guard
        # passes and the verify runs (no drift refusal, accept-all commit).
        result = gb._step()
        assert result[0] == [7]
        assert bg._suffix_stats["hybrid_drifts"] == 0
        assert bg._suffix_stats["verify_steps"] == 1

    def test_guard_refuses_on_drift(self, monkeypatch):
        # Force drift: a fake model whose chunked (multi-token) forward
        # disagrees with its own stepwise single-token forward.

        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import mlx.core as mx

        from vllm_mlx import scheduler
        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.speculative import suffix_decoding

        calls = {"verify": 0, "step": 0}

        class DriftingModel:
            def __call__(self, tokens, cache=None, **kw):
                L = tokens.shape[1]
                if L > 1:  # chunked verify forward
                    calls["verify"] += 1
                    logits = mx.full((1, L, 8), -10.0)
                    # chunked predicts draft[0]==draft[1] at each position
                    for i in range(L):
                        logits[0, i, 5] = 10.0  # always predicts token 5
                    return logits
                # single token stepwise
                calls["step"] += 1
                logits = mx.full((1, 1, 8), -10.0)
                logits[0, 0, 6] = 10.0  # stepwise predicts token 6
                return logits

        cache = [SimpleNamespace(offset=0, max_size=100, size=lambda: 0)]

        class Drafter:
            max_draft_tokens = 2

            def __init__(self, **_kw):
                pass

            def add_prompt_tokens(self, _t):
                pass

            def add_generated_token(self, _t):
                pass

            def record_acceptance(self, _c):
                pass

            def get_draft(self):
                return [5, 5]

        gb = SimpleNamespace(
            _next_tokens=mx.array([7], dtype=mx.int32),
            _next_logprobs=[mx.zeros((8,))],
            uids=[1],
            tokens=[[]],
            logits_processors=[],
            prompt_cache=cache,
            _num_tokens=[0],
            max_tokens=[10],
            state_machines=[SimpleNamespace(match=lambda s, _t: (s, None, None))],
            _matcher_states=[None],
            extract_cache=lambda _r: [],
            model=DriftingModel(),
        )
        gb.Response = SimpleNamespace
        gb._step = lambda: ([7], [])
        gb.next = lambda: []
        bg = MagicMock()
        bg._generation_batch = gb
        bg.remove = lambda *a, **k: {}
        profile = ModelConfig(is_hybrid=True, supports_spec_decode=False)
        # Seed the closure's drafter via monkeypatch BEFORE install, because
        # ``_install_suffix_decoding`` binds SuffixDecodingDrafter into the
        # closure at install time.
        monkeypatch.setattr(suffix_decoding, "SuffixDecodingDrafter", Drafter)
        scheduler._install_suffix_decoding(
            bg,
            model=gb.model,
            profile=profile,
            max_draft=2,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
            suffix_hybrid=True,
            suffix_min_match_len=2,
            suffix_hybrid_bit_exact=True,
        )
        result = gb._step()
        # Drift detected: refuse, fall through (no wrong tokens committed).
        assert bg._suffix_stats["hybrid_drifts"] == 1
        assert bg._suffix_stats["ft_hybrid_drift"] == 1
        assert result[0] == [7]

    def test_guard_catches_drift_beyond_position_four(self, monkeypatch):
        """The bit-exactness guard must probe EVERY committed position.

        The old guard replayed only ``suffix_hybrid_probe_len`` (default 4)
        positions, so a quantized-hybrid drift that only appears AFTER the
        fourth accepted token committed silently under the "bit-exact"
        contract. Here the drift first shows up at position 5: the chunked
        verify accepts a 6-token draft (n_accepted == 6), and stepwise greedy
        matches for positions 0..3 but diverges at position 4. The guard must
        still refuse, proving it probes the full accepted prefix.
        """
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import mlx.core as mx

        from vllm_mlx import scheduler
        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.speculative import suffix_decoding

        class LateDriftModel:
            def __init__(self):
                self._stepwise_calls = 0

            def __call__(self, tokens, cache=None, **kw):
                L = tokens.shape[1]
                logits = mx.full((1, L, 16), -10.0)
                if L > 1:  # chunked verify forward: predict draft token 10
                    for i in range(L):
                        logits[0, i, 10] = 10.0
                    return logits
                # stepwise probe: agree with the chunked pred for the first
                # four committed positions (j = 0..3), then diverge at j = 4
                # (the 5th stepwise call) — exactly the position the OLD
                # probe window (suffix_hybrid_probe_len=4) never inspected.
                self._stepwise_calls += 1
                if self._stepwise_calls >= 5:
                    logits[0, 0, 11] = 10.0  # drift beyond old window
                else:
                    logits[0, 0, 10] = 10.0  # matches chunked
                return logits

        cache = [SimpleNamespace(offset=0, max_size=100, size=lambda: 0)]

        class Drafter:
            max_draft_tokens = 6

            def __init__(self, **_kw):
                pass

            def add_prompt_tokens(self, _t):
                pass

            def add_generated_token(self, _t):
                pass

            def record_acceptance(self, _c):
                pass

            def get_draft(self):
                return [10, 10, 10, 10, 10, 10]

        gb = SimpleNamespace(
            _next_tokens=mx.array([7], dtype=mx.int32),
            _next_logprobs=[mx.zeros((8,))],
            uids=[1],
            tokens=[[]],
            logits_processors=[],
            prompt_cache=cache,
            _num_tokens=[0],
            max_tokens=[10],
            state_machines=[SimpleNamespace(match=lambda s, _t: (s, None, None))],
            _matcher_states=[None],
            extract_cache=lambda _r: [],
            model=LateDriftModel(),
        )
        gb.Response = SimpleNamespace
        gb._step = lambda: ([7], [])
        gb.next = lambda: []
        bg = MagicMock()
        bg._generation_batch = gb
        bg.remove = lambda *a, **k: {}
        profile = ModelConfig(is_hybrid=True, supports_spec_decode=False)
        monkeypatch.setattr(suffix_decoding, "SuffixDecodingDrafter", Drafter)
        scheduler._install_suffix_decoding(
            bg,
            model=gb.model,
            profile=profile,
            max_draft=6,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
            suffix_hybrid=True,
            suffix_min_match_len=6,
            suffix_hybrid_bit_exact=True,
        )
        result = gb._step()
        # Drift at position 4 (beyond the old fixed window) is caught.
        assert bg._suffix_stats["hybrid_drifts"] == 1
        assert bg._suffix_stats["ft_hybrid_drift"] == 1
        assert result[0] == [7]


class TestBitExactDefault:
    """The bit-exactness guard is DEFAULT ON (finding #1: hybrid is only
    lossless when the drift guard is on). Disabling is a non-lossless/unsafe
    mode for eval only."""

    def test_scheduler_config_defaults_guard_on(self):
        assert SchedulerConfig().suffix_hybrid_bit_exact is True

    def test_install_default_arg_guard_on(self, monkeypatch):
        """A hybrid install WITHOUT passing ``suffix_hybrid_bit_exact`` must
        default the bit-exactness guard ON — the drift path is armed by
        default, not the lossless-bypassing fast path.

        Behaviorally: a model whose chunked verify drifts from stepwise must
        be refused (hybrid_drifts == 1) even though the caller never passed
        ``suffix_hybrid_bit_exact=True``. This is exactly
        ``TestBitExactnessGuard.test_guard_refuses_on_drift`` minus the
        explicit True flag."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import mlx.core as mx

        from vllm_mlx import scheduler
        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.speculative import suffix_decoding

        class DriftingModel:
            def __call__(self, tokens, cache=None, **kw):
                L = tokens.shape[1]
                if L > 1:  # chunked verify forward
                    logits = mx.full((1, L, 8), -10.0)
                    for i in range(L):
                        logits[0, i, 5] = 10.0
                    return logits
                logits = mx.full((1, 1, 8), -10.0)
                logits[0, 0, 6] = 10.0  # stepwise differs -> drift
                return logits

        cache = [SimpleNamespace(offset=0, max_size=100, size=lambda: 0)]

        class Drafter:
            max_draft_tokens = 2

            def __init__(self, **_kw):
                pass

            def add_prompt_tokens(self, _t):
                pass

            def add_generated_token(self, _t):
                pass

            def record_acceptance(self, _c):
                pass

            def get_draft(self):
                return [5, 5]

        gb = SimpleNamespace(
            _next_tokens=mx.array([7], dtype=mx.int32),
            _next_logprobs=[mx.zeros((8,))],
            uids=[1],
            tokens=[[]],
            logits_processors=[],
            prompt_cache=cache,
            _num_tokens=[0],
            max_tokens=[10],
            state_machines=[SimpleNamespace(match=lambda s, _t: (s, None, None))],
            _matcher_states=[None],
            extract_cache=lambda _r: [],
            model=DriftingModel(),
        )
        gb.Response = SimpleNamespace
        gb._step = lambda: ([7], [])
        gb.next = lambda: []
        bg = MagicMock()
        bg._generation_batch = gb
        bg.remove = lambda *a, **k: {}
        profile = ModelConfig(is_hybrid=True, supports_spec_decode=False)
        monkeypatch.setattr(suffix_decoding, "SuffixDecodingDrafter", Drafter)
        scheduler._install_suffix_decoding(
            bg,
            model=gb.model,
            profile=profile,
            max_draft=2,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
            suffix_hybrid=True,
            suffix_min_match_len=2,
            # NOTE: suffix_hybrid_bit_exact intentionally omitted -> must be True.
        )
        result = gb._step()
        # Guard armed by default: drift is refused even without the flag.
        assert bg._suffix_stats["hybrid_drifts"] == 1
        assert bg._suffix_stats["ft_hybrid_drift"] == 1
        assert result[0] == [7]

    def test_default_guard_misfire_free_on_step_exact_model(self, monkeypatch):
        """With the guard now default-on, the real step-exact tiny hybrid must
        still accept fully (no spurious drift) — mirrors
        ``TestBitExactnessGuard.test_guard_does_not_misfire_on_step_exact_model``
        but WITHOUT passing ``suffix_hybrid_bit_exact=True``."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from vllm_mlx import scheduler
        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.speculative import suffix_decoding

        model, _ = _model_and_prompt()
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))
        l0 = model(mx.array([[7]]), cache=probe)
        mx.eval(l0)
        d0 = int(mx.argmax(l0[:, -1], axis=-1).item())
        l1 = model(mx.array([[d0]]), cache=probe)
        mx.eval(l1)
        d1 = int(mx.argmax(l1[:, -1], axis=-1).item())
        greedy_draft = [d0, d1]

        drafter = [greedy_draft]

        class MockDrafter:
            max_draft_tokens = 2

            def __init__(self, **_kw):
                pass

            def add_prompt_tokens(self, _t):
                pass

            def add_generated_token(self, _t):
                pass

            def record_acceptance(self, _c):
                pass

            def get_draft(self):
                return list(drafter[0])

        cache = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=cache))
        gb = SimpleNamespace(
            _next_tokens=mx.array([7], dtype=mx.int32),
            _next_logprobs=[mx.zeros((64,))],
            uids=[1],
            tokens=[[]],
            logits_processors=[],
            prompt_cache=cache,
            _num_tokens=[0],
            max_tokens=[10],
            state_machines=[SimpleNamespace(match=lambda s, _t: (s, None, None))],
            _matcher_states=[None],
            extract_cache=lambda _r: [],
            model=model,
        )
        gb.Response = SimpleNamespace
        gb._step = lambda: ([7], [])
        gb.next = lambda: []
        bg = MagicMock()
        bg._generation_batch = gb
        bg.remove = lambda *a, **k: {}
        profile = ModelConfig(is_hybrid=True, supports_spec_decode=False)
        monkeypatch.setattr(suffix_decoding, "SuffixDecodingDrafter", MockDrafter)
        scheduler._install_suffix_decoding(
            bg,
            model=model,
            profile=profile,
            max_draft=2,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
            suffix_hybrid=True,
            suffix_min_match_len=2,
        )
        result = gb._step()
        assert result[0] == [7]
        assert bg._suffix_stats["hybrid_drifts"] == 0
        assert bg._suffix_stats["verify_steps"] == 1


class TestHybridCLI:
    """CLI-level wiring: bit-exact default ON + positive integer validation."""

    @staticmethod
    def _parse(argv):
        import sys
        from unittest import mock

        from vllm_mlx import cli

        captured = {}

        def _capture(args):
            captured["args"] = args

        with (
            mock.patch.object(cli, "serve_command", _capture),
            mock.patch.object(sys, "argv", ["rapid-mlx", *argv]),
        ):
            cli.main()
        assert "args" in captured
        args = captured["args"]
        cli._normalize_speculative_config_or_exit(args)
        return args

    def test_cli_default_guard_on(self):
        args = self._parse(["serve", "qwen3-1.7b", "--suffix-hybrid"])
        assert args.suffix_hybrid_bit_exact is True
        assert args.suffix_min_match_len == 24

    def test_cli_guard_off_via_no_flag(self):
        args = self._parse(
            ["serve", "qwen3-1.7b", "--suffix-hybrid", "--no-suffix-hybrid-bit-exact"]
        )
        assert args.suffix_hybrid_bit_exact is False

    def test_cli_min_match_len_rejects_non_positive_at_parse_time(self):
        import sys
        from unittest import mock

        from vllm_mlx import cli

        # argparse ``type=positive_int`` rejects <=0 BEFORE any engine boot.
        for bad in ("0", "-1", "-5"):
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "rapid-mlx",
                        "serve",
                        "qwen3-1.7b",
                        "--suffix-min-match-len",
                        bad,
                    ],
                ),
                mock.patch.object(cli, "serve_command", lambda args: None),
                pytest.raises(SystemExit) as excinfo,
            ):
                cli.main()
            assert excinfo.value.code == 2

    def test_cli_min_match_len_accepts_positive(self):
        args = self._parse(["serve", "qwen3-1.7b", "--suffix-min-match-len", "16"])
        assert args.suffix_min_match_len == 16


class TestSingleSnapshotCommit:
    """Finding #4: the successful hybrid verify must use ONE pre-verify
    snapshot for probe AND commit (no double deepcopy / redundant replay).

    With the bit-exactness guard ON (the default) and no drift, the commit
    reuses the guard's already-replayed `probe_head` by installing it directly
    onto the live cache — so ``_commit_scratch_accepted`` (which would deep-copy
    the cache AGAIN and re-step every accepted token) must NOT run on that path.
    When the guard is explicitly OFF (non-lossless/debug mode) no probe ran, so
    the stepwise commit helper IS still used.
    """

    @staticmethod
    def _install_and_step(monkeypatch, *, guard_on):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from vllm_mlx import scheduler
        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.speculative import suffix_decoding

        model, _ = _model_and_prompt()
        # Greedy accept-all draft (step-exact model -> no drift).
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))
        l0 = model(mx.array([[7]]), cache=probe)
        mx.eval(l0)
        d0 = int(mx.argmax(l0[:, -1], axis=-1).item())
        l1 = model(mx.array([[d0]]), cache=probe)
        mx.eval(l1)
        d1 = int(mx.argmax(l1[:, -1], axis=-1).item())

        drafter = [[d0, d1]]

        class MockDrafter:
            max_draft_tokens = 2

            def __init__(self, **_kw):
                pass

            def add_prompt_tokens(self, _t):
                pass

            def add_generated_token(self, _t):
                pass

            def record_acceptance(self, _c):
                pass

            def get_draft(self):
                return list(drafter[0])

        cache = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=cache))
        gb = SimpleNamespace(
            _next_tokens=mx.array([7], dtype=mx.int32),
            _next_logprobs=[mx.zeros((64,))],
            uids=[1],
            tokens=[[]],
            logits_processors=[],
            prompt_cache=cache,
            _num_tokens=[0],
            max_tokens=[10],
            state_machines=[SimpleNamespace(match=lambda s, _t: (s, None, None))],
            _matcher_states=[None],
            extract_cache=lambda _r: [],
            model=model,
        )
        gb.Response = SimpleNamespace
        gb._step = lambda: ([7], [])
        gb.next = lambda: []
        bg = MagicMock()
        bg._generation_batch = gb
        bg.remove = lambda *a, **k: {}
        profile = ModelConfig(is_hybrid=True, supports_spec_decode=False)
        monkeypatch.setattr(suffix_decoding, "SuffixDecodingDrafter", MockDrafter)

        calls = {"commit": 0}

        def spy_commit(*a, **k):
            calls["commit"] += 1
            # Keep real behavior so state equality stays meaningful.
            return _commit_scratch_accepted(*a, **k)

        monkeypatch.setattr(scheduler, "_commit_scratch_accepted", spy_commit)
        scheduler._install_suffix_decoding(
            bg,
            model=model,
            profile=profile,
            max_draft=2,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
            suffix_hybrid=True,
            suffix_min_match_len=2,
            suffix_hybrid_bit_exact=guard_on,
        )
        result = gb._step()
        assert result[0] == [7]
        return calls, bg

    def test_commit_reuses_probe_head_on_default_guard(self, monkeypatch):
        calls, bg = self._install_and_step(monkeypatch, guard_on=True)
        # Guard passed (accept-all, no drift): the commit must reuse the
        # guard's probe head NOT call _commit_scratch_accepted again.
        assert bg._suffix_stats["hybrid_drifts"] == 0
        assert calls["commit"] == 0

    def test_commit_uses_step_replay_when_guard_off(self, monkeypatch):
        calls, bg = self._install_and_step(monkeypatch, guard_on=False)
        # Guard off -> no probe -> stepwise commit helper replays once.
        assert calls["commit"] == 1

    def test_post_commit_exception_restores_pristine_cache_then_falls_through(
        self, monkeypatch
    ):
        """Codex BLOCKING (round-9c): an exception raised AFTER the hybrid
        commit head is swapped in must NOT run ``_orig_step`` against the
        already-advanced cache. The hybrid ``except`` in ``_suffix_step``
        restores ``gb.prompt_cache`` to the pristine pre-verify snapshot before
        falling through.

        We inject a POST-commit exception by patching the module-level
        ``_retain_hybrid_replay`` (the in-try step that runs after
        ``result[\"committed\"] = True``) to raise. The commit already swapped
        the probe head onto the live cache, so the except MUST restore the
        pristine snapshot before ``_orig_step`` — and the wrapped ``_orig_step``
        observer records that it saw the pristine cache."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import vllm_mlx.scheduler as scheduler
        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.speculative import suffix_decoding

        model, _ = _model_and_prompt()
        # Greedy accept-all draft (step-exact model -> no drift, n_accepted>0).
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))
        l0 = model(mx.array([[7]]), cache=probe)
        mx.eval(l0)
        d0 = int(mx.argmax(l0[:, -1], axis=-1).item())
        l1 = model(mx.array([[d0]]), cache=probe)
        mx.eval(l1)
        d1 = int(mx.argmax(l1[:, -1], axis=-1).item())

        class MockDrafter:
            max_draft_tokens = 2

            def __init__(self, **_kw):
                pass

            def add_prompt_tokens(self, _t):
                pass

            def add_generated_token(self, _t):
                pass

            def record_acceptance(self, _c):
                pass

            def get_draft(self):
                return [d0, d1]

        pristine = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=pristine))

        gb = SimpleNamespace(
            _next_tokens=mx.array([7], dtype=mx.int32),
            _next_logprobs=[mx.zeros((64,))],
            uids=[1],
            tokens=[[]],
            logits_processors=[],
            prompt_cache=pristine,
            _num_tokens=[0],
            max_tokens=[10],
            state_machines=[SimpleNamespace(match=lambda s, _t: (s, None, None))],
            _matcher_states=[None],
            extract_cache=lambda _r: [],
            model=model,
        )
        gb.Response = SimpleNamespace
        gb._step = lambda: ([7], [])
        gb.next = lambda: []
        bg = MagicMock()
        bg._generation_batch = gb
        bg.remove = lambda *a, **k: {}

        # Wrap the vanilla step to record which cache it observes. Install
        # captures ``gb._step`` as its ``_orig_step``, so we put the observer
        # IN ``gb._step`` (pre-install). This is the crux of the codex finding:
        # after a post-commit exception + restore, the fall-through _orig_step
        # must run against the PRISTINE pre-verify cache, not the advanced
        # (committed probe) head.
        observed = {"cache": None}

        def _orig_step():
            observed["cache"] = list(gb.prompt_cache)
            return ([7], [])

        gb._step = _orig_step

        def _boom_retain(*a, **k):
            raise MemoryError("simulated post-commit OOM in replay retain")

        monkeypatch.setattr(suffix_decoding, "SuffixDecodingDrafter", MockDrafter)
        monkeypatch.setattr(scheduler, "_retain_hybrid_replay", _boom_retain)
        profile = ModelConfig(is_hybrid=True, supports_spec_decode=False)
        scheduler._install_suffix_decoding(
            bg,
            model=model,
            profile=profile,
            max_draft=2,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
            suffix_hybrid=True,
            suffix_min_match_len=2,
            suffix_hybrid_bit_exact=True,
        )
        # The post-commit exception is absorbed by the hybrid except -> clean
        # fall-through, and the wrapped _orig_step observed the PRISTINE cache.
        gb._step()
        _assert_state_equal(observed["cache"], pristine)

    def test_restore_hybrid_cache_helper(self):
        """Direct unit test of ``_restore_hybrid_cache_after_exception``: a
        committed result (``committed`` + ``replay_snapshot``) restores the
        cache; an empty / pre-commit result is a no-op."""
        from types import SimpleNamespace

        import vllm_mlx.scheduler as scheduler

        gb = SimpleNamespace(prompt_cache="committed_head")
        # (a) post-commit exception -> restore to pristine snapshot.
        result = {"committed": True, "replay_snapshot": "pristine_snapshot"}
        scheduler._restore_hybrid_cache_after_exception(gb, result)
        assert gb.prompt_cache == "pristine_snapshot"
        # (b) pre-commit exception (empty result) -> no restore.
        gb2 = SimpleNamespace(prompt_cache="live")
        scheduler._restore_hybrid_cache_after_exception(gb2, {})
        assert gb2.prompt_cache == "live"
        # (c) committed but no snapshot (shouldn't happen) -> no restore.
        gb3 = SimpleNamespace(prompt_cache="live2")
        scheduler._restore_hybrid_cache_after_exception(
            gb3, {"committed": True, "replay_snapshot": None}
        )
        assert gb3.prompt_cache == "live2"

    def test_post_commit_phase_exception_restores_cache(self, monkeypatch):
        """Codex round-9e finding #3: a failure AFTER the hybrid commit but in
        the shared post-commit preparation phase (cooldown bookkeeping / logprob
        construction / pending-emission creation) must restore the pristine
        pre-verify cache, never leaving ``gb.prompt_cache`` advanced beyond the
        surfaced tokens.

        We inject the failure in the logprob construction (``mx.logsumexp``),
        which runs inside the post-commit envelope AFTER the commit head was
        swapped in, and assert the re-raised exception propagates to the wrapper
        while the live cache has been restored to the pristine snapshot."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        import vllm_mlx.scheduler as scheduler
        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.speculative import suffix_decoding

        model, _ = _model_and_prompt()
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))
        l0 = model(mx.array([[7]]), cache=probe)
        mx.eval(l0)
        d0 = int(mx.argmax(l0[:, -1], axis=-1).item())
        l1 = model(mx.array([[d0]]), cache=probe)
        mx.eval(l1)
        d1 = int(mx.argmax(l1[:, -1], axis=-1).item())

        class MockDrafter:
            max_draft_tokens = 2

            def __init__(self, **_kw):
                pass

            def add_prompt_tokens(self, _t):
                pass

            def add_generated_token(self, _t):
                pass

            def record_acceptance(self, _c):
                pass

            def get_draft(self):
                return [d0, d1]

        pristine = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=pristine))

        gb = SimpleNamespace(
            _next_tokens=mx.array([7], dtype=mx.int32),
            _next_logprobs=[mx.zeros((64,))],
            uids=[1],
            tokens=[[]],
            logits_processors=[],
            prompt_cache=pristine,
            _num_tokens=[0],
            max_tokens=[10],
            state_machines=[SimpleNamespace(match=lambda s, _t: (s, None, None))],
            _matcher_states=[None],
            extract_cache=lambda _r: [],
            model=model,
        )
        gb.Response = SimpleNamespace
        gb._step = lambda: ([7], [])
        gb.next = lambda: []
        bg = MagicMock()
        bg._generation_batch = gb
        bg.remove = lambda *a, **k: {}
        monkeypatch.setattr(suffix_decoding, "SuffixDecodingDrafter", MockDrafter)
        profile = ModelConfig(is_hybrid=True, supports_spec_decode=False)
        scheduler._install_suffix_decoding(
            bg,
            model=model,
            profile=profile,
            max_draft=2,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
            suffix_hybrid=True,
            suffix_min_match_len=2,
            suffix_hybrid_bit_exact=True,
        )
        # Fail inside the post-commit logprob construction (runs after the
        # commit head swap). The wrapped _step must re-raise AND restore the
        # pristine cache.
        with (
            patch.object(scheduler.mx, "logsumexp", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            gb._step()
        _assert_state_equal(gb.prompt_cache, pristine)

    def test_zero_accept_hybrid_commit_still_restores_pristine_on_post_commit_error(
        self, monkeypatch
    ):
        """Codex round-9f finding: ``_hybrid_pc_snapshot`` used to be captured
        only when ``n_accepted > 0``. But even a ZERO-accept hybrid commit swaps
        the commit head in (advancing the live cache through the primary X),
        so a post-commit exception with no accepted drafts left ``gb.prompt_cache``
        advanced with no snapshot to restore. Fix captures the pristine snapshot
        unconditionally after every hybrid commit; the replay HEAD stays gated on
        ``n_accepted > 0`` (no synthetic emits to drain when nothing was accepted).

        We force a mismatch draft (draft[0] != pred for X) so ``n_accepted == 0``,
        then fail inside the shared post-commit logprob construction and assert
        the re-raised exception restores the pristine cache.
        """
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        import vllm_mlx.scheduler as scheduler
        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.speculative import suffix_decoding

        model, _ = _model_and_prompt()
        # Probe what X predicts so we can build a DELIBERATELY WRONG draft that
        # guarantees zero acceptance.
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))
        l0 = model(mx.array([[7]]), cache=probe)
        mx.eval(l0)
        x_pred = int(mx.argmax(l0[:, -1], axis=-1).item())
        vocab_size = _ple_args().vocab_size or 4**6
        wrong_draft = [(x_pred + 1) % vocab_size, x_pred + 2]

        class MockDrafter:
            max_draft_tokens = 2

            def __init__(self, **_kw):
                pass

            def add_prompt_tokens(self, _t):
                pass

            def add_generated_token(self, _t):
                pass

            def record_acceptance(self, _c):
                pass

            def get_draft(self):
                return list(wrong_draft)

        pristine = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=pristine))

        gb = SimpleNamespace(
            _next_tokens=mx.array([7], dtype=mx.int32),
            _next_logprobs=[mx.zeros((64,))],
            uids=[1],
            tokens=[[]],
            logits_processors=[],
            prompt_cache=pristine,
            _num_tokens=[0],
            max_tokens=[10],
            state_machines=[SimpleNamespace(match=lambda s, _t: (s, None, None))],
            _matcher_states=[None],
            extract_cache=lambda _r: [],
            model=model,
        )
        gb.Response = SimpleNamespace
        gb._step = lambda: ([7], [])
        gb.next = lambda: []
        bg = MagicMock()
        bg._generation_batch = gb
        bg.remove = lambda *a, **k: {}
        monkeypatch.setattr(suffix_decoding, "SuffixDecodingDrafter", MockDrafter)
        profile = ModelConfig(is_hybrid=True, supports_spec_decode=False)
        scheduler._install_suffix_decoding(
            bg,
            model=model,
            profile=profile,
            max_draft=2,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
            suffix_hybrid=True,
            suffix_min_match_len=2,
            suffix_hybrid_bit_exact=True,
        )
        # Fail after the zero-accept commit head is swapped in (logprob
        # construction). Must re-raise AND restore the pristine cache — the
        # regression this guards is the live cache staying advanced through X.
        with (
            patch.object(scheduler.mx, "logsumexp", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            gb._step()
        _assert_state_equal(gb.prompt_cache, pristine)

    def test_post_commit_failure_deferres_mutation_tail(self, monkeypatch):
        """Codex round-9g finding: the post-commit phase must not leave ANY
        bookkeeping half-mutated when a fallible MLX op raises. The fix does
        this by reordering so all fallible compute (logprob construction) runs
        BEFORE the mutation tail (cooldown/adaptive width set_state, drafter,
        counter, gb fields, emit stash). We prove the deferral by spying on the
        counter's ``set_state`` — it is only invoked in the mutation tail, so
        if a ``logsumexp`` raise happens first, ``set_state`` must never run."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        import vllm_mlx.scheduler as scheduler
        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.speculative import suffix_counter, suffix_decoding

        class SpyCounter(suffix_counter.SuffixAcceptCounter):
            def __init__(self):
                super().__init__()
                self.set_state_calls = 0

            def set_state(self, current_k, backoff_level):
                self.set_state_calls += 1
                return super().set_state(current_k, backoff_level)

        spy = SpyCounter()
        monkeypatch.setattr(suffix_counter, "get_global_counter", lambda: spy)

        model, _ = _model_and_prompt()
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))
        l0 = model(mx.array([[7]]), cache=probe)
        mx.eval(l0)
        d0 = int(mx.argmax(l0[:, -1], axis=-1).item())
        d1 = d0  # accept-all draft so the step commits

        class MockDrafter:
            max_draft_tokens = 2

            def __init__(self, **_kw):
                pass

            def add_prompt_tokens(self, _t):
                pass

            def add_generated_token(self, _t):
                pass

            def record_acceptance(self, _c):
                pass

            def get_draft(self):
                return [d0, d1]

        pristine = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=pristine))

        gb = SimpleNamespace(
            _next_tokens=mx.array([7], dtype=mx.int32),
            _next_logprobs=[mx.zeros((64,))],
            uids=[1],
            tokens=[[]],
            logits_processors=[],
            prompt_cache=pristine,
            _num_tokens=[0],
            max_tokens=[10],
            state_machines=[SimpleNamespace(match=lambda s, _t: (s, None, None))],
            _matcher_states=[None],
            extract_cache=lambda _r: [],
            model=model,
        )
        gb.Response = SimpleNamespace
        gb._step = lambda: ([7], [])
        gb.next = lambda: []
        bg = MagicMock()
        bg._generation_batch = gb
        bg.remove = lambda *a, **k: {}
        monkeypatch.setattr(suffix_decoding, "SuffixDecodingDrafter", MockDrafter)
        profile = ModelConfig(is_hybrid=True, supports_spec_decode=False)
        scheduler._install_suffix_decoding(
            bg,
            model=model,
            profile=profile,
            max_draft=2,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
            suffix_hybrid=True,
            suffix_min_match_len=2,
            suffix_hybrid_bit_exact=True,
        )
        # Seed the per-uid state so a later step does not trigger ``_state_for``
        # lazy-init (which publishes the at-rest gauge via ``set_state`` — that
        # is pre-verify and not part of the post-commit tail). Run one full
        # successful hybrid step to populate ``_uid_state`` for uid 1. It also
        # ratifies the wiring (verify commits + emits are stashed).
        gb._step()
        # After the warm-up step the live cache has advanced past the primary;
        # capture a FRESH pristine snapshot that the failing step must restore.
        import copy as _copy

        pre_fail = _copy.deepcopy(gb.prompt_cache)
        # Now the uid state exists, so ``_state_for`` returns without calling
        # ``set_state``. Any ``set_state`` from here on can only come from the
        # post-commit mutation tail.
        spy.set_state_calls = 0
        # Fail in the fallible logprob compute (which now runs FIRST, before
        # the mutation tail). Because the mutation tail never runs, the counter
        # ``set_state`` is never called: no cooldown/width state was mutated on
        # the way to the exception, and the pristine cache is restored. The step
        # falls through and the caller re-derives state on the re-run.
        with (
            patch.object(scheduler.mx, "logsumexp", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            gb._step()
        assert spy.set_state_calls == 0
        _assert_state_equal(gb.prompt_cache, pre_fail)

    def test_post_commit_exception_drops_retained_replay_entry(self, monkeypatch):
        """Codex round-9h finding #1: a post-commit exception must pop the
        uid's retained ``_pending_hybrid_replay`` entry. If it leaked, a later
        terminal response for that uid would rebuild its delivered cache from a
        NOW-STALE pristine snapshot (the cache was restored to that snapshot and
        re-advanced by the fall-through re-run) and the full duplicate snapshot
        would leak until UID cleanup.

        We prove the entry is dropped by reaching the terminal branch of the
        wrapped ``_suffix_next`` after a post-commit exception: if the entry
        were still present, the terminal would be repaired (delivered cache =
        pristine + X); because the fix pops it, the terminal passes through
        with the live (restored) cache untouched."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        import vllm_mlx.scheduler as scheduler
        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.speculative import suffix_decoding

        model, _ = _model_and_prompt()
        # Accepted draft ([d0] matches greedy), so the hybrid commit retains a
        # replay entry — the leak codex round-9h finding #1 guards against.
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))
        l0 = model(mx.array([[7]]), cache=probe)
        mx.eval(l0)
        d0 = int(mx.argmax(l0[:, -1], axis=-1).item())
        l1 = model(mx.array([[d0]]), cache=probe)
        mx.eval(l1)
        d1 = int(mx.argmax(l1[:, -1], axis=-1).item())

        class MockDrafter:
            max_draft_tokens = 2

            def __init__(self, **_kw):
                pass

            def add_prompt_tokens(self, _t):
                pass

            def add_generated_token(self, _t):
                pass

            def record_acceptance(self, _c):
                pass

            def get_draft(self):
                return [d0, d1]

        pristine = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=pristine))

        gb = SimpleNamespace(
            _next_tokens=mx.array([7], dtype=mx.int32),
            _next_logprobs=[mx.zeros((64,))],
            uids=[1],
            tokens=[[]],
            logits_processors=[],
            prompt_cache=pristine,
            _num_tokens=[0],
            max_tokens=[10],
            state_machines=[SimpleNamespace(match=lambda s, _t: (s, None, None))],
            _matcher_states=[None],
            extract_cache=lambda _r: [],
            model=model,
        )
        gb.Response = SimpleNamespace
        gb._step = lambda: ([7], [])
        gb.next = lambda: [SimpleNamespace(uid=1, finish_reason="length")]
        bg = MagicMock()
        bg._generation_batch = gb
        bg.remove = lambda *a, **k: {}
        monkeypatch.setattr(suffix_decoding, "SuffixDecodingDrafter", MockDrafter)
        profile = ModelConfig(is_hybrid=True, supports_spec_decode=False)
        scheduler._install_suffix_decoding(
            bg,
            model=model,
            profile=profile,
            max_draft=2,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
            suffix_hybrid=True,
            suffix_min_match_len=2,
            suffix_hybrid_bit_exact=True,
        )
        # Force a post-commit exception on the hybrid commit. The handler must
        # (a) restore the pristine cache AND (b) drop the retained replay entry
        # so the subsequent terminal response is NOT rebuilt from a stale
        # snapshot.
        with (
            patch.object(scheduler.mx, "logsumexp", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            gb._step()
        _assert_state_equal(gb.prompt_cache, pristine)
        # A terminal (length) primary now flows through _suffix_next. With the
        # replay entry correctly dropped, the passed-through response is left
        # UNTOUCHED by the terminal repair — i.e. no ``prompt_cache`` attr was
        # set on it (the repair is what would set it). Had the stale entry
        # leaked, the repair would have assigned pristine + X here, rebuilding
        # from a now-obsolete snapshot.
        outs = gb.next()
        terminals = [o for o in outs if o.finish_reason is not None]
        assert len(terminals) == 1
        assert not hasattr(terminals[0], "prompt_cache")
        _assert_state_equal(gb.prompt_cache, pristine)


class TestHybridTerminalRepair:
    """Finding #1: a hybrid finish part-way through the accepted drafts must
    drop the un-surfaced accepted tail from the delivered response cache.

    On the pure-attention path the terminal branch trims un-surfaced accepted
    drafts via ``trim_all``. The Qwen4 recurrent cache cannot trim a rejected
    tail (the whole reason B2.1 uses scratch+replay instead of trim_all), so a
    hybrid terminal must instead rebuild the response cache from the retained
    pristine pre-verify snapshot plus ONLY the surfaced tokens — dropping the
    tail by never replaying it. This mirrors the DSpark ``_pending_replay``
    terminal-repair pattern and prevents a poisoned (cache-ahead-of-tokens)
    saved prefix from poisoning prefix-cache reuse for the next request.
    """

    def test_hybrid_terminal_drops_unsurfaced_accepted_tail(self, monkeypatch):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from vllm_mlx import scheduler
        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.speculative import suffix_decoding

        model, _ = _model_and_prompt()
        # Greedy accept-all draft of 2 tokens (step-exact -> no drift).
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))
        l0 = model(mx.array([[7]]), cache=probe)
        mx.eval(l0)
        d0 = int(mx.argmax(l0[:, -1], axis=-1).item())
        l1 = model(mx.array([[d0]]), cache=probe)
        mx.eval(l1)
        d1 = int(mx.argmax(l1[:, -1], axis=-1).item())

        drafter = [[d0, d1]]

        class MockDrafter:
            max_draft_tokens = 2

            def __init__(self, **_kw):
                pass

            def add_prompt_tokens(self, _t):
                pass

            def add_generated_token(self, _t):
                pass

            def record_acceptance(self, _c):
                pass

            def get_draft(self):
                return list(drafter[0])

        cache = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=cache))
        # max_tokens=1 forces a length-finish after the FIRST synthetic draft
        # (X), leaving the second accepted draft (d1) un-surfaced.
        gb = SimpleNamespace(
            _next_tokens=mx.array([7], dtype=mx.int32),
            _next_logprobs=[mx.zeros((64,))],
            uids=[1],
            tokens=[[]],
            logits_processors=[],
            prompt_cache=cache,
            _num_tokens=[0],
            max_tokens=[1],
            state_machines=[SimpleNamespace(match=lambda s, _t: (s, None, None))],
            _matcher_states=[None],
            extract_cache=lambda _r: gb.prompt_cache,
            # Faithful-ish filter: production ``GenerationBatch.filter`` compacts
            # the live cache list in place and drops finished rows. We clear
            # the cache list to simulate the finishing row being compacted away,
            # then assert the delivered response cache is still intact+decodable.
            filter=lambda keep: (
                setattr(gb, "uids", list(keep)),
                gb.prompt_cache.clear() if not keep else None,
            ),
            model=model,
        )
        gb.Response = SimpleNamespace
        # Baseline ``_step``/``next`` placeholders (install wraps them).
        gb._step = lambda: ([7], [])
        # ``_orig_next`` returns the primary response (X, not yet finished);
        # the terminal fires later while draining the synthetic emits.
        gb.next = lambda: [SimpleNamespace(uid=1, finish_reason=None)]
        bg = MagicMock()
        bg._generation_batch = gb
        bg.remove = lambda *a, **k: {}
        profile = ModelConfig(is_hybrid=True, supports_spec_decode=False)
        monkeypatch.setattr(suffix_decoding, "SuffixDecodingDrafter", MockDrafter)
        scheduler._install_suffix_decoding(
            bg,
            model=model,
            profile=profile,
            max_draft=2,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
            suffix_hybrid=True,
            suffix_min_match_len=2,
            suffix_hybrid_bit_exact=True,
        )

        gb._step()  # hybrid verify: X + d0 accepted, d1 accepted too
        # The verify committed prompt+[7,d0,d1]; pending emits = [d0, d1].
        # A length-terminal fires when emitting d0 (emit_idx 0 -> unused 1).
        outs = gb.next()
        # The finishing response's cache must equal prompt + [7, d0] only —
        # d1 was never surfaced and must be dropped.
        finish = [o for o in outs if o.finish_reason is not None]
        assert len(finish) == 1
        delivered = finish[0].prompt_cache
        # ``delivered`` is the repaired cache list (per-layer cells) built by
        # the terminal-repair branch. Even after the faithful filter CLEARED
        # the shared live cache list (simulating the finishing row being
        # compacted away), the delivered response cache must remain intact,
        # independent, and decodable. Build the step-replayed gold cache list.
        gold = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=gold))
        mx.eval(model(mx.array([[7]]), cache=gold))
        mx.eval(model(mx.array([[d0]]), cache=gold))
        _assert_state_equal(delivered, gold)
        # Independent/decodable: continuing from the delivered cache produces
        # the same next token as continuing from the gold cache.
        nxt_raw = model(mx.array([[8]]), cache=delivered)
        nxt_gold = model(mx.array([[8]]), cache=gold)
        mx.eval(nxt_raw, nxt_gold)
        assert int(mx.argmax(nxt_raw[:, -1], axis=-1).item()) == int(
            mx.argmax(nxt_gold[:, -1], axis=-1).item()
        )

    def test_primary_terminal_drops_every_queued_accepted_draft(self, monkeypatch):
        """Codex round-9e finding #2: the PRIMARY can be terminal (a length
        finish on the primary token X itself), leaving every queued accepted
        draft un-surfaced. The delivered response cache must then hold ONLY
        pristine + X (the Finding-1 repair), dropping all accepted drafts.

        Model the real generation path: the primary _step advances
        ``_num_tokens`` to 1, so the synthetic emits are never reached (the
        primary's length limit stops the request). The returning primary
        response is terminal, and ``_suffix_next`` must repair its cache to
        pristine + X via the primary-terminal repair (not the synthetic-drain
        loop, which never runs)."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from vllm_mlx import scheduler
        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.speculative import suffix_decoding

        model, _ = _model_and_prompt()
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))
        l0 = model(mx.array([[7]]), cache=probe)
        mx.eval(l0)
        d0 = int(mx.argmax(l0[:, -1], axis=-1).item())
        l1 = model(mx.array([[d0]]), cache=probe)
        mx.eval(l1)
        d1 = int(mx.argmax(l1[:, -1], axis=-1).item())

        class MockDrafter:
            max_draft_tokens = 2

            def __init__(self, **_kw):
                pass

            def add_prompt_tokens(self, _t):
                pass

            def add_generated_token(self, _t):
                pass

            def record_acceptance(self, _c):
                pass

            def get_draft(self):
                return [d0, d1]

        pristine = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=pristine))

        gb = SimpleNamespace(
            _next_tokens=mx.array([7], dtype=mx.int32),
            _next_logprobs=[mx.zeros((64,))],
            uids=[1],
            tokens=[[]],
            logits_processors=[],
            prompt_cache=pristine,
            _num_tokens=[0],
            max_tokens=[1],  # length-limit: the PRIMARY X saturates it
            state_machines=[SimpleNamespace(match=lambda s, _t: (s, None, None))],
            _matcher_states=[None],
            extract_cache=lambda _r: [],
            model=model,
        )
        gb.Response = SimpleNamespace
        # The PRIMARY _step commits X (increments _num_tokens to 1) and
        # stashes [d0, d1] as pending emits. The primary is terminal (length).
        gb._step = lambda: ([7], [])
        # The returned primary is the terminal length response.
        gb.next = lambda: [SimpleNamespace(uid=1, finish_reason="length")]
        bg = MagicMock()
        bg._generation_batch = gb
        bg.remove = lambda *a, **k: {}
        profile = ModelConfig(is_hybrid=True, supports_spec_decode=False)
        monkeypatch.setattr(suffix_decoding, "SuffixDecodingDrafter", MockDrafter)
        scheduler._install_suffix_decoding(
            bg,
            model=model,
            profile=profile,
            max_draft=2,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
            suffix_hybrid=True,
            suffix_min_match_len=2,
            suffix_hybrid_bit_exact=True,
        )

        # Commit the hybrid verify: X + d0 + d1 all accepted, [d0, d1] queued.
        gb._step()
        # The next() wrapper returns the terminal primary; Finding-1 repair
        # must rebuild its cache to pristine + X ONLY (d0, d1 dropped because
        # they were never surfaced).
        outs = gb.next()
        terminals = [o for o in outs if o.finish_reason is not None]
        assert len(terminals) == 1
        delivered = terminals[0].prompt_cache
        # Gold = pristine + X (no accepted drafts surfaced).
        gold = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=gold))
        mx.eval(model(mx.array([[7]]), cache=gold))
        _assert_state_equal(delivered, gold)
