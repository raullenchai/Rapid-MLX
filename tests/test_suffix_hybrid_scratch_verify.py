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
    _commit_scratch_accepted,
    _verify_scratch,
)


def _offsets(cache):
    return [
        None
        if not isinstance(lc, CacheList)
        else (int(lc[0].offset), int(lc[1].offset))
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
        # gold: commit [4,5] stepwise
        gold = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=gold))
        for t in ([4], [5]):
            mx.eval(model(mx.array([t]), cache=gold))

        draft = [5, 6]
        commit_head = copy.deepcopy(cache)
        result = _verify_scratch(
            model, cache, mx.array([[4, 5, 6]]), draft, commit_head
        )
        # Accept-all only if both draft tokens match greedy. Probe first.
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))
        vlog = model(mx.array([[4, 5, 6]]), cache=probe)
        mx.eval(vlog)
        preds = mx.argmax(vlog, axis=-1).tolist()[0]
        if preds[0] == 5 and preds[1] == 6:
            assert result["n_accepted"] == 2
        # Commit accepted (whatever it is) and compare to the stepwise gold
        # of THAT same accepted prefix: gold = prompt + committed [4,5,..].
        na = result["n_accepted"]
        _commit_scratch_accepted(model, cache, mx.array([[4, 5, 6]]), na)
        g2 = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=g2))
        for t in ([4], [5], [6])[: na + 1]:
            mx.eval(model(mx.array([t]), cache=g2))
        _assert_state_equal(cache, g2)

    def test_partial_reject_commits_only_accepted(self):
        import copy

        model, _cache = _model_and_prompt()
        # Probe the verify predictions to choose a draft whose token 0 is
        # accepted and token 1 is rejected.
        probe = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=probe))
        vlog = model(mx.array([[4, 5, 6]]), cache=probe)
        mx.eval(vlog)
        preds = mx.argmax(vlog, axis=-1).tolist()[0]
        d0 = preds[0]  # accepted at position 0
        d1 = (preds[1] + 1) % 100  # rejected at position 1

        live = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=live))
        commit_head = copy.deepcopy(live)
        verify_input = mx.array([[4, d0, d1]])
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
        d0 = preds[0] + 1  # guaranteed rejected at position 0
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
        assert gb._step is not None  # old step preserved

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
        d0 = int(mx.argmax(l0[:, -1], axis=-1).item()) + 1  # rejected
        d1 = d0 + 1
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
        # Live cache must equal prompt + [7] (X committed despite reject).
        gold = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=gold))
        mx.eval(model(mx.array([[7]]), cache=gold))
        _assert_state_equal(cache, gold)

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
        d0 = self._stepwise_next(model, probe, 7) + 1
        verify_input = mx.array([[7, d0, d0 + 1]])
        draft = [d0, d0 + 1]
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
        verify_input = mx.array([[7, d0, d0 + 1]])
        draft = [d0, d0 + 1]
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
