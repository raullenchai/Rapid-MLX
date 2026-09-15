"""Media-aware boundary store/lookup contract tests.

Pins the first media-prefix milestone on the serialized MLLM lane:

* the rollback flag is validated at every config layer;
* the boundary plan fails closed (flag off, text-only, no boundary,
  unqualified leaves, no structural rope-deltas consumption, extra
  kwargs, partial mask, unverifiable placeholder position) into the
  cold single forward;
* identity is content-keyed even when the vision-feature-cache key is
  absent — two turns with the same token prefix but different image
  bytes never share an entry;
* the store path splits the prefill at the derived boundary, snapshots
  a detached clone with the recorded MRoPE delta, and keeps strict
  ``token_ids`` for next-turn verification;
* the resume path re-clones the *stored* leaves into the request cache
  (state equality asserted, not just object identity), installs the
  delta, and forwards only the strict suffix with ``pixel_values=None``;
* a prefix mismatch is a clean counted miss — never a trim or a guess;
* the MRoPE transaction restores prior model state (sentinel-aware,
  verified against a real ``mlx.nn.Module``);
* the byte-bounded LRU evicts oldest-first and always keeps the newest.

Design note: docs/engineering/design/2026-09-15-mllm-media-prefix-cache.md.
"""

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from vllm_mlx.mllm_batch_generator import (  # noqa: E402
    MLLMBatchGenerator,
    MLLMBatchRequest,
    MLLMBatchStats,
    _media_clone_leaves,
)
from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig  # noqa: E402
from vllm_mlx.scheduler import SchedulerConfig  # noqa: E402

VOCAB = 8
# Stands in for the model's processor-expanded image placeholder token.
_PLACEHOLDER_ID = 99


def _make_request(uid: int = 0, **overrides) -> MLLMBatchRequest:
    fields = {
        "uid": uid,
        "request_id": f"r{uid}",
        "prompt": "user turn",
        "images": ["img.png"],
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
        "prefix_boundary": 10,
        "vision_feature_key": "img-hash-1",
    }
    fields.update(overrides)
    return MLLMBatchRequest(**fields)


def _full_ids(n: int = 30, placeholder_at: int = 10) -> list[int]:
    """An expanded sequence with one placeholder well inside the boundary."""
    ids = list(range(n))
    ids[placeholder_at] = _PLACEHOLDER_ID
    return ids


class _BareGenerator:
    """Attribute bag for unbound ``__init__`` validation tests."""


class _FakeTokenizer:
    def encode(self, text):
        # Deterministic: one token per character (bounded by tests).
        return list(range(len(text)))


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()


class _RecordingModel:
    """Records forwards; consumes ``rope_deltas`` through ``**kwargs``.

    Real mlx-vlm models pop ``rope_deltas`` from ``**kwargs`` inside the
    call body — the kwarg is not an explicit signature parameter — and the
    production gate detects exactly that shape structurally.
    """

    def __init__(self, vocab: int = VOCAB):
        self.vocab = vocab
        self.calls: list[tuple[int, int, bool]] = []  # (start, end, has_pixels)
        self.config = type("Config", (), {"image_token_id": _PLACEHOLDER_ID})()

    def __call__(self, ids, cache=None, pixel_values=None, **kwargs):
        rope_deltas = kwargs.pop("rope_deltas", None)
        start = int(ids[0, 0]) if ids.size else -1
        self.calls.append((start, start + int(ids.shape[1]), pixel_values is not None))
        return _Output(mx.zeros((1, int(ids.shape[1]), self.vocab)))


class _NoRopeModel:
    """A wrapper whose call neither accepts ``**kwargs`` nor references
    ``"rope_deltas"`` — the structural gate must never open for it."""

    def __init__(self, vocab: int = VOCAB):
        self.vocab = vocab
        self.calls: list[tuple[int, int, bool]] = []

    def __call__(self, ids, cache=None, pixel_values=None, rope_deltas=None):
        start = int(ids[0, 0]) if ids.size else -1
        self.calls.append((start, start + int(ids.shape[1]), pixel_values is not None))
        return _Output(mx.zeros((1, int(ids.shape[1]), self.vocab)))


class _Output:
    def __init__(self, logits):
        self.logits = logits


class _FakeLanguageModel:
    def __init__(self):
        self._position_ids = None
        self._rope_deltas = None


class _RealLanguageModel(nn.Module):
    """A real ``mlx.nn.Module`` — plain array attributes live outside
    ``__dict__`` (verified: ``__dict__.pop`` is a silent no-op for them)."""

    def __init__(self):
        super().__init__()
        self.w = mx.zeros((2,))


def _kv_leaves(n_layers: int = 2, rows: int = 1, slen: int = 4):
    from mlx_vlm.models.cache import KVCache

    leaves = []
    for _ in range(n_layers):
        leaf = KVCache()
        leaf.keys = mx.ones((rows, 2, slen, 4))
        leaf.values = mx.ones((rows, 2, slen, 4)) * 2
        leaf.offset = slen
        leaves.append(leaf)
    return leaves


def _stub_generator(model=None, *, media_prefix_cache: str = "auto"):
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._stats = MLLMBatchStats()
    gen._stream = mx.default_stream(mx.cpu)
    gen.allow_arrays_cache = True
    gen.singleton_fastpath = "auto"
    gen.media_prefix_cache = media_prefix_cache
    gen._media_boundary_entries = {}
    gen._media_boundary_hits = 0
    gen._media_boundary_misses = 0
    gen._media_boundary_stores = 0
    gen._media_boundary_budget_evictions = 0
    gen._media_boundary_max_bytes = 0
    gen._media_mrope_saved = None
    gen.model = model if model is not None else _RecordingModel()
    gen.language_model = _FakeLanguageModel()
    gen.processor = _FakeProcessor()
    gen.vision_prefill_token_budget = 8192
    return gen


def _ids(values: list[int]):
    return mx.array([values], dtype=mx.uint32)


class TestConfigValidation:
    def test_scheduler_config_rejects_unknown_values(self):
        with pytest.raises(ValueError, match="mllm_media_prefix_cache"):
            SchedulerConfig(mllm_media_prefix_cache="on")

    def test_scheduler_config_accepts_off(self):
        assert (
            SchedulerConfig(mllm_media_prefix_cache="off").mllm_media_prefix_cache
            == "off"
        )

    def test_mllm_config_rejects_unknown_values(self):
        with pytest.raises(ValueError, match="mllm_media_prefix_cache"):
            MLLMSchedulerConfig(mllm_media_prefix_cache="on")

    def test_generator_rejects_unknown_values(self):
        # The media flag is validated first in ``__init__``, so an unbound
        # call against a bare namespace raises before touching the model.
        with pytest.raises(ValueError, match="media_prefix_cache"):
            MLLMBatchGenerator.__init__(
                _BareGenerator(), model=None, processor=None, media_prefix_cache="on"
            )


class TestRopeKwargGate:
    def test_kwargs_popped_rope_deltas_opens_the_gate(self):
        gen = _stub_generator()
        assert gen._media_model_supports_rope_kwarg() is True

    def test_explicit_param_without_kwargs_fails_closed(self):
        gen = _stub_generator(model=_NoRopeModel())
        assert gen._media_model_supports_rope_kwarg() is False

    def test_plan_fails_closed_without_structural_support(self):
        gen = _stub_generator(model=_NoRopeModel())
        req = _make_request(pixel_values=mx.zeros((1, 2)))
        assert gen._media_boundary_plan(req, _ids([1, 2, 3]), _kv_leaves()) is None
        # Gate fires before planning: no miss is counted.
        assert gen._media_boundary_misses == 0


class TestPlanGates:
    def test_off_flag_is_cold(self):
        gen = _stub_generator(media_prefix_cache="off")
        req = _make_request(pixel_values=mx.zeros((1, 2)))
        assert gen._media_boundary_plan(req, _ids([1, 2, 3]), _kv_leaves()) is None

    def test_text_only_request_never_plans(self):
        gen = _stub_generator()
        req = _make_request(pixel_values=None)
        assert gen._media_boundary_plan(req, _ids([1, 2, 3]), _kv_leaves()) is None

    def test_zero_boundary_never_plans(self):
        gen = _stub_generator()
        req = _make_request(pixel_values=mx.zeros((1, 2)), prefix_boundary=0)
        assert gen._media_boundary_plan(req, _ids([1, 2, 3]), _kv_leaves()) is None

    def test_unqualified_leaves_fail_closed(self):
        gen = _stub_generator()

        class NotACache:
            pass

        req = _make_request(pixel_values=mx.zeros((1, 2)))
        assert gen._media_boundary_plan(req, _ids([1, 2, 3]), [NotACache()]) is None

    def test_nonempty_extra_kwargs_fail_closed(self):
        gen = _stub_generator()
        req = _make_request(
            pixel_values=mx.zeros((1, 2)),
            extra_kwargs={"image_grid_thw": mx.zeros((1, 3))},
        )
        assert gen._media_boundary_plan(req, _ids(_full_ids()), _kv_leaves()) is None
        assert gen._media_boundary_misses == 1

    def test_partial_attention_mask_fails_closed(self):
        gen = _stub_generator()
        mask = mx.ones((1, 30))
        mask[0, 3] = 0
        req = _make_request(pixel_values=mx.zeros((1, 2)), attention_mask=mask)
        assert gen._media_boundary_plan(req, _ids(_full_ids()), _kv_leaves()) is None
        assert gen._media_boundary_misses == 1

    def test_placeholder_after_boundary_fails_closed(self):
        gen = _stub_generator()
        full_ids = _full_ids()
        full_ids[27] = _PLACEHOLDER_ID  # derived boundary is 26
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,  # marker width 4 → boundary 26
        )
        assert gen._media_boundary_plan(req, _ids(full_ids), _kv_leaves()) is None
        assert gen._media_boundary_misses == 1

    def test_missing_placeholder_config_fails_closed(self):
        gen = _stub_generator()
        gen.model.config = None
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
        )
        assert gen._media_boundary_plan(req, _ids(_full_ids()), _kv_leaves()) is None
        assert gen._media_boundary_misses == 1


class TestMediaIdentity:
    def test_digest_falls_back_to_image_content_hash(self):
        gen = _stub_generator()
        same_a = _make_request(vision_feature_key=None, images=["shot-a.png"])
        same_b = _make_request(vision_feature_key=None, images=["shot-a.png"])
        other = _make_request(vision_feature_key=None, images=["shot-b.png"])
        digest_a = gen._media_identity_digest(same_a)
        assert digest_a is not None
        assert digest_a == gen._media_identity_digest(same_b)
        assert digest_a != gen._media_identity_digest(other)

    def test_stamped_feature_key_wins(self):
        gen = _stub_generator()
        keyed = _make_request(vision_feature_key="stamped-key")
        assert gen._media_identity_digest(keyed) == hash(("stamped-key",)) & (
            0xFFFFFFFFFFFF
        )


class TestStorePath:
    def test_store_plan_derives_expanded_boundary(self):
        gen = _stub_generator()
        # prompt "user turn ok" encodes to 12 rendered tokens;
        # prefix_boundary = 10 → marker width 2; expanded length 12 →
        # boundary 10, below the min-tokens gate: reject, counted miss.
        req = _make_request(
            prompt="user turn ok",
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=10,
            max_tokens=8,
        )
        full_ids = _full_ids(12, placeholder_at=5)
        plan = gen._media_boundary_plan(req, _ids(full_ids), _kv_leaves())
        assert plan is None
        assert gen._media_boundary_misses == 1

    def test_store_splits_forward_and_snapshots(self):
        gen = _stub_generator()
        req = _make_request(
            prompt="a" * 24,  # 24 rendered tokens
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,  # marker width 4
            max_tokens=8,
        )
        full_ids = _full_ids()
        ids = _ids(full_ids)
        cache = _kv_leaves()
        plan = gen._media_boundary_plan(req, ids, cache)
        assert plan is not None and plan[0] == "store"
        boundary = plan[2]
        assert boundary == len(full_ids) - 4

        rope_delta = mx.array([3])
        gen.language_model._rope_deltas = rope_delta
        kwargs = {"pixel_values": req.pixel_values}
        out = gen._media_forward(req, ids, cache, kwargs)

        model = gen.model
        # Prefix forward saw the vision inputs; suffix forward did not.
        assert model.calls[0] == (full_ids[0], boundary, True)
        assert model.calls[1] == (full_ids[boundary], len(full_ids), False)
        # Entry stored with the strict prefix and the recorded delta.
        digest = gen._media_identity_digest(req)
        entry = gen._media_boundary_entries[digest]
        assert entry.token_ids == full_ids[:boundary]
        assert entry.rope_delta is rope_delta
        assert entry.cache_bytes > 0
        assert gen._media_boundary_stores == 1
        assert out is not None

    def test_clone_failure_stores_nothing(self, monkeypatch):
        import mlx_vlm.apc_adapters as apc_adapters

        monkeypatch.setattr(apc_adapters, "clone_cache_entry", lambda *a, **k: None)
        gen = _stub_generator()
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        ids = _ids(_full_ids())
        cache = _kv_leaves()
        out = gen._media_forward(req, ids, cache, {"pixel_values": req.pixel_values})
        assert gen._media_boundary_stores == 0
        assert not gen._media_boundary_entries
        assert out is not None

    def test_below_min_tokens_boundary_never_stores(self):
        gen = _stub_generator()
        req = _make_request(
            prompt="abcdefgh",  # 8 rendered tokens
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=6,  # marker width 2
        )
        # Expanded sequence is short: boundary = 3 - 2 = 1 token, below the
        # min-boundary gate → never stored, counted miss.
        assert gen._media_boundary_plan(req, _ids([0, 1, 2]), _kv_leaves()) is None
        assert gen._media_boundary_misses == 1

    def test_mismatch_then_short_boundary_counts_one_miss(self):
        gen = _stub_generator()
        gen._media_mrope_save()
        gen.language_model._rope_deltas = mx.array([7])
        stored_ids = _full_ids()
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        gen._media_store(
            req, _kv_leaves(), _ids(stored_ids), len(stored_ids) - 4, mx.array([7])
        )
        # Same identity, but this turn's boundary is below the min-tokens
        # gate after the stored prefix failed verification: exactly one miss.
        short = _make_request(
            prompt="user turn ok",  # marker width 2, expanded 12 → boundary 10
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=10,
            max_tokens=8,
        )
        assert (
            gen._media_boundary_plan(short, _ids(_full_ids(12, 5)), _kv_leaves())
            is None
        )
        assert gen._media_boundary_misses == 1


class TestResumePath:
    def _gen_with_entry(self, full_ids):
        gen = _stub_generator()
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        ids = _ids(full_ids)
        cache = _kv_leaves()
        gen._media_mrope_save()
        gen.language_model._rope_deltas = mx.array([7])
        boundary = len(full_ids) - 4
        gen._media_store(req, cache, ids, boundary, gen.language_model._rope_deltas)
        return gen, req, ids, boundary

    def test_resume_installs_stored_state_and_forwards_suffix(self):
        full_ids = _full_ids()
        gen, req, ids, boundary = self._gen_with_entry(full_ids)
        digest = gen._media_identity_digest(req)
        entry = gen._media_boundary_entries[digest]
        stored_leaves = list(entry.leaves)

        fresh_cache = _kv_leaves()
        hits_before = gen._media_boundary_hits
        out = gen._media_forward(req, ids, fresh_cache, {"pixel_values": None})

        assert gen._media_boundary_hits == hits_before + 1
        # The live cache now holds the STORED boundary state (re-cloned,
        # not aliased): same offsets and equal KV contents.
        assert len(fresh_cache) == len(stored_leaves)
        for live, stored_leaf in zip(fresh_cache, stored_leaves):
            assert live is not stored_leaf
            assert live.offset == stored_leaf.offset
            span = int(stored_leaf.keys.shape[2])
            assert mx.array_equal(live.keys[:, :, :span, :], stored_leaf.keys)
            assert mx.array_equal(live.values[:, :, :span, :], stored_leaf.values)
        # Delta installed for decode; position ids reset.
        assert gen.language_model._rope_deltas is entry.rope_delta
        assert gen.language_model._position_ids is None
        # Only the strict suffix was forwarded, without vision inputs.
        model_call = gen.model.calls[-1]
        assert model_call[0] == full_ids[boundary]
        assert model_call[2] is False
        assert req.cached_tokens == boundary
        assert out is not None

    def test_prefix_mismatch_is_clean_miss(self):
        full_ids = _full_ids()
        gen, _, _, _ = self._gen_with_entry(full_ids)
        diverged = _full_ids()
        diverged[5] = 999
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        misses_before = gen._media_boundary_misses
        stores_before = gen._media_boundary_stores
        cold_cache = _kv_leaves()
        gen._media_forward(req, _ids(diverged), cold_cache, {"pixel_values": None})
        # Clean counted miss; the diverged prompt's own boundary is still
        # storable, so the forward splits and re-snapshots instead of
        # resuming — never a trim of the stored entry.
        assert gen._media_boundary_misses == misses_before + 1
        assert gen._media_boundary_stores == stores_before + 1
        assert gen.model.calls[0][0] == diverged[0]

    def test_resume_clone_failure_degrades_to_cold(self, monkeypatch):
        import mlx_vlm.apc_adapters as apc_adapters

        full_ids = _full_ids()
        gen, req, ids, _ = self._gen_with_entry(full_ids)
        monkeypatch.setattr(apc_adapters, "clone_cache_entry", lambda *a, **k: None)
        misses_before = gen._media_boundary_misses
        stores_before = gen._media_boundary_stores
        cache = _kv_leaves()
        gen._media_forward(req, ids, cache, {"pixel_values": None})
        # The plan counted the hit, the failed re-clone reverted it to a
        # miss: net hits back at the baseline, one counted miss.
        assert gen._media_boundary_hits == 0
        assert gen._media_boundary_misses == misses_before + 1
        # No usable snapshot: the fallback is the cold whole-sequence
        # forward, not a re-store. The seeded entry's store count stands.
        assert gen._media_boundary_stores == stores_before
        assert gen.model.calls[-1][1] - gen.model.calls[-1][0] == len(full_ids)


class TestMropeTransaction:
    def test_restore_deletes_absent_attributes(self):
        gen = _stub_generator()
        lm = gen.language_model
        del lm._position_ids
        del lm._rope_deltas
        gen._media_mrope_save()
        lm._position_ids = mx.array([1])
        lm._rope_deltas = mx.array([2])
        gen._media_mrope_restore()
        assert "_position_ids" not in lm.__dict__
        assert "_rope_deltas" not in lm.__dict__

    def test_restore_deletes_absent_attributes_on_real_module(self):
        gen = _stub_generator()
        gen.language_model = _RealLanguageModel()
        gen._media_mrope_save()
        gen._media_mrope_install(mx.array([5]))
        # nn.Module keeps plain array attributes outside __dict__; the
        # sentinel restore must still remove them (not silently no-op).
        assert hasattr(gen.language_model, "_rope_deltas")
        gen._media_mrope_restore()
        assert not hasattr(gen.language_model, "_rope_deltas")
        assert not hasattr(gen.language_model, "_position_ids")

    def test_restore_returns_prior_values(self):
        gen = _stub_generator()
        lm = gen.language_model
        prev_pos = mx.array([1])
        prev_delta = mx.array([2])
        lm._position_ids = prev_pos
        lm._rope_deltas = prev_delta
        gen._media_mrope_save()
        gen._media_mrope_install(mx.array([9]))
        assert lm._rope_deltas is not prev_delta
        gen._media_mrope_restore()
        assert lm._position_ids is prev_pos
        assert lm._rope_deltas is prev_delta

    def test_restore_noop_without_save(self):
        gen = _stub_generator()
        gen._media_mrope_restore()  # must not raise
        assert gen._media_mrope_saved is None


class TestBudget:
    def test_oldest_evicted_newest_kept(self, monkeypatch):
        gen = _stub_generator()
        gen._media_boundary_max_bytes = 10
        full_ids = _full_ids()

        def fake_clone(leaves, *, min_capacity_tokens):
            return _kv_leaves()

        monkeypatch.setattr(
            "vllm_mlx.mllm_batch_generator._media_clone_leaves", fake_clone
        )
        for image in ("img1.png", "img2.png", "img3.png"):
            req = _make_request(
                prompt="a" * 24,
                images=[image],
                vision_feature_key=f"hash-{image}",
                pixel_values=mx.zeros((1, 2)),
                prefix_boundary=20,
                max_tokens=8,
            )
            gen._media_mrope_save()
            gen._media_store(req, _kv_leaves(), _ids(full_ids), 26, mx.array([1]))
        stats = gen.get_media_prefix_stats()
        # Budget is tiny: only the newest entry survives.
        assert stats["entries"] == 1
        assert stats["budget_evictions"] == 2
        digest = gen._media_identity_digest(
            _make_request(images=["img3.png"], vision_feature_key="hash-img3.png")
        )
        assert digest in gen._media_boundary_entries


class TestDetachContract:
    def test_clone_is_detached_and_evaluated(self):
        leaves = _kv_leaves()
        cloned = _media_clone_leaves(leaves, min_capacity_tokens=64)
        assert cloned is not None
        for src, dst in zip(leaves, cloned):
            assert src is not dst
            assert not mx.array_equal(dst.keys, src.keys + 100.0)
            src.keys = src.keys + 100.0
            mx.eval(src.keys)
            assert not mx.array_equal(dst.keys, src.keys)


class TestStatsSurface:
    def test_stats_shape(self):
        gen = _stub_generator()
        stats = gen.get_media_prefix_stats()
        assert set(stats) == {
            "hits",
            "misses",
            "stores",
            "budget_evictions",
            "entries",
            "bytes",
            "budget_bytes",
        }
        assert stats["entries"] == 0
