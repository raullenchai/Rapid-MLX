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
* only the qualified model family may split; the byte-bounded store
  evicts oldest-first under the shared ceiling (media-preferred order,
  text may empty once media holds room).

Design note: docs/engineering/design/2026-09-15-mllm-media-prefix-cache.md.
"""

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

from collections import OrderedDict  # noqa: E402
from typing import Any  # noqa: E402

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from vllm_mlx.mllm_batch_generator import (  # noqa: E402
    MLLMBatchGenerator,
    MLLMBatchRequest,
    MLLMBatchStats,
    _media_clone_leaves,
    _media_leaf_bytes,
    _media_leaves_bytes,
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


def _full_ids(n: int = 196, placeholder_at: int = 10) -> list[int]:
    """An expanded sequence with one placeholder well inside the boundary.

    Long enough that the derived boundary stays above the 64-token
    alignment grid after flooring (``_MEDIA_BOUNDARY_ALIGN_TOKENS``).
    """
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
        # ``model_type`` matches the qualified family so the family gate
        # opens; the gate itself is tested separately below.
        self.config = type(
            "Config",
            (),
            {"image_token_id": _PLACEHOLDER_ID, "model_type": "qwen3_5_moe"},
        )()

    def __call__(self, ids, cache=None, pixel_values=None, **kwargs):
        rope_deltas = kwargs.pop("rope_deltas", None)
        start = int(ids[0, 0]) if ids.size else -1
        self.calls.append((start, start + int(ids.shape[1]), pixel_values is not None))
        return _Output(mx.zeros((1, int(ids.shape[1]), self.vocab)))


class _NoRopeModel:
    """A wrapper that declares ``rope_deltas`` and plumbs it onward, but
    accepts no ``**kwargs`` — the gate opens only with a consuming LM."""

    def __init__(self, vocab: int = VOCAB):
        self.vocab = vocab
        self.calls: list[tuple[int, int, bool]] = []
        self.config = type("Config", (), {"model_type": "qwen3_5_moe"})()

    def __call__(self, ids, cache=None, pixel_values=None, rope_deltas=None):
        forwarded = {"rope_deltas": rope_deltas}
        start = int(ids[0, 0]) if ids.size else -1
        self.calls.append((start, start + int(ids.shape[1]), pixel_values is not None))
        return _Output(mx.zeros((1, int(ids.shape[1]), self.vocab)))


class _CommentRopeModel(_NoRopeModel):
    """False-positive shape: the call body *mentions* ``"rope_deltas"`` in
    a comment but consumes nothing — no ``**kwargs`` to carry it through."""

    def __call__(self, ids, cache=None, pixel_values=None, rope_deltas=None):
        # legacy callers used to pass "rope_deltas" positionally
        start = int(ids[0, 0]) if ids.size else -1
        self.calls.append((start, start + int(ids.shape[1]), pixel_values is not None))
        return _Output(mx.zeros((1, int(ids.shape[1]), self.vocab)))


class _IgnoringExplicitRopeModel:
    """Declares a ``rope_deltas`` parameter and silently ignores it: the
    gate must not ride on the language model's consumption when the
    wrapper itself never plumbs the value."""

    def __init__(self, vocab: int = VOCAB):
        self.vocab = vocab
        self.calls: list[tuple[int, int, bool]] = []
        self.config = type("Config", (), {"model_type": "qwen3_5_moe"})()

    def __call__(self, ids, cache=None, pixel_values=None, rope_deltas=None):
        start = int(ids[0, 0]) if ids.size else -1
        self.calls.append((start, start + int(ids.shape[1]), pixel_values is not None))
        return _Output(mx.zeros((1, int(ids.shape[1]), self.vocab)))


class _KwargsCommentRopeModel:
    """Worst false-positive shape: the call *accepts* ``**kwargs`` and the
    body only *mentions* ``"rope_deltas"`` (comment/log line) without ever
    consuming it — an installed delta would be silently ignored, so the
    gate must stay closed for consumption-shaped matching."""

    def __init__(self, vocab: int = VOCAB):
        self.vocab = vocab
        self.calls: list[tuple[int, int, bool]] = []

    def __call__(self, ids, cache=None, pixel_values=None, **kwargs):
        # historical note: "rope_deltas" used to arrive positionally here
        start = int(ids[0, 0]) if ids.size else -1
        self.calls.append((start, start + int(ids.shape[1]), pixel_values is not None))
        return _Output(mx.zeros((1, int(ids.shape[1]), self.vocab)))


class _DroppingKwargsModel:
    """Accepts ``**kwargs`` and drops them; declares no rope_deltas
    parameter — an installed delta would silently vanish."""

    def __init__(self, vocab: int = VOCAB):
        self.vocab = vocab
        self.calls: list[tuple[int, int, bool]] = []
        self.config = type("Config", (), {"model_type": "qwen3_5_moe"})()

    def __call__(self, ids, cache=None, pixel_values=None, **kwargs):
        start = int(ids[0, 0]) if ids.size else -1
        self.calls.append((start, start + int(ids.shape[1]), pixel_values is not None))
        return _Output(mx.zeros((1, int(ids.shape[1]), self.vocab)))


class _SpreadingKwargsModel(_DroppingKwargsModel):
    """Forwards ``**kwargs`` into an inner call, so a passed delta reaches
    the (consuming) language model."""

    def __call__(self, ids, cache=None, pixel_values=None, **kwargs):
        forwarded = dict(**kwargs)
        start = int(ids[0, 0]) if ids.size else -1
        self.calls.append((start, start + int(ids.shape[1]), pixel_values is not None))
        return _Output(mx.zeros((1, int(ids.shape[1]), self.vocab)))


class _EmbedFeatures:
    def __init__(self, inputs_embeds, rope_deltas=None):
        self.inputs_embeds = inputs_embeds
        self.rope_deltas = rope_deltas

    def to_dict(self):
        return {"inputs_embeds": self.inputs_embeds}


class _DirectLanguageModel:
    """An mlx-vlm-shaped language model: accepts ``inputs_embeds`` and
    consumes ``rope_deltas`` through ``**kwargs``."""

    def __init__(self, vocab: int = VOCAB):
        self.vocab = vocab
        self.calls: list[tuple[int, int, Any]] = []
        self.config = type("Config", (), {"model_type": "qwen3_5_moe_text"})()
        self._position_ids = None
        self._rope_deltas = None

    def __call__(self, tokens, inputs_embeds=None, mask=None, cache=None, **kwargs):
        rope_deltas = kwargs.pop("rope_deltas", None)
        start = int(tokens[0, 0]) if tokens.size else -1
        self.calls.append((start, start + int(tokens.shape[1]), rope_deltas))
        return _Output(mx.zeros((1, int(tokens.shape[1]), self.vocab)))


class _PositionOverrideModel(_RecordingModel):
    """qwen3-vl-shaped wrapper: merges ``InputEmbeddingsFeatures.to_dict()``
    — including a freshly recomputed (0-based) position payload — into the
    LM call. On a ``pixel_values=None`` suffix forward that merge overrides
    the installed boundary delta, so the split must bypass this wrapper."""

    def __init__(self, vocab: int = VOCAB):
        super().__init__(vocab)
        self.embed_calls: list[int] = []

    def __call__(self, ids, cache=None, pixel_values=None, **kwargs):
        rope_deltas = kwargs.pop("rope_deltas", None)
        start = int(ids[0, 0]) if ids.size else -1
        self.calls.append((start, start + int(ids.shape[1]), pixel_values is not None))
        feats = self.get_input_embeddings(ids, pixel_values)
        kwargs.update({"pixel_values": pixel_values, **feats.to_dict()})
        return _Output(mx.zeros((1, int(ids.shape[1]), self.vocab)))

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        self.embed_calls.append(int(input_ids.shape[1]))
        return _EmbedFeatures(mx.zeros((1, int(input_ids.shape[1]), 4)))


class _OverrideNoEmbedsModel(_PositionOverrideModel):
    """The corrupting wrapper shape WITHOUT the LM-direct escape hatch."""

    get_input_embeddings = None  # type: ignore[assignment]


class _TalkerOverrideModel(_PositionOverrideModel):
    """Thinker/talker-shaped wrapper (qwen3_omni_moe): the wrapper performs
    talker bookkeeping around the LM call, so the LM-direct bypass must
    never engage even though the position-override probe matches."""

    def __call__(self, ids, cache=None, pixel_values=None, **kwargs):
        rope_deltas = kwargs.pop("rope_deltas", None)
        start = int(ids[0, 0]) if ids.size else -1
        self.calls.append((start, start + int(ids.shape[1]), pixel_values is not None))
        feats = self.get_input_embeddings(ids, pixel_values)
        kwargs.update({"pixel_values": pixel_values, **feats.to_dict()})
        # The wrapper must sync talker state before returning; a direct
        # language_model call would skip that bookkeeping entirely.
        return _Output(mx.zeros((1, int(ids.shape[1]), self.vocab)))


class _Output:
    def __init__(self, logits):
        self.logits = logits


class _FakeLanguageModel:
    def __init__(self):
        # Qualified text family so the family gate stays open for tests
        # that exercise later gates (placeholder, boundary, verification).
        self.config = type("Config", (), {"model_type": "qwen3_5_moe_text"})()
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
    gen._media_singleton_turn = True
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

    def test_plan_gated_by_prefix_cache_opt_out(self):
        # ``enable_prefix_cache=False`` covers every form of prefix reuse on
        # this lane — the media boundary resume included.
        gen = _stub_generator()
        gen._prefix_cache_enabled = False
        req = _make_request(pixel_values=mx.zeros((1, 2)))
        assert gen._media_boundary_plan(req, _ids([1, 2, 3]), _kv_leaves()) is None
        assert gen._media_boundary_misses == 0

    def test_plan_gated_by_unqualified_family(self):
        # Only the qualified family (Qwen3.6 hybrid) may split: the 64-token
        # alignment and MRoPE transaction are qualified per family, so any
        # other model — even one that consumes rope_deltas — stays cold.
        gen = _stub_generator()
        gen.model.config = type("Config", (), {"model_type": "qwen3_vl"})()
        gen.language_model.config = type("Config", (), {"model_type": "qwen2_vl"})()
        req = _make_request(pixel_values=mx.zeros((1, 2)))
        assert gen._media_boundary_plan(req, _ids(_full_ids()), _kv_leaves()) is None
        # Structural gate: fires before planning, no miss counted.
        assert gen._media_boundary_misses == 0

    def test_family_probe_memoizes_per_model(self):
        gen = _stub_generator()
        first = gen._media_family_qualified()
        second = gen._media_family_qualified()
        assert first is second is True
        assert gen._media_family_probe[0] == (
            type(gen.model),
            type(gen.language_model),
        )

    def test_unsupported_wrapper_with_supporting_lm_opens_the_gate(self):
        # The production qwen3-vl shape: the wrapper declares rope_deltas as
        # an explicit parameter and plumbs it onward, and the language model
        # that ultimately consumes it pops it from **kwargs.
        gen = _stub_generator(model=_NoRopeModel())
        gen.language_model = _DirectLanguageModel()
        assert gen._media_model_supports_rope_kwarg() is True

    def test_declared_but_ignored_rope_param_fails_closed(self):
        # Declaring the parameter proves nothing: a wrapper that accepts
        # rope_deltas and silently drops it must stay closed even with a
        # consuming language model behind it.
        gen = _stub_generator(model=_IgnoringExplicitRopeModel())
        gen.language_model = _DirectLanguageModel()
        assert gen._media_model_supports_rope_kwarg() is False

    def test_kwargs_dropping_wrapper_fails_closed_despite_consuming_lm(self):
        # A wrapper that accepts **kwargs and drops them cannot ride on its
        # inner language model's consumption: the installed delta would
        # never reach the LM and the resumed suffix would be mispositioned.
        gen = _stub_generator(model=_DroppingKwargsModel())
        gen.language_model = _DirectLanguageModel()
        assert gen._media_model_supports_rope_kwarg() is False

    def test_kwargs_spreading_wrapper_opens_the_gate_with_consuming_lm(self):
        # A wrapper that forwards **kwargs into an inner call hands the
        # delta to the consuming language model.
        gen = _stub_generator(model=_SpreadingKwargsModel())
        gen.language_model = _DirectLanguageModel()
        assert gen._media_model_supports_rope_kwarg() is True

    def test_comment_mention_without_kwargs_fails_closed(self):
        # A quoted "rope_deltas" in a comment must not open the gate when
        # the call has no **kwargs to carry it.
        gen = _stub_generator(model=_CommentRopeModel())
        assert gen._media_model_supports_rope_kwarg() is False

    def test_comment_mention_with_kwargs_fails_closed(self):
        # **kwargs alone is not enough: a call that never *consumes*
        # rope_deltas would silently drop an installed delta on resume.
        # The probe matches consumption shapes, not a bare mention.
        gen = _stub_generator(model=_KwargsCommentRopeModel())
        assert gen._media_model_supports_rope_kwarg() is False

    def test_source_read_failure_fails_closed(self, monkeypatch):
        # inspect.getsource can raise SyntaxError from a stale linecache;
        # every source-derived gate must fail closed, not propagate.
        def boom(target):
            raise SyntaxError("stale linecache")

        monkeypatch.setattr("vllm_mlx.mllm_batch_generator.inspect.getsource", boom)
        gen = _stub_generator(model=_PositionOverrideModel())
        gen.language_model = _DirectLanguageModel()
        assert gen._media_model_supports_rope_kwarg() is False
        assert gen._media_wrapper_overrides_positions() is False
        assert gen._media_lm_direct_available() is False


class TestWrapperPositionOverride:
    def test_probe_detects_to_dict_merging_wrapper(self):
        override = _stub_generator(model=_PositionOverrideModel())
        assert override._media_wrapper_overrides_positions() is True
        plain = _stub_generator(model=_RecordingModel())
        assert plain._media_wrapper_overrides_positions() is False

    def test_talker_wrapper_never_bypasses(self):
        # qwen3_omni_moe-shaped wrappers do talker bookkeeping a direct LM
        # call would skip: the bypass must refuse them (plan fails closed).
        gen = _stub_generator(model=_TalkerOverrideModel())
        gen.language_model = _DirectLanguageModel()
        assert gen._media_wrapper_overrides_positions() is True
        assert gen._media_lm_direct_available() is False
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        assert gen._media_boundary_plan(req, _ids(_full_ids()), _kv_leaves()) is None

    def test_probes_memoize_per_model(self):
        gen = _stub_generator(model=_PositionOverrideModel())
        gen.language_model = _DirectLanguageModel()
        first = gen._media_wrapper_overrides_positions()
        second = gen._media_wrapper_overrides_positions()
        assert first is second is True
        # The source probe ran once: the memo survives further calls.
        assert gen._media_wrapper_source[0] is type(gen.model)
        rope_first = gen._media_model_supports_rope_kwarg()
        rope_second = gen._media_model_supports_rope_kwarg()
        assert rope_first is rope_second is True
        assert gen._media_rope_targets[type(gen.language_model)] is True

    def test_plan_fails_closed_without_lm_direct_escape(self):
        # The corrupting wrapper shape without get_input_embeddings: the
        # suffix cannot bypass the wrapper, so planning fails closed before
        # any forward (structural gate — no miss counted).
        gen = _stub_generator(model=_OverrideNoEmbedsModel())
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        assert gen._media_boundary_plan(req, _ids(_full_ids()), _kv_leaves()) is None
        assert gen._media_boundary_misses == 0

    def test_store_suffix_routes_through_language_model(self):
        gen = _stub_generator(model=_PositionOverrideModel())
        gen.language_model = _DirectLanguageModel()
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        full_ids = _full_ids()
        ids = _ids(full_ids)
        cache = _kv_leaves()
        rope_delta = mx.array([3])
        gen.language_model._rope_deltas = rope_delta
        gen._media_forward(req, ids, cache, {"pixel_values": req.pixel_values})

        # Wrapper saw only the prefix forward (with vision inputs); the
        # suffix went to the language model with the boundary delta.
        assert [call[:2] for call in gen.model.calls] == [
            (full_ids[0], len(full_ids) - 4)
        ]
        suffix_call = gen.language_model.calls[-1]
        assert suffix_call[0] == full_ids[len(full_ids) - 4]
        assert suffix_call[1] == len(full_ids)
        # The install passes the entry's own detached delta copy.
        assert mx.array_equal(suffix_call[2], rope_delta)

    def test_resume_suffix_routes_through_language_model(self):
        full_ids = _full_ids()
        gen = _stub_generator(model=_PositionOverrideModel())
        gen.language_model = _DirectLanguageModel()
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        rope_delta = mx.array([7])
        boundary = len(full_ids) - 4
        gen._media_store(req, _kv_leaves(), _ids(full_ids), boundary, rope_delta)

        fresh_cache = _kv_leaves()
        gen._media_forward(req, _ids(full_ids), fresh_cache, {"pixel_values": None})
        # Resume forwards the strict suffix via the LM-direct path only —
        # the wrapper is never invoked on the resumed suffix.
        assert gen.model.calls == []
        suffix_call = gen.language_model.calls[-1]
        assert suffix_call[0] == full_ids[boundary]
        # The install passes the entry's own detached delta copy.
        assert mx.array_equal(suffix_call[2], rope_delta)
        assert req.cached_tokens == boundary


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
        full_ids[193] = _PLACEHOLDER_ID  # derived boundary 196, aligned 192
        req = _make_request(
            prompt="a" * 192,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=188,  # marker width 4
        )
        assert gen._media_boundary_plan(req, _ids(full_ids), _kv_leaves()) is None
        assert gen._media_boundary_misses == 1

    def test_boundary_aligns_down_to_tile_grid(self):
        # marker width 4 → derived boundary 196 → floored to the 64-token
        # recurrent-scan grid: the split's tiles then coincide with the
        # single forward's (bit-exact store/resume on hybrid models).
        gen = _stub_generator()
        req = _make_request(
            prompt="a" * 196,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=192,
            max_tokens=8,
        )
        plan = gen._media_boundary_plan(req, _ids(_full_ids(200)), _kv_leaves())
        assert plan is not None and plan[0] == "store"
        assert plan[2] == 192

    def test_alignment_below_placeholder_fails_closed(self):
        # Flooring must never push the boundary onto a placeholder.
        gen = _stub_generator()
        ids = _full_ids(200)
        ids[193] = _PLACEHOLDER_ID  # aligned boundary 192 < 193
        req = _make_request(
            prompt="a" * 196,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=192,
            max_tokens=8,
        )
        assert gen._media_boundary_plan(req, _ids(ids), _kv_leaves()) is None
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

    def test_wide_batch_never_plans(self):
        # The plan is keyed to the actual B=1 turn: inside a wider batch the
        # cache list is shared, so planning must fail closed (silently —
        # structural gate, not a modeling miss).
        gen = _stub_generator()
        gen._media_singleton_turn = False
        req = _make_request(pixel_values=mx.zeros((1, 2)))
        assert gen._media_boundary_plan(req, _ids(_full_ids()), _kv_leaves()) is None
        assert gen._media_boundary_misses == 0


class TestResumePlaceholderGate:
    def _seed_entry(self, stored_ids=None):
        gen = _stub_generator()
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        gen._media_mrope_save()
        gen.language_model._rope_deltas = mx.array([7])
        if stored_ids is None:
            stored_ids = _full_ids(80)
        gen._media_store(
            req, _kv_leaves(), _ids(stored_ids), len(stored_ids) - 4, mx.array([7])
        )
        return gen

    def test_suffix_placeholder_never_resumes(self):
        # The image re-sent on a later turn re-expands placeholders below
        # the boundary: the strict token prefix matches, but the resumed
        # suffix forwards with pixel_values=None — it must be rejected.
        gen = self._seed_entry()  # stored prefix = first 76 tokens
        extended = _full_ids(200)  # prefix matches the stored 76 tokens
        extended[150] = _PLACEHOLDER_ID  # inside the aligned boundary 192
        req = _make_request(
            prompt="a" * 196,  # marker width 4 → derived 196, aligned 192
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=192,
            max_tokens=8,
        )
        hits_before = gen._media_boundary_hits
        plan = gen._media_boundary_plan(req, _ids(extended), _kv_leaves())
        assert plan is not None and plan[0] == "store"
        assert gen._media_boundary_hits == hits_before
        assert gen._media_boundary_misses == 1

    def test_unresolvable_placeholder_ids_fail_resume_closed(self):
        gen = self._seed_entry()
        gen.model.config = None
        req = _make_request(
            prompt="a" * 196,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=192,
            max_tokens=8,
        )
        plan = gen._media_boundary_plan(req, _ids(_full_ids(200)), _kv_leaves())
        # Cannot verify the suffix invariant → clean miss; without the ids
        # the store gate stays closed too.
        assert plan is None
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

    def test_effective_pixel_cap_rides_in_the_digest(self):
        # Identical image bytes preprocessed under different effective pixel
        # caps (the auto ceiling varies with image count; the budget loop
        # reduces it further) resize to different pixel tensors — their KV
        # state must never share an entry.
        gen = _stub_generator()
        low = _make_request(vision_feature_key="stamped-key", media_pixel_cap=1024)
        high = _make_request(vision_feature_key="stamped-key", media_pixel_cap=4096)
        uncapped = _make_request(vision_feature_key="stamped-key")
        digest_low = gen._media_identity_digest(low)
        assert digest_low is not None
        assert digest_low != gen._media_identity_digest(high)
        assert digest_low == gen._media_identity_digest(
            _make_request(vision_feature_key="stamped-key", media_pixel_cap=1024)
        )
        # A request that never went through preprocessing (cap 0) keys
        # distinctly from any capped one.
        assert digest_low != gen._media_identity_digest(uncapped)

    def test_stamped_feature_key_wins(self):
        gen = _stub_generator()
        keyed = _make_request(vision_feature_key="stamped-key")
        # The stamped key leads the digest verbatim (no truncated hash that
        # could alias distinct media sets), suffixed by the semantic salt
        # so a semantics-changing configuration never shares an entry.
        digest = gen._media_identity_digest(keyed)
        assert digest is not None and digest.startswith("stamped-key#")
        # The salt is stable per model and folds the pixel bounds.
        assert gen._media_identity_digest(keyed) == digest
        assert gen._media_semantics_salt().endswith(
            f"#{int(getattr(gen, 'vision_min_pixels', 0) or 0)}"
            f"#{int(getattr(gen, 'vision_max_pixels', 0) or 0)}"
        )
        # A different media identity never collides with the stamped one.
        other = _make_request(images=["other.png"], vision_feature_key=None)
        assert gen._media_identity_digest(other) != digest


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
        # Stored as an evaluated detached copy, not the model-owned array.
        assert entry.rope_delta is not rope_delta
        assert mx.array_equal(entry.rope_delta, rope_delta)
        assert entry.cache_bytes > 0
        assert gen._media_boundary_stores == 1
        assert out is not None

    def test_clone_failure_redoes_a_cold_full_forward(self, monkeypatch):
        import mlx_vlm.apc_adapters as apc_adapters

        monkeypatch.setattr(apc_adapters, "clone_cache_entry", lambda *a, **k: None)
        gen = _stub_generator()
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        full_ids = _full_ids()
        ids = _ids(full_ids)
        cache = _kv_leaves()
        gen.language_model._rope_deltas = mx.array([3])
        gen.language_model.layers = []
        out = gen._media_forward(req, ids, cache, {"pixel_values": req.pixel_values})
        assert gen._media_boundary_stores == 0
        assert not gen._media_boundary_entries
        assert gen._media_boundary_misses == 1
        assert len(gen.model.calls) == 2
        # Second call: the cold redo over the whole sequence — never a
        # snapshotless split suffix.
        assert gen.model.calls[-1] == (full_ids[0], len(full_ids), True)
        assert out is not None

    def test_over_budget_snapshot_redoes_a_cold_full_forward(self):
        gen = _stub_generator()
        gen._media_boundary_max_bytes = 1
        full_ids = _full_ids()
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        gen.language_model._rope_deltas = mx.array([3])
        gen.language_model.layers = []
        out = gen._media_forward(
            req, _ids(full_ids), _kv_leaves(), {"pixel_values": req.pixel_values}
        )
        assert out is not None
        assert not gen._media_boundary_entries
        assert gen._media_boundary_stores == 0
        assert gen._media_boundary_misses == 1
        assert gen.model.calls[-1] == (full_ids[0], len(full_ids), True)

    def test_unmeasurable_snapshot_refuses_the_store(self, monkeypatch):
        import vllm_mlx.mllm_batch_generator as mlbg

        gen = _stub_generator()
        full_ids = _full_ids()
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        gen.language_model._rope_deltas = mx.array([3])
        gen.language_model.layers = []
        # A cloned leaf that exposes neither state, nbytes, nor array
        # attributes: its footprint against the shared ceiling is unknown,
        # so the store must refuse it rather than charge it as zero bytes.
        monkeypatch.setattr(mlbg, "_media_clone_leaves", lambda *a, **k: [object()])
        out = gen._media_forward(
            req, _ids(full_ids), _kv_leaves(), {"pixel_values": req.pixel_values}
        )
        assert out is not None
        assert not gen._media_boundary_entries
        assert gen._media_boundary_stores == 0
        assert gen._media_boundary_misses == 1
        assert gen.model.calls[-1] == (full_ids[0], len(full_ids), True)

    def test_media_leaves_bytes_measures_arrays_not_wrapper_truthiness(self):
        leaf = _kv_leaves()[0]
        nbytes_only = type("SnapshotLeaf", (), {"nbytes": 777})()
        assert _media_leaf_bytes(leaf) is not None and _media_leaf_bytes(leaf) > 0
        assert _media_leaf_bytes(nbytes_only) == 777
        # No state, no nbytes, no array attributes: unmeasurable.
        assert _media_leaf_bytes(object()) is None
        assert _media_leaves_bytes([leaf, nbytes_only]) == (
            _media_leaf_bytes(leaf) + 777
        )
        assert _media_leaves_bytes([leaf, object()]) is None

    def test_store_without_rope_delta_redoes_a_cold_full_forward(self):
        gen = _stub_generator()
        full_ids = _full_ids()
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        # No delta lands on the language model: the split suffix cannot be
        # positioned (and _media_store refuses to snapshot it), so the
        # request must restart as one cold full forward on a fresh cache.
        gen.language_model.layers = []
        out = gen._media_forward(
            req, _ids(full_ids), _kv_leaves(), {"pixel_values": req.pixel_values}
        )
        assert out is not None
        assert not gen._media_boundary_entries
        assert gen._media_boundary_stores == 0
        assert gen._media_boundary_misses == 1
        model = gen.model
        assert len(model.calls) == 2
        # First call: the split prefix. Last call: the cold redo over the
        # whole sequence, vision inputs included.
        assert model.calls[0][1] < len(full_ids)
        assert model.calls[-1] == (full_ids[0], len(full_ids), True)
        # The redo restored the transaction the prefix opened.
        assert gen._media_mrope_saved is None

    def test_store_snapshots_an_evaluated_detached_delta(self):
        gen = _stub_generator()
        full_ids = _full_ids()
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        # A lazily-computed delta: the model owns it and its graph is still
        # pending when the snapshot is taken.
        lazy = mx.ones((2,)) * 5
        entry = gen._media_store(
            req, _kv_leaves(), _ids(full_ids), len(full_ids) - 4, lazy
        )
        assert entry is not None
        assert entry.rope_delta is not lazy
        assert mx.array_equal(entry.rope_delta, mx.array([5, 5]))

    def test_store_suffix_failure_discards_the_published_boundary(self, monkeypatch):
        gen = _stub_generator()
        full_ids = _full_ids()
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        gen.language_model._rope_deltas = mx.array([3])

        def broken_suffix(*args, **kwargs):
            raise RuntimeError("suffix exploded")

        monkeypatch.setattr(gen, "_media_suffix_forward", broken_suffix)
        with pytest.raises(RuntimeError, match="suffix exploded"):
            gen._media_forward(
                req,
                _ids(full_ids),
                _kv_leaves(),
                {"pixel_values": req.pixel_values},
            )
        digest = gen._media_identity_digest(req)
        assert digest not in gen._media_boundary_entries
        # ``stores`` stays monotonic — it counts publishes, not live
        # entries; the failed request leaves no reusable boundary.
        assert gen._media_boundary_stores == 1

    def test_promotion_after_concurrent_clear_does_not_raise(self):
        gen = _stub_generator()
        full_ids = _full_ids()
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        gen._media_store(
            req, _kv_leaves(), _ids(full_ids), len(full_ids) - 4, mx.array([3])
        )
        digest = gen._media_identity_digest(req)
        gen._media_promote_entry(digest)
        assert list(gen._media_boundary_entries)[-1] == digest
        # A concurrent clear removed the key between the lookup and the
        # promotion: no KeyError, no promotion.
        gen._media_boundary_entries.clear()
        gen._media_promote_entry(digest)

    def test_clear_during_resume_clone_forces_a_cold_miss(self, monkeypatch):
        import rapid_mlx.mllm_batch_generator as mlbg

        gen = _stub_generator()
        full_ids = _full_ids()
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        gen._media_store(
            req, _kv_leaves(), _ids(full_ids), len(full_ids) - 4, mx.array([3])
        )
        gen.language_model._rope_deltas = mx.array([3])
        gen.language_model.layers = []
        real_clone = mlbg._media_clone_leaves

        def clearing_clone(source, **kwargs):
            # A concurrent clear lands between the plan and the install.
            gen.clear_prefix_cache(reset_stats=False)
            return real_clone(source, **kwargs)

        monkeypatch.setattr(mlbg, "_media_clone_leaves", clearing_clone)
        out = gen._media_forward(
            req, _ids(full_ids), _kv_leaves(), {"pixel_values": req.pixel_values}
        )
        assert out is not None
        # The stale-generation resume degraded to a counted cold miss: the
        # plan's hit was rolled back and the request ran one full forward.
        assert gen._media_boundary_hits == 0
        assert gen._media_boundary_misses == 1
        assert len(gen.model.calls) == 1
        assert gen.model.calls[0] == (full_ids[0], len(full_ids), True)

    def test_rope_probe_requires_a_real_consumption_operation(self):
        consume = MLLMBatchGenerator._media_rope_consumes_key
        assert consume(
            "def f(**kwargs):\n    rope_deltas = kwargs.pop('rope_deltas', None)\n"
        )
        assert consume(
            'def f(**kwargs):\n    rope_deltas = kwargs.get("rope_deltas")\n'
        )
        assert consume("def f(**kwargs):\n    d = kwargs['rope_deltas']\n")
        # Comments, log lines, and dead mentions prove nothing.
        assert not consume(
            "def f(**kwargs):\n"
            '    # kwargs.pop("rope_deltas", None) used to happen here\n'
            "    return None\n"
        )
        assert not consume(
            "def f(**kwargs):\n"
            '    logger.info("rope_deltas = kwargs is the legacy shape")\n'
            "    return None\n"
        )
        # Dead code never consumes: constant-false branches and statements
        # after an unconditional return are not live.
        assert not consume(
            "def f(**kwargs):\n"
            "    if False:\n"
            "        d = kwargs.pop('rope_deltas', None)\n"
            "    return None\n"
        )
        assert not consume(
            "def f(**kwargs):\n    return None\n    d = kwargs['rope_deltas']\n"
        )
        # A nested function's body never runs in the probed scope.
        assert not consume(
            "def f(**kwargs):\n"
            "    def g(**inner):\n"
            "        return kwargs.pop('rope_deltas', None)\n"
            "    return None\n"
        )
        # typing.TYPE_CHECKING blocks are dead at runtime.
        assert not consume(
            "def f(**kwargs):\n"
            "    if TYPE_CHECKING:\n"
            "        d = kwargs.pop('rope_deltas', None)\n"
            "    return None\n"
        )
        # An access confined to an exception handler runs only after
        # forwarding has already failed — the success path never consumes.
        assert not consume(
            "def f(**kwargs):\n"
            "    try:\n"
            "        return forward(**kwargs)\n"
            "    except Exception:\n"
            "        return kwargs.pop('rope_deltas', None)\n"
        )
        # Consumption on the success path (try body) still counts.
        assert consume(
            "def f(**kwargs):\n"
            "    try:\n"
            "        d = kwargs.pop('rope_deltas', None)\n"
            "        return forward(d, **kwargs)\n"
            "    except Exception:\n"
            "        d = None\n"
        )

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
    def test_store_path_failure_restores_mrope_state(self, monkeypatch):
        # The save transaction spans the whole request: a failing prefix
        # forward must restore the prior model-global position state
        # before the exception propagates.
        gen = _stub_generator()
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        gen._media_mrope_save()
        lm = gen.language_model
        prior_delta = mx.array([2])
        lm._rope_deltas = prior_delta
        lm._position_ids = None
        lm.calls = None  # sentinel: raise on any forward

        def broken_forward(*args, **kwargs):
            raise RuntimeError("prefill exploded")

        gen.model = type(
            "Broken",
            (),
            {"__call__": staticmethod(broken_forward)},
        )()
        with pytest.raises(RuntimeError, match="prefill exploded"):
            gen._media_forward(req, _ids(_full_ids()), _kv_leaves(), {})
        assert lm._rope_deltas is prior_delta

    def test_resume_suffix_failure_restores_mrope_state(self, monkeypatch):
        gen = _stub_generator()
        full_ids = _full_ids()
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        prior_delta = mx.array([2])
        gen.language_model._rope_deltas = prior_delta
        gen._media_mrope_save()
        gen._media_store(
            req, _kv_leaves(), _ids(full_ids), len(full_ids) - 4, mx.array([3])
        )
        assert gen._media_boundary_stores == 1

        def broken_suffix(*args, **kwargs):
            raise RuntimeError("suffix exploded")

        monkeypatch.setattr(gen, "_media_suffix_forward", broken_suffix)
        with pytest.raises(RuntimeError, match="suffix exploded"):
            gen._media_forward(req, _ids(full_ids), _kv_leaves(), {})
        assert gen.language_model._rope_deltas is prior_delta

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
        # Ceiling = exactly one entry's bytes: the admission cap lets every
        # store through (entry == ceiling), and enforcement keeps only the
        # newest.
        gen._media_boundary_max_bytes = _media_leaves_bytes(_kv_leaves())
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

    def test_text_budget_credits_media_bytes(self, monkeypatch):
        # Symmetric accounting: the text eviction must charge the media
        # store's bytes against the shared ceiling, exactly as the media
        # store credits the text footprint.
        import threading

        gen = _stub_generator()
        gen._prefix_cache_max_bytes = 12
        gen._prefix_cache_budget_evictions = 0
        gen._media_boundary_max_bytes = 12
        gen._media_boundary_entries["media"] = type("Entry", (), {"cache_bytes": 10})()
        lock = threading.Lock()
        entries: OrderedDict[str, Any] = OrderedDict()
        for key in ("a", "b"):
            entries[key] = type(
                "ExactEntry", (), {"prompt_cache": [type("L", (), {"nbytes": 5})()]}
            )()
        monkeypatch.setattr(gen, "_exact_entries", lambda cache: (lock, entries))
        monkeypatch.setattr(
            "vllm_mlx.mllm_batch_generator.checkpoint_bytes", lambda stored: 0
        )
        # Effective budget 12 - 10 = 2 against 10 text bytes: the text side
        # empties completely — the combined footprint must fit the shared
        # ceiling strictly, and the surviving state is the media entry.
        gen._enforce_exact_cache_budget(object())
        assert list(entries) == []
        assert gen._prefix_cache_budget_evictions == 2

    def test_text_budget_media_over_ceiling_empties_text(self, monkeypatch):
        # One media entry alone meeting the shared ceiling must not freeze
        # text eviction: the text side empties completely, leaving the
        # media entry as the sole survivor under the strict ceiling.
        import threading

        gen = _stub_generator()
        gen._prefix_cache_max_bytes = 12
        gen._prefix_cache_budget_evictions = 0
        gen._media_boundary_max_bytes = 12
        gen._media_boundary_entries["media"] = type("Entry", (), {"cache_bytes": 15})()
        lock = threading.Lock()
        entries: OrderedDict[str, Any] = OrderedDict()
        for key in ("a", "b", "c"):
            entries[key] = type(
                "ExactEntry", (), {"prompt_cache": [type("L", (), {"nbytes": 5})()]}
            )()
        monkeypatch.setattr(gen, "_exact_entries", lambda cache: (lock, entries))
        monkeypatch.setattr(
            "vllm_mlx.mllm_batch_generator.checkpoint_bytes", lambda stored: 0
        )
        gen._enforce_exact_cache_budget(object())
        assert list(entries) == []
        assert gen._prefix_cache_budget_evictions == 3

    def test_text_budget_zero_config_disables_eviction(self, monkeypatch):
        # A zero ceiling means the feature is off: no eviction either way.
        import threading

        gen = _stub_generator()
        gen._prefix_cache_max_bytes = 0
        gen._prefix_cache_budget_evictions = 0
        lock = threading.Lock()
        entries: OrderedDict[str, Any] = OrderedDict()
        entries["a"] = type(
            "ExactEntry", (), {"prompt_cache": [type("L", (), {"nbytes": 5})()]}
        )()
        monkeypatch.setattr(gen, "_exact_entries", lambda cache: (lock, entries))
        gen._enforce_exact_cache_budget(object())
        assert list(entries) == ["a"]
        assert gen._prefix_cache_budget_evictions == 0

    def test_media_insertion_evicts_text_oldest_first(self, monkeypatch):
        # Coordinated eviction: when the media store's own entries are
        # exhausted and the combined footprint still exceeds the ceiling,
        # the media side reclaims room from the text side (oldest first).
        import threading

        gen = _stub_generator()
        gen._prefix_cache_max_bytes = 0  # text budget step inert
        gen._media_boundary_max_bytes = 12
        gen._prefix_cache_budget_evictions = 0
        gen._media_boundary_entries["media"] = type("Entry", (), {"cache_bytes": 10})()
        lock = threading.Lock()
        entries: OrderedDict[str, Any] = OrderedDict()
        for key in ("a", "b"):
            entries[key] = type(
                "ExactEntry", (), {"prompt_cache": [type("L", (), {"nbytes": 5})()]}
            )()
        monkeypatch.setattr(gen, "_exact_entries", lambda cache: (lock, entries))
        monkeypatch.setattr(
            "vllm_mlx.mllm_batch_generator.checkpoint_bytes", lambda stored: 0
        )
        gen._prefix_cache = object()
        gen._media_enforce_budget()
        # 10 media + 10 text > 12: the text side empties so the combined
        # footprint fits the shared ceiling strictly (media entry survives).
        assert list(entries) == []
        assert gen._prefix_cache_budget_evictions == 2

    def test_media_store_refuses_entry_over_ceiling(self, monkeypatch):
        # Admission cap: a single snapshot larger than the whole shared
        # ceiling is discarded — it can never fit beside anything.
        gen = _stub_generator()
        gen._media_boundary_max_bytes = 4
        full_ids = _full_ids()

        def fake_clone(leaves, *, min_capacity_tokens):
            return _kv_leaves()

        monkeypatch.setattr(
            "vllm_mlx.mllm_batch_generator._media_clone_leaves", fake_clone
        )
        req = _make_request(
            prompt="a" * 24,
            pixel_values=mx.zeros((1, 2)),
            prefix_boundary=20,
            max_tokens=8,
        )
        gen._media_mrope_save()
        stored = gen._media_store(req, _kv_leaves(), _ids(full_ids), 26, mx.array([1]))
        assert stored is None
        assert not gen._media_boundary_entries
        assert gen._media_boundary_stores == 0

    def test_footprint_zero_without_text_cache(self):
        gen = _stub_generator()
        assert gen._exact_cache_footprint_bytes() == 0

    def test_footprint_charges_the_ceiling_for_an_unintrospectable_store(self):
        # A present but malformed exact-entry store cannot be walked; the
        # media side must assume the text side holds the whole ceiling —
        # never admit media against unaccounted room.
        gen = _stub_generator()
        gen._prefix_cache_max_bytes = 1 << 20
        gen._prefix_cache = type(
            "Cache", (), {"_exact_cache": ["not", "an", "ordereddict"], "lock": None}
        )()
        assert gen._exact_cache_footprint_bytes() == gen._media_resolved_budget()

    def test_footprint_zero_for_a_manager_without_an_exact_store(self):
        # No exact-entry store at all: the text side holds nothing chargeable.
        gen = _stub_generator()
        gen._prefix_cache = type("Cache", (), {})()
        assert gen._exact_cache_footprint_bytes() == 0


class TestClearPrefixCache:
    def test_clear_drops_media_entries_without_text_cache(self, monkeypatch):
        gen = _stub_generator()
        req = _make_request()
        gen._media_mrope_save()
        gen._media_store(req, _kv_leaves(), _ids(_full_ids()), 26, mx.array([7]))
        assert gen._media_boundary_entries
        monkeypatch.setattr(gen, "_media_enforce_budget", lambda: None)
        # No text-APC manager attached, but real media bytes were dropped —
        # the return must report that reusable state was held and released.
        assert gen.clear_prefix_cache() is True
        assert not gen._media_boundary_entries
        # Nothing held at all: nothing was cleared.
        assert gen.clear_prefix_cache() is False

    def test_clear_reset_stats_zeroes_media_counters(self, monkeypatch):
        # ``reset_stats=True`` is the method's documented contract: media
        # counters zero with the text counters, keeping the stats shape
        # consistent. ``reset_stats=False`` preserves lifetime totals.
        gen = _stub_generator()
        req = _make_request()
        gen._media_boundary_hits = 5
        gen._media_boundary_misses = 3
        gen._media_boundary_stores = 2
        gen._media_boundary_budget_evictions = 1
        monkeypatch.setattr(gen, "_media_enforce_budget", lambda: None)
        gen.clear_prefix_cache(reset_stats=False)
        assert (gen._media_boundary_hits, gen._media_boundary_misses) == (5, 3)
        assert (gen._media_boundary_stores, gen._media_boundary_budget_evictions) == (
            2,
            1,
        )
        gen.clear_prefix_cache(reset_stats=True)
        assert gen._media_boundary_hits == 0
        assert gen._media_boundary_misses == 0
        assert gen._media_boundary_stores == 0
        assert gen._media_boundary_budget_evictions == 0

    def test_clear_with_text_cache_drops_both(self, monkeypatch):
        gen = _stub_generator()
        req = _make_request()
        gen._media_mrope_save()
        gen._media_store(req, _kv_leaves(), _ids(_full_ids()), 26, mx.array([7]))
        monkeypatch.setattr(gen, "_media_enforce_budget", lambda: None)
        cache = type("Cache", (), {"clear": lambda self: None})()
        cache.stats_snapshot = lambda: {"evictions": 0}
        gen._prefix_cache = cache
        assert gen.clear_prefix_cache() is True
        assert not gen._media_boundary_entries


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

    def test_stats_never_raise_during_concurrent_stores(self, monkeypatch):
        # get_media_prefix_stats runs off the worker thread while the step
        # executor inserts/evicts: the store lock must make the snapshot
        # atomic instead of racing the OrderedDict mutation.
        import threading

        gen = _stub_generator()
        gen._media_boundary_entries = OrderedDict()
        gen._media_boundary_lock = threading.Lock()
        full_ids = _full_ids()
        errors: list[BaseException] = []
        stop = threading.Event()

        def store_loop():
            try:
                while not stop.is_set():
                    gen._media_mrope_save()
                    gen._media_store(
                        _make_request(
                            prompt="a" * 24,
                            pixel_values=mx.zeros((1, 2)),
                            prefix_boundary=20,
                            max_tokens=8,
                        ),
                        _kv_leaves(),
                        _ids(full_ids),
                        26,
                        mx.array([1]),
                    )
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        worker = threading.Thread(target=store_loop, daemon=True)
        worker.start()
        try:
            for _ in range(500):
                stats = gen.get_media_prefix_stats()
                assert stats["entries"] >= 0
        finally:
            stop.set()
            worker.join(timeout=5)
        assert not errors
