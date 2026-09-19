"""Probes for the vendored generation core (step 3a).

Mechanical guarantee: every function/class body in the vendored
``generate/ar.py``, ``generate/common.py``, ``generate/types.py``, and
``sample_utils.py`` is byte-identical to the pinned upstream
``mlx-vlm==0.7.1`` source. The only permitted differences are the
documented import redirects (see the package inventory), which live at
module level and therefore never enter a function's ``getsource``.

Behavioral guarantee: the vendored AR core binds the vendored primitives
(cache, inputs helpers, sampler) so the module graph stays inside the
vendored package except for the documented pinned redirects.
"""

import inspect
import types

import mlx.core as mx
import pytest

import rapid_mlx.models.mlx_vlm_vendored as vendored_pkg
import rapid_mlx.models.mlx_vlm_vendored.generate as vendored_generate
import rapid_mlx.models.mlx_vlm_vendored.generate.ar as vendored_ar
import rapid_mlx.models.mlx_vlm_vendored.generate.common as vendored_common
import rapid_mlx.models.mlx_vlm_vendored.generate.types as vendored_types
import rapid_mlx.models.mlx_vlm_vendored.inputs as vendored_inputs
import rapid_mlx.models.mlx_vlm_vendored.sample_utils as vendored_sample_utils

pytest.importorskip("mlx_vlm")

# Functions whose BODY carries a documented deviation hunk (the deviation
# lives on lines inside the body, so getsource differs by exactly those
# sentinel lines):
# - ``prepare_inputs``: the inputs.py hunks (bytes-path fsdecode; see the
#   inputs.py inventory entry; behavior-tested in
#   tests/test_mlx_vlm_vendored_inputs.py).
# - ``kv_quant_from_legacy``: kv_quant.py's documented lazy ``.turboquant``
#   redirect (see the kv_quant.py inventory entry).
# - ``generate_step`` / ``batch_generate``: ar.py's redirects to the
#   pinned speculative drafters helper and to the vendored
#   ``inputs.process_image``.
# - ``_generate_batch``: the capture-release + None-token bugfix hunks
#   (finally-close; skip token=None terminal responses), repro-tested below.
# - ``BatchGenerator``: the class body carries the APC matched_blocks
#   release-on-failed-merge bugfix hunk, repro-tested below.
_DOCUMENTED_HUNK_BODIES = {
    "BatchGenerator",
    "prepare_inputs",
    "kv_quant_from_legacy",
    "generate_step",
    "batch_generate",
    "_generate_batch",
}


def _body_divergences(vendored_module, upstream_module):
    diverged = []
    for name, obj in vars(vendored_module).items():
        if name.startswith("__") or name in _DOCUMENTED_HUNK_BODIES:
            continue
        upstream_obj = getattr(upstream_module, name, None)
        if upstream_obj is None:
            if inspect.isfunction(obj) or inspect.isclass(obj):
                # A vendored-only function/class means the copy is not a
                # faithful verbatim region — flag it.
                diverged.append(f"{name}: no upstream symbol")
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


def test_vendored_ar_bodies_match_upstream():
    from mlx_vlm.generate import ar as upstream_ar

    assert _body_divergences(vendored_ar, upstream_ar) == []


def test_vendored_common_bodies_match_upstream():
    from mlx_vlm.generate import common as upstream_common

    assert _body_divergences(vendored_common, upstream_common) == []


def test_vendored_types_and_sample_utils_bodies_match_upstream():
    from mlx_vlm import sample_utils as upstream_sample_utils
    from mlx_vlm.generate import types as upstream_types

    assert _body_divergences(vendored_types, upstream_types) == []
    assert _body_divergences(vendored_sample_utils, upstream_sample_utils) == []


def test_vendored_ar_binds_vendored_primitives():
    assert vendored_ar.cache is vendored_pkg.cache
    assert vendored_ar.prepare_inputs is vendored_inputs.prepare_inputs
    assert vendored_ar.group_images_by_shape is (vendored_inputs.group_images_by_shape)
    assert vendored_ar.should_add_special_tokens is (
        vendored_inputs.should_add_special_tokens
    )
    assert vendored_ar.make_sampler is vendored_sample_utils.make_sampler
    assert vendored_ar.DEFAULT_KV_GROUP_SIZE is (vendored_common.DEFAULT_KV_GROUP_SIZE)
    assert vendored_ar.GenerateKwargs is vendored_types.GenerateKwargs


def test_generate_shim_exports_text_ar_surface_only():
    """The shim re-exports the vendored surface and nothing from the
    (not-yet-vendored) modality modules — the subset-exports deviation."""
    for name in (
        "BatchGenerator",
        "BatchResponse",
        "BatchStats",
        "PromptProcessingBatch",
        "batch_generate",
        "generate_step",
        "GenerationResult",
        "PromptCacheState",
        "generation_stream",
        "maybe_quantize_kv_cache",
        "wired_limit",
        "GenerateKwargs",
        "ProcessorLike",
    ):
        assert hasattr(vendored_generate, name), name
    # The upstream init eagerly pulls dispatch (generate/stream_generate),
    # image/audio/video/diffusion/edit_image; none is vendored in 3a.
    for name in ("generate", "stream_generate", "generate_image"):
        assert not hasattr(vendored_generate, name), name


def test_generate_step_signature_matches_upstream():
    from mlx_vlm.generate import ar as upstream_ar

    assert (
        inspect.signature(vendored_ar.generate_step).parameters.keys()
        == inspect.signature(upstream_ar.generate_step).parameters.keys()
    )


def test_generate_batch_closes_generator_on_exception(monkeypatch):
    """upstream-bugfix: _generate_batch must close the generator (and its
    wired_limit context) even when the generation loop raises."""
    closed = []

    class _FakeGen:
        has_work = True

        def __init__(self, *args, **kwargs):
            pass

        def insert(self, *args, **kwargs):
            return ["u1"]

        def next(self):
            raise RuntimeError("generation boom")

        def close(self):
            closed.append(True)

    class _FakeEmbedding:
        def to_dict(self):
            return {}

    class _FakeModel:
        config = type("C", (), {"model_type": "fake"})()
        language_model = None

        def get_input_embeddings(self, *args, **kwargs):
            return _FakeEmbedding()

    class _FakeProcessor:
        tokenizer = None

    monkeypatch.setattr(vendored_ar, "BatchGenerator", _FakeGen)
    monkeypatch.setattr(vendored_ar, "apply_chat_template", lambda *a, **k: "p")
    monkeypatch.setattr(vendored_ar, "should_add_special_tokens", lambda *a, **k: False)
    monkeypatch.setattr(
        vendored_ar,
        "prepare_inputs",
        lambda *a, **k: {"input_ids": mx.array([[1]])},
    )
    monkeypatch.setattr(
        vendored_ar, "_default_prefill_step_size_for_offload", lambda *a, **k: None
    )

    with pytest.raises(RuntimeError, match="generation boom"):
        vendored_ar._generate_batch(_FakeModel(), _FakeProcessor(), ["p"])
    assert closed == [True]


def test_thinking_budget_criteria_default_start_token_does_not_crash():
    """upstream-bugfix: the documented thinking_start_token=None default
    must construct (pinned upstream crashes in tokenizer.encode(None))."""

    class _HFStyleTokenizer:
        def encode(self, text, add_special_tokens=False):
            if text is None:
                raise TypeError(
                    "the following arguments are required of type str: 'text'"
                )
            return [1, 2]

    criteria = vendored_inputs.ThinkingBudgetCriteria(
        _HFStyleTokenizer(),
        thinking_budget=8,
        thinking_start_token=None,
        enable_thinking=True,
    )
    assert criteria.thinking_start_token_id is None
    # The span-entry comparison is guarded: no crash, no span entry.
    assert criteria(5) is None

    pytest.importorskip("mlx_vlm.utils")
    from mlx_vlm.utils import ThinkingBudgetCriteria as UpstreamCriteria

    with pytest.raises(TypeError):
        UpstreamCriteria(
            _HFStyleTokenizer(),
            thinking_budget=8,
            thinking_start_token=None,
            enable_thinking=True,
        )


def test_mixed_prompt_batch_releases_picks_on_warm_merge_failure(monkeypatch):
    """upstream-bugfix: acquired APC matched_blocks are released when
    warm-cache merging fails and the caller falls back to cold prefill."""
    released = []

    class _FakeManager:
        def release(self, blocks):
            released.append(list(blocks))

    pick = {"matched_blocks": ["blk1"], "prefix_len": 2, "warm_cache": None}
    fake_self = types.SimpleNamespace(
        apc_manager=_FakeManager(),
        apc=None,
        apc_mode="block",
        kv_bits=None,
        kv_quant_scheme=None,
        kv_group_size=None,
        kv_key_bits=None,
        kv_value_bits=None,
        kv_key_scheme=None,
        model=types.SimpleNamespace(make_cache=lambda: []),
        _APC_PRIVATE_KEYS=getattr(
            vendored_ar.BatchGenerator, "_APC_PRIVATE_KEYS", set()
        ),
        _apc_pick_for=lambda sequence: pick,
    )
    fake_self._assemble_mixed_prompt_batch = lambda sequences, picks: (
        vendored_ar.BatchGenerator._assemble_mixed_prompt_batch(
            fake_self, sequences, picks
        )
    )
    monkeypatch.setattr(
        vendored_ar._apc,
        "make_warm_batch_kv_cache_multi",
        lambda *args, **kwargs: (None, None),
    )

    out = vendored_ar.BatchGenerator._build_mixed_prompt_batch(
        fake_self,
        [
            (
                "u1",
                [1, 2, 3],
                10,
                {"inputs_embeds": mx.zeros((1, 3, 4))},
                None,
                None,
            )
        ],
    )
    assert out is None
    assert released == [["blk1"]]


def test_mixed_prompt_batch_releases_picks_on_assembly_exception(monkeypatch):
    """upstream-bugfix: exceptions after the APC lookups release the
    acquired matched_blocks before propagating."""

    class _FakeManager:
        def __init__(self):
            self.released = []

        def release(self, blocks):
            self.released.append(list(blocks))

    manager = _FakeManager()
    pick = {"matched_blocks": ["blk9"], "prefix_len": 2, "warm_cache": None}
    fake_self = types.SimpleNamespace(
        apc_manager=manager,
        apc=None,
        apc_mode="block",
        kv_bits=None,
        kv_quant_scheme=None,
        kv_group_size=None,
        kv_key_bits=None,
        kv_value_bits=None,
        kv_key_scheme=None,
        model=types.SimpleNamespace(make_cache=lambda: []),
        _APC_PRIVATE_KEYS=getattr(
            vendored_ar.BatchGenerator, "_APC_PRIVATE_KEYS", set()
        ),
        _apc_pick_for=lambda sequence: pick,
    )
    fake_self._assemble_mixed_prompt_batch = lambda sequences, picks: (
        vendored_ar.BatchGenerator._assemble_mixed_prompt_batch(
            fake_self, sequences, picks
        )
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("merge boom")

    monkeypatch.setattr(vendored_ar._apc, "make_warm_batch_kv_cache_multi", _boom)

    with pytest.raises(RuntimeError, match="merge boom"):
        vendored_ar.BatchGenerator._build_mixed_prompt_batch(
            fake_self,
            [
                (
                    "u1",
                    [1, 2, 3],
                    10,
                    {"inputs_embeds": mx.zeros((1, 3, 4))},
                    None,
                    None,
                )
            ],
        )
    assert manager.released == [["blk9"]]
