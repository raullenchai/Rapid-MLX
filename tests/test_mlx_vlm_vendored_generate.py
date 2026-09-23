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

import ast
import importlib
import inspect
import types

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

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
#   ``inputs.process_image``; generate_step also keeps cache quantization in
#   the dual vendored/upstream cache namespace.
# - ``maybe_quantize_kv_cache``: accept fallback vendored caches and caches
#   returned by still-upstream model implementations.
# - ``_generate_module_override``: ignore upstream's default AR exports while
#   retaining support for an explicitly patched public override.
# - ``_is_batch_cache_entry`` / ``_make_cache``: accept and preserve both
#   cache namespaces during continuous-batch conversion.
# - ``_generate_batch``: the capture-release + None-token bugfix hunks
#   (finally-close; skip token=None terminal responses), repro-tested below.
# - ``_merge_prefill_prompt_kwargs``: reject tensor kwargs that are absent
#   from any row instead of concatenating a row-shifted batch.
# - ``BatchGenerator``: the class body carries the APC matched_blocks
#   release-on-failed-merge bugfix hunks, repro-tested below.
# - ``GenerationBatch``: the decode sampling hunk passes the per-row int
#   uids as row_ids so seeded draws stay independent (upstream passes
#   row_ids=[0]*n and correlates same-position rows), repro-tested below.
# - ``SpeculativeGenerationBatch``: same row_ids fix on the speculative
#   rounds kickoff, repro-tested below.
# - ``PromptProcessingBatch``: same row_ids fix on the first-token sample,
#   plus the constructor prepare-guard no longer releases the APC meta
#   blocks before raising (the mixed-assembly caller owns release; the
#   upstream double-release underflows shared refcounts), repro-tested
#   below.
# - ``run_speculative_server_rounds``: imported into ar's namespace from
#   vendored ``speculative/utils.py``, whose server-rounds call site
#   threads the server's per-request row ID into singleton dflash
#   positioned sampling (step-3b inventory; upstream hard-codes row 0).
_DOCUMENTED_HUNK_BODIES = {
    "BatchGenerator",
    "GenerationBatch",
    "SpeculativeGenerationBatch",
    "PromptProcessingBatch",
    "maybe_quantize_kv_cache",
    "_generate_module_override",
    "_is_batch_cache_entry",
    "_make_cache",
    "prepare_inputs",
    "kv_quant_from_legacy",
    "generate_step",
    "batch_generate",
    "_generate_batch",
    "_merge_prefill_prompt_kwargs",
    "run_speculative_server_rounds",
}


def _body_divergences(vendored_module, upstream_module):
    def defined_bodies(module):
        # Read the installed source rather than live module attributes. Runtime
        # compatibility hooks intentionally replace a few upstream symbols;
        # parity is against the pinned file, not that process-global mutation.
        source = inspect.getsource(module)
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)
        return {
            node.name: "".join(lines[node.lineno - 1 : node.end_lineno])
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }

    diverged = []
    vendored_bodies = defined_bodies(vendored_module)
    upstream_bodies = defined_bodies(upstream_module)
    for name, vendored_src in vendored_bodies.items():
        if name.startswith("__") or name in _DOCUMENTED_HUNK_BODIES:
            continue
        upstream_src = upstream_bodies.get(name)
        if upstream_src is None:
            # A vendored-only function/class means the copy is not a faithful
            # verbatim region — flag it.
            diverged.append(f"{name}: no upstream symbol")
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


def test_generate_override_keeps_vendored_defaults_but_honors_patches(monkeypatch):
    """Default upstream exports must not pull vendored batching upstream."""
    upstream_generate = importlib.import_module("mlx_vlm.generate")

    assert (
        vendored_ar._generate_module_override(
            "PromptProcessingBatch", vendored_ar.PromptProcessingBatch
        )
        is vendored_ar.PromptProcessingBatch
    )

    patched = type("PatchedPromptProcessingBatch", (), {})
    monkeypatch.setattr(upstream_generate, "PromptProcessingBatch", patched)
    assert (
        vendored_ar._generate_module_override(
            "PromptProcessingBatch", vendored_ar.PromptProcessingBatch
        )
        is patched
    )


def test_generate_step_keeps_quantizer_in_vendored_cache_namespace(monkeypatch):
    """The loaded upstream package must not steal quantization from caches
    constructed by the vendored generation core."""
    upstream_generate = importlib.import_module("mlx_vlm.generate")

    calls = []

    def vendored_quantizer(prompt_cache, **kwargs):
        calls.append((prompt_cache, kwargs))

    def upstream_quantizer(*args, **kwargs):
        raise AssertionError("upstream quantizer cannot recognize vendored caches")

    class _EmbeddingOutput:
        inputs_embeds = mx.zeros((1, 1, 4))

        def to_dict(self):
            return {"inputs_embeds": self.inputs_embeds}

    class _LanguageModel:
        def __call__(self, *args, **kwargs):
            return types.SimpleNamespace(
                logits=mx.zeros((1, 1, 8)),
                cross_attention_states=None,
                encoder_outputs=None,
            )

    class _Model:
        language_model = _LanguageModel()

        def get_input_embeddings(self, *args, **kwargs):
            return _EmbeddingOutput()

    monkeypatch.setattr(vendored_ar, "maybe_quantize_kv_cache", vendored_quantizer)
    monkeypatch.setattr(
        upstream_generate, "maybe_quantize_kv_cache", upstream_quantizer
    )

    list(
        vendored_ar.generate_step(
            mx.array([[1]], dtype=mx.int32),
            _Model(),
            None,
            None,
            max_tokens=0,
            prompt_cache=[],
        )
    )

    assert len(calls) == 1
    assert calls[0][0] == []


@pytest.mark.parametrize(
    "cache_module",
    [vendored_pkg.cache, importlib.import_module("mlx_vlm.models.cache")],
)
def test_vendored_quantizer_accepts_both_cache_namespaces(monkeypatch, cache_module):
    """Model-owned upstream and fallback vendored caches must both quantize."""

    class _FakeTurboQuantKVCache:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(vendored_common, "kv_quant_from_legacy", lambda *a: None)
    monkeypatch.setattr(vendored_common, "turboquant_enabled", lambda *a: True)
    monkeypatch.setattr(vendored_common, "TurboQuantKVCache", _FakeTurboQuantKVCache)

    nested = cache_module.CacheList(cache_module.KVCache())
    prompt_cache = [nested]
    vendored_common.maybe_quantize_kv_cache(
        prompt_cache,
        quantized_kv_start=5000,
        kv_group_size=64,
        kv_bits=4,
        kv_quant_scheme="turboquant",
    )

    assert isinstance(prompt_cache[0], cache_module.CacheList)
    assert isinstance(prompt_cache[0].caches[0], _FakeTurboQuantKVCache)


@pytest.mark.parametrize(
    "cache_module",
    [vendored_pkg.cache, importlib.import_module("mlx_vlm.models.cache")],
)
def test_batch_cache_conversion_preserves_cache_namespace(cache_module):
    """Continuous batching accepts caches from either producer namespace."""

    arrays = cache_module.ArraysCache(size=2)
    model_cache = [
        cache_module.KVCache(),
        cache_module.CacheList(cache_module.KVCache()),
        cache_module.PoolingCache(2),
        cache_module.RotatingKVCache(16),
        arrays,
    ]
    model = types.SimpleNamespace(make_cache=lambda: model_cache)

    converted = vendored_ar._make_cache(model, [0, 1])

    assert isinstance(converted[0], cache_module.BatchKVCache)
    assert isinstance(converted[1], cache_module.CacheList)
    assert isinstance(converted[1].caches[0], cache_module.BatchKVCache)
    assert vendored_ar._is_batch_cache_entry(converted[1])
    assert isinstance(converted[2], cache_module.BatchPoolingCache)
    assert isinstance(converted[3], cache_module.BatchRotatingKVCache)
    assert converted[4] is arrays
    assert converted[4].left_padding.tolist() == [0, 1]

    quantized = vendored_ar._make_cache(
        types.SimpleNamespace(make_cache=lambda: [cache_module.KVCache()]),
        [0, 1],
        kv_bits=4,
    )
    assert isinstance(quantized[0], cache_module.BatchQuantizedKVCache)


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


def test_merge_prefill_prompt_kwargs_rejects_sparse_tensor_keys():
    """upstream-bugfix: a tensor kwarg present on only part of a mixed batch
    must not be concatenated into a smaller, row-shifted tensor."""
    rows = [
        {
            "inputs_embeds": mx.zeros((1, 3, 4)),
            "attention_mask": mx.ones((1, 3), dtype=mx.int32),
        },
        {"inputs_embeds": mx.zeros((1, 2, 4))},
    ]

    with pytest.raises(
        ValueError,
        match="batched prompt kwarg 'attention_mask' must be present for every row",
    ):
        vendored_ar._merge_prefill_prompt_kwargs(rows, [[1, 2, 3], [4, 5]])


def test_assemble_mixed_prompt_batch_rejects_sparse_tensor_keys():
    """The APC warm/cold assembly path enforces the same row-alignment
    contract as the cold-only helper."""
    fake_self = types.SimpleNamespace(
        _APC_PRIVATE_KEYS=vendored_ar.APC_PRIVATE_PROMPT_KEYS
    )
    sequences = [
        (
            "u1",
            [1, 2, 3],
            8,
            {
                "inputs_embeds": mx.zeros((1, 3, 4)),
                "attention_mask": mx.ones((1, 3), dtype=mx.int32),
            },
            None,
            None,
        ),
        (
            "u2",
            [4, 5],
            8,
            {"inputs_embeds": mx.zeros((1, 2, 4))},
            None,
            None,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="batched prompt kwarg 'attention_mask' must be present for every row",
    ):
        vendored_ar.BatchGenerator._assemble_mixed_prompt_batch(
            fake_self,
            sequences,
            [None, {"prefix_len": 0}],
        )


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


def test_generation_batch_decode_passes_uid_row_ids():
    """upstream-bugfix: seeded batched sampling must give every row its own
    key. Upstream passes row_ids=[0]*n, so rows sharing a generated position
    fold to the same _position_keys entry and their draws correlate; the
    vendored core passes the per-row int uids (unique per generator, stable
    across filter()/extend())."""

    class _RecordingSampler:
        def __init__(self):
            self.row_ids = None
            self.positions = None

        def sample_target(self, logprobs, *, row_ids, positions):
            self.row_ids = list(row_ids)
            self.positions = list(positions)
            return mx.zeros((logprobs.shape[0],), dtype=mx.int32)

    class _LM:
        def __call__(self, inputs, cache=None, **kwargs):
            return types.SimpleNamespace(logits=mx.zeros((inputs.shape[0], 1, 8)))

    sampler = _RecordingSampler()
    batch = vendored_ar.GenerationBatch(
        model=types.SimpleNamespace(language_model=_LM()),
        uids=[7, 11],
        inputs=mx.array([[5, 9]], dtype=mx.int32),
        prompt_cache=[],
        sampler=sampler,
        stop_criteria=lambda token: False,
        max_tokens=[10, 10],
    )
    batch._step()
    assert sampler.row_ids == [7, 11]
    assert sampler.positions == [1, 1]


def test_speculative_rounds_pass_uid_row_ids(monkeypatch):
    """upstream-bugfix: the speculative rounds kickoff must pass distinct
    per-row sampling identities too."""

    captured = {}

    def _fake_rounds(*args, **kwargs):
        captured["row_ids"] = kwargs.get("row_ids")
        return iter([])

    monkeypatch.setattr(vendored_ar, "run_speculative_server_rounds", _fake_rounds)
    batch = vendored_ar.SpeculativeGenerationBatch.__new__(
        vendored_ar.SpeculativeGenerationBatch
    )
    batch.model = types.SimpleNamespace()
    batch.draft_model = None
    batch.draft_kind = "mtp"
    batch.prompt_cache = []
    batch.hidden = None
    batch.first_tokens = mx.zeros((2,), dtype=mx.int32)
    batch.max_tokens = [4, 4]
    batch.sampler = lambda logprobs: logprobs
    batch.draft_block_size = 2
    batch.token_dtype = mx.int32
    batch.stop_criteria = lambda token: False
    batch.greedy_sampling = True
    batch.shared_kv_states = None
    batch.prompt_tokens = mx.zeros((2, 3), dtype=mx.int32)
    batch._all_uids = [3, 9]
    batch._finished = [False, False]
    batch._num_tokens = [0, 0]
    batch._rounds_iter = None
    batch._start_rounds()
    assert captured["row_ids"] == [3, 9]


def test_batch_generator_does_not_count_tokenless_terminal_responses():
    """Speculative exhaustion sentinels complete rows without generating tokens."""

    class _ExhaustedBatch:
        logits_processors = []
        prompt_cache = []

        def __init__(self):
            self.active = True

        def __len__(self):
            return int(self.active)

        def next(self):
            self.active = False
            return [
                types.SimpleNamespace(token=7),
                types.SimpleNamespace(token=None, finish_reason="length"),
            ]

    generator = types.SimpleNamespace(
        _generation_batch=_ExhaustedBatch(),
        _gen_tokens_counter=0,
        _steps_counter=0,
        _cache_eval_interval=0,
        completion_batch_size=8,
        _prompt_batch=None,
        _unprocessed_sequences=[],
        prefill_batch_size=1,
    )

    _, responses = vendored_ar.BatchGenerator._next(generator)
    assert [response.token for response in responses] == [7, None]
    assert generator._gen_tokens_counter == 1


def test_prompt_batch_first_token_passes_uid_row_ids():
    """upstream-bugfix: the post-prefill first-token sample must carry the
    per-row uids as row_ids (upstream passed row_ids=[0]*n)."""

    class _RecordingSampler:
        def __init__(self):
            self.row_ids = None

        def sample_target(self, logprobs, *, row_ids, positions):
            self.row_ids = list(row_ids)
            return mx.zeros((logprobs.shape[0],), dtype=mx.int32)

    class _Model:
        layers = []

        def __call__(self, input_ids, cache=None, inputs_embeds=None, **kwargs):
            return types.SimpleNamespace(
                logits=mx.zeros((input_ids.shape[0], input_ids.shape[1], 8))
            )

    batch = vendored_ar.PromptProcessingBatch(
        model=_Model(),
        uids=[7, 11],
        input_ids=[[1, 2, 3], [4, 5, 6]],
        max_tokens=[4, 4],
        inputs_embeds=None,
        prompt_kwargs={},
    )
    # Keep the harvest/transition machinery inert for this probe.
    batch._apc_manager = None
    sampler = _RecordingSampler()
    gen_batch = batch.generate(sampler, lambda token: False, compute_logprobs=False)
    assert sampler.row_ids == [7, 11]
    assert list(gen_batch.uids) == [7, 11]


def test_mixed_prompt_batch_ctor_guard_leaves_release_to_caller(monkeypatch):
    """upstream-bugfix: the PromptProcessingBatch prepare-guard used to
    release the APC meta blocks before raising while
    BatchGenerator._build_mixed_prompt_batch's failure handler released the
    same acquired blocks again — a double release underflowing shared
    refcounts. Release ownership stays with the caller: exactly one release
    must hit the manager."""

    class _FakeManager:
        def __init__(self):
            self.released = []

        def release(self, blocks):
            self.released.append(list(blocks))

    manager = _FakeManager()
    pick = {
        "matched_blocks": ["blk7"],
        "prefix_len": 2,
        "warm_cache": None,
        "extra_hash": 0,
    }
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
        _apc_exact_checkpoint_len=lambda ids: 0,
        _apc_exact_checkpoint_lengths=lambda ids: [],
        prefill_step_size=None,
    )
    fake_self._assemble_mixed_prompt_batch = lambda sequences, picks: (
        vendored_ar.BatchGenerator._assemble_mixed_prompt_batch(
            fake_self, sequences, picks
        )
    )

    # Warm-cache merge succeeds, handing the raw no-prepare cache objects to
    # the constructor, whose right-pad prepare-guard raises (the shorter of
    # the two rows carries real right-padding, so the guard branch runs).
    monkeypatch.setattr(
        vendored_ar._apc,
        "make_warm_batch_kv_cache_multi",
        lambda *a, **k: ([object(), object()], None),
    )

    with pytest.raises(RuntimeError, match="requires a prompt cache with prepare"):
        vendored_ar.BatchGenerator._build_mixed_prompt_batch(
            fake_self,
            [
                (
                    "u1",
                    [1, 2, 3, 4],
                    10,
                    {"inputs_embeds": mx.zeros((1, 4, 4))},
                    None,
                    None,
                ),
                (
                    "u2",
                    [1, 2, 3],
                    10,
                    {"inputs_embeds": mx.zeros((1, 3, 4))},
                    None,
                    None,
                ),
            ],
        )
    # Exactly one release carrying the acquired blocks per pick: the caller
    # handler's. The ctor guard may still record no-op empty releases on the
    # stripped metas, but 'blk7' must never be released twice.
    assert [blocks for blocks in manager.released if blocks] == [
        ["blk7"],
        ["blk7"],
    ]


def test_mixed_prompt_batch_reattaches_blocks_on_success(monkeypatch):
    """upstream-bugfix companion: after a successful mixed assembly the
    acquired block references must be back on the batch's metas so the
    post-prefill harvest/commit lifecycle still owns them."""

    class _PrepareCache:
        def __init__(self):
            self.prepared = None

        def prepare(self, right_padding=None, lengths=None):
            self.prepared = (right_padding, lengths)

    class _Manager:
        def __init__(self):
            self.released = []

        def release(self, blocks):
            self.released.append(list(blocks))

    manager = _Manager()
    pick = {
        "matched_blocks": ["blk7"],
        "prefix_len": 2,
        "warm_cache": None,
        "extra_hash": 0,
    }
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
        _apc_exact_checkpoint_len=lambda ids: 0,
        _apc_exact_checkpoint_lengths=lambda ids: [],
        prefill_step_size=None,
    )
    fake_self._assemble_mixed_prompt_batch = lambda sequences, picks: (
        vendored_ar.BatchGenerator._assemble_mixed_prompt_batch(
            fake_self, sequences, picks
        )
    )

    caches = [_PrepareCache(), _PrepareCache()]
    monkeypatch.setattr(
        vendored_ar._apc,
        "make_warm_batch_kv_cache_multi",
        lambda *a, **k: (caches, None),
    )

    batch = vendored_ar.BatchGenerator._build_mixed_prompt_batch(
        fake_self,
        [
            (
                "u1",
                [1, 2, 3, 4],
                10,
                {"inputs_embeds": mx.zeros((1, 4, 4))},
                None,
                None,
            ),
            ("u2", [1, 2, 3], 10, {"inputs_embeds": mx.zeros((1, 3, 4))}, None, None),
        ],
    )
    assert batch is not None
    # The guard declared the right-padding on every cache.
    assert caches[0].prepared == ([0, 1], [2, 1])
    assert caches[1].prepared == ([0, 1], [2, 1])
    # The stripped references are back on the metas for the harvest phase.
    assert [m["apc_blocks"] for m in batch._apc_meta] == [["blk7"], ["blk7"]]
    # Nothing was released on the success path.
    assert manager.released == []


def test_thinking_budget_reset_clears_pending_forced_token():
    """upstream-bugfix: reset_thinking_state() must clear a pending forced
    token captured by the previous generation; upstream left it populated
    so a later pop_forced_token_id() injected a stale forced token into the
    new generation."""

    class _Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return {"\n": [4], "</think>": [9]}[text]

    criteria = vendored_inputs.ThinkingBudgetCriteria(
        _Tokenizer(),
        thinking_budget=1,
        thinking_start_token=None,
        enable_thinking=True,
        prompt_preopens_thinking=True,
    )
    # Budget (1) exceeded after two thinking tokens: the closer is pending.
    criteria(20)
    criteria(21)
    assert criteria.forced_token_id == 4
    # Generation boundary without popping the pending token.
    criteria.reset_thinking_state()
    assert criteria.pop_forced_token_id() is None
    # The next generation starts clean and can still force on its own budget.
    criteria(30)
    criteria(31)
    assert criteria.pop_forced_token_id() == 4
