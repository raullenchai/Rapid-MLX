# SPDX-License-Identifier: Apache-2.0
"""Vendored APC engine and cache primitives from mlx-vlm, pinned at 0.7.1.

The native serialized MLLM lane (``MLLMBatchGenerator``) leans on mlx-vlm
cache primitives. Vendoring this first slice brings those code paths under
this repo's review and tests while the Rapid-owned compatibility seam remains
under the repository's mypy budget. The upstream dependency stays pinned for
everything else it provides (model loading, processors, templating, vision
preprocessing, and the speculative-decode runtime).

Design note: ``docs/engineering/design/2026-09-18-vendor-mllm-primitives.md``.
Upstream: https://github.com/Blaizzy/mlx-vlm/tree/v0.7.1

Provenance contract: every file here is a copy of the upstream
``mlx-vlm==0.7.1`` source except *import redirects* and *in-source
``VENDOR-DEVIATION`` hunks*, each documented in the file's own body and in
the inventory below. A future behavior change must re-qualify against the
pinned tag: ``diff`` against upstream may show only the documented hunks.

Inventory (digests are of the UPSTREAM file, not the vendored bytes — the
vendored copy differs by exactly the deviations listed):

- ``cache.py`` — identical to ``mlx_vlm/models/cache.py`` @ v0.7.1
  (upstream sha256
  ``b736c299bc576f4bdf8d6edf3e1ac6fa3019ca888a2f3cba115d4252f84d5b29``)
  **except three in-source ``VENDOR-DEVIATION(upstream-bugfix)`` hunks**,
  each reproducible against the pinned upstream:
  1. ``BatchRotatingKVCache.merge`` called the zero-arg in-place
     ``_temporal_order`` with an argument, raising TypeError on every
     merge with content (latent upstream; mlx-lm 0.31.3 carries the same
     defect). It also selected the allocation tail of an unrotated,
     preallocated decode cache, merging unused zeros instead of its live
     prefix. Fixed to call the zero-arg form and copy the first ``length``
     temporal entries.
  2. ``BatchPoolingCache.make_mask``'s scalar-offset branch added
     ``offset`` twice to the absolute query positions, admitting pooled
     tokens earlier than the causal contract allows. Fixed to match the
     ``mx.array`` branch semantics.
  3. ``BatchRotatingKVCache.meta_state`` serializes ``rotated`` via
     ``str()`` but the setter read it with ``bool()``, so the string
     ``"False"`` restored as True and corrupted every unrotated cache
     round-tripped through ``from_state`` (the lane's memory-cache restore
     path; mlx-lm 0.31.3 carries the same defect). Fixed to parse the
     serialized spelling.
  A diff against the pinned tag must show only these hunks (plus this
  note in the inventory).

- ``apc_storage.py`` — byte-identical to ``mlx_vlm/apc_storage.py`` @ v0.7.1
  (upstream sha256
  ``e58b3a5aa5fa875764194712e38a7752006aeb31db5abe132057c667759e7010``).
- ``_stream_cleanup.py`` — byte-identical to ``mlx_vlm/_stream_cleanup.py``
  @ v0.7.1 (upstream sha256
  ``00bf5797510f088cfe4cea7798dce0f2902af99a956888748ff0ddc11de21c5d``).
- ``vision_cache.py`` — identical to ``mlx_vlm/vision_cache.py`` @ v0.7.1
  (upstream sha256
  ``5db081a4ef9ee07bb1102c6a87b5fb4a0895821bb2c05d19e110ba6f700e4561``)
  **except in-source ``VENDOR-DEVIATION(upstream-bugfix)`` hunks** covering
  ``_make_key`` (plus the ``import os`` it needs), reproducible against the
  pinned upstream:
  1. Upstream's key derivation was ambiguous and could serve one image
     set's cached features to another: list sources joined the recursively
     derived keys with a bare ``"|"`` (``["a|b", "c"]`` vs
     ``["a", "b|c"]`` collide), and even length-prefixing collides across
     nesting boundaries (``["1:a", "b"]`` vs ``[["a"], "b"]``). Fixed with
     a count-delimited, type-tagged encoding (``s``=str, ``l``=list with a
     child count, ``p``=content hash) that is injective. Upstream also
     documented Path
     sources but only accepted ``str`` (a ``Path`` fell into the
     ``obj:{id}`` fallback); ``os.PathLike`` is now normalized via
     ``os.fsdecode`` — ``os.fspath`` alone can return ``bytes`` for
     byte-valued paths, which fell into the image-content hash branch and
     collided with a raw image payload equal to the path bytes.
  2. Unsupported source types fell back to ``obj:{id(...)}``; Python may
     hand that id to an unrelated object after collection — a silent
     stale-feature hit. Fixed to raise ``TypeError`` (the str/PathLike and
     bytes-like branches cover every real caller; the lane passes
     pre-hashed string keys).
  3. ``tobytes()`` sources were hashed on their raw bytes alone, so two
     images with identical byte sequences but different mode or size (a
     2x1 ``"L"`` vs a 1x1 ``"RGB"``) collided and one image's cached
     features were served for the other. Fixed to hash stable
     type/mode/size metadata together with the raw bytes (bytes-like
     sources carry no such metadata and stay content-addressed). Palette
     images (``"P"``) hash to palette indices in ``tobytes()``, so
     same-sized images with identical indices but different palettes
     rendered different content yet still collided; the effective
     ``getpalette()`` bytes and palette transparency metadata are folded
     into the digest as well.
  4. ``put()`` with ``max_size <= 0`` evaluated
     ``len(self._cache) >= self.max_size`` against an empty mapping and
     called ``popitem()`` on it, raising KeyError. Fixed so zero (or
     negative) ``max_size`` disables storage instead of crashing.
  The lane's ``VisionFeatureCache`` import resolves here.
- ``kv_quant.py`` — identical to ``mlx_vlm/kv_quant.py`` @ v0.7.1 (upstream
  sha256
  ``2936878096435dd2540e7a029b986259e5b7101c972a5be7168495a58e7fbfa3``)
  **except one import redirect**: ``from_legacy()``'s lazy
  ``from .turboquant import ...`` resolves the pinned upstream
  ``mlx_vlm.turboquant`` instead. TurboQuant itself (7k lines with its own
  ``.models.cache`` dependency) is NOT vendored — it can never fit a
  reviewable diff, so it stays a pinned-dependency redirect for the whole
  transition.
- ``apc_coordinator.py`` — identical to ``mlx_vlm/apc_coordinator.py`` @
  v0.7.1 (upstream sha256
  ``8c3939a15b8bee2c4ac8f1144a1048c3463d6cf935537d63eb21403b6f63773b``)
  **except documented redirects**: its module-level
  ``from .apc_adapters import ...`` resolves the vendored sibling; its lazy
  ``from .apc import ...`` engine calls and the ``fresh_cache`` fallback
  ``make_prompt_cache`` remain on upstream mlx-vlm until producers flip
  (step 3). Every site carries a ``VENDOR-DEVIATION`` comment. Additionally,
  return-value locals
  are explicitly annotated where the redirected engine calls type-resolve to
  ``Any`` (the repo's mypy ``no-any-return`` discipline; no behavior
  change).
- ``apc_adapters.py`` — identical to ``mlx_vlm/apc_adapters.py`` @ v0.7.1
  (upstream sha256
  ``9ce11d3c420d983281faf229cfc06ae4a8dce28469dd1536a05d26e4e980c101``)
  **except documented ``VENDOR-DEVIATION`` hunks**, all part of one
  transition mechanism (type-namespace duality — see the design note):
  1. Module-level ``_cache_namespaces`` / ``_cache_namespace_of`` helpers:
     every type table, capability registration, contract probe and
     constructor covers the vendored AND upstream cache namespaces, so the
     adapters behave identically whichever namespace produced a cache. A
     stripped installation without mlx-vlm yields one namespace (no None
     entries); ownership is decided by base-class identity so third-party
     subclasses resolve to their own namespace.
  2. Constructors route through ``_cache_namespace_of`` so cloned/merged
     results keep the producer's cache types (upstream-typed inputs yield
     upstream-typed results — byte-identical behavior today; correct
     typing once producers emit vendored caches in step 3). Bare tuples
     are namespace-agnostic: ``clone_cache_entry`` clones a tuple before
     the owning-namespace lookup (each element resolves its own), and
     ``merge_cache_entries`` derives the container namespace from the
     first tuple element — a stripped install cannot resolve a namespace
     for a tuple itself, which silently dropped composite caches that
     ``apc_exact_eligible`` declares supported. The lazy
     ``_apc_type_tables`` / ``_clone_rules`` builders also publish their
     globals only after both namespaces are processed — as one immutable
     assignment for the two type tables — so a concurrent first caller
     can never observe a partially built (or half-published) table.
  3. Redirects: ``_apc_array_helpers``' lazy ``.apc`` import and the
     ``build_prefix_cache_plan`` fallback ``make_prompt_cache`` remain on
     upstream mlx-vlm until producers flip (step 3); the turboquant
     registration resolves the
     pinned upstream ``mlx_vlm.turboquant`` (not vendored — see above).
  The lane's ``clone_cache_entry`` / ``Capability`` / ``resolve_capability``
  imports now resolve here; the four test modules that stub
  ``clone_cache_entry`` were re-pointed at this module in the same commit.

- ``apc.py`` — identical to ``mlx_vlm/apc.py`` @ v0.7.1 (upstream sha256
  ``5b2b940852f11f34f7b4daf627bc31fc701f8abffc72d40189bc3e5ac57f878c``)
  except documented deviations, each carrying a ``# VENDOR-DEVIATION``
  sentinel:

  - 14 one-line import redirects: the top-level relative imports bind the
    vendored siblings; the lazy ``.models*`` (11) and ``.turboquant`` (3)
    sites resolve upstream. (2b-3 folded the four cache-typed sites — the
    ``_dense_checkpoint*`` pair and both exact-snapshot ladders — into the
    dual-namespace helpers below.)
  - dual-namespace recognition helpers (``_cache_ns_*``, 2b-3) so the
    engine's exact-type tables, snapshot/restore constructors, and
    ``_resolve_checkpoint_class`` accept both cache namespaces; exact
    snapshots record ``_ns="v"`` for vendored-typed entries (absent key =
    upstream, so pre-2b-3 shards stay readable). The step-3 mechanical
    revert drops the helpers.
  - three ``upstream-bugfix`` hunks (2b-3), each repro-tested in
    ``tests/test_mlx_vlm_vendored_apc_engine.py``: ``_rebuild_index``
    disk-bytes accounting (unreadable shards inflated ``_disk_bytes``),
    ``_finish_write`` in-flight entry ownership (partially overlapping
    writers could erase another writer's entry), and
    ``_save_layer_major_shard`` temp-file cleanup on write failure.

- ``inputs.py`` — verbatim from ``mlx_vlm/utils.py`` @ v0.7.1 lines
  1714-2807 (``load_image`` .. ``prepare_inputs`` ..
  ``group_images_by_shape`` .. ``should_add_special_tokens``, an unbroken
  region; the tail past ``prepare_inputs`` was appended in the step-3a
  slice to satisfy ``generate/ar.py``'s helper imports; region sha256
  ``ac610b0e2c157de878b17ec9f5ebaa8bf2c75000e44c09d84b5c17dbaf7c7b5f``),
  except the import block: exactly the names the region references
  (ruff F821 closure), with ``mlx_vlm.models.base`` (``
  BaseImageProcessor``) still resolving upstream and the logger pinned to
  the upstream ``mlx_vlm.utils`` name (``mlx_vlm.apc`` precedent), plus
  four documented hunks (all repro-tested in
  ``tests/test_mlx_vlm_vendored_inputs.py``):

  - ``VENDOR-DEVIATION(dual-namespace)`` in ``processor_video_sampling``:
    a processor hook may return the upstream ``VideoSampling`` dataclass
    while the vendored class is in effect, so matching-shape objects
    normalize by their known fields instead of class identity.
  - ``VENDOR-DEVIATION(upstream-bugfix)`` in ``load_video``: the cv2
    capture handle is released via try/finally (upstream leaked it on
    failed opens, frame-sampler errors, index validation, and read
    failures).
  - ``VENDOR-DEVIATION(upstream-bugfix)`` in ``prepare_inputs``: bytes
    video paths are ``os.fsdecode``-d (upstream str()-ed them into
    ``"b'/tmp/a.mp4'"``-style garbage filenames).
  - ``VENDOR-DEVIATION(upstream-bugfix)`` in ``load_audio``: the streamed
    URL response is closed via ``with`` (upstream left it open).

  The
  lane's ``prepare_inputs`` call sites (``multimodal_processor.py``,
  ``mllm_batch_generator.py``) resolve this module; per-function upstream
  parity is probed in ``tests/test_mlx_vlm_vendored_inputs.py`` (all
  functions except ``processor_video_sampling``, ``load_video``,
  ``load_audio``, and ``prepare_inputs``, which carry the hunks above).
  A diff against the pinned tag must show only the header/import block
  and the documented hunks.

- ``sample_utils.py`` — verbatim from ``mlx_vlm/sample_utils.py`` @ v0.7.1
  (upstream sha256
  ``b3057b6dcaefe5b0a7c50cb96b70baad4334a2f88f70adc04fe7ec2060f7851c``),
  no deviations (mlx + stdlib imports only). Consumed by
  ``generate/ar.py``.

- ``generate/`` — verbatim from ``mlx_vlm/generate`` @ v0.7.1, text AR
  core only (step-3a slice):

  - ``ar.py`` (upstream sha256
    ``3ede5d76b292cdecc0da479a0807d081b1918bdc6dfe7e47755b725299888ac2``)
    except five import redirects:
    ``..models import cache`` → vendored package root;
    ``..prompt_utils`` → pinned upstream (design-doc step-2 boundary);
    ``..speculative.utils`` (both top-level and the lazy
    ``validate_drafter_compatibility``) → pinned upstream until the
    speculative slice lands; ``..turboquant`` → pinned upstream
    (``kv_quant.py`` precedent); ``..utils`` helpers → vendored
    ``inputs.py`` (incl. the lazy ``process_image``).
    It also carries in-source ``VENDOR-DEVIATION(upstream-bugfix)`` hunks,
    each repro-tested in ``tests/test_mlx_vlm_vendored_generate.py``:
    ``_generate_batch`` closes the wired-limit generator in a ``finally``
    and skips ``token=None`` terminal responses;
    ``BatchGenerator._build_mixed_prompt_batch``/``_assemble_mixed_prompt_batch``
    release acquired APC picks on every failed warm assembly and strip the
    block references from the metas handed to the constructor so the
    constructor's prepare-guard cannot double-release them (re-attached on
    success);
    ``BatchGenerator.remove`` releases a cancelled sole-prefill batch's APC
    blocks;
    and the three batched sampling sites (``GenerationBatch._step``,
    ``SpeculativeGenerationBatch._start_rounds``,
    ``PromptProcessingBatch.generate``) pass the per-row int uids as
    ``row_ids`` instead of upstream's ``[0]*n``, so seeded draws stay
    independent across rows sharing a generated position.
  - ``common.py`` (upstream sha256
    ``c69e7e38a09990404d299b0a8d4be55c8220e60652e67a2456e8177633813ba0``)
    except two redirects: ``..models import cache`` → vendored root,
    ``..turboquant`` → pinned upstream.
  - ``types.py`` (upstream sha256
    ``dff487807bedaa3549c39dea02986ada31a45e9e4c27ab447a207e3fff009f6c``)
    — verbatim, no deviations.
  - ``__init__.py`` — **not verbatim**: a reduced-export shim
    (``VENDOR-DEVIATION(subset-exports)``) re-exporting only the vendored
    text-AR surface; the upstream init eagerly imports every modality
    module (dispatch/image/audio/video/diffusion/edit_image), none of
    which is vendored yet.

  Consumers: ``speculative/native_mtp/runtime.py`` and
  ``speculative/native_mtp/transaction.py`` import ``ar`` from this
  package; byte-identical per-function parity is probed in
  ``tests/test_mlx_vlm_vendored_generate.py`` (every top-level symbol
  except the documented-hunk bodies listed in that module). A diff against
  the pinned tag must show only the import-block redirects and the
  documented hunks above.

- ``speculative/`` — verbatim from ``mlx_vlm/speculative`` @ v0.7.1,
  coordinator core only (step-3b slice):
  ``cache_state.py``/``common.py``/``ddtree.py``/``dflash.py``/``mtp.py``/
  ``utils.py`` with three redirects: ``cache_state``'s
  ``..models.cache`` → vendored root; ``mtp``'s
  ``..models.quantized_verifier`` → pinned upstream (2k-line verifier,
  ``decode_quantized_argmax`` is a pure array function);
  ``utils``'s ``.eagle3`` → pinned upstream (eagle3 backend not vendored —
  Rapid serves the dflash and mtp kinds; note the eagle3 round
  coordinators are cache-coupled, not pure-array — they duck-type on the
  passed prompt cache, whose vendored API is upstream-identical plus the
  2a merge bugfix; no Rapid lane dispatches eagle3 through this package,
  and a future one must vendor the coordinator with its cache contract);
  1 documented bugfix hunk: ``run_speculative_server_rounds`` threads the
  server's per-request row ID into singleton dflash positioned sampling. Upstream digests: cache_state
  ``39d35ef0aae0c9298f2fcd3ff6b91b106c6b6c1c3ab6becae216a8fafb2c5300``,
  common ``e3d3c0294a6d6fc915a32e96460370bfe57baef19539c5a290d4254525717cf1``
  (1 documented bugfix hunk:
  ``_speculative_walk_batch_uniform_acceptance`` clamps over rows with a
  positive budget — pinned upstream mins over every row, letting a
  retained finished row collapse the batch to zero acceptance),
  ddtree ``5e3651fe81aad1adee8ab6de7e15bd97d59845cf05724ae26d86af4eb982a342``
  (1 documented bugfix hunk: ``build_ddtree`` validates with ``ValueError``
  instead of ``assert``, which ``python -O`` strips),
  dflash ``39244ec611b38caacd706474722b2997e50954fe48219e26baf9017ade881ad0``
  (2 documented bugfix hunks: continuous-batch compaction shrinks
  ``active_idx`` only when every cache is filterable — pinned upstream
  filters selectively but always shrinks, misaligning mixed cache lists;
  and ``_dflash_rounds`` threads a ``row_id`` into positioned sampling —
  pinned upstream hard-codes row 0, mirroring mtp's existing threading),
  mtp ``4ed467918bd24e26c60730d6d3529c6a9e633448b2fc6028829965bb1f4daad1``
  (1 documented bugfix hunk: ``_mtp_rounds_batch`` budgets the block size
  from unfinished rows only — pinned upstream lets a retained finished row
  force ``bs <= 1`` and terminate the whole batched loop when compaction
  is skipped),
  utils ``93d2ed29ac7b7c378536bf09d22eb570d5abb1f2d338dc550e59a84468b6a9ff``.
  ``__init__.py`` is a reduced shim (``VENDOR-DEVIATION(subset-exports)``):
  the upstream init also re-exports ``load_drafter``; the drafter registry
  lives in this package under ``speculative/drafters/`` (below).

- ``speculative/drafters/`` — the drafter registry and the served drafter
  families, verbatim from ``mlx_vlm/speculative/drafters`` @ v0.7.1 with
  the documented upstream-bugfix deviations below
  (step-3c slice): ``__init__.py`` (registry; upstream digest
  ``7e9fd507dd4ec5aa880d7b0cd04a9f09fbc60ce34e7471b7fbc61863f68a714f``;
  redirects: the ``dspark``/``laguna_dflash``/``muse_glimmer_assistant``
  class imports → pinned upstream (drafter families outside the served
  set, their closures are not vendored), and ``load_drafter``'s lazy
  ``...utils`` → pinned ``mlx_vlm.utils`` (load/get_model_path are step-3e
  scope); upstream-bugfix: ``DRAFTER_KIND_BY_MODEL_TYPE`` gains
  ``dflash2``/``qwen3_dflash`` → ``dflash`` — pinned 0.7.1 omits the served
  DFlash model types, so an explicit wrong ``--draft-kind`` dispatched them
  through the wrong round loop), ``compatibility.py`` (digest
  ``e360a7f03f25da810229ab04f5a68c667cc3831d291c3c22c03e1a0efa0ee2c4``),
  ``mtp_base.py`` (digest
  ``3e071843a4fabca2f7be20c15b04ab1e45ac178d4fa63d7f108684787a2262ab``;
  upstream-bugfix: ``draft_block`` returns the DFlash2-shaped empty
  proposal for ``block_size <= 1`` — pinned 0.7.1 crashes on an empty
  concatenate (reachable through externally supplied drafter repos);
  upstream-bugfix: ``accept_verified_tokens_batch`` raises on mixed
  bonus-token presence BEFORE any cache or position mutation — pinned
  0.7.1 silently skipped every row's replay),
  ``mtp_split.py`` (digest
  ``55afe4b6341ee97d764da90af3d404b0cc86a2a355d198b1d4d299be8040ed2f``;
  upstream-bugfix: ``iter_selected`` resolves index shard paths and rejects
  entries outside the model directory — pinned 0.7.1 joins untrusted
  ``weight_map`` filenames directly; upstream-bugfix: ``split`` validates
  every configuration argument before creating the output directory,
  defaults ``block_size`` only when ``None``, and rejects values below 2
  (the drafting loops crash on block_size 1 with an empty concatenate) —
  pinned 0.7.1 replaced an explicit 0 via ``or`` and accepted negatives;
  upstream-bugfixes: ``split`` stages the checkpoint in a unique sibling
  temporary directory (``tempfile.mkdtemp``, removed in a ``finally``)
  and swaps it into place only after every save and copy succeeds — the
  old destination is preserved as a unique, nonexistent backup path and
  restored if the install rename fails (a symlinked destination moves
  aside cleanly — pinned 0.7.1's pre-created backup directory rejected
  it with ``IsADirectoryError``), ``output == source`` is rejected, and the install runs
  under a per-destination advisory lock so concurrent splits cannot
  interleave destination moves, shard filenames from the safetensors
  index are validated lexically (absolute paths and ``..`` traversal
  rejected) and symlinked shards must resolve inside the model
  directory or the repository's own HF blob cache — pinned 0.7.1
  followed any symlink — and a malformed index (non-object document,
  missing or non-object ``weight_map``, non-string filename entries)
  raises a clear ``ValueError`` — pinned 0.7.1
  writes directly into the destination, so a pre-existing directory
  keeps stale tokenizer files and a mid-way failure pairs new weights
  with an old ``config.json``;
  redirects: the unserved ``deepseek_v4_dspark`` detection import → pinned
  upstream (family not vendored));
  ``glm5_next_mtp/`` (digests ``__init__``
  ``ce16dd3c620b86198ba0a616dc845e3feb424b53f27ff827edf05a88147f4085``,
  ``config``
  ``8cd9c04959ab93441968c18199a94d5bad006ccdb02cf58208bb47d4aed4268e``,
  ``glm5_next_mtp``
  ``6c9a65e7925ceb92b9d3d086abbd814e289c3b92d9f951dda7fae0c71c23ddaf``,
  ``split``
  ``fd44ac09875312a9a3afae793ba7e38671d45555b8d34eb4de4361e3dbfe5f95``;
  redirects: ``models.glm5_next.{config,language}`` and
  ``models.cache`` → pinned upstream, the glm5_next family lands in
  step-3c-2 and ``models/cache.py`` in a later slice);
  ``qwen3_5_mtp/`` (digests ``__init__``
  ``2dac026a94d20fac3247e98a14f823d9a2721cd272ff1df73a2bda93d5213191``,
  ``config``
  ``4555b3973a8dd77fade607b27d79471b7241fa3eb94b4073887a363b0c367409``,
  ``qwen3_5_mtp``
  ``3b2cf5cf0e83393a331fdd95394b812cda749d207a96b9bd4905bc025df85876``,
  ``split``
  ``0cd02dacee282ed6702ab49863f4c300f1acab9e5d2ed3c56a835cad0c2dc128``;
  redirects: ``models.qwen3_5{,_moe}.{config,language}`` and
  ``models.cache`` → pinned upstream, step-3c-3 scope; ``config.py``
  upstream-bugfix: ``TextConfig.from_dict`` routes the Qwen3-Next model
  types to the MoE config — pinned 0.7.1 keyed the decision on "moe" in
  the model type, so Qwen3-Next checkpoints resolved dense decoder
  layers; ``qwen3_5_mtp.py`` upstream-bugfixes: the decoder class routes
  the Qwen3-Next family to ``Qwen3_5MoeDecoderLayer`` (pinned 0.7.1 keyed
  the choice on "moe" in the model type, instantiating dense layers over
  MoE checkpoints), and ``draft_block`` returns the DFlash2-shaped empty
  proposal for ``block_size <= 1`` before consuming seed state — pinned
  0.7.1 crashes on an empty concatenate; ``split.py``
  ``postprocess`` rejects a partially present expert group with the
  missing keys listed instead of silently saving an incomplete
  checkpoint — pinned 0.7.1 skipped the group. The drafter registry also
  installs a Rapid binding hook: ``load_drafter`` pre-registers a
  package-compatible ``sys.modules`` shim for the loaded family (from
  ``glm5_next_mtp``, ``qwen3_5_mtp``, ``qwen3_dflash``, ``dflash2``)
  exposing the vendored package's ``Model``/``ModelConfig`` so the
  pinned ``load_model`` dispatch constructs the vendored classes and
  the runtime fixes reach production drafters; the shim preserves the
  canonical module's exports (``__path__``/``__spec__``) so submodule
  imports keep working, and entries that do not match the vendored
  classes (an earlier pinned import, a pre-swap GLM shim) are re-bound
  so no stale implementation is served; DFlash2 checkpoints that
  declare the backbone model type with a nested ``dflash_config`` are
  bound as ``dflash2`` — pinned 0.7.1 peeked the raw type and skipped
  the shim, constructing the backbone architecture from drafter
  weights; unvendored families fall
  through to the pinned modules. The registry's ``_read_drafter_config``
  degrades a non-object ``config.json`` to the documented empty dict —
  pinned 0.7.1 returns any decoded JSON value and crashes
  ``resolve_drafter_kind`` on ``config.get()``. The DFlash runtime loads
  through the vendored registry (vendored-first, pinned availability
  guard); ``split.py``
  upstream-bugfix: ``Qwen3NextMTPSplitter.postprocess`` stacks per-expert
  ``scales``/``biases`` into the ``switch_mlp`` layout
  alongside the weights — pinned 0.7.1 stacked only weights, so
  quantized Qwen3-Next checkpoints kept per-expert quantization metadata the
  runtime cannot resolve;
  upstream-bugfix: ``accept_verified_tokens_batch`` promotes a scalar
  ``_next_position`` to per-row positions before applying heterogeneous
  replay right-padding — pinned 0.7.1 skips the correction for scalars,
  leaving shorter rows with too-large position ids);
  ``qwen3_dflash/`` (digests ``__init__``
  ``929c03a2169b49c25974f4d292d74b35f3f511acdf459e7e29a0c4bf3083f06b``,
  ``config``
  ``d7ab8dd8742b2232ece0e1240a316e8df5a647f62a6042614cb966fd146ac31e``,
  ``dflash``
  ``4e92910a4f364cccab07c2de1c242177bf62dd849896ba80de32279fc7002cb3``,
  ``parity_check``
  ``1712776c25dbeb045190397e3bc683d68f072ecc81c543457f432a033a68b77f``;
  redirects: ``models.{activations,cache,rope_utils}`` → pinned
  upstream; upstream-bugfix: ``bind`` re-resolves the target embeddings on every
  call — pinned 0.7.1 resolved only when unset, so resetting with a
  different target kept stale embeddings)); ``dflash2/`` (digests ``__init__``
  ``94b557b7ab3de885bbe98ead9ba9e48828330fc0f76a0bb37d7e396f37d72683``,
  ``config``
  ``af864e1190a2eca902adb31cf2ad21e88dc5d2892b3a3d21b2e75b37798930f3``,
  ``dflash2``
  ``19287b0e436c6750ccabfcdfb1388dea4123d6da506adea511ce16269a95e8f0``) —
  verbatim, internal imports only; ``config.py`` upstream-bugfix:
  ``from_dict`` derives ``runtime_block_size`` from the dataclass
  ``block_size`` default when the config omits it — pinned 0.7.1
  indexes ``flat["block_size"]`` and crashes with ``KeyError`` — and
  propagates an inherited ``dflash_config.causal`` to ``is_causal`` so
  causal checkpoints hit the documented rejection — pinned 0.7.1
  dropped the flag and silently served a non-causal drafter. The
  MTP/dflash round-loop fixes from
  the step-3b coordinator slices apply unchanged: the drafters consume
  the vendored ``cache_state``/``common`` via package-relative imports.
  Consumers: ``speculative/native_mtp/runtime.py`` and
  ``speculative/native_mtp/glm5_compat.py`` bind this registry
  (vendored-first, pinned mlx-vlm fallback retained for the transition).

- ``models/`` — verbatim model foundations (step-3b slice): ``base.py``
  (657 lines; upstream digest
  ``4f915923c591faf4b25603b5b5cab4511296a37c7d141553506e6b0760a03a41``;
  redirects: ``..turboquant`` → pinned upstream (kv_quant precedent),
  ``.cache`` → vendored root) and ``linear.py`` (74 lines; upstream digest
  ``148a56193feaf170097ff1c5edccca8165c1b5df878f1605778c4d3642a4d2fa``; 1
  function-level redirect: ``native_batch_linear``'s lazy
  ``.quantized_verifier`` → pinned upstream (verifier not vendored,
  mirrors the ``mtp`` redirect)). ``__init__.py`` is a marker shim
  (upstream's is empty).

- ``fp8.py`` (upstream digest
  ``36ded0f7d5b031fbaaf9a522a6df71e477cc260102092c7f7599862367d70ee7``) —
  verbatim, no deviations (mlx + stdlib only) — and ``quant_utils.py``
  (upstream digest
  ``e323189054be767e0945c29167ad9de7a5992b24c51c0be98c65477260d454af``) —
  verbatim with 2 function-level redirects in ``dequantize_model``:
  ``.models.mla``/``.models.switch_layers`` → pinned upstream (model
  modules not vendored; mlx-nn isinstance dispatch only). Foundations for
  the drafter/model slices.
"""
