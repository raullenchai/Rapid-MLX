# SPDX-License-Identifier: Apache-2.0
"""
Scheduler for rapid-mlx continuous batching.

This module provides a Scheduler class that manages request scheduling
using mlx-lm's BatchGenerator for efficient continuous batching.

The scheduler follows vLLM's design with:
- Waiting queue for pending requests
- Running set for active requests
- Continuous batching via BatchGenerator
"""

import inspect
import logging
import math
import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

# MUST install the MLX hardware-compat shim BEFORE importing mlx_lm.generate.
# mlx_lm/generate.py captures `mx.new_thread_local_stream(mx.default_device())`
# at module-import time; on M5 single-stream GPUs that stream is unusable
# (#404). The shim is idempotent and a no-op on hardware where the original
# API works.
from . import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.generate import BatchGenerator  # noqa: E402
from mlx_lm.sample_utils import make_logits_processors, make_sampler  # noqa: E402
from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer  # noqa: E402

# ...and the batch-slot guard AFTER, because it patches a class that lives
# inside the module imported above. Mixing a request that carries logits
# processors with one that doesn't otherwise puts ``None`` in a per-sequence
# slot that mlx-lm later iterates, killing the engine loop and 503-ing every
# in-flight request (#1525).
_mlx_compat.install_batch_slot_guard()

from ._sampler_fast_path import (  # noqa: E402
    is_fused_top_p_eligible,
    make_fused_top_p_temp_sampler,
)
from ._seeded_sampler import make_seeded_sampler  # noqa: E402
from .kv_estimation import (  # noqa: E402
    _cfg_get,
    _valid_layer_types,
    estimate_kv_footprint,
    rotating_cache_slots,
)


def _read_kv_dims(model):
    """Permissively resolve ``(num_layers, kv_heads, head_dim, struct_cfg)``
    from a loaded model, or ``None``.

    mlx-lm models expose the HF dims on ``.args`` (a ModelArgs dataclass), not
    a ``.config``; multimodal checkpoints nest the text tower under
    ``text_config`` and may carry DECOY vision dims on the outer config. This
    only runs when the scheduler's own config-based terms could not resolve,
    but it still mirrors the admission path's tower selection so it can't pick
    the wrong one: a config whose OWN ``layer_types`` validates against its OWN
    ``num_hidden_layers`` (``_pick_structural_config``'s rule) wins, then the
    nested ``text_config``, then the outer holder (dense: neither validates and
    only the outer carries dims). Returns the triple plus the config the dims
    came from, so ``estimate_kv_footprint`` reads its hybrid structural fields
    from the SAME place.
    """

    def _pos_int(value):
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            return None
        return ivalue if ivalue > 0 else None

    def _dims_of(cfg):
        if cfg is None:
            return None
        layers = _pos_int(_cfg_get(cfg, "num_hidden_layers"))
        kv_heads = _pos_int(_cfg_get(cfg, "num_key_value_heads")) or _pos_int(
            _cfg_get(cfg, "num_attention_heads")
        )
        head_dim = _pos_int(_cfg_get(cfg, "head_dim"))
        if head_dim is None:
            hidden = _pos_int(_cfg_get(cfg, "hidden_size"))
            heads = _pos_int(_cfg_get(cfg, "num_attention_heads"))
            if hidden and heads:
                head_dim = hidden // heads
        if layers and kv_heads and head_dim:
            return layers, kv_heads, head_dim, cfg
        return None

    for holder in (getattr(model, "config", None), getattr(model, "args", None)):
        if holder is None:
            continue
        text_cfg = _cfg_get(holder, "text_config")
        # 1) The structural tower: the config whose own layer_types validates
        #    against its own num_hidden_layers (the text tower for a hybrid).
        for cfg in (holder, text_cfg):
            if cfg is None:
                continue
            n = _pos_int(_cfg_get(cfg, "num_hidden_layers"))
            if n and _valid_layer_types(_cfg_get(cfg, "layer_types"), n):
                dims = _dims_of(cfg)
                if dims is not None:
                    return dims
        # 2) No validating layer_types (dense / non-hybrid multimodal): prefer
        #    the nested text tower over the outer (possibly-decoy) config.
        for cfg in (text_cfg, holder):
            dims = _dims_of(cfg)
            if dims is not None:
                return dims
    return None


from .memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig  # noqa: E402
from .paged_cache import PagedCacheManager
from .pflash import PFlashConfig, compress_request_tokens
from .prefix_cache import BlockAwarePrefixCache, PrefixCacheManager
from .repetition_guard import (
    AgentRepetitionLogitsProcessor,
    detect_repeated_token_suffix,
)
from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .utils.decode import IncrementalDecoder
from .utils.mamba_cache import ensure_mamba_support

logger = logging.getLogger(__name__)

# Functional hybrid cache updates retain their prior lazy graph until an eval
# barrier. Eight steps bounds that graph to a few hundred live Metal handles on
# current 40-layer families while amortizing the synchronization cost.
_RECURRENT_CACHE_MATERIALIZE_INTERVAL = 8
# Handle-budget for the recurrent-state barrier (#1834/#1827). The lazy
# graph retains at most ``active_seqs × interval`` un-materialized updates
# between barriers, so the Metal handle count scales with that product, NOT
# with the interval alone. The flat every-8 barrier holds it at ``B × 8``;
# at batch 8 that is 64 handle-units, a load the engine already sustains in
# production. A single stream pays the barrier's ``mx.eval`` host sync as a
# visible decode-rate tax (≈3% at B=1), so at low batch we widen the interval
# to keep ``active_seqs × interval`` at that same 64-unit steady-state budget
# — far fewer host stalls. A batch-GROWTH boundary (e.g. B=1 depth 63 → B=8)
# can transiently reach ``MAX_INTERVAL + active − 1`` units for the single
# step before the depth-keyed barrier fires; that spike is bounded and cleared
# in one step, and even at its worst (~71 units → a few thousand handles) sits
# ~150× below the 499000-handle ceiling — only UNBOUNDED chains exhaust Metal.
_RECURRENT_MATERIALIZE_HANDLE_BUDGET = 64
_RECURRENT_MATERIALIZE_MAX_INTERVAL = 64


def _pflash_compressed(request: Request) -> bool:
    """Whether PFlash replaced this request's prompt with a compressed
    subsequence. Used to gate every prefix-cache store/fetch site so the
    compressed token sequence — which is positionally non-faithful to
    the original prompt — never enters the shared trie.
    """
    return bool(
        request.pflash_metadata is not None
        and request.pflash_metadata.get("compressed", False)
    )


# Enable MambaCache batching support for models like Nemotron
ensure_mamba_support()

# Error patterns that indicate cache corruption.
# Each pattern must be specific enough to avoid false positives.
# The bare word "cache" was removed because it matched unrelated TypeErrors
# (e.g. "unsupported operand type for cache_size"), masking real bugs and
# triggering unnecessary cache wipes + request reschedules.
CACHE_CORRUPTION_PATTERNS = [
    "'NoneType' object is not subscriptable",
    "BatchKVCache",
    "KVCache",
    "cache is not subscriptable",
    "cache has no attribute",
]


def _assemble_stop_tokens(
    sampling_params: SamplingParams, model_stop_tokens: set[int]
) -> set[int]:
    """Build the stop-token set the BatchGenerator should respect for one request.

    Contract (locked by ``tests/test_community_bench.py::test_scheduler_honours_ignore_eos``):

    - ``sampling_params.ignore_eos=True`` → suppress every token in
      ``model_stop_tokens`` (the model's own EOS + chat-template terminators).
      Matches llama.cpp ``llama-bench --no-eos`` and vLLM upstream semantics.
      Used by community-bench's ``tg128`` / ``tg512`` rounds where the
      contract is "decode exactly N tokens", not "decode until the model
      decides to stop".
    - ``sampling_params.stop_token_ids`` is **always** unioned in.
      Those are *caller intent*, not model intent, so they survive
      ``ignore_eos=True``.
    - ``sampling_params.ignore_eos=False`` (default) → return the union
      of model stop tokens and any caller stop ids. Normal serve / chat
      behaviour.

    Why this is a free function: extracted from ``_create_batch_generator``
    so the test exercises the production assembly directly. A local
    stand-in in the test could pass even if this function deleted the
    ``ignore_eos`` branch.
    """
    stop_tokens: set[int] = (
        set() if sampling_params.ignore_eos else set(model_stop_tokens)
    )
    if sampling_params.stop_token_ids:
        stop_tokens.update(sampling_params.stop_token_ids)
    return stop_tokens


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""

    # Maximum number of concurrent requests in the batch
    max_num_seqs: int = 256
    # BatchGenerator settings
    prefill_batch_size: int = 8
    completion_batch_size: int = 32
    prefill_step_size: int = 2048

    # Prefix cache settings
    enable_prefix_cache: bool = True
    prefix_cache_size: int = 100  # Max cached entries (legacy, ignored if memory-aware)

    # R15-P1 (task #303): radix-tree prefix-cache index. ``"radix"`` (the
    # default) wraps the memory-aware cache in a token trie that gives
    # O(prefix_len) lookups and accounts for cross-request prefix dedup
    # (the headline win on shared-system-prompt multi-tenant workloads:
    # Cursor / Claude-Code-style backends). ``"hash"`` falls back to the
    # legacy bisect-over-sorted-keys path — kept as an escape hatch in
    # case a regression is found in production. Has no effect unless
    # ``use_memory_aware_cache`` is True.
    prefix_cache_index: str = "radix"

    # Memory-aware cache settings (recommended for large models)
    use_memory_aware_cache: bool = True  # Use memory-based eviction
    cache_memory_mb: int | None = None  # None = auto-detect (20% of available RAM)
    cache_memory_percent: float = 0.20  # Fraction of available RAM if auto-detecting

    # KV cache quantization (reduces prefix cache memory). The
    # ``kv_cache_dtype`` field is the canonical R15 #300 knob — it
    # carries the operator-facing dtype string (``bf16`` / ``int8`` /
    # ``int4``) for observability (Prometheus gauge, startup banner).
    # ``kv_cache_quantization`` + ``_bits`` remain the wire-level
    # toggles that drive ``mlx_lm.QuantizedKVCache``; setters in
    # ``vllm_mlx.cli`` resolve dtype → (quantization, bits) via
    # :func:`vllm_mlx.kv_cache_dtype.dtype_to_quantization_bits` so the
    # two stay coherent.
    kv_cache_dtype: str = "bf16"
    kv_cache_quantization: bool = False
    kv_cache_quantization_bits: int = 8
    kv_cache_quantization_group_size: int = 64
    kv_cache_min_quantize_tokens: int = 256

    # TurboQuant KV cache compression (R15 Phase 4).
    #
    # ``kv_cache_turboquant`` is the legacy boolean toggle (PR #157).
    # ``kv_cache_turboquant_mode`` carries the V-only vs K8V4 selection:
    #   * ``"v4"``  — K=FP16, V=3-4bit Lloyd-Max (PR #157).
    #   * ``"k8v4"`` — K=8-bit Walsh-Hadamard, V=4-bit (this PR).
    # The boolean is kept for downstream callers that pre-date the mode
    # field; treat ``kv_cache_turboquant=True`` + mode unset as ``"v4"``.
    kv_cache_turboquant: bool = False
    kv_cache_turboquant_bits: int | None = None  # None = auto-select by head_dim
    kv_cache_turboquant_group_size: int = 32
    kv_cache_turboquant_mode: str = "v4"

    # R15-P1 (task #296): disk-backed KV checkpointing.
    # ``0`` (default) disables the feature so the scheduler hot-path never
    # touches the disk module. Opt-in only: each snapshot serializes the
    # FULL KV cache synchronously on the decode thread, which costs
    # O(context) per boundary — at 16k context on a 4B model one snapshot
    # is ~0.6s, degrading long-context decode by up to 45% (#1853). When
    # enabling, use a multiple of 256 to match MLX-LM's ``KVCache.step``
    # so the on-disk shape lines up with the in-memory shape on reload.
    # The disk cap is resolved at runtime via
    # ``RAPID_MLX_KV_CHECKPOINT_MAX_BYTES`` so a single field on the
    # SchedulerConfig is enough — see
    # :mod:`vllm_mlx.runtime.disk_kv_checkpoint`.
    kv_disk_checkpoint_interval: int = 0

    # Paged cache settings (experimental - for memory efficiency)
    use_paged_cache: bool = (
        False  # Use BlockAwarePrefixCache instead of PrefixCacheManager
    )
    paged_cache_block_size: int = 64  # Tokens per block
    max_cache_blocks: int = 1000  # Maximum number of cache blocks

    # #1103: bounded trim-free prefix reuse for hybrid (GatedDeltaNet /
    # Mamba MoE) models. 0 (default) keeps the #1075 drop-at-store policy;
    # N > 0 retains at most N recurrent-state entries for exact /
    # prefix-extension reuse (LRU-evicted among themselves). Appended AFTER
    # the pre-existing cache fields (codex #1103 NIT-5): keeps every field
    # that predates this PR at its original dataclass position so no future
    # positional ``SchedulerConfig(...)`` construction can silently rebind.
    hybrid_cache_entries: int = 0

    # Speculative decoding selection. "none" is baseline decode; "mtp"
    # installs the vendored mlx-lm PR #990 MTP draft/verify path through
    # the common speculative-config frontend.
    # Validated at SchedulerConfig construction in cli.py.
    spec_decode: str = "none"
    # Deprecated compatibility fields for external callers that constructed
    # SchedulerConfig(enable_mtp=...) before the unified speculative-config
    # migration. __post_init__ translates enable_mtp=True into spec_decode="mtp".
    enable_mtp: bool = False
    mtp_num_draft_tokens: int = 1
    mtp_optimistic: bool = False
    # R15-P1 #313: DFlash drafter HF path override. Empty string is the
    # "no override; defer to the side-registry" sentinel matching the
    # argparse default. When non-empty, the DFlash boot eligibility
    # check uses this path regardless of what the alias-side registry
    # would resolve.
    dflash_drafter_path: str = ""

    # (0.9.13 PR-A ``mtp_sidecar`` / ``mtp_model_type`` fields
    # intentionally live at the END of this dataclass — see the
    # tail comment below ``pflash_config``. Codex round-F BLOCKING
    # #1 flagged that adding them in the middle shifted every
    # subsequent positional argument.)

    # SuffixDecoding — drafter-free speculative decoding using a suffix
    # tree over prompt + generated tokens. Predicts repeated patterns
    # (tool boilerplate, JSON schemas, ReAct loops) at zero drafter
    # cost. Pure-attention only; the architecture allowlist is enforced
    # via ``ModelConfig.supports_spec_decode`` at install time.
    enable_suffix_decoding: bool = False
    suffix_max_draft: int = 8  # Max draft tokens per step (verify cost ∝ this)
    suffix_max_suffix_len: int = 4  # Longest k-gram indexed for matching
    suffix_min_confidence: float = 0.3  # Vote confidence floor before truncating
    # Skip the verify forward when the drafter returned fewer than this
    # many tokens. Single-token drafts are common on free-form chat where
    # the drafter sees a weak match — verify cost dominates the small
    # win. Default 2 keeps chat near regression-floor while still
    # accepting most useful drafts on tool/JSON workloads.
    suffix_min_draft_len: int = 2

    # Admission control: hard cap on concurrent in-flight requests
    # (queued + running). A buggy client (or simple fork bomb) used to
    # be able to OOM the Metal allocator and crash the server for all
    # other clients; ``add_request`` now raises ``BackpressureError``
    # at the cap and routes return 503 with Retry-After. Default 256
    # provides ample queue depth on top of ``max_num_seqs`` — waiting
    # requests only carry their tokenised prompt, not KV cache state,
    # so the memory cost of a queue is small even when ``max_num_seqs``
    # is constrained. Operators who want admission to mirror
    # ``max_num_seqs`` exactly can pass ``--max-concurrent-requests``
    # (codex R7 flagged the gap; the explicit override resolves it
    # without breaking existing tests that intentionally send more
    # requests than ``max_num_seqs`` to exercise the queue).
    max_concurrent_requests: int = 256

    # D-METAL-CAP: GPU memory utilization cap used for admission-time
    # enforcement. ``mx.set_memory_limit`` is documented as a guideline —
    # MLX will quietly grow PAST the limit while system RAM is
    # available, so the user's ``--gpu-memory-utilization 0.45`` request
    # is silently violated on big-RAM hosts (a 256 GB M3 Ultra actually
    # grew Metal active to ~179 GB on a single 32k-prefill before macOS
    # paged). The scheduler therefore re-enforces the cap in Python at
    # admission and at the periodic memory-pressure check. ``0.0``
    # disables the soft check (back-compat default; engines populate
    # this from ``EngineConfig.gpu_memory_utilization`` via
    # ``BatchedEngine``).
    gpu_memory_utilization: float = 0.0
    # D-METAL-PFX: pressure threshold above which the scheduler
    # proactively evicts prefix-cache entries (LRU) to release Metal
    # slabs. Expressed as a fraction of the hard cap. Default 0.9 keeps
    # a 10% safety margin below the cap, wide enough that one large
    # prefill on a half-empty cache will not trigger a thrash loop.
    # The D-METAL-PFX regression repro was tracked in the 0.8-era
    # local TODO, since removed — see git history.
    metal_pressure_evict_fraction: float = 0.9

    # D-METAL-CAP (codex round 3 BLOCKING #1): conservative per-token
    # KV-cache reservation, in bytes per (prompt+output) token. When
    # ``> 0``, ``_enforce_metal_cap_at_admission`` adds
    # ``(num_prompt_tokens + max_tokens) × kv_bytes_per_token`` to the
    # current ``mx.get_active_memory`` reading and compares the SUM
    # to the cap — so a single large prefill that would have grown
    # active PAST the cap is rejected BEFORE the allocation happens,
    # not just after. Without this, admission compares only current
    # active vs cap, which lets a 32k-prefill request slip through
    # when active is currently below cap and then allocate past it.
    # Default ``0`` keeps back-compat (the cheap current-active-only
    # check still runs); operators who want belt-and-suspenders can
    # set a model-tuned value (e.g. 35B-8bit ≈ 1_300_000). Sanity
    # tip: ``num_layers × 2 × hidden_dim × dtype_bytes`` is the
    # per-token KV size for an attention-only model.
    metal_cap_kv_bytes_per_token: int = 0

    # PFlash long-prompt prefill compression (#287). Disabled by default;
    # see vllm_mlx/pflash.py for the design notes and the prefix-cache
    # bypass on compressed requests.
    pflash_config: PFlashConfig = field(default_factory=PFlashConfig)

    # External MTP sidecar path for the Gemma 4 assistant-drafter route
    # (``--speculative-config '{"method":"mtp","model":"<path>"}'``).
    # ``None`` (the default) matches the pre-0.9.13 shape where
    # a missing sidecar only supported Qwen3.5/3.6 native-MTP
    # (i.e. MTP baked into the target checkpoint). When set, the
    # scheduler routes through ``dispatch_mtp_inject(model,
    # model_type, mtp_sidecar=<path>)`` at boot, which grafts the
    # sidecar's ~4-layer drafter onto the target before the
    # server-side MTP hot loop is installed. Accepts either a local
    # safetensors directory or an HF repo id — resolution is deferred
    # to ``dispatch_mtp_inject`` (which itself defers to
    # ``mlx_lm.utils.load`` for HF resolution). See
    # ``vllm_mlx/spec_decode/mtp/detect.py::detect_mtp_eligibility``
    # for how CLI eligibility flips on a non-None value.
    #
    # Codex round-F BLOCKING #1: this field (and ``mtp_model_type``
    # below) live at the very END of the dataclass so no earlier
    # positional argument gets shifted. Adding them in the middle
    # of the field list would silently rebind any positional caller
    # that follows the SuffixDecoding fields.
    mtp_sidecar: str | None = None

    # 0.9.13 PR-A codex round-E blocker #2: CLI-resolved
    # ``config.json::model_type`` for the target model, threaded down
    # from the CLI so the engine's model-load-thread dispatch step
    # does not re-read config.json (which can race with the CLI's
    # asyncio-thread read in offline / gated-cache environments and
    # spuriously report the model_type as unresolvable). ``None`` is
    # the "not yet resolved" sentinel — the engine will fall back to
    # a best-effort HF cache lookup, which preserves pre-0.9.13
    # behaviour for callers who never set this field.
    #
    # When the CLI populates this field, the engine can hard-fail
    # on ANY dispatch mismatch (unresolved / no-inject / rejected)
    # because the CLI has already vetted the config; a soft-fail
    # there would silently downgrade an operator-requested feature.
    mtp_model_type: str | None = None

    # 0.9.13 PR-B: Ollama-style EV depth controller knobs. ``mtp_max_k``
    # is the hard ceiling on the per-round draft depth the controller
    # may select. K=0 parks the drafter; K=1..3 use chained MTP with
    # per-position rollback on hybrid SSM targets.
    # ``mtp_disable_auto_k`` fixes depth at ``mtp_max_k`` for A/B benching.
    mtp_max_k: int = 3
    mtp_disable_auto_k: bool = False

    # Opt-in prompt-deterministic RESPONSE CACHE (exact-match short-circuit).
    # 0 (default) disables it entirely — no store, no lookup, zero behavior
    # change. N > 0 LRU-bounds the cache to N fully-computed deterministic
    # responses so a completely repeated greedy request returns the stored
    # completion verbatim, zero GPU decode. Distinct from the prefix/KV cache
    # (which reuses prefix STATE); this returns the whole stored completion.
    # Consumed at the route layer (``vllm_mlx/response_cache.py`` singleton,
    # configured at serve boot), not by the scheduler hot loop — kept on
    # SchedulerConfig purely so the CLI plumbs it through the same
    # construction sites as every other cache knob.
    #
    # Keep newly-added fields below the historical configuration surface:
    # positional ``SchedulerConfig(...)`` construction binds by position, so
    # appending avoids shifting the position of any existing field.
    response_cache_entries: int = 0

    # Enables the N-1 snapshot strategy for models whose cache state can be
    # continued at an exact prefix but cannot safely trim an exact full-prompt
    # hit by one token. Kept separate from ``hybrid_cache_entries`` because
    # that retention bound is also useful to callers with ordinary caches.
    non_trimmable_exact_prefix_reuse: bool = False

    # DeepSeek V4's checkpoint-native DSpark block drafter. This stays at the
    # end of the dataclass field list to preserve positional compatibility.
    dspark_num_speculative_tokens: int = 5

    # Long-context prefill guard. mlx-lm already processes one bounded prompt
    # chunk per BatchGenerator.next() and materializes cache state between
    # chunks. What it cannot know is how much unified-memory headroom the
    # surrounding server (model + retained prefixes + other requests) has.
    # Appended here to preserve positional SchedulerConfig compatibility.
    adaptive_prefill: bool = True
    adaptive_prefill_min_tokens: int = 32_768
    adaptive_prefill_min_chunk_size: int = 256

    # APPEND-ONLY TAIL: this dataclass is constructed positionally by
    # external callers, so every new field from here onward must stay at
    # the end (see the note above ``mtp_sidecar``).
    #
    # Checkpoint identity for the MTP depth-controller registry, which is
    # process-global and survives a model swap. Without it the key falls
    # back to the model's SHAPE, and two checkpoints that share an
    # architecture, quantization and MTP-head size — different weights,
    # different acceptance profiles — collapse onto one controller, so
    # one model's observations steer the other's depth selection. Also
    # published as the ``model_id`` label on
    # ``rapid_mlx_spec_decode_k_cost_ms``.
    model_name: str | None = None

    # Opt-in dynamic-resolution bounds for MLLM image preprocessing.
    # Appended to preserve positional SchedulerConfig compatibility.
    # Zero preserves the processor/model defaults exactly.
    vision_min_pixels: int = 0
    vision_max_pixels: int = 0

    # Opt-in idle prefix-cache release. ``None`` lets the engine read
    # RAPID_MLX_IDLE_CACHE_CLEAR_SECONDS; 0 explicitly disables it.
    idle_cache_clear_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.vision_min_pixels < 0 or self.vision_max_pixels < 0:
            raise ValueError("vision pixel bounds must be non-negative")
        if (
            self.vision_min_pixels
            and self.vision_max_pixels
            and self.vision_min_pixels > self.vision_max_pixels
        ):
            raise ValueError("vision_min_pixels must not exceed vision_max_pixels")
        if self.response_cache_entries < 0:
            raise ValueError("response_cache_entries must be >= 0")
        if self.idle_cache_clear_seconds is not None and (
            not math.isfinite(self.idle_cache_clear_seconds)
            or self.idle_cache_clear_seconds < 0
        ):
            raise ValueError("idle_cache_clear_seconds must be finite and >= 0")
        if self.enable_mtp:
            import warnings

            if self.spec_decode not in ("none", "mtp"):
                raise ValueError(
                    "SchedulerConfig(enable_mtp=True) conflicts with "
                    f"spec_decode={self.spec_decode!r}; pass only one "
                    "speculative decoding method."
                )
            warnings.warn(
                "SchedulerConfig(enable_mtp=True) is deprecated; pass "
                "SchedulerConfig(spec_decode='mtp') instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.spec_decode = "mtp"
            self.mtp_max_k = max(1, int(self.mtp_num_draft_tokens))

        if self.spec_decode == "suffix":
            # Enable the drafter-free suffix path implicitly (matches the
            # public docs' promise for ``spec_decode='suffix'``), but keep
            # ``spec_decode='suffix'`` as the canonical selector so callers
            # reading the value back observe what they passed in
            # (codex R3: silent rewrite to ``'none'`` was UX drift).
            self.enable_suffix_decoding = True

        if self.spec_decode not in (
            None,
            "none",
            "mtp",
            "dflash",
            "dspark",
            "suffix",
        ):
            raise ValueError(
                f"SchedulerConfig(spec_decode={self.spec_decode!r}) is not "
                "supported; expected one of 'none', 'mtp', 'dflash', "
                "'dspark', or 'suffix'."
            )
        if self.dspark_num_speculative_tokens <= 0:
            raise ValueError("dspark_num_speculative_tokens must be > 0")
        if self.mtp_optimistic and self.spec_decode == "mtp":
            # Unified spec-decode interface (PR #1050) always routes MTP
            # through the vendored ``mtp_generate_step`` hot loop, which
            # does not honour the pre-migration ``mtp_optimistic`` knob.
            # Silently ignoring it was a UX drift (codex R2); hard-reject
            # so callers cannot mistakenly believe optimistic MTP is live.
            raise ValueError(
                "SchedulerConfig(mtp_optimistic=True) is not supported "
                "under the unified spec-decode interface — the vendored "
                "MTP installer does not implement optimistic mode. "
                "Remove the flag."
            )

        active_methods: list[str] = []
        if self.spec_decode not in (None, "none"):
            active_methods.append(str(self.spec_decode))
        if self.enable_suffix_decoding:
            active_methods.append("suffix")
        if len(set(active_methods)) > 1:
            raise ValueError(
                "SchedulerConfig selects multiple speculative decoding "
                f"methods ({', '.join(active_methods)}); pass only one "
                "speculative decoding method."
            )
        if (self.dflash_drafter_path or "").strip() and (
            self.enable_suffix_decoding
            or self.spec_decode not in (None, "none", "dflash")
        ):
            raise ValueError(
                "SchedulerConfig(dflash_drafter_path=...) conflicts with "
                f"spec_decode={self.spec_decode!r}; pass only one "
                "speculative decoding method."
            )

        # PFlashConfig is dataclass(frozen=True), so .validate() returns
        # a new instance; reassign so the SchedulerConfig holds the
        # validated copy. Done in __post_init__ to keep callers from
        # threading .validate() through every construction site.
        self.pflash_config = self.pflash_config.validate()


class BackpressureError(Exception):
    """Raised when admission control rejects a new request.

    Caught by route handlers and converted to HTTP 503 with a
    Retry-After header so well-behaved clients back off and retry.
    Distinguished from ``ValueError`` so the scheduler's narrow
    batch-error catch path doesn't swallow it.
    """


@dataclass
class SchedulerOutput:
    """
    Output from a scheduling step.

    Contains information about what was scheduled and results.
    """

    # Requests scheduled in this step
    scheduled_request_ids: list[str] = field(default_factory=list)
    # Total tokens scheduled
    num_scheduled_tokens: int = 0
    # Requests that finished in this step
    finished_request_ids: set[str] = field(default_factory=set)
    # Request outputs (tokens generated)
    outputs: list[RequestOutput] = field(default_factory=list)
    # Whether any work was done
    has_work: bool = False


def _install_dense_sampler_fastpath(batch_gen: "BatchGenerator") -> None:
    """Swap to mlx-lm's batched sampler fast path when the running batch
    is homogeneous in sampling params.

    mlx-lm's ``GenerationBatch._step`` (``mlx_lm/generate.py:1320``) takes
    a per-row Python loop + ``mx.concatenate`` whenever
    ``any(self.samplers)`` is True. The Scheduler attaches a per-request
    sampler on every ``insert(...)``, so that branch is taken for every
    multi-request batch — bypassing the fast ``fallback_sampler(logprobs)``
    path that runs sampling once on ``[B, vocab]``.

    When every entry in ``self.samplers`` is the same callable instance,
    sampling is mathematically identical to invoking that one callable on
    the full ``[B, vocab]`` matrix (mlx-lm's ``apply_top_p`` /
    ``apply_min_p`` / ``apply_top_k`` / ``categorical_sampling`` all
    operate row-wise along ``axis=-1``). The Scheduler interns samplers
    via ``_get_request_sampler``, so identity-equality of the entries in
    ``self.samplers`` already implies value-equality of the sampling
    params — no separate key check needed.

    Heterogeneous batches (mixed temp/top_p across requests) fall back to
    mlx-lm's original per-row loop — correctness preserved.

    Companion to ``MLLMBatchGenerator._step`` fast path in
    ``mllm_batch_generator.py`` (PR #519). This installs the same shape
    on the dense LLM path that lives inside mlx-lm.
    """
    import types

    gen_batch = getattr(batch_gen, "_generation_batch", None)
    if gen_batch is None or not hasattr(gen_batch, "_step"):
        return

    # ``gen_batch._step`` may already be a bound method (vanilla mlx-lm)
    # OR a plain closure replaced by ``_install_suffix_decoding`` (which
    # writes ``gb._step = _suffix_step`` — see the assignment in that
    # function). Both shapes accept zero args (the closure closes over
    # ``gb``; the bound method already carries ``self``), so calling
    # ``orig_step()`` without args works for either.
    orig_step = gen_batch._step

    def patched_step(self):
        samplers = self.samplers
        # B=1 included (bench-tuning 2026-08-12): a single per-request
        # sampler still forces mlx-lm's per-row branch — one slice, one
        # sampler closure call, one mx.concatenate of a single element —
        # every decode step. The fallback swap below is mathematically
        # identical at any batch size.
        if samplers and len(samplers) >= 1:
            first = samplers[0]
            if first is not None and all(s is first for s in samplers[1:]):
                saved_samplers = self.samplers
                saved_fallback = self.fallback_sampler
                self.samplers = [None] * len(samplers)
                self.fallback_sampler = first
                try:
                    return orig_step()
                finally:
                    self.samplers = saved_samplers
                    self.fallback_sampler = saved_fallback
        return orig_step()

    gen_batch._step = types.MethodType(patched_step, gen_batch)
    logger.info("[dense_sampler_fastpath] installed on BatchGenerator")


def _mtp_controller_key(model_name: str | None, sidecar: str | None) -> str | None:
    """Combine target and drafter identity into one controller key.

    The depth controller learns an acceptance profile, and acceptance is
    a property of the target/drafter PAIR — the same target served with a
    different sidecar head accepts differently. The registry is
    process-global and is never reset in production, so keying on the
    target alone would let the first sidecar's profile drive depth
    selection for a second one after a reload.

    ``None`` when there is no target name, which hands the caller over to
    the shape-derived fallback rather than keying on a bare sidecar path.

    The pair is length-prefixed so the encoding is injective: a bare
    ``"{target}+mtp:{sidecar}"`` join made target ``"a+mtp:b"`` (no sidecar)
    alias target ``"a"`` with sidecar ``"b"`` — two unrelated models sharing
    one controller. Prefixing the target with its length makes the target
    boundary exact, so no two distinct (target, sidecar) inputs — including
    the no-sidecar case — can collide (codex #1441).
    """
    if not model_name:
        return None
    if not sidecar:
        return f"{len(model_name)}:{model_name}"
    return f"{len(model_name)}:{model_name}+mtp:{sidecar}"


def _install_mtp_vendored(
    batch_gen: "BatchGenerator",
    model: Any,
    requests: dict[str, Any] | None = None,
    uid_to_request_id: dict[int, str] | None = None,
    max_k: int = 3,
    disable_auto_k: bool = False,
    controller_key: str | None = None,
) -> bool:
    """Install the vendored PR #990 ``mtp_generate_step`` hot loop into
    ``GenerationBatch._step``.

    This is the SERVER-SIDE wiring for
    ``--speculative-config '{"method":"mtp"}'`` (Gemma 4 external assistant
    + Qwen3.5 baked-in MTP).

    Gate (all required):
      * ``model`` exposes the ``mtp_generate_step`` protocol:
        ``mtp_forward``, ``make_mtp_cache`` (installed by
        :func:`~vllm_mlx.spec_decode.mtp.dispatch.dispatch_mtp_inject`).
      * ``batch_gen._generation_batch`` exists (mlx-lm 0.31+).

    On a gate miss, logs a WARN and returns ``False`` — the request
    continues on plain autoregressive decode.

    Hook shape: replaces ``GenerationBatch._step`` (mlx-lm 0.31+ shape:
    ``() -> (List[int], List[mx.array])``). Per-step, exactly one primary
    token is returned to keep the mlx-lm ``next()`` contract intact.
    Multi-token gains come from the generator's internal batched
    backbone+MTP passes (up to K+1 tokens per pass), not from returning
    multiple tokens per ``_step`` call. Extra tokens produced by the
    generator are queued and drained on the following ``_step`` calls.

    Adaptive chain-of-K scope:

    * Single-request only (``len(gb.uids) == 1``). Multi-request batches
      fall through to ``_orig_step`` — Gemma 4's MTP fast-path is
      batch=1-only (``mtp_forward`` raises on B>1) and the vendored
      generator maintains its own per-request state.

    * Greedy sampling only (temperature == 0). Non-greedy falls through
      to ``_orig_step`` — the byte-lossless verify contract lives in the
      generator's residual-distribution sampling on reject, which the
      MVP does not exercise. Non-greedy support is a follow-up.

    * No logits processors. If any position of ``gb.logits_processors``
      is truthy we fall through — the generator has its own logits-
      processor plumbing but wiring the mlx-lm per-uid processor list
      through to the generator is out of MVP scope.

    * On the very first ``_step`` call we short-circuit and return the
      token that mlx-lm's fresh ``GenerationBatch.__init__._step()``
      already sampled and stashed in ``_next_tokens``. This preserves
      byte-equal output vs. baseline: the FIRST generated token is the
      argmax(prefill-final-logits), identical to plain decode. We seed
      the generator with that same token so its first backbone step
      produces the SECOND generated token.
    """
    gb = getattr(batch_gen, "_generation_batch", None)
    if gb is None:
        logger.warning(
            "[MTP-vendored] disabled: BatchGenerator has no _generation_batch "
            "attribute (mlx-lm version mismatch — expected >=0.31)."
        )
        return False

    def _has_mtp_surface(candidate: Any) -> bool:
        return (
            hasattr(candidate, "mtp_forward")
            and hasattr(candidate, "make_mtp_cache")
            and hasattr(candidate, "mtp")
        )

    mtp_model = model
    if not _has_mtp_surface(mtp_model):
        inner = getattr(model, "language_model", None)
        if inner is not None and _has_mtp_surface(inner):
            mtp_model = inner

    if not _has_mtp_surface(mtp_model):
        logger.warning(
            "[MTP-vendored] disabled: model lacks mtp_forward / make_mtp_cache / "
            "mtp attributes — dispatch_mtp_inject did not run or returned False. "
            "MTP speculative decoding will be a no-op; requests continue on plain "
            "autoregressive decode."
        )
        return False

    # Lazy import — the generator module pulls in mlx-lm's sample_utils and
    # patches ArraysCache; keep the import off the scheduler boot path so a
    # non-MTP build has zero cost.
    from .spec_decode.mtp.draft_k_controller_v2 import derive_controller_key
    from .spec_decode.mtp.generator import mtp_generate_step

    # Derive the structural controller key ONCE at install. It walks the model
    # tree to discover quantization, so recomputing it per generation request
    # (the unnamed-model fallback in ``_mtp_step`` below) would put
    # O(model-size) work on the decode hot path. ``mtp_model`` is fixed for
    # this generator's lifetime, so the key is stable — cache it in the closure
    # and let the per-request path read it (codex #1441 NIT).
    _derived_controller_key = derive_controller_key(mtp_model)

    _orig_step = gb._step

    # Per-uid MTP state. Each entry:
    #   {
    #     "gen": the mtp_generate_step generator instance (or None on FIRST call),
    #     "queue": deque of pending (tok_int, lp_array, from_draft_bool),
    #     "primed": True after we emit the vanilla-sampled first token,
    #     "request_id": the request_id captured at construction time —
    #       codex round-K BLOCKING #1. mlx-lm reuses uid ints when a
    #       request completes; without tracking the owning request
    #       here, a new request that draws the same uid would resume
    #       the OLD generator (built for the old prompt/prompt_cache)
    #       and emit stale tokens from the previous request — a data
    #       corruption bug. On every ``_mtp_step`` call, we compare
    #       ``_state[uid]["request_id"]`` against the current
    #       ``uid_to_request_id[uid]``; on mismatch we treat the state
    #       as stale and reset to the FIRST-call branch.
    #   }
    # Only one uid is ever active at a time under the batch=1 gate.
    _state: dict[int, dict[str, Any]] = {}

    # Codex round-D blocker #2 + round-E blocker #1: permanent-skip
    # map, keyed by uid with the request_id at the time of disabling
    # as the value. Used to:
    #
    # 1. Skip retrying MTP construction on a uid whose first-call
    #    construction failed (round-D — otherwise a bad sidecar or
    #    weight-shape mismatch would DoS the request with one failed
    #    construction attempt per token).
    #
    # 2. Detect uid reuse across requests and re-enable MTP for the
    #    new request. mlx-lm reuses uid ints when a request completes;
    #    keying only by uid (round-D's initial fix) let a bad sidecar
    #    state from request N permanently disable MTP for request
    #    N+1, N+2, … that happened to draw the same uid.
    #    (round-E BLOCKER #1). Storing the request_id lets us
    #    distinguish "same request, still disabled" from "uid was
    #    reused, forget the stale disable."
    #
    # The value can be None (as a placeholder) when the outer install
    # was called with ``uid_to_request_id=None`` — that case is
    # unavoidable and we accept the pre-round-E uid-lifetime scope
    # (this only happens under bench harness callers, where uids are
    # not reused across requests anyway).
    _disabled_uids: dict[int, str | None] = {}

    _stats = {
        "vendored_steps": 0,
        "fallthrough_steps": 0,
        "ft_batch_size": 0,
        "ft_non_greedy": 0,
        "ft_logits_processors": 0,
        "ft_disabled": 0,
        "gen_exhausted": 0,
        "gen_raised": 0,
        # Codex round-L BLOCKING #2-4: track uids that have been
        # handed off from MTP to plain decode mid-stream so subsequent
        # fallback branches can log ONCE per uid rather than once per
        # step. Silent degradation is per Ollama's depth-0 park
        # behavior; but we still want the operator to see the
        # degradation happened, without log spam if the batch stays
        # B>1 (or non-greedy, or has an lp) for many tokens.
        "ft_mid_stream_handoff": 0,
    }

    # Codex round-L BLOCKING #2-4: log-once bookkeeping for mid-stream
    # MTP → plain-decode handoffs. Keyed by (uid, reason) so the same
    # uid can log both "B>1" and "non-greedy" if it hits both, but
    # each reason surfaces at most once per uid lifetime.
    _handoff_logged: set[tuple[int, str]] = set()

    def _log_mtp_mid_stream_handoff_once(uid: int, reason: str, detail: str) -> None:
        """Emit a WARN log for a mid-stream MTP → plain-decode handoff,
        at most once per (uid, reason).

        Codex round-L BLOCKING #2-4: the fallback design matches
        Ollama's depth-0 park behavior — MTP silently degrades to
        plain decode when the current step is incompatible (B>1,
        non-greedy, or has a logits processor) instead of aborting
        the request with a RuntimeError. But the tradeoff is real:
        ``gb._next_tokens`` currently holds the last-MTP-emitted
        token (see ``_sync_next_tokens_after_emit``) rather than a
        fresh baseline sample, so ``_orig_step`` may emit a
        duplicated token or sample from a slightly stale cache
        position for one step before the request continues on plain
        decode. Log the handoff so the operator can correlate the
        potential stream artifact with the load-balancing event.
        """
        key = (uid, reason)
        if key in _handoff_logged:
            return
        _handoff_logged.add(key)
        logger.warning(
            "[MTP-vendored] uid=%s handoff to plain decode (%s): %s. "
            "The MTP generator was closed; the request continues on "
            "baseline mlx-lm _step. gb._next_tokens still holds the "
            "last-MTP-emitted token so the next _orig_step call may "
            "produce a duplicated token or a token sampled from a "
            "slightly stale cache position for one step — a bounded, "
            "known tradeoff for not killing the request "
            "(Ollama-style depth-0 park behavior).",
            uid,
            reason,
            detail,
        )

    def _cleanup_uid(uid: int) -> None:
        # Codex round-G BLOCKING #1: DO NOT clear _disabled_uids here.
        # This helper runs on every fallthrough branch (B>1, non-greedy,
        # logits-processors, mid-stream failure), so unconditionally
        # popping _disabled_uids would silently "un-disable" a uid the
        # very next step. That would re-enable retry of MTP construction
        # (or of the vendored generator) on a request whose earlier
        # ``_mtp_step`` call has already proven the path is broken —
        # a slow-loss loop that codex round-G rightly called out.
        #
        # _disabled_uids has exactly TWO valid clear paths:
        #   1. Reuse detection in the ``uid in _disabled_uids`` gate
        #      inside _mtp_step (the round-E fix): a NEW request_id for
        #      the same uid means mlx-lm reused the uid; clear and let
        #      MTP re-arm for the new request.
        #   2. Never for the current request. The disable is a permanent
        #      marker for the request's lifetime.
        #
        # State (the per-uid MTP generator + queue) is cleaned here as
        # usual — that's per-generator lifecycle, not per-request.
        state = _state.pop(uid, None)
        if state is None:
            return
        gen = state.get("gen")
        if gen is not None:
            try:
                gen.close()
            except Exception:  # noqa: BLE001
                pass

    def _is_greedy_for_uid(uid: int) -> bool:
        """Return True when the request behind ``uid`` sampled at temp=0.

        Matches the greedy contract that
        ``vllm_mlx/spec_decode/mtp/generator.py::mtp_generate_step``
        implements with ``temp=0.0``. Under temp>0, the vendored
        generator can still preserve the lossless marginal via its
        residual-distribution sample on reject — but the MVP install
        hard-codes ``temp=0.0`` into the generator constructor, so any
        request with temperature>0 would silently receive a
        different sampled marginal.

        Codex round-A blocker #1: fail closed on unresolvable metadata.
        Prior revision returned ``True`` when ``uid_to_request_id`` or
        ``requests`` were ``None`` (or the request lookup failed) —
        that would silently apply greedy sampling to a temp>0 request
        whose bookkeeping had just been evicted. Return ``False`` here
        so the caller falls through to ``_orig_step()`` (which reads
        the real sampler from ``gb.samplers[0]``) instead of applying
        the MTP-hardcoded greedy path.

        Codex round-B blocker: also fail closed when ``temperature is
        None``. ``vllm_mlx.request.SamplingParams`` defaults
        ``temperature=0.7`` (not zero) and ``None`` is not a normal
        value — it typically signals "use the server / OpenAI-route
        default," which is likewise nonzero. Treating a bare ``None``
        as greedy would silently apply the MTP-hardcoded ``temp=0.0``
        marginal to a request the operator meant to sample stochast-
        ically. Only an EXPLICIT ``0.0`` passes the gate; every other
        shape falls through to plain decode.
        """
        if uid_to_request_id is None or requests is None:
            return False
        req_id = uid_to_request_id.get(uid)
        req = requests.get(req_id) if req_id else None
        if req is None or getattr(req, "sampling_params", None) is None:
            return False
        temp = getattr(req.sampling_params, "temperature", None)
        return temp == 0.0

    def _mtp_step():
        """Wrapped ``GenerationBatch._step`` for MTP speculative decoding.

        See :func:`_install_mtp_vendored` docstring for the gate matrix
        and MVP caveats.
        """

        # --- Gate matrix ---
        # Batch=1 only. mlx-lm's ``PromptProcessingBatch.generate``
        # constructs a fresh ``GenerationBatch`` with size 1 per request
        # split; the persistent ``_generation_batch`` then extends
        # in-place. Under the smoke script's single-request load this
        # stays at 1 throughout.
        #
        # Codex round-A blocker #3 (initial cleanup requirement)
        # + codex round-H BLOCKING #1-3 (fallthrough safety):
        #
        # When B>1 (or non-greedy / logits-proc appears MID-stream):
        # if MTP has already emitted tokens for the affected uid,
        # falling through to ``_orig_step()`` is UNSAFE. The
        # wrapper never updates ``gb._next_tokens`` — it still
        # holds ``first_gen_tok`` from the priming ``_step`` in
        # ``__init__`` — so ``_orig_step()`` would emit
        # ``first_gen_tok`` AGAIN, duplicating the stream.
        #
        # Two-way split for every fallthrough branch:
        #   * ``_state`` empty for the affected uid → soft-fall-
        #     through to ``_orig_step()``. MTP hasn't primed
        #     anything, ``gb._next_tokens`` is the fresh sample
        #     baseline ``_step`` needs. Also mark the uid as
        #     disabled so subsequent steps in this request skip the
        #     wrapper entirely.
        #   * ``_state`` non-empty for the affected uid → TERMINAL.
        #     Record the disable marker (so any retry short-
        #     circuits) and raise ``RuntimeError``. Recovering to
        #     plain decode would require synthesising
        #     ``gb._next_tokens`` from the last MTP-emitted token,
        #     which we don't stage anywhere.
        def _record_terminal_disable(u: int) -> None:
            """Record a terminal disable marker for uid ``u`` and
            drop any per-generator state. Used on the "MTP already
            emitted, fallthrough is unsafe" path."""
            _term_req_id = None
            if uid_to_request_id is not None:
                _term_req_id = uid_to_request_id.get(u)
            _disabled_uids[u] = _term_req_id
            _state_entry = _state.pop(u, None)
            if _state_entry is not None:
                _gen = _state_entry.get("gen")
                if _gen is not None:
                    try:
                        _gen.close()
                    except Exception:  # noqa: BLE001
                        pass

        def _mark_disabled(u: int) -> None:
            """Mark uid ``u`` as disabled (for pre-MTP soft-fall-
            through paths). No state to clean up because state was
            empty at this branch."""
            _term_req_id = None
            if uid_to_request_id is not None:
                _term_req_id = uid_to_request_id.get(u)
            _disabled_uids[u] = _term_req_id

        def _sync_next_tokens_after_emit(
            gb_ref: Any,
            emitted_tok: int,
            emitted_lp: Any,
        ) -> None:
            """Sync ``gb._next_tokens`` / ``gb._next_logprobs`` shape
            with the token the wrapper just emitted.

            Codex round-I BLOCKING #2 / round-J BLOCKING #2+#3: mlx-lm's
            ``GenerationBatch._step`` contract maintains
            ``_next_tokens`` in a canonical shape so ``.filter(keep)``
            slicing and ``.extend(batch)`` concatenation see a live
            tensor at every step (initialized from ``inputs`` in
            ``__init__``, sliced by ``keep`` on request completion).
            The vendored wrapper's queue-driven emission path never
            touched those fields, leaving them frozen at the
            ``first_gen_tok`` staged by ``__init__``'s priming
            ``_step`` — a rank-1 uint32 that gets increasingly stale
            across the whole request.

            Round-J review: a prior revision drove the MTP generator
            one step ahead (a "prefetch") to publish the NEXT
            to-be-emitted token here, but that changed
            ``gb.prompt_cache`` state behind ``GenerationBatch``'s
            bookkeeping and swallowed generator exceptions (delaying
            the terminal-raise). Both were correctly flagged as
            unsafe.

            Simpler contract that satisfies round-I without the
            round-J side effects: stash the JUST-EMITTED token as the
            placeholder. Shape / dtype match mlx-lm's expected
            ``mx.array([tok], dtype=uint32)`` invariant so
            ``.filter`` / ``.extend`` slicing succeeds; the VALUE is
            semantically stale ("last emitted" rather than "next to
            feed"), but that's tolerated:

            * ``.filter(keep)`` / ``.extend`` don't forward through
              the model — they mutate the tensor in place. No
              downstream cache interaction.
            * Codex round-L BLOCKING #2-4 relaxed the round-H
              terminal-raise contract: the B>1 / non-greedy /
              logits-processor fallthrough branches now delegate to
              ``_orig_step()`` instead of aborting the request. In
              that handoff path, ``_orig_step()`` will read the
              stale ``_next_tokens`` and may emit a duplicated token
              or sample from a slightly stale cache position for one
              step. The wrapper logs a WARN (once per uid+reason)
              on the handoff so the operator can correlate the
              artifact with the load-balancing event. This is the
              accepted tradeoff for not killing the request — see
              :func:`_log_mtp_mid_stream_handoff_once` and the
              round-L rationale comments in the three fallthrough
              branches.

            Cache state stays under the MTP generator's control — the
            wrapper never advances ``prompt_cache`` outside a
            ``next(gen)`` call driven by an actual mlx-lm ``_step``
            request.
            """
            gb_ref._next_tokens = mx.array([int(emitted_tok)], dtype=mx.uint32)
            gb_ref._next_logprobs = [emitted_lp]

        if not gb.uids or len(gb.uids) != 1:
            _stats["fallthrough_steps"] += 1
            _stats["ft_batch_size"] += 1
            # Codex round-L BLOCKING #2: prior round-H revision raised
            # ``RuntimeError`` here when any uid in ``_state`` had in-
            # flight MTP emissions. That killed the request whenever
            # normal continuous-batching load added a second uid to
            # the batch — hostile behavior for a multi-request server
            # where B>1 is the norm, not the exception.
            #
            # Round-L fix: hand off to ``_orig_step`` regardless of
            # whether MTP has emitted. The MTP generator is closed and
            # the affected uid(s) are marked disabled so we don't
            # retry MTP on subsequent steps. The stream may briefly
            # exhibit a duplicated token or a token sampled from a
            # slightly stale cache position (``gb._next_tokens`` still
            # holds the last-MTP-emitted token) — a bounded, known
            # tradeoff that matches Ollama's ``depth=0`` park behavior
            # when speculation cannot proceed. See
            # :func:`_log_mtp_mid_stream_handoff_once` for the operator-
            # facing warning contract.
            if _state:
                terminal_uids = list(_state)
                _stats["ft_mid_stream_handoff"] += len(terminal_uids)
                for stale_uid in terminal_uids:
                    _log_mtp_mid_stream_handoff_once(
                        stale_uid,
                        "b_gt_1",
                        f"batch grew to size {len(gb.uids)}",
                    )
                    _record_terminal_disable(stale_uid)
            return _orig_step()

        uid = gb.uids[0]

        # Codex round-D blocker #2 + round-E blocker #1: honour the
        # permanent-skip map BEFORE re-entering FIRST-call
        # construction, but detect uid reuse across requests. mlx-lm
        # can recycle uid ints once a request completes; without the
        # request-id cross-check a bad sidecar state from a completed
        # request could silently disable MTP for every subsequent
        # request that happened to draw the same uid.
        if uid in _disabled_uids:
            disabled_req_id = _disabled_uids[uid]
            current_req_id = None
            if uid_to_request_id is not None:
                current_req_id = uid_to_request_id.get(uid)
            # Same request: still disabled — skip MTP for the rest of
            # its lifetime.
            #
            # Different request (uid reused): the disable state is
            # stale; drop it and re-enter normal MTP path. The new
            # request may be pointed at a working sidecar even if the
            # previous one wasn't.
            #
            # Missing bookkeeping (both sides None or the map itself
            # is None): can't distinguish. Fall back to the round-D
            # behaviour of honouring the disable — under bench-harness
            # callers uids aren't reused anyway, and treating this as
            # "still disabled" is the safe default.
            if (
                disabled_req_id is not None
                and current_req_id is not None
                and disabled_req_id != current_req_id
            ):
                # uid was reused for a new request — forget the stale
                # disable and fall through to normal MTP path.
                del _disabled_uids[uid]
            else:
                _stats["fallthrough_steps"] += 1
                _stats["ft_disabled"] += 1
                return _orig_step()

        if not _is_greedy_for_uid(uid):
            _stats["fallthrough_steps"] += 1
            _stats["ft_non_greedy"] += 1
            # Codex round-L BLOCKING #3: prior round-H revision raised
            # ``RuntimeError`` here when sampling switched to non-
            # greedy after MTP had already emitted. That killed the
            # request on a legitimate runtime sampling-param change.
            #
            # Round-L fix: hand off to ``_orig_step`` regardless of
            # state. The MTP generator is closed and the uid is
            # marked disabled so subsequent steps skip MTP entirely.
            # Same bounded stream-artifact tradeoff as the B>1 handoff
            # above; see :func:`_log_mtp_mid_stream_handoff_once` for
            # the operator-facing WARN contract.
            if uid in _state:
                _stats["ft_mid_stream_handoff"] += 1
                _log_mtp_mid_stream_handoff_once(
                    uid,
                    "non_greedy",
                    "sampling switched to temperature > 0 mid-stream",
                )
                _record_terminal_disable(uid)
            else:
                _mark_disabled(uid)
            return _orig_step()

        _lp = getattr(gb, "logits_processors", None)
        if _lp and any(p for p in _lp if p):
            _stats["fallthrough_steps"] += 1
            _stats["ft_logits_processors"] += 1
            # Codex round-L BLOCKING #4: prior round-H revision raised
            # ``RuntimeError`` here when a logits processor was added
            # mid-stream after MTP had already emitted. That killed
            # the request whenever an operator toggled a per-request
            # processor (e.g., a guided-decoding grammar) after the
            # first tokens streamed.
            #
            # Round-L fix: hand off to ``_orig_step`` regardless of
            # state. Same handoff pattern as B>1 and non-greedy: log
            # once per uid, mark disabled, delegate.
            if uid in _state:
                _stats["ft_mid_stream_handoff"] += 1
                _log_mtp_mid_stream_handoff_once(
                    uid,
                    "logits_processor",
                    "logits processor appeared mid-stream",
                )
                _record_terminal_disable(uid)
            else:
                _mark_disabled(uid)
            return _orig_step()

        state = _state.get(uid)

        # Codex round-K BLOCKING #1: uid reuse detection for the
        # ACTIVE (non-disabled) state map. mlx-lm reuses uid ints
        # when a request completes and a new one joins the batch.
        # Without this check the wrapper would resume the OLD
        # request's generator (built for a different prompt +
        # prompt_cache state) on the NEW request's next _step call
        # — a data corruption bug because the SUBSEQUENT branch
        # pulls tokens from the stale generator and appends them
        # to gb.tokens[0]. The round-E fix wired this exact
        # detection into ``_disabled_uids``; codex round-K
        # correctly notes the same treatment is missing here.
        #
        # If ``uid_to_request_id`` is not plumbed (bench harness)
        # we can't distinguish reuse from continuation and fall
        # back to the pre-round-K behaviour; this only matters
        # for harnesses that DON'T reuse uids anyway.
        if state is not None and uid_to_request_id is not None:
            stashed_req_id = state.get("request_id")
            current_req_id = uid_to_request_id.get(uid)
            if (
                stashed_req_id is not None
                and current_req_id is not None
                and stashed_req_id != current_req_id
            ):
                # uid was reused for a NEW request. Close the OLD
                # generator + drop the queue, then fall through to
                # FIRST-call construction so the new request gets a
                # fresh MTP path.
                _cleanup_uid(uid)
                state = None

        if state is None:
            # --- FIRST call for this uid ---
            # mlx-lm's fresh ``GenerationBatch.__init__`` ran its
            # ORIGINAL ``_step`` once (before our patch took effect on
            # the persistent gb), which fed ``last_prompt_token``
            # through the model, advanced ``prompt_cache`` by 1
            # position, and stashed the sampled FIRST generated token
            # in ``_next_tokens``. Emit that token now to preserve
            # byte-equality with plain-decode baseline: the argmax at
            # the prompt-end hidden state is deterministic.
            #
            # Then set up the vendored generator seeded with that same
            # token as the "prompt" — the generator's first backbone
            # step feeds it, advances the cache to +1, and samples the
            # SECOND generated token.
            first_tok_arr = gb._next_tokens
            first_lp_list = gb._next_logprobs
            if first_tok_arr is None or not first_lp_list:
                # Shouldn't happen — the fresh __init__ always calls _step.
                # But fall back defensively rather than crashing.
                _stats["fallthrough_steps"] += 1
                return _orig_step()
            first_tok = int(first_tok_arr[0].item())
            first_lp = first_lp_list[0]

            # Compute a generous max_tokens for the generator. Even
            # when the request's max_tokens is small (e.g. 80), the
            # generator uses this as an internal upper bound. Overshoot
            # is fine — mlx-lm's ``next()`` enforces the true max via
            # ``_num_tokens[i] >= self.max_tokens[i]``.
            gen_max = int(gb.max_tokens[0]) if gb.max_tokens else 4096

            # Codex round-A blocker #2: construct the generator BEFORE
            # mutating ``gb.tokens[0]``. Prior revision appended the
            # first token first, then constructed the generator; on
            # construction failure the fallthrough path called
            # ``_orig_step()`` which appended the SAME token again,
            # double-booking bookkeeping and emitting a duplicated
            # token to the stream.
            #
            # Codex round-D blocker #2: on failure here the invariant
            # is that we have NOT yet advanced any state — ``_next_
            # tokens`` still contains ``first_gen_tok`` (staged by
            # the ``GenerationBatch.__init__._step()`` priming call),
            # ``prompt_cache`` is still at position ``prompt_len``,
            # and ``gb.tokens[0]`` still ends at the last prompt
            # token. Delegating to ``_orig_step()`` is byte-equal to
            # plain decode because ``_orig_step`` will read
            # ``_next_tokens = first_gen_tok``, feed it through the
            # target (advancing cache to ``prompt_len+1``), sample
            # ``second_gen_tok``, stage it into ``_next_tokens``, and
            # append ``first_gen_tok`` to ``gb.tokens[0]``. The
            # request-visible first output is ``first_gen_tok``,
            # exactly as it would be under baseline. Mark the uid as
            # permanently disabled so we don't retry construction on
            # every subsequent step.
            try:
                gen = mtp_generate_step(
                    prompt=first_tok_arr.astype(mx.uint32),
                    model=mtp_model,
                    max_tokens=gen_max,
                    prompt_cache=gb.prompt_cache,
                    temp=0.0,
                    # 0.9.13 PR-B: EV depth controller.
                    # Fallback key derived from the model's SHAPE, not
                    # its address. ``SchedulerConfig`` carries no model
                    # name, so this path is the common one, not an edge
                    # case. It has to satisfy two things at once: stable
                    # across restarts (the string is published as the
                    # ``model_id`` label on
                    # ``rapid_mlx_spec_decode_k_cost_ms``, and
                    # ``id(mtp_model)`` would mint a new series every
                    # boot) and distinct between different models (equal
                    # keys share one DepthController, so a collision lets
                    # one model's learned costs drive another's depth).
                    model_id=controller_key or _derived_controller_key,
                    max_k=max_k,
                    disable_auto_k=disable_auto_k,
                    # 0.9.13 PR-C: EOS holdout — feed the
                    # BatchGenerator's assembled stop set to the
                    # controller so positions past EOS are not
                    # logged as (nonexistent) rejections. Emitted
                    # tokens are unchanged; only the acceptance
                    # model's training window shrinks.
                    stop_tokens=getattr(batch_gen, "stop_tokens", None),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[MTP-vendored] mtp_generate_step construction failed "
                    "(%s); disabling MTP for uid=%s and falling back to "
                    "plain decode for the rest of the request. "
                    "_next_tokens is untouched, so the baseline _step "
                    "will correctly emit the first generated token.",
                    e,
                    uid,
                )
                # Codex round-E blocker #1: record the request_id at
                # disable time so uid reuse across requests re-enables
                # MTP for the new request. Store ``None`` if the outer
                # bookkeeping map is None (bench-harness path) — the
                # gate above treats that as "keep disabled" which is
                # the safe default for callers without request IDs.
                _disabled_req_id = None
                if uid_to_request_id is not None:
                    _disabled_req_id = uid_to_request_id.get(uid)
                _disabled_uids[uid] = _disabled_req_id
                _stats["fallthrough_steps"] += 1
                return _orig_step()

            # Generator built successfully — safe to record the first
            # token now. Match the bookkeeping mlx-lm's original _step
            # performs on the ``self.tokens`` list per emitted token.
            gb.tokens[0].append(first_tok)

            # Codex round-K BLOCKING #1: capture the owning
            # request_id so the uid-reuse gate at wrapper entry can
            # detect when mlx-lm reassigns this uid to a different
            # request. ``None`` when ``uid_to_request_id`` isn't
            # plumbed (bench harness); the reuse gate treats
            # ``None`` as "cannot distinguish, keep existing
            # state" — safe because harnesses that don't plumb
            # uid_to_request_id also don't reuse uids.
            _first_call_req_id: str | None = None
            if uid_to_request_id is not None:
                _first_call_req_id = uid_to_request_id.get(uid)

            _state[uid] = {
                "gen": gen,
                "queue": [],
                "primed": True,
                "request_id": _first_call_req_id,
            }
            _stats["vendored_steps"] += 1
            # Codex round-I BLOCKING #2 / round-J BLOCKING #2+#3:
            # keep ``gb._next_tokens`` / ``gb._next_logprobs`` in a
            # coherent shape for downstream ``.filter`` /
            # ``.extend`` slicing. Uses the just-emitted token as
            # the placeholder (round-J review: a prior revision
            # prefetched the next generator token here, which
            # changed ``gb.prompt_cache`` state behind mlx-lm's
            # bookkeeping and swallowed generator exceptions —
            # both correctly flagged as unsafe). See
            # ``_sync_next_tokens_after_emit`` docstring for the
            # full "stale value is safe" argument (short version:
            # round-H terminal-raise fires before any
            # ``_orig_step`` can consume the stale value).
            _sync_next_tokens_after_emit(gb, first_tok, first_lp)
            return [first_tok], [first_lp]

        # --- SUBSEQUENT calls: drain queue, else pull from generator ---
        queue = state["queue"]
        if not queue:
            gen = state["gen"]
            try:
                tok_int, lp_arr, _from_draft = next(gen)
                queue.append((int(tok_int), lp_arr))
            except StopIteration:
                _stats["gen_exhausted"] += 1
                # Codex round-G BLOCKING #2: preserve the terminal
                # disabled marker for the current request BEFORE
                # dropping any state. If mlx-lm somehow re-enters
                # ``_mtp_step`` for this uid after the raise (e.g.,
                # the caller decides to retry on a scheduler tick
                # instead of failing the request), the disabled-uid
                # gate MUST fire; otherwise the wrapper would try
                # to construct a fresh generator and hit the same
                # bug again. Record the request_id so uid reuse
                # for a NEW request still re-enables MTP.
                _terminal_req_id = None
                if uid_to_request_id is not None:
                    _terminal_req_id = uid_to_request_id.get(uid)
                _disabled_uids[uid] = _terminal_req_id
                # Generator's own state can go — nothing to close
                # here (StopIteration means it already tore down).
                _state.pop(uid, None)
                # Codex round-D blocker #3: falling back to
                # ``_orig_step()`` mid-stream is UNSAFE — see the
                # comment on the ``Exception`` branch below.
                # ``StopIteration`` before mlx-lm hits max_tokens is
                # a plumbing bug; the generator's internal
                # ``max_tokens`` should always overshoot the
                # request's ``max_tokens``. Terminating the request
                # is safer than emitting a duplicate token.
                raise RuntimeError(
                    "[MTP-vendored] internal generator exhausted "
                    f"for uid={uid} before mlx-lm hit max_tokens. "
                    "This is a plumbing bug — the generator's "
                    "internal max_tokens should always overshoot "
                    "the request's max_tokens. Failing request to "
                    "avoid duplicate-token stream corruption on "
                    "fallback."
                )
            except Exception as e:  # noqa: BLE001
                # Codex round-D blocker #3: mid-stream generator
                # failure. Baseline ``_orig_step()`` here would
                # dutifully read ``gb._next_tokens`` (STALE — still
                # ``first_gen_tok`` from the priming ``_step``),
                # feed it through the model, and emit
                # ``first_gen_tok`` again — a duplicate. ``gb.tokens
                # [0]`` would also gain a duplicated ``first_gen_tok``
                # entry, corrupting the KV/token log invariant.
                #
                # The safe options are (a) terminate the request or
                # (b) rebuild the baseline state before delegating.
                # (b) is impossible without the next-token — we
                # never staged one — so (a) is the only clean path.
                _stats["gen_raised"] += 1
                logger.exception(
                    "[MTP-vendored] generator raised on uid=%s mid-stream: "
                    "%s. Terminating request — cannot fall back to plain "
                    "decode because gb._next_tokens is stale relative to "
                    "the tokens already emitted by the vendored path.",
                    uid,
                    e,
                )
                # Codex round-G BLOCKING #2: same terminal-marker
                # treatment as the StopIteration branch. Ensures a
                # retry on this uid+request_id hits the disable
                # gate and short-circuits to plain decode instead
                # of re-arming the vendored path.
                _terminal_req_id = None
                if uid_to_request_id is not None:
                    _terminal_req_id = uid_to_request_id.get(uid)
                _disabled_uids[uid] = _terminal_req_id
                # Close the (broken) generator; don't touch
                # _disabled_uids from inside _cleanup_uid.
                _state_entry = _state.pop(uid, None)
                if _state_entry is not None:
                    _gen = _state_entry.get("gen")
                    if _gen is not None:
                        try:
                            _gen.close()
                        except Exception:  # noqa: BLE001
                            pass
                raise RuntimeError(
                    f"[MTP-vendored] uid={uid} generator raised mid-"
                    f"stream ({type(e).__name__}: {e}); cannot fall "
                    "back to plain decode without corrupting the "
                    "output stream. Original exception logged above."
                ) from e

        tok_int, lp_arr = queue.pop(0)
        gb.tokens[0].append(tok_int)
        _stats["vendored_steps"] += 1
        # Codex round-I BLOCKING #2 / round-J BLOCKING #2+#3:
        # mirror the FIRST-call branch — sync ``gb._next_tokens`` /
        # ``gb._next_logprobs`` with the just-emitted token so
        # ``.filter`` / ``.extend`` see a coherent shape. No
        # generator prefetch here — that would advance
        # ``gb.prompt_cache`` behind mlx-lm's bookkeeping (round-J
        # BLOCKING #2) and could swallow generator exceptions
        # (round-J BLOCKING #3). See ``_sync_next_tokens_after_emit``
        # docstring for why the stale placeholder is safe under
        # round-H's terminal-raise regime.
        _sync_next_tokens_after_emit(gb, tok_int, lp_arr)
        return [tok_int], [lp_arr]

    # Patch onto the persistent _generation_batch. New GenerationBatch
    # instances created inside PromptProcessingBatch.generate() use the
    # CLASS _step for their priming call (which is exactly what we want:
    # the first sampled token comes from mlx-lm's plain argmax path so
    # it matches baseline byte-for-byte). The state transfer to the
    # persistent gb happens via .extend(), after which our patched
    # _step takes over.
    gb._step = _mtp_step
    batch_gen._mtp_vendored_stats = _stats

    logger.info(
        "[MTP-vendored] installed on GenerationBatch._step "
        "(single-request greedy adaptive chain-of-K; falls through on B>1 / "
        "non-greedy / logits-processors)."
    )
    return True


def _config_vetted_mtp_supports_spec_decode(model_type: str | None) -> bool:
    """Return True for model types that passed config-driven MTP eligibility.

    Some older alias profiles still carry ``supports_spec_decode=False`` even
    when the checkpoint config advertises a Qwen MTP head. The CLI promotes the
    eligibility gate's model_type into SchedulerConfig only after
    ``detect_mtp_eligibility`` accepts the config; keep the scheduler override
    narrowly tied to the model families this MTP runtime supports.
    """

    return model_type in {"qwen3_5", "qwen3_5_moe", "hy_v3"}


def _replay_dspark_committed(
    model: Any,
    cache_snapshot: list[Any],
    verify_input: mx.array,
    token_count: int,
) -> list[Any]:
    """Replay only committed target tokens into a pre-verify cache snapshot."""

    committed = verify_input[:, :token_count]
    replay_logits = model(committed, cache=cache_snapshot)
    mx.eval(replay_logits, model._last_dspark_hidden)
    return cache_snapshot


def _adapt_dspark_depth(
    current_depth: int,
    max_depth: int,
    accepted: int,
    proposed: int,
    low_accept_streak: int,
) -> tuple[int, int, bool]:
    """Adjust DSpark depth and report whether a baseline cooldown is due."""

    if proposed > 0 and accepted == proposed:
        return min(max_depth, current_depth + 1), 0, False
    if accepted <= 1:
        next_depth = max(1, current_depth - 1)
        next_streak = low_accept_streak + 1
        if next_streak >= 3:
            return 1, 0, True
        return next_depth, next_streak, False
    return current_depth, 0, False


def _dspark_sampling_logprobs(logits: mx.array, params: Any) -> mx.array:
    """Return the exact normalized distribution used by mlx-lm sampling."""
    from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    top_p = float(getattr(params, "top_p", 0.0) or 0.0)
    min_p = float(getattr(params, "min_p", 0.0) or 0.0)
    top_k = int(getattr(params, "top_k", 0) or 0)
    temperature = float(getattr(params, "temperature", 0.0) or 0.0)
    if 0.0 < top_p < 1.0:
        logprobs = apply_top_p(logprobs, top_p)
    if min_p:
        logprobs = apply_min_p(logprobs, min_p)
    if 0 < top_k < logprobs.shape[-1]:
        logprobs = apply_top_k(logprobs, top_k)
    if temperature > 0.0:
        logprobs = logprobs / temperature
    return logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)


def _install_dspark(
    batch_gen: "BatchGenerator",
    model: Any,
    requests: dict[str, Any],
    uid_to_request_id: dict[int, str],
    max_draft: int,
) -> bool:
    """Install DeepSeek V4's checkpoint-native DSpark draft/verify loop.

    V1 is deliberately single-request and greedy. Target verification remains
    authoritative, including request logits processors; finite-precision batch
    evaluation may still choose a different token when target logits are tied.
    """

    if not all(
        hasattr(model, name)
        for name in ("dspark_forward", "make_dspark_cache", "_last_dspark_hidden")
    ):
        logger.warning("[DSpark] disabled: loaded model lacks DSpark runtime hooks")
        return False
    if not getattr(model, "mtp", None):
        logger.warning("[DSpark] disabled: checkpoint did not load DSpark weights")
        return False

    checkpoint_k = int(getattr(model.args, "dspark_block_size", 0))
    if checkpoint_k <= 0:
        logger.warning("[DSpark] disabled: checkpoint has no valid block size")
        return False
    draft_k = min(int(max_draft), checkpoint_k)
    gb = getattr(batch_gen, "_generation_batch", None)
    if gb is None:
        logger.warning("[DSpark] disabled: incompatible mlx-lm BatchGenerator")
        return False

    _orig_step = gb._step
    _orig_next = gb.next
    _caches: dict[int, list[Any]] = {}
    _pending: dict[int, list[tuple[int, mx.array]]] = {}
    # Retain enough state to roll back a verify round if a stop condition is
    # reached part-way through its accepted tokens. DeepSeek V4's pooling
    # caches are not trimmable, so exact replay is the only safe rollback.
    _pending_replay: dict[int, tuple[Any, mx.array]] = {}
    _depths: dict[int, int] = {}
    _cooldowns: dict[int, int] = {}
    _dspark_parameters = inspect.signature(model.dspark_forward).parameters
    _supports_variable_depth = "max_draft_tokens" in _dspark_parameters
    _supports_draft_sampler = "draft_sampler" in _dspark_parameters
    # The 0731 checkpoint predicts five rows, but the target verify kernel has
    # a measured K=5 shape cliff on Apple Silicon: compared with K=4 it costs
    # ~54% more for only ~6% more committed tokens on a coding workload. Keep
    # the checkpoint capability intact while capping the adaptive runtime at
    # the empirically efficient shape. Non-variable third-party hooks retain
    # their original fixed depth.
    runtime_max_k = min(draft_k, 4) if _supports_variable_depth else draft_k
    _stats = {
        "verify_steps": 0,
        "draft_tokens_proposed": 0,
        "tokens_accepted": 0,
        "fallthrough_steps": 0,
        "errors": 0,
        "full_accept_rounds": 0,
    }
    _uid_stats: dict[int, dict[str, Any]] = {}

    def _request_stats(uid: int) -> dict[str, Any]:
        return _uid_stats.setdefault(
            uid,
            {
                "verify_steps": 0,
                "draft_tokens_proposed": 0,
                "tokens_accepted": 0,
                "full_accept_rounds": 0,
                "fallback_events": 0,
                "draft_ms": 0.0,
                "verify_ms": 0.0,
                "sample_ms": 0.0,
                "rollback_ms": 0.0,
                "low_accept_streak": 0,
                "min_depth": runtime_max_k,
                "max_depth": runtime_max_k,
                "depth_stats": {},
            },
        )

    def _sampling_params(uid: int):
        request_id = uid_to_request_id.get(uid)
        request = requests.get(request_id) if request_id else None
        return getattr(request, "sampling_params", None)

    def _dspark_step():
        if (
            gb._next_tokens is None
            or gb._next_tokens.shape[0] != 1
            or len(gb.uids) != 1
        ):
            _stats["fallthrough_steps"] += 1
            return _orig_step()
        uid = gb.uids[0]
        uid_stats = _request_stats(uid)
        cooldown = _cooldowns.get(uid, 0)
        if cooldown > 0:
            _cooldowns[uid] = cooldown - 1
            _stats["fallthrough_steps"] += 1
            return _orig_step()
        params = _sampling_params(uid)
        # Seeded decoding has a stricter baseline-stream reproducibility
        # contract. Keep it on the ordinary path until DSpark owns a separate
        # per-request acceptance key without perturbing the target sampler.
        if params is not None and getattr(params, "seed", None) is not None:
            _stats["fallthrough_steps"] += 1
            return _orig_step()
        processors = getattr(gb, "logits_processors", None)
        request_processors = processors[0] if processors and processors[0] else []
        stochastic = params is not None and params.temperature not in (None, 0.0)
        if stochastic:
            # The residual sampler is mathematically correct for one isolated
            # proposal, but real multi-round Codex runs exposed state/RNG
            # divergence and severe repetition at temperature=1. Keep user
            # output authoritative by taking mlx-lm's ordinary sampler until
            # stochastic DSpark has a per-request RNG stream plus a real-model
            # multi-round distribution-equivalence gate. Greedy DSpark remains
            # enabled and retains its speedup.
            _stats["fallthrough_steps"] += 1
            return _orig_step()

        inputs = gb._next_tokens
        last_token = int(inputs[0].item())
        hidden = getattr(model, "_last_dspark_hidden", None)
        if hidden is None or hidden.shape[0] != 1:
            _stats["fallthrough_steps"] += 1
            return _orig_step()
        dspark_cache = _caches.get(uid)
        if dspark_cache is None:
            take_primed = getattr(model, "take_dspark_primed", None)
            dspark_cache = take_primed(gb.prompt_cache) if take_primed else None
            if dspark_cache is None:
                dspark_cache = model.make_dspark_cache()
            _caches[uid] = dspark_cache
        current_k = _depths.setdefault(uid, runtime_max_k)
        offsets_before = [cache.offset for cache in dspark_cache]
        q_logprobs: list[mx.array] = []
        q_context = None
        if request_processors:
            q_context = mx.concatenate([gb._token_context[0].tokens, inputs])

        def _sample_draft(row_logits, _idx):
            nonlocal q_context
            for processor in request_processors:
                row_logits = processor(q_context, row_logits)
            row_q = _dspark_sampling_logprobs(row_logits, params)
            token = mx.random.categorical(row_q)
            q_logprobs.append(row_q[0])
            if request_processors:
                q_context = mx.concatenate([q_context, token.reshape(-1)])
            return token

        phase_t0 = time.perf_counter()
        try:
            draft_kwargs: dict[str, Any] = {}
            if _supports_variable_depth:
                draft_kwargs["max_draft_tokens"] = current_k
            if _supports_draft_sampler:
                draft_kwargs["draft_sampler"] = _sample_draft if stochastic else None
            proposal = model.dspark_forward(
                inputs[:, None], hidden, dspark_cache, **draft_kwargs
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[DSpark] draft failed; falling back: %r", exc)
            _stats["errors"] += 1
            _caches.pop(uid, None)
            return _orig_step()
        if proposal is None:
            return _orig_step()

        output_ids, _ = proposal
        draft = [int(v) for v in output_ids[0, 1 : current_k + 1].tolist()]
        uid_stats["draft_ms"] = (
            float(uid_stats["draft_ms"]) + (time.perf_counter() - phase_t0) * 1000
        )
        K = len(draft)
        _stats["verify_steps"] += 1
        _stats["draft_tokens_proposed"] += K
        uid_stats["verify_steps"] += 1
        uid_stats["draft_tokens_proposed"] += K
        import copy

        target_cache_snapshot = copy.deepcopy(gb.prompt_cache)
        phase_t0 = time.perf_counter()
        try:
            verify_input = mx.concatenate(
                [inputs[:, None], mx.array([draft], dtype=inputs.dtype)], axis=1
            )
            from .models.deepseek_v4_rollback import armed

            with armed():
                verify_logits = model(verify_input, cache=gb.prompt_cache)
            verify_hidden = model._last_dspark_hidden
            mx.eval(verify_logits, verify_hidden)
            verify_elapsed_ms = (time.perf_counter() - phase_t0) * 1000
            uid_stats["verify_ms"] = float(uid_stats["verify_ms"]) + verify_elapsed_ms
        except Exception as exc:  # noqa: BLE001
            logger.warning("[DSpark] verify failed; falling back: %r", exc)
            _stats["errors"] += 1
            # Verification may have advanced a non-trimmable target cache
            # before raising.  Baseline decoding must resume from the exact
            # pre-verify state, never from speculative state.
            gb.prompt_cache = target_cache_snapshot
            for cache, old_offset in zip(dspark_cache, offsets_before):
                delta = cache.offset - old_offset
                if delta > 0 and cache.is_trimmable():
                    cache.trim(delta)
            _caches.pop(uid, None)
            return _orig_step()

        phase_t0 = time.perf_counter()
        accepted = 0
        preds_list: list[int] = []
        processed_logprobs: list[mx.array] = []
        token_context = None
        if request_processors:
            token_context = gb._token_context[0].update_and_fetch(inputs)
        for idx in range(K + 1):
            sample_logits = verify_logits[0:1, idx]
            for processor in request_processors:
                sample_logits = processor(token_context, sample_logits)
            logprobs = (
                _dspark_sampling_logprobs(sample_logits, params)
                if stochastic
                else sample_logits - mx.logsumexp(sample_logits, axis=-1, keepdims=True)
            )
            if stochastic:
                processed_logprobs.append(logprobs[0])
                if idx < K and request_processors:
                    token_context = mx.concatenate(
                        [token_context, mx.array([draft[idx]], dtype=inputs.dtype)]
                    )
                continue
            else:
                pred = int(mx.argmax(logprobs[0], axis=-1).item())
            preds_list.append(pred)
            processed_logprobs.append(logprobs[0])
            if idx == K or pred != draft[idx]:
                break
            accepted += 1
            if request_processors:
                token_context = gb._token_context[0].update_and_fetch(
                    mx.array([draft[idx]], dtype=inputs.dtype)
                )
        if stochastic:
            p_rows = mx.stack(processed_logprobs[:K])
            q_rows = mx.stack(q_logprobs)
            draft_idx = mx.array(draft, dtype=mx.uint32)[:, None]
            p_token = mx.take_along_axis(mx.exp(p_rows), draft_idx, axis=1).squeeze(1)
            q_token = mx.take_along_axis(mx.exp(q_rows), draft_idx, axis=1).squeeze(1)
            accept_flags = mx.random.uniform(shape=(K,)) < mx.minimum(
                1.0, p_token / q_token
            )
            residual = mx.maximum(mx.exp(p_rows) - mx.exp(q_rows), 0)
            residual_sum = mx.sum(residual, axis=-1, keepdims=True)
            residual = mx.where(
                residual_sum > 0,
                residual / residual_sum,
                mx.exp(p_rows),
            )
            residual_tokens = mx.random.categorical(mx.log(residual))
            bonus_token = mx.random.categorical(processed_logprobs[K])
            resolved = mx.concatenate(
                [
                    accept_flags.astype(mx.int32),
                    residual_tokens.astype(mx.int32),
                    bonus_token.reshape(1).astype(mx.int32),
                ]
            ).tolist()
            flags = resolved[:K]
            accepted = next((idx for idx, flag in enumerate(flags) if not flag), K)
            bonus = resolved[K + accepted] if accepted < K else resolved[-1]
            preds_list = draft[:accepted] + [bonus]
        uid_stats["sample_ms"] = (
            float(uid_stats["sample_ms"]) + (time.perf_counter() - phase_t0) * 1000
        )
        rejected = K - accepted
        if accepted == K:
            _stats["full_accept_rounds"] += 1
            uid_stats["full_accept_rounds"] += 1
        if rejected:
            from .models.deepseek_v4_rollback import trim_all

            phase_t0 = time.perf_counter()
            rollback_error = None
            try:
                rollback_ok = trim_all(gb.prompt_cache, rejected)
            except Exception as exc:  # noqa: BLE001
                rollback_error = exc
                rollback_ok = False
            if not rollback_ok:
                # Compatibility fallback for an unknown/new cache class. It
                # is exact but expensive; supported DeepSeek caches use their
                # short-window undo records and never replay the backbone.
                logger.warning(
                    "[DSpark] cache rollback %s; replaying committed prefix%s",
                    "failed" if rollback_error is not None else "unsupported",
                    f": {rollback_error!r}" if rollback_error is not None else "",
                )
                gb.prompt_cache = target_cache_snapshot
                committed_input = verify_input[:, : accepted + 1]
                replay_logits = model(committed_input, cache=gb.prompt_cache)
                replay_hidden = model._last_dspark_hidden
                mx.eval(replay_logits, replay_hidden)
                verify_hidden = replay_hidden
            uid_stats["rollback_ms"] = (
                float(uid_stats["rollback_ms"])
                + (time.perf_counter() - phase_t0) * 1000
            )

        # Only target states for committed inputs may seed the next proposal.
        model._last_dspark_hidden = verify_hidden[:, : accepted + 1]
        primary_lp = (
            gb._next_logprobs[0]
            if gb._next_logprobs is not None and len(gb._next_logprobs) > 0
            else processed_logprobs[0]
        )
        extras = draft[:accepted]
        extra_lps = processed_logprobs[:accepted]
        bonus = preds_list[accepted]
        bonus_lp = processed_logprobs[accepted]
        gb._next_tokens = mx.array([bonus], dtype=inputs.dtype)
        gb._next_logprobs = [bonus_lp]
        mx.async_eval(gb._next_tokens, bonus_lp)
        gb.tokens[0].append(last_token)
        _pending[uid] = list(zip(extras, extra_lps))
        if extras:
            _pending_replay[uid] = (target_cache_snapshot, verify_input)
        _stats["tokens_accepted"] += accepted
        uid_stats["tokens_accepted"] += accepted
        depth_stats = uid_stats["depth_stats"]
        depth_bucket = depth_stats.setdefault(
            current_k, {"rounds": 0, "accepted": 0, "proposed": 0, "verify_ms": 0.0}
        )
        depth_bucket["rounds"] += 1
        depth_bucket["accepted"] += accepted
        depth_bucket["proposed"] += K
        depth_bucket["verify_ms"] += verify_elapsed_ms
        # oMLX's most important operational lesson is that a fixed speculative
        # depth is wrong for coding agents: acceptance changes sharply across
        # prose, JSON and tool-call boundaries.  Use a conservative AIMD-like
        # controller here. Full accepts grow depth; weak rounds shrink it; three
        # weak rounds take a short baseline cooldown and then probe again at K=1
        # instead of permanently disabling DSpark for the rest of the request.
        current_k, low_streak, needs_cooldown = _adapt_dspark_depth(
            current_k,
            runtime_max_k,
            accepted,
            K,
            int(uid_stats["low_accept_streak"]),
        )
        uid_stats["low_accept_streak"] = low_streak
        _depths[uid] = current_k
        uid_stats["min_depth"] = min(int(uid_stats["min_depth"]), current_k)
        uid_stats["max_depth"] = max(int(uid_stats["max_depth"]), current_k)
        if needs_cooldown:
            _cooldowns[uid] = 16
            uid_stats["fallback_events"] = int(uid_stats["fallback_events"]) + 1
        return [last_token], [primary_lp]

    def _finish_uid(uid: int) -> None:
        request_stats = _uid_stats.pop(uid, {})
        attempts = int(request_stats.get("draft_tokens_proposed", 0))
        accepts = int(request_stats.get("tokens_accepted", 0))
        rounds = int(request_stats.get("verify_steps", 0))
        depth_stats = request_stats.get("depth_stats", {})
        depth_summary = ",".join(
            f"k{k}:{v['rounds']}r/{v['accepted']}a/{v['proposed']}p/{v['verify_ms']:.0f}ms"
            for k, v in sorted(depth_stats.items())
        )
        logger.info(
            "[DSpark] completed uid=%s rounds=%d accepted=%d/%d "
            "avg_accept=%.2f full_accept=%d/%d depth=%d..%d "
            "fallback_events=%d timing[draft=%.1fms verify=%.1fms "
            "sample=%.1fms rollback=%.1fms] by_depth=[%s]",
            uid,
            rounds,
            accepts,
            attempts,
            accepts / rounds if rounds else 0.0,
            int(request_stats.get("full_accept_rounds", 0)),
            rounds,
            int(request_stats.get("min_depth", 0)),
            int(request_stats.get("max_depth", 0)),
            int(request_stats.get("fallback_events", 0)),
            float(request_stats.get("draft_ms", 0.0)),
            float(request_stats.get("verify_ms", 0.0)),
            float(request_stats.get("sample_ms", 0.0)),
            float(request_stats.get("rollback_ms", 0.0)),
            depth_summary,
        )
        _pending.pop(uid, None)
        _pending_replay.pop(uid, None)
        _caches.pop(uid, None)
        _depths.pop(uid, None)
        _cooldowns.pop(uid, None)

    def _dspark_next():
        responses = _orig_next()
        if not responses:
            return responses
        for response in responses:
            if response.finish_reason is not None:
                replay_state = _pending_replay.get(response.uid)
                if replay_state is not None:
                    snapshot, verify_input = replay_state
                    # ``GenerationBatch.next`` has already extracted the
                    # finishing row and filtered it out of the live batch.
                    # Rebuild the response-owned cache with only the primary
                    # token that was actually surfaced, excluding queued
                    # accepted draft tokens.
                    restored = _replay_dspark_committed(
                        model, snapshot, verify_input, 1
                    )
                    response.prompt_cache = [cache.extract(0) for cache in restored]
                _finish_uid(response.uid)
        if not _pending:
            return responses

        augmented = list(responses)
        for response in responses:
            if response.finish_reason is not None:
                continue
            pending = _pending.pop(response.uid, None)
            replay_state = _pending_replay.pop(response.uid, None)
            if not pending:
                continue
            try:
                row = gb.uids.index(response.uid)
            except ValueError:
                continue
            for emit_idx, (token, logprobs) in enumerate(pending):
                gb.tokens[row].append(token)
                gb._num_tokens[row] += 1
                finish_reason = None
                match_sequence = None
                current_state = None
                try:
                    new_state, match_sequence, current_state = gb.state_machines[
                        row
                    ].match(gb._matcher_states[row], token)
                    gb._matcher_states[row] = new_state
                    if match_sequence is not None and current_state is None:
                        finish_reason = "stop"
                except Exception:  # noqa: BLE001
                    pass
                if finish_reason is None and gb._num_tokens[row] >= gb.max_tokens[row]:
                    finish_reason = "length"
                if finish_reason is not None:
                    unused = len(pending) - emit_idx - 1
                    if unused and replay_state is not None:
                        snapshot, verify_input = replay_state
                        gb.prompt_cache = snapshot
                        # Commit the primary plus only the accepted tokens that
                        # have actually been surfaced through this response.
                        committed = verify_input[:, : emit_idx + 2]
                        replay_logits = model(committed, cache=gb.prompt_cache)
                        mx.eval(replay_logits, model._last_dspark_hidden)
                    augmented.append(
                        gb.Response(
                            uid=response.uid,
                            token=token,
                            logprobs=logprobs,
                            finish_reason=finish_reason,
                            current_state=current_state,
                            match_sequence=match_sequence,
                            prompt_cache=gb.extract_cache(row),
                            all_tokens=gb.tokens[row],
                        )
                    )
                    gb.filter([i for i in range(len(gb.uids)) if i != row])
                    _finish_uid(response.uid)
                    break
                augmented.append(
                    gb.Response(
                        uid=response.uid,
                        token=token,
                        logprobs=logprobs,
                        finish_reason=None,
                        current_state=current_state,
                        match_sequence=match_sequence,
                        prompt_cache=None,
                        all_tokens=None,
                    )
                )
        return augmented

    gb._step = _dspark_step
    gb.next = _dspark_next
    batch_gen._dspark_stats = _stats
    logger.info(
        "[DSpark] installed (greedy-only; stochastic requests use plain "
        "decode, checkpoint K=%d, runtime max K=%d)",
        checkpoint_k,
        runtime_max_k,
    )
    return True


def _install_suffix_decoding(
    batch_gen: "BatchGenerator",
    model: Any,
    profile: Any | None,
    max_draft: int,
    max_suffix_len: int,
    min_confidence: float,
    requests: dict[str, Any],
    uid_to_request_id: dict[int, str],
    min_draft_len: int = 2,
) -> None:
    """Monkey-patch BatchGenerator's GenerationBatch to add SuffixDecoding.

    Drafter-free spec-decode: a suffix-tree index over prompt + emitted
    tokens predicts repeated patterns. This is workload-specific, not a
    general accelerator: it can help long high-overlap copy/code-edit/
    repeated tool-XML traffic, and can regress ordinary chat or model
    families whose generated token path does not match the suffix drafts.

    The hot path lives in ``GenerationBatch._step`` (mlx-lm 0.31+):

      1. Drafter builds up to ``max_draft`` candidate tokens.
      2. We run ``model([X, d_0..d_{K-1}])`` of shape (1, K+1).
      3. Greedy compare argmax(logits[i]) vs draft[i]; accept up to
         first mismatch. ``n_accepted ∈ [0, K]``.
      4. Trim trimmable cache layers by ``K - n_accepted``.
      5. Emit ``n_accepted + 1`` new tokens: ``[d_0..d_{n-1}, bonus]``
         where ``bonus = preds[n_accepted]``.

    Wrapped ``GenerationBatch.next()`` augments the single Response
    that ``_step`` returns with ``n_accepted`` extra synthetic Responses
    so the engine sees the full token burst.

    Falls through to ``_orig_step`` when:
      - batch size != 1 (multi-request not handled in v1),
      - sampler is non-greedy (temperature > 0 / top_p < 1 / top_k > 0),
      - logits processors are configured (would need per-position apply),
      - drafter returns empty (low repetition).

    The architecture allowlist is enforced upstream via
    ``ModelConfig.supports_spec_decode``: hybrid linear-attention models
    (Qwen3.5/3.6 GatedDeltaNet, Granite 4 Mamba2) skip install entirely
    because chunked-batched verify isn't numerically equivalent to
    step-update on recurrent layers — see SUFFIX_POC_REPORT.md.
    """
    from .speculative.suffix_counter import get_global_counter as _suffix_counter
    from .speculative.suffix_decoding import SuffixDecodingDrafter

    if profile is not None and not profile.supports_spec_decode:
        logger.warning(
            "[SuffixDecoding] disabled: model is hybrid (linear-attention/"
            "Mamba). Multi-token verify path is not numerically equivalent "
            "to step-update on recurrent layers. See "
            "evals/results/SUFFIX_POC_REPORT.md."
        )
        return

    # mlx-lm 0.31+ moved the actual generation step from BatchGenerator
    # to GenerationBatch. The _generation_batch instance is created once
    # in BatchGenerator.__init__ and is mutated (extend/filter) in place
    # — so a single instance-level patch persists across all sequences.
    gb = getattr(batch_gen, "_generation_batch", None)
    if gb is None:
        logger.warning(
            "[SuffixDecoding] disabled: BatchGenerator has no _generation_batch "
            "attribute (mlx-lm version mismatch — expected ≥0.31)."
        )
        return

    _orig_step = gb._step
    _orig_next = gb.next

    # Per-uid drafter state. Lazy-init on first encounter (we need the
    # request's prompt_token_ids to seed the suffix index).
    _counter = _suffix_counter()
    _drafters: dict[int, SuffixDecodingDrafter] = {}
    # When _step does a verify forward, it stashes the extra emitted
    # tokens here (one entry per accepted draft + bonus). The wrapped
    # ``next()`` then drains the queue, producing one synthetic Response
    # per token so the engine surface stays consistent.
    _pending_emits: dict[int, list[tuple[int, mx.array]]] = {}

    _stats = {
        "verify_steps": 0,
        "fallthrough_steps": 0,
        # Total draft TOKENS proposed across all verify steps (i.e., the
        # sum of K over verify_steps), not the count of verify proposals.
        # Mirrors ``DraftStats.total_draft_tokens_proposed`` naming.
        "draft_tokens_proposed": 0,
        "tokens_accepted": 0,
        "errors": 0,
        # Diagnostic breakdown of WHY we fell through. Sum should equal
        # ``fallthrough_steps``. Useful when debugging "no drafts, no
        # speedup" reports — points at the specific guard.
        "ft_batch_size": 0,
        "ft_uids_size": 0,
        "ft_non_greedy": 0,
        "ft_logits_processors": 0,
        "ft_no_draft": 0,
        "ft_cooldown": 0,
        "ft_non_trimmable_cache": 0,
        # Error fallbacks are fallthroughs too. Without this key the
        # breakdown stops summing to ``fallthrough_steps`` exactly when
        # something is going wrong — which is when the breakdown is being
        # read. (The exported counter already tracks ``ft_error``; this is
        # the local dict catching up.)
        "ft_error": 0,
        # Backoff observability: how many times the skip window re-armed,
        # and the current level (0 = eager). A low-overlap request should
        # show a handful of trips and then go quiet; a high-overlap one
        # should show zero.
        "cooldown_trips": 0,
        "cooldown_level": 0,
        # Current adaptive draft width (see _current_k).
        "k_current": 0,
    }

    # Per-UID drafting state. MUST be keyed by request: the drafter itself
    # already is (``_drafters``), and sharing the window / width / queued
    # history across requests means a request that finishes mid-cooldown
    # hands the next one up to ``_COOLDOWN_MAX`` skipped steps it never
    # earned — and, worse, ``pending`` would flush one request's tokens into
    # another's suffix tree, silently corrupting its drafts.
    #
    #   cooldown : steps left in the current skip window
    #   level    : backoff level, 0 = eager (draft every step)
    #   zeros    : consecutive zero-accept verifies
    #   k        : current adaptive draft width
    #   pending  : tokens emitted while skipping, held as unread mlx arrays
    #
    # Cleared alongside ``_drafters`` when the request finishes.
    _uid_state: dict[int, dict] = {}

    def _reset_state_gauges_if_idle() -> None:
        """Return the state gauges to their at-rest values when no request
        is left holding them.

        ``draft_width`` and ``backoff_level`` describe the CURRENTLY
        drafting request. Once the last ``_uid_state`` entry is reaped they
        describe nothing, and a scrape taken while the server is idle would
        otherwise keep reporting whatever the final request happened to end
        on — a deeply backed-off value looks like a server still in trouble
        long after the traffic that caused it stopped.
        """
        if not _uid_state:
            _counter.set_state(_K_MIN, 0)

    def _reap_uid(uid: int) -> None:
        """Drop every per-uid structure this installer owns.

        One helper so the three exit paths (normal finish, synthetic-emit
        finish, abort) cannot drift apart — the abort path was already
        missing two of the three.
        """
        _pending_emits.pop(uid, None)
        _drafters.pop(uid, None)
        _uid_state.pop(uid, None)

    def _state_for(uid: int) -> dict:
        st = _uid_state.get(uid)
        if st is None:
            st = {
                "cooldown": 0,
                "level": 0,
                "zeros": 0,
                "k": _K_MIN,
                "pending": [],
            }
            _uid_state[uid] = st
            # Publish immediately. A request that only ever takes
            # ``no_draft`` fallthroughs, or ends before its first successful
            # verify, would otherwise leave the gauges showing the PREVIOUS
            # request's width and backoff level.
            _counter.set_state(_K_MIN, 0)
        return st

    # Backoff: each re-trip doubles the skip window, so a request whose
    # traffic has no drafter signal stops paying verify overhead after a
    # handful of probes instead of forever.
    #
    # The fixed-10 cooldown this replaces re-armed on a 3-miss trigger every
    # time, so low-overlap traffic paid 3 wasted verifies per 13 steps —
    # ~23% of steps, and a verify costs ~3.4x a plain forward on M3 Pro.
    # Measured on gemma-4-12b-4bit: accept_ratio 0.044 on free-form
    # generation, 12.6 vs 18.5 tok/s (-32%) with the cooldown "working".
    _COOLDOWN_TRIGGER = 3
    _COOLDOWN_BASE = 10
    # Cap the window so a request that turns high-overlap late (a chat that
    # starts emitting a big code block) still re-probes within a bounded
    # number of tokens rather than never. Kept deliberately small: at one
    # probe per window, a 320-step cap costs ~0.3% of steps, and a deeper
    # cap made recovery too slow — a low-then-high request measured 22.8
    # tok/s against 33.2 for the same work on a fresh drafter, because it
    # could not climb out of the window it had entered.
    _COOLDOWN_MAX = 320
    # Minimum accepted draft tokens for a verify to count as "this traffic
    # has drafter signal" and walk the backoff level down. 1-of-K is noise —
    # crediting it kept low-overlap traffic oscillating back to eager.
    _BACKOFF_DECAY_MIN_ACCEPT = 2

    # Adaptive draft width. The cost of a verify is superlinear in K — on
    # gemma-4-12b-4bit / M3 Pro a 9-wide forward costs 3.44x a 1-wide one —
    # so a FIXED K=8 makes every probe expensive. That is what a short
    # response pays and never amortises: backoff needs 3 misses to engage,
    # 3 x 3.44 ~ 10 forward-equivalents, which on a 74-token answer is 13%
    # before the window ever helps.
    #
    # So ramp instead: probe narrow, widen only once the traffic has proven
    # it accepts. Full acceptance doubles K (2 -> 4 -> 8, reaching the cap
    # within a few verifies on genuinely high-overlap traffic); a partial
    # accept drops K to just above what landed, which is the width that
    # would have been free.
    #
    # The floor has to clear ``min_draft_len``: a draft shorter than that is
    # discarded before it is ever verified, and width only grows AFTER a
    # verify — so starting below it deadlocks at the floor and silently
    # disables suffix decoding for the whole request. Clamped to
    # ``max_draft`` on the other side, since ``num_speculative_tokens``
    # accepts 1 and a floor of 2 would issue two-token drafts against a
    # configured cap of one.
    _K_MIN = min(max(2, min_draft_len), max_draft)

    def _is_greedy_for_uid(uid: int) -> bool:
        """Detect whether the request's sampler is effectively greedy.

        With ``temperature == 0`` mlx-lm short-circuits to argmax, so
        top_p / top_k are no-ops in that regime — we only check the
        temperature. (Defaults of top_p=0.9 / top_k=0 are common and
        don't actually change the sampler when temp=0.)

        Greedy verify only matches the user-requested distribution
        when the actual sampler is greedy; otherwise we fall through to
        keep token-stream stochasticity intact.
        """
        req_id = uid_to_request_id.get(uid)
        req = requests.get(req_id) if req_id else None
        if req is None or req.sampling_params is None:
            return True
        sp = req.sampling_params
        if sp.temperature is None or sp.temperature == 0.0:
            return True
        return False

    def _suffix_step():
        """Wrapped GenerationBatch._step.

        Original signature: ``() -> (List[int], List[mx.array])``.
        We preserve that contract — return the **single** primary token
        (= the input that was just fed through the model) plus its
        logprobs. Additional emitted tokens (accepted drafts + bonus)
        are stashed in ``_pending_emits`` for ``_suffix_next`` to drain.
        """
        # Single-request guard. _next_tokens has shape (B,).
        if gb._next_tokens is None or gb._next_tokens.shape[0] != 1:
            _stats["fallthrough_steps"] += 1
            _stats["ft_batch_size"] += 1
            _counter.record_fallthrough("batch_size")
            return _orig_step()

        if len(gb.uids) != 1:
            _stats["fallthrough_steps"] += 1
            _stats["ft_uids_size"] += 1
            _counter.record_fallthrough("uids_size")
            return _orig_step()

        uid = gb.uids[0]
        if not _is_greedy_for_uid(uid):
            _stats["fallthrough_steps"] += 1
            _stats["ft_non_greedy"] += 1
            _counter.record_fallthrough("non_greedy")
            return _orig_step()

        # Skip when logits_processors are set — applying them at every
        # speculative position would change the math in a way the
        # standalone PoC didn't validate. Defer to a follow-up.
        # Defensive ``getattr``: GenerationBatch grew this attribute in
        # mlx-lm 0.31; older builds would AttributeError here and silently
        # disable the entire suffix-decoding install.
        _lp = getattr(gb, "logits_processors", None)
        if _lp and any(p for p in _lp if p):
            _stats["fallthrough_steps"] += 1
            _stats["ft_logits_processors"] += 1
            _counter.record_fallthrough("logits_processors")
            return _orig_step()

        # Lazy-init drafter on first encounter for this uid.
        drafter = _drafters.get(uid)
        if drafter is None:
            req_id = uid_to_request_id.get(uid)
            req = requests.get(req_id) if req_id else None
            prompt_ids = (
                list(req.prompt_token_ids)
                if req is not None and req.prompt_token_ids
                else []
            )
            drafter = SuffixDecodingDrafter(
                max_draft_tokens=max_draft,
                max_suffix_len=max_suffix_len,
                min_confidence=min_confidence,
            )
            drafter.add_prompt_tokens(prompt_ids)
            # Catch up any tokens already in gb.tokens[0] (rare path —
            # only if suffix decoding were enabled mid-stream).
            try:
                for t in gb.tokens[0]:
                    drafter.add_generated_token(int(t))
            except Exception:  # noqa: BLE001
                pass
            _drafters[uid] = drafter

        st = _state_for(uid)

        # The token we're about to feed (= last step's sampled token).
        # Also the one ``_orig_step`` would return as ``inputs.tolist()``.
        inputs = gb._next_tokens

        # Cooldown check FIRST — before anything that costs.
        #
        # Note what is NOT done here: ``int(inputs[0].item())``. That is a
        # device->host sync, and mlx-lm's step loop is otherwise fully async
        # (``async_eval`` overlaps device work with engine bookkeeping), so
        # forcing a sync on EVERY step stalls the pipeline. It is the fixed
        # per-step cost that kept low-overlap traffic ~13% slow even after
        # the backoff had stopped both the verify forwards and the draft
        # construction — the drafter's own CPU work is negligible by
        # comparison (measured: 0.0006 ms/token for the tree insert,
        # 0.008 ms for a draft, against a ~62 ms forward).
        #
        # Instead the array is queued unread and drained in one batch when
        # the window ends, which is the only point the tree is needed.
        if st["cooldown"] > 0:
            st["cooldown"] -= 1
            st["pending"].append(inputs)
            _stats["fallthrough_steps"] += 1
            _stats["ft_cooldown"] += 1
            _counter.record_fallthrough("cooldown")
            return _orig_step()

        # Coming out of a window (or never in one): bring the suffix tree up
        # to date. One sync for the whole batch rather than one per step.
        if st["pending"]:
            for arr in st["pending"]:
                drafter.add_generated_token(int(arr[0].item()))
            st["pending"].clear()
        last_token = int(inputs[0].item())
        drafter.add_generated_token(last_token)

        # Build draft at the current adaptive width.
        drafter.max_draft_tokens = st["k"]
        try:
            draft = drafter.get_draft()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[SuffixDecoding] drafter error: {e!r}")
            _stats["errors"] += 1
            _stats["fallthrough_steps"] += 1
            _stats["ft_error"] += 1
            _counter.record_error()
            return _orig_step()

        if not draft or len(draft) < min_draft_len:
            # No (or too-short) repetition signal — vanilla step.
            # Short drafts on free-form text would pay verify-forward
            # overhead for almost no acceptance gain (chat regression-
            # floor). Skip them.
            _stats["fallthrough_steps"] += 1
            _stats["ft_no_draft"] += 1
            _counter.record_fallthrough("no_draft")
            return _orig_step()

        # Defense-in-depth: even though ``profile.supports_spec_decode``
        # already gates installation on hybrid arches, verify that EVERY
        # cache layer is trimmable before paying the verify-forward cost.
        # If any layer can't trim and we end up needing to roll back, the
        # cache state would silently diverge — better to fall through.
        for c in gb.prompt_cache:
            if not (
                hasattr(c, "is_trimmable") and c.is_trimmable() and hasattr(c, "trim")
            ):
                _stats["fallthrough_steps"] += 1
                _stats["ft_non_trimmable_cache"] += 1
                _counter.record_fallthrough("non_trimmable_cache")
                return _orig_step()

        K = len(draft)
        _stats["verify_steps"] += 1
        _stats["draft_tokens_proposed"] += K

        # Verify forward: [last_token, d_0..d_{K-1}] of shape (1, K+1).
        try:
            draft_arr = mx.array([draft], dtype=inputs.dtype)
            verify_input = mx.concatenate([inputs[:, None], draft_arr], axis=1)
            verify_logits = gb.model(verify_input, cache=gb.prompt_cache)
            # logits shape (1, K+1, V); greedy verify.
            preds = mx.argmax(verify_logits, axis=-1)
            mx.eval(preds)
            preds_list = preds.tolist()[0]
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[SuffixDecoding] verify forward failed: {e!r}")
            _stats["errors"] += 1
            _stats["fallthrough_steps"] += 1
            _stats["ft_error"] += 1
            # The attempt happened — ``_stats`` counted it above the forward
            # and the exported totals must agree, or a model that fails
            # verification repeatedly shows an attempt rate that quietly
            # understates the work being done.
            _counter.record_verify(K, 0)
            _counter.record_error()
            # Cache was not advanced because the forward raised; safe to
            # retry via vanilla path below.
            return _orig_step()

        # Accept up to first mismatch (greedy).
        n_accepted = 0
        for i in range(K):
            if preds_list[i] == draft[i]:
                n_accepted += 1
            else:
                break

        # Cooldown bookkeeping: track consecutive zero-accept verifies
        # so workloads with weak drafter signal (e.g., free-form chat)
        # automatically stop paying verify overhead.
        #
        # First trip needs ``_COOLDOWN_TRIGGER`` misses so a brief stumble in
        # otherwise-accepting traffic doesn't cost a skip window. Once we
        # HAVE backed off, a single miss re-arms: the previous window already
        # established this traffic has no drafter signal, so waiting for two
        # more misses just buys two more wasted verifies. Each re-trip
        # doubles the window (10, 20, 40 … capped), so a low-overlap request
        # converges to ~no drafting after a handful of probes.
        # Adaptive width update. Full acceptance means the draft was too
        # SHORT — we left tokens on the table — so double. A partial accept
        # means we paid for K but only n landed; retarget just above n.
        if n_accepted >= K:
            st["k"] = min(st["k"] * 2, max_draft)
        else:
            st["k"] = max(_K_MIN, min(n_accepted + 1, max_draft))
        _stats["k_current"] = st["k"]

        if n_accepted == 0:
            st["zeros"] += 1
            trigger = _COOLDOWN_TRIGGER if st["level"] == 0 else 1
            if st["zeros"] >= trigger:
                st["level"] += 1
                st["cooldown"] = min(
                    _COOLDOWN_BASE * (2 ** (st["level"] - 1)),
                    _COOLDOWN_MAX,
                )
                st["zeros"] = 0
                _stats["cooldown_trips"] += 1
                _stats["cooldown_level"] = st["level"]
                _counter.record_cooldown_trip(st["level"])
        else:
            st["zeros"] = 0
            # DECAY one level, don't reset to eager. A single lucky accept in
            # otherwise-signalless traffic must not undo several levels of
            # backoff — with a full reset, low-overlap traffic kept
            # re-arming (measured: 8 trips, level back to 0, still -24%).
            # Sustained acceptance walks the level back down within a few
            # verifies, so a chat that starts emitting a repeated code block
            # still reaches full speed quickly.
            #
            # A 1-of-K accept is noise, not signal: require at least
            # ``_BACKOFF_DECAY_MIN_ACCEPT`` accepted draft tokens before
            # crediting the traffic with having drafter signal.
            #
            # The noise floor has to be checked BEFORE the strong-signal
            # branch, not alongside it. After a back-off the adaptive width
            # is ``_K_MIN`` (2), where a single accepted token satisfies
            # ``n_accepted * 2 >= K`` — so without this guard the one
            # outcome the policy calls noise would reset the whole level,
            # and low-overlap traffic with the occasional 1-of-2 accept
            # would bounce back to eager drafting instead of converging.
            # That is precisely the regression the back-off exists to stop.
            # Clamped to the configured width, not absolute: with
            # ``max_draft=1`` a 1-of-1 accept IS full acceptance, and an
            # absolute floor of 2 would make that configuration unable to
            # ever leave a back-off — every later isolated miss would arm a
            # longer cooldown however well the drafts in between landed.
            _decay_floor = min(_BACKOFF_DECAY_MIN_ACCEPT, K)
            if st["level"] and n_accepted >= _decay_floor:
                if n_accepted * 2 >= K:
                    # STRONG signal — at least half the draft landed. This is
                    # unambiguously high-overlap traffic; go straight back to
                    # eager rather than walking down one level per verify,
                    # which a deep window gives too few chances to do.
                    st["level"] = 0
                else:
                    # Real but weak signal — walk down one level.
                    st["level"] -= 1
                _stats["cooldown_level"] = st["level"]

        # Publish AFTER the backoff transition. Publishing before it left the
        # gauge showing the pre-reset level, permanently so if the request
        # finished on that burst.
        _counter.set_state(st["k"], st["level"])

        n_rejected = K - n_accepted
        if n_rejected > 0:
            # Pre-checked above — every layer here is trimmable.
            for c in gb.prompt_cache:
                c.trim(n_rejected)

        # Token emission accounting.
        #
        # _orig_step emits one token per call: the ``inputs`` it just
        # fed through the model (= what was previously in
        # ``_next_tokens``). The newly-sampled token is stashed in
        # ``_next_tokens`` for the next step.
        #
        # For spec-decode the verify forward consumed K+1 tokens
        # (last_token + K drafts), so we have committed to the cache
        # ``[..., last_token, d_0..d_{n_accepted-1}]`` after trim.
        # Tokens that NEED to surface on the response stream:
        #
        #   - last_token   ← primary, returned by this _step (1 token)
        #   - d_0..d_{n-1} ← accepted drafts (n tokens, drained by
        #                    _suffix_next as synthetic responses)
        #
        # The bonus (= preds[n_accepted], the correction at the
        # rejection point or the post-K bonus) is **NOT** emitted this
        # step — it gets stashed in _next_tokens and surfaces as the
        # primary of the NEXT _step call. Otherwise it would duplicate
        # (see early bug: every-other-token doubling).
        bonus = preds_list[n_accepted]

        full_logprobs = verify_logits - mx.logsumexp(
            verify_logits, axis=-1, keepdims=True
        )
        # The primary's logprobs come from the PREVIOUS step (saved in
        # gb._next_logprobs). Passing them through preserves the same
        # contract as _orig_step.
        primary_logprobs = (
            gb._next_logprobs[0]
            if gb._next_logprobs is not None and len(gb._next_logprobs) > 0
            else full_logprobs[0, 0, :]
        )
        extra_tokens = list(draft[:n_accepted])
        extra_logprobs: list[mx.array] = []
        for i in range(n_accepted):
            # full_logprobs[0, i, :] is the logprobs row that PRODUCED
            # the token at sequence position N+i+1, i.e. d_i.
            extra_logprobs.append(full_logprobs[0, i, :])
        # logprobs row at position n_accepted is the one that produced
        # the bonus — used for the bonus surfacing in the next step.
        bonus_logprobs = full_logprobs[0, n_accepted, :]

        # Drafter history += newly-committed tokens. We add ONLY the
        # accepted drafts here; ``bonus`` will be added on the next
        # ``_suffix_step`` call (line ~1235 ``drafter.add_generated_token
        # (last_token)`` where ``last_token = bonus`` since we just
        # stashed it in ``_next_tokens``). Adding it here too would
        # double-index it in the suffix tree and skew future drafts.
        for tok in extra_tokens:
            drafter.add_generated_token(tok)
        drafter.record_acceptance(n_accepted)
        _stats["tokens_accepted"] += n_accepted
        _counter.record_verify(K, n_accepted)

        # Update gb state for the next _step call. Bonus becomes the
        # next step's primary input. async_eval overlaps device work
        # with engine bookkeeping (matches _orig_step's pattern).
        bonus_arr = mx.array([bonus], dtype=inputs.dtype)
        gb._next_tokens = bonus_arr
        gb._next_logprobs = [bonus_logprobs]
        mx.async_eval(bonus_arr, bonus_logprobs)

        # _step normally appends inputs.tolist()[i] to gb.tokens[i].
        # We do the same for last_token (the primary that we return).
        # The extra tokens get appended in the next() wrapper as each
        # synthetic Response is built, mirroring _orig_step's flow.
        gb.tokens[0].append(last_token)

        # Stash extras for next() to drain.
        _pending_emits[uid] = list(zip(extra_tokens, extra_logprobs))

        return [last_token], [primary_logprobs]

    def _suffix_next():
        """Wrapped GenerationBatch.next.

        Calls ``_orig_next`` (which calls our wrapped ``_step``) for the
        primary Response, then for each pending extra token builds a
        synthetic Response, handling stop-token / max-tokens like the
        original ``next()`` does.
        """
        responses = _orig_next()

        # Drop drafters for finished uids unconditionally — each drafter
        # holds up to ``max_history`` indexed tokens, so a leak here adds
        # up over a long-running server even on workloads that never hit
        # the synthetic-emit path. Run this before the early-return so
        # plain (non-spec-decode) finishes are also reaped.
        if responses:
            for r in responses:
                if r.finish_reason is not None:
                    _reap_uid(r.uid)
            _reset_state_gauges_if_idle()

        if not _pending_emits or not responses:
            return responses

        augmented = list(responses)
        for r in responses:
            uid = r.uid
            if r.finish_reason is not None:
                # Already reaped above — just skip.
                continue

            pending = _pending_emits.pop(uid, None)
            if not pending:
                continue

            # Find this uid's row in gb (post _orig_next, gb may have
            # been filtered if the primary finished — but we already
            # filtered out finished primaries above).
            try:
                row = gb.uids.index(uid)
            except ValueError:
                # Sequence already gone (filtered by _orig_next somehow);
                # bail out for this uid.
                continue

            for emit_idx, (tok, lp) in enumerate(pending):
                # Append to gb.tokens[row] for the synthetic emit; matches
                # the bookkeeping our wrapped _step already does for the
                # primary token (mlx-lm's original _step does NOT append).
                gb.tokens[row].append(tok)
                gb._num_tokens[row] += 1

                # Run the stop-machine on this token to detect stop seqs.
                finish_reason = None
                match_sequence = None
                current_state = None
                try:
                    new_state, match_sequence, current_state = gb.state_machines[
                        row
                    ].match(gb._matcher_states[row], tok)
                    gb._matcher_states[row] = new_state
                    if match_sequence is not None and current_state is None:
                        finish_reason = "stop"
                except Exception:  # noqa: BLE001
                    # If the matcher is in an unexpected state for any
                    # reason, treat the synthetic emit as plain. We'd
                    # rather emit a token than crash the request.
                    pass

                if finish_reason is None and gb._num_tokens[row] >= gb.max_tokens[row]:
                    finish_reason = "length"

                if finish_reason is not None:
                    # Roll back KV cache for any *unconsumed* accepted
                    # drafts. The verify forward in ``_suffix_step``
                    # advanced the cache through ALL ``n_accepted``
                    # drafts; if we stop early at ``emit_idx``, the
                    # remaining ``len(pending) - emit_idx - 1`` drafts
                    # were never surfaced — their KV state must come
                    # back out of the cache or it'll poison prefix-cache
                    # reuse for the next request that hits this prefix.
                    unused = len(pending) - emit_idx - 1
                    if unused > 0:
                        for c in gb.prompt_cache:
                            if (
                                hasattr(c, "is_trimmable")
                                and c.is_trimmable()
                                and hasattr(c, "trim")
                            ):
                                c.trim(unused)
                    augmented.append(
                        gb.Response(
                            uid=uid,
                            token=tok,
                            logprobs=lp,
                            finish_reason=finish_reason,
                            current_state=current_state,
                            match_sequence=match_sequence,
                            prompt_cache=gb.extract_cache(row),
                            all_tokens=gb.tokens[row],
                        )
                    )
                    # Filter the finished sequence out of gb.
                    keep = [i for i in range(len(gb.uids)) if i != row]
                    if keep:
                        gb.filter(keep)
                    else:
                        # Cleared the only sequence; reset the batch.
                        gb.filter([])
                    # Drop the drafter — sequence is done, its history
                    # would otherwise live in _drafters until the
                    # BatchGenerator itself is replaced.
                    _reap_uid(uid)
                    _reset_state_gauges_if_idle()
                    # No more pending to emit for this uid.
                    break

                augmented.append(
                    gb.Response(
                        uid=uid,
                        token=tok,
                        logprobs=lp,
                        finish_reason=None,
                        current_state=current_state,
                        match_sequence=match_sequence,
                        prompt_cache=None,
                        all_tokens=None,
                    )
                )

        return augmented

    _orig_remove = getattr(batch_gen, "remove", None)

    def _departed(uid_list: list[int]) -> list[int]:
        """Which of ``uid_list`` the generator no longer holds.

        Only used on the ``remove`` failure path. When membership cannot
        be established, report none: a leaked drafter is recoverable (the
        BatchGenerator is rebuilt on the next model load), silently
        discarding a live request's queued emits is not.
        """
        finder = getattr(batch_gen, "_find_uids", None)
        if finder is None:
            return []
        try:
            still_present = set(finder(uid_list))
        except Exception:  # noqa: BLE001
            return []
        return [uid for uid in uid_list if uid not in still_present]

    def _suffix_remove(uids, return_prompt_caches=False):
        """Wrapped ``BatchGenerator.remove`` — the abort path's reap hook.

        Every ordinary exit runs through ``next()``, which reaps on
        ``finish_reason``. An abort does not: ``Scheduler._do_abort_request``
        calls ``remove()`` directly and the uid never produces a Response,
        so the finish sweep never sees it. Without this, each client
        cancellation leaks that uid's drafter (up to the suffix index's
        full token history) plus its ``_uid_state`` entry — whose
        ``pending`` list can be holding as many as ``_COOLDOWN_MAX``
        unread mlx arrays, i.e. live GPU buffers, on exactly the path a
        flaky or impatient client hits over and over.

        Deliberately NOT reaping in ``finally``. ``remove`` can raise
        before the uid leaves the batch — with ``return_prompt_caches`` it
        calls ``extract_cache`` first, and the index bookkeeping can fail
        part-way through a multi-uid removal. Reaping a still-live uid is
        worse than the leak this wrapper exists to fix: its ``pending``
        entries are accepted draft tokens whose KV is already committed to
        the cache, so dropping them loses output that was going to be
        emitted and leaves the rebuilt drafter history disagreeing with
        the cache. On failure we reap only what actually left.
        """
        uid_list = list(uids)
        try:
            result = _orig_remove(uid_list, return_prompt_caches=return_prompt_caches)
        except Exception:
            for uid in _departed(uid_list):
                _reap_uid(uid)
            _reset_state_gauges_if_idle()
            raise
        for uid in uid_list:
            _reap_uid(uid)
        _reset_state_gauges_if_idle()
        return result

    gb._step = _suffix_step
    gb.next = _suffix_next
    if _orig_remove is not None:
        batch_gen.remove = _suffix_remove
    else:
        # Not fatal: without ``remove`` there is no abort path to leak
        # through. Logged rather than silent so a real version skew shows
        # up as a named gap instead of as unexplained growth.
        logger.warning(
            "[SuffixDecoding] BatchGenerator has no remove(); abort-path "
            "state reaping is not installed (mlx-lm version mismatch?)."
        )
    # Telemetry attached to the BatchGenerator (where the rest of the
    # engine looks for it) and to gb for direct inspection.
    batch_gen._suffix_stats = _stats
    gb._suffix_stats = _stats
    # Expose the per-uid drafter dict for tests to assert lifecycle
    # cleanup. Production code should not mutate this directly.
    gb._suffix_drafters = _drafters
    gb._suffix_uid_state = _uid_state

    logger.info(
        "[SuffixDecoding] installed: max_draft=%d, max_suffix_len=%d, "
        "min_confidence=%.2f (single-request fast path; B>1 falls through)",
        max_draft,
        max_suffix_len,
        min_confidence,
    )


class Scheduler:
    """
    Scheduler for continuous batching using mlx-lm BatchGenerator.

    This scheduler manages the lifecycle of requests:
    1. Requests arrive and are added to the waiting queue
    2. Scheduler moves requests from waiting to running (via BatchGenerator)
    3. BatchGenerator processes all running requests together
    4. Finished requests are removed and outputs returned

    The key insight is that mlx-lm's BatchGenerator already implements
    continuous batching at the token level, so we use it as the backend.
    """

    # Class-level default so ``__new__``-built test stubs that bypass
    # __init__ still step cleanly; __init__ resolves the real value once
    # (an os.environ lookup per step would sit on the decode hot path).
    _step_timing_enabled = False
    # Decode steps accumulated since the last recurrent-state barrier. The
    # barrier fires off THIS depth (not the global step counter) so a batch
    # size change re-evaluates the interval against the actual live chain:
    # a deep low-batch chain materializes immediately when concurrency rises
    # instead of waiting for the next global-step multiple.
    _recurrent_chain_depth = 0
    # Running-sequence count at the previous barrier check. An idle->active
    # edge (this was 0, now >0) arms the barrier so a sequence admitted to a
    # fresh OR long-idle scheduler materializes its prefill-inherited graph
    # on its first decode step — the #1834 step-zero barrier generalized to
    # every activation, not just construction (codex #1895 r2+r3).
    _recurrent_prev_running = 0

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: SchedulerConfig | None = None,
        tool_logits_processor_factory: Any | None = None,
        model_config: Any | None = None,
    ):
        """
        Initialize the scheduler.

        Args:
            model: The MLX model
            tokenizer: The tokenizer
            config: Scheduler configuration
            tool_logits_processor_factory: Optional callable that creates a
                logits processor for tool call structural token biasing.
                Called with no args, returns a processor or None.
            model_config: Optional ``ModelConfig`` from
                ``vllm_mlx.model_auto_config``. Used as a capability gate for
                spec-decoding installs (SuffixDecoding refuses to enable on
                hybrid linear-attention models).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SchedulerConfig()
        self._tool_logits_processor_factory = tool_logits_processor_factory
        self.model_config = model_config
        if os.environ.get("RAPID_DUMP_SCHED_CONFIG"):
            import dataclasses as _dc

            logger.warning(
                "[SCHEDCONFIG] %s",
                {k: v for k, v in sorted(_dc.asdict(self.config).items())},
            )

        # Detect if tokenizer is a processor (MLLM) and get the actual tokenizer
        self._actual_tokenizer = self._get_actual_tokenizer(tokenizer)

        # #1049 — harmony family gate for channel-scoped user stops.
        # When True, ``stop=[...]`` sequences match only inside the
        # ``<|channel|>final<|message|>`` body of the decoded surface;
        # the analysis channel (CoT) is stop-agnostic. Non-harmony
        # models keep raw-stream stop matching unchanged. Computed
        # once at scheduler init from the tokenizer identity so per-
        # step cost is one boolean check on the hot decode loop.
        from .reasoning.harmony_stop import is_harmony_family_tokenizer

        self._is_harmony_family = is_harmony_family_tokenizer(self._actual_tokenizer)

        # Per-request streaming detokenizers for UTF-8-safe incremental decode
        self._detokenizer_pool: dict[str, Any] = {}

        # Request management - following vLLM's design
        self.waiting: deque[Request] = deque()  # Waiting queue (FCFS)
        self.running: dict[str, Request] = {}  # Running requests by ID
        self.requests: dict[str, Request] = {}  # All requests by ID
        self.finished_req_ids: set[str] = set()  # Recently finished
        # Debug aid (#1878): resolved ONCE — an os.environ lookup per
        # step would put dict access on the decode hot path advertised
        # as zero-cost when off.
        self._step_timing_enabled = bool(os.environ.get("RAPID_STEP_TIMING"))
        # prev_running = 0 means the first activation is an idle->active edge,
        # so the first decode step arms the barrier (materializes any
        # prefill-inherited recurrent graph) — see the class-level comment.
        self._recurrent_chain_depth = 0
        self._recurrent_prev_running = 0

        # Mapping between our request IDs and BatchGenerator UIDs
        self.request_id_to_uid: dict[str, int] = {}
        self.uid_to_request_id: dict[int, str] = {}

        # #558 PR-3: authoritative per-uid logits-processor state. mlx-lm's
        # ``GenerationBatch`` keys ``logits_processors`` positionally by uid
        # index, and that list can DESYNC from ``uids`` when a no-processor
        # request finishes while another is mid-flight (a stale entry survives
        # the filter). A positional desync would apply a grammar mask to the
        # wrong request — or, on a length mismatch, permanently drop a
        # bystander's penalty processors when we rebuild.
        #
        # We keep the COMPLETE per-request processor list (grammar + penalties,
        # in insert order) keyed by uid here for EVERY live uid that carries any
        # processor — not just grammar ones — so
        # ``_realign_grammar_logits_processors`` can reconstruct each slot
        # entirely from uid-keyed state on a desync. This preserves a
        # penalty-only request's repetition/frequency/presence processors (they
        # would otherwise be zeroed when the positional list is length-mismatched
        # and cannot be trusted), and never applies one request's grammar to
        # another (codex #558-PR3).
        self.uid_to_request_processors: dict[int, list] = {}
        # Which of the tracked uids actually carry a grammar processor. The
        # realign guard only needs to fire while a grammar is in flight; a
        # penalty-only uid is recorded above for correct desync repair but does
        # NOT by itself arm the guard.
        self._uids_with_grammar: set[int] = set()
        # Which tracked uids carry a generation-time reasoning-budget processor
        # (force-close </think>). Like a grammar, a budget processor is a
        # per-request logits processor whose slot MUST stay aligned to its uid
        # across ticks: mlx-lm's positional ``logits_processors`` list can desync
        # when a NO-processor request finishes while this one is mid-flight,
        # silently dropping the budget so the </think> is never forced (observed
        # on a live model: a plain greedy request preceding a budget request left
        # the budget inert). So a live budget uid ALSO arms the realign guard,
        # which rebuilds every slot from ``uid_to_request_processors`` (the source
        # of truth that already holds the budget processor). Penalty-only uids
        # still do NOT arm it — only grammar and budget carry a hard per-token
        # guarantee that a desync would violate.
        #
        # Maps uid -> budget processor (not a bare set) so the realign guard can
        # see each processor's ``_ended`` latch: a budget processor arms the guard
        # ONLY while it can still act (before it has forced ``</think>``). Once it
        # is ``_ended`` (thinking closed, now emitting the answer) it is inert —
        # keeping the guard armed for it would add an O(batch) slot rebuild to
        # EVERY answer token for nothing (codex R10 #4). A finished uid's leaked
        # processor is still handled: ``_forget_uid_grammar`` tombstones it, and
        # the tombstone re-arms the guard for the scrub tick.
        self._uids_with_reasoning_budget: dict[int, Any] = {}
        # Stateless, but still a hard per-request invariant: positional slot
        # desync must never leak a suppression mask to another request/model.
        self._uids_with_suppressed_tokens: dict[int, Any] = {}
        # Identity set of every STATEFUL per-request processor currently in
        # flight (by ``id()``) — a grammar OR a reasoning-budget force-close.
        # Lets a slot be scrubbed of ANY such processor before its uid's
        # authoritative list is applied — recognizing even one whose owning uid
        # already finished but that still lingers in a leaked slot. An id is
        # dropped only once its owning uid is removed AND the processor is absent
        # from every live batch slot (tombstoned until then; see
        # ``_forget_uid_grammar`` + the sweep in the realign guard).
        self._known_stateful_processors: set[int] = set()
        # Keeps each tracked processor OBJECT alive while its ``id()`` is in the
        # set above, so Python can't reuse that id for a different object while
        # the processor may still sit in a batch slot (id-reuse-after-GC guard).
        self._stateful_processor_objs: dict[int, Any] = {}
        # Stateful processors (grammar/budget) whose owning uid has left the
        # tracking maps but that may still linger in a leaked batch slot. Kept in
        # ``_known_stateful_processors`` (so they're still scrubbed) until the
        # realign guard confirms they're absent from every live slot, then
        # forgotten. Closes the cleanup-ordering gap where the LAST such processor
        # finishing would disarm the guard before its leaked slot was cleaned.
        self._stateful_tombstones: set[int] = set()

        # BatchGenerator - the actual batching engine
        self.batch_generator: BatchGenerator | None = None
        self._current_sampler_params: tuple | None = None

        # Sampler cache: interns ``make_sampler`` results keyed on
        # ``(temp, top_p, min_p, top_k)``. Homogeneous concurrent
        # batches end up sharing one callable, which lets
        # ``_install_dense_sampler_fastpath`` detect them by identity and
        # swap to mlx-lm's batched fast path.
        #
        # Bounded LRU (``OrderedDict``) because the cache key is
        # request-controlled: an adversarial client could otherwise
        # stream many unique float values for ``(temp, top_p, min_p,
        # top_k)`` and grow the cache without bound. Production traffic
        # almost always converges to one or two distinct keys, so a
        # small cap is more than enough; evicting an entry just costs
        # one ``make_sampler`` call the next time that key reappears.
        self._sampler_cache: OrderedDict[tuple, Any] = OrderedDict()
        self._sampler_cache_max = 32

        # #1197: resolve the shared KV-quant group size + per-cache enable flags.
        self._init_kv_quantization(model)

        # Prefix cache for KV state reuse
        self.prefix_cache: PrefixCacheManager | None = None
        self.memory_aware_cache: MemoryAwarePrefixCache | None = None
        self.paged_cache_manager: PagedCacheManager | None = None
        self.block_aware_cache: BlockAwarePrefixCache | None = None

        if self.config.enable_prefix_cache:
            if self.config.use_paged_cache:
                # Use paged cache for memory efficiency
                self.paged_cache_manager = PagedCacheManager(
                    block_size=self.config.paged_cache_block_size,
                    max_blocks=self.config.max_cache_blocks,
                )
                self.block_aware_cache = BlockAwarePrefixCache(
                    model=model,
                    paged_cache_manager=self.paged_cache_manager,
                )
                logger.info(
                    f"Paged cache enabled: block_size={self.config.paged_cache_block_size}, "
                    f"max_blocks={self.config.max_cache_blocks}"
                )
            elif self.config.use_memory_aware_cache:
                # Use memory-aware cache (recommended for large models)
                cache_config = MemoryCacheConfig(
                    max_memory_mb=self.config.cache_memory_mb,
                    max_memory_percent=self.config.cache_memory_percent,
                    # #1197: the retained prefix cache keeps its original enable
                    # flag and configured group size — ``_quantize_cache`` coerces
                    # the group size PER LAYER against the real stored dims at
                    # quantize time (and keeps a layer bf16 when none fits), so no
                    # config-level head_dim probe can wrongly disable or mis-size
                    # it (that probe misreads MLA models like DeepSeek-V3).
                    kv_quantize=self.config.kv_cache_quantization,
                    kv_bits=self.config.kv_cache_quantization_bits,
                    kv_group_size=self.config.kv_cache_quantization_group_size,
                    kv_min_quantize_tokens=self.config.kv_cache_min_quantize_tokens,
                    kv_turboquant=self.config.kv_cache_turboquant,
                    kv_turboquant_bits=self.config.kv_cache_turboquant_bits,
                    kv_turboquant_group_size=self.config.kv_cache_turboquant_group_size,
                    kv_turboquant_mode=self.config.kv_cache_turboquant_mode,
                    # #1103: bounded trim-free hybrid reuse (0 = #1075 policy).
                    hybrid_reuse_max_entries=self.config.hybrid_cache_entries,
                )
                # R15-P1 (task #303): radix-tree prefix-cache index.
                # Constructed when ``prefix_cache_index == "radix"`` and
                # threaded into the memory-aware cache so store/fetch
                # stay coherent. ``"hash"`` skips construction entirely.
                radix_idx = None
                if self.config.prefix_cache_index == "radix":
                    try:
                        from .runtime.radix_index import RadixPrefixIndex

                        radix_idx = RadixPrefixIndex()
                    except Exception as exc:  # pragma: no cover — defensive
                        logger.warning(
                            f"[radix] failed to construct RadixPrefixIndex: {exc}; "
                            "falling back to hash index"
                        )
                        radix_idx = None
                self.memory_aware_cache = MemoryAwarePrefixCache(
                    model=model,
                    config=cache_config,
                    radix_index=radix_idx,
                )
                logger.info(
                    f"Memory-aware cache enabled: "
                    f"limit={self.memory_aware_cache.memory_limit_mb:.1f}MB, "
                    f"index={'radix' if radix_idx is not None else 'hash'}"
                )
            else:
                # Use legacy entry-count based prefix cache
                self.prefix_cache = PrefixCacheManager(
                    model=model,
                    max_entries=self.config.prefix_cache_size,
                )
                logger.info(
                    f"Prefix cache enabled with max_entries={self.config.prefix_cache_size}"
                )

        # Thread-safe set for deferred aborts (main thread → executor thread)
        # CPython GIL guarantees set.add() and `x in set` are atomic.
        self._pending_abort_ids: set[str] = set()
        # #1759: targeted second edge from engine cleanup to the executor.
        # ``remove_finished_request`` adds an id only when its running slot is
        # still live; normal completions remove ``running`` first.  This avoids
        # an O(batch) reconciliation scan on every decode tick.
        self._orphaned_running_candidates: dict[str, Request] = {}
        # M-01 codex r2 BLOCKING #1: lifetime de-dup set for the
        # cancellation counter. ``_pending_abort_ids`` is the wrong
        # ledger to dedupe against — it's a DEFERRED-ABORT QUEUE that
        # gets drained on every step via ``_process_pending_aborts``.
        # Once drained, a later ``abort_request(rid)`` for a request
        # that's still resident (e.g. a sequence of cancel attempts
        # while the request lives in ``running``, or request_id reuse
        # across distinct lifetimes) would see ``already_pending=False``
        # again and double-count. ``_cancelled_request_ids`` is a
        # lifetime ledger — every id that has ever advanced the
        # counter stays in it for the process lifetime. Memory is
        # bounded by the cancel traffic (one ~36-byte uuid per cancel),
        # which is the same scale as ``finished_req_ids`` and not a
        # concern. The set is wiped only on ``reset()`` (matches the
        # _pending_abort_ids treatment there).
        self._cancelled_request_ids: set[str] = set()
        # M-01: once-per-request guard for the disconnect-cause
        # sub-counter. ``_force_abort_request`` calls
        # ``record_disconnect_abort`` from BOTH the disconnect branch
        # AND the GeneratorExit branch AND the finally belt-and-
        # suspenders; without this de-dup the sub-counter would over-
        # count by up to 3x per disconnect. Lifetime ledger like
        # ``_cancelled_request_ids`` above — never drained between
        # cancels.
        self._disconnect_abort_ids: set[str] = set()
        # M-01 codex r1 BLOCKING #2/#3: serialize the cancellation-
        # counter mutations against the dedupe-set membership checks.
        # ``set.add`` and ``x in set`` are individually GIL-atomic,
        # but the check-add-increment sequence is NOT — two threads
        # calling ``abort_request(rid)`` concurrently can both observe
        # ``already_counted=False`` and double-count the same request.
        # The disconnect_guard fires from up to three branches per
        # disconnect (potentially on different async tasks) and the
        # explicit cancel route can race with engine_core's own
        # cleanup-abort enqueue — both real concurrency surfaces. The
        # lock cost is negligible (microseconds per abort), well below
        # the existing per-step Metal latency.
        self._cancel_counter_lock = threading.Lock()

        # Statistics
        self.num_requests_processed = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # Last observed text-path throughput. The upstream BatchGenerator has
        # no stats object, so derive rates from request timing in our wrapper.
        self._last_prompt_tps = 0.0
        self._last_generation_tps = 0.0
        # Agent-only safety stop for exact token loops.  This remains separate
        # from normal completed-request accounting so operators can distinguish
        # healthy EOS stops from model degeneration.
        self.num_repetition_loop_stops = 0
        self.num_repetition_loop_breaks = 0
        # PFlash observability (M-02 reframe). When PFlash compresses a
        # prompt the request bypasses the prefix-cache fetch + store
        # paths entirely (positional-fiction safety; see comment block
        # near ``compress_request_tokens``). That bypass is correct but
        # silences ``rapid_mlx_prefix_cache_*`` on PFlash-always tiers
        # (e.g. verified-tier aliases), making /metrics look frozen at
        # ``hits=0/misses=1``. These two counters let operators see
        # PFlash is doing meaningful work even when the cache series
        # stays flat. Observability only — bypass semantics unchanged.
        self.pflash_bypass_count = 0
        self.pflash_compressed_tokens_dropped = 0
        # Cancellation observability (M-01). ``num_requests_processed``
        # deliberately excludes aborted requests, so operators staring at
        # ``rapid_mlx_requests_processed_total = 0`` after fifty bailed-
        # out clients can't tell whether the route is broken, the model
        # is idle, or every caller is disconnecting before EOS. The total
        # counter increments inside ``abort_request`` the moment a
        # newly-known request_id transitions into the pending-abort set
        # (idempotent re-enqueues do NOT double-count), so it reflects
        # accepted public-API aborts irrespective of cause. The disconnect
        # sub-counter is bumped separately by ``_force_abort_request`` in
        # the disconnect-guard path via ``record_disconnect_abort`` so
        # the (total - disconnect) gap surfaces explicit-cancel-route +
        # timeout traffic. Both observability only — abort semantics are
        # untouched.
        self.num_requests_cancelled = 0
        self.num_requests_cancelled_via_disconnect = 0
        # D-METAL-CAP observability. Increments once per request that
        # ``add_request`` rejected because Metal active memory already
        # exceeded the soft cap. Surfaced as
        # ``rapid_mlx_metal_cap_violations_total`` so operators can see
        # ``--gpu-memory-utilization`` is doing meaningful work
        # (pre-fix, the cap was silently violated and there was no
        # series to alert on).
        self.num_metal_cap_violations = 0
        # D-METAL-PFX observability. Increments once per prefix-cache
        # entry that was evicted by the Metal-pressure trigger (separate
        # series from the LRU-capacity evictions reported by the cache
        # itself). Surfaced as
        # ``rapid_mlx_prefix_cache_pressure_evictions_total``.
        self.num_prefix_cache_pressure_evictions = 0
        # D-METAL-CAP: once-per-process WARNING gate. The log noise of
        # a sustained over-cap admit storm would otherwise drown the
        # rest of the engine output; we want exactly one operator-
        # visible WARNING when the cap first trips, and then rely on
        # the Prometheus counter for ongoing visibility.
        self._metal_cap_warning_logged = False

        # Memory management: periodic mx.clear_cache() to free Metal command buffers
        # Lower interval = less VRAM spike during generation but slight throughput cost
        self._step_count = 0
        self._clear_cache_interval = 32
        self._memory_log_interval = 256
        self._last_adaptive_prefill_size = self.config.prefill_step_size
        self._adaptive_prefill_protected_chunks = 0
        self._adaptive_prefill_reduced_chunks = 0
        # D-METAL-CAP / D-METAL-PFX: cached hard cap in bytes for fast
        # admission checks. Computed lazily on first use so unit tests
        # that build a Scheduler against a fake model with no Metal
        # device pay zero cost. ``0`` means "no cap" (see
        # ``gpu_memory_utilization`` doc on SchedulerConfig).
        self._metal_cap_bytes: int = 0
        self._metal_cap_bytes_resolved: bool = False
        # D-METAL-CAP: cached per-token KV-cache size for the
        # projection-based admission gate. Auto-derived from the
        # model config on first use (operator override via
        # ``SchedulerConfig.metal_cap_kv_bytes_per_token`` wins). See
        # ``_resolve_kv_bytes_per_token`` for the formula. ``0``
        # means "auto-derive failed / no model config" which
        # disables the projection branch (back-compat for unit
        # tests built against MagicMock models).
        self._kv_bytes_per_token: int = 0
        self._kv_bytes_per_token_resolved: bool = False
        # D-METAL-CAP: cached per-sequence FIXED KV baseline (bytes) for
        # architecture-aware hybrids — the conservative recurrent-state term of a
        # hybrid's linear-attention (GatedDeltaNet) layers. Allocated once per
        # sequence, not per token (an SSM state does not grow with the token
        # budget), so ``_estimate_request_kv_bytes`` charges it as a flat add-on
        # rather than multiplying by the token count. ``0`` for dense models and
        # for the byte-identical uniform fallback (see
        # ``_resolve_kv_bytes_per_token`` and ``kv_estimation``).
        self._kv_fixed_baseline_bytes: int = 0
        # D-METAL-CAP: cached SLIDING-window term for hybrids with rotating-cache
        # layers (GPT-OSS, Gemma-4 local). ``_kv_sliding_slot_bytes`` is the
        # per-SLOT bytes summed across the window-bounded sliding layers and
        # ``_kv_sliding_window`` is their shared rotating window. Unlike the fixed
        # baseline, this term is REQUEST-DEPENDENT: a sliding buffer grows with
        # the token budget up to the window, so ``_estimate_request_kv_bytes``
        # multiplies ``_kv_sliding_slot_bytes`` by
        # ``kv_estimation.rotating_cache_slots(window, T)`` per request — the
        # over-count-safe upper bound of the real ``RotatingKVCache`` allocation
        # that grows with ``T`` and caps at the full window (codex round 11
        # BLOCKING #1: a flat whole-window baseline over-counts short requests).
        # Both ``0`` for dense models and the uniform fallback.
        self._kv_sliding_slot_bytes: int = 0
        self._kv_sliding_window: int = 0

        # Prompt-boundary cache snapshot callback for the new mlx-lm 0.31+ API.
        # Built lazily once memory_aware_cache exists and reused per step.
        # Without this hook, hybrid models can't satisfy repeated identical
        # prompts via supersequence fallback (issue #163).
        self._prompt_cache_save_cb = (
            self._make_prompt_cache_save_callback()
            if self.memory_aware_cache is not None
            else None
        )

    def _get_actual_tokenizer(self, tokenizer: Any) -> Any:
        """
        Get the actual tokenizer from a processor or tokenizer.

        MLLM models use processors (e.g., Qwen3VLProcessor) which wrap
        the tokenizer. This method extracts the actual tokenizer.
        """
        # If it has encode method, it's already a tokenizer
        if hasattr(tokenizer, "encode") and callable(tokenizer.encode):
            return tokenizer
        # If it's a processor, get the wrapped tokenizer
        if hasattr(tokenizer, "tokenizer"):
            return tokenizer.tokenizer
        # Fallback to the original
        return tokenizer

    def _decode_tokens(self, token_ids: list[int]) -> str:
        """
        Decode token IDs to text, handling both tokenizers and processors.
        """
        return self._actual_tokenizer.decode(token_ids)

    def _get_detokenizer(self, request_id: str) -> Any:
        """Get or create a streaming detokenizer for a request."""
        if request_id not in self._detokenizer_pool:
            if hasattr(self.tokenizer, "detokenizer"):
                detok = self.tokenizer.detokenizer
            else:
                detok = NaiveStreamingDetokenizer(self._actual_tokenizer)
            detok.reset()
            self._detokenizer_pool[request_id] = detok
        return self._detokenizer_pool[request_id]

    def _cleanup_detokenizer(self, request_id: str) -> None:
        """Remove the streaming detokenizer for a finished request."""
        self._detokenizer_pool.pop(request_id, None)

    def _get_stop_tokens(self) -> set[int]:
        """Get stop token IDs from tokenizer or processor.

        Resolution order (all sources unioned — set semantics make
        overlap harmless):

        1. ``TokenizerWrapper._eos_token_ids`` — the curated set
           mlx-lm's own ``BatchGenerator`` uses to halt generation.
           Grown at load time by
           ``augment_eos_token_ids_from_generation_config`` to
           include the chat-template terminator (Gemma 3
           ``<end_of_turn>``, Qwen3 ``<|endoftext|>``, Llama 3
           ``<|eot_id|>``, etc.).
        2. ``tok.eos_token_id`` — the underlying HF tokenizer's
           primary id. Required for non-wrapped tokenizers (custom
           fallback paths, mlx-vlm processor objects).
        3. ``tok.eos_token_ids`` — some processors expose the plural
           form natively.
        4. ``tok._rapid_extra_eos_token_ids`` — the union stashed by
           ``augment_eos_token_ids_from_generation_config`` on raw
           HF tokenizers whose ``eos_token_ids`` is a property that
           rejects non-string assignment. This is the surface that
           rescues mlx-vlm processors (Gemma 3 VL etc.).
        """
        from .utils.tokenizer import RAPID_EXTRA_EOS_ATTR

        stop_tokens: set[int] = set()
        # Check both the processor/tokenizer and the actual tokenizer
        for tok in [self.tokenizer, self._actual_tokenizer]:
            if tok is None:
                continue
            # Source 1: mlx-lm TokenizerWrapper's curated set.
            wrapper_ids = getattr(tok, "_eos_token_ids", None)
            if wrapper_ids:
                stop_tokens.update(wrapper_ids)
            # Source 2: legacy singular path.
            if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
                if isinstance(tok.eos_token_id, list):
                    stop_tokens.update(tok.eos_token_id)
                else:
                    stop_tokens.add(tok.eos_token_id)
            # Source 3: processor-style plural path.
            if hasattr(tok, "eos_token_ids") and tok.eos_token_ids is not None:
                if isinstance(tok.eos_token_ids, (list, set, tuple)):
                    stop_tokens.update(tok.eos_token_ids)
                else:
                    stop_tokens.add(tok.eos_token_ids)
            # Source 4: Rapid-MLX extras stash (see RAPID_EXTRA_EOS_ATTR).
            extras = getattr(tok, RAPID_EXTRA_EOS_ATTR, None)
            if extras:
                stop_tokens.update(extras)
        return stop_tokens

    def _get_request_sampler(self, sampling_params: SamplingParams) -> Any:
        """Return a cached sampler for these sampling params.

        Interning samplers by ``(temp, top_p, min_p, top_k)`` is what
        lets ``_install_dense_sampler_fastpath`` detect homogeneous
        batches via identity comparison on ``GenerationBatch.samplers``.
        Without this, every request would carry its own
        ``make_sampler`` closure even when the params are identical,
        forcing the slow per-row loop in mlx-lm.

        WARNING: the cache key intentionally covers only the four
        knobs threaded through to ``make_sampler``. If we ever start
        forwarding xtc_probability / xtc_threshold / xtc_special_tokens
        per request, the key MUST grow accordingly — otherwise
        homogeneous-looking batches would silently share an incorrect
        sampler.
        """
        # H-11: per-request seed requests bypass the shared sampler cache
        # because the seeded sampler carries mutable per-call PRNG state.
        # Two requests with the same ``seed`` MUST still each get their
        # own closure — otherwise the second request would resume from
        # wherever the first left off (so its first token would be the
        # first request's second token). The mlx-lm fast-path interning
        # (identity-equality on ``GenerationBatch.samplers``) is also
        # incorrect for seeded requests because the dense-batch fast
        # path replaces the per-row dispatch with a single shared
        # sampler call — which would lose the seed isolation. Seeded
        # requests therefore route through ``_mtp_step``'s explicit
        # per-row loop and skip the dense sampler fast path naturally
        # (the identity-equality check fails when each row has its own
        # closure).
        #
        # ``getattr`` defaults to ``None`` so legacy callers (community
        # bench harness, embedded test stubs) that construct
        # ``SamplingParams`` look-alikes via attribute set without the
        # H-11 ``seed`` field still route through the unchanged cache
        # path — no behaviour change for the pre-H-11 surface.
        _seed = getattr(sampling_params, "seed", None)
        if _seed is not None:
            # Log once per process so operators can confirm the H-11
            # plumbing is engaged on a deployment without spamming the
            # request log on every seeded request. Mirrors the
            # ``_fused_top_p_logged`` belt below.
            #
            # Codex r1 NIT: do NOT include the raw seed value here. Seeds
            # are caller-controlled and routinely come from eval / audit
            # harnesses where leakage to an operator log would let a
            # reviewer replay the exact graded outputs. Operators just
            # need to know the per-request RNG path is engaged; the
            # request itself can still be correlated by the request id
            # on the surrounding scheduler log line.
            if not getattr(self, "_seeded_sampler_logged", False):
                logger.info(
                    "[seeded_sampler] H-11 engaged — per-request "
                    "seeds are honoured (sample shape: temp=%.3f "
                    "top_p=%.3f)",
                    sampling_params.temperature,
                    sampling_params.top_p,
                )
                self._seeded_sampler_logged = True
            return make_seeded_sampler(
                seed=_seed,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                min_p=sampling_params.min_p,
                top_k=sampling_params.top_k,
            )
        # Codex round-2 BLOCKER #3 fix: read the env-var BEFORE the cache
        # lookup so that flipping ``RAPID_MLX_DISABLE_FUSED_SAMPLER`` in a
        # long-lived process can disable the fast path on the next request
        # without us serving a stale cached fused sampler. The disabled
        # state is folded into the cache key so the two branches don't
        # collide either.
        # Codex round-5 NIT: accept a small set of truthy values so operators
        # who set ``RAPID_MLX_DISABLE_FUSED_SAMPLER=true`` (the more natural
        # form for a boolean knob) actually get the fast path disabled,
        # instead of silently leaving it on.
        _fused_disabled = os.environ.get(
            "RAPID_MLX_DISABLE_FUSED_SAMPLER", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        key = (
            sampling_params.temperature,
            sampling_params.top_p,
            sampling_params.min_p,
            sampling_params.top_k,
            _fused_disabled,
        )
        cached = self._sampler_cache.get(key)
        if cached is not None:
            # LRU bookkeeping — keep the hot key warm.
            self._sampler_cache.move_to_end(key)
            return cached
        # Fast path for the dominant chat config (temp + top_p, with or
        # without top_k). See ``vllm_mlx/_sampler_fast_path.py`` for the
        # math-equivalence argument and the perf data behind it (Qwen 3.6
        # 35B 4-bit B=1: 65 -> 92 tok/s). Falls through to mlx-lm's chain
        # whenever the request enables min_p, xtc, sets temperature == 0
        # (mlx-lm already short-circuits to argmax), is top-k-only with no
        # nucleus cut (mlx-lm uses a cheaper partition primitive there),
        # or whenever the operator sets
        # ``RAPID_MLX_DISABLE_FUSED_SAMPLER=1`` as an escape hatch.
        if not _fused_disabled and is_fused_top_p_eligible(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            min_p=sampling_params.min_p,
            top_k=sampling_params.top_k,
        ):
            sampler = make_fused_top_p_temp_sampler(
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                top_k=sampling_params.top_k,
            )
            if not getattr(self, "_fused_top_p_logged", False):
                logger.info(
                    "[fused_top_p_sampler] engaged for temp=%.3f top_p=%.3f top_k=%d",
                    sampling_params.temperature,
                    sampling_params.top_p,
                    sampling_params.top_k,
                )
                self._fused_top_p_logged = True
        else:
            sampler = make_sampler(
                temp=sampling_params.temperature,
                top_p=sampling_params.top_p,
                min_p=sampling_params.min_p,
                top_k=sampling_params.top_k,
            )
        self._sampler_cache[key] = sampler
        # Evict the least-recently-used entry once we exceed the cap.
        # Identity-sharing only matters for live in-flight batches; a
        # freshly evicted sampler that reappears just costs one
        # ``make_sampler`` call.
        if len(self._sampler_cache) > self._sampler_cache_max:
            self._sampler_cache.popitem(last=False)
        return sampler

    def _init_kv_quantization(self, model) -> None:
        """Resolve the LIVE cache's group size + install gate (#1197).

        Only the LIVE continuous-batching cache (``QuantizedBatchKVCache``) needs
        this up-front, config-level decision: its type is fixed before any token
        and cannot fall back to bf16 mid-stream, so an incompatible first write
        would crash the request. ``mx.quantize`` requires the quantized dim
        divisible by a group size in {32,64,128}: a head_dim=96 model drops
        64->32, and head_dim=80 (or an unprobeable model) can't be quantized, so
        the live cache stays bf16 (``_kv_quant_live_disabled``).

        The retained prefix cache is deliberately NOT gated here — it self-coerces
        per layer against the real stored dims at quantize time
        (``memory_cache._quantize_cache``), which also correctly handles MLA
        models (DeepSeek-V3) whose cached dims differ from any config head dim, so
        the fragile config-level probe never disables or mis-sizes it.

        Extracted from ``__init__`` so the scheduler -> live-hook wiring is
        unit-testable without loading a model.
        """
        self._kv_quant_group_size = self.config.kv_cache_quantization_group_size
        self._kv_quant_live_disabled = False
        if not (
            self.config.kv_cache_quantization
            and not getattr(self.config, "kv_cache_turboquant", None)
        ):
            return

        from .quantized_batch_cache import (
            probe_kv_head_dims,
            resolve_kv_quantization,
        )

        k_hd, v_hd = probe_kv_head_dims(model)
        requested = self._kv_quant_group_size
        self._kv_quant_group_size, self._kv_quant_live_disabled = (
            resolve_kv_quantization(k_hd, v_hd, requested)
        )

        if self._kv_quant_live_disabled:
            logger.warning(
                "[kv-cache] live KV quantization disabled: head_dim (K=%s, V=%s) "
                "unknown or not divisible by any supported group_size "
                "(32/64/128); serving a bf16 live cache (retained prefix cache "
                "quantizes independently). Pass --kv-cache-dtype bf16 to silence.",
                k_hd,
                v_hd,
            )
        elif self._kv_quant_group_size != requested:
            logger.info(
                "[kv-cache] adjusted live group_size %d->%d for head_dim (K=%s, V=%s)",
                requested,
                self._kv_quant_group_size,
                k_hd,
                v_hd,
            )

    def _create_batch_generator(
        self, sampling_params: SamplingParams
    ) -> BatchGenerator:
        """Create a BatchGenerator with the given sampling parameters."""
        sampler = make_sampler(
            temp=sampling_params.temperature,
            top_p=sampling_params.top_p,
            min_p=sampling_params.min_p,
            top_k=sampling_params.top_k,
        )

        stop_tokens = _assemble_stop_tokens(sampling_params, self._get_stop_tokens())

        # mlx-lm 0.31.3+: BatchGenerator captures generation_stream at __init__
        # via a thread-local Stream; without an explicit stream= the captured
        # stream is whatever the import-thread had — which on the asyncio loop
        # thread is unreachable from the mlx-step worker that runs .next(),
        # so every request fails with "There is no Stream(gpu, 1) in current
        # thread" (#170 hot path; complements the warmup fix in PR #173).
        # _create_batch_generator runs on the mlx-step thread so default_stream
        # here is the worker's stream (our `_init_mlx_step_thread` sets it).
        bg_kwargs = dict(
            model=self.model,
            max_tokens=sampling_params.max_tokens,
            stop_tokens=stop_tokens,
            sampler=sampler,
            prefill_batch_size=self.config.prefill_batch_size,
            completion_batch_size=self.config.completion_batch_size,
            prefill_step_size=self.config.prefill_step_size,
        )
        try:
            import mlx.core as _mx

            bg = BatchGenerator(
                **bg_kwargs, stream=_mx.default_stream(_mx.default_device())
            )
        except TypeError:
            # mlx-lm < 0.31.3 — no `stream` kwarg; fall back to legacy path.
            bg = BatchGenerator(**bg_kwargs)

        # ``--kv-cache-dtype int8/int4`` must reach the LIVE continuous-batching
        # KV cache, not just the retained prefix cache (#1197). Swap the
        # generator's plain ``BatchKVCache`` for a quantized, dequant-on-read
        # ``QuantizedBatchKVCache``. TurboQuant has its own path, so this only
        # runs for the plain quantization toggle. The head_dim-compatible group
        # size (and the disable-on-incompatible decision) were resolved once in
        # __init__ so the live and retained caches never diverge.
        # ``getattr`` guards: some unit paths build a Scheduler via
        # ``__new__`` + a stub config (bypassing __init__), so these attributes
        # may be absent — treat missing as "no live quantization".
        self._live_kv_quant: tuple[int, int] | None = None
        if (
            getattr(self.config, "kv_cache_quantization", False)
            and not getattr(self.config, "kv_cache_turboquant", None)
            and not getattr(self, "_kv_quant_live_disabled", True)
        ):
            from .quantized_batch_cache import install_quantized_batch_cache

            bits = getattr(self.config, "kv_cache_quantization_bits", 8)
            eff_gs = getattr(self, "_kv_quant_group_size", 64)
            if install_quantized_batch_cache(bg, group_size=eff_gs, bits=bits):
                # Remember the effective params so prefix-cache HITS get
                # normalized to the same quantized type as MISSES (#1197).
                self._live_kv_quant = (eff_gs, bits)
                logger.info(
                    "[kv-cache] live continuous-batching KV cache quantized to "
                    "int%d (group_size=%d)",
                    bits,
                    eff_gs,
                )
            else:
                # Running mlx-lm lacks the ``BatchGenerator._make_new_cache``
                # hook (need mlx-lm>=0.31.3). Keep the bf16 live cache — the
                # retained prefix cache still quantizes — rather than letting a
                # per-step AttributeError wedge the server into a silent hang.
                self._live_kv_quant = None
                logger.warning(
                    "[kv-cache] live KV quantization skipped: this mlx-lm build "
                    "lacks BatchGenerator._make_new_cache (need mlx-lm>=0.31.3). "
                    "Live continuous-batching cache stays bf16; upgrade mlx-lm "
                    "to quantize it."
                )

        # Server-side wiring for ``--speculative-config '{"method":"mtp"}'``.
        # This installs the vendored PR #990 ``mtp_generate_step`` hot
        # loop as ``GenerationBatch._step``, gated on the target having
        # the ``mtp_forward`` / ``make_mtp_cache`` protocol installed
        # by ``dispatch_mtp_inject`` (which runs during engine boot in
        # ``BatchedEngine._start_llm`` before this scheduler is built).
        #
        # Single-request adaptive chain-of-K with batched verify.
        if getattr(self.config, "spec_decode", "none") == "mtp":
            mtp_model_type = getattr(self.config, "mtp_model_type", None)
            config_vetted_mtp = _config_vetted_mtp_supports_spec_decode(mtp_model_type)
            if (
                getattr(self, "model_config", None) is not None
                and not self.model_config.supports_spec_decode
                and not config_vetted_mtp
            ):
                logger.warning(
                    "[MTP-vendored] MTP speculative-config requested but "
                    "profile says supports_spec_decode=False and "
                    "model_type=%r is not in the config-vetted MTP allowlist. "
                    "MTP will be disabled.",
                    mtp_model_type,
                )
            else:
                _install_mtp_vendored(
                    bg,
                    model=self.model,
                    requests=self.requests,
                    uid_to_request_id=self.uid_to_request_id,
                    # 0.9.13 PR-B: EV depth controller knobs.
                    max_k=getattr(self.config, "mtp_max_k", 3),
                    disable_auto_k=getattr(self.config, "mtp_disable_auto_k", False),
                    # Preferred (named) identity for the controller
                    # registry. The previous spelling ended in a bare
                    # conditional, and Python binds that loosest, so
                    # ``a or b if c else None`` collapsed the whole
                    # expression to ``None`` whenever ``model_config`` was
                    # absent. ``None`` here is not fatal —
                    # ``_install_mtp_vendored`` derives a structural key —
                    # but it discarded the good name whenever one existed.
                    controller_key=_mtp_controller_key(
                        getattr(self, "_model_name", None)
                        or getattr(getattr(self, "model_config", None), "name", None)
                        or getattr(self.config, "model_name", None),
                        getattr(self.config, "mtp_sidecar", None),
                    ),
                )

        if getattr(self.config, "spec_decode", "none") == "dspark":
            _install_dspark(
                bg,
                model=self.model,
                requests=self.requests,
                uid_to_request_id=self.uid_to_request_id,
                max_draft=getattr(self.config, "dspark_num_speculative_tokens", 5),
            )

        # Install SuffixDecoding (drafter-free spec-decode).
        if self.config.enable_suffix_decoding:
            _install_suffix_decoding(
                bg,
                model=self.model,
                profile=self.model_config,
                max_draft=self.config.suffix_max_draft,
                max_suffix_len=self.config.suffix_max_suffix_len,
                min_confidence=self.config.suffix_min_confidence,
                min_draft_len=self.config.suffix_min_draft_len,
                requests=self.requests,
                uid_to_request_id=self.uid_to_request_id,
            )

        # Install batched-sampler fast path. Must run AFTER MTP /
        # SuffixDecoding since they may replace _step on the
        # GenerationBatch — our wrapper has to sit at the outermost
        # layer so it can short-circuit the per-row loop wherever the
        # final _step ends up. SuffixDecoding/MTP wrappers themselves
        # call into the original ``_step`` and ignore ``self.samplers``,
        # so this layering is safe.
        _install_dense_sampler_fastpath(bg)

        # Singleton-cache fast path: keep per-request KV caches in their
        # single-sequence form while the batch holds one row (plain causal
        # mask -> mx.fast SDPA's native fast path), promoting to batched
        # caches only when a second row joins. Idempotent module-level
        # patch on mlx_lm.generate; measured +4-8% B=1 decode (bench
        # 2026-08-12, oMLX parity study).
        from .singleton_cache_fastpath import install_singleton_cache_fastpath

        install_singleton_cache_fastpath()

        return bg

    def _make_prompt_cache_save_callback(self):
        """Create a callback that stores prompt-only KV/Mamba cache.

        Called from ``_generation_step`` right before the first output token
        is fed into the model.  At that point ``num_tokens == 0`` and the
        batch cache contains the exact prompt-only state (correct for both
        KVCache and MambaCache/ArraysCache layers).

        The cache is stored with key = prompt_token_ids so that a future
        request with the identical prompt gets an exact hit.
        """
        import time as _time

        def _prompt_cache_save(uid, extracted_cache):
            request_id = self.uid_to_request_id.get(uid)
            if not request_id:
                return
            request = self.requests.get(request_id)
            if not request or not request.prompt_token_ids:
                return
            # PFlash bypass: see scheduler.add_request — compressed
            # prompt_token_ids are not positionally faithful so storing
            # KV under this key would poison the trie.
            if _pflash_compressed(request):
                return

            # Bounded trim-free reuse deliberately segments a cold request at
            # N-1. Saving an additional exact N-token entry would win lookup
            # priority, fail trim-one kickoff on non-trimmable caches, and mask
            # the reusable N-1 prefix. The boundary snapshot is the prompt
            # cache for this path; completion still stores prompt + output.
            if (
                getattr(self.config, "non_trimmable_exact_prefix_reuse", False)
                and getattr(request, "_cache_snapshot_is_internal", False)
                and getattr(request, "_cache_snapshot_stored", False)
            ):
                return

            prompt_tokens = list(request.prompt_token_ids)
            _t0 = _time.monotonic()
            # evict_prefixes=False: keep mid-prefill boundary entries so
            # that future requests with the same prefix but different
            # suffix get a prefix cache hit (critical for agentic multi-turn).
            stored = self.memory_aware_cache.store(
                prompt_tokens, extracted_cache, evict_prefixes=False
            )
            _dt = _time.monotonic() - _t0
            if stored:
                logger.info(
                    f"[prompt_cache_save] request={request_id[:12]} "
                    f"prompt_tokens={len(prompt_tokens)} "
                    f"store_time={_dt:.3f}s"
                )

        return _prompt_cache_save

    def _snapshot_promoted_prompts(self, prompt_responses) -> None:
        """Snapshot prompt-only cache for sequences just promoted to generation.

        Reads the public ``end_of_prompt`` flag from mlx-lm 0.31+'s prompt
        responses, then uses the public ``BatchGenerator.extract_cache`` API
        to capture the per-uid cache state. Each capture is forwarded to the
        prompt-cache-save callback so a future request with the identical
        prompt finds an exact-match entry in the prefix cache.

        Without it, hybrid models (Mamba/DeltaNet+Transformer) MISS
        the prefix cache forever because their non-trimmable cache layers
        cannot satisfy the supersequence fallback path (issue #163).
        """
        if self._prompt_cache_save_cb is None or not prompt_responses:
            return

        promoted_uids = [
            resp.uid
            for resp in prompt_responses
            if getattr(resp, "end_of_prompt", False)
        ]
        if not promoted_uids:
            return

        try:
            extracted = self.batch_generator.extract_cache(promoted_uids)
        except Exception as exc:
            logger.debug("[prompt_cache_save] extract_cache failed: %s", exc)
            return

        for uid, payload in extracted.items():
            # Promoted sequences (stage == 2) return (cache, tokens). Any
            # other shape means the uid was already removed before the
            # snapshot — skip silently.
            if isinstance(payload, tuple) and len(payload) == 2:
                cache, _tokens = payload
                try:
                    self._prompt_cache_save_cb(uid, cache)
                except Exception as exc:
                    logger.debug(
                        "[prompt_cache_save] callback failed for uid=%s: %s",
                        uid,
                        exc,
                    )

    def _snapshot_boundary_segments(self, prompt_responses) -> None:
        """Snapshot KV/Mamba cache at ``prefix_boundary`` for multi-turn workloads.

        Issue #427: hybrid models (linear-attention/Mamba + Transformer)
        MISS the LCP-based prefix cache on every turn of a growing
        conversation because the prior turn's cached entry has a tail
        that diverges from the new turn (e.g. ``<think>\\n`` template
        sentinel emitted by ``add_generation_prompt=True`` gets replaced
        by actual assistant content on the next turn) and Mamba layers
        are non-trimmable, so the supersequence fallback can't reuse
        the prefix either.

        Fix: when a request arrives with ``prefix_boundary > 0``,
        ``_schedule_waiting`` inserts it via ``insert_segments(
        [[prefix_seg, tail_seg]])`` so BatchGenerator processes the
        prefix segment as its own boundary. When that segment finishes,
        the response carries ``end_of_segment=True`` **without**
        ``end_of_prompt=True`` (the tail still has work to do). That's
        our cue to extract the cache via the public
        ``BatchGenerator.extract_cache`` API and store it under the
        ``prefix_boundary`` token prefix — so the *next* turn's lookup
        finds the boundary entry and skips re-prefilling the shared
        prefix.

        This uses mlx-lm 0.31+'s public cache-extraction API.
        """
        if self.memory_aware_cache is None or not prompt_responses:
            return

        boundary_uids: list[int] = []
        for resp in prompt_responses:
            # end_of_prompt promotions are handled by
            # _snapshot_promoted_prompts (whole-prompt entry, issue #163).
            # We only want the *inter*-segment boundary here.
            if getattr(resp, "end_of_prompt", False):
                continue
            request_id = self.uid_to_request_id.get(resp.uid)
            if not request_id:
                continue
            request = self.requests.get(request_id)
            if not request:
                continue
            snapshot_boundary = getattr(
                request,
                "_cache_snapshot_boundary",
                getattr(request, "prefix_boundary", 0),
            )
            if snapshot_boundary <= 0:
                continue
            # Defense-in-depth: validate progress[0] equals the
            # expected boundary offset. mlx-lm 0.31+ rewrites
            # ``[[prefix, tail]]`` into ``[[prefix, tail[:-1], tail[-1:]]]``
            # when ``len(tail) > 1`` (generate.py:1646-1648), so
            # end_of_segment fires THREE times — once at prefix done,
            # once at tail[:-1] done, and end_of_prompt at tail[-1:].
            # The `_boundary_snapshot_taken` guard below blocks the
            # second fire, but this progress check skips it deterministically.
            progress = getattr(resp, "progress", None)
            expected_offset = snapshot_boundary - (request.cached_tokens or 0)
            aligned_internal_chunk = (
                getattr(request, "_cache_snapshot_is_internal", False)
                and progress is not None
                and isinstance(progress, tuple)
                and len(progress) >= 1
                and progress[0] == expected_offset
            )
            if not (getattr(resp, "end_of_segment", False) or aligned_internal_chunk):
                continue
            if (
                progress is not None
                and isinstance(progress, tuple)
                and len(progress) >= 1
                and progress[0] != expected_offset
            ):
                continue
            # Once-per-request guard: prevents a future API change that
            # repeats end_of_segment from producing duplicate stores.
            if getattr(request, "_boundary_snapshot_taken", False):
                continue
            boundary_uids.append(resp.uid)

        if not boundary_uids:
            return

        try:
            extracted = self.batch_generator.extract_cache(boundary_uids)
        except Exception as exc:
            logger.debug("[boundary_snapshot] extract_cache failed: %s", exc)
            return

        import time as _time

        for uid, payload in extracted.items():
            # Stage-1 (in-prompt) and stage-2 (promoted) both return
            # ``(cache, tokens)``. Anything else means the uid was
            # already removed before the snapshot — skip silently.
            if not (isinstance(payload, tuple) and len(payload) == 2):
                continue
            cache, _tokens = payload

            request_id = self.uid_to_request_id.get(uid)
            request = self.requests.get(request_id) if request_id else None
            if not request:
                continue
            # PFlash bypass: defensive guard. add_request also zeros
            # prefix_boundary for compressed requests so the next
            # condition would short-circuit anyway, but a future change
            # touching prefix_boundary must not silently start poisoning
            # the trie.
            if _pflash_compressed(request):
                continue
            prefix_boundary = getattr(
                request,
                "_cache_snapshot_boundary",
                getattr(request, "prefix_boundary", 0),
            )
            if prefix_boundary <= 0:
                continue

            states = self._extract_cache_states(cache)
            if not states:
                continue
            reconstructed = self._reconstruct_cache_from_states(states)
            if not reconstructed:
                continue

            prefix_tokens = list(request.prompt_token_ids[:prefix_boundary])
            _t0 = _time.monotonic()
            stored = False
            try:
                # evict_prefixes=False matches the prompt-cache save
                # path — keep boundary entries so later turns with the
                # same prefix but different suffix still hit.
                stored = self.memory_aware_cache.store(
                    prefix_tokens,
                    reconstructed,
                    evict_prefixes=False,
                    message_boundary=True,
                )
            except Exception as exc:
                logger.debug(
                    "[boundary_snapshot] store failed for uid=%s: %s", uid, exc
                )
            _dt = _time.monotonic() - _t0
            # Mark the guard after the attempt (success OR failure) so a
            # repeated end_of_segment doesn't redo the expensive
            # extract+reconstruct cycle. A failed store usually means
            # the entry already exists (returns False) or the cache is
            # busy — retrying every step would be pure waste. DeepSeek
            # finding #2 on PR #435.
            request._boundary_snapshot_taken = True
            request._cache_snapshot_stored = stored

            if stored:
                logger.info(
                    f"[boundary_snapshot] request={request_id[:12]} "
                    f"saved {prefix_boundary} tokens at message boundary "
                    f"store_time={_dt:.3f}s"
                )

    def _close_batch_generator(self) -> None:
        """Properly close BatchGenerator to restore wired_limit."""
        if self.batch_generator is not None:
            try:
                if hasattr(self.batch_generator, "close"):
                    self.batch_generator.close()
            except Exception as e:
                logger.debug(f"Error closing BatchGenerator: {e}")
            self.batch_generator = None

    def _ensure_batch_generator(self, sampling_params: SamplingParams) -> bool:
        """Ensure BatchGenerator exists with compatible stop token configuration.

        Returns:
            ``True`` if a compatible generator is ready for the caller
            to insert this request into. ``False`` if the caller MUST
            requeue the request — the current generator's stop_tokens
            differ from what this request requires, and there are active
            requests still draining, so admitting now would silently bind
            this request to the wrong stop-token set. The contract is a hard
            refusal, not advisory; ``_schedule_waiting`` requeues on ``False``
            to preserve stop_token and ignore_eos semantics across overlapping
            batches. Per-request samplers (temperature, top_p, etc.) do NOT
            trigger a new generator — they are passed directly to insert().
        """
        # Per-request samplers (temperature, top_p, min_p, top_k) are passed
        # directly to batch_gen.insert(..., samplers=[request_sampler]), so
        # they do NOT need to match the global BatchGenerator sampler. The
        # only generator-level invariant is stop_tokens, which is computed at
        # init time via _assemble_stop_tokens(sampling_params, model_stop_tokens).
        #
        # Two requests must share the same BatchGenerator iff they produce the
        # same final stop_tokens set. Since model_stop_tokens is invariant
        # (fixed per model/server run), the key is:
        # (frozenset(request.stop_token_ids), request.ignore_eos).
        #
        # Requests with different temperatures but same stop config can now
        # batch together, fixing issue where Claude Code's big agentic request
        # (33K tokens, default temp) blocked smaller concurrent requests with
        # different temps for 115 seconds (Rapid-MLX #611 follow-up).
        sampler_params = (
            frozenset(sampling_params.stop_token_ids or ()),
            bool(sampling_params.ignore_eos),
        )

        # Create new generator if needed or if sampling params changed
        if (
            self.batch_generator is None
            or self._current_sampler_params != sampler_params
        ):
            # If we have an existing generator with requests, the new
            # request's stop_tokens / sampler are incompatible with the
            # live generator. Refuse admission — the caller (
            # ``_schedule_waiting``) requeues and retries on the next
            # step, after the running batch has had a chance to drain.
            # Previously we returned without recreating but left the
            # stale generator in place; ``_schedule_waiting`` would then
            # insert the new request into it, silently inheriting the
            # wrong ignore_eos behavior (codex P2 on PR #612).
            if self.batch_generator is not None and self.running:
                logger.warning(
                    "Stop token configuration changed with active requests. "
                    "Requeuing request — admission deferred until current "
                    "batch drains so stop_tokens remain consistent."
                )
                return False

            # Keep prefix cache across BatchGenerator recreations.
            # KV cache entries depend only on the input tokens, not on
            # sampling params (temperature, top_p, min_p).  Since the
            # server runs a single model, the cache is always valid.
            if self.batch_generator is not None:
                n_entries = 0
                if self.memory_aware_cache is not None:
                    n_entries = len(self.memory_aware_cache._entries)
                elif self.prefix_cache is not None:
                    n_entries = (
                        len(self.prefix_cache)
                        if hasattr(self.prefix_cache, "__len__")
                        else 0
                    )
                logger.info(
                    f"[batch_generator] recreating (sampler params changed), "
                    f"keeping {n_entries} cache entries"
                )

            self._close_batch_generator()
            self.batch_generator = self._create_batch_generator(sampling_params)
            self._current_sampler_params = sampler_params

        return True

    def _validate_cache(self, cache: Any) -> bool:
        """
        Validate that a cache object is usable.

        Checks for None references AND shape compatibility.  Restored
        cache entries must have batch_size == 1 (single sequence) so
        they can be merged into the running batch by _merge_caches.
        A shape mismatch here (e.g. batch=2 from a stale entry) would
        cause a concatenation crash inside _merge_caches.

        Args:
            cache: The cache object to validate

        Returns:
            True if cache is valid and usable
        """
        if cache is None:
            return False

        # Check if it's a list of cache layers
        if isinstance(cache, list):
            if len(cache) == 0:
                return False
            # Check each layer
            for layer_cache in cache:
                if layer_cache is None:
                    return False
                # Check if layer has expected structure
                if hasattr(layer_cache, "keys") and layer_cache.keys is None:
                    return False
                if hasattr(layer_cache, "values") and layer_cache.values is None:
                    return False
                # Validate batch dimension == 1 for KVCache layers
                if hasattr(layer_cache, "keys") and layer_cache.keys is not None:
                    if (
                        hasattr(layer_cache.keys, "shape")
                        and layer_cache.keys.shape[0] != 1
                    ):
                        logger.debug(
                            f"Cache layer invalid: keys batch={layer_cache.keys.shape[0]}, expected 1"
                        )
                        return False
                # Validate batch dimension for MambaCache layers
                if hasattr(layer_cache, "cache") and isinstance(
                    layer_cache.cache, list
                ):
                    for arr in layer_cache.cache:
                        if arr is not None and arr.shape[0] != 1:
                            logger.debug(
                                f"Cache layer invalid: mamba batch={arr.shape[0]}, expected 1"
                            )
                            return False

        # Check BatchKVCache structure
        if hasattr(cache, "caches"):
            if cache.caches is None:
                return False
            for c in cache.caches:
                if c is None:
                    return False

        return True

    def _extract_cache_states(self, raw_cache: list[Any]) -> list[dict[str, Any]]:
        """
        Extract actual tensor state from each layer cache.

        This extracts the real KV data using mlx-lm's cache.state property,
        allowing the data to be stored and reconstructed later even after
        the BatchGenerator is recreated.

        Args:
            raw_cache: List of KVCache objects from mlx-lm

        Returns:
            List of dicts with {state: (keys, values), meta_state: (offset,), class_name: str}
        """
        if not raw_cache:
            return []

        extracted = []
        for layer_cache in raw_cache:
            try:
                if hasattr(layer_cache, "state") and hasattr(layer_cache, "meta_state"):
                    state = layer_cache.state  # (keys, values) or more for Mamba
                    meta = layer_cache.meta_state  # (offset,) as strings
                    extracted.append(
                        {
                            "state": state,
                            "meta_state": meta,
                            "class_name": type(layer_cache).__name__,
                            "class_ref": type(layer_cache),
                        }
                    )
            except Exception as e:
                logger.debug(f"Failed to extract state from cache layer: {e}")
                continue

        return extracted if len(extracted) == len(raw_cache) else []

    def _reconstruct_cache_from_states(
        self, extracted_states: list[dict[str, Any]]
    ) -> list[Any] | None:
        """
        Reconstruct cache objects from extracted cache states.

        This is the inverse of _extract_cache_states(). Uses mlx-lm's
        _BaseCache.from_state() to reconstruct any cache type (KVCache,
        MambaCache, etc.) from its state/meta_state.

        Args:
            extracted_states: List of dicts from _extract_cache_states()

        Returns:
            List of cache objects, or None if reconstruction fails
        """
        if not extracted_states:
            return None

        try:
            caches = []
            for layer_state in extracted_states:
                state = layer_state.get("state")
                meta_state = layer_state.get("meta_state")
                cache_cls = layer_state.get("class_ref")
                if state is None:
                    return None

                if cache_cls is not None and hasattr(cache_cls, "from_state"):
                    # BatchKVCache doesn't inherit from KVCache, so
                    # _merge_caches can't handle it. Convert to KVCache
                    # (safe because mid-prefill save is always batch_size=1).
                    from mlx_lm.models.cache import (
                        BatchKVCache as _BatchKVCache,
                    )
                    from mlx_lm.models.cache import (
                        KVCache as _KVCache,
                    )

                    from vllm_mlx.quantized_batch_cache import (
                        QuantizedBatchKVCache as _QuantizedBatchKVCache,
                    )

                    if cache_cls is _QuantizedBatchKVCache:
                        # state = (k_triple, v_triple, offset, left_padding).
                        # Mid-prefill save is always batch_size=1, so dequantize
                        # the quantized triples to a plain bf16 KVCache (#1197).
                        cache = _KVCache()
                        if state[0] is None:
                            # Empty cache: QuantizedBatchKVCache.state returns
                            # (None, None, ...) before its first write. Restore an
                            # empty KVCache rather than dequantizing None.
                            cache.offset = 0
                        else:
                            from vllm_mlx.quantized_batch_cache import _dequantize

                            gs, bits = (
                                (int(meta_state[0]), int(meta_state[1]))
                                if meta_state
                                else (64, 8)
                            )
                            keys = _dequantize(state[0], gs, bits)
                            values = _dequantize(state[1], gs, bits)
                            cache.keys = keys
                            cache.values = values
                            cache.offset = keys.shape[2]
                    elif cache_cls is _BatchKVCache:
                        # BatchKVCache.state = (keys, values, offset, left_padding)
                        keys, values = state[0], state[1]
                        cache = _KVCache()
                        cache.keys = keys
                        cache.values = values
                        cache.offset = keys.shape[2] if hasattr(keys, "shape") else 0
                    elif cache_cls.__name__ == "CacheList":
                        # CacheList.from_state resolves nested class names only
                        # from mlx_lm.models.cache globals. Vendored DeepSeek
                        # pooling caches intentionally do not live there, so
                        # reconstruct the wrapper with an explicit local map.
                        from mlx_lm.models import cache as _mlx_cache
                        from mlx_lm.models.cache import CacheList as _CacheList

                        from vllm_mlx.models.deepseek_v4_cache import (
                            BatchDeepseekV4PoolingCache,
                            BatchPoolingCache,
                            DeepseekV4PoolingCache,
                            PoolingCache,
                        )

                        vendored = {
                            cls.__name__: cls
                            for cls in (
                                PoolingCache,
                                BatchPoolingCache,
                                DeepseekV4PoolingCache,
                                BatchDeepseekV4PoolingCache,
                            )
                        }
                        names, nested_meta = meta_state
                        if not (len(state) == len(names) == len(nested_meta)):
                            raise ValueError(
                                "CacheList state/metadata length mismatch: "
                                f"state={len(state)}, names={len(names)}, "
                                f"metadata={len(nested_meta)}"
                            )
                        nested = []
                        for nested_state, name, nested_m in zip(
                            state, names, nested_meta
                        ):
                            nested_cls = getattr(
                                _mlx_cache, name, None
                            ) or vendored.get(name)
                            if nested_cls is None:
                                raise ValueError(f"Unknown nested cache class {name!r}")
                            nested.append(nested_cls.from_state(nested_state, nested_m))
                        cache = _CacheList(*nested)
                    else:
                        cache = cache_cls.from_state(state, meta_state)
                else:
                    # Fallback: try KVCache manual reconstruction
                    from mlx_lm.models.cache import KVCache

                    if len(state) != 2:
                        return None
                    cache = KVCache()
                    cache.keys, cache.values = state
                    cache.offset = (
                        int(meta_state[0])
                        if meta_state
                        else (
                            cache.keys.shape[2] if hasattr(cache.keys, "shape") else 0
                        )
                    )

                caches.append(cache)

            return caches

        except Exception as e:
            logger.info(f"[mid_prefill_cache] reconstruct EXCEPTION: {e}")
            return None

    def _resolve_metal_cap_bytes(self) -> int:
        """Compute the admission-time Metal cap in bytes (cached after first call).

        D-METAL-CAP root cause: ``mx.set_memory_limit`` is documented as
        a *guideline* — MLX will silently grow past the limit while
        system RAM is available. On a 256 GB M3 Ultra with
        ``--gpu-memory-utilization 0.45`` (≈ 115 GB cap) the user saw
        Metal active grow to 179 GB on a single 32k prefill with no
        warning. This helper materializes the same per-device cap the
        BatchedEngine boot path uses for ``mx.set_memory_limit`` so the
        scheduler can enforce it at admission with no race against the
        allocator's leniency window.

        Returns ``0`` when the cap should be considered disabled (the
        SchedulerConfig default ``gpu_memory_utilization=0.0``, or the
        Metal device probe failed). Callers MUST treat ``0`` as "do not
        check" rather than "no headroom".
        """
        if self._metal_cap_bytes_resolved:
            return self._metal_cap_bytes
        cap = 0
        util = float(getattr(self.config, "gpu_memory_utilization", 0.0) or 0.0)
        if util > 0.0:
            try:
                if mx.metal.is_available():
                    info = mx.device_info()
                    base = info.get(
                        "max_recommended_working_set_size",
                        info.get("memory_size", 0),
                    )
                    if base and base > 0:
                        cap = int(base * util)
            except Exception:
                cap = 0
        self._metal_cap_bytes = cap
        self._metal_cap_bytes_resolved = True
        return cap

    def _current_metal_active_bytes(self) -> int:
        """Best-effort snapshot of MLX-reported Metal active memory.

        Wrapped in try/except so a non-Metal host (CI, Linux GPU shim,
        unit-test fake) doesn't take down the admission path.
        """
        try:
            return int(mx.get_active_memory())
        except Exception:
            return 0

    def _current_process_resident_bytes(self) -> int:
        """Best-effort process footprint for unified-memory pressure.

        MLX active memory omits Python objects, token buffers and some Metal
        driver allocations. macOS ``phys_footprint`` is the authoritative
        kernel ledger; RSS is retained as a portable fallback for Linux CI.
        """
        try:
            from .runtime.process_memory import get_phys_footprint

            footprint = get_phys_footprint()
            if footprint > 0:
                return footprint
        except Exception:
            pass
        try:
            import psutil

            return int(psutil.Process().memory_info().rss)
        except Exception:
            return 0

    def _active_prefill_token_count(self) -> int:
        """Return the largest final context offset currently prefetched.

        Workspace at a long attention offset can be large even when a prefix
        hit leaves only a tiny suffix. Count cached/all-token history plus the
        remaining segments, not merely the suffix workload.
        """
        bg = getattr(self, "batch_generator", None)
        processing = getattr(bg, "_currently_processing", ()) if bg else ()
        prompt_batch = getattr(bg, "_prompt_batch", None) if bg else None
        prior_tokens = getattr(prompt_batch, "tokens", ()) if prompt_batch else ()
        largest = 0
        for index, item in enumerate(processing):
            try:
                already_cached = (
                    len(prior_tokens[index]) if index < len(prior_tokens) else 0
                )
                processed = int(item[1])
                remaining = max(0, int(item[2]) - processed)
                largest = max(largest, already_cached + remaining)
            except (IndexError, TypeError, ValueError):
                continue
        # Requests have not entered ``_currently_processing`` on their first
        # scheduler tick yet. Include queued BatchGenerator segments so the
        # first cold chunk is guarded too.
        queued = getattr(bg, "_unprocessed_sequences", ()) if bg else ()
        for item in queued:
            try:
                largest = max(
                    largest,
                    len(item[4]) + sum(len(seg) for seg in item[1]),
                )
            except (IndexError, TypeError):
                continue
        # mlx-lm may represent a prefix-cache hit as a fresh prompt batch
        # containing only the uncached suffix. In that transition window
        # ``prior_tokens`` therefore understates the resident KV offset. The
        # public Request retains the full model prompt length; use it while a
        # prefill queue is actually active, but never during decode (where it
        # would keep the low cache ceiling armed for the whole completion).
        if processing or queued:
            for request in getattr(self, "running", {}).values():
                try:
                    largest = max(
                        largest,
                        int(request.model_prompt_tokens or request.num_prompt_tokens),
                    )
                except (AttributeError, TypeError, ValueError):
                    continue
        return largest

    def _select_adaptive_prefill_size(self) -> int:
        """Choose a memory-safe size for the next long-prompt chunk.

        The thresholds intentionally start tightening at 70%: the 97k-token
        DeepSeek-V4 repro showed a roughly 50 GB transient gap between steady
        active memory and the prefill peak, so waiting until 90% is too late.
        """
        configured = max(1, int(getattr(self.config, "prefill_step_size", 2048)))
        if not getattr(self.config, "adaptive_prefill", True):
            return configured
        minimum_prompt = max(
            1, int(getattr(self.config, "adaptive_prefill_min_tokens", 32_768))
        )
        if self._active_prefill_token_count() < minimum_prompt:
            return configured
        cap = self._resolve_metal_cap_bytes()
        if cap <= 0:
            return configured
        footprint = max(
            self._current_metal_active_bytes(), self._current_process_resident_bytes()
        )
        ratio = footprint / cap
        floor = max(
            1, int(getattr(self.config, "adaptive_prefill_min_chunk_size", 256))
        )
        if ratio >= 0.88:
            target = 256
        elif ratio >= 0.80:
            target = 512
        elif ratio >= 0.70:
            target = 1024
        else:
            target = configured
        return min(configured, max(floor, target))

    def _apply_adaptive_prefill_size(self) -> int:
        """Apply the selected size to mlx-lm's current and future batches."""
        bg = getattr(self, "batch_generator", None)
        if bg is None:
            return 0
        configured = max(1, int(getattr(self.config, "prefill_step_size", 2048)))
        if not getattr(self.config, "adaptive_prefill", True):
            bg.prefill_step_size = configured
            prompt_batch = getattr(bg, "_prompt_batch", None)
            if prompt_batch is not None:
                prompt_batch.prefill_step_size = configured
            self._last_adaptive_prefill_size = configured
            return configured
        selected = self._select_adaptive_prefill_size()
        prompt_tokens = self._active_prefill_token_count()
        minimum_prompt = max(
            1, int(getattr(self.config, "adaptive_prefill_min_tokens", 32_768))
        )
        if prompt_tokens >= minimum_prompt:
            # Pressure samples can wobble around a threshold as allocator
            # slabs are materialized and released. Growing the chunk again in
            # the same prefill causes oscillation and can recreate the peak we
            # just avoided. A completed prompt naturally restores configured
            # size on the next decode/idle tick.
            previous = getattr(self, "_last_adaptive_prefill_size", configured)
            selected = min(selected, previous)
            self._adaptive_prefill_protected_chunks += 1
            if selected < configured:
                self._adaptive_prefill_reduced_chunks += 1
        bg.prefill_step_size = selected
        prompt_batch = getattr(bg, "_prompt_batch", None)
        if prompt_batch is not None:
            prompt_batch.prefill_step_size = selected
        previous = getattr(self, "_last_adaptive_prefill_size", None)
        if previous != selected:
            logger.info(
                "[adaptive_prefill] chunk %s -> %s tokens prompt=%s "
                "metal=%.1fGB rss=%.1fGB cap=%.1fGB",
                previous,
                selected,
                prompt_tokens,
                self._current_metal_active_bytes() / 1e9,
                self._current_process_resident_bytes() / 1e9,
                self._resolve_metal_cap_bytes() / 1e9,
            )
            self._last_adaptive_prefill_size = selected
        return selected

    def _infer_kv_dtype_bytes(self, model_config: Any) -> int:
        """Best-effort KV-cache dtype-bytes inference.

        Codex round 5 BLOCKING #2: returns the size in bytes of the
        KV-cache element dtype. Falls back to ``4`` (fp32) ONLY when the
        dtype genuinely cannot be determined — over-estimating is the
        safe direction for a TRULY unknown dtype (admission rejects a
        borderline request rather than letting it slip past the cap).

        Reads the element dtype from, in priority order:
          1. ``dtype`` — the MODERN HuggingFace/transformers key.
             ``torch_dtype`` was renamed to ``dtype`` upstream, so newer
             configs (e.g. Gemma 4) carry ONLY ``dtype``.
          2. ``torch_dtype`` — the legacy key, still emitted by older
             configs.
          3. The same two keys on a nested ``text_config`` — multimodal
             configs (Gemma 4, Qwen-VL) nest the language-model config
             there and may leave the top level without a usable dtype.

        Why this matters (gemma4-on-18GB false-rejection bug): reading
        ONLY ``torch_dtype`` meant a config that uses the modern
        ``dtype`` key fell through to the fp32 fallback, DOUBLING the
        projected KV (bf16 2 B/elem mis-read as fp32 4 B/elem) and
        rejecting at admission a request whose real usage fits under the
        cap. An MLX fp16/bf16 model's KV cache is 2 bytes/elem; fp32 KV
        is not a real MLX deployment, so the fallback should be the rare
        last resort, not the default for every modern config.

        Quantized KV-cache deployments are not auto-detected — the
        operator-tuned ``metal_cap_kv_bytes_per_token`` knob is the
        right escape hatch for those.
        """
        mapping = {
            "float64": 8,
            "fp64": 8,
            "double": 8,
            "float32": 4,
            "fp32": 4,
            "float16": 2,
            "fp16": 2,
            "half": 2,
            "bfloat16": 2,
            "bf16": 2,
            "int8": 1,
            "uint8": 1,
            "float8": 1,
            "fp8": 1,
        }

        def _bytes_from(obj: Any) -> int:
            # ``dtype`` (modern) and ``torch_dtype`` (legacy) may each be
            # a string (``"bfloat16"``) or a ``torch.dtype`` whose
            # ``str()`` is e.g. ``"torch.bfloat16"``. Note ``"float16"``
            # is a substring of ``"bfloat16"`` — harmless here since both
            # map to 2. Read through ``_cfg_get`` so a dict-backed config
            # resolves its dtype identically to an attribute config (codex
            # round 11 NIT: dict configs must behave the same end-to-end,
            # else they silently over-estimate at the fp32 fallback).
            for attr in ("dtype", "torch_dtype"):
                raw = _cfg_get(obj, attr)
                if raw is None:
                    continue
                dtype_str = str(raw).lower()
                for needle, n in mapping.items():
                    if needle in dtype_str:
                        return n
            return 0

        try:
            n = _bytes_from(model_config)
            if n > 0:
                return n
            # Multimodal configs nest the LM dtype under ``text_config``.
            text_config = _cfg_get(model_config, "text_config")
            if text_config is not None:
                n = _bytes_from(text_config)
                if n > 0:
                    return n
        except Exception:
            pass
        # Default: assume the LARGEST plausible dtype (fp32 = 4) so we
        # over-estimate KV usage and err toward rejection rather than
        # admitting a request that exceeds the cap. Reached only when
        # NEITHER ``dtype`` nor ``torch_dtype`` is present on the config
        # or its ``text_config`` — rare on real HF/MLX configs.
        return 4

    def _resolve_kv_bytes_per_token(self) -> int:
        """Compute the per-token KV-cache size (cached after first call).

        Codex round 4 BLOCKING #1+#2 closure: the operator-tuned
        ``metal_cap_kv_bytes_per_token`` is still honored when
        explicitly set, but when it is 0 (default) we auto-derive a
        conservative estimate from the model config so the projection-
        based admission gate works OUT OF THE BOX without operators
        having to thread a per-model knob. Pre-fix, defaulting to 0
        meant the projection branch was effectively dead code unless
        operators set the field — which contradicted the PR's claim
        to fix the "currently below cap, one large prefill allocates
        past cap" failure mode by default.

        Auto-derivation formula (uniform baseline):
            ``2 (K + V) × num_layers × num_kv_heads × head_dim × dtype_bytes``

        ``dtype_bytes=2`` (fp16) is the conservative default for the
        dominant case; 8-bit / 4-bit KV-cache deployments OVER-estimate,
        which is the safe direction (a 4-bit user pays the price of an
        admission rejection at half the actual cap headroom — still
        better than the D-METAL-CAP cliff). Operators on quantized-KV
        deployments can pin a tighter value via the SchedulerConfig field
        to recover precision.

        Architecture-aware refinement (``kv_estimation``): the uniform
        formula counts EVERY layer as a full-growth attention layer,
        which over-estimates the per-token figure by up to ~2× for the
        hybrid architectures we ship — sliding-window layers (GPT-OSS,
        Gemma-4 local) are window-bounded, KV-sharing borrower layers
        (Gemma-4 ``num_kv_shared_layers``) allocate nothing, and
        recurrent / linear-attention layers (Qwen3.5 GatedDeltaNet)
        carry a fixed state. Once the uniform figure is computed we
        route it through :func:`kv_estimation.estimate_kv_footprint`,
        which decomposes the real footprint into THREE terms that the
        admission projection recombines per request:

          1. per-token GROWTH bytes — the full-attention layers only;
             the one term that scales linearly and unbounded with
             ``prompt + max_tokens``.
          2. a per-request SLIDING term — the sliding-window layers,
             charged as ``sliding_slot_bytes ×
             rotating_cache_slots(window, tokens)``. This is
             request-dependent and SUBLINEAR: it grows in ``step``
             blocks up to the window, then stays flat, so short
             requests are not charged the whole window. Cached here as
             ``_kv_sliding_slot_bytes`` / ``_kv_sliding_window`` and
             applied in :meth:`_estimate_request_kv_bytes`.
          3. a per-sequence FIXED baseline — recurrent /
             linear-attention state ONLY (a constant that does not grow
             with sequence length). Note: window buffers are NOT in the
             baseline; they live in term 2.

        The estimator is over-estimate-safe: a dense model or an
        unknown/stub config that exposes no ``layer_types`` /
        ``num_kv_shared_layers`` yields a BYTE-IDENTICAL uniform result
        (per-token unchanged, sliding + baseline both 0), so only
        recognized hybrids get the smaller, accurate number and the
        D-METAL-CAP OOM cliff is never re-introduced.

        Returns ``0`` only when the model config is missing entirely
        (e.g. unit-test ``MagicMock`` model) so back-compat unit
        tests that build a Scheduler against a stub model don't
        suddenly start rejecting requests that previously admitted.
        """
        if self._kv_bytes_per_token_resolved:
            return self._kv_bytes_per_token
        # Operator override wins — it short-circuits ALL architecture-aware
        # refinement, so the fixed baseline and the sliding term are both zero
        # (the operator's flat per-token figure is the whole projection).
        configured = int(getattr(self.config, "metal_cap_kv_bytes_per_token", 0) or 0)
        if configured > 0:
            self._kv_bytes_per_token = configured
            self._kv_fixed_baseline_bytes = 0
            self._kv_sliding_slot_bytes = 0
            self._kv_sliding_window = 0
            self._kv_bytes_per_token_resolved = True
            return configured
        # Auto-derive directly from model.config. Defensive
        # ``isinstance(..., int)`` filtering keeps a MagicMock model from
        # returning mock objects that produce a phantom positive
        # estimate. Pre-fix this was a real surprise during testing:
        # ``int(MagicMock())`` coerces to ``1``, so a stub model
        # yielded a 4-byte-per-token estimate that turned every
        # unit-test admission into a projection rejection. Requiring
        # ints filters that path.
        per_tok = 0
        fixed_baseline = 0
        sliding_slot_bytes = 0
        sliding_window = 0
        try:
            model_config = getattr(self.model, "config", None)
            if model_config is not None:
                # Multimodal architectures nest the text-tower dims
                # (num_hidden_layers, num_key_value_heads, head_dim, layer_types)
                # under ``text_config``, and the OUTER config may still carry a
                # DECOY ``num_hidden_layers`` describing the vision tower. Read
                # the base dims from the config that actually carries the
                # text-layer structure — the SAME rule ``_pick_structural_config``
                # uses: pick the first of (outer, text_config) whose OWN
                # ``layer_types`` is valid and length-matches its OWN
                # ``num_hidden_layers`` (codex round 8 + round 9 BLOCKING #1).
                # Reusing ``_valid_layer_types`` keeps the scheduler's selection
                # from drifting away from the estimator's.
                #
                # BYTE-IDENTICAL guard: if NEITHER config exposes a valid
                # ``layer_types`` (a dense / unknown model), fall back to the
                # outer config exactly as before — dense/text admission
                # accounting is unchanged.
                #
                # Read every field through the estimator's dict-aware
                # ``_cfg_get`` (not bare ``getattr``): ``estimate_kv_footprint``
                # explicitly supports dict-backed configs (the offline hybrid
                # probe parses ``config.json`` into a dict), so the scheduler
                # wiring must resolve dict and attribute configs identically —
                # bare ``getattr`` on a dict returns ``None`` for every field
                # and silently collapses the whole estimate to 0 (codex round 11
                # NIT: dict-backed configs produced a zero estimate through the
                # scheduler even though the estimator handles them).
                struct_config = model_config
                for _candidate in (
                    model_config,
                    _cfg_get(model_config, "text_config"),
                ):
                    if _candidate is None:
                        continue
                    _cand_layers = _cfg_get(_candidate, "num_hidden_layers")
                    if isinstance(_cand_layers, int) and _valid_layer_types(
                        _cfg_get(_candidate, "layer_types"), _cand_layers
                    ):
                        struct_config = _candidate
                        break

                def _read_int(name: str, fallback: int = 0) -> int:
                    raw = _cfg_get(struct_config, name, fallback)
                    return raw if isinstance(raw, int) else 0

                num_layers = _read_int("num_hidden_layers")
                num_kv_heads = _read_int("num_key_value_heads")
                if num_kv_heads <= 0:
                    num_kv_heads = _read_int("num_attention_heads")
                head_dim = _read_int("head_dim")
                if head_dim <= 0:
                    hidden_size = _read_int("hidden_size")
                    num_heads = _read_int("num_attention_heads")
                    if num_heads > 0:
                        head_dim = hidden_size // num_heads
                # Codex round 5 BLOCKING #2: derive ``dtype_bytes``
                # from the model dtype when available. The pre-fix
                # constant ``2`` (fp16) underestimated fp32 KV
                # caches by 2× and could admit requests that exceed
                # the cap despite the projection guard. Fallback is
                # ``4`` (the largest plausible dtype: fp32) — over-
                # estimating in the safe direction. Operators on
                # quantized-KV deployments can still pin a tighter
                # value via ``metal_cap_kv_bytes_per_token``.
                dtype_bytes = self._infer_kv_dtype_bytes(model_config)
                if num_layers > 0 and num_kv_heads > 0 and head_dim > 0:
                    per_tok = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
                    # Architecture-aware refinement. For dense / unknown
                    # configs the estimator returns this exact uniform
                    # ``per_tok`` with a 0 baseline and no sliding term
                    # (byte-identical); only a recognized hybrid (sliding-window
                    # / KV-sharing / recurrent layers) reduces the per-token
                    # growth, reserves recurrent state in the per-sequence fixed
                    # baseline, and exposes the sliding layers as a
                    # request-dependent slot term
                    # (``sliding_slot_bytes`` × ``rotating_cache_slots(window,
                    # T)`` — charged in ``_estimate_request_kv_bytes``). Wrapped
                    # in its OWN try/except so a failure inspecting the newer
                    # hybrid fields degrades to the SAFE uniform ``per_tok`` (no
                    # baseline, no sliding term) instead of the outer handler
                    # resetting to 0 — which would disable admission accounting
                    # entirely (codex round 3 BLOCKING #3).
                    try:
                        estimate = estimate_kv_footprint(
                            struct_config,
                            dtype_bytes=dtype_bytes,
                            uniform_per_token_bytes=per_tok,
                            base_num_layers=num_layers,
                            base_kv_heads=num_kv_heads,
                            base_head_dim=head_dim,
                        )
                        per_tok = estimate.per_token_growth_bytes
                        fixed_baseline = estimate.fixed_baseline_bytes
                        sliding_slot_bytes = estimate.sliding_slot_bytes
                        sliding_window = estimate.sliding_window
                    except Exception as est_err:
                        logger.debug(
                            "[D-METAL-CAP] architecture-aware KV estimate "
                            "failed (%s); keeping the uniform per-token "
                            "estimate with no fixed baseline / sliding term",
                            est_err,
                        )
                        # ``per_tok`` already holds the uniform value; keep it.
                        fixed_baseline = 0
                        sliding_slot_bytes = 0
                        sliding_window = 0
        except Exception as e:
            logger.debug(
                "[D-METAL-CAP] failed to auto-derive kv_bytes_per_token "
                "from model.config (%s); admission projection will use "
                "0 — operator can set "
                "SchedulerConfig.metal_cap_kv_bytes_per_token explicitly",
                e,
            )
            per_tok = 0
            fixed_baseline = 0
            sliding_slot_bytes = 0
            sliding_window = 0
        self._kv_sliding_slot_bytes = sliding_slot_bytes
        self._kv_sliding_window = sliding_window
        self._kv_bytes_per_token = per_tok
        self._kv_fixed_baseline_bytes = fixed_baseline
        self._kv_bytes_per_token_resolved = True
        return per_tok

    def _resolve_kv_fixed_baseline_bytes(self) -> int:
        """Per-sequence FIXED KV baseline (bytes) for hybrid architectures.

        Companion to :meth:`_resolve_kv_bytes_per_token`: the token-independent
        footprint an architecture-aware hybrid allocates ONCE per sequence (not
        per token) — the fixed recurrent state of each sizeable recurrent
        (GatedDeltaNet) layer. Sliding-window buffers are NOT here: they are
        request-dependent and charged via the sliding slot term in
        ``_estimate_request_kv_bytes``. ``0`` for dense models, for the uniform
        fallback, and under the operator override — so the projection stays
        byte-identical wherever the baseline is not populated. Resolution and
        caching happen inside ``_resolve_kv_bytes_per_token`` (all terms are
        computed from the same estimate), so we drive it first, then read the
        cached baseline.
        """
        self._resolve_kv_bytes_per_token()
        return self._kv_fixed_baseline_bytes

    def projected_memory_max_context(
        self, native_context: int | None = None
    ) -> int | None:
        """Largest context length whose projected KV footprint fits this
        device's memory budget right now, or ``None`` when it can't be
        estimated.

        This is the number behind the ``max_model_len`` model-card field:
        vLLM/SGLang expose ``max_model_len`` as the served ceiling but crash
        if it won't fit KV memory; on unified-memory Apple silicon we can go
        one better and REPORT the memory-fitted ceiling instead of aborting.

        The footprint is the exact per-request projection the admission gate
        uses (``_estimate_request_kv_bytes``), so the advertised number lines
        up with what the scheduler would actually admit::

            footprint(T) = fixed_baseline
                           + per_token_growth * T
                           + sliding_slot_bytes * rotating_cache_slots(window, T)

        Budget = device working set × utilization − currently-resident bytes.
        Utilization is the operator's ``gpu_memory_utilization`` when a Metal
        cap is configured, else a conservative reporting default so the field
        is populated on the default (cap-disabled) config too. Residency is a
        point-in-time ``get_active_memory()`` reading — weights plus any live
        KV/prefix cache — so the number reflects what could be added NOW; it
        is advisory, re-derived per call, and capped at ``native_context``.

        Returns ``None`` (not a guess) on any failure — a stub/unknown model
        with no resolvable footprint, no Metal device, or a non-positive
        budget — so the ``/v1/models`` builder falls through to leaving the
        field absent rather than advertising a fabricated cap.
        """
        try:
            # Prefer the scheduler's OWN resolved footprint terms so the
            # advertised ceiling can never diverge from what admission would
            # actually charge: these honor the operator
            # ``metal_cap_kv_bytes_per_token`` override and use
            # ``_pick_structural_config`` to select the right tower on
            # multimodal/hybrid configs. They are populated whenever the model
            # exposes a ``.config`` (the admission path's own source).
            per_tok = self._resolve_kv_bytes_per_token()
            fixed_baseline = self._resolve_kv_fixed_baseline_bytes()
            sliding_slot_bytes = self._kv_sliding_slot_bytes
            sliding_window = self._kv_sliding_window
            if per_tok <= 0 and fixed_baseline <= 0 and sliding_slot_bytes <= 0:
                # The scheduler could not resolve terms — mlx-lm models expose
                # dims on ``.args``, not ``.config``, so its config-only read
                # yields 0 (and the admission cap is likewise a no-op there, so
                # there is nothing to diverge from). Fall back to reading the
                # dims off the model and running the SAME hybrid-aware
                # estimator, preferring the text tower over any decoy outer
                # config (``_read_kv_dims``).
                dims = _read_kv_dims(self.model)
                if dims is None:
                    return None
                num_layers, kv_heads, head_dim, struct_cfg = dims
                dtype_bytes = self._infer_kv_dtype_bytes(struct_cfg)
                uniform_per_token = 2 * num_layers * kv_heads * head_dim * dtype_bytes
                if uniform_per_token <= 0:
                    return None
                estimate = estimate_kv_footprint(
                    struct_cfg,
                    dtype_bytes=dtype_bytes,
                    uniform_per_token_bytes=uniform_per_token,
                    base_num_layers=num_layers,
                    base_kv_heads=kv_heads,
                    base_head_dim=head_dim,
                )
                per_tok = estimate.per_token_growth_bytes
                fixed_baseline = estimate.fixed_baseline_bytes
                sliding_slot_bytes = estimate.sliding_slot_bytes
                sliding_window = estimate.sliding_window
                if per_tok <= 0 and fixed_baseline <= 0 and sliding_slot_bytes <= 0:
                    return None

            if not mx.metal.is_available():
                return None
            info = mx.device_info()
            base = int(
                info.get("max_recommended_working_set_size", info.get("memory_size", 0))
                or 0
            )
            if base <= 0:
                return None

            util = float(getattr(self.config, "gpu_memory_utilization", 0.0) or 0.0)
            if util <= 0.0:
                # The Metal admission cap is disabled by default, but the
                # reporting field should still be populated. 0.90 mirrors the
                # engine's allocation-side default working-set fraction.
                util = 0.90
            budget = int(base * util)
            resident = int(mx.get_active_memory())
            available = budget - resident
            if available <= 0:
                return None

            def _footprint(tokens: int) -> int:
                return (
                    fixed_baseline
                    + per_tok * tokens
                    + sliding_slot_bytes * rotating_cache_slots(sliding_window, tokens)
                )

            # Search up to the native window when known (the field is capped
            # there anyway), else a generous ceiling covering 1M-context models.
            upper = (
                native_context
                if isinstance(native_context, int) and native_context > 0
                else 8_000_000
            )
            if _footprint(upper) <= available:
                # Memory is not the binding constraint — the native window fits.
                return upper
            # Monotonic non-decreasing footprint → binary-search the largest T.
            lo, hi = 0, upper
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if _footprint(mid) <= available:
                    lo = mid
                else:
                    hi = mid - 1
            return lo if lo > 0 else None
        except Exception:  # noqa: BLE001
            # Never let a memory-probe failure break the models endpoint.
            return None

    def _estimate_request_kv_bytes(self, request: Request) -> int:
        """Project KV-cache memory the new request would consume.

        With ``T = num_prompt_tokens + max_tokens`` this returns::

            fixed_baseline
              + per_token_growth * T
              + sliding_slot_bytes * rotating_cache_slots(sliding_window, T)

        (auto-derived from model config or operator-overridden via
        ``metal_cap_kv_bytes_per_token``). Used by the admission gate to reject
        prefill requests that would push Metal active PAST the cap before the
        allocation happens (codex round 3 BLOCKING #1 + round 4 BLOCKING #1+#2 +
        round 9/10/11).

        The three terms model the three footprint shapes:

        * ``per_token_growth`` — the unbounded full-attention layers, growing
          with every token → multiplied by ``T``.
        * ``sliding_slot_bytes`` — the window-bounded sliding layers. Their
          buffer grows SUBLINEARLY (up to a rotating window, then flat), so we
          multiply the per-slot bytes by ``rotating_cache_slots(window, T)`` — an
          over-count-safe upper bound of the real ``RotatingKVCache`` allocation
          that rises with ``T`` and caps at the full window. Charging per request
          (not a flat whole-window baseline) stops a SHORT request from being
          over-charged the whole buffer and spuriously rejected (codex round 11
          BLOCKING #1), while a long request still reserves the full window.
        * ``fixed_baseline`` — the token-independent recurrent state.

        For a dense model the last two terms are 0 and this reduces to the
        historical ``per_tok × T`` — byte-identical. The sum is always >= the
        true architecture-aware footprint of the counted layers, so the gate can
        only over-estimate, never under-count.

        Conservative by design: we count both the prompt and the
        full ``max_tokens`` budget (rather than the much smaller
        first-step prefill), so the gate errs toward rejecting a
        borderline request rather than letting it slip through and
        grow past the cap mid-generation.
        """
        per_tok = self._resolve_kv_bytes_per_token()
        fixed_baseline = self._resolve_kv_fixed_baseline_bytes()
        sliding_slot_bytes = self._kv_sliding_slot_bytes
        sliding_window = self._kv_sliding_window
        if per_tok <= 0 and fixed_baseline <= 0 and sliding_slot_bytes <= 0:
            return 0
        # ``num_prompt_tokens`` is populated by either the route layer
        # (when prompt_token_ids was supplied) or zero at this point
        # (tokenization runs AFTER admission). Fall back to the length
        # of the raw prompt as a best-effort proxy in the zero case so
        # the cap still bites on a 32k-string prompt that has not been
        # tokenized yet.
        prompt_tokens = int(getattr(request, "num_prompt_tokens", 0) or 0)
        if prompt_tokens <= 0:
            raw_prompt = getattr(request, "prompt_token_ids", None)
            if raw_prompt is not None:
                prompt_tokens = len(raw_prompt)
            else:
                raw = getattr(request, "prompt", "")
                # Codex round 6 HIGH #1: ``len(str)`` counts Python
                # code points, NOT tokenizer tokens. For ASCII English
                # this OVER-estimates (3–5 chars/token typical), but
                # adversarial inputs can UNDER-estimate:
                # - byte-fallback BPE turns a single code point like
                #   ``💀`` (1 char, 4 UTF-8 bytes) into 4 byte tokens
                # - sentencepiece on rare CJK glyphs can emit 2+
                #   tokens per code point
                # Both reintroduce the D-METAL-CAP under-estimate path
                # codex flagged. The byte-length of the UTF-8 encoding
                # is a strict upper bound for every byte-level
                # tokenizer (one byte ≥ one byte token) and a safe
                # ceiling for SentencePiece (worst-case 1 byte → 1
                # token). Lists and tuples we already trust as token
                # IDs.
                if isinstance(raw, str):
                    try:
                        prompt_tokens = len(raw.encode("utf-8"))
                    except (UnicodeError, AttributeError):
                        prompt_tokens = len(raw)
                elif isinstance(raw, (list, tuple, bytes, bytearray)):
                    prompt_tokens = len(raw)
        max_tokens = int(
            getattr(getattr(request, "sampling_params", None), "max_tokens", 0) or 0
        )
        tokens = prompt_tokens + max_tokens
        # Sliding-window term: per-slot bytes × the request's slot count, an
        # over-count-safe upper bound of the real RotatingKVCache allocation that
        # grows with ``tokens`` up to the full window and then caps (0 when there
        # is no window-bounded sliding layer). Charged per request so a short
        # request is not over-charged the whole window.
        sliding_bytes = 0
        if sliding_slot_bytes > 0 and sliding_window > 0:
            sliding_bytes = sliding_slot_bytes * rotating_cache_slots(
                sliding_window, tokens
            )
        # Fixed baseline (recurrent state) charged once, plus the unbounded
        # per-token growth of the full-attention layers over the projected token
        # budget, plus the request-dependent sliding term. For a dense model the
        # baseline and sliding term are 0, so this is byte-identical to the
        # historical ``per_tok × tokens``.
        return fixed_baseline + per_tok * tokens + sliding_bytes

    def _sum_in_flight_kv_bytes(self) -> int:
        """Sum projected KV reservations of WAITING-only requests.

        Codex round 5 BLOCKING #1: ``mx.get_active_memory()`` only
        reflects the allocator AT THE INSTANT we read it — admitted
        requests that have not yet been picked up by the BatchGenerator
        contribute 0 to ``active`` even though they will allocate KV
        on their first step. Without including their reservations,
        a burst of concurrent admits each individually under cap will
        STACK and blow the cap collectively (the multi-client repro
        path).

        Codex round 6 BLOCKING #3: critically, we must EXCLUDE
        ``self.running`` requests — their KV has already been
        allocated by the BatchGenerator and is therefore already
        counted in ``mx.get_active_memory()``. Including them would
        double-count and reject all new admits after even ONE
        in-flight large request, when real Metal headroom is fine.
        Only ``self.waiting`` (admitted but never stepped) contains
        reservations not yet visible in ``active``.

        Cheap: dict iteration + arithmetic only — no Metal device
        probe, no lock acquisition (the caller already holds the
        scheduler lock).
        """
        per_tok = self._resolve_kv_bytes_per_token()
        fixed_baseline = self._resolve_kv_fixed_baseline_bytes()
        sliding_slot_bytes = self._kv_sliding_slot_bytes
        # Cheap short-circuit only: a hybrid with all-sliding layers has zero
        # per-token growth and zero fixed baseline but a real per-request sliding
        # term, so ALL THREE must be 0 (dense MagicMock / no config) to skip. The
        # per-request charge itself — including the sliding term once per waiting
        # request — is computed by ``_estimate_request_kv_bytes`` in the loop
        # below, NOT here, so no term is ever dropped from the aggregate.
        if per_tok <= 0 and fixed_baseline <= 0 and sliding_slot_bytes <= 0:
            return 0
        try:
            waiting = self.waiting
        except AttributeError:
            return 0
        total = 0
        for req in waiting:
            # ``_estimate_request_kv_bytes`` returns the full per-request
            # projection (baseline + per_tok × tokens + sliding term) — each
            # waiting sequence allocates its own buffers.
            total += self._estimate_request_kv_bytes(req)
        return int(total)

    def _enforce_metal_cap_at_admission(self, request: Request) -> None:
        """D-METAL-CAP: reject the request if Metal active exceeds the cap.

        Runs as the second admission check in ``add_request`` —
        immediately after the concurrent-requests cap and BEFORE any
        tokenization / prefix-cache lookup so the rejection cost is a
        single ``mx.get_active_memory`` syscall plus a dict lookup.

        Two-stage check (codex round 3 BLOCKING #1 closure):
        1. Cheap path — ``active >= cap`` rejects when the allocator
           is ALREADY over budget. This is the original guard and
           covers sustained over-cap storms.
        2. Projection path — when
           ``metal_cap_kv_bytes_per_token > 0`` and the
           ``active + projected_kv >= cap``, reject the request
           BEFORE its prefill grows active past the cap. Without this
           leg, a single large 32k-prefill admitted while current
           active sits at e.g. 60% of cap could allocate the
           remaining 70% and still slip through (the documented
           D-METAL-CAP failure mode the bug repro hit).

        Behavior on cap violation:
        - Increment ``num_metal_cap_violations`` counter (exposed via
          /metrics).
        - Log a single WARNING the first time the cap trips in this
          process — subsequent violations rely on the Prometheus
          counter to keep the log readable on a sustained over-cap
          storm (#D-METAL-CAP repro showed thousands of attempted
          admits within a single minute).
        - Raise ``BackpressureError`` so the existing route plumbing
          translates the failure to HTTP 503 with Retry-After.

        No-op when ``gpu_memory_utilization`` is 0 (default) or the
        Metal device probe failed — preserves the engine_core
        soft-pressure check as the only line of defence on those
        configurations (back-compat).
        """
        cap = self._resolve_metal_cap_bytes()
        if cap <= 0:
            return
        active = self._current_metal_active_bytes()
        projected_kv = self._estimate_request_kv_bytes(request)
        # Codex round 5 BLOCKING #1: count KV reservations of every
        # request already admitted but not yet finished. Without this,
        # a burst of small admits each individually fitting under the
        # cap stacks up to BLOW the cap collectively — ``active`` lags
        # the allocator until prefill actually runs.
        reserved_kv = self._sum_in_flight_kv_bytes()
        # Reject when ALREADY over cap OR when admitting the request
        # would push the allocator over cap on its own KV grow path
        # OR when the sum of in-flight reservations + this request
        # would exceed the cap.
        if active < cap and (active + reserved_kv + projected_kv) < cap:
            return

        # ── Hermes patch: force-evict paged KV before rejecting ──
        # D-METAL-CAP wedge root cause: the paged cache keeps KV
        # tensor memory resident on FREE blocks for reuse, so once
        # active exceeds cap every request is rejected and nothing
        # ever frees — permanent 503 until restart. Try to release
        # free-block KV tensor memory first; if enough returns, admit
        # the request instead of wedging the server.
        if self.paged_cache_manager is not None:
            try:
                # Bounded sweep: one admission must not synchronously
                # discard the entire free-block cache (unbounded latency).
                # 64 blocks is enough to clear a typical over-cap spike and
                # re-measure; sustained pressure keeps draining via the
                # engine tick.
                freed = self.paged_cache_manager.release_pressure_blocks(max_blocks=64)
            except Exception:
                freed = 0
                # A broken reclamation path must not masquerade as an
                # ordinary capacity 503 — surface it once (with traceback)
                # so recurrence of the wedge is diagnosable.
                if not getattr(self, "_admission_release_error_logged", False):
                    self._admission_release_error_logged = True
                    logger.warning(
                        "[D-METAL-CAP-force-evict] release_pressure_blocks "
                        "failed during admission; treating as no memory freed",
                        exc_info=True,
                    )
            if freed:
                active = self._current_metal_active_bytes()
                reserved_kv = self._sum_in_flight_kv_bytes()
                if active < cap and (active + reserved_kv + projected_kv) < cap:
                    logger.info(
                        "[D-METAL-CAP-force-evict] released %d paged KV "
                        "block(s); admitted request after pressure drop",
                        freed,
                    )
                    return
        self.num_metal_cap_violations += 1
        if not self._metal_cap_warning_logged:
            self._metal_cap_warning_logged = True
            # Codex round 3 NIT #4: defensively coerce ``request_id`` to
            # ``str`` before slicing. ``Request.request_id`` is typed as
            # ``str`` but unit-test fakes / malformed callers occasionally
            # pass through bytes or numbers — those would turn the
            # backpressure log path into an unrelated ``TypeError`` that
            # masks the real D-METAL-CAP signal. ``str(getattr(...))``
            # keeps the warning sane on every input shape.
            rid_str = str(getattr(request, "request_id", ""))[:12]
            logger.warning(
                "[D-METAL-CAP] Metal active %.1f GB + reserved KV "
                "%.1f GB + projected KV %.1f GB ≥ cap %.1f GB "
                "(gpu_memory_utilization=%.2f) — rejecting new "
                "request %s with backpressure. Further violations "
                "will be tracked by "
                "rapid_mlx_metal_cap_violations_total only.",
                active / 1e9,
                reserved_kv / 1e9,
                projected_kv / 1e9,
                cap / 1e9,
                getattr(self.config, "gpu_memory_utilization", 0.0),
                rid_str,
            )
        raise BackpressureError(
            f"Metal active {active / 1e9:.1f}GB + reserved KV "
            f"{reserved_kv / 1e9:.1f}GB + projected KV "
            f"{projected_kv / 1e9:.1f}GB would exceed "
            f"gpu_memory_utilization cap {cap / 1e9:.1f}GB "
            f"(D-METAL-CAP); retry after pressure drops"
        )

    def _resolve_pressure_evict_fraction(self) -> float:
        """Return the clamped ``(0, 1]`` fraction used for pressure thresholds.

        Codex round 2 NIT (kept across the R6-H6 expansion): a zero or
        negative configured fraction would compute ``threshold <= 0``
        and trip the eviction loop on every tick even when memory is
        quiet. A value > 1.0 would push the threshold ABOVE the cap
        itself, so eviction would never run before the admission gate
        started rejecting requests. Both shapes are clamped (not
        rejected) so a misconfigured operator gets a working default.
        """
        raw_fraction = float(getattr(self.config, "metal_pressure_evict_fraction", 0.9))
        if not (raw_fraction > 0.0):
            raw_fraction = 0.9
        if raw_fraction > 1.0:
            raw_fraction = 1.0
        return raw_fraction

    def _cache_self_pressure_threshold_bytes(self) -> int:
        """Return the prefix-cache memory threshold above which the
        scheduler proactively evicts, INDEPENDENT of ``gpu_memory_utilization``.

        R6-H6 root cause: pre-fix the pressure path was gated solely
        on ``mx.get_active_memory() > fraction × metal_cap``. When
        ``gpu_memory_utilization`` is unset (the default 0.0 — the
        configuration the 0.8.7 dogfood actually ran with),
        ``_resolve_metal_cap_bytes`` returns 0, the function early-
        returns, and ``num_prefix_cache_pressure_evictions`` never
        ticks even though the cache itself crept to 31 GB / 35.5 GB
        Metal allocated. This helper surfaces a second, always-on
        trigger so the pressure counter ticks whenever the cache's
        own memory ledger crosses the same configured fraction of
        its OWN budget (driven by ``RAPID_MLX_PREFIX_CACHE_MAX_BYTES``
        when set, or the heuristic 20%-of-RAM default otherwise).

        Returns ``0`` when no memory-aware cache is configured or it
        reports a non-positive max — callers treat ``0`` as "do not
        check on this path" and the legacy Metal-cap path still runs.

        ``getattr`` (not direct attribute access) on ``self`` makes the
        helper robust against partially initialised / older Scheduler
        instances missing the ``memory_aware_cache`` attribute (codex
        round-1 NIT on the R6-H6 patch).
        """
        cache = getattr(self, "memory_aware_cache", None)
        if cache is None:
            return 0
        # ``_max_memory`` is the bytes budget compute_memory_limit()
        # resolved at cache init (env override > programmatic >
        # heuristic > 8 GiB fallback). Pull it through a getattr so
        # this helper is robust against fakes / older cache variants
        # that don't expose the attribute.
        max_memory = int(getattr(cache, "_max_memory", 0) or 0)
        if max_memory <= 0:
            return 0
        return int(max_memory * self._resolve_pressure_evict_fraction())

    def _cache_self_pressure_current_bytes(self) -> int:
        """Snapshot of memory_aware_cache's current ledger in bytes.

        Returns ``0`` when no memory-aware cache is configured so
        callers short-circuit cleanly on engines that route through
        the block-aware / trie-based variants instead.

        ``getattr`` on ``self`` for the same defensive reason as
        :meth:`_cache_self_pressure_threshold_bytes` — a partially
        initialised Scheduler must NOT 500 the engine-loop pressure
        tick on attribute lookup.
        """
        cache = getattr(self, "memory_aware_cache", None)
        if cache is None:
            return 0
        return int(getattr(cache, "_current_memory", 0) or 0)

    def evict_prefix_cache_under_pressure(self, max_evict: int = 64) -> int:
        """LRU-evict prefix-cache entries while memory pressure persists.

        Two independent triggers can fire this loop:

        * **D-METAL-PFX (Metal active pressure):** when
          ``mx.get_active_memory()`` climbs above
          ``metal_pressure_evict_fraction × _resolve_metal_cap_bytes()``.
          Requires ``gpu_memory_utilization > 0`` so the Metal soft cap
          is configured.
        * **R6-H6 (cache-self pressure):** when the memory-aware
          cache's own ``_current_memory`` ledger climbs above
          ``metal_pressure_evict_fraction × _max_memory``. Fires
          INDEPENDENTLY of ``gpu_memory_utilization`` — this was the
          missing trigger in 0.8.7 dogfood (31 GB cache / 35.5 GB Metal
          allocated, zero pressure evictions because the Metal cap
          was never configured).

        After each eviction we call ``mx.clear_cache()`` so the
        allocator actually returns slabs to MLX's free pool rather
        than holding wired memory pinned to the now-dead CacheEntry.

        Returns the number of entries evicted (0 if neither trigger
        fired, no cache is configured, or the cache had nothing
        eligible). Increments ``num_prefix_cache_pressure_evictions``
        for each eviction so operators can attribute pressure-driven
        eviction separately from the cache's own LRU-capacity
        eviction in /metrics.

        Implementation note: ``max_evict`` is bounded so a single
        pressure tick cannot evict the entire prefix cache and trash
        every in-flight hit-rate stat on a transient spike. The
        engine_core loop calls this method every 16 steps, so a
        sustained-pressure scenario still drains the cache within a
        few hundred ms.
        """
        metal_cap = self._resolve_metal_cap_bytes()
        cache_self_threshold = self._cache_self_pressure_threshold_bytes()
        # Short-circuit when NEITHER trigger is configured. This keeps
        # the no-op cost (one ``mx.get_active_memory()`` syscall +
        # one dict lookup) off the path on engines that disabled both
        # the Metal soft cap AND the cache-self trigger (e.g. legacy
        # trie-based PrefixCacheManager engines).
        if metal_cap <= 0 and cache_self_threshold <= 0:
            return 0
        fraction = self._resolve_pressure_evict_fraction()
        metal_threshold = int(metal_cap * fraction) if metal_cap > 0 else 0
        evicted = 0
        # Track which trigger fired at least once during this tick so
        # the closing log line attributes the eviction wave to the
        # actual cause rather than defaulting to "Metal" whenever the
        # cap is merely configured (codex round-1 NIT on the R6-H6
        # patch). Both flags can be true on the same tick if pressure
        # crosses both thresholds simultaneously.
        triggered_metal = False
        triggered_cache_self = False
        for _ in range(max(0, int(max_evict))):
            should_evict = False
            if metal_threshold > 0:
                active = self._current_metal_active_bytes()
                if active >= metal_threshold:
                    should_evict = True
                    triggered_metal = True
            if not should_evict and cache_self_threshold > 0:
                current_cache = self._cache_self_pressure_current_bytes()
                if current_cache >= cache_self_threshold:
                    should_evict = True
                    triggered_cache_self = True
            if not should_evict:
                break
            if not self._evict_one_prefix_cache_entry():
                break
            # The entry has been removed from the cache trie — count
            # this as a successful eviction REGARDLESS of whether the
            # allocator-cache-flush step below succeeds. Codex round 4
            # BLOCKING #3: do NOT delay the counter past
            # ``mx.clear_cache()`` because if clear_cache raises, the
            # entry has already been removed but the counter would
            # never tick, leaving cache state and metrics in
            # disagreement (the trie says "108 → 107 entries" but
            # the metric still reads 0 cumulative evictions).
            evicted += 1
            self.num_prefix_cache_pressure_evictions += 1
            # Force the MLX allocator to actually return the slab now
            # that the trie / dict no longer pins the CacheEntry.
            # Without this, the underlying Metal allocation lingers in
            # the free-cache list and ``get_active_memory`` does not
            # drop on the next tick — exactly the D-METAL-PFX symptom
            # (allocator cache stuck at 0 while active stayed pinned).
            #
            # Codex round 3 BLOCKING #2 + round 4 BLOCKING #3 reconciled:
            # a failing ``mx.clear_cache`` MUST still propagate to the
            # engine_core warning path (so operators see the underlying
            # MLX failure), but the counter has ALREADY ticked above
            # because the cache mutation already happened — so the
            # metric reflects ground truth even when the allocator
            # flush blows up. This satisfies both invariants codex
            # flagged: surface clear_cache failures, AND keep
            # cache-state-vs-metric in sync on failure.
            mx.clear_cache()
        if evicted:
            if triggered_metal and triggered_cache_self:
                trigger = "Metal+cache-self"
            elif triggered_metal:
                trigger = "Metal"
            elif triggered_cache_self:
                trigger = "cache-self"
            else:
                # Belt-and-suspenders: ``evicted > 0`` only happens
                # after at least one ``should_evict = True``, so this
                # branch is unreachable. Keeping the explicit fallback
                # rather than asserting so a future refactor that
                # changes the loop structure doesn't crash the engine
                # loop on a log-only side effect.
                trigger = "unknown"
            logger.info(
                "[prefix-pressure-evict] evicted %d entries under %s pressure "
                "(metal_cap=%.1fGB, cache_max=%.1fGB)",
                evicted,
                trigger,
                metal_cap / 1e9 if metal_cap > 0 else 0.0,
                cache_self_threshold / 1e9 if cache_self_threshold > 0 else 0.0,
            )
        return evicted

    def _evict_one_prefix_cache_entry(self) -> bool:
        """Evict a single LRU prefix-cache entry across all cache variants.

        Returns True if an entry was actually removed. Encapsulates the
        cache-variant dispatch so ``evict_prefix_cache_under_pressure``
        stays variant-agnostic.

        The three cache variants share an LRU policy but their
        internal data structures differ:
        - ``memory_aware_cache``: OrderedDict-based, exposes
          ``_evict_lru`` under its own lock.
        - ``prefix_cache``: trie + OrderedDict LRU.
        - ``block_aware_cache``: paged block table; out of scope for
          the pressure trigger (blocks are released by
          ``PagedCacheManager`` ref-counts), so we no-op.

        Exception policy (codex round 2 BLOCKING #1): a failing
        ``_evict_lru`` call MUST propagate to
        ``evict_prefix_cache_under_pressure`` and from there to
        engine_core's rate-limited warning. Pre-fix this method
        swallowed every exception and returned False, making a broken
        cache variant indistinguishable from "nothing eligible" — the
        engine_core ``logger.warning(...)`` path could then never fire
        because the caller saw ``evicted=0`` and returned cleanly. By
        propagating the exception, the engine_core ``except
        Exception as evict_exc:`` block surfaces the underlying
        failure on the first occurrence per process.
        """
        if self.memory_aware_cache is not None:
            with self.memory_aware_cache._lock:  # noqa: SLF001 — coordinated eviction
                if not self.memory_aware_cache._entries:  # noqa: SLF001
                    return False
                self.memory_aware_cache._evict_lru()  # noqa: SLF001
            return True
        if self.prefix_cache is not None:
            if not getattr(self.prefix_cache, "_lru", None):
                return False
            self.prefix_cache._evict_lru()  # noqa: SLF001 — coordinated eviction
            return True
        if self.block_aware_cache is not None:
            # Hermes patch: pressure-trigger eviction for the paged
            # (block-aware) prefix cache. Previously this branch was a
            # deliberate no-op on the theory that PagedCacheManager
            # ref-counts release blocks — but completed requests never
            # call release_cache (blocks persist for sharing, see the
            # store_cache call site), so their blocks keep ref_count=1,
            # never re-enter the free queue, and every release path
            # (release_pressure_blocks / evict_lru_blocks / free_block)
            # only scans the free queue. Under Metal pressure the
            # admission gate then rejects ALL new requests while active
            # stays pinned above cap — a permanent 503 wedge (the
            # D-METAL-CAP symptom) with no in-process recovery. Fix:
            # evict the LRU prefix-index entry and drop its resident
            # block tensors so active memory falls back under cap and
            # admission resumes. Stale index entries are guarded by the
            # fetch-side live check (blk.cache_data is None => miss).
            index = getattr(self.block_aware_cache, "_prefix_index", None)
            paged = self.block_aware_cache.paged_cache
            # 1) Release the FULL-KV tensor pinned by the oldest
            # request-table entry FIRST — this is the actual Metal
            # allocation owner. The block slices are only views into
            # the same underlying buffer; the entry's ``cache_data`` is
            # what pins it. Without this, pressure eviction clears
            # slices forever while active memory never drops (the
            # observed D-METAL-CAP wedge: evictions_total grows, active
            # stays pinned above cap). Pop the entry so its block
            # ref-counts drop and the blocks can re-enter the free
            # queue. Doing this before the index pop also guarantees
            # full-KV entries drain even when the LRU index is empty.
            rt = getattr(self.block_aware_cache, "_request_tables", None)
            if rt:
                # True LRU: evict the least-recently-used entry
                # (BlockCacheEntry tracks last_access), not merely the first
                # in insertion order.
                evictable = [
                    (rid, rentry)
                    for rid in list(rt.keys())
                    if (rentry := rt.get(rid)) is not None
                    and rentry.cache_data is not None
                ]
                if evictable:
                    rid, rentry = min(evictable, key=lambda kv: kv[1].last_access)
                    block_ids = rentry.block_table.block_ids
                    # Drop this request's OWN full-KV buffer. Its per-block
                    # tensors are independent MLX slice arrays, so this does
                    # not invalidate a pinned or shared block's tensor — the
                    # buffer's memory stays live through whichever slices
                    # still reference it. Reclaiming the entry's own buffer
                    # is worthwhile even when every block is pinned/shared.
                    rentry.cache_data = None
                    for bid in block_ids:
                        blk = paged.allocated_blocks.get(bid)
                        if blk is None or blk.cache_data is None:
                            continue
                        # Keep the KV of blocks that must survive: pinned
                        # (system prompt) or still referenced by another
                        # live request (ref_count > 1). Clearing a shared
                        # block would corrupt the request that still holds
                        # it — pressure must not reach into live KV.
                        if blk.is_pinned or blk.ref_count > 1:
                            continue
                        # Remove hash mappings that still point at this
                        # block BEFORE reset_hash clears them — otherwise
                        # the maps keep pointing at an emptied block and a
                        # later hash lookup resurrects it. Mirror the index
                        # path below.
                        if (
                            blk.hash_value is not None
                            and paged.hash_to_block.get(blk.hash_value) == blk.block_id
                        ):
                            del paged.hash_to_block[blk.hash_value]
                        if blk.block_hash is not None:
                            paged.cached_block_hash_to_block.pop(
                                blk.block_hash, blk.block_id
                            )
                        blk.reset_hash()
                        blk.cache_data = None
                        blk.cache_class_name = None
                        paged.stats.evictions += 1
                    # release_cache pops the rt entry first, then frees the
                    # block table — so a raise here cannot leave the entry
                    # stuck in rt, but guard it anyway so one bad table can't
                    # tear down the pressure tick, and drop the rt entry
                    # ourselves as a backstop.
                    try:
                        self.block_aware_cache.release_cache(rid)
                    except Exception:
                        rt.pop(rid, None)
                        if not getattr(self, "_pfx_release_error_logged", False):
                            self._pfx_release_error_logged = True
                            logger.warning(
                                "[D-METAL-PFX-evict] release_cache(%s) failed; "
                                "dropped the entry to keep eviction live",
                                rid,
                                exc_info=True,
                            )
                    logger.debug(
                        "[D-METAL-PFX-evict] dropped full-KV entry %s "
                        "(request_tables=%d)",
                        rid,
                        len(rt),
                    )
                    return True
            # 2) Then evict the LRU prefix-index entry and drop its
            # resident block tensors, so stale index entries stop
            # pinning block slices. Stale entries are guarded by the
            # fetch-side live check (blk.cache_data is None => miss).
            if index:
                # dict preserves insertion order: first key = LRU. Walk in
                # LRU order and skip any entry whose blocks are ALL pinned
                # or absent: popping such an entry would make the prefix
                # unreachable while freeing nothing, and returning False on
                # it would halt the caller's eviction loop prematurely.
                for oldest_hash in list(index.keys()):
                    cached_tokens, block_ids = index[oldest_hash]
                    # An index entry can outlive its physical blocks. If one
                    # of those slots was subsequently reallocated, ref_count
                    # alone cannot distinguish the new owner's live KV from
                    # the old prefix. Prune the stale index entry without
                    # touching any block unless every slot still owns the
                    # token slice recorded by this prefix.
                    stale = False
                    bs = self.block_aware_cache.block_size
                    for j, bid in enumerate(block_ids):
                        blk = paged.allocated_blocks.get(bid)
                        block_tokens = cached_tokens[j * bs : (j + 1) * bs]
                        expected_hash = paged.compute_block_hash(block_tokens)
                        # Missing blocks are handled by the eligibility
                        # check below. A ``None`` hash is retained for
                        # compatibility with legacy/unhashed fixtures; all
                        # newly stored production blocks record ownership.
                        if (
                            blk is not None
                            and blk.hash_value is not None
                            and blk.hash_value != expected_hash
                        ):
                            stale = True
                            break
                    if stale:
                        index.pop(oldest_hash)
                        return True
                    # Skip an entry only when NONE of its blocks are
                    # clearable — absent, pinned, or still shared with a
                    # live request (ref_count > 1). Popping such an entry
                    # would make the prefix unreachable while freeing
                    # nothing and would halt the caller's eviction loop.
                    if all(
                        (blk := paged.allocated_blocks.get(bid)) is None
                        or blk.is_pinned
                        or blk.ref_count > 1
                        for bid in block_ids
                    ):
                        continue
                    index.pop(oldest_hash)
                    evicted = 0
                    for bid in block_ids:
                        blk = paged.allocated_blocks.get(bid)
                        # Never clear a pinned or still-shared block: its KV
                        # is live for another request.
                        if blk is None or blk.is_pinned or blk.ref_count > 1:
                            continue
                        # Release the block's resident KV tensor regardless
                        # of hash registration variant. store_cache
                        # registers blocks via register_block_hash (legacy
                        # hash_value only, block_hash stays None), so the
                        # chain-hash gated _maybe_evict_cached_block would
                        # short-circuit and leak the tensor. Mirror its
                        # cleanup directly: drop any hash mapping, clear
                        # the tensor.
                        if (
                            blk.hash_value is not None
                            and paged.hash_to_block.get(blk.hash_value) == blk.block_id
                        ):
                            del paged.hash_to_block[blk.hash_value]
                        if blk.block_hash is not None:
                            paged.cached_block_hash_to_block.pop(
                                blk.block_hash, blk.block_id
                            )
                        if blk.cache_data is not None:
                            blk.reset_hash()
                            blk.cache_data = None  # Free tensor memory
                            blk.cache_class_name = None
                            paged.stats.evictions += 1
                            evicted += 1
                        # Return the block slot to the free queue. Unlike the
                        # request-table path (which frees via release_cache ->
                        # delete_block_table), the index path holds its own
                        # ref, so clearing the tensor alone would leak the
                        # block from the pool and eventually exhaust
                        # max_cache_blocks. free_block decrements ref and
                        # enqueues the (now-empty) block when it reaches 0.
                        paged.free_block(bid)
                    return evicted > 0
                return False
            return False
        return False

    def release_paged_cache_blocks_under_pressure(self, max_blocks: int = 32) -> int:
        """Drop reusable free-block KV only while Metal pressure is active.

        The engine calls this on its periodic pressure tick. Keeping the
        threshold decision here prevents normal, below-threshold operation
        from continually destroying reusable paged-cache blocks.
        """
        paged = self.paged_cache_manager
        if paged is None:
            return 0
        cap = self._resolve_metal_cap_bytes()
        if cap <= 0:
            # The explicit admission cap is optional (default util=0), but
            # paged-cache slabs still need an emergency pressure ceiling.
            # Fall back to Metal's recommended working set so default
            # deployments reclaim near device exhaustion instead of either
            # reclaiming constantly or retaining free KV until OOM.
            try:
                if not mx.metal.is_available():
                    return 0
                info = mx.device_info()
                cap = int(
                    info.get(
                        "max_recommended_working_set_size",
                        info.get("memory_size", 0),
                    )
                    or 0
                )
            except Exception:
                return 0
            if cap <= 0:
                return 0
        threshold = int(cap * self._resolve_pressure_evict_fraction())
        if self._current_metal_active_bytes() < threshold:
            return 0
        return paged.release_pressure_blocks(max_blocks=max_blocks)

    def add_request(self, request: Request) -> None:
        """
        Add a new request to the scheduler.

        Args:
            request: The request to add

        Raises:
            BackpressureError: If the in-flight request count is at or
                above ``config.max_concurrent_requests``. Routes catch
                this and return 503 with Retry-After.
        """
        if request.request_id in self.requests:
            raise ValueError(f"Request {request.request_id} already exists")

        # Admission control: cap concurrent in-flight requests so a
        # buggy/abusive client can't OOM Metal and crash the server
        # for everyone else. Check BEFORE tokenization so the cost of
        # being over the cap is just a dict lookup.
        cap = self.config.max_concurrent_requests
        if cap is not None and cap > 0 and len(self.requests) >= cap:
            raise BackpressureError(
                f"max_concurrent_requests={cap} reached "
                f"(currently {len(self.requests)} in-flight)"
            )

        # D-METAL-CAP: enforce the gpu_memory_utilization cap that
        # ``mx.set_memory_limit`` silently let MLX violate on big-RAM
        # hosts. Raises ``BackpressureError`` so the existing route
        # plumbing returns 503 + Retry-After instead of marching the
        # allocator past the operator-configured limit.
        self._enforce_metal_cap_at_admission(request)

        # Tokenize if needed
        if request.prompt_token_ids is None:
            if isinstance(request.prompt, str):
                # Handle both tokenizers and processors (for MLLM models)
                if hasattr(self.tokenizer, "encode"):
                    request.prompt_token_ids = self.tokenizer.encode(request.prompt)
                elif hasattr(self.tokenizer, "tokenizer") and hasattr(
                    self.tokenizer.tokenizer, "encode"
                ):
                    # Processor wraps tokenizer (e.g., Qwen3VLProcessor)
                    request.prompt_token_ids = self.tokenizer.tokenizer.encode(
                        request.prompt
                    )
                else:
                    raise AttributeError(
                        f"Tokenizer {type(self.tokenizer)} has no 'encode' method. "
                        "Continuous batching requires a tokenizer with encode support."
                    )
            else:
                request.prompt_token_ids = list(request.prompt)
            request.num_prompt_tokens = len(request.prompt_token_ids)

        # Logical-vs-model prompt-length split (#287). num_prompt_tokens
        # is what gets reported to clients / usage tracking; PFlash may
        # shorten prompt_token_ids before prefill, so model_prompt_tokens
        # tracks the post-transform length used by the scheduler.
        if request.prompt_token_ids is not None and request.model_prompt_tokens == 0:
            request.model_prompt_tokens = len(request.prompt_token_ids)

        # PFlash long-prompt compression — must run before any cache
        # lookup. When compression engages, prompt_token_ids is replaced
        # by the kept-token subsequence and the prefix cache is bypassed
        # entirely (both fetch and store, see below) because the
        # compressed token sequence is a positional fiction: position i
        # in compressed land does NOT correspond to position i in the
        # original prompt, so reusing KV computed for the uncompressed
        # prefix would inject position-shifted state into a later
        # uncompressed request that shares the same sink prefix.
        pflash_compressed = False
        if self.config.pflash_config.mode != "off" and request.prompt_token_ids:
            original_tokens = list(request.prompt_token_ids)
            original_prefix_boundary = request.prefix_boundary
            scoring_start = time.monotonic()
            compressed_tokens, metadata = compress_request_tokens(
                original_tokens,
                self.config.pflash_config,
                has_tools=request.has_tools,
                requires_prompt_integrity=request.requires_prompt_integrity,
            )
            metadata["scoring_seconds"] = time.monotonic() - scoring_start
            metadata["logical_prompt_tokens"] = len(original_tokens)
            metadata["model_prompt_tokens"] = len(compressed_tokens)
            metadata["prefix_boundary_original"] = original_prefix_boundary
            metadata["prefix_boundary_disabled"] = False
            request.pflash_metadata = metadata
            if metadata["compressed"]:
                pflash_compressed = True
                # M-02: count every prompt that took the PFlash bypass
                # so /metrics surfaces the work that prefix-cache
                # counters can't (the compressed sequence skips both
                # fetch and store — see the explanation block above).
                # ``tokens_dropped`` = logical prompt length minus kept
                # length, i.e. the saving operators want for capacity
                # planning.
                self.pflash_bypass_count += 1
                self.pflash_compressed_tokens_dropped += max(
                    0, len(original_tokens) - len(compressed_tokens)
                )
                request.original_prompt_token_ids = original_tokens
                request.prompt_token_ids = compressed_tokens
                request.model_prompt_tokens = len(compressed_tokens)
                # prefix_boundary indexes into the ORIGINAL prompt; the
                # compressed sequence is non-prefix so a boundary save
                # would point at meaningless tokens. Force-disable.
                if original_prefix_boundary > 0:
                    request.prefix_boundary = 0
                    metadata["prefix_boundary_disabled"] = True
                logger.info(
                    f"[pflash] request={request.request_id[:12]} "
                    f"compressed {metadata['original_tokens']} -> "
                    f"{metadata['kept_tokens']} tokens "
                    f"ratio={metadata['compression_ratio']:.3f} "
                    f"scoring_ms={metadata['scoring_seconds'] * 1000.0:.2f}"
                )
            else:
                logger.debug(
                    f"[pflash] request={request.request_id[:12]} skipped "
                    f"reason={metadata['reason']} tokens={metadata['original_tokens']}"
                )

        # Check prefix cache for cached KV state. Compressed requests
        # MUST skip the lookup — see PFlash comment above for the
        # positional-fiction explanation.
        if pflash_compressed:
            request.cache_hit_type = "miss"
            request.remaining_tokens = request.prompt_token_ids
        elif self.block_aware_cache is not None:
            # Use paged cache
            block_table, remaining = self.block_aware_cache.fetch_cache(
                request.request_id,
                request.prompt_token_ids,
            )
            if block_table and block_table.num_tokens > 0:
                request.cache_hit_type = "hit"
                # Reconstruct actual KVCache objects from stored tensor data
                reconstructed = self.block_aware_cache.reconstruct_cache(block_table)
                if reconstructed:
                    request.prompt_cache = reconstructed
                    request.block_table = block_table
                    request.cached_tokens = block_table.num_tokens
                    request.shared_prefix_blocks = len(block_table.block_ids)
                    request.remaining_tokens = remaining
                    logger.debug(
                        f"Request {request.request_id}: paged cache hit, "
                        f"{request.cached_tokens} tokens in {request.shared_prefix_blocks} blocks, "
                        f"{len(remaining)} tokens remaining, cache reconstructed"
                    )
                else:
                    # Reconstruction failed, treat as cache miss
                    request.cache_hit_type = "miss"
                    request.remaining_tokens = request.prompt_token_ids
                    logger.debug(
                        f"Request {request.request_id}: paged cache reconstruction failed"
                    )
            else:
                request.cache_hit_type = "miss"
                request.remaining_tokens = request.prompt_token_ids
        elif self.memory_aware_cache is not None:
            # Use memory-aware prefix cache
            import time as _time

            _fetch_t0 = _time.monotonic()
            cache, remaining = self.memory_aware_cache.fetch(request.prompt_token_ids)
            _fetch_dt = _time.monotonic() - _fetch_t0
            request.cache_hit_type = self.memory_aware_cache._last_match_type
            if cache:
                request.prompt_cache = cache
                request.cached_tokens = len(request.prompt_token_ids) - len(remaining)
                request.remaining_tokens = remaining
                logger.info(
                    f"[cache_fetch] request={request.request_id[:12]} HIT "
                    f"prompt_tokens={len(request.prompt_token_ids)} "
                    f"cached={request.cached_tokens} remaining={len(remaining)} "
                    f"time={_fetch_dt:.3f}s"
                )
            else:
                request.remaining_tokens = request.prompt_token_ids
                logger.info(
                    f"[cache_fetch] request={request.request_id[:12]} MISS "
                    f"prompt_tokens={len(request.prompt_token_ids)} "
                    f"time={_fetch_dt:.3f}s entries={len(self.memory_aware_cache._entries)}"
                )
        elif self.prefix_cache is not None:
            # Use legacy prefix cache
            cache, remaining = self.prefix_cache.fetch_cache(request.prompt_token_ids)
            if cache:
                request.cache_hit_type = "hit"
                request.prompt_cache = cache
                request.cached_tokens = len(request.prompt_token_ids) - len(remaining)
                request.remaining_tokens = remaining
                logger.debug(
                    f"Request {request.request_id}: cache hit, "
                    f"{request.cached_tokens} tokens cached, "
                    f"{len(remaining)} tokens remaining"
                )
            else:
                request.cache_hit_type = "miss"
                request.remaining_tokens = request.prompt_token_ids
        else:
            request.cache_hit_type = "miss"
            request.remaining_tokens = request.prompt_token_ids

        # Add to tracking. D-M01-2X (0.8.2 dogfood, codex r10
        # BLOCKING follow-up): the cancellation dedupe ledgers
        # (``_cancelled_request_ids`` / ``_disconnect_abort_ids``)
        # are LIFETIME-PERSISTENT across the
        # abort+cleanup window (see ``remove_finished_request``
        # docstring for the multi-branch race repro). Clearing
        # them at fresh admit preserves the request_id-reuse
        # counting semantics — but the clear MUST run atomically
        # with the ``self.requests[...] = request`` commit, NOT
        # earlier in this method. An earlier clear (e.g. right
        # after the admission gate) would erase the prior
        # lifetime's dedupe even if tokenization / cache lookup /
        # PFlash compression subsequently raised, opening a
        # double-count window for the OLD lifetime should a late
        # ``abort_request`` arrive between the failed admit and
        # the next successful one. By gating the clear on the
        # same critical section as the actual commit, every
        # exception path between admission and tracking leaves
        # the ledger intact and the prior lifetime's dedupe
        # stays effective.
        with self._cancel_counter_lock:
            self._cancelled_request_ids.discard(request.request_id)
            self._disconnect_abort_ids.discard(request.request_id)
            self._orphaned_running_candidates.pop(request.request_id, None)
            self.requests[request.request_id] = request
        self.waiting.append(request)

        logger.debug(
            f"Added request {request.request_id} with {request.num_prompt_tokens} prompt tokens"
        )

    def abort_request(self, request_id: str) -> bool:
        """
        Queue request for abort. Thread-safe, called from any thread.

        The actual abort is deferred to the executor thread (inside step())
        to avoid race conditions with in-flight Metal GPU operations.

        Args:
            request_id: The request ID to abort

        Returns:
            True when an active/queued request was enqueued for abort, False
            when ``request_id`` is unknown to this scheduler. F-151 hardening:
            previously this method returned True unconditionally — including
            for arbitrary attacker-supplied strings — which let the route
            layer respond ``{"cancelled": true}`` for any id. The route uses
            the False return as the 404 signal.
        """
        # Consider the request "known" if it lives in any of: the canonical
        # ``requests`` dict (admitted but not finished), the BatchGenerator
        # uid map (admitted into a live batch — may already have been popped
        # from ``requests`` by an in-flight ``_cleanup_request``), the
        # ``running`` map (currently scheduled), or ``_pending_abort_ids``
        # (a concurrent abort enqueue made this method idempotent — return
        # True so a double-cancel doesn't 404 the second caller). We do NOT
        # treat ``finished_req_ids`` as "known" because the abort would be
        # a no-op and the route contract is "404 when already finished".
        # M-01 codex r1 BLOCKING #2 + r2 BLOCKING #1 + r6 BLOCKING #1:
        # the membership check AND the check-add-increment sequence
        # MUST be atomic together — checking ``request_id in
        # self.requests`` outside the lock leaves a window where
        # ``remove_finished_request`` can race in, pop ``self.requests``,
        # clear the dedupe ledger, and let THIS path then re-add the id
        # to ``_pending_abort_ids`` and increment ``num_requests_cancelled``
        # for an already-removed request lifetime. By re-validating
        # membership INSIDE the lock against the same maps
        # ``remove_finished_request`` mutates (``self.requests``) and
        # the abort-state maps (``request_id_to_uid`` / ``running`` /
        # ``_pending_abort_ids``), we guarantee:
        #   * any abort that passes the inside-lock predicate has a
        #     live referent that can't be popped concurrently;
        #   * the dedupe-ledger check + add + counter increment
        #     remain serialized across all callers.
        # The lock cost is negligible (microseconds per abort).
        with self._cancel_counter_lock:
            if not (
                request_id in self.requests
                or request_id in self.request_id_to_uid
                or request_id in self.running
                or request_id in self._pending_abort_ids
            ):
                logger.info(
                    "[abort_request] unknown request_id (rejected without enqueue)"
                )
                return False
            already_counted = request_id in self._cancelled_request_ids
            self._cancelled_request_ids.add(request_id)
            self._pending_abort_ids.add(request_id)
            if not already_counted:
                self.num_requests_cancelled += 1
        logger.info(f"[abort_request] {request_id[:12]} enqueued for deferred abort")
        return True

    def record_disconnect_abort(self, request_id: str) -> None:
        """M-01: attribute a previously-accepted abort to client disconnect.

        Called by ``_force_abort_request`` (service/helpers.py) AFTER
        the sync ``abort_request`` returned True (or the async fallback
        scheduled the abort), so the total counter was already bumped
        exactly once on the public entry-point. The ``request_id`` is
        recorded into the dedicated ``_disconnect_abort_ids`` set so
        concurrent disconnect-guard + finally belt-and-suspenders
        paths (both fire the helper) only attribute once per request
        — matching the once-per-request semantics of the total
        counter.

        Codex r1 BLOCKING #3: the check-add-increment sequence is
        serialized against the same ``_cancel_counter_lock`` that
        guards the total counter, because the disconnect_guard fires
        from up to three branches per disconnect across potentially
        different async tasks. Without the lock two threads could
        both observe ``request_id not in _disconnect_abort_ids`` and
        double-count the sub-counter. The lock cost is microseconds
        per call, negligible against the existing disconnect-path
        latency.

        Codex r7 NIT #3: validate against ``_cancelled_request_ids``
        BEFORE incrementing so a future caller (or a bug) that
        records a disconnect for an id the scheduler never accepted
        as a cancel cannot push the ``via_disconnect`` sub-counter
        above the total. The contract is now "disconnect
        attribution is only valid for ids the scheduler ALSO
        accepted via ``abort_request``"; ids not in the lifetime
        ledger silently no-op. This guarantees the dashboard
        invariant ``via_disconnect_total <= cancelled_total`` holds
        even on programmer error in callers that record without
        first hitting the public abort path.

        Safe to call from any thread, never raises. Empty / None ids
        are no-ops.
        """
        try:
            if not request_id:
                return
            with self._cancel_counter_lock:
                # Codex r7 NIT #3: gate on the lifetime ledger so
                # ``via_disconnect_total <= cancelled_total`` holds
                # by construction. Callers MUST hit
                # ``abort_request`` first; this method only
                # attributes a previously-accepted abort.
                if request_id not in self._cancelled_request_ids:
                    return
                if request_id not in self._disconnect_abort_ids:
                    self._disconnect_abort_ids.add(request_id)
                    self.num_requests_cancelled_via_disconnect += 1
        except Exception:  # pragma: no cover - belt-and-suspenders
            # Observability must never break a live disconnect path —
            # a counter that fails to advance is preferable to one
            # that escapes back through ``_force_abort_request`` and
            # masks the abort in the caller's exception handler.
            pass

    def _register_uid_processors(
        self,
        uid: int,
        request: Any,
        request_processors: list | None,
        grammar_lp: Any,
    ) -> None:
        """Record a uid's logits-processor bookkeeping at admission.

        Single source of truth for the per-tick realign guard (the COUNTERPART
        of :meth:`_forget_uid_grammar`). Both the scheduler's admission path and
        the realign unit tests call THIS method, so a regression in the recorded
        state (a dropped budget/grammar arm, a lost penalty list) fails a test
        rather than only surfacing on live hardware (codex #558 NIT).

        * ``uid_to_request_processors`` remembers EVERY uid that carries any
          processor (grammar + penalties), the authoritative list the realign
          rebuilds each slot from — immune to mlx-lm's stale-entry desync.
        * A grammar-carrying uid also registers the grammar's identity in the
          stateful-processor set (so a leaked slot can be scrubbed even after the
          uid finishes) and arms the guard.
        * A reasoning-budget uid arms the guard so its force-close ``</think>``
          processor is realigned every tick, AND registers the budget
          processor's identity in the SAME stateful set — a budget processor is
          per-request stateful (a ``</think>`` force-close latch), so if a
          finished budget uid's processor leaks into a stale positional slot
          after cancellation/truncation it must be tombstoned and scrubbed
          exactly like a leaked grammar (codex: track + tombstone budget like
          grammar). Its object already lives in ``uid_to_request_processors[uid]``
          too.
        """
        if request_processors:
            self.uid_to_request_processors[uid] = list(request_processors)
        if grammar_lp is not None:
            self._uids_with_grammar.add(uid)
            self._known_stateful_processors.add(id(grammar_lp))
            # Keep the object alive so its ``id()`` can't be reused by a
            # different object while it may still linger in a slot.
            self._stateful_processor_objs[id(grammar_lp)] = grammar_lp
        _rblp = getattr(request, "reasoning_budget_logits_processor", None)
        if _rblp is not None:
            self._uids_with_reasoning_budget[uid] = _rblp
            # Same tombstone/scrub treatment as grammar (see docstring): register
            # identity + keep the object alive so a leaked force-close processor
            # is scrubbed from a foreign slot and the guard stays armed for that
            # scrubbing tick.
            self._known_stateful_processors.add(id(_rblp))
            self._stateful_processor_objs[id(_rblp)] = _rblp
        _stlp = getattr(request, "suppressed_tokens_logits_processor", None)
        if _stlp is not None:
            self._uids_with_suppressed_tokens[uid] = _stlp
            self._known_stateful_processors.add(id(_stlp))
            self._stateful_processor_objs[id(_stlp)] = _stlp

    def _forget_uid_grammar(self, uid: int) -> None:
        """Drop #558 PR-3 per-uid processor state for a uid leaving the batch.

        Called from the abort + finish cleanup paths. Those paths remove ``uid``
        from the generation batch's ``uids`` *before* the next tick's realign,
        but mlx-lm's own positional ``logits_processors`` filter can lag by a
        tick and leave the just-finished STATEFUL processor (grammar OR
        reasoning-budget force-close) in a LEAKED slot. So we do NOT immediately
        forget its identity here: we TOMBSTONE it (keep it in
        ``_known_stateful_processors`` so the realign guard stays armed and still
        scrubs it) until the guard confirms it's absent from every live slot.
        This is what stops a finished budget uid's force-close processor from
        leaking onto a later no-processor request after the last budget uid
        disarms the guard (codex). Penalty-only uids carry no stateful processor
        and are simply dropped.
        """
        self._uids_with_grammar.discard(uid)
        self._uids_with_reasoning_budget.pop(uid, None)
        self._uids_with_suppressed_tokens.pop(uid, None)
        procs = self.uid_to_request_processors.pop(uid, None)
        if not procs:
            return
        # A stateful processor (grammar/budget) in this uid's list that's still
        # referenced by ANOTHER live uid stays fully live; otherwise it becomes a
        # tombstone — still known (and thus still scrubbable) but pending removal
        # once no slot holds it. A request never shares its stateful processor, so
        # the cross-uid check is defensive; the tombstone is the load-bearing part.
        still_referenced: set[int] = set()
        for plist in self.uid_to_request_processors.values():
            for p in plist:
                if id(p) in self._known_stateful_processors:
                    still_referenced.add(id(p))
        for p in procs:
            pid = id(p)
            if pid in self._known_stateful_processors and pid not in still_referenced:
                self._stateful_tombstones.add(pid)

    def _realign_guard_armed(self) -> bool:
        """True when the per-tick logits-processor realign must run.

        Armed by any live GRAMMAR uid, a not-yet-flushed stateful tombstone, or
        any reasoning-BUDGET uid whose processor is still ACTIVE (has not yet
        forced ``</think>``). All carry a HARD per-token guarantee (a constrained
        tool call / a pending forced ``</think>``) that mlx-lm's positional
        ``logits_processors`` desync would silently violate, so their slots must
        be rebuilt from ``uid_to_request_processors`` every tick.

        A budget processor that has already ``_ended`` (thinking closed; now
        emitting the answer) is INERT — it returns logits unchanged — so it no
        longer needs its slot realigned, and keeping the guard armed for it would
        add an O(batch) rebuild to every remaining ANSWER token for no benefit
        (codex R10 #4). Disarming is safe: if such an inert processor later leaks
        into a foreign slot, applying it is a no-op, and when its uid finishes
        ``_forget_uid_grammar`` tombstones it — which re-arms the guard for the
        scrub tick. Penalty-only uids never arm it (plain decode hot path).

        This is the single source of truth for the arming condition so it can be
        unit-tested directly (a deleted budget arm fails a test rather than only
        surfacing on live hardware).
        """
        return bool(
            self._uids_with_grammar
            or self._stateful_tombstones
            or self._uids_with_suppressed_tokens
            or any(
                not getattr(p, "_ended", False)
                for p in self._uids_with_reasoning_budget.values()
            )
        )

    def _realign_grammar_logits_processors(self) -> None:
        """Rebuild the generation batch's ``logits_processors`` aligned to uids.

        #558 PR-3. mlx-lm's ``GenerationBatch`` stores ``logits_processors``
        as a positional per-uid list. When a NO-processor request finishes
        while a grammar request is mid-flight, a stale entry can survive the
        filter and DESYNC that list from ``uids`` — mlx-lm's ``_step`` then
        iterates ``range(len(uids))`` and applies the wrong (or no) processor,
        silently leaving an explicit ``required``/named call unconstrained.

        Rebuild strategy — authoritative, never lossy (codex #558-PR3):

          * ``uid_to_request_processors`` is the SOURCE OF TRUTH for EVERY live
            uid that carries any processor — grammar AND penalty-only. Each such
            slot is rebuilt verbatim from its stored list, so a bystander's
            repetition/frequency/presence penalties are preserved even when the
            positional list is length-desynced (previously those uids got ``[]``,
            silently deleting their penalties, because running requests are not
            re-inserted).
          * A uid we do NOT track genuinely has no processors, so its slot is
            ``[]``. This also scrubs any grammar processor (by identity, via the
            ``_known_stateful_processors`` set — which still contains tombstoned
            grammars whose owning uid finished, closing the cleanup-ordering
            gap) that leaked into that slot via a desync.

        After rebuilding, sweep tombstoned grammar identities: any that is now
        absent from every live slot is fully forgotten. This keeps the guard
        armed across the tick that finally drops a leaked grammar even when it
        was the LAST in-flight grammar.

        Idempotent and cheap: only writes back when something actually changed.
        """
        bg = getattr(self.batch_generator, "_generation_batch", None)
        if bg is None:
            # No generation batch == no live slots at all, so no tombstoned
            # grammar can be lingering in one. Flush them here too (codex
            # #558-PR3 nit): otherwise a finished grammar's processor + matcher
            # state would be retained indefinitely across an idle period where
            # the batch object itself is absent.
            self._flush_stateful_tombstones(present=set())
            return
        uids = getattr(bg, "uids", None)
        if not uids:
            # No live slots. A stale positional ``logits_processors`` list can
            # still linger here if the batch emptied its ``uids`` without the
            # filter dropping the parallel processor list (mlx-lm keys the two
            # separately). Left in place, the NEXT admitted plain request would
            # inherit a leaked grammar processor at position 0 and be silently
            # constrained (codex #558-PR3 blocking). A batch with zero uids must
            # carry zero processors — scrub the list to ``[]`` while the grammar
            # identities are still known, THEN forget the tombstones.
            existing = getattr(bg, "logits_processors", None)
            if existing:
                bg.logits_processors = []
            self._flush_stateful_tombstones(present=set())
            return
        existing = getattr(bg, "logits_processors", None)
        aligned = existing is not None and len(existing) == len(uids)
        rebuilt: list = []
        present_ids: set[int] = set()
        changed = not aligned
        for i, uid in enumerate(uids):
            authoritative = self.uid_to_request_processors.get(uid)
            if authoritative is not None:
                # Source of truth: rebuild this slot verbatim (grammar +
                # penalties). Copy so mlx-lm's in-place filters can't mutate
                # our stored list.
                slot = list(authoritative)
            else:
                # Untracked uid == no processors. Empty slot (which also drops
                # any grammar that leaked here via a desync).
                slot = []
            for p in slot:
                if id(p) in self._known_stateful_processors:
                    present_ids.add(id(p))
            # Detect whether this slot differs from what's there now.
            if not aligned or existing[i] != slot:
                changed = True
            rebuilt.append(slot)
        if changed:
            bg.logits_processors = rebuilt
        self._flush_stateful_tombstones(present=present_ids)

    def _flush_stateful_tombstones(self, present: set[int]) -> None:
        """Forget tombstoned grammar identities no longer in any live slot.

        A tombstone (see ``_forget_uid_grammar``) is a grammar whose owning uid
        has left the tracking maps but that may still linger in a leaked batch
        slot. Once the realign guard confirms it's absent from every live slot,
        it's safe to fully drop — releasing the id-reuse guard object too.
        """
        for pid in list(self._stateful_tombstones):
            if pid in present:
                continue
            self._stateful_tombstones.discard(pid)
            self._known_stateful_processors.discard(pid)
            self._stateful_processor_objs.pop(pid, None)

    def _process_pending_aborts(self) -> None:
        """Drain and process pending abort requests. Called from executor thread."""
        while self._pending_abort_ids:
            request_id = self._pending_abort_ids.pop()
            self._do_abort_request(request_id)

    def _reconcile_orphaned_running_requests(self) -> list[str]:
        """Reap running slots whose engine-side request was already released.

        ``EngineCore._cleanup_request`` removes the canonical ``requests``
        entry when a streaming consumer disconnects.  The scheduler normally
        consumes the corresponding deferred abort at the start of the next
        step.  Production issue #1759 demonstrated that relying on that single
        edge is not sufficient: after a rare disconnect race, the abort edge
        can be gone while ``running`` and the BatchGenerator uid remain.  The
        empty consumer can then be decoded forever and permanently occupies a
        ``max_num_seqs`` slot.

        After pending aborts have been drained, a running id missing from
        ``requests`` is impossible for a live request: normal completions are
        removed from ``running`` on the executor thread *before* engine cleanup,
        while disconnect cleanup deliberately removes ``requests`` first.  Use
        that invariant as a narrow executor-thread safety net and route cleanup
        through the same BatchGenerator-aware abort implementation.
        """
        with self._cancel_counter_lock:
            candidates = self._orphaned_running_candidates
            self._orphaned_running_candidates = {}
        orphaned = []
        for request_id, request in candidates.items():
            if self._do_abort_request(request_id, expected_orphan=request):
                orphaned.append(request_id)
                logger.warning(
                    "[abort_reconcile] reaped orphaned running request %s",
                    request_id[:12],
                )
        return orphaned

    def _do_abort_request(
        self, request_id: str, *, expected_orphan: Request | None = None
    ) -> bool:
        """Serialize abort cleanup and optionally bind it to one lifetime.

        The orphan reconciler crosses from engine cleanup back onto the
        scheduler thread.  An ID alone is insufficient because callers may
        reuse one after cleanup.  Validate the exact ``Request`` identity and
        perform the ID-based teardown while holding the same lock used by the
        admission commit, closing the check-to-abort window.
        """
        with self._cancel_counter_lock:
            if expected_orphan is not None and not (
                self.running.get(request_id) is expected_orphan
                and request_id not in self.requests
            ):
                return False
            return self._do_abort_request_impl(request_id)

    def _do_abort_request_impl(self, request_id: str) -> bool:
        """
        Actually abort a request. Must be called from the executor thread.

        Handles the case where the request was already removed from
        self.requests by _cleanup_request() but still lives in the
        BatchGenerator (e.g. in _partial or active_batch).

        Args:
            request_id: The request ID to abort

        Returns:
            True if any cleanup was performed, False otherwise
        """
        request = self.requests.get(request_id)
        was_waiting = False
        was_running = False
        removed_from_batch = False

        # Remove from waiting queue.
        # When request is not None we can remove by identity; when it's None
        # (already popped by _cleanup_request) we must scan by request_id so
        # the deque entry doesn't survive the abort.
        if request is not None and request.status == RequestStatus.WAITING:
            was_waiting = True
            try:
                self.waiting.remove(request)
            except ValueError:
                pass
        elif request is None:
            # Scan waiting deque by request_id — request object was already
            # removed from self.requests but may still sit in the deque.
            for waiting_req in list(self.waiting):
                if waiting_req.request_id == request_id:
                    was_waiting = True
                    try:
                        self.waiting.remove(waiting_req)
                    except ValueError:
                        pass
                    break

        # Remove from running (BatchGenerator) — do this even if request
        # was already cleaned up from self.requests, because the UID may
        # still be live inside the BatchGenerator (_partial / active_batch).
        if request_id in self.request_id_to_uid:
            was_running = True
            uid = self.request_id_to_uid[request_id]
            if self.batch_generator is not None:
                self.batch_generator.remove([uid])
                removed_from_batch = True
            del self.uid_to_request_id[uid]
            # #558 PR-3: drop the aborted uid's grammar processor state (the
            # uid is already out of the batch via ``remove`` above).
            self._forget_uid_grammar(uid)
            del self.request_id_to_uid[request_id]

        if request_id in self.running:
            del self.running[request_id]

        # Credit in-flight tokens so dashboard metrics stay accurate
        # (without this, aborted requests' tokens vanish from /v1/status).
        if request is not None and request.num_output_tokens > 0:
            self.total_completion_tokens += request.num_output_tokens
            self.total_prompt_tokens += request.num_prompt_tokens

        if request is not None:
            request.set_finished(RequestStatus.FINISHED_ABORTED)
            # Release cache references so Metal buffers can be freed
            request.prompt_cache = None
            request._extracted_cache = None
        self.finished_req_ids.add(request_id)
        self._cleanup_detokenizer(request_id)

        # M-01 codex r4 BLOCKING #1: do NOT discard the dedupe
        # ledgers here. The text scheduler intentionally keeps the
        # ``Request`` object in ``self.requests`` between
        # ``_do_abort_request`` and the later
        # ``remove_finished_request`` call (engine_core cleanup),
        # so ``abort_request`` would still observe
        # ``request_id in self.requests`` and admit a redundant
        # enqueue. Discarding the dedupe ledger HERE would let
        # that redundant enqueue double-count the same request
        # lifetime. The discard happens in ``remove_finished_request``
        # instead — by which point the request has truly left every
        # admit-able map and a fresh ``abort_request`` could only
        # land via a new admit() with the same id (a distinct
        # lifetime).

        # Flush Metal encoders after removing arrays from batch
        mx.clear_cache()

        logger.info(
            f"[abort_request] {request_id[:12]} ABORTED "
            f"was_waiting={was_waiting} was_running={was_running} "
            f"removed_from_batch={removed_from_batch} "
            f"remaining_running={len(self.running)} remaining_waiting={len(self.waiting)}"
        )
        return True

    def has_requests(self) -> bool:
        """Check if there are any pending or running requests."""
        return bool(self.waiting or self.running)

    def get_num_waiting(self) -> int:
        """Get number of waiting requests."""
        return len(self.waiting)

    def get_num_running(self) -> int:
        """Get number of running requests."""
        return len(self.running)

    def _resolve_exact_hit_tokens(self, request) -> list:
        """Resolve ``tokens_to_process`` for an exact cache hit (remaining == []).

        A warm exact-repeat re-forwards the last prompt token; to stay
        byte-equal to a cold prefill the saved cache must be trimmed by 1 (it
        un-writes the doubled last token). That trim requires EVERY layer to be
        trimmable. A rotated ``RotatingKVCache`` (sliding-window: Gemma 4 /
        GPT-OSS, once ``offset >= max_size`` → ``is_trimmable()`` False) is NOT
        trimmable, and ``RotatingKVCache.trim`` cannot un-write the rotated slot
        — so a skipped trim would double-count the last token and drift the
        first generated token (verified on gpt-oss-20b: a borderline-confidence
        greedy token can flip).

        Correctness-first fallback: on a non-trimmable exact hit, drop the
        reused cache and full-prefill (byte-equal to cold by construction). The
        agent-loop win — a stable prefix + a NEW suffix (prefix-EXTENSION,
        remaining >= 1) — is trim-free and already byte-exact, handled by the
        caller's ``elif request.remaining_tokens`` branch and untouched by this
        fallback. Only an identical re-request of a rotated prompt loses reuse.
        """
        tokens_to_process = request.prompt_token_ids[-1:]
        if request.prompt_cache is None:
            return tokens_to_process
        try:
            from mlx_lm.models.cache import (
                can_trim_prompt_cache,
                trim_prompt_cache,
            )

            if can_trim_prompt_cache(request.prompt_cache):
                trim_prompt_cache(request.prompt_cache, 1)
            else:
                logger.debug(
                    "[cache_fetch] exact-hit on non-trimmable "
                    "(rotated sliding-window) cache for request=%s: "
                    "trim(1) impossible, full-prefilling to stay "
                    "byte-equal to cold",
                    request.request_id[:12],
                )
                request.prompt_cache = None
                request.cached_tokens = 0
                request.remaining_tokens = request.prompt_token_ids
                tokens_to_process = request.prompt_token_ids
        except Exception as _trim_exc:  # noqa: BLE001
            # Any trim failure (inspection raised, or a partial/half-applied
            # trim) leaves the reused cache in an unknown state. Re-forwarding
            # only the last token on top of an un-trimmed cache reintroduces the
            # exact-hit drift this helper exists to prevent, so fall back to the
            # SAME cold full-prefill as the non-trimmable branch — byte-equal to
            # cold by construction, never drifting.
            logger.debug(
                "[cache_fetch] exact-hit trim(1) failed for "
                "request=%s: %s (dropping reused cache and full-prefilling "
                "to stay byte-equal to cold)",
                request.request_id[:12],
                _trim_exc,
            )
            request.prompt_cache = None
            request.cached_tokens = 0
            request.remaining_tokens = request.prompt_token_ids
            return request.prompt_token_ids
        return tokens_to_process

    def _schedule_waiting(self) -> list[Request]:
        """
        Move requests from waiting queue to running.

        Returns:
            List of requests that were scheduled
        """
        scheduled = []

        while self.waiting and len(self.running) < self.config.max_num_seqs:
            request = self.waiting.popleft()

            # Ensure we have a batch generator. The False return means
            # the live generator has incompatible stop_tokens / sampler
            # for this request and is still draining — we must NOT admit
            # into the stale generator (Rapid-MLX #611 / codex P2 on
            # PR #612). Requeue and break so the next ``step`` retries
            # once the running batch completes.
            if not self._ensure_batch_generator(request.sampling_params):
                self.waiting.appendleft(request)
                break

            if self.batch_generator is None:
                # Put back and try again later
                self.waiting.appendleft(request)
                break

            # Determine tokens to process and cache to use
            # Note: Don't use `remaining_tokens or prompt_token_ids` because empty list
            # is falsy in Python. For exact cache match, remaining_tokens=[] but we should
            # pass just the last token so BatchGenerator can start generation.
            if (
                request.remaining_tokens is not None
                and len(request.remaining_tokens) == 0
            ):
                # Exact cache match — pass only the last token for
                # generation kickoff. The saved cache captured state at
                # offset=N (all N prompt tokens processed).
                # ``PromptProcessingBatch.generate([last_token])`` then
                # calls ``GenerationBatch.__init__(inputs=last_token)``
                # which invokes ``_step()``. That step forwards the last
                # token through the model with ``cache=prompt_cache``,
                # writing K/V at position N and advancing offset to N+1.
                # The result: the last prompt token appears at TWO cache
                # positions (N-1 from the saved cache, N from the re-fed
                # step), the sampling query is emitted at position N+1
                # (with a shifted RoPE), and the softmax denominator
                # includes an extra spurious K/V. That drifts the first
                # output token vs. the fresh-prefill baseline (which
                # samples at position N with cache offset=N-1 → N).
                #
                # Fix: trim the cache offset by 1 before the batch
                # generator picks it up. The last prompt token's K/V is
                # discarded from cache; ``_step()``'s forward then
                # overwrites position N-1 in-place, ending at offset=N.
                # Position and softmax denominator now match the fresh
                # path exactly, restoring byte-equal output between a
                # cold prompt and a warm-cache repeat.
                # Trim vs non-trimmable fallback is factored into a helper so
                # the correctness-critical branch is unit-testable in isolation
                # (see tests/test_sliding_window_prefix_reuse.py).
                tokens_to_process = self._resolve_exact_hit_tokens(request)
            elif request.remaining_tokens:
                tokens_to_process = request.remaining_tokens
            else:
                tokens_to_process = request.prompt_token_ids
            cache_to_use = request.prompt_cache  # May be None

            # Validate cache before using it
            if cache_to_use is not None and not self._validate_cache(cache_to_use):
                logger.debug(
                    f"Request {request.request_id}: invalid cache detected, "
                    f"proceeding without cache"
                )
                cache_to_use = None
                request.prompt_cache = None
                request.cached_tokens = 0
                request.remaining_tokens = request.prompt_token_ids
                tokens_to_process = request.prompt_token_ids

            # Prefix-cache HIT with live quantization on: normalize the restored
            # KVCache layers into the same _QuantizableKVCache the MISS path uses,
            # so mlx-lm merges them into a QuantizedBatchKVCache rather than a bf16
            # BatchKVCache that then crashes when extended against a quantized
            # batch (#1197).
            _lq = getattr(self, "_live_kv_quant", None)
            if _lq is not None and cache_to_use:
                from .quantized_batch_cache import (
                    normalize_caches_for_quantization,
                )

                cache_to_use = normalize_caches_for_quantization(
                    cache_to_use, _lq[0], _lq[1]
                )

            # Insert into BatchGenerator with optional cache.
            # Wrap in try/except: if cache shapes are incompatible
            # (e.g. stale entry after BatchGenerator recreation),
            # fall back to no-cache insert instead of crashing.
            # Create per-request logits processors. The tool-bias factory is
            # gated on ``has_tools``: a per-token Python processor on every
            # row is pure decode overhead for plain-chat requests (minimax/
            # gpt-oss attached one unconditionally before, ~every request on
            # a --tool-call-parser minimax server).
            request_processors: list = []
            if self._tool_logits_processor_factory and request.has_tools:
                processor = self._tool_logits_processor_factory()
                if processor is not None:
                    request_processors.append(processor)
            # Grammar-constrained tool calling (#558): a per-request
            # ``GrammarLogitsProcessor`` set on the Request masks decoding to
            # the tool-call grammar. Same per-token slot as the (unused today)
            # MiniMax soft-bias factory above, so it composes with penalties.
            _glp = getattr(request, "grammar_logits_processor", None)
            if _glp is not None:
                request_processors.append(_glp)
            # Prevent an exact agent loop before it reaches the streaming
            # hard-stop below.  The processor is deliberately tool-request
            # only and masks a single predicted token only after the output is
            # one full copy short of the conservative abort threshold.
            if request.has_tools:
                _loop_breaker = AgentRepetitionLogitsProcessor(request.output_token_ids)
                request._repetition_logits_processor = _loop_breaker
                request_processors.append(_loop_breaker)
            # Penalty knobs (#355) — only add the processor when at least
            # one penalty is non-default. mlx-lm's make_logits_processors
            # returns an empty list when all knobs are at defaults, but
            # constructing it unconditionally would still allocate the
            # context-tracking arrays for every request.
            #
            # OpenAI-spec penalties (frequency/presence) are defined over
            # the entire generated sequence, not a sliding window. mlx-lm's
            # default context_size of 20 truncates the visibility window so
            # aggressively that callers report the penalty "feels like a
            # no-op" on chat-length outputs (#470). We bump the OpenAI-spec
            # ones to 4096 — enough to cover the vast majority of chat
            # responses without bloating per-request arrays. Repetition
            # penalty stays at mlx-lm's default 20 since it's a rapid-mlx
            # extension (not OpenAI-spec) and is documented as multiplicative
            # over a rolling window.
            sp = request.sampling_params
            if (
                sp.repetition_penalty != 1.0
                or sp.presence_penalty != 0.0
                or sp.frequency_penalty != 0.0
            ):
                request_processors.extend(
                    make_logits_processors(
                        repetition_penalty=(
                            sp.repetition_penalty
                            if sp.repetition_penalty != 1.0
                            else None
                        ),
                        presence_penalty=(
                            sp.presence_penalty if sp.presence_penalty != 0.0 else None
                        ),
                        presence_context_size=4096,
                        frequency_penalty=(
                            sp.frequency_penalty
                            if sp.frequency_penalty != 0.0
                            else None
                        ),
                        frequency_context_size=4096,
                    )
                )
            # Generation-time thinking-token budget (force-close </think>).
            # Appended LAST so its force-close mask (all but </think> -> -inf)
            # has final say over any penalty/grammar bias in the same step;
            # it is inert (returns logits unchanged) once thinking has ended,
            # so a chained grammar processor owns the generation phase.
            _rblp = getattr(request, "reasoning_budget_logits_processor", None)
            if _rblp is not None:
                request_processors.append(_rblp)
            _stlp = getattr(request, "suppressed_tokens_logits_processor", None)
            if _stlp is not None:
                request_processors.append(_stlp)
            request_logits_processors = (
                [request_processors] if request_processors else None
            )

            # Per-request sampler (temperature/top_p/top_k/min_p may differ
            # per request). Without this, all requests use the BatchGenerator's
            # default sampler (argmax), ignoring the requested temperature.
            # ``_get_request_sampler`` interns by sampling-param tuple so that
            # homogeneous batches share one callable — required for
            # ``_install_dense_sampler_fastpath`` to detect them by identity.
            request_sampler = self._get_request_sampler(request.sampling_params)

            # Issue #427: split the insert at prefix_boundary so the
            # per-message cache snapshot can fire after the prefix
            # segment prefills (see _snapshot_boundary_segments). Only
            # useful when (a) we have somewhere to save, (b) the request
            # has a multi-turn shared prefix set, and (c) the boundary
            # lies strictly inside the tokens we're about to process —
            # otherwise there's nothing new to capture at the boundary.
            boundary_local_split: int | None = None
            snapshot_boundary = getattr(request, "prefix_boundary", 0)
            # A non-trimmable exact hit cannot use the usual trim-one then
            # re-forward-last-token kickoff. Capture the cold prompt at N-1;
            # an identical repeat becomes a safe one-token prefix extension.
            if (
                snapshot_boundary <= 0
                and self.memory_aware_cache is not None
                and getattr(self.config, "hybrid_cache_entries", 0) > 0
                and getattr(self.config, "non_trimmable_exact_prefix_reuse", False)
                and not request.prompt_cache
                and len(request.prompt_token_ids) > 1
            ):
                snapshot_boundary = len(request.prompt_token_ids) - 1
                request._cache_snapshot_boundary = snapshot_boundary
                request._cache_snapshot_is_internal = True
            if (
                self.memory_aware_cache is not None
                and snapshot_boundary > 0
                and len(tokens_to_process) > 1
            ):
                _pb = snapshot_boundary
                _cached = request.cached_tokens or 0
                _local = _pb - _cached
                if 0 < _local < len(tokens_to_process):
                    boundary_local_split = _local

            try:
                if boundary_local_split is not None:
                    uids = self.batch_generator.insert_segments(
                        [
                            [
                                tokens_to_process[:boundary_local_split],
                                tokens_to_process[boundary_local_split:],
                            ]
                        ],
                        max_tokens=[request.sampling_params.max_tokens],
                        caches=[cache_to_use] if cache_to_use else None,
                        samplers=[request_sampler],
                        logits_processors=request_logits_processors,
                    )
                else:
                    uids = self.batch_generator.insert(
                        [tokens_to_process],
                        max_tokens=[request.sampling_params.max_tokens],
                        caches=[cache_to_use] if cache_to_use else None,
                        samplers=[request_sampler],
                        logits_processors=request_logits_processors,
                    )
            except Exception as e:
                if cache_to_use is not None:
                    logger.warning(
                        f"[cache_insert_error] request={request.request_id[:12]} "
                        f"cache insert failed ({e}), retrying without cache"
                    )
                    cache_to_use = None
                    request.prompt_cache = None
                    request.cached_tokens = 0
                    request.remaining_tokens = request.prompt_token_ids
                    tokens_to_process = request.prompt_token_ids
                    # Recompute split against the now-full prompt
                    # (cached_tokens=0 so boundary == split).
                    retry_boundary = getattr(
                        request,
                        "_cache_snapshot_boundary",
                        getattr(request, "prefix_boundary", 0),
                    )
                    if self.memory_aware_cache is not None and 0 < retry_boundary < len(
                        tokens_to_process
                    ):
                        uids = self.batch_generator.insert_segments(
                            [
                                [
                                    tokens_to_process[:retry_boundary],
                                    tokens_to_process[retry_boundary:],
                                ]
                            ],
                            max_tokens=[request.sampling_params.max_tokens],
                            caches=None,
                            samplers=[request_sampler],
                            logits_processors=request_logits_processors,
                        )
                    else:
                        uids = self.batch_generator.insert(
                            [tokens_to_process],
                            max_tokens=[request.sampling_params.max_tokens],
                            caches=None,
                            samplers=[request_sampler],
                            logits_processors=request_logits_processors,
                        )
                else:
                    raise

            if uids:
                uid = uids[0]
                self.request_id_to_uid[request.request_id] = uid
                self.uid_to_request_id[uid] = request.request_id
                request.batch_uid = uid
                request.status = RequestStatus.RUNNING
                request._prefill_started_at = time.time()
                # #558 PR-3 / #558 budget: record this request's FULL processor
                # list (grammar + penalties + budget) by uid as the authoritative
                # state the per-tick realign guard rebuilds from — immune to
                # mlx-lm's stale-entry desync. Extracted into a production helper
                # so the realign tests exercise the SAME registration (codex NIT).
                self._register_uid_processors(uid, request, request_processors, _glp)
                # Attach incremental decoder for multi-byte safe streaming
                request._decoder = IncrementalDecoder(self._actual_tokenizer)
                # Release the prompt cache reference now that BatchGenerator
                # has its own copy.  Holding this reference prevents MLX from
                # freeing the Metal buffers until the request object is GC'd,
                # which under sustained traffic can accumulate hundreds of GB
                # of wired memory (issue #442).
                request.prompt_cache = None
                self.running[request.request_id] = request
                scheduled.append(request)

                self.total_prompt_tokens += request.num_prompt_tokens
                cache_info = (
                    f", {request.cached_tokens} cached"
                    if request.cached_tokens > 0
                    else ""
                )
                tokens_to_prefill = len(tokens_to_process)
                logger.info(
                    f"[schedule] request={request.request_id[:12]} uid={uid} "
                    f"prompt_tokens={request.num_prompt_tokens} "
                    f"tokens_to_prefill={tokens_to_prefill}{cache_info} "
                    f"max_tokens={request.sampling_params.max_tokens} "
                    f"running={len(self.running)} waiting={len(self.waiting)}"
                )

        return scheduled

    def _process_batch_responses(
        self, responses: list[Any]
    ) -> tuple[list[RequestOutput], set[str]]:
        """
        Process responses from BatchGenerator.

        Args:
            responses: List of BatchGenerator.Response objects

        Returns:
            Tuple of (outputs, finished_request_ids)
        """
        outputs = []
        finished_ids = set()
        prompt_tps_this_batch = 0.0

        for response in responses:
            request_id = self.uid_to_request_id.get(response.uid)
            if request_id is None:
                continue

            request = self.running.get(request_id)
            if request is None:
                continue

            # Append token to request
            request.append_output_token(response.token)

            # R15-P1 (task #296): trigger disk-backed KV checkpoint at
            # 256-tok boundaries. Cheap when disabled — the helper
            # short-circuits on ``interval <= 0`` so the only cost on
            # the hot path for operators who haven't opted in is one int
            # comparison. The actual cache extraction + safetensors
            # write happens off the response loop; failures are logged
            # and never tear the response down (best-effort persistence,
            # mirrors the in-process prefix-cache contract). Delegates
            # to ``_safe_disk_checkpoint`` so the silent-swallow
            # regression guard has a tested entry point — see the
            # method docstring for the wrong-attribute typos #919
            # shipped that motivated this split.
            self._safe_disk_checkpoint(request, response)

            # Record first token time for TTFT metric
            if request.first_token_time is None and request.num_output_tokens > 0:
                import time as _time

                request.first_token_time = _time.time()
                prefill_s = request.first_token_time - getattr(
                    request, "_prefill_started_at", request.arrival_time
                )
                if prefill_s > 0:
                    prompt_tps_this_batch += request.num_prompt_tokens / prefill_s

            if request.first_token_time is not None and request.num_output_tokens > 0:
                generation_s = time.time() - request.first_token_time
                if generation_s > 0:
                    self._last_generation_tps = request.num_output_tokens / generation_s

            # Decode the new token using IncrementalDecoder for multi-byte
            # safety (emoji, CJK). Skip stop tokens — they are not content.
            if response.finish_reason == "stop":
                new_text = ""
            else:
                decoder = getattr(request, "_decoder", None)
                if decoder is not None:
                    new_text = decoder.add_token(response.token)
                else:
                    new_text = self._decode_tokens([response.token])

            # output_token_ids is a live reference (not a defensive copy):
            # consumers read it synchronously; the per-decode list() was O(n).
            output = RequestOutput(
                request_id=request_id,
                new_token_ids=[response.token],
                new_text=new_text,
                output_token_ids=request.output_token_ids,
                prompt_tokens=request.num_prompt_tokens,
                completion_tokens=request.num_output_tokens,
                cached_tokens=request.cached_tokens,
                logprobs=response.logprobs,
            )

            # Check text-based stop sequences. ``SamplingParams.stop`` is a
            # list of user-supplied strings (OpenAI-API contract); mlx-lm's
            # BatchGenerator only honours ``stop_token_ids``, so we have to
            # match-and-truncate on the decoded output here. MLLMScheduler
            # has had the equivalent check since launch; the text scheduler
            # was silently dropping ``request.stop`` until #354 / regression
            # tests 1, 2, 4, 5 surfaced the gap.
            #
            # Surface choice: the IncrementalDecoder's ``get_full_text()`` is
            # the AUTHORITATIVE surface for stop matching — it is what the
            # client sees on the streaming path, byte-for-byte. The previous
            # implementation called ``self._decode_tokens()`` (a fresh
            # ``tokenizer.decode(token_ids)`` with the wrapper's default
            # ``skip_special_tokens=True``), which on tokenizer families
            # whose default decoding strips text the streaming detokenizer
            # preserves (Phi-3.5 / Gemma-3n SentencePiece variants surfaced
            # in the 2026-06-18 fuzz battery) produced a window that did
            # NOT contain the literal stop string the user was looking
            # for — even though the streamed surface DID contain it. Using
            # the incremental decoder closes this skew at the sampler
            # layer without tokenizer-family-specific casing.
            finish_reason = response.finish_reason
            stop_trimmed = False
            _loop_breaker = getattr(request, "_repetition_logits_processor", None)
            if isinstance(_loop_breaker, AgentRepetitionLogitsProcessor):
                _reported = getattr(request, "_reported_repetition_breaks", 0)
                if _loop_breaker.interventions > _reported:
                    _new_breaks = _loop_breaker.interventions - _reported
                    self.num_repetition_loop_breaks += _new_breaks
                    request._reported_repetition_breaks = _loop_breaker.interventions
                    _match = _loop_breaker.last_match
                    logger.warning(
                        "Broke exact token loop for agent request %s "
                        "before stream abort (period_tokens=%s repeats=%s "
                        "interventions=%d)",
                        request_id,
                        getattr(_match, "period_tokens", "unknown"),
                        getattr(_match, "repeats", "unknown"),
                        _loop_breaker.interventions,
                    )
            # Agent models can occasionally enter a perfectly periodic decode
            # loop after a tool interaction.  Once streamed, those deltas cannot
            # be retracted, so stop it in the scheduler before thousands of
            # useless tokens consume minutes of wall time.  Restrict this to
            # tool-bearing requests and exact, long token repetition: ordinary
            # chat/creative completions keep their historical semantics.
            repetition_match = None
            repetition_error: str | None = None
            if (
                finish_reason is None
                and request.has_tools
                and request.num_output_tokens % 8 == 0
            ):
                repetition_match = detect_repeated_token_suffix(
                    request.output_token_ids
                )
                if repetition_match is not None:
                    finish_reason = "abort"
                    repetition_error = (
                        "Model generation aborted: exact repetition loop "
                        "detected "
                        f"(period_tokens={repetition_match.period_tokens}, "
                        f"repeats={repetition_match.repeats})"
                    )
                    self.num_repetition_loop_stops += 1
                    logger.warning(
                        "Stopping agent request %s after exact token loop "
                        "(period_tokens=%d repeats=%d completion_tokens=%d)",
                        request_id,
                        repetition_match.period_tokens,
                        repetition_match.repeats,
                        request.num_output_tokens,
                    )
            stop_params = request.sampling_params.stop or []
            if finish_reason is None and stop_params:
                decoder = getattr(request, "_decoder", None)
                if decoder is not None:
                    decoded_so_far = decoder.get_full_text()
                else:
                    decoded_so_far = self._decode_tokens(request.output_token_ids)
                # #1049 — for harmony-format models the stop-string
                # search must be scoped to the ``final`` channel body:
                # analysis-channel CoT routinely mentions user-supplied
                # stop markers while reasoning (agents like OpenHands
                # CodeActAgent set ``stop=['</execute_ipython>', ...]``
                # and the CoT names those markers verbatim), so a raw-
                # stream match prematurely terminates the request
                # before the final channel emits any content. Non-
                # harmony models keep the raw-stream match unchanged.
                # Non-harmony path preserves the pre-#1049 iteration-
                # order semantics (first stop-string in ``stop_params``
                # that appears anywhere in ``decoded_so_far`` wins) so
                # this change is a strict superset for harmony models
                # and a no-op for everyone else.
                stop_match: tuple[str, int] | None = None
                if self._is_harmony_family:
                    from .reasoning.harmony_stop import find_stop_in_final_channel

                    stop_match = find_stop_in_final_channel(decoded_so_far, stop_params)
                else:
                    for stop_str in stop_params:
                        if stop_str and stop_str in decoded_so_far:
                            stop_match = (
                                stop_str,
                                decoded_so_far.index(stop_str),
                            )
                            break
                if stop_match is not None:
                    stop_str, idx = stop_match
                    finish_reason = "stop"
                    # H-03: pin WHICH user-supplied stop fired so
                    # the Anthropic adapter can surface
                    # ``stop_reason="stop_sequence"`` +
                    # ``stop_sequence: <str>`` per the public spec.
                    # OpenAI's ``finish_reason="stop"`` bucket
                    # already lumps EOS and stop-string together so
                    # the OpenAI surface ignores this field.
                    output.matched_stop = stop_str
                    trimmed_total = decoded_so_far[:idx]
                    request.output_text = trimmed_total
                    stop_trimmed = True
                    # Adjust new_text so streaming clients only see the
                    # valid prefix, never the stop marker itself.
                    # Pre-token streaming surface ≡ the decoder's
                    # ``prev_text``  — what the client has seen so
                    # far. Computing it as ``decoded_so_far -
                    # new_text`` is fragile: when the incremental
                    # decoder holds back a U+FFFD-incomplete
                    # sequence ``new_text == ""`` but
                    # ``decoded_so_far`` grew, so the subtraction
                    # math reset to the wrong boundary and could
                    # leak or drop text on multibyte streams
                    # (codex r8 BLOCKING). Falling back to
                    # ``decoded_so_far - new_text`` only when no
                    # decoder is attached (text-only paths that
                    # decode in bulk).
                    if decoder is not None:
                        prev_text = decoder.prev_text
                    else:
                        prev_text = (
                            decoded_so_far[: -len(new_text)]
                            if new_text
                            else decoded_so_far
                        )
                    if len(trimmed_total) > len(prev_text):
                        output.new_text = trimmed_total[len(prev_text) :]
                    else:
                        output.new_text = ""

            # Check if finished
            if finish_reason is not None:
                response.finish_reason = finish_reason
                if response.finish_reason == "stop":
                    request.set_finished(RequestStatus.FINISHED_STOPPED)
                elif response.finish_reason == "length":
                    request.set_finished(RequestStatus.FINISHED_LENGTH_CAPPED)
                elif response.finish_reason == "abort":
                    request.set_finished(RequestStatus.FINISHED_ABORTED)

                output.finished = True
                output.finish_reason = response.finish_reason
                if repetition_error is not None:
                    output.error = repetition_error
                    # Mark this abort as the graceful repetition-guard stop so
                    # the engine returns the partial output as 200 (finish_reason
                    # remapped) instead of raising 503 like a genuine runtime
                    # abort. Other abort sources (Metal/OOM recovery) leave
                    # error_kind None → they stay on the raise→503 path.
                    output.error_kind = "repetition"
                finished_ids.add(request_id)

                if stop_trimmed:
                    # request.output_text was already truncated to the prefix
                    # before the stop string — using that as the final output
                    # preserves the truncation; re-decoding here would put the
                    # stop marker back in.
                    output.output_text = request.output_text
                    self._cleanup_detokenizer(request_id)
                else:
                    # Decode full output using decoder if available (ensures
                    # any held-back multi-byte chars are flushed)
                    decoder = getattr(request, "_decoder", None)
                    if decoder is not None:
                        output.output_text = decoder.get_full_text()
                    else:
                        output.output_text = self._decode_tokens(
                            request.output_token_ids
                        )
                    request.output_text = output.output_text
                    self._cleanup_detokenizer(request_id)

                # Extract cache for future reuse (critical for agentic multi-turn)
                if hasattr(response, "prompt_cache"):
                    try:
                        # prompt_cache may be callable or direct attribute
                        if callable(response.prompt_cache):
                            raw_cache = response.prompt_cache()
                        else:
                            raw_cache = response.prompt_cache

                        if raw_cache:
                            # For paged cache, extract actual tensor states
                            # This allows cache to survive BatchGenerator recreation
                            if self.block_aware_cache is not None:
                                extracted_cache = self._extract_cache_states(raw_cache)
                                if extracted_cache:
                                    request._extracted_cache = extracted_cache
                                    logger.debug(
                                        f"Extracted {len(extracted_cache)} layer states "
                                        f"for request {request_id}"
                                    )
                            else:
                                # Standard cache stores object references
                                request._extracted_cache = raw_cache
                    except Exception as e:
                        logger.debug(f"Failed to extract cache for {request_id}: {e}")

                self.total_completion_tokens += request.num_output_tokens
                self.num_requests_processed += 1

                logger.debug(
                    f"Request {request_id} finished: {response.finish_reason}, "
                    f"{request.num_output_tokens} tokens"
                )

            outputs.append(output)

        if prompt_tps_this_batch > 0:
            self._last_prompt_tps = prompt_tps_this_batch
        return outputs, finished_ids

    def _safe_disk_checkpoint(self, request: Request, response: Any) -> None:
        """Wrap ``_maybe_disk_checkpoint`` in a never-raise contract.

        Every *expected* skip path inside ``_maybe_disk_checkpoint`` is
        an explicit early-return — interval disabled, request has no
        active batch, cache extraction not yet available, etc. Any
        exception that reaches this wrapper is by definition
        unexpected: the wrong-attribute typos shipped in PR #919
        (``self.scheduler_config`` for the config and ``self.batch_gen``
        for the BatchGenerator) raised AttributeError here every step
        and stayed silent at ``logger.debug`` for two releases. The
        wrapper surfaces them at ``warning`` and bumps a Prometheus
        error counter so the next class of bug is visible in both the
        log and ``/metrics``. The wrapper itself never raises — a bug
        in disk-IO must not crash the decode path on a live server.

        Tested in ``tests/test_scheduler_disk_kv_hook.py`` —
        ``test_safe_disk_checkpoint_records_silent_failure`` is the
        explicit regression guard for the silent-swallow class of bug.
        """
        try:
            self._maybe_disk_checkpoint(request, response)
        except Exception as _ckpt_err:  # pragma: no cover — defensive
            # Late import so a broken disk_kv_checkpoint module
            # (e.g. ImportError on a stripped-down deployment) never
            # bubbles up from the error path itself.
            try:
                from .runtime import disk_kv_checkpoint as _dkc_err

                _dkc_err.record_hook_error()
            except Exception:  # pragma: no cover — defensive of the defensive
                pass
            logger.warning(
                "[kv_checkpoint] hook raised for %s: %r",
                request.request_id,
                _ckpt_err,
            )

    def _maybe_disk_checkpoint(self, request: Request, response: Any) -> None:
        """Trigger a disk-backed KV checkpoint at the next 256-tok boundary.

        R15-P1 (task #296) hook. Called once per response from
        ``_process_batch_responses`` after the token has been appended to
        the request. Gated tightly so disabled servers pay nothing:

        - ``self.config.kv_disk_checkpoint_interval == 0`` →
          immediate return. This is the dominant case for the first wave
          of deploys (operators opt in deliberately).
        - Lazy per-request bookkeeping: a ``_kv_checkpoint_state`` attr
          is attached to ``request`` on first crossing so the watermark
          survives across steps.
        - Cache extraction is best-effort — the active batch is the
          authoritative source via ``batch.extract_cache(e)``; when the
          batch doesn't expose it (e.g. between steps, during
          chunked-prefill finalization, or on a hybrid generator without
          the upstream API), the watermark stays put and the next step
          gets another shot.

        Any failure is swallowed with a debug log; the caller wraps this
        whole method in a broad try/except as belt-and-suspenders.
        """
        interval = getattr(self.config, "kv_disk_checkpoint_interval", 0)
        if interval is None or interval <= 0:
            return

        # Lazy import keeps the module-load cost of vllm_mlx.scheduler
        # zero when the disk checkpoint feature is never used (the runtime
        # subpackage imports mlx_lm symbols that aren't free).
        from .runtime import disk_kv_checkpoint as _dkc

        # Total tokens already in the cache — prompt + every output token
        # we have already appended. ``num_tokens`` is the canonical sum
        # on the Request dataclass and accounts for PFlash's bypass shape.
        num_tokens = request.num_tokens

        state = getattr(request, "_kv_checkpoint_state", None)
        if state is None:
            state = _dkc.RequestCheckpointState(
                req_hash=_dkc.request_hash(
                    request.request_id, model_name=getattr(self, "_model_name", None)
                ),
                interval=interval,
                last_checkpoint_at=0,
                requires_full_checkpoint=_dkc.model_requires_full_checkpoint(
                    getattr(self, "_model_name", None)
                ),
                kv_dtype=getattr(self.config, "kv_cache_dtype", "bf16") or "bf16",
                model_name=getattr(self, "_model_name", None),
            )
            request._kv_checkpoint_state = state

        # A serializer failure is cache-shape specific and will not recover on
        # the next decode token.  Retrying every step would repeatedly walk a
        # potentially huge cache and flood logs for the rest of the request.
        if state.disabled:
            return

        if not _dkc.should_checkpoint(num_tokens, state.last_checkpoint_at, interval):
            return

        # Try to pull the cache off the active batch. The mlx-lm 0.31+
        # GenerationBatch lives on ``batch_gen._generation_batch`` and
        # exposes ``extract_cache(e)``; older builds expose the same
        # method directly on ``batch_gen.active_batch``. Walking both
        # surfaces keeps this hook portable across the mlx-lm versions
        # rapid-mlx supports.
        batch = getattr(self, "batch_generator", None)
        if batch is None:
            return
        gen_batch = getattr(batch, "_generation_batch", None) or getattr(
            batch, "active_batch", None
        )
        if gen_batch is None:
            return
        try:
            uids = list(gen_batch.uids)
        except AttributeError:
            return
        try:
            e = uids.index(request.batch_uid)
        except (ValueError, AttributeError):
            return

        try:
            cache = gen_batch.extract_cache(e)
        except Exception:
            return
        if not cache:
            return

        new_offset, _path = _dkc.maybe_write_checkpoint(
            cache,
            root=_dkc.get_default_root(),
            req_hash=state.req_hash,
            num_tokens=num_tokens,
            last_checkpoint_at=state.last_checkpoint_at,
            interval=interval,
            kv_dtype=state.kv_dtype,
            requires_full_checkpoint=state.requires_full_checkpoint,
            model_name=state.model_name,
        )
        state.last_checkpoint_at = new_offset
        if _path is None:
            state.disabled = True
            return

        # Cheap disk-cap check: only fires when bytes actually moved.
        # The enforce_disk_cap helper is itself lock-guarded so racing
        # write/evict callers serialize correctly.
        try:
            _dkc.enforce_disk_cap(_dkc.get_default_root())
        except Exception as _evict_err:  # pragma: no cover — defensive
            # Promoted from debug to warning + error counter for the
            # same reason as the outer wrapper at the call site: every
            # expected skip is an early-return inside enforce_disk_cap,
            # so anything reaching here is an unexpected fault.
            try:
                _dkc.record_hook_error()
            except Exception:  # pragma: no cover — defensive of the defensive
                pass
            logger.warning("[kv_checkpoint] enforce_disk_cap failed: %r", _evict_err)

    def _cleanup_finished(self, finished_ids: set[str]) -> None:
        """Clean up finished requests and store caches for reuse."""
        for request_id in finished_ids:
            request = self.running.get(request_id)

            # PFlash bypass: compressed requests skip the prefix-cache
            # store entirely. Their prompt_token_ids holds the
            # compressed subsequence so a stored entry would be keyed by
            # positions that do not match any real prompt prefix.
            pflash_skip_store = request is not None and _pflash_compressed(request)

            # Store cache for future reuse
            if (
                request is not None
                and request.prompt_token_ids
                and not pflash_skip_store
            ):
                if self.block_aware_cache is not None:
                    # Store in paged cache
                    # Key includes both prompt and output tokens for multi-turn chat caching
                    if (
                        hasattr(request, "_extracted_cache")
                        and request._extracted_cache is not None
                    ):
                        try:
                            full_token_sequence = list(request.prompt_token_ids) + list(
                                request.output_token_ids
                            )
                            self.block_aware_cache.store_cache(
                                request_id,
                                full_token_sequence,
                                request._extracted_cache,
                            )
                            logger.debug(
                                f"Stored paged cache for request {request_id} "
                                f"({len(full_token_sequence)} tokens: {len(request.prompt_token_ids)} prompt + {len(request.output_token_ids)} output)"
                            )
                        except Exception as e:
                            logger.debug(
                                f"Failed to store paged cache for {request_id}: {e}"
                            )
                    # NOTE: Do NOT call release_cache here - blocks should persist
                    # for future requests to share. The LRU eviction will clean up
                    # unused blocks when under memory pressure.

                elif self.memory_aware_cache is not None:
                    # Keep mid-prefill entry as prefix cache for future
                    # requests that share a common prefix (e.g. same system
                    # prompt + tools but different user message).  LRU
                    # eviction handles memory pressure.

                    # Store in memory-aware prefix cache
                    # Key includes both prompt and output tokens for multi-turn chat caching
                    if (
                        hasattr(request, "_extracted_cache")
                        and request._extracted_cache is not None
                    ):
                        try:
                            full_token_sequence = list(request.prompt_token_ids) + list(
                                request.output_token_ids
                            )
                            import time as _time

                            _store_t0 = _time.monotonic()
                            stored = self.memory_aware_cache.store(
                                full_token_sequence,
                                request._extracted_cache,
                                evict_prefixes=False,
                            )
                            _store_dt = _time.monotonic() - _store_t0
                            # NOTE: We intentionally do NOT store a prompt-only
                            # cache entry.  Hybrid Mamba+Transformer models
                            # (like Qwen3-Coder-Next) have MambaCache layers
                            # whose state is cumulative and cannot be trimmed
                            # back to "prompt only".  Reusing such state causes
                            # the model to immediately produce EOS.
                            # The full prompt+output entry is stored above; a
                            # future request with the same prompt will hit the
                            # supersequence match path in the fetch, which is
                            # now disabled for safety (see memory_cache.py).

                            logger.info(
                                f"[cache_store] request={request_id[:12]} "
                                f"tokens={len(full_token_sequence)} "
                                f"({len(request.prompt_token_ids)} prompt + {len(request.output_token_ids)} output) "
                                f"stored={stored} time={_store_dt:.3f}s "
                                f"cache_entries={len(self.memory_aware_cache._entries)} "
                                f"cache_mem={self.memory_aware_cache._current_memory / 1e6:.0f}MB"
                            )
                            # Release the original FP16 cache reference so
                            # memory can be reclaimed (the quantized copy
                            # lives inside the prefix cache now).
                            request._extracted_cache = None
                        except Exception as e:
                            logger.debug(
                                f"Failed to store memory-aware cache for {request_id}: {e}"
                            )

                elif self.prefix_cache is not None:
                    # Store in legacy prefix cache
                    # Key includes both prompt and output tokens for multi-turn chat caching
                    # The next turn's prompt will include the previous response
                    if (
                        hasattr(request, "_extracted_cache")
                        and request._extracted_cache is not None
                    ):
                        try:
                            full_token_sequence = list(request.prompt_token_ids) + list(
                                request.output_token_ids
                            )
                            self.prefix_cache.store_cache(
                                full_token_sequence,
                                request._extracted_cache,
                            )
                            logger.debug(
                                f"Stored cache for request {request_id} "
                                f"({len(full_token_sequence)} tokens: {len(request.prompt_token_ids)} prompt + {len(request.output_token_ids)} output)"
                            )
                        except Exception as e:
                            logger.debug(f"Failed to store cache for {request_id}: {e}")

            # Evaluate stored cache tensors incrementally (per-layer) to prevent
            # a deferred batch evaluation spike when all lazy ops resolve at once.
            # This spreads the VRAM cost across smaller per-layer evaluations.
            if (
                request is not None
                and hasattr(request, "_extracted_cache")
                and request._extracted_cache
            ):
                for layer in request._extracted_cache:
                    if isinstance(layer, dict) and "state" in layer:
                        keys, values = layer["state"]
                        mx.eval(keys, values)
                    elif hasattr(layer, "keys") and hasattr(layer, "values"):
                        keys_attr = layer.keys
                        values_attr = layer.values
                        if not callable(keys_attr) and not callable(values_attr):
                            mx.eval(keys_attr, values_attr)

            # Release all cache references on the request so Metal buffers
            # can be freed.  The prefix cache (if any) holds its own copy;
            # keeping a second reference here pins the buffers in wired memory
            # until the request object is GC'd (issue #442).
            if request is not None:
                request.prompt_cache = None
                request._extracted_cache = None

            # Remove from running
            if request_id in self.running:
                del self.running[request_id]

            # Remove UID mappings
            if request_id in self.request_id_to_uid:
                uid = self.request_id_to_uid[request_id]
                if uid in self.uid_to_request_id:
                    del self.uid_to_request_id[uid]
                # #558 PR-3: drop the finished uid's grammar processor state so
                # it can't linger and re-map onto a reused uid (the uid is
                # already out of the batch by the time cleanup runs).
                self._forget_uid_grammar(uid)
                del self.request_id_to_uid[request_id]

            # Track as finished
            self.finished_req_ids.add(request_id)

        # Free Metal command buffers after cleanup (prevents end-of-generation spike)
        if finished_ids:
            mx.clear_cache()

    def _is_cache_corruption_error(self, error: Exception) -> bool:
        """Check if an error indicates cache corruption."""
        error_str = str(error)
        return any(pattern in error_str for pattern in CACHE_CORRUPTION_PATTERNS)

    def _recover_from_cache_error(self) -> None:
        """Recover from cache corruption error."""
        # Properly close batch generator (this is the source of the corruption)
        self._close_batch_generator()
        self._current_sampler_params = None

        # Clear caches
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear()
        if self.memory_aware_cache is not None:
            self.memory_aware_cache.clear()
        if self.prefix_cache is not None:
            self.prefix_cache.clear()

        # Clear UID mappings
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()

        logger.info("Cache recovery completed")

    def _recover_from_generation_error(self) -> set[str]:
        """Recover from fatal generation error (OOM, Metal crash).

        Aborts all running requests and resets batch state.
        Unlike cache corruption recovery, does NOT reschedule —
        the request that OOMed would just OOM again.

        Returns:
            Set of aborted request IDs.
        """
        # Close batch generator (clears _partial state, active_batch)
        self._close_batch_generator()
        self._current_sampler_params = None

        # Abort all running requests
        aborted_ids: set[str] = set()
        for request_id in list(self.running):
            request = self.running.get(request_id)
            if request is not None:
                request.set_finished(RequestStatus.FINISHED_ABORTED)
            aborted_ids.add(request_id)
            self.finished_req_ids.add(request_id)
        self.running.clear()
        self._detokenizer_pool.clear()

        # Clear UID mappings (batch generator is gone)
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()

        # Release Metal memory
        mx.clear_cache()

        logger.warning(
            f"[generation_error_recovery] aborted {len(aborted_ids)} running requests, "
            f"batch generator closed, Metal cache cleared"
        )
        return aborted_ids

    def _reschedule_running_requests(self) -> None:
        """Move running requests back to waiting queue for retry."""
        count = len(self.running)
        for request_id, request in list(self.running.items()):
            # Reset request state
            request.status = RequestStatus.WAITING
            request.batch_uid = None
            request.prompt_cache = None
            request.cached_tokens = 0
            request.remaining_tokens = request.prompt_token_ids

            # Move to waiting queue (at front for priority)
            self.waiting.appendleft(request)
            del self.running[request_id]

        if count > 0:
            logger.info(f"Rescheduled {count} requests for retry")

    def step(self, max_retries: int = 1) -> SchedulerOutput:
        """
        Execute one scheduling step with automatic error recovery.

        This method:
        1. Schedules waiting requests into the batch
        2. Runs one generation step via BatchGenerator
        3. Processes outputs and handles finished requests
        4. Automatically recovers from cache corruption errors

        Args:
            max_retries: Number of times to retry on cache errors (default 1)

        Returns:
            SchedulerOutput with results of this step
        """
        output = SchedulerOutput()

        # Process pending aborts FIRST (in executor thread, safe for MLX)
        self._process_pending_aborts()
        # #1759: the engine has no consumer for a running request once its
        # canonical tracking entry is gone.  Reconcile that invariant before
        # invoking BatchGenerator.next(), otherwise a lost disconnect-abort can
        # spend this and every later tick decoding a ghost slot.
        self._reconcile_orphaned_running_requests()

        for attempt in range(max_retries + 1):
            try:
                # Schedule waiting requests
                scheduled = self._schedule_waiting()
                output.scheduled_request_ids = [r.request_id for r in scheduled]
                # Use model_prompt_tokens — when PFlash engages the
                # prefill workload is the compressed length, not the
                # logical (client-visible) prompt length.
                output.num_scheduled_tokens = sum(
                    r.model_prompt_tokens or r.num_prompt_tokens for r in scheduled
                )

                # Run generation step if we have running requests
                if self.batch_generator is not None and self.running:
                    # #558 PR-3: repair any grammar processor↔uid positional
                    # desync before decoding this tick. Armed ONLY while a
                    # grammar is actually in flight, or a finished grammar is
                    # still tombstoned (so its leaked slot is swept even when it
                    # was the last one). Penalty-only requests do NOT arm this —
                    # their entries in ``uid_to_request_processors`` are passive
                    # reconstruction state used when a grammar realign fires;
                    # arming on them would run an O(batch) rebuild every token on
                    # the plain decode hot path (codex #558-PR3).
                    if self._realign_guard_armed():
                        self._realign_grammar_logits_processors()
                    # mlx-lm consumes at most one prompt chunk in this call.
                    # Tighten that chunk before dispatch when a long cold or
                    # cache-miss prefill is approaching the unified-memory cap.
                    self._apply_adaptive_prefill_size()
                    if self._step_timing_enabled:
                        st = getattr(self, "_steptime", None)
                        if st is None:
                            # [next_samples, outside_samples, count, end_stamp]
                            st = self._steptime = [[], [], 0, 0.0]
                        _t0 = time.perf_counter()
                        if st[3]:
                            st[1].append(_t0 - st[3])
                            st[3] = 0.0
                        raw_next = self.batch_generator.next()
                        st[0].append(time.perf_counter() - _t0)
                        st[2] += 1
                        if st[2] % 256 == 0:
                            _nx = sorted(st[0])
                            _ou = sorted(st[1]) or [0.0]
                            logger.warning(
                                "[STEPTIME] n=%d next mean=%.2f p50=%.2f "
                                "p90=%.2f max=%.1f | outside mean=%.2f "
                                "p50=%.2f p90=%.2f max=%.1f (ms, window)",
                                st[2],
                                sum(_nx) / len(_nx) * 1e3,
                                _nx[len(_nx) // 2] * 1e3,
                                _nx[int(len(_nx) * 0.9)] * 1e3,
                                _nx[-1] * 1e3,
                                sum(_ou) / len(_ou) * 1e3,
                                _ou[len(_ou) // 2] * 1e3,
                                _ou[int(len(_ou) * 0.9)] * 1e3,
                                _ou[-1] * 1e3,
                            )
                            st[0], st[1] = [], []
                    else:
                        raw_next = self.batch_generator.next()
                    # Bound functional recurrent-state graphs without forcing
                    # a host synchronization on every token. The barrier fires
                    # off the live chain DEPTH (steps since the last barrier),
                    # evaluated against an interval that widens at low batch so
                    # a single stream stops paying the host sync ~8× more often
                    # than a batch of eight. Keying off depth (not the global
                    # step counter) makes a batch-size change re-check the
                    # accumulated chain immediately: a deep B=1 chain that
                    # meets the tighter high-concurrency interval materializes
                    # on the very next step instead of drifting to the next
                    # global-step multiple (codex #1895).
                    _active_seqs = len(self.running) if self.running else 1
                    _raw_running = len(self.running) if self.running else 0
                    if self._recurrent_prev_running == 0 and _raw_running > 0:
                        # Idle -> active edge: a sequence just entered an empty
                        # batch and may carry an unmaterialized prefill graph.
                        # Seed the depth to the interval so THIS step arms the
                        # barrier (fresh scheduler or long-idle one alike).
                        self._recurrent_chain_depth = (
                            self._recurrent_materialize_interval(_active_seqs)
                        )
                    self._recurrent_prev_running = _raw_running
                    self._recurrent_chain_depth += 1
                    if (
                        self._recurrent_chain_depth
                        >= self._recurrent_materialize_interval(_active_seqs)
                    ):
                        self._materialize_active_recurrent_cache()
                        self._recurrent_chain_depth = 0
                    output.has_work = True

                    # mlx-lm 0.31+ returns (prompt_responses, generation_responses) tuple
                    # older versions return a flat list of responses
                    if isinstance(raw_next, tuple):
                        prompt_responses, responses = raw_next
                        self._snapshot_promoted_prompts(prompt_responses)
                        # issue #427: per-message boundary snapshot for
                        # multi-turn hybrid workloads (segment finished
                        # but prompt still has tail to process).
                        self._snapshot_boundary_segments(prompt_responses)
                    else:
                        responses = raw_next

                    if responses:
                        outputs, finished_ids = self._process_batch_responses(responses)
                        output.outputs = outputs
                        output.finished_request_ids = finished_ids
                        self._cleanup_finished(finished_ids)

                # #558 PR-3: reconcile grammar processors when idle. The realign
                # guard (which scrubs leaked slots AND flushes tombstones) only
                # runs while requests are running, so if the LAST grammar request
                # finished this tick and the batch is now empty, (a) a stale
                # positional ``logits_processors`` list could linger and be
                # inherited by the next admitted plain request, and (b) the
                # tombstoned processor would stay strongly referenced through the
                # idle period. Route through the realign method (not a bare
                # flush): with zero uids it takes the no-live-slots branch, which
                # scrubs ``bg.logits_processors`` to ``[]`` before forgetting the
                # tombstones — closing the leaked-slot inheritance gap (codex
                # #558-PR3 blocking).
                if self._stateful_tombstones and not self.running:
                    self._realign_grammar_logits_processors()

                # Success - break out of retry loop
                break

            except TypeError as e:
                # Catch the NoneType error specifically
                if self._is_cache_corruption_error(e):
                    if attempt < max_retries:
                        logger.warning(
                            f"Cache corruption detected (attempt {attempt + 1}), "
                            f"performing recovery and retry..."
                        )
                        # Deep reset to recover
                        self._recover_from_cache_error()
                        # Re-add any running requests back to waiting
                        self._reschedule_running_requests()
                    else:
                        logger.error(
                            f"Cache corruption not recoverable after "
                            f"{max_retries + 1} attempts"
                        )
                        raise
                else:
                    raise
            except Exception as e:
                import traceback

                abort_error = (
                    f"Inference aborted after generation error: {type(e).__name__}: {e}"
                )
                logger.error(
                    f"Error in batch generation step: {e}\n{traceback.format_exc()}"
                )
                # Recover from fatal errors (OOM, Metal crash) instead of
                # re-raising, which would cause infinite loop in engine_core.
                aborted_ids = self._recover_from_generation_error()
                for rid in aborted_ids:
                    output.outputs.append(
                        RequestOutput(
                            request_id=rid,
                            finished=True,
                            # This is an internal terminal output.  The engine
                            # raises on ``error`` before route serializers see
                            # it, so keep its state truthful rather than
                            # masquerading as a successful token-budget cap.
                            finish_reason="abort",
                            error=abort_error,
                        )
                    )
                output.finished_request_ids = aborted_ids
                break

        if self._step_timing_enabled and hasattr(self, "_steptime"):
            self._steptime[3] = time.perf_counter()

        # Clear finished tracking for next step
        self.finished_req_ids = set()

        # Adaptive interval: scale inversely with concurrency to prevent
        # Metal resource handle exhaustion under high-concurrency workloads.
        active_seqs = len(self.running)
        min_interval = max(4, self._clear_cache_interval // 4)
        effective_interval = max(
            min_interval, self._clear_cache_interval // max(1, active_seqs // 8)
        )

        self._step_count += 1
        if self._step_count % effective_interval == 0:
            # Evaluate batch tokens to collapse lazy concatenation chains
            # mlx-lm 0.31+ renamed active_batch to _generation_batch
            _active = None
            if self.batch_generator is not None:
                _active = getattr(
                    self.batch_generator, "active_batch", None
                ) or getattr(self.batch_generator, "_generation_batch", None)
            if _active is not None and hasattr(_active, "tokens"):
                tokens = _active.tokens
                if tokens:
                    mx.eval(*tokens)
            mx.clear_cache()

        # Periodically log memory stats for monitoring
        if self._step_count % self._memory_log_interval == 0:
            try:
                if mx.metal.is_available():
                    active_gb = mx.get_active_memory() / 1e9
                    peak_gb = mx.get_peak_memory() / 1e9
                    cache_gb = mx.get_cache_memory() / 1e9
                    logger.info(
                        f"[Metal memory] active={active_gb:.1f}GB "
                        f"peak={peak_gb:.1f}GB cache={cache_gb:.1f}GB "
                        f"step={self._step_count} "
                        f"running={len(self.running)} waiting={len(self.waiting)}"
                    )
            except Exception:
                pass

        return output

    def _recurrent_materialize_interval(self, active_seqs: int) -> int:
        """Steps between recurrent-state barriers for the current batch depth.

        The barrier's cost is a host ``mx.eval`` sync that cannot overlap
        the decode pipeline; at B=1 it is a visible ~3% decode-rate tax.
        Its *purpose* is to cap the lazy-graph handle count, which grows as
        ``active_seqs × steps_since_barrier``. Holding that product at the
        same 64-unit budget the flat every-8 barrier already sustains at
        batch 8 lets a single stream materialize every 64 steps instead of
        every 8 — 8× fewer stalls, identical peak handles. Concurrency keeps
        the #1834 every-8 floor.
        """
        if active_seqs < 1:
            active_seqs = 1
        interval = _RECURRENT_MATERIALIZE_HANDLE_BUDGET // active_seqs
        if interval < _RECURRENT_CACHE_MATERIALIZE_INTERVAL:
            return _RECURRENT_CACHE_MATERIALIZE_INTERVAL
        if interval > _RECURRENT_MATERIALIZE_MAX_INTERVAL:
            return _RECURRENT_MATERIALIZE_MAX_INTERVAL
        return interval

    def _materialize_active_recurrent_cache(self) -> int:
        """Detach lazy recurrent-cache updates from prior decode steps.

        MLX cache implementations such as ``ArraysCache`` update recurrent
        state functionally.  Without an evaluation barrier, the live batch
        keeps the head of a graph that references every earlier decode step.
        The byte footprint can stay flat while Metal's live buffer-handle
        count grows until its 499000-resource ceiling is reached (#1827).

        The caller runs this barrier on the first decode step and every eight
        steps thereafter. That keeps graph depth O(1) while avoiding a costly
        host synchronization on every token. Dense KV caches use in-place
        writes and do not need it. Evaluate only non-trimmable/unknown states:
        materializing dense KV layers too would add an unnecessary per-token
        synchronization cost to hybrid models.
        """
        batch_generator = self.batch_generator
        if batch_generator is None:
            return 0
        generation_batch = getattr(batch_generator, "_generation_batch", None)
        if generation_batch is None:
            generation_batch = getattr(batch_generator, "active_batch", None)
        cache = getattr(generation_batch, "prompt_cache", None)
        if not cache:
            return 0

        states = []
        for layer in cache:
            is_trimmable = getattr(layer, "is_trimmable", None)
            if callable(is_trimmable):
                try:
                    materialize = not bool(is_trimmable())
                except Exception:
                    # Classification is a safety gate: treating an unknown
                    # cache as dense would silently disable the only barrier
                    # preventing an unbounded lazy-state graph.  An extra eval
                    # is safe; skipping one for a recurrent cache is not.
                    materialize = True
            else:
                # Supported modern dense caches affirmatively implement this
                # method. Missing/non-callable classification is unknown, so
                # retain the same safety-first behavior as a raised classifier.
                materialize = True
            if materialize:
                state = getattr(layer, "state", None)
                if state is not None:
                    states.append(state)
        if not states:
            return 0
        mx.eval(states)
        return len(states)

    def get_request(self, request_id: str) -> Request | None:
        """Get a request by ID."""
        return self.requests.get(request_id)

    def remove_finished_request(self, request_id: str) -> Request | None:
        """Remove a finished request from tracking.

        D-M01-2X + D-M01-DEAD (0.8.2 dogfood): this method MUST
        NOT discard ``_cancelled_request_ids`` /
        ``_disconnect_abort_ids``. Those are LIFETIME ledgers — the
        ``__init__`` comment block explicitly documents them as
        "every id that has ever advanced the counter stays in it
        for the process lifetime" with memory bounded by cancel
        traffic.

        Why the prior discard was a regression
        --------------------------------------
        On the production ``BatchedEngine`` over ``AsyncEngineCore``
        shape, an aborted request follows this sequence:

          1. ``stream_outputs.finally`` (or the deferred
             ``_await_and_record`` coroutine) calls
             ``scheduler.abort_request(rid)`` → adds to
             ``_cancelled_request_ids``, increments
             ``num_requests_cancelled``, queues into
             ``_pending_abort_ids``. Returns True.
          2. ``EngineCore._cleanup_request`` calls THIS method →
             previously discarded both ledgers. ``_pending_abort_ids``
             still contains the id (it's drained on the executor
             thread by ``_process_pending_aborts``).
          3. The other branch's ``scheduler.abort_request(rid)``
             (the disconnect_guard fires from up to three places,
             the async-fallback coroutine adds a fourth)
             re-enters the public abort. The membership predicate
             ``request_id in self._pending_abort_ids`` still
             evaluates True, so the abort is accepted. The
             ``already_counted`` read on the WIPED
             ``_cancelled_request_ids`` returns False, and the
             counter increments AGAIN — the 2x over-count.
          4. ``record_disconnect_abort`` then runs through the gate
             ``request_id not in self._cancelled_request_ids`` —
             which AGAIN reads from the wiped ledger and silently
             returns. ``via_disconnect_total`` stays flat-zero
             through every real disconnect — D-M01-DEAD.

        The fix: leave the ledgers populated for the process
        lifetime. Membership in ``_cancelled_request_ids`` is now
        a true "this id has already advanced the counter once"
        marker that survives ``_cleanup_request`` AND any number
        of redundant abort calls from the disconnect_guard's
        multi-branch fire pattern. The only paths that clear them
        are ``reset()`` / ``deep_reset()`` — see the codex r8
        BLOCKING #1 comment block in ``reset()`` for why those
        clear AFTER the abort loop.

        Memory: one ~36-byte uuid per cancel, same scale as
        ``finished_req_ids`` (which also persists across
        ``_cleanup_request``). The PR #783 docstring claim that
        "the only way a future ``abort_request(rid)`` can hit
        True is through a fresh ``add_request``" was wrong: the
        ``_pending_abort_ids`` membership branch passes True for
        the entire window between abort enqueue and
        executor-thread drain, opening exactly this race.

        Returns the popped Request (or None if already gone).
        """
        # Pop under the lock so a concurrent ``abort_request`` either
        # observes the id present in ``self.requests`` (admits, hits
        # the lifetime ledger, dedupes — no double count) or absent
        # AND with all admission predicates ruling it out (returns
        # False per F-151). The ledgers stay populated indefinitely.
        with self._cancel_counter_lock:
            popped = self.requests.pop(request_id, None)
            if popped is not None and self.running.get(request_id) is popped:
                self._orphaned_running_candidates[request_id] = popped
        return popped

    def get_running_requests_info(self) -> list[dict[str, Any]]:
        """Per-request details for status endpoint."""
        import time as _time

        now = _time.time()
        result = []

        # Waiting requests
        for req in self.waiting:
            result.append(
                {
                    "request_id": req.request_id,
                    "status": "waiting",
                    "phase": "queued",
                    "elapsed_s": round(now - req.arrival_time, 2),
                    "prompt_tokens": req.num_prompt_tokens,
                    "completion_tokens": 0,
                    "max_tokens": req.max_tokens,
                    "progress": 0.0,
                    "tokens_per_second": None,
                    "ttft_s": None,
                    "cache_hit_type": req.cache_hit_type,
                    "cached_tokens": req.cached_tokens,
                }
            )

        # Running requests
        for req in self.running.values():
            n_out = req.num_output_tokens
            elapsed = now - req.arrival_time

            # Phase detection
            if n_out == 0:
                phase = "prefill"
            else:
                phase = "generation"

            # Tokens per second (generation phase only)
            tok_s = None
            ttft = None
            if req.first_token_time is not None:
                ttft = round(req.first_token_time - req.arrival_time, 3)
                gen_elapsed = now - req.first_token_time
                if gen_elapsed > 0 and n_out > 0:
                    tok_s = round(n_out / gen_elapsed, 1)

            # Progress: completion_tokens / max_tokens
            progress = round(n_out / req.max_tokens, 3) if req.max_tokens > 0 else 0.0

            result.append(
                {
                    "request_id": req.request_id,
                    "status": "running",
                    "phase": phase,
                    "elapsed_s": round(elapsed, 2),
                    "prompt_tokens": req.num_prompt_tokens,
                    "completion_tokens": n_out,
                    "max_tokens": req.max_tokens,
                    "progress": min(progress, 1.0),
                    "tokens_per_second": tok_s,
                    "ttft_s": ttft,
                    "cache_hit_type": req.cache_hit_type,
                    "cached_tokens": req.cached_tokens,
                }
            )

        return result

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        now = time.time()
        active_generation_rates = []
        for request in self.running.values():
            if request.first_token_time is not None and request.num_output_tokens > 0:
                elapsed = now - request.first_token_time
                if elapsed > 0:
                    active_generation_rates.append(request.num_output_tokens / elapsed)
        if active_generation_rates:
            self._last_generation_tps = sum(active_generation_rates)
        stats = {
            "num_waiting": len(self.waiting),
            "num_running": len(self.running),
            "num_requests_processed": self.num_requests_processed,
            "num_repetition_loop_stops": self.num_repetition_loop_stops,
            "num_repetition_loop_breaks": self.num_repetition_loop_breaks,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "batch_generator": {
                "prompt_tps": round(self._last_prompt_tps, 2),
                "generation_tps": round(self._last_generation_tps, 2),
            },
            # M-02: PFlash observability counters. ``bypass_count`` is
            # the number of requests where PFlash compression engaged
            # and the prefix-cache fetch/store was skipped;
            # ``compressed_tokens_dropped`` is the cumulative number of
            # prompt tokens removed by the compressor (logical minus
            # kept). Both default to zero on engines without PFlash so
            # /metrics renders a flat-line 0 instead of an absent
            # series.
            "pflash_bypass_count": self.pflash_bypass_count,
            "pflash_compressed_tokens_dropped": self.pflash_compressed_tokens_dropped,
            # M-01: cancellation observability. ``num_requests_cancelled``
            # is the total count of public-API aborts the scheduler
            # accepted (one increment per unique request_id transitioning
            # into ``_pending_abort_ids``). ``num_requests_cancelled_via_
            # disconnect`` is the subset attributed to client disconnect
            # via ``_force_abort_request``. Both default to zero on
            # engines that never see traffic so /metrics stays at a
            # flat-line series rather than an absent one. See the init
            # comment for the rationale on why ``num_requests_processed``
            # alone is insufficient.
            "num_requests_cancelled": self.num_requests_cancelled,
            "num_requests_cancelled_via_disconnect": (
                self.num_requests_cancelled_via_disconnect
            ),
            # D-METAL-CAP / D-METAL-PFX observability — pre-fix, both
            # were silent: the cap was violated with no warning and the
            # prefix cache pinned slabs through one 32k prefill that
            # then cratered decode-tps for the rest of the session.
            "num_metal_cap_violations": self.num_metal_cap_violations,
            "num_prefix_cache_pressure_evictions": (
                self.num_prefix_cache_pressure_evictions
            ),
            "adaptive_prefill_chunk_size": getattr(
                self, "_last_adaptive_prefill_size", self.config.prefill_step_size
            ),
            "adaptive_prefill_protected_chunks": getattr(
                self, "_adaptive_prefill_protected_chunks", 0
            ),
            "adaptive_prefill_reduced_chunks": getattr(
                self, "_adaptive_prefill_reduced_chunks", 0
            ),
        }
        # R15-P1 (task #296): disk-backed KV checkpoint counters.
        # Folded straight from the module-level ``disk_kv_checkpoint``
        # stats so /metrics can render writes / loads / bytes / evictions
        # without the scheduler having to track them per-instance.
        # Guarded by an import-try so a fresh test harness that doesn't
        # exercise the runtime module still gets a sane scheduler stats
        # dict (gracefully degrades to an empty sub-dict, matching every
        # other optional cache feature here).
        try:
            from .runtime import disk_kv_checkpoint as _dkc

            stats["kv_checkpoint"] = _dkc.get_stats()
        except Exception:  # pragma: no cover — defensive
            pass
        # Include Metal memory stats
        try:
            if mx.metal.is_available():
                stats["metal_active_memory_gb"] = round(mx.get_active_memory() / 1e9, 2)
                stats["metal_peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
                stats["metal_cache_memory_gb"] = round(mx.get_cache_memory() / 1e9, 2)
        except Exception:
            pass

        # Include cache stats
        if self.block_aware_cache is not None:
            stats["paged_cache"] = self.block_aware_cache.get_stats()
        elif self.memory_aware_cache is not None:
            stats["memory_aware_cache"] = self.memory_aware_cache.get_stats()
        elif self.prefix_cache is not None:
            stats["prefix_cache"] = self.prefix_cache.get_stats()
        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics."""
        if self.block_aware_cache is not None:
            return self.block_aware_cache.get_stats()
        elif self.memory_aware_cache is not None:
            return self.memory_aware_cache.get_stats()
        elif self.prefix_cache is not None:
            return self.prefix_cache.get_stats()
        return None

    def clear_prefix_cache(self, *, reset_stats: bool = True) -> bool:
        """Clear scheduler-owned reusable KV state without touching requests."""
        if self.has_requests():
            raise RuntimeError("cannot clear prefix cache while requests are active")
        cleared = False
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear(reset_stats=reset_stats)
            cleared = True
        elif self.memory_aware_cache is not None:
            self.memory_aware_cache.clear(reset_stats=reset_stats)
            cleared = True
        elif self.prefix_cache is not None:
            self.prefix_cache.clear(reset_stats=reset_stats)
            cleared = True
        return cleared

    def reset(self) -> None:
        """Reset the scheduler state.

        M-01 codex r8 BLOCKING #1: the cancellation dedupe ledgers
        (``_cancelled_request_ids`` / ``_disconnect_abort_ids``)
        MUST be cleared AFTER the abort loop, not before. Clearing
        before means a concurrent ``record_disconnect_abort`` for a
        still-live in-flight request could either no-op (id removed
        from ledger ahead of its lifetime ending) or, worse, re-add
        the id after ``_do_abort_request`` runs (because that path
        also does a discard, and the ``add`` came from
        ``abort_request`` racing against the loop). Clearing AFTER
        the abort loop AND under the lock means any concurrent
        ``record_disconnect_abort`` either ran fully BEFORE reset
        started (correct) or sees the cleared state and no-ops
        (correct: the request is gone, attribution is meaningless).
        """
        # Drain both cross-thread cleanup edges under their shared lock.
        # Snapshot the union before teardown: a disconnect orphan has already
        # left ``requests`` and would otherwise survive a reset that iterated
        # only the canonical map.
        with self._cancel_counter_lock:
            self._pending_abort_ids.clear()
            self._orphaned_running_candidates.clear()
            teardown_ids = list(dict.fromkeys((*self.requests, *self.running)))

        # Abort all requests directly (reset is synchronous)
        for request_id in teardown_ids:
            self._do_abort_request(request_id)

        self.waiting.clear()
        self.running.clear()
        self.requests.clear()
        self.finished_req_ids.clear()
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()
        self._detokenizer_pool.clear()
        self._close_batch_generator()
        self._current_sampler_params = None

        # M-01: drop the cancellation lifetime ledgers AFTER the
        # tear-down loop completes. The counters themselves
        # (``num_requests_cancelled`` /
        # ``num_requests_cancelled_via_disconnect``) are NOT zeroed —
        # they're lifetime-cumulative Prometheus counters and
        # resetting them would make /metrics report a non-monotonic
        # step change to scrapers. The sticky-counter accumulator in
        # routes/metrics.py would then fold the apparent reset into
        # a baseline, which is the right behaviour for the cache
        # series but here we'd rather never trip it. Wiping the
        # dedupe ledgers AFTER the abort loop is safe because the
        # request_ids they tracked have all been torn down by then,
        # and is correct against the codex r8 BLOCKING #1 race
        # (clearing before reset's _do_abort_request loop ran would
        # have re-opened the dedupe window during the tear-down).
        with self._cancel_counter_lock:
            self._cancelled_request_ids.clear()
            self._disconnect_abort_ids.clear()

        # Clear caches
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear()
        if self.memory_aware_cache is not None:
            self.memory_aware_cache.clear()
        if self.prefix_cache is not None:
            self.prefix_cache.clear()

    def deep_reset(self) -> None:
        """
        Deep reset that clears ALL cache state including model-level caches.

        This is more aggressive than reset() and should be used when
        switching engines or recovering from errors.
        """
        # Standard reset first
        self.reset()

        # Clear any model-level cache state
        # MLX models may have internal cache references
        if hasattr(self.model, "cache"):
            self.model.cache = None

        # Some MLX models store cache in layers
        if hasattr(self.model, "layers"):
            for layer in self.model.layers:
                if hasattr(layer, "cache"):
                    layer.cache = None
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "cache"):
                    layer.self_attn.cache = None

        # Force garbage collection of any lingering cache objects
        import gc

        gc.collect()

        logger.info("Deep reset completed - all caches cleared")

    # -----------------------------------------------------------------
    # Cache persistence
    # -----------------------------------------------------------------

    def save_cache_to_disk(self, cache_dir: str, should_abort=None) -> bool:
        """Save prefix cache to disk for persistence across restarts.

        ``should_abort`` is an optional ``Callable[[float], bool]``
        (the ``float`` is the next entry's predicted write duration)
        that signals the lifespan SIGTERM-grace deadline to the per-
        entry loop inside ``MemoryAwarePrefixCache.save_to_disk``.
        Zero-arg callables are accepted via auto-detection for
        backwards compatibility. See that method's docstring for the
        partial-commit guarantee.
        """
        if self.memory_aware_cache is not None:
            return self.memory_aware_cache.save_to_disk(
                cache_dir, should_abort=should_abort
            )
        logger.info("[cache_persist] no memory-aware cache to save")
        return False

    def load_cache_from_disk(
        self, cache_dir: str, replace: bool = False, protected_import: bool = True
    ) -> int:
        """Load prefix cache from disk. Returns number of entries loaded.

        ``replace=True`` (export/import "replace" strategy, #476) clears
        the in-memory cache atomically inside the load, after the on-disk
        index is validated — see ``MemoryAwarePrefixCache.load_from_disk``.

        ``protected_import`` (#1111 codex r3): True for the explicit HTTP import
        (pin loaded entries), False for the startup auto-load (loaded entries
        obey the hybrid retention bound) — see ``load_from_disk``.
        """
        if self.memory_aware_cache is not None:
            return self.memory_aware_cache.load_from_disk(
                cache_dir, replace=replace, protected_import=protected_import
            )
        logger.info("[cache_persist] no memory-aware cache to load into")
        return 0
