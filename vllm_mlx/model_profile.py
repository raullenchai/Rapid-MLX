# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for per-model profile data.

Unifies the former ``AliasProfile`` (alias-keyed, resolved from
``aliases.json``) and ``ModelConfig`` (regex-detected, from
``model_auto_config``) into ONE frozen dataclass. Before this module the
two dataclasses mirrored a shared field set by hand, and
``detect_model_config`` copied a *subset* of ``AliasProfile``'s fields
into a fresh ``ModelConfig`` — so any field that existed on the alias
profile but was forgotten in that copy (``is_moe``, ``dflash_draft_model``,
``recommended_sampling``, ``modality``, ``ddtree_*``, ``min_memory_gb``)
silently never reached ``rapid-mlx info`` / serve defaults. Adding a field
meant editing two dataclasses AND the copy site; missing any one dropped
the field. Now there is exactly one class + one field set: add a field
here and every consumer sees it, with no copy step to drift out of sync.

``AliasProfile`` (in ``model_aliases``) and ``ModelConfig`` (in
``model_auto_config``) remain as module-level aliases of this class for
backward compatibility with the ~20 call sites that import them by name —
they are the *same* class, not two shapes.

Import-light on purpose: only stdlib + typing, so lightweight callers
(the CLI alias map, dflash / ddtree eligibility checks) can import the
profile shape without pulling in ``model_auto_config``'s regex detection
tables. Locating the class here (rather than in ``model_auto_config``)
also avoids an import cycle: ``model_auto_config`` already imports
``model_aliases.resolve_profile``, so the profile dataclass cannot live in
``model_auto_config`` without ``model_aliases`` importing it back.
"""

from dataclasses import dataclass
from typing import Literal

# NOTE: deliberately NO ``from __future__ import annotations`` here. The
# out-of-band-routing guard (tests/test_no_out_of_band_routing.py) inspects
# ``dataclasses.fields(...).type`` to flag any ``str`` field as a possible
# covert routing switch. Under stringized annotations that check sees the
# raw source text, so tuple fields whose element type mentions ``str``
# (``recommended_sampling``, ``suffix_bench_speedup``) get misread as str
# fields. Keeping real type objects matches the pre-unification AliasProfile
# behaviour. All annotations below are valid at runtime on Python 3.10+
# (PEP 604 ``X | None`` + PEP 585 ``tuple[...]``).

# Canonical modality enum. Default is ``"text"`` so every legacy alias
# (and every external JSON snippet that pre-dates this field) keeps the
# auto-regressive LLM lane untouched. New modalities branch the runtime
# at startup — see ``runtime/diffusion_lane.py`` for the discrete
# text-diffusion path used by DiffusionGemma. ``"vision"`` and
# ``"image-gen"`` are reserved for forthcoming VLM / image-gen
# integrations. (A vision-config checkpoint we serve text-only — e.g.
# Ternary-Bonsai-27B — stays ``modality="text"`` and sets
# ``is_text_only=True`` instead; it is not a ``"vision"`` alias because
# we do not serve its vision tower.)
# Adding a new value requires editing this Literal AND the dispatch table
# in cli.py / routes/models.py so the surface-level UX (info, ls, chat)
# doesn't silently expose LLM-only columns on a non-LLM alias.
Modality = Literal["text", "text-diffusion", "vision", "image-gen"]


@dataclass(frozen=True, kw_only=True)
class ModelProfile:
    """Per-model profile — parser defaults + capability gates.

    The single dataclass behind both ``AliasProfile`` (alias-keyed) and
    ``ModelConfig`` (regex-detected). Frozen so a resolved profile can be
    shared safely across threads and cached without a defensive copy.

    ``hf_path`` defaults to ``""`` so a regex-detected profile (no owning
    alias) can be constructed without one; alias construction always
    passes a validated non-empty path via ``model_aliases._coerce``.

    Construction is KEYWORD-ONLY (``kw_only=True``). This is deliberate
    and load-bearing for the unification: the pre-unification ``ModelConfig``
    began with ``tool_call_parser`` and carried no ``hf_path`` field at
    all, whereas the unified profile begins with ``hf_path``. Under
    positional construction a legacy call like ``ModelConfig("hermes")``
    would silently bind ``hf_path="hermes"`` instead of the intended
    ``tool_call_parser`` (pr_validate codex flagged exactly this on PR
    #1108). ``kw_only=True`` makes any positional construction a loud
    ``TypeError`` instead of a silent field-misbind, so the field-order
    difference between the two former dataclasses can never surface as a
    routing bug. Every construction site in-tree already passes keywords
    (``model_aliases._coerce``, the ``model_auto_config`` regex table,
    and all tests), so this is a no-op for existing callers.

    Migration paths for the two shape changes the unification introduced
    (both flagged by codex as needing an explicit deprecation path):
      * ``ModelConfig`` was mutable; a resolved profile is now frozen.
        Produce a modified copy with ``dataclasses.replace(profile,
        field=value)`` instead of ``profile.field = value``
        (``model_auto_config.enrich_model_config`` and
        ``engine_core`` already do this).
      * ``suffix_bench_speedup`` was a ``dict``; it is now a tuple of
        ``(workload, speedup)`` pairs (frozen dataclasses need immutable,
        hashable fields). Read it as a mapping via the ``speedup_dict``
        property, never ``profile.suffix_bench_speedup.get(...)``.

    Defaults err on the side of "supported" — known-incompatible families
    set the flag explicitly.
    """

    hf_path: str = ""

    # --- Parser defaults ---
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None

    # --- Architecture / capability gates ---
    # ``is_hybrid`` = the model uses linear-attention or recurrent layers
    # (GatedDeltaNet, Mamba, Jamba, ...). Hybrid models need request
    # throttling and disable optimizations that rely on chunked-batched
    # forward — verified on Qwen3.5-4B where spec decode produces
    # corrupted output (see evals/results/SUFFIX_POC_REPORT.md).
    is_hybrid: bool = False
    # r6-A R6-C1: when an aliases.json entry explicitly declares
    # ``is_hybrid``, the runtime ArraysCache probe in
    # ``model_auto_config.enrich_model_config`` MUST NOT override the
    # declared value. Without this flag, the probe one-way-promotes
    # ``is_hybrid`` to ``True`` for any model whose ``make_cache()``
    # returns linear-attention layers — which is exactly what dense
    # Qwen3.5 / Qwen3.6 weights do (model_type=qwen3_5 uses
    # GatedDeltaNet), forcing the alias-level routing decision (hybrid
    # throttle + prefix-boundary snapshot) back on even after the JSON
    # marked the model as non-hybrid. That re-promotion is what
    # wedges ``rapid-mlx serve qwen3.5-4b-4bit`` on metal::malloc with a
    # 499000 byte limit; ``--no-hybrid`` was the only workaround.
    #
    # Default ``False`` keeps the existing safety-net behaviour for
    # legacy aliases that haven't opted into the explicit contract —
    # those still pick up the probe's hybrid promotion as before.
    is_hybrid_explicit: bool = False
    # MoE / sparse-expert architecture (A3B, A10B, A17B Qwen3.5/3.6 variants,
    # plus future Mixtral/Granite-MoE families). Tracked separately from
    # ``is_hybrid`` because the two attributes gate different downstream
    # paths — hybrid affects ArraysCache/GDN rollback, MoE affects DFlash
    # acceptance rate (the drafter's hidden-state fusion misfires on
    # expert-routing churn; PoC measured 0.76-0.82× regression regardless
    # of precision on Qwen3.6-35B-A3B).
    is_moe: bool = False
    # ``supports_spec_decode`` controls SuffixDecoding / draft-model
    # speculative decoding. Disabled for hybrid models because the
    # batched-verify path through GatedDeltaNet derails generation.
    # Pure-attention models (llama, qwen3, mistral, gemma3, gpt-oss,
    # phi, ...) are safe.
    supports_spec_decode: bool = True
    default_max_tokens: int | None = None  # Per-model default when user omits

    # SuffixDecoding eligibility — populated from cross-model bench (issue #269).
    # ``None`` for ``suffix_bench_speedup`` means "not benched yet"; the tier
    # then defaults to ``"unknown"`` and the startup hint stays silent.
    # When populated, ``suffix_bench_speedup`` is a tuple of ``(workload, speedup)``
    # pairs (tuple, not dict, so frozen dataclass instances stay safely shareable).
    # Consumers that index by workload key use the ``speedup_dict`` property.
    suffix_decoding_tier: str = "unknown"
    suffix_bench_speedup: tuple[tuple[str, float], ...] | None = None
    # DFlash speculative-decoding eligibility (issue #264). Explicit opt-in
    # per alias rather than auto-derived because the PoC showed
    # precision-dependent regressions: 4-bit kills acceptance even on dense
    # models. Aliases keep ``supports_dflash=False`` until benched to win
    # by ≥1.3× on the canonical Fibonacci/Quicksort/HashTable prompts.
    # ``dflash_draft_model`` is the matching drafter HF path (e.g.
    # ``z-lab/Qwen3.5-27B-DFlash``); required if ``supports_dflash=True``.
    supports_dflash: bool = False
    dflash_draft_model: str | None = None
    # Recommended sampling defaults — curated per-family overrides that
    # sit above HF ``generation_config.json`` in the resolve chain (see
    # ``service/helpers.py``). Tuple-of-pairs (not dict) because the
    # dataclass is frozen and dict defaults are mutable. Keys are
    # restricted to the sampling subset: ``temperature``, ``top_p``,
    # ``top_k``, ``min_p``, ``repetition_penalty``, ``presence_penalty``,
    # ``frequency_penalty``. ``None`` means "no curated value" → fall
    # through to ``generation_config.json``.
    recommended_sampling: tuple[tuple[str, float], ...] | None = None
    # Inference modality. Default ``"text"`` covers every legacy LLM
    # alias and keeps the auto-regressive scheduler/runtime path
    # unchanged. Non-text modalities branch into dedicated lanes:
    # ``"text-diffusion"`` → ``runtime/diffusion_lane.py`` (block
    # denoising, no spec-decode, no DFlash); ``"vision"`` /
    # ``"image-gen"`` reserved for upcoming integrations.
    modality: Modality = "text"
    # PFlash long-prompt compression eligibility (#287). Default
    # ``"unknown"`` keeps the engine's PFlash mode at ``"off"`` so a
    # brand-new alias never silently enables compression on an
    # unbenched architecture. ``"verified"`` flips the engine's default
    # to ``"always"`` — used only for aliases where we've measured both
    # the TTFT speedup AND the needle-recall floor (Qwen3.5 / Qwen3.6
    # families per #287). Explicit CLI ``--pflash {off,auto,always}``
    # still wins; this only changes the no-flag default.
    pflash_tier: str = "unknown"
    # TurboQuant K8V4 default-on tier. See ``VALID_TURBOQUANT_TIERS``.
    turboquant_tier: str = "unknown"
    # DDTree speculative-decoding eligibility (#879). Intentionally
    # separate from DFlash even though it uses the same DFlash draft
    # weights: DDTree verifies a tree of candidate continuations and
    # needs model-family specific verifier support.
    supports_ddtree: bool = False
    ddtree_draft_model: str | None = None
    ddtree_speculative_tokens: int | None = None
    ddtree_tree_budget: int | None = None
    # codex round 3 [NIT #3]: minimum unified-memory floor (in GB) for
    # aliases that are unfit for smaller machines. ``None`` means "no
    # hardware gate" — the default for every text/vision model we ship
    # under 100 GB weights. Populated for the flagship-tier Ultra-only
    # entries (currently ``hy3-preview-4bit`` at 166 GB weights + ~156
    # GB peak RSS — needs 192 GB+ M3 Ultra). Enforced as a boot-time
    # WARNING (not a hard block) in ``vllm_mlx/cli.py``.
    min_memory_gb: float | None = None
    # ``is_text_only`` = this checkpoint is served through the
    # auto-regressive text (mlx-lm) lane even though its ``config.json``
    # declares a ``vision_config`` (and may ship ``vision_tower`` weights)
    # that would make ``is_mllm_model`` auto-detection route it to the
    # mlx-vlm MLLM engine. Same shape and spirit as ``is_hybrid_explicit``:
    # a STATE description of the model (parallel to ``is_hybrid`` /
    # ``is_moe``), not an imperative ``force_*`` switch — it pins what the
    # served capability IS so the runtime name/config probe can't route it
    # to a lane we don't support. Canonical example: PrismML
    # Ternary-Bonsai-27B — a Qwen3.5-class checkpoint whose bundled vision
    # tower our mlx-vlm loader can't drive (its GatedDeltaNet/SSM forward
    # garbles output), but whose text backbone is coherent via mlx-lm's
    # ``qwen3_5``.
    #
    # This is the per-alias declarative form of the existing, fully
    # governed ``--no-mllm`` / ``force_text`` routing override (#393,
    # registered in ``tests/test_no_mllm_flag.py::AUTO_ROUTING_FLAG_PAIRS``
    # under the ``--mllm`` / ``--no-mllm`` pair): ``server.load_model``
    # translates ``is_text_only=True`` UNCONDITIONALLY into the registered
    # ``force_text`` kwarg, so the routing decision still flows through the
    # same audited kwarg surface — no new escape hatch. It is applied even
    # when the operator passes ``--mllm``: that collides with the resulting
    # ``force_text=True`` at the ``force_mllm``/``force_text`` mutual-
    # exclusion guard in ``load_model`` and raises loudly, so an operator
    # who insists on the (broken) MLLM path for a text-only-pinned alias
    # gets a clear error rather than a silent flip to the garbling MLLM
    # engine (codex #1116). Default ``False`` leaves every legacy alias on
    # auto-detection untouched — real VLM aliases (Qwen-VL, gemma vision,
    # UI-TARS, …) never set it and keep routing to mlx-vlm exactly as
    # before.
    #
    # Placed LAST in the field list deliberately: although the frozen
    # dataclass is ``kw_only=True`` (positional construction is a loud
    # TypeError, not a silent misbind — see the class docstring), keeping
    # new fields at the end preserves the positional signature for any
    # out-of-tree caller and quiets the recurring "mid-dataclass insert"
    # review flag.
    is_text_only: bool = False

    @property
    def speedup_dict(self) -> dict[str, float]:
        """``suffix_bench_speedup`` materialized as a fresh dict.

        The field is stored as a tuple-of-pairs so the frozen dataclass
        stays hashable and safely shareable; consumers that index by
        workload key (``_suffix_tier_cell``) call this to get a plain
        dict. ``None`` (not benched) yields an empty dict.
        """
        return dict(self.suffix_bench_speedup or ())
