# SPDX-License-Identifier: Apache-2.0
"""Model alias registry — single source of truth for known models.

Each entry in ``aliases.json`` is a per-alias profile: HF path + parser +
capability gates. Code that just needs ``alias → hf_path`` calls
``resolve_model``; code that needs the full profile (parser, hybrid
flag, spec-decode gate, …) calls ``resolve_profile``.

The legacy short form (``"alias": "hf_path"``) is still accepted for
backward compatibility with any external tool that hand-edits the file —
that entry just gets default capability flags.
"""

import difflib
import json
import os

from .model_profile import Modality, ModelProfile

# ``Modality`` and the unified ``ModelProfile`` dataclass live in the
# import-light ``model_profile`` module — the single source of truth for
# per-model profile shape. Re-exported here so existing
# ``from vllm_mlx.model_aliases import Modality`` / ``AliasProfile`` call
# sites keep resolving. See ``model_profile`` for why the class lives
# there (import-light + avoids the model_auto_config↔model_aliases cycle).
# Implemented lanes — what ``load_model`` can actually dispatch to today.
# ``image-gen`` routes to the mflux image lane (runtime/image_lane.py).
# ``vision`` stays RESERVED in the type alias so routing code can pattern-match
# on it once its dispatch path lands, but loading a ``vision`` alias MUST fail
# loud right now — otherwise an aliases.json typo would pass schema validation
# and crash at request time with an unrouted lane (pr_validate codex r13 NIT).
_VALID_MODALITIES: frozenset[str] = frozenset(
    {"text", "text-diffusion", "video-gen", "image-gen"}
)
_RESERVED_MODALITIES: frozenset[str] = frozenset({"vision"})

# Canonical enum for ``suffix_decoding_tier``. Kept here so the contract
# test (tests/test_aliases_contract.py) and any future loader / CLI
# renderer share one source of truth — drift between the two has shipped
# silently before (an alias with tier=``good`` would have been a no-op if
# the loader's allow-list and the CLI's display map disagreed).
#
# - ``unknown``: not benched yet (default)
# - ``neutral``: benched, mixed results, no recommendation either way
# - ``good``:    benched, clearly profitable, hint user to enable
# - ``avoid``:   benched, regression on at least one canonical workload
VALID_SUFFIX_TIERS: frozenset[str] = frozenset({"unknown", "neutral", "good", "avoid"})

# Canonical enum for ``pflash_tier``. PFlash long-prompt compression
# (#287) is a per-model decision: the bench evidence showed 3.87x-8.5x
# TTFT speedups with 100% needle recall on the Qwen3.5 / Qwen3.6 family
# at keep_ratio=0.20, but we have no evidence for other families. To
# avoid a silent quality regression on an unbenched arch, an alias must
# be explicitly tagged ``"verified"`` before the engine defaults
# ``--pflash`` to ``always`` for it; everything else stays ``"unknown"``
# and the engine keeps PFlash off (preserving v0.7.x behaviour). Any
# explicit ``--pflash {off,auto,always}`` flag on the CLI still wins
# over the tier-based default.
#
# - ``unknown``:  not benched / no decision (default, engine keeps PFlash off)
# - ``verified``: bench-validated speedup + recall on this alias; engine
#                 defaults PFlash to ``always`` unless the user overrides
#
# The recall validation is AT the keep_ratio the alias will actually run:
# by default 0.20, or the per-alias ``pflash_keep_ratio`` override when set.
# Some arches (e.g. Ternary-Bonsai-27B) collapse mid-prompt recall at 0.20
# (1/5 needle) but pass 5/5 at 0.50 — such an alias is verified *with* a
# ``pflash_keep_ratio`` pin, never bare at the lossy default.
VALID_PFLASH_TIERS: frozenset[str] = frozenset({"unknown", "verified"})

# Canonical enum for ``turboquant_tier``. ``"k8v4_verified"`` flips the
# no-flag default to ``--kv-cache-turboquant k8v4`` on that alias only;
# explicit CLI still wins. Mirrors ``pflash_tier`` in shape.
VALID_TURBOQUANT_TIERS: frozenset[str] = frozenset({"unknown", "k8v4_verified"})


# Canonical names for block-diffusion speculative-decoding drafter kinds.
# Kept as module constants so eligibility checks, CLI flag handlers, and
# alias validation all reference the same strings.
DFLASH_KIND: str = "dflash"
DDTREE_KIND: str = "ddtree"

_aliases: dict[str, "AliasProfile"] | None = None
# Reverse index: hf_path → first alias that references it. Built once
# alongside ``_aliases`` so reverse lookups in ``resolve_profile`` are
# O(1) instead of scanning all 50+ profiles on every cache-miss.
# When two aliases share the same hf_path (e.g. ``nemotron-30b-4bit`` and
# ``nemotron-30b-4bit`` both pointing at the same MLX repo), the first one
# in JSON order wins. The contract is "any profile valid for this
# path" rather than "the canonical alias", so this is fine.
_hf_to_alias: dict[str, str] | None = None


class RetiredModelAliasError(ValueError):
    """Raised when a known-broken short alias must fail before model loading."""


_RETIRED_MODEL_ALIASES: dict[str, str] = {
    "ministral-3b-4bit": (
        "The 'ministral-3b-4bit' alias was retired because its default "
        "multimodal route can hang on the first text completion. For explicit "
        "text-only testing, use 'mlx-community/Ministral-3-3B-Instruct-2512-4bit' "
        "with --no-mllm."
    ),
}


# ``AliasProfile`` is a DEPRECATED alias of the unified ``ModelProfile``
# (defined in the import-light ``model_profile`` module). Retained so the
# ~20 call sites that import ``AliasProfile`` by name keep resolving; a
# follow-up rename PR migrates them to ``ModelProfile`` and drops this
# shim. It is the SAME class object, not a subclass — construction and
# ``isinstance`` behave identically to ``ModelProfile``.
AliasProfile = ModelProfile


def _coerce(alias: str, value: object) -> AliasProfile:
    """Build an ``AliasProfile`` from a raw JSON value.

    Accepts both the rich dict form and the legacy bare-string form so a
    file edited by hand or carried over from an old release still loads.

    Validates that ``hf_path`` is a non-empty string regardless of the
    schema flavor — an empty path slips silently through every
    downstream check (``resolve_model`` returns ``""``, downloads fail
    with confusing 404s) and the loader is the only honest place to
    catch it.
    """
    if isinstance(value, str):
        if not value:
            raise ValueError(f"alias {alias!r}: hf_path string is empty")
        return AliasProfile(hf_path=value)
    if not isinstance(value, dict) or "hf_path" not in value:
        raise ValueError(
            f"alias {alias!r}: value must be a string or an object with "
            f"'hf_path', got {type(value).__name__}"
        )
    # Closed-key schema: any unknown key is rejected at load time so a
    # contributor can't sneak a covert routing flip into aliases.json
    # (round-4 env-config attack #5). Adding a NEW field requires
    # editing this set AND the dataclass — surfacing the change in
    # review.
    _ALLOWED_PROFILE_KEYS = frozenset(
        {
            "hf_path",
            # Directory inside ``hf_path`` holding this alias's checkpoint.
            # For publishers who ship every quantization as a sibling
            # folder of ONE repo (LiquidAI/LFM2.5-2.6B-MLX → ``4bit/``,
            # ``8bit/``, ``bf16/``, …) rather than one repo per quant.
            # See ModelProfile.subfolder for why this is not folded into
            # ``hf_path``.
            "subfolder",
            "modality",
            # State-pin (parallel to ``is_hybrid`` / ``is_moe``): serve
            # this checkpoint through the text mlx-lm lane even though its
            # config declares a vision tower. server.load_model translates
            # it into the pre-existing, registered ``force_text`` kwarg
            # (``--mllm`` / ``--no-mllm`` pair in
            # tests/test_no_mllm_flag.py::AUTO_ROUTING_FLAG_PAIRS), so the
            # routing decision still flows through the audited kwarg
            # surface. Used by Ternary-Bonsai-27B (mlx-vlm can't drive its
            # bundled vision tower; mlx-lm's qwen3_5 serves the text
            # backbone coherently).
            "is_text_only",
            "tool_call_parser",
            "reasoning_parser",
            "is_hybrid",
            # r6-A R6-C1: pin the JSON-declared is_hybrid value so the
            # runtime ArraysCache probe in
            # ``enrich_model_config`` cannot one-way-flip it to True.
            # See AliasProfile.is_hybrid_explicit for the full
            # rationale.
            "is_hybrid_explicit",
            "is_moe",
            "supports_spec_decode",
            "mtp_draft_model",
            "mtp_speculative_tokens",
            "default_max_tokens",
            "suffix_decoding_tier",
            "suffix_bench_speedup",
            "supports_dflash",
            "dflash_draft_model",
            "supports_ddtree",
            "ddtree_draft_model",
            "ddtree_speculative_tokens",
            "ddtree_tree_budget",
            "min_memory_gb",
            "recommended_sampling",
            "pflash_tier",
            "pflash_keep_ratio",
            "turboquant_tier",
        }
    )
    unknown_keys = set(value.keys()) - _ALLOWED_PROFILE_KEYS
    if unknown_keys:
        raise ValueError(
            f"alias {alias!r}: unknown key(s) {sorted(unknown_keys)}; allowed: "
            f"{sorted(_ALLOWED_PROFILE_KEYS)}. If you intend to add a new "
            "field, update both AliasProfile and _ALLOWED_PROFILE_KEYS — and "
            "if the field is a routing decision (force_*/no_*), it must be "
            "registered in tests/test_no_mllm_flag.py::AUTO_ROUTING_FLAG_PAIRS."
        )
    hf_path = value["hf_path"]
    if not isinstance(hf_path, str) or not hf_path:
        raise ValueError(
            f"alias {alias!r}: 'hf_path' must be a non-empty string, "
            f"got {type(hf_path).__name__}={hf_path!r}"
        )

    # ``subfolder`` becomes a path segment joined onto a snapshot
    # directory, so it is validated as a *relative, downward* path here —
    # the one place that sees the raw JSON. An absolute path or a ``..``
    # segment would let an aliases.json edit point the loader at an
    # arbitrary directory outside the HF cache; a backslash would do the
    # same on a case-insensitive filesystem that normalises it. Rejecting
    # at load time keeps every downstream consumer free to treat the value
    # as a trusted relative path.
    subfolder = value.get("subfolder")
    if subfolder is not None:
        if not isinstance(subfolder, str) or not subfolder:
            raise ValueError(
                f"alias {alias!r}: 'subfolder' must be a non-empty string when "
                f"present, got {type(subfolder).__name__}={subfolder!r}"
            )
        # ``os.path.join`` DISCARDS everything to its left when the right
        # operand is absolute, so an absolute ``subfolder`` silently
        # relocates the load outside the HF cache instead of erroring.
        # ``os.path.isabs`` alone is not enough: it is platform-dependent
        # and returns False for ``C:/x`` on POSIX, so a drive-qualified
        # path would pass validation on the machine that reviews the JSON
        # and escape on the machine that loads it. Reject both spellings
        # everywhere.
        # The value is BOTH a path segment and the literal prefix of an HF
        # ``allow_patterns`` glob. Validating it only as a path would let
        # ``4bit*``, ``[48]bit`` or ``**`` through — each still a legal
        # relative path, each widening the download past the one directory
        # the alias declares, which is the entire guarantee this field
        # exists to make. Reject the metacharacters instead of escaping
        # them: no real publisher names a folder ``model[0]``, and a
        # rejected alias is a loud edit-time error.
        glob_meta = set("*?[]!")
        drive_qualified = len(subfolder) >= 2 and subfolder[1] == ":"
        if (
            subfolder.startswith("/")
            or os.path.isabs(subfolder)
            or drive_qualified
            or "\\" in subfolder
            or ".." in subfolder.split("/")
            or subfolder.endswith("/")
            or glob_meta & set(subfolder)
        ):
            raise ValueError(
                f"alias {alias!r}: 'subfolder' must be a relative path inside "
                f"the repo with no '..' segment and no trailing slash, got "
                f"{subfolder!r}"
            )
    raw_speedup = value.get("suffix_bench_speedup")
    speedup: tuple[tuple[str, float], ...] | None
    if raw_speedup is None:
        speedup = None
    elif isinstance(raw_speedup, dict):
        try:
            speedup = tuple(sorted((k, float(v)) for k, v in raw_speedup.items()))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"alias {alias!r}: suffix_bench_speedup values must be numbers"
            ) from e
    else:
        raise ValueError(
            f"alias {alias!r}: suffix_bench_speedup must be an object, "
            f"got {type(raw_speedup).__name__}"
        )
    tier = value.get("suffix_decoding_tier", "unknown")
    if not isinstance(tier, str):
        raise ValueError(f"alias {alias!r}: suffix_decoding_tier must be a string")

    # PFlash tier — validated against the closed enum here so a typo
    # in aliases.json fails loud at load time. ``_coerce`` is the only
    # place that sees the raw JSON; downstream readers trust the
    # dataclass.
    pflash_tier = value.get("pflash_tier", "unknown")
    if not isinstance(pflash_tier, str):
        raise ValueError(f"alias {alias!r}: pflash_tier must be a string")
    if pflash_tier not in VALID_PFLASH_TIERS:
        raise ValueError(
            f"alias {alias!r}: pflash_tier={pflash_tier!r} not in "
            f"{sorted(VALID_PFLASH_TIERS)}"
        )

    # Optional per-alias PFlash keep_ratio override. Absent → ``None`` →
    # engine default 0.20. When present it must be a real fraction in (0, 1]
    # AND the alias must be pflash_tier=verified — the field's ONLY purpose is
    # to pin the ratio a verified alias was recall-validated at (the resolver
    # applies it whenever PFlash runs, incl. an explicit ``--pflash always``,
    # so allowing it on an unknown-tier alias would silently shift explicitly
    # enabled PFlash behaviour). Both validated here so a typo / misuse fails
    # loud at load time next to ``pflash_tier``.
    pflash_keep_ratio = value.get("pflash_keep_ratio")
    if pflash_keep_ratio is not None:
        if isinstance(pflash_keep_ratio, bool) or not isinstance(
            pflash_keep_ratio, (int, float)
        ):
            raise ValueError(f"alias {alias!r}: pflash_keep_ratio must be a number")
        pflash_keep_ratio = float(pflash_keep_ratio)
        if not (0.0 < pflash_keep_ratio <= 1.0):
            raise ValueError(
                f"alias {alias!r}: pflash_keep_ratio={pflash_keep_ratio!r} "
                "must be > 0.0 and <= 1.0"
            )
        if pflash_tier != "verified":
            raise ValueError(
                f"alias {alias!r}: pflash_keep_ratio is only valid with "
                f"pflash_tier='verified' (got {pflash_tier!r}). The override "
                "pins the ratio a VERIFIED alias was recall-validated at; it "
                "has no meaning on an unbenched alias."
            )

    turboquant_tier = value.get("turboquant_tier", "unknown")
    if not isinstance(turboquant_tier, str):
        raise ValueError(f"alias {alias!r}: turboquant_tier must be a string")
    if turboquant_tier not in VALID_TURBOQUANT_TIERS:
        raise ValueError(
            f"alias {alias!r}: turboquant_tier={turboquant_tier!r} not in "
            f"{sorted(VALID_TURBOQUANT_TIERS)}"
        )

    # Strict bool coercion — bare ``bool(...)`` treats the string
    # ``"false"`` as True and silently flips a careful maintainer's
    # intent. Validate the JSON type explicitly so a typo in
    # aliases.json fails loud at load time.
    def _strict_bool(key: str, default: bool) -> bool:
        raw = value.get(key, default)
        if not isinstance(raw, bool):
            raise ValueError(
                f"alias {alias!r}: {key} must be a JSON boolean, "
                f"got {type(raw).__name__}={raw!r}"
            )
        return raw

    supports_dflash = _strict_bool("supports_dflash", False)
    dflash_draft_model = value.get("dflash_draft_model")
    if supports_dflash and not dflash_draft_model:
        # Fail loud here, not at server-start — a half-populated DFlash
        # alias would silently fall back to AR and look like a perf bug.
        raise ValueError(
            f"alias {alias!r}: supports_dflash=true requires "
            f"dflash_draft_model to be set"
        )
    if dflash_draft_model is not None and not isinstance(dflash_draft_model, str):
        raise ValueError(
            f"alias {alias!r}: dflash_draft_model must be a string, "
            f"got {type(dflash_draft_model).__name__}"
        )
    supports_ddtree = _strict_bool("supports_ddtree", False)
    ddtree_draft_model = value.get("ddtree_draft_model")
    if supports_ddtree and not ddtree_draft_model:
        raise ValueError(
            f"alias {alias!r}: supports_ddtree=true requires "
            f"ddtree_draft_model to be set"
        )
    if ddtree_draft_model is not None and not isinstance(ddtree_draft_model, str):
        raise ValueError(
            f"alias {alias!r}: ddtree_draft_model must be a string, "
            f"got {type(ddtree_draft_model).__name__}"
        )

    def _optional_positive_int(key: str) -> int | None:
        raw = value.get(key)
        if raw is None:
            return None
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(
                f"alias {alias!r}: {key} must be a positive integer, "
                f"got {type(raw).__name__}={raw!r}"
            )
        if raw <= 0:
            raise ValueError(
                f"alias {alias!r}: {key} must be a positive integer, got {raw}"
            )
        return raw

    ddtree_speculative_tokens = _optional_positive_int("ddtree_speculative_tokens")
    ddtree_tree_budget = _optional_positive_int("ddtree_tree_budget")

    # ``min_memory_gb`` (codex #1069 round 3 [NIT #3]) — accepted as a
    # positive number (int or float). ``None`` = no hardware gate;
    # rejected on non-numeric / zero / negative so a typo fails at load
    # time instead of silently disabling the guard for an Ultra-only
    # alias.
    raw_min_mem = value.get("min_memory_gb")
    min_memory_gb: float | None
    if raw_min_mem is None:
        min_memory_gb = None
    elif isinstance(raw_min_mem, bool) or not isinstance(raw_min_mem, (int, float)):
        raise ValueError(
            f"alias {alias!r}: min_memory_gb must be a positive number, "
            f"got {type(raw_min_mem).__name__}={raw_min_mem!r}"
        )
    elif raw_min_mem <= 0:
        raise ValueError(
            f"alias {alias!r}: min_memory_gb must be a positive number, "
            f"got {raw_min_mem}"
        )
    else:
        min_memory_gb = float(raw_min_mem)
    raw_sampling = value.get("recommended_sampling")
    recommended_sampling: tuple[tuple[str, float], ...] | None
    if raw_sampling is None:
        recommended_sampling = None
    elif isinstance(raw_sampling, dict):
        _ALLOWED_SAMPLING_KEYS = {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
        }
        items: list[tuple[str, float]] = []
        for k, v in raw_sampling.items():
            if k not in _ALLOWED_SAMPLING_KEYS:
                raise ValueError(
                    f"alias {alias!r}: recommended_sampling has "
                    f"unsupported key {k!r}; allowed: "
                    f"{sorted(_ALLOWED_SAMPLING_KEYS)}"
                )
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(
                    f"alias {alias!r}: recommended_sampling[{k!r}] "
                    f"must be a number, got {type(v).__name__}"
                )
            if k == "top_k":
                # ``top_k`` is an integer count; silently truncating
                # 20.5 → 20 would hide a typo in a hand-edited
                # aliases.json. Mirror the same guard the loader at
                # utils/generation_config.py applies to the JSON layer.
                if isinstance(v, float) and not v.is_integer():
                    raise ValueError(
                        f"alias {alias!r}: recommended_sampling['top_k'] "
                        f"must be a whole number, got {v!r}"
                    )
            items.append((k, float(v)))
        recommended_sampling = tuple(sorted(items)) if items else None
    else:
        raise ValueError(
            f"alias {alias!r}: recommended_sampling must be an object, "
            f"got {type(raw_sampling).__name__}"
        )
    raw_modality = value.get("modality", "text")
    if not isinstance(raw_modality, str):
        raise ValueError(
            f"alias {alias!r}: modality must be one of "
            f"{sorted(_VALID_MODALITIES)}, got {raw_modality!r}"
        )
    if raw_modality in _RESERVED_MODALITIES:
        # Type alias keeps these for forward compat, but loading
        # fails loud until their dispatch lands (pr_validate codex
        # r13 NIT).
        raise ValueError(
            f"alias {alias!r}: modality={raw_modality!r} is reserved but "
            "not yet implemented — there is no dispatch path for it. "
            f"Use one of {sorted(_VALID_MODALITIES)} or wait for the "
            "matching engine to land."
        )
    if raw_modality not in _VALID_MODALITIES:
        raise ValueError(
            f"alias {alias!r}: modality must be one of "
            f"{sorted(_VALID_MODALITIES)}, got {raw_modality!r}"
        )
    modality: Modality = raw_modality  # type: ignore[assignment]
    # ``is_text_only`` — state-pin that serves a vision-config checkpoint
    # through the AR text mlx-lm lane (translated to the ``force_text``
    # routing kwarg in server.load_model). Only meaningful on the ``text``
    # modality: a non-``text`` modality already picks its own dedicated
    # lane (text-diffusion → DiffusionEngine), so combining the two is a
    # contradiction that must fail loud rather than silently pick one.
    is_text_only = _strict_bool("is_text_only", False)
    # Capability gates that only make sense for the auto-regressive LLM
    # lane. Catching the mismatch here keeps the diffusion / vision /
    # image-gen lanes from silently inheriting a routing decision that
    # would never apply to them — and makes a bad aliases.json entry
    # fail loud at load instead of misroute at request time.
    if modality != "text":
        if is_text_only:
            raise ValueError(
                f"alias {alias!r}: is_text_only=true is only valid when "
                f"modality='text' (it serves the checkpoint through the AR "
                f"text mlx-lm lane); got modality={modality!r}"
            )
        if _strict_bool("supports_spec_decode", True):
            raise ValueError(
                f"alias {alias!r}: supports_spec_decode must be false when "
                f"modality={modality!r} (only the text lane runs the AR "
                "speculative-decoding stack)"
            )
        if supports_dflash:
            raise ValueError(
                f"alias {alias!r}: supports_dflash must be false when "
                f"modality={modality!r} (DFlash is AR-only)"
            )
        if supports_ddtree:
            raise ValueError(
                f"alias {alias!r}: supports_ddtree must be false when "
                f"modality={modality!r} (DDTree is AR-only)"
            )

    mtp_draft_model = value.get("mtp_draft_model")
    if mtp_draft_model is not None and (
        not isinstance(mtp_draft_model, str)
        or not mtp_draft_model.strip()
        or "/" not in mtp_draft_model
    ):
        raise ValueError(
            f"alias {alias!r}: mtp_draft_model must use non-empty 'org/repo' format"
        )
    if "mtp_speculative_tokens" in value and mtp_draft_model is None:
        raise ValueError(
            f"alias {alias!r}: mtp_speculative_tokens requires mtp_draft_model"
        )
    mtp_speculative_tokens = value.get("mtp_speculative_tokens", 3)
    if (
        isinstance(mtp_speculative_tokens, bool)
        or not isinstance(mtp_speculative_tokens, int)
        or mtp_speculative_tokens <= 0
    ):
        raise ValueError(
            f"alias {alias!r}: mtp_speculative_tokens must be a positive integer"
        )

    return AliasProfile(
        hf_path=hf_path,
        subfolder=subfolder,
        modality=modality,
        is_text_only=is_text_only,
        tool_call_parser=value.get("tool_call_parser"),
        reasoning_parser=value.get("reasoning_parser"),
        is_hybrid=_strict_bool("is_hybrid", False),
        is_hybrid_explicit=_strict_bool("is_hybrid_explicit", False),
        is_moe=_strict_bool("is_moe", False),
        supports_spec_decode=_strict_bool("supports_spec_decode", True),
        mtp_draft_model=mtp_draft_model,
        mtp_speculative_tokens=mtp_speculative_tokens,
        default_max_tokens=value.get("default_max_tokens"),
        suffix_decoding_tier=tier,
        suffix_bench_speedup=speedup,
        supports_dflash=supports_dflash,
        dflash_draft_model=dflash_draft_model,
        supports_ddtree=supports_ddtree,
        ddtree_draft_model=ddtree_draft_model,
        ddtree_speculative_tokens=ddtree_speculative_tokens,
        ddtree_tree_budget=ddtree_tree_budget,
        recommended_sampling=recommended_sampling,
        pflash_tier=pflash_tier,
        pflash_keep_ratio=pflash_keep_ratio,
        turboquant_tier=turboquant_tier,
        min_memory_gb=min_memory_gb,
    )


def _load() -> dict[str, AliasProfile]:
    global _aliases, _hf_to_alias
    if _aliases is None:
        path = os.path.join(os.path.dirname(__file__), "aliases.json")
        with open(path) as f:
            raw = json.load(f)
        parsed = {alias: _coerce(alias, v) for alias, v in raw.items()}
        # Validate BEFORE publishing to the module globals. Assigning first
        # would leave a caught exception with a populated ``_aliases``, and
        # every later ``_load()`` would take the memoized fast path and use
        # the registry this check just rejected — failing open exactly once
        # and then silently forever.
        _assert_subfolder_is_unambiguous(parsed)
        # Build reverse index in JSON-insertion order so the "first alias
        # wins" rule is deterministic.
        index: dict[str, str] = {}
        for alias, profile in parsed.items():
            index.setdefault(profile.hf_path, alias)
        _aliases, _hf_to_alias = parsed, index
    return _aliases


def _assert_subfolder_is_unambiguous(profiles: dict[str, AliasProfile]) -> None:
    """Reject an aliases.json where ``subfolder`` cannot be recovered.

    ``resolve_subfolder`` reaches the loader through the reverse
    ``hf_path → first alias`` index, because by the time the text lane
    loads, the user-typed alias has already been resolved to a bare repo
    id. "First alias wins" is harmless for every other profile field —
    two aliases on one repo agree about the parsers and capability gates
    — but ``subfolder`` is the one field where they would legitimately
    DISAGREE: ``…-2.6b-4bit`` and ``…-2.6b-8bit`` are the same repo and
    different directories.

    Rather than silently serve 8-bit weights to someone who asked for
    4-bit, fail at registry load. Lifting this restriction means
    threading the alias (not just the resolved path) down to
    ``load_model_with_fallback`` — a deliberate piece of work, not
    something to back into by adding a JSON line.
    """
    by_path: dict[str, dict[str, str | None]] = {}
    for alias, profile in profiles.items():
        by_path.setdefault(profile.hf_path, {})[alias] = profile.subfolder
    for hf_path, members in by_path.items():
        if len(set(members.values())) > 1:
            raise ValueError(
                f"aliases {sorted(members)} share hf_path {hf_path!r} but "
                f"declare different 'subfolder' values "
                f"({ {a: s for a, s in members.items()} }). The loader "
                "recovers the subfolder by reverse-lookup from the resolved "
                "hf_path, so it cannot tell them apart. Give each quant its "
                "own repo, or thread the alias through to "
                "utils.tokenizer.load_model_with_fallback."
            )


def checkpoint_prefix(name: str) -> str:
    """``"4bit/"`` when ``name``'s checkpoint lives in a repo subfolder.

    The single implementation shared by every consumer that has to reach
    a file inside the checkpoint — cache probes (``config.json``),
    size estimates, and completeness checks. Returns ``""`` for the
    ordinary flat repo, so callers can prepend it unconditionally.

    Fail-soft: an unreadable registry yields ``""``, i.e. the historical
    whole-repo behaviour, never an exception. These callers run on the
    startup path and several of them are explicitly best-effort.
    """
    try:
        subfolder = resolve_subfolder(name)
    except Exception:
        return ""
    return f"{subfolder}/" if subfolder else ""


def subfolder_allow_patterns(name: str) -> list[str] | None:
    """``allow_patterns`` that fetch only ``name``'s checkpoint, or ``None``.

    A repo that ships every quantization side by side is many times larger
    than the one directory a given alias needs — ``LiquidAI/LFM2.5-2.6B-MLX``
    is ~20 GB across eight quants, of which ``4bit/`` is 1.6 GB. Every
    caller that downloads or measures such a repo must pass these
    patterns, or the user waits for (and stores) seven checkpoints they
    did not ask for.
    """
    subfolder = resolve_subfolder(name)
    return [f"{subfolder}/*"] if subfolder else None


def resolve_subfolder(name: str) -> str | None:
    """The in-repo directory holding ``name``'s checkpoint, if any.

    Accepts either a user-typed alias or the already-resolved HF repo id
    (``resolve_profile`` handles both). Returns ``None`` for the ordinary
    case where the repo root IS the checkpoint.
    """
    profile = resolve_profile(name)
    return profile.subfolder if profile is not None else None


def resolve_model(name: str) -> str:
    """Resolve a model alias to its full HuggingFace path.

    If a local file/directory with the name exists, prefer that.
    If a configured external-model root contains the repo, serve it in place.
    If name contains '/' it's already a full Hugging Face path — pass through.
    If name is a retired, known-broken alias, raise before any download or load.
    If name matches an alias, return the mapped HF path.
    Otherwise return unchanged.
    """
    if os.path.exists(name):
        return name
    if reason := _RETIRED_MODEL_ALIASES.get(name):
        raise RetiredModelAliasError(reason)
    from .user_aliases import validated_user_aliases

    builtins = {alias: profile.hf_path for alias, profile in _load().items()}
    if target := validated_user_aliases(builtins, user_alias_reserved_names()).get(
        name
    ):
        profile = _load().get(target)
        return profile.hf_path if profile is not None else target
    # Preserve the historical resolver hot path exactly unless the external
    # feature is configured. In particular, ordinary chat/serve resolution
    # must not introduce a cache/download-gate probe.
    if not os.environ.get("RAPID_MLX_EXTRA_MODEL_ROOTS", "").strip():
        if "/" in name:
            return name
        profile = _load().get(name)
        return profile.hf_path if profile is not None else name
    if "/" in name:
        if not _managed_hub_model_is_runnable(name):
            if external := _resolve_external_model_path(name):
                return external
        return name
    if _managed_hub_model_is_runnable(name):
        profile = _load().get(name)
        return profile.hf_path if profile is not None else name
    if external := _resolve_external_model_path(name):
        return external
    profile = _load().get(name)
    return profile.hf_path if profile is not None else name


def _managed_hub_model_is_runnable(name: str) -> bool:
    """Apply the cached-listing's managed-hub precedence at launch too."""
    profile = _load().get(name)
    repo = profile.hf_path if profile is not None else name
    try:
        from ._download_gate import _snapshot_is_complete_split_model, is_repo_cached

        return is_repo_cached(repo) or _snapshot_is_complete_split_model(repo)
    except Exception:
        return False


def _resolve_external_model_path(name: str) -> str | None:
    """Resolve a discovered external repo back to its canonical directory.

    Discovery renders a stable ``publisher/repo`` identifier for the CLI and
    desktop UI, but the loader must receive the local directory or it will
    interpret that identifier as a Hugging Face repo and download the model
    again.  Search the same ordered roots used by discovery and accept only
    the one- or two-component layouts that scanner emits.

    ``realpath`` + ``commonpath`` keeps ``..`` and symlinked candidates from
    escaping the user-nominated root.  Completeness is rechecked at launch so
    a row that became partial after catalog refresh cannot be served as ready.
    """
    # A root-level store may use the Rapid alias literally, while another
    # runtime may use the profile's canonical publisher/repo layout. Probe the
    # displayed identifier first, then that canonical fallback.
    profile = _load().get(name)
    external_names = [name]
    if profile is not None and profile.hf_path != name:
        external_names.append(profile.hf_path)

    raw_roots = os.environ.get("RAPID_MLX_EXTRA_MODEL_ROOTS", "")
    if not raw_roots:
        return None

    from ._download_gate import _snapshot_is_complete

    trusted_roots = [
        os.path.realpath(os.path.expanduser(value.strip()))
        for value in _external_model_root_values(raw_roots)
        if value.strip()
    ]
    for root in trusted_roots:
        for external_name in external_names:
            parts = _external_model_identifier_parts(external_name)
            if parts is None:
                continue
            candidate = os.path.realpath(os.path.join(root, *parts))
            try:
                if os.path.commonpath((root, candidate)) != root:
                    continue
            except ValueError:
                continue
            if os.path.isdir(candidate):
                try:
                    if _external_model_tree_is_contained(
                        candidate, trusted_roots
                    ) and _snapshot_is_complete(candidate):
                        return candidate
                except OSError:
                    continue
    return None


def _external_model_tree_is_contained(directory: str, roots: list[str]) -> bool:
    """Require a readable tree whose links stay in explicitly trusted roots."""
    canonical_roots = [os.path.realpath(root) for root in roots]

    def contained(path: str) -> bool:
        target = os.path.realpath(path)
        for root in canonical_roots:
            try:
                if os.path.commonpath((root, target)) == root:
                    return True
            except (OSError, ValueError):
                continue
        return False

    if not contained(directory):
        return False
    try:

        def inaccessible(error: OSError) -> None:
            raise error

        for current, directories, files in os.walk(
            directory, followlinks=False, onerror=inaccessible
        ):
            for name in (*directories, *files):
                path = os.path.join(current, name)
                if os.path.islink(path) and not contained(path):
                    return False
    except OSError:
        return False
    return True


def _external_model_root_values(raw: str) -> list[str]:
    """Decode the shared desktop/engine external-root environment value."""
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            import json

            decoded = json.loads(raw)
            if isinstance(decoded, list):
                return [value for value in decoded if isinstance(value, str)]
        except (TypeError, ValueError):
            pass
        # A legacy path may itself begin with ``[``. Invalid/non-array JSON
        # therefore falls through to the pathsep protocol instead of
        # silently erasing a valid configured root.
    return raw.split(os.pathsep)


def _external_model_identifier_parts(name: str) -> list[str] | None:
    """Return safe path/display components for an external model id.

    The identifier crosses a terminal-oriented CLI boundary before Swift
    parses it.  Restrict it to the same conservative alphabet Hugging Face
    repository ids use so a directory name cannot inject a row, ANSI escape,
    or extra whitespace-delimited columns into that protocol.
    """
    parts = name.split("/")
    if len(parts) not in (1, 2):
        return None
    for part in parts:
        if (
            not part
            or part in (".", "..")
            or part.startswith("-")
            or not part.isascii()
            or not all(character.isalnum() or character in "._-" for character in part)
        ):
            return None
    return parts


def list_aliases() -> dict[str, str]:
    """Return all aliases as ``{alias: hf_path}`` (legacy view)."""
    builtins = {alias: profile.hf_path for alias, profile in _load().items()}
    from .user_aliases import validated_user_aliases

    users = validated_user_aliases(builtins, user_alias_reserved_names())
    return builtins | {
        alias: builtins.get(target, target) for alias, target in users.items()
    }


def list_builtin_aliases() -> dict[str, str]:
    """Return the immutable catalog aliases, excluding user mappings."""
    return {alias: profile.hf_path for alias, profile in _load().items()}


def user_alias_reserved_names() -> frozenset[str]:
    """Names user aliases may not shadow, including retired catalog names."""
    return frozenset(_load()) | frozenset(_RETIRED_MODEL_ALIASES)


def list_profiles() -> dict[str, AliasProfile]:
    """Return all alias profiles. Use this when you need parser/capability
    info, not just the HF path."""
    profiles = dict(_load())
    from .user_aliases import validated_user_aliases

    users = validated_user_aliases(
        {alias: profile.hf_path for alias, profile in profiles.items()},
        user_alias_reserved_names(),
    )
    for alias, target in users.items():
        target_profile = profiles.get(target)
        if target_profile is None and _hf_to_alias is not None:
            canonical = _hf_to_alias.get(target)
            if canonical is not None:
                target_profile = profiles[canonical]
        profiles[alias] = target_profile or AliasProfile(hf_path=target)
    return profiles


def resolve_profile(name: str) -> AliasProfile | None:
    """Return the profile for an alias name or full HF path.

    Two lookups in order:
    1. Direct alias name match (``qwen3.5-4b-4bit``).
    2. Reverse HF-path match (``mlx-community/Qwen3.5-4B-MLX-4bit``)
       via the pre-built ``_hf_to_alias`` index — O(1).

    Returns ``None`` if no alias covers this name/path — caller should
    then fall back to the regex-based ``detect_model_config``.
    """
    profiles = _load()  # also populates _hf_to_alias on first call
    direct = profiles.get(name)
    if direct is not None:
        return direct
    from .user_aliases import validated_user_aliases

    users = validated_user_aliases(
        {alias: profile.hf_path for alias, profile in profiles.items()},
        user_alias_reserved_names(),
    )
    if target := users.get(name):
        target_profile = profiles.get(target)
        if target_profile is None and _hf_to_alias is not None:
            canonical = _hf_to_alias.get(target)
            if canonical is not None:
                target_profile = profiles[canonical]
        return target_profile or AliasProfile(hf_path=target)
    if "/" in name and _hf_to_alias is not None:
        canonical = _hf_to_alias.get(name)
        if canonical is not None:
            return profiles[canonical]
    return None


def _family_prefix(name: str) -> str:
    """Strip trailing size/quant tokens to get the model-family prefix.

    ``deepseek-v4-27b`` → ``deepseek-v4`` (drop ``27b``)
    ``qwen3.5-122b-8bit`` → ``qwen3.5`` (drop ``8bit`` then ``122b``)
    ``hermes`` → ``hermes`` (single token, no change)

    Used to keep typo suggestions inside the same family — ``deepseek-v4-27b``
    suggests ``deepseek-v4-flash-8bit``, not ``deepseek-r1-32b-4bit``.
    """
    parts = name.split("-")
    while parts:
        tail = parts[-1]
        if not tail:
            break
        # size token (``27b``, ``1.5b``), quant token (``8bit``, ``mxfp4``),
        # or pure-digit version segment.
        if tail[-1].lower() == "b" or "bit" in tail.lower() or tail.isdigit():
            parts.pop()
            continue
        break
    return "-".join(parts)


def _letters_only_prefix(name: str) -> str:
    """Extract the leading ``[a-z]+`` run from ``name`` (lowercased).

    Used as a fallback family hint when the dash-aware ``_family_prefix``
    returns nothing useful — handles cases where the user collapses or
    inserts separators we don't use (``gemma4-27b`` → ``gemma``, matches
    our ``gemma-4-*`` and ``gemma3-*`` aliases; ``mistral24b`` →
    ``mistral``, matches ``mistral-24b-4bit``).
    """
    out = []
    for ch in name.lower():
        if ch.isalpha():
            out.append(ch)
        else:
            break
    return "".join(out)


def suggest_similar(name: str, n: int = 3, cutoff: float = 0.5) -> list[str]:
    """Return up to ``n`` aliases similar to ``name`` for typo suggestions.

    Family-aware in two passes:
    1. **Strict family match** — uses ``_family_prefix`` (drops trailing
       size/quant tokens). Keeps the wrong-family bait-and-switch (typing
       ``deepseek-v4-27b`` and being told ``deepseek-r1-32b-4bit``) from
       happening, and prevents legitimate single-segment HuggingFace IDs
       like ``gpt2`` or ``bert-base-uncased`` from spuriously matching.
    2. **Letter-only prefix fallback** — if step 1 finds nothing, retry
       using the ``[a-z]+`` prefix (e.g. ``gemma4-27b`` → ``gemma``). The
       cutoff is dropped here because we already filtered by family
       overlap; difflib just orders by closeness within the family.

    Returns ``[]`` only when neither pass finds anything in the same
    letter family — at which point the caller should show a curated
    "popular models" fallback rather than leave the user empty-handed.
    """
    aliases = list(_load().keys())

    # Pass 1: strict family prefix.
    fam = _family_prefix(name)
    same_fam: list[str] = []
    if fam:
        if "-" in fam:
            same_fam = [a for a in aliases if a.startswith(fam + "-") or a == fam]
        elif len(fam) >= 3:
            same_fam = [a for a in aliases if a.startswith(fam)]
        if same_fam and same_fam != [fam]:
            # If we found candidates in the same strict family, trust the
            # cutoff — even if it filters everything out. The cutoff
            # rejecting ``gpt2`` against ``gpt-oss-20b-mxfp4-q8`` is the
            # legitimate-HF-ID guarantee at work; the letter-only
            # fallback below would override that and is wrong here.
            return difflib.get_close_matches(name, same_fam, n=n, cutoff=cutoff)
        # If the strict pass found ONLY the bare-prefix alias itself
        # (e.g. user typed ``gemma4-26b``, fam stripped to ``gemma4``
        # which is the new short alias), fall through to the letter-only
        # pass below so the size-qualified variants surface instead of
        # bait-and-switching the user onto the bare default.

    # Pass 2: letter-only prefix fallback. Gated to inputs where the
    # strict family parser *had to strip something* (signal that the user
    # typed a name following our size/quant naming convention) — handles
    # ``gemma4-27b`` (fam stripped to ``gemma4``, no exact match) and
    # ``mistral24b`` (fam stripped to empty by the trailing ``-b``-ish
    # token). Untouched inputs like ``gpt2``, ``bert-base-uncased`` or
    # ``qwen-coder`` skip this fallback so legit single-segment HF repo
    # IDs aren't bait-and-switched.
    if fam == name:
        return []
    letter_fam = _letters_only_prefix(name)
    if len(letter_fam) < 3:
        return []
    same_letter_fam = [a for a in aliases if _letters_only_prefix(a) == letter_fam]
    if not same_letter_fam:
        return []
    # Within a family, order by similarity to the typed name. No cutoff —
    # any same-letter-family alias is a sane suggestion.
    ranked = sorted(
        same_letter_fam,
        key=lambda a: difflib.SequenceMatcher(None, name, a).ratio(),
        reverse=True,
    )
    return ranked[:n]


# Curated "what should a brand-new user try" list. Surfaced when the user
# typed a name we couldn't match to anything (or even fuzzy-match within a
# family). Hand-picked rather than auto-generated so it always leads with
# the small/fast tier and one well-known representative per category —
# auto-generation would spit out alphabetic noise like ``bonsai-*`` first.
POPULAR_ALIASES: tuple[str, ...] = (
    "qwen3.5-4b-4bit",  # default smoke / small
    "qwen3.5-9b-4bit",  # mid-size general
    "qwen3.6-27b-4bit",  # latest hybrid family
    "qwen3-coder-30b-4bit",  # coding
    "gemma-4-12b-qat-4bit",  # gemma family rep (12B QAT 4-bit)
    "gpt-oss-20b",  # 0.10.0: OpenAI open-weights harmony family
    "llama3-3b-4bit",  # tiny llama
    "mistral-24b-4bit",  # mistral
    "deepseek-r1-32b-4bit",  # reasoning
)
