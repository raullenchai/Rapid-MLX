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
from dataclasses import dataclass

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


# Canonical name for the DFlash speculative-decoding drafter kind. Kept
# as a module constant so eligibility checks, CLI flag handlers, and
# alias validation all reference the same string.
DFLASH_KIND: str = "dflash"

_aliases: dict[str, "AliasProfile"] | None = None
# Reverse index: hf_path → first alias that references it. Built once
# alongside ``_aliases`` so reverse lookups in ``resolve_profile`` are
# O(1) instead of scanning all 50+ profiles on every cache-miss.
# When two aliases share the same hf_path (e.g. ``nemotron-30b`` and
# ``nemotron-nano`` both pointing at the same MLX repo), the first one
# in JSON order wins. The contract is "any profile valid for this
# path" rather than "the canonical alias", so this is fine.
_hf_to_alias: dict[str, str] | None = None


@dataclass(frozen=True)
class AliasProfile:
    """Per-alias profile — resolved from ``aliases.json``.

    Mirrors a subset of ``ModelConfig``'s fields. Kept separate so this
    module stays import-light (``model_auto_config`` pulls in regex,
    typing, etc. that aren't needed when callers only want the alias →
    hf_path map).
    """

    hf_path: str
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    is_hybrid: bool = False
    # MoE / sparse-expert architecture (A3B, A10B, A17B Qwen3.5/3.6 variants,
    # plus future Mixtral/Granite-MoE families). Tracked separately from
    # ``is_hybrid`` because the two attributes gate different downstream
    # paths — hybrid affects ArraysCache/GDN rollback, MoE affects DFlash
    # acceptance rate (the drafter's hidden-state fusion misfires on
    # expert-routing churn; PoC measured 0.76-0.82× regression regardless
    # of precision on Qwen3.6-35B-A3B).
    is_moe: bool = False
    supports_spec_decode: bool = True
    default_max_tokens: int | None = None
    # SuffixDecoding eligibility — populated from cross-model bench (issue #269).
    # ``None`` for ``suffix_bench_speedup`` means "not benched yet"; the tier
    # then defaults to ``"unknown"`` and the startup hint stays silent.
    # When populated, ``suffix_bench_speedup`` is a tuple of ``(workload, speedup)``
    # pairs (tuple, not dict, so frozen dataclass instances stay safely shareable).
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
    hf_path = value["hf_path"]
    if not isinstance(hf_path, str) or not hf_path:
        raise ValueError(
            f"alias {alias!r}: 'hf_path' must be a non-empty string, "
            f"got {type(hf_path).__name__}={hf_path!r}"
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
    return AliasProfile(
        hf_path=hf_path,
        tool_call_parser=value.get("tool_call_parser"),
        reasoning_parser=value.get("reasoning_parser"),
        is_hybrid=_strict_bool("is_hybrid", False),
        is_moe=_strict_bool("is_moe", False),
        supports_spec_decode=_strict_bool("supports_spec_decode", True),
        default_max_tokens=value.get("default_max_tokens"),
        suffix_decoding_tier=tier,
        suffix_bench_speedup=speedup,
        supports_dflash=supports_dflash,
        dflash_draft_model=dflash_draft_model,
        recommended_sampling=recommended_sampling,
    )


def _load() -> dict[str, AliasProfile]:
    global _aliases, _hf_to_alias
    if _aliases is None:
        path = os.path.join(os.path.dirname(__file__), "aliases.json")
        with open(path) as f:
            raw = json.load(f)
        _aliases = {alias: _coerce(alias, v) for alias, v in raw.items()}
        # Build reverse index in JSON-insertion order so the "first alias
        # wins" rule is deterministic.
        _hf_to_alias = {}
        for alias, profile in _aliases.items():
            _hf_to_alias.setdefault(profile.hf_path, alias)
    return _aliases


def resolve_model(name: str) -> str:
    """Resolve a model alias to its full HuggingFace path.

    If name contains '/' it's already a full path — pass through.
    If a local file/directory with the name exists, prefer that.
    If name matches an alias, return the mapped HF path.
    Otherwise return unchanged.
    """
    if "/" in name:
        return name
    if os.path.exists(name):
        return name
    profile = _load().get(name)
    return profile.hf_path if profile is not None else name


def list_aliases() -> dict[str, str]:
    """Return all aliases as ``{alias: hf_path}`` (legacy view)."""
    return {alias: profile.hf_path for alias, profile in _load().items()}


def list_profiles() -> dict[str, AliasProfile]:
    """Return all alias profiles. Use this when you need parser/capability
    info, not just the HF path."""
    return dict(_load())


def resolve_profile(name: str) -> AliasProfile | None:
    """Return the profile for an alias name or full HF path.

    Two lookups in order:
    1. Direct alias name match (``qwen3.5-4b``).
    2. Reverse HF-path match (``mlx-community/Qwen3.5-4B-MLX-4bit``)
       via the pre-built ``_hf_to_alias`` index — O(1).

    Returns ``None`` if no alias covers this name/path — caller should
    then fall back to the regex-based ``detect_model_config``.
    """
    profiles = _load()  # also populates _hf_to_alias on first call
    direct = profiles.get(name)
    if direct is not None:
        return direct
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
    suggests ``deepseek-v4-flash``, not ``deepseek-r1-32b``.
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
    ``mistral``, matches ``mistral-24b``).
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
       ``deepseek-v4-27b`` and being told ``deepseek-r1-32b``) from
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
        if same_fam:
            # If we found candidates in the same strict family, trust the
            # cutoff — even if it filters everything out. The cutoff
            # rejecting ``gpt2`` against ``gpt-oss-20b`` is the
            # legitimate-HF-ID guarantee at work; the letter-only
            # fallback below would override that and is wrong here.
            return difflib.get_close_matches(name, same_fam, n=n, cutoff=cutoff)

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
    "qwen3.5-4b",  # default smoke / small
    "qwen3.5-9b",  # mid-size general
    "qwen3.6-27b",  # latest hybrid family
    "qwen3-coder-30b",  # coding
    "gemma3-12b",  # gemma family rep
    "llama3-3b",  # tiny llama
    "mistral-24b",  # mistral
    "deepseek-r1-32b",  # reasoning
)
