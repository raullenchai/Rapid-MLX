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
    supports_spec_decode: bool = True
    default_max_tokens: int | None = None
    # SuffixDecoding eligibility — populated from cross-model bench (issue #269).
    # ``None`` for ``suffix_bench_speedup`` means "not benched yet"; the tier
    # then defaults to ``"unknown"`` and the startup hint stays silent.
    # When populated, ``suffix_bench_speedup`` is a tuple of ``(workload, speedup)``
    # pairs (tuple, not dict, so frozen dataclass instances stay safely shareable).
    suffix_decoding_tier: str = "unknown"
    suffix_bench_speedup: tuple[tuple[str, float], ...] | None = None


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
    return AliasProfile(
        hf_path=hf_path,
        tool_call_parser=value.get("tool_call_parser"),
        reasoning_parser=value.get("reasoning_parser"),
        is_hybrid=bool(value.get("is_hybrid", False)),
        supports_spec_decode=bool(value.get("supports_spec_decode", True)),
        default_max_tokens=value.get("default_max_tokens"),
        suffix_decoding_tier=tier,
        suffix_bench_speedup=speedup,
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


def suggest_similar(name: str, n: int = 3, cutoff: float = 0.5) -> list[str]:
    """Return up to ``n`` aliases similar to ``name`` for typo suggestions.

    Family-aware: only suggests aliases that share a family prefix with the
    input. This keeps the wrong-family bait-and-switch (typing
    ``deepseek-v4-27b`` and being told ``deepseek-r1-32b``) from happening,
    and also prevents legitimate single-segment HuggingFace IDs like
    ``gpt2`` or ``bert-base-uncased`` from spuriously matching.

    Behaviour:
    - Multi-segment family (``fam`` contains ``-``): match aliases that
      start with ``fam-`` or are exactly ``fam``.
    - Single-segment family (``hermes``, ``qwen3.5``, ``gpt2``): match
      aliases whose name starts with ``fam``. Requires ``fam`` to be at
      least 3 chars to avoid one- or two-letter false positives.
    - No same-family match: return ``[]`` (let the loader's 404 handle it).
    """
    fam = _family_prefix(name)
    if not fam:
        return []
    aliases = list(_load().keys())
    if "-" in fam:
        same_fam = [a for a in aliases if a.startswith(fam + "-") or a == fam]
    else:
        if len(fam) < 3:
            return []
        same_fam = [a for a in aliases if a.startswith(fam)]
    if not same_fam:
        return []
    return difflib.get_close_matches(name, same_fam, n=n, cutoff=cutoff)
