#!/usr/bin/env python3.12
"""One-shot rename of every alias in vllm_mlx/aliases.json to the canonical
explicit form ``<family>-<version>-<params>-<modality?>-<technique?>-<quant>``.

Drops three legacy short-form codename aliases that violate the spec
(``deepseek-v4-flash``, ``gemma4``, ``nemotron-nano``) and fixes the
``phi4-14b`` schema bug where the alias name claimed 14B but the hf_path
pointed at phi-4-mini (~4B).

Also dumps ``rename_map.json`` so the repo-wide reference sweep can
mechanically rewrite occurrences in tests, docs, scripts.

Run from repo root:

    python3.12 scripts/rename_aliases.py
"""

import json
import re
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALIASES_PATH = ROOT / "vllm_mlx" / "aliases.json"
RENAME_MAP_PATH = ROOT / "scripts" / "rename_map.json"


def detect_quant(hf: str) -> str | None:
    """Inspect an hf_path and return the canonical quant suffix.

    Order matters: longer / more specific markers first so we don't
    misclassify ``mxfp4-q8`` as ``mxfp4`` or ``-4bit-DWQ`` as ``-4bit``.
    """
    h = hf.lower()
    if "mxfp4-q8" in h or "mxfp4_q8" in h:
        return "mxfp4-q8"
    if "mxfp4" in h:
        return "mxfp4"
    if "dwq" in h:
        return "dwq"
    if "ud-mlx" in h or "-ud-" in h:
        return "ud"
    if m := re.search(r"-(\d+)bit", h):
        return f"{m.group(1)}bit"
    if "unpacked" in h:
        return "unpacked"
    if "bf16" in h:
        return "bf16"
    if "fp16" in h:
        return "fp16"
    return None


# 1. Aliases whose hf_path is unambiguous and only need the quant suffix added.
# These are determined automatically by detect_quant.

# 2. Manual overrides — aliases that need name changes beyond just adding a suffix,
#    or that need their hf_path corrected. ``drop_to`` is the equivalent
#    explicit alias the repo-wide sweep should rewrite references to.
MANUAL: dict[str, dict[str, str] | None] = {
    # Codename aliases — duplicate hf_path of an explicit entry. Drop entirely,
    # but tell the sweep where to redirect references.
    "deepseek-v4-flash": {"drop": True, "redirect_to": "deepseek-v4-flash-8bit"},
    "gemma4": {"drop": True, "redirect_to": "gemma-4-12b-qat-4bit"},
    "nemotron-nano": {"drop": True, "redirect_to": "nemotron-30b-4bit"},
    # phi4-14b: schema bug. hf_path was pointing at phi-4-mini (~4B).
    # Fix: rename to phi-4-14b-4bit AND swap hf_path to the real Phi-4 14B.
    # The 4B mini variant moves to its own new alias (added below).
    "phi4-14b": {
        "new_name": "phi-4-14b-4bit",
        "new_hf_path": "mlx-community/phi-4-4bit",
    },
}

# 3. Brand-new aliases to add at the end (preserves old phi-4-mini coverage).
NEW_ALIASES: list[tuple[str, dict]] = [
    (
        "phi-4-mini-4bit",
        {
            "hf_path": "mlx-community/phi-4-mini-instruct-4bit",
            "tool_call_parser": "hermes",
            "reasoning_parser": None,
            "is_hybrid": False,
            "is_moe": False,
            "supports_spec_decode": True,
            "suffix_decoding_tier": "unknown",
        },
    ),
]


def compute_new_name(old: str, hf: str) -> str:
    quant = detect_quant(hf)
    if quant is None:
        raise ValueError(f"alias {old!r}: cannot detect quant from {hf!r}")
    # If the old name already ends in a known quant suffix, replace it.
    stem = re.sub(
        r"-(2bit|3bit|4bit|6bit|8bit|mxfp4-q8|mxfp4|dwq|ud|unpacked|bf16|fp16)$",
        "",
        old,
    )
    return f"{stem}-{quant}"


def main() -> None:
    with open(ALIASES_PATH) as fp:
        data = json.load(fp, object_pairs_hook=OrderedDict)

    new_data: OrderedDict[str, object] = OrderedDict()
    rename_map: dict[str, str | None] = {}

    for old, profile in data.items():
        # Handle manual overrides first.
        if old in MANUAL:
            spec = MANUAL[old]
            if spec is None:
                # Drop entirely (legacy form — kept for the type checker).
                rename_map[old] = None
                continue
            if spec.get("drop"):
                # Codename alias — drop from aliases.json but tell the sweep
                # where to point references.
                rename_map[old] = spec["redirect_to"]
                continue
            new_name = spec["new_name"]
            if "new_hf_path" in spec:
                profile = OrderedDict(profile)
                profile["hf_path"] = spec["new_hf_path"]
            new_data[new_name] = profile
            rename_map[old] = new_name
            continue

        # Default path: keep the entry, rewrite the key.
        hf = profile["hf_path"] if isinstance(profile, dict) else profile
        new_name = compute_new_name(old, hf)
        if new_name in new_data:
            raise ValueError(f"rename collision: {old!r} -> {new_name!r} already used")
        new_data[new_name] = profile
        rename_map[old] = new_name

    # Append brand-new aliases (skip if already present so the script is
    # idempotent — useful when iterating on the rename rules).
    for name, profile in NEW_ALIASES:
        if name not in new_data:
            new_data[name] = profile

    # Write back.
    with open(ALIASES_PATH, "w") as fp:
        json.dump(new_data, fp, indent=2)
        fp.write("\n")

    with open(RENAME_MAP_PATH, "w") as fp:
        json.dump(rename_map, fp, indent=2, sort_keys=True)
        fp.write("\n")

    # Counters from the input data's perspective so the three lines add
    # back up to the input alias count. Each input alias is exactly one
    # of: dropped (MANUAL says ``drop=True``), renamed (name changed),
    # or kept (name unchanged because the old key already carried the
    # canonical quant suffix). Counting drops via ``rename_map[o] is None``
    # would always print 0 because dropped codename aliases store their
    # redirect target — non-None — in the rename map.
    def _is_drop(old: str) -> bool:
        spec = MANUAL.get(old)
        return isinstance(spec, dict) and bool(spec.get("drop"))

    dropped = sum(1 for old in data if _is_drop(old))
    renamed = sum(1 for old in data if not _is_drop(old) and rename_map[old] != old)
    kept = sum(1 for old in data if not _is_drop(old) and rename_map[old] == old)
    print(f"  renamed: {renamed}")
    print(f"  dropped: {dropped}")
    print(f"  kept (already explicit): {kept}")
    print(f"  new aliases added: {len(NEW_ALIASES)}")
    print(f"  total: {len(new_data)}")


if __name__ == "__main__":
    main()
