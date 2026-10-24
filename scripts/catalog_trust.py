#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline differential policy for reviewed catalog identities; no model downloads."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

CATALOGS = frozenset({"rapid_mlx/aliases.json", "rapid_mlx/audio/aliases.json"})
COSMETIC_FIELDS = frozenset({"notes", "description", "display_name"})
_CONTROL_PATHS = (
    "scripts/check_vendored_provenance.py",
    "scripts/vendor_ltx25.py",
    "ltx_core_mlx/",
    "ltx_pipelines_mlx/",
    "rapid_mlx/kernels/affine8_attention.py",
    "rapid_mlx/kernels/affine8_attention.metal",
    "scripts/vendored-provenance.json",
    "rapid_mlx/models/mlx_vlm_vendored/",
    "scripts/catalog_trust.py",
    "scripts/catalog_trust/",
    "scripts/merge_on_green/",
    "scripts/merge_on_green.py",
    "scripts/__init__.py",
    "sitecustomize.py",
    "usercustomize.py",
    "scripts/pr_validate/",
    ".github/workflows/",
    "rapid_mlx/catalog/",
    "proto/model-catalog/",
    "rapid_mlx/model_aliases.py",
    "rapid_mlx/model_catalog.py",
    "rapid_mlx/model_catalog_lock.json",
    "scripts/lock_model_catalog.py",
    "rapid_mlx/model_downloads.py",
    "rapid_mlx/utils/model_file_guard.py",
    "rapid_mlx/utils/torch_checkpoint.py",
    "rapid_mlx/utils/tokenizer.py",
    "rapid_mlx/utils/kimi_tokenizer/",
    "rapid_mlx/models/kimi_vlm.py",
    "rapid_mlx/audio/registry.py",
    "rapid_mlx/audio/runtime_catalog.py",
    "rapid_mlx/audio/runtime_requirements.py",
    "rapid_mlx/audio/tts.py",
    "rapid_mlx/audio/stt.py",
    "rapid_mlx/audio/vad_artifacts.py",
    "rapid_mlx/audio/processor.py",
    "rapid_mlx/audio/sam_security.py",
    "rapid_mlx/audio/f5_security.py",
    "rapid_mlx/audio/voxcpm_security.py",
    "rapid_mlx/server_exposure.py",
    "rapid_mlx/model_integrity.py",
    "rapid_mlx/server.py",
    "rapid_mlx/cli.py",
    "rapid_mlx/audio/sa3/models/defs/checkpoint_security.py",
    "rapid_mlx/model_load_errors.py",
    "rapid_mlx/_env.py",
    ".github/CODEOWNERS",
    ".mergify.yml",
    "pyproject.toml",
    "install.sh",
)
SHA = re.compile(r"[0-9a-f]{40}\Z")


def is_trust_control(path: str) -> bool:
    return any(
        path == prefix or (prefix.endswith("/") and path.startswith(prefix))
        for prefix in _CONTROL_PATHS
    )


def parse_catalog(text: str) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate catalog key: {key}")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("Catalog must be a JSON object")
    return value


def _semantic(entry):
    if isinstance(entry, dict):
        return {
            key: value for key, value in entry.items() if key not in COSMETIC_FIELDS
        }
    return entry


def catalog_changes(
    before: dict, after: dict, path: str
) -> tuple[list[str], list[str]]:
    """Return semantic aliases and invalid new/changed text checkpoint identities.

    Existing unpinned entries are grandfathered, not silently treated as reviewed.
    Pins identify proposed content; runtime pin consumption needs separate review.
    """
    changed, errors = [], []
    # Repo-id reverse lookup selects the first matching alias. Reordering
    # existing entries can change effective behavior without any value diff.
    old_order = [key for key in before if key in after and key != "_comment"]
    new_order = [key for key in after if key in before and key != "_comment"]
    if old_order != new_order:
        changed.append("<alias-order>")
    for alias in sorted(before.keys() | after.keys()):
        if alias == "_comment":
            continue
        old, new = before.get(alias), after.get(alias)
        if alias in before and alias in after and _semantic(old) == _semantic(new):
            continue
        changed.append(alias)
        if alias == "_runtime_assets" and alias in after:
            if not isinstance(new, dict):
                errors.append("Runtime assets must be an object of asset lists")
                continue
            for family, assets in new.items():
                if not isinstance(assets, list):
                    errors.append(f"{alias}.{family}: runtime assets must be a list")
                    continue
                old_assets = old.get(family, []) if isinstance(old, dict) else []
                if not isinstance(old_assets, list):
                    old_assets = []
                for asset in assets:
                    if (
                        not isinstance(asset, dict)
                        or not isinstance(asset.get("repo_id"), str)
                        or not asset["repo_id"].strip()
                    ):
                        errors.append(
                            f"{alias}.{family}: runtime asset requires a repository"
                        )
                    elif asset in old_assets:
                        continue
                    pin = asset.get("revision") if isinstance(asset, dict) else None
                    if not isinstance(pin, str) or not SHA.fullmatch(pin):
                        errors.append(
                            f"{alias}.{family}: changed runtime asset requires a full 40-hex revision"
                        )
        if alias not in after or alias.startswith("_"):
            continue
        source_field = "hf_id" if path == "rapid_mlx/audio/aliases.json" else "hf_path"
        old_source = old.get(source_field) if isinstance(old, dict) else old
        new_source = new.get(source_field) if isinstance(new, dict) else new
        if not isinstance(new_source, str) or not new_source.strip():
            errors.append(f"{alias}: catalog source must be a nonempty repository")
        old_pin = old.get("revision") if isinstance(old, dict) else None
        new_pin = new.get("revision") if isinstance(new, dict) else None
        identity_changed = (
            old is None
            or old_source != new_source
            or old_pin != new_pin
            or (
                isinstance(old, dict)
                and isinstance(new, dict)
                and old.get("subfolder") != new.get("subfolder")
            )
        )
        if identity_changed and (
            not isinstance(new_pin, str) or not SHA.fullmatch(new_pin)
        ):
            errors.append(
                f"{alias}: new/changed source requires a reviewed full 40-hex revision"
            )
    return changed, errors


def read_git_catalog(root: Path, ref: str, path: str) -> dict:
    if not SHA.fullmatch(ref):
        raise ValueError("Catalog audit requires an exact base/head commit SHA")
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_catalog(result.stdout)


def audit_worktree(
    root: Path, base: str, paths: list[str]
) -> tuple[list[str], list[str]]:
    changed, errors = [], []
    for path in sorted(set(paths) & CATALOGS):
        try:
            before = read_git_catalog(root, base, path)
            target = root / path
            if target.is_symlink():
                raise ValueError("Catalog must be a regular file, not a symlink")
            after = parse_catalog(target.read_text()) if target.exists() else {}
            aliases, invalid = catalog_changes(before, after, path)
            changed.extend(f"{path}:{alias}" for alias in aliases)
            errors.extend(invalid)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            errors.append(f"Cannot establish catalog trust for {path}: {exc}")
    return changed, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base")
    parser.add_argument("--report-unpinned", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    if args.report_unpinned:
        catalog = parse_catalog((root / "rapid_mlx/aliases.json").read_text())
        print(
            json.dumps(
                {
                    alias: entry.get("hf_path") if isinstance(entry, dict) else entry
                    for alias, entry in catalog.items()
                    if alias != "_comment"
                    and not (isinstance(entry, dict) and entry.get("revision"))
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.base:
        parser.error("--base or --report-unpinned is required")
    changed, errors = audit_worktree(root, args.base, list(CATALOGS))
    print(json.dumps({"human_review_required": changed, "errors": errors}, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
