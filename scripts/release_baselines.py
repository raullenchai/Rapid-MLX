#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit committed performance baselines used by pr_validate."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jsonschema
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "scripts" / "pr_validate" / "golden_models.yaml"
BASELINE_DIR = REPO_ROOT / "harness" / "baselines"
SCHEMA_PATH = REPO_ROOT / "harness" / "baseline.schema.json"


@dataclass(frozen=True)
class Candidate:
    family: str
    model_id: str

    @property
    def filename(self) -> str:
        return f"bench-{self.model_id.replace('/', '--')}.json"


def load_candidates(path: Path = REGISTRY_PATH) -> tuple[Candidate, ...]:
    data = yaml.safe_load(path.read_text())
    families = data.get("families") if isinstance(data, dict) else None
    if not isinstance(families, list) or not families:
        raise ValueError("golden model registry needs a non-empty families list")

    candidates: list[Candidate] = []
    seen: set[str] = set()
    seen_filenames: dict[str, str] = {}
    for family in families:
        if (
            not isinstance(family, dict)
            or not isinstance(family.get("family"), str)
            or not family["family"]
        ):
            raise ValueError("every registry family needs a name")
        raw_candidates = family.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError(f"{family['family']}: candidates must be non-empty")
        for raw in raw_candidates:
            model_id = raw.get("id") if isinstance(raw, dict) else None
            if not isinstance(model_id, str) or not model_id:
                raise ValueError(f"{family['family']}: candidate needs an id")
            if model_id in seen:
                raise ValueError(f"duplicate benchmark candidate {model_id!r}")
            seen.add(model_id)
            candidate = Candidate(family["family"], model_id)
            if previous := seen_filenames.get(candidate.filename):
                raise ValueError(
                    f"benchmark candidates {previous!r} and {model_id!r} "
                    f"collide at {candidate.filename!r}"
                )
            seen_filenames[candidate.filename] = model_id
            candidates.append(candidate)
    return tuple(candidates)


def latest_release() -> tuple[str, datetime] | None:
    describe = subprocess.run(
        [
            "git",
            "describe",
            "--tags",
            "--match",
            "v[0-9]*",
            "--abbrev=0",
            "HEAD",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if describe.returncode != 0 or not describe.stdout.strip():
        return None
    tag = describe.stdout.strip()
    timestamp = subprocess.run(
        [
            "git",
            "for-each-ref",
            "--format=%(taggerdate:iso-strict)%00%(creatordate:iso-strict)",
            f"refs/tags/{tag}",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if timestamp.returncode != 0 or not timestamp.stdout.strip():
        return None
    tagger_date, _, creator_date = timestamp.stdout.strip().partition("\x00")
    captured_at = tagger_date or creator_date
    if not captured_at:
        return None
    return tag, datetime.fromisoformat(captured_at)


def audit(
    *,
    registry_path: Path = REGISTRY_PATH,
    baseline_dir: Path = BASELINE_DIR,
    schema_path: Path = SCHEMA_PATH,
    release: tuple[str, datetime] | None = None,
) -> dict[str, Any]:
    candidates = load_candidates(registry_path)
    schema = json.loads(schema_path.read_text())
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    expected = {candidate.filename: candidate for candidate in candidates}
    errors: list[str] = []
    stale: list[str] = []
    warnings: list[str] = []
    covered = 0

    for filename, candidate in expected.items():
        path = baseline_dir / filename
        if not path.exists():
            errors.append(f"missing {filename} ({candidate.family})")
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{filename}: invalid JSON ({exc})")
            continue
        schema_errors = sorted(
            validator.iter_errors(payload), key=lambda error: list(error.path)
        )
        if schema_errors:
            detail = "; ".join(error.message for error in schema_errors[:3])
            errors.append(f"{filename}: schema invalid ({detail})")
            continue
        if payload["family"] != candidate.family:
            errors.append(
                f"{filename}: family {payload['family']!r} != {candidate.family!r}"
            )
            continue
        if payload["model"]["id"] != candidate.model_id:
            errors.append(
                f"{filename}: model {payload['model']['id']!r} "
                f"!= {candidate.model_id!r}"
            )
            continue
        captured = datetime.fromisoformat(payload["captured_at"].replace("Z", "+00:00"))
        if release is not None and captured < release[1]:
            stale.append(
                f"{filename}: captured {payload['captured_at']} before {release[0]}"
            )
        covered += 1

    for path in sorted(baseline_dir.glob("bench-*.json")):
        if path.name not in expected:
            errors.append(f"orphan baseline {path.name}")

    return {
        "candidate_count": len(candidates),
        "covered_count": covered,
        "errors": errors,
        "stale": stale,
        "warnings": warnings,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-stale",
        action="store_true",
        help="fail when a baseline predates the latest release",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    release = latest_release()
    report = audit(release=release)
    if release is None:
        report["warnings"].append("latest release tag unavailable; age not checked")
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(
            "baseline audit: "
            f"{report['covered_count']}/{report['candidate_count']} candidates covered"
        )
        for message in report["errors"]:
            print(f"ERROR: {message}")
        for message in report["stale"]:
            print(f"WARNING: stale {message}")
        for message in report["warnings"]:
            print(f"WARNING: {message}")
    return 1 if report["errors"] or (args.strict_stale and report["stale"]) else 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
