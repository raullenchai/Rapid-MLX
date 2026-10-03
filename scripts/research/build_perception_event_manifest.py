#!/usr/bin/env python3
"""Build a role-preserving manifest from Perception Test action annotations.

This utility copies official temporal boundaries verbatim.  It does not infer
first-visible times, split compound actions, or change an existing video role.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


PICKUP_PUTDOWN_LABELS = {
    "Lifting something and placing it back down": "compound_pickup_putdown",
    "Putting something on top of something": "putdown_or_placement",
    "Putting something into something": "put_into",
    "Taking something out of something": "take_out",
    "Dropping something on top of something": "drop_on",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_segment(segment: dict, video: dict) -> None:
    required = {"id", "label_id", "label", "parent_objects", "timestamps", "frame_ids"}
    if set(segment) != required:
        raise ValueError(f"unexpected segment fields: {set(segment)!r}")
    frames = segment["frame_ids"]
    times = segment["timestamps"]
    if len(frames) != 2 or len(times) != 2 or frames[0] > frames[1] or times[0] > times[1]:
        raise ValueError(f"invalid boundaries in {video['metadata']['video_id']}:{segment['id']}")
    if frames[0] < 0 or frames[1] >= video["metadata"]["num_frames"]:
        raise ValueError(f"frame out of range in {video['metadata']['video_id']}:{segment['id']}")


def build(annotations: dict, prior_cases: list, opened_cases: list, video_dirs: list[Path]) -> dict:
    roles: dict[str, str] = {}
    for case in prior_cases:
        video_id, role = case["id"], case["split"]
        if role not in {"train", "dev", "sealed"}:
            raise ValueError(f"unexpected prior role {role!r}")
        roles[video_id] = {"train": "train", "dev": "dev", "sealed": "opened_prior"}[role]
    for case in opened_cases:
        video_id = case["id"]
        if video_id in roles:
            raise ValueError(f"role collision for {video_id}")
        roles[video_id] = "opened_diagnostic"

    cases = []
    role_counts: dict[str, Counter] = defaultdict(Counter)
    label_counts: dict[str, Counter] = defaultdict(Counter)
    pickup_counts: dict[str, Counter] = defaultdict(Counter)
    pickup_videos: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for video_id, role in sorted(roles.items()):
        source = annotations.get(video_id)
        if source is None:
            role_counts[role]["videos_without_action_annotations"] += 1
            continue
        segments = source["action_localisation"]
        for segment in segments:
            validate_segment(segment, source)
            label_counts[role][segment["label"]] += 1
            if segment["label"] in PICKUP_PUTDOWN_LABELS:
                bucket = PICKUP_PUTDOWN_LABELS[segment["label"]]
                pickup_counts[role][bucket] += 1
                pickup_videos[role][bucket].add(video_id)
        path = next((root / f"{video_id}.mp4" for root in video_dirs if (root / f"{video_id}.mp4").is_file()), None)
        role_counts[role]["videos_with_action_annotations"] += 1
        role_counts[role]["segments"] += len(segments)
        role_counts[role]["local_mp4_present"] += path is not None
        cases.append(
            {
                "video_id": video_id,
                "role": role,
                "local_video_path": str(path) if path is not None else None,
                "metadata": source["metadata"],
                "segments": segments,
            }
        )

    return {
        "schema_version": 1,
        "boundary_contract": {
            "source": "official Perception Test temporal action localisation train annotations",
            "timestamps_unit": "microseconds",
            "boundaries": "copied verbatim; no first-visible or derived event times",
            "parent_objects": "official object annotation IDs; object-track annotations are not included",
            "negative_windows": "not generated; annotation exhaustiveness for arbitrary windows is unproven",
            "unknown_windows": "not generated; unknown requires an explicit insufficient-evidence sampling rule",
        },
        "role_contract": {
            "train": "existing 180-video training subset; only role eligible for training",
            "dev": "existing 40-video development subset; evaluation only",
            "opened_prior": "existing 80-video opened former confirmation subset; diagnostic only",
            "opened_diagnostic": "new 60-video confirmation subset now opened; diagnostic only",
            "disjointness": "video_id only; actor, participant, scene, and source-recording disjointness unproven",
        },
        "pickup_putdown_mapping": {
            "contract": "exact-label inventory aid only; compound labels are not split into fabricated events",
            "labels": PICKUP_PUTDOWN_LABELS,
        },
        "statistics": {
            "official_annotation_videos": len(annotations),
            "official_segments": sum(len(v["action_localisation"]) for v in annotations.values()),
            "by_role": {role: dict(counts) for role, counts in sorted(role_counts.items())},
            "labels_by_role": {role: dict(sorted(counts.items())) for role, counts in sorted(label_counts.items())},
            "pickup_putdown_segments_by_role": {
                role: dict(sorted(counts.items())) for role, counts in sorted(pickup_counts.items())
            },
            "pickup_putdown_videos_by_role": {
                role: {bucket: len(ids) for bucket, ids in sorted(groups.items())}
                for role, groups in sorted(pickup_videos.items())
            },
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--prior-cases", type=Path, required=True)
    parser.add_argument("--opened-manifest", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotations = load_json(args.annotations)
    prior_cases = load_json(args.prior_cases)
    opened_cases = load_json(args.opened_manifest)["cases"]
    result = build(annotations, prior_cases, opened_cases, args.video_dir)
    result["source_files"] = {
        "annotations": {"path": str(args.annotations), "sha256": sha256(args.annotations)},
        "prior_cases": {"path": str(args.prior_cases), "sha256": sha256(args.prior_cases)},
        "opened_manifest": {"path": str(args.opened_manifest), "sha256": sha256(args.opened_manifest)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    positives = []
    review = []
    for case in result["cases"]:
        if case["role"] != "train":
            continue
        for segment in case["segments"]:
            row = {
                "video_id": case["video_id"],
                "role": "train",
                "label": segment["label"],
                "label_id": segment["label_id"],
                "segment_id": segment["id"],
                "frame_ids": segment["frame_ids"],
                "timestamps_us": segment["timestamps"],
                "parent_objects": segment["parent_objects"],
                "supervision": "official_positive_segment",
            }
            positives.append(row)
            if segment["label"] in PICKUP_PUTDOWN_LABELS:
                review.append(
                    {
                        **row,
                        "candidate_product_event": PICKUP_PUTDOWN_LABELS[segment["label"]],
                        "requires_human_review": True,
                        "review_reason": "official action may be compound or differ from product event semantics",
                    }
                )

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    write_jsonl(args.output.with_name("TRAIN-EVENT-POSITIVES.jsonl"), positives)
    write_jsonl(args.output.with_name("PICKUP-PUTDOWN-REVIEW-CANDIDATES.jsonl"), review)


if __name__ == "__main__":
    main()
