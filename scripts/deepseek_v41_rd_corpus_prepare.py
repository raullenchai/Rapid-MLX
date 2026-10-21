#!/usr/bin/env python3
"""Validate owner-prepared local corpus exports and provenance. No network."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def prepare(manifest, calibration, heldout):
    specs = {s["name"]: s for s in manifest["slices"]}
    cal, held = calibration["slices"], heldout["slices"]
    expected = {n for n, s in specs.items() if not s["heldout_only"]}
    if set(cal) != expected or set(held) != set(specs):
        raise ValueError(
            "exports must cover exactly the manifest calibration/held-out slices"
        )
    seen = set()
    for split in (cal, held):
        for name, files in split.items():
            if not files:
                raise ValueError(f"empty slice {name}")
            if (
                sum(Path(file).stat().st_size for file in files)
                > manifest["max_export_bytes_per_slice"]
            ):
                raise ValueError("export slice exceeds byte cap")
            for file in files:
                path = Path(file)
                if (
                    not path.is_absolute()
                    or path.stat().st_size > manifest["max_export_bytes_per_slice"]
                ):
                    raise ValueError("exports require bounded absolute local paths")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen:
                    raise ValueError("duplicate file content across slices/splits")
                seen.add(digest)


def validate_provenance(manifest, calibration, heldout, provenance, tokenizer_path):
    """Verify the exported text and source-document split contract offline."""
    from tokenizers import Tokenizer

    def digest(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    if digest(tokenizer_path) != provenance["tokenizer_sha256"]:
        raise ValueError("source tokenizer hash mismatch")
    if provenance["tokenizer_sha256"] != manifest["tokenizer"]["sha256"]:
        raise ValueError("tokenizer differs from manifest pin")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    specs = {s["name"]: s for s in manifest["slices"]}
    records = {(r["split"], r["name"]): r for r in provenance["slices"]}
    sources_by_slice = {
        name: source for source in manifest["sources"] for name in source["slices"]
    }
    expected = {
        (split, name)
        for split, export in (("calibration", calibration), ("heldout", heldout))
        for name in export["slices"]
    }
    if len(records) != len(provenance["slices"]) or set(records) != expected:
        raise ValueError("provenance must cover every export exactly once")
    groups, documents, text_hashes = {}, set(), set()
    totals = {"calibration": 0, "heldout": 0}
    raw_paths = {r["file"] for r in provenance["raw_files"]}
    for split, export in (("calibration", calibration), ("heldout", heldout)):
        for name, files in export["slices"].items():
            record = records[split, name]
            if files != [record["file"]] or digest(files[0]) != record["sha256"]:
                raise ValueError(f"export hash/path mismatch: {split}/{name}")
            text = Path(files[0]).read_text(encoding="utf-8")
            count = len(tokenizer.encode(text).ids)
            if count != record["tokens"] or count != specs[name][split + "_tokens"]:
                raise ValueError(f"token budget mismatch: {split}/{name}: {count}")
            if record["role"] != specs[name]["role"] or not record["documents"]:
                raise ValueError("missing documents or incorrect slice role")
            pieces = []
            position = 0
            for doc in record["documents"]:
                revision = doc["revision"]
                if not re.fullmatch(r"[0-9a-f]{40}|[0-9]+", revision):
                    raise ValueError(
                        "source revision must be a commit or article revision"
                    )
                if not doc["license"] or not doc["row_id"] or not doc["source_split"]:
                    raise ValueError("missing source license, row ID or split")
                if doc["dataset"] not in specs[name]["resolved_datasets"]:
                    raise ValueError("source dataset differs from slice manifest")
                if doc["license"] not in sources_by_slice[name]["approved_licenses"]:
                    raise ValueError(
                        f"source license is not approved for dataset {doc['dataset']}"
                    )
                pin = manifest["pinned_sources"].get(doc["dataset"])
                if pin is not None and revision != pin:
                    raise ValueError("source commit differs from manifest pin")
                identity = (doc["dataset"], doc["document_id"])
                group = doc["group_id"]
                if identity in documents or doc["text_sha256"] in text_hashes:
                    raise ValueError("duplicate source document/text")
                if group in groups and groups[group] != split:
                    raise ValueError("source task/article/translation crosses splits")
                documents.add(identity)
                text_hashes.add(doc["text_sha256"])
                groups[group] = split
                if digest(doc["rendered_file"]) != doc["text_sha256"]:
                    raise ValueError("rendered document hash mismatch")
                if (
                    doc["source_file"] not in raw_paths
                    or doc["rendered_file"] not in raw_paths
                ):
                    raise ValueError("source/document missing from raw hash inventory")
                full = Path(doc["rendered_file"]).read_text(encoding="utf-8")
                selected = doc["selected_char_count"]
                if selected <= 0 or selected > len(full):
                    raise ValueError("invalid document selection range")
                if pieces:
                    position += 2
                if doc["export_char_start"] != position:
                    raise ValueError("invalid document start")
                position += selected
                if doc["export_char_end"] != position:
                    raise ValueError("invalid document end")
                pieces.append(full[:selected])
            terminator = record.get("export_terminator", "")
            if terminator not in ("", "\n", "\n\n"):
                raise ValueError("invalid export terminator")
            if "\n\n".join(pieces) + terminator != text:
                raise ValueError("export cannot be reconstructed from source documents")
            totals[split] += count
    for raw in provenance["raw_files"]:
        if digest(raw["file"]) != raw["sha256"]:
            raise ValueError(f"raw source hash mismatch: {raw['file']}")
    raw_bytes = sum(Path(r["file"]).stat().st_size for r in provenance["raw_files"])
    if raw_bytes > manifest["max_total_download_bytes"]:
        raise ValueError("raw sources exceed download budget")
    return {
        "slices": {
            "calibration": len(calibration["slices"]),
            "heldout": len(heldout["slices"]),
        },
        "tokens": totals,
        "documents": len(documents),
        "raw_bytes": raw_bytes,
        "document_overlap": 0,
        "duplicate_document_texts": 0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("deepseek_v41_rd_corpus_manifest.json"),
    )
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--provenance", type=Path)
    p.add_argument("--tokenizer", type=Path)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    calibration = json.loads(a.calibration.read_text())
    heldout = json.loads(a.heldout.read_text())
    prepare(
        manifest,
        calibration,
        heldout,
    )
    provenance_path = a.provenance or a.calibration.with_name("provenance.json")
    provenance = json.loads(provenance_path.read_text())
    summary = validate_provenance(
        manifest,
        calibration,
        heldout,
        provenance,
        a.tokenizer or Path(provenance["tokenizer_file"]),
    )
    print(json.dumps({"validation": "passed", **summary}, indent=2))


if __name__ == "__main__":
    main()
