#!/usr/bin/env python3
"""Reproducible public rulers and CPU compact-reference math (no model execution)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.deepseek_v41_rd_common import file_hash

MAX_SOURCE_BYTES = 64 * 1024 * 1024

WIKI_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"
COMPACT_FIELDS = ("ids", "logprobs", "tail_mass", "logsumexp", "argmax")
SOURCE_HASHES = {
    "wikitext2-test.parquet": "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91",
    "pg1342.txt": "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75",
    "pg23950.txt": "ed422e10c16c19af7ca7b09ca0320012e60776b3d7cb054769b8f7d9a19aea7d",
}
TOKENIZER_HASH = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"


PUBLIC_RULERS = {
    "wiki1100": {f"wiki1100_{i:02d}": 1100 for i in range(24)},
    "long17k": {"long17k_en": 17000, "long17k_zh": 17000},
}


def read_slice_manifest(path):
    raw = path.read_bytes()
    manifest = json.loads(raw)
    specs = manifest.get("slices") if isinstance(manifest, dict) else None
    if not isinstance(specs, dict) or not specs:
        raise ValueError("each slice needs at least one local text file")
    root = path.parent.resolve()
    slices = {}
    for name, files in specs.items():
        if not isinstance(files, list) or not files:
            raise ValueError("each slice needs at least one local text file")
        resolved = []
        for filename in files:
            if (
                not isinstance(filename, str)
                or not filename
                or Path(filename).is_absolute()
            ):
                raise ValueError("slice paths must be nonempty relative paths")
            candidate = (root / filename).resolve()
            if not candidate.is_relative_to(root):
                raise ValueError("slice path escapes manifest directory")
            resolved.append(candidate)
        slices[name] = resolved
    return raw, manifest, slices


def verify_input_hashes(hashes):
    if any(file_hash(path) != expected for path, expected in hashes.items()):
        raise ValueError("slice inputs changed during inference")


def reconstruct_public_slices(root, name):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    if name == "wiki1100":
        import pyarrow.parquet as pq

        rows = pq.read_table(root / "raw/wikitext2-test.parquet")["text"].to_pylist()
        ids = tokenizer.encode("\n\n".join(rows)).ids
        return {
            f"wiki1100_{i:02d}": np.asarray(ids[i * 1100 : (i + 1) * 1100], np.int32)
            for i in range(24)
        }
    result = {}
    for book, language in ((1342, "en"), (23950, "zh")):
        text = (
            (root / "raw" / f"pg{book}.txt")
            .read_bytes()
            .decode("utf-8-sig")
            .replace("\r\n", "\n")
        )
        start = re.search(r"\*\*\* START OF .*?\*\*\*[^\n]*\n", text)
        end = re.search(r"\*\*\* END OF .*?\*\*\*", text)
        if not start or not end or start.end() >= end.start():
            raise ValueError("public ruler source body markers missing")
        result[f"long17k_{language}"] = np.asarray(
            tokenizer.encode(text[start.end() : end.start()]).ids[:17000], np.int32
        )
    return result


def validate_ruler_manifest(path):
    """Authenticate named public rulers against their retained build provenance."""
    manifest = json.loads(path.read_bytes())
    name = manifest.get("ruler")
    if name not in PUBLIC_RULERS:
        return None
    root = path.parent
    provenance_path = root / "provenance.json"
    if file_hash(provenance_path) != manifest.get("provenance_sha256"):
        raise ValueError("public ruler provenance hash mismatch")
    provenance = json.loads(provenance_path.read_bytes())
    if file_hash(root / "tokenizer.json") != TOKENIZER_HASH:
        raise ValueError("public ruler tokenizer file mismatch")
    if (
        manifest.get("tokenizer_sha256") != TOKENIZER_HASH
        or provenance.get("tokenizer_sha256") != TOKENIZER_HASH
    ):
        raise ValueError("public ruler tokenizer mismatch")
    audit_path = root / "overlap.json"
    audit_hash = file_hash(audit_path)
    audit = json.loads(audit_path.read_bytes())
    expected_slices = {n for entries in PUBLIC_RULERS.values() for n in entries}
    calibration_hash = audit.get("calibration_manifest_sha256")
    if (
        audit_hash != manifest.get("overlap_sha256")
        or audit_hash != provenance.get("overlap_sha256")
        or calibration_hash != manifest.get("calibration_manifest_sha256")
        or calibration_hash != provenance.get("calibration_manifest_sha256")
        or not isinstance(calibration_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", calibration_hash) is None
        or type(audit.get("shared_windows_total")) is not int
        or audit["shared_windows_total"] != 0
        or audit.get("shared_windows_by_slice") != {n: 0 for n in expected_slices}
        or audit.get("window_tokens") != 64
        or audit.get("stride") != 1
        or not audit.get("calibration_files")
        or audit.get("calibration_unique_windows", 0) < 1
    ):
        raise ValueError("public ruler overlap qualification mismatch")
    if set(manifest.get("slices", {})) != set(PUBLIC_RULERS[name]):
        raise ValueError("public ruler slice coverage mismatch")
    sources = {s.get("file"): s.get("sha256") for s in provenance.get("sources", [])}
    expected_sources = {"raw/" + n: digest for n, digest in SOURCE_HASHES.items()}
    if sources != expected_sources or any(
        file_hash(root / n) != digest for n, digest in sources.items()
    ):
        raise ValueError("public ruler source provenance mismatch")
    reconstructed = reconstruct_public_slices(root, name)
    for slice_name, count in PUBLIC_RULERS[name].items():
        filename = slice_name + ".npy"
        entry = provenance.get("slices", {}).get(slice_name, {})
        if manifest["slices"][slice_name] != [filename] or file_hash(
            root / filename
        ) != entry.get("sha256"):
            raise ValueError("public ruler authoritative token hash mismatch")
        ids = np.load(root / filename, allow_pickle=False)
        if (
            ids.shape != (count,)
            or ids.dtype != np.dtype("int32")
            or np.any(ids < 0)
            or entry.get("tokens") != count
        ):
            raise ValueError("public ruler token shape mismatch")
        if not np.array_equal(ids, reconstructed[slice_name]):
            raise ValueError("public ruler tokens differ from pinned source recipe")
    return file_hash(path)


def retain_ruler_manifest(path, cache, expected_digest):
    if expected_digest is None:
        return
    if validate_ruler_manifest(path) != expected_digest:
        raise ValueError("public ruler manifest changed during collection")
    manifest = json.loads(path.read_bytes())
    destination = cache / "ruler"
    destination.mkdir()
    (destination / "raw").mkdir()
    for filename in [
        "provenance.json",
        "overlap.json",
        "tokenizer.json",
        *["raw/" + n for n in SOURCE_HASHES],
        *[n + ".npy" for n in manifest["slices"]],
    ]:
        shutil.copyfile(path.parent / filename, destination / filename)
    shutil.copyfile(path, destination / "manifest.json")
    if validate_ruler_manifest(destination / "manifest.json") != expected_digest:
        raise ValueError("public ruler snapshot mismatch")


def slice_tokens(paths, tokenizer, count):
    """NPY slices are authoritative token IDs; text retains the existing recipe."""
    if any(p.suffix == ".npy" for p in paths):
        if len(paths) != 1 or paths[0].suffix != ".npy":
            raise ValueError("token slices need exactly one NPY file")
        ids = np.load(paths[0], allow_pickle=False)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0):
            raise ValueError("invalid token slice")
        if len(ids) != count:
            raise ValueError(f"token slice has {len(ids)} IDs, requested {count}")
        if np.any(ids >= tokenizer.get_vocab_size(with_added_tokens=True)):
            raise ValueError("token ID outside tokenizer vocabulary")
    else:
        ids = tokenizer.encode("\n\n".join(p.read_text() for p in paths)).ids
        if len(ids) < count:
            raise ValueError(f"slice has {len(ids)} tokens; need {count}")
    return np.asarray(ids[:count], np.int32)


def log_softmax(x):
    x = np.asarray(x, np.float64)
    if x.ndim != 2 or 0 in x.shape or not np.isfinite(x).all():
        raise ValueError("logits must be finite [tokens, vocabulary]")
    maximum = x.max(-1, keepdims=True)
    lse = maximum + np.log(np.exp(x - maximum).sum(-1, keepdims=True))
    return x - lse, lse[:, 0]


def compact_logits(logits, k):
    lp, lse = log_softmax(logits)
    if not 1 <= k <= lp.shape[1]:
        raise ValueError("invalid compact k")
    ids = np.argpartition(-lp, k - 1, axis=-1)[:, :k]
    # A tied top-1 may fall outside argpartition's arbitrary tie selection.
    argmax = np.argmax(logits, -1)
    missing = ~np.any(ids == argmax[:, None], axis=1)
    ids[missing, -1] = argmax[missing]
    selected = np.take_along_axis(lp, ids, axis=-1)
    order = np.argsort(-selected, axis=-1, kind="stable")
    ids = np.take_along_axis(ids, order, axis=-1)
    selected = np.take_along_axis(lp, ids, axis=-1)
    # Sum the tail directly, avoiding catastrophic cancellation near unit mass.
    prob = np.exp(lp)
    np.put_along_axis(prob, ids, 0.0, axis=-1)
    return dict(
        ids=ids.astype(np.int32),
        logprobs=selected.astype(np.float32),
        tail_mass=prob.sum(-1).astype(np.float32),
        logsumexp=lse.astype(np.float32),
        argmax=np.argmax(logits, -1).astype(np.int32),
    )


def compact_metrics(ref, candidate):
    """KL after coarsening the complement of reference top-k to one bucket."""
    q, _ = log_softmax(candidate)
    ids, p_log = ref["ids"], np.asarray(ref["logprobs"], np.float64)
    if (
        ids.ndim != 2
        or ids.dtype.kind not in "iu"
        or ids.shape[1] == 0
        or ids.shape != p_log.shape
        or len(ids) != len(q)
        or np.any(ids < 0)
        or np.any(ids >= q.shape[1])
        or np.any(np.diff(np.sort(ids, axis=-1), axis=-1) == 0)
    ):
        raise ValueError("invalid compact support")
    pt = np.asarray(ref["tail_mass"], np.float64)
    if (
        pt.shape != (len(q),)
        or not np.isfinite(p_log).all()
        or not np.isfinite(pt).all()
        or np.any((pt < 0) | (pt > 1))
        or np.any(p_log > 0)
        or (ids.shape[1] == q.shape[1] and np.any(pt != 0))
        or not np.allclose(np.exp(p_log).sum(-1) + pt, 1, atol=2e-6)
    ):
        raise ValueError("invalid compact probabilities")
    argmax = np.asarray(ref["argmax"])
    if (
        argmax.shape != (len(q),)
        or argmax.dtype.kind not in "iu"
        or np.any(argmax < 0)
        or np.any(argmax >= q.shape[1])
        or not np.all(np.any(ids == argmax[:, None], axis=1))
    ):
        raise ValueError("invalid compact argmax")
    top = (np.exp(p_log) * (p_log - np.take_along_axis(q, ids, axis=-1))).sum(-1)
    # Direct logsumexp over tail logits remains stable for extremely small tails.
    tail_logits = np.array(candidate, dtype=np.float64, copy=True)
    np.put_along_axis(tail_logits, ids, -np.inf, axis=-1)
    maximum = tail_logits.max(-1)
    active = pt > 0
    tail = np.zeros(len(q), np.float64)
    if active.any():
        _, q_lse = log_softmax(candidate)
        log_qt = (
            maximum[active]
            + np.log(np.exp(tail_logits[active] - maximum[active, None]).sum(-1))
            - q_lse[active]
        )
        tail[active] = pt[active] * (np.log(pt[active]) - log_qt)
    return np.maximum(0, top + tail), ref["argmax"] != np.argmax(candidate, -1)


def compact_arrays(root, name, start=None, end=None):
    return {
        f: np.load(root / f"{name}-{f}.npy", mmap_mode="r", allow_pickle=False)[
            start:end
        ]
        for f in COMPACT_FIELDS
    }


def create_compact(root, name, tokens, vocab, k):
    return {
        f: np.lib.format.open_memmap(
            root / f"{name}-{f}.npy",
            mode="w+",
            dtype=np.int32 if f in ("ids", "argmax") else np.float32,
            shape=(tokens, k) if f in ("ids", "logprobs") else (tokens,),
        )
        for f in COMPACT_FIELDS
    }


def compact_hashes(root, name):
    return {f: file_hash(root / f"{name}-{f}.npy") for f in COMPACT_FIELDS}


def windows(ids, size=64):
    raw = np.asarray(ids, dtype="<u4").tobytes()
    return {
        hashlib.sha256(raw[i * 4 : (i + size) * 4]).digest()
        for i in range(len(ids) - size + 1)
    }


def fetch(root, name, url):
    p = root / name
    if not p.exists():
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{name}.", suffix=".partial", dir=root
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            # Bound both transfer size and duration; never buffer the response in RAM.
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--max-time",
                    "120",
                    "--max-filesize",
                    str(MAX_SOURCE_BYTES),
                    "--output",
                    str(temporary),
                    url,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=125,
            )
            if temporary.stat().st_size > MAX_SOURCE_BYTES:
                raise ValueError("raw source exceeds size limit")
            if file_hash(temporary) != SOURCE_HASHES[name]:
                raise ValueError(f"source changed: {url}; restore the pinned raw bytes")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.link(temporary, p)
        finally:
            temporary.unlink(missing_ok=True)
    if file_hash(p) != SOURCE_HASHES[name]:
        raise ValueError(f"raw source hash mismatch: {name}")
    return p


def build(a):
    """Use a new destination; failures retain inputs/audits for diagnosis.

    A directory is qualified only when its ruler manifests exist. After a failure,
    inspect the retained overlap audit/raw inputs, then choose a fresh destination.
    """
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    root = a.output
    if root.exists():
        raise FileExistsError("ruler destination must not exist")
    root.mkdir(parents=True)
    raw = root / "raw"
    raw.mkdir(exist_ok=True)
    tok = Tokenizer.from_file(str(a.tokenizer))
    tokenizer_hash = file_hash(a.tokenizer)
    if tokenizer_hash != TOKENIZER_HASH:
        raise ValueError("rulers require the pinned source tokenizer")
    shutil.copyfile(a.tokenizer, root / "tokenizer.json")
    if file_hash(root / "tokenizer.json") != tokenizer_hash:
        raise ValueError("tokenizer changed during ruler construction")
    wiki_url = (
        f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{WIKI_REV}/"
        "wikitext-2-raw-v1/test-00000-of-00001.parquet"
    )
    wiki = fetch(raw, "wikitext2-test.parquet", wiki_url)
    rows = pq.read_table(wiki)["text"].to_pylist()
    # Standard evaluation concatenation: double-newline join of test text rows,
    # including empty rows, with no strip/normalization or article deduplication.
    text = "\n\n".join(rows)
    wiki_ids = tok.encode(text).ids
    sources = [
        dict(
            dataset="Salesforce/wikitext",
            config="wikitext-2-raw-v1",
            split="test",
            revision=WIKI_REV,
            url=wiki_url,
            license="CC-BY-SA-3.0",
            license_url="https://huggingface.co/datasets/Salesforce/wikitext/blob/"
            + WIKI_REV
            + "/README.md",
            file="raw/" + wiki.name,
            sha256=file_hash(wiki),
            rows=len(rows),
            joined_utf8_sha256=hashlib.sha256(text.encode()).hexdigest(),
            total_tokens=len(wiki_ids),
        )
    ]
    manifests = {"wiki1100": dict(slices={}), "long17k": dict(slices={})}
    for ruler in manifests:
        manifests[ruler].update(
            tokenizer_sha256=tokenizer_hash,
            ruler=ruler,
            token_format="int32 NPY; authoritative, no retokenization",
        )
    spans = {}

    def save(ruler, name, ids, source, start):
        np.save(root / f"{name}.npy", np.asarray(ids, np.int32))
        manifests[ruler]["slices"][name] = [f"{name}.npy"]
        spans[name] = dict(
            source=source,
            token_start=start,
            token_end=start + len(ids),
            tokens=len(ids),
            sha256=file_hash(root / f"{name}.npy"),
        )

    if len(wiki_ids) < 26400:
        raise ValueError("insufficient WikiText tokens")
    for i in range(24):
        save(
            "wiki1100",
            f"wiki1100_{i:02d}",
            wiki_ids[i * 1100 : (i + 1) * 1100],
            0,
            i * 1100,
        )
    for book, lang, title in [
        (1342, "en", "Pride and Prejudice"),
        (23950, "zh", "三國志演義"),
    ]:
        url = f"https://www.gutenberg.org/ebooks/{book}.txt.utf-8"
        p = fetch(raw, f"pg{book}.txt", url)
        original = p.read_bytes().decode("utf-8-sig").replace("\r\n", "\n")
        start = re.search(r"\*\*\* START OF .*?\*\*\*[^\n]*\n", original)
        end = re.search(r"\*\*\* END OF .*?\*\*\*", original)
        if not start or not end or start.end() >= end.start():
            raise ValueError("Gutenberg body markers missing")
        body = original[start.end() : end.start()]
        ids = tok.encode(body).ids
        if len(ids) < 17000:
            raise ValueError("book too short")
        sources.append(
            dict(
                title=title,
                language=lang,
                url=url,
                license="Public domain in the USA; Gutenberg license retained in raw file",
                license_url=f"https://www.gutenberg.org/ebooks/{book}",
                file="raw/" + p.name,
                sha256=file_hash(p),
                body_utf8_sha256=hashlib.sha256(body.encode()).hexdigest(),
                body_char_start=start.end(),
                body_char_end=end.start(),
                total_tokens=len(ids),
            )
        )
        save("long17k", f"long17k_{lang}", ids[:17000], len(sources) - 1, 0)
    cal_windows = set()
    cal_files = {}
    calibration_raw, calibration_manifest, calibration_slices = read_slice_manifest(
        a.calibration
    )
    calibration_manifest_hash = hashlib.sha256(calibration_raw).hexdigest()
    if calibration_manifest.get("tokenizer_sha256") != TOKENIZER_HASH:
        raise ValueError("calibration tokenizer identity mismatch")
    for name, paths in calibration_slices.items():
        cal_files[name] = [file_hash(p) for p in paths]
        ids = slice_tokens(paths, tok, 40960)
        cal_windows.update(windows(ids))
    overlaps = {}
    for manifest in manifests.values():
        for name, paths in manifest["slices"].items():
            overlaps[name] = len(windows(np.load(root / paths[0])) & cal_windows)
    audit = dict(
        window_tokens=64,
        stride=1,
        token_hash="SHA256(little-endian uint32 IDs)",
        calibration_manifest_sha256=calibration_manifest_hash,
        calibration_files=cal_files,
        calibration_unique_windows=len(cal_windows),
        shared_windows_by_slice=overlaps,
        shared_windows_total=sum(overlaps.values()),
        scope=f"every scored slice vs all {len(cal_files)} full 40960-token calibration exports, including joins",
    )
    provenance = dict(
        schema_version=1,
        tokenizer_sha256=tokenizer_hash,
        model_revision="1bae0e1088993b9b82de08f59de5be5a34e42170",
        sources=sources,
        slices=spans,
        recipe=dict(
            wiki1100="double-newline join ALL test rows including empty rows in parquet order; tokenizer.encode with default special-token processing; first 24 consecutive nonoverlapping 1100-token spans, discard remainder",
            long17k="UTF-8-sig decode; CRLF to LF; remove Gutenberg header/footer at markers; first 17000 body tokens, default tokenizer special-token processing; one uninterrupted context per book",
        ),
    )
    (root / "overlap.json").write_text(json.dumps(audit, indent=2) + "\n")
    provenance.update(
        overlap_sha256=file_hash(root / "overlap.json"),
        calibration_manifest_sha256=calibration_manifest_hash,
    )
    (root / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    if audit["shared_windows_total"]:
        raise ValueError("calibration overlap; audit saved, rulers NOT qualified")
    if file_hash(a.calibration) != calibration_manifest_hash or any(
        [file_hash(p) for p in paths] != cal_files[name]
        for name, paths in calibration_slices.items()
    ):
        raise ValueError("calibration inputs changed during overlap audit")
    prepared = []
    for ruler, manifest in manifests.items():
        manifest["provenance_sha256"] = file_hash(root / "provenance.json")
        manifest["overlap_sha256"] = provenance["overlap_sha256"]
        manifest["calibration_manifest_sha256"] = calibration_manifest_hash
        temporary = root / f".{ruler}.json"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        validate_ruler_manifest(temporary)
        prepared.append((temporary, root / f"{ruler}.json"))
    for temporary, final in prepared:
        os.link(temporary, final)
        temporary.unlink()
    print(
        json.dumps(
            dict(
                tokens=60400,
                overlap=audit["shared_windows_total"],
                calibration_unique_windows=len(cal_windows),
                output=str(root),
            )
        )
    )


def verify(a):
    from scripts.evaluate_deepseek_v41_rd import token_metrics

    r = np.load(a.logits, mmap_mode="r", allow_pickle=False)
    candidate = (
        np.load(a.candidate, mmap_mode="r", allow_pickle=False) if a.candidate else None
    )
    for values in (r, candidate):
        if values is None:
            continue
        if values.ndim != 2 or 0 in values.shape or values.dtype.kind not in "fiu":
            raise ValueError("logits must be nonempty real [tokens, vocabulary] arrays")
        for start in range(0, len(values), 8):
            if not np.isfinite(values[start : start + 8]).all():
                raise ValueError("logits must be finite")
    if candidate is not None and candidate.shape != r.shape:
        raise ValueError("candidate shape mismatch")
    rng = np.random.default_rng(1017)
    totals = {k: [] for k in a.ks}
    full, exact = [], {k: True for k in a.ks}
    for start in range(0, len(r), 8):
        rr = r[start : start + 8]
        # Seeded iid additive logit noise; no reordering/temperature-only shortcut.
        c = (
            candidate[start : start + 8]
            if candidate is not None
            else (rr.astype(np.float64) + rng.normal(0, a.sigma, rr.shape)).astype(
                np.float32
            )
        )
        kl, flip = token_metrics(rr, c)
        full.extend(kl.tolist())
        for k in a.ks:
            compact = compact_logits(rr, k)
            ck, cf = compact_metrics(compact, c)
            totals[k].extend(ck.tolist())
            exact[k] &= bool(np.array_equal(flip, cf))
    report = dict(
        reference=str(a.logits),
        reference_sha256=file_hash(a.logits),
        candidate=str(a.candidate)
        if a.candidate
        else "synthetic iid Gaussian additive logits",
        candidate_sha256=file_hash(a.candidate) if a.candidate else None,
        seed=1017,
        sigma=a.sigma,
        tokens=len(r),
        full_mean_kl=float(np.mean(full)),
        trials=[],
    )
    for k in a.ks:
        error = np.asarray(full) - totals[k]
        report["trials"].append(
            dict(
                k=k,
                compact_mean_kl=float(np.mean(totals[k])),
                mean_absolute_error=float(np.mean(np.abs(error))),
                max_position_error=float(np.max(np.abs(error))),
                top1_exact=exact[k],
            )
        )
    passed = [
        x["k"]
        for x in report["trials"]
        if x["mean_absolute_error"] < 1e-3 and x["top1_exact"]
    ]
    report["chosen_k"] = min(passed) if passed else None
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise ValueError("no tested k meets error tolerance")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--tokenizer", type=Path, required=True)
    b.add_argument("--calibration", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--logits", type=Path, required=True)
    v.add_argument("--candidate", type=Path)
    v.add_argument("--sigma", type=float, default=0.25)
    v.add_argument(
        "--ks", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048, 4096]
    )
    v.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    (build if a.command == "build" else verify)(a)


if __name__ == "__main__":
    main()
