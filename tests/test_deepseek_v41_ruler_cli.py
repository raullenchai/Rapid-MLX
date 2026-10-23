"""Deterministic CPU CLI contracts; public inputs are local pinned fixtures."""

import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import deepseek_v41_public_ruler as ruler


def test_fetch_checks_cached_and_downloaded_bytes(tmp_path, monkeypatch):
    data = b"public fixture"
    monkeypatch.setitem(
        ruler.SOURCE_HASHES, "fixture", hashlib.sha256(data).hexdigest()
    )
    calls = []

    def download(args, **kwargs):
        calls.append(args)
        assert kwargs == dict(check=True, capture_output=True)
        return SimpleNamespace(stdout=data)

    monkeypatch.setattr(ruler.subprocess, "run", download)
    p = ruler.fetch(tmp_path, "fixture", "https://example.test/data")
    assert p.read_bytes() == data
    assert ruler.fetch(tmp_path, "fixture", "unused") == p and len(calls) == 1
    p.write_bytes(b"bad")
    with pytest.raises(ValueError, match="raw source hash"):
        ruler.fetch(tmp_path, "fixture", "unused")
    p.unlink()
    monkeypatch.setitem(ruler.SOURCE_HASHES, "fixture", "bad")
    with pytest.raises(ValueError, match="source changed"):
        ruler.fetch(tmp_path, "fixture", "https://example.test/data")
    assert not p.exists()


def build_inputs(tmp_path, monkeypatch, failure=None):
    import tokenizers

    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    monkeypatch.setattr(ruler, "TOKENIZER_HASH", ruler.file_hash(tokenizer))

    class Tokenizer:
        def encode(self, text):
            size = 10 if failure == "short_wiki" and text == "wiki" else 26400
            if "book" in text:
                size = 10 if failure == "short_book" else 17000
            offset = 0 if text == "wiki" else 50000
            return SimpleNamespace(ids=list(range(offset, offset + size)))

        def get_vocab_size(self, **kw):
            return 200000

    monkeypatch.setattr(tokenizers.Tokenizer, "from_file", lambda _: Tokenizer())
    pq = SimpleNamespace(
        read_table=lambda _: {"text": SimpleNamespace(to_pylist=lambda: ["wiki"])}
    )
    monkeypatch.setitem(sys.modules, "pyarrow", SimpleNamespace(parquet=pq))
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", pq)

    def fetch(root, name, url):
        p = root / name
        p.write_text(
            "book"
            if failure == "markers"
            else "*** START OF BOOK ***\nbook\n*** END OF BOOK ***"
        )
        return p

    monkeypatch.setattr(ruler, "fetch", fetch)
    ids = np.arange(40960, dtype=np.int32) + (0 if failure == "overlap" else 100000)
    np.save(tmp_path / "cal.npy", ids)
    calibration = tmp_path / "cal.json"
    calibration.write_text(json.dumps({"slices": {"cal": ["cal.npy"]}}))
    if failure == "tokenizer":
        monkeypatch.setattr(ruler, "TOKENIZER_HASH", "bad")
    return SimpleNamespace(
        output=tmp_path / "output", tokenizer=tokenizer, calibration=calibration
    )


def test_build_public_rulers_records_provenance_and_disjointness(tmp_path, monkeypatch):
    a = build_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ruler",
            "build",
            "--tokenizer",
            str(a.tokenizer),
            "--calibration",
            str(a.calibration),
            "--output",
            str(a.output),
        ],
    )
    ruler.main()
    provenance = json.loads((a.output / "provenance.json").read_text())
    assert len(provenance["sources"]) == 3
    assert len(provenance["slices"]) == 26
    assert sum(x["tokens"] for x in provenance["slices"].values()) == 60400
    assert (
        json.loads((a.output / "overlap.json").read_text())["shared_windows_total"] == 0
    )
    assert len(json.loads((a.output / "wiki1100.json").read_text())["slices"]) == 24


@pytest.mark.parametrize(
    "failure,match",
    [
        ("tokenizer", "pinned source tokenizer"),
        ("short_wiki", "insufficient WikiText"),
        ("markers", "body markers"),
        ("short_book", "book too short"),
        ("overlap", "calibration overlap"),
    ],
)
def test_build_rejects_unqualified_inputs(tmp_path, monkeypatch, failure, match):
    a = build_inputs(tmp_path, monkeypatch, failure)
    with pytest.raises(ValueError, match=match):
        ruler.build(a)
    assert not (a.output / "wiki1100.json").exists()


@pytest.mark.parametrize("candidate", [False, True])
def test_verify_cli_selects_exact_full_support(tmp_path, monkeypatch, candidate):
    logits = tmp_path / "logits.npy"
    np.save(logits, np.random.default_rng(8).normal(size=(11, 4)).astype(np.float32))
    output = tmp_path / "report.json"
    argv = [
        "ruler",
        "verify",
        "--logits",
        str(logits),
        "--ks",
        "4",
        "--output",
        str(output),
    ]
    if candidate:
        argv += ["--candidate", str(logits)]
    monkeypatch.setattr(sys, "argv", argv)
    ruler.main()
    report = json.loads(output.read_text())
    assert report["chosen_k"] == 4 and report["trials"][0]["top1_exact"]
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((2, 3)))
    with pytest.raises(ValueError, match="shape mismatch"):
        ruler.verify(
            SimpleNamespace(
                logits=logits, candidate=bad, ks=[4], sigma=0.25, output=output
            )
        )
    with pytest.raises(ValueError, match="no tested k"):
        ruler.verify(
            SimpleNamespace(
                logits=logits, candidate=None, ks=[], sigma=0.25, output=output
            )
        )


def test_token_and_compact_input_rejections(tmp_path):
    tok = SimpleNamespace(
        get_vocab_size=lambda **kw: 3, encode=lambda _: SimpleNamespace(ids=[0])
    )
    p = tmp_path / "tokens.npy"
    for ids in [np.zeros((2, 2)), np.array([-1]), np.array([3])]:
        np.save(p, ids)
        with pytest.raises(ValueError):
            ruler.slice_tokens([p], tok, 1)
    with pytest.raises(ValueError, match="exactly one"):
        ruler.slice_tokens([p, p], tok, 1)
    t = tmp_path / "text.txt"
    t.write_text("hello")
    with pytest.raises(ValueError, match="need"):
        ruler.slice_tokens([t], tok, 2)
    for bad in [np.array([1]), np.array([[np.nan]])]:
        with pytest.raises(ValueError, match="finite"):
            ruler.log_softmax(bad)
    with pytest.raises(ValueError, match="compact k"):
        ruler.compact_logits([[1, 2]], 0)
    compact = ruler.compact_logits([[1, 2]], 1)
    compact["ids"][0, 0] = 3
    with pytest.raises(ValueError, match="support"):
        ruler.compact_metrics(compact, [[1, 2]])
    compact = ruler.compact_logits([[1, 2]], 1)
    compact["tail_mass"][0] = np.nan
    with pytest.raises(ValueError, match="probabilities"):
        ruler.compact_metrics(compact, [[1, 2]])


@pytest.mark.parametrize(
    "script",
    [
        "deepseek_v41_public_ruler",
        "evaluate_deepseek_v41_rd",
        "deepseek_v41_streaming_teacher",
        "verify_deepseek_v41_streaming_teacher",
    ],
)
def test_cli_help_exercises_entrypoint(monkeypatch, script):
    monkeypatch.setattr(sys, "argv", [script, "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(Path("scripts") / (script + ".py")), run_name="__main__")
    assert exc.value.code == 0


def test_compact_rejects_invalid_argmax_and_full_support_tail():
    ref = ruler.compact_logits([[1.0, 2.0]], 1)
    ref["argmax"] = np.array([-1])
    with pytest.raises(ValueError, match="argmax"):
        ruler.compact_metrics(ref, [[1.0, 2.0]])
    ref = ruler.compact_logits([[1.0, 2.0]], 2)
    ref["tail_mass"][:] = 1e-7
    with pytest.raises(ValueError, match="probabilities"):
        ruler.compact_metrics(ref, [[1.0, 2.0]])
    tied = np.tile([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], (3, 1))
    compact = ruler.compact_logits(tied, 2)
    assert np.all(np.any(compact["ids"] == 1, axis=1))
    kl, flips = ruler.compact_metrics(compact, tied)
    assert np.max(kl) < 1e-7 and not flips.any()


def test_build_refuses_existing_destination(tmp_path, monkeypatch):
    a = build_inputs(tmp_path, monkeypatch)
    a.output.mkdir()
    old = a.output / "wiki1100.json"
    old.write_text("preserve")
    with pytest.raises(FileExistsError):
        ruler.build(a)
    assert old.read_text() == "preserve"
