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
        assert kwargs == dict(
            check=True,
            stdout=ruler.subprocess.DEVNULL,
            stderr=ruler.subprocess.PIPE,
            timeout=125,
        )
        assert args[args.index("--max-filesize") + 1] == str(ruler.MAX_SOURCE_BYTES)
        Path(args[args.index("--output") + 1]).write_bytes(data)
        return SimpleNamespace(returncode=0)

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
        monkeypatch.setitem(ruler.SOURCE_HASHES, name, ruler.file_hash(p))
        return p

    monkeypatch.setattr(ruler, "fetch", fetch)
    ids = np.arange(40960, dtype=np.int32) + (0 if failure == "overlap" else 100000)
    np.save(tmp_path / "cal.npy", ids)
    calibration = tmp_path / "cal.json"
    calibration.write_text(
        json.dumps(
            {"slices": {"cal": ["cal.npy"]}, "tokenizer_sha256": ruler.TOKENIZER_HASH}
        )
    )
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
    output = tmp_path / "nested" / "report.json"
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


def test_build_rejects_empty_calibration_manifest(tmp_path, monkeypatch):
    a = build_inputs(tmp_path, monkeypatch)
    a.calibration.write_text('{"slices": {}}')
    with pytest.raises(ValueError, match="each slice needs"):
        ruler.build(a)
    assert not (a.output / "wiki1100.json").exists()
    assert not (a.output / "long17k.json").exists()


def test_verify_absolute_errors_do_not_cancel(tmp_path, monkeypatch):
    from scripts import evaluate_deepseek_v41_rd as evaluator

    logits = tmp_path / "logits.npy"
    np.save(logits, np.zeros((2, 2), np.float32))
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        evaluator,
        "token_metrics",
        lambda *args: (np.array([0.1, 0.1]), np.array([False, False])),
    )
    monkeypatch.setattr(
        ruler,
        "compact_metrics",
        lambda *args: (np.array([0.2, 0.0]), np.array([False, False])),
    )
    with pytest.raises(ValueError, match="no tested k"):
        ruler.verify(
            SimpleNamespace(
                logits=logits, candidate=None, ks=[1], sigma=0.25, output=output
            )
        )
    trial = json.loads(output.read_text())["trials"][0]
    assert trial["mean_absolute_error"] == pytest.approx(0.1)
    assert trial["max_position_error"] == pytest.approx(0.1)


@pytest.mark.parametrize("manifest", [False, True])
def test_build_rechecks_calibration_inputs_before_qualification(
    tmp_path, monkeypatch, manifest
):
    a = build_inputs(tmp_path, monkeypatch)
    original = ruler.slice_tokens

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if manifest:
            a.calibration.write_bytes(a.calibration.read_bytes() + b" ")
        else:
            path = tmp_path / "cal.npy"
            values = np.load(path)
            np.save(path, values + 1)
        return result

    monkeypatch.setattr(ruler, "slice_tokens", mutate)
    with pytest.raises(ValueError, match="calibration inputs changed"):
        ruler.build(a)
    assert not (a.output / "wiki1100.json").exists()
    assert not (a.output / "long17k.json").exists()


@pytest.mark.parametrize(
    "bad",
    [
        np.empty((0, 2)),
        np.empty((2, 0)),
        np.array([1.0, 2.0]),
        np.array([[np.nan, 0.0]]),
        np.array([[np.inf, 0.0]]),
        np.array([[1j, 0j]]),
    ],
)
@pytest.mark.parametrize("candidate", [False, True])
def test_verify_rejects_empty_or_nonfinite_arrays(tmp_path, bad, candidate):
    reference = tmp_path / "reference.npy"
    other = tmp_path / "candidate.npy"
    np.save(reference, np.zeros((1, 2)) if candidate else bad)
    np.save(other, bad)
    output = tmp_path / "report.json"
    with pytest.raises(ValueError, match="logits must be"):
        ruler.verify(
            SimpleNamespace(
                logits=reference,
                candidate=other if candidate else None,
                ks=[1],
                sigma=0.25,
                output=output,
            )
        )
    assert not output.exists()


@pytest.mark.parametrize("failure", ["oversize", "transfer", "collision"])
def test_fetch_bounds_and_cleans_failed_transfers(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(ruler, "MAX_SOURCE_BYTES", 4)
    monkeypatch.setitem(
        ruler.SOURCE_HASHES, "fixture", hashlib.sha256(b"safe").hexdigest()
    )

    def transfer(argv, **kwargs):
        temporary = Path(argv[argv.index("--output") + 1])
        temporary.write_bytes(b"too large" if failure == "oversize" else b"safe")
        if failure == "transfer":
            raise ruler.subprocess.CalledProcessError(63, argv)
        if failure == "collision":
            (tmp_path / "fixture").write_bytes(b"preserve")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ruler.subprocess, "run", transfer)
    with pytest.raises(
        (ValueError, ruler.subprocess.CalledProcessError, FileExistsError)
    ):
        ruler.fetch(tmp_path, "fixture", "https://example.test/data")
    assert not list(tmp_path.glob("*.partial"))
    if failure == "collision":
        assert (tmp_path / "fixture").read_bytes() == b"preserve"
    else:
        assert not (tmp_path / "fixture").exists()


@pytest.mark.parametrize("identity", [None, "wrong"])
def test_calibration_tokenizer_identity_is_required(tmp_path, monkeypatch, identity):
    a = build_inputs(tmp_path, monkeypatch)
    manifest = json.loads(a.calibration.read_text())
    manifest["tokenizer_sha256"] = identity
    a.calibration.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="calibration tokenizer identity"):
        ruler.build(a)
    assert not (a.output / "long17k.json").exists()


def test_ruler_tokenizer_copy_must_match_pin(tmp_path, monkeypatch):
    a = build_inputs(tmp_path, monkeypatch)
    original = ruler.shutil.copyfile

    def mutate(source, destination):
        result = original(source, destination)
        destination.write_bytes(destination.read_bytes() + b" ")
        return result

    monkeypatch.setattr(ruler.shutil, "copyfile", mutate)
    with pytest.raises(ValueError, match="tokenizer changed"):
        ruler.build(a)
    assert not (a.output / "wiki1100.json").exists()


@pytest.mark.parametrize(
    "bad",
    ["absolute", "parent", "symlink", "string", "nonstring", "empty", "empty_list"],
)
def test_slice_manifest_cannot_escape_its_directory(tmp_path, bad):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private fixture")
    (root / "link.txt").symlink_to(outside)
    files = {
        "absolute": [str(outside)],
        "parent": ["../outside.txt"],
        "symlink": ["link.txt"],
        "string": "outside.txt",
        "nonstring": [7],
        "empty": [""],
        "empty_list": [],
    }[bad]
    manifest = root / "slices.json"
    manifest.write_text(json.dumps({"slices": {"a": files}}))
    with pytest.raises(ValueError, match="slice"):
        ruler.read_slice_manifest(manifest)


def test_relative_nested_slice_manifest_is_supported(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    text = nested / "slice.txt"
    text.write_text("fixture")
    manifest = tmp_path / "slices.json"
    raw = json.dumps({"slices": {"a": ["nested/slice.txt"]}}).encode()
    manifest.write_bytes(raw)
    retained, parsed, slices = ruler.read_slice_manifest(manifest)
    assert retained == raw and parsed == json.loads(raw)
    assert slices == {"a": [text.resolve()]}
