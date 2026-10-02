# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx import``: preflight, keyed cache, atomic + cancel-safe publish.

MLX never runs here: the conversion/smoke workers and the Hub download are
injected or mocked, and the worker module is driven with a fake ``mlx_lm``.
A real end-to-end run (Qwen/Qwen3-0.6B bf16 → 4-bit, served by name, Ctrl-C
mid-conversion) is recorded in the PR description.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest

from rapid_mlx import cli
from rapid_mlx.byom import _import_worker as worker
from rapid_mlx.byom import imports as im
from rapid_mlx.byom import preflight as pf

GIB = 1 << 30
SUPPORTED = frozenset({"qwen3"})


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("RAPID_MLX_HOME", str(tmp_path / "rmlx"))
    return tmp_path / "rmlx"


def _insp(**kw) -> pf.Inspection:
    fields = {
        "ref": "o/My-FT-bf16",
        "is_local": False,
        "files": ("model.safetensors", "config.json"),
        "weight_bytes": 2 * GIB,
        "config": {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
        "dtype": "bf16",
        "revision": "sha1",
    }
    fields.update(kw)
    return pf.Inspection(**fields)


def _plan(insp=None, *, bits=4, name=None, hub_cached=False):
    return im.plan_import(
        (insp or _insp()).ref,
        bits=bits,
        name=name,
        inspection=insp or _insp(),
        supported=SUPPORTED,
        hub_cached=hub_cached,
    )


def _publish(home: Path, name: str, key: str = "k", source: str = "o/r") -> Path:
    path = home / "imports" / name
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"x" * 10)
    (path / im.MANIFEST).write_text(
        json.dumps({"key": key, "source": source, "bits": 4})
    )
    return path


# ------------------------------------------------------------------ naming


def test_imports_root_defaults_to_home(monkeypatch, tmp_path):
    monkeypatch.delenv("RAPID_MLX_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert im.imports_root() == tmp_path / ".rapid-mlx" / "imports"


def test_imported_model_path(home):
    assert im.imported_model_path("nope") is None
    path = _publish(home, "my-ft-4bit")
    assert im.imported_model_path("my-ft-4bit") == str(path)
    for bad in ("a/b", "..x", "", 3):
        assert im.imported_model_path(bad) is None


@pytest.mark.parametrize(
    "source,expected",
    [
        ("o/My-FT-bf16", "my-ft-4bit"),
        ("./my-ft", "my-ft-4bit"),
        ("o/Weird Name!!-hf", "weird-name-4bit"),
        ("o/---", "model-4bit"),
    ],
)
def test_default_name(source, expected):
    assert im._default_name(source, 4) == expected


def test_default_name_never_shadows_an_alias():
    plan = _plan(_insp(ref="Qwen/Qwen3.5-4B-MLX"), bits=4)
    assert not im._name_is_taken(plan.name)
    assert im._name_is_taken("qwen3.5-4b-4bit")
    taken = _plan(_insp(ref="o/qwen3.5-4b"), bits=4)
    assert taken.name == "qwen3.5-4b-4bit-local"
    assert im._name_is_taken("kokoro")


def test_explicit_name_rules():
    assert _plan(name="mine").name == "mine"
    with pytest.raises(im.ImportRefusedError, match="not a valid import name"):
        _plan(name="bad/name")
    with pytest.raises(im.ImportRefusedError, match="already a model alias"):
        _plan(name="qwen3.5-4b-4bit")


# ---------------------------------------------------------------- planning


@pytest.mark.parametrize(
    "insp,message",
    [
        (_insp(files=("a.gguf",)), "no safetensors"),
        (
            _insp(config={"model_type": "qwen3", "quantization": {"bits": 4}}),
            "already quantized",
        ),
        (
            _insp(
                config={
                    "model_type": "qwen3",
                    "quantization_config": {"quant_method": "awq"},
                }
            ),
            "already quantized",
        ),
        (_insp(config={}), "no config.json model_type"),
        (
            _insp(config={"model_type": "spark_x", "architectures": ["SForCausalLM"]}),
            "no mlx-lm converter",
        ),
        (
            _insp(config={"model_type": "odd", "architectures": ["OddModel"]}),
            "no mlx-lm",
        ),
        (_insp(config={"model_type": "qwen3", "model_file": "m.py"}), "own model code"),
        (
            _insp(config={"model_type": "qwen3", "auto_map": {"x": "y"}}),
            "own model code",
        ),
        (_insp(revision=None), "current revision"),
        (_insp(weight_bytes=0), "no model\\*.safetensors"),
    ],
)
def test_plan_refusals(insp, message):
    with pytest.raises(im.ImportRefusedError, match=message):
        _plan(insp)


def test_plan_sizes_and_key(monkeypatch):
    monkeypatch.setattr(pf, "_mlx_lm_version", lambda: "0.31.3")
    plan = _plan()
    assert plan.output_bytes == int(GIB * (4 + 0.5) / 8)
    assert plan.download_bytes == 2 * GIB
    assert plan.dtype == "bf16"
    assert _plan(hub_cached=True).download_bytes == 0
    assert _plan(bits=8).key != plan.key
    f32 = _plan(_insp(dtype="float32"))
    assert f32.output_bytes == int(GIB / 2 * 4.5 / 8)
    assert _plan(_insp(dtype=None)).dtype == "bf16"
    monkeypatch.setattr(pf, "_mlx_lm_version", lambda: "0.32.0")
    assert _plan().key != plan.key


def test_local_plan_uses_a_content_fingerprint(tmp_path):
    src = tmp_path / "ft"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"x" * 8)
    (src / "sub").mkdir()
    insp = _insp(ref=str(src), is_local=True)
    first = _plan(insp)
    assert first.revision.startswith("local:")
    assert first.download_bytes == 0
    (src / "model.safetensors").write_bytes(b"y" * 9)
    assert _plan(insp).key != first.key


# --------------------------------------------------------------- resources


@pytest.fixture
def disk(monkeypatch, home):
    state = {"free": {}, "dev": {}}

    def _free(path):
        return state["free"].get(
            "cache" if "hfcache" in str(path) else "out", 100 * GIB
        )

    monkeypatch.setattr(im, "_free_bytes", _free)
    monkeypatch.setattr(im, "_hf_cache_dir", lambda: Path("/hfcache"))
    monkeypatch.setattr(im, "_same_filesystem", lambda a, b: state.get("same", True))
    return state


def test_check_resources_plan_lines(disk):
    lines = im.check_resources(_plan(), 64 * GIB)
    assert lines[0] == "Source   bf16 · 2.0 GB · qwen3 (supported)"
    assert lines[1].startswith("Needs    ~2.8 GB free disk")
    assert lines[2] == "Output   ~0.6 GB, fits in 64 GB ✓"
    assert im.check_resources(_plan(), None)[-1] == "Output   ~0.6 GB"


def test_check_resources_disk_refusals(disk):
    disk["free"]["out"] = GIB
    with pytest.raises(im.ImportRefusedError, match="free disk"):
        im.check_resources(_plan(), None)
    disk["free"]["out"] = 100 * GIB
    disk["same"] = False
    disk["free"]["cache"] = GIB
    with pytest.raises(im.ImportRefusedError, match="Hugging Face cache"):
        im.check_resources(_plan(), None)
    disk["free"]["cache"] = 100 * GIB
    assert im.check_resources(_plan(), None)[1].startswith("Needs    ~0.6 GB")


def test_check_resources_memory_refusal(disk):
    big = _plan(_insp(weight_bytes=60 * GIB))
    with pytest.raises(im.ImportRefusedError, match="enough memory"):
        im.check_resources(big, 16 * GIB)


def test_free_bytes_and_filesystem_probe_walk_up(tmp_path):
    missing = tmp_path / "a" / "b"
    assert im._free_bytes(missing) > 0
    assert im._same_filesystem(missing, tmp_path)


def test_hf_cache_dir():
    from huggingface_hub.constants import HF_HUB_CACHE

    assert im._hf_cache_dir() == Path(HF_HUB_CACHE)


# --------------------------------------------------------------- execution


def _fake_worker(calls, *, fail_on=None, interrupt_on=None):
    def run(step, *argv):
        calls.append((step, *argv))
        if step == interrupt_on:
            raise KeyboardInterrupt
        if step == fail_on:
            raise im.ImportRefusedError(f"{step} step failed: boom")
        if step == "convert":
            out = Path(argv[1])
            out.mkdir()
            (out / "model.safetensors").write_bytes(b"q")

    return run


def _execute(plan, calls, **kw):
    return im.execute(
        plan,
        force=kw.pop("force", False),
        spinner_factory=lambda label: nullcontext(),
        run_worker=_fake_worker(calls, **kw),
        download=lambda p: "/hf/snapshot",
    )


def test_execute_publishes_atomically_and_reuses(home):
    calls = []
    plan = _plan()
    final, reused = _execute(plan, calls)
    assert not reused
    assert final == home / "imports" / plan.name
    manifest = json.loads((final / im.MANIFEST).read_text())
    assert manifest["key"] == plan.key and manifest["bits"] == 4
    assert [c[0] for c in calls] == ["convert", "smoke"]
    assert calls[0][1] == "/hf/snapshot"
    assert not list((home / "imports").glob(".tmp-*"))
    # Same key: no work at all.
    calls.clear()
    assert _execute(plan, calls) == (final, True)
    assert calls == []


def test_execute_uses_a_local_source_in_place(home, tmp_path):
    calls = []
    plan = _plan(_insp(ref=str(tmp_path), is_local=True))
    _execute(plan, calls)
    assert calls[0][1] == str(tmp_path)


def test_execute_refuses_a_different_import_under_the_same_name(home):
    _publish(home, "my-ft-4bit", key="other")
    with pytest.raises(im.ImportRefusedError, match="already exists"):
        _execute(_plan(), [])
    final, _ = _execute(_plan(), [], force=True)
    assert json.loads((final / im.MANIFEST).read_text())["key"] == _plan().key
    assert not (final / "replaced").exists()


@pytest.mark.parametrize("failure", ["fail_on", "interrupt_on"])
@pytest.mark.parametrize("step", ["convert", "smoke"])
def test_failures_and_cancel_leave_the_cache_untouched(home, failure, step):
    old = _publish(home, "my-ft-4bit", key="old")
    with pytest.raises((im.ImportRefusedError, KeyboardInterrupt)):
        _execute(_plan(), [], force=True, **{failure: step})
    assert json.loads((old / im.MANIFEST).read_text())["key"] == "old"
    assert not list((home / "imports").glob(".tmp-*"))


def test_stale_temp_dirs_are_reclaimed_under_the_lock(home):
    stale = home / "imports" / ".tmp-my-ft-4bit-abandoned"
    stale.mkdir(parents=True)
    _execute(_plan(), [])
    assert not stale.exists()


def test_lock_excludes_a_concurrent_import(home):
    lock_path = home / "imports" / ".locks" / "x.lock"
    outer = im._Lock(lock_path)
    outer.__enter__()
    with pytest.raises(im.ImportRefusedError, match="running"), im._Lock(lock_path):
        pass
    outer.__exit__(None, None, None)
    with im._Lock(lock_path):
        pass


def test_existing_key_tolerates_garbage(home):
    path = home / "imports" / "g"
    path.mkdir(parents=True)
    (path / im.MANIFEST).write_text("{not json")
    assert im._existing_key(path) is None
    (path / im.MANIFEST).write_text("[1]")
    assert im._existing_key(path) is None


def test_run_worker_success_and_failure(monkeypatch):
    seen = {}

    def _run(argv, **kw):
        seen["argv"] = argv
        seen["timeout"] = kw["timeout"]
        return types.SimpleNamespace(returncode=seen.get("rc", 0), stderr="line1\nboom")

    monkeypatch.setattr(subprocess, "run", _run)
    im._run_worker("smoke", "/out")
    assert seen["argv"][:3] == [sys.executable, "-c", im.WORKER_BOOTSTRAP]
    assert seen["argv"][3:] == ["smoke", "/out"]
    seen["rc"] = 1
    with pytest.raises(
        im.ImportRefusedError, match="smoke step failed:\n    line1\n    boom"
    ):
        im._run_worker("smoke", "/out")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kw: types.SimpleNamespace(returncode=1, stderr=""),
    )
    with pytest.raises(im.ImportRefusedError, match="no output"):
        im._run_worker("convert", "a", "b", "4", "64")


def test_download_source_fetches_only_loadable_files(monkeypatch):
    import huggingface_hub

    seen = {}

    def _snap(repo, *, revision, allow_patterns):
        seen.update(repo=repo, revision=revision, allow=allow_patterns)
        return "/snap"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snap)
    assert im._download_source(_plan()) == "/snap"
    assert seen["repo"] == "o/My-FT-bf16" and seen["revision"] == "sha1"
    assert "model*.safetensors" in seen["allow"]
    assert "*.py" not in seen["allow"]


# ---------------------------------------------------------------- worker


def test_worker_steps_with_fake_mlx_lm(monkeypatch):
    calls = []
    fake = types.ModuleType("mlx_lm")
    fake.convert = lambda **kw: calls.append(("convert", kw))
    fake.load = lambda path: calls.append(("load", path)) or ("model", "tok")
    fake.generate = lambda model, tok, **kw: calls.append(("generate", kw)) or "Hi"
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    assert worker.main(["convert", "/src", "/out", "4", "64"]) == 0
    assert calls[0] == (
        "convert",
        {
            "hf_path": "/src",
            "mlx_path": "/out",
            "quantize": True,
            "q_bits": 4,
            "q_group_size": 64,
        },
    )
    assert worker.main(["smoke", "/out"]) == 0
    assert calls[-1][1]["max_tokens"] == 1
    fake.generate = lambda *a, **kw: None
    with pytest.raises(SystemExit, match="no text"):
        worker.main(["smoke", "/out"])
    with pytest.raises(SystemExit, match="unknown step"):
        worker.main(["nope"])


# -------------------------------------------------------------- listing/rm


def test_list_imports_skips_junk(home):
    _publish(home, "b-model")
    _publish(home, "a-model", source="./local")
    (home / "imports" / ".locks").mkdir()
    bad = home / "imports" / "bad"
    bad.mkdir()
    (bad / im.MANIFEST).write_text("[1]")
    listed = im.list_imports()
    assert [i.name for i in listed] == ["a-model", "b-model"]
    assert listed[0].source == "./local" and listed[0].bits == 4
    assert listed[0].size > 0


def test_list_imports_without_a_root(home):
    assert im.list_imports() == []


def test_print_imports_section(home, capsys):
    im.print_imports_section()
    assert capsys.readouterr().out == ""
    _publish(home, "my-ft-4bit")
    im.print_imports_section()
    out = capsys.readouterr().out
    assert "── Imported with `rapid-mlx import`" in out
    assert "  • my-ft-4bit  " in out


def test_remove_import(home, monkeypatch, capsys):
    path = _publish(home, "my-ft-4bit")
    assert im.remove_import("not-an-import", assume_yes=True) is False
    assert im.remove_import("/elsewhere/x", assume_yes=True) is False
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert im.remove_import("my-ft-4bit", assume_yes=False) is True
    assert path.exists() and "Cancelled" in capsys.readouterr().out

    def _eof(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    assert im.remove_import(str(path), assume_yes=False) is True
    assert path.exists()
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert im.remove_import(str(path), assume_yes=False) is True
    assert not path.exists()
    assert "Removed imported model my-ft-4bit" in capsys.readouterr().out


def test_import_dir_for_bad_paths(monkeypatch):
    def _boom(self, *a, **kw):
        raise OSError

    monkeypatch.setattr(Path, "resolve", _boom)
    assert im._import_dir_for("/x/y") is None


# --------------------------------------------------------- CLI wiring


def test_resolve_model_serves_an_import_by_name(home):
    from rapid_mlx.model_aliases import resolve_model

    path = _publish(home, "my-ft-4bit")
    assert resolve_model("my-ft-4bit") == str(path)
    assert resolve_model("qwen3.5-4b-4bit") != str(path)


def test_resolve_model_import_with_external_roots(home, monkeypatch, tmp_path):
    from rapid_mlx.model_aliases import resolve_model

    monkeypatch.setenv("RAPID_MLX_EXTRA_MODEL_ROOTS", str(tmp_path / "ext"))
    path = _publish(home, "my-ft-4bit")
    assert resolve_model("my-ft-4bit") == str(path)


def test_rm_command_removes_an_import(home, capsys):
    path = _publish(home, "my-ft-4bit")
    cli.rm_command(argparse.Namespace(model=str(path), yes=True))
    assert not path.exists()


def test_models_cached_lists_imports(home, monkeypatch, capsys):
    _publish(home, "my-ft-4bit")
    monkeypatch.setattr(cli, "_print_cached_models", lambda: print("TABLE"))
    monkeypatch.setattr(
        cli, "print_staleness_warning_if_any", lambda: None, raising=False
    )
    import rapid_mlx._version_check as vc

    monkeypatch.setattr(vc, "print_staleness_warning_if_any", lambda: None)
    cli.models_command(argparse.Namespace(cached=True, json=False))
    out = capsys.readouterr().out
    assert out.index("TABLE") < out.index("my-ft-4bit")


def test_parser_import_defaults():
    ns = cli.build_parser().parse_args(["import", "o/r"])
    assert (ns.source, ns.quantize, ns.name, ns.force) == ("o/r", 4, None, False)
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["import", "o/r", "--quantize", "5"])


@pytest.fixture
def run_import(monkeypatch, disk):
    monkeypatch.setattr(im, "convertible_model_types", lambda: SUPPORTED)
    monkeypatch.setattr(pf, "physical_ram_bytes", lambda: 64 * GIB)
    monkeypatch.setattr(im, "revision_cached", lambda repo, insp: False)
    state = {"insp": _insp(), "execute": None}
    monkeypatch.setattr(pf, "inspect_hub", lambda ref: state["insp"])
    monkeypatch.setattr(pf, "inspect_local", lambda ref: state["insp"])

    def _execute(plan, *, force, spinner_factory):
        if isinstance(state["execute"], BaseException):
            raise state["execute"]
        return Path("/imports") / plan.name, state["execute"] == "reused"

    monkeypatch.setattr(im, "execute", _execute)

    def run(source="o/My-FT-bf16", **kw):
        args = argparse.Namespace(
            source=source, quantize=kw.get("bits", 4), name=None, force=False
        )
        im.import_command(args, spinner_factory=lambda label: nullcontext())

    run.state = state
    return run


def test_import_command_happy_path(run_import, capsys):
    run_import()
    out = capsys.readouterr().out
    assert "  Source   bf16 · 2.0 GB · qwen3 (supported)" in out
    assert "  ✓ Smoke test passed\n  rapid-mlx serve my-ft-4bit" in out
    run_import.state["execute"] = "reused"
    run_import()
    assert "✓ Already imported" in capsys.readouterr().out


def test_import_command_local_source(run_import, tmp_path, capsys):
    run_import.state["insp"] = _insp(ref=str(tmp_path), is_local=True)
    run_import(str(tmp_path))
    assert "rapid-mlx serve" in capsys.readouterr().out


def test_import_command_errors(run_import, capsys):
    with pytest.raises(SystemExit) as exc:
        run_import("not-a-repo")
    assert exc.value.code == 2
    run_import.state["insp"] = None
    with pytest.raises(SystemExit) as exc:
        run_import()
    assert exc.value.code == 1
    assert "Could not read" in capsys.readouterr().err
    run_import.state["insp"] = _insp()
    run_import.state["execute"] = KeyboardInterrupt()
    with pytest.raises(SystemExit) as exc:
        run_import()
    assert exc.value.code == 130
    assert "import cache is unchanged" in capsys.readouterr().err


def test_main_dispatches_import(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        im, "import_command", lambda args, spinner_factory: seen.update(src=args.source)
    )
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "--no-telemetry", "import", "o/r"])
    cli.main()
    assert seen == {"src": "o/r"}


def test_run_worker_bounds_only_the_smoke_test(monkeypatch):
    timeouts = []

    def _run(argv, *, timeout, **kw):
        timeouts.append(timeout)
        if argv[3] == "smoke":
            raise subprocess.TimeoutExpired(argv, timeout)
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subprocess, "run", _run)
    im._run_worker("convert", "a", "b", "4", "64")
    with pytest.raises(im.ImportRefusedError, match="did not finish"):
        im._run_worker("smoke", "/out")
    assert timeouts == [None, im.SMOKE_TIMEOUT_SECONDS]


def test_local_fingerprint_is_recursive_and_content_aware(tmp_path):
    src = tmp_path / "ft"
    (src / "tok").mkdir(parents=True)
    (src / "tok" / "tokenizer.json").write_text("a")
    first = im._local_revision(src)
    (src / "tok" / "tokenizer.json").write_text("b")
    assert im._local_revision(src) != first


def test_large_files_are_fingerprinted_by_metadata_only(tmp_path, monkeypatch):
    src = tmp_path / "ft"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"x" * 10)
    monkeypatch.setattr(im, "_HASHED_FILE_MAX_BYTES", 4)
    reads = []
    real = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda self: reads.append(self) or real(self)
    )
    im._local_revision(src)
    assert reads == []


def test_convertible_model_types_use_mlx_lm_only(monkeypatch):
    monkeypatch.setattr(
        pf, "_installed_types", lambda pkg, dirs: {"llama"} if pkg == "mlx_lm" else None
    )
    assert im.convertible_model_types() == frozenset({"llama"})
    monkeypatch.setattr(pf, "_installed_types", lambda pkg, dirs: None)
    assert im.convertible_model_types() is None


def test_revision_cached_checks_the_exact_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "_hf_cache_dir", lambda: tmp_path)
    insp = _insp(files=("model-1.safetensors", "model-2.safetensors", "config.json"))
    snap = tmp_path / "models--o--My-FT-bf16" / "snapshots" / "sha1"
    snap.mkdir(parents=True)
    (snap / "model-1.safetensors").write_text("x")
    assert im.revision_cached("o/My-FT-bf16", insp) is False
    (snap / "model-2.safetensors").write_text("x")
    assert im.revision_cached("o/My-FT-bf16", insp) is True
    assert im.revision_cached("o/My-FT-bf16", _insp(revision=None)) is False
    assert im.revision_cached("o/My-FT-bf16", _insp(files=("config.json",))) is False


def test_force_replacement_crash_is_recovered(home):
    plan = _plan()
    backup = home / "imports" / f".old-{plan.name}-123-456"
    backup.mkdir(parents=True)
    (backup / im.MANIFEST).write_text(json.dumps({"key": plan.key}))
    # A replacement died after moving the old import aside: it comes back.
    final, reused = _execute(plan, [])
    assert reused and final.exists() and not backup.exists()


def test_force_replacement_recovery_then_rebuild(home):
    plan = _plan()
    backup = home / "imports" / f".old-{plan.name}-1-2"
    backup.mkdir(parents=True)
    (backup / im.MANIFEST).write_text(json.dumps({"key": "older"}))
    older = home / "imports" / f".old-{plan.name}-0-1"
    older.mkdir()
    calls = []
    final, reused = _execute(plan, calls, force=True)
    assert not reused and [c[0] for c in calls] == ["convert", "smoke"]
    assert not backup.exists() and not older.exists()
    assert json.loads((final / im.MANIFEST).read_text())["key"] == plan.key


def test_alias_set_refuses_an_import_name(home, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("RAPID_MLX_USER_ALIASES_FILE", str(tmp_path / "ua.json"))
    _publish(home, "my-ft-4bit")
    with pytest.raises(SystemExit):
        cli.alias_command(
            argparse.Namespace(alias_action="set", name="my-ft-4bit", target="o/r")
        )
    assert "imported model" in capsys.readouterr().err
