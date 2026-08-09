# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/release_check_m3_random.py — G12 release-gauntlet
random-coverage gate. The orchestrator script lives outside the
``vllm_mlx`` package so we import it via importlib."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "release_check_m3_random.py"
)


@pytest.fixture(scope="module")
def g12():
    """Load the orchestrator script as a module so its helpers can be
    unit-tested without spawning subprocesses."""
    spec = importlib.util.spec_from_file_location("g12", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _fake_aliases() -> dict[str, dict]:
    """A shrunken aliases.json fixture covering every eligibility
    branch (size in/out of band, vision marker, kimi marker, gemma-4
    marker, 8bit vs 4bit, missing hf_path, fail-closed name-without-
    size token)."""
    return {
        # In-band (6-12 B, 4-bit, no special excludes)
        "qwen3.5-9b-4bit": {"hf_path": "mlx-community/Qwen3.5-9B-4bit"},
        "qwen3-8b-4bit": {"hf_path": "mlx-community/Qwen3-8B-4bit"},
        "hermes3-8b-4bit": {"hf_path": "mlx-community/Hermes-3-Llama-3.1-8B-4bit"},
        # Excluded: the 4 B tier cannot hold an agentic loop. It does not
        # fail fast either — it retries malformed tool calls until the
        # per-profile clock runs out, so the gauntlet goes red on a healthy
        # engine (#1672).
        "qwen3-4b-instruct-2507-4bit": {
            "hf_path": "mlx-community/Qwen3-4B-Instruct-2507-4bit"
        },
        "qwen3.5-4b-4bit": {"hf_path": "mlx-community/Qwen3.5-4B-MLX-4bit"},
        # Fail-closed: the repo name carries no parameter-count token
        # (``Air`` is a variant name, ``4.5`` is the version). The old
        # parser regex extracted ``4`` from ``-4bit`` and admitted this
        # entry as a 4 B model — codex PR #693 review caught it. The
        # post-fix parser refuses to guess and skips the entry.
        "glm4.5-air-4bit": {"hf_path": "mlx-community/GLM-4.5-Air-4bit"},
        # Out-of-band: far below the floor — harnesses would false-fail.
        "qwen3-0.6b-4bit": {"hf_path": "mlx-community/Qwen3-0.6B-4bit"},
        "llama3-1b-4bit": {"hf_path": "mlx-community/Llama-3.2-1B-Instruct-4bit"},
        "smollm3-3b-4bit": {"hf_path": "mlx-community/SmolLM3-3B-4bit"},
        # Out-of-band: too large (> 12 B).
        "qwen3.5-27b-4bit": {"hf_path": "mlx-community/Qwen3.5-27B-4bit"},
        # Excluded: vision variant.
        "qwen3-vl-8b-4bit": {"hf_path": "mlx-community/Qwen3-VL-8B-Instruct-4bit"},
        # Excluded: kimi family (heavy + user-flagged).
        "kimi-k2-9b-4bit": {"hf_path": "fake/Kimi-K2-9B-4bit"},
        # Excluded: gemma-4 family (issue #686 thought-loop hang).
        "gemma-4-12b-4bit": {"hf_path": "mlx-community/gemma-4-12B-it-4bit"},
        "gemma-4-e4b-4bit": {"hf_path": "mlx-community/gemma-4-e4b-it-4bit"},
        # Excluded: 8-bit quant (we sample 4-bit only).
        "qwen3.5-9b-8bit": {"hf_path": "mlx-community/Qwen3.5-9B-8bit"},
        # Skipped silently: missing hf_path field.
        "broken-9b-4bit": {"tool_call_parser": "hermes"},
    }


def test_eligible_aliases_filters_correctly(g12, tmp_path):
    """The 3 in-band entries survive; everything else is filtered.

    Note ``glm4.5-air-4bit`` is **expected** to be excluded — its repo
    name (``GLM-4.5-Air-4bit``) carries no parameter-count token, so
    the post-codex-#693 parser fails closed rather than mis-attributing
    a 4 B size from the ``-4bit`` quantization suffix.
    """
    p = tmp_path / "aliases.json"
    p.write_text(json.dumps(_fake_aliases()))
    eligible = g12._eligible_aliases(p)
    names = {name for name, _ in eligible}
    assert names == {
        "qwen3.5-9b-4bit",
        "qwen3-8b-4bit",
        "hermes3-8b-4bit",
    }


def test_eligible_aliases_does_not_parse_quant_suffix_as_size(g12, tmp_path):
    """Regression: the size parser MUST NOT match ``-4bit`` / ``-8bit``
    as a fake 4 B / 8 B model size.

    Round-1 codex review of PR #693 caught this — the original regex
    ``(\\d+(?:\\.\\d+)?)b`` greedily matched the ``4b`` inside ``4bit``,
    so any 4-bit alias without a real size token in its repo name
    (e.g. ``GLM-4.5-Air-4bit``) slipped past the 4-12 B disk filter as
    a phantom 4 B model. The post-fix parser requires the ``b`` token
    to be bounded by name separators so the quant suffix can't match.
    """
    aliases = {
        # Repo name has NO parameter-count token. Must fail closed.
        "phantom-air-4bit": {"hf_path": "fake/Phantom-Air-4bit"},
        # Same idea, 8-bit suffix.
        "phantom-air-8bit": {"hf_path": "fake/Phantom-Air-8bit"},
        # Real size token IS present — must survive (control case).
        "good-9b-4bit": {"hf_path": "fake/Good-9B-4bit"},
    }
    p = tmp_path / "aliases.json"
    p.write_text(json.dumps(aliases))
    eligible = dict(g12._eligible_aliases(p))
    assert "phantom-air-4bit" not in eligible, (
        "size parser must not extract 4 from -4bit quant suffix"
    )
    assert "phantom-air-8bit" not in eligible, (
        "the 8-bit alias is filtered by quant-only rule anyway, but its "
        "repo name also has no real size token — both filters apply"
    )
    assert eligible.get("good-9b-4bit") == "fake/Good-9B-4bit"


def test_eligible_aliases_sorted_for_reproducible_sampling(g12, tmp_path):
    """``random.sample`` is order-sensitive — eligibility must return
    in deterministic order so the same seed picks the same models
    across machines / future aliases.json additions to unrelated
    entries.

    The function sorts by ``(size_B, name)``. We verify determinism by
    invoking twice and checking order also doesn't depend on dict
    insertion order in the source JSON.
    """
    p = tmp_path / "aliases.json"
    aliases = _fake_aliases()
    p.write_text(json.dumps(aliases))
    eligible_a = g12._eligible_aliases(p)
    eligible_b = g12._eligible_aliases(p)
    assert eligible_a == eligible_b
    # Write the same aliases in REVERSED insertion order — sort must
    # produce the same output regardless of dict-iteration order.
    p_reversed = tmp_path / "aliases_reversed.json"
    p_reversed.write_text(json.dumps(dict(reversed(list(aliases.items())))))
    eligible_c = g12._eligible_aliases(p_reversed)
    assert eligible_a == eligible_c, (
        "eligibility must be order-stable across source-file dict orderings"
    )


def test_the_measured_agentic_failure_is_out_of_the_pool(g12):
    """#1672, at the source rather than at the verdict.

    ``qwen3-4b-instruct-2507-4bit`` × ``hermes`` burns the entire 1020 s
    per-profile cap while the engine answers without a single 5xx, and v0.12.7
    behaves identically — the model's ceiling, not a regression. G12 is a
    coverage sweep, so the answer is to stop drawing a model that cannot do
    the work, NOT to teach the gate to forgive a red round: from the log of a
    failed round, "weak model" and "wedged scheduler" are the same evidence,
    and any rule lenient enough to pass the first ships the second.

    All four 4 B aliases go, not just the measured one. Waiting for each to be
    drawn and measured means a red gauntlet on a random calendar day, which is
    how a gate stops being read.
    """
    real = Path(__file__).resolve().parent.parent / "vllm_mlx" / "aliases.json"
    names = {name for name, _ in g12._eligible_aliases(real)}
    four_b = {n for n in names if re.search(r"(?:^|[-_.])4b(?=[-_.]|$)", n)}
    assert not four_b, f"the 4 B tier cannot run agentic profiles (#1672): {four_b}"
    assert "qwen3.5-9b-4bit" in names, (
        "the 9 B tier completes hermes in ~740-800 s and is where the floor's "
        "upper measurement comes from — it must stay in the pool"
    )


def test_multimodal_models_are_out_of_the_pool(g12):
    """The harness profiles are text-only, and this filter has always said so —
    but it said it by looking for ``-vl-`` in the alias, which is exactly the
    test the engine documents as insufficient.

    UI-TARS is a Qwen2.5-VL-based GUI agent whose public name carries no
    ``VL``; Gemma 3 is multimodal with no marker at all. Four of the nine
    eligible aliases were vision models, and the default seed sampled one. On
    an install without the vision extra that is a crash; with it, a text-only
    agentic harness measures nothing about a model whose whole contract is
    screenshots.
    """
    real = SCRIPT_PATH.parent.parent / "vllm_mlx" / "aliases.json"
    names = {name for name, _ in g12._eligible_aliases(real)}
    offenders = {n for n in names if "ui-tars" in n or "gemma3" in n or "-vl-" in n}
    assert not offenders, f"multimodal aliases in a text-only sweep: {offenders}"


def test_the_multimodal_mirror_still_covers_the_engines_own_list():
    """``MLLM_NAME_PATTERNS`` is a copy — the script cannot import the package
    without pulling mlx_lm at module load. Parsed, not imported, so this runs on
    a machine with no MLX at all: a family added upstream must not be able to
    reappear in the sweep silently."""
    tree = ast.parse(
        (SCRIPT_PATH.parent.parent / "vllm_mlx" / "api" / "utils.py").read_text()
    )
    upstream = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "MLLM_PATTERNS" for t in node.targets
        ):
            upstream = [e.value for e in node.value.elts]
    assert upstream, "MLLM_PATTERNS not found in vllm_mlx/api/utils.py"
    mirrored = {p.lower() for p in g12_module_patterns()}
    missing = {p.lower() for p in upstream} - mirrored
    assert not missing, f"engine patterns absent from the G12 mirror: {missing}"


def g12_module_patterns():
    """Load just the mirror, without importing anything heavyweight."""
    spec = importlib.util.spec_from_file_location("g12_mirror", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MLLM_NAME_PATTERNS


def test_the_floor_applies_to_active_params_not_total(g12, tmp_path):
    """A MoE's ability to hold an agentic loop follows its ACTIVE parameters,
    so the SAME floor applies to them.

    The cases straddle the new floor rather than the old one: A4 and A5 were
    admitted before this change and are not now, A6 is the boundary and stays.
    Picking A1-vs-A8 would pass either way and prove nothing about the move.
    """
    p = tmp_path / "aliases.json"
    p.write_text(
        json.dumps(
            {
                "moe-12b-a4b-4bit": {"hf_path": "fake/MoE-12B-A4B-4bit"},
                "moe-12b-a5b-4bit": {"hf_path": "fake/MoE-12B-A5B-4bit"},
                "moe-12b-a6b-4bit": {"hf_path": "fake/MoE-12B-A6B-4bit"},
            }
        )
    )
    names = {name for name, _ in g12._eligible_aliases(p)}
    assert names == {"moe-12b-a6b-4bit"}


def test_real_aliases_json_yields_nonzero_pool(g12):
    """Sanity check against the in-tree aliases.json: at least 5 eligible
    models must exist or the gauntlet has nothing to sample.

    The pool sits at EXACTLY 5 as of #1671 — the 6B floor took the four 4B
    aliases and the multimodal filter took three UI-TARS plus Gemma 3. There is
    no headroom left: the next exclusion, or an aliases prune, trips this, and
    the answer then is a gate sized for the models being excluded (#1677), not
    a lower bar here."""
    real = Path(__file__).resolve().parent.parent / "vllm_mlx" / "aliases.json"
    eligible = g12._eligible_aliases(real)
    assert len(eligible) >= 5, (
        f"need ≥5 sample-eligible aliases for meaningful G12 random "
        f"coverage; only {len(eligible)} pass the filter — "
        f"check the size band / new exclude rules in "
        f"release_check_m3_random.py"
    )


def test_hf_cache_dir_shape(g12, monkeypatch):
    """Cleanup path must match HuggingFace's actual snapshot layout
    (``models--<owner>--<repo>``) so the rm -rf at end-of-model
    actually deletes the right tree, not a sibling.

    Pin the env to the default branch — other tests in the module may
    leave ``HF_HOME`` / ``HF_HUB_CACHE`` set in this process.
    """
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(env, raising=False)
    p = g12._hf_cache_dir("mlx-community/Qwen3.5-9B-4bit")
    assert p.name == "models--mlx-community--Qwen3.5-9B-4bit"
    assert p.parent.name == "hub"
    assert p.parent.parent.name == "huggingface"


def test_free_disk_gb_walks_to_existing_ancestor(g12, tmp_path):
    """``_free_disk_gb`` must tolerate a non-existent leaf — the cache
    root may be on a custom mount whose ``models--owner--repo`` leaf
    hasn't been created until the first download. Without the ancestor
    walk, ``shutil.disk_usage`` raises ``FileNotFoundError`` on a
    brand-new ``HF_HUB_CACHE=/data/hf-cache`` rig where ``/data/``
    exists but ``hf-cache`` doesn't. Codex round-2 PR #693 review.
    """
    missing_leaf = tmp_path / "nonexistent" / "deeper" / "still-missing"
    assert not missing_leaf.exists()
    # Should return a real positive number, not raise.
    free_gb = g12._free_disk_gb(missing_leaf)
    assert free_gb > 0


def test_hf_cache_root_honors_env_vars(g12, tmp_path, monkeypatch):
    """``_hf_cache_root`` must respect ``HF_HUB_CACHE``,
    ``HUGGINGFACE_HUB_CACHE`` and ``HF_HOME`` in the same precedence
    order as ``huggingface_hub.constants.HF_HUB_CACHE`` — otherwise
    G12 downloads into one place and tries to ``rm -rf`` another,
    leaving the actual snapshots on disk to balloon across releases.
    Codex round-1 review of PR #693 caught this.
    """
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(env, raising=False)

    # Modern override
    target = tmp_path / "modern"
    monkeypatch.setenv("HF_HUB_CACHE", str(target))
    assert g12._hf_cache_root() == target
    monkeypatch.delenv("HF_HUB_CACHE")

    # Legacy override
    target = tmp_path / "legacy"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(target))
    assert g12._hf_cache_root() == target
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE")

    # HF_HOME — cache root is ``$HF_HOME/hub``
    target = tmp_path / "home"
    monkeypatch.setenv("HF_HOME", str(target))
    assert g12._hf_cache_root() == target / "hub"
    monkeypatch.delenv("HF_HOME")

    # Default
    assert g12._hf_cache_root() == Path.home() / ".cache" / "huggingface" / "hub"


# ---------------------------------------------------------------------------
# The port G12 benchmarks must belong to the server G12 started.
#
# Not hypothetical: while a gauntlet was running, a desktop app on this machine
# swept port 8000, SIGTERM'd the gauntlet's server and bound its own sidecar
# (#1618). Every later request went there, and nothing noticed, because the
# replacement answered perfectly well.
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, pid: int) -> None:
        self.pid = pid


def _stub_tree(g12, monkeypatch, *, listeners, parents):
    monkeypatch.setattr(g12, "_listening_pids", lambda port: list(listeners))
    monkeypatch.setattr(g12, "_parent_pid", lambda pid: parents.get(pid))


def test_our_own_process_owns_the_port(g12, monkeypatch):
    _stub_tree(g12, monkeypatch, listeners=[900], parents={})
    assert g12._owns_port(_FakeProc(900), 8000)


def test_a_descendant_of_ours_owns_the_port(g12, monkeypatch):
    """`rapid-mlx serve` may bind from a child of the process we spawned."""
    _stub_tree(g12, monkeypatch, listeners=[903], parents={903: 902, 902: 900})
    assert g12._owns_port(_FakeProc(900), 8000)


def test_a_stranger_does_not_own_the_port(g12, monkeypatch):
    """The #1618 shape: something else is listening and answering 200."""
    _stub_tree(g12, monkeypatch, listeners=[555], parents={555: 1})
    assert not g12._owns_port(_FakeProc(900), 8000)


def test_a_port_shared_with_a_stranger_is_refused(g12, monkeypatch):
    """One of ours plus one of theirs is not ownership — requests can land on
    either, so the numbers cannot be attributed to the sampled alias."""
    _stub_tree(g12, monkeypatch, listeners=[900, 555], parents={555: 1})
    assert not g12._owns_port(_FakeProc(900), 8000)


def test_an_unverifiable_port_is_refused(g12, monkeypatch):
    """Fails CLOSED. The caller only asks after a 200, so an empty listener
    list means `lsof` could not tell us — not that nothing is there. Reading
    "cannot tell" as "it's ours" reinstates exactly the bug."""
    _stub_tree(g12, monkeypatch, listeners=[], parents={})
    assert not g12._owns_port(_FakeProc(900), 8000)


def test_an_unreadable_parent_is_refused(g12, monkeypatch):
    """`ps` failing mid-walk proves nothing about the rest of the chain."""
    _stub_tree(g12, monkeypatch, listeners=[903], parents={903: None})
    assert not g12._owns_port(_FakeProc(900), 8000)


def test_the_ancestor_walk_is_bounded(g12, monkeypatch):
    """A pid whose parent chain loops must not spin forever."""
    _stub_tree(g12, monkeypatch, listeners=[10], parents={10: 11, 11: 10})
    assert not g12._owns_port(_FakeProc(900), 8000)


# ---------------------------------------------------------------------------
# Two writers, one of them offset-based, corrupts the artifact.
# ---------------------------------------------------------------------------


def test_the_bench_transcript_is_not_written_to_the_servers_log(g12):
    """The server holds its log open ``"w"`` — no O_APPEND — so it writes at
    an offset it tracks itself. Appending a bench transcript to that same path
    moves the end of the file without moving that offset, and the server's
    next line overwrites what we wrote."""
    assert g12._serve_log_path("qwen3-8b-4bit") != g12._bench_log_path("qwen3-8b-4bit")


def test_a_round_appends_its_transcript_to_the_bench_log(g12, tmp_path, monkeypatch):
    bench_log = tmp_path / "bench.log"
    bench_log.write_text("")

    class _Result:
        returncode = 0
        stdout = "harness ok\n"
        stderr = ""

    monkeypatch.setattr(g12.subprocess, "run", lambda *a, **k: _Result())
    ok, _, _ = g12._run_harness_round(
        alias="qwen3-8b-4bit",
        harness="hermes",
        base_url="http://127.0.0.1:8000",
        bench_log=bench_log,
    )
    assert ok
    assert "harness ok" in bench_log.read_text()


# ---------------------------------------------------------------------------
# The ownership check has to be WIRED IN, and it has to keep being asked.
#
# Testing `_owns_port` alone proves nothing: delete its call site and every
# test above stays green while the sweep goes back to benchmarking strangers.
# ---------------------------------------------------------------------------


class _FakeResponse:
    status = 200

    def read(self):
        return b'{"object":"list","data":[{"id":"x"}]}'

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _AliveProc:
    """A child that never exits — readiness must be decided by the port."""

    def __init__(self, pid: int = 900) -> None:
        self.pid = pid

    def poll(self):
        return None


def test_readiness_refuses_a_port_answering_from_someone_elses_process(
    g12, monkeypatch, tmp_path
):
    """A 200 is not ownership. Our child is alive the whole time it loads
    weights, so a stranger already on the port answers first (#1618).

    The second return value is the part that matters downstream: this is not
    "the model would not boot", it is "somebody else has the port", and that
    stays true for every model after this one.
    """
    monkeypatch.setattr(g12.urllib.request, "urlopen", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(g12, "_owns_port", lambda proc, port: False)
    monkeypatch.setattr(g12.time, "sleep", lambda *_: None)
    ready, saw_stranger = g12._wait_for_server(
        _AliveProc(), 8000, 0.3, tmp_path / "serve.log"
    )
    assert not ready
    assert saw_stranger


def test_readiness_accepts_the_port_once_it_is_ours(g12, monkeypatch, tmp_path):
    monkeypatch.setattr(g12.urllib.request, "urlopen", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(g12, "_owns_port", lambda proc, port: True)
    monkeypatch.setattr(g12.time, "sleep", lambda *_: None)
    ready, saw_stranger = g12._wait_for_server(
        _AliveProc(), 8000, 5, tmp_path / "serve.log"
    )
    assert ready
    assert not saw_stranger


def test_a_model_that_simply_will_not_boot_is_not_reported_as_a_takeover(
    g12, monkeypatch, tmp_path
):
    """The other direction. One model failing to load is that model's problem;
    treating it as environment contamination would abort a sweep that could
    still have covered the rest."""

    def _refused(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(g12.urllib.request, "urlopen", _refused)
    monkeypatch.setattr(g12.time, "sleep", lambda *_: None)
    ready, saw_stranger = g12._wait_for_server(
        _AliveProc(), 8000, 0.3, tmp_path / "serve.log"
    )
    assert not ready
    assert not saw_stranger


def test_a_stolen_port_during_boot_aborts_the_sweep(g12, monkeypatch, tmp_path):
    """A stranger on the port during boot must stop the sweep, not just log it.

    Driven through `main()` on purpose. The previous version of this test read
    the source and asserted that `drifted = True` appears near `if
    saw_stranger:` — which it did, while a `continue` on the same path ran the
    `finally` and jumped straight over the `if drifted: break` underneath it.
    The text was right and the behaviour was wrong, so the test could not tell.
    Counting how many models actually get booted can.
    """
    booted: list[str] = []
    ran_rounds: list[str] = []

    class _Proc:
        pid = 4242

        def poll(self):
            return None

    def _popen(cmd, **kwargs):
        # `serve <alias>` — the alias is the argument after "serve".
        booted.append(cmd[cmd.index("serve") + 1])
        return _Proc()

    aliases = tmp_path / "aliases.json"
    aliases.write_text(json.dumps(_fake_aliases()))
    report = tmp_path / "report.log"

    monkeypatch.setattr(g12.subprocess, "Popen", _popen)
    monkeypatch.setattr(g12, "_free_disk_gb", lambda path: 10_000.0)
    # The stranger arrives while the FIRST model is booting.
    monkeypatch.setattr(
        g12, "_wait_for_server", lambda proc, port, timeout, log: (False, True)
    )
    monkeypatch.setattr(
        g12,
        "_run_model_rounds",
        lambda **kw: ran_rounds.append(kw["alias"]) or ([], False),
    )
    monkeypatch.setattr(g12, "_stop_server", lambda proc, port: None)
    monkeypatch.setattr(g12, "_hf_cache_dir", lambda hf_path: tmp_path / "nonexistent")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "release_check_m3_random.py",
            "--models",
            "3",
            "--harnesses",
            "1",
            "--rounds",
            "1",
            "--seed",
            "7",
            "--aliases-json",
            str(aliases),
            "--report",
            str(report),
        ],
    )

    rc = g12.main()

    assert len(booted) == 1, (
        f"the sweep booted {len(booted)} models after detecting a stranger on "
        f"the port during the first one's boot: {booted} — every model after "
        "the first is measured against a machine we know is contaminated"
    )
    assert ran_rounds == [], (
        f"benchmark rounds ran against the contaminated port: {ran_rounds}"
    )
    assert rc != 0, "a drifted sweep must not report success"


def test_a_takeover_is_noticed_and_named(g12, monkeypatch):
    """`_still_ours` is what the round loop asks before and after every round.
    Its answer has to be a reason, not a bare False — a sweep that says "round
    3 failed" without saying the port changed hands sends someone hunting a
    regression that is not there."""
    monkeypatch.setattr(g12, "_owns_port", lambda proc, port: False)
    monkeypatch.setattr(g12, "_listening_pids", lambda port: [555])
    reason = g12._still_ours(_AliveProc(), 8000)
    assert "no longer served by our process" in reason
    assert "555" in reason


def test_a_dead_server_is_noticed_by_the_same_check(g12, monkeypatch):
    class _Dead(_AliveProc):
        def poll(self):
            return 137

    monkeypatch.setattr(g12, "_owns_port", lambda proc, port: True)
    assert "exited (rc=137)" in g12._still_ours(_Dead(), 8000)


def _drive_rounds(g12, monkeypatch, tmp_path, *, ownership, rounds=2, harnesses=None):
    """Run `_run_model_rounds` with ownership answers fed in order."""
    answers = list(ownership)
    asked: list[str] = []
    ran: list[tuple[str, int]] = []

    def _ours(proc, port):
        reply = answers.pop(0) if answers else ""
        asked.append(reply)
        return reply

    def _round(*, alias, harness, base_url, bench_log):
        ran.append((harness, len(ran)))
        return True, 1.0, ""

    monkeypatch.setattr(g12, "_still_ours", _ours)
    monkeypatch.setattr(g12, "_run_harness_round", _round)
    report = tmp_path / "report.log"
    report.write_text("")
    failures, drifted = g12._run_model_rounds(
        proc=_AliveProc(),
        port=8000,
        alias="qwen3-8b-4bit",
        harnesses=list(harnesses or ["hermes", "aider"]),
        rounds=rounds,
        bench_log=tmp_path / "bench.log",
        report_path=report,
    )
    return failures, drifted, asked, ran, report.read_text()


def test_ownership_is_asked_twice_per_round(g12, monkeypatch, tmp_path):
    """Once before and once after. Before alone cannot catch a takeover that
    happens DURING a round, and after alone measures a stranger first."""
    failures, drifted, asked, ran, _ = _drive_rounds(
        g12, monkeypatch, tmp_path, ownership=[""] * 8
    )
    assert not failures and not drifted
    assert len(ran) == 4, "2 harnesses x 2 rounds"
    assert len(asked) == 8, "one check before and one after each round"


def test_a_takeover_during_a_round_discards_that_round(g12, monkeypatch, tmp_path):
    """The dangerous direction: the round finished, so there IS a result — it
    just belongs to whoever took the port. It must not be reported as this
    model's."""
    failures, drifted, _, ran, report = _drive_rounds(
        g12, monkeypatch, tmp_path, ownership=["", "taken"]
    )
    assert drifted
    assert len(ran) == 1, "the round ran, and is the one being discarded"
    assert "PASS" not in report, "a round whose port changed hands is not a PASS"
    assert len(failures) == 1
    assert "during the round" in failures[0]


def test_a_takeover_before_a_round_does_not_run_it(g12, monkeypatch, tmp_path):
    failures, drifted, _, ran, _ = _drive_rounds(
        g12, monkeypatch, tmp_path, ownership=["taken"]
    )
    assert drifted and not ran
    assert "before the round" in failures[0]


def test_drift_skips_the_remaining_harnesses(g12, monkeypatch, tmp_path):
    """Continuing after a takeover measures the stranger under this alias's
    name for every remaining round."""
    _, drifted, _, ran, _ = _drive_rounds(
        g12,
        monkeypatch,
        tmp_path,
        ownership=["", "", "", "taken"],
        harnesses=["hermes", "aider", "codex"],
    )
    assert drifted
    assert len(ran) == 2, "stopped inside the first harness, not after it"


def test_the_sweep_aborts_rather_than_booting_the_next_model_into_a_stranger():
    """Pin the caller's half. `_run_model_rounds` reporting drift is useless if
    `main` shrugs and boots the next model into the environment that took the
    port — which can also overlap its GPU allocation with whatever is tearing
    down."""
    source = SCRIPT_PATH.read_text()
    sweep = source[source.index("    # ===== Sweep =====") :]
    sweep = sweep[: sweep.index("    # ===== Verdict =====")]
    assert "if drifted:" in sweep and "break" in sweep, (
        "the model loop must stop on ownership drift, not continue"
    )


def test_a_partial_lsof_run_is_not_read_as_a_listener_list(g12, monkeypatch):
    """lsof can print some rows and then fail. Half a listener list reads as
    "all of these are ours", so the row it did not print is exactly the
    stranger the caller asked about."""

    class _Partial:
        returncode = 1
        stdout = "900\n"

    monkeypatch.setattr(g12.subprocess, "run", lambda *a, **k: _Partial())
    assert g12._listening_pids(8000) == []


def test_a_timed_out_round_still_leaves_a_transcript(g12, tmp_path, monkeypatch):
    """The failure class that most needs evidence produced none: the timeout
    path returned before anything was written."""
    bench_log = tmp_path / "bench.log"
    bench_log.write_text("")

    def _boom(*a, **k):
        raise g12.subprocess.TimeoutExpired(
            cmd="bench", timeout=1, output=b"got as far as tier=harness\n", stderr=b""
        )

    monkeypatch.setattr(g12.subprocess, "run", _boom)
    ok, _, excerpt = g12._run_harness_round(
        alias="qwen3-8b-4bit",
        harness="hermes",
        base_url="http://127.0.0.1:8000",
        bench_log=bench_log,
    )
    assert not ok
    assert "timed out" in excerpt
    assert "got as far as tier=harness" in bench_log.read_text()


# ---------------------------------------------------------------------------
# The same question asked of the real machine.
#
# Every ownership test above stubs `_listening_pids` and `_parent_pid`, so none
# of them exercises the `lsof`/`ps` invocations or the actual process tree — and
# that blind spot hid a live bug: `_owns_port` returned True for pid 1, because
# every process descends from init and nothing rejected that. Measured here, not
# reasoned about.
# ---------------------------------------------------------------------------


# The child binds its own port and reports it, rather than the parent asking
# the kernel for a free one and handing over the number. Closing the probe
# socket before the child binds leaves a window in which any other process on
# the machine can take that port — a flake that would show up as this gate
# failing for reasons nobody can reproduce.
_PORT_HOLDER = """
import socket, sys, time
s = socket.socket()
s.bind(("127.0.0.1", 0))
s.listen(1)
sys.stdout.write(str(s.getsockname()[1]) + chr(10))
sys.stdout.flush()
time.sleep(60)
"""


@pytest.mark.skipif(
    shutil.which("lsof") is None or shutil.which("ps") is None,
    reason="ownership is established with lsof + ps",
)
def test_ownership_against_a_real_listener(g12):
    class _Pid:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def poll(self):
            return None

    proc = subprocess.Popen(
        [sys.executable, "-c", _PORT_HOLDER],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        line = proc.stdout.readline().strip()
        assert line.isdigit(), f"the listener did not report a port: {line!r}"
        port = int(line)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not g12._listening_pids(port):
            time.sleep(0.1)
        assert g12._listening_pids(port), "the child never bound the port"

        assert g12._owns_port(proc, port), "our own child must count as ours"
        # We are the listener's parent, so we own it too — the walk has to
        # cross at least one edge for a `serve` that binds from a subprocess.
        assert g12._owns_port(_Pid(os.getpid()), port)
        # init is an ancestor of everything, which is exactly why it must not
        # satisfy ownership.
        assert not g12._owns_port(_Pid(1), port)
        assert "no longer served by our process" in g12._still_ours(_Pid(1), port)
        assert g12._still_ours(proc, port) == ""
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and g12._listening_pids(port):
        time.sleep(0.1)
    assert not g12._listening_pids(port)
    assert not g12._owns_port(proc, port), "a freed port belongs to nobody"


def test_logs_live_in_a_private_directory_not_a_guessable_tmp_path(g12):
    """A predictable name in a world-writable directory is a truncation gadget.

    The per-alias logs used to be `/tmp/release-check-m3-random-<alias>.log`.
    Any other local process can pre-create that exact name as a symlink, and
    this script opens the path with `write_text("")` — which follows the link
    and truncates whatever it points at, as the user running the gauntlet.
    Owning the directory removes the guess: nobody else can create entries in
    it.
    """
    serve = g12._serve_log_path("qwen3-8b-4bit")
    bench = g12._bench_log_path("qwen3-8b-4bit")

    assert serve.parent == bench.parent, "both logs belong to the same run"
    parent = serve.parent
    assert parent != Path("/tmp"), (
        f"logs are written straight into a world-writable directory: {serve}"
    )
    assert parent.is_dir()
    mode = stat.S_IMODE(parent.stat().st_mode)
    assert mode == 0o700, (
        f"log directory is {mode:o}, not 0700 — another user can plant "
        f"symlinks in it: {parent}"
    )
    # Same run, same directory: a second call must not mint a new one, or the
    # per-alias logs of one sweep end up scattered.
    assert g12._serve_log_path("another-alias").parent == parent
    assert serve != bench, "the serve and bench logs must not share a path"


def test_a_listener_exactly_max_depth_below_us_is_ours(g12, monkeypatch):
    """`max_depth` counts ancestry EDGES, so the walk needs max_depth + 1 checks.

    Checking only `max_depth` times tested every generation except the last
    one it walked to, so a legitimate listener exactly `max_depth` edges below
    the server we started was reported as a stranger — and a stranger aborts
    the release as ownership drift.
    """
    chain = {900: 800, 800: 700, 700: 600, 600: 500, 500: 4242, 4242: 1}
    monkeypatch.setattr(g12, "_listening_pids", lambda port: [900])
    monkeypatch.setattr(g12, "_parent_pid", lambda pid: chain.get(pid))
    # 900 -> 800 -> 700 -> 600 -> 500 -> 4242 is five edges. _AliveProc
    # defaults to pid 900, which IS the listener — pass the server pid
    # explicitly or the walk matches at depth 0 and proves nothing.
    assert g12._owns_port(_AliveProc(4242), 8000, max_depth=5) is True
    # One edge short of reaching us is correctly a stranger.
    assert g12._owns_port(_AliveProc(4242), 8000, max_depth=4) is False


def test_the_bench_transcript_is_wired_to_the_bench_log_not_the_server_log(
    g12, monkeypatch, tmp_path
):
    """Drive `main()`: the round runner must receive the BENCH log.

    Asserting that the two helper functions return different names says
    nothing about which one `main()` actually hands down. If it passed the
    serve log, two writers would share one file — the server tracking its own
    byte offset — and corrupt the artifact someone needs on exactly the runs
    that failed.
    """
    seen: list[Path] = []

    class _Proc:
        pid = 4242

        def poll(self):
            return None

    aliases = tmp_path / "aliases.json"
    aliases.write_text(json.dumps(_fake_aliases()))

    monkeypatch.setattr(g12.subprocess, "Popen", lambda cmd, **kw: _Proc())
    monkeypatch.setattr(g12, "_free_disk_gb", lambda path: 10_000.0)
    monkeypatch.setattr(
        g12, "_wait_for_server", lambda proc, port, timeout, log: (True, False)
    )
    monkeypatch.setattr(g12, "_still_ours", lambda proc, port: "")
    monkeypatch.setattr(g12, "_stop_server", lambda proc, port: None)
    monkeypatch.setattr(g12, "_hf_cache_dir", lambda hf_path: tmp_path / "nonexistent")

    def _round(*, alias, harness, base_url, bench_log):
        seen.append(Path(bench_log))
        return True, 1.0, ""

    monkeypatch.setattr(g12, "_run_harness_round", _round)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "release_check_m3_random.py",
            "--models",
            "1",
            "--harnesses",
            "1",
            "--rounds",
            "1",
            "--seed",
            "7",
            "--aliases-json",
            str(aliases),
        ],
    )

    g12.main()

    assert seen, "no harness round ran, so nothing was verified"
    for bench_log in seen:
        assert bench_log.name.endswith(".bench.log"), (
            f"the round runner was handed {bench_log}, which is not a bench log"
        )
        assert bench_log != g12._serve_log_path(bench_log.name.split(".")[0]), (
            "the round transcript would be appended to the server's own log"
        )


def test_the_report_defaults_into_the_private_directory(
    g12, monkeypatch, tmp_path, capsys
):
    """An unspecified --report must not land on a fixed name in /tmp."""
    aliases = tmp_path / "aliases.json"
    aliases.write_text(json.dumps(_fake_aliases()))
    monkeypatch.setattr(g12, "_free_disk_gb", lambda path: 10_000.0)
    monkeypatch.setattr(
        g12, "_wait_for_server", lambda proc, port, timeout, log: (False, False)
    )
    monkeypatch.setattr(g12.subprocess, "Popen", lambda cmd, **kw: _AliveProc())
    monkeypatch.setattr(g12, "_stop_server", lambda proc, port: None)
    monkeypatch.setattr(g12, "_hf_cache_dir", lambda hf_path: tmp_path / "nonexistent")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "release_check_m3_random.py",
            "--models",
            "1",
            "--harnesses",
            "1",
            "--rounds",
            "1",
            "--seed",
            "7",
            "--aliases-json",
            str(aliases),
        ],
    )
    g12.main()
    written = g12._log_dir() / "report.log"
    assert written.is_file(), f"no report written into {g12._log_dir()}"
    assert written.parent != Path("/tmp")
    assert f"Full log: {written}" in capsys.readouterr().out


def test_main_fails_fast_when_lsof_is_unavailable(g12, monkeypatch, capsys):
    """Standalone G12 must not spend ten minutes misdiagnosing missing lsof."""
    monkeypatch.setattr(g12.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        g12,
        "_port_free",
        lambda port: pytest.fail("port probing must happen after the lsof preflight"),
    )
    monkeypatch.setattr(sys, "argv", ["release_check_m3_random.py"])

    assert g12.main() == 2
    assert "lsof is required" in capsys.readouterr().err
