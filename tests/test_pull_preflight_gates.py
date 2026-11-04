"""Pull must admit the complete selected payload before any transfer (offline)."""

from argparse import Namespace

import pytest

from rapid_mlx import cli
from rapid_mlx.byom import preflight
from rapid_mlx.model_aliases import resolve_profile

pytestmark = pytest.mark.catalog_memory_preflight
GIB = 1 << 30
ORIGINAL_MISSING_BYTES = cli._missing_download_bytes


@pytest.fixture
def host(monkeypatch):
    calls = []
    priced = []
    monkeypatch.setattr(preflight, "physical_ram_bytes", lambda: 512 * GIB)
    monkeypatch.setattr(preflight, "mlx_recommended_working_set_bytes", lambda: None)
    monkeypatch.setattr(
        preflight, "_same_task_fitting_pick", lambda *a, **k: (None, None)
    )
    monkeypatch.setattr(preflight, "_emit_rejection", lambda *a: None)
    monkeypatch.setattr(cli, "_cache_runnability", lambda *a: False)
    monkeypatch.setattr(
        "rapid_mlx.runtime.video_lane.video_runtime_issue", lambda *a, **k: None
    )
    monkeypatch.setattr(cli, "_prepare_audio_runtime_requirements", lambda *a: None)
    monkeypatch.setattr(cli, "_emit_pull_activation", lambda: None)
    monkeypatch.setattr(
        cli, "_pull_repository", lambda args, **kw: calls.append((args.model, kw))
    )
    monkeypatch.setattr(
        "rapid_mlx._mirror._cache_disk_space", lambda: ("cache", 1000 * 10**9)
    )

    def size(repo, *, revision_override=None, allow_patterns=None):
        priced.append(
            (
                repo,
                revision_override,
                list(allow_patterns) if allow_patterns is not None else None,
            )
        )
        return 40 * 10**9

    monkeypatch.setattr(cli, "_missing_download_bytes", size)
    return calls, priced


def pull(alias, **kw):
    cli.pull_command(Namespace(model=alias, **kw))


@pytest.mark.parametrize(
    "alias,extra",
    [
        ("pika-h3-flex", {}),
        ("minimax-h3", {}),
        ("sd35-large-4bit", {}),
        ("hidream-o1-dev", {}),
        ("sdxl-base", {}),
        ("bonsai-image-4b-2bit", {}),
        ("ltx-2.5-mlx-q8", {}),
        ("qwen-image", {}),
        ("mlx-community/Kokoro-82M-bf16", {}),
        ("qwen3.5-4b-4bit", {}),
    ],
)
def test_disk_refuses_complete_plan_before_download(
    host, monkeypatch, capsys, alias, extra
):
    calls, priced = host
    from rapid_mlx.audio.registry import runtime_assets_for

    assert resolve_profile(alias) is not None or runtime_assets_for(alias)
    # One repository fits; the complete dependency plan must still be checked.
    monkeypatch.setattr(
        "rapid_mlx._mirror._cache_disk_space", lambda: ("cache", 24 * 10**9)
    )
    with pytest.raises(SystemExit) as exc:
        pull(alias, **extra)
    assert exc.value.code == 1
    assert calls == []
    assert priced
    output = capsys.readouterr()
    assert "Insufficient disk space" in output.err
    assert "Nothing was downloaded." in output.err
    assert "Free space:" in output.err
    assert "10%" in output.err


@pytest.mark.parametrize("alias", ["pika-h3-flex", "minimax-h3"])
def test_pull_memory_floor_and_bypass(host, monkeypatch, capsys, alias):
    calls, priced = host
    monkeypatch.setattr(preflight, "physical_ram_bytes", lambda: 48 * GIB)
    with pytest.raises(SystemExit) as exc:
        pull(alias)
    assert exc.value.code == 1
    assert calls == priced == []
    error = capsys.readouterr().err
    floor = resolve_profile(alias).min_memory_gb
    assert (
        f"needs a Mac with at least {floor:g} GB of unified memory; this Mac has 48 GB."
        in error
    )
    assert "Nothing was downloaded." in error
    assert error.count("--no-preflight") == 1
    assert (
        "(To download anyway, for example to move the files to a bigger Mac, "
        "re-run with --no-preflight.)" in error
    )
    pull(alias, no_preflight=True)
    assert len(calls) == len(priced) > 1


@pytest.mark.parametrize("command", ["pull", "serve"])
def test_memory_refusal_suggestion_matches_command(host, monkeypatch, capsys, command):
    monkeypatch.setattr(preflight, "physical_ram_bytes", lambda: 48 * GIB)
    pick = "qwen3.5-4b-4bit"
    monkeypatch.setattr(
        preflight, "_same_task_fitting_pick", lambda *a, **k: (pick, "text")
    )
    monkeypatch.setattr(preflight, "_candidate_disk_fit", lambda *a: (True, True, None))
    with pytest.raises(SystemExit) as exc:
        preflight._check_catalog_memory_fit(
            Namespace(command=command, model="pika-h3-flex")
        )
    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert f"A model that fits this Mac: rapid-mlx {command} {pick}" in error
    assert error.count("--no-preflight") == 1
    if command == "serve":
        assert (
            "(Think this is wrong? Re-run with --no-preflight to skip this check.)"
            in error
        )


def test_metadata_failure_keeps_pull_available(host, monkeypatch):
    calls, _ = host
    monkeypatch.setattr("rapid_mlx._mirror._cache_disk_space", lambda: ("cache", 0))
    with pytest.raises(SystemExit):
        pull("minimax-h3")
    assert calls == []
    # Exercise the real best-effort estimator, rather than replacing its
    # documented return-zero-on-metadata-failure contract with an exception.
    monkeypatch.setattr(cli, "_missing_download_bytes", ORIGINAL_MISSING_BYTES)
    monkeypatch.setattr("rapid_mlx._mirror._packaged_digest_info", lambda *a: None)

    def unavailable(*a, **k):
        raise OSError("Hub metadata unavailable")

    monkeypatch.setattr("huggingface_hub.model_info", unavailable)
    pull("minimax-h3")
    assert len(calls) == 2


def test_cached_pull_skips_memory_floor(host, monkeypatch):
    calls, _ = host
    monkeypatch.setattr(preflight, "physical_ram_bytes", lambda: 48 * GIB)
    monkeypatch.setattr(cli, "_cache_runnability", lambda *a: False)
    with pytest.raises(SystemExit):
        pull("minimax-h3")
    assert calls == []
    monkeypatch.setattr(cli, "_cache_runnability", lambda *a: True)
    pull("minimax-h3")
    assert len(calls) == 2


@pytest.mark.parametrize(
    "alias",
    ["minimax-h3", "pika-h3-flex", "sd35-large-4bit", "mlx-community/Kokoro-82M-bf16"],
)
def test_dependency_sum_can_refuse_when_each_repository_fits(host, monkeypatch, alias):
    calls, priced = host
    monkeypatch.setattr(
        "rapid_mlx._mirror._cache_disk_space", lambda: ("cache", 45 * 10**9)
    )
    with pytest.raises(SystemExit):
        pull(alias)
    assert len(priced) > 1
    assert calls == []


def test_memory_bypass_keeps_disk_gate(host, monkeypatch):
    calls, priced = host
    monkeypatch.setattr(preflight, "physical_ram_bytes", lambda: 48 * GIB)
    monkeypatch.setattr("rapid_mlx._mirror._cache_disk_space", lambda: ("cache", 0))
    with pytest.raises(SystemExit):
        pull("pika-h3-flex", no_preflight=True)
    assert priced
    assert calls == []


def test_unreadable_disk_keeps_pull_available(host, monkeypatch):
    calls, _ = host

    monkeypatch.setattr("rapid_mlx._mirror._cache_disk_space", lambda: ("cache", 0))
    with pytest.raises(SystemExit):
        pull("minimax-h3")
    assert calls == []

    def unavailable():
        raise OSError("cache filesystem unavailable")

    monkeypatch.setattr("rapid_mlx._mirror._cache_disk_space", unavailable)
    pull("minimax-h3")
    assert len(calls) == 2


def test_force_disk_check_warns_and_downloads(host, monkeypatch, capsys):
    calls, _ = host
    monkeypatch.setattr("rapid_mlx._mirror._cache_disk_space", lambda: ("cache", 0))
    with pytest.raises(SystemExit):
        pull("minimax-h3")
    assert calls == []
    capsys.readouterr()
    pull("minimax-h3", force_disk_check=True)
    assert len(calls) == 2
    assert "Insufficient disk space" in capsys.readouterr().err
