# SPDX-License-Identifier: Apache-2.0
"""Immutable catalog checkpoints across discovery, download and load boundaries."""

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rapid_mlx import _download_gate as gate
from rapid_mlx import cli, model_downloads, model_metadata
from rapid_mlx import model_aliases as aliases
from tests.pull_integrity_helpers import hub_integrity_metadata, snapshot_tree  # noqa: F401

PIN = "3b93f9053b530488405ae9fca51d4900d6ce21f0"
OTHER = "a" * 40
ALIAS = "flint-40b-a2.5b-4bit"
REPO = "RecursiveSI/flint-int4"
UNPINNED = "mlx-community/Qwen3-0.6B-4bit"


@pytest.fixture(params=[REPO, ALIAS, "my-flint"])
def pinned_name(request, monkeypatch, tmp_path):
    from rapid_mlx import user_aliases

    monkeypatch.setenv(
        "RAPID_MLX_USER_ALIASES_FILE", str(tmp_path / "user-aliases.json")
    )
    user_aliases.set_user_alias("my-flint", ALIAS, aliases.list_aliases())
    return request.param


@pytest.mark.parametrize(
    "extra,expected", [({}, None), ({"revision": None}, None), ({"revision": PIN}, PIN)]
)
def test_coerce_optional_revision(extra, expected):
    assert (
        aliases._coerce("example", {"hf_path": "org/repo", **extra}).revision
        == expected
    )
    assert aliases._coerce("legacy", "org/repo").revision is None


@pytest.mark.parametrize(
    "revision", ["abc", "A" * 40, "main", 1, True, [], {}, " " + PIN]
)
def test_coerce_rejects_mutable_or_invalid_revision(revision):
    with pytest.raises(ValueError, match="revision must"):
        aliases._coerce("example", {"hf_path": "org/repo", "revision": revision})


@pytest.mark.parametrize("kind", ["dflash", "tensorfold"])
def test_target_pin_must_agree(kind):
    data = {"hf_path": "org/repo", "revision": PIN}
    if kind == "dflash":
        data.update(
            dflash_algorithm="dflash2",
            dflash_draft_model="org/draft",
            dflash_draft_revision=OTHER,
            dflash_target_revision=OTHER,
        )
    else:
        data.update(
            tensorfold_mtp=True,
            tensorfold_target_revision=OTHER,
            tensorfold_runtime_revision=OTHER,
        )
    with pytest.raises(ValueError, match="disagrees"):
        aliases._coerce("example", data)
    data[f"{kind}_target_revision"] = PIN
    assert aliases._coerce("example", data).revision == PIN


@pytest.mark.parametrize("second", [OTHER, None, PIN])
def test_cross_alias_revision_guard(second):
    profiles = {
        "one": aliases.AliasProfile(hf_path="ORG/repo", revision=PIN),
        "two": aliases.AliasProfile(hf_path="org/REPO", revision=second),
    }
    if second == PIN:
        aliases._assert_revision_is_unambiguous(profiles)
    else:
        with pytest.raises(ValueError, match="different revision"):
            aliases._assert_revision_is_unambiguous(profiles)


def test_resolution_and_user_store_compatibility(monkeypatch, tmp_path):
    from rapid_mlx import user_aliases

    store = tmp_path / "user-aliases.json"
    monkeypatch.setenv("RAPID_MLX_USER_ALIASES_FILE", str(store))
    builtins = aliases.list_aliases()
    # Persistence remains the existing version-1 string-to-string contract.
    user_aliases.set_user_alias("my-flint", ALIAS, builtins)
    user_aliases.set_user_alias("my-repo", REPO.upper(), builtins)
    assert json.loads(store.read_text()) == {
        "version": 1,
        "aliases": {"my-flint": ALIAS, "my-repo": REPO.upper()},
    }
    for name in (ALIAS, REPO, REPO.upper(), "my-flint", "my-repo"):
        assert aliases.resolve_revision(name) == PIN
        assert aliases.resolve_hub_model(name) == REPO
    assert aliases.resolve_revision(UNPINNED) is None
    assert aliases.resolve_revision("unknown/repo") is None
    assert aliases.resolve_revision(str(tmp_path)) is None
    monkeypatch.setattr(aliases, "external_model_root_paths", lambda: [str(tmp_path)])
    monkeypatch.setattr(
        aliases,
        "_resolve_external_model_path",
        lambda name: pytest.fail("pin cannot use unversioned external copy"),
    )
    assert aliases.resolve_model(ALIAS) == REPO
    assert aliases.resolve_model(REPO.upper()) == REPO


def snapshot(root, sha, *, subfolder=None, complete=True):
    path = root / "snapshots" / sha
    if subfolder:
        path /= subfolder
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
            }
        )
    )
    (path / "tokenizer_config.json").write_text(json.dumps({"chat_template": sha}))
    (path / "tokenizer.json").write_text("{}")
    if complete:
        (path / "model.safetensors").write_bytes(b"weights")
    return path


@pytest.fixture
def cache(monkeypatch, tmp_path):
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    root = tmp_path / ("models--" + REPO.replace("/", "--"))
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text(OTHER)
    return root


def test_exact_cache_readiness_metadata_and_offline(cache, monkeypatch):
    from rapid_mlx import local_models
    from rapid_mlx.utils.tokenizer import (
        _local_snapshot_if_cached,
        _resolve_subfolder_checkpoint,
    )

    other = snapshot(cache, OTHER)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert not gate.is_repo_cached(REPO)
    assert not gate.is_repo_cached(ALIAS)
    assert cli._cache_runnability(REPO) is False
    assert model_metadata.read_cached_model_metadata(REPO) is None
    assert model_metadata.resolve_unreferenced_cached_snapshot(REPO) is None
    assert not local_models.catalog_model_ready(ALIAS)
    pinned = snapshot(cache, PIN, complete=False)
    assert not gate.is_repo_cached(REPO)
    # Metadata-only routing may read the pin, never the other snapshot.
    assert model_metadata.read_cached_model_metadata(REPO).snapshot_dir == pinned
    (pinned / "model.safetensors").write_bytes(b"weights")
    assert gate.is_repo_cached(REPO)
    assert gate.is_repo_cached(ALIAS)
    assert cli._cache_runnability(REPO) is True
    assert model_metadata.resolve_unreferenced_cached_snapshot(REPO) == pinned
    assert model_metadata.resolve_offline_cached_snapshot(REPO) == pinned
    assert model_metadata.read_cached_model_metadata(REPO).chat_template == PIN
    assert _local_snapshot_if_cached(REPO) == str(pinned)
    assert _resolve_subfolder_checkpoint(REPO) == str(pinned)
    assert local_models._checkpoint_dir(REPO, False) == str(pinned)
    # A selected ordinary repo still follows refs/main.
    monkeypatch.setattr(aliases, "resolve_revision", lambda _: None)
    assert gate.is_repo_cached(REPO)
    assert model_metadata.read_cached_model_metadata(REPO).snapshot_dir == other
    assert model_metadata.resolve_pinned_cached_snapshot(REPO) is None


@pytest.mark.parametrize("method", ["snapshot_download", "hf_hub_download"])
@pytest.mark.parametrize("disabled", [False, True])
def test_shared_download_wrappers_enforce_catalog_pin(
    monkeypatch, method, disabled, pinned_name
):
    import huggingface_hub

    download = Mock(return_value="/snapshot")
    monkeypatch.setattr(huggingface_hub, method, download)
    monkeypatch.setattr(model_downloads, "disabled", lambda: disabled)
    fn = getattr(model_downloads, method)
    rest = () if method == "snapshot_download" else ("config.json",)
    for repo, kwargs, expected in [
        (pinned_name, {}, PIN),
        (pinned_name, {"revision": None}, PIN),
        (pinned_name, {"revision": PIN}, PIN),
        (UNPINNED, {}, None),
        (UNPINNED, {"revision": OTHER}, OTHER),
    ]:
        assert fn(repo, *rest, **kwargs) == "/snapshot"
        assert download.call_args.args == (REPO if expected == PIN else repo, *rest)
        actual = download.call_args.kwargs
        assert actual.get("revision") == expected
        assert actual.get("local_files_only", False) == disabled

    download.reset_mock()
    with pytest.raises(ValueError, match=rf"{pinned_name}.*{PIN}"):
        fn(pinned_name, *rest, revision=OTHER)
    download.assert_not_called()


@pytest.mark.parametrize("name", [ALIAS, REPO])
def test_warm_pinned_subfolder_checkpoint_uses_cache(cache, monkeypatch, name):
    from rapid_mlx import _mirror
    from rapid_mlx.utils.tokenizer import _resolve_subfolder_checkpoint

    profile = aliases.AliasProfile(hf_path=REPO, revision=PIN, subfolder="4bit")
    monkeypatch.setattr(aliases, "_aliases", {ALIAS: profile})
    monkeypatch.setattr(aliases, "_hf_to_alias", {REPO.lower(): ALIAS})
    pinned = snapshot(cache, PIN, subfolder="4bit")
    fetch = Mock(side_effect=AssertionError("warm cache must not download"))
    monkeypatch.setattr(_mirror, "pinned_snapshot_download", fetch)

    assert _resolve_subfolder_checkpoint(name) == str(pinned)
    fetch.assert_not_called()


@pytest.mark.parametrize("subfolder", [None, "4bit"])
def test_cold_checkpoint_and_server_use_pinned_snapshot(
    monkeypatch, tmp_path, subfolder
):
    from rapid_mlx import _mirror, server
    from rapid_mlx.utils.tokenizer import _resolve_subfolder_checkpoint

    profile = aliases.AliasProfile(hf_path=REPO, revision=PIN, subfolder=subfolder)
    monkeypatch.setattr(aliases, "resolve_profile", lambda _: profile)
    monkeypatch.setattr(
        model_metadata, "resolve_pinned_cached_snapshot", lambda _: None
    )
    root = tmp_path / "repo"
    path = snapshot(root, PIN, subfolder=subfolder)
    fetch = Mock(return_value=str(root / "snapshots" / PIN))
    monkeypatch.setattr(_mirror, "pinned_snapshot_download", fetch)
    assert Path(_resolve_subfolder_checkpoint(REPO)) == path
    fetch.assert_called_with(
        REPO, PIN, allow_patterns=["4bit/*"] if subfolder else None
    )
    monkeypatch.setattr(server, "_preflight_vision_runtime", lambda *a, **k: None)
    result = server._resolve_serving_checkpoint(REPO, force_text=True)
    assert Path(result.load_path) == path
    # A missing pin or incomplete pin is a hard failure, never a bare repo.
    fetch.side_effect = RuntimeError("pin unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        _resolve_subfolder_checkpoint(REPO)
    fetch.side_effect = None
    (path / "model.safetensors").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        _resolve_subfolder_checkpoint(REPO)


@pytest.mark.parametrize("repo,pin", [(REPO, PIN), (UNPINNED, None)])
@pytest.mark.parametrize("mirror_ok", [True, False])
def test_ensure_download_threads_pin_and_preserves_main(
    monkeypatch, cache, repo, pin, mirror_ok
):
    import huggingface_hub

    from rapid_mlx import _mirror

    mirror = Mock(return_value=mirror_ok)
    hf = Mock(return_value="/snapshot")
    info = Mock(return_value=SimpleNamespace(sha=OTHER, siblings=[]))
    disk = Mock()
    monkeypatch.setattr(cli, "_cache_runnability", lambda _: False)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(cli, "_check_disk_space", disk)
    monkeypatch.setattr(cli, "_model_pull_blob_identifier", lambda _: ())
    monkeypatch.setattr(_mirror, "download_with_mirror_fallback", mirror)
    monkeypatch.setattr(huggingface_hub, "model_info", info)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", hf)
    main_ref = Mock()
    monkeypatch.setattr(gate, "pin_main_ref", main_ref)
    cli._ensure_model_downloaded(repo)
    assert mirror.call_args.kwargs["revision"] == pin
    assert disk.call_args.kwargs.get("revision_override") == pin
    if not mirror_ok:
        assert info.call_args.kwargs.get("revision") == pin
        assert hf.call_args.kwargs["revision"] == (pin or OTHER)
    if pin:
        main_ref.assert_not_called()


def test_ensure_pin_survives_metadata_failure_and_download_failure(monkeypatch, capsys):
    import huggingface_hub

    monkeypatch.setattr(cli, "_cache_runnability", lambda _: False)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_retry_hub_download", lambda fn, _: fn())
    monkeypatch.setattr(
        huggingface_hub, "model_info", Mock(side_effect=RuntimeError("no metadata"))
    )
    hf = Mock(return_value="/snapshot")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", hf)
    cli._ensure_model_downloaded(ALIAS)
    assert hf.call_args.args == (REPO,)
    assert hf.call_args.kwargs["revision"] == PIN
    hf.side_effect = RuntimeError("pin gone")
    capsys.readouterr()
    assert cli._ensure_model_downloaded(REPO) is None
    assert hf.call_args.args == (REPO,)
    assert hf.call_args.kwargs["revision"] == PIN
    assert (
        "Pre-download skipped (RuntimeError); server will retry."
        in capsys.readouterr().out
    )


@pytest.mark.parametrize("mirror_ok", [True, False])
@pytest.mark.parametrize("name,pin", [(ALIAS, PIN), (UNPINNED, None)])
def test_pull_uses_revision_override_in_existing_pipeline(
    hub_online_env, monkeypatch, tmp_path, mirror_ok, name, pin
):
    import huggingface_hub

    from rapid_mlx import _mirror

    monkeypatch.setattr(cli, "_emit_pull_activation", lambda: None)
    monkeypatch.setattr(cli, "_print_pull_summary", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_sync_pulled_variant_marker", lambda *a: None)
    monkeypatch.setattr(
        "rapid_mlx._version_check.print_staleness_warning_if_any", lambda: None
    )
    mirror = Mock(return_value=mirror_ok)
    repo = aliases.resolve_hub_model(name)
    repo_root = tmp_path / "hub" / f"models--{repo.replace('/', '--')}"
    snapshot = snapshot_tree(repo_root, pin or OTHER)
    if pin is None:
        (repo_root / "refs").mkdir()
        (repo_root / "refs" / "main").write_text(OTHER)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(repo_root.parent))
    hf = Mock(return_value=str(snapshot))
    monkeypatch.setattr(_mirror, "download_with_mirror_fallback", mirror)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", hf)
    cli.pull_command(SimpleNamespace(model=name, bits=None, format=None, json=True))
    assert mirror.call_args.kwargs.get("revision") == pin
    if not mirror_ok:
        assert hf.call_args.kwargs.get("revision") == pin
    assert (repo_root / "refs" / "main").exists() is (pin is None)


def test_metadata_and_variant_probes_use_pin(monkeypatch, cache, pinned_name):
    import huggingface_hub

    from rapid_mlx import server

    info = Mock(return_value=SimpleNamespace(sha=OTHER, siblings=[]))
    monkeypatch.setattr(
        huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=info)
    )
    gate._model_info_with_timeout(pinned_name, 1)
    info.assert_called_once_with(REPO, files_metadata=True, revision=PIN)
    hf = Mock(return_value="/snapshot")
    monkeypatch.setattr(huggingface_hub, "model_info", info)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", hf)
    assert server._prefetch_routing_metadata(pinned_name).startswith("/snapshot")
    assert info.call_args.kwargs["revision"] == PIN
    assert hf.call_args.kwargs["revision"] == PIN
    assert info.call_args.args == (REPO,)
    assert hf.call_args.args == (REPO,)
    cli._check_disk_space(pinned_name)
    info.assert_called_with(REPO, revision=PIN, files_metadata=True)
    tree = Mock(return_value=[huggingface_hub.RepoFolder(path="8bit", oid="tree")])
    monkeypatch.setattr(
        huggingface_hub, "HfApi", lambda: SimpleNamespace(list_repo_tree=tree)
    )
    assert cli._resolve_variant_allow_patterns(REPO, "8", None, revision=PIN) == [
        "8bit/*"
    ]
    assert tree.call_args.kwargs["revision"] == PIN


def test_cached_config_prefers_pin_even_with_main(cache):
    snapshot(cache, OTHER)
    path = snapshot(cache, PIN)
    assert cli._resolve_cached_hf_config(REPO, PIN) == (str(path / "config.json"), None)
    (path / "config.json").unlink()
    assert cli._resolve_cached_hf_config(REPO, PIN)[0] is None


def test_models_json_unpinned_rows_and_swift_compatibility():
    payload = cli._available_models_json_payload()
    rows = {
        row["alias"]: row
        for bucket in ("text", "video", "image")
        for row in payload[bucket]
    }
    assert rows[ALIAS]["revision"] == PIN
    for name, profile in aliases.list_profiles().items():
        if profile.revision is None:
            assert "revision" not in rows[name]
    # These are the required keys selected by the existing dictionary reader.
    swift = Path("apps/rapid-mac/Sources/Rapid/Server/ModelCatalog.swift").read_text()
    reader = swift[
        swift.index("static func parseAvailableJSON") : swift.index(
            "private static func parseAtomicCatalogJSON"
        )
    ]
    required = {"alias", "hf_path", "is_builtin", "is_text_only"}
    assert required <= set(re.findall(r'row\["([^"]+)"\]', reader))
    for row in payload["text"]:
        assert required <= row.keys()
        # Projecting selected fields ignores the additive pin as Swift does.
        assert {key: row[key] for key in required} == {
            key: value for key, value in row.items() if key in required
        }
    assert "JSONSerialization.jsonObject" in reader


def test_kv_and_memory_config_reads_use_the_catalog_pin(
    cache, monkeypatch, pinned_name
):
    import huggingface_hub

    from rapid_mlx.byom import preflight

    snapshot(cache, OTHER)
    pinned = snapshot(cache, PIN)
    (pinned / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "pinned": True})
    )
    config, profile = cli._gather_kv_cache_dtype_inputs(pinned_name)
    assert config == {"model_type": "qwen3", "pinned": True}
    assert profile["hf_path"] == REPO
    cli._check_memory_capacity(pinned_name)
    (pinned / "config.json").unlink()
    info = Mock(return_value=SimpleNamespace(siblings=[]))
    monkeypatch.setattr(huggingface_hub, "model_info", info)
    cli._check_memory_capacity(pinned_name)
    info.assert_called_with(REPO, files_metadata=True, revision=PIN)
    (pinned / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    monkeypatch.setattr(
        preflight, "supported_model_types", lambda: frozenset({"qwen3"})
    )
    monkeypatch.setattr(preflight, "physical_ram_bytes", lambda: 64 << 30)
    from rapid_mlx.local_models import catalog_model_ready

    assert catalog_model_ready(ALIAS)
    assert cli._active_hf_snapshot_path(pinned_name) == pinned
    (cache / "refs" / "main").unlink()
    assert gate.is_repo_cached(REPO)
    assert model_metadata.resolve_pinned_cached_snapshot(REPO) == pinned


def test_pinned_alias_subfolder_inventory_uses_pin(cache, monkeypatch, pinned_name):
    profile = aliases.AliasProfile(hf_path=REPO, revision=PIN, subfolder="4bit")
    monkeypatch.setattr(aliases, "resolve_profile", lambda _: profile)
    snapshot(cache, OTHER, subfolder="4bit")
    assert cli._cached_subfolder_size(pinned_name, "4bit") is None
    pinned = snapshot(cache, PIN, subfolder="4bit")
    assert cli._cached_subfolder_size(pinned_name, "4bit") == sum(
        path.stat().st_size for path in pinned.iterdir()
    )


def test_revision_resolution_matches_local_model_precedence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    local = tmp_path / ALIAS
    local.mkdir()
    (local / "benchmark.txt").write_text("stray output")
    assert aliases.resolve_model(ALIAS) == REPO
    assert aliases.resolve_revision(ALIAS) == PIN
    for explicit in (f"./{ALIAS}", str(local)):
        assert aliases.resolve_model(explicit) == explicit
        assert aliases.resolve_revision(explicit) is None
    (local / "config.json").write_text("{}")
    assert aliases.resolve_model(ALIAS) == ALIAS
    assert aliases.resolve_revision(ALIAS) is None


def test_active_pinned_snapshot_requires_directory(cache):
    snapshot(cache, OTHER)
    assert cli._active_hf_snapshot_path(REPO) is None
    pinned = cache / "snapshots" / PIN
    pinned.write_text("not a directory")
    assert cli._active_hf_snapshot_path(REPO) is None
    pinned.unlink()
    pinned.mkdir()
    assert cli._active_hf_snapshot_path(REPO) == pinned


def test_cached_file_resolves_pinned_alias(monkeypatch, pinned_name):
    import huggingface_hub

    cached = Mock(return_value="/snapshot/config.json")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cached)
    assert model_metadata._cached_file(pinned_name, "config.json") == Path(
        "/snapshot/config.json"
    )
    cached.assert_called_once_with(REPO, "config.json", revision=PIN)


@pytest.mark.parametrize(
    "name,expected",
    [(ALIAS, PIN), (REPO, PIN), (UNPINNED, None), ("owner/model", None)],
)
def test_profile_lookup_failure_preserves_catalog_revision(monkeypatch, name, expected):
    def broken_profile(_name):
        raise RuntimeError("bad catalog")

    monkeypatch.setattr(aliases, "resolve_profile", broken_profile)
    assert aliases.resolve_revision(name) == expected


@pytest.mark.parametrize("method", ["snapshot_download", "hf_hub_download"])
def test_profile_lookup_failure_does_not_unpin_download(monkeypatch, method):
    import huggingface_hub

    def broken_profile(_name):
        raise RuntimeError("bad catalog")

    download = Mock(return_value="/snapshot")
    monkeypatch.setattr(aliases, "resolve_profile", broken_profile)
    monkeypatch.setattr(huggingface_hub, method, download)
    rest = () if method == "snapshot_download" else ("config.json",)
    getattr(model_downloads, method)(ALIAS, *rest)
    assert download.call_args.args == (REPO, *rest)
    assert download.call_args.kwargs["revision"] == PIN
    download.reset_mock()
    with pytest.raises(ValueError, match="conflicting revision"):
        getattr(model_downloads, method)(ALIAS, *rest, revision=OTHER)
    download.assert_not_called()


def test_revision_lookup_does_not_swallow_registry_validation_failure(monkeypatch):
    def invalid_registry():
        raise ValueError("revision must be a full commit")

    monkeypatch.setattr(aliases, "_load", invalid_registry)
    with pytest.raises(ValueError, match="revision must"):
        aliases.resolve_revision("owner/model")


@pytest.mark.parametrize("name", [ALIAS, REPO])
def test_launch_warm_pinned_cache_has_no_download_label(
    cache, monkeypatch, tmp_path, capsys, name
):
    import socket
    from argparse import ArgumentParser

    from rapid_mlx.agents import cli_preflight, isolated
    from rapid_mlx.launch import cli as launch_cli

    snapshot(cache, PIN)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(cli_preflight, "agent_cli_found", lambda _: True)
    monkeypatch.setattr(isolated, "configure", lambda *a, **kw: "agent-command")
    monkeypatch.setattr(isolated, "stable_key", lambda **kw: "test-key")
    monkeypatch.setattr(isolated, "running_model", lambda *a: None)
    status = Mock(side_effect=["stopped", "ready"])
    monkeypatch.setattr(isolated, "connection_status", status)
    monkeypatch.setattr("rapid_mlx.run.cli._fits_host", lambda *a: True)
    monkeypatch.setattr(cli, "_check_alias_min_memory", lambda *a: None)
    start = Mock(return_value=12345)
    monkeypatch.setattr(launch_cli, "_start_server_background", start)
    run = Mock(return_value=0)
    monkeypatch.setattr(launch_cli.subprocess, "call", run)
    forbidden = Mock(side_effect=AssertionError("warm cache must not download"))
    monkeypatch.setattr(cli, "_check_disk_space", forbidden)
    monkeypatch.setattr("huggingface_hub.snapshot_download", forbidden)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", forbidden)
    monkeypatch.setattr("rapid_mlx._mirror.download_with_mirror_fallback", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)

    # Exercise real confirmation, pin resolution and ensure-download against a
    # complete pinned snapshot while refs/main points at a different commit.
    assert cli._cache_runnability(REPO) is True
    parser = ArgumentParser()
    launch_cli.register(parser.add_subparsers())
    args = parser.parse_args(
        [
            "launch",
            "goose",
            "--isolated",
            "--run",
            "--start-server",
            "--model",
            name,
            "--port",
            "18742",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        launch_cli.launch_command(args)
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert f"Preparing {name}…" in output
    assert "download" not in output.lower()
    forbidden.assert_not_called()
    start.assert_called_once_with(name, 18742, api_key="test-key")
    run.assert_called_once()
