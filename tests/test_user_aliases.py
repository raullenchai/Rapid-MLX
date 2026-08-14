from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_mlx.user_aliases import (
    UserAliasError,
    config_path,
    load_user_aliases,
    remove_user_alias,
    set_user_alias,
    validated_user_aliases,
)


@pytest.fixture
def alias_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "config" / "user-aliases.json"
    monkeypatch.setenv("RAPID_MLX_USER_ALIASES_FILE", str(path))
    return path


BUILTINS = {
    "smart": "mlx-community/Qwen3.5-9B-4bit",
    "fast": "mlx-community/LFM2.5-1.2B-Instruct-4bit",
}


def test_set_list_remove_round_trip_is_atomic_and_private(alias_file: Path) -> None:
    set_user_alias("daily", "smart", BUILTINS)

    assert validated_user_aliases(BUILTINS) == {"daily": "smart"}
    payload = json.loads(alias_file.read_text())
    assert payload == {"version": 1, "aliases": {"daily": "smart"}}
    assert alias_file.stat().st_mode & 0o777 == 0o600
    assert alias_file.parent.stat().st_mode & 0o777 == 0o700
    assert not list(alias_file.parent.glob("*.tmp"))

    assert remove_user_alias("daily", BUILTINS) is True
    assert load_user_aliases() == {}
    assert remove_user_alias("daily", BUILTINS) is False


@pytest.mark.parametrize(
    "name",
    ["-option", "../escape", "two words", "tool\nspoof", "éclair", "a" * 65],
)
def test_rejects_unsafe_alias_names(alias_file: Path, name: str) -> None:
    with pytest.raises(UserAliasError):
        set_user_alias(name, "smart", BUILTINS)
    assert not alias_file.exists()


@pytest.mark.parametrize(
    "target", ["-option", "../escape", "org/repo/extra", "two words", "org/répo"]
)
def test_rejects_unsafe_targets(alias_file: Path, target: str) -> None:
    with pytest.raises(UserAliasError):
        set_user_alias("mine", target, BUILTINS)


def test_rejects_builtin_collision_and_alias_chains(alias_file: Path) -> None:
    with pytest.raises(UserAliasError, match="reserved built-in"):
        set_user_alias("smart", "fast", BUILTINS)

    set_user_alias("mine", "smart", BUILTINS)
    with pytest.raises(UserAliasError, match="chains are not allowed"):
        set_user_alias("second", "mine", BUILTINS)


def test_rejects_separately_reserved_retired_name(alias_file: Path) -> None:
    with pytest.raises(UserAliasError, match="reserved built-in"):
        set_user_alias("retired", "fast", BUILTINS, frozenset({"retired"}))


def test_corrupt_config_fails_closed(alias_file: Path) -> None:
    alias_file.parent.mkdir(parents=True)
    alias_file.write_text('{"version": 1, "aliases": ')
    with pytest.raises(UserAliasError, match="cannot read"):
        validated_user_aliases(BUILTINS)


def test_symlinked_config_is_rejected(alias_file: Path, tmp_path: Path) -> None:
    alias_file.parent.mkdir(parents=True)
    victim = tmp_path / "victim.json"
    victim.write_text("do not replace")
    alias_file.symlink_to(victim)

    with pytest.raises(UserAliasError, match="symlinked"):
        set_user_alias("mine", "smart", BUILTINS)
    assert victim.read_text() == "do not replace"


def test_remove_never_touches_target_weights(alias_file: Path, tmp_path: Path) -> None:
    weights = tmp_path / "models--org--repo" / "weights.safetensors"
    weights.parent.mkdir()
    weights.write_bytes(b"weights")
    set_user_alias("mine", "org/repo", BUILTINS)

    assert remove_user_alias("mine", BUILTINS)
    assert weights.read_bytes() == b"weights"


def test_config_path_override_is_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAPID_MLX_USER_ALIASES_FILE", "~/aliases-test.json")
    assert config_path() == Path.home() / "aliases-test.json"


def test_model_registry_resolves_user_alias_through_shared_choke_point(
    alias_file: Path,
) -> None:
    from vllm_mlx.model_aliases import (
        list_builtin_aliases,
        list_profiles,
        resolve_model,
        resolve_profile,
    )

    builtins = list_builtin_aliases()
    set_user_alias("my-smart", "qwen3.5-9b-4bit", builtins)
    set_user_alias("my-repo", "example/model", builtins)

    assert resolve_model("my-smart") == builtins["qwen3.5-9b-4bit"]
    assert resolve_model("my-repo") == "example/model"
    assert resolve_profile("my-smart") == resolve_profile("qwen3.5-9b-4bit")
    assert resolve_profile("my-repo").hf_path == "example/model"
    assert list_profiles()["my-smart"].hf_path == builtins["qwen3.5-9b-4bit"]


def test_cli_alias_commands_share_the_same_store(
    alias_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from vllm_mlx.cli import alias_command, build_parser

    parser = build_parser()
    alias_command(parser.parse_args(["alias", "set", "daily", "qwen3.5-9b-4bit"]))
    alias_command(parser.parse_args(["alias", "list"]))
    assert "daily -> qwen3.5-9b-4bit" in capsys.readouterr().out

    alias_command(parser.parse_args(["alias", "remove", "daily"]))
    assert "cached weights were not changed" in capsys.readouterr().out
    assert json.loads(alias_file.read_text())["aliases"] == {}
