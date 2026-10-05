# SPDX-License-Identifier: Apache-2.0
"""Setup reports saved identity and distinguishes it from live/client checks."""

import json
from argparse import Namespace

import pytest

from rapid_mlx.agents import get_profile
from rapid_mlx.agents.adapter import codex_setup_report, setup_agent_config


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    profile = get_profile("codex")
    setup_agent_config(profile, "http://127.0.0.1:18872/v1", "local-qwen")
    return profile, tmp_path


def test_report_reads_saved_config_and_preserved_profile(configured):
    profile, root = configured
    config = root / "config.toml"
    config.write_text('profile = "work"\n' + config.read_text())
    before = config.read_bytes()
    report = codex_setup_report(profile)
    assert "Model: local-qwen" in report
    assert "Endpoint: http://127.0.0.1:18872/v1" in report
    assert "model ID matches" in report
    assert "Default profile: work (preserved; may override" in report
    assert "--model/--config flags can override" in report
    assert config.read_bytes() == before


def test_report_rejects_alias_catalog_mismatch(configured):
    profile, root = configured
    catalog_path = root / "rapid-mlx-model-catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["models"][0]["slug"] = "organization/local-qwen"
    catalog_path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="missing from its Codex model catalog"):
        codex_setup_report(profile)


def test_report_does_not_print_credentials_or_url_query(configured):
    profile, root = configured
    config_path = root / "config.toml"
    config_path.write_text(
        config_path.read_text().replace(
            "http://127.0.0.1:18872/v1",
            "http://user:private-password@127.0.0.1:18872/v1?key=private-query#private-fragment",
        )
    )
    report = codex_setup_report(profile)
    assert "Endpoint: http://127.0.0.1:18872/v1" in report
    assert "private-" not in report


@pytest.mark.parametrize(
    "contents,reason",
    [
        ('model = "local-qwen"\n', "no model or model_provider"),
        ('model = "local-qwen"\nmodel_provider = "missing"\n', "no base_url"),
    ],
)
def test_report_rejects_incomplete_saved_config(configured, contents, reason):
    profile, root = configured
    (root / "config.toml").write_text(contents)
    with pytest.raises(ValueError, match=reason):
        codex_setup_report(profile)


@pytest.mark.parametrize("catalog", [{"models": "invalid"}, ["invalid"]])
def test_report_rejects_invalid_catalog_shape(configured, catalog):
    profile, root = configured
    (root / "rapid-mlx-model-catalog.json").write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="missing from its Codex model catalog"):
        codex_setup_report(profile)


def test_report_warns_when_custom_config_omits_catalog(configured):
    profile, root = configured
    path = root / "config.toml"
    path.write_text(
        "\n".join(
            line
            for line in path.read_text().splitlines()
            if not line.startswith("model_catalog_json")
        )
    )
    assert "not configured; Codex may use fallback" in codex_setup_report(profile)


def test_cli_fresh_setup_reports_readback(tmp_path, monkeypatch, capsys):
    from rapid_mlx.cli import agents_command

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr(
        "rapid_mlx.agents.adapter.fetch_context_window", lambda *_: 32768
    )
    agents_command(_args(no_check=True))
    output = capsys.readouterr().out
    assert "configured!" in output
    assert "Saved Codex defaults" in output


def _args(**overrides):
    values = dict(
        agent_name="codex",
        setup=True,
        test=False,
        model="local-qwen",
        base_url="http://127.0.0.1:18872/v1",
        agent_version=None,
        dry_run=False,
        no_check=False,
        yes=True,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize("no_check", [False, True])
def test_cli_reports_saved_defaults_and_real_check_status(
    configured, monkeypatch, capsys, no_check
):
    from rapid_mlx.cli import agents_command

    monkeypatch.setattr(
        "rapid_mlx.agents.adapter.fetch_context_window", lambda *_: 32768
    )
    checks = []

    def verify(*args, **kwargs):
        checks.append(args)
        return "local-qwen"

    monkeypatch.setattr("rapid_mlx.agents.setup.verify_server", verify)
    agents_command(_args(no_check=no_check))
    output = capsys.readouterr().out
    assert "Saved Codex defaults" in output
    assert "Model: local-qwen" in output
    assert bool(checks) is not no_check
    assert (
        "Connection check skipped" if no_check else "Connection check passed"
    ) in output


def test_cli_dry_run_does_not_report_readback_or_write(configured, monkeypatch, capsys):
    from rapid_mlx.cli import agents_command

    profile, root = configured
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    monkeypatch.setattr(
        "rapid_mlx.agents.adapter.fetch_context_window", lambda *_: 32768
    )
    agents_command(_args(dry_run=True, model="other-model"))
    output = capsys.readouterr().out
    assert "Dry run only" in output
    assert "Saved Codex defaults" not in output
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_cli_does_not_claim_success_if_saved_catalog_is_unreadable(
    configured, monkeypatch, capsys
):
    from rapid_mlx.cli import agents_command

    monkeypatch.setattr(
        "rapid_mlx.agents.adapter.fetch_context_window", lambda *_: 32768
    )

    # Simulate a read-back failure after the writer returned, rather than
    # corrupting an operator's real configuration or changing its permissions.
    def unreadable(*_):
        raise OSError("catalog unavailable")

    monkeypatch.setattr("rapid_mlx.agents.adapter.codex_setup_report", unreadable)
    with pytest.raises(SystemExit) as raised:
        agents_command(_args(no_check=True))
    assert raised.value.code == 1
    output = capsys.readouterr().out
    assert "config verification failed" in output
    assert "configured!" not in output
