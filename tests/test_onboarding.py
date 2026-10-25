"""Short first-success guidance and consent-gated integration attribution."""

import os
from types import SimpleNamespace

import pytest

from rapid_mlx import onboarding as ob
from rapid_mlx.telemetry import registry, track


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(tmp_path))


def test_readiness_is_once_and_unwritable_is_harmless(monkeypatch, tmp_path):
    ob.record_serve_success()
    marker = tmp_path / "first_serve_success"
    os.utime(marker, (100, 100))
    ob.record_serve_success()
    assert marker.stat().st_mtime == 100
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(marker))
    ob.record_serve_success()


@pytest.mark.parametrize("age,expected", [(0, 1), (86400, 1), (86401, 0), (-1, 0)])
def test_attribution_window_and_dedupe(monkeypatch, tmp_path, age, expected):
    ob.record_serve_success()
    start = (tmp_path / "first_serve_success").stat().st_mtime
    monkeypatch.setattr(ob.time, "time", lambda: start + age)
    events = []
    monkeypatch.setattr(track, "would_accept", lambda name, props: (name, props))
    monkeypatch.setattr(
        track, "_enqueue_accepted", lambda event: events.append(event) or True
    )
    ob.track_first_integration("unknown-private-name")
    ob.track_first_integration("codex")
    assert events == [("first_integration", {"agent": "other"})] * expected


def test_no_success_or_optout_does_not_claim(monkeypatch, tmp_path):
    monkeypatch.setattr(track, "would_accept", lambda *_: None)
    ob.track_first_integration("codex")
    ob.record_serve_success()
    ob.track_first_integration("codex")
    assert not (tmp_path / "first_integration_seen").exists()


def test_shown_event_and_failure_suppression(monkeypatch):
    events = []
    monkeypatch.setattr(track, "track", lambda *args: events.append(args))
    ob.track_next_step()
    assert events == [("onboarding_next_step_shown", {})]

    def fail(*args):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(track, "track", fail)
    ob.track_next_step()


def test_discovery_uses_capabilities_and_installed_binaries(monkeypatch):
    monkeypatch.setattr(
        "rapid_mlx.integrations.list_launch_targets",
        lambda: [
            SimpleNamespace(id="codex", capability="launchable"),
            SimpleNamespace(id="aider", capability="launchable"),
            SimpleNamespace(id="cline", capability="config-only"),
            SimpleNamespace(id="missing", capability="launchable"),
        ],
    )
    monkeypatch.setattr(
        "rapid_mlx.agents.get_profile",
        lambda name: (
            None
            if name == "missing"
            else SimpleNamespace(testing=SimpleNamespace(binary=name))
        ),
    )
    monkeypatch.setattr(
        "shutil.which", lambda name: "/bin/tool" if name == "codex" else None
    )
    assert ob.detected_launch_agents() == ["codex"]

    def fail():
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr("rapid_mlx.integrations.list_launch_targets", fail)
    assert ob.detected_launch_agents() == []


def test_closed_event_shapes():
    assert registry.validate("onboarding_next_step_shown", {}) is not None
    assert registry.validate("first_integration", {"agent": "codex"}) is not None
    assert registry.validate("first_integration", {"agent": "/private/path"}) is None
    assert registry.validate("onboarding_next_step_shown", {"model": "secret"}) is None


@pytest.mark.parametrize("raises", [False, True])
def test_enqueue_failure_allows_retry(monkeypatch, tmp_path, raises):
    ob.record_serve_success()
    monkeypatch.setattr(track, "would_accept", lambda *_: object())

    def fail(_event):
        if raises:
            raise RuntimeError("queue unavailable")
        return False

    monkeypatch.setattr(track, "_enqueue_accepted", fail)
    ob.track_first_integration("codex")
    assert not (tmp_path / "first_integration_seen").exists()
    monkeypatch.setattr(track, "_enqueue_accepted", lambda _: True)
    ob.track_first_integration("codex")
    assert (tmp_path / "first_integration_seen").exists()


def test_existing_configuration_success_attributes_integration(monkeypatch):
    from rapid_mlx.agents.telemetry import track_agent_configured

    events = []
    integrations = []
    monkeypatch.setattr(track, "track", lambda *args: events.append(args))
    monkeypatch.setattr(ob, "track_first_integration", integrations.append)
    track_agent_configured("codex")
    assert events == [("agent_configured", {"agent": "codex"})]
    assert integrations == ["codex"]
