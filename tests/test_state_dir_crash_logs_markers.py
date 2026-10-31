# SPDX-License-Identifier: Apache-2.0
from rapid_mlx import _signal_observability as observability
from rapid_mlx.telemetry import server_start


def test_override_routes_crash_logs_and_markers(monkeypatch, tmp_path):
    home = tmp_path / "home"
    root = tmp_path / "state-root"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(root))

    assert observability._crash_logs_dir() == root / "logs"
    assert server_start._marker_path().parent == root / "state"

    _, owns_marker = server_start._begin_inflight_marker()
    assert owns_marker
    monkeypatch.setattr(server_start, "_owns_inflight_marker", owns_marker)
    try:
        assert server_start._marker_path().is_file()
        observability._install_crash_file()
        assert len(list((root / "logs").glob("crash-*.txt"))) == 1
        assert not (home / ".rapid-mlx").exists()
    finally:
        observability.release_crash_file_after_clean_shutdown()
        server_start._remove_inflight_marker()
    assert not server_start._marker_path().exists()
    assert not (home / ".rapid-mlx").exists()


def test_default_paths_use_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("RAPID_MLX_STATE_DIR", raising=False)

    assert observability._crash_logs_dir() == home / ".rapid-mlx" / "logs"
    assert server_start._marker_path().parent == home / ".rapid-mlx" / "state"
