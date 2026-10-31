# SPDX-License-Identifier: Apache-2.0
"""CLI startup applies download progress policy before command dispatch."""

import json
import os
import subprocess
import sys
from types import ModuleType

import pytest

from rapid_mlx import cli


class DownloadReachedError(Exception):
    pass


@pytest.fixture
def cli_environment(monkeypatch, tmp_path, capsys):
    # Register the key with monkeypatch even when initially absent: main may set it.
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "temporary")
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("RAPID_MLX_PROCESS_ROLE", raising=False)
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *args, **kwargs: None)
    yield
    if "huggingface_hub" in sys.modules:
        from huggingface_hub.utils import enable_progress_bars

        enable_progress_bars()


@pytest.mark.parametrize("command", ["serve", "chat", "bench", "pull"])
def test_downloads_suppress_progress_at_entry(monkeypatch, cli_environment, command):
    from huggingface_hub.utils import enable_progress_bars

    enable_progress_bars()
    observed = []

    if command == "serve":
        server = ModuleType("rapid_mlx.server")
        server._resolve_api_key = lambda key: key
        monkeypatch.setitem(sys.modules, "rapid_mlx.server", server)
    if command == "bench":
        from rapid_mlx import _mlx_compat

        monkeypatch.setattr(_mlx_compat, "install", lambda: None)
        # These imports occur before the download but need native MLX hardware.
        for name, attributes in {
            "engine_core": ("AsyncEngineCore", "EngineConfig"),
            "pflash": ("resolve_pflash_config", "validate_model_support"),
            "request": ("SamplingParams",),
            "scheduler": ("SchedulerConfig",),
            "utils.model_file_guard": ("validate_local_model_file",),
            "utils.tokenizer": ("load_model_with_fallback",),
        }.items():
            module = ModuleType(f"rapid_mlx.{name}")
            for attribute in attributes:
                setattr(module, attribute, None)
            monkeypatch.setitem(sys.modules, module.__name__, module)

    def download(*args, **kwargs):
        env_set = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"
        live = None
        if "huggingface_hub" in sys.modules:
            from huggingface_hub.utils import are_progress_bars_disabled

            live = are_progress_bars_disabled()
        observed.append((env_set, live))
        raise DownloadReachedError

    monkeypatch.setattr(cli, "_ensure_model_downloaded", download)
    monkeypatch.setattr(cli, "_pull_repository", download)
    monkeypatch.setattr(
        sys, "argv", ["rapid-mlx", command, "mlx-community/Qwen3-0.6B-4bit"]
    )
    with pytest.raises(DownloadReachedError):
        cli.main()
    assert observed == [(True, True)]


@pytest.mark.parametrize(
    ("role", "setting", "expected"),
    [("desktop-sidecar", None, None), (None, "0", "0")],
)
def test_startup_respects_overrides(
    monkeypatch, cli_environment, role, setting, expected
):
    if role:
        monkeypatch.setenv("RAPID_MLX_PROCESS_ROLE", role)
    if setting is not None:
        monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", setting)
    monkeypatch.setattr(cli, "chat_command", lambda args: None)
    monkeypatch.setattr(
        sys, "argv", ["rapid-mlx", "chat", "mlx-community/Qwen3-0.6B-4bit"]
    )
    cli.main()
    assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == expected


def test_startup_sets_inheritable_policy_without_importing_hub(tmp_path):
    script = """
import json
import os
import sys
from rapid_mlx import cli

def hub_loaded():
    return any(m == "huggingface_hub" or m.startswith("huggingface_hub.") for m in sys.modules)

assert not hub_loaded()

from rapid_mlx import _hf_logging

original_quiet = _hf_logging.quiet_hf_progress_bars_off_terminal
loaded_at_policy = []

def quiet():
    if not loaded_at_policy:
        loaded_at_policy.append(hub_loaded())
    return original_quiet()

_hf_logging.quiet_hf_progress_bars_off_terminal = quiet

def dispatch(args):
    loaded = hub_loaded()
    bars_disabled = None
    if loaded:
        from huggingface_hub.utils import are_progress_bars_disabled
        bars_disabled = are_progress_bars_disabled()
    print(json.dumps({"env": os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"), "hub_loaded": loaded, "bars_disabled": bars_disabled, "loaded_at_policy": loaded_at_policy}))

cli.chat_command = dispatch
sys.argv = ["rapid-mlx", "chat", "mlx-community/Qwen3-0.6B-4bit"]
try:
    cli.main()
except SystemExit:
    pass
"""
    env = os.environ.copy()
    env.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    env.pop("RAPID_MLX_PROCESS_ROLE", None)
    env.update(
        RAPID_MLX_TELEMETRY="0",
        DO_NOT_TRACK="1",
        HOME=str(tmp_path),
        RAPID_MLX_STATE_DIR=str(tmp_path),
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )
    observed = json.loads(result.stdout)
    assert observed["env"] == "1"
    assert observed["loaded_at_policy"] == [False]
    if observed["hub_loaded"]:
        assert observed["bars_disabled"] is True
