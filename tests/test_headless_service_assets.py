"""Static contracts for the documented headless launchd deployment."""

from __future__ import annotations

import os
import plistlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
PLIST = ROOT / "examples/launchd/com.rapidmlx.server.plist"
SMOKE = ROOT / "scripts/headless_service_smoke.sh"
GUIDE = ROOT / "docs/guides/headless-macos-service.md"


def test_launchdaemon_template_is_valid_and_safe_by_default() -> None:
    with PLIST.open("rb") as handle:
        config = plistlib.load(handle)

    assert config["Label"] == "com.rapidmlx.server"
    assert config["UserName"] == "serveuser"
    assert config["EnvironmentVariables"]["HOME"] == "/Users/serveuser"
    assert config["ProgramArguments"][0].startswith("/Users/serveuser/")
    host_index = config["ProgramArguments"].index("--host")
    assert config["ProgramArguments"][host_index + 1] == "127.0.0.1"
    assert config["KeepAlive"] is True
    assert config["ThrottleInterval"] >= 10
    assert config["Umask"] == 0o27
    assert "ProcessType" not in config
    assert "RAPID_MLX_API_KEY" not in config["EnvironmentVariables"]
    assert config["StandardOutPath"] != config["StandardErrorPath"]


def test_smoke_script_is_syntactically_valid_and_does_not_accept_key_argv() -> None:
    subprocess.run(["bash", "-n", str(SMOKE)], check=True)
    source = SMOKE.read_text()
    assert "RAPID_MLX_API_KEY" in source
    assert "RAPID_MLX_SERVICE_DOMAIN" in source
    assert "--api-key" not in source
    assert "unsafe for a curl config" in source
    assert 'curl -q --config "$CURL_CONFIG"' in source
    assert '--max-time 1 "$BASE_URL/readyz"' in source
    assert '--max-time 1 "$BASE_URL/livez"' in source
    assert "READY_DEADLINE=$((SECONDS + 120))" in source
    assert "kill -0" not in source
    assert "launchctl print" in source
    assert "/livez" in source
    assert "/readyz" in source
    assert "did not become ready within 120 seconds" in source
    assert "/v1/chat/completions" in source


def test_smoke_script_rejects_loopback_userinfo_url() -> None:
    env = {
        **os.environ,
        "RAPID_MLX_BASE_URL": "http://127.0.0.1:8000@attacker.example",
        "RAPID_MLX_API_KEY": "secret",
    }
    result = subprocess.run([str(SMOKE)], env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "invalid or unsafe base URL" in result.stderr


def test_guide_pins_operational_and_security_boundaries() -> None:
    guide = GUIDE.read_text()
    for required in (
        "FileVault",
        "launchctl bootstrap system",
        "launchctl bootout system/com.rapidmlx.server",
        "HOME",
        "KeepAlive",
        "RAPID_MLX_API_KEY",
        "rapid-mlx doctor",
        "headless_service_smoke.sh",
        "autorestart 1",
    ):
        assert required in guide
