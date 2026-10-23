"""Pin resolution is read-only and fails independently of mirror availability."""

import runpy
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError, RevisionNotFoundError

from scripts import check_image_pins


def test_image_pins_includes_auxiliary_and_unlisted_repositories(monkeypatch):
    from rapid_mlx import _download_gate as gate

    monkeypatch.setattr(gate, "IMAGE_MODEL_REVISIONS", {"org/primary": "a" * 40})
    monkeypatch.setattr(
        gate,
        "IMAGE_MODEL_RUNTIME_ASSETS",
        {"org/primary": (("org/auxiliary", "b" * 40, ("tokenizer.json",)),)},
    )
    assert check_image_pins.image_pins() == [
        ("org/auxiliary", "b" * 40),
        ("org/primary", "a" * 40),
    ]


@pytest.mark.parametrize(
    "result,expected",
    [
        (SimpleNamespace(sha="a" * 40), None),
        (SimpleNamespace(sha="b" * 40), "IMAGE PIN CHECK FAILED"),
        (
            RevisionNotFoundError(
                "gone",
                response=httpx.Response(
                    404, request=httpx.Request("GET", "https://example.test")
                ),
            ),
            "DANGLING IMAGE PIN",
        ),
        (
            HfHubHTTPError(
                "outage",
                response=httpx.Response(
                    503, request=httpx.Request("GET", "https://example.test")
                ),
            ),
            "IMAGE PIN CHECK FAILED",
        ),
        (TimeoutError("timeout"), "IMAGE PIN CHECK FAILED"),
    ],
)
def test_resolution_checks_exact_pin_and_alerts(result, expected, monkeypatch, capsys):
    monkeypatch.setattr(
        check_image_pins, "image_pins", lambda: [("org/image", "a" * 40)]
    )
    with patch("huggingface_hub.model_info") as info:
        if isinstance(result, Exception):
            info.side_effect = result
        else:
            info.return_value = result
        assert check_image_pins.main() == int(expected is not None)
        info.assert_called_once_with("org/image", revision="a" * 40)
    if expected:
        assert expected in capsys.readouterr().err


def test_script_entrypoint(monkeypatch):
    from rapid_mlx import _download_gate as gate

    monkeypatch.setattr(gate, "IMAGE_MODEL_REVISIONS", {})
    monkeypatch.setattr(gate, "IMAGE_MODEL_RUNTIME_ASSETS", {})
    monkeypatch.setattr(
        "sys.path", [str(check_image_pins.ROOT / "scripts"), *__import__("sys").path]
    )
    with pytest.raises(SystemExit, match="0"):
        runpy.run_path(check_image_pins.__file__, run_name="__main__")
