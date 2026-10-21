"""DDTree missing-runtime errors provide the active Python install command."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.requires_mlx


def test_missing_ddtree_runtime_shows_active_python_install(monkeypatch, capsys):
    from rapid_mlx import cli
    from rapid_mlx.speculative.ddtree import runtime, server

    monkeypatch.setattr(
        "rapid_mlx.speculative.ddtree.eligibility.have_runtime", lambda: False
    )
    monkeypatch.setattr(runtime, "have_runtime", lambda: False)
    monkeypatch.setattr(server, "have_runtime", lambda: False)

    args = SimpleNamespace(
        model="qwen3.5-9b-8bit",
        _original_alias=None,
        speculative_config='{"method":"ddtree"}',
        enable_ddtree=False,
        enable_dflash=False,
        spec_decode="none",
        suffix_decoding=False,
        no_spec_decode=False,
    )
    cli._normalize_speculative_config_or_exit(args)
    with pytest.raises(SystemExit, match="1"):
        cli._preflight_ddtree_or_exit(args)
    assert runtime.DDTREE_INSTALL_COMMAND in capsys.readouterr().out

    with pytest.raises(RuntimeError, match="DDTree server requires") as error:
        server.run_ddtree_server(
            main_model_repo="unused",
            drafter_repo="unused",
            speculative_tokens=1,
            tree_budget=1,
            host="127.0.0.1",
            port=0,
            served_model_name="unused",
            default_max_tokens=1,
            cors_origins=[],
            uvicorn_log_level="warning",
        )
    assert runtime.DDTREE_INSTALL_COMMAND in str(error.value)
