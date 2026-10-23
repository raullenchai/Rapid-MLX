# SPDX-License-Identifier: Apache-2.0
"""Production model entry points must enforce the shared code-consent gate."""

import json

import pytest


@pytest.mark.requires_mlx
@pytest.mark.parametrize("entry", ["openjev", "rerank", "hidream"])
def test_alternate_loaders_reject_repository_python(tmp_path, monkeypatch, entry):
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    marker = tmp_path / "executed"
    (tmp_path / "custom.py").write_text(
        f"open({str(marker)!r}, 'w').write('executed')\n"
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "model_file": "custom.py",
            }
        )
    )
    if entry == "openjev":
        from rapid_mlx.system_one.openjev import load_openjev

        invoke = lambda: load_openjev(str(tmp_path))
    elif entry == "rerank":
        from rapid_mlx.rerank import RerankEngine

        monkeypatch.setattr(
            "rapid_mlx.model_downloads.require_local", lambda *_a, **_kw: None
        )
        invoke = lambda: RerankEngine(str(tmp_path), model_family="qwen3").load()
    else:
        from rapid_mlx.image.hidream_runtime.runtime import HiDreamO1

        invoke = lambda: HiDreamO1(str(tmp_path))
    with pytest.raises(ValueError, match="model_file.*executable Python"):
        invoke()
    assert not marker.exists()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("entry", ["dflash", "native_mtp"])
def test_speculative_target_refuses_code_before_listener(tmp_path, monkeypatch, entry):
    from dataclasses import replace

    import mlx_vlm

    from rapid_mlx.speculative.dflash import server as dflash
    from rapid_mlx.speculative.native_mtp import server as native
    from rapid_mlx.speculative.native_mtp.eligibility import QWEN36_35B_4BIT

    (tmp_path / "config.json").write_text(
        '{"model_type":"qwen3_5","model_file":"custom.py"}'
    )
    (tmp_path / "custom.py").write_text('raise AssertionError("must not import")')
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    monkeypatch.setattr(dflash, "have_runtime", lambda: True)
    monkeypatch.setattr(
        mlx_vlm, "load", lambda *a, **k: pytest.fail("unchecked model load")
    )
    monkeypatch.setattr(
        "rapid_mlx._uvicorn._run_server",
        lambda *a, **k: pytest.fail("listener must not start"),
    )
    options = dict(
        host="127.0.0.1",
        port=58998,
        served_model_name="local",
        default_max_tokens=1,
        cors_origins=[],
        uvicorn_log_level="warning",
    )
    with pytest.raises(ValueError, match="executable Python"):
        if entry == "dflash":
            dflash.run_dflash_server(
                main_model_repo=str(tmp_path), drafter_repo="local/draft", **options
            )
        else:
            native.run_native_mtp_server(
                pair=replace(QWEN36_35B_4BIT, target_repo=str(tmp_path)), **options
            )
