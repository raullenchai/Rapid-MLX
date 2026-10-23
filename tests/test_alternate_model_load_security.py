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
