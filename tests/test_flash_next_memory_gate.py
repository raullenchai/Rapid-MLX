"""Flash-Next must reject undersized hosts before an implicit download."""

from types import SimpleNamespace

import pytest

from rapid_mlx import cli


@pytest.mark.parametrize(
    "model_name",
    ["qwen3.8-flash-next-4bit", "rapid-mlx/Qwen3.8-Flash-Next-4bit"],
)
def test_flash_next_refuses_before_cache_probe(monkeypatch, capsys, model_name):
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: SimpleNamespace(total=64 * 1024**3)
    )
    monkeypatch.setattr(
        cli,
        "_cache_runnability",
        lambda _name: (_ for _ in ()).throw(AssertionError("cache probe ran")),
    )

    with pytest.raises(SystemExit) as exc:
        cli._ensure_model_downloaded(model_name)

    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert "requires at least 128 GB" in error
    assert "reports 64.0 GB" in error
    assert "218 GB" not in error


def test_flash_next_floor_allows_qualified_host(monkeypatch, capsys):
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: SimpleNamespace(total=128 * 1024**3)
    )

    cli._check_alias_min_memory("qwen3.8-flash-next-4bit")

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""
