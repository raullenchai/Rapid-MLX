"""Image Hub upload failures point to the active Python environment."""

import builtins
import runpy
import sys
from pathlib import Path

import pytest

from rapid_mlx.connect import package_install_command


def test_missing_hub_dependency_for_image_upload(monkeypatch):
    hub_path = Path(__file__).parents[1] / "rapid_mlx/image/sdxl_runtime/hub.py"
    push_folder = runpy.run_path(str(hub_path))["push_folder"]
    original_import = builtins.__import__

    def missing_hub(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("hub unavailable for this test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_hub)

    with pytest.raises(ImportError) as error:
        push_folder("unused", "unused/repo")

    assert package_install_command("huggingface_hub") in str(error.value)


@pytest.mark.requires_mlx
def test_missing_sdxl_tokenizer_dependency(monkeypatch):
    from rapid_mlx.image.sdxl_runtime.pipelines.sdxl import _load_tokenizer

    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(ImportError) as error:
        _load_tokenizer(Path("unused"))

    assert package_install_command("transformers") in str(error.value)
