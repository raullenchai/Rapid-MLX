"""Share's missing WebSocket dependency identifies the running Python."""

import builtins
import runpy

import pytest

from rapid_mlx.connect import package_install_command


def test_missing_websockets_dependency(monkeypatch):
    original_import = builtins.__import__

    def missing_websockets(name, *args, **kwargs):
        if name == "websockets":
            raise ImportError("websockets unavailable for this test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_websockets)

    with pytest.raises(ImportError) as error:
        runpy.run_module("rapid_mlx.share.ws_tunnel")

    assert package_install_command("websockets") in str(error.value)
