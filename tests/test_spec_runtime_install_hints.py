"""Optional speculative runtimes give commands for the running Python."""

import shlex
import sys

import pytest

from rapid_mlx.speculative.ddtree import runtime
from rapid_mlx.speculative.tensorfold_runtime import INSTALL_HINT


def test_missing_ddtree_runtime_hint(monkeypatch):
    monkeypatch.setattr(runtime, "have_runtime", lambda: False)

    with pytest.raises(RuntimeError) as error:
        runtime.load_runtime(
            main_model_repo="unused",
            drafter_repo="unused",
            speculative_tokens=1,
            tree_budget=1,
        )

    assert runtime.DDTREE_INSTALL_COMMAND in str(error.value)
    assert shlex.split(runtime.DDTREE_INSTALL_COMMAND)[-1].startswith(
        "dtree-mlx @ git+"
    )


def test_tensorfold_hint_uses_active_python():
    assert shlex.quote(sys.executable) in INSTALL_HINT
