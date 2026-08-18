# SPDX-License-Identifier: Apache-2.0
"""The installer's quick-start must not recommend a command that errors (#2029).

A fresh base install followed the printed quick-start and hit:

    $ rapid-mlx-chat
    Error: gradio is required for the chat UI.

The base wheel deliberately omits the [chat] extra, so the installer was
recommending an out-of-the-box failure as the FIRST thing to try. Two surfaces,
two contracts pinned here:

* ``install.sh`` recommends the zero-extra terminal REPL first and labels the
  web UI with its prerequisite (the script is served from rapidmlx.com and
  synced from main, so this fix reaches users without a release);
* ``rapid-mlx-chat`` invoked BARE without gradio falls back to that same REPL,
  pointed at the same default server the web UI would have used — while any
  explicit argument (a share link, another --server-url) keeps the hard error,
  because the user asked for THIS UI specifically.
"""

import importlib
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ install.sh
def _quickstart_block() -> str:
    return (REPO / "install.sh").read_text()


def test_installer_recommends_a_command_that_works_on_a_base_install():
    s = _quickstart_block()
    assert "rapid-mlx chat --port 8000" in s, (
        "quick-start must lead with the terminal REPL, which needs no extras"
    )


def test_installer_labels_the_web_ui_with_its_extra():
    s = _quickstart_block()
    line = next(
        (
            ln
            for ln in s.splitlines()
            if "rapid-mlx-chat" in ln and ln.strip().startswith("echo")
        ),
        "",
    )
    assert "rapid-mlx[chat]" in line, (
        "if the quick-start mentions rapid-mlx-chat at all, the same line must "
        "say it needs the [chat] extra — recommending an out-of-the-box error "
        "is the exact papercut this guards against"
    )


# ------------------------------------------------------- gradio_app fallback
@pytest.fixture
def gradio_app_without_gradio(monkeypatch):
    """Reload vllm_mlx.gradio_app in a world where gradio is unimportable."""
    monkeypatch.setitem(sys.modules, "gradio", None)  # import -> ImportError
    sys.modules.pop("vllm_mlx.gradio_app", None)
    import vllm_mlx.gradio_app as mod

    mod = importlib.reload(mod)
    yield mod
    sys.modules.pop("vllm_mlx.gradio_app", None)


def test_import_without_gradio_has_no_side_effects(gradio_app_without_gradio):
    """Importing the module must never print, exec a REPL, or exit."""
    assert gradio_app_without_gradio.gr is None


def test_bare_invocation_falls_back_to_the_terminal_repl(
    gradio_app_without_gradio, monkeypatch, capsys
):
    calls = {}

    def fake_main(argv=None):
        calls["argv"] = list(sys.argv)
        return 0

    import vllm_mlx.cli as cli

    monkeypatch.setattr(cli, "main", fake_main)
    monkeypatch.setattr(sys, "argv", ["rapid-mlx-chat"])
    with pytest.raises(SystemExit) as exc:
        gradio_app_without_gradio.main()
    assert exc.value.code == 0
    # Same default target as the web UI (http://localhost:8000).
    assert calls["argv"] == ["rapid-mlx", "chat", "--port", "8000"]
    err = capsys.readouterr().err
    assert "rapid-mlx[chat]" in err and "terminal chat" in err, (
        "the fallback must say what happened and how to get the web UI"
    )


def test_explicit_arguments_keep_the_hard_error(
    gradio_app_without_gradio, monkeypatch, capsys
):
    """--share (or any arg) means the user wanted THIS UI — no substitute."""
    monkeypatch.setattr(sys, "argv", ["rapid-mlx-chat", "--share"])
    with pytest.raises(SystemExit) as exc:
        gradio_app_without_gradio.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "gradio is required" in out and "rapid-mlx[chat]" in out


def test_with_gradio_present_module_exposes_it():
    """Control: in an env WITH gradio the module must bind it, not None."""
    pytest.importorskip("gradio")
    sys.modules.pop("vllm_mlx.gradio_app", None)
    import vllm_mlx.gradio_app as mod

    mod = importlib.reload(mod)
    assert mod.gr is not None
    sys.modules.pop("vllm_mlx.gradio_app", None)


# ----------------------------------------------------------- end to end (real)
def test_real_subprocess_bare_fallback_reaches_the_repl_path():
    """Belt-and-braces: a REAL interpreter with gradio blocked, bare argv.

    The fake cli.main above proves the wiring; this proves the import-block
    trick didn't diverge from reality. The REPL would try to reach a server,
    so cut it off at cli.main with an injected stub via sitecustomize-less
    -c driver: we only assert the fallback NOTICE appears on stderr and the
    process does NOT die with the hard gradio error.
    """
    code = (
        "import sys, types\n"
        "sys.modules['gradio'] = None\n"
        "import vllm_mlx.cli as cli\n"
        "cli.main = lambda argv=None: 0\n"
        "sys.argv = ['rapid-mlx-chat']\n"
        "import vllm_mlx.gradio_app as g\n"
        "try:\n"
        "    g.main()\n"
        "except SystemExit as e:\n"
        "    sys.exit(e.code)\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO),
    )
    assert r.returncode == 0, r.stderr[-400:]
    assert "terminal chat" in r.stderr
    assert "Error: gradio is required" not in r.stdout
