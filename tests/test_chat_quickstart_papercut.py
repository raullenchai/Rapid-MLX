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


def _chat_echo_lines() -> list[str]:
    """Every echo line in install.sh that mentions either chat command."""
    return [
        ln
        for ln in _quickstart_block().splitlines()
        if ln.strip().startswith("echo")
        and ("rapid-mlx chat" in ln or "rapid-mlx-chat" in ln)
    ]


def test_installer_recommends_a_command_that_works_on_a_base_install():
    """ORDER matters, not mere presence (codex on #2030): a restored bare
    rapid-mlx-chat line ABOVE the terminal one would re-open the papercut
    while a substring check stayed green. The zero-extra terminal REPL must
    be the FIRST chat command the quick-start prints — and it must name the
    SAME model the serve line recommends: a fresh-install dogfood showed the
    bare ``chat --port 8000`` resolving the client-side default alias and
    404ing against the just-served model (#2035)."""
    lines = _chat_echo_lines()
    assert lines, "quick-start no longer prints any chat command at all"
    first = lines[0]
    assert "rapid-mlx chat " in first and "--port 8000" in first, (
        "the first chat command the installer prints must be the terminal "
        f"REPL (works on a base install); got: {first!r}"
    )
    assert "${RECOMMENDED_MODEL}" in first, (
        "the chat line must reuse the exact alias the serve line recommends "
        "— an unpaired bare chat resolves the client default and 404s: "
        f"{first!r}"
    )


def test_installer_labels_every_web_ui_mention_with_its_extra():
    """EVERY echo line naming rapid-mlx-chat must carry the prerequisite on
    that same line — one unlabeled mention anywhere is the papercut back."""
    web_lines = [ln for ln in _chat_echo_lines() if "rapid-mlx-chat" in ln]
    assert web_lines, (
        "the quick-start no longer mentions rapid-mlx-chat at all — the web "
        "UI must stay discoverable, labeled with its prerequisite"
    )
    for ln in web_lines:
        assert "rapid-mlx[chat]" in ln, (
            "web-UI mention without its [chat]-extra prerequisite on the same "
            f"line: {ln!r}"
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


def test_import_without_gradio_has_no_side_effects(gradio_app_without_gradio, capsys):
    """Importing the module must never print, exec a REPL, or exit."""
    importlib.reload(gradio_app_without_gradio)  # top-level runs under capsys
    assert gradio_app_without_gradio.gr is None
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == "", (
        "importing gradio_app produced output — the module must stay "
        "side-effect free so console_scripts resolution never prints"
    )


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
    """Belt-and-braces: a REAL interpreter, the REAL argument parser.

    Codex (r3 on #2030) rightly flagged that a bare ``cli.main`` lambda left
    the fallback argv unvalidated: rename ``chat`` or ``--port`` and the stub
    stays green while the fallback dies for users. The replacement main here
    runs the fallback argv through the REAL ``build_parser()`` strictly —
    an unknown subcommand or flag makes argparse exit 2 and the test red —
    and asserts the parsed port. It deliberately does NOT run the real
    ``main()`` body: its pre-dispatch model auto-select for ``chat`` with no
    positional model touches the machine's model catalog (env-dependent —
    it failed on the hosted validate runner, which has no models), and
    routing ``chat`` to ``chat_command`` is cli's own dispatch contract,
    covered by the CLI suite.

    Building the real parser needs the package's own core deps:
    ``build_parser`` registers the ``share`` subcommand, whose module hard-
    imports ``websockets`` (a declared install dep — every pip install has
    it). pr_validate's minimal CI env does not, so probe and skip there
    rather than fail on an environment no user runs.
    """
    pytest.importorskip(
        "websockets",
        reason="cli.build_parser() needs the share subcommand's core dep",
    )
    code = (
        "import sys\n"
        "sys.modules['gradio'] = None\n"
        "import vllm_mlx.cli as cli\n"
        "def _parse_with_real_parser(argv=None):\n"
        "    args = cli.build_parser().parse_args(sys.argv[1:])\n"
        "    assert args.command == 'chat', args.command\n"
        "    print(f'REPL-DISPATCH port={args.port}')\n"
        "    return 0\n"
        "cli.main = _parse_with_real_parser\n"
        "sys.argv = ['rapid-mlx-chat']\n"
        "import vllm_mlx.gradio_app as g\n"
        "try:\n"
        "    g.main()\n"
        "except SystemExit as e:\n"
        "    sys.exit(e.code if isinstance(e.code, int) or e.code is None else 1)\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO),
    )
    assert r.returncode == 0, (r.stdout + r.stderr)[-600:]
    assert "REPL-DISPATCH port=8000" in r.stdout, (
        "the fallback argv did not reach chat_command through the real "
        f"parser; stdout={r.stdout[-300:]!r} stderr={r.stderr[-300:]!r}"
    )
    assert "terminal chat" in r.stderr
    assert "Error: gradio is required" not in r.stdout
