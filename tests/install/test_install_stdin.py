# SPDX-License-Identifier: Apache-2.0
"""Offline end-to-end installer tests with children that drain inherited stdin."""

import os
import pty
import subprocess
import tempfile
import unittest
from pathlib import Path

INSTALL_SH = Path(__file__).resolve().parents[2] / "install.sh"


class InstallerStdinTests(unittest.TestCase):
    def run_installer(
        self,
        *,
        bootstrap=True,
        uv=False,
        target="stable",
        file=False,
        drain="brew",
        fail=False,
        fallback=False,
    ):
        with tempfile.TemporaryDirectory(prefix="installer-4536-") as work:
            root = Path(work)
            binaries = root / "bin"
            binaries.mkdir()
            home = root / "home"
            home.mkdir()
            events = root / "events"

            def binary(name, source):
                path = binaries / name
                path.write_text(
                    "#!/bin/bash\nset -eu\n"
                    'if [ "${1-}" = --test-ready ]; then exit 0; fi\n' + source
                )
                path.chmod(0o755)
                return path

            # Deliberately exclude the host PATH: no real Python, brew, uv or curl.
            for name in ("mkdir", "cp", "chmod", "ln", "cut", "awk", "tr", "basename"):
                path = Path("/bin") / name
                if not path.exists():
                    path = Path("/usr/bin") / name
                (binaries / name).symlink_to(path)
            binary("curl", "exit 99\n")  # discovery only; network is forbidden
            binary("uname", 'case "$1" in -s) echo Darwin;; -m) echo arm64;; esac\n')
            binary("sw_vers", "echo 14.0\n")
            binary(
                "sysctl",
                'case "$*" in *hw.memsize*) echo 34359738368;; *) echo "Apple M4";; esac\n',
            )
            drain_stdin = """
if [ "$DRAIN" = "$ROLE" ]; then
    case "$*" in
        *json.load*) ;;  # this child intentionally reads the cache JSON pipe
        *) while IFS= read -r line; do echo "consumed:$ROLE" >> "$EVENTS"; done ;;
    esac
fi
"""
            binary(
                "brew",
                "ROLE=brew\n"
                + drain_stdin
                + """
case "$1" in
    install)
        echo bootstrap >> "$EVENTS"
        [ "$FAIL" = 0 ] || exit 7
        mkdir -p "$KEG/bin"
        cp "$FAKE_PY" "$KEG/bin/python3.12"
        ;;
    --prefix) echo "$KEG" ;;
esac
""",
            )
            fake_py = binary(
                "fake-python",
                """
ROLE=python
case "$*" in "-m venv "*) ROLE=venv ;; esac
"""
                + drain_stdin
                + """
case "$*" in
    *json.load*)
        IFS= read -r json
        [ "$json" = '{"cached":[]}' ]
        echo cache-json >> "$EVENTS"
        ;;
    *version_info*) echo 3.12 ;;
    *platform.machine*) echo arm64 ;;
    *"import sys"*) exit 0 ;;
    --version) echo "Python 3.12.0" ;;
    "-m venv "*)
        echo venv >> "$EVENTS"
        mkdir -p "$3/bin"
        cp "$FAKE_PY" "$3/bin/python"
        cp "$FAKE_PY" "$3/bin/python3"
        cp "$FAKE_PIP" "$3/bin/pip"
        ;;
esac
""",
            )
            fake_pip = binary(
                "fake-pip",
                """
ROLE=pip
case "$*" in *rapid-mlx*) ROLE=pip-package ;; esac
"""
                + drain_stdin
                + """
if [ "$1" = --version ]; then echo "pip 25"; exit 0; fi
case "$*" in
    *"--upgrade pip"*) echo pip-upgrade >> "$EVENTS" ;;
    *)
        echo "package:$*" >> "$EVENTS"
        [ "$FAIL" = 0 ] || exit 7
        [ "$FALLBACK" = 0 ] || exit 7
        cp "$FAKE_CLI" "$HOME/.rapid-mlx/bin/rapid-mlx"
        ;;
esac
""",
            )
            fake_cli = binary(
                "fake-cli",
                """
case "$1" in
    models) echo '{"cached":[]}' ;;
    --version) echo 0.16.1 ;;
esac
""",
            )
            if not bootstrap:
                (binaries / "python3.12").symlink_to(fake_py)
            if uv:
                binary(
                    "uv",
                    "ROLE=uv\n"
                    + drain_stdin
                    + """
echo "uv:$*" >> "$EVENTS"
[ "$FAIL" = 0 ] || exit 7
[ "$FALLBACK" = 0 ] || exit 7
cp "$FAKE_CLI" "$HOME/.rapid-mlx/bin/rapid-mlx"
""",
                )
            env = {
                "PATH": str(binaries),
                "HOME": str(home),
                "SHELL": "/bin/bash",
                "EVENTS": str(events),
                "KEG": str(root / "keg"),
                "FAKE_PY": str(fake_py),
                "FAKE_PIP": str(fake_pip),
                "FAKE_CLI": str(fake_cli),
                "DRAIN": drain,
                "FAIL": str(int(fail)),
                "FALLBACK": str(int(fallback)),
                "RAPID_MLX_GIT_COMMIT": "a" * 40,
                "TMPDIR": os.environ.get("TMPDIR", work),
            }
            # Newly generated executables can incur macOS first-launch validation.
            # Establish fixture readiness before the installer's stdin-hang budget;
            # the installer itself retains the same ten-second timeout below.
            for path in binaries.iterdir():
                if path.is_symlink():
                    continue
                subprocess.run(
                    [str(path), "--test-ready"],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    env=env,
                    check=True,
                    timeout=30,
                )
            command = (
                ["/bin/bash", str(INSTALL_SH), target]
                if file
                else ["/bin/bash", "-s", "--", target]
            )
            result = subprocess.run(
                command,
                input=None if file else INSTALL_SH.read_text(),
                text=True,
                capture_output=True,
                env=env,
                timeout=10,
            )
            log = events.read_text() if events.exists() else ""
            if fallback:
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("git+", log)
                self.assertIn("refusing", result.stderr)
                self.assertNotIn("Rapid-MLX installed!", result.stdout)
                return
            if fail:
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("Rapid-MLX installed!", result.stdout)
                return
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            if bootstrap:
                self.assertIn("bootstrap\n", log)
            self.assertIn(
                "venv\n", log, f"exit={result.returncode}\n{log}\n{result.stdout}"
            )
            self.assertIn("pip-upgrade\n", log)
            self.assertIn("uv:" if uv else "package:", log)
            package_commands = [
                line
                for line in log.splitlines()
                if line.startswith("uv:" if uv else "package:")
            ]
            if target == "stable":
                self.assertIn(" --upgrade ", package_commands[0])
                self.assertIn(" rapid-mlx", package_commands[0])
            elif target == "latest":
                flag = "--reinstall" if uv else "--force-reinstall"
                self.assertIn(flag, package_commands[0])
                self.assertIn(
                    "rapid-mlx @ git+https://github.com/", package_commands[0]
                )
                self.assertIn("@" + "a" * 40, package_commands[0])
            else:
                self.assertIn(f"rapid-mlx=={target}", package_commands[0])
            self.assertNotIn("consumed:", log)
            self.assertIn(
                "cache-json\n", log
            )  # legitimate pipeline still supplies JSON
            self.assertTrue((home / ".local/bin/rapid-mlx").is_symlink())
            self.assertIn("Rapid-MLX installed!", result.stdout)
            self.assertIn("Quick start:", result.stdout)
            self.assertIn("rapid-mlx chat qwen3.5-4b-4bit", result.stdout)

    def test_streamed_bootstrap(self):
        self.run_installer()

    def test_streamed_existing_python(self):
        self.run_installer(bootstrap=False, drain="python")

    def test_streamed_venv(self):
        self.run_installer(bootstrap=False, drain="venv")

    def test_streamed_pip_targets(self):
        for target in ("stable", "latest", "0.16.1"):
            with self.subTest(target=target):
                self.run_installer(bootstrap=False, drain="pip", target=target)

    def test_streamed_package_targets(self):
        for target in ("stable", "latest", "0.16.1"):
            with self.subTest(target=target):
                self.run_installer(bootstrap=False, drain="pip-package", target=target)

    def test_streamed_package_failure_does_not_change_sources(self):
        for uv in (False, True):
            for target in ("stable", "0.16.1"):
                with self.subTest(target=target, uv=uv):
                    self.run_installer(
                        bootstrap=False,
                        uv=uv,
                        drain="pip-package",
                        target=target,
                        fallback=True,
                    )

    def test_source_failure(self):
        for uv in (False, True):
            with self.subTest(uv=uv):
                self.run_installer(bootstrap=False, uv=uv, target="latest", fail=True)

    def test_streamed_uv_targets(self):
        for target in ("stable", "latest", "0.16.1"):
            with self.subTest(target=target):
                self.run_installer(bootstrap=False, uv=True, drain="uv", target=target)

    def test_file_invocation(self):
        self.run_installer(bootstrap=False, file=True, drain="none")

    def test_bootstrap_failure(self):
        self.run_installer(fail=True)

    def test_package_failure(self):
        self.run_installer(bootstrap=False, drain="none", fail=True)

    def test_terminal_input_is_preserved(self):
        master, slave = pty.openpty()
        try:
            os.write(master, b"interactive-input\n")
            result = subprocess.run(
                [
                    "/bin/bash",
                    "-c",
                    """
RAPID_INSTALL_LIB=1 source "$1"
run_install_command /bin/bash -c 'IFS= read -r reply; printf "%s" "$reply"'
""",
                    "test",
                    str(INSTALL_SH),
                ],
                stdin=slave,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "interactive-input")
        finally:
            os.close(master)
            os.close(slave)


if __name__ == "__main__":
    unittest.main()
