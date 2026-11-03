"""Release toolchain selection and ordering contracts."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SELECTOR = "./.github/actions/select-release-xcode"
DESKTOP = "./.github/actions/desktop-releasable"


def load(root, path):
    return yaml.safe_load((root / path).read_text())


def assert_selected_before_builds(root, steps, selected=False):
    for step in steps:
        uses = step.get("uses", "")
        if uses == SELECTOR:
            assert "if" not in step
            selected = True
        elif uses.startswith("./"):
            selected = assert_selected_before_builds(
                root, load(root, uses[2:] + "/action.yml")["runs"]["steps"], selected
            )
        # Comments describing a build do not execute toolchain commands.
        commands = "\n".join(
            line.split("#", 1)[0] for line in step.get("run", "").splitlines()
        )
        if re.search(r"swift |xcodebuild|build\.sh", commands):
            assert selected, f"Release Xcode must precede {step.get('name', commands)}"
    return selected


def test_release_xcode_has_one_non_overridable_pin():
    path = ROOT / SELECTOR[2:] / "action.yml"
    text = path.read_text()
    action = yaml.safe_load(text)
    assert not action.get("inputs")
    assert re.findall(r"\b\d+\.\d+(?:\.\d+)?\b", text) == ["16.4"]
    assert action["runs"]["using"] == "composite"
    steps = action["runs"]["steps"]
    assert len(steps) == 1
    assert steps[0]["shell"] == "bash"
    script = steps[0]["run"]
    for required in (
        "set -euo pipefail",
        'readonly RELEASE_XCODE_VERSION="16.4"',
        'app="/Applications/Xcode_${RELEASE_XCODE_VERSION}.app"',
        '[[ ! -d "$app" ]]',
        'sudo xcode-select -s "$app/Contents/Developer"',
        'version="$(xcodebuild -version)"',
        "swift --version",
        '''"${version%%$'\\n'*}" != "Xcode $RELEASE_XCODE_VERSION"''',
    ):
        assert required in script
    assert script.count("::error::") == 2
    assert script.count("exit 1") == 2


def test_release_lanes_select_before_toolchain_use():
    composite = load(ROOT, DESKTOP[2:] + "/action.yml")
    assert composite["runs"]["steps"][0] == {"uses": SELECTOR}
    for workflow, name in (
        ("rapid-mac-ci.yml", "release-toolchain-build"),
        ("auto-release.yml", "desktop-candidate-gate"),
        ("rapid-mac-release.yml", "build"),
    ):
        job = load(ROOT, ".github/workflows/" + workflow)["jobs"][name]
        assert job["runs-on"] == "macos-15"
        steps = job["steps"]
        assert steps[0]["uses"].startswith("actions/checkout@")
        assert steps[0].get("with", {}).get("path", ".") == "."
        assert "if" not in steps[0]
        expected = SELECTOR if name == "release-toolchain-build" else DESKTOP
        assert any(s.get("uses") == expected for s in steps)
        assert assert_selected_before_builds(ROOT, steps)
