from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python 3.10 test lane
    import tomli as tomllib


ROOT = Path(__file__).parents[1]
PYOBJC_PACKAGES = {
    "pyobjc-core",
    "pyobjc-framework-ApplicationServices",
    "pyobjc-framework-Cocoa",
    "pyobjc-framework-CoreText",
    "pyobjc-framework-Quartz",
}


def test_computer_use_extra_is_darwin_only_and_in_all() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    expected = {
        "pyobjc-framework-ApplicationServices==12.2.2; platform_system == 'Darwin'",
        "pyobjc-framework-Quartz==12.2.2; platform_system == 'Darwin'",
    }
    assert set(extras["computer-use"]) == expected
    assert expected <= set(extras["all"])


def test_desktop_sidecar_installs_and_smokes_computer_use_frameworks() -> None:
    script = (ROOT / "apps/rapid-mac/scripts/build-sidecar.sh").read_text()
    assert "[audio-desktop,computer-use]" in script
    assert "import ApplicationServices, Quartz" in script
    assert 'MACHO_BASELINE_COUNT="${MACHO_BASELINE_COUNT:-194}"' in script
    assert 'rm -rf "$STAGE/site-packages/PyObjCTest"' in script
    assert "-name '*.dSYM'" in script
    assert 'cp "$PYOBJC_LICENSE" "$STAGE/licenses/PyObjC-MIT.txt"' in script


def test_pyobjc_license_notice_and_desktop_inventory_are_complete() -> None:
    notice = (ROOT / "apps/rapid-mac/licenses/PyObjC-MIT.txt").read_text()
    assert "Copyright 2002, 2003 - Bill Bumgarner, Ronald Oussoren" in notice
    assert "Copyright 2003-2025 - Ronald Oussoren" in notice
    assert "Permission is hereby granted, free of charge" in notice
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in notice

    inventory = (ROOT / "apps/rapid-mac/THIRD_PARTY.md").read_text()
    for package in PYOBJC_PACKAGES:
        assert f"| {package} |" in inventory
    assert "licenses/PyObjC-MIT.txt" in inventory


def test_non_macos_import_contract_stays_lazy() -> None:
    # The package metadata excludes native frameworks off Darwin. Keep the
    # computer-use package importable there so server route registration and
    # capability discovery can report unsupported_platform at request time.
    if sys.platform == "darwin":
        return
    import rapid_mlx.computer_use  # noqa: F401
