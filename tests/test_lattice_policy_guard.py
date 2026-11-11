"""Keep device/OS optimization policy in the lattice registry."""

import ast
import re
from pathlib import Path


def inline_hardware_policy(source):
    violations = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        if all(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
            continue  # Optional-argument selection is not hardware policy.
        text = ast.unparse(node)
        literals = [
            n.value
            for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        ]
        if (
            any(re.search(r"\b(?:Apple )?M[1-9]\d*\b|applegpu_", s) for s in literals)
            or any(
                getattr(n, "attr", getattr(n, "id", ""))
                in {
                    "gpu_generation",
                    "chip_generation",
                    "macos",
                    "macos_version",
                    "metal_architecture",
                    "QUALIFIED_METAL_ARCHITECTURES",
                }
                for n in ast.walk(node)
                if isinstance(n, (ast.Name, ast.Attribute))
            )
            or "mac_ver(" in text
            or any(s in {"device_name", "architecture"} for s in literals)
        ):
            violations.append(node.lineno)
    return violations


def test_no_inline_hardware_policy():
    root = Path(__file__).resolve().parents[1] / "rapid_mlx"
    # These existing fact readers classify/report hardware, rather than choose
    # optimization policy. They are explicitly outside this migration's scope.
    fact_readers = {"chip_tier.py", "optimizations.py", "telemetry/chip.py"}
    found = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if relative.parts[0] == "lattice" or str(relative) in fact_readers:
            continue
        found.extend(
            f"{relative}:{line}" for line in inline_hardware_policy(path.read_text())
        )
    assert not found, "Move hardware policy to lattice: " + ", ".join(found)


def test_guard_detects_reintroduced_policy():
    for source in [
        'if info.get("device_name") == "Apple M5 Max": pass',
        'if arch in {"applegpu_g15d"}: pass',
        'if platform.mac_ver()[0] == "26.5.2": pass',
        'if "M5" in name: pass',
        "if ctx.hardware.gpu_generation >= 17: pass",
    ]:
        assert inline_hardware_policy(source)
