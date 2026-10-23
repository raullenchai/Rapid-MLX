"""LTX-2.5 external MLX runtime integration tests."""

from __future__ import annotations

import importlib.util
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from rapid_mlx.model_aliases import resolve_profile
from rapid_mlx.runtime import video_lane
from rapid_mlx.runtime.optional_runtime import OptionalRuntimeMissing
from rapid_mlx.runtime.video_lane import VideoEngine, VideoRuntimeError
from rapid_mlx.video import ltx25

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_SCRIPT = REPO_ROOT / "scripts" / "vendor_ltx25.py"


def test_ltx25_alias_routes_to_video_lane() -> None:
    profile = resolve_profile("ltx-2.5-mlx-q8")
    assert profile is not None
    assert profile.hf_path == "MrMofer/ltx-2.5-mlx-q8"
    assert profile.modality == "video-gen"
    assert profile.min_memory_gb == 24
    assert ltx25.is_ltx25_model("org/my-ltx25-experiment") is False


def test_ltx25_capabilities_match_distilled_controls() -> None:
    from rapid_mlx.routes.video import _video_capabilities

    capabilities = _video_capabilities(
        SimpleNamespace(model_name="MrMofer/ltx-2.5-mlx-q8", video_family="ltx-2.5")
    )
    assert capabilities["family"] == "ltx-2.5"
    assert capabilities["limits"]["size"]["width"]["multiple_of"] == 32
    assert capabilities["limits"]["workload"]["dimension_rounding"] == "ceil_to_32"
    assert capabilities["controls"]["guidance_scale"] is None
    assert capabilities["controls"]["negative_prompt"] is False
    assert capabilities["controls"]["conditioning_strength"] == {
        "minimum": 0.0,
        "maximum": 1.0,
    }


@pytest.fixture(autouse=True)
def _reset_ltx25_runtime_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep process-private caches out of other tests."""
    monkeypatch.setattr(ltx25, "_RUNTIME_CACHE", {})
    monkeypatch.setattr(ltx25, "_TREE_DIGEST_CACHE", {})


def _load_vendor_script():
    spec = importlib.util.spec_from_file_location("vendor_ltx25", VENDOR_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _provenance_text(package: str, tree_sha256: str, **overrides: str) -> str:
    values = {
        "REPOSITORY": ltx25.LTX25_RUNTIME_REPOSITORY,
        "COMMIT": ltx25.LTX25_RUNTIME_COMMIT,
        "VERSION": ltx25.LTX25_RUNTIME_VERSION,
        "PACKAGE": package,
        "TREE_SHA256": tree_sha256,
    }
    values.update(overrides)
    body = "".join(f"{name} = {value!r}\n" for name, value in values.items())
    return '"""Provenance for the vendored LTX-2.5 runtime package."""\n\n' + body


def _vendored_tree(
    root: Path,
    *,
    packages: tuple[str, ...] = ltx25.LTX25_VENDORED_PACKAGES,
    provenance: dict[str, dict[str, str]] | None = None,
    omit_provenance: frozenset[str] = frozenset(),
    with_cli: bool = True,
) -> dict[str, Path]:
    """Create vendored package directories with real content and real digests.

    The provenance digest is computed by the same vendoring script that ships
    the runtime, so a ``trusted`` fixture proves the probe validates real file
    content rather than four labels.
    """
    vendor = _load_vendor_script()
    directories: dict[str, Path] = {}
    for package in packages:
        directory = root / package
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text(f'"""Vendored {package}."""\n')
        (directory / "core.py").write_text(f"VALUE = {package!r}\n")
        if package == "ltx_pipelines_mlx" and with_cli:
            (directory / "cli.py").write_text("def main():\n    return 0\n")
        # Byte-code caches are not part of the audited tree; writing them here
        # proves the digest ignores both ``__pycache__`` and loose ``.pyc``
        # files exactly as the vendoring script does.
        (directory / "__pycache__").mkdir(exist_ok=True)
        (directory / "__pycache__" / "stale.pyc").write_bytes(b"\0")
        (directory / "stray.pyc").write_bytes(b"\0")
        if package not in omit_provenance:
            overrides = dict((provenance or {}).get(package, {}))
            tree_sha256 = overrides.pop("TREE_SHA256", vendor.tree_digest(directory))
            (directory / ltx25.LTX25_PROVENANCE_FILE).write_text(
                _provenance_text(package, tree_sha256, **overrides)
            )
        directories[package] = directory
    return directories


def _fake_package_specs(
    monkeypatch: pytest.MonkeyPatch,
    directories: dict[str, Path],
    *,
    install_root: Path,
    origins: dict[str, str] | None = None,
    locations: dict[str, list[str]] | None = None,
) -> None:
    """Resolve the vendored packages from *directories* inside *install_root*."""

    def find_spec(name: str, *_args: object, **_kwargs: object) -> object:
        directory = directories.get(name)
        if directory is None:
            return None
        origin = (origins or {}).get(name, str(directory / "__init__.py"))
        search = (locations or {}).get(name, [str(directory)])
        return SimpleNamespace(origin=origin, submodule_search_locations=search)

    monkeypatch.setattr(ltx25.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(ltx25, "_install_root", lambda: install_root)


def test_ltx25_runtime_trusted_vendored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "site-packages"
    directories = _vendored_tree(root)
    _fake_package_specs(monkeypatch, directories, install_root=root)

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "trusted"
    assert runtime.source == "vendored"
    assert runtime.executable == ltx25.sys.executable
    assert runtime.detail is None
    assert ltx25.embedded_ltx25_interpreter() == ltx25.sys.executable


def test_ltx25_runtime_absent_when_no_vendored_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "site-packages"
    root.mkdir()
    _fake_package_specs(monkeypatch, {}, install_root=root)

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "absent"
    assert runtime.source == "vendored"
    assert runtime.executable is None
    assert "ltx_core_mlx" in (runtime.detail or "")
    assert "ltx_pipelines_mlx" in (runtime.detail or "")
    assert ltx25.embedded_ltx25_interpreter() is None


def test_ltx25_runtime_partially_missing_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "site-packages"
    directories = _vendored_tree(root, packages=("ltx_core_mlx",))
    _fake_package_specs(monkeypatch, directories, install_root=root)

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "corrupt"
    assert runtime.detail == "missing vendored package(s): ltx_pipelines_mlx"
    assert ltx25.embedded_ltx25_interpreter() is None


def test_ltx25_runtime_shadowed_copy_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_root = tmp_path / "site-packages"
    install_root.mkdir()
    shadow = tmp_path / "elsewhere"
    directories = _vendored_tree(shadow)
    _fake_package_specs(monkeypatch, directories, install_root=install_root)

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "corrupt"
    assert runtime.reason == "shadowed"
    assert "resolves outside this rapid-mlx installation" in (runtime.detail or "")
    assert ltx25.embedded_ltx25_interpreter() is None


def test_ltx25_runtime_foreign_origin_is_shadowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A foreign module origin must not pass just because a search location is in-root."""
    root = tmp_path / "site-packages"
    directories = _vendored_tree(root)
    foreign = tmp_path / "foreign" / "ltx_core_mlx" / "__init__.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("")
    _fake_package_specs(
        monkeypatch,
        directories,
        install_root=root,
        origins={"ltx_core_mlx": str(foreign)},
    )

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "corrupt"
    assert runtime.reason == "shadowed"
    assert "module origin" in (runtime.detail or "")


def test_ltx25_runtime_multiple_search_locations_is_shadowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "site-packages"
    directories = _vendored_tree(root)
    extra = tmp_path / "extra" / "ltx_pipelines_mlx"
    extra.mkdir(parents=True)
    _fake_package_specs(
        monkeypatch,
        directories,
        install_root=root,
        locations={
            "ltx_pipelines_mlx": [
                str(directories["ltx_pipelines_mlx"]),
                str(extra),
            ]
        },
    )

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "corrupt"
    assert runtime.reason == "shadowed"
    assert "import locations" in (runtime.detail or "")


def test_ltx25_runtime_overwritten_vendored_file_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "site-packages"
    directories = _vendored_tree(root)
    _fake_package_specs(monkeypatch, directories, install_root=root)
    assert ltx25.ltx25_runtime().state == "trusted"

    (directories["ltx_core_mlx"] / "core.py").write_text("VALUE = 'replaced'\n")

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "corrupt"
    assert runtime.reason == "damaged"
    assert "no longer match the audited snapshot" in (runtime.detail or "")
    assert ltx25.embedded_ltx25_interpreter() is None


def test_ltx25_runtime_deleted_vendored_file_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "site-packages"
    directories = _vendored_tree(root)
    _fake_package_specs(monkeypatch, directories, install_root=root)
    assert ltx25.ltx25_runtime().state == "trusted"

    (directories["ltx_pipelines_mlx"] / "core.py").unlink()

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "corrupt"
    assert runtime.reason == "damaged"
    assert "no longer match the audited snapshot" in (runtime.detail or "")


def test_ltx25_runtime_added_vendored_file_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "site-packages"
    directories = _vendored_tree(root)
    _fake_package_specs(monkeypatch, directories, install_root=root)
    assert ltx25.ltx25_runtime().state == "trusted"

    (directories["ltx_core_mlx"] / "smuggled.py").write_text("VALUE = 1\n")

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "corrupt"
    assert runtime.reason == "damaged"
    assert "no longer match the audited snapshot" in (runtime.detail or "")


def test_ltx25_runtime_renamed_vendored_file_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rename keeps mtime, size and file count, so the cache key must name files."""
    root = tmp_path / "site-packages"
    directories = _vendored_tree(root)
    _fake_package_specs(monkeypatch, directories, install_root=root)
    assert ltx25.ltx25_runtime().state == "trusted"

    os.rename(
        directories["ltx_core_mlx"] / "core.py",
        directories["ltx_core_mlx"] / "renamed_core.py",
    )

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "corrupt"
    assert runtime.reason == "damaged"
    assert "no longer match the audited snapshot" in (runtime.detail or "")
    assert ltx25.embedded_ltx25_interpreter() is None


def test_ltx25_runtime_unreadable_vendored_files_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "site-packages"
    directories = _vendored_tree(root)
    _fake_package_specs(monkeypatch, directories, install_root=root)

    def unreadable(_package_dir: Path) -> str:
        raise OSError("permission denied")

    monkeypatch.setattr(ltx25, "_tree_digest", unreadable)

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "corrupt"
    assert runtime.reason == "damaged"
    assert "files could not be read for verification" in (runtime.detail or "")


def test_ltx25_runtime_missing_provenance_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "site-packages"
    directories = _vendored_tree(root, omit_provenance=frozenset({"ltx_core_mlx"}))
    _fake_package_specs(monkeypatch, directories, install_root=root)

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "corrupt"
    assert "no usable _vendored.py provenance" in (runtime.detail or "")


def test_ltx25_runtime_malformed_provenance_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "site-packages"
    directories = _vendored_tree(root)
    (directories["ltx_pipelines_mlx"] / ltx25.LTX25_PROVENANCE_FILE).write_text(
        f"COMMIT = {ltx25.LTX25_RUNTIME_COMMIT!r}\n"
    )
    _fake_package_specs(monkeypatch, directories, install_root=root)

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "corrupt"
    assert "no usable _vendored.py provenance" in (runtime.detail or "")


@pytest.mark.parametrize(
    "override",
    [
        {"REPOSITORY": "https://example.invalid/other.git"},
        {"COMMIT": "0" * 40},
        {"VERSION": "0.14.16"},
        {"PACKAGE": "ltx_pipelines_mlx"},
    ],
)
def test_ltx25_runtime_mismatched_provenance_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: dict[str, str]
) -> None:
    root = tmp_path / "site-packages"
    directories = _vendored_tree(root, provenance={"ltx_core_mlx": override})
    _fake_package_specs(monkeypatch, directories, install_root=root)

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "corrupt"
    ((name, value),) = override.items()
    assert f"provenance {name} is {value!r}" in (runtime.detail or "")


def test_ltx25_runtime_missing_cli_entry_point_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "site-packages"
    directories = _vendored_tree(root, with_cli=False)
    _fake_package_specs(monkeypatch, directories, install_root=root)

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "corrupt"
    assert (
        runtime.detail == "ltx_pipelines_mlx.cli is missing from the vendored package."
    )


def test_ltx25_runtime_find_spec_failure_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "site-packages"
    root.mkdir()

    def broken_find_spec(*_args: object, **_kwargs: object) -> object:
        raise ImportError("broken metadata")

    monkeypatch.setattr(ltx25.importlib.util, "find_spec", broken_find_spec)
    monkeypatch.setattr(ltx25, "_install_root", lambda: root)

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "absent"
    assert "missing vendored package(s)" in (runtime.detail or "")


def test_ltx25_runtime_explicit_override_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_path = tmp_path / "ltx-2-mlx"
    runtime_path.write_text("#!/bin/sh\n")
    runtime_path.chmod(0o755)
    monkeypatch.setenv("RAPID_MLX_LTX25_RUNTIME", str(runtime_path))
    monkeypatch.setattr(
        ltx25, "_runtime_revision", lambda _: ltx25.LTX25_RUNTIME_COMMIT
    )
    monkeypatch.setattr(
        ltx25,
        "_vendored_runtime",
        lambda: pytest.fail("an explicit override must not consult the bundled copy"),
    )

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "trusted"
    assert runtime.source == "override"
    assert runtime.executable == str(runtime_path)


def test_ltx25_runtime_invalid_override_never_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAPID_MLX_LTX25_RUNTIME", str(tmp_path / "missing"))
    monkeypatch.setattr(
        ltx25,
        "_vendored_runtime",
        lambda: pytest.fail("a broken override must not fall through to the bundle"),
    )

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "override_invalid"
    assert runtime.source == "override"
    assert ltx25.LTX25_RUNTIME_COMMIT in (runtime.detail or "")


def test_ltx25_runtime_python_floor_precedes_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ltx25.sys, "version_info", (3, 10))
    monkeypatch.setenv("RAPID_MLX_LTX25_RUNTIME", "/anywhere")
    monkeypatch.setattr(
        ltx25,
        "resolve_ltx25_runtime",
        lambda: pytest.fail("the Python floor must fail before resolving a runtime"),
    )

    runtime = ltx25.ltx25_runtime()

    assert runtime.state == "python_unsupported"
    assert runtime.source is None


def test_ltx25_runtime_unavailable_message_for_vendored_failure() -> None:
    for state in ("absent", "corrupt"):
        message = ltx25.runtime_unavailable_message(
            ltx25.LTX25Runtime(state=state, source="vendored")
        )
        assert "missing or damaged" in message
        assert "Reinstall rapid-mlx" in message


def test_ltx25_runtime_unavailable_message_for_python_floor() -> None:
    message = ltx25.runtime_unavailable_message(
        ltx25.LTX25Runtime(state="python_unsupported")
    )
    assert "Python 3.11 or newer" in message
    assert "missing or damaged" not in message
    assert "pip install --upgrade --force-reinstall" not in message


def test_ltx25_runtime_unavailable_message_for_invalid_override() -> None:
    message = ltx25.runtime_unavailable_message(
        ltx25.LTX25Runtime(
            state="override_invalid", source="override", detail="custom detail"
        )
    )
    assert "custom detail" in message
    assert "unset RAPID_MLX_LTX25_RUNTIME" in message
    assert "Reinstall rapid-mlx" not in message

    fallback = ltx25.runtime_unavailable_message(
        ltx25.LTX25Runtime(state="override_invalid", source="override")
    )
    assert "The LTX-2.5 runtime override is invalid." in fallback


def test_ltx25_runtime_unavailable_message_for_shadowed_copy() -> None:
    message = ltx25.runtime_unavailable_message(
        ltx25.LTX25Runtime(
            state="corrupt",
            source="vendored",
            detail=(
                "ltx_pipelines_mlx resolves outside this rapid-mlx installation "
                "(/opt/other/ltx_pipelines_mlx); a shadowing copy is not trusted."
            ),
            reason="shadowed",
        )
    )
    assert "Remove the shadowing path from PYTHONPATH" in message
    assert "/opt/other/ltx_pipelines_mlx" in message
    assert "Reinstall rapid-mlx" not in message


def test_ltx25_runtime_preflight_needs_no_checkout_or_uv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_lane.sys, "version_info", (3, 11))
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: ltx25.LTX25Runtime(
            state="trusted", source="vendored", executable="/embedded/python"
        ),
    )
    monkeypatch.setattr(
        ltx25,
        "resolve_ltx25_runtime",
        lambda: pytest.fail("the bundled runtime must not inspect a checkout"),
    )
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "/sidecar/ffmpeg")
    monkeypatch.setattr(video_lane.shutil, "which", lambda _: None)
    video_lane.require_video_runtime_or_exit("ltx-2.5-mlx-q8")


def test_ltx25_embedded_generation_preserves_sidecar_python_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: ltx25.LTX25Runtime(
            state="trusted", source="vendored", executable="/embedded/python"
        ),
    )
    monkeypatch.setenv("PYTHONHOME", "/signed-sidecar/python")
    monkeypatch.setenv("PYTHONPATH", "/signed-sidecar/site-packages")
    captured: dict = {}

    class Process:
        returncode = 0

        def __init__(self, command: list[str], **kwargs) -> None:
            self.command = command
            captured.update(command=command, kwargs=kwargs)

        def communicate(self, *, input: str, timeout: int) -> None:
            captured.update(prompt=input, timeout=timeout)
            Path(self.command[self.command.index("--output") + 1]).write_bytes(b"mp4")

    monkeypatch.setattr(ltx25.subprocess, "Popen", Process)
    output = tmp_path / "result.mp4"
    ltx25.LTX25VideoEngine("ltx-2.5-mlx-q8").generate(
        prompt="private prompt",
        output_path=output,
        width=704,
        height=480,
        num_frames=97,
        fps=24,
        seed=7,
        image=None,
    )

    assert captured["command"][0] == "/embedded/python"
    # The bundled runtime pins the audited package root and keeps the working
    # directory off ``sys.path``.
    assert captured["command"][1] == "-P"
    assert captured["command"][3] == ltx25._STDIN_PROMPT_RUNNER
    assert captured["command"][4] == str(ltx25._install_root())
    assert "private prompt" not in captured["command"]
    assert captured["prompt"] == "private prompt"
    assert captured["kwargs"]["env"]["PYTHONHOME"] == "/signed-sidecar/python"
    assert captured["kwargs"]["env"]["PYTHONPATH"] == "/signed-sidecar/site-packages"
    assert output.read_bytes() == b"mp4"


def test_ltx25_direct_engine_rejects_conditioning_without_image() -> None:
    engine = VideoEngine.__new__(VideoEngine)
    engine._ltx25_engine = object()

    with pytest.raises(VideoRuntimeError, match="requires an input image"):
        engine.generate(
            prompt="fox",
            output_path=Path("unused.mp4"),
            width=704,
            height=480,
            num_frames=97,
            fps=24,
            seed=7,
            image=None,
            conditioning_strength=0.5,
        )


def _runtime(
    state: str,
    source: str | None = None,
    executable: str | None = None,
    detail: str | None = None,
    reason: str | None = None,
) -> ltx25.LTX25Runtime:
    return ltx25.LTX25Runtime(
        state=state,
        source=source,
        executable=executable,
        detail=detail,
        reason=reason,
    )


def test_ltx25_runtime_preflight_absent_uses_reinstall_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_lane.sys, "version_info", (3, 11))
    monkeypatch.setattr(ltx25, "ltx25_runtime", lambda: _runtime("absent", "vendored"))
    monkeypatch.setattr(
        video_lane, "_resolve_ffmpeg", lambda: "/opt/homebrew/bin/ffmpeg"
    )
    monkeypatch.setattr(video_lane.shutil, "which", lambda name: "/usr/bin/uv")

    with pytest.raises(OptionalRuntimeMissing) as exc:
        video_lane.require_video_runtime_or_exit("MrMofer/ltx-2.5-mlx-q8")

    error = exc.value.format_user_message()
    assert "ships with rapid-mlx but is missing or damaged" in error
    assert "git clone" not in error
    assert exc.value.status == "broken"
    assert exc.value.marker_reason == "runtime_dependency_missing"


def test_ltx25_runtime_preflight_corrupt_uses_reinstall_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_lane.sys, "version_info", (3, 11))
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: _runtime(
            "corrupt", "vendored", detail="ltx_core_mlx provenance COMMIT is wrong"
        ),
    )
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video_lane.shutil, "which", lambda name: "/usr/bin/uv")

    with pytest.raises(OptionalRuntimeMissing) as exc:
        video_lane.require_video_runtime_or_exit("MrMofer/ltx-2.5-mlx-q8")

    error = exc.value.format_user_message()
    assert "the LTX-2.5 runtime shipped with rapid-mlx" in error
    assert "ships with rapid-mlx but is missing or damaged" in error
    assert exc.value.status == "broken"


def test_ltx25_runtime_preflight_shadowed_uses_removal_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shadowing copy must be removed, not reinstalled over."""
    monkeypatch.setattr(video_lane.sys, "version_info", (3, 11))
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: _runtime(
            "corrupt",
            "vendored",
            detail=(
                "ltx_pipelines_mlx resolves outside this rapid-mlx installation "
                "(/opt/other/ltx_pipelines_mlx); a shadowing copy is not trusted."
            ),
            reason="shadowed",
        ),
    )
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video_lane.shutil, "which", lambda name: "/usr/bin/uv")

    with pytest.raises(OptionalRuntimeMissing) as exc:
        video_lane.require_video_runtime_or_exit("MrMofer/ltx-2.5-mlx-q8")

    error = exc.value.format_user_message()
    assert "a different copy shadows it" in error
    assert "Remove the conflicting LTX-2.5 copy from PYTHONPATH" in error
    assert "/opt/other/ltx_pipelines_mlx" in error
    assert "reinstall" not in error.lower()
    assert "git clone" not in error
    assert exc.value.status == "broken"


def test_ltx25_runtime_preflight_invalid_override_points_at_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_lane.sys, "version_info", (3, 11))
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: _runtime(
            "override_invalid",
            "override",
            detail="RAPID_MLX_LTX25_RUNTIME does not point at the pinned revision.",
        ),
    )
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "/usr/bin/ffmpeg")

    with pytest.raises(OptionalRuntimeMissing) as exc:
        video_lane.require_video_runtime_or_exit("MrMofer/ltx-2.5-mlx-q8")

    error = exc.value.format_user_message()
    assert "RAPID_MLX_LTX25_RUNTIME" in error
    assert ltx25.LTX25_RUNTIME_COMMIT in error
    assert "docs/guides/video-generation.md" in error
    assert "ships with rapid-mlx" not in error
    assert exc.value.status == "broken"


def test_ltx25_runtime_preflight_python_floor_uses_upgrade_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections import namedtuple

    Version = namedtuple("Version", "major minor")
    monkeypatch.setattr(video_lane.sys, "version_info", Version(3, 10))
    monkeypatch.setattr(ltx25, "ltx25_runtime", lambda: _runtime("python_unsupported"))

    with pytest.raises(OptionalRuntimeMissing) as exc:
        video_lane.require_video_runtime_or_exit("MrMofer/ltx-2.5-mlx-q8")

    error = exc.value.format_user_message()
    assert "Python 3.11 or newer" in error
    assert "uv tool install --force --python 3.12" in error
    assert "pip install --upgrade --force-reinstall" not in error
    assert "ships with rapid-mlx" not in error
    assert exc.value.status == "incompatible"
    assert exc.value.marker_reason == "python_version_unsupported"


def test_ltx25_runtime_preflight_requires_uv_for_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_lane.sys, "version_info", (3, 11))
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: _runtime("trusted", "override", executable="/runtime/ltx-2-mlx"),
    )
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video_lane.shutil, "which", lambda name: None)

    with pytest.raises(OptionalRuntimeMissing) as exc:
        video_lane.require_video_runtime_or_exit("MrMofer/ltx-2.5-mlx-q8")

    error = exc.value.format_user_message()
    assert "uv (`brew install uv`)" in error
    # The runtime itself resolved, so the clone/checkout walkthrough is noise.
    assert "git clone" not in error


def test_ltx25_runtime_preflight_ffmpeg_only_uses_ffmpeg_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_lane.sys, "version_info", (3, 11))
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: _runtime("trusted", "vendored", executable="/embedded/python"),
    )
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: None)

    with pytest.raises(OptionalRuntimeMissing) as exc:
        video_lane.require_video_runtime_or_exit("MrMofer/ltx-2.5-mlx-q8")

    error = exc.value.format_user_message()
    assert "ffmpeg (`brew install ffmpeg`)" in error
    assert "Install ffmpeg so LTX-2.5 can encode" in error
    # The bundled runtime is present; reinstall guidance would be a lie.
    assert "ships with rapid-mlx" not in error
    assert "Reinstall rapid-mlx" not in error


def test_ltx25_runtime_preflight_runtime_blocker_dominates_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_lane.sys, "version_info", (3, 11))
    monkeypatch.setattr(ltx25, "ltx25_runtime", lambda: _runtime("absent", "vendored"))
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: None)

    with pytest.raises(OptionalRuntimeMissing) as exc:
        video_lane.require_video_runtime_or_exit("MrMofer/ltx-2.5-mlx-q8")

    error = exc.value.format_user_message()
    assert "the LTX-2.5 runtime shipped with rapid-mlx" in error
    assert "ffmpeg (`brew install ffmpeg`)" in error
    assert "ships with rapid-mlx but is missing or damaged" in error


def test_ltx25_runtime_preflight_catalog_is_side_effect_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_lane.sys, "version_info", (3, 11))
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: _runtime("trusted", "override", executable="/runtime/ltx-2-mlx"),
    )
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video_lane.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(
        ltx25,
        "prepare_ltx25_runtime",
        lambda _: pytest.fail("the model catalog must not provision a runtime"),
    )

    assert (
        video_lane.video_runtime_issue("MrMofer/ltx-2.5-mlx-q8", provision=False)
        is None
    )
    assert video_lane.video_alias_runtime_ready("ltx-2.5-mlx-q8") is True


def test_ltx25_runtime_preflight_catalog_marks_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_lane.sys, "version_info", (3, 11))
    monkeypatch.setattr(ltx25, "ltx25_runtime", lambda: _runtime("absent", "vendored"))
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "/usr/bin/ffmpeg")

    assert video_lane.video_alias_runtime_ready("ltx-2.5-mlx-q8") is False


def test_ltx25_provisioning_failure_surfaces_cause_not_clone_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_lane.sys, "version_info", (3, 11))
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: _runtime("trusted", "override", executable="/runtime/ltx-2-mlx"),
    )

    def boom(executable: str) -> None:
        # Mirrors the message prepare_ltx25_runtime() actually raises for a
        # failed `uv sync` (see test_ltx25_provisioning_error_includes_uv_
        # failure_detail, which pins the production construction).
        raise ltx25.LTX25BackendError(
            "The pinned LTX-2.5 runtime could not be provisioned: "
            "`uv sync --frozen` failed with exit code 2 "
            "(error: lockfile out of date)"
        )

    monkeypatch.setattr(ltx25, "prepare_ltx25_runtime", boom)
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video_lane.shutil, "which", lambda name: "/usr/bin/uv")

    with pytest.raises(OptionalRuntimeMissing) as exc:
        video_lane.require_video_runtime_or_exit("MrMofer/ltx-2.5-mlx-q8")

    error = exc.value.format_user_message()
    assert "a provisioned pinned LTX-2.5 runtime" in error
    # The underlying failure reason is the actionable part.
    assert "`uv sync --frozen` failed with exit code 2" in error
    assert "lockfile out of date" in error
    assert "docs/guides/video-generation.md" in error
    # The checkout already resolved — re-cloning is not the fix.
    assert "git clone" not in error


def test_ltx25_provisioning_error_includes_uv_failure_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ltx25, "_RUNTIME_CACHE", {})
    monkeypatch.setattr(ltx25.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(ltx25, "_materialize_runtime", lambda repo, dest: None)

    def failing_run(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(
            2, ["uv", "sync"], stderr="error: lockfile out of date\n"
        )

    monkeypatch.setattr(ltx25.subprocess, "run", failing_run)

    with pytest.raises(ltx25.LTX25BackendError) as excinfo:
        ltx25.prepare_ltx25_runtime("/checkout/.venv/bin/ltx-2-mlx")

    message = str(excinfo.value)
    assert "could not be provisioned" in message
    assert "`uv sync --frozen` failed with exit code 2" in message
    assert "lockfile out of date" in message


def test_ltx25_provisioning_detail_sanitizes_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control sequences and index credentials in uv output must not reach
    the terminal (escape injection / credential exposure)."""
    monkeypatch.setattr(ltx25, "_RUNTIME_CACHE", {})
    monkeypatch.setattr(ltx25.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(ltx25, "_materialize_runtime", lambda repo, dest: None)

    def failing_run(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(
            1,
            ["uv", "sync"],
            stderr=(
                "\x1b[31merror\x1b[0m: failed to fetch "
                "https://build:s3cret@index.example/simple/foo\n"
            ),
        )

    monkeypatch.setattr(ltx25.subprocess, "run", failing_run)

    with pytest.raises(ltx25.LTX25BackendError) as excinfo:
        ltx25.prepare_ltx25_runtime("/checkout/.venv/bin/ltx-2-mlx")

    message = str(excinfo.value)
    assert "\x1b" not in message
    assert "s3cret" not in message
    assert "https://***@index.example/simple/foo" in message


def test_ltx25_sanitize_redacts_percent_encoded_userinfo() -> None:
    from rapid_mlx.video.ltx25 import _sanitize_diagnostic

    out = _sanitize_diagnostic(
        "failed to fetch https://build%40corp:s3cret@index.example/simple/"
    )
    assert "s3cret" not in out
    assert "build%40corp" not in out
    assert "https://***@index.example/simple/" in out


def test_ltx25_sanitize_redacts_query_tokens_and_bearer() -> None:
    from rapid_mlx.video.ltx25 import _sanitize_diagnostic

    out = _sanitize_diagnostic(
        "fetch https://index.example/simple?token=s3cret&x=1 failed; "
        "api_key=abc123 Authorization: Bearer eyJhbGci.xyz"
    )
    assert "s3cret" not in out
    assert "abc123" not in out
    assert "eyJhbGci" not in out
    assert "https://index.example/simple?token=***" in out


def test_ltx25_sanitize_redacts_signed_url_params() -> None:
    """Signed-URL auth params don't end in the classic credential
    suffixes (codex on #2166): AWS presigned ``X-Amz-Signature``, Azure
    SAS ``sig``/``sas``, generic ``auth``/``jwt`` must all redact, since
    uv stderr can echo the full index URL query string."""
    from rapid_mlx.video.ltx25 import _sanitize_diagnostic

    out = _sanitize_diagnostic(
        "GET https://bucket.s3.example/wheel.whl"
        "?X-Amz-Credential=AKIA%2F20260820&X-Amz-Signature=deadbeefcafe"
        "&X-Amz-Expires=300 and https://acct.blob.example/pkg?sv=2024"
        "&sig=Zm9vYmFy%3D and auth=topsecret9 jwt=eyJ0eXAi.abc"
    )
    assert "deadbeefcafe" not in out
    assert "Zm9vYmFy" not in out
    assert "topsecret9" not in out
    assert "eyJ0eXAi" not in out
    assert "X-Amz-Signature=***" in out
    assert "sig=***" in out
    # Non-credential params survive so the diagnostic stays useful.
    assert "X-Amz-Expires=300" in out


def test_ltx25_sanitize_redacts_quoted_credential_values() -> None:
    from rapid_mlx.video.ltx25 import _sanitize_diagnostic

    out = _sanitize_diagnostic(
        "config error: token=\"quoted-s3cret\" password='single-s3cret' left"
    )
    assert "quoted-s3cret" not in out
    assert "single-s3cret" not in out
    assert "left" in out


def test_ltx25_sanitize_redacts_prefixed_credential_names() -> None:
    from rapid_mlx.video.ltx25 import _sanitize_diagnostic

    out = _sanitize_diagnostic(
        "access_token=at-s3cret client_secret=cs-s3cret "
        "HF_TOKEN=hf-s3cret AWS_SECRET_ACCESS_KEY=aws-s3cret api-key: ak-s3cret"
    )
    for leak in ("at-s3cret", "cs-s3cret", "hf-s3cret", "aws-s3cret", "ak-s3cret"):
        assert leak not in out
    assert "access_token=***" in out
    assert "client_secret=***" in out


def test_ltx25_sanitize_redacts_basic_auth_header() -> None:
    from rapid_mlx.video.ltx25 import _sanitize_diagnostic

    out = _sanitize_diagnostic(
        "request failed; Authorization: Basic dXNlcjpwYXNz and "
        "authorization: Digest response-s3cret"
    )
    assert "dXNlcjpwYXNz" not in out
    assert "response-s3cret" not in out
    assert "Authorization: Basic ***" in out


def test_ltx25_oserror_detail_is_sanitized_and_bounded() -> None:
    from rapid_mlx.video.ltx25 import _provisioning_failure_detail

    exc = OSError(
        "\x1b[31mdisk full\x1b[0m at https://user:tok3n@mirror.example/x " + "p" * 500
    )
    detail = _provisioning_failure_detail(exc)
    assert "\x1b" not in detail
    assert "tok3n" not in detail
    assert len(detail) <= 301


def test_ltx25_stderr_tail_is_bounded() -> None:
    """Only a bounded tail of uv stderr is ever read back into memory."""
    import tempfile

    with tempfile.TemporaryFile() as f:
        f.write(b"x" * 100_000 + b"\nfinal line\n")
        tail = ltx25._stderr_tail(f)

    assert len(tail) <= 4096
    assert "final line" in tail


def test_ltx25_provisioning_timeout_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ltx25, "_RUNTIME_CACHE", {})
    monkeypatch.setattr(ltx25.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(ltx25, "_materialize_runtime", lambda repo, dest: None)

    def timing_out_run(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(["uv", "sync"], 1800)

    monkeypatch.setattr(ltx25.subprocess, "run", timing_out_run)

    with pytest.raises(ltx25.LTX25BackendError) as excinfo:
        ltx25.prepare_ltx25_runtime("/checkout/.venv/bin/ltx-2-mlx")

    assert "timed out after 1800s" in str(excinfo.value)


def test_ltx25_runtime_override_must_be_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "ltx-2-mlx"
    runtime.write_text("#!/bin/sh\n")
    monkeypatch.setenv("RAPID_MLX_LTX25_RUNTIME", str(runtime))
    monkeypatch.setattr(
        ltx25, "_runtime_revision", lambda _: ltx25.LTX25_RUNTIME_COMMIT
    )
    monkeypatch.setattr(
        ltx25.shutil,
        "which",
        lambda _: pytest.fail(
            "an invalid explicit override must not fall back to PATH"
        ),
    )

    assert ltx25.resolve_ltx25_runtime() is None

    runtime.chmod(0o755)
    assert ltx25.resolve_ltx25_runtime() == str(runtime)


def test_ltx25_runtime_rejects_unpinned_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / ".venv" / "bin" / "ltx-2-mlx"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o755)
    monkeypatch.setenv("RAPID_MLX_LTX25_RUNTIME", str(runtime))
    monkeypatch.setattr(ltx25, "_runtime_revision", lambda _: "wrong-revision")

    assert ltx25.resolve_ltx25_runtime() is None


def test_ltx25_materialization_ignores_checkout_and_replace_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "runtime"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    tracked = repository / "tracked.py"
    tracked.write_text("safe = True\n")
    (repository / "uv.lock").write_text("version = 1\n")
    subprocess.run(
        ["git", "-C", str(repository), "add", "tracked.py", "uv.lock"], check=True
    )
    tree = subprocess.run(
        ["git", "-C", str(repository), "write-tree"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "-C", str(repository), "commit-tree", tree, "-m", "fixture"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(repository), "update-ref", "HEAD", commit], check=True
    )
    tracked.write_text("raise RuntimeError('replaced')\n")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.py"], check=True)
    replacement_tree = subprocess.run(
        ["git", "-C", str(repository), "write-tree"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    replacement_commit = subprocess.run(
        ["git", "-C", str(repository), "commit-tree", replacement_tree, "-m", "evil"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(repository), "replace", commit, replacement_commit],
        check=True,
    )
    (repository / "untracked.py").write_text("raise RuntimeError('unsafe')\n")
    destination = tmp_path / "snapshot"
    destination.mkdir()

    monkeypatch.setattr(ltx25, "LTX25_RUNTIME_COMMIT", commit)
    ltx25._materialize_runtime(repository, destination)

    assert (destination / "tracked.py").read_text() == "safe = True\n"
    assert not (destination / "untracked.py").exists()


def test_ltx25_materialization_wraps_malformed_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    malformed = subprocess.CompletedProcess([], 0, stdout=b"not a tar", stderr=b"")
    monkeypatch.setattr(ltx25.subprocess, "run", lambda *args, **kwargs: malformed)

    with pytest.raises(ltx25.LTX25BackendError, match="could not be materialized"):
        ltx25._materialize_runtime(tmp_path, tmp_path / "snapshot")


def test_ltx25_materialization_falls_back_without_data_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"safe = True\n"
    archive = ltx25.io.BytesIO()
    with ltx25.tarfile.open(fileobj=archive, mode="w:") as source:
        member = ltx25.tarfile.TarInfo("tracked.py")
        member.size = len(payload)
        source.addfile(member, ltx25.io.BytesIO(payload))
    completed = subprocess.CompletedProcess(
        [], 0, stdout=archive.getvalue(), stderr=b""
    )
    monkeypatch.setattr(ltx25.subprocess, "run", lambda *args, **kwargs: completed)
    monkeypatch.delattr(ltx25.tarfile, "data_filter", raising=False)
    extracted_without_filter = []

    def legacy_extractall(source, path, *, members, **kwargs):
        assert kwargs == {}
        extracted_without_filter.append(True)
        destination_path = Path(path)
        destination_path.mkdir(parents=True)
        for member in members:
            extracted = source.extractfile(member)
            assert extracted is not None
            (destination_path / member.name).write_bytes(extracted.read())

    monkeypatch.setattr(ltx25.tarfile.TarFile, "extractall", legacy_extractall)
    destination = tmp_path / "snapshot"

    ltx25._materialize_runtime(tmp_path, destination)

    assert extracted_without_filter == [True]
    assert (destination / "tracked.py").read_bytes() == payload


def test_ltx25_runtime_is_provisioned_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "checkout" / ".venv" / "bin" / "ltx-2-mlx"
    runtime.parent.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(ltx25, "_RUNTIME_CACHE", {})
    monkeypatch.setattr(ltx25.shutil, "which", lambda _: "/trusted/uv")
    monkeypatch.setattr(ltx25, "_materialize_runtime", lambda *args: None)

    def provision(command: list[str], **kwargs) -> subprocess.CompletedProcess:
        calls.append((command, kwargs))
        workspace = Path(command[command.index("--project") + 1])
        interpreter = workspace / ".venv/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("#!/bin/sh\n")
        interpreter.chmod(0o755)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(ltx25.subprocess, "run", provision)

    first = ltx25.prepare_ltx25_runtime(str(runtime))
    second = ltx25.prepare_ltx25_runtime(str(runtime))

    assert first == second
    assert len(calls) == 1
    assert calls[0][0][:3] == ["/trusted/uv", "sync", "--frozen"]


def test_ltx25_provisioning_does_not_leak_bundled_python_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "checkout" / ".venv" / "bin" / "ltx-2-mlx"
    runtime.parent.mkdir(parents=True)
    monkeypatch.setattr(ltx25, "_RUNTIME_CACHE", {})
    monkeypatch.setattr(ltx25.shutil, "which", lambda _: "/trusted/uv")
    monkeypatch.setattr(ltx25, "_materialize_runtime", lambda *args: None)
    monkeypatch.setenv("PYTHONHOME", "/signed-sidecar/python")
    monkeypatch.setenv("PYTHONPATH", "/signed-sidecar/site-packages")
    captured: dict[str, str] = {}

    def provision(command: list[str], **kwargs) -> subprocess.CompletedProcess:
        captured.update(kwargs["env"])
        workspace = Path(command[command.index("--project") + 1])
        interpreter = workspace / ".venv/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("#!/bin/sh\n")
        interpreter.chmod(0o755)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(ltx25.subprocess, "run", provision)
    ltx25.prepare_ltx25_runtime(str(runtime))

    assert "PYTHONHOME" not in captured
    assert "PYTHONPATH" not in captured


def _generate(engine: ltx25.LTX25VideoEngine, output_path: Path) -> None:
    engine.generate(
        prompt="fox",
        output_path=output_path,
        width=704,
        height=480,
        num_frames=97,
        fps=24,
        seed=7,
        image=None,
    )


def test_ltx25_generation_reuses_workspace_for_the_selected_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prepared workspace is reused, but only for the runtime it was built for."""
    prepared: list[str] = []
    workspace = tmp_path / "runtime-cache"
    monkeypatch.setattr(
        ltx25,
        "_prepared_ltx25_runtime",
        lambda executable: prepared.append(executable) or workspace,
    )
    monkeypatch.setattr(
        ltx25,
        "prepare_ltx25_runtime",
        lambda _: pytest.fail("a prepared workspace must be reused"),
    )
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: ltx25.LTX25Runtime(
            state="trusted", source="override", executable="/runtime/ltx-2-mlx"
        ),
    )
    commands: list[list[str]] = []

    class Process:
        returncode = 0

        def __init__(self, command: list[str], **kwargs) -> None:
            self.command = command
            commands.append(command)

        def communicate(self, *, input: str, timeout: int) -> None:
            Path(self.command[self.command.index("--output") + 1]).write_bytes(b"mp4")

    monkeypatch.setattr(ltx25.subprocess, "Popen", Process)
    engine = ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8")
    _generate(engine, tmp_path / "first.mp4")
    _generate(engine, tmp_path / "second.mp4")

    assert prepared == ["/runtime/ltx-2-mlx", "/runtime/ltx-2-mlx"]
    interpreter = str(workspace / ".venv/bin/python")
    assert [command[0] for command in commands] == [interpreter, interpreter]


def test_ltx25_generation_reselects_runtime_when_override_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing or unsetting the override must not run a stale workspace."""
    selected: dict[str, ltx25.LTX25Runtime] = {}
    monkeypatch.setattr(ltx25, "ltx25_runtime", lambda: selected["runtime"])
    prepared: list[str] = []
    monkeypatch.setattr(
        ltx25,
        "_prepared_ltx25_runtime",
        lambda executable: prepared.append(executable) or None,
    )
    provisioned: list[str] = []
    install_root = tmp_path / "install"
    monkeypatch.setattr(ltx25, "_install_root", lambda: install_root)

    def prepare(executable: str) -> Path:
        provisioned.append(executable)
        return tmp_path / f"cache-{len(provisioned)}"

    monkeypatch.setattr(ltx25, "prepare_ltx25_runtime", prepare)
    commands: list[list[str]] = []

    class Process:
        returncode = 0

        def __init__(self, command: list[str], **kwargs) -> None:
            self.command = command
            commands.append(command)

        def communicate(self, *, input: str, timeout: int) -> None:
            Path(self.command[self.command.index("--output") + 1]).write_bytes(b"mp4")

    monkeypatch.setattr(ltx25.subprocess, "Popen", Process)
    engine = ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8")
    for index, runtime in enumerate(
        (
            ltx25.LTX25Runtime(
                state="trusted", source="override", executable="/runtime/first"
            ),
            ltx25.LTX25Runtime(
                state="trusted", source="override", executable="/runtime/second"
            ),
            ltx25.LTX25Runtime(
                state="trusted", source="vendored", executable=sys.executable
            ),
        )
    ):
        selected["runtime"] = runtime
        _generate(engine, tmp_path / f"result-{index}.mp4")

    assert prepared == ["/runtime/first", "/runtime/second"]
    assert provisioned == ["/runtime/first", "/runtime/second"]
    assert commands[0][0] == str(tmp_path / "cache-1" / ".venv/bin/python")
    assert commands[1][0] == str(tmp_path / "cache-2" / ".venv/bin/python")
    # Unsetting the override falls back to the bundled runtime, never the cache.
    assert commands[2][:5] == [
        sys.executable,
        "-P",
        "-c",
        ltx25._STDIN_PROMPT_RUNNER,
        str(install_root),
    ]


def test_ltx25_generation_invalid_override_never_provisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: ltx25.LTX25Runtime(
            state="override_invalid",
            source="override",
            detail="RAPID_MLX_LTX25_RUNTIME does not point at the pinned revision.",
        ),
    )
    monkeypatch.setattr(
        ltx25,
        "_prepared_ltx25_runtime",
        lambda _: pytest.fail("an invalid override must not reuse a workspace"),
    )
    monkeypatch.setattr(
        ltx25,
        "prepare_ltx25_runtime",
        lambda _: pytest.fail("an invalid override must not be provisioned"),
    )

    with pytest.raises(ltx25.LTX25BackendError, match="RAPID_MLX_LTX25_RUNTIME"):
        _generate(
            ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8"),
            tmp_path / "result.mp4",
        )


def test_ltx25_generation_maps_child_provenance_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child that refuses the pinned tree gets the refusal message, not a
    generic exit-code error."""
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: ltx25.LTX25Runtime(
            state="trusted", source="vendored", executable=sys.executable
        ),
    )

    class Process:
        returncode = ltx25._CHILD_PROVENANCE_EXIT

        def __init__(self, command: list[str], **kwargs) -> None:
            pass

        def communicate(self, *, input: str, timeout: int) -> None:
            pass

    monkeypatch.setattr(ltx25.subprocess, "Popen", Process)
    monkeypatch.setattr(
        ltx25.LTX25VideoEngine,
        "_terminate_process",
        staticmethod(lambda candidate: None),
    )

    with pytest.raises(ltx25.LTX25BackendError) as exc:
        _generate(
            ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8"),
            tmp_path / "result.mp4",
        )

    assert "no longer matches the audited snapshot" in str(exc.value)
    assert "Reinstall rapid-mlx" in str(exc.value)


def test_ltx25_generation_rejects_untrusted_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ltx25, "_prepared_ltx25_runtime", lambda _: None)
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: ltx25.LTX25Runtime(
            state="corrupt", source="vendored", detail="provenance mismatch"
        ),
    )

    with pytest.raises(ltx25.LTX25BackendError, match="Reinstall rapid-mlx"):
        _generate(
            ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8"),
            tmp_path / "result.mp4",
        )


def test_serve_routes_ltx25_model_to_specific_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rapid_mlx import cli

    class PreflightReachedError(RuntimeError):
        pass

    def stop_at_preflight(model_name: str) -> None:
        assert model_name == "ltx-2.5-mlx-q8"
        raise PreflightReachedError

    monkeypatch.setattr(video_lane, "require_video_runtime_or_exit", stop_at_preflight)
    args = SimpleNamespace(model="ltx-2.5-mlx-q8", max_tokens=None, watchdog_ppid=None)
    with pytest.raises(PreflightReachedError):
        cli.serve_command(args)


def test_ltx25_child_pin_beats_a_shadowing_working_directory(
    tmp_path: Path,
) -> None:
    """The pinned child resolves the audited copy even when PYTHONPATH points
    at a fake package next to the working directory."""
    shadow = tmp_path / "shadow"
    (shadow / "ltx_pipelines_mlx").mkdir(parents=True)
    (shadow / "ltx_pipelines_mlx" / "__init__.py").write_text("FAKE = True\n")
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {"PYTHONHOME", "PYTHONPATH"}
    }
    environment["PYTHONPATH"] = str(shadow)
    # Resolve rather than import: the vendored package needs MLX at import time,
    # and the pin is what decides which copy the import machinery would pick.
    probe = (
        ltx25._RUNNER_PIN
        + ltx25._CHILD_PROVENANCE_CHECK
        + "import importlib.util\n"
        + "print(importlib.util.find_spec('ltx_pipelines_mlx').origin)\n"
    )

    # Control: without the pin, the shadowing copy wins.
    unpinned = subprocess.run(
        [sys.executable, "-P", "-c", probe, ""],
        cwd=shadow,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert unpinned.stdout.strip() == str(shadow / "ltx_pipelines_mlx" / "__init__.py")

    pinned = subprocess.run(
        [sys.executable, "-P", "-c", probe, str(REPO_ROOT)],
        cwd=shadow,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert pinned.returncode == 0, pinned.stderr
    assert pinned.stdout.strip() == str(REPO_ROOT / "ltx_pipelines_mlx" / "__init__.py")


def test_ltx25_child_refuses_a_pin_without_audited_provenance(
    tmp_path: Path,
) -> None:
    """The provenance re-check runs inside the child before importing LTX."""
    fake_root = tmp_path / "install"
    for package in ltx25.LTX25_VENDORED_PACKAGES:
        (fake_root / package).mkdir(parents=True)
        (fake_root / package / "__init__.py").write_text("FAKE = True\n")
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {"PYTHONHOME", "PYTHONPATH"}
    }
    probe = ltx25._RUNNER_PIN + ltx25._CHILD_PROVENANCE_CHECK + "print('imported')\n"

    completed = subprocess.run(
        [sys.executable, "-P", "-c", probe, str(fake_root)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == ltx25._CHILD_PROVENANCE_EXIT
    assert "imported" not in completed.stdout
    assert "ltx_core_mlx: provenance file is missing" in completed.stderr


def test_ltx25_child_refuses_a_pin_that_does_not_contain_the_import(
    tmp_path: Path,
) -> None:
    """A pin that resolves elsewhere (here via PYTHONPATH) is refused, so the
    child never imports a copy outside the audited install root."""
    empty_root = tmp_path / "install"
    empty_root.mkdir()
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {"PYTHONHOME", "PYTHONPATH"}
    }
    environment["PYTHONPATH"] = str(REPO_ROOT)
    probe = ltx25._RUNNER_PIN + ltx25._CHILD_PROVENANCE_CHECK + "print('imported')\n"

    completed = subprocess.run(
        [sys.executable, "-P", "-c", probe, str(empty_root)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == ltx25._CHILD_PROVENANCE_EXIT
    assert "imported" not in completed.stdout
    assert "resolved outside the pinned runtime" in completed.stderr


def test_ltx25_engine_invokes_pinned_runtime_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONHOME", "/signed-sidecar/python")
    monkeypatch.setenv("PYTHONPATH", "/signed-sidecar/site-packages")
    output = tmp_path / "result.mp4"
    image = tmp_path / "reference.png"
    image.write_bytes(b"png")
    calls: list[tuple[list[str], dict]] = []
    communicates: list[tuple[str, int]] = []
    runtime = tmp_path / ".venv" / "bin" / "ltx-2-mlx"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o755)
    runtime_python = runtime.with_name("python")
    runtime_python.write_text("#!/bin/sh\n")
    runtime_python.chmod(0o755)
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: ltx25.LTX25Runtime(
            state="trusted", source="override", executable=str(runtime)
        ),
    )
    runtime_cache = tmp_path / "runtime-cache"
    monkeypatch.setattr(ltx25, "prepare_ltx25_runtime", lambda _: runtime_cache)

    class Process:
        returncode = 0

        def __init__(self, command: list[str], **kwargs) -> None:
            self.command = command
            calls.append((command, kwargs))

        def communicate(self, *, input: str, timeout: int) -> tuple[str, str]:
            communicates.append((input, timeout))
            generated = Path(self.command[self.command.index("--output") + 1])
            generated.write_bytes(b"mp4-with-audio")
            return "", ""

    monkeypatch.setattr(ltx25.subprocess, "Popen", Process)
    ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8").generate(
        prompt="a fox",
        output_path=output,
        width=704,
        height=480,
        num_frames=97,
        fps=24,
        seed=7,
        image=image,
        conditioning_strength=0.6,
    )

    command, run_kwargs = calls[0]
    child_environment = run_kwargs.pop("env")
    readiness_fds = run_kwargs.pop("pass_fds")
    assert "PYTHONHOME" not in child_environment
    assert "PYTHONPATH" not in child_environment
    assert child_environment[ltx25._READINESS_FD_ENV] == str(readiness_fds[0])
    assert command[:2] == [str(runtime_cache / ".venv/bin/python"), "-P"]
    assert command[2] == "-c"
    assert command[3] == ltx25._STDIN_PROMPT_RUNNER
    # An override runtime is isolated, so no bundled package root is pinned.
    assert command[4] == ""
    assert command[5:7] == ["generate", "--model"]
    generated = Path(command[command.index("--output") + 1])
    assert generated != output
    assert generated.parent.parent == output.parent
    assert output.read_bytes() == b"mp4-with-audio"
    assert "--distilled" in command
    assert "--low-ram" in command
    assert "a fox" not in command
    assert command[command.index("--image") + 1 :] == [str(image), "0", "0.6"]
    assert run_kwargs == {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "text": True,
        "start_new_session": True,
    }
    assert communicates == [("a fox", 7200)]
    assert output.read_bytes() == b"mp4-with-audio"


def test_ltx25_engine_reports_subprocess_failure_without_leaking_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / ".venv" / "bin" / "ltx-2-mlx"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o755)
    runtime.with_name("python").write_text("#!/bin/sh\n")
    runtime.with_name("python").chmod(0o755)
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: ltx25.LTX25Runtime(
            state="trusted", source="override", executable=str(runtime)
        ),
    )
    monkeypatch.setattr(
        ltx25, "prepare_ltx25_runtime", lambda _: tmp_path / "runtime-cache"
    )

    class Process:
        returncode = 1

        def __init__(self, command: list[str], **kwargs) -> None:
            pass

        def communicate(self, *, input: str, timeout: int) -> tuple[str, str]:
            return "private-output-sentinel", "private-error-sentinel"

    monkeypatch.setattr(ltx25.subprocess, "Popen", Process)
    terminated = []
    monkeypatch.setattr(
        ltx25.LTX25VideoEngine,
        "_terminate_process",
        staticmethod(lambda candidate: terminated.append(candidate)),
    )
    engine = ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8")
    with pytest.raises(ltx25.LTX25BackendError) as exc:
        engine.generate(
            prompt="a fox",
            output_path=tmp_path / "result.mp4",
            width=704,
            height=480,
            num_frames=97,
            fps=24,
            seed=7,
            image=None,
        )
    assert str(exc.value) == (
        "LTX-2.5 runtime exited with code 1; runtime output is not retained "
        "because it may contain request data."
    )
    assert "private-output-sentinel" not in str(exc.value)
    assert "private-error-sentinel" not in str(exc.value)
    assert len(terminated) == 1


def test_ltx25_zero_exit_cannot_reuse_stale_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.mp4"
    output.write_bytes(b"stale-video")
    runtime = tmp_path / ".venv" / "bin" / "ltx-2-mlx"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o755)
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: ltx25.LTX25Runtime(
            state="trusted", source="override", executable=str(runtime)
        ),
    )
    monkeypatch.setattr(
        ltx25, "prepare_ltx25_runtime", lambda _: tmp_path / "runtime-cache"
    )

    class Process:
        returncode = 0

        def __init__(self, command: list[str], **kwargs) -> None:
            pass

        def communicate(self, *, input: str, timeout: int) -> tuple[str, str]:
            return "", ""

    monkeypatch.setattr(ltx25.subprocess, "Popen", Process)
    terminated = []
    monkeypatch.setattr(
        ltx25.LTX25VideoEngine,
        "_terminate_process",
        staticmethod(lambda candidate: terminated.append(candidate)),
    )

    with pytest.raises(ltx25.LTX25BackendError, match="without an MP4 output"):
        ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8").generate(
            prompt="a fox",
            output_path=output,
            width=704,
            height=480,
            num_frames=97,
            fps=24,
            seed=7,
            image=None,
        )

    assert output.read_bytes() == b"stale-video"
    assert list(tmp_path.glob(".result.mp4.*")) == []
    assert len(terminated) == 1


def test_ltx25_timeout_terminates_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / ".venv" / "bin" / "ltx-2-mlx"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o755)
    runtime.with_name("python").write_text("#!/bin/sh\n")
    runtime.with_name("python").chmod(0o755)
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: ltx25.LTX25Runtime(
            state="trusted", source="override", executable=str(runtime)
        ),
    )
    monkeypatch.setattr(
        ltx25, "prepare_ltx25_runtime", lambda _: tmp_path / "runtime-cache"
    )
    terminated = []

    class Process:
        returncode = None

        def __init__(self, command: list[str], **kwargs) -> None:
            pass

        def communicate(self, *, input: str, timeout: int) -> None:
            raise subprocess.TimeoutExpired("ltx-2-mlx", timeout)

    process = Process([], stdin=subprocess.PIPE, text=True)
    monkeypatch.setattr(ltx25.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        ltx25.LTX25VideoEngine,
        "_terminate_process",
        staticmethod(lambda candidate: terminated.append(candidate)),
    )
    engine = ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8")

    with pytest.raises(ltx25.LTX25BackendError, match="time limit"):
        engine.generate(
            prompt="a fox",
            output_path=tmp_path / "result.mp4",
            width=704,
            height=480,
            num_frames=97,
            fps=24,
            seed=7,
            image=None,
        )

    assert terminated == [process]
    assert engine._process is None


def test_ltx25_termination_does_not_reuse_exited_leader_pgid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals = []

    class Process:
        pid = 123

        def poll(self) -> int:
            return 1

        def wait(self, *, timeout: int) -> None:
            pytest.fail("an exited group leader must not be waited on again")

    monkeypatch.setattr(ltx25.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    ltx25.LTX25VideoEngine._terminate_process(Process())  # type: ignore[arg-type]

    assert signals == []


def test_ltx25_termination_escalates_while_leader_is_unreaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals = []

    class Process:
        pid = 123
        waits = 0

        def poll(self) -> None:
            return None

        def wait(self, *, timeout: int) -> int:
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("ltx", timeout)
            return -signal.SIGKILL

    process = Process()
    monkeypatch.setattr(ltx25.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    ltx25.LTX25VideoEngine._terminate_process(process)  # type: ignore[arg-type]

    assert signals == [(123, signal.SIGTERM), (123, signal.SIGKILL)]
    assert process.waits == 2


def test_ltx25_unexpected_communication_error_terminates_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / ".venv" / "bin" / "ltx-2-mlx"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o755)
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: ltx25.LTX25Runtime(
            state="trusted", source="override", executable=str(runtime)
        ),
    )
    monkeypatch.setattr(
        ltx25, "prepare_ltx25_runtime", lambda _: tmp_path / "runtime-cache"
    )
    terminated = []

    class Process:
        returncode = None

        def communicate(self, *, input: str, timeout: int) -> None:
            raise UnicodeEncodeError("utf-8", "bad surrogate \ud800", 14, 15, "invalid")

        def poll(self) -> None:
            return None

    process = Process()
    monkeypatch.setattr(ltx25.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        ltx25.LTX25VideoEngine,
        "_terminate_process",
        staticmethod(lambda candidate: terminated.append(candidate)),
    )
    engine = ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8")

    with pytest.raises(ltx25.LTX25BackendError, match="isolated runtime"):
        engine.generate(
            prompt="bad surrogate \ud800",
            output_path=tmp_path / "result.mp4",
            width=704,
            height=480,
            num_frames=97,
            fps=24,
            seed=7,
            image=None,
        )

    assert terminated == [process]
    assert engine._process is None


def test_ltx25_spawn_failure_closes_readiness_pipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: ltx25.LTX25Runtime(
            state="trusted", source="vendored", executable="/python"
        ),
    )
    opened: list[int] = []
    closed: list[int] = []
    original_pipe = ltx25.os.pipe
    original_close = ltx25.os.close

    def tracked_pipe() -> tuple[int, int]:
        descriptors = original_pipe()
        opened.extend(descriptors)
        return descriptors

    def tracked_close(descriptor: int) -> None:
        closed.append(descriptor)
        original_close(descriptor)

    monkeypatch.setattr(ltx25.os, "pipe", tracked_pipe)
    monkeypatch.setattr(ltx25.os, "close", tracked_close)
    monkeypatch.setattr(
        ltx25.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("spawn failed")),
    )

    with pytest.raises(ltx25.LTX25BackendError, match="isolated runtime"):
        ltx25.LTX25VideoEngine("ltx-2.5-mlx-q8").generate(
            prompt="x",
            output_path=tmp_path / "output.mp4",
            width=64,
            height=64,
            num_frames=5,
            fps=24,
            seed=1,
            image=None,
        )

    assert set(opened).issubset(closed)


def test_ltx25_invalid_timeout_does_not_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / ".venv" / "bin" / "ltx-2-mlx"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o755)
    runtime.with_name("python").write_text("#!/bin/sh\n")
    runtime.with_name("python").chmod(0o755)
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: ltx25.LTX25Runtime(
            state="trusted", source="override", executable=str(runtime)
        ),
    )
    monkeypatch.setattr(ltx25.shutil, "which", lambda _: "/trusted/uv")
    monkeypatch.setenv("RAPID_MLX_LTX25_TIMEOUT_SEC", "invalid")
    monkeypatch.setattr(
        ltx25.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("invalid config must fail before spawn"),
    )

    with pytest.raises(ltx25.LTX25BackendError, match="integer"):
        ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8").generate(
            prompt="a fox",
            output_path=tmp_path / "result.mp4",
            width=704,
            height=480,
            num_frames=97,
            fps=24,
            seed=7,
            image=None,
        )


def test_video_engine_selects_ltx25_and_preserves_generated_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def generate(self, **kwargs) -> None:
        captured.update(kwargs)
        kwargs["output_path"].write_bytes(b"mp4-with-audio")

    monkeypatch.setattr(ltx25.LTX25VideoEngine, "generate", generate)
    monkeypatch.setattr(
        VideoEngine,
        "_crop_generated_output",
        lambda *args, **kwargs: captured.update(crop=kwargs),
    )
    engine = VideoEngine("MrMofer/ltx-2.5-mlx-q8")
    output = tmp_path / "result.mp4"
    engine.generate(
        prompt="a fox",
        output_path=output,
        width=704,
        height=512,
        num_frames=97,
        fps=24,
        seed=7,
        image=None,
    )

    assert engine.video_family == "ltx-2.5"
    assert output.read_bytes() == b"mp4-with-audio"
    assert captured["crop"]["family"] == "LTX-2.5"


def test_ltx25_real_crop_preserves_audio_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ffmpeg = video_lane._resolve_ffmpeg()
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg and ffprobe are required for the real media assertion")
    output = tmp_path / "with-audio.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:d=0.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=0.2",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(output),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    VideoEngine._crop_generated_output(
        output_path=output,
        width=64,
        height=64,
        output_width=32,
        output_height=32,
        family="LTX-2.5",
    )
    streams = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert streams.stdout.strip() == "audio"


def test_ltx25_distilled_rejects_cfg_controls() -> None:
    engine = VideoEngine("MrMofer/ltx-2.5-mlx-q8")
    with pytest.raises(VideoRuntimeError, match="does not support"):
        engine.generate(
            prompt="a fox",
            output_path=Path("result.mp4"),
            width=704,
            height=480,
            num_frames=97,
            fps=24,
            seed=7,
            image=None,
            guidance_scale=4.0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsupported",
    [{"guidance_scale": 4.0}, {"negative_prompt": "blurry"}],
)
async def test_ltx25_route_rejects_unsupported_cfg_before_queueing(
    monkeypatch: pytest.MonkeyPatch, unsupported: dict[str, object]
) -> None:
    from rapid_mlx.routes import video

    engine = SimpleNamespace(
        model_name="MrMofer/ltx-2.5-mlx-q8", video_family="ltx-2.5"
    )
    monkeypatch.setattr(video, "_video_engine", lambda: engine)
    monkeypatch.setattr(video, "_accepting_jobs", True)
    before = set(video._jobs)

    with pytest.raises(HTTPException, match="does not support") as exc:
        await video.create_video(
            prompt="a fox",
            model="ltx-2.5-mlx-q8",
            seconds="1",
            size="704x512",
            seed=7,
            input_reference=None,
            **unsupported,
        )

    assert exc.value.status_code == 400
    assert set(video._jobs) == before
