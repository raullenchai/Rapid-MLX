# SPDX-License-Identifier: Apache-2.0
"""Subfolder selection for ``scripts/mirror_to_r2.py``.

Upstreams have started shipping every quantisation of a model inside ONE
repo as sibling directories. ``LiquidAI/LFM2.5-2.6B-MLX`` is the first one
we serve: 60 files, 20.05 GB, of which the 4-bit variant anyone actually
pulls is 7 files and 1.60 GB. Mirroring it wholesale would put 18.45 GB of
unused weights on R2 — 92% waste — which is why nothing mirrored it and the
8 GB tier's only recommended model 404s.

Every other repo we mirror is one-quantisation-per-repo (the quantisation
is in the repo NAME: ``…-MLX-4bit``), where "the repo" and "what we serve"
are the same thing. That is why this selector did not exist before, and why
the default must stay "mirror everything".

Fixtures mirror the real layouts, taken from the HF API:

  * LiquidAI/LFM2.5-2.6B-MLX — 3 root files + 8 quantisation subfolders
  * openbmb/MiniCPM5-1B-MLX — 10 files, all at the root (flat, despite the
    identical ``-MLX`` naming, which is why the selector keys on layout
    rather than on the repo name)
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

# ``scripts/`` is not a package, so load the module by path. It must be
# registered in ``sys.modules`` BEFORE exec: ``@dataclass`` resolves its
# annotations through ``sys.modules[cls.__module__]``, which is None for an
# unregistered module and fails collection with a bare AttributeError.
_SPEC = importlib.util.spec_from_file_location(
    "mirror_to_r2",
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "mirror_to_r2.py",
)
mirror = importlib.util.module_from_spec(_SPEC)
sys.modules["mirror_to_r2"] = mirror
_SPEC.loader.exec_module(mirror)


def _f(relpath: str, size: int = 1000) -> mirror.FileMeta:
    return mirror.FileMeta(relpath=relpath, size=size, key=f"repo/{relpath}")


_QUANTS = ("4bit", "5bit", "6bit", "8bit", "bf16", "mxfp4", "mxfp8", "nvfp4")

SUBFOLDER_REPO = [
    _f(".gitattributes", 1),
    _f("LICENSE", 10_574),
    _f("README.md", 5_000),
    *[
        _f(f"{q}/{name}", 200_000_000 if name.endswith(".safetensors") else 1_000)
        for q in _QUANTS
        for name in (
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    ],
]

FLAT_REPO = [
    _f(".gitattributes"),
    _f("README.md"),
    _f("config.json"),
    _f("model.safetensors"),
    _f("tokenizer.json"),
]


def test_selects_only_the_requested_quantisation():
    got = mirror._select_subfolder(SUBFOLDER_REPO, "4bit")
    quant_dirs = {f.relpath.split("/")[0] for f in got if "/" in f.relpath}
    assert quant_dirs == {"4bit"}, f"leaked other quantisations: {quant_dirs}"


def test_keeps_repo_root_license_and_readme():
    """The weights are in the subfolder; the terms are not.

    A mirror that ships only ``4bit/`` hands users the weights with no
    statement of what they may do with them — and these repos are not all
    Apache-2.0.
    """
    got = {f.relpath for f in mirror._select_subfolder(SUBFOLDER_REPO, "4bit")}
    assert "LICENSE" in got
    assert "README.md" in got


def test_drops_root_noise():
    got = {f.relpath for f in mirror._select_subfolder(SUBFOLDER_REPO, "4bit")}
    assert ".gitattributes" not in got


def test_selection_is_the_expected_size():
    """Guards the number that motivates the whole feature."""
    got = mirror._select_subfolder(SUBFOLDER_REPO, "4bit")
    full_bytes = sum(f.size for f in SUBFOLDER_REPO)
    kept_bytes = sum(f.size for f in got)
    assert len(got) == 9, [f.relpath for f in got]  # 7 weights + LICENSE + README
    assert kept_bytes < full_bytes / 7, (
        f"selection kept {kept_bytes / 1e9:.2f} GB of {full_bytes / 1e9:.2f} GB — "
        f"expected roughly one eighth"
    )


def test_keys_keep_the_subfolder_path():
    """R2 layout must stay a faithful copy of the source tree.

    ``vllm_mlx/_mirror.py`` resolves ``<owner>/<repo>/<filename>``; flattening
    ``4bit/config.json`` to ``config.json`` would silently collide with any
    other quantisation mirrored later into the same repo prefix.
    """
    got = mirror._select_subfolder(SUBFOLDER_REPO, "4bit")
    weights = [f for f in got if f.relpath.endswith(".safetensors")]
    assert weights and all(f.relpath.startswith("4bit/") for f in weights)


def test_trailing_slash_is_accepted():
    assert mirror._select_subfolder(SUBFOLDER_REPO, "4bit/") == (
        mirror._select_subfolder(SUBFOLDER_REPO, "4bit")
    )


def test_unknown_subfolder_fails_loudly_and_lists_options():
    """Typos must not silently mirror the root metadata and nothing else.

    Without the explicit check the selector would return just LICENSE +
    README, the run would report success, and the model would 404 exactly
    as it does today — a failure that looks like a completed mirror.
    """
    with pytest.raises(ValueError) as exc:
        mirror._select_subfolder(SUBFOLDER_REPO, "4-bit")
    assert "4bit" in str(exc.value), "error should list the available subfolders"


def test_flat_repo_rejects_subfolder_rather_than_mirroring_nothing():
    with pytest.raises(ValueError) as exc:
        mirror._select_subfolder(FLAT_REPO, "4bit")
    assert "flat repo" in str(exc.value)


# --- review round 1 -------------------------------------------------------
#
# Three ways the selector reported success while producing something
# unusable. Each is a "completed mirror that still 404s" — the exact
# failure this flag exists to prevent, arriving through a different door.


def test_empty_subfolder_is_rejected_not_treated_as_no_filter():
    """``--subfolder "$QUANT"`` with QUANT unset must not mirror 20 GB.

    An unset shell variable expands to an empty string, and a truthiness
    check reads that as "no subfolder requested". The operator asked for
    one quantisation and would have silently got the whole repo.
    """
    for empty in ("", "/", "//"):
        with pytest.raises(ValueError) as exc:
            mirror._select_subfolder(SUBFOLDER_REPO, empty)
        assert "empty" in str(exc.value).lower(), str(exc.value)


def test_non_checkpoint_directory_is_rejected():
    """A directory that exists is not necessarily a checkpoint.

    ``--subfolder docs`` used to upload the documentation, verify the
    objects it had just written, and exit 0 — a mirror that completed and
    served no model.
    """
    repo = [*SUBFOLDER_REPO, _f("docs/example.md"), _f("docs/img/diagram.png")]
    with pytest.raises(ValueError) as exc:
        mirror._select_subfolder(repo, "docs")
    assert "checkpoint" in str(exc.value)


def test_subfolder_needs_both_config_and_weights():
    """Neither half alone is enough to load a model."""
    config_only = [_f("LICENSE"), _f("4bit/config.json")]
    with pytest.raises(ValueError):
        mirror._select_subfolder(config_only, "4bit")

    weights_only = [_f("LICENSE"), _f("4bit/model.safetensors")]
    with pytest.raises(ValueError):
        mirror._select_subfolder(weights_only, "4bit")


def test_nested_config_does_not_satisfy_the_checkpoint_check():
    """``4bit/extra/config.json`` is not the checkpoint's own config."""
    nested = [
        _f("LICENSE"),
        _f("4bit/extra/config.json"),
        _f("4bit/model.safetensors"),
    ]
    with pytest.raises(ValueError):
        mirror._select_subfolder(nested, "4bit")


@pytest.mark.parametrize(
    "name",
    [
        "LICENSE",
        "LICENSE.txt",
        "LICENSE.md",
        "LICENSE-MODEL",
        "license",
        "NOTICE",
        "NOTICE.md",
        "NOTICE.txt",
        "README.md",
        "COPYING",
    ],
)
def test_root_terms_are_kept_whatever_the_upstream_calls_them(name: str):
    """Redistribution terms must not be dropped over a filename variant.

    An exact-name allowlist omitted ``NOTICE.md`` while keeping
    ``NOTICE.txt`` and ``LICENSE.md`` — an arbitrary distinction that
    silently ships weights without their terms.
    """
    repo = [_f(name), *[f for f in SUBFOLDER_REPO if "/" in f.relpath]]
    got = {f.relpath for f in mirror._select_subfolder(repo, "4bit")}
    assert name in got


@pytest.mark.parametrize(
    "name", [".gitattributes", "config.json", "model.safetensors", "licenses.py"]
)
def test_root_noise_is_still_dropped(name: str):
    """Widening the terms match must not start sweeping in root weights."""
    repo = [_f(name), *[f for f in SUBFOLDER_REPO if "/" in f.relpath]]
    got = {f.relpath for f in mirror._select_subfolder(repo, "4bit")}
    assert name not in got
