# SPDX-License-Identifier: Apache-2.0
"""Review forced code-trust routes when the installed VLM dependency changes."""

import ast
import importlib.util
from pathlib import Path

import pytest

from rapid_mlx.utils.model_file_guard import (
    EXTERNAL_TOKENIZER_VLM_MODEL_TYPES,
    FORCED_TRUST_VLM_MODEL_TYPES,
)

# Reviewed against 0.7.2. Keep reasons explicit: a literal True alone does not
# establish whether a route bypasses consent at the checkpoint boundary.
CLASSIFICATION = {
    "models/got/processing_got.py": (
        "forced: guarded by FORCED_TRUST_VLM_MODEL_TYPES",
        {"got"},
    ),
    "models/inkling/processing_inkling.py": (
        "forced: guarded by FORCED_TRUST_VLM_MODEL_TYPES",
        {"inkling", "inkling_mm_model"},
    ),
    "models/z_image/pipeline.py": (
        "forced: guarded by FORCED_TRUST_VLM_MODEL_TYPES",
        {"z_image"},
    ),
    "models/moondream2/processing_moondream2.py": (
        "external: guarded by EXTERNAL_TOKENIZER_VLM_MODEL_TYPES",
        {"moondream1", "moondream2"},
    ),
    "models/minicpmo/processing_minicpmo.py": (
        "consent-gated: only after checking caller consent; local tokenizer fallback",
        set(),
    ),
    "models/minicpmv4_6/processing_minicpmv4_6.py": (
        "consent-gated: only after checking caller consent; local tokenizer fallback",
        set(),
    ),
    "models/internvl_chat/processor.py": (
        "constructor-only: string tokenizer branch; from_pretrained forwards caller "
        "trust to AutoTokenizer and passes a tokenizer object to the constructor",
        set(),
    ),
}
# models/base.py checks kwargs.get("trust_remote_code") is True before invoking
# its composable hook; that comparison is consent gating, not a forced assignment.
VENDORED_CLASSIFICATION = {}


def _literal_true(node):
    return isinstance(node, ast.Constant) and node.value is True


def _trust_target(node):
    return (isinstance(node, ast.Name) and node.id == "trust_remote_code") or (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "trust_remote_code"
    )


def _forced_trust_hits(root):
    hits = {}
    for path in sorted((root / "models").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        lines = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and any(
                kw.arg == "trust_remote_code" and _literal_true(kw.value)
                for kw in node.keywords
            ):
                lines.append(node.lineno)
            elif isinstance(node, ast.Assign) and _literal_true(node.value):
                if any(_trust_target(target) for target in node.targets):
                    lines.append(node.lineno)
            elif isinstance(node, ast.AnnAssign) and _literal_true(node.value):
                if _trust_target(node.target):
                    lines.append(node.lineno)
        if lines:
            hits[path.relative_to(root).as_posix()] = sorted(lines)
    return hits


@pytest.mark.parametrize("vendored", [False, True], ids=["installed", "vendored"])
def test_vlm_processor_trust_drift(vendored):
    if vendored:
        root = Path(__file__).resolve().parents[1] / "rapid_mlx/models/mlx_vlm_vendored"
        classified = VENDORED_CLASSIFICATION
    else:
        spec = importlib.util.find_spec("mlx_vlm")
        if spec is None:
            pytest.skip("mlx_vlm is not installed")
        root = Path(next(iter(spec.submodule_search_locations)))
        classified = CLASSIFICATION
    assert (root / "models").is_dir()
    hits = _forced_trust_hits(root)
    assert hits.keys() == classified.keys(), (
        "Review the new or changed tokenizer/processor route and update the guard "
        f"and classification. Unclassified: {hits.keys() - classified.keys()}; "
        f"stale classifications: {classified.keys() - hits.keys()}; hits: {hits}"
    )
    # Every reviewed file currently has exactly one hit. A new forced-trust
    # route inside an already classified file must also receive review.
    assert all(len(lines) == 1 for lines in hits.values()), (
        f"Review the additional route and update the guard and classification: {hits}"
    )
    for reason, model_types in classified.values():
        if reason.startswith("forced:"):
            assert model_types <= FORCED_TRUST_VLM_MODEL_TYPES
        elif reason.startswith("external:"):
            assert model_types <= EXTERNAL_TOKENIZER_VLM_MODEL_TYPES


def test_scan_detects_forced_trust_forms(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "processor.py").write_text(
        "load(trust_remote_code=True)\n"
        'kwargs["trust_remote_code"] = True\n'
        "trust_remote_code = True\n"
        "trust_remote_code: bool = True\n"
        "load(trust_remote_code=False)\n"
        'consented = kwargs.get("trust_remote_code") is True\n',
        encoding="utf-8",
    )
    assert _forced_trust_hits(tmp_path) == {"models/processor.py": [1, 2, 3, 4]}
