"""Vendored drafter registry parity and binding probes (step 3c-1).

The vendored ``speculative/drafters`` package must stay byte-verbatim
against pinned ``mlx_vlm.speculative.drafters`` @ 0.7.1 except the
documented import redirects listed in the package ``__init__.py``
inventory. Every probe here fails closed: an undocumented edit anywhere in
a drafter module diverges.
"""

import inspect
import json
from pathlib import Path

import pytest

pytest.importorskip("mlx_vlm")

VENDORED_ROOT = (
    Path(__file__).resolve().parent.parent
    / "rapid_mlx"
    / "models"
    / "mlx_vlm_vendored"
    / "speculative"
    / "drafters"
)

# Documented redirects: (file, vendored line, upstream line). Applying the
# replacements below to the vendored sources must reproduce the pinned
# upstream bytes exactly; anything else diverges.
REDIRECTS = {
    "__init__.py": [
        (
            "from mlx_vlm.speculative.drafters.dspark import DSparkDraftModel\n",
            "from .dspark import DSparkDraftModel\n",
        ),
        (
            "from mlx_vlm.speculative.drafters.laguna_dflash import LagunaDFlashDraftModel\n",
            "from .laguna_dflash import LagunaDFlashDraftModel\n",
        ),
        (
            "from mlx_vlm.speculative.drafters.muse_glimmer_assistant import MuseGlimmerAssistantDraftModel\n",
            "from .muse_glimmer_assistant import MuseGlimmerAssistantDraftModel\n",
        ),
        (
            "    from mlx_vlm.utils import get_model_path, load_model\n",
            "    from ...utils import get_model_path, load_model\n",
        ),
    ],
    "glm5_next_mtp/config.py": [
        (
            "from mlx_vlm.models.glm5_next.config import TextConfig as Glm5NextTextConfig\n",
            "from ....models.glm5_next.config import TextConfig as Glm5NextTextConfig\n",
        ),
    ],
    "glm5_next_mtp/glm5_next_mtp.py": [
        (
            "from mlx_vlm.models.cache import (\n",
            "from ....models.cache import (\n",
        ),
        (
            "from mlx_vlm.models.glm5_next.language import Glm5NextAttention, Glm5NextMoE\n",
            "from ....models.glm5_next.language import Glm5NextAttention, Glm5NextMoE\n",
        ),
    ],
    "glm5_next_mtp/split.py": [
        (
            "from mlx_vlm.models.glm5_next.config import TextConfig\n",
            "from ....models.glm5_next.config import TextConfig\n",
        ),
    ],
    "qwen3_5_mtp/config.py": [
        (
            "from mlx_vlm.models.qwen3_5.config import TextConfig as DenseTextConfig\n",
            "from ....models.qwen3_5.config import TextConfig as DenseTextConfig\n",
        ),
        (
            "from mlx_vlm.models.qwen3_5_moe.config import TextConfig as MoeTextConfig\n",
            "from ....models.qwen3_5_moe.config import TextConfig as MoeTextConfig\n",
        ),
    ],
    "qwen3_5_mtp/qwen3_5_mtp.py": [
        (
            "from mlx_vlm.models.cache import BatchKVCache, KVCache\n",
            "from ....models.cache import BatchKVCache, KVCache\n",
        ),
        (
            "from mlx_vlm.models.qwen3_5.language import Qwen3_5DecoderLayer\n",
            "from ....models.qwen3_5.language import Qwen3_5DecoderLayer\n",
        ),
        (
            "from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeDecoderLayer\n",
            "from ....models.qwen3_5_moe.language import Qwen3_5MoeDecoderLayer\n",
        ),
    ],
    "qwen3_dflash/dflash.py": [
        (
            "from mlx_vlm.models.activations import swiglu\n",
            "from ....models.activations import swiglu\n",
        ),
        (
            "from mlx_vlm.models.cache import (\n",
            "from ....models.cache import (\n",
        ),
        (
            "from mlx_vlm.models.rope_utils import initialize_rope\n",
            "from ....models.rope_utils import initialize_rope\n",
        ),
    ],
}

FILES = [
    "__init__.py",
    "compatibility.py",
    "mtp_base.py",
    "mtp_split.py",
    "glm5_next_mtp/__init__.py",
    "glm5_next_mtp/config.py",
    "glm5_next_mtp/glm5_next_mtp.py",
    "glm5_next_mtp/split.py",
    "qwen3_5_mtp/__init__.py",
    "qwen3_5_mtp/config.py",
    "qwen3_5_mtp/qwen3_5_mtp.py",
    "qwen3_5_mtp/split.py",
    "qwen3_dflash/__init__.py",
    "qwen3_dflash/config.py",
    "qwen3_dflash/dflash.py",
    "qwen3_dflash/parity_check.py",
    "dflash2/__init__.py",
    "dflash2/config.py",
    "dflash2/dflash2.py",
]


def test_vendored_drafter_files_match_upstream_bytes():
    """Byte parity after reverting the documented import redirects."""
    import mlx_vlm.speculative.drafters as pinned_pkg

    upstream_root = Path(inspect.getsourcefile(pinned_pkg)).parent
    diverged = []
    for rel in FILES:
        vendored = (VENDORED_ROOT / rel).read_text()
        for vendored_line, upstream_line in REDIRECTS.get(rel, []):
            if vendored.count(vendored_line) != 1:
                diverged.append(f"{rel}: documented redirect line not found")
                vendored = None
                break
            vendored = vendored.replace(vendored_line, upstream_line, 1)
        if vendored is None:
            continue
        upstream = (upstream_root / rel).read_text()
        if vendored != upstream:
            diverged.append(rel)
    assert diverged == []


def test_registry_tables_and_exports():
    import mlx_vlm.speculative.drafters as pinned_pkg

    from rapid_mlx.models.mlx_vlm_vendored.speculative import drafters as reg

    assert reg.KNOWN_DRAFTER_KINDS == pinned_pkg.KNOWN_DRAFTER_KINDS
    assert reg.DRAFTER_KIND_BY_MODEL_TYPE == pinned_pkg.DRAFTER_KIND_BY_MODEL_TYPE
    assert reg.DEFAULT_DRAFTER_KIND == pinned_pkg.DEFAULT_DRAFTER_KIND
    # Served families resolve from the vendored package; out-of-scope
    # families resolve through the documented pinned redirect.
    assert reg.DFlash2DraftModel.__module__.startswith(
        "rapid_mlx.models.mlx_vlm_vendored"
    )
    assert reg.DFlashDraftModel.__module__.startswith(
        "rapid_mlx.models.mlx_vlm_vendored"
    )
    assert reg.DSparkDraftModel.__module__ == (
        "mlx_vlm.speculative.drafters.dspark.dspark"
    )


def test_resolve_drafter_kind_auto_detects_served_families(tmp_path):
    from rapid_mlx.models.mlx_vlm_vendored.speculative.drafters import (
        resolve_drafter_kind,
    )

    for model_type, expected in (
        ("glm5_next_mtp", "mtp"),
        ("qwen3_5_mtp", "mtp"),
        ("qwen3_dflash", "dflash"),
        ("dflash2", "dflash"),
    ):
        repo = tmp_path / model_type
        repo.mkdir()
        (repo / "config.json").write_text(json.dumps({"model_type": model_type}))
        assert resolve_drafter_kind(repo) == expected, model_type


def test_load_drafter_rejects_unknown_kind(tmp_path):
    from rapid_mlx.models.mlx_vlm_vendored.speculative.drafters import load_drafter

    with pytest.raises(ValueError, match="Unknown drafter kind"):
        load_drafter(str(tmp_path), kind="teleport")


def test_runtime_binds_vendored_registry():
    """The native-MTP runtime must import the vendored registry, not pinned."""
    import rapid_mlx.speculative.native_mtp.runtime as runtime

    source = inspect.getsource(runtime)
    assert (
        "from rapid_mlx.models.mlx_vlm_vendored.speculative.drafters import" in source
    )
    assert "    from mlx_vlm.speculative.drafters import load_drafter\n" not in source
