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
    "mtp_split.py": [
        (
            '    "qwen3_5": "rapid_mlx.models.mlx_vlm_vendored'
            '.speculative.drafters.qwen3_5_mtp.split:Qwen3_5MTPSplitter",\n',
            '    "qwen3_5": "mlx_vlm.speculative.drafters.qwen3_5_mtp.split:Qwen3_5MTPSplitter",\n',
        ),
        (
            '    "qwen3_5_moe": "rapid_mlx.models.mlx_vlm_vendored'
            '.speculative.drafters.qwen3_5_mtp.split:Qwen3_5MTPSplitter",\n',
            '    "qwen3_5_moe": "mlx_vlm.speculative.drafters.qwen3_5_mtp.split:Qwen3_5MTPSplitter",\n',
        ),
        (
            '    "qwen3_next": "rapid_mlx.models.mlx_vlm_vendored'
            '.speculative.drafters.qwen3_5_mtp.split:Qwen3NextMTPSplitter",\n',
            '    "qwen3_next": "mlx_vlm.speculative.drafters.qwen3_5_mtp.split:Qwen3NextMTPSplitter",\n',
        ),
        (
            '    "glm5_next": "rapid_mlx.models.mlx_vlm_vendored'
            '.speculative.drafters.glm5_next_mtp.split:Glm5NextMTPSplitter",\n',
            '    "glm5_next": "mlx_vlm.speculative.drafters.glm5_next_mtp.split:Glm5NextMTPSplitter",\n',
        ),
        (
            '    "glm5_next_text": "rapid_mlx.models.mlx_vlm_vendored'
            '.speculative.drafters.glm5_next_mtp.split:Glm5NextMTPSplitter",\n',
            '    "glm5_next_text": "mlx_vlm.speculative.drafters.glm5_next_mtp.split:Glm5NextMTPSplitter",\n',
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

# Documented upstream-bugfix deviations (see the package inventory):
# (file, vendored hunk, upstream hunk). Applying redirects then reverting
# these hunks must reproduce the pinned upstream bytes exactly.
DEVIATIONS = {
    "qwen3_5_mtp/qwen3_5_mtp.py": [
        (
            """                # Rapid upstream-bugfix (documented deviation): pinned
                # 0.7.1 skips the padding correction for a scalar
                # _next_position, so shorter rows keep too-large position
                # ids for the next round. Promote to per-row positions when
                # the padding is heterogeneous.
                padding = mx.array(right_padding, dtype=mx.int32)
                if isinstance(self._next_position, mx.array):
                    self._next_position = self._next_position - padding
                elif int(padding.min()) == int(padding.max()):
                    self._next_position = self._next_position - int(padding.min())
                else:
                    self._next_position = mx.full(
                        (len(right_padding),),
                        self._next_position,
                        dtype=mx.int32,
                    ) - padding
""",
            """                if isinstance(self._next_position, mx.array):
                    self._next_position = self._next_position - mx.array(
                        right_padding, dtype=mx.int32
                    )
""",
        ),
    ],
    "__init__.py": [
        (
            """    "qwen3_dspark": "dflash",
    # Rapid upstream-bugfix (documented deviation): pinned 0.7.1 omits the
    # served DFlash families' model types, so an explicit wrong --draft-kind
    # (e.g. "mtp") dispatched them through the wrong round loop instead of
    # being overridden here.
    "dflash2": "dflash",
    "qwen3_dflash": "dflash",
}
""",
            """    "qwen3_dspark": "dflash",
}
""",
        ),
    ],
    "mtp_split.py": [
        (
            """# Documented pinned redirects: quant_utils/utils live at the mlx_vlm root
# and are vendored by later slices (quant_utils exists in this package;
# utils is step-3e scope).
from mlx_vlm.utils import get_model_path

from ...quant_utils import get_quantization_params
""",
            """from ...quant_utils import get_quantization_params
from ...utils import get_model_path
""",
        ),
        (
            """                for filename, keys in by_file.items():
                    # Rapid upstream-bugfix (documented deviation): shard
                    # filenames come from an untrusted safetensors index;
                    # resolve and reject anything outside the model dir.
                    shard = (source_path / filename).resolve()
                    if not shard.is_relative_to(source_path.resolve()):
                        raise ValueError(
                            "safetensors index entry escapes the model "
                            f"directory: {filename!r}"
                        )
                    yield shard, keys
""",
            """                for filename, keys in by_file.items():
                    yield source_path / filename, keys
""",
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
        for vendored_hunk, upstream_hunk in DEVIATIONS.get(rel, []):
            if vendored.count(vendored_hunk) != 1:
                diverged.append(f"{rel}: documented deviation hunk not found")
                vendored = None
                break
            vendored = vendored.replace(vendored_hunk, upstream_hunk, 1)
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
    # The vendored table is the pinned table plus the two documented
    # upstream-bugfix entries (served DFlash model types).
    assert pinned_pkg.DRAFTER_KIND_BY_MODEL_TYPE.items() <= (
        reg.DRAFTER_KIND_BY_MODEL_TYPE.items()
    )
    assert set(reg.DRAFTER_KIND_BY_MODEL_TYPE) - set(
        pinned_pkg.DRAFTER_KIND_BY_MODEL_TYPE
    ) == {"dflash2", "qwen3_dflash"}
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


def test_resolve_drafter_kind_overrides_explicit_wrong_kind(tmp_path):
    """A DFlash repo given kind="mtp" must be overridden to "dflash"."""
    from rapid_mlx.models.mlx_vlm_vendored.speculative.drafters import (
        resolve_drafter_kind,
    )

    for model_type in ("qwen3_dflash", "dflash2"):
        repo = tmp_path / model_type
        repo.mkdir()
        (repo / "config.json").write_text(json.dumps({"model_type": model_type}))
        assert resolve_drafter_kind(repo, kind="mtp") == "dflash", model_type
        assert resolve_drafter_kind(repo, kind="dflash") == "dflash", model_type


def test_mtp_splitter_rejects_index_shards_outside_model_dir(tmp_path):
    """weight_map filenames from an untrusted index must stay in the dir."""
    from rapid_mlx.models.mlx_vlm_vendored.speculative.drafters.mtp_split import (
        MTPSplitter,
    )

    class AllKeys(MTPSplitter):
        def select_keys(self, key, text_config):
            return True

    source = tmp_path / "model"
    source.mkdir()
    outside = tmp_path / "evil.safetensors"
    outside.write_bytes(b"x")

    benign = AllKeys()
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"blk.0.mlp": "model-00001.safetensors"}})
    )
    yielded = list(benign.iter_selected(source, {}))
    assert yielded == [(source / "model-00001.safetensors", ["blk.0.mlp"])]

    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"blk.0.mlp": "../evil.safetensors"}})
    )
    with pytest.raises(ValueError, match="escapes the model directory"):
        list(benign.iter_selected(source, {}))

    absolute = AllKeys()
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"blk.0.mlp": str(outside)}})
    )
    with pytest.raises(ValueError, match="escapes the model directory"):
        list(absolute.iter_selected(source, {}))


def test_qwen_mtp_batch_replay_corrects_scalar_position_for_ragged_rows():
    """Heterogeneous right_padding must promote a scalar _next_position.

    Regression probe for the documented upstream-bugfix: pinned 0.7.1
    skips the padding correction when the tracked position is a scalar,
    so the shorter replayed row keeps too-large position ids.
    """
    import mlx.core as mx

    from rapid_mlx.models.mlx_vlm_vendored.speculative.drafters.qwen3_5_mtp import (
        qwen3_5_mtp as qwen_module,
    )

    drafter = qwen_module.Qwen3_5MTPDraftModel.__new__(qwen_module.Qwen3_5MTPDraftModel)
    drafter._cache = []
    drafter._round_appended = 0
    drafter._next_position = 7
    seeds = []
    drafter._forward_tokens = lambda tokens, hiddens, token_dtype: mx.zeros((2, 3, 2))
    drafter._set_seed_from_hidden = lambda last_hidden, sampler, greedy: seeds.append(
        last_hidden
    )

    verify_hidden = mx.zeros((2, 3, 2))
    draft_tokens = mx.array([[10, 11, 0], [20, 21, 0]], dtype=mx.int32)
    drafter.accept_verified_tokens_batch(
        verify_hidden,
        draft_tokens,
        accepted=[2, 1],
        new_tokens=[[7], [9]],
        sampler=None,
        token_dtype=mx.int32,
        greedy=True,
    )

    position = drafter._next_position
    assert isinstance(position, mx.array)
    assert position.tolist() == [7, 6]
    assert len(seeds) == 1


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
