"""Vendored drafter registry parity and binding probes (step 3c-1).

The vendored ``speculative/drafters`` package must stay byte-verbatim
against pinned ``mlx_vlm.speculative.drafters`` @ 0.7.1 except the
documented import redirects listed in the package ``__init__.py``
inventory. Every probe here fails closed: an undocumented edit anywhere in
a drafter module diverges.
"""

import inspect
import json
import sys
from pathlib import Path
from types import ModuleType

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
            "        # Documented pinned redirect: the deepseek_v4_dspark family is\n"
            "        # outside the served set (not vendored); detection must resolve\n"
            "        # the pinned splitter module.\n"
            "        from mlx_vlm.speculative.drafters.deepseek_v4_dspark.split import (\n"
            "            DeepseekV4DsparkSplitter,\n"
            "        )\n",
            "        from .deepseek_v4_dspark.split import DeepseekV4DsparkSplitter\n",
        ),
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
    "mtp_base.py": [
        (
            """        del cache
        if self._input_embed is None or self._lm_head_fn is None:
            raise RuntimeError(
                "bind(target_model) must be called before draft_block() "
                "so the drafter can use the target embeddings and LM head."
            )
        if block_size <= 1:
            # Rapid upstream-bugfix (documented deviation): pinned 0.7.1
            # crashes on mx.concatenate with an empty token list when
            # block_size <= 1 (also reachable through externally supplied
            # drafter repos that load_drafter cannot validate). Return the
            # DFlash2-shaped empty proposal instead.
            batch = 1 if isinstance(last_bonus, int) else int(last_bonus.shape[0])
            return mx.zeros((batch, 0), dtype=token_dtype)
""",
            """        del cache
        if self._input_embed is None or self._lm_head_fn is None:
            raise RuntimeError(
                "bind(target_model) must be called before draft_block() "
                "so the drafter can use the target embeddings and LM head."
            )
""",
        ),
        (
            """        # Rapid upstream-bugfix (documented deviation): pinned 0.7.1
        # dropped every row's bonus replay whenever any row lacked one,
        # leaving the other rows' caches and seeds stale. Mixed presence
        # is unsupported by the shared uniform-acceptance replay; fail
        # loudly instead of silently skipping.
        if any(new_tokens) and not all(new_tokens):
            raise ValueError(
                "mixed MTP bonus-token presence across replay rows is "
                "unsupported; all rows must carry a verifier bonus token"
            )
        if all(new_tokens):
            bonus = mx.array(
                [[int(row_tokens[-1])] for row_tokens in new_tokens],
                dtype=token_dtype,
            )""",
            """        if all(new_tokens):
            bonus = mx.array(
                [[int(row_tokens[-1])] for row_tokens in new_tokens],
                dtype=token_dtype,
            )""",
        ),
    ],
    "qwen3_dflash/dflash.py": [
        (
            """    def bind(self, target_model) -> "DFlashDraftModel":
        # Rapid upstream-bugfix (documented deviation): pinned 0.7.1
        # resolved the embeddings only when unset, so resetting with a
        # different target kept the previous target's embeddings while
        # swapping its LM head. Force re-resolution on every bind.
        self.embed_tokens = None
        if self.embed_tokens is None:""",
            """    def bind(self, target_model) -> "DFlashDraftModel":
        if self.embed_tokens is None:""",
        ),
    ],
    "qwen3_5_mtp/split.py": [
        (
            """            for proj in ("gate_proj", "up_proj", "down_proj"):
                # Rapid upstream-bugfix (documented deviation): quantized
                # checkpoints carry per-expert ``_scales``/``_biases``;
                # stack them alongside the weights so the runtime sees a
                # consistent switch_mlp layout (mirrors the gate_up_proj
                # handling above).
                for suffix in ("weight", "scales", "biases"):
                    keys = [
                        f"{prefix}.{e}.{proj}.{suffix}" for e in range(n_experts)
                    ]
                    if all(k in tensors for k in keys):
                        tensors[f"{base}.switch_mlp.{proj}.{suffix}"] = mx.stack(
                            [tensors.pop(k) for k in keys]
                        )""",
            """            for proj in ("gate_proj", "up_proj", "down_proj"):
                keys = [f"{prefix}.{e}.{proj}.weight" for e in range(n_experts)]
                if all(k in tensors for k in keys):
                    tensors[f"{base}.switch_mlp.{proj}.weight"] = mx.stack(
                        [tensors.pop(k) for k in keys]
                    )""",
        ),
    ],
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
            """        text_config = self.read_text_config(source_config)

        # Rapid upstream-bugfix (documented deviation): validate every
        # configuration argument BEFORE creating or writing the output —
        # rejected input must not leave a partially generated directory.
        # Minimum supported block size is 2: with 1 the MTP drafting loops
        # feed an empty token list into ``mx.concatenate`` and crash.
        resolved_block_size = (
            self.depth(text_config) + self.block_size_extra
            if block_size is None
            else int(block_size)
        )
        if resolved_block_size < 2:
            raise ValueError(f"block_size must be >= 2, got {block_size!r}")

        output_path.mkdir(parents=True, exist_ok=True)
""",
            """        text_config = self.read_text_config(source_config)

""",
        ),
        (
            """        draft_config = {
            "model_type": self.output_model_type,
            "text_config": text_config,
            "block_size": resolved_block_size,""",
            """        depth = self.depth(text_config)
        draft_config = {
            "model_type": self.output_model_type,
            "text_config": text_config,
            "block_size": int(block_size or depth + self.block_size_extra),""",
        ),
        (
            """        )
        output_path = Path(output)

        with open(source_path / "config.json") as f:""",
            """        )
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(source_path / "config.json") as f:""",
        ),
        (
            """from ...fp8 import transform_fp8_weights

# Documented pinned redirects: quant_utils/utils live at the mlx_vlm root
# and are vendored by later slices (quant_utils exists in this package;
# utils is step-3e scope).
from mlx_vlm.utils import get_model_path

from ...quant_utils import get_quantization_params
""",
            """from ...fp8 import transform_fp8_weights
from ...quant_utils import get_quantization_params
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


def test_mtp_split_block_size_resolution(tmp_path):
    """block_size defaults only when None; zero/negative are rejected."""
    from rapid_mlx.models.mlx_vlm_vendored.speculative.drafters.mtp_split import (
        MTPSplitter,
    )

    class StubSplitter(MTPSplitter):
        output_model_type = "qwen3_5_mtp"
        tokenizer_files = []

        def select_keys(self, key, text_config):
            return True

        def depth(self, text_config):
            return 3

        def transform(self, tensors, text_config, source_is_mlx):
            return {"w": mx.zeros((1,))}

        def quantization(self, weights, source_config, text_config, quant_opts):
            return None

    source = tmp_path / "src"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {"model_type": "qwen3_5", "num_hidden_layers": 4},
            }
        )
    )
    import mlx.core as mx

    mx.save_safetensors(
        str(source / "model.safetensors"),
        {"w": mx.zeros((1,))},
        metadata={"format": "mlx"},
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model.safetensors"}})
    )

    splitter = StubSplitter()
    for bad in (0, 1, -3):
        out_bad = tmp_path / f"out-bad-{bad}"
        with pytest.raises(ValueError, match="block_size must be >= 2"):
            splitter.split(str(source), str(out_bad), block_size=bad)
        # rejected input must not create the output directory
        assert not out_bad.exists()

    default_out = tmp_path / "out-default"
    splitter.split(str(source), str(default_out))
    assert json.loads((default_out / "config.json").read_text())["block_size"] == 4

    explicit_out = tmp_path / "out-explicit"
    splitter.split(str(source), str(explicit_out), block_size=2)
    assert json.loads((explicit_out / "config.json").read_text())["block_size"] == 2


def test_qwen3_next_postprocess_stacks_quantized_expert_metadata():
    """Per-expert scales/biases stack into the switch_mlp layout."""
    import mlx.core as mx

    from rapid_mlx.models.mlx_vlm_vendored.speculative.drafters.qwen3_5_mtp import (
        split as qwen_split_module,
    )

    splitter = qwen_split_module.Qwen3NextMTPSplitter()
    tensors = {}
    for expert in range(2):
        for proj, shape in (
            ("gate_proj", (2, 1)),
            ("up_proj", (2, 1)),
            ("down_proj", (1, 2)),
        ):
            tensors[f"blk.0.experts.{expert}.{proj}.weight"] = mx.full(
                shape, expert + 1
            )
            tensors[f"blk.0.experts.{expert}.{proj}.scales"] = mx.full((1,), expert + 1)
            tensors[f"blk.0.experts.{expert}.{proj}.biases"] = mx.zeros((1,))
    splitter.postprocess(tensors, {"num_experts": 2})

    for proj in ("gate_proj", "up_proj", "down_proj"):
        stacked = tensors[f"blk.0.switch_mlp.{proj}.weight"]
        assert stacked.shape[0] == 2
        assert f"blk.0.switch_mlp.{proj}.scales" in tensors
        assert f"blk.0.switch_mlp.{proj}.biases" in tensors
        for expert in range(2):
            for suffix in ("weight", "scales", "biases"):
                assert f"blk.0.experts.{expert}.{proj}.{suffix}" not in tensors


def test_dflash_bind_re_resolves_target_embeddings():
    """bind() must not keep a previous target's embeddings (r8 finding)."""
    from types import SimpleNamespace

    from rapid_mlx.models.mlx_vlm_vendored.speculative.drafters.qwen3_dflash import (
        dflash as dflash_module,
    )

    drafter = dflash_module.DFlashDraftModel.__new__(dflash_module.DFlashDraftModel)
    stale = object()
    drafter.embed_tokens = stale
    new_embed = object()
    head = lambda value: value
    drafter.bind(SimpleNamespace(embed_tokens=new_embed, lm_head=head))
    assert drafter.embed_tokens is new_embed
    assert drafter.lm_head is head


def test_mtp_base_rejects_mixed_bonus_presence():
    """Mixed bonus presence must fail loudly, not skip rows (r8 finding)."""
    import mlx.core as mx

    from rapid_mlx.models.mlx_vlm_vendored.speculative.drafters.mtp_base import (
        AutoregressiveMTPDraftModel,
    )

    drafter = AutoregressiveMTPDraftModel.__new__(AutoregressiveMTPDraftModel)
    drafter._cache = []
    drafter._next_position = 5
    drafter._round_appended = 0
    with pytest.raises(ValueError, match="mixed MTP bonus-token"):
        drafter.accept_verified_tokens_batch(
            mx.zeros((2, 2, 1)),
            mx.zeros((2, 2), dtype=mx.int32),
            [1, 1],
            [[5], []],
            None,
        )


def test_mtp_base_block_size_one_returns_empty_proposal():
    """block_size <= 1 returns a shaped empty proposal (r8 finding)."""
    import mlx.core as mx

    from rapid_mlx.models.mlx_vlm_vendored.speculative.drafters.mtp_base import (
        AutoregressiveMTPDraftModel,
    )

    drafter = AutoregressiveMTPDraftModel.__new__(AutoregressiveMTPDraftModel)
    drafter._input_embed = object()
    drafter._lm_head_fn = lambda value: value
    out = drafter.draft_block(5, mx.zeros((1, 1, 1)), None, 1, None, greedy=True)
    assert out.shape == (1, 0)


def test_detect_mtp_splitter_resolves_pinned_dspark_module(tmp_path, monkeypatch):
    """The unserved deepseek_v4_dspark family resolves through pinned."""
    from rapid_mlx.models.mlx_vlm_vendored.speculative.drafters.mtp_split import (
        detect_mtp_splitter,
    )

    class FakeDsparkSplitter:
        def read_text_config(self, source_config):
            return {}

        def iter_selected(self, model_path, text_config):
            yield model_path / "model.safetensors", ["blk.0.weight"]

    split_module = ModuleType("mlx_vlm.speculative.drafters.deepseek_v4_dspark.split")
    split_module.DeepseekV4DsparkSplitter = FakeDsparkSplitter
    monkeypatch.setitem(sys.modules, split_module.__name__, split_module)

    source = tmp_path / "src"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "dspark_target_layer_ids": [1, 8],
            }
        )
    )
    assert isinstance(detect_mtp_splitter(source), FakeDsparkSplitter)


def test_load_drafter_rejects_unknown_kind(tmp_path):
    from rapid_mlx.models.mlx_vlm_vendored.speculative.drafters import load_drafter

    with pytest.raises(ValueError, match="Unknown drafter kind"):
        load_drafter(str(tmp_path), kind="teleport")


def test_runtime_binds_vendored_registry(monkeypatch, tmp_path):
    """load_runtime must dispatch through the vendored registry seam."""
    from types import ModuleType

    import mlx_vlm as real_mlx_vlm

    import rapid_mlx.models.mlx_vlm_vendored.speculative.drafters as vendored_registry
    import rapid_mlx.speculative.native_mtp.runtime as runtime

    calls = []

    def vendored_marker(path, kind=None, **kwargs):
        calls.append("vendored")
        raise RuntimeError("VENDORED-SEAM-CALLED")

    def pinned_marker(path, kind=None, **kwargs):
        calls.append("pinned")
        raise RuntimeError("PINNED-SEAM-CALLED")

    root = ModuleType("mlx_vlm")
    root.__path__ = list(real_mlx_vlm.__path__)
    drafters = ModuleType("mlx_vlm.speculative.drafters")
    drafters.__path__ = []
    drafters.load_drafter = pinned_marker
    utils = ModuleType("mlx_vlm.utils")
    utils.get_model_path = lambda repo_id, revision=None: str(tmp_path)
    for name, module in {
        "mlx_vlm": root,
        "mlx_vlm.speculative.drafters": drafters,
        "mlx_vlm.utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(vendored_registry, "load_drafter", vendored_marker)

    with pytest.raises(RuntimeError, match="VENDORED-SEAM-CALLED"):
        runtime.load_runtime(
            str(tmp_path),
            target_revision="t",
            drafter_revision="d",
            block_size=8,
        )
    assert calls == ["vendored"]
