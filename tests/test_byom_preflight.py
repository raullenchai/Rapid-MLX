# SPDX-License-Identifier: Apache-2.0
"""BYOM preflight: refuse provably unrunnable models before any download.

Every Hub call is mocked; no weights or network are touched. The golden
section at the bottom pins that models the preflight passes (or does not apply
to) reach ``serve_command`` / ``pull_command`` with byte-identical output to
``origin/main`` (fixtures captured there with the same harness).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest

from rapid_mlx import cli
from rapid_mlx.byom import preflight as pf
from rapid_mlx.telemetry import model_events

REPO_ROOT = Path(__file__).resolve().parents[1]
GIB = 1 << 30


# ---------------------------------------------------------------- fixtures


def _sibling(name: str, size: int = 10) -> types.SimpleNamespace:
    return types.SimpleNamespace(rfilename=name, size=size, lfs=None)


def _info(
    files: dict[str, int],
    *,
    config: dict | None = None,
    params: dict | None = None,
    sha: str = "abc123",
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        siblings=[_sibling(name, size) for name, size in files.items()],
        config=config,
        safetensors=(
            types.SimpleNamespace(parameters=params) if params is not None else None
        ),
        sha=sha,
    )


def _no_deadline(monkeypatch) -> None:
    monkeypatch.setattr(
        pf, "_call_with_deadline", lambda fn, *a, **kw: fn(*a, **kw), raising=True
    )


SUPPORTED = frozenset({"qwen3", "llama", "mistral"})
MLX_CONFIG = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "quantization_config": {"bits": 4},
    "tokenizer_config": {"chat_template": "{{ x }}"},
}


# ------------------------------------------------------------ file helpers


def test_gguf_quants_extracts_unique_labels_in_order():
    files = [
        "Model-Q4_K_M.gguf",
        "sub/Model-q5_k_m.gguf",
        "Model-Q4_K_M-00002.gguf",
        "Model-IQ3_XS.gguf",
        "model-bf16.gguf",
        "plain.gguf",
        "README.md",
    ]
    assert pf.gguf_quants(files) == ("Q4_K_M", "Q5_K_M", "IQ3_XS", "BF16")


@pytest.mark.parametrize(
    "files,expected",
    [
        (["a.gguf", "README.md"], "gguf"),
        (["a.gguf", "model.safetensors"], None),
        (["q4/model.safetensors", "a.gguf"], None),
        (["weights.npz"], None),
        (["pytorch_model.bin", "config.json"], "pytorch"),
        (["pytorch_model-00001-of-00002.bin"], "pytorch"),
        (["sub/pytorch_model.bin"], None),
        (["training_args.bin", "config.json"], None),
        (["config.json"], None),
    ],
)
def test_classify_format(files, expected):
    assert pf.classify_format(files) == expected


# -------------------------------------------------------- loader inventory


def test_dict_literal_keys_branches(tmp_path):
    good = tmp_path / "good.py"
    good.write_text("X = 1\nMODEL_REMAPPING = {'a': 'b', 'c': 'd'}\n")
    assert pf._dict_literal_keys(good, "MODEL_REMAPPING") == {"a", "c"}
    assert pf._dict_literal_keys(good, "MISSING") is None
    assert pf._dict_literal_keys(tmp_path / "absent.py", "X") is None
    bad = tmp_path / "bad.py"
    bad.write_text("MODEL_REMAPPING = {\n")
    assert pf._dict_literal_keys(bad, "MODEL_REMAPPING") is None
    computed = tmp_path / "computed.py"
    computed.write_text("MODEL_REMAPPING = dict(a=1)\n")
    assert pf._dict_literal_keys(computed, "MODEL_REMAPPING") is None
    not_dict = tmp_path / "list.py"
    not_dict.write_text("MODEL_REMAPPING = ['a']\n")
    assert pf._dict_literal_keys(not_dict, "MODEL_REMAPPING") is None


def test_module_names(tmp_path):
    (tmp_path / "llama.py").write_text("")
    (tmp_path / "_private.py").write_text("")
    (tmp_path / "notes.txt").write_text("")
    pkg = tmp_path / "gemma4"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (tmp_path / "not_a_pkg").mkdir()
    assert pf._module_names(tmp_path) == {"llama", "gemma4"}
    assert pf._module_names(tmp_path / "missing") == set()


def test_package_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(
        pf.importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace(submodule_search_locations=[str(tmp_path)]),
    )
    assert pf._package_dir("x") == tmp_path
    monkeypatch.setattr(pf.importlib.util, "find_spec", lambda name: None)
    assert pf._package_dir("x") is None
    monkeypatch.setattr(
        pf.importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace(submodule_search_locations=None),
    )
    assert pf._package_dir("x") is None

    def _boom(name):
        raise ImportError(name)

    monkeypatch.setattr(pf.importlib.util, "find_spec", _boom)
    assert pf._package_dir("x") is None


def _fake_package(root: Path, models: list[str], remap: dict) -> Path:
    (root / "models").mkdir(parents=True)
    for name in models:
        (root / "models" / f"{name}.py").write_text("")
    (root / "utils.py").write_text(f"MODEL_REMAPPING = {remap!r}\n")
    return root


def test_installed_types(monkeypatch, tmp_path):
    root = _fake_package(tmp_path / "pkg", ["llama"], {"mistral": "llama"})
    monkeypatch.setattr(pf, "_package_dir", lambda name: root)
    assert pf._installed_types("pkg", ("models",)) == {"llama", "mistral"}
    monkeypatch.setattr(pf, "_package_dir", lambda name: None)
    assert pf._installed_types("pkg", ("models",)) is None
    (root / "utils.py").write_text("")
    monkeypatch.setattr(pf, "_package_dir", lambda name: root)
    assert pf._installed_types("pkg", ("models",)) is None


def test_supported_model_types(monkeypatch, tmp_path):
    lm = _fake_package(tmp_path / "lm", ["llama"], {"mistral": "llama"})
    vlm = _fake_package(tmp_path / "vlm", ["gemma4"], {})
    dirs = {"mlx_lm": lm, "mlx_vlm": vlm}
    monkeypatch.setattr(pf, "_package_dir", lambda name: dirs.get(name))
    types_ = pf.supported_model_types()
    assert {"llama", "mistral", "gemma4", "deepseek_v4"} <= types_
    assert "qwen3_vl" not in types_

    # mlx-vlm (the optional [vision] extra) absent: the pinned snapshot stands
    # in so a model the extra would load is never called unsupported.
    del dirs["mlx_vlm"]
    assert "qwen3_vl" in pf.supported_model_types()

    audio = tmp_path / "audio"
    for sub, name in (("stt", "whisperish"), ("tts", "kokoro")):
        (audio / sub / "models" / name).mkdir(parents=True)
        (audio / sub / "models" / name / "__init__.py").write_text("")
    dirs["mlx_audio"] = audio
    assert {"whisperish", "kokoro"} <= pf.supported_model_types()

    del dirs["mlx_lm"]
    assert pf.supported_model_types() is None


def test_mlx_lm_version(monkeypatch):
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.9.9")
    assert pf._mlx_lm_version() == "9.9.9"

    def _missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)
    assert pf._mlx_lm_version() is None


@pytest.mark.parametrize(
    "config,supported,expected",
    [
        ({"model_type": "qwen3"}, SUPPORTED, True),
        ({"model_type": "Llama"}, SUPPORTED, True),
        ({"model_type": "qwen3"}, None, None),
        ({}, SUPPORTED, None),
        ({"model_type": ""}, SUPPORTED, None),
        ({"model_type": 3}, SUPPORTED, None),
        (
            {
                "model_type": "x",
                "architectures": ["XForCausalLM"],
                "model_file": "m.py",
            },
            SUPPORTED,
            None,
        ),
        (
            {"model_type": "x", "architectures": ["XForCausalLM"], "dflash_config": {}},
            SUPPORTED,
            None,
        ),
        ({"model_type": "x", "architectures": "XForCausalLM"}, SUPPORTED, None),
        ({"model_type": "x", "architectures": ["XModel"]}, SUPPORTED, None),
        ({"model_type": "x"}, SUPPORTED, None),
        ({"model_type": "x", "architectures": ["XForCausalLM"]}, SUPPORTED, False),
        (
            {"model_type": "x", "architectures": [1, "XForConditionalGeneration"]},
            SUPPORTED,
            None,
        ),
        ({"model_type": "x", "architectures": ["XLMHeadModel"]}, SUPPORTED, False),
    ],
)
def test_architecture_supported(config, supported, expected):
    assert pf.architecture_supported(config, supported) is expected


def test_conditional_generation_needs_a_chat_template_to_be_refused():
    """Whisper-style speech models share the suffix with VLMs: never refuse
    them on the architecture name alone (they may be ``pull``ed for audio)."""
    whisper = {
        "model_type": "whisper",
        "architectures": ["WhisperForConditionalGeneration"],
    }
    assert pf.architecture_supported(whisper, SUPPORTED) is None
    assert pf.architecture_supported(whisper, SUPPORTED, chat_template=False) is None
    assert pf.architecture_supported(whisper, SUPPORTED, chat_template=True) is False


def test_live_inventory_knows_common_architectures():
    """Against the installed loaders (when present), mainstream types pass."""
    supported = pf.supported_model_types()
    if supported is None:
        pytest.skip("mlx-lm is not installed in this lane")
    assert {"llama", "mistral", "qwen3", "gemma3", "qwen3_5"} <= supported


def test_vendored_types_cover_every_rapid_registration():
    """Every ``mlx_lm.models.<type>`` Rapid-MLX registers is known here."""
    pattern = re.compile(
        r"sys\.modules(?:\.setdefault\(\s*|\[\s*)[\"']mlx_lm\.models\.([a-z0-9_]+)[\"']"
    )
    found: set[str] = set()
    for path in (REPO_ROOT / "rapid_mlx").rglob("*.py"):
        found |= set(pattern.findall(path.read_text(encoding="utf-8")))
    assert found, "registration scan found nothing; pattern drifted"
    assert found <= pf.RAPID_MODEL_TYPES


def test_vlm_snapshot_matches_pinned_runtime():
    from rapid_mlx.byom import _vlm_model_types as snap

    pin = re.search(
        r'"mlx-vlm==([^"]+)"', (REPO_ROOT / "pyproject.toml").read_text()
    ).group(1)
    assert pin == snap.MLX_VLM_PIN
    live = pf._installed_types(
        "mlx_vlm", ("models", str(Path("speculative") / "drafters"))
    )
    if live is None:
        pytest.skip("mlx-vlm is not installed in this lane")
    import importlib.metadata

    if importlib.metadata.version("mlx-vlm") != pin:
        pytest.skip("installed mlx-vlm is not the pinned release")
    assert set(snap.MLX_VLM_MODEL_TYPES) == live


# --------------------------------------------------------------- inspection


def test_quant_bits():
    assert pf._quant_bits({"quantization": {"bits": 4}}) == 4
    assert pf._quant_bits({"quantization_config": {"bits": 8}}) == 8
    assert pf._quant_bits({"quantization_config": {"quant_method": "awq"}}) is None
    assert pf._quant_bits({}) is None


def test_hub_dtype():
    assert pf._hub_dtype(_info({}, params={"BF16": 10, "F32": 2})) == "bf16"
    assert pf._hub_dtype(_info({}, params={"U32": 10, "BF16": 2})) is None
    assert pf._hub_dtype(_info({}, params={})) is None
    assert pf._hub_dtype(_info({})) is None


def test_call_with_deadline_uses_the_shared_helper(monkeypatch):
    from rapid_mlx import _download_gate

    seen = {}

    def _fake(fn, timeout, *a, **kw):
        seen["timeout"] = timeout
        return fn(*a, **kw)

    monkeypatch.setattr(_download_gate, "call_with_deadline", _fake)
    assert pf._call_with_deadline(lambda x, y=0: x + y, 1, y=2) == 3
    assert seen["timeout"] == pf._HUB_TIMEOUT_SECONDS


def test_fetch_hub_config(monkeypatch):
    _no_deadline(monkeypatch)
    import huggingface_hub

    seen = {}

    def _download(repo, filename, *, revision, local_dir):
        seen.update(repo=repo, filename=filename, revision=revision)
        path = Path(local_dir) / filename
        path.write_text(json.dumps({"model_type": "x"}))
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    assert pf._fetch_hub_config("o/r", "sha1") == {"model_type": "x"}
    assert seen == {"repo": "o/r", "filename": "config.json", "revision": "sha1"}
    # The worker removes its own temp dir (a timed-out caller must not).
    dirs = []

    def _tracking(repo, filename, *, revision, local_dir):
        dirs.append(local_dir)
        return _download(repo, filename, revision=revision, local_dir=local_dir)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _tracking)
    pf._fetch_hub_config("o/r", None)
    assert dirs and not Path(dirs[0]).exists()

    def _list_download(repo, filename, *, revision, local_dir):
        path = Path(local_dir) / filename
        path.write_text("[1]")
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _list_download)
    assert pf._fetch_hub_config("o/r", None) is None

    def _fail(*a, **kw):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fail)
    assert pf._fetch_hub_config("o/r", None) is None


def test_inspect_hub_reads_one_metadata_call(monkeypatch):
    _no_deadline(monkeypatch)
    import huggingface_hub

    calls = []

    def _model_info(repo, files_metadata):
        calls.append((repo, files_metadata))
        return _info(
            {
                "model-00001.safetensors": 3 * GIB,
                "model-00002.safetensors": 1 * GIB,
                "adapter_model.safetensors": 20 * GIB,
                "consolidated.safetensors": 4 * GIB,
                "q8/model.safetensors": 9 * GIB,
                "README.md": 5,
            },
            config=dict(MLX_CONFIG),
            params={"BF16": 1},
        )

    monkeypatch.setattr(huggingface_hub, "model_info", _model_info)
    insp = pf.inspect_hub("o/r")
    assert calls == [("o/r", True)]
    assert insp.weight_bytes == 4 * GIB
    assert insp.has_chat_template is True
    assert "tokenizer_config" not in insp.config
    assert insp.dtype == "bf16"
    assert insp.revision == "abc123"
    assert not insp.is_local


@pytest.mark.parametrize(
    "files,config,expected",
    [
        ({"chat_template.jinja": 1}, None, True),
        ({"chat_template.json": 1}, {"model_type": "x"}, True),
        ({}, {"tokenizer_config": {"chat_template": ""}}, False),
        ({}, {"tokenizer_config": "odd"}, None),
        ({}, None, None),
    ],
)
def test_inspect_hub_chat_template(monkeypatch, files, config, expected):
    _no_deadline(monkeypatch)
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "model_info", lambda *a, **kw: _info(files, config=config)
    )
    assert pf.inspect_hub("o/r").has_chat_template is expected


def test_inspect_hub_failure_is_no_verdict(monkeypatch):
    _no_deadline(monkeypatch)
    import huggingface_hub

    def _boom(*a, **kw):
        raise TimeoutError

    monkeypatch.setattr(huggingface_hub, "model_info", _boom)
    assert pf.inspect_hub("o/r") is None


def test_inspect_hub_tolerates_missing_siblings(monkeypatch):
    _no_deadline(monkeypatch)
    import huggingface_hub

    info = types.SimpleNamespace(siblings=None, config=None, safetensors=None, sha=None)
    monkeypatch.setattr(huggingface_hub, "model_info", lambda *a, **kw: info)
    insp = pf.inspect_hub("o/r")
    assert insp.files == ()
    assert insp.weight_bytes == 0


def test_inspect_local_directory(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"x" * 999)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "torch_dtype": "bfloat16"})
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ x }}"})
    )
    insp = pf.inspect_local(str(tmp_path))
    assert insp.is_local
    assert insp.weight_bytes == 100
    assert insp.has_chat_template is True
    assert insp.dtype == "bfloat16"
    assert "config.json" in insp.files


def test_inspect_local_without_config(tmp_path):
    (tmp_path / "model-q4.gguf").write_bytes(b"x")
    insp = pf.inspect_local(str(tmp_path))
    assert insp.config == {}
    assert insp.has_chat_template is None
    assert insp.dtype is None


def test_inspect_local_file_and_missing(tmp_path, monkeypatch):
    weight = tmp_path / "model-Q4_K_M.gguf"
    weight.write_bytes(b"x")
    insp = pf.inspect_local(str(weight))
    assert insp.files == ("model-Q4_K_M.gguf",)
    assert pf.inspect_local(str(tmp_path / "nope")) is None

    def _deny(self):
        raise PermissionError

    monkeypatch.setattr(Path, "iterdir", _deny)
    assert pf.inspect_local(str(tmp_path)) is None


def test_inspect_local_metadata_unreadable(tmp_path, monkeypatch):
    import rapid_mlx.model_metadata as mm

    monkeypatch.setattr(mm, "read_local_model_metadata", lambda path: None)
    insp = pf.inspect_local(str(tmp_path))
    assert insp.config == {}
    assert insp.has_chat_template is None


# ------------------------------------------------------------------ verdict


def _insp(**overrides) -> pf.Inspection:
    fields = {
        "ref": "o/r",
        "is_local": False,
        "files": ("model.safetensors", "config.json"),
        "weight_bytes": 4 * GIB,
        "config": {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
        "has_chat_template": True,
    }
    fields.update(overrides)
    return pf.Inspection(**fields)


@pytest.mark.parametrize(
    "insp,label",
    [
        (_insp(config={"quantization": {"bits": 4}}), "MLX · 4-bit"),
        (_insp(dtype="bfloat16"), "safetensors · bf16"),
        (_insp(dtype="F16"), "safetensors · f16"),
        (_insp(), "safetensors"),
    ],
)
def test_format_label(insp, label):
    assert pf._format_label(insp, None) == label


def test_physical_ram_bytes(monkeypatch):
    psutil = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(total=36 * GIB)
    )
    monkeypatch.setitem(sys.modules, "psutil", psutil)
    assert pf.physical_ram_bytes() == 36 * GIB
    psutil.virtual_memory = lambda: types.SimpleNamespace(total=0)
    assert pf.physical_ram_bytes() is None

    def _boom():
        raise RuntimeError

    psutil.virtual_memory = _boom
    assert pf.physical_ram_bytes() is None


def _evaluate(insp, *, refuse=True, ram=36 * GIB):
    return pf.evaluate(insp, refuse_oversize=refuse, supported=SUPPORTED, ram_bytes=ram)


def test_evaluate_gguf_only():
    verdict = _evaluate(_insp(files=("m-Q4_K_M.gguf", "m-Q5_K_M.gguf"), config={}))
    assert verdict.failure == pf.UNSUPPORTED_FORMAT
    assert verdict.format_label == "GGUF"
    assert verdict.gguf_quants == ("Q4_K_M", "Q5_K_M")
    assert verdict.model_type is None


def test_evaluate_pytorch_only():
    verdict = _evaluate(_insp(files=("pytorch_model.bin",)))
    assert verdict.failure == pf.UNSUPPORTED_FORMAT
    assert verdict.format_label == "PyTorch .bin"
    assert verdict.model_type == "qwen3"


def test_evaluate_unsupported_architecture():
    config = {"model_type": "spark_x", "architectures": ["SparkXForCausalLM"]}
    verdict = _evaluate(_insp(config=config))
    assert verdict.failure == pf.UNSUPPORTED_ARCHITECTURE
    assert verdict.model_type == "spark_x"
    assert verdict.architecture_supported is False


def test_evaluate_fit_levels():
    assert _evaluate(_insp(weight_bytes=4 * GIB)).fit == "yes"
    tight = _evaluate(_insp(weight_bytes=30 * GIB))
    assert tight.fit == "tight" and tight.failure is None
    too_big = _evaluate(_insp(weight_bytes=40 * GIB))
    assert too_big.fit == "no" and too_big.failure == pf.INSUFFICIENT_MEMORY
    assert too_big.est_memory_bytes == int(40 * GIB * 1.2 + GIB)
    stored = _evaluate(_insp(weight_bytes=40 * GIB), refuse=False)
    assert stored.fit == "no" and stored.failure is None
    assert _evaluate(_insp(weight_bytes=0)).fit == "unknown"
    assert _evaluate(_insp(), ram=None).fit == "unknown"


# ---------------------------------------------------------------- rendering


def test_render_gguf_hub_and_local():
    verdict = _evaluate(_insp(ref="someone/Cool-12B-GGUF", files=("a-Q4_K_M.gguf",)))
    lines = pf.render_failure(
        _insp(ref="someone/Cool-12B-GGUF", files=("a-Q4_K_M.gguf",)), verdict
    )
    assert lines[0] == "! someone/Cool-12B-GGUF only has GGUF files (Q4_K_M)."
    assert "Nothing was downloaded." in lines[1]
    assert lines[-1].strip() == "https://huggingface.co/models?search=Cool-12B%20mlx"

    local = _insp(ref="/m/x.gguf", is_local=True, files=("x.gguf",))
    lines = pf.render_failure(local, _evaluate(local))
    assert lines == [
        "! /m/x.gguf only has GGUF files.",
        "  Rapid-MLX runs MLX and safetensors weights, not GGUF.",
    ]


def test_hf_search_url_keeps_names_without_gguf_suffix():
    assert pf._hf_search_url("a/Model 7B") == (
        "https://huggingface.co/models?search=Model%207B%20mlx"
    )
    assert pf._hf_search_url("a/GGUF").endswith("search=GGUF%20mlx")


def test_render_pytorch():
    insp = _insp(files=("pytorch_model.bin",))
    lines = pf.render_failure(insp, _evaluate(insp))
    assert lines[0] == "! o/r only has PyTorch .bin weights."


def test_render_architecture(monkeypatch):
    insp = _insp(config={"model_type": "spark_x", "architectures": ["SForCausalLM"]})
    verdict = _evaluate(insp)
    monkeypatch.setattr(pf, "_mlx_lm_version", lambda: "0.31.3")
    lines = pf.render_failure(insp, verdict)
    assert lines[0] == "✗ Architecture spark_x is not supported yet."
    assert "(mlx-lm 0.31.3)" in lines[1]
    monkeypatch.setattr(pf, "_mlx_lm_version", lambda: None)
    assert "mlx-lm" not in pf.render_failure(insp, verdict)[1]


def test_render_memory():
    insp = _insp(weight_bytes=40 * GIB)
    lines = pf.render_failure(insp, _evaluate(insp))
    assert lines[0] == "✗ o/r needs ~49 GB of memory; this Mac has 36 GB."
    assert lines[1] == "  Its weights alone are 40 GB. Nothing was downloaded."


def test_render_pass_full_and_sparse():
    insp = _insp(weight_bytes=5 * GIB, config=dict(MLX_CONFIG))
    lines = pf.render_pass(insp, _evaluate(insp))
    assert lines == [
        "✓ Checked before download",
        "  Format        MLX · 4-bit · 5.0 GB",
        "  Architecture  qwen3 (supported)",
        "  Chat template found",
        "  Fits your Mac yes · ~7.0 GB of 36 GB",
    ]
    sparse = _insp(weight_bytes=0, config={}, has_chat_template=None)
    assert pf.render_pass(sparse, _evaluate(sparse)) == [
        "✓ Checked before download",
        "  Format        safetensors",
    ]
    unknown = _insp(
        config={"model_type": "odd", "architectures": ["OddModel"]},
        has_chat_template=False,
    )
    lines = pf.render_pass(unknown, _evaluate(unknown))
    assert "  Architecture  odd (not verified)" in lines
    assert any(line.startswith("  Chat template none") for line in lines)


# ------------------------------------------------------------------ CLI hook


def _args(**kw) -> argparse.Namespace:
    base = {"command": "serve", "model": "o/r"}
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def uncached(monkeypatch):
    from rapid_mlx import _download_gate

    monkeypatch.setattr(_download_gate, "is_repo_cached", lambda name: False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)


def test_preflight_target_applies_to_uncataloged_repos(uncached):
    assert pf.preflight_target(_args()) == ("o/r", False)
    assert pf.preflight_target(_args(command="pull", bits=None, format=None)) == (
        "o/r",
        False,
    )


@pytest.mark.parametrize(
    "args",
    [
        _args(command="chat"),
        _args(model=None),
        _args(model=""),
        _args(no_preflight=True),
        _args(command="pull", bits="4", format=None),
        _args(command="pull", bits=None, format="mxfp4"),
        _args(model="not-a-repo"),
        _args(model="mlx-community/Qwen3.5-9B-4bit"),
        _args(model="mlx-community/whisper-tiny-mlx"),
    ],
)
def test_preflight_target_skips(uncached, args):
    assert pf.preflight_target(args) is None


def test_preflight_target_skips_cached_and_offline(monkeypatch, uncached):
    from rapid_mlx import _download_gate

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert pf.preflight_target(_args()) is None
    monkeypatch.delenv("HF_HUB_OFFLINE")
    monkeypatch.setattr(_download_gate, "is_repo_cached", lambda name: True)
    assert pf.preflight_target(_args()) is None


def test_preflight_target_local_path(tmp_path):
    assert pf.preflight_target(_args(model=str(tmp_path))) == (str(tmp_path), True)


def test_emit_rejection_serve(monkeypatch):
    from rapid_mlx.telemetry import model_events as me
    from rapid_mlx.telemetry import server_start

    seen = {}
    monkeypatch.setattr(server_start, "set_failure_stage", lambda s: seen.update(s=s))
    monkeypatch.setattr(
        me,
        "emit_model_serve_failed",
        lambda exc, alias_or_path: seen.update(exc=exc, ref=alias_or_path),
    )
    exc = pf.PreflightRejectedError(pf.UNSUPPORTED_FORMAT)
    pf._emit_rejection(_args(_original_alias="my-alias"), exc)
    assert seen == {"s": "preflight", "exc": exc, "ref": "my-alias"}


def test_emit_rejection_pull(monkeypatch):
    from rapid_mlx.telemetry import model_events as me

    seen = {}
    monkeypatch.setattr(
        me,
        "emit_model_pull_failed",
        lambda exc, model_ref: seen.update(exc=exc, ref=model_ref),
    )
    exc = pf.PreflightRejectedError(pf.UNSUPPORTED_ARCHITECTURE)
    pf._emit_rejection(_args(command="pull"), exc)
    assert seen == {"exc": exc, "ref": "o/r"}


@pytest.fixture
def hook(monkeypatch, uncached):
    """Run the CLI hook against a mocked Hub; returns the emitted rejections."""
    _no_deadline(monkeypatch)
    emitted: list[pf.PreflightRejectedError] = []
    monkeypatch.setattr(pf, "_emit_rejection", lambda args, exc: emitted.append(exc))
    monkeypatch.setattr(pf, "supported_model_types", lambda: SUPPORTED)
    monkeypatch.setattr(pf, "physical_ram_bytes", lambda: 36 * GIB)
    monkeypatch.setattr(pf, "_mlx_lm_version", lambda: "0.31.3")
    from rapid_mlx.byom import alternatives

    hints: dict[str, object] = {"value": ([], False)}
    monkeypatch.setattr(alternatives, "suggest", lambda *a, **kw: hints["value"])
    from rapid_mlx.byom import support_request

    offered: list[tuple] = []
    monkeypatch.setattr(support_request, "offer", lambda *a: offered.append(a[1:]))

    def run(info=None, args=None, *, tty=False, full_config=None):
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "model_info", lambda *a, **kw: info)
        monkeypatch.setattr(pf, "_fetch_hub_config", lambda repo, rev: full_config)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: tty)
        pf.run_cli_preflight(
            args if args is not None else _args(),
            spinner_factory=lambda label: nullcontext(),
        )

    run.emitted = emitted
    run.hints = hints
    run.offered = offered
    return run


def test_hook_refuses_gguf_only_repo(hook, capsys):
    info = _info({"m-Q4_K_M.gguf": 7 * GIB, "README.md": 1})
    with pytest.raises(SystemExit) as exc:
        hook(info)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "! o/r only has GGUF files (Q4_K_M)." in err
    assert "Nothing was downloaded." in err
    assert "--no-preflight" in err
    assert [e.failure_class for e in hook.emitted] == [pf.UNSUPPORTED_FORMAT]


def test_hook_confirms_architecture_against_full_config(hook, capsys):
    config = {"model_type": "spark_x", "architectures": ["SparkXForCausalLM"]}
    info = _info({"model.safetensors": GIB}, config=config)
    # The full config.json routes it elsewhere (or cannot be read): no verdict.
    hook(info, full_config=None)
    hook(info, full_config={**config, "model_file": "custom.py"})
    assert hook.emitted == []
    with pytest.raises(SystemExit):
        hook(info, full_config=config)
    assert "Architecture spark_x is not supported yet." in capsys.readouterr().err
    assert [e.failure_class for e in hook.emitted] == [pf.UNSUPPORTED_ARCHITECTURE]


def test_hook_refuses_oversize_serve_but_not_pull_or_disk_stream(hook, capsys):
    info = _info({"model.safetensors": 40 * GIB}, config=dict(MLX_CONFIG))
    hook(info, _args(command="pull", bits=None, format=None))
    hook(info, _args(disk_stream=True))
    assert hook.emitted == []
    with pytest.raises(SystemExit):
        hook(info)
    assert [e.failure_class for e in hook.emitted] == [pf.INSUFFICIENT_MEMORY]


def test_hook_pass_is_silent_off_tty_and_summarises_on_tty(hook, capsys):
    info = _info({"model.safetensors": 5 * GIB}, config=dict(MLX_CONFIG))
    args = _args()
    before = dict(vars(args))
    hook(info, args)
    assert capsys.readouterr() == ("", "")
    assert vars(args) == before
    hook(info, args, tty=True)
    out = capsys.readouterr().out
    assert "✓ Checked before download" in out
    assert "Fits your Mac yes · ~7.0 GB of 36 GB" in out


def test_hook_unreadable_metadata_is_silent(hook, capsys, monkeypatch):
    def _boom(*a, **kw):
        raise OSError("down")

    import huggingface_hub

    hook(None, _args(command="chat"))
    monkeypatch.setattr(pf, "inspect_hub", lambda ref: None)
    pf.run_cli_preflight(_args(), spinner_factory=lambda label: nullcontext())
    assert capsys.readouterr() == ("", "")
    assert hook.emitted == []
    del huggingface_hub, _boom


def test_hook_local_paths(hook, capsys, tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    (good / "model.safetensors").write_bytes(b"x")
    (good / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    hook(None, _args(model=str(good)), tty=True)
    assert capsys.readouterr() == ("", "")

    gguf = tmp_path / "m-Q8_0.gguf"
    gguf.write_bytes(b"x")
    with pytest.raises(SystemExit):
        hook(None, _args(model=str(gguf)))
    err = capsys.readouterr().err
    assert "only has GGUF files (Q8_0)." in err
    assert "Nothing was downloaded" not in err


def test_hook_local_path_that_vanishes_is_silent(hook, capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "inspect_local", lambda ref: None)
    hook(None, _args(model=str(tmp_path)))
    assert capsys.readouterr() == ("", "")


def test_hook_rejects_pull_format_gguf_before_anything(hook, capsys):
    with pytest.raises(SystemExit) as exc:
        hook(None, _args(command="pull", bits=None, format="GGUF"))
    assert exc.value.code == 1
    assert "`--format gguf` is not available" in capsys.readouterr().err
    assert [e.failure_class for e in hook.emitted] == [pf.UNSUPPORTED_FORMAT]


def test_resolve_variant_rejects_gguf_without_listing_the_repo(monkeypatch):
    import huggingface_hub

    def _no_listing(*a, **kw):
        raise AssertionError("must not list the repo")

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_tree", _no_listing)
    with pytest.raises(ValueError, match="cannot run GGUF"):
        cli._resolve_variant_allow_patterns("o/r", None, "gguf")


def test_parsers_expose_no_preflight():
    parser = cli.build_parser()
    assert parser.parse_args(["serve", "o/r", "--no-preflight"]).no_preflight
    assert parser.parse_args(["pull", "o/r", "--no-preflight"]).no_preflight
    assert parser.parse_args(["pull", "o/r"]).no_preflight is False
    help_text = parser._subparsers._group_actions[0].choices["pull"].format_help()
    assert "--format gguf" not in help_text


# ---------------------------------------------------------------- telemetry


def test_pull_error_class_for_preflight_rejections():
    for cls in (pf.UNSUPPORTED_FORMAT, pf.UNSUPPORTED_ARCHITECTURE):
        assert model_events.pull_error_class(pf.PreflightRejectedError(cls)) == cls
    assert (
        model_events.pull_error_class(pf.PreflightRejectedError(pf.INSUFFICIENT_MEMORY))
        == "other"
    )


def test_serve_error_class_for_preflight_rejections():
    for cls in (
        pf.UNSUPPORTED_FORMAT,
        pf.UNSUPPORTED_ARCHITECTURE,
        pf.INSUFFICIENT_MEMORY,
    ):
        assert model_events.serve_error_class(pf.PreflightRejectedError(cls)) == cls


def test_preflight_classes_are_registry_values():
    from rapid_mlx.telemetry import registry

    for cls in (pf.UNSUPPORTED_FORMAT, pf.UNSUPPORTED_ARCHITECTURE):
        assert registry.validate("model_pull_failed", {"error_class": cls})
    for cls in (
        pf.UNSUPPORTED_FORMAT,
        pf.UNSUPPORTED_ARCHITECTURE,
        pf.INSUFFICIENT_MEMORY,
    ):
        assert registry.validate("model_serve_failed", {"error_class": cls})


# ------------------------------------------------------------------- golden
#
# ``main()`` for models the preflight passes or does not apply to must print
# exactly what origin/main printed. The expected strings were captured by
# running this harness against origin/main 9221ff773 (where the Hub mock is
# simply never consulted).


def _run_main_golden(monkeypatch, capsys, argv, info):
    import huggingface_hub

    from rapid_mlx import _download_gate

    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    monkeypatch.setenv("RAPID_MLX_AUTO_PULL", "1")
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "--no-telemetry", *argv])
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(_download_gate, "is_repo_cached", lambda name: False)
    monkeypatch.setattr(huggingface_hub, "model_info", lambda *a, **kw: info)
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        cli, "serve_command", lambda args: seen.append(("serve", args.model))
    )
    monkeypatch.setattr(
        cli, "pull_command", lambda args: seen.append(("pull", args.model))
    )
    capsys.readouterr()
    cli.main()
    out, err = capsys.readouterr()
    return seen, out, err


GOLDEN_PASSING = _info(
    {"model.safetensors": 2 * GIB, "config.json": 1},
    config=dict(MLX_CONFIG),
)


@pytest.mark.parametrize(
    "argv,dispatched,expected_out",
    [
        (
            ["pull", "someone/Custom-4B-MLX-4bit"],
            ("pull", "someone/Custom-4B-MLX-4bit"),
            "",
        ),
        (
            ["serve", "someone/Custom-4B-MLX-4bit"],
            ("serve", "someone/Custom-4B-MLX-4bit"),
            "",
        ),
        (
            ["pull", "qwen3.5-4b-4bit"],
            ("pull", "mlx-community/Qwen3.5-4B-MLX-4bit"),
            "  Alias: qwen3.5-4b-4bit → mlx-community/Qwen3.5-4B-MLX-4bit\n",
        ),
    ],
)
def test_golden_working_paths_are_unchanged(
    monkeypatch, capsys, argv, dispatched, expected_out
):
    seen, out, err = _run_main_golden(monkeypatch, capsys, argv, GOLDEN_PASSING)
    assert seen == [dispatched]
    assert out == expected_out
    assert err == ""


def test_fetch_hub_config_timeout_is_no_verdict(monkeypatch):
    def _late(fn, *a, **kw):
        raise TimeoutError

    monkeypatch.setattr(pf, "_call_with_deadline", _late)
    assert pf._fetch_hub_config("o/r", None) is None


def test_golden_interactive_pass_summary(monkeypatch, capsys):
    """The one intended output change: an interactive terminal sees the
    pre-download summary from the approved mock before the download starts."""
    info = _info(
        {"model.safetensors": 5 * GIB, "adapter_model.safetensors": 9 * GIB},
        config=dict(MLX_CONFIG),
    )
    _no_deadline(monkeypatch)
    import huggingface_hub

    from rapid_mlx import _download_gate

    monkeypatch.setattr(_download_gate, "is_repo_cached", lambda name: False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(huggingface_hub, "model_info", lambda *a, **kw: info)
    monkeypatch.setattr(pf, "supported_model_types", lambda: SUPPORTED)
    monkeypatch.setattr(pf, "physical_ram_bytes", lambda: 36 * GIB)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    pf.run_cli_preflight(_args(), spinner_factory=lambda label: nullcontext())
    assert capsys.readouterr() == (
        "\n"
        "  ✓ Checked before download\n"
        "    Format        MLX · 4-bit · 5.0 GB\n"
        "    Architecture  qwen3 (supported)\n"
        "    Chat template found\n"
        "    Fits your Mac yes · ~7.0 GB of 36 GB\n",
        "",
    )


def test_hook_prints_alternatives_and_drops_search_hint(hook, capsys):
    info = _info({"m-Q4_K_M.gguf": 7 * GIB})
    hook.hints["value"] = (
        ["  Same model in MLX format: x", "    rapid-mlx serve x"],
        True,
    )
    with pytest.raises(SystemExit):
        hook(info)
    err = capsys.readouterr().err
    assert "    Same model in MLX format: x" in err
    assert "huggingface.co/models?search" not in err


def test_render_gguf_caps_the_quant_list():
    files = tuple(f"m-Q{i}_K_M.gguf" for i in range(2, 8))
    insp = _insp(files=files, config={})
    line = pf.render_failure(insp, _evaluate(insp))[0]
    assert line.endswith("(Q2_K_M, Q3_K_M, Q4_K_M, Q5_K_M, …).")


def test_hub_provenance_fields(monkeypatch):
    _no_deadline(monkeypatch)
    import huggingface_hub

    info = _info({"m-Q8_0.gguf": GIB})
    info.gguf = {"total": 596049920}
    info.card_data = {"license": "apache-2.0"}
    info.tags = [
        "gguf",
        "base_model:Qwen/Qwen3-0.6B",
        "base_model:quantized:Qwen/Qwen3-0.6B",
        "base_model:finetune:other/x",
        3,
    ]
    monkeypatch.setattr(huggingface_hub, "model_info", lambda *a, **kw: info)
    insp = pf.inspect_hub("Qwen/Qwen3-0.6B-GGUF")
    assert insp.params == 596049920
    assert insp.quantized_from == ("Qwen/Qwen3-0.6B",)
    assert insp.license == "apache-2.0"
    assert "gguf" in insp.tags


def test_hub_params_and_card_helpers():
    assert pf._hub_params(_info({}, params={"BF16": 1})) is None  # total missing
    st = types.SimpleNamespace(safetensors=types.SimpleNamespace(total=7), gguf=None)
    assert pf._hub_params(st) == 7
    assert pf._hub_params(types.SimpleNamespace(gguf={"total": 0})) is None
    assert pf._hub_params(types.SimpleNamespace(gguf="odd")) is None
    assert pf.hub_tags(types.SimpleNamespace(tags=None)) == ()
    assert pf.quantized_from(("base_model:quantized:plain",)) == ()

    class _Exploding:
        def get(self, key):
            raise RuntimeError

    assert pf.card_license(_Exploding()) is None
    assert pf.card_license(None) is None
    assert pf.card_license({"license": ""}) is None


def test_hook_offers_a_support_request_on_refusal(hook, capsys, monkeypatch):
    info = _info({"m-Q4_K_M.gguf": 7 * GIB})
    info.private = False
    info.gated = False
    monkeypatch.setattr(pf, "_cli_version", lambda: "0.15.4")
    with pytest.raises(SystemExit):
        hook(info)
    ((insp, verdict, version),) = hook.offered
    assert insp.public is True
    assert verdict.failure == pf.UNSUPPORTED_FORMAT
    assert version == "0.15.4"


def test_cli_version(monkeypatch):
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    assert pf._cli_version() == "1.2.3"

    def _missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)
    assert pf._cli_version() == "dev"


@pytest.mark.parametrize(
    "private,gated,public",
    [
        (False, False, True),
        (True, False, False),
        (False, "auto", False),
        (None, None, False),
    ],
)
def test_inspect_hub_public_flag(monkeypatch, private, gated, public):
    _no_deadline(monkeypatch)
    import huggingface_hub

    info = _info({"model.safetensors": 1})
    info.private = private
    info.gated = gated
    monkeypatch.setattr(huggingface_hub, "model_info", lambda *a, **kw: info)
    assert pf.inspect_hub("o/r").public is public


def test_parsers_expose_request():
    parser = cli.build_parser()
    assert parser.parse_args(["serve", "o/r", "--request"]).request
    assert parser.parse_args(["pull", "o/r"]).request is False
