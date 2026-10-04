# SPDX-License-Identifier: Apache-2.0
"""BYOM alternatives: matching MLX builds first, then a catalog model.

All Hub calls are mocked. The matching rules are the product: a candidate
that is not provably the same model, would not run or fit here, or carries
an adult-content tag the user's own source does not carry must never be
printed.
"""

from __future__ import annotations

import types

import pytest

from rapid_mlx.byom import alternatives as alt
from rapid_mlx.byom import preflight as pf

GIB = 1 << 30
RAM = 36 * GIB
SUPPORTED = frozenset({"qwen3", "llama"})
PARAMS = 596_049_920
BASE = "Qwen/Qwen3-0.6B"


@pytest.fixture(autouse=True)
def _no_deadline(monkeypatch):
    from rapid_mlx import _download_gate

    monkeypatch.setattr(
        _download_gate, "call_with_deadline", lambda fn, timeout, *a, **kw: fn(*a, **kw)
    )


def _gguf_insp(**kw) -> pf.Inspection:
    fields = {
        "ref": "Qwen/Qwen3-0.6B-GGUF",
        "is_local": False,
        "files": ("Qwen3-0.6B-Q8_0.gguf",),
        "params": PARAMS,
        "quantized_from": (BASE,),
        "tags": ("gguf",),
    }
    fields.update(kw)
    return pf.Inspection(**fields)


def _verdict(insp: pf.Inspection) -> pf.Verdict:
    return pf.evaluate(insp, refuse_oversize=True, supported=SUPPORTED, ram_bytes=None)


def _model(
    repo_id: str,
    *,
    model_type: str = "qwen3",
    quant: dict | None = None,
    total: int | None = PARAMS,
    downloads: int = 10,
    license_: str | None = "apache-2.0",
    tags: tuple[str, ...] = ("mlx", "safetensors"),
) -> types.SimpleNamespace:
    config: dict = {"model_type": model_type, "architectures": ["Qwen3ForCausalLM"]}
    if quant is not False:
        config["quantization_config"] = quant if quant is not None else {"bits": 4}
    return types.SimpleNamespace(
        id=repo_id,
        config=config,
        safetensors=types.SimpleNamespace(total=total),
        card_data={"license": license_},
        downloads=downloads,
        tags=list(tags),
    )


@pytest.fixture
def hub(monkeypatch):
    """Mock the base-model lookup and the MLX search."""
    import huggingface_hub

    state = {"base": _model(BASE, quant=False), "builds": [], "calls": []}

    def _model_info(repo):
        state["calls"].append(("info", repo))
        if isinstance(state["base"], Exception):
            raise state["base"]
        return state["base"]

    class _Api:
        def list_models(self, **kw):
            state["calls"].append(("list", kw["filter"]))
            if isinstance(state["builds"], Exception):
                raise state["builds"]
            return iter(state["builds"])

    monkeypatch.setattr(huggingface_hub, "model_info", _model_info)
    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    return state


def _find(insp=None, *, ram=RAM, supported=SUPPORTED):
    return alt.find_mlx_builds(insp or _gguf_insp(), supported=supported, ram_bytes=ram)


def test_matching_builds_are_ranked_and_capped(hub):
    hub["builds"] = [
        _model("mlx-community/Qwen3-0.6B-8bit", quant={"bits": 8}, downloads=900),
        _model("mlx-community/Qwen3-0.6B-4bit", downloads=300),
        _model("lmstudio-community/Qwen3-0.6B-MLX-4bit", downloads=50),
        _model("other/Qwen3-0.6B-odd", quant={"bits": 7}, downloads=5000),
    ]
    found = _find()
    assert [c.repo_id for c in found] == [
        "mlx-community/Qwen3-0.6B-4bit",
        "lmstudio-community/Qwen3-0.6B-MLX-4bit",
    ]
    assert ("list", ["mlx", f"base_model:quantized:{BASE}"]) in hub["calls"]
    assert found[0].publisher == "mlx-community"
    assert found[0].approx_bytes == int(PARAMS * 4.5 / 8)


def test_odd_bit_widths_rank_last(hub):
    hub["builds"] = [
        _model("a/odd", quant={"bits": 7}, downloads=9),
        _model("a/eight", quant={"quantization": 1, "bits": 8}),
    ]
    assert [c.repo_id for c in _find()] == ["a/eight", "a/odd"]


@pytest.mark.parametrize(
    "build",
    [
        _model("Qwen/Qwen3-0.6B-GGUF"),  # the refused repo itself
        _model("a/wrong-type", model_type="llama"),
        _model("a/not-quantized", quant=False),
        _model("a/awq", quant={"quant_method": "awq", "bits": 4}),
        _model("a/bool-bits", quant={"bits": True}),
        _model("a/no-params", total=None),
        _model("a/too-small", total=PARAMS // 2),
        _model("a/too-big", total=int(PARAMS * 1.2)),
        _model("a/untagged", tags=("safetensors",)),
        _model("a/tagged", tags=("mlx", "Not-For-All-Audiences")),
        _model("a/tagged2", tags=("mlx", "nsfw")),
        types.SimpleNamespace(id=None, config={}),
        types.SimpleNamespace(id="a/no-config", config=None),
    ],
)
def test_unmatched_candidates_are_never_shown(hub, build):
    hub["builds"] = [build]
    assert _find() == []


def test_candidates_must_be_positively_supported_and_fit_comfortably(hub):
    hub["builds"] = [_model("a/x")]
    assert _find(supported=frozenset({"llama"})) == []  # loader unknown here
    assert _find(supported=None) == []  # inventory unknown: not proven
    assert _find(ram=None) == []
    # ~0.33 GB of weights needs ~1.4 GB served: 1.5 GB of RAM is not "comfortable".
    assert _find(ram=int(1.5 * GIB)) == []
    assert [c.repo_id for c in _find(ram=4 * GIB)] == ["a/x"]


@pytest.mark.parametrize(
    "insp",
    [
        _gguf_insp(quantized_from=()),  # finetune/merge or no declared base
        _gguf_insp(quantized_from=("a/b", "c/d")),
        _gguf_insp(params=None),
    ],
)
def test_unknown_provenance_has_no_candidates(hub, insp):
    hub["builds"] = [_model("a/x")]
    assert _find(insp) == []
    assert hub["calls"] == []


def test_base_lookup_failures_mean_no_candidates(hub):
    hub["builds"] = [_model("a/x")]
    hub["base"] = OSError("down")
    assert _find() == []
    hub["base"] = types.SimpleNamespace(config=None)
    assert _find() == []
    hub["base"] = types.SimpleNamespace(config={"model_type": ""})
    assert _find() == []


def test_search_failure_means_no_candidates(hub):
    hub["builds"] = RuntimeError("rate limited")
    assert _find() == []


def test_one_budget_covers_every_hub_call(hub, monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr(alt.time, "monotonic", lambda: clock["now"])
    timeouts = []

    from rapid_mlx import _download_gate

    def _deadline(fn, timeout, *a, **kw):
        timeouts.append(timeout)
        clock["now"] += 5.0  # each call eats 5 s of the 8 s budget
        return fn(*a, **kw)

    monkeypatch.setattr(_download_gate, "call_with_deadline", _deadline)
    hub["builds"] = [_model("a/x")]
    assert [c.repo_id for c in _find()] == ["a/x"]
    assert timeouts == [8.0, 3.0]
    clock["now"] = 100.0
    timeouts.clear()

    def _slow(fn, timeout, *a, **kw):
        clock["now"] += 9.0
        return fn(*a, **kw)

    monkeypatch.setattr(_download_gate, "call_with_deadline", _slow)
    assert _find() == []  # base lookup spent the whole budget; no search


# ------------------------------------------------------------- catalog side


def test_similar_catalog_model_uses_the_shipped_policy():
    assert alt.similar_catalog_model(1 * GIB, 18 * GIB) == "lfm2.5-1b-4bit"
    assert alt.similar_catalog_model(30 * GIB, 64 * GIB) in {
        "qwen3.8-27b-4bit",
        "qwen3.6-35b-4bit",
    }
    assert alt.similar_catalog_model(None, 18 * GIB) == "qwen3.5-9b-4bit"
    assert alt.similar_catalog_model(GIB, None) is None
    assert alt.similar_catalog_model(GIB, 4 * GIB) is None


def test_similar_catalog_model_survives_policy_errors(monkeypatch):
    import rapid_mlx.recommendations as rec

    def _boom(**kw):
        raise ValueError("bad policy")

    monkeypatch.setattr(rec, "load_recommendation_tiers", _boom)
    assert alt.similar_catalog_model(GIB, 18 * GIB) is None


def test_target_bytes():
    insp = _gguf_insp()
    assert alt._target_bytes(insp, _verdict(insp)) == int(PARAMS * 0.55)
    weights = pf.Inspection(ref="a/b", is_local=False, files=(), weight_bytes=5)
    assert alt._target_bytes(weights, _verdict(weights)) == 5
    bare = pf.Inspection(ref="a/b", is_local=False, files=())
    assert alt._target_bytes(bare, _verdict(bare)) is None
    oversize = pf.Verdict(
        failure=pf.INSUFFICIENT_MEMORY,
        format_label="safetensors",
        model_type=None,
        architecture_supported=True,
    )
    assert alt._target_bytes(weights, oversize) is None


# ------------------------------------------------------------------- suggest


def _suggest(insp, command="serve", ram=RAM):
    return alt.suggest(
        insp, _verdict(insp), command=command, supported=SUPPORTED, ram_bytes=ram
    )


def test_suggest_presents_builds_as_unreviewed_third_party(hub):
    hub["builds"] = [_model("mlx-community/Qwen3-0.6B-4bit", license_=None)]
    lines, found = _suggest(_gguf_insp())
    assert found is True
    assert lines == [
        "  MLX builds of the same base model (third-party, not reviewed by Rapid-MLX;",
        "  matched by base model, architecture and size):",
        "    mlx-community/Qwen3-0.6B-4bit · 4-bit · ~0.3 GB · license unknown",
        "  To try one, check its model card first, then: rapid-mlx serve "
        "mlx-community/Qwen3-0.6B-4bit",
    ]


def test_tagged_source_may_get_a_matching_tagged_build(hub):
    hub["builds"] = [
        _model("a/tagged-4bit", tags=("mlx", "not-for-all-audiences")),
        _model("a/plain-8bit", quant={"bits": 8}),
    ]
    tagged_source = _gguf_insp(tags=("gguf", "Not-For-All-Audiences"))
    lines, found = _suggest(tagged_source)
    assert found is True
    assert any("a/tagged-4bit" in line for line in lines)
    assert any("a/plain-8bit" in line for line in lines)
    assert any("third-party, not reviewed" in line for line in lines)


def test_untagged_source_never_gets_a_tagged_build(hub, monkeypatch):
    monkeypatch.setattr(alt, "similar_catalog_model", lambda target, ram: "q")
    hub["builds"] = [_model("a/tagged-4bit", tags=("mlx", "nsfw"))]
    lines, found = _suggest(_gguf_insp())
    assert found is False
    assert not any("a/tagged-4bit" in line for line in lines)


def test_is_adult_tagged():
    assert alt.is_adult_tagged(["gguf", "NSFW"])
    assert not alt.is_adult_tagged(("gguf",))


def test_suggest_falls_back_to_the_catalog(hub, monkeypatch):
    monkeypatch.setattr(
        alt, "similar_catalog_model", lambda target, ram: "qwen3.5-4b-4bit"
    )
    lines, found = _suggest(_gguf_insp(), command="pull")
    assert found is False
    assert lines == [
        "  A catalog model of similar size that fits your Mac:",
        "    rapid-mlx pull qwen3.5-4b-4bit",
    ]
    local = pf.Inspection(ref="/m", is_local=True, files=("x.gguf",))
    lines, _ = _suggest(local)
    assert lines[0] == "  A catalog model that runs well on your Mac:"
    # PyTorch-only repos have no trustworthy parameter count: catalog only.
    pytorch = pf.Inspection(ref="a/b", is_local=False, files=("pytorch_model.bin",))
    hub["calls"].clear()
    _suggest(pytorch)
    assert hub["calls"] == []


def test_suggest_without_anything_to_offer(monkeypatch):
    monkeypatch.setattr(alt, "similar_catalog_model", lambda target, ram: None)
    insp = pf.Inspection(ref="/m", is_local=True, files=("x.gguf",))
    assert _suggest(insp, ram=None) == ([], False)

    def _boom(*a, **kw):
        raise RuntimeError

    monkeypatch.setattr(alt, "_target_bytes", _boom)
    assert _suggest(insp, ram=None) == ([], False)


def test_suggest_reports_the_suggested_targets(hub, monkeypatch):
    hub["builds"] = [_model("mlx-community/Qwen3-0.6B-4bit")]
    insp = _gguf_insp()
    targets: list[str] = []
    alt.suggest(
        insp,
        _verdict(insp),
        command="serve",
        supported=SUPPORTED,
        ram_bytes=RAM,
        targets=targets,
    )
    assert targets == ["mlx-community/Qwen3-0.6B-4bit"]
    monkeypatch.setattr(alt, "similar_catalog_model", lambda target, ram: "q-4bit")
    local = pf.Inspection(ref="/m", is_local=True, files=("x.gguf",))
    targets.clear()
    alt.suggest(
        local,
        _verdict(local),
        command="pull",
        supported=SUPPORTED,
        ram_bytes=RAM,
        targets=targets,
    )
    assert targets == ["q-4bit"]
