"""Source remapping and explicit summary truncation regressions."""

import json
from types import SimpleNamespace

import pytest

from rapid_mlx.chat_runtime import delegation


def source(number, label, size=0):
    return {
        "type": "source",
        "source_id": str(number),
        "call_id": "child-search",
        "title": label,
        "url": "https://example.com/" + label + "a" * size,
    }


@pytest.mark.asyncio
async def test_grouped_citations_remap_to_parent_sources():
    emitted = []

    async def emit(event):
        emitted.append(event)

    previous = source(1, "previous")
    alpha, beta = source(1, "alpha"), source(2, "beta")
    parent = SimpleNamespace(sources={previous["url"]: previous}, emit=emit)
    manager = delegation.DelegationManager(SimpleNamespace(store=None))
    summary, promoted = await manager._promote_sources(
        parent,
        "helper",
        "Facts [1, 2]; compact [1,2]; spaced [ 1 , 2 ]; single [2]; unknown [9].",
        {s["url"]: s for s in [alpha, beta]},
    )
    assert (
        summary
        == "Facts [2, 3]; compact [2,3]; spaced [ 2 , 3 ]; single [3]; unknown [9]."
    )
    assert [s["source_id"] for s in promoted] == ["2", "3"]
    assert emitted == promoted


def test_grouped_citations_get_source_appendix_priority():
    sources = [source(i, str(i), 220) for i in range(1, 8)]
    data = json.loads(
        delegation.DelegationManager._result(
            "child", "completed", "Supported facts [7, 6].", sources, 1800
        )
    )
    assert [s["source_id"] for s in data["sources"]][:2] == ["6", "7"]


def test_summary_json_budget_marks_truncation_and_keeps_complete_words():
    text = 'Price/Availability café "quoted" \\ details\n' * 1000
    result = delegation.DelegationManager._result(
        "child", "completed", text, [source(1, "reference", 120)], 1200
    )
    assert len(result.encode()) <= 1200
    summary = json.loads(result)["summary"]
    assert summary.endswith(
        "[Summary truncated; see the helper transcript for omitted details.]"
    )
    before = summary.split("\n[Summary truncated;")[0]
    assert before.endswith(("Price/Availability", "café", '"quoted"', "\\", "details"))


def test_initial_hidden_summary_budget_marks_truncation():
    result = delegation._clip("Availability café " * 100, 200)
    assert len(result.encode()) <= 200
    assert result.endswith(
        "[Summary truncated; see the helper transcript for omitted details.]"
    )


def test_short_summary_is_preserved_without_marker():
    text = "Complete summary [1, 2]."
    assert delegation._clip(text, 200) == text
    data = json.loads(
        delegation.DelegationManager._result("child", "completed", text, [], 1000)
    )
    assert data["summary"] == text
