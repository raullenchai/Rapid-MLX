import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("httpx")


def _load(name: str) -> Any:  # noqa: ANN401
    module_path = Path(__file__).parents[1] / "tools" / "flow_tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load("file_organizer")
DIGEST = _load("digest")


def test_sanitize_folder_rejects_bad_names_and_neutralizes_traversal():
    # Traversal characters are stripped, so ../escape becomes a safe in-root name.
    assert MODULE._sanitize_folder("../escape") == "escape"
    with pytest.raises(ValueError, match="unsafe folder"):
        MODULE._sanitize_folder("..")
    with pytest.raises(ValueError, match="unsafe folder"):
        MODULE._sanitize_folder("   ")
    with pytest.raises(ValueError, match="unsafe folder"):
        MODULE._sanitize_folder("///")
    assert MODULE._sanitize_folder("My: Invoices/") == "My-Invoices"


def test_validate_plan_rejects_unknown_and_duplicate_files(tmp_path):
    plan = {
        "moves": [
            {"file": "a.pdf", "folder": "Invoices", "reason": "bill"},
            {"file": "a.pdf", "folder": "Invoices", "reason": "bill"},
        ],
        "summary": "x",
    }
    with pytest.raises(ValueError, match="unknown file"):
        MODULE._validate_plan(plan, {"b.pdf"}, tmp_path)
    with pytest.raises(ValueError, match="duplicate move"):
        MODULE._validate_plan(plan, {"a.pdf"}, tmp_path)


def test_validate_plan_rejects_destination_outside_root(tmp_path):
    plan = {
        "moves": [{"file": "a.pdf", "folder": "..", "reason": ""}],
        "summary": "",
    }
    with pytest.raises(ValueError, match="unsafe folder"):
        MODULE._validate_plan(plan, {"a.pdf"}, tmp_path)


def test_validate_plan_accepts_clean_moves(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"x")
    plan = {
        "moves": [{"file": "a.pdf", "folder": "Invoices 2026", "reason": "bill"}],
        "summary": "one move",
    }
    clean = MODULE._validate_plan(plan, {"a.pdf"}, tmp_path)
    assert clean["moves"][0]["folder"] == "Invoices-2026"


def test_execute_moves_and_undo_restores(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"x")
    plan = {"moves": [{"file": "a.pdf", "folder": "Invoices", "reason": ""}]}
    clean = MODULE._validate_plan(plan, {"a.pdf"}, tmp_path)
    result = MODULE._execute(tmp_path, clean["moves"])
    assert (tmp_path / "Invoices" / "a.pdf").exists()
    restored = MODULE._undo(tmp_path, result["undo"])
    assert restored == ["a.pdf"]
    assert (tmp_path / "a.pdf").exists()


def test_scan_skips_hidden_and_unknown_extensions(tmp_path):
    (tmp_path / "keep.txt").write_text("x")
    (tmp_path / ".hidden.txt").write_text("x")
    (tmp_path / "skip.exe").write_text("x")
    names = [item["name"] for item in MODULE._scan(tmp_path)]
    assert names == ["keep.txt"]


def test_digest_validate_item_normalizes_urgency_and_actions():
    item = DIGEST._validate_item(
        {
            "title": "Invoice overdue",
            "summary": "The invoice is overdue.",
            "action_items": ["Pay today", 42],
            "urgency": "CRITICAL",
        },
        "fallback.md",
    )
    assert item["urgency"] == "low"
    assert item["action_items"] == ["Pay today", "42"]


def test_digest_render_sorts_by_urgency():
    items = [
        {"title": "B", "summary": "low prio", "action_items": [], "urgency": "low"},
        {
            "title": "A",
            "summary": "due today",
            "action_items": ["act"],
            "urgency": "high",
        },
    ]
    rendered = DIGEST._render_digest("docs", items)
    assert rendered.index("[HIGH] A") < rendered.index("[LOW] B")
    assert "- act" in rendered


def test_digest_collect_limits_extensions_and_size(tmp_path):
    (tmp_path / "a.md").write_text("x" * 10)
    (tmp_path / "b.exe").write_text("x")
    (tmp_path / "c.md").write_text("x" * (DIGEST.MAX_BYTES + 1))
    names = [path.name for path in DIGEST._collect(tmp_path)]
    assert names == ["a.md"]


def test_digest_rejects_non_loopback_planner_url():
    with pytest.raises(ValueError, match="loopback"):
        DIGEST._validate_loopback_url("http://10.0.0.1:8888/v1")
    assert (
        DIGEST._validate_loopback_url("http://127.0.0.1:18730/v1/chat/completions")
        == "http://127.0.0.1:18730/v1/chat/completions"
    )


def test_execute_moves_with_collision_stamp_and_undo(tmp_path):
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "b.txt").write_text("B")
    (tmp_path / "Docs").mkdir()
    (tmp_path / "Docs" / "a.txt").write_text("existing")

    result = MODULE._execute(
        tmp_path,
        [
            {"file": "a.txt", "folder": "Docs"},
            {"file": "b.txt", "folder": "Docs"},
            {"file": "missing.txt", "folder": "Docs"},  # skipped silently
        ],
    )
    assert len(result["executed"]) == 2
    moved = tmp_path / "Docs" / "b.txt"
    assert moved.read_text() == "B"
    stamped = [p for p in (tmp_path / "Docs").glob("a-*.txt")]
    assert stamped, "collision must be stamped, not overwritten"
    assert result["undo"][0]["to"] == "a.txt"


def test_undo_restores_original_layout(tmp_path):
    (tmp_path / "a.txt").write_text("A")
    MODULE._execute(tmp_path, [{"file": "a.txt", "folder": "Docs"}])
    assert not (tmp_path / "a.txt").exists()
    undo = [{"from": "Docs/a.txt", "to": "a.txt"}]
    restored = MODULE._undo(tmp_path, undo)
    assert (tmp_path / "a.txt").read_text() == "A"
    assert restored == ["a.txt"]


def test_validate_plan_rejects_unknown_duplicate_and_escape(tmp_path):
    known = {"a.txt"}
    with pytest.raises(ValueError, match="unknown file"):
        MODULE._validate_plan(
            {"moves": [{"file": "ghost.txt", "folder": "X"}]}, known, tmp_path
        )
    with pytest.raises(ValueError, match="duplicate"):
        MODULE._validate_plan(
            {
                "moves": [
                    {"file": "a.txt", "folder": "X"},
                    {"file": "a.txt", "folder": "Y"},
                ]
            },
            known,
            tmp_path,
        )


def test_digest_validate_item_normalizes_and_truncates():
    item = DIGEST._validate_item(
        {
            "title": "T" * 200,
            "summary": "S" * 2000,
            "action_items": ["x" * 300, 42, {"bad": "type"}],
            "urgency": "CRITICAL",  # not in order -> low
        },
        "fallback.md",
    )
    assert item["urgency"] == "low"
    assert len(item["title"]) == 120
    assert len(item["summary"]) == 1200
    assert len(item["action_items"]) == 3
    assert all(isinstance(a, str) for a in item["action_items"])


def test_render_digest_orders_by_urgency():
    items = [
        {"title": "weekly", "summary": "s", "action_items": [], "urgency": "medium"},
        {
            "title": "incident",
            "summary": "s",
            "action_items": ["page on-call"],
            "urgency": "high",
        },
    ]
    text = DIGEST._render_digest("goal", items)
    assert text.index("incident") < text.index("weekly")
    assert "[HIGH]" in text and "[MEDIUM]" in text
    assert "- page on-call" in text
