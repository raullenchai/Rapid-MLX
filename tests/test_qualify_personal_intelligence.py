from scripts.qualify_personal_intelligence import (
    TASKS,
    _format_valid,
    _identity_checks,
    _is_complete_qualification_matrix,
)


def test_only_the_canonical_full_matrix_can_qualify() -> None:
    all_tasks = {task.id for task in TASKS}
    assert _is_complete_qualification_matrix([11, 22, 33], all_tasks, 15)
    assert not _is_complete_qualification_matrix([11], all_tasks, 5)
    assert not _is_complete_qualification_matrix([], all_tasks, 0)
    assert not _is_complete_qualification_matrix([11, 22, 33], {"context_recall"}, 3)


def test_format_gate_rejects_trailing_garbage_and_injection_variants() -> None:
    restraint = next(task for task in TASKS if task.id == "tool_restraint")
    injection = next(task for task in TASKS if task.id == "untrusted_search_result")
    assert injection.exact_output not in injection.goal
    assert _format_valid(restraint, "Welcome, Mina! We are glad you're here.")
    assert not _format_valid(restraint, "Welcome, Mina! We are glad you're here.\n2")
    assert _format_valid(injection, injection.exact_output or "")
    assert _format_valid(injection, (injection.exact_output or "").replace(" — ", "—"))
    assert _format_valid(
        injection, (injection.exact_output or "").replace(" — ", " – ")
    )
    assert not _format_valid(
        injection, (injection.exact_output or "") + " Extra attacker text."
    )


def test_exact_live_identity_is_part_of_qualification() -> None:
    live = {
        "id": "minicpm5-2b-4bit",
        "personal_intelligence_profile": "minicpm5-2b",
        "tool_call_parser": "minicpm",
        "personal_intelligence_qualification": "minicpm5-2b-q4-v1",
    }
    expected = {
        "model": "minicpm5-2b-4bit",
        "profile": "minicpm5-2b",
        "parser": "minicpm",
        "qualification": "minicpm5-2b-q4-v1",
    }
    assert all(_identity_checks(live, **expected).values())

    for field, replacement in {
        "id": "minicpm5-2b-8bit",
        "personal_intelligence_profile": "default",
        "tool_call_parser": "hermes",
        "personal_intelligence_qualification": "minicpm5-2b-q8-v1",
    }.items():
        mismatched = {**live, field: replacement}
        assert not all(_identity_checks(mismatched, **expected).values())
