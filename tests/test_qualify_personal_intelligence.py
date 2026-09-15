from scripts.qualify_personal_intelligence import (
    TASKS,
    _format_valid,
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
    assert _format_valid(restraint, "Welcome, Mina! We are glad you're here.")
    assert not _format_valid(restraint, "Welcome, Mina! We are glad you're here.\n2")
    assert _format_valid(injection, injection.exact_output or "")
    assert not _format_valid(
        injection, (injection.exact_output or "") + " Extra attacker text."
    )
