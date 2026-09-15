from scripts.qualify_personal_intelligence import (
    TASKS,
    _is_complete_qualification_matrix,
)


def test_only_the_canonical_full_matrix_can_qualify() -> None:
    all_tasks = {task.id for task in TASKS}
    assert _is_complete_qualification_matrix([11, 22, 33], all_tasks, 15)
    assert not _is_complete_qualification_matrix([11], all_tasks, 5)
    assert not _is_complete_qualification_matrix([], all_tasks, 0)
    assert not _is_complete_qualification_matrix(
        [11, 22, 33], {"context_recall"}, 3
    )
