"""Fatal generation errors must release request-owned Metal state."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mlx")

from rapid_mlx.scheduler import Scheduler  # noqa: E402


def test_generation_error_releases_prompt_and_checkpoint_references():
    scheduler = object.__new__(Scheduler)
    cache = object()
    request = SimpleNamespace(
        prompt_cache=cache,
        _extracted_cache=cache,
        _continuous_mtp_state=cache,
        set_finished=MagicMock(),
    )
    scheduler._close_batch_generator = MagicMock()
    scheduler._current_sampler_params = object()
    scheduler.performance = SimpleNamespace(record_failed_performance=MagicMock())
    scheduler.running = {"failed": request}
    scheduler.requests = {"failed": request}
    scheduler.finished_req_ids = set()
    scheduler._detokenizer_pool = {}
    scheduler.request_id_to_uid = {"failed": 7}
    scheduler.uid_to_request_id = {7: "failed"}
    scheduler._hybrid_checkpoints = {7: [cache]}
    scheduler.block_aware_cache = SimpleNamespace(release_cache=MagicMock())

    with patch("rapid_mlx.scheduler.mx.clear_cache"):
        aborted = scheduler._recover_from_generation_error()

    assert aborted == {"failed"}
    assert request.prompt_cache is None
    assert request._extracted_cache is None
    assert request._continuous_mtp_state is None
    assert scheduler._hybrid_checkpoints == {}
    scheduler.block_aware_cache.release_cache.assert_called_once_with("failed")


def test_generation_error_continues_after_paged_release_failure():
    scheduler = object.__new__(Scheduler)
    failed = SimpleNamespace(
        prompt_cache=object(),
        _extracted_cache=object(),
        _continuous_mtp_state=object(),
        set_finished=MagicMock(),
    )
    second = SimpleNamespace(
        prompt_cache=object(),
        _extracted_cache=object(),
        _continuous_mtp_state=object(),
        set_finished=MagicMock(),
    )
    scheduler._close_batch_generator = MagicMock()
    scheduler._current_sampler_params = object()
    scheduler.performance = SimpleNamespace(record_failed_performance=MagicMock())
    scheduler.running = {"failed": failed, "second": second}
    scheduler.finished_req_ids = set()
    scheduler._detokenizer_pool = {}
    scheduler.request_id_to_uid = {"failed": 7, "second": 8}
    scheduler.uid_to_request_id = {7: "failed", 8: "second"}
    scheduler._hybrid_checkpoints = {7: [object()], 8: [object()]}
    release = MagicMock(side_effect=[RuntimeError("broken"), None])
    scheduler.block_aware_cache = SimpleNamespace(release_cache=release)

    with patch("rapid_mlx.scheduler.mx.clear_cache"):
        aborted = scheduler._recover_from_generation_error()

    assert aborted == {"failed", "second"}
    assert scheduler.running == {}
    assert failed.prompt_cache is None
    assert second.prompt_cache is None
    assert release.call_count == 2
