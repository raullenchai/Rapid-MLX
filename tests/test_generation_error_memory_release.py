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

    with patch("rapid_mlx.scheduler.mx.clear_cache"):
        aborted = scheduler._recover_from_generation_error()

    assert aborted == {"failed"}
    assert request.prompt_cache is None
    assert request._extracted_cache is None
    assert scheduler._hybrid_checkpoints == {}
