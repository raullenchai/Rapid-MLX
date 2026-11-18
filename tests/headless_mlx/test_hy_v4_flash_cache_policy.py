"""Hy4's prefix-cache opt-out is applied before creating any reuse store."""

from unittest.mock import MagicMock

import pytest


def test_model_cache_capability_disables_prefix_backends():
    from rapid_mlx.scheduler import Scheduler, SchedulerConfig

    model = MagicMock()
    model.supports_prefix_cache = False
    config = SchedulerConfig(enable_prefix_cache=True, use_memory_aware_cache=False)
    scheduler = Scheduler(model=model, tokenizer=MagicMock(), config=config)
    assert scheduler.config.enable_prefix_cache is False
    assert config.enable_prefix_cache is True
    assert scheduler.prefix_cache is None
    assert scheduler.memory_aware_cache is None


def test_explicit_paged_cache_cannot_silently_override_model_opt_out():
    from rapid_mlx.errors import PagedCacheUnsupportedLayoutError
    from rapid_mlx.scheduler import Scheduler, SchedulerConfig

    model = MagicMock()
    model.supports_prefix_cache = False
    with pytest.raises(PagedCacheUnsupportedLayoutError, match="prefix cache"):
        Scheduler(
            model=model,
            tokenizer=MagicMock(),
            config=SchedulerConfig(use_paged_cache=True),
        )
