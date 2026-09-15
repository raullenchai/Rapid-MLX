import os
import sys

from rapid_mlx.runtime.process_memory import get_phys_footprint


def test_phys_footprint_is_nonnegative_for_current_process():
    assert get_phys_footprint() >= 0


def test_phys_footprint_accepts_explicit_pid():
    assert get_phys_footprint(os.getpid()) >= 0


def test_phys_footprint_is_available_on_darwin():
    if sys.platform == "darwin":
        assert get_phys_footprint() > 0
