"""Measure the complete parser path, including finalization."""

from .benchmark import measure


class ScalingRegressionError(Exception):
    pass


def test_full_path_cpu_scaling():
    # Stop at the first violation to keep the known quadratic implementation
    # affordable in CI; a passing implementation must reach the 64K case.
    previous = measure(1024)
    for size in (4096, 16384, 65536):
        current = measure(size)
        if current["ns_per_chunk"] > previous["ns_per_chunk"] * 2:
            raise ScalingRegressionError((previous, current))
        previous = current
