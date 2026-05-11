# SPDX-License-Identifier: Apache-2.0
"""Regression guard for the doctor-harness ↔ test_hermes.py contract.

The doctor harness loads agent-specific integration tests via
``importlib.util.spec_from_file_location`` + ``spec.loader.exec_module``
(see ``vllm_mlx/agents/testing.py:_run_specific_tests``). It then reads
``mod.results`` to extract per-test PASS/FAIL entries.

For five weeks (PR #99 → PR #354) ``tests/integrations/test_hermes.py``
gated every ``run_test(...)`` invocation behind
``if __name__ == "__main__":``. Under ``exec_module``, the module name
becomes ``specific_test_test_hermes`` (not ``__main__``), so the gate
never opened — no tests ran, ``results`` stayed empty, and the harness
silently reported "No test results found (missing 'results' dict or
all tests skipped)" on every full-tier run.

This test pins the contract: when loaded the way the harness loads it,
``mod.results`` must be populated. Mocks ``httpx`` so each API call
fails fast — every test ends up FAIL'd, but the dict is non-empty,
which is the actual invariant the harness depends on.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_HERMES = REPO_ROOT / "tests" / "integrations" / "test_hermes.py"


def test_test_hermes_populates_results_under_exec_module(monkeypatch):
    """Load test_hermes.py the way the doctor harness does and verify the
    module-level ``results`` dict gets populated. Regression guard for
    the PR #99 + PR #125 mismatch — see module docstring.
    """
    assert TEST_HERMES.exists(), f"missing fixture: {TEST_HERMES}"

    # Mock httpx so each API call fails immediately rather than waiting
    # on a real server. The point of this test is the harness invocation
    # contract — not the integration tests' substance.
    class _FailHTTPX:
        def get(self, *a, **kw):
            raise RuntimeError("mocked: no server")

        def post(self, *a, **kw):
            raise RuntimeError("mocked: no server")

    import httpx

    fake = _FailHTTPX()
    monkeypatch.setattr(httpx, "get", fake.get)
    monkeypatch.setattr(httpx, "post", fake.post)

    # Force HERMES_BIN to a path that cannot exist so the E2E branch
    # (which would write ~/.hermes/config.yaml and invoke real hermes
    # subprocesses) never fires. The contract we're pinning is only
    # the 10 API-level tests at the top of the file — running the E2E
    # block would be a side effect, not part of the contract.
    monkeypatch.setenv("HERMES_BIN", "/nonexistent/hermes-binary-for-contract-test")
    monkeypatch.setenv("RAPID_MLX_BASE_URL", "http://localhost:0/v1")

    # Mirror vllm_mlx/agents/testing.py:_run_specific_tests exactly.
    spec = importlib.util.spec_from_file_location(
        "specific_test_test_hermes", str(TEST_HERMES)
    )
    mod = importlib.util.module_from_spec(spec)
    orig_exit = sys.exit
    sys.exit = lambda *a: None
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    finally:
        sys.exit = orig_exit

    results = getattr(mod, "results", None)
    assert isinstance(results, dict), (
        f"test_hermes.py must define a module-level `results` dict; "
        f"got {type(results).__name__}"
    )
    # 10 API-level tests run unconditionally (hermes_chat... only run if
    # the hermes binary is installed). At minimum we expect those 10.
    assert len(results) >= 10, (
        f"Expected ≥10 entries in `results` (the 10 API-level tests). "
        f"Got {len(results)}: {list(results.keys())}. "
        f"If empty, test_hermes.py's run_test() calls are gated on "
        f'`if __name__ == "__main__":` and never execute under the '
        f"harness's exec_module path."
    )
    # Every entry should carry a status string (PASS / FAIL / ERROR).
    for name, status in results.items():
        assert isinstance(status, str), f"non-str status for {name!r}: {status!r}"
