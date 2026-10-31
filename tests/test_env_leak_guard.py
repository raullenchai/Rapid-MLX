import inspect
import os

from tests import conftest

pytest_plugins = ["pytester"]


def test_env_leak_guard_restores_and_reports_names(pytester):
    fixture = inspect.getsource(conftest._aaa_env_leak_guard.__wrapped__)
    pytester.makeconftest(
        "import os\nimport pytest\n\n"
        f"_IMPORT_SIDE_EFFECT_ENV_VARS = {conftest._IMPORT_SIDE_EFFECT_ENV_VARS!r}\n\n"
        + fixture
    )
    pytester.makepyfile(
        """
        import os

        def test_leak():
            os.environ["RAPID_TEST_GUARD_LEAK"] = "secret-value"

        def test_after_leak():
            assert "RAPID_TEST_GUARD_LEAK" not in os.environ

        def test_monkeypatch(monkeypatch):
            monkeypatch.setenv("RAPID_TEST_GUARD_PATCH", "value")
        """
    )
    result = pytester.runpytest("-q", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=3, errors=1)
    output = result.stdout.str()
    assert "RAPID_TEST_GUARD_LEAK" in output
    assert "secret-value" not in output
    assert "RAPID_TEST_GUARD_PATCH" not in output


def test_env_leak_guard_ignores_import_side_effect(pytester, monkeypatch):
    fixture = inspect.getsource(conftest._aaa_env_leak_guard.__wrapped__)
    pytester.makeconftest(
        "import os\nimport pytest\n\n"
        f"_IMPORT_SIDE_EFFECT_ENV_VARS = {conftest._IMPORT_SIDE_EFFECT_ENV_VARS!r}\n\n"
        + fixture
    )
    pytester.makepyfile(
        """
        import os

        def test_import_side_effect():
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        def test_import_side_effect_was_restored():
            assert "TOKENIZERS_PARALLELISM" not in os.environ
        """
    )
    # Restore the outer process environment after the in-process test runs.
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    result = pytester.runpytest("-q", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=2)


def test_env_leak_guard_restores_existing_import_side_effect(pytester, monkeypatch):
    fixture = inspect.getsource(conftest._aaa_env_leak_guard.__wrapped__)
    pytester.makeconftest(
        "import os\nimport pytest\n\n"
        f"_IMPORT_SIDE_EFFECT_ENV_VARS = {conftest._IMPORT_SIDE_EFFECT_ENV_VARS!r}\n\n"
        + fixture
    )
    pytester.makepyfile(
        """
        import os

        def test_change_import_side_effect():
            os.environ["TOKENIZERS_PARALLELISM"] = "changed"
        """
    )
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "preset")
    result = pytester.runpytest("-q", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)
    assert os.environ["TOKENIZERS_PARALLELISM"] == "preset"
