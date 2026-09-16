import argparse
import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "deepseek_context_replay.py"
SPEC = importlib.util.spec_from_file_location("deepseek_context_replay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_corpus_is_deterministic_and_bounded(tmp_path):
    (tmp_path / "rapid_mlx").mkdir()
    (tmp_path / "rapid_mlx" / "large.py").write_text("a" * 80)
    (tmp_path / "rapid_mlx" / "small.py").write_text("b" * 40)
    first = MODULE.build_corpus(tmp_path, 75)
    second = MODULE.build_corpus(tmp_path, 75)
    assert first == second
    assert len(first) == 75
    assert "large.py" in first


def test_build_corpus_rejects_undersized_source(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "tiny.py").write_text("pass")
    try:
        MODULE.build_corpus(tmp_path, 10_000)
    except ValueError as exc:
        assert "source corpus" in str(exc)
    else:
        raise AssertionError("expected undersized corpus to fail")


def test_build_corpus_does_not_follow_source_symlinks(tmp_path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("TOP_SECRET")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "linked.py").symlink_to(outside)
    (tmp_path / "scripts" / "safe.py").write_text("s" * 100)
    corpus = MODULE.build_corpus(tmp_path, 60)
    assert "TOP_SECRET" not in corpus
    assert "safe.py" in corpus


def test_detect_repetition_finds_exact_repeated_suffix():
    prefix = "unique context "
    cycle = "one two three four five six seven eight "
    repeated, period = MODULE.detect_repetition(prefix + cycle * 3)
    assert repeated
    assert period == 8


def test_detect_repetition_finds_short_period_degradation():
    assert MODULE.detect_repetition("prefix " + "error " * 20) == (True, 1)


def test_detect_repetition_ignores_normal_short_answer():
    assert MODULE.detect_repetition("rapid_mlx/model_profile.py") == (False, None)


def test_build_conversation_preserves_complete_prior_turns():
    items = MODULE.build_conversation("a" * 12, turn_chars=5)
    assert [item["role"] for item in items] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert items[1]["content"][0]["text"] == MODULE.TURN_ACKNOWLEDGEMENT
    assert MODULE.TURN_ACKNOWLEDGEMENT in items[0]["content"][0]["text"]
    assert MODULE.TURN_ACKNOWLEDGEMENT in items[-1]["content"][0]["text"]
    assert "a" * 5 in items[0]["content"][0]["text"]
    assert "a" * 2 in items[-1]["content"][0]["text"]


def test_larger_conversation_extends_completed_smaller_request():
    smaller = MODULE.build_conversation("a" * 10, turn_chars=5)
    completed_smaller = smaller + [
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": MODULE.TURN_ACKNOWLEDGEMENT}],
        }
    ]
    larger = MODULE.build_conversation("a" * 15, turn_chars=5)
    assert larger[: len(completed_smaller)] == completed_smaller


def test_default_target_sizes_are_complete_turn_boundaries():
    for size in (320_000, 480_000, 640_000):
        assert size % MODULE.TURN_CHARS == 0
    MODULE.validate_target_sizes([320_000, 480_000])
    try:
        MODULE.validate_target_sizes([400_000])
    except ValueError as exc:
        assert "complete turn boundary" in str(exc)
    else:
        raise AssertionError("expected partial-turn target to fail")


def test_build_conversation_rejects_non_positive_turn_size():
    try:
        MODULE.build_conversation("source", turn_chars=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected invalid turn size to fail")


def test_positive_int_rejects_non_positive_targets():
    assert MODULE.positive_int("1") == 1
    for value in ("0", "-1"):
        try:
            MODULE.positive_int(value)
        except argparse.ArgumentTypeError as exc:
            assert "positive" in str(exc)
        else:
            raise AssertionError("expected invalid target size to fail")


def test_run_replay_records_transport_failure():
    class BrokenClient:
        def post(self, *args, **kwargs):
            raise RuntimeError("endpoint unavailable")

    result = MODULE.run_replay(
        BrokenClient(),
        endpoint="local",
        url="http://test/v1",
        model="model",
        corpus="source",
        target_chars=6,
    )
    assert result.status == "error"
    assert result.failure == "RuntimeError: endpoint unavailable"
    assert result.answer == ""


def test_run_replay_records_malformed_usage():
    response = type(
        "Response",
        (),
        {
            "raise_for_status": lambda self: None,
            "json": lambda self: {
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": "not-a-number"},
            },
        },
    )()
    client = type("Client", (), {"post": lambda self, *args, **kwargs: response})()
    result = MODULE.run_replay(
        client,
        endpoint="local",
        url="http://test/v1",
        model="model",
        corpus="source",
        target_chars=6,
    )
    assert result.status == "error"
    assert result.input_tokens == 0
    assert result.failure.startswith("ValueError:")


def test_replay_endpoint_validation_is_credential_safe_and_unique():
    assert MODULE.parse_endpoint("cloud=https://example.test/v1,CLOUD_KEY") == (
        "cloud",
        "https://example.test/v1",
        "CLOUD_KEY",
    )
    try:
        MODULE.parse_endpoint("cloud=https://example.test/v1,not-a-valid-env-name")
    except argparse.ArgumentTypeError as exc:
        assert "environment-variable name" in str(exc)
        assert "not-a-valid" not in str(exc)
    else:
        raise AssertionError("expected literal credential suffix to fail")
    try:
        MODULE.parse_endpoint("cloud=http://example.test/v1,CLOUD_KEY")
    except argparse.ArgumentTypeError as exc:
        assert "HTTPS" in str(exc)
    else:
        raise AssertionError("expected plaintext credential endpoint to fail")
    for unsafe in (
        "https://user:secret@example.test/v1",
        "https://example.test/v1?api_key=secret",
        "https://example.test/v1#secret",
    ):
        try:
            MODULE.parse_endpoint(f"cloud={unsafe}")
        except argparse.ArgumentTypeError as exc:
            assert "must not contain" in str(exc)
        else:
            raise AssertionError("expected credential-bearing URL to fail")
    try:
        MODULE.reject_duplicate_endpoint_names(
            [("cloud", "http://one", None), ("cloud", "http://two", None)]
        )
    except ValueError as exc:
        assert "duplicate endpoint" in str(exc)
    else:
        raise AssertionError("expected duplicate endpoint names to fail")
