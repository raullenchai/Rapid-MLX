"""Compatibility coverage for mid-conversation system messages (#1543)."""

import pytest

from vllm_mlx.utils.chat_template import apply_chat_template

MESSAGES = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello."},
    {"role": "system", "content": "From now on answer only BLUE."},
    {"role": "user", "content": "What colour is the sky?"},
]


class _RecordingTemplate:
    def __init__(
        self, *, reject_mid_system: bool, retry_error: Exception | None = None
    ):
        self.reject_mid_system = reject_mid_system
        self.retry_error = retry_error
        self.calls: list[list[dict]] = []

    def apply_chat_template(self, messages, **_kwargs):
        self.calls.append(messages)
        has_mid_system = any(message["role"] == "system" for message in messages[1:])
        if self.reject_mid_system and has_mid_system:
            raise RuntimeError("System message must be at the beginning.")
        if len(self.calls) > 1 and self.retry_error is not None:
            raise self.retry_error
        return messages


def test_retries_rejecting_template_with_one_leading_system_message():
    template = _RecordingTemplate(reject_mid_system=True)

    rendered = apply_chat_template(template, MESSAGES)

    assert len(template.calls) == 2
    assert rendered == [
        {
            "role": "system",
            "content": "You are terse.\n\nFrom now on answer only BLUE.",
        },
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "What colour is the sky?"},
    ]


def test_does_not_coerce_template_that_accepts_original_order():
    template = _RecordingTemplate(reject_mid_system=False)

    rendered = apply_chat_template(template, MESSAGES)

    assert len(template.calls) == 1
    assert rendered == MESSAGES


def test_retry_preserves_structured_system_content_parts():
    template = _RecordingTemplate(reject_mid_system=True)
    structured = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Inspect this."},
                {"type": "image", "image": "first.png"},
            ],
        },
        {"role": "user", "content": "Hi"},
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Compare it."},
                {"type": "image", "image": "second.png"},
            ],
        },
        {"role": "user", "content": "Continue"},
    ]

    rendered = apply_chat_template(template, structured)

    assert rendered[0] == {
        "role": "system",
        "content": [
            {"type": "text", "text": "Inspect this."},
            {"type": "image", "image": "first.png"},
            {"type": "text", "text": "\n\n"},
            {"type": "text", "text": "Compare it."},
            {"type": "image", "image": "second.png"},
        ],
    }


def test_unrelated_template_error_is_not_retried():
    class _BrokenTemplate:
        calls = 0

        def apply_chat_template(self, _messages, **_kwargs):
            self.calls += 1
            raise RuntimeError("unknown template failure")

    template = _BrokenTemplate()
    with pytest.raises(RuntimeError, match="unknown template failure"):
        apply_chat_template(template, MESSAGES)
    assert template.calls == 1


def test_retry_failure_surfaces_original_template_error():
    template = _RecordingTemplate(
        reject_mid_system=True,
        retry_error=RuntimeError("retry-only failure"),
    )

    with pytest.raises(RuntimeError, match="System message must be at the beginning"):
        apply_chat_template(template, MESSAGES)


def test_no_mid_system_does_not_retry_matching_error():
    class _RejectsForAnotherReason:
        calls = 0

        def apply_chat_template(self, _messages, **_kwargs):
            self.calls += 1
            raise RuntimeError("System message must be at the beginning.")

    template = _RejectsForAnotherReason()
    leading_only = MESSAGES[:3]
    with pytest.raises(RuntimeError, match="System message must be at the beginning"):
        apply_chat_template(template, leading_only)
    assert template.calls == 1


def test_system_metadata_is_not_silently_discarded_by_retry():
    template = _RecordingTemplate(reject_mid_system=True)
    named_system = [dict(message) for message in MESSAGES]
    named_system[3]["name"] = "policy-update"

    with pytest.raises(RuntimeError, match="System message must be at the beginning"):
        apply_chat_template(template, named_system)

    assert len(template.calls) == 1
    assert template.calls[0][3]["name"] == "policy-update"
