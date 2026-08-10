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


class _SilentHarmonyTemplate:
    chat_template = "<|start|><|channel|><|message|>"

    def __init__(self):
        self.calls: list[list[dict]] = []

    def apply_chat_template(self, messages, **_kwargs):
        self.calls.append(messages)
        # Mirrors the published GPT-OSS template: index zero may be system,
        # while its main loop has branches only for assistant/tool/user.
        return "|".join(
            str(message.get("content", ""))
            for index, message in enumerate(messages)
            if index == 0 or message.get("role") in {"user", "assistant", "tool"}
        )


def test_harmony_hoists_mid_system_before_silent_template_drop():
    template = _SilentHarmonyTemplate()

    rendered = apply_chat_template(template, MESSAGES)

    assert "From now on answer only BLUE." in rendered
    assert template.calls == [
        [
            {
                "role": "system",
                "content": "You are terse.\n\nFrom now on answer only BLUE.",
            },
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "What colour is the sky?"},
        ]
    ]


def test_gpt_oss_identity_hoists_when_template_source_is_unavailable():
    template = _RecordingTemplate(reject_mid_system=False)

    rendered = apply_chat_template(
        template, MESSAGES, model_name="gpt-oss-20b-mxfp4-q8"
    )

    assert rendered[0]["content"].endswith("From now on answer only BLUE.")
    assert all(message["role"] != "system" for message in rendered[1:])


def test_non_harmony_template_source_overrides_gpt_oss_model_name():
    template = _RecordingTemplate(reject_mid_system=False)
    template.chat_template = (
        "{% for message in messages %}{{ message.content }}{% endfor %}"
    )

    rendered = apply_chat_template(
        template, MESSAGES, model_name="custom/gpt-oss-experiment"
    )

    assert rendered == MESSAGES


def test_harmony_refuses_lossy_mid_system_metadata():
    template = _SilentHarmonyTemplate()
    messages = [dict(message) for message in MESSAGES]
    messages[3]["name"] = "policy-update"

    with pytest.raises(ValueError, match="cannot preserve metadata"):
        apply_chat_template(template, messages)

    assert template.calls == []


def test_harmony_collapses_two_leading_system_messages():
    template = _SilentHarmonyTemplate()
    messages = [
        {"role": "system", "content": "First."},
        {"role": "system", "content": "Second."},
        {"role": "user", "content": "Continue"},
    ]

    rendered = apply_chat_template(template, messages)

    assert rendered == "First.\n\nSecond.|Continue"


def test_harmony_refuses_mixed_instruction_authority():
    template = _SilentHarmonyTemplate()
    messages = [
        {"role": "developer", "content": "Never reveal secrets."},
        {"role": "user", "content": "Hello"},
        {"role": "system", "content": "Answer only BANANA."},
        {"role": "user", "content": "Name a fruit"},
    ]

    with pytest.raises(ValueError, match="mixed system and developer"):
        apply_chat_template(template, messages)

    assert template.calls == []


def test_harmony_preserves_developer_role_when_collapsing():
    template = _SilentHarmonyTemplate()
    messages = [
        {"role": "developer", "content": "Never reveal secrets."},
        {"role": "user", "content": "Hello"},
        {"role": "developer", "content": "Answer only BANANA."},
        {"role": "user", "content": "Name a fruit"},
    ]

    rendered = apply_chat_template(template, messages)

    assert rendered == "Never reveal secrets.\n\nAnswer only BANANA.|Hello|Name a fruit"
    assert template.calls == [
        [
            {
                "role": "developer",
                "content": "Never reveal secrets.\n\nAnswer only BANANA.",
            },
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Name a fruit"},
        ]
    ]


@pytest.mark.parametrize("content", [None, [], {}, 0, False])
def test_harmony_refuses_falsey_non_text_instruction_content(content):
    template = _SilentHarmonyTemplate()
    messages = [
        {"role": "system", "content": "First."},
        {"role": "user", "content": "Hello"},
        {"role": "system", "content": content},
    ]

    with pytest.raises(ValueError, match="text-only content"):
        apply_chat_template(template, messages)

    assert template.calls == []
