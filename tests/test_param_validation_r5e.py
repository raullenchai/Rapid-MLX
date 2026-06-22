# SPDX-License-Identifier: Apache-2.0
"""
r5-E validation-gap tightenings (B-7 / B-8 / B-9).

Three Pydantic-schema fixes confirmed open against v0.8.6 by the
DGF-v080 sweep (rapid-desktop/.claude/loop/bug_report.md):

  * **B-7 / F-DGF-V080-C-4**: ``top_k`` upper-bound silent accept
    (``999999999`` / ``2**63-1`` returned HTTP 200).
  * **B-8**: ``seed=-1`` silently accepted (silent-correctness — the
    backend folds the value to a valid uint32 PRNG key, so a caller
    that thought they passed a sentinel actually pinned a fixed
    state).
  * **B-9 / F-DGF-V080-V1**: ``stream_options.include_usage:"yes"``
    truthy-string silently treated as ``true`` (lax-mode bool
    coercion).

All three reproduce on the request-model layer, so the tests below
exercise the Pydantic models directly — that's the same gate every
route hits before request handlers run.
"""
import pytest
from pydantic import ValidationError

from vllm_mlx.api.anthropic_models import AnthropicRequest
from vllm_mlx.api.models import (
    _TOP_K_SENTINEL_CAP,
    ChatCompletionRequest,
    CompletionRequest,
    StreamOptions,
)
from vllm_mlx.api.responses_models import ResponsesRequest


def _user_msg():
    return [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# B-7: top_k upper bound
# ---------------------------------------------------------------------------


class TestTopKUpperBound:
    @pytest.mark.parametrize(
        "Model,extra",
        [
            (ChatCompletionRequest, {"messages": _user_msg()}),
            (CompletionRequest, {"prompt": "hi"}),
        ],
    )
    @pytest.mark.parametrize("bad_value", [999_999_999, 2**63 - 1])
    def test_top_k_above_sentinel_cap_rejected(self, Model, extra, bad_value):
        with pytest.raises(ValidationError) as excinfo:
            Model(model="x", top_k=bad_value, **extra)
        # Error message must name the field AND the cap so the client
        # can act on it without guessing.
        msg = str(excinfo.value)
        assert "top_k" in msg
        assert str(_TOP_K_SENTINEL_CAP) in msg

    @pytest.mark.parametrize(
        "Model,extra",
        [
            (ChatCompletionRequest, {"messages": _user_msg()}),
            (CompletionRequest, {"prompt": "hi"}),
        ],
    )
    def test_top_k_zero_accepted_by_design(self, Model, extra):
        """``top_k=0`` means "disabled" on mlx-lm — must NOT 4xx
        (v0.8.2 own error string read "use 0 to disable")."""
        req = Model(model="x", top_k=0, **extra)
        assert req.top_k == 0

    @pytest.mark.parametrize(
        "Model,extra",
        [
            (ChatCompletionRequest, {"messages": _user_msg()}),
            (CompletionRequest, {"prompt": "hi"}),
        ],
    )
    @pytest.mark.parametrize("good_value", [1, 64, 128, 100_000])
    def test_top_k_typical_values_accepted(self, Model, extra, good_value):
        req = Model(model="x", top_k=good_value, **extra)
        assert req.top_k == good_value

    @pytest.mark.parametrize(
        "Model,extra",
        [
            (ChatCompletionRequest, {"messages": _user_msg()}),
            (CompletionRequest, {"prompt": "hi"}),
        ],
    )
    def test_top_k_at_sentinel_cap_accepted(self, Model, extra):
        """The cap itself (``2**20``) is inclusive — wide enough to
        cover every shipped vocab size (Gemma 3 at 262144 is the
        widest)."""
        req = Model(model="x", top_k=_TOP_K_SENTINEL_CAP, **extra)
        assert req.top_k == _TOP_K_SENTINEL_CAP

    def test_top_k_anthropic_surface_also_capped(self):
        """Anthropic ``/v1/messages`` shares the same validator — the
        fix must mirror across surfaces."""
        with pytest.raises(ValidationError):
            AnthropicRequest(
                model="x",
                messages=_user_msg(),
                max_tokens=10,
                top_k=999_999_999,
            )
        # at-cap accepted
        req = AnthropicRequest(
            model="x",
            messages=_user_msg(),
            max_tokens=10,
            top_k=_TOP_K_SENTINEL_CAP,
        )
        assert req.top_k == _TOP_K_SENTINEL_CAP

    def test_top_k_negative_still_rejected_no_regression(self):
        """H-10's negative-int gate still fires — r5-E's upper cap is
        an additive tightening, NOT a relaxation of the lower gate.

        Pass ``model="x"`` and all other required fields so the only
        possible source of ``ValidationError`` is the ``top_k`` validator
        (codex pr_validate BLOCKING-4: previously this test could have
        passed for the wrong reason — missing-``model`` error — even
        if the negative ``top_k`` gate had been removed).
        """
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionRequest(model="x", messages=_user_msg(), top_k=-5)
        # Assert the validation error specifically cites top_k.
        errors = excinfo.value.errors()
        assert any("top_k" in err.get("loc", ()) for err in errors), (
            f"Expected top_k validation error, got: {errors}"
        )


# ---------------------------------------------------------------------------
# B-8: seed=-1 silent accept
# ---------------------------------------------------------------------------


class TestSeedNonNegative:
    @pytest.mark.parametrize(
        "Model,extra",
        [
            (ChatCompletionRequest, {"messages": _user_msg()}),
            (CompletionRequest, {"prompt": "hi"}),
            (ResponsesRequest, {"input": "hi"}),
        ],
    )
    @pytest.mark.parametrize("bad_value", [-1, -42, -(2**31)])
    def test_negative_seed_rejected(self, Model, extra, bad_value):
        with pytest.raises(ValidationError) as excinfo:
            Model(model="x", seed=bad_value, **extra)
        msg = str(excinfo.value)
        assert "seed" in msg
        assert ">= 0" in msg or "non-negative" in msg

    @pytest.mark.parametrize(
        "Model,extra",
        [
            (ChatCompletionRequest, {"messages": _user_msg()}),
            (CompletionRequest, {"prompt": "hi"}),
            (ResponsesRequest, {"input": "hi"}),
        ],
    )
    def test_seed_zero_accepted(self, Model, extra):
        """``seed=0`` is a legitimate PRNG key used by every eval
        harness — preserve it."""
        req = Model(model="x", seed=0, **extra)
        assert req.seed == 0

    @pytest.mark.parametrize(
        "Model,extra",
        [
            (ChatCompletionRequest, {"messages": _user_msg()}),
            (CompletionRequest, {"prompt": "hi"}),
            (ResponsesRequest, {"input": "hi"}),
        ],
    )
    def test_seed_omitted_accepted(self, Model, extra):
        """No seed → ``None`` (process-global PRNG)."""
        req = Model(model="x", **extra)
        assert req.seed is None

    @pytest.mark.parametrize(
        "Model,extra",
        [
            (ChatCompletionRequest, {"messages": _user_msg()}),
            (CompletionRequest, {"prompt": "hi"}),
        ],
    )
    def test_seed_large_positive_still_accepted(self, Model, extra):
        """64-bit positive seeds must pass — backend uint32 fold is
        downstream of validation (codex round-6 contract)."""
        req = Model(model="x", seed=2**63 - 1, **extra)
        assert req.seed == 2**63 - 1

    @pytest.mark.parametrize(
        "Model,extra",
        [
            (ChatCompletionRequest, {"messages": _user_msg()}),
            (CompletionRequest, {"prompt": "hi"}),
            (ResponsesRequest, {"input": "hi"}),
        ],
    )
    def test_seed_bool_still_rejected_no_regression(self, Model, extra):
        """``seed=True`` still 4xx's (H-11's contract) — r5-E is
        additive."""
        with pytest.raises(ValidationError):
            Model(model="x", seed=True, **extra)


# ---------------------------------------------------------------------------
# B-9: stream_options.include_usage strict bool
# ---------------------------------------------------------------------------


class TestStreamOptionsIncludeUsageStrict:
    @pytest.mark.parametrize(
        "bad_value",
        ["yes", "true", "no", "false", "1", "0", "on", "off", 1, 0],
    )
    def test_non_bool_include_usage_rejected_direct(self, bad_value):
        """StreamOptions itself must 4xx on non-bool wire values."""
        with pytest.raises(ValidationError) as excinfo:
            StreamOptions(include_usage=bad_value)
        msg = str(excinfo.value)
        assert "include_usage" in msg or "boolean" in msg

    @pytest.mark.parametrize(
        "Model,extra",
        [
            (ChatCompletionRequest, {"messages": _user_msg()}),
            (CompletionRequest, {"prompt": "hi"}),
        ],
    )
    @pytest.mark.parametrize(
        "bad_value", ["yes", "true", "false", "on", "1", 1, 0]
    )
    def test_request_nested_include_usage_rejected(self, Model, extra, bad_value):
        """The same gate must fire through the parent request model."""
        with pytest.raises(ValidationError):
            Model(
                model="x",
                stream_options={"include_usage": bad_value},
                **extra,
            )

    @pytest.mark.parametrize(
        "Model,extra",
        [
            (ChatCompletionRequest, {"messages": _user_msg()}),
            (CompletionRequest, {"prompt": "hi"}),
        ],
    )
    @pytest.mark.parametrize("good_value", [True, False])
    def test_proper_bool_include_usage_accepted(self, Model, extra, good_value):
        req = Model(
            model="x",
            stream_options={"include_usage": good_value},
            **extra,
        )
        assert req.stream_options is not None
        assert req.stream_options.include_usage is good_value

    @pytest.mark.parametrize(
        "Model,extra",
        [
            (ChatCompletionRequest, {"messages": _user_msg()}),
            (CompletionRequest, {"prompt": "hi"}),
        ],
    )
    def test_omitted_stream_options_accepted(self, Model, extra):
        """No ``stream_options`` → None (legacy default — no usage on
        the wire)."""
        req = Model(model="x", **extra)
        assert req.stream_options is None
