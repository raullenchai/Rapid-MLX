# SPDX-License-Identifier: Apache-2.0
"""Regression tests for F-155 — ``n=0`` / ``n=-1`` silently accepted
as HTTP 200 with one choice on /v1/chat/completions and
/v1/completions, asymmetric with ``n > 1`` which already returned
400. Both forms are almost always client-side bugs:

* ``n=0``: typo for ``n=1`` (off-by-one)
* ``n=-1``: SDK sentinel for "use server default" — the user means
  ``n=1`` but the SDK serialized ``-1`` on the wire

Accepting them as 200 hid the bug; this test pins the new contract:
omitted (``None``) or ``1`` is the only legal surface; anything
else → 422 at parse time. The route-level ``n > 1`` reject in
``vllm_mlx/routes/chat.py`` and ``routes/completions.py`` stays as
a belt-and-braces guard in case an in-process caller skips the
Pydantic layer.
"""

import pytest
from pydantic import ValidationError

from vllm_mlx.api.models import (
    ChatCompletionRequest,
    CompletionRequest,
    _reject_non_one_n,
)


class TestRejectNonOneN:
    """Direct contract on the shared validator helper."""

    def test_none_flows_through(self):
        assert _reject_non_one_n(None) is None

    def test_one_flows_through(self):
        assert _reject_non_one_n(1) == 1

    @pytest.mark.parametrize("bad_n", [0, -1, 2, 5, -100, 100])
    def test_non_one_rejected(self, bad_n):
        with pytest.raises(ValueError, match="must equal 1"):
            _reject_non_one_n(bad_n)

    @pytest.mark.parametrize("bool_v", [True, False])
    def test_bool_rejected(self, bool_v):
        """``bool`` is a Python ``int`` subclass; without an explicit
        reject Pydantic would silently coerce ``True`` → 1 / ``False``
        → 0. The latter would re-introduce the F-155 hazard via the
        ``False`` wire form."""
        with pytest.raises(ValueError, match="not bool"):
            _reject_non_one_n(bool_v)


class TestChatCompletionRequestN:
    """End-to-end through Pydantic — every silent-200 wire form
    must surface a ``ValidationError`` at parse time."""

    def _base(self, **overrides):
        base = {
            "model": "qwen3-0.6b-8bit",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        }
        base.update(overrides)
        return base

    @pytest.mark.parametrize("n", [0, -1, 2, -100, 1000])
    def test_invalid_n_rejected(self, n):
        with pytest.raises(ValidationError) as ei:
            ChatCompletionRequest.model_validate(self._base(n=n))
        assert "n" in str(ei.value)
        assert "must equal 1" in str(ei.value)

    @pytest.mark.parametrize("n", [None, 1])
    def test_valid_n_accepted(self, n):
        req = ChatCompletionRequest.model_validate(self._base(n=n))
        assert req.n == n

    def test_omitted_n_accepted(self):
        req = ChatCompletionRequest.model_validate(self._base())
        assert req.n is None

    @pytest.mark.parametrize("bool_v", [True, False])
    def test_bool_n_rejected_through_pydantic(self, bool_v):
        """Codex round-2 BLOCKING: ``bool`` is an ``int`` subclass
        and Pydantic v2 coerces ``True`` → 1 / ``False`` → 0 BEFORE
        an after-mode validator runs — so a JSON ``"n": true`` wire
        form silently passed as ``n=1`` even though the helper-level
        ``_reject_non_one_n`` rejects booleans. Wire as
        ``mode="before"`` so the bool check fires on the raw value.
        """
        with pytest.raises(ValidationError) as ei:
            ChatCompletionRequest.model_validate(self._base(n=bool_v))
        assert "bool" in str(ei.value).lower() or "must equal 1" in str(ei.value)


class TestCompletionRequestN:
    """Mirror surface on the legacy completions schema —
    ``n > 1`` already 400'd at the route, but ``n=0`` / ``n=-1``
    fell through. Pin the schema-layer contract symmetrically with
    the chat surface so a single rule covers both endpoints."""

    def _base(self, **overrides):
        base = {
            "model": "qwen3-0.6b-8bit",
            "prompt": "hi",
            "max_tokens": 5,
        }
        base.update(overrides)
        return base

    @pytest.mark.parametrize("n", [0, -1, 2, -100, 1000])
    def test_invalid_n_rejected(self, n):
        with pytest.raises(ValidationError) as ei:
            CompletionRequest.model_validate(self._base(n=n))
        assert "n" in str(ei.value)
        assert "must equal 1" in str(ei.value)

    @pytest.mark.parametrize("n", [None, 1])
    def test_valid_n_accepted(self, n):
        req = CompletionRequest.model_validate(self._base(n=n))
        assert req.n == n

    @pytest.mark.parametrize("bool_v", [True, False])
    def test_bool_n_rejected_through_pydantic(self, bool_v):
        """Mirror surface: bool wire form must also fail at parse
        time on the legacy completions schema, same codex round-2
        rationale as on the chat surface."""
        with pytest.raises(ValidationError) as ei:
            CompletionRequest.model_validate(self._base(n=bool_v))
        assert "bool" in str(ei.value).lower() or "must equal 1" in str(ei.value)
