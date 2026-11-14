# SPDX-License-Identifier: Apache-2.0
"""Run prompt scoring contracts in CI's isolated no-MLX process.

The same tests also run with real MLX in tests/test_prompt_logprobs.py. Only
array arithmetic and tokenizer dependencies are replaced here; scoring, route,
admission, abort, and cache ownership logic remain the production code.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests import test_completions_spec_parity as completion_contracts
from tests import test_prompt_logprobs as contracts

# Enroll the shared contracts without their Apple-only module marker.
for module in (completion_contracts, contracts):
    for name in dir(module):
        if name.startswith(("test_", "Test")) or name == "patched_config":
            globals()[name] = getattr(module, name)


@pytest.fixture(autouse=True)
def numpy_scoring_runtime(monkeypatch):
    import mlx.core as mx

    tokenizer_utils = sys.modules["mlx_lm.tokenizer_utils"]

    if not isinstance(mx, MagicMock):
        return  # The same contracts exercise real Metal on Apple hosts.

    monkeypatch.setitem(
        sys.modules["mlx_lm"].__dict__, "tokenizer_utils", tokenizer_utils
    )
    caches = sys.modules["mlx_lm.models.cache"]
    monkeypatch.setitem(
        sys.modules["mlx_lm"].__dict__, "models", sys.modules["mlx_lm.models"]
    )
    monkeypatch.setitem(sys.modules["mlx_lm.models"].__dict__, "cache", caches)
    monkeypatch.setitem(caches.__dict__, "make_prompt_cache", lambda model: [])

    # Dictionary patches restore lazy MagicMock attributes without leaving
    # deleted-attribute sentinels that would break later lifecycle tests.
    for name in (
        "array",
        "zeros",
        "full",
        "arange",
        "abs",
        "cumsum",
        "float32",
        "take_along_axis",
        "argpartition",
        "argsort",
        "exp",
    ):
        monkeypatch.setitem(mx.__dict__, name, getattr(np, name))
    monkeypatch.setitem(mx.__dict__, "eval", lambda *args: None)
    monkeypatch.setitem(
        mx.__dict__,
        "logsumexp",
        lambda x, axis=-1, keepdims=False: np.logaddexp.reduce(
            x, axis=axis, keepdims=keepdims
        ),
    )

    class Detokenizer:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.reset()

        def reset(self):
            self.tokens = []

        def add_token(self, token):
            self.tokens.append(token)

        @property
        def text(self):
            return self.tokenizer.decode(self.tokens)

        def finalize(self):
            pass

    monkeypatch.setitem(
        tokenizer_utils.__dict__, "NaiveStreamingDetokenizer", Detokenizer
    )
    monkeypatch.setitem(
        tokenizer_utils.__dict__, "SPMStreamingDetokenizer", Detokenizer
    )
