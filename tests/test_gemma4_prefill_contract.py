# SPDX-License-Identifier: Apache-2.0
"""Consumer opt-in tests run on Linux too, without importing MLX."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from rapid_mlx.gemma4_prefill import install_gemma4_prefill


@pytest.fixture
def setup(monkeypatch):
    module = ModuleType("rapid_mlx.models.gemma4_text")

    class Wrapper:
        def __init__(self, shared=2):
            self.config = SimpleNamespace(num_kv_shared_layers=shared)
            self.calls = []

        def __call__(self, inputs, cache=None, **kwargs):
            self.calls.append((inputs, cache, kwargs))
            return "logits"

    module.Gemma4TextWrapper = Wrapper
    monkeypatch.setitem(sys.modules, module.__name__, module)
    model = Wrapper()
    batch = SimpleNamespace(model=model)

    def prompt(tokens):
        batch.model(tokens, cache="prefix-cache")

    batch.prompt = prompt
    return model, batch, SimpleNamespace(_prompt_batch=batch)


def test_only_prompt_processing_opts_in(setup):
    model, batch, generator = setup
    assert install_gemma4_prefill(generator, model, speculative=False)
    batch.prompt([1, 2, 3])
    assert model.calls == [([1, 2, 3], "prefix-cache", {"logits_to_keep": 1})]
    assert batch.model is model
    # Sampling, prompt scoring, and speculative verification call the actual
    # model, not the temporary output-discarding prompt callable.
    model([4, 5, 6], cache="verify-cache")
    assert model.calls[-1] == ([4, 5, 6], "verify-cache", {})


@pytest.mark.parametrize("reason", ["speculative", "no-sharing", "other-family"])
def test_ineligible_consumers_keep_original_prompt(setup, reason):
    model, batch, generator = setup
    if reason == "no-sharing":
        model.config.num_kv_shared_layers = 0
    elif reason == "other-family":
        model = object()
    original = batch.prompt
    assert not install_gemma4_prefill(
        generator, model, speculative=reason == "speculative"
    )
    assert batch.prompt is original


def test_prompt_failure_restores_full_position_model(setup):
    model, batch, generator = setup

    def failing_prompt(tokens):
        batch.model(tokens)
        raise ValueError("prefill failed")

    batch.prompt = failing_prompt
    assert install_gemma4_prefill(generator, model, speculative=False)
    with pytest.raises(ValueError, match="prefill failed"):
        batch.prompt([1, 2, 3])
    assert batch.model is model
    assert model.calls[-1] == ([1, 2, 3], None, {"logits_to_keep": 1})
