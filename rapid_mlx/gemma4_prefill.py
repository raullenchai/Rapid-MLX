# SPDX-License-Identifier: Apache-2.0
"""Opt in only the batch generator's output-discarding prompt forward.

PromptProcessingBatch.prompt fills caches and discards every output. Its
generation/verification forwards use the original model with full outputs.
The hook is local to this generator; no model or dependency class is patched.
"""

from functools import partial


def _prompt_last_logits(batch, prompt, tokens):
    model = batch.model

    def forward(inputs, cache=None):
        return model(inputs, cache=cache, logits_to_keep=1)

    batch.model = forward
    try:
        return prompt(tokens)
    finally:
        batch.model = model


def install_gemma4_prefill(generator, model, *, speculative: bool) -> bool:
    """Install on ordinary Gemma 4 text generation only, including cache hits.

    Splitting the prompt batch creates an ordinary copy without this hook;
    verification, sampling, and logprob extraction therefore retain their
    original model. Multi-row and hidden-state calls also fail closed in the
    model itself. Speculative installers may add implicit capture consumers,
    so their entire prompt lane keeps the full forward.
    """
    from .models.gemma4_text import Gemma4TextWrapper

    if (
        speculative
        or not isinstance(model, Gemma4TextWrapper)
        or getattr(model.config, "num_kv_shared_layers", 0) <= 0
    ):
        return False
    batch = generator._prompt_batch
    batch.prompt = partial(_prompt_last_logits, batch, batch.prompt)
    return True
