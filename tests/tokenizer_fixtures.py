# SPDX-License-Identifier: Apache-2.0
"""Small real tokenizers for offline unit tests, independent of the HF cache."""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def byte_level_tokenizer():
    """Cover every UTF-8 byte, including partial multibyte streaming tokens."""
    vocab = {
        token: i for i, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
    }
    vocab["<eos>"] = len(vocab)
    # Keep merged words and leading-space pieces as well as byte fallbacks.
    # A byte-only vocabulary would miss bugs that truncate multi-byte-map
    # token strings in the optimized detokenizer's token table.
    merges = []
    for word in ("Hello", "Ġworld", "Test", "Ġmessage", "Goodbye", "Hi"):
        prefix = word[0]
        for character in word[1:]:
            merges.append((prefix, character))
            prefix += character
            if prefix not in vocab:
                vocab[prefix] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>")
