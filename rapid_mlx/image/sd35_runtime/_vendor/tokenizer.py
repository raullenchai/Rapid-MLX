# reference: https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion

import logging
from pathlib import Path
from typing import List

import mlx.core as mx
import numpy as np
import regex
from transformers import AutoTokenizer, T5Config

logger = logging.getLogger(__name__)


class Tokenizer:
    """A simple port of CLIPTokenizer from https://github.com/huggingface/transformers/ ."""

    def __init__(self, bpe_ranks, vocab, pad_with_eos=False):
        self.bpe_ranks = bpe_ranks
        self.vocab = vocab
        self.pat = regex.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            regex.IGNORECASE,
        )

        self.pad_to_max_length = True
        self.max_length = 77

        self._cache = {self.bos: self.bos, self.eos: self.eos}
        self.pad_with_eos = pad_with_eos

    @property
    def bos(self):
        return "<|startoftext|>"

    @property
    def bos_token(self):
        return self.vocab[self.bos]

    @property
    def eos(self):
        return "<|endoftext|>"

    @property
    def eos_token(self):
        return self.vocab[self.eos]

    def bpe(self, text):
        if text in self._cache:
            return self._cache[text]

        unigrams = list(text[:-1]) + [text[-1] + "</w>"]
        unique_bigrams = set(zip(unigrams, unigrams[1:]))

        if not unique_bigrams:
            return unigrams

        # In every iteration try to merge the two most likely bigrams. If none
        # was merged we are done.
        #
        # Ported from https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/tokenization_clip.py
        while unique_bigrams:
            bigram = min(
                unique_bigrams, key=lambda pair: self.bpe_ranks.get(pair, float("inf"))
            )
            if bigram not in self.bpe_ranks:
                break

            new_unigrams = []
            skip = False
            for a, b in zip(unigrams, unigrams[1:]):
                if skip:
                    skip = False
                    continue

                if (a, b) == bigram:
                    new_unigrams.append(a + b)
                    skip = True

                else:
                    new_unigrams.append(a)

            if not skip:
                new_unigrams.append(b)

            unigrams = new_unigrams
            unique_bigrams = set(zip(unigrams, unigrams[1:]))

        self._cache[text] = unigrams

        return unigrams

    def tokenize(self, text, prepend_bos=True, append_eos=True):
        if isinstance(text, list):
            return [self.tokenize(t, prepend_bos, append_eos) for t in text]

        # Lower case cleanup and split according to self.pat. Hugging Face does
        # a much more thorough job here but this should suffice for 95% of
        # cases.
        clean_text = regex.sub(r"\s+", " ", text.lower())
        tokens = regex.findall(self.pat, clean_text)

        # Split the tokens according to the byte-pair merge file
        bpe_tokens = [ti for t in tokens for ti in self.bpe(t)]

        # Map to token ids and return
        tokens = [self.vocab[t] for t in bpe_tokens]

        # Truncate
        max_length = self.max_length - int(prepend_bos) - int(append_eos)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            logger.warning(
                f"Length of tokens exceeds {self.max_length}. Truncating to {self.max_length}."
            )
        if prepend_bos:
            tokens = [self.bos_token] + tokens
        if append_eos:
            tokens.append(self.eos_token)

        return tokens


class T5Tokenizer:
    def __init__(
        self,
        config: T5Config,
        max_context_length: int,
        tokenizer_path: str | Path,
    ):
        self.max_length = max_context_length
        self._decoder_start_id = config.decoder_start_token_id
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            legacy=False,
            model_max_length=self.max_length,
            local_files_only=True,
        )

        self.pad_to_max_length = True
        self.pad_with_eos = False

    @property
    def eos_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def decoder_start_id(self) -> int:
        return self._decoder_start_id

    def encode(self, s: str) -> mx.array:
        return mx.array(
            self._tokenizer(
                s,
                return_tensors="np",
                return_attention_mask=False,
                max_length=self.max_length,
                truncation=True,
            )["input_ids"]
        )

    def decode(self, t: List[int], with_sep: bool = True) -> str:
        tokens = self._tokenizer.convert_ids_to_tokens(t)
        return "".join(t.replace("▁", " " if with_sep else "") for t in tokens)

    def tokenize(self, s: str) -> np.array:
        return [t.item() for t in self.encode(s)[0]]
