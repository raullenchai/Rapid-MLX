"""T5 tokenizer wrapper using sentencepiece."""

import os
from typing import List, Union

import mlx.core as mx


class T5Tokenizer:
    """Simple T5 tokenizer using sentencepiece.

    Args:
        model_path: Path to directory containing tokenizer_spiece.model.
        max_length: Maximum sequence length.
    """

    def __init__(self, model_path: str, max_length: int = 226):
        import sentencepiece as spm

        # Try multiple locations
        candidates = [
            os.path.join(model_path, "tokenizer_spiece.model"),
            os.path.join(model_path, "tokenizer", "spiece.model"),
            os.path.join(model_path, "spiece.model"),
        ]
        spiece_file = None
        for c in candidates:
            if os.path.exists(c):
                spiece_file = c
                break
        if spiece_file is None:
            raise FileNotFoundError(f"No spiece.model found in {model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spiece_file)
        self.max_length = max_length
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(
        self,
        text: Union[str, List[str]],
        max_length: int = None,
    ) -> mx.array:
        """Tokenize text to padded token IDs.

        Args:
            text: Input string or list of strings.
            max_length: Override default max length.

        Returns:
            (B, L) int32 array of token IDs, padded to max_length.
        """
        max_length = max_length or self.max_length

        if isinstance(text, str):
            text = [text]

        batch_ids = []
        for t in text:
            ids = self.sp.Encode(t)
            ids = ids + [self.eos_token_id]  # T5 appends EOS
            ids = ids[:max_length]
            # Pad
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
            batch_ids.append(ids)

        return mx.array(batch_ids, dtype=mx.int32)
