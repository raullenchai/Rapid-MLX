# SPDX-License-Identifier: Apache-2.0
"""Read tiktoken vocabulary data locally, without URL or cache resolution."""

import base64
from pathlib import Path


def load_local_bpe(filename):
    path = Path(filename)
    if not path.is_file():
        raise ValueError("Kimi requires a local tiktoken.model file")
    ranks = {}
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            token, rank = line.split()
            token = base64.b64decode(token, validate=True)
            rank = int(rank)
            if token in ranks or rank < 0:
                raise ValueError("Invalid or duplicate Kimi BPE token")
            ranks[token] = rank
    if set(ranks.values()) != set(range(len(ranks))):
        raise ValueError("Kimi BPE ranks must be unique and contiguous")
    if any(bytes([value]) not in ranks for value in range(256)):
        raise ValueError("Kimi BPE must include every byte")
    return ranks
