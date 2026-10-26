# SPDX-License-Identifier: Apache-2.0
"""Request-local detokenization policy for the selected grammar families."""

from .selected_grammar import ToolPolicy, terminals
from .session import supported


def control_token_kwargs(cfg, engine, request):
    """Omit proven structural IDs, never ordinary text with the same spelling.

    Grammar markers remain decoded for the sole output demultiplexer. Native
    families retain their existing token routing and detokenization policy.
    """
    if not supported(cfg, bool(getattr(request, "tools", None))):
        return {}
    tokenizer = getattr(engine, "tokenizer", None)
    ids = getattr(tokenizer, "all_special_ids", ())
    if not isinstance(ids, (list, tuple, set, frozenset)) or not ids:
        return {}
    grammar_text = {
        terminal.text
        for fmt in ("hermes", "qwen", "glm47", "muse", "seed_oss")
        for terminal in terminals(ToolPolicy(fmt, enabled=True))
    }
    omitted = []
    for token_id in ids:
        spelling = tokenizer.decode([token_id], skip_special_tokens=False)
        if spelling and spelling not in grammar_text:
            omitted.append(token_id)
    return {"output_control_token_ids": tuple(omitted)} if omitted else {}
