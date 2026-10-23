# Checkpoint code consent

Main text and vision loading inspect the resolved local snapshot before loading
that same snapshot. Checkpoint-bundled Python is refused by default. Ordinary
local model directories and supported installed architectures need no prompt.

After reviewing checkpoint code, callers can explicitly pass
`trust_remote_code=True` or set `RAPID_MLX_TRUST_REMOTE_CODE=1`.
`RAPID_MLX_TRUST_REMOTE_CODE=0` always disables repository code, including when
an API caller requests it. Consent does not permit Python paths outside the
checkpoint, except normal blob links within its configured Hub cache repository.

Supported GLM and Kimi checkpoints use installed model implementations. Kimi's
packaged tokenizer reads local vocabulary, token configuration and chat-template
data, preserving text, vision, tools and thinking without checkpoint Python.
Tool declarations use the pinned chat template's JSON format, preserving the
original callable names and complete schemas without a TypeScript translation.

The same policy applies to rerank (including local Jina), DFlash, DSpark, native
MTP, drafters, HiDream, Clef, OpenJev, diffusion and model conversion. Decider
uses its installed architecture and always disables checkpoint selectors and
repository tokenizer code. Ordinary supported model directories keep loading
without a prompt. This does not certify arbitrary checkpoint contents or change
catalog revisions.
