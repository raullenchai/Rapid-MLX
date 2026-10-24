# Reviewed Kimi data-only tokenizer

`tokenization.py` is adapted from reviewed Moonshot AI source, licensed under
the accompanying Modified MIT LICENSE. Its source is `tokenization_kimi.py` from
`mlx-community/Kimi-K2.6-mlx-DQ3_K_M-q8` at immutable commit
`9cb5961bf22facced678b894ee5e9eb1538f7cba`, SHA256
`2ab1ffb6f5c4380758bd8d9752ff1041c09024182676a4311528fbdf92fb9599`.

LICENSE is from the original `moonshotai/Kimi-K2.6` repository at commit
`7eb5002f6aadc958aed6a9177b7ed26bb94011bb`. It includes the upstream large
commercial deployment attribution condition; downstream redistributors must
retain the license.

Review covered text encoding/decoding, byte mappings, special tokens, chunk
limits, and sandboxed Transformers chat templates. No checkpoint-owned Python
executes. The upstream BPE resolver is replaced with a strictly local parser;
initialization uses an explicit class and selected JSON fields instead of
AutoTokenizer. Explicit local checkpoints remain operator-owned data.

Native MLX `kimi_k25` handles model architecture. This adapter preserves the
upstream thinking defaults and special-token behavior. Tool declarations use
the pinned template's existing JSON branch, preserving exact callable names,
references and constraints. The optional TypeScript-style formatter is not
included or executed, and caller-supplied `tools_ts_str` is ignored.

Other local deviations are Ruff formatting, explicit optional/container/return
types, indexed token lookup, compatible positional encode/decode options and
explicit text/local-file validation and array-like decode normalization. Complete special tokens are recognized before ordinary
text is divided into bounded chunks. Special-token control is a keyword-only extension. Reviewed packaged SHA256:

- `tokenization.py`: `21bfbe6f53235954c1cc17db42aa5a5de86004943fb3dffb20afc818bd656781`
