# Reviewed Kimi data-only tokenizer

`tokenization.py` and `tool_declaration_ts.py` are reviewed Moonshot AI source,
licensed under the accompanying Modified MIT LICENSE. They come from
`mlx-community/Kimi-K2.6-mlx-DQ3_K_M-q8` at immutable commit
`9cb5961bf22facced678b894ee5e9eb1538f7cba`:

- `tokenization_kimi.py`: SHA256 `2ab1ffb6f5c4380758bd8d9752ff1041c09024182676a4311528fbdf92fb9599`
- `tool_declaration_ts.py`: SHA256 `97e09ed262f1cd38f87584eaa46c5d4d5b3c9725dd299b21642f389040794ebc`

LICENSE is from the original `moonshotai/Kimi-K2.6` repository at commit
`7eb5002f6aadc958aed6a9177b7ed26bb94011bb`. It includes the upstream large
commercial deployment attribution condition; downstream redistributors must
retain the license.

Review covered all tokenizer and tool serializer source: text encoding/decoding,
byte mappings, special tokens, chunk limits, JSON Schema to TypeScript formatting,
and sandboxed Transformers chat templates. No checkpoint-owned Python executes.
The upstream BPE resolver is replaced with a strictly local parser; initialization
uses an explicit class and selected JSON fields instead of AutoTokenizer. Explicit local checkpoints remain operator-owned data.

Native MLX `kimi_k25` handles model architecture. This adapter preserves the
upstream thinking defaults, tool declarations, and special-token behavior.

Local deviations also include Ruff formatting, explicit optional/container/return
types and indexed token lookup. Reviewed packaged SHA256 values:

- `tokenization.py`: `a1e79c8652393ad06c92d5a2e53fc90d87ddc6768fd06d5ab2a20df253a4a89d`
- `tool_declaration_ts.py`: `fefa602eba382758873622271972f252b48d5e8382652094d929cf953446ace8`
