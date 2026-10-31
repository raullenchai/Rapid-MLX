# VLM processor code consent

Some processor routes in the pinned VLM dependency force tokenizer repository
code trust on. A catalog pin fixes an identity; it does not grant code consent.

At the load boundary, inspect the resolved checkpoint with the existing
`RAPID_MLX_TRUST_REMOTE_CODE` or explicit `trust_remote_code` consent policy.
An environment value of `0` overrides explicit opt-in. Without consent, refuse
forced-trust routes when metadata declares repository code, and always refuse
routes that load a tokenizer from an auxiliary repository. Reviewed routes and
owned native loaders remain unchanged. No processor consent work runs per request.

## Forced-trust routes

The pinned VLM dependency forces trust for model types `got`, `inkling`, and
`inkling_mm_model` when loading tokenizer code from the primary snapshot. It
also forces trust for `moondream1` and `moondream2`, which load a tokenizer
from a separate, unpinned auxiliary repository. The `minicpmv4_6` and
`minicpmo` routes force trust only after the caller has opted in.
`tests/test_vlm_processor_inventory.py` scans the installed dependency and fails
when a new forced-trust route appears.

## Catalog qualification

The metadata-only fixture `tests/fixtures/vlm_processor_inventory.json` contains
126 catalog VLM aliases: those flagged `supports_image_input`, whose pinned
config is multimodal, or which are routed to the multimodal lane. Fourteen are
pinned to the text lane. All metadata was reachable at the pinned revisions;
nothing was missing. The fixture contains no weights.

| Route | Aliases |
| --- | ---: |
| Dependency route honoring caller consent | 117 |
| Owned native loaders | 9 |
| Forced-trust routes | 0 |

The native loaders comprise five prism, two embedding, one kimi, and one media
backend alias. Five aliases declare repository code metadata and remain loadable
without consent:

- `kimi-k2.6` declares `auto_map` in `config.json` and
  `tokenizer_config.json`; its owned native loader uses a data-only tokenizer.
- `lfm3-vl-2b-8bit` declares `config.json` `auto_map` and uses the owned native
  media backend.
- `spark-x2.6-4b-4bit`, `lfm3-1.7b-8bit`, and
  `nemotron-labs-diffusion-3b-4bit` declare only `config.json` `auto_map`. Their
  processor route passes caller consent through, so repository code is not
  loaded without opt-in.

None declares `model_file`. Refusing every `auto_map` would break these five;
refusal is therefore limited to forced-trust routes.

VLM aliases reference 12 draft repositories: three pinned and nine inspected at
`main` because their alias entries have no draft revision. Two DFlash drafts
declare `config.json` `auto_map` `AutoModel`. Drafters load weights only through
the drafter loader, which already enforces `model_file` consent, and never load
a processor. The only auxiliary processor identity in scope is the moondream
tokenizer repository, which is always refused without consent.

After catalog or dependency changes, re-qualify with
`python -m scripts.vlm_processor_inventory`, then rerun
`tests/test_vlm_processor_inventory.py`. The test fails if a
`supports_image_input` alias is missing or a recorded pin is stale.

A refusal happens before the loader runs and leaves the model unloaded. After
review and opt-in, retrying the load can succeed. Malformed metadata fails
closed; correct the metadata before retrying.
