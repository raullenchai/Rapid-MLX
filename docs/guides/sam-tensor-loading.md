# SAM-Audio tensor loading

SAM-Audio and its nested T5 encoder use owned tensor-loading adapters. Local
model directories remain supported, and remote model assets resolve once to a
local snapshot before configuration, weights, or tokenizers are loaded.

Safetensors weights are preferred. Numeric `checkpoint.pt` and
`pytorch_model.bin` archives remain supported through the shared restricted
loader on stable PyTorch 2.10 or newer. Unsupported object data and missing
weights are refused; a corrupt safetensors file does not trigger a legacy
fallback. T5 tokenizers load locally with repository code disabled.

The nested encoder's configured name is retained. Its data is loaded through
the same restrictions, without a new prompt for normal tensor checkpoints.
