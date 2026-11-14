# Legacy tensor loading

The shared legacy tensor loader requires stable PyTorch 2.10 or newer and
explicitly requests tensor-only deserialization on the CPU. It never retries
with unrestricted deserialization, including when a process-wide environment
setting requests it. Unknown and prerelease runtime versions are refused.

Numeric tensor checkpoints remain supported. Safetensors loading does not use
this helper. SAM and VoxCPM adapters use it for PyTorch tensor checkpoints;
SA3 starts with the same restricted loader and has a separate
[checkpoint consent policy](sa3-checkpoint-consent.md) for reviewed files
requiring object loading.
