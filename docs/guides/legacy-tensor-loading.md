# Legacy tensor loading

The shared legacy tensor loader requires stable PyTorch 2.10 or newer and
explicitly requests tensor-only deserialization on the CPU. It never retries
with unrestricted deserialization, including when a process-wide environment
setting requests it. Unknown and prerelease runtime versions are refused.

Numeric tensor checkpoints remain supported. Safetensors loading does not use
this helper. Audio adapters adopt this helper in separate changes; adding the
helper alone does not change existing adapter behavior. The Python dependency
floor for PyTorch is now 2.10, so existing installations should update their
runtime before using adapters that adopt it.
