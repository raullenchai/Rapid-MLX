# SA3 legacy checkpoint consent

Numeric SA3 tensor checkpoints load without additional consent on stable
PyTorch 2.10 or newer. The loader first attempts restricted tensor-only loading.
A process-wide setting alone no longer authorizes an unrestricted retry.

For an intentionally reviewed legacy checkpoint requiring object loading, set
both `RAPID_MLX_ALLOW_UNSAFE_SA3_PICKLE=1` and
`RAPID_MLX_SA3_TRUSTED_SHA256` to that file's SHA-256 digest. This explicitly
permits code-bearing deserialization of those bytes. Use tensor-only formats
where possible and do not enable this policy for unreviewed files.

The loader copies the file to private temporary storage, checks the digest,
and reads that same copy. A mismatching digest is refused. The temporary copy
is removed automatically, and the mechanism uses the platform's temporary
storage location on Linux and macOS. Consent never bypasses the PyTorch runtime
version requirement.
