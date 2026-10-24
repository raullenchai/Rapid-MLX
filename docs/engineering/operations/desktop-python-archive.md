# Desktop Python archive verification

Desktop sidecar builds verify the Python distribution archive against a reviewed
SHA-256 digest before listing or extracting it. Cached archives use the same
check as new downloads. An invalid cache is removed and downloaded again; a
failed download or verification stops the build before extraction and removes
the temporary archive.

The default Python version and build tag have a pinned digest. Builders changing
`PBS_VERSION` or `PBS_TAG` must also set `PBS_SHA256` to the independently reviewed
64-character lowercase digest for that exact arm64 stripped archive. Updating a
version requires reviewing its publisher checksum together with the build inputs.
The digest is a build input; it is never downloaded dynamically during the build.

This changes custom build configuration only. Installed Desktop engines, model
loading, and normal local model directories keep their existing behavior.
