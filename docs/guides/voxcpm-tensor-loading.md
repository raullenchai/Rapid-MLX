# VoxCPM tensor sidecars

VoxCPM and VoxCPM2 retain numeric `audiovae.pth` sidecar support through the
shared tensor-only loader on stable PyTorch 2.10 or newer. Embedded VAE weights
are preferred; VoxCPM2 also prefers an available `audiovae.safetensors` sidecar.
Legacy archives must contain non-empty named tensors, optionally under
`state_dict`. Native VAE sanitization runs once before quantization and loading.

Normal local directories and downloaded snapshots need no additional consent.
Unsupported object data is refused without an unrestricted retry. A legacy VAE
sidecar on an unrecognized architecture is refused. Safetensors-only model
packs continue through the normal audio loader.
