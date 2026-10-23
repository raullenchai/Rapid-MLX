# Model pull integrity

`rapid-mlx pull` (also used by the desktop downloader) verifies selected weight
and config files before recording a successful pull or changing the served
variant. Catalog repositories use the packaged `model_catalog_lock.json` SHA256
manifest for the pinned commit, without a metadata network request. Every
selected manifest entry must exist and match. Other repositories use Hugging
Face LFS SHA256 or Git blob SHA1 metadata resolved at the exact downloaded
snapshot commit. These checks establish content identity, not a publisher
signature or a claim that the model itself is safe.

The first verification streams files in 8 MiB chunks. Receipts live alongside
HF snapshots under `.rapid-mlx-integrity`; subsequent pulls reuse them only
while file device, inode, size, modification time and change time match. The
receipt key includes repository, commit, selection and packaged manifest
identity. The local cache and receipts remain locally trusted; this is not
protection against a local account that can rewrite both. Download resume
behavior is unchanged. A mismatch stops completion without deleting blobs or
resume data; repair the named cached file and rerun the pull.

SAM-Audio's main checkpoint and lazy T5 encoder prefer safetensors. Numeric
legacy `checkpoint.pt` / `pytorch_model.bin` state dictionaries remain supported
through explicit `weights_only=True` on stable PyTorch 2.10 or newer. Unsafe
objects, unpatched runtimes and malformed preferred safetensors fail without
an unrestricted fallback. No model weights were downloaded for regression
tests; tiny real tensor archives and harmless malicious pickle sentinels cover
both accepted and rejected paths.
