# Model catalog identity lock

The shipped text catalog (239 aliases) and audio catalog (44 aliases) resolve to
immutable Hugging Face commit IDs. `rapid_mlx/model_catalog_lock.json` records 279
repository identities including auxiliary audio models, Whisper processors,
Kokoro voices, and existing image/video runtime assets. Existing explicit image,
video, draft and target pins remain unchanged. Alias names, repository IDs,
subfolders and capability fields are preserved.

The migration selected 21 existing explicit pins, 69 locally cached main
revisions and 189 repository metadata revisions. This is a reproducible identity
lock, **not a model-quality audit or approval to execute repository Python**.
The manifest retains config SHA-256 digests, model/tokenizer executable selectors
and complete Git blob identity/size and Hub LFS SHA-256/size metadata for verification. No weights were downloaded to
collect it. Remote-code consent is a separate boundary.

To reproduce metadata collection for the pinned catalog:

```sh
python -m scripts.lock_model_catalog --output /private/tmp/rapid-mlx-catalog-lock/recheck.json
```

The collector reads exact revisions and small configuration files only. Compare
its repository identities and hashes with the packaged lock; collection timestamps
will differ. Updating a repository requires changing its catalog pin and manifest
in the same reviewed change. Audio runtime assets also carry enforced revisions.
The catalog review gate rejects newly introduced mutable text/audio identities
and treats the collector, packaged lock, runtime security adapters and downloader
as trust controls.

The shared downloader canonicalizes both text and audio aliases and applies pins
to default auxiliary repository IDs. Explicit user local directories remain local.
For multi-variant repositories, a raw repository ID still honors the previous
`pull --bits/--format` selection within the pinned snapshot; an explicit alias
retains its declared variant.
