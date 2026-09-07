# Qwen Image Edit q8 release dogfood (2026-09-05)

## Candidate and environment

- Alias: `qwen-image-edit`
- Repository: `OsaurusAI/Qwen-Image-Edit-mflux-q8`
- Revision: `a458969f2a612433cf036bfc3d8d818ceba29fab`
- Cataloged payload: 37,472,689,129 bytes (34.9 GiB)
- Runtime: Rapid-MLX 0.13.4 from the candidate worktree, Python 3.12.14,
  mflux 0.19.1, MLX 0.32.2
- Desktop candidate: Rapid-MLX Desktop 0.13.4 with its packaged mflux 0.19.0,
  launched with `RAPID_BIN` pointing to the candidate worktree environment
- Host: Apple M3 Ultra Mac Studio, 256 GB unified memory, macOS 26.5.2
- Source: `docs/assets/logo.png` (683 x 606 PNG)
- License provenance: the pinned checkpoint card declares Apache-2.0 and names
  the Qwen Image Edit 2509 source model

The public minimum is 96 GB rather than 64 GB. The measured server peak already
exceeds a 64 GB machine before allowing useful headroom for macOS and the
Desktop app.

## Reproduction

Create an isolated environment so an older globally installed mflux does not
change the result:

```bash
uv venv /private/tmp/rapid-qwen-edit-venv --python python3.12
uv pip install --python /private/tmp/rapid-qwen-edit-venv/bin/python -e '.[image]'
/usr/bin/time -l /private/tmp/rapid-qwen-edit-venv/bin/rapid-mlx pull qwen-image-edit
/usr/bin/time -l /private/tmp/rapid-qwen-edit-venv/bin/rapid-mlx serve \
  qwen-image-edit --host 127.0.0.1 --port 8127
```

The default edit omitted `steps` deliberately:

```bash
curl --fail-with-body --max-time 1800 \
  http://127.0.0.1:8127/v1/images/edits \
  -F image=@docs/assets/logo.png \
  -F model=qwen-image-edit \
  -F 'prompt=Place this cheetah logo on a clean warm sunset background while preserving the cheetah shape and black line art' \
  -F response_format=b64_json \
  -o /private/tmp/qwen-edit-response.json
```

Inspect the live process after the request with `vmmap -summary <pid>`. Restart
with `HF_HUB_OFFLINE=1` and repeat an edit to exercise the verified local cache.

## Results

| Check | Result |
| --- | --- |
| Exact cold pull | 34.9 GiB in 5m 6s; pinned snapshot path matched the revision |
| Cold default edit | HTTP 200; 20/20 steps; 174.98 s |
| Warm default edit | HTTP 200; 20/20 steps; 173.65 s |
| Output | Decodable non-uniform RGB PNG, 672 x 592 |
| Visual instruction | Warm-sunset and cool-moonlight edits both preserved the cheetah subject and line art |
| Peak server footprint | 70,506,019,848 bytes (65.66 GiB), zero process swap |
| Cancellation | Cancelled at step 3; HTTP 200 with `cancelled=true`; next default edit succeeded |
| Wrong endpoint | `/v1/images/generations` returned 409 `wrong_image_endpoint` |
| Offline restart | `HF_HUB_OFFLINE=1`; model card and a one-step edit returned HTTP 200 |
| Discovery | Alias and repository id both reported `capabilities=["image.editing"]` |
| Desktop | Imported the source, selected the cached alias, started its sidecar, displayed 1-20 step progress and ETA, and added the completed edit as selected gallery item 1 |

The resident-model admission charge is 68 GiB, just above the measured peak.
The catalog minimum is 96 GB so users retain operating-system and application
headroom instead of discovering the incompatibility after a 34.9 GiB download.
