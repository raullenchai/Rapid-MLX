# Verified local ComfyUI demo run

On 2026-10-10, ComfyUI submitted six selected 1024×1024, 40-step Qwen-Image 2.1 jobs through Rapid-MLX on an M3 Ultra with 256 GB unified memory. Eight images were generated in total; two candidates had subtitle spelling errors and were retained locally before shorter-prompt reruns. Requested English headlines, subtitles and brand footers were visually reviewed. Decorative background sign lettering is not validated publication copy.

The API node → Rapid-MLX → SaveImage path was exercised with real weights. Four node contract tests passed, covering pixels, errors, cancellation and connectivity; Ruff and whitespace checks passed. Re-running the batch against the six completed outputs verified hashes and skipped generation.

| Destination | Seed | Server total seconds | Client wall seconds including queue |
| --- | ---: | ---: | ---: |
| lunar-night-market | 4300 | 204.81 | 333.36 |
| cloud-ocean | 4201 | 201.97 | 202.69 |
| mushroom-forest | 4202 | 188.58 | 393.43 |
| saturn-riviera | 4303 | 194.04 | 353.21 |
| deep-sea-express | 4204 | 192.10 | 192.55 |
| origami-city | 4205 | 194.69 | 389.17 |

These measurements describe this demo session, not a hardware comparison. Per-image server times come from Rapid-MLX completion logs and include image inference overhead. Client times include queue waits and polling; the two streams of requests used for retries explain the larger waits. Full versions, model revision, hashes and job IDs are in [verified-run.json](verified-run.json).

The final MP4 is 20 seconds, 1920×1080, 30 fps, H.264 without audio. It animates generated stills and ends with a two-second brand card. Original PNGs, JSON execution records, rejected candidates, contact sheet, workflow screenshot and MP4 are in ignored `media/`. They are local artifacts and are not included in the Git push. Blog and tweet files remain unpublished drafts.

Owner and handoff: Vector integration scope on Studio, branch `raullenchai/demo-twitter`; changed only this example. No server API, model engine, Desktop, CI or release changes. Human owner can review the media and publication drafts. Start FYI was sent through Orca to the discoverable Harbor desk; Atlas, Pixel and Echo were not available as live messaging targets on this host.
