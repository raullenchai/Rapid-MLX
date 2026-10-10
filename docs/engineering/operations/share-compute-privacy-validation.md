# Pool privacy validation (2026-10-10)

PR #4458 was validated on an Apple M3 Ultra with 256 GiB memory, macOS
26.5.2, and Python 3.11.16. Models were already in the canonical HF cache;
offline mode was enabled and no model downloads were performed. Test HOME,
provider registration, relay endpoints, credentials, and launchd label were
isolated from production. The relay/provider fixture used localhost; inference
used real MLX models, not mocked responses.

## Reproduction and results

At baseline `dfaa9cc40`, serve `mlx-community/Qwen3-0.6B-4bit` under a disposable
HOME, submit a chat request with a unique canary and a repeated private-text
fixture, then stop the server cleanly. Two `*_tokens.bin` snapshots remained
under that HOME's `.cache/rapid-mlx/prefix_cache`. Their other-read permission
bit was set. Reading each snapshot with `_read_tokens_bin`, using its
`index.json` token count/save UUID, and decoding with the cached tokenizer
recovered the canary. The request contained 582 prompt tokens and generated
24 completion tokens. This demonstrates prompt reconstruction, rather than
only the existence of opaque cache files.

With the fix, run `rapid-mlx share MODEL --quicksilver` against a disposable
HTTP registration/heartbeat fixture and WebSocket relay. Send both streaming
and nonstreaming `/v1/chat/completions` requests through the relay, stop the
supervisor, restart without a provider key, and repeat. This passed for both
`mlx-community/Qwen3-0.6B-4bit` and `mlx-community/Qwen3.6-35B-A3B-8bit`:

- All eight requests returned HTTP 200 with positive prompt/completion usage;
  streaming responses included final usage and `[DONE]`.
- Restart reused the registered node. Heartbeats reached the fixture.
- A seeded, provably owned legacy cache was removed before serving, while an
  unrelated-model fixture remained intact.
- After each stop there were no token snapshots, safetensors snapshots, or
  radix index in the isolated prefix-cache root.

The 35B model required `mlx-vlm==0.7.2`; this was installed without dependencies
in a scratch venv using system packages. The global installation was unchanged.
The baseline 35B request did not produce a snapshot, so the baseline privacy
reproduction above relies on the 0.6B model.

## Service and TLS evidence

Real `--install-service` and reinstall calls successfully bootstrapped a
disposable GUI LaunchAgent with an absolute stable launcher. `launchctl print`
confirmed the program and running/scheduled state; `bootout` removed the label.
The service's Python process blocked in `open()` on the canonical HF cache's
`refs/main` before model startup. A process sample and Python stack dump
confirmed the location; the underlying macOS access cause remains undiagnosed.
Service inference is therefore **not validated** on this host. Foreground
inference passed, and install/reinstall/stop mechanics were validated separately.
Harbor owns checking service inference in a GUI account with appropriate cache
access before production rollout.

An independent adversarial reviewer reproduced heartbeat certificate failure
using a real localhost HTTPS server with a self-signed certificate. The first
failure now sets a fatal CA-repair message; subsequent heartbeat attempts do
not send requests. The supervisor stops its child and reports the certificate
repair hint instead of a revoked-key message. The reviewer approved the fix. A subsequent validator review also prompted
exclusive random radix staging files, no-follow descriptor permission changes,
and refusal of symlinked cache ancestors; four regressions cover these cases.

## Regression command

```sh
python3.11 -m pytest -q tests/test_share_quicksilver.py tests/test_share_cli.py \
  tests/test_share_privacy.py tests/test_prefix_cache_persistence.py \
  tests/test_radix_index.py tests/test_prefix_cache_radix_e2e.py
```

Result: 474 passed. Ruff formatting/lint and `git diff --check` passed.
The scratch harness and logs live under `/private/tmp/compute-share-dogfood`
and are ephemeral; this document records the conclusions and reproduction
procedure without retaining prompts or host credentials.

## Review dispositions

The validator's later static review proposed descriptor-relative protection
against concurrent ancestor replacement. The privacy boundary here is other
local users reading snapshots under the default owner-controlled HOME/cache
directories. Those readers cannot rename cache ancestors. Malicious same-UID
code and deliberately writable cache ancestry require broader filesystem
hardening and are not claimed to be contained by this change. The independent
reviewer confirmed this disposition.

A separate finding claimed an existing `.new` directory symlink reaches the
new chmod call. The full save method already removes stale staging and checks
for survivors before that call; a surviving symlink returns False before
writes/chmod. Both independent reproduction and the added
`test_snapshot_preclean_rejects_existing_staging_symlink` prove target contents
and permissions remain unchanged. The diff-only reviewer lacked that unchanged
pre-clean context. Concurrent malicious same-UID path replacement remains
outside the boundary above.

Later review fixes add restoration of the previous plist and re-bootstrap of
a stopped previous job if replacement bootstrap fails. Recovery failures are
reported explicitly. Radix temporary descriptors are closed if wrapping the
file object fails. Focused regressions cover both recovery outcomes and the
descriptor failure path.

Trust roots are initialized lazily inside the HTTPS request error boundary.
Invalid CA bundles raise the same terminal certificate error as an untrusted
peer, preserving supervisor cleanup and repair guidance. A real malformed
bundle and lazy handler initialization have regression coverage.
