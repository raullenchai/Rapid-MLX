# Mobile-client integration outreach — 2026-11-06

Owner: Echo. Host: Studio / local Mac. Branch: `echo/mobile-clients-20261106`.
The user authorized progressing mobile chat integrations and community outreach.
All five targets were checked for existing Rapid-MLX issues and discussions;
none were found before this batch. Each submission discloses the provider
affiliation, AI assistance, and absence of physical-device validation.

## Submitted

| Project | Submission | Concrete scope | Current disposition |
| --- | --- | --- | --- |
| Conduit | [Discussion #940](https://github.com/cogwheel0/conduit/discussions/940) | Existing Direct mode; phone-to-Mac setup guide | Submitted; awaiting response |
| RikkaHub | [Discussion #2300](https://github.com/rikkahub/rikkahub/discussions/2300) | Existing OpenAI-compatible configuration; app-owned QR sharing workflow | Submitted; awaiting response |
| Chatbox | [Issue #4032](https://github.com/chatboxai/chatbox/issues/4032) | Provider import JSON and mobile setup documentation | Open proposal; not accepted |
| ChatterUI | [Issue #663](https://github.com/Vali-98/ChatterUI/issues/663) | 38-line candidate template and setup guide | Open proposal; not accepted |
| Oriveo | [Issue #20](https://github.com/oriveo/oriveo/issues/20) | Manually configured existing Relay connection | Open proposal; not accepted |

No upstream PR was created. Conduit's README directs contributors to discussion
first; ChatterUI and Oriveo require an issue before substantive PRs. Chatbox's
feature-request structure was followed. RikkaHub's Ideas category was used for
the configuration/sharing proposal. These are five tailored integration proposals,
not five completed integrations or endorsements.

## Reviewable materials

Committed at `298308a74` under `docs/integrations/mobile-clients/`:

- Setup guide covering server binding, authentication, phone localhost versus
  Mac address, trusted LAN versus HTTPS/private VPN, and metadata-telemetry opt-out.
- Credential-free Chatbox provider-import JSON with a documentation-only IP.
- Credential-free ChatterUI API template with matching model/SSE field paths.

The public configurations deliberately omit secrets and do not claim image,
audio, maximum-context or tool support on the mobile apps. No real chat or
machine address was uploaded.

## Verification

Rapid-MLX 0.16.8, Apple M3 Ultra, cached Qwen3.5-4B-MLX-4bit revision
`32f3e8ecf65426fc3306969496342d504bfa13f3`, served as `rapid-local`, loopback port
18943. Correct Bearer key: models 200, plain text and SSE text succeeded.
Missing/wrong key: 401. Synthetic completion marker: `MOBILE_BACKEND_OK`;
stream included `[DONE]`. Model downloads, disk caches and telemetry were
disabled for the run. JSON parsing, expected fields, URL composition and absence
of embedded credentials passed local checks. The task-owned server was stopped.

There was no available iOS simulator or Android debugging device in this session.
No physical phone, importer, LAN connection, QR import, background/resume or
mobile multimodal test is claimed. Chatbox #3516 is an existing authentication
report, not a bug reproduced by this work.

## Next actions and ownership

Echo: inspect replies before additional outreach; continue in existing threads,
with no duplicate pings. Atlas/Pixel: qualify the actual iOS/Android client
versions once devices are available, including import, authentication and
foreground/background streaming. RikkaHub QR generation should use the app's
actual format and be tested before publishing a usable import link. If upstream
maintainers welcome a change, prepare the smallest requested docs/template PR
and perform the applicable review workflow. No timeline or acceptance was promised.
