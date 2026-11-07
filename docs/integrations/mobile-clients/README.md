# Mobile chat clients with a Mac-hosted Rapid-MLX server

Use a phone as the chat interface and an Apple Silicon Mac for inference.
This is a provider-maintained candidate guide, not an endorsement by any app.
The server protocol is tested; installation, import and chat on physical iOS or
Android devices have **not** been tested in this work.

## Start the Mac server

Install Rapid-MLX using its [setup instructions](https://rapidmlx.com/docs/).
Choose an already downloaded supported model. In a terminal on your Mac:

```sh
export RAPID_MLX_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export RAPID_MLX_TELEMETRY=0
rapid-mlx serve <your-model-or-cached-snapshot> \
  --host <your-Mac-LAN-IP> --port 8000 --served-model-name rapid-local
```

Replace angle-bracket placeholders. Keep this terminal running. Enter the same
generated key into the phone app's API-key field using your own secure transfer
method; do not put it in a public configuration, screenshot or shared QR code.
Rapid-MLX enables metadata telemetry by default; the explicit environment setting
above opts out for this recipe.

For the first connection, put both devices on the same trusted Wi-Fi network,
allow the Mac process through its firewall, and grant the phone app local-network
permission when requested. The phone must use the Mac's LAN address: `localhost`
and `127.0.0.1` refer to the phone itself. Guest-network client isolation can block
the connection even when both devices appear on the same Wi-Fi.

HTTP is a local-network example and does not encrypt the key or chat. For remote
access, use a private VPN or an HTTPS endpoint; do not publicly port-forward the
example HTTP server. If an app refuses cleartext HTTP, use HTTPS/private-network
setup supported by that app rather than disabling API authentication.

## Check the endpoint before configuring an app

On the Mac, set `RAPID_BASE_URL` to `http://<your-Mac-LAN-IP>:8000/v1`:

```sh
curl --fail "$RAPID_BASE_URL/models" \
  -H "Authorization: Bearer $RAPID_MLX_API_KEY"
curl --fail "$RAPID_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $RAPID_MLX_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"rapid-local","messages":[{"role":"user","content":"Say hello."}],"max_tokens":64,"stream":false}'
```

Confirm `rapid-local` appears in `/models`. A Mac-side check does not establish
that the phone can reach it. Start with plain text and the selected model;
vision, audio, reasoning and tool capabilities depend on the model and client.

## App-specific setup candidates

### Conduit

Choose **Direct**, then an **OpenAI-compatible** profile using Chat Completions.
Use base URL `http://<Mac-LAN-IP>:8000/v1`, the generated key, and `rapid-local`.
This uses Direct mode, not an Open WebUI login or account. Confirm model listing
and streaming before trying media or voice. Source: [Conduit README](https://github.com/cogwheel0/conduit#readme).

### RikkaHub

Add an OpenAI-compatible provider with base URL
`http://<Mac-LAN-IP>:8000/v1`, Chat Completions path `/chat/completions`,
your generated key, and model `rapid-local`. Use Chat Completions rather than
Responses for this initial check. These fields match the current
[provider settings](https://github.com/rikkahub/rikkahub/blob/master/ai/src/main/java/me/rerere/ai/provider/ProviderSetting.kt).
RikkaHub supports provider QR export/import; generate the QR in the app only after
the connection works. Inspect whether an export includes credentials before
sharing it. No app-imported QR payload has been qualified here.

### Chatbox

The [chatbox.json](chatbox.json) candidate follows the published
[provider import format](https://docs.chatboxai.app/en/guides/providers/import-config).
Replace the documentation-only IP `192.0.2.10` with your Mac's reachable address.
The host excludes `/v1`; the separate `apiPath` supplies `/v1/chat/completions`.
Import the edited JSON, then fill the key in the app. No credential is embedded.
The icon URL is a public image request. No vision/tool/reasoning capability is
pre-enabled by this minimal text profile.

There is an open [iOS custom-endpoint authentication report](https://github.com/chatboxai/chatbox/issues/3516).
It is not reproduced or diagnosed here. If affected, record the app version and
request status with secrets removed; do not turn off the server key as a fix.

### ChatterUI (Android)

The [chatterui.json](chatterui.json) candidate follows the current
[API template format](https://github.com/Vali-98/ChatterUI/blob/master/docs/CustomTemplates.md).
Replace both `192.0.2.10` addresses. Use **API Manager → Gear → Add Template**,
create an API entry, enter the key and fetch/select `rapid-local`.
The template uses Bearer authentication, the `data` model list and
`choices.0.delta.content` streaming text. Images/audio are disabled in this
candidate. An existing OpenAI template may also suffice; a built-in new provider
is not necessary for the proposed guide.

### Oriveo

Use a **Relay** connection with OpenAI-compatible Chat Completions and the Mac's
`/v1` URL, generated key and `rapid-local`. Configure the host manually for this
candidate; automatic discovery has not been verified. This needs no Oriveo account,
hosted inference or shared catalog entry. Source: [Oriveo README](https://github.com/oriveo/oriveo#readme).

## Verification record — 2026-11-06

On Apple M3 Ultra, Rapid-MLX 0.16.8 used cached
`mlx-community/Qwen3.5-4B-MLX-4bit` revision
`32f3e8ecf65426fc3306969496342d504bfa13f3`, served as `rapid-local`.
The test server bound **127.0.0.1:18943**, with a random environment-based key,
telemetry disabled, downloads disabled, disk caches disabled and thinking off.
No new model was downloaded. Synthetic requests established:

| Probe | Result |
| --- | --- |
| `/v1/models`, no key | HTTP 401 |
| `/v1/models`, wrong key | HTTP 401 |
| `/v1/models`, correct Bearer key | HTTP 200; `rapid-local` listed |
| Non-streaming text | `MOBILE_BACKEND_OK` |
| Streaming text | `MOBILE_BACKEND_OK` reconstructed from deltas; `[DONE]` terminator |

JSON syntax and expected template fields were checked locally. These checks do
not run either app's importer. LAN reachability, mobile HTTP restrictions,
background/resume, actual imports, media, maximum context and mobile tool loops
remain unverified. Do not promote the candidates as certified mobile integrations.
