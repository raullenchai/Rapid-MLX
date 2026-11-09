# Use Rapid from your phone

Keep the Mac running and connect your iPhone to the same Wi-Fi. No cloud
service, tunnel, account, or mobile app is needed.

## From the CLI

Start a chat model with LAN access and an API key (choose a private key):

```sh
rapid-mlx serve qwen3.5-4b-4bit --host 0.0.0.0 --api-key YOUR_PRIVATE_KEY
rapid-mlx connect
```

If your existing server is loopback-only or has no key, `connect` prints a
restart command containing a freshly generated strong key, once. Stop the
old server before running that command. The command preserves the model and
port; reapply any custom serving flags you need.

Scan the terminal QR with iPhone Camera and open the link in Safari. Choose
a model and chat. Stop cancels the response; New chat clears the conversation.
The current conversation and key stay in that browser's local storage.

For multiple servers or network interfaces:

```sh
rapid-mlx connect --port 8000 --host-ip 192.168.1.5
rapid-mlx connect --port 8000 --json
```

The primary IPv4 interface is preferred; alternatives are printed. `--json`
emits `url`, `base_url`, `api_key`, and `alternatives` (or an error and, when
appropriate, `restart_command`, with exit code 1). This output contains a
secret. The CLI inspects only the selected local Rapid process; if macOS denies
access to its listener or environment, use Desktop's sheet instead.

Existing tool instructions still work: `rapid-mlx connect openai-python`,
`rapid-mlx connect claude-code`, and `rapid-mlx connect continue`.
An explicit `--base-url` or `--host` retains the existing connection banner.

## From Desktop

Open the LAN controls in API settings or the integration connection panel,
then click **Connect phone**. If necessary, enable the existing LAN access
setting in the sheet. This persists the existing API key and restarts a running
model, interrupting active requests. Start a chat model, then scan the QR.
Choose another network address in the sheet if the Mac has several interfaces.

## Use a native chat app

Expand **Use a chat app instead** in Safari. Copy the Base URL and API key,
then paste them into an OpenAI-compatible connection in Conduit, Pal Chat,
Chapper, or OpenClient. Links and short setup instructions are on the page.
OpenClient requires the `/v1` suffix. These apps do not import the QR or a
Rapid deep link; it opens the browser page.

## Home screen and privacy

In Safari, choose **Share → Add to Home Screen**. If iOS gives the saved app
separate website storage, paste the API key under **Connection key**. There is
no offline inference or service worker. Use **Forget key and chat** to clear
this device; clearing Safari website data also removes the saved conversation.

The QR URL puts the key after `#k=`, never in the HTTP query or served HTML.
The page removes the fragment from browser history after saving the key. API
requests send `Authorization: Bearer …` to the same origin only. Desktop
continues to expose only its LAN inference allowlist; management stays local.

LAN traffic uses **unencrypted HTTP**. Use a trusted Wi-Fi network, treat the
QR/URL/key as a password, and do not port-forward the server. Anyone with the
key and network access can use the API. To revoke a phone's access, change the
server key (Desktop: rotate the persistent key and restart). This MVP has no
per-device keys. Guest Wi-Fi isolation, a VPN, or a Mac firewall may prevent
connection; allow local access and check the selected IP. Rescan after changing
network address or port.
