# Security Policy

## Supported Versions

Security fixes land on the latest release line. We do not maintain separate
patch branches for older versions — always run the newest release
(`brew upgrade rapid-mlx` / `pip install -U rapid-mlx`).

| Version            | Supported |
| ------------------ | --------- |
| Latest release     | ✅        |
| Older releases     | ❌        |

## Reporting a Vulnerability

**Please do not open a public issue for security reports.**

Report privately through GitHub:
[Security → Advisories → Report a vulnerability](https://github.com/raullenchai/Rapid-MLX/security/advisories/new).

- You will get an acknowledgement within 72 hours.
- For confirmed issues we aim to ship a fix or documented mitigation within
  14 days.
- We credit reporters in the release notes unless you prefer to stay
  anonymous.

## Scope

**In scope**

- Remote code execution or privilege escalation via the HTTP server
  (`rapid-mlx serve`) or the CLI
- Authentication bypass on the server when `--api-key` is set
- Path traversal or arbitrary file read/write outside the model cache
- Supply-chain integrity of the distributed artifacts (PyPI package,
  homebrew-core formula, `install.sh`)
- Unsafe behaviour in `install.sh` (unverified downloads, privilege misuse)

**Out of scope**

- Vulnerabilities in third-party model weights or upstream libraries
  (MLX, mlx-lm, transformers, FastAPI, …) — please report those upstream
- Prompt injection and model-output misbehaviour — inherent to LLMs, not a
  defect of this engine
- Denial of service from sending expensive requests to a server you
  deliberately exposed to the network
- Anything requiring physical access to the machine

## Verifying Our Artifacts

- **PyPI** — releases are published with Sigstore attestations (PEP 740)
  from this repository's release workflow. Inspect them under the
  "Integrity" panel on <https://pypi.org/project/rapid-mlx/>, or verify
  locally with [`pypi-attestations`](https://pypi.org/project/pypi-attestations/).
- **`install.sh`** — releases cut after this signing workflow landed ship
  `SHA256SUMS.txt` plus a cosign (keyless, GitHub-OIDC) signature bundle as
  release assets. Pick a release whose **Assets** list includes
  `SHA256SUMS.txt` (older releases predate the workflow and won't have it),
  then:

  ```bash
  TAG=<release-tag>   # a release whose Assets include SHA256SUMS.txt
  curl -fsSLO "https://github.com/raullenchai/Rapid-MLX/releases/download/$TAG/install.sh"
  curl -fsSLO "https://github.com/raullenchai/Rapid-MLX/releases/download/$TAG/SHA256SUMS.txt"
  curl -fsSLO "https://github.com/raullenchai/Rapid-MLX/releases/download/$TAG/SHA256SUMS.txt.sigstore.json"

  cosign verify-blob --bundle SHA256SUMS.txt.sigstore.json \
    --certificate-oidc-issuer https://token.actions.githubusercontent.com \
    --certificate-identity \
      "https://github.com/raullenchai/Rapid-MLX/.github/workflows/publish.yml@refs/tags/$TAG" \
    SHA256SUMS.txt
  shasum -a 256 -c SHA256SUMS.txt   # confirms the downloaded install.sh matches the signed sum

  # Only after both checks pass, run the verified copy you just downloaded —
  # not the website pipe (`curl … rapidmlx.com/install.sh | bash`), whose
  # bytes this recipe does not attest.
  bash install.sh
  ```

- **Homebrew** — the homebrew-core formula pins the PyPI sdist SHA256 and
  Homebrew CI re-verifies it on every build.

## Hardening Notes

- `rapid-mlx serve` binds to `127.0.0.1` by default. Only pass
  `--host 0.0.0.0` when you understand the exposure, and always set
  `--api-key` when binding beyond localhost (the server enforces OpenAI
  Bearer / Anthropic `x-api-key` auth).
- `rapid-mlx share` intentionally exposes your server beyond localhost —
  treat the URL it prints as a secret.
- Anonymous telemetry is **opt-in only** and never collects prompts,
  completions, file paths, IPs, or API keys. See
  <https://rapidmlx.com/docs/telemetry.html>.
- If your threat model forbids piping remote scripts into a shell, use
  `brew install rapid-mlx` or `pip install rapid-mlx` instead of the
  curl one-liner; the curl installer is optional sugar, not the only path.
