# Build on Rapid-MLX

Build menubar apps, editor plugins, scripts, and model pickers on an Apple
Silicon Mac by starting `rapid-mlx serve` as a child process or connecting to
an existing server. Use its HTTP control plane to manage models; one server
can switch models without starting another process.

**Stable** below means the documented commands, request fields, response
field types, and success codes are covered by
[`test_build_on_rapid_mlx_contract.py`](../../tests/test_build_on_rapid_mlx_contract.py).
JSON examples show subsets of responses; accept additional fields. **Experimental**
surfaces may change. This guide covers the standard `rapid-mlx serve` server;
specialized serving lanes may expose different controls.

## Run a background server

**Stable:** `rapid-mlx serve <model> --host 127.0.0.1 --port 8000` and the
readiness endpoint below. For tools that manage their own server lifetime,
run `serve` as a child process without sudo. The equivalent terminal command is:

```bash
rapid-mlx serve qwen3.5-4b-4bit --host 127.0.0.1 --port 8000
```

This Python example uses `httpx` (installed with Rapid-MLX). First probe the
known port: reuse an existing server, including one returning HTTP 503 while
starting, and start a child only when the connection fails. Unexpected HTTP
errors or an unresponsive listener are surfaced instead of starting a second
server. Reserve a port for your integration; a readiness response alone does
not establish the listener's identity or guarantee a particular model.

```python
import subprocess
import time

import httpx

child = None
with httpx.Client(base_url="http://127.0.0.1:8000", timeout=2.0, trust_env=False) as probe:
    try:
        try:
            response = probe.get("/health/ready")
        except httpx.ConnectError:
            child = subprocess.Popen([
                "rapid-mlx", "serve", "qwen3.5-4b-4bit",
                "--host", "127.0.0.1", "--port", "8000",
            ])
        else:
            if response.status_code != 503:
                response.raise_for_status()

        deadline = time.monotonic() + 300.0
        while True:
            if child is not None and child.poll() is not None:
                raise RuntimeError("Rapid-MLX exited before becoming ready")
            if time.monotonic() >= deadline:
                raise TimeoutError("Rapid-MLX did not become ready within 300 seconds")
            try:
                response = probe.get("/health/ready")
            except httpx.ConnectError:
                pass  # The child may not have bound its listener yet.
            else:
                if response.status_code != 503:
                    response.raise_for_status()
                    if response.json()["ready"] is True:
                        break
            time.sleep(0.5)

        # Keep your controller's work here (see the model/status examples below).
        # The finally block runs when that work finishes or raises an exception.
    finally:
        if child is not None:  # Never stop a server this controller did not start.
            child.terminate()  # SIGTERM requests graceful server shutdown on macOS.
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
```

Use an installed `rapid-mlx` on your controller's PATH and a model appropriate
for the Mac. Initial downloads can exceed the example's five-minute readiness
budget; adjust it for your deployment. Readiness can also mean lazy standby,
with weights not yet loaded. Keep the child handle for your app's lifetime,
and run the cleanup on cancellation or normal exit. The thirty-second shutdown
budget is an example policy: allow longer for active work before forcing exit.
Abrupt termination of the controller cannot run Python's `finally` block.

To inspect a known listener manually (HTTP 503 means it is not ready yet):

```bash
curl -fsS http://127.0.0.1:8000/health/ready
```

## Connect your controller

The examples use `http://127.0.0.1:8000`. Model controls and `/v1/status` require
`Authorization: Bearer ...` when an API key is configured. Set
`RAPID_MLX_API_KEY` in your controller's environment to the configured key.
The Python examples use `httpx` (installed with Rapid-MLX):

```python
import os
from urllib.parse import quote

import httpx

api = httpx.Client(
    base_url="http://127.0.0.1:8000",
    headers={"Authorization": f"Bearer {os.environ['RAPID_MLX_API_KEY']}"},
    timeout=1800.0,
    trust_env=False,  # Never send the API key through an HTTP(S)_PROXY.
)
```

Health probes and `/metrics` are unauthenticated, even when an API key is set.
Keep the listener on loopback or behind your own access controls. Check HTTP
status codes before using a response; a transport timeout is not proof that a
model load was cancelled. Re-read residency before retrying.

## Manage resident models

### Stable: `GET /v1/models/residency`

List the process's model records and memory ceiling. This is a residency view,
not a catalog of downloadable models. A record can remain registered while its
weights are unloaded: inspect `state` rather than assuming every row is loaded.
`primary` identifies the configured primary; `pinned` and `active_requests`
help explain why eviction may be refused.

```bash
curl -fsS -H "Authorization: Bearer $RAPID_MLX_API_KEY" http://127.0.0.1:8000/v1/models/residency
```

```python
response = api.get("/v1/models/residency")
response.raise_for_status()
models = response.json()["models"]
```

HTTP 200, example subset (byte counts and IDs vary):

```json
{
  "memory_limit_bytes": 12884901888,
  "models": [{"id": "qwen3.5-4b-4bit", "state": "resident", "primary": true, "pinned": true, "active_requests": 0}]
}
```

Use the returned `id` for deletion, including any repository slash.

### Stable: `POST /v1/models/load`

Load a model using an alias or repository ID. `model` is required.
`replace_group: "assistant"` switches the assistant model group; it can replace
the primary. `replace_mode: "reject"` refuses replacement when work is active.
Idle, unpinned models may be evicted to fit the memory ceiling. Loading may
require a download and can take substantially longer than a status request.

```bash
curl -fsS -H "Authorization: Bearer $RAPID_MLX_API_KEY" -H 'Content-Type: application/json' -d '{"model":"qwen3.5-9b-4bit","replace_group":"assistant","replace_mode":"reject"}' http://127.0.0.1:8000/v1/models/load
```

```python
response = api.post("/v1/models/load", json={
    "model": "qwen3.5-9b-4bit",
    "replace_group": "assistant",
    "replace_mode": "reject",
})
response.raise_for_status()
model_id = response.json()["id"]
```

HTTP 200 returns the loaded model's residency record, for example:

```json
{"id": "qwen3.5-9b-4bit", "state": "resident", "primary": true, "pinned": true, "active_requests": 0}
```

HTTP 409 means a residency conflict (including active work); HTTP 507 means
insufficient capacity. Do not assume the previous model remains loaded after
a failed replacement: the default memory policy can evict before loading.
Re-read residency to reconcile the controller's state. See the
[server guide](server.md#low-memory-model-switching) for further options and error handling.

### Stable: `DELETE /v1/models/{id}`

Evict an idle, unpinned model. Pinned models, including a protected primary,
and busy models return HTTP 409. An unknown ID returns HTTP 404. A successful
deletion returns HTTP 204 with an empty body; it does not delete cached weights.
For this example, choose an evictable ID from residency (the switched primary
above may be pinned).

```bash
curl -fsS -X DELETE -H "Authorization: Bearer $RAPID_MLX_API_KEY" http://127.0.0.1:8000/v1/models/org/model
```

```python
model_id = "org/model"  # Replace with an evictable ID from residency.
response = api.delete("/v1/models/" + quote(model_id, safe=""))
response.raise_for_status()
assert response.status_code == 204
```

## Observe the server

### Stable: `GET /v1/status`

Read the primary engine's activity. With an engine installed, `status` is
`idle` or `generating`; `num_running` and `num_waiting` describe work in flight.
This is not a readiness probe or a list of every resident engine.

```bash
curl -fsS -H "Authorization: Bearer $RAPID_MLX_API_KEY" http://127.0.0.1:8000/v1/status
```

```python
response = api.get("/v1/status")
response.raise_for_status()
status = response.json()
```

HTTP 200, engine-present subset:

```json
{"status": "idle", "model": "qwen3.5-9b-4bit", "num_running": 0, "num_waiting": 0}
```

With no primary engine the entire response is
`{"status":"not_loaded","model":null,"requests":[]}`; the counts above are
absent. An installed lazy engine can exist while its weights are unloaded.

### Stable: `GET /healthz`

A lightweight public liveness probe. HTTP 200 means the process responds;
inspect `ready` separately. During graceful draining it returns HTTP 503 with
`status: "draining"` and `ready: false`.

```bash
curl -fsS http://127.0.0.1:8000/healthz
```

```python
response = api.get("/healthz")
response.raise_for_status()
health = response.json()
```

HTTP 200 example:

```json
{"status": "healthy", "ready": true, "model_loaded": true, "model_name": "qwen3.5-9b-4bit"}
```

### Stable: `GET /health/ready`

Poll this public readiness probe before sending work. HTTP 503 means startup
or another readiness blocker prevents work; retry later. HTTP 200 means the
endpoint accepts work. A lazy endpoint can be ready with `state: "standby"`
and `model_loaded: false`; readiness does not guarantee resident weights.

```bash
curl -fsS http://127.0.0.1:8000/health/ready
```

```python
response = api.get("/health/ready")
response.raise_for_status()
ready = response.json()
```

HTTP 200 example:

```json
{"ready": true, "model": "qwen3.5-9b-4bit", "state": "ready", "model_loaded": true}
```

### Stable: `GET /metrics`

Scrape public Prometheus text exposition (HTTP 200,
`text/plain; version=0.0.4`). `rapid_mlx_model_loaded` is a gauge: 1 when the
primary weights are loaded, 0 otherwise. It is not a count of resident models.

```bash
curl -fsS http://127.0.0.1:8000/metrics
```

```python
response = api.get("/metrics")
response.raise_for_status()
prometheus_text = response.text
```

Example sample:

```text
rapid_mlx_model_loaded 1
```

**Experimental:** metric families beyond the sample guaranteed here; consult
[the server guide](server.md) before depending on individual series.

## Always-on Mac (system daemon)

For an unattended, always-on Mac, use the system daemon instead of tying the
server lifetime to your app. **Stable:** the service commands shown here.
`rapid-mlx service` manages a macOS system LaunchDaemon, which starts at boot
without a graphical login.
First create a dedicated non-administrator account (`serveuser` below), install
Rapid-MLX as that account, and download the intended model once as that account.
Then run installation from an administrator session:

```bash
sudo rapid-mlx service install --service-user serveuser --model qwen3.5-4b-4bit --host 127.0.0.1 --port 8000
rapid-mlx service status --json
sudo rapid-mlx service restart
```

From Python, run the same CLI with an argument list. Run installation/restart
from an administrator terminal where `sudo` can request authorization:

```python
import subprocess

subprocess.run([
    "sudo", "rapid-mlx", "service", "install",
    "--service-user", "serveuser", "--model", "qwen3.5-4b-4bit",
    "--host", "127.0.0.1", "--port", "8000",
], check=True)
subprocess.run(["rapid-mlx", "service", "status", "--json"], check=True)
```

After installation, use the curl health example above to check the HTTP listener;
there is no HTTP service-install operation. To remove the daemon registration:

```bash
sudo rapid-mlx service uninstall
```

See the [headless macOS service guide](headless-macos-service.md) for account
setup, FileVault boot limits, credentials, persistent configuration changes,
and recovery. Runtime model loading below changes the running process only;
it does not rewrite the service's boot configuration.

## Discover running servers

**Experimental / forthcoming:** `ps --json` (#5884) is **coming in the next release**.
This guide intentionally makes no schema promise for it. Until then, use the
readiness probe on known server addresses as above. For a system daemon,
use the stable service status command above as well.
Close the Python client when your controller shuts down:

```python
api.close()
```
