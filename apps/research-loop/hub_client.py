"""Opt-in Hub integration; standalone requests retain their original headers."""

import contextlib
import contextvars
import json
import os
import secrets
import subprocess
import threading
from http.cookies import SimpleCookie
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

_project = contextvars.ContextVar("rapid_project", default=None)


@contextlib.contextmanager
def project_scope(project):
    token = _project.set(str(getattr(project, "name", project)))
    try:
        yield
    finally:
        _project.reset(token)


def headers(project=None, priority=None):
    result = {"Content-Type": "application/json"}
    if os.environ.get("RAPID_HUB_URL") and os.environ.get("RAPID_HUB_TOKEN"):
        result.update(
            {
                "Authorization": "Bearer " + os.environ["RAPID_HUB_TOKEN"],
                "X-Rapid-Project": str(
                    project or _project.get() or os.environ.get("RAPID_PROJECT", "")
                ),
                "X-Rapid-Priority": priority
                or os.environ.get("RAPID_PRIORITY", "interactive"),
            }
        )
    return result


def call(path, body=None, method=None):
    request = Request(
        os.environ["RAPID_HUB_URL"] + "/hub/v1/" + path,
        json.dumps(body).encode() if body is not None else None,
        headers(),
        method=method,
    )
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def discover():
    catalog = call("models")
    models, embedding, embedding_model = {}, None, None
    for model in catalog["data"]:
        url = os.environ["RAPID_HUB_URL"] + "/hub/v1"
        ident = model["engine"] + "/" + model["id"]
        if "chat" in model["capabilities"]:
            models[ident] = url
        if "embeddings" in model["capabilities"] and embedding is None:
            embedding, embedding_model = url, ident
    if not models:
        raise ValueError("Hub has no chat-capable models")
    return models, embedding, embedding_model


def subprocess_env(project):
    env = dict(os.environ)
    if env.get("RAPID_HUB_URL"):
        env.update(RAPID_PROJECT=project.name, RAPID_PRIORITY="background")
    return env


def enable_capture(project):
    if os.environ.get("RAPID_HUB_URL"):
        call(
            "projects/" + project.name + "/capture",
            {"payload": True, "ttl_hours": 24},
            "PUT",
        )


def ui_authorized(handler):
    """Host-only plugin sessions, enabled only for the --hub server."""
    if not getattr(handler.server, "hub_mode", False):
        return True
    parsed = urlparse(handler.path)
    if parsed.path == "/healthz" and handler.command == "GET":
        return True
    expected = os.environ["RAPID_PLUGIN_ID"] + ".localhost"
    if urlparse("http://" + handler.headers.get("Host", "")).hostname != expected:
        handler.json({"error": "Plugin hostname required"}, 403)
        return False
    if parsed.path == "/hub-auth" and handler.command == "GET":
        try:
            code = parse_qs(parsed.query).get("code", [""])[0]
            call("ui-codes/redeem", {"code": code}, "POST")
        except (OSError, ValueError):
            handler.json({"error": "Invalid or expired login"}, 401)
            return False
        session = secrets.token_urlsafe(32)
        handler.server.hub_sessions.add(session)
        handler.send_response(303)
        handler.send_header(
            "Set-Cookie",
            "research_loop=" + session + "; Path=/; HttpOnly; SameSite=Strict",
        )
        handler.send_header("Location", "/")
        handler.send_header("Referrer-Policy", "no-referrer")
        handler.send_header("Content-Length", "0")
        handler.end_headers()
        return False
    cookie = SimpleCookie()
    try:
        cookie.load(handler.headers.get("Cookie", ""))
        value = cookie.get("research_loop")
        if value and value.value in handler.server.hub_sessions:
            return True
    except Exception:
        pass
    handler.close_connection = True
    handler.json({"error": "Open Research Loop from the Hub dashboard"}, 401)
    return False


def terminate_run(process):
    """Loop's SIGTERM handler kills its separate experiment process groups."""
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def watch_run(project, run, process):
    """Heartbeat jobs, acknowledge cancellation, and wait for loop termination."""
    ident = "research-loop-" + project.name + "-" + run.name

    def worker():
        import time

        with project_scope(project):
            registered = False
            cancelled = False
            while True:
                code = process.poll()
                try:
                    if not registered:
                        call(
                            "jobs",
                            dict(
                                id=ident,
                                title=f"{project.name} / {run.name}",
                                project=project.name,
                                state="running",
                                progress="Starting experiment",
                            ),
                        )
                        registered = True
                    job = call("jobs/" + ident)
                    if (
                        job["state"] in ("cancel_requested", "cancelling")
                        and not cancelled
                    ):
                        call(
                            "jobs/" + ident,
                            dict(
                                state="cancelling",
                                progress="Stopping experiment process groups",
                            ),
                            "PATCH",
                        )
                        cancelled = True
                        (run / "STOP").touch()
                        (run / "PAUSE").unlink(missing_ok=True)
                        terminate_run(process)
                        code = process.poll()
                    if code is not None:
                        call(
                            "jobs/" + ident,
                            dict(
                                state="cancelled"
                                if cancelled
                                else ("done" if code == 0 else "failed"),
                                progress=f"Process exited ({code})",
                            ),
                            "PATCH",
                        )
                        return
                    path = run / "events.jsonl"
                    progress = "Running"
                    if path.exists():
                        with path.open("rb") as stream:
                            stream.seek(max(0, path.stat().st_size - 8192))
                            lines = stream.read().splitlines()
                        if lines:
                            try:
                                last = json.loads(lines[-1])
                                progress = f"{last.get('event', 'running')} · round {last.get('round', '?')}"
                            except ValueError:
                                pass
                    call("jobs/" + ident, dict(progress=progress), "PATCH")
                except (OSError, ValueError):
                    # Re-read on the next attempt: cancellation may have raced
                    # the terminal PATCH, or its response may have been lost.
                    pass
                time.sleep(1)

    threading.Thread(target=worker, daemon=True).start()
