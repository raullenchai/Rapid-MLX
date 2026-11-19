"""Opt-in Hub integration; standalone requests retain their original headers."""

import contextlib
import contextvars
import json
import os
import threading
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
        url = os.environ["RAPID_HUB_URL"] + "/hub/v1/engines/" + model["engine"] + "/v1"
        if "chat" in model["capabilities"]:
            models.setdefault(model["id"], url)
        if "embeddings" in model["capabilities"] and embedding is None:
            embedding, embedding_model = url, model["id"]
    if not models:
        raise ValueError("Hub has no chat-capable models")
    return models, embedding, embedding_model


def subprocess_env(project):
    env = dict(os.environ)
    if env.get("RAPID_HUB_URL"):
        env.update(RAPID_PROJECT=project.name, RAPID_PRIORITY="background")
    return env


def watch_run(project, run, process):
    """Use the same cooperative stop files as the Research Loop stop button."""
    ident = "research-loop-" + project.name + "-" + run.name

    def worker():
        import time

        with project_scope(project):
            registered = False
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
                    if job["state"] == "cancel_requested":
                        (run / "STOP").touch()
                        (run / "PAUSE").unlink(missing_ok=True)
                    if code is not None:
                        call(
                            "jobs/" + ident,
                            dict(
                                state="done" if code == 0 else "failed",
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
                    if code is not None:
                        return
                time.sleep(1)

    threading.Thread(target=worker, daemon=True).start()
