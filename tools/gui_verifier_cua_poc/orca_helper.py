"""Python client for Orca's native macOS computer-use helper.

Orca (com.stablyai.orca) ships a signed Swift helper app that speaks a
line-delimited JSON-RPC protocol over a Unix socket:

    orca-computer-use-macos --agent <socket> --token-file <token>

Methods (protocol v1, observed from Orca's computer-sidecar.js):
    handshake, listApps, listWindows, getAppState, click, performSecondaryAction,
    scroll, drag, typeText, pressKey, hotkey, pasteText, setValue, terminate

This gives us the Muse-on-Mac shape without writing our own Swift: AX snapshot
with element indexes, screenshot in the same call, setValue with read-back
verification on Orca's side, and typed error codes (element_not_found,
value_not_settable, permission_denied, ...).

POC scope: single request in flight, no reconnect, no background observer.
"""

from __future__ import annotations

import base64
import json
import secrets
import socket
import subprocess
import tempfile
import time
from pathlib import Path

DEFAULT_HELPER = (
    "/Applications/Orca.app/Contents/Resources/"
    "Orca Computer Use.app/Contents/MacOS/orca-computer-use-macos"
)
REQUEST_TIMEOUT = 60.0


class OrcaHelperError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class OrcaComputer:
    def __init__(self, helper_path: str = DEFAULT_HELPER):
        self.helper_path = helper_path
        self._tmpdir = Path(tempfile.mkdtemp(prefix="orca-computer-"))
        self.token = secrets.token_hex(32)
        self.token_file = self._tmpdir / "provider.token"
        self.token_file.write_text(self.token, encoding="utf-8")
        self.token_file.chmod(0o600)
        self.socket_path = self._tmpdir / "provider.sock"
        self._proc = subprocess.Popen(
            [
                self.helper_path,
                "--agent",
                str(self.socket_path),
                "--token-file",
                str(self.token_file),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._sock = self._connect(10.0)
        self._next_id = 1
        self.capabilities = self.call("handshake", {})["capabilities"]

    def _connect(self, timeout: float) -> socket.socket:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise OrcaHelperError(
                    "accessibility_error",
                    f"helper exited early with code {self._proc.returncode}",
                )
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(str(self.socket_path))
                sock.settimeout(REQUEST_TIMEOUT)
                return sock
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.2)
        raise OrcaHelperError("action_timeout", "helper socket never appeared")

    def call(self, method: str, params: dict) -> dict:
        request_id = self._next_id
        self._next_id += 1
        line = (
            json.dumps(
                {
                    "id": request_id,
                    "method": method,
                    "params": params,
                    "token": self.token,
                }
            )
            + "\n"
        )
        self._sock.sendall(line.encode())
        buffer = b""
        while True:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise OrcaHelperError("accessibility_error", "helper closed connection")
            buffer += chunk
            if b"\n" in buffer:
                payload, _, rest = buffer.partition(b"\n")
                if rest:
                    raise OrcaHelperError(
                        "accessibility_error", "unexpected extra data after response"
                    )
                break
        response = json.loads(payload.decode())
        if response.get("id") != request_id:
            raise OrcaHelperError("accessibility_error", "response id mismatch")
        if not response.get("ok"):
            error = response.get("error", {})
            raise OrcaHelperError(
                error.get("code", "accessibility_error"),
                error.get("message", "unknown helper error"),
            )
        return response.get("result", {})

    # --- observation -----------------------------------------------------

    def list_apps(self) -> list[dict]:
        return self.call("listApps", {}).get("apps", [])

    def list_windows(self, app: str) -> list[dict]:
        return self.call("listWindows", {"app": app}).get("windows", [])

    def get_app_state(self, app: str, screenshot: bool = True) -> dict:
        """Return {elements[], treeText, window, screenshot?} for one app window."""
        result = self.call("getAppState", {"app": app, "noScreenshot": not screenshot})
        snapshot = result.get("snapshot", result)
        if screenshot and snapshot.get("screenshotPngBase64"):
            snapshot["screenshot_png"] = base64.b64decode(
                snapshot["screenshotPngBase64"]
            )
        return snapshot

    # --- actions ---------------------------------------------------------

    def click(self, app: str, element_index: int | None = None, **kwargs) -> dict:
        params: dict = {"app": app}
        if element_index is not None:
            params["elementIndex"] = element_index
        params.update(kwargs)
        return self.call("click", params)

    def set_value(self, app: str, element_index: int, value: str, **kwargs) -> dict:
        params = {"app": app, "elementIndex": element_index, "value": value}
        params.update(kwargs)
        return self.call("setValue", params)

    def type_text(self, app: str, text: str, **kwargs) -> dict:
        params = {"app": app, "text": text}
        params.update(kwargs)
        return self.call("typeText", params)

    def press_key(self, app: str, key: str, **kwargs) -> dict:
        params = {"app": app, "key": key}
        params.update(kwargs)
        return self.call("pressKey", params)

    def hotkey(self, app: str, key: str, **kwargs) -> dict:
        params = {"app": app, "key": key}
        params.update(kwargs)
        return self.call("hotkey", params)

    def scroll(
        self, app: str, direction: str, element_index: int | None = None, **kwargs
    ) -> dict:
        params = {"app": app, "direction": direction}
        if element_index is not None:
            params["elementIndex"] = element_index
        params.update(kwargs)
        return self.call("scroll", params)

    def perform_secondary_action(
        self, app: str, element_index: int, action: str, **kwargs
    ) -> dict:
        params = {"app": app, "elementIndex": element_index, "action": action}
        params.update(kwargs)
        return self.call("performSecondaryAction", params)

    def terminate(self) -> None:
        try:
            self.call("terminate", {})
        except Exception:  # noqa: BLE001 - best-effort shutdown
            pass
        self._sock.close()
        self._proc.terminate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Orca native computer-use smoke")
    parser.add_argument("--list-apps", action="store_true")
    parser.add_argument("--state", default=None, help="app name for snapshot")
    parser.add_argument(
        "--click", type=int, default=None, help="element index to click"
    )
    parser.add_argument(
        "--png-out", default=None, help="write snapshot screenshot here"
    )
    args = parser.parse_args()

    computer = OrcaComputer()
    print("provider:", computer.capabilities.get("provider"))
    try:
        if args.list_apps:
            for app in computer.list_apps()[:15]:
                print(" ", app)
        if args.state:
            state = computer.get_app_state(args.state)
            elements = state.get("elements", [])
            print(f"{len(elements)} elements; window={state.get('windowTitle')!r}")
            for element in elements[:25]:
                print(
                    f"  [{element.get('index')}] {element.get('role')} "
                    f"{str(element.get('label') or element.get('value') or '')[:60]}"
                )
            if args.png_out and state.get("screenshot_png"):
                Path(args.png_out).write_bytes(state["screenshot_png"])
                print("screenshot ->", args.png_out)
    finally:
        computer.terminate()
