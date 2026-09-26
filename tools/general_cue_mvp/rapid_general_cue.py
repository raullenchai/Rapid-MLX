"""Launch Meta's macOS CUA harness against local Rapid-MLX services."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "muse-glimmer-30b-4bit"
DEFAULT_JEV_URL = "http://127.0.0.1:8700/v1/systemone"


def _loopback_url(value: str, *, label: str) -> str:
    parsed = urlsplit(value.rstrip("/"))
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise argparse.ArgumentTypeError(
            f"{label} must be an unauthenticated http:// loopback URL"
        )
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{label} host must be a loopback IP"
        ) from error
    if not address.is_loopback:
        raise argparse.ArgumentTypeError(f"{label} host must be a loopback IP")
    if parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError(
            f"{label} must not contain a query or fragment"
        )
    return value.rstrip("/")


def _loopback_base_url(value: str) -> str:
    return _loopback_url(value, label="base URL")


def _loopback_jev_url(value: str) -> str:
    return _loopback_url(value, label="JEV URL")


@dataclass(frozen=True)
class JevConfig:
    mode: str
    url: str
    api_key: str | None
    execute_threshold: float
    planner_max_output_tokens: int


@dataclass(frozen=True)
class JevDecision:
    execute_probability: float
    replan_probability: float
    latency_ms: float


def _parse_arguments(
    argv: list[str] | None = None,
) -> tuple[list[str], JevConfig]:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded local desktop Computer Use session with Rapid-MLX. "
            "The upstream Meta harness captures and controls the current macOS desktop."
        )
    )
    parser.add_argument("--goal", required=True)
    parser.add_argument("--base-url", type=_loopback_base_url, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("RAPID_MLX_API_KEY"),
        help="Rapid server bearer; defaults to RAPID_MLX_API_KEY",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--screenshot-scale", type=float, default=0.75)
    parser.add_argument("--effort", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--planner-max-output-tokens", type=int, default=768)
    parser.add_argument(
        "--jev-mode",
        choices=("off", "shadow", "guard"),
        default="off",
        help=(
            "Score Muse actions with a local System One service. Shadow records "
            "scores only; guard replaces low-scoring actions with re-observation."
        ),
    )
    parser.add_argument("--jev-url", type=_loopback_jev_url, default=DEFAULT_JEV_URL)
    parser.add_argument(
        "--jev-api-key",
        default=os.environ.get("RAPID_MLX_SYSTEM_ONE_API_KEY"),
        help="System One bearer; defaults to RAPID_MLX_SYSTEM_ONE_API_KEY",
    )
    parser.add_argument("--jev-execute-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    if not args.api_key:
        parser.error("provide --api-key or set RAPID_MLX_API_KEY")
    if not (1 <= args.max_steps <= 40):
        parser.error("--max-steps must be between 1 and 40")
    if not (1 <= args.max_images <= 5):
        parser.error("--max-images must be between 1 and 5")
    if not (0.25 <= args.screenshot_scale <= 1.0):
        parser.error("--screenshot-scale must be between 0.25 and 1.0")
    if not (0.0 <= args.jev_execute_threshold <= 1.0):
        parser.error("--jev-execute-threshold must be between 0 and 1")
    if not (128 <= args.planner_max_output_tokens <= 4096):
        parser.error("--planner-max-output-tokens must be between 128 and 4096")

    forwarded = [
        "metacua",
        "agent",
        "--goal",
        args.goal,
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--api-key",
        args.api_key,
        "--syntax",
        "function",
        "--coords",
        "normalized",
        "--effort",
        args.effort,
        "--max-steps",
        str(args.max_steps),
        "--max-images",
        str(args.max_images),
        "--screenshot-scale",
        str(args.screenshot_scale),
        "--allow-bash",
        "false",
    ]
    forwarded.append("--no-overlay" if args.no_overlay else "--overlay")
    return forwarded, JevConfig(
        mode=args.jev_mode,
        url=args.jev_url,
        api_key=args.jev_api_key,
        execute_threshold=args.jev_execute_threshold,
        planner_max_output_tokens=args.planner_max_output_tokens,
    )


def build_metacua_arguments(argv: list[str] | None = None) -> list[str]:
    """Return only the arguments understood by the pinned Meta harness."""
    return _parse_arguments(argv)[0]


class SystemOneGuard:
    """Small synchronous client for Rapid's typed System One endpoint."""

    def __init__(self, config: JevConfig):
        self.config = config

    def assess(self, state: dict[str, Any]) -> JevDecision:
        body = {
            "state": state,
            "questions": {
                "action_gate": {
                    "type": "choice",
                    "instructions": (
                        "Choose how the desktop agent should handle the proposed action."
                    ),
                    "criteria": {
                        "execute": (
                            "Execute it now: it directly advances the user goal, agrees "
                            "with the described screen, and has acceptable impact."
                        ),
                        "replan": (
                            "Do not execute it: inspect the screen again and choose a "
                            "safer or more relevant next action."
                        ),
                    },
                }
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(
            self.config.url,
            data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=3.0) as response:
                payload = json.load(response)
        except (OSError, ValueError, urllib.error.HTTPError) as error:
            raise RuntimeError(f"System One action gate failed: {error}") from error
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            answer = payload["answers"]["action_gate"]
            probabilities = answer["probabilities"]
            execute = float(probabilities["execute"])
            replan = float(probabilities["replan"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "System One action gate returned an invalid body"
            ) from error
        if not all(
            math.isfinite(value) and 0 <= value <= 1 for value in (execute, replan)
        ):
            raise RuntimeError("System One action gate returned invalid probabilities")
        return JevDecision(
            execute_probability=execute,
            replan_probability=replan,
            latency_ms=latency_ms,
        )


class JevGuardBackend:
    """Score the visual planner's proposed action before Meta executes it.

    Shadow mode leaves the action unchanged and records the decision in the
    metacua trace. Guard mode converts a rejected action into a screenshot-only
    action, causing the visual planner to observe again on the next step.
    """

    def __init__(self, backend: Any, guard: SystemOneGuard, config: JevConfig):
        self._backend = backend
        self._guard = guard
        self._config = config
        self._goal = ""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    @property
    def label(self) -> str:
        return f"{self._backend.label} + Laya {self._config.mode}"

    def initial_conversation(self, goal_text: str, screenshot: Any) -> list[dict]:
        self._goal = goal_text
        return self._backend.initial_conversation(goal_text, screenshot)

    def send(self, system: str, conversation: list[dict]) -> Any:
        result = self._backend.send(system, conversation)
        calls = [
            call
            for call in result.tool_calls
            if call.name in ("computer.computer", "computer", "computer_computer")
        ]
        if not calls:
            return result

        state = {
            "goal": self._goal,
            "recent_context": _recent_text_context(conversation),
            "planner_reasoning": result.thinking,
            "planner_message": result.text,
            "proposed_tools": [
                {"name": call.name, "input": _json_safe(call.input)} for call in calls
            ],
        }
        try:
            decision = self._guard.assess(state)
        except RuntimeError as error:
            _record_gate_metadata(
                result, {"mode": self._config.mode, "error": str(error)}
            )
            return result

        rejected = decision.execute_probability < self._config.execute_threshold
        metadata = {
            "mode": self._config.mode,
            "execute_probability": decision.execute_probability,
            "replan_probability": decision.replan_probability,
            "threshold": self._config.execute_threshold,
            "latency_ms": decision.latency_ms,
            "rejected": rejected,
        }
        _record_gate_metadata(result, metadata)
        note = (
            f"[Laya {self._config.mode}: execute={decision.execute_probability:.3f}, "
            f"replan={decision.replan_probability:.3f}, {decision.latency_ms:.1f} ms]"
        )
        result.thinking = (
            note if not result.thinking else result.thinking + "\n\n" + note
        )

        if self._config.mode == "guard" and rejected:
            _replace_calls_with_reobservation(result, calls)
        return result

    def tool_result_items(
        self, runs: list[Any], screenshot: Any, notes=None
    ) -> list[dict]:
        return self._backend.tool_result_items(runs, screenshot, notes=notes)


def _record_gate_metadata(result: Any, metadata: dict[str, Any]) -> None:
    if not isinstance(result.raw_response, dict):
        result.raw_response = {}
    result.raw_response["_rapid_jev"] = metadata


def _replace_calls_with_reobservation(result: Any, rejected_calls: list[Any]) -> None:
    rejected_ids = {call.id for call in rejected_calls}
    for call in result.tool_calls:
        if call.id in rejected_ids:
            call.input = {"action": "screenshot"}
    for item in result.assistant_items:
        if item.get("type") != "function_call":
            continue
        if (item.get("call_id") or "") not in rejected_ids:
            continue
        item["arguments"] = json.dumps({"action": "screenshot"}, separators=(",", ":"))


def _recent_text_context(conversation: list[dict]) -> list[Any]:
    return [_redact_images(item) for item in conversation[-6:]]


def _redact_images(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("type") == "input_image":
            return {"type": "input_image", "image_url": "[current screenshot]"}
        return {key: _redact_images(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_redact_images(child) for child in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return "[current screenshot]"
    return _json_safe(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _make_rapid_compatible(backend: Any) -> Any:
    """Adapt Meta's wire details to Rapid's OpenAI Responses contract."""
    original_send = getattr(backend, "send", None)
    if callable(original_send):

        def rapid_send(system, conversation):
            result = original_send(system, conversation)
            if result.tool_calls:
                # Rapid already exposes Muse's self-directed reasoning as a
                # message item containing channel scaffolding. Replaying that
                # message teaches the next turn a malformed wire prefix and,
                # after a few steps, the model emits prose instead of a
                # function_call. Function calls are the only assistant items
                # needed to pair with the following outputs.
                result.assistant_items = [
                    item
                    for item in result.assistant_items
                    if item.get("type") == "function_call"
                ]
            return result

        backend.send = rapid_send

    original_tools = getattr(backend, "_tools", None)
    if callable(original_tools):

        def rapid_tools() -> list[dict]:
            tools = original_tools()
            for tool in tools:
                name = tool.get("name")
                if isinstance(name, str):
                    tool["name"] = name.replace(".", "_")
            return tools

        backend._tools = rapid_tools

    image_block = getattr(backend, "_image_block", None)
    if callable(image_block):

        def rapid_tool_result_items(runs, screenshot, notes=None) -> list[dict]:
            del notes
            items = []
            for run in runs:
                output = ("ERROR: " + run.output) if run.is_error else run.output
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": run.call_id,
                        "output": output,
                    }
                )
            if screenshot is not None:
                items.append(
                    {
                        "role": "user",
                        "type": "message",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Screen observation after the action:",
                            },
                            image_block(screenshot),
                        ],
                    }
                )
            return items

        backend.tool_result_items = rapid_tool_result_items
    return backend


def main() -> None:
    forwarded, jev_config = _parse_arguments()
    import metacua.agent as agent_module
    import metacua.muse_spark as muse_module

    original_http_post_json = muse_module.http_post_json

    def bounded_http_post_json(url, headers, body, *args, **kwargs):
        body = dict(body)
        body["max_output_tokens"] = jev_config.planner_max_output_tokens
        return original_http_post_json(url, headers, body, *args, **kwargs)

    muse_module.http_post_json = bounded_http_post_json

    original_make_backend = agent_module.make_backend

    def make_backend(config: Any) -> Any:
        backend = _make_rapid_compatible(original_make_backend(config))
        if jev_config.mode != "off":
            backend = JevGuardBackend(
                backend,
                SystemOneGuard(jev_config),
                jev_config,
            )
        return backend

    agent_module.make_backend = make_backend

    from metacua.__main__ import main as metacua_main

    sys.argv = forwarded
    metacua_main()


if __name__ == "__main__":
    main()
