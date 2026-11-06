# SPDX-License-Identifier: Apache-2.0
"""Chat eligibility from /v1/models metadata, without loading model engines."""

from __future__ import annotations

import shlex
from functools import cached_property


def preferred_model(models: list[dict]) -> dict:
    """Prefer a public alias over an HF id in user-facing diagnostics."""
    return min(models, key=lambda model: ("/" in model["id"], len(model["id"])))


def non_chat_kind(model: dict) -> str | None:
    """Return the advertised non-chat kind; absent metadata stays compatible.

    Vision models advertise ``image`` (input), distinct from ``image-gen``.
    Embeddings/rerank advertise ``text`` input but exclusive capabilities.
    """
    modality = model.get("modality")
    kinds = {
        "video-gen": "video",
        "image-gen": "image generation",
        "tts": "speech synthesis",
        "stt": "speech recognition",
        "audio": "audio",
        "embedding": "embedding",
        "rerank": "rerank",
    }
    if isinstance(modality, str) and modality:
        if modality not in {"text", "image", "vision", "mllm", "text-diffusion"}:
            return kinds.get(modality, modality)
    capabilities = model.get("capabilities")
    if isinstance(capabilities, list) and capabilities and "text" not in capabilities:
        for capability in capabilities:
            if isinstance(capability, str):
                return kinds.get(capability, capability)
    return None


class NonChatModelError(ValueError, RuntimeError):
    """A healthy server advertises a model without a chat surface."""

    def __init__(self, base_url: str, model: dict, *, agent: str = "codex"):
        from rapid_mlx.connect import _parse_base_url

        self.port = _parse_base_url(base_url)[1]
        self.model = model
        self.agent = agent
        kind = non_chat_kind(model) or "non-chat"
        article = "an" if kind[0].lower() in "aeiou" else "a"
        description = f"{article} {kind} model ({model.get('id', '(unknown)')})"
        self.skip_hint = (
            f"Skipping the server on :{self.port}: it serves {description}, "
            "not a chat model."
        )
        self.message = (
            f"the server on :{self.port} is serving {description}; "
            "coding agents need a chat model"
        )
        super().__init__(self.message)

    @cached_property
    def recovery_target(self) -> tuple[str, int | None]:
        from rapid_mlx.agents import get_profile
        from rapid_mlx.cli import _port_is_busy
        from rapid_mlx.run.cli import _select_model

        model = (
            _select_model(
                explicit=None,
                profile=get_profile(self.agent),
                no_download=False,
                quiet=True,
            )
            or "<chat-model>"
        )
        # The rejected endpoint is occupied even when a bind probe cannot see it.
        port = next(
            (
                port
                for port in range(8000, 8100)
                if port != self.port and not _port_is_busy("127.0.0.1", port)
            ),
            None,
        )
        return model, port

    def recovery_hint(self, *, agent_version: str | None = None) -> str:
        model, port = self.recovery_target
        if port is None:
            return "Start a chat model on a free port, then re-run setup with --port <port>."
        model = model if model == "<chat-model>" else shlex.quote(model)
        setup = f"rapid-mlx agents {shlex.quote(self.agent)} --setup --yes"
        if port != 8000:
            setup += f" --port {port}"
        if agent_version:
            setup += f" --agent-version {shlex.quote(agent_version)}"
        return (
            f"Start a chat model: rapid-mlx serve {model} --port {port}, "
            f"then run: {setup}"
        )

    def __str__(self) -> str:
        return f"{self.message}. {self.recovery_hint()}"
