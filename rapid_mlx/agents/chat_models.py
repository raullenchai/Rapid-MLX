# SPDX-License-Identifier: Apache-2.0
"""Chat eligibility from /v1/models metadata, without loading model engines."""

from __future__ import annotations


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
        from rapid_mlx.agents import get_profile
        from rapid_mlx.connect import _parse_base_url

        self.port = _parse_base_url(base_url)[1]
        self.model = model
        self.skip_hint = (
            f"Skipping the server on :{self.port}: it serves a {non_chat_kind(model)} "
            f"model ({model.get('id', '(unknown)')}), not a chat model."
        )
        profile = get_profile(agent)
        recommended = (
            profile.recommended_models[0]
            if profile and profile.recommended_models
            else "<chat-model>"
        )
        super().__init__(
            f"the server on :{self.port} is serving a {non_chat_kind(model)} model "
            f"({model.get('id', '(unknown)')}); coding agents need a chat model — "
            f"start one with rapid-mlx serve {recommended}"
        )
