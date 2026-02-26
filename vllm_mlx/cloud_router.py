# SPDX-License-Identifier: Apache-2.0
"""
Cloud routing for large-context requests.

Routes requests with many new (uncached) tokens to a cloud LLM via litellm,
keeping short requests on local MLX inference. This avoids the O(n^2) attention
prefill bottleneck on Apple Silicon for cold, large-context prompts.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)


class CloudRouter:
    """
    Routes requests to a cloud LLM when new token count exceeds a threshold.

    Uses litellm for provider-agnostic cloud calls (Anthropic, OpenAI, etc.).
    litellm is imported lazily — only when a request actually routes to cloud.
    """

    def __init__(self, cloud_model: str, threshold: int = 20000):
        """
        Args:
            cloud_model: litellm model string (e.g. "anthropic/claude-sonnet-4-5-20250929")
            threshold: Route to cloud when new_tokens exceeds this value
        """
        self.cloud_model = cloud_model
        self.threshold = threshold
        self._litellm = None

    def _get_litellm(self):
        """Lazy import of litellm."""
        if self._litellm is None:
            import litellm

            self._litellm = litellm
        return self._litellm

    def should_route_to_cloud(self, new_tokens: int) -> bool:
        """Return True if new_tokens exceeds the cloud routing threshold."""
        return new_tokens > self.threshold

    async def completion(
        self,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> dict:
        """
        Non-streaming cloud completion. Returns OpenAI-format response dict.

        Args:
            messages: OpenAI chat format messages
            **kwargs: temperature, max_tokens, top_p, tools, etc.
        """
        litellm = self._get_litellm()

        call_kwargs = self._build_call_kwargs(messages, stream=False, **kwargs)

        start = time.perf_counter()
        response = await litellm.acompletion(**call_kwargs)
        elapsed = time.perf_counter() - start

        logger.info(
            f"[CLOUD] Non-streaming completion from {self.cloud_model} "
            f"in {elapsed:.2f}s"
        )

        # litellm returns a ModelResponse — convert to dict
        return response.model_dump()

    async def stream_completion(
        self,
        messages: list[dict[str, Any]],
        model_name: str = "cloud-model",
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Streaming cloud completion. Yields SSE-formatted chunks.

        Args:
            messages: OpenAI chat format messages
            model_name: Model name to use in SSE chunk IDs
            **kwargs: temperature, max_tokens, top_p, tools, etc.
        """
        litellm = self._get_litellm()

        call_kwargs = self._build_call_kwargs(messages, stream=True, **kwargs)

        start = time.perf_counter()
        response = await litellm.acompletion(**call_kwargs)

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        token_count = 0

        async for chunk in response:
            # Extract the delta from litellm's streaming response
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            # Build SSE chunk in OpenAI format
            sse_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            }

            # Populate delta fields
            if hasattr(delta, "role") and delta.role:
                sse_chunk["choices"][0]["delta"]["role"] = delta.role
            if hasattr(delta, "content") and delta.content:
                sse_chunk["choices"][0]["delta"]["content"] = delta.content
                token_count += 1
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                sse_chunk["choices"][0]["delta"]["tool_calls"] = [
                    tc.model_dump() for tc in delta.tool_calls
                ]

            yield f"data: {json.dumps(sse_chunk)}\n\n"

        elapsed = time.perf_counter() - start
        logger.info(
            f"[CLOUD] Streamed {token_count} tokens from {self.cloud_model} "
            f"in {elapsed:.2f}s"
        )

        yield "data: [DONE]\n\n"

    def _build_call_kwargs(
        self,
        messages: list[dict[str, Any]],
        stream: bool,
        **kwargs,
    ) -> dict:
        """Build kwargs dict for litellm.acompletion."""
        call_kwargs: dict[str, Any] = {
            "model": self.cloud_model,
            "messages": messages,
            "stream": stream,
            "drop_params": True,  # Let litellm drop unsupported params per provider
        }

        # Pass through standard OpenAI params (litellm handles provider compatibility)
        for key in (
            "temperature", "max_tokens", "top_p", "stop",
            "frequency_penalty", "presence_penalty",
            "tools", "tool_choice",
        ):
            if key in kwargs and kwargs[key] is not None:
                call_kwargs[key] = kwargs[key]

        return call_kwargs
