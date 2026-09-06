# SPDX-License-Identifier: Apache-2.0
"""
MLX Model Runner for vLLM.

This module implements the model runner that bridges vLLM's request
handling with mlx-lm's inference capabilities.

Includes low-level optimizations:
- Memory bandwidth optimization
- Prefill chunking for L2 cache efficiency
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from vllm_mlx.patches.deepseek_v32_indexer_gate import (
    install_deepseek_v32_indexer_gate as _install_dsv32_indexer_gate,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput

logger = logging.getLogger(__name__)

# Install the per-layer Indexer gate so REAP-pruned DeepseekV32 configs
# (e.g. mlx-community/pipenetwork-GLM-5.2-REAP50-MLX-4bit) load via mlx_lm.
# Idempotent + no-op on configs that don't publish ``indexer_types``.
_install_dsv32_indexer_gate()


@dataclass
class SamplerOutput:
    """Output from sampling."""

    token_ids: list[int]
    logprobs: list[dict] | None = None


@dataclass
class MLXModelRunnerOutput:
    """Output from MLX model runner, compatible with vLLM's ModelRunnerOutput."""

    # Request ID to sampled token IDs
    req_id_to_token_ids: dict[str, list[int]]

    # Request ID to logprobs (if requested)
    req_id_to_logprobs: dict[str, list[dict]] | None = None

    # Number of tokens generated
    num_tokens_generated: int = 0

    # Time taken for generation
    generation_time_s: float = 0.0


# --------------------------------------------------------------------------
# Compiled-decode-replay lane (opt-in, fail-closed to eager generate_step)
# --------------------------------------------------------------------------
#
# mlx-lm's ``generate_step`` can trace the width-1 decode step once with
# ``mx.compile`` and replay it instead of rebuilding ~250 kernels per token
# (``mlx_lm.compiled_decode``). The gain is exact (bit-identical to eager on
# the qualified class-1 bucket ladder) and single-digit percent, because
# decode is near the memory-bandwidth floor. This lane is OFF by default; when
# on, it declines cleanly to eager for anything it does not cover (a quantized
# or rotating cache, batched/speculative/PLD requests, an out-of-policy
# context, or a build whose mlx-lm lacks the module). It is a plain-decode
# accelerator only.
_COMPILED_DECODE_ENV = "VLLM_MLX_COMPILED_DECODE"
_COMPILED_QUALIFICATION_ENV = "MLX_LM_COMPILED_DECODE_QUALIFICATION"


def _env_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() not in ("", "0", "false", "no", "off")


def _compiled_decode_available() -> tuple[bool, str | None]:
    """Whether the loaded mlx-lm exposes the compiled-decode surface.

    A build without the module (or with an older module missing the
    request-private plumbing) declines the lane; the request still serves on
    the eager path.
    """
    try:
        from mlx_lm.compiled_decode import (  # noqa: F401
            compiled_decode_context_policy,
            model_is_compilable,
        )
        from mlx_lm.generate import generate_step  # noqa: F401
    except Exception as exc:  # ImportError, or a partial/old module
        return False, f"mlx-lm build has no compiled-decode surface ({exc})"
    # The request-private prompt-cache argument is required to engage compiled
    # replay without leaking RingKVCache's class into a shared cache. Older
    # builds route the same call through the eager path, which is still correct.
    import inspect

    try:
        params = inspect.signature(generate_step).parameters
    except (TypeError, ValueError):
        return False, "mlx-lm generate_step signature is not introspectable"
    if "compiled_decode" not in params or "_prompt_cache_is_request_private" not in params:
        return False, "mlx-lm generate_step predates request-private compiled replay"
    return True, None


class MLXModelRunner:
    """
    Model runner that uses mlx-lm for inference.

    This class handles:
    - Model loading via mlx-lm
    - Converting vLLM requests to mlx-lm format
    - Running inference and returning results in vLLM format
    - KV cache management (delegated to mlx-lm)

    Optimizations:
    - Memory optimization for bandwidth efficiency
    - Prefill chunking for L2 cache utilization
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        enable_optimizations: bool = True,
        enable_compiled_decode: bool | None = None,
    ):
        """
        Initialize MLX model runner.

        Args:
            vllm_config: vLLM configuration
            enable_optimizations: Whether to enable low-level optimizations
            enable_compiled_decode: Opt into the compiled-decode-replay
                plain-decode lane. ``None`` reads ``VLLM_MLX_COMPILED_DECODE``
                (the ``serve`` CLI sets it from ``--compiled-decode``); the
                default is off. Enabling it never forces compiled replay: the
                lane still declines per request to the eager path for anything
                it does not cover, and refuses to serve compiled without a
                bound reviewed-qualification manifest.
        """
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config

        # mlx-lm model and tokenizer
        self.model = None
        self.tokenizer = None
        self._loaded = False

        # Sampler for generation
        self._sampler = None

        # Cache for prompt processing
        self._prompt_cache = None

        # Compiled-decode-replay lane (opt-in, fail-closed to eager). When
        # enabled, plain width-1 decode is routed through mlx-lm's compiled
        # replay path with a request-private cache.
        if enable_compiled_decode is None:
            enable_compiled_decode = _env_truthy(os.environ.get(_COMPILED_DECODE_ENV))
        self._enable_compiled_decode = bool(enable_compiled_decode)
        # Single-entry APC (publish-back on hit): the stock-cache form of the
        # last compiled turn's KV, plus the exact token ids it covers, so a
        # later turn that extends the same prefix warm-restores instead of
        # re-prefilling. The RingKVCache the compiled step owns is
        # request-private and is NEVER stored here (mirrors the mlx-lm server's
        # publish-back discipline: the APC only ever holds a stock KVCache).
        self._compiled_apc_tokens: tuple[int, ...] | None = None
        self._compiled_apc_cache: list | None = None

        # KV cache blocks
        self._num_cache_blocks = 0

        # Optimization settings
        self._enable_optimizations = enable_optimizations
        self._hardware_info = None  # Detected hardware profile
        # Set True only after ``_apply_optimizations`` completes without
        # raising. ``_hardware_info`` being populated is NOT proof of
        # success: ``configure_memory_optimization()`` can raise AFTER the
        # hardware probe, and that exception is caught + logged. Derive the
        # ``optimized`` status from this flag, not from ``_hardware_info``.
        self._optimizations_applied = False

        logger.info(f"MLXModelRunner initialized for model: {self.model_config.model}")
        logger.info(
            f"Low-level optimizations: {'ENABLED' if enable_optimizations else 'disabled'}"
        )
        logger.info(
            "Compiled-decode-replay lane: "
            f"{'ENABLED (opt-in)' if self._enable_compiled_decode else 'disabled'}"
        )

    def load_model(self) -> None:
        """Load model using mlx-lm with optimizations."""
        if self._loaded:
            return

        try:
            from mlx_lm import load

            model_name = self.model_config.model

            logger.info(f"Loading model with mlx-lm: {model_name}")
            start_time = time.time()

            self.model, self.tokenizer = load(
                model_name,
                tokenizer_config={
                    "trust_remote_code": self.model_config.trust_remote_code,
                },
            )

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s")

            self._loaded = True

            # Create default sampler
            self._create_default_sampler()

            # Apply low-level optimizations
            if self._enable_optimizations:
                self._apply_optimizations()

        except ImportError:
            raise ImportError(
                "mlx-lm is required for MLX model runner. "
                "Install with: pip install mlx-lm"
            )

    def _apply_optimizations(self) -> None:
        """Apply low-level optimizations for maximum performance."""
        # Reset at entry so a failed RE-attempt after a prior success does
        # not leave the runner falsely reporting ``optimized`` (codex
        # #1112 [NIT] round 8). The flag is re-set True only if this
        # attempt completes.
        self._optimizations_applied = False
        try:
            from vllm_mlx.optimizations import (
                configure_memory_optimization,
                detect_hardware,
            )

            # Detect hardware and apply memory optimization
            self._hardware_info = detect_hardware()
            logger.info(f"Hardware detected: {self._hardware_info.chip_name}")
            logger.info(f"Memory: {self._hardware_info.total_memory_gb:.1f} GB")
            logger.info(f"Bandwidth: {self._hardware_info.memory_bandwidth_gbs} GB/s")

            # Configure memory settings
            configure_memory_optimization()

            # Reached only if every step above succeeded.
            self._optimizations_applied = True

        except Exception as e:
            logger.warning(f"Failed to apply optimizations: {e}")

    def _create_default_sampler(self) -> None:
        """Create default sampler for generation."""
        try:
            from mlx_lm.sample_utils import make_sampler

            self._sampler = make_sampler(
                temp=0.7,
                top_p=0.9,
            )
        except ImportError:
            logger.warning("Could not create sampler, using defaults")

    def initialize_cache(self, num_blocks: int) -> None:
        """Initialize KV cache."""
        self._num_cache_blocks = num_blocks
        logger.info(f"KV cache initialized with {num_blocks} blocks")

        # mlx-lm manages its own KV cache internally
        # We just track the configuration here

    def get_kv_cache_spec(self) -> dict:
        """Get KV cache specification."""
        return {
            "num_blocks": self._num_cache_blocks,
            "block_size": self.cache_config.block_size,
        }

    def get_cache_block_size_bytes(self) -> int:
        """Calculate cache block size in bytes."""
        if not self._loaded or self.model is None:
            return 0

        # Get model config
        config = getattr(self.model, "config", None)
        if config is None:
            return 0

        head_size = getattr(config, "head_dim", 64)
        num_kv_heads = getattr(
            config, "num_key_value_heads", getattr(config, "num_attention_heads", 32)
        )
        num_layers = getattr(config, "num_hidden_layers", 32)
        block_size = self.cache_config.block_size

        # 2 for K and V, 2 bytes for float16
        return 2 * block_size * num_layers * num_kv_heads * head_size * 2

    def warm_up(self) -> None:
        """Warm up model with a test generation."""
        if not self._loaded:
            self.load_model()

        logger.info("Warming up model...")

        try:
            from mlx_lm import generate

            # Simple warm-up generation
            _ = generate(
                self.model,
                self.tokenizer,
                prompt="Hello",
                max_tokens=5,
                verbose=False,
            )
            logger.info("Model warm-up complete")

        except Exception as e:
            logger.warning(f"Warm-up failed (non-critical): {e}")

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> MLXModelRunnerOutput:
        """
        Execute model inference for scheduled requests.

        Args:
            scheduler_output: Contains requests to process

        Returns:
            MLXModelRunnerOutput with generated tokens
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()
        req_id_to_token_ids: dict[str, list[int]] = {}
        total_tokens = 0

        # Process new requests
        for req_data in scheduler_output.scheduled_new_reqs:
            req_id = req_data.req_id
            prompt_token_ids = req_data.prompt_token_ids

            # Generate tokens for this request
            generated_ids = self._generate_for_request(
                prompt_token_ids=prompt_token_ids,
                sampling_params=req_data.sampling_params,
                max_tokens=1,  # Generate one token at a time for streaming
            )

            req_id_to_token_ids[req_id] = generated_ids
            total_tokens += len(generated_ids)

        # Process running requests (continue generation)
        for req_id in scheduler_output.scheduled_running_reqs:
            # For running requests, we continue generation
            # This is simplified - in practice we'd use KV cache
            generated_ids = self._continue_generation(req_id)
            if generated_ids:
                req_id_to_token_ids[req_id] = generated_ids
                total_tokens += len(generated_ids)

        generation_time = time.time() - start_time

        return MLXModelRunnerOutput(
            req_id_to_token_ids=req_id_to_token_ids,
            num_tokens_generated=total_tokens,
            generation_time_s=generation_time,
        )

    def _prefill_with_chunking(
        self,
        input_ids: mx.array,
        cache: Any | None = None,
    ) -> tuple[mx.array, Any]:
        """
        Process prompt with optimal chunking for L2 cache efficiency.

        Long prompts are broken into chunks that fit in L2 cache,
        maximizing memory bandwidth utilization during prefill.

        Args:
            input_ids: Input token IDs [1, seq_len]
            cache: Optional existing KV cache

        Returns:
            Tuple of (logits, updated_cache)
        """
        try:
            from vllm_mlx.optimizations import get_optimal_prefill_size
        except ImportError:
            # Fallback if optimizations module not available
            def get_optimal_prefill_size(seq_len):
                return min(512, seq_len)

        seq_len = input_ids.shape[-1] if len(input_ids.shape) > 1 else len(input_ids)
        chunk_size = get_optimal_prefill_size(seq_len)

        # Reshape if needed
        if len(input_ids.shape) == 1:
            input_ids = input_ids.reshape(1, -1)

        forward_fn = self.model

        if seq_len <= chunk_size:
            # Process entire sequence at once
            return forward_fn(input_ids, cache=cache)

        # Process in chunks for large prompts
        for i in range(0, seq_len, chunk_size):
            chunk = input_ids[:, i : i + chunk_size]
            logits, cache = forward_fn(chunk, cache=cache)
            mx.eval(cache)  # Force evaluation to free intermediate memory

        return logits, cache

    def _compiled_decode_decline_reason(
        self,
        prompt_token_ids: list[int],
        sampling_params: Any,
        max_tokens: int,
    ) -> str | None:
        """Why this request may NOT use the compiled-replay lane, or ``None``.

        Returns a reason string for anything the lane does not cover; the
        caller then serves the request on the eager path. This gate is
        deliberately conservative and mirrors ``generate_step``'s own
        preconditions so a declined request never allocates a private cache
        it will not use.
        """
        if not self._enable_compiled_decode:
            return "compiled-decode lane disabled"
        available, why = _compiled_decode_available()
        if not available:
            return why
        # The lane must NOT serve compiled without an operator-approved,
        # loader-bound checkpoint manifest. Family eligibility alone is only
        # for direct research use of CompiledDecodeStep. mlx-lm binds a
        # manifest either from a file (MLX_LM_COMPILED_DECODE_QUALIFICATION,
        # set by --compiled-decode-qualification) or from an in-process
        # registered serving record; honor both, but hard-decline when neither
        # is present. generate_step remains the authoritative gate — it
        # verifies the bound manifest against the loaded weights and declines
        # to eager if it does not match, so this check never hashes weights.
        manifest = os.environ.get(_COMPILED_QUALIFICATION_ENV, "").strip()
        if not manifest:
            try:
                from mlx_lm import compiled_qualification as _cq

                registered = bool(getattr(_cq, "SERVING_QUALIFICATIONS", {}))
            except Exception:
                registered = False
            if not registered:
                return (
                    "no reviewed qualification manifest bound "
                    f"({_COMPILED_QUALIFICATION_ENV} unset and no registered "
                    "serving record; pass --compiled-decode-qualification)"
                )
        if max_tokens is not None and 0 <= max_tokens < 1:
            return "no decode step requested (max_tokens < 1)"
        # Width-1, batch-1, non-speculative only.
        if getattr(sampling_params, "n", 1) not in (None, 1):
            return "n>1 (compiled replay is width 1, batch 1)"
        if (getattr(sampling_params, "best_of", 1) or 1) > 1:
            return "best_of>1 (compiled replay is width 1, batch 1)"
        if getattr(self.vllm_config, "speculative_config", None) is not None:
            return "speculative decoding is configured (not shape-stable)"
        if _env_truthy(os.environ.get("VLLM_MLX_PROMPT_LOOKUP")):
            return "prompt-lookup (PLD) is enabled (not shape-stable)"
        # A quantized / bounded KV cache is not shape-stable. This runner never
        # passes kv_bits/max_kv_size to generate_step, but decline defensively
        # if a caller ever wires them onto the sampling params.
        if getattr(sampling_params, "kv_bits", None) is not None:
            return "kv_bits (a quantized cache is not shape-stable)"
        if getattr(sampling_params, "max_kv_size", None) is not None:
            return "max_kv_size (rotating caches are not shape-stable)"
        # Context must be within the configured compiled context policy. The
        # env default is ``short`` (4096); generate_step re-checks this too.
        try:
            from mlx_lm.compiled_decode import compiled_decode_context_policy

            budget = 0 if max_tokens is None or max_tokens < 0 else max_tokens
            why, _policy = compiled_decode_context_policy(
                len(prompt_token_ids), budget
            )
        except Exception as exc:
            return f"context policy check failed ({exc})"
        if why is not None:
            return why
        return None

    def _plan_compiled_decode(
        self,
        prompt_token_ids: list[int],
        sampling_params: Any,
        max_tokens: int,
    ) -> dict | None:
        """Build the compiled-lane plan (cache + tokens to feed), or ``None``.

        On a warm-restore hit the stored stock cache from the previous compiled
        turn is reused and only the new suffix is prefilled; otherwise a fresh
        request-private cache is created. Returning ``None`` means serve eager.
        """
        reason = self._compiled_decode_decline_reason(
            prompt_token_ids, sampling_params, max_tokens
        )
        if reason is not None:
            logger.debug("compiled decode declined: %s", reason)
            return None

        from mlx_lm.models.cache import make_prompt_cache

        tokens = tuple(prompt_token_ids)
        # Warm restore: the previous compiled turn's cache is reusable when its
        # covered tokens are a strict prefix of this request and at least one
        # new token remains to prefill. The stored form is a stock KVCache list
        # (never a RingKVCache), which generate_step re-converts in place.
        stored_tokens = self._compiled_apc_tokens
        stored_cache = self._compiled_apc_cache
        if (
            stored_cache is not None
            and stored_tokens is not None
            and 0 < len(stored_tokens) < len(tokens)
            and tokens[: len(stored_tokens)] == stored_tokens
        ):
            prefix_len = len(stored_tokens)
            # Hand the stored list to generate_step, which mutates it in place.
            # Clear our reference so a mid-flight failure cannot leave a stale,
            # half-converted cache visible to the next turn; we re-publish on
            # a clean finish.
            self._compiled_apc_tokens = None
            self._compiled_apc_cache = None
            return {
                "cache": stored_cache,
                "feed": list(tokens[prefix_len:]),
                "hit": True,
                "prefix_len": prefix_len,
            }
        return {
            "cache": make_prompt_cache(self.model),
            "feed": list(tokens),
            "hit": False,
            "prefix_len": 0,
        }

    def _record_compiled_result(
        self,
        plan: dict,
        prompt_token_ids: list[int],
        generated_ids: list[int],
        status: dict | None,
    ) -> None:
        """Publish the finished turn's cache back for later warm restore.

        Mirrors the mlx-lm server's publish-back discipline: convert any
        RingKVCache to its stock KVCache form (the APC never holds a
        RingKVCache) and record the exact tokens it covers. A turn that
        declined compiled still leaves a valid stock cache, so it is published
        too, preserving prefix reuse across a policy-driven decline.
        """
        used = bool(status and status.get("used"))
        if status is not None and not used and status.get("decline_reason"):
            logger.debug(
                "compiled decode declined at prefill: %s",
                status.get("decline_reason"),
            )
        try:
            from mlx_lm.models.cache import RingKVCache

            cache = plan["cache"]
            published = [
                c.to_kv_cache() if isinstance(c, RingKVCache) else c for c in cache
            ]
            if any(isinstance(c, RingKVCache) for c in published):
                raise RuntimeError("a RingKVCache must never be published")
            self._compiled_apc_cache = published
            self._compiled_apc_tokens = tuple(prompt_token_ids) + tuple(generated_ids)
        except Exception as exc:
            # Never let publish-back break generation; just skip the warm-restore
            # opportunity for the next turn.
            logger.debug("compiled decode publish-back skipped: %s", exc)
            self._compiled_apc_cache = None
            self._compiled_apc_tokens = None

    def _generate_for_request(
        self,
        prompt_token_ids: list[int],
        sampling_params: Any,
        max_tokens: int = 1,
    ) -> list[int]:
        """
        Generate tokens for a single request.

        Uses optimizations when enabled:
        - Prefill chunking for long prompts
        - Compiled-decode-replay lane for plain width-1 decode (opt-in)

        Args:
            prompt_token_ids: Input token IDs
            sampling_params: Sampling parameters
            max_tokens: Maximum tokens to generate

        Returns:
            List of generated token IDs
        """
        try:
            # Install MLX hardware-compat shim (#404 M5 single-stream guard)
            # BEFORE importing mlx_lm.generate — that module captures
            # `mx.new_thread_local_stream(mx.default_device())` at top level.
            # Idempotent: no-op once installed.
            from vllm_mlx import _mlx_compat as _mlx_compat

            _mlx_compat.install()

            from mlx_lm.generate import generate_step
            from mlx_lm.sample_utils import make_sampler

            # Create sampler from sampling params
            temp = getattr(sampling_params, "temperature", 0.7)
            top_p = getattr(sampling_params, "top_p", 0.9)
            sampler = make_sampler(temp=temp, top_p=top_p)

            # Decide whether this request may use the compiled-replay lane.
            plan = self._plan_compiled_decode(
                prompt_token_ids, sampling_params, max_tokens
            )

            gen_kwargs: dict[str, Any] = dict(
                prompt=mx.array(prompt_token_ids if plan is None else plan["feed"]),
                model=self.model,
                max_tokens=max_tokens,
                sampler=sampler,
            )
            status: dict | None = None
            if plan is not None:
                # A request-private cache lets generate_step convert to a
                # RingKVCache and replay a traced step without leaking that
                # numerical/performance class into any shared cache. Any
                # decline inside generate_step falls back to the eager step
                # for this same request — the lane is fail-closed by
                # construction.
                status = {}
                gen_kwargs.update(
                    prompt_cache=plan["cache"],
                    _prompt_cache_is_request_private=True,
                    compiled_decode=True,
                    _compiled_decode_status=status,
                )

            generated_ids = []

            # Generate tokens
            for token_info in generate_step(**gen_kwargs):
                if hasattr(token_info, "token"):
                    generated_ids.append(token_info.token)
                elif isinstance(token_info, tuple) and len(token_info) > 0:
                    generated_ids.append(token_info[0])

                if len(generated_ids) >= max_tokens:
                    break

            if plan is not None:
                self._record_compiled_result(
                    plan, prompt_token_ids, generated_ids, status
                )

            return generated_ids

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # A compiled attempt that raised may have left a poisoned/partial
            # cache; drop the warm-restore entry so the next turn re-prefills.
            self._compiled_apc_cache = None
            self._compiled_apc_tokens = None
            return []

    def _continue_generation(self, req_id: str) -> list[int]:
        """
        Continue generation for an existing request.

        This is a placeholder - in a full implementation, we would
        use cached KV states to continue generation efficiently.
        """
        # For now, return empty - full implementation would track state
        return []

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        if self.tokenizer is None:
            return ""
        return self.tokenizer.decode(token_ids)

    def get_model_info(self) -> dict:
        """Get information about the loaded model and optimizations."""
        info = {
            "loaded": self._loaded,
            "model_name": self.model_config.model,
            "optimizations_enabled": self._enable_optimizations,
        }

        if self._loaded and self.model is not None:
            config = getattr(self.model, "config", None)
            if config:
                info.update(
                    {
                        "vocab_size": getattr(config, "vocab_size", None),
                        "hidden_size": getattr(config, "hidden_size", None),
                        "num_layers": getattr(config, "num_hidden_layers", None),
                        "num_heads": getattr(config, "num_attention_heads", None),
                    }
                )

            # Add optimization status. ``kernel_fusion`` is retained as a
            # permanently-``False`` compatibility key: the always-on
            # ``mx.compile`` forward-pass fusion it reported was rejected
            # (A5 — bucketed compile regressed batch=1 decode; the win was
            # already captured by ``mx.async_eval``) and its dead code was
            # removed, but any external caller that indexes this key keeps
            # a stable dict shape instead of hitting a ``KeyError``.
            info["optimizations"] = {
                "kernel_fusion": False,
                "memory_optimized": self._optimizations_applied,
                "compiled_decode": self._enable_compiled_decode,
            }

            if self._hardware_info:
                info["hardware"] = {
                    "chip": self._hardware_info.chip_name,
                    "memory_gb": self._hardware_info.total_memory_gb,
                    "bandwidth_gbs": self._hardware_info.memory_bandwidth_gbs,
                    "gpu_cores": self._hardware_info.gpu_cores,
                    "prefill_chunk_size": self._hardware_info.optimal_prefill_size,
                }

        return info

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        opt_status = "optimized" if self._optimizations_applied else "standard"
        return f"<MLXModelRunner model={self.model_config.model} status={status} mode={opt_status}>"
