# SPDX-License-Identifier: Apache-2.0
"""
MLX Platform implementation for vLLM.

This module provides the MLXPlatform class that integrates Apple's MLX
framework with vLLM's platform system, enabling native Apple Silicon
GPU acceleration.
"""

import logging
import platform
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _get_apple_chip_name() -> str:
    """Get the name of the Apple Silicon chip."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "Apple Silicon"


def _get_unified_memory_size() -> int:
    """Get the total unified memory size in bytes."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip())
    except Exception:
        # Fallback: return 8GB
        return 8 * 1024 * 1024 * 1024


def _is_mlx_available() -> bool:
    """Check if MLX is available and working."""
    try:
        import mlx.core as mx

        # Verify we can actually use MLX
        _ = mx.array([1.0, 2.0, 3.0])
        return True
    except Exception as e:
        logger.debug("MLX not available: %s", e)
        return False


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


class MLXPlatform:
    """
    Platform implementation for Apple Silicon using MLX.

    This platform uses Apple's MLX framework for GPU-accelerated
    inference on Apple Silicon Macs. It integrates with mlx-lm for
    LLM inference and mlx-vlm for vision-language models.

    Key features:
    - Unified memory model (no CPU<->GPU transfers)
    - Native Metal GPU acceleration
    - Optimized kernels for Apple Silicon
    - Support for quantized models (4-bit, 8-bit)
    """

    # Platform identification
    # Using OOT (Out-of-Tree) since MLX is not a built-in vLLM platform
    # Import here to avoid circular imports at module level
    @property
    def _enum(self):
        from vllm.platforms.interface import PlatformEnum

        return PlatformEnum.OOT

    device_name: str = "mlx"
    device_type: str = "mlx"

    # MLX uses CPU dispatch key since it's not registered in PyTorch
    dispatch_key: str = "CPU"

    # Ray device key (empty = not supported)
    ray_device_key: str = ""

    # No device visibility control like CUDA_VISIBLE_DEVICES
    device_control_env_var: str = "MLX_VISIBLE_DEVICES"

    # Use eager backend for torch.compile (inductor not supported on MLX)
    simple_compile_backend: str = "eager"

    # Distributed backend (gloo works on Mac)
    dist_backend: str = "gloo"

    # Supported quantization methods
    supported_quantization: list[str] = ["mlx-4bit", "mlx-8bit"]

    # Additional environment variables
    additional_env_vars: list[str] = []

    _global_graph_pool: Any | None = None

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        """Return supported dtypes for MLX."""
        # MLX supports bfloat16, float16, and float32
        # bfloat16 may not be available on all Apple Silicon chips
        try:
            import mlx.core as mx

            # Check if bfloat16 is supported
            test = mx.array([1.0], dtype=mx.bfloat16)
            return [torch.bfloat16, torch.float16, torch.float32]
        except Exception:
            return [torch.float16, torch.float32]

    def is_cuda(self) -> bool:
        return False

    def is_rocm(self) -> bool:
        return False

    def is_tpu(self) -> bool:
        return False

    def is_xpu(self) -> bool:
        return False

    def is_cpu(self) -> bool:
        return False

    def is_mlx(self) -> bool:
        return True

    def is_out_of_tree(self) -> bool:
        return True

    def is_cuda_alike(self) -> bool:
        return False

    def is_sleep_mode_available(self) -> bool:
        return False

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the Apple Silicon chip name."""
        return _get_apple_chip_name()

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        """Get device UUID (not applicable for MLX)."""
        return "mlx-0"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get total unified memory in bytes."""
        return _get_unified_memory_size()

    @classmethod
    def inference_mode(cls):
        """Return inference mode context manager."""
        # MLX doesn't need a special inference mode
        # Return torch.no_grad() for compatibility
        return torch.no_grad()

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """Set the device (no-op for MLX, uses default device)."""
        # MLX automatically uses the GPU
        pass

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        """Set random seeds for reproducibility."""
        import random
        import numpy as np

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Set MLX seed
            try:
                import mlx.core as mx

                mx.random.seed(seed)
            except Exception:
                pass

    @classmethod
    def import_kernels(cls) -> None:
        """Import MLX kernels (no custom C kernels)."""
        # MLX uses its own Metal kernels, no vllm._C needed
        pass

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype,
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        attn_type: str | None = None,
    ) -> str:
        """Return MLX attention backend class path."""
        # Use our custom MLX attention backend
        return "vllm_mlx.attention.MLXAttentionBackend"

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Check and update vLLM configuration for MLX."""
        logger.info("Configuring vLLM for MLX backend on Apple Silicon")

        # Get chip info
        chip_name = _get_apple_chip_name()
        memory_gb = _get_unified_memory_size() / (1024**3)
        logger.info(f"Detected: {chip_name} with {memory_gb:.1f}GB unified memory")

        # Disable CUDA-specific features
        if hasattr(vllm_config, "compilation_config"):
            # Disable CUDA graphs
            vllm_config.compilation_config.cudagraph_capture_sizes = []

        # Set worker class to MLX worker
        if hasattr(vllm_config, "parallel_config"):
            parallel_config = vllm_config.parallel_config
            if parallel_config.worker_cls == "auto":
                parallel_config.worker_cls = "vllm_mlx.worker.MLXWorker"

            # Disable features not supported on MLX
            if parallel_config.enable_dbo:
                logger.warning("Dual-Batch Overlap not supported on MLX, disabling")
                parallel_config.enable_dbo = False

        # Configure cache
        if hasattr(vllm_config, "cache_config"):
            cache_config = vllm_config.cache_config
            if cache_config.block_size is None:
                cache_config.block_size = 16  # Default for MLX

    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        """Verify model architecture is supported on MLX."""
        # mlx-lm supports most transformer architectures
        # Log a warning for potentially unsupported models
        unsupported_hints = ["mamba", "rwkv", "retnet"]
        for hint in unsupported_hints:
            if hint.lower() in model_arch.lower():
                logger.warning(
                    f"Model architecture {model_arch} may not be fully "
                    f"supported on MLX. Please verify with mlx-lm."
                )
                break

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        """Verify quantization method is supported."""
        supported = ["mlx-4bit", "mlx-8bit", None, ""]
        if quant and quant not in supported:
            raise ValueError(
                f"Quantization '{quant}' not supported on MLX. "
                f"Supported: {supported}"
            )

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Pin memory not needed with unified memory."""
        return False

    @classmethod
    def get_current_memory_usage(cls, device=None) -> float:
        """Get current memory usage in bytes."""
        try:

            # MLX doesn't have a direct memory query
            # Use system memory info instead
            import psutil

            process = psutil.Process()
            return float(process.memory_info().rss)
        except Exception:
            return 0.0

    @classmethod
    def supports_fp8(cls) -> bool:
        """FP8 not supported on MLX."""
        return False

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        """Custom allreduce not available."""
        return False

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        """Static graph mode (CUDA graphs) not supported."""
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """Return the communicator class for distributed."""
        return "vllm_mlx.distributed.MLXCommunicator"

    @classmethod
    def get_punica_wrapper(cls) -> str:
        """Return LoRA wrapper (not yet implemented for MLX)."""
        raise NotImplementedError("LoRA not yet supported on MLX backend")

    def __repr__(self) -> str:
        return f"<MLXPlatform device={self.device_name}>"
