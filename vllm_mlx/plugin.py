# SPDX-License-Identifier: Apache-2.0
"""
vLLM Platform Plugin for MLX.

This module provides the entry point for vLLM's platform plugin system,
enabling automatic detection and activation of the MLX platform on
Apple Silicon Macs.
"""

import logging
import platform
import sys

logger = logging.getLogger(__name__)


def mlx_platform_plugin() -> str | None:
    """
    Platform plugin entry point for vLLM.

    This function is called by vLLM's platform detection system to
    determine if the MLX platform should be activated.

    Returns:
        str: Fully qualified class name of MLXPlatform if conditions are met
        None: If MLX platform should not be activated
    """
    logger.debug("Checking if MLX platform is available")

    # Check if running on macOS
    if sys.platform != "darwin":
        logger.debug("MLX platform not available: not running on macOS")
        return None

    # Check if running on Apple Silicon (ARM)
    if platform.machine() != "arm64":
        logger.debug("MLX platform not available: not running on Apple Silicon")
        return None

    # Check if MLX is installed and working
    try:
        import mlx.core as mx

        # Verify MLX can actually run
        test_array = mx.array([1.0, 2.0, 3.0])
        _ = mx.sum(test_array)

        # Check default device
        default_device = mx.default_device()
        logger.debug(f"MLX default device: {default_device}")

    except ImportError:
        logger.debug("MLX platform not available: mlx not installed")
        return None
    except Exception as e:
        logger.debug(f"MLX platform not available: {e}")
        return None

    # Check if mlx-lm is available
    try:
        import mlx_lm

        logger.debug(f"mlx-lm version: {getattr(mlx_lm, '__version__', 'unknown')}")
    except ImportError:
        logger.warning("mlx-lm not installed. Install with: pip install mlx-lm")
        # Still allow MLX platform, but functionality will be limited
        pass

    logger.info("MLX platform is available on Apple Silicon")
    return "vllm_mlx.vllm_platform.MLXPlatform"


def is_mlx_available() -> bool:
    """
    Check if MLX platform can be used.

    Returns:
        bool: True if MLX is available and working
    """
    return mlx_platform_plugin() is not None


def get_mlx_device_info() -> dict:
    """
    Get information about the MLX device.

    Returns:
        dict: Device information including chip name, memory, etc.
    """
    import subprocess

    info = {
        "platform": "mlx",
        "available": False,
        "chip_name": "Unknown",
        "memory_gb": 0,
        "mlx_version": "Unknown",
        "mlx_lm_version": "Unknown",
        "mlx_vlm_version": "Unknown",
    }

    if not is_mlx_available():
        return info

    info["available"] = True

    # Get chip name
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        info["chip_name"] = result.stdout.strip()
    except Exception:
        pass

    # Get memory
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
        )
        info["memory_gb"] = int(result.stdout.strip()) / (1024**3)
    except Exception:
        pass

    # Get MLX version
    try:
        import mlx

        info["mlx_version"] = getattr(mlx, "__version__", "Unknown")
    except Exception:
        pass

    # Get mlx-lm version
    try:
        import mlx_lm

        info["mlx_lm_version"] = getattr(mlx_lm, "__version__", "Unknown")
    except Exception:
        pass

    # Get mlx-vlm version
    try:
        import mlx_vlm

        info["mlx_vlm_version"] = getattr(mlx_vlm, "__version__", "Unknown")
    except Exception:
        pass

    return info
