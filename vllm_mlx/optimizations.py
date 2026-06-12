# SPDX-License-Identifier: Apache-2.0
"""
Hardware detection and system information for rapid-mlx.

This module provides:
- Hardware detection for Apple Silicon (M1, M2, M3, M4 series)
- System memory detection
- Memory bandwidth benchmarking

Note: mlx-lm already includes optimized implementations internally:
- Flash Attention via mx.fast.scaled_dot_product_attention
- Efficient memory management
- Optimized Metal kernels

No additional optimization is needed - mlx-lm is already fast out of the box.

Usage:
    from vllm_mlx.optimizations import (
        detect_hardware,
        get_optimization_status,
        benchmark_memory_bandwidth,
    )
"""

import logging
from dataclasses import dataclass

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Hardware information for Apple Silicon."""

    chip_name: str
    total_memory_gb: float
    memory_bandwidth_gbs: float  # GB/s
    gpu_cores: int


# Hardware profiles for Apple Silicon chips
HARDWARE_PROFILES = {
    # M1 Series
    "M1": {"bandwidth": 68.25, "gpu_cores": 8},
    "M1 Pro": {"bandwidth": 200, "gpu_cores": 16},
    "M1 Max": {"bandwidth": 400, "gpu_cores": 32},
    "M1 Ultra": {"bandwidth": 800, "gpu_cores": 64},
    # M2 Series
    "M2": {"bandwidth": 100, "gpu_cores": 10},
    "M2 Pro": {"bandwidth": 200, "gpu_cores": 19},
    "M2 Max": {"bandwidth": 400, "gpu_cores": 38},
    "M2 Ultra": {"bandwidth": 800, "gpu_cores": 76},
    # M3 Series
    "M3": {"bandwidth": 100, "gpu_cores": 10},
    "M3 Pro": {"bandwidth": 150, "gpu_cores": 18},
    "M3 Max": {"bandwidth": 400, "gpu_cores": 40},
    "M3 Ultra": {"bandwidth": 800, "gpu_cores": 80},
    # M4 Series
    "M4": {"bandwidth": 120, "gpu_cores": 10},
    "M4 Pro": {"bandwidth": 273, "gpu_cores": 20},
    "M4 Max": {"bandwidth": 546, "gpu_cores": 40},
    "M4 Ultra": {"bandwidth": 800, "gpu_cores": 80},
}


def get_system_memory_gb() -> float:
    """
    Get actual system memory in GB.

    Returns:
        Total system memory in GB (unified memory on Apple Silicon)
    """
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
        )
        mem_bytes = int(result.stdout.strip())
        return mem_bytes / (1024**3)
    except Exception:
        # Fallback: try to get from MLX device info
        try:
            device_info = mx.device_info()
            if "memory_size" in device_info:
                return device_info["memory_size"] / (1024**3)
        except Exception:
            pass
        return 16.0  # Conservative default


def detect_hardware() -> HardwareInfo:
    """
    Detect Apple Silicon hardware and return info.

    Memory is detected dynamically from the system.
    Other specs (bandwidth, GPU cores) come from known chip profiles.

    Returns:
        HardwareInfo with detected hardware specifications
    """
    try:
        device_info = mx.device_info()
        device_name = device_info.get("device_name", "")
        actual_memory_gb = get_system_memory_gb()

        # Match with known profiles (check longest names first)
        sorted_profiles = sorted(
            HARDWARE_PROFILES.items(), key=lambda x: len(x[0]), reverse=True
        )

        for chip_name, profile in sorted_profiles:
            if chip_name in device_name:
                return HardwareInfo(
                    chip_name=chip_name,
                    total_memory_gb=actual_memory_gb,
                    memory_bandwidth_gbs=profile["bandwidth"],
                    gpu_cores=profile["gpu_cores"],
                )

        # Unknown chip
        return HardwareInfo(
            chip_name="Unknown",
            total_memory_gb=actual_memory_gb,
            memory_bandwidth_gbs=200,
            gpu_cores=16,
        )

    except Exception as e:
        logger.warning(f"Failed to detect hardware: {e}")
        return HardwareInfo(
            chip_name="Unknown",
            total_memory_gb=get_system_memory_gb(),
            memory_bandwidth_gbs=200,
            gpu_cores=16,
        )


def benchmark_memory_bandwidth() -> dict:
    """
    Benchmark actual memory bandwidth achieved.

    Returns:
        dict with bandwidth measurements for different array sizes
    """
    import time

    sizes_mb = [1, 4, 16]
    results = {}

    for size_mb in sizes_mb:
        elements = (size_mb * 1024 * 1024) // 4  # float32
        a = mx.random.normal((elements,))
        b = mx.random.normal((elements,))
        mx.eval(a, b)

        start = time.perf_counter()
        for _ in range(10):
            c = a + b
            mx.eval(c)
        elapsed = time.perf_counter() - start

        # read a, read b, write c = 3x size
        total_bytes = 3 * size_mb * 1024 * 1024 * 10
        bandwidth_gbs = (total_bytes / elapsed) / 1e9

        results[f"{size_mb}MB"] = f"{bandwidth_gbs:.1f} GB/s"

    return results


def get_optimization_status() -> dict:
    """
    Get current hardware and MLX status.

    Returns:
        dict with hardware info and MLX configuration
    """
    hw = detect_hardware()
    device_info = mx.device_info()
    flash_available = hasattr(mx, "fast") and hasattr(
        mx.fast, "scaled_dot_product_attention"
    )

    return {
        "hardware": {
            "chip": hw.chip_name,
            "total_memory_gb": hw.total_memory_gb,
            "memory_bandwidth_gbs": hw.memory_bandwidth_gbs,
            "gpu_cores": hw.gpu_cores,
            "device_name": device_info.get("device_name", "Unknown"),
        },
        "mlx_memory": {
            "active_bytes": mx.get_active_memory(),
            "cache_bytes": mx.get_cache_memory(),
            "peak_bytes": mx.get_peak_memory(),
        },
        "mlx_lm_features": {
            "flash_attention": "built-in" if flash_available else "not available",
            "metal_kernels": "optimized for Apple Silicon",
            "kv_cache": "managed by mlx-lm",
            "quantization": "4-bit and 8-bit supported",
        },
    }
