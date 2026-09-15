"""DeepSeek V4 checkpoint-native DSpark speculative decoding."""

from .detect import DSparkMetadata, detect_dspark_metadata

__all__ = ["DSparkMetadata", "detect_dspark_metadata"]
