"""DeepSeek-native and qualified companion DSpark runtimes."""

from .detect import DSparkMetadata, detect_dspark_metadata
from .eligibility import (
    LFM25_VL_3B,
    CompanionDSparkError,
    CompanionDSparkPair,
    resolve_companion_dspark_pair,
)

__all__ = [
    "CompanionDSparkError",
    "CompanionDSparkPair",
    "DSparkMetadata",
    "LFM25_VL_3B",
    "detect_dspark_metadata",
    "resolve_companion_dspark_pair",
]
