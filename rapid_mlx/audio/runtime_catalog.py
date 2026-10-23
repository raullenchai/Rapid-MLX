# SPDX-License-Identifier: Apache-2.0
"""Code-owned install targets; audio alias metadata may only select these keys.

Pipeline pins follow the publisher's spaCy compatibility table. New pipelines
or spaCy minor versions require a reviewed code change, never a catalog update.
"""

from types import MappingProxyType

SPACY_PIPELINE_PINS = MappingProxyType(
    {
        "en_core_web_sm": MappingProxyType({"3.7": "3.7.0", "3.8": "3.8.0"}),
    }
)
