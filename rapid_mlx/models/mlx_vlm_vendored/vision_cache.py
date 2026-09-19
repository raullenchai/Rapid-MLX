"""Vision feature cache for multi-turn conversations.

Caches the output of vision_tower + embed_vision (projected image features
in language model space) keyed by image path or content hash, avoiding
expensive re-computation when the same image is discussed across turns.
"""

import hashlib
import os
from collections import OrderedDict
from typing import Any, Optional, Sequence, Union

import mlx.core as mx

VisionFeatures = Union[mx.array, Sequence[mx.array]]


class VisionFeatureCache:
    """LRU cache for vision features projected into language model space.

    Cache keys are derived from image paths (for file/URL images) or content
    hashes (for PIL images). Cached values are either one feature array or an
    ordered collection of per-image feature arrays after vision projection.

    Cleanup is handled by three mechanisms:
    - **LRU eviction**: oldest entry is dropped when max_size is exceeded.
    - **Model unload**: server calls clear() when the model is swapped.
    - **Process exit**: in-memory cache is freed automatically.

    Args:
        max_size: Maximum number of cached image features. Default 20.
    """

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self._cache: OrderedDict[str, VisionFeatures] = OrderedDict()

    def _make_key(self, image_source: Any) -> str:
        """Derive a cache key from an image source.

        For str/Path: use the string directly (path or URL).
        For lists: create a composite key from individual keys.
        For PIL images: hash the image bytes.
        """
        # VENDOR-DEVIATION(upstream-bugfix): upstream's key derivation was
        # ambiguous and could serve one image set's cached features to
        # another — a bare "|" join collided distinct lists (["a|b", "c"] vs
        # ["a", "b|c"]), and length-prefixing alone still collides across
        # nesting boundaries (["1:a", "b"] vs [["a"], "b"]). Every branch now
        # emits a count-delimited, type-tagged encoding (``s``=str/Path,
        # ``l``=list with a child count, ``p``=content hash), which is
        # injective. Upstream also documented Path sources but only accepted
        # ``str`` (a Path fell into the ``obj:{id}`` fallback); PathLike is
        # normalized via ``os.fsdecode`` — ``os.fspath`` alone can return
        # ``bytes`` for byte-valued paths, which then fell into the
        # image-content hash branch and collided with a raw image payload
        # equal to the path bytes; fsdecode's surrogateescape mapping is
        # injective and lands byte paths on the same key as the equivalent
        # str path.
        if isinstance(image_source, os.PathLike):
            image_source = os.fsdecode(image_source)
        if isinstance(image_source, str):
            return f"s{len(image_source)}:{image_source}"
        if isinstance(image_source, list):
            # The child count makes empty children unambiguous: bare "l"
            # tags alone would collapse ([[], []] vs [[[]]]).
            return f"l{len(image_source)}:" + "".join(map(self._make_key, image_source))
        if isinstance(image_source, (bytes, bytearray, memoryview)):
            payload = bytes(image_source)
        elif hasattr(image_source, "tobytes"):
            # VENDOR-DEVIATION(upstream-bugfix): upstream hashed only
            # ``tobytes()``, so two images with identical raw bytes but
            # different mode or size (a 2x1 "L" vs a 1x1 "RGB" of the same
            # byte string) collided and one image's features were served
            # for the other. Hash stable type/mode/size metadata together
            # with the raw bytes; bytes-like sources carry no such metadata
            # and stay content-addressed above.
            digest = hashlib.sha256()
            digest.update(
                f"{type(image_source).__module__}.{type(image_source).__name__}\x00".encode()
            )
            digest.update(f"{getattr(image_source, 'mode', '')!s}\x00".encode())
            digest.update(repr(tuple(getattr(image_source, "size", ()) or ())).encode())
            digest.update(b"\x00")
            # VENDOR-DEVIATION(upstream-bugfix): palette images ("P") hash
            # to palette INDICES in tobytes(), so two same-sized images with
            # identical indices but different palettes rendered different
            # content yet collided. Fold the effective palette into the
            # digest (getpalette() is None for non-palette modes).
            get_palette = getattr(image_source, "getpalette", None)
            if callable(get_palette):
                palette = get_palette()
                if palette is not None:
                    digest.update(b"palette\x00")
                    digest.update(repr(bytes(palette)).encode())
                    digest.update(b"\x00")
            # Palette indices plus RGB palette bytes still do not fully
            # describe rendered pixels: Pillow stores a transparent palette
            # index (or per-entry alpha table) in ``info``.  Keep its type in
            # the digest as well so integer indices and byte tables cannot
            # alias one another.
            info = getattr(image_source, "info", None)
            if isinstance(info, dict) and "transparency" in info:
                transparency = info["transparency"]
                digest.update(b"transparency\x00")
                digest.update(
                    f"{type(transparency).__module__}."
                    f"{type(transparency).__name__}\x00".encode()
                )
                if isinstance(transparency, (bytes, bytearray, memoryview)):
                    digest.update(bytes(transparency))
                else:
                    digest.update(repr(transparency).encode())
                digest.update(b"\x00")
            digest.update(image_source.tobytes())
            return f"p:{digest.hexdigest()[:16]}"
        else:
            # Upstream fell back to ``obj:{id(...)}``; Python may hand that
            # id to an unrelated object after the original is collected — a
            # silent stale-feature hit. Fail loudly instead; the branches
            # above cover every real caller (the lane passes pre-hashed
            # string keys).
            raise TypeError(
                "unsupported image source type for the vision feature "
                f"cache: {type(image_source).__name__}; pass a path/URL "
                "string, a list of them, or a bytes-like image object"
            )
        return f"p:{hashlib.sha256(payload).hexdigest()[:16]}"

    def get(self, image_source: Any) -> Optional[VisionFeatures]:
        """Look up cached features. Returns None on miss."""
        key = self._make_key(image_source)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, image_source: Any, features: VisionFeatures) -> None:
        """Store features in the cache, evicting LRU if full."""
        # VENDOR-DEVIATION(upstream-bugfix): upstream accepted any
        # ``max_size`` but ``put`` with ``max_size <= 0`` evaluated
        # ``len(self._cache) >= self.max_size`` against an empty mapping and
        # called ``popitem()`` on it, raising KeyError. Zero (or negative)
        # now disables storage instead of crashing.
        if self.max_size <= 0:
            return
        key = self._make_key(image_source)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
        self._cache[key] = features

    def clear(self) -> None:
        """Clear all cached features."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, image_source: Any) -> bool:
        key = self._make_key(image_source)
        return key in self._cache
