# SPDX-License-Identifier: Apache-2.0
"""
MLX hardware-compatibility shims.

Currently handles one upstream issue:

**M5 single-stream GPU (#404)**: `mlx_lm/generate.py` does

    generation_stream = mx.new_thread_local_stream(mx.default_device())

at module-import time. On M1–M4, this returns a usable thread-local stream.
On M5, the call appears to succeed (returning a Stream handle), but later
``with mx.stream(generation_stream):`` raises

    RuntimeError: There is no Stream(gpu, 1) in current thread.

because the M5 GPU only exposes a single stream slot. Every pure-attention
model crashes at first prompt evaluation. Hybrid models (Qwen3.5/3.6) work
because their custom path doesn't import ``mlx_lm.generate``.

Fix: monkey-patch ``mx.new_thread_local_stream`` with a probe-and-cache
wrapper. On the first call we attempt a trivial op inside ``mx.stream(s)``;
if it raises, we cache that fact and return ``mx.default_stream(device)``
for all subsequent calls. Single-stream devices then run with the default
stream, losing parallel-issue throughput but staying functional. Hardware
that supports multiple streams gets the original behavior — the probe is
one-time per device.

This patch must execute *before* any ``import mlx_lm.generate``, since
that module captures the returned stream at module level. The install
hook is called at the top of every consumer that imports
``mlx_lm.generate`` (currently ``vllm_mlx/scheduler.py``) — *not* from
``vllm_mlx/__init__.py``.
We deliberately keep ``import vllm_mlx`` free of any ``mlx.core`` import
so the package stays usable for metadata-only access on systems where
``mlx`` is installed but Metal is unavailable (``import mlx.core``
SIGABRTs there with an uncatchable NSException).

Upstream tracking: file mlx-lm bug + remove this shim when upstream lands
a device-capability check.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def install() -> None:
    """Install M5-compat shim. Safe to call multiple times (idempotent).

    No-op when mlx.core can't be imported (non-Apple-Silicon CI). Logging
    is at debug level on success to keep startup quiet for the 99% of
    users on hardware where the original API works.
    """
    try:
        import mlx.core as mx
    except ImportError:
        return  # Linux CI / no MLX

    if getattr(mx, "_rapid_mlx_compat_installed", False):
        return

    # No-op on builds that predate ``mx.new_thread_local_stream`` (#408): the
    # M5 single-stream bug only manifests when ``mlx_lm.generate`` captures
    # this symbol at module import. Older mlx never had it, so neither
    # ``mlx_lm.generate`` nor the bug it triggers can be present here. We
    # intentionally do NOT set ``_rapid_mlx_compat_installed`` here — if
    # the symbol later appears (importlib.reload, dynamic upgrade), the
    # next install() call should re-evaluate and apply the wrap.
    if not hasattr(mx, "new_thread_local_stream"):
        return

    original = mx.new_thread_local_stream

    # Tri-state cache per device:
    #   None  → not probed yet
    #   True  → original works on this device
    #   False → original is unusable, must fall back to default_stream
    _probe_cache: dict = {}

    def _probe(stream, device) -> bool:
        """True if `with mx.stream(stream)` can run a trivial op."""
        try:
            with mx.stream(stream):
                # Force evaluation so the stream is actually exercised.
                _ = (mx.array([0.0]) + mx.array([1.0])).item()
            return True
        except RuntimeError as e:
            msg = str(e)
            # Be permissive about the exact wording: upstream may change it.
            if "Stream" in msg and ("no Stream" in msg or "not exist" in msg):
                logger.warning(
                    "MLX device %s rejects secondary streams (%s) — "
                    "falling back to default_stream. "
                    "Throughput on parallel ops may be reduced. "
                    "This is the #404 M5 single-stream workaround.",
                    device,
                    msg,
                )
                return False
            raise

    def patched_new_thread_local_stream(device):
        cached = _probe_cache.get(repr(device))
        if cached is False:
            return mx.default_stream(device)
        if cached is True:
            return original(device)

        # First call for this device — probe. Log at INFO so that anyone
        # filing a hardware-shaped bug report has the device family in
        # their startup output. Future M5+/Apple chip families will land
        # here first; greppable on "rapid-mlx compat".
        stream = original(device)
        if _probe(stream, device):
            _probe_cache[repr(device)] = True
            logger.info(
                "rapid-mlx compat: device %s supports thread-local streams (no shim).",
                device,
            )
            return stream
        _probe_cache[repr(device)] = False
        logger.info(
            "rapid-mlx compat: device %s uses default_stream fallback (#404 M5 path).",
            device,
        )
        return mx.default_stream(device)

    mx.new_thread_local_stream = patched_new_thread_local_stream
    mx._rapid_mlx_compat_installed = True
    logger.debug("MLX compat shim installed (#404 M5 single-stream guard).")


def install_batch_slot_guard() -> None:
    """Stop mlx-lm writing ``None`` into a batch's ``logits_processors``.

    Separate from ``install()`` and separately idempotent: that shim must
    run BEFORE ``mlx_lm.generate`` is imported (it patches a symbol that
    module captures at import time), whereas this one patches a class
    inside it and therefore must run AFTER. Callers import
    ``mlx_lm.generate`` and then call this.

    ``PromptProcessingBatch.extend`` normalizes "this batch has no logits
    processors" to ``None`` slots::

        if not any(self.logits_processors):
            self.logits_processors = [None] * len(self.uids)
        logits_processors = (
            batch.logits_processors
            if any(batch.logits_processors)
            else [None] * len(batch.uids)
        )

    ``any()`` is a whole-list question but the slots are per-sequence, so
    merging a no-processor batch into one that HAS a processor yields
    ``[None, None, [proc]]``. From then on ``any(...)`` is True, so
    ``filter`` keeps every slot verbatim and ``GenerationBatch._step``
    reaches ``for processor in self.logits_processors[e]`` on a ``None``::

        TypeError: 'NoneType' object is not iterable

    which kills the engine loop and 503s every in-flight request. Only the
    MIXED batch is affected — all-empty takes the ``else`` branch and
    becomes ``[[]] * n``, all-present is iterable throughout. That is
    exactly "plain chat concurrent with a tool call", i.e. every agent we
    support, and it needs a split/merge to reshuffle the batch, which is
    why it is intermittent: 3 of 5 consecutive stress runs on main.

    ``[]`` is the correct normalization at both sites — it keeps ``any()``
    False for the all-empty case, so the fast path still skips the whole
    processing loop, and it stays iterable when mixed. We wrap rather than
    reimplement ``extend`` so upstream stays authoritative for everything
    else it does; we only clean up after it.

    Upstream tracking: rapid-mlx #1525. Remove when a released mlx-lm
    writes ``[]`` instead of ``None``.
    """
    try:
        # ``import_module``, NOT ``from mlx_lm import generate``: mlx_lm's
        # package namespace binds a top-level ``generate()`` FUNCTION that
        # shadows the submodule of the same name, so the ``from`` form
        # hands back a callable and every attribute lookup below silently
        # misses.
        from importlib import import_module

        _generate = import_module("mlx_lm.generate")
    except ImportError as exc:
        # "mlx_lm isn't here" is the expected case off Apple Silicon and
        # stays quiet. "mlx_lm is here but importing it blew up on one of
        # its dependencies" is a broken install that would otherwise leave
        # the guard silently disabled in production (raised in review).
        missing = getattr(exc, "name", None) or ""
        if missing == "mlx_lm" or missing.startswith("mlx_lm.") or missing == "mlx":
            return  # Linux CI / no MLX
        logger.warning(
            "rapid-mlx compat: could not import mlx_lm.generate (%s) — the "
            "#1525 logits_processors slot guard is NOT installed. Mixed "
            "chat+tool-call traffic may 503. Check the mlx-lm install.",
            exc,
        )
        return

    batch_cls = getattr(_generate, "PromptProcessingBatch", None)
    original = getattr(batch_cls, "extend", None)
    if original is None:
        # Renamed or restructured upstream. A shim that silently patches
        # the wrong thing is worse than one that declines — but declining
        # quietly is how a crash-prevention guard disappears across an
        # unattended dependency bump, so say so (raised in review).
        logger.warning(
            "rapid-mlx compat: mlx_lm.generate has no "
            "PromptProcessingBatch.extend to guard (mlx-lm %s) — the #1525 "
            "slot guard is NOT installed. Either upstream fixed it (then "
            "drop this shim) or it moved (then retarget it).",
            getattr(import_module("mlx_lm"), "__version__", "unknown"),
        )
        return
    if getattr(batch_cls, "_rapid_mlx_slot_guard", False):
        return

    def extend(self, batch):
        original(self, batch)
        slots = getattr(self, "logits_processors", None)
        if slots is not None and any(slot is None for slot in slots):
            self.logits_processors = [[] if s is None else s for s in slots]

    extend.__doc__ = original.__doc__
    extend.__name__ = original.__name__
    batch_cls.extend = extend
    batch_cls._rapid_mlx_slot_guard = True
    logger.debug("MLX compat shim installed (#1525 logits_processors slot guard).")
