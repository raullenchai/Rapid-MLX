"""Serialized, process-local qualification for lazy native kernels."""

import logging
from threading import Lock

logger = logging.getLogger(__name__)


class SelfCheck:
    """Evaluate a synthetic probe once per specialization, failing closed."""

    def __init__(self, name):
        self.name = name
        self.verdicts = {}
        self.lock = Lock()

    def qualified(self, key, probe, *args):
        verdict = self.verdicts.get(key)
        if verdict is not None:
            return verdict
        with self.lock:
            if key in self.verdicts:
                return self.verdicts[key]
            try:
                verdict = bool(probe(*args))
            except Exception:  # Includes errors surfaced by evaluating the probe.
                verdict = False
            self.verdicts[key] = verdict
            if not verdict:
                logger.warning("%s self-check failed; using reference path", self.name)
            return verdict
