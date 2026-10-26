# SPDX-License-Identifier: Apache-2.0
"""Opt-in, nonpublishing channel comparison for the lexer migration."""

from __future__ import annotations

import hashlib
import logging
import os

from .output_demux import (
    ContentDelta,
    OutputDemultiplexer,
    ParseContext,
    ReasoningDelta,
)

logger = logging.getLogger(__name__)


class _OrderedDigest:
    """Partition-independent digest of consecutive lane runs, without raw text."""

    def __init__(self):
        self._digest = hashlib.sha256()
        self._run = hashlib.sha256()
        self._lane = None
        self._length = 0

    def _commit(self, digest):
        if self._lane is not None:
            digest.update(bytes([self._lane]))
            digest.update(self._length.to_bytes(8, "big"))
            digest.update(self._run.digest())

    def update(self, lane, text):
        if not text:
            return
        if lane != self._lane:
            self._commit(self._digest)
            self._run = hashlib.sha256()
            self._length = 0
            self._lane = lane
        data = text.encode("utf-8")
        self._run.update(data)
        self._length += len(data)

    def digest(self):
        result = self._digest.copy()
        self._commit(result)
        return result.digest()


class ShadowComparison:
    """Hash ordered channel text, not chunk boundaries or private transcript data.

    Phase three observes text-only Qwen-family requests. Tool grammar and native
    router comparisons become eligible when their adapters are migrated. Errors
    disable this diagnostic request only and never alter published events.
    """

    def __init__(self):
        self.parser = None
        self.disabled = False
        self.report = None
        self._old = [hashlib.sha256(), hashlib.sha256()]
        self._new = [hashlib.sha256(), hashlib.sha256()]
        self._old_order = _OrderedDigest()
        self._new_order = _OrderedDigest()

    def _record_new(self, events):
        for event in events:
            if isinstance(event, (ContentDelta, ReasoningDelta)):
                index = int(isinstance(event, ReasoningDelta))
                self._new[index].update(event.text.encode("utf-8"))
                self._new_order.update(index, event.text)

    def _record_old(self, events):
        for event in events:
            self._old[0].update((event.content or "").encode("utf-8"))
            self._old[1].update((event.reasoning or "").encode("utf-8"))
            self._old_order.update(0, event.content or "")
            self._old_order.update(1, event.reasoning or "")

    def observe(self, output, events):
        if self.disabled or self.report is not None:
            return
        try:
            if output.channel is not None or output.tool_calls:
                self.disabled = True
                return
            if self.parser is None:
                self.parser = OutputDemultiplexer(
                    ParseContext(
                        prompt=output.prompt_parse_context, default_channel="reasoning"
                    )
                )
            self._record_new(self.parser.feed(output.new_text))
            self._record_old(events)
        except Exception:
            self.disabled = True
            logger.debug("Parser shadow disabled after diagnostic failure")

    def finish(self, events):
        if self.disabled or self.report is not None:
            return
        try:
            if self.parser is None:
                self.parser = OutputDemultiplexer(ParseContext())
            self._record_new(self.parser.finish())
            self._record_old(events)
            self.report = {
                "content_equal": self._old[0].digest() == self._new[0].digest(),
                "reasoning_equal": self._old[1].digest() == self._new[1].digest(),
                "order_equal": self._old_order.digest() == self._new_order.digest(),
                "source_chars": self.parser.offset,
                "examined_chars": self.parser.examined_chars,
            }
            logger.debug("Parser shadow comparison: %s", self.report)
        except Exception:
            self.disabled = True
            logger.debug("Parser shadow disabled after diagnostic failure")


def make_shadow(cfg, tools_requested):
    if (
        os.environ.get("RAPID_MLX_PARSER_SHADOW") == "1"
        and cfg.reasoning_parser_name in {"qwen3", "hy_v3", "hy3"}
        and not tools_requested
    ):
        return ShadowComparison()
    return None
