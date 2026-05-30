# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for MiniMax models.

MiniMax models (e.g., MiniMax-M2.5) generate inline reasoning text without
explicit <think> tags. This parser detects and strips common reasoning patterns
that leak into visible output, such as:
  - "The user asks..." / "The user wants..."
  - "I need to..." / "Let me think..."
  - Step-by-step analysis before the actual answer

Unlike tag-based parsers, this uses heuristic pattern matching on the
beginning of the output to separate reasoning preamble from content.
"""

from __future__ import annotations

import re

from .base import DeltaMessage, ReasoningParser


class MiniMaxReasoningParser(ReasoningParser):
    """
    Reasoning parser for MiniMax models that think inline without tags.

    Strategy:
    - Buffer the first N characters of output
    - Detect reasoning preamble patterns
    - Find the transition point where real content begins
    - Emit reasoning as reasoning_content, rest as content
    """

    # Max chars to buffer before deciding (covers typical reasoning preamble)
    BUFFER_SIZE = 512

    # Patterns that indicate the START of reasoning (at beginning of output)
    _REASONING_START_RE = re.compile(
        r"^(?:\s*)"  # optional whitespace
        r"(?:"
        # English reasoning patterns
        r"(?:The\s+user\s+(?:asks|wants|is\s+asking|requests|said|query|question))"
        r"|(?:I\s+(?:need\s+to|should|will|can|want\s+to|have\s+to|must|am\s+going\s+to))"
        r"|(?:Let\s+me\s+(?:think|check|analyze|figure|consider|look|read|review|process))"
        r"|(?:This\s+(?:is\s+a|requires|seems|looks\s+like|appears))"
        r"|(?:First,?\s+(?:I|let|we))"
        r"|(?:(?:So|Now|OK|Okay|Alright|Well),?\s+(?:the\s+user|I\s+need|let\s+me|I\s+should))"
        r"|(?:what's\s+worth\s+storing)"
        r"|(?:(?:Analyzing|Thinking|Processing|Considering|Evaluating|Extracting)\s)"
        # Chinese reasoning patterns (MiniMax is a Chinese model)
        r"|(?:用户(?:想|要|需要|问|请求|说|希望|让我))"
        r"|(?:我(?:需要|应该|将|可以|要|得|必须))"
        r"|(?:让我(?:想|看|分析|检查|考虑|读|处理|review))"
        r"|(?:这(?:是一个|需要|似乎|看起来|个))"
        r"|(?:首先[，,]?(?:我|让|我们))"
        r"|(?:(?:好的|那么|现在|所以)[，,]?(?:用户|我需要|让我|我应该))"
        r"|(?:(?:分析|思考|处理|考虑|评估|提取)(?:一下|中|着))"
        r")",
        re.IGNORECASE,
    )

    # Patterns that indicate transition FROM reasoning TO content
    # These mark where the actual answer/response begins
    _CONTENT_TRANSITION_RE = re.compile(
        r"(?:"
        # Common answer starters after reasoning
        r"(?:^|\n\n)(?:(?:The\s+)?(?:answer|result|output|response|solution)\s*(?:is|:))"
        # MiniMax-specific patterns: "Thus answer:", "Thus final"
        r"|(?:^|\n)(?:Thus\s+(?:answer|final|the\s+answer|response)\s*[:\.])"
        # Direct content markers
        r"|(?:^|\n\n)(?:```)"  # code block start
        r"|(?:^|\n\n)(?:Here\s+(?:is|are)\s)"
        r"|(?:^|\n\n)(?:(?:Sure|Of\s+course|Absolutely)[!,.]?\s)"
        r"|(?:^|\n\n)(?:I'(?:d|ll|m)\s+(?:happy|glad)\s+to\s)"
        # Tool call markers (should NOT be stripped)
        r"|(?:<minimax:tool_call>)"
        r"|(?:<tool_call>)"
        r"|(?:<invoke\s)"
        # Structured output after reasoning
        r"|(?:^|\n\n)(?:\d+\.\s+\*\*)"  # numbered bold list
        r"|(?:^|\n\n)(?:##\s)"  # markdown heading
        # MiniMax meta-reasoning followed by actual answer
        r"|(?:^|\n\n)\*\*"  # bold text start (often the answer)
        # Chinese transition patterns
        r"|(?:^|\n\n)(?:(?:答案|结果|输出|响应|解决方案)\s*(?:是|：|:))"
        r"|(?:^|\n\n)(?:(?:好的|当然|没问题)[！!，,]?\s)"
        r"|(?:^|\n\n)(?:以下是)"
        r")",
        re.IGNORECASE | re.MULTILINE,
    )

    # If the output starts with these, it's NOT reasoning (direct content)
    _DIRECT_CONTENT_RE = re.compile(
        r"^(?:\s*)"
        r"(?:"
        r"```"  # code block
        r"|(?:<minimax:tool_call>)"
        r"|(?:<tool_call>)"
        r"|(?:<invoke\s)"
        r"|(?:#+\s)"  # markdown heading
        r"|(?:\{)"  # JSON object
        r"|(?:\[)"  # JSON array
        r")",
    )

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._buffer = ""
        self._decided = False
        self._is_reasoning = False
        self._transition_pos = 0

    def reset_state(self):
        """Reset state for a new stream."""
        self._buffer = ""
        self._decided = False
        self._is_reasoning = False
        self._transition_pos = 0

    def extract_reasoning(self, model_output: str) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete MiniMax output.

        Returns:
            (reasoning, content) tuple.
        """
        # Handle explicit <think> tags first (MiniMax sometimes uses them)
        if "<think>" in model_output or "</think>" in model_output:
            if "</think>" in model_output:
                parts = model_output.split("</think>", 1)
                reasoning = parts[0].replace("<think>", "").strip()
                content = parts[1].strip() if len(parts) > 1 else None
                return reasoning or None, content or None

        # Check for direct content (no reasoning)
        if self._DIRECT_CONTENT_RE.match(model_output):
            return None, model_output

        # Check if output starts with reasoning patterns
        if not self._REASONING_START_RE.match(model_output):
            return None, model_output

        # Find transition point
        match = self._CONTENT_TRANSITION_RE.search(model_output)
        if match:
            reasoning = model_output[: match.start()].strip()
            content = model_output[match.start() :].strip()
            # Don't strip if the "reasoning" is very short (likely false positive)
            if len(reasoning) < 10:
                return None, model_output
            return reasoning or None, content or None

        # No clear transition found - if the whole thing looks like reasoning
        # followed by a short answer, try splitting on double newline
        parts = model_output.split("\n\n", 1)
        if len(parts) == 2:
            first, second = parts
            # Only split if first part matches reasoning and second is shorter
            if self._REASONING_START_RE.match(first) and len(second.strip()) > 0:
                return first.strip(), second.strip()

        # Can't separate - return as content (conservative)
        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming delta for MiniMax models.
        """
        # Handle explicit </think> tag transition
        if "</think>" in delta_text:
            idx = delta_text.find("</think>")
            reasoning_part = delta_text[:idx]
            content_part = delta_text[idx + len("</think>") :]
            self._decided = True
            self._is_reasoning = False
            return DeltaMessage(
                reasoning=reasoning_part if reasoning_part else None,
                content=content_part if content_part else None,
            )

        # Skip <think> tag itself
        if "<think>" in delta_text:
            cleaned = delta_text.replace("<think>", "")
            self._decided = True
            self._is_reasoning = True
            if cleaned:
                return DeltaMessage(reasoning=cleaned)
            return None  # Skip the tag

        if self._decided:
            if self._is_reasoning:
                # Still in reasoning phase - check for transition
                match = self._CONTENT_TRANSITION_RE.search(
                    current_text[self._transition_pos :]
                )
                if match:
                    # Found transition to content
                    abs_pos = self._transition_pos + match.start()
                    self._decided = True
                    self._is_reasoning = False

                    # The delta might contain the transition
                    prev_len = len(previous_text)
                    if abs_pos >= prev_len:
                        # Transition is in this delta
                        reasoning_part = delta_text[: abs_pos - prev_len]
                        content_part = delta_text[abs_pos - prev_len :]
                        # Strip any leading newlines from content
                        content_part = content_part.lstrip("\n")
                        return DeltaMessage(
                            reasoning=reasoning_part if reasoning_part else None,
                            content=content_part if content_part else None,
                        )
                    else:
                        # Transition was before this delta - emit as content
                        return DeltaMessage(content=delta_text)

                # No transition yet - emit as reasoning
                return DeltaMessage(reasoning=delta_text)
            else:
                # In content phase - pass through
                return DeltaMessage(content=delta_text)

        # Still buffering - accumulate and decide
        self._buffer = current_text

        if len(self._buffer) < min(self.BUFFER_SIZE, 80):
            # Not enough text yet to decide
            # But check for direct content markers early
            if self._DIRECT_CONTENT_RE.match(self._buffer):
                self._decided = True
                self._is_reasoning = False
                # Flush entire buffer as content
                return DeltaMessage(content=current_text)
            # Buffer silently — don't emit anything until we decide
            return None

        # Enough text to decide
        self._decided = True

        if not self._REASONING_START_RE.match(self._buffer):
            # Not reasoning - flush entire buffer as content
            self._is_reasoning = False
            return DeltaMessage(content=current_text)

        # It IS reasoning - check for transition already in buffer
        self._is_reasoning = True
        match = self._CONTENT_TRANSITION_RE.search(self._buffer)
        if match:
            self._is_reasoning = False
            abs_pos = match.start()
            # Flush entire buffer: reasoning before transition, content after
            reasoning_part = current_text[:abs_pos].strip()
            content_part = current_text[abs_pos:].lstrip("\n")
            return DeltaMessage(
                reasoning=reasoning_part if reasoning_part else None,
                content=content_part if content_part else None,
            )

        self._transition_pos = max(0, len(self._buffer) - 20)
        # Flush entire buffer as reasoning
        return DeltaMessage(reasoning=current_text)

    def finalize_streaming(self, accumulated_text: str) -> DeltaMessage | None:
        """
        Finalize streaming - handle cases where content was never emitted:
        1. Still buffering (never decided) - emit buffer as content
        2. Everything classified as reasoning - try to extract answer
        """
        if not self._decided:
            # Never reached decision threshold — emit as content
            return DeltaMessage(content=accumulated_text) if accumulated_text else None

        if not self._is_reasoning:
            return None

        # Try to extract answer from accumulated reasoning text
        reasoning, content = self.extract_reasoning(accumulated_text)
        if content and content != accumulated_text:
            return DeltaMessage(content=content)

        # Can't separate - reclassify as content to avoid empty response
        return DeltaMessage(content=accumulated_text)
