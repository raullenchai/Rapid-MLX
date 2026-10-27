# Output parser migration core

`OutputDemultiplexer` is a request-local lexer, grammar and commit cursor. It was
introduced in shadow mode; the selected production adapters described below now
use this contract.

- `ParseContext` carries rendered current-turn evidence and an explicit grammar
  default for unknown prompts. Requested thinking mode alone is not evidence.
- `feed(text, token_id=...)` accepts one aligned decoded token span when an ID is
  available. Recognized atomic controls override spelling. Unaligned multi-token
  chunks use text lexing. Prefix-free terminal spellings have bounded holdback:
  strictly less than the longest terminal, with no response-prefix rescanning.
- Offsets are half-open Unicode code-point ranges in decoded generation, starting
  at zero. They are not UTF-8 byte or tokenizer offsets. `input_bytes` records UTF-8
  input size; `examined_chars` counts characters in tested bounded prefixes.
- A grammar consumes each lexeme once. Events own consecutive source spans;
  controls own their wire markers and zero-width lifecycle events own no text.
  Content/reasoning delta lengths must match their spans; converted tool
  arguments may differ from their source spelling.
  Candidate adapters may defer publication, but must account for every span by
  finish. Header limits and argument storage are separate from marker lookahead.
- `finish(reason)` and `cancel()` drain once. Unresolved text marker prefixes stay
  literal in their current channel, including on truncation/cancellation. No
  completed output is replayed. Input after terminal state is rejected. An
  adapter exception during feed or finish permanently fails that parser; retry
  requires a new instance rather than reusing partially advanced grammar state.
- `accept_native` bypasses text grammar. A stream cannot mix already routed events
  with raw-text reclassification. Native adapters must supply source offsets
  and cover their complete batch extent; incomplete batches are rejected before
  any cursor or mode changes.
- `EventReducer` accumulates the same events for buffered consumers. Only ended,
  nonfailed calls are eligible; no incomplete call receives fabricated arguments.
  Indexed failures reserve that index even before a start event, so a failed
  candidate cannot later become an executable call under the same index.
  Name authorization and argument validation belong to the format adapter before
  `ToolEnd`. This phase supplies the event contract, not new executable grammars.

Set the internal diagnostic environment switch `RAPID_MLX_PARSER_SHADOW=1` before
creating a request to compare text-only Qwen-family postprocessor channels. The
comparison stores only incremental hashes and logs equality flags and work
counters at debug level. It never publishes candidate events, logs transcript
text, or changes the existing response on a diagnostic failure. Tool requests,
native routing and other families remain ineligible until their adapters move.
The independent Responses/Messages parsing paths are not consolidated here.

Tests exercise split points, randomized partitions, token/text boundaries,
source ownership, terminal lifecycles, native bypass, reducer failures, request
reset, diagnostic failures, and real-route switch invariance. The 1K/4K/16K/64K
work-counter test is deterministic; the existing full-postprocessor CPU contract
remains a strict expected failure until the tool grammar migration.

## Prompt-owned Flint and LFM2.5 output

`PromptOutputOwner` activates from the rendered Flint template contract or the
LFM2.5 model identity plus `PromptParseContext`. The last current-turn thinking
opener/closer determines the initial channel; a request's thinking switch does
not. A bare Flint assistant prefix starts in content. The template renderer no
longer adds an empty thinking block.

The existing `OutputDemultiplexer` owns split-marker holdback and finalization.
Its selected grammar preserves tool-looking reasoning and shields content tool
arguments from reasoning transitions. Chat, Responses and Messages pass only
content to tool extraction, including buffered replies. Owned reasoning bypasses
serializer stripping and EOF promotion. Native channel events and other model
families retain their existing paths.

LFM2.5's native tool-envelope terminals produce control events in content and
remain literal in reasoning. The existing tool parser receives the pythonic
payload. Reset restores legacy parser configuration before the next stream.

## Prompt-owned Qwen tool envelopes

`PromptOutputOwner.for_request` also selects one owner for Qwen-family
reasoning (`qwen3`, `hy_v3`, `hy3`) paired with a `<tool_call>` envelope parser
(Hermes/Qwen JSON, Qwen XML, GLM 4.x) when the request declares tools, enables
automatic tool choice, asks for thinking and sets no reasoning budget. Disabled
tools, thinking-off templates and reasoning budgets keep their existing paths.

Reasoning is lexed first; the start marker is honored once and a closer ends
only an open block. Every closer outside an envelope is a reasoning control, as
in the existing thinking parser, so a repeated closer is not answer text. A thinking request whose rendered prompt state is unknown
starts in reasoning; there is no end-of-output rescue. A block the output
opened resumes after an envelope; reasoning inherited from the prompt ends at
the tool phase. Markers inside a JSON string or a markup argument value (GLM `<arg_value>`,
Qwen XML `<parameter=...>`) are argument text; the decoder skips its own
reasoning-tag stripping because reasoning is already separated. Each
envelope, from reasoning or content, is one tool phase:
its source span is decoded once by the configured parser's complete-output
extractor and published as typed tool events, so every API mode receives the
same calls through its existing structured-call lane. An envelope that yields
no call stays verbatim (opener, body and closer) in the channel it arrived in,
as does a partial envelope interrupted by a new opener outside an argument value,
or by the closer of the reasoning block it interrupted, and an envelope still
open at finish. A closed GLM argument list without its
call opener is an unpublished parse failure (#5274). Routes' existing
streaming markup filters still apply to that literal text. Buffered fields
keep the existing trimmed normalization.

## Prompt-owned Qwen reasoning without tools

A tool-free Qwen-family request that asks for thinking and sets no reasoning
budget selects the same grammar without tool terminals, but only when the
engine reports the rendered prompt's initial channel. A prompt that opens
`<think>` starts the output in reasoning even when the model never repeats the
opener; a prompt whose block is already closed starts in content. Unknown
prompt state, declared tools the envelope owner does not select, and budgets
keep their existing parser policy.

A length stop before the reasoning end marker keeps the unfinished output in
reasoning and leaves content empty in every API mode (#5295). No notice and no
reasoning tail is copied into content; clients read `finish_reason="length"`,
Responses `status="incomplete"` or Messages `stop_reason="max_tokens"`.

Remaining migration work includes other families' reasoning parsers and think
routers, content sanitizers, format-specific tool filters and fallback parsing,
and API-specific tool serialization/validation. These remain active; this change
does not install a grammar dependency or introduce new request errors. The
prompt-owner differential ledger records every change from v0.16.10, including
unchanged Qwen control cases. The source comments identify the reference rules.

For Flint and LFM2.5, a closing thinking marker without an open reasoning
block remains literal output, matching the reference detector. Live acceptance must reject such model output;
the adapter does not hide it with a sanitizer.

Existing non-reasoning control cleanup remains in place. Foreign marker spellings
are not automatically controls of the selected grammar: for example, LFM's native
tool envelope is ordinary literal text for Flint. The differential ledger records
this removal of broad cross-format sanitizer rewriting.
