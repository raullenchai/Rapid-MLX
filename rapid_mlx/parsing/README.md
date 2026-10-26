# Output parser migration core

`OutputDemultiplexer` is a request-local lexer, grammar and commit cursor. Phase
three introduces it in shadow mode; production parsing remains unchanged until
the selected format adapters move onto this contract.

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
