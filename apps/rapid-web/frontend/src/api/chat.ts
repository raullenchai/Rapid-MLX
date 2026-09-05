import { ApiError } from './errors';
import { request } from './client';
import { readJsonEventStream } from './sse';

/** One tool call, in the OpenAI shape. `arguments` stays a JSON-encoded
 *  STRING and is never re-encoded, so the envelope round-trips unchanged. */
export interface ToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

/** A tool definition as the engine takes it. `parameters` is an open JSON
 *  Schema blob — the server owns the shapes, this is a passthrough. */
export interface ToolDefinition {
  type: 'function';
  function: { name: string; description: string; parameters: unknown };
}

/** A turn as it goes on the wire. Nothing local (stats, ids, reasoning) rides
 *  along — the engine only ever sees role, content and the tool envelope. */
export interface WireTurn {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  /** On an assistant turn that called tools. */
  tool_calls?: ToolCall[];
  /** On a tool turn, naming the call it answers. */
  tool_call_id?: string;
}

export interface ChatRequest {
  turns: WireTurn[];
  /** Which resident model answers. Omitting it routes to the engine's primary,
   *  which is whatever was loaded last — an image model there answers a chat
   *  request with a 500. */
  model: string | null;
  temperature: number;
  topP: number;
  maxTokens: number;
  /** Omitted entirely when empty: sending `tools: []` is not the same as
   *  sending no tools, and some templates emit a tool preamble either way. */
  tools?: ToolDefinition[];
  signal: AbortSignal;
}

/** One decoded piece of a streamed answer. */
export type ChatDelta =
  | { kind: 'content'; text: string }
  | { kind: 'reasoning'; text: string }
  | { kind: 'usage'; completionTokens: number }
  | { kind: 'toolCalls'; calls: ToolCall[] };

/** On-the-wire shape of one streamed tool-call fragment. */
interface ToolCallDelta {
  index: number;
  id?: string | null;
  function?: { name?: string | null; arguments?: string | null };
}

/** The subset of the OpenAI streaming shape this client reads. */
interface ChatChunk {
  choices?: Array<{
    delta?: {
      content?: string | null;
      /** Reasoning models stream their scratchpad here, ahead of the answer. */
      reasoning_content?: string | null;
      tool_calls?: ToolCallDelta[] | null;
    };
    finish_reason?: string | null;
  }>;
  usage?: { completion_tokens?: number };
  error?: { message?: string; type?: string };
}

/**
 * Accumulates `delta.tool_calls` fragments into finished calls.
 *
 * One call is split across many chunks: the first carries `id` and
 * `function.name`, later ones only append slices of `function.arguments`.
 * `index` is the only key stable across them.
 */
export class ToolCallAccumulator {
  #byIndex = new Map<number, { id: string; name: string; arguments: string }>();

  accept(delta: ToolCallDelta): void {
    const builder = this.#byIndex.get(delta.index) ?? { id: '', name: '', arguments: '' };
    // Missing fields are left alone, so an arguments-only chunk extends the
    // builder rather than erasing the id and name that arrived first.
    if (delta.id) builder.id = delta.id;
    if (delta.function?.name) builder.name = delta.function.name;
    if (delta.function?.arguments) builder.arguments += delta.function.arguments;
    this.#byIndex.set(delta.index, builder);
  }

  /**
   * The finished calls, ordered by index.
   *
   * BOTH id and name must be non-empty. A malformed stream otherwise yields
   * calls the executor can route nowhere, and the round trip fails silently
   * instead of the caller seeing that no usable call was produced.
   */
  finalize(): ToolCall[] {
    return [...this.#byIndex.entries()]
      .sort(([a], [b]) => a - b)
      .filter(([, builder]) => builder.id !== '' && builder.name !== '')
      .map(([, builder]) => ({
        id: builder.id,
        type: 'function' as const,
        function: { name: builder.name, arguments: builder.arguments },
      }));
  }
}

/**
 * Stream a chat completion.
 *
 * A POST rather than an ``EventSource`` for two independent reasons: the
 * request carries a body, and ``EventSource`` cannot set the ``Authorization``
 * header this server requires.
 */
export async function* streamChat(options: ChatRequest): AsyncGenerator<ChatDelta> {
  const response = await request('/v1/chat/completions', {
    method: 'POST',
    signal: options.signal,
    body: {
      ...(options.model ? { model: options.model } : {}),
      messages: options.turns,
      stream: true,
      temperature: options.temperature,
      top_p: options.topP,
      max_tokens: options.maxTokens,
      ...(options.tools?.length ? { tools: options.tools, tool_choice: 'auto' } : {}),
      // Without this the engine sends no usage frame and the token count has
      // to be estimated from character length, which is off by a wide and
      // model-dependent margin.
      stream_options: { include_usage: true },
    },
  });

  if (!response.body) return;

  const accumulator = new ToolCallAccumulator();
  let sawToolCalls = false;

  for await (const chunk of readJsonEventStream<ChatChunk>(response.body, options.signal)) {
    // An error can arrive mid-stream, after a 200 and after real content: the
    // engine hit something partway through generating. It is a hard stop, not
    // a frame to skip.
    if (chunk.error) {
      throw new ApiError(
        502,
        chunk.error.type ?? 'engine_error',
        chunk.error.message ?? 'engine error',
      );
    }

    // A usage frame carries no delta, so it must return rather than fall
    // through into the choices lookup.
    if (chunk.usage) {
      yield {
        kind: 'usage',
        completionTokens: chunk.usage.completion_tokens ?? 0,
      };
      continue;
    }

    const choice = chunk.choices?.[0];
    // The delta is optional: some engines send a terminal chunk carrying only
    // `finish_reason`, and requiring a delta drops that reason with it.
    const delta = choice?.delta;

    if (delta?.reasoning_content) yield { kind: 'reasoning', text: delta.reasoning_content };
    if (delta?.content) yield { kind: 'content', text: delta.content };
    for (const fragment of delta?.tool_calls ?? []) accumulator.accept(fragment);

    if (choice?.finish_reason === 'tool_calls' && !sawToolCalls) {
      sawToolCalls = true;
      const calls = accumulator.finalize();
      if (calls.length > 0) yield { kind: 'toolCalls', calls };
    }
  }

  // Not every engine sets `finish_reason: "tool_calls"`. Emitting whatever
  // accumulated is what keeps those from looking like an empty answer.
  if (!sawToolCalls) {
    const calls = accumulator.finalize();
    if (calls.length > 0) yield { kind: 'toolCalls', calls };
  }
}
