/**
 * Server-sent event framing over ``fetch`` + ``ReadableStream``.
 *
 * Not ``EventSource``: it cannot set an ``Authorization`` header, and this
 * server's only credential is a bearer. That is also why the chat stream is a
 * POST — ``EventSource`` is GET-only.
 */

/** A parsed frame. Comment frames and ``[DONE]`` never reach a caller. */
export interface SseFrame {
  data: string;
}

/**
 * Split a byte stream into SSE frames, yielding each ``data:`` payload.
 * Three behaviours are load-bearing:
 *
 * * **Partial frames survive chunk boundaries.** A chunk can end mid-``data:``,
 *   so the trailing segment is retained rather than parsed.
 * * **Comment frames are skipped** — the keepalives that hold a tunnel open
 *   have no ``data:`` prefix and simply do not match.
 * * **``[DONE]`` is swallowed.** OpenAI's terminator, not a payload.
 */
export async function* readEventStream(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<SseFrame> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (signal?.aborted) break;

      // `{ stream: true }` so a multi-byte character split across two chunks
      // is held back rather than emitted as a replacement character.
      buffer += decoder.decode(value, { stream: true });

      const segments = buffer.split('\n\n');
      // The last segment is either empty (the chunk ended on a boundary) or a
      // partial frame. Either way it stays in the buffer.
      buffer = segments.pop() ?? '';

      for (const segment of segments) {
        const frame = parseFrame(segment);
        if (frame) yield frame;
      }
    }

    // A stream that ends without a trailing blank line still has one whole
    // frame left in the buffer. `[DONE]` normally makes this moot, but a
    // server that closes cleanly after a final frame would otherwise lose it.
    const trailing = parseFrame(buffer);
    if (trailing) yield trailing;
  } finally {
    // Releasing the lock lets an abort actually tear the connection down.
    // Without it the reader holds the stream and the request lingers.
    reader.releaseLock();
  }
}

/**
 * Extract the ``data:`` payload from one frame, or null if there is none.
 *
 * A frame may carry several lines (``event:``, ``id:``, ``retry:``); only
 * ``data:`` is used here, and per the spec multiple ``data:`` lines in one
 * frame join with newlines.
 */
function parseFrame(segment: string): SseFrame | null {
  const parts: string[] = [];
  for (const line of segment.split('\n')) {
    if (!line.startsWith('data:')) continue;
    // One optional space after the colon is part of the framing, not the data.
    parts.push(line.slice(line.startsWith('data: ') ? 6 : 5));
  }
  if (parts.length === 0) return null;

  const data = parts.join('\n').trim();
  if (data === '' || data === '[DONE]') return null;
  return { data };
}

/**
 * ``readEventStream`` with JSON decoding.
 *
 * A frame that fails to parse is DROPPED, not thrown. A single malformed
 * frame mid-stream must not destroy an answer the user is already reading;
 * the terminal state of a stream is decided by the exit path, never by one
 * frame going missing.
 */
export async function* readJsonEventStream<T>(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<T> {
  for await (const frame of readEventStream(body, signal)) {
    let parsed: T;
    try {
      parsed = JSON.parse(frame.data) as T;
    } catch {
      continue;
    }
    yield parsed;
  }
}
