import { describe, expect, it } from 'vitest';
import { readEventStream, readJsonEventStream } from './sse';

/**
 * Build a ReadableStream that delivers exactly the given chunks, so a test can
 * place a chunk boundary anywhere it likes. That placement is the point: a
 * boundary landing mid-frame is the failure mode this parser exists to
 * survive, and it is not reproducible with a single-chunk fixture.
 */
function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

async function collect<T>(source: AsyncGenerator<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const item of source) out.push(item);
  return out;
}

describe('readEventStream', () => {
  it('yields each frame in order', async () => {
    const frames = await collect(
      readEventStream(streamOf(['data: one\n\ndata: two\n\ndata: three\n\n'])),
    );
    expect(frames.map((f) => f.data)).toEqual(['one', 'two', 'three']);
  });

  it('reassembles a frame split across chunk boundaries', async () => {
    // The boundary falls inside the word `payload` — the classic truncation
    // bug, where a token arrives as "pay" and the rest is lost.
    const frames = await collect(readEventStream(streamOf(['data: pay', 'load\n\n'])));
    expect(frames.map((f) => f.data)).toEqual(['payload']);
  });

  it('reassembles a frame split inside the delimiter itself', async () => {
    const frames = await collect(readEventStream(streamOf(['data: a\n', '\ndata: b\n\n'])));
    expect(frames.map((f) => f.data)).toEqual(['a', 'b']);
  });

  it('survives a boundary inside the "data:" prefix', async () => {
    const frames = await collect(readEventStream(streamOf(['dat', 'a: split\n\n'])));
    expect(frames.map((f) => f.data)).toEqual(['split']);
  });

  it('reassembles a multi-byte character split across chunks', async () => {
    // "é" is two bytes in UTF-8. Decoding each chunk independently would emit
    // a replacement character; the streaming decoder holds the first byte.
    const bytes = new TextEncoder().encode('data: caf\u00e9\n\n');
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        // Split between the two bytes of "é".
        controller.enqueue(bytes.slice(0, bytes.length - 3));
        controller.enqueue(bytes.slice(bytes.length - 3));
        controller.close();
      },
    });
    const frames = await collect(readEventStream(stream));
    expect(frames.map((f) => f.data)).toEqual(['caf\u00e9']);
  });

  it('ignores keepalive comment frames', async () => {
    // app.py:566 sends these every 15 s so a tunnel does not time the
    // download feed out. They must be invisible to callers.
    const frames = await collect(
      readEventStream(streamOf([': keepalive\n\ndata: real\n\n: keepalive\n\n'])),
    );
    expect(frames.map((f) => f.data)).toEqual(['real']);
  });

  it('swallows the [DONE] terminator', async () => {
    const frames = await collect(readEventStream(streamOf(['data: last\n\ndata: [DONE]\n\n'])));
    expect(frames.map((f) => f.data)).toEqual(['last']);
  });

  it('yields a final frame that arrives without a trailing blank line', async () => {
    const frames = await collect(readEventStream(streamOf(['data: only'])));
    expect(frames.map((f) => f.data)).toEqual(['only']);
  });

  it('accepts "data:" with and without the optional space', async () => {
    const frames = await collect(readEventStream(streamOf(['data:tight\n\ndata: loose\n\n'])));
    expect(frames.map((f) => f.data)).toEqual(['tight', 'loose']);
  });

  it('joins multiple data lines in one frame with newlines', async () => {
    const frames = await collect(readEventStream(streamOf(['data: first\ndata: second\n\n'])));
    expect(frames.map((f) => f.data)).toEqual(['first\nsecond']);
  });

  it('ignores non-data fields around a data line', async () => {
    const frames = await collect(
      readEventStream(streamOf(['event: message\nid: 7\ndata: body\nretry: 1000\n\n'])),
    );
    expect(frames.map((f) => f.data)).toEqual(['body']);
  });

  it('emits nothing for an empty stream', async () => {
    expect(await collect(readEventStream(streamOf([])))).toEqual([]);
  });
});

describe('readJsonEventStream', () => {
  it('decodes each frame', async () => {
    const frames = await collect(
      readJsonEventStream<{ n: number }>(streamOf(['data: {"n":1}\n\ndata: {"n":2}\n\n'])),
    );
    expect(frames).toEqual([{ n: 1 }, { n: 2 }]);
  });

  it('drops a malformed frame without killing the stream', async () => {
    // A single corrupt frame must not destroy an answer the user is already
    // reading. The terminal state of a stream is decided by its exit path,
    // never by one frame going missing.
    const frames = await collect(
      readJsonEventStream<{ n: number }>(
        streamOf(['data: {"n":1}\n\ndata: {not json\n\ndata: {"n":3}\n\n']),
      ),
    );
    expect(frames).toEqual([{ n: 1 }, { n: 3 }]);
  });

  it('surfaces an engine error frame with its type intact', async () => {
    const frames = await collect(
      readJsonEventStream<{ error?: { message: string; type: string } }>(
        streamOf(['data: {"error":{"message":"boom","type":"engine_transport"}}\n\n']),
      ),
    );
    expect(frames[0]?.error).toEqual({
      message: 'boom',
      type: 'engine_transport',
    });
  });
});
