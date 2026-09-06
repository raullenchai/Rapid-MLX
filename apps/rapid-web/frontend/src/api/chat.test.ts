import { describe, expect, it } from 'vitest';
import { ToolCallAccumulator } from './chat';

describe('ToolCallAccumulator', () => {
  it('joins argument fragments spread across chunks', () => {
    const accumulator = new ToolCallAccumulator();
    accumulator.accept({ index: 0, id: 'call_1', function: { name: 'weather' } });
    accumulator.accept({ index: 0, function: { arguments: '{"loca' } });
    accumulator.accept({ index: 0, function: { arguments: 'tion":"Paris"}' } });

    expect(accumulator.finalize()).toEqual([
      {
        id: 'call_1',
        type: 'function',
        function: { name: 'weather', arguments: '{"location":"Paris"}' },
      },
    ]);
  });

  it('keeps the id and name a later fragment omits', () => {
    const accumulator = new ToolCallAccumulator();
    accumulator.accept({ index: 0, id: 'call_1', function: { name: 'weather', arguments: '{}' } });
    accumulator.accept({ index: 0, id: null, function: { name: null, arguments: '' } });

    expect(accumulator.finalize()[0]?.function.name).toBe('weather');
    expect(accumulator.finalize()[0]?.id).toBe('call_1');
  });

  it('orders by index, not by arrival', () => {
    const accumulator = new ToolCallAccumulator();
    accumulator.accept({ index: 1, id: 'b', function: { name: 'browse', arguments: '{}' } });
    accumulator.accept({ index: 0, id: 'a', function: { name: 'weather', arguments: '{}' } });

    expect(accumulator.finalize().map((call) => call.id)).toEqual(['a', 'b']);
  });

  // A call the executor could route nowhere must not reach it: the caller's
  // "no usable calls" branch is the honest outcome, a silent no-op is not.
  it('drops a call missing either the id or the name', () => {
    const accumulator = new ToolCallAccumulator();
    accumulator.accept({ index: 0, function: { name: 'weather', arguments: '{}' } });
    accumulator.accept({ index: 1, id: 'call_2', function: { arguments: '{}' } });

    expect(accumulator.finalize()).toEqual([]);
  });

  it('produces nothing when no delta arrived', () => {
    expect(new ToolCallAccumulator().finalize()).toEqual([]);
  });
});
