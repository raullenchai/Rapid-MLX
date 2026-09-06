import { describe, expect, it } from 'vitest';
import { withoutToolCallEcho } from './toolEcho';
import type { ToolCall } from '@/api/chat';

const call = (name: string): ToolCall => ({
  id: 'call_1',
  type: 'function',
  function: { name, arguments: '{}' },
});

describe('withoutToolCallEcho', () => {
  it('drops a call the model also wrote out as prose', () => {
    const content = '{"name": "web_search", "arguments": {"query": "weather"}}';
    expect(withoutToolCallEcho(content, [call('web_search')])).toBe('');
  });

  it('keeps the prose around an echoed line', () => {
    const content = 'Let me look.\n{"name": "web_search", "arguments": {}}\nOne moment.';
    expect(withoutToolCallEcho(content, [call('web_search')])).toBe(
      'Let me look.\nOne moment.',
    );
  });

  it('keeps JSON that is the actual answer', () => {
    // Matched against the calls this turn made, never on shape alone: a turn
    // answering with JSON must survive intact.
    const content = '{"name": "Ada", "born": 1815}';
    expect(withoutToolCallEcho(content, [call('web_search')])).toBe(content);
  });

  it('leaves a turn that called nothing alone', () => {
    const content = '{"name": "web_search"}';
    expect(withoutToolCallEcho(content, undefined)).toBe(content);
    expect(withoutToolCallEcho(content, [])).toBe(content);
  });

  it('leaves ordinary prose alone', () => {
    expect(withoutToolCallEcho('It is 18°C.', [call('weather')])).toBe('It is 18°C.');
  });
});
