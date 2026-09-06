import { describe, expect, it } from 'vitest';
import type { ToolCall } from '@/api/chat';
import { advertised, displaySafe, gate, originOf } from './tools';

function call(name: string, args = '{}'): ToolCall {
  return { id: 'call_1', type: 'function', function: { name, arguments: args } };
}

const definitions = [
  { type: 'function' as const, function: { name: 'weather', description: '', parameters: {} } },
  { type: 'function' as const, function: { name: 'browse', description: '', parameters: {} } },
];

describe('advertised', () => {
  it('keeps only the enabled tools', () => {
    expect(advertised(definitions, ['weather']).map((t) => t.function.name)).toEqual(['weather']);
  });

  it('is empty when nothing is enabled', () => {
    expect(advertised(definitions, [])).toEqual([]);
  });
});

describe('originOf', () => {
  it('makes the default port explicit so a bare host and its port agree', () => {
    expect(originOf('http://example.com/a')).toBe('http://example.com:80');
    expect(originOf('http://example.com:80/b')).toBe('http://example.com:80');
    expect(originOf('https://example.com/a')).toBe('https://example.com:443');
  });

  it('treats a path change as the same origin and a host change as a new one', () => {
    expect(originOf('https://a.example/one')).toBe(originOf('https://a.example/two'));
    expect(originOf('https://a.example/')).not.toBe(originOf('https://b.example/'));
  });

  it('refuses anything that is not http(s)', () => {
    expect(originOf('file:///etc/passwd')).toBeNull();
    expect(originOf('javascript:alert(1)')).toBeNull();
    expect(originOf('not a url')).toBeNull();
  });
});

describe('displaySafe', () => {
  it('leaves an ordinary URL alone', () => {
    expect(displaySafe('https://example.com/a?b=1')).toBe('https://example.com/a?b=1');
  });

  // A bidi override renders a hostile host as a trusted one, and the prompt
  // is the only thing standing between the model's choice and the request.
  it('escapes a bidi override so the shown host is the fetched host', () => {
    expect(displaySafe('https://evil\u202Eexample.com')).toContain('\\u{202E}');
  });

  it('escapes a zero-width joiner', () => {
    expect(displaySafe('https://a\u200Bb.example')).toContain('\\u{200B}');
  });
});

describe('gate', () => {
  const base = {
    advertised: new Set(['weather', 'browse']),
    approvalRequired: new Set(['browse']),
    approvedOrigins: new Set<string>(),
  };

  it('runs a tool that needs no approval', () => {
    expect(gate(call('weather'), base)).toEqual({ kind: 'run' });
  });

  // The load-bearing check: leaving a tool out of the request body does not
  // stop a malformed model emitting a call for it.
  it('refuses a tool that was not advertised this round', () => {
    const decision = gate(call('browse'), { ...base, advertised: new Set(['weather']) });
    expect(decision.kind).toBe('refuse');
    expect(decision.kind === 'refuse' && decision.reason).toContain('weather');
  });

  it('asks before fetching a host that has not been approved', () => {
    const decision = gate(call('browse', '{"url":"https://example.com/a"}'), base);
    expect(decision).toEqual({
      kind: 'approve',
      url: 'https://example.com/a',
      host: 'example.com',
      origin: 'https://example.com:443',
    });
  });

  it('does not ask twice for the same origin', () => {
    const decision = gate(call('browse', '{"url":"https://example.com/b"}'), {
      ...base,
      approvedOrigins: new Set(['https://example.com:443']),
    });
    expect(decision).toEqual({ kind: 'run' });
  });

  it('still asks when the host changes', () => {
    const decision = gate(call('browse', '{"url":"https://other.example/b"}'), {
      ...base,
      approvedOrigins: new Set(['https://example.com:443']),
    });
    expect(decision.kind).toBe('approve');
  });

  it('refuses a non-http scheme rather than prompting for it', () => {
    const decision = gate(call('browse', '{"url":"file:///etc/passwd"}'), base);
    expect(decision.kind).toBe('refuse');
  });

  it('refuses arguments that are not valid JSON', () => {
    const decision = gate(call('browse', '{"url":'), base);
    expect(decision.kind).toBe('refuse');
    expect(decision.kind === 'refuse' && decision.reason).toContain('valid JSON');
  });

  it('refuses a missing url', () => {
    expect(gate(call('browse', '{}'), base).kind).toBe('refuse');
  });

  it('skips the prompt when auto-approve is on', () => {
    const origins = new Set<string>();
    const decision = gate(call('browse', '{"url":"https://example.com/a"}'), {
      ...base,
      approvedOrigins: origins,
      autoApprove: true,
    });

    expect(decision).toEqual({ kind: 'run' });
    // Recorded, so the server is told which origin the fetch was allowed for.
    expect([...origins]).toEqual(['https://example.com:443']);
  });

  // Auto-approve waives the human, not the guard: the scheme check runs first
  // and a file:// URL never becomes a fetch.
  it('still refuses a non-http scheme with auto-approve on', () => {
    const decision = gate(call('browse', '{"url":"file:///etc/passwd"}'), {
      ...base,
      autoApprove: true,
    });
    expect(decision.kind).toBe('refuse');
  });

  it('still refuses an unadvertised tool with auto-approve on', () => {
    const decision = gate(call('browse', '{"url":"https://example.com"}'), {
      ...base,
      advertised: new Set(['weather']),
      autoApprove: true,
    });
    expect(decision.kind).toBe('refuse');
  });
});
