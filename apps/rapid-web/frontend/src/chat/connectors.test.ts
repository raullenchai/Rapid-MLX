import { describe, expect, it } from 'vitest';
import type { ConnectorState } from '@/api/connectors';
import type { ToolCall } from '@/api/chat';
import {
  advertisedConnectorTools,
  displaySafe,
  displaySafeResult,
  formatArguments,
  gateConnectorCall,
  isConnectorTool,
  serverOf,
  shortToolName,
} from './connectors';

function state(patch: Partial<ConnectorState> = {}): ConnectorState {
  return {
    enabled: true,
    servers: [],
    load_error: null,
    config_path: '/tmp/mcp.json',
    engine_servers: [],
    engine_reachable: true,
    subsystem_error: null,
    configured: true,
    needs_restart: false,
    engine_running: true,
    tools: [
      { name: 'fs__read_file', description: 'Read a file', server: 'fs', parameters: null },
      { name: 'fs__write_file', description: 'Write a file', server: 'fs', parameters: null },
    ],
    disabled_tools: [],
    granted_tools: [],
    auto_approve_all: false,
    ...patch,
  };
}

function call(name: string, args = '{}'): ToolCall {
  return { id: 'call_1', type: 'function', function: { name, arguments: args } };
}

describe('namespaced names', () => {
  it('splits a name into its server and tool halves', () => {
    expect(serverOf('fs__read_file')).toBe('fs');
    expect(shortToolName('fs__read_file')).toBe('read_file');
  });

  it('splits on the FIRST separator', () => {
    // The server half is what the engine prefixed and never contains one, so
    // a tool whose own name has a double underscore keeps it.
    expect(serverOf('fs__read__file')).toBe('fs');
    expect(shortToolName('fs__read__file')).toBe('read__file');
  });

  it('leaves an unnamespaced name alone', () => {
    expect(serverOf('weather')).toBe('');
    expect(shortToolName('weather')).toBe('weather');
  });
});

describe('advertising', () => {
  it('offers every connected tool', () => {
    expect(advertisedConnectorTools(state()).map((t) => t.function.name)).toEqual([
      'fs__read_file',
      'fs__write_file',
    ]);
  });

  it('withholds a tool the user switched off', () => {
    const tools = advertisedConnectorTools(state({ disabled_tools: ['fs__write_file'] }));
    expect(tools.map((t) => t.function.name)).toEqual(['fs__read_file']);
  });

  it('offers nothing while the master switch is off', () => {
    // The running child keeps its connectors loaded until it is restarted,
    // so "off" has to be enforced rather than inferred from an empty list.
    expect(advertisedConnectorTools(state({ enabled: false }))).toEqual([]);
  });

  it('offers nothing when the state could not be fetched', () => {
    expect(advertisedConnectorTools(null)).toEqual([]);
  });

  it('substitutes an empty schema for a tool that declares none', () => {
    // An absent `parameters` still has to be a legal object or the engine
    // rejects the whole tools array.
    const [first] = advertisedConnectorTools(state());
    expect(first?.function.parameters).toEqual({ type: 'object', properties: {} });
  });

  it('keeps a declared schema verbatim', () => {
    const schema = { type: 'object', properties: { path: { type: 'string' } } };
    const tools = advertisedConnectorTools(
      state({
        tools: [{ name: 'fs__read', description: 'd', server: 'fs', parameters: schema }],
      }),
    );
    expect(tools[0]?.function.parameters).toBe(schema);
  });
});

describe('isConnectorTool', () => {
  it('recognises a connector tool', () => {
    expect(isConnectorTool(state(), 'fs__read_file')).toBe(true);
  });

  it('does not claim a built-in', () => {
    // `weather` is dispatched to /api/tools/call, not the connector route.
    expect(isConnectorTool(state(), 'weather')).toBe(false);
  });
});

describe('the gate', () => {
  it('asks before a tool runs for the first time', () => {
    const decision = gateConnectorCall(call('fs__read_file'), state());
    expect(decision).toMatchObject({ kind: 'approve', server: 'fs', short: 'read_file' });
  });

  it('runs a tool with a remembered grant', () => {
    const decision = gateConnectorCall(
      call('fs__read_file'),
      state({ granted_tools: ['fs__read_file'] }),
    );
    expect(decision.kind).toBe('run');
  });

  it('does not let one tool\u2019s grant cover another', () => {
    // "Always allow time__get_current_time" must not silently grant
    // shell__run.
    const decision = gateConnectorCall(
      call('fs__write_file'),
      state({ granted_tools: ['fs__read_file'] }),
    );
    expect(decision.kind).toBe('approve');
  });

  it('runs everything under the blanket switch', () => {
    const decision = gateConnectorCall(call('fs__read_file'), state({ auto_approve_all: true }));
    expect(decision.kind).toBe('run');
  });

  it('refuses a tool the user switched off', () => {
    // Leaving it out of the request body does not stop a malformed model
    // emitting the name anyway.
    const decision = gateConnectorCall(
      call('fs__write_file'),
      state({ disabled_tools: ['fs__write_file'] }),
    );
    expect(decision).toMatchObject({ kind: 'refuse' });
  });

  it('refuses everything while the master switch is off', () => {
    const decision = gateConnectorCall(call('fs__read_file'), state({ enabled: false }));
    expect(decision).toMatchObject({ kind: 'refuse' });
  });

  it('refuses when the state could not be fetched', () => {
    expect(gateConnectorCall(call('fs__read_file'), null).kind).toBe('refuse');
  });

  it('carries the arguments in full, never truncated', () => {
    // The sheet scrolls; truncating would let whatever is past the cutoff be
    // approved unseen, which is what the gate exists to prevent.
    const args = JSON.stringify({ path: 'x'.repeat(500) });
    const decision = gateConnectorCall(call('fs__read_file', args), state());
    expect(decision).toMatchObject({ kind: 'approve', args });
  });

  it('names the server even when the tool is not namespaced', () => {
    const decision = gateConnectorCall(
      call('bare'),
      state({ tools: [{ name: 'bare', description: 'd', server: '', parameters: null }] }),
    );
    expect(decision).toMatchObject({ kind: 'approve', server: 'unknown' });
  });
});

describe('displaySafe', () => {
  it('leaves ordinary text alone', () => {
    expect(displaySafe('read_file')).toBe('read_file');
  });

  it('escapes a bidi override', () => {
    // A connector supplies its own tool names and descriptions, and a bidi
    // override can make a prompt read as a tool the user trusts.
    expect(displaySafe('read\u202Eelif')).toBe('read\\u{202E}elif');
  });

  it('escapes a zero-width character', () => {
    expect(displaySafe('fs\u200B__read')).toBe('fs\\u{200B}__read');
  });

  it('keeps non-Latin text readable', () => {
    // Escaping by codepoint range must not mangle a legitimate description.
    expect(displaySafe('读取文件')).toBe('读取文件');
  });
});

describe('displaySafeResult', () => {
  it('preserves the line and column layout of tool output', () => {
    expect(displaySafeResult('[FILE] a.txt\n[DIR]\tfolder')).toBe(
      '[FILE] a.txt\n[DIR]\tfolder',
    );
  });

  it('normalizes CRLF without exposing a carriage return', () => {
    expect(displaySafeResult('one\r\ntwo')).toBe('one\ntwo');
  });

  it('still escapes characters that can spoof displayed output', () => {
    expect(displaySafeResult('safe\nread\u202Eelif\u0000')).toBe(
      'safe\nread\\u{202E}elif\\u{0}',
    );
  });
});

describe('formatArguments', () => {
  it('pretty-prints a JSON object', () => {
    expect(formatArguments('{"path":"/tmp"}')).toBe('{\n  "path": "/tmp"\n}');
  });

  it('says so when there are none', () => {
    expect(formatArguments('  ')).toBe('(no arguments)');
  });

  it('passes malformed arguments through verbatim', () => {
    // What will be sent is what has to be shown, even when it is not JSON.
    expect(formatArguments('{oops')).toBe('{oops');
  });
});
