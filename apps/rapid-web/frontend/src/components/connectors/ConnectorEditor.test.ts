import { describe, expect, it } from 'vitest';
import { draftError } from './ConnectorEditor';

/**
 * Mirrors `connectors.py`'s `validation_error`. The server's copy is the one
 * that decides; these assert the two say the same thing, so a name refused
 * on save is refused in the field first.
 */

function draft(patch: Record<string, unknown> = {}) {
  return {
    name: 'fs',
    transport: 'stdio' as const,
    command: 'npx',
    args: [],
    env: {},
    url: '',
    enabled: true,
    timeout: 30,
    ...patch,
  };
}

describe('draftError', () => {
  it('accepts a well-formed command connector', () => {
    expect(draftError(draft(), [])).toBeNull();
  });

  it('accepts a well-formed URL connector', () => {
    const entry = draft({ transport: 'sse', command: '', url: 'http://localhost:3001/sse' });
    expect(draftError(entry, [])).toBeNull();
  });

  it('requires a name', () => {
    expect(draftError(draft({ name: '  ' }), [])).toMatch(/name/i);
  });

  it('refuses a space in the name', () => {
    // `my server__read_file` is not a legal function name, so the model
    // could never call it.
    expect(draftError(draft({ name: 'my server' }), [])).toMatch(/letters, numbers/);
  });

  it('refuses a double underscore in the name', () => {
    // Both sides split `server__tool` on the first one, so `my__server`
    // would dispatch as server `my` and never resolve.
    expect(draftError(draft({ name: 'my__server' }), [])).toMatch(/letters, numbers/);
    expect(draftError(draft({ name: 'my_server' }), [])).toBeNull();
  });

  it('caps the name so the tool half still fits', () => {
    expect(draftError(draft({ name: 'x'.repeat(32) }), [])).toBeNull();
    expect(draftError(draft({ name: 'x'.repeat(33) }), [])).toMatch(/letters, numbers/);
  });

  it('refuses a duplicate name', () => {
    expect(draftError(draft(), ['fs'])).toMatch(/already exists/);
  });

  it('requires a command on a command connector', () => {
    expect(draftError(draft({ command: '' }), [])).toMatch(/command to run/);
  });

  it('requires an http(s) URL on a URL connector', () => {
    const base = { transport: 'sse' as const, command: '' };
    expect(draftError(draft({ ...base, url: '' }), [])).toMatch(/needs a URL/);
    expect(draftError(draft({ ...base, url: 'ftp://x/y' }), [])).toMatch(/http/);
    expect(draftError(draft({ ...base, url: 'not a url' }), [])).toMatch(/http/);
  });

  it('requires a positive timeout', () => {
    expect(draftError(draft({ timeout: 0 }), [])).toMatch(/greater than zero/);
    expect(draftError(draft({ timeout: Number.NaN }), [])).toMatch(/greater than zero/);
  });
});
