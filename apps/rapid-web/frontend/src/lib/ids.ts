/**
 * Id generation.
 *
 * `crypto.randomUUID` is only available in a SECURE CONTEXT, which a tunnel
 * without TLS or a plain `--host 0.0.0.0` bind is not. There it is
 * `undefined`, and calling it throws on every message and conversation.
 *
 * The fallback need not be cryptographically strong — these are local storage
 * keys, and nothing authenticates on them. It needs to be unique, and
 * monotonic-ish so `precedes` has a sensible tie-break.
 */

let counter = 0;

export function newId(): string {
  const uuid = globalThis.crypto?.randomUUID;
  if (typeof uuid === 'function') return globalThis.crypto.randomUUID();
  counter += 1;
  return `${Date.now().toString(36)}-${counter.toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
