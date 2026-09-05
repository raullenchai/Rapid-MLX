import type { ToolCall } from '@/api/chat';

/**
 * Drop a tool call the model also wrote out as prose.
 *
 * Small models frequently emit `{"name": "web_search", "arguments": {…}}` into
 * their content AS WELL AS in the tool-call channel. It is the same dispatch
 * twice, and the chip below already says what ran — so the echo is noise the
 * user has to read past, and it looks like the model malfunctioned.
 *
 * Matched against the calls this turn ACTUALLY made, never on JSON shape
 * alone: a turn that legitimately answers with JSON must survive intact.
 */
export function withoutToolCallEcho(content: string, calls: ToolCall[] | undefined): string {
  if (!calls?.length) return content;
  const names = new Set(calls.map((call) => call.function.name));

  if (echoesACall(content, names)) return '';

  const kept = content
    .split('\n')
    .filter((line) => !echoesACall(line, names))
    .join('\n');
  return kept.trim() === '' ? '' : kept;
}

/** Does this text parse as an object naming one of the calls? */
function echoesACall(text: string, names: Set<string>): boolean {
  const trimmed = text.trim();
  if (!trimmed.startsWith('{') || !trimmed.endsWith('}')) return false;
  try {
    const parsed: unknown = JSON.parse(trimmed);
    if (parsed === null || typeof parsed !== 'object') return false;
    const name = (parsed as { name?: unknown }).name;
    return typeof name === 'string' && names.has(name);
  } catch {
    return false;
  }
}
