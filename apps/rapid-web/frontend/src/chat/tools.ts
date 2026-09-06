import { callTool, fetchTools, type ToolCallResult } from '@/api/tools';
import type { ToolCall, ToolDefinition } from '@/api/chat';

/**
 * The tool loop's policy, separate from the streaming that drives it.
 *
 * Executing a call is a server round trip; deciding whether it may run at
 * all is local. Keeping the two apart is what lets the budget, the allowlist
 * and the approval gate be tested without a stream.
 */

/**
 * How many calls one turn may execute.
 *
 * Counted per requested call, not per round: a model can batch several into
 * one turn, and a per-round cap would let it run unbounded work by widening
 * each batch.
 */
export const MAX_TOOL_EXECUTIONS = 3;

export const TOOL_BUDGET_MESSAGE = `The model could not finish after ${MAX_TOOL_EXECUTIONS} tool calls. Try rephrasing, or turn a tool off.`;

let cached: { tools: ToolDefinition[]; approvalRequired: Set<string> } | null = null;

/** The server's tool catalogue, fetched once per page load. */
export async function loadTools(): Promise<{
  tools: ToolDefinition[];
  approvalRequired: Set<string>;
}> {
  if (cached) return cached;
  const response = await fetchTools();
  cached = {
    tools: response.tools,
    approvalRequired: new Set(response.approval_required),
  };
  return cached;
}

/** Test seam, and what a failed fetch falls back to. */
export function resetToolCache(): void {
  cached = null;
}

/** The definitions to advertise this round. */
export function advertised(all: ToolDefinition[], enabled: string[]): ToolDefinition[] {
  const wanted = new Set(enabled);
  return all.filter((tool) => wanted.has(tool.function.name));
}

/**
 * scheme://host:port, default port made explicit.
 *
 * An approval is granted to an ORIGIN, not a URL: approving one article on a
 * host and then re-prompting for the next link on the same host is noise, but
 * a host change is exactly what the prompt exists to catch.
 */
export function originOf(url: string): string | null {
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return null;
    const port = parsed.port || (parsed.protocol === 'https:' ? '443' : '80');
    return `${parsed.protocol}//${parsed.hostname.toLowerCase()}:${port}`;
  } catch {
    return null;
  }
}

/**
 * Escape everything that could disguise where a URL actually points.
 *
 * A model chooses the URL it asks to fetch, and a bidi override renders
 * `evil.example` as a host the user trusts. The approval prompt is the only
 * defence against that, so what it shows has to be what will be fetched.
 */
export function displaySafe(url: string): string {
  return [...url]
    .map((character) => {
      const code = character.codePointAt(0) ?? 0;
      const printable =
        code >= 0x20 && code !== 0x7f && !(code >= 0x200b && code <= 0x206f) && code !== 0xfeff;
      return printable ? character : `\\u{${code.toString(16).toUpperCase()}}`;
    })
    .join('');
}

/** What a call needs before it may run. */
export type Gate =
  | { kind: 'run' }
  | { kind: 'approve'; url: string; host: string; origin: string }
  | { kind: 'refuse'; reason: string };

/**
 * Decide whether `call` may run now.
 *
 * The allowlist check is here as well as on the server. Leaving a tool out of
 * the request body does not stop a malformed model emitting a call for it,
 * and a local refusal costs no round trip.
 */
export function gate(
  call: ToolCall,
  options: {
    advertised: Set<string>;
    approvalRequired: Set<string>;
    approvedOrigins: Set<string>;
    /** Skip the prompt for a public destination. The scheme and private-range
     *  checks still apply — this waives the human, not the guard. */
    autoApprove?: boolean;
  },
): Gate {
  const name = call.function.name;
  if (!options.advertised.has(name)) {
    const listed = [...options.advertised].sort().join(', ');
    return {
      kind: 'refuse',
      reason: listed
        ? `unknown tool '${name}' — available: ${listed}. Answer directly instead.`
        : `unknown tool '${name}'. Answer directly instead.`,
    };
  }
  if (!options.approvalRequired.has(name)) return { kind: 'run' };

  let url: unknown;
  try {
    url = (JSON.parse(call.function.arguments || '{}') as { url?: unknown }).url;
  } catch {
    return { kind: 'refuse', reason: `tool '${name}' error: arguments were not valid JSON` };
  }
  if (typeof url !== 'string' || url === '') {
    return { kind: 'refuse', reason: `tool '${name}' error: 'url' must be a non-empty string` };
  }
  const origin = originOf(url);
  if (origin === null) {
    return { kind: 'refuse', reason: `tool '${name}' error: only http(s) URLs can be fetched` };
  }
  if (options.approvedOrigins.has(origin)) return { kind: 'run' };
  // Auto-approve is waived only for a destination that already passed the
  // scheme check above; the server's private-range guard still runs.
  if (options.autoApprove) {
    options.approvedOrigins.add(origin);
    return { kind: 'run' };
  }

  return { kind: 'approve', url, host: new URL(url).hostname, origin };
}

export function execute(
  call: ToolCall,
  options: { advertised: string[]; approvedOrigins: string[]; signal: AbortSignal },
): Promise<ToolCallResult> {
  return callTool({
    name: call.function.name,
    arguments: call.function.arguments,
    advertised: options.advertised,
    approvedOrigins: options.approvedOrigins,
    signal: options.signal,
  });
}

/**
 * Prepended once a tool RESULT is in the history, never merely because a tool
 * is advertised.
 *
 * Attaching it to every first turn taught a small model to answer "I don't
 * have access to external data" to questions that needed none.
 */
export const TOOL_GUIDANCE = [
  'You have just received results from a tool call.',
  'Answer from those results. Do not claim you lack access to real-time or external data — you have it.',
  'Quote concrete figures, dates and names from the results rather than paraphrasing vaguely.',
  'Present lists and records as readable Markdown lists or tables, and state a requested total before a long listing.',
  'If the results do not answer the question, say exactly what is missing.',
  'Never invent a result the tool did not return.',
].join(' ');
