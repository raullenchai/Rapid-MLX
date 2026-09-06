import { fetchConnectors, toolDefinition, type ConnectorState } from '@/api/connectors';
import type { ToolCall, ToolDefinition } from '@/api/chat';

/**
 * Connector tools in the chat loop.
 *
 * The engine never injects MCP tools into `/v1/chat/completions` on its own,
 * so the loop stays on this side — which is what lets the consent gate live in
 * the UI, where the decision belongs. Policy is separated from execution for
 * the same reason `chat/tools.ts` is: the gate has to be testable without a
 * stream.
 */

/** The namespace separator the engine builds `server__tool` with. */
const SEPARATOR = '__';

/**
 * The server half of a namespaced name.
 *
 * "Run read_file?" is not a question anyone can answer without knowing whose
 * `read_file` it is.
 */
export function serverOf(toolName: string): string {
  const index = toolName.indexOf(SEPARATOR);
  return index === -1 ? '' : toolName.slice(0, index);
}

/**
 * The tool half.
 *
 * Splits on the FIRST separator, so a tool whose own name contains a double
 * underscore keeps it — the server half is what the engine prefixed and never
 * contains one.
 */
export function shortToolName(toolName: string): string {
  const index = toolName.indexOf(SEPARATOR);
  return index === -1 ? toolName : toolName.slice(index + SEPARATOR.length);
}

/**
 * What the model may be shown this round.
 *
 * Nothing at all while the master switch is off, even if the state still
 * carries tools: the running child keeps its connectors loaded until it is
 * restarted, so "connectors are off" has to be enforced here rather than
 * inferred from an empty list.
 */
export function advertisedConnectorTools(state: ConnectorState | null): ToolDefinition[] {
  if (state === null || !state.enabled) return [];
  const disabled = new Set(state.disabled_tools);
  return state.tools.filter((tool) => !disabled.has(tool.name)).map(toolDefinition);
}

/** Whether `name` is a connector tool rather than one of the built-ins. */
export function isConnectorTool(state: ConnectorState | null, name: string): boolean {
  return state !== null && state.tools.some((tool) => tool.name === name);
}

export type ConnectorGate =
  | { kind: 'run' }
  | { kind: 'approve'; tool: string; server: string; short: string; args: string }
  | { kind: 'refuse'; reason: string };

/**
 * Decide whether one connector call may run.
 *
 * Both checks are load-bearing and neither subsumes the other: a tool the user
 * switched off is refused even though it was never advertised (a malformed
 * model can still emit the name), and a tool that was advertised does not run
 * until it has been approved.
 */
export function gateConnectorCall(
  call: ToolCall,
  state: ConnectorState | null,
): ConnectorGate {
  const name = call.function.name;
  if (state === null || !state.enabled) {
    return {
      kind: 'refuse',
      reason: `Connectors are turned off in Settings → Connectors; '${name}' was not run.`,
    };
  }
  if (state.disabled_tools.includes(name)) {
    return {
      kind: 'refuse',
      reason: `tool '${name}' is turned off in Settings → Connectors and was not run.`,
    };
  }
  if (state.auto_approve_all || state.granted_tools.includes(name)) return { kind: 'run' };

  return {
    kind: 'approve',
    tool: name,
    server: serverOf(name) || 'unknown',
    short: shortToolName(name),
    // The FULL arguments, not a capped preview: the sheet scrolls, and
    // truncating would let whatever is past the cutoff be approved unseen —
    // which is what the gate exists to prevent. Model output is
    // token-bounded, so there is no unbounded size to guard against.
    args: call.function.arguments,
  };
}

/**
 * Escape anything that could disguise what a row actually says.
 *
 * A connector supplies its own tool names, descriptions and error strings, and
 * a bidi override or a zero-width scalar in any of them can make an approval
 * prompt read as a tool the user trusts. Everything server-supplied goes
 * through this before it is shown.
 */
export function displaySafe(text: string): string {
  return escapeUnsafeCharacters(text, new Set());
}

/**
 * Make a connector result safe without flattening its layout.
 *
 * Results are shown in a preformatted tool detail and then passed back to the
 * model. Newlines and tabs are meaningful there: the filesystem server, for
 * example, returns one directory entry per line. Approval metadata still goes
 * through `displaySafe`, where every control character is escaped because a
 * forged line break could disguise what the user is approving.
 */
export function displaySafeResult(text: string): string {
  return escapeUnsafeCharacters(text.replaceAll('\r\n', '\n'), new Set([0x09, 0x0a]));
}

function escapeUnsafeCharacters(text: string, allowedControls: ReadonlySet<number>): string {
  return [...text]
    .map((character) => {
      const code = character.codePointAt(0) ?? 0;
      const printable =
        allowedControls.has(code) ||
        (code >= 0x20 &&
          code !== 0x7f &&
          !(code >= 0x200b && code <= 0x206f) &&
          code !== 0xfeff);
      return printable ? character : `\\u{${code.toString(16).toUpperCase()}}`;
    })
    .join('');
}

/** Pretty-print the model's argument string for the prompt, or pass it back
 *  verbatim when it is not JSON — what will be sent is what must be shown. */
export function formatArguments(raw: string): string {
  const text = raw.trim();
  if (text === '') return '(no arguments)';
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

/**
 * The connector state a turn runs against.
 *
 * Fetched per turn rather than per page: the panel can arm a connector
 * mid-session, and a page-lifetime cache would hide it from the model until
 * a reload.
 */
export async function loadConnectorState(): Promise<ConnectorState | null> {
  try {
    return await fetchConnectors();
  } catch {
    // A server that cannot answer has no connectors to offer; the turn still
    // runs with the built-ins.
    return null;
  }
}
