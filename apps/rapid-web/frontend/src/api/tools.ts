import { requestJson } from './client';
import type { ToolDefinition } from './chat';

/**
 * Tools run on the Mac, not in this browser.
 *
 * Every provider behind them (Open-Meteo, DuckDuckGo, an arbitrary page) is
 * cross-origin and sends no CORS headers, so a fetch from here is blocked
 * before it leaves. The page owns the LOOP — it is what streams the answer —
 * and the server owns each call.
 */

export interface ToolsResponse {
  tools: ToolDefinition[];
  /** Tools the user must approve per call. Server-declared so the page cannot
   *  quietly disagree with what the server actually gates. */
  approval_required: string[];
}

export interface ToolCallResult {
  content: string;
  is_error: boolean;
  /** A redirect left every approved origin. The page prompts and retries with
   *  the new origin added, rather than the server holding a socket open
   *  waiting on a human. */
  needs_approval?: { url: string; host: string };
}

export function fetchTools(): Promise<ToolsResponse> {
  return requestJson<ToolsResponse>('/api/tools');
}

export function callTool(body: {
  name: string;
  arguments: string;
  /** What the model was shown THIS round. The server refuses anything absent
   *  from it — leaving a tool out of the request body does not stop a
   *  malformed model emitting a call for it. */
  advertised: string[];
  approvedOrigins: string[];
  signal: AbortSignal;
}): Promise<ToolCallResult> {
  return requestJson<ToolCallResult>('/api/tools/call', {
    method: 'POST',
    signal: body.signal,
    body: {
      name: body.name,
      arguments: body.arguments,
      advertised: body.advertised,
      approved_origins: body.approvedOrigins,
    },
  });
}
