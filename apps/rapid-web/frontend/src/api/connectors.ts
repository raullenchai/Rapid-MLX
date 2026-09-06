import { requestJson } from './client';
import type { ToolDefinition } from './chat';

/**
 * MCP connectors: programs on the Mac that expose tools to the model.
 *
 * The engine spawns and validates them; this package owns the config file it
 * reads. Every mutating call answers the WHOLE snapshot rather than an
 * acknowledgement, so the panel cannot render a server it just toggled beside
 * a tool list from before the reload that followed.
 */

export interface ConnectorServer {
  name: string;
  transport: 'stdio' | 'sse';
  command: string | null;
  args: string[];
  env: Record<string, string>;
  url: string | null;
  enabled: boolean;
  timeout: number;
  /** The command that will run, or the URL that will be contacted. */
  summary: string;
}

/** One server's live state, as the engine reports it. */
export interface EngineServer {
  name: string;
  state: string;
  transport: string;
  tools_count: number;
  error: string | null;
}

export interface ConnectorTool {
  /** Engine-namespaced `server__tool`. */
  name: string;
  description: string;
  server: string;
  parameters: unknown;
}

export interface ConnectorState {
  /** Master switch. Off means the engine has no MCP subsystem at all. */
  enabled: boolean;
  servers: ConnectorServer[];
  /** The config file exists but could not be read, or names an illegal server. */
  load_error: string | null;
  config_path: string;
  engine_servers: EngineServer[];
  engine_reachable: boolean;
  /** MCP could not start AT ALL, as opposed to one server failing. */
  subsystem_error: string | null;
  /** The running child was given a config path. */
  configured: boolean;
  /** Connectors are on, but the running model predates them. */
  needs_restart: boolean;
  engine_running: boolean;
  tools: ConnectorTool[];
  disabled_tools: string[];
  granted_tools: string[];
  auto_approve_all: boolean;
}

export function fetchConnectors(): Promise<ConnectorState> {
  return requestJson<ConnectorState>('/api/connectors');
}

/** The switches. Every field is optional; only what is sent is changed. */
export function updateConnectorSettings(patch: {
  enabled?: boolean;
  auto_approve_all?: boolean;
  tool?: string;
  tool_enabled?: boolean;
  grant?: boolean;
  reset_grants?: boolean;
}): Promise<ConnectorState> {
  return requestJson<ConnectorState>('/api/connectors/settings', {
    method: 'POST',
    body: patch,
  });
}

/** `replacing` is the name being edited, so a rename does not read as a
 *  duplicate of its own old name. */
export function saveConnector(
  server: Omit<ConnectorServer, 'summary'>,
  replacing?: string,
): Promise<ConnectorState> {
  return requestJson<ConnectorState>('/api/connectors/servers', {
    method: 'POST',
    body: { server, ...(replacing === undefined ? {} : { replacing }) },
  });
}

export function removeConnector(name: string): Promise<ConnectorState> {
  return requestJson<ConnectorState>('/api/connectors/servers/remove', {
    method: 'POST',
    body: { name },
  });
}

export function setConnectorEnabled(name: string, enabled: boolean): Promise<ConnectorState> {
  return requestJson<ConnectorState>('/api/connectors/servers/enabled', {
    method: 'POST',
    body: { name, enabled },
  });
}

/** Respawn the loaded model so the child is started WITH `--mcp-config`. */
export function restartForConnectors(): Promise<{ restarting: boolean; model: string }> {
  return requestJson<{ restarting: boolean; model: string }>('/api/connectors/restart', {
    method: 'POST',
    body: {},
  });
}

export function callConnectorTool(body: {
  name: string;
  arguments: string;
  signal: AbortSignal;
}): Promise<{ content: string; is_error: boolean }> {
  return requestJson<{ content: string; is_error: boolean }>('/api/connectors/execute', {
    method: 'POST',
    signal: body.signal,
    body: { name: body.name, arguments: body.arguments },
  });
}

/** A connector tool in the shape the chat request body takes. */
export function toolDefinition(tool: ConnectorTool): ToolDefinition {
  return {
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      // The connector owns the schema; an absent one still has to be a legal
      // object or the engine rejects the whole tools array.
      parameters: tool.parameters ?? { type: 'object', properties: {} },
    },
  };
}
