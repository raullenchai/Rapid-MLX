import { useEffect, useState } from 'react';
import { MoreHorizontal, RotateCw, Wrench } from 'lucide-react';
import {
  fetchConnectors,
  removeConnector,
  restartForConnectors,
  saveConnector,
  setConnectorEnabled,
  updateConnectorSettings,
  type ConnectorServer,
  type ConnectorState,
  type EngineServer,
} from '@/api/connectors';
import { asApiError } from '@/api/errors';
import { displaySafe, shortToolName } from '@/chat/connectors';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { cn } from '@/lib/utils';
import { ConfirmDialog } from '../common/ConfirmDialog';
import {
  PageHeader,
  PanelBody,
  SettingsRow,
  SettingsRowDivider,
  SettingsSection,
} from '../common/SettingsSection';
import { Switch } from '../common/Switch';
import { ConnectorEditor } from './ConnectorEditor';

/**
 * Settings → Connectors.
 *
 * The four things it does that hand-editing the config file cannot: add and
 * remove servers and show whether each actually connected (including why not),
 * list the tools they expose with a per-tool off switch, show and revoke the
 * consent record, and apply an edit without restarting the model — or say
 * plainly when it can't.
 *
 * Every mutation answers the whole state, so there is one fetch shape and the
 * panel can never render a server it just changed beside a tool list from
 * before the reload that followed.
 */
export function ConnectorsPanel() {
  const [state, setState] = useState<ConnectorState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<{ original: ConnectorServer | null } | null>(null);
  const [removing, setRemoving] = useState<ConnectorServer | null>(null);
  const [restarting, setRestarting] = useState(false);

  useEffect(() => {
    let live = true;
    // Reflect reality on open: this is the one place a user comes to ask
    // "did it work?", and a stale list is worse than a blank one.
    fetchConnectors()
      .then((next) => live && setState(next))
      .catch((cause: unknown) => live && setError(asApiError(cause).message));
    return () => {
      live = false;
    };
  }, []);

  /** Every write goes through here, so one failure path serves all of them. */
  const run = (action: () => Promise<ConnectorState>) => {
    setError(null);
    action()
      .then(setState)
      .catch((cause: unknown) => setError(asApiError(cause).message));
  };

  const restart = () => {
    setRestarting(true);
    setError(null);
    restartForConnectors()
      .then(() => fetchConnectors())
      .then(setState)
      .catch((cause: unknown) => setError(asApiError(cause).message))
      .finally(() => setRestarting(false));
  };

  if (state === null) {
    return (
      <PanelBody>
        <PageHeader title="Connectors" />
        {error === null ? (
          <p className="text-muted-foreground text-sm">Loading…</p>
        ) : (
          <InlineNotice tone="error">{error}</InlineNotice>
        )}
      </PanelBody>
    );
  }

  return (
    <PanelBody>
      <PageHeader
        title="Connectors"
        subtitle="Connect the model to MCP servers — programs on this Mac that expose tools like file access, databases or search. Off by default: a connector is a program that runs on your machine and that the model can invoke."
      />

      <SettingsSection>
        <SettingsRow
          title="Enable connectors"
          description="The local server only loads connectors when this is on."
          control={
            <Switch
              label="Enable connectors"
              checked={state.enabled}
              onChange={(enabled) => run(() => updateConnectorSettings({ enabled }))}
            />
          }
        />
      </SettingsSection>

      {state.enabled ? (
        <>
          {state.load_error ? <InlineNotice tone="warning">{state.load_error}</InlineNotice> : null}
          {/* The restart case owns its own banner, and the engine's string is
              suppressed under it on purpose: with no config path the engine
              says "start the server with --mcp-config", which is operator
              language for a state this user reaches without a command line. */}
          {state.needs_restart ? (
            <RestartBanner busy={restarting} onRestart={restart} />
          ) : state.subsystem_error ? (
            <InlineNotice tone="warning">{`Connectors couldn't start: ${state.subsystem_error}`}</InlineNotice>
          ) : null}
          {error ? <InlineNotice tone="error">{error}</InlineNotice> : null}

          <SettingsSection
            title="Servers"
            subtitle="Each server runs as its own program and exposes a set of tools."
            accessory={
              <Button variant="outline" size="sm" onClick={() => setEditing({ original: null })}>
                Add…
              </Button>
            }
          >
            {state.servers.length === 0 ? (
              <p className="text-muted-foreground m-0 text-sm">
                No connectors yet. Add one to give the model tools beyond the built-ins.
              </p>
            ) : (
              state.servers.map((server, index) => (
                <div key={server.name}>
                  {index > 0 ? <SettingsRowDivider /> : null}
                  <ServerRow
                    server={server}
                    status={state.engine_servers.find((row) => row.name === server.name) ?? null}
                    state={state}
                    onToggle={(enabled) =>
                      run(() => setConnectorEnabled(server.name, enabled))
                    }
                    onEdit={() => setEditing({ original: server })}
                    onRemove={() => setRemoving(server)}
                  />
                </div>
              ))
            )}
          </SettingsSection>

          {state.tools.length > 0 ? (
            <SettingsSection
              title="Tools"
              subtitle="What the connected servers expose. Turn one off and it is never offered to the model — and never runs, even if the model asks for it by name."
            >
              {state.tools.map((tool, index) => (
                <div key={tool.name}>
                  {index > 0 ? <SettingsRowDivider /> : null}
                  <ToolRow
                    name={tool.name}
                    description={tool.description}
                    server={tool.server}
                    granted={state.granted_tools.includes(tool.name)}
                    checked={!state.disabled_tools.includes(tool.name)}
                    onChange={(on) =>
                      run(() => updateConnectorSettings({ tool: tool.name, tool_enabled: on }))
                    }
                  />
                </div>
              ))}
            </SettingsSection>
          ) : null}

          <SettingsSection
            title="Approvals"
            subtitle="The first time the model calls a connector tool, Rapid asks. Your answer is remembered per tool."
          >
            <SettingsRow
              title="Auto-approve all tool calls"
              description="Skips every prompt, including for connectors added later. For unattended use only."
              control={
                <Switch
                  label="Auto-approve all tool calls"
                  checked={state.auto_approve_all}
                  onChange={(on) => run(() => updateConnectorSettings({ auto_approve_all: on }))}
                />
              }
            />
            <SettingsRowDivider />
            <SettingsRow
              title={
                state.granted_tools.length === 0
                  ? 'No tools are permanently allowed.'
                  : `${state.granted_tools.length} tool${state.granted_tools.length === 1 ? '' : 's'} permanently allowed.`
              }
              description="Resetting makes Rapid ask again the next time each one is called."
              control={
                <Button
                  variant="outline"
                  size="sm"
                  disabled={state.granted_tools.length === 0}
                  onClick={() => run(() => updateConnectorSettings({ reset_grants: true }))}
                >
                  Reset
                </Button>
              }
            />
          </SettingsSection>

          <p className="text-muted-foreground m-0 font-mono text-[11px] [overflow-wrap:anywhere]">
            {state.config_path}
          </p>
        </>
      ) : null}

      {editing ? (
        <ConnectorEditor
          original={editing.original}
          existingNames={state.servers.map((server) => server.name)}
          onCancel={() => setEditing(null)}
          onSave={(server) => {
            setEditing(null);
            run(() => saveConnector(server, editing.original?.name));
          }}
        />
      ) : null}

      <ConfirmDialog
        open={removing !== null}
        title={`Remove “${removing?.name ?? ''}”?`}
        body="Its tools stop being offered to the model. The program itself isn't uninstalled."
        confirmLabel="Remove"
        onCancel={() => setRemoving(null)}
        onConfirm={() => {
          const target = removing;
          setRemoving(null);
          if (target) run(() => removeConnector(target.name));
        }}
      />
    </PanelBody>
  );
}

/** An inline message about this panel's own state. The notice STACK is for
 *  app-level events; a failure to save a connector belongs beside the form
 *  that caused it, not floating over the transcript behind the window. */
function InlineNotice({ tone, children }: { tone: 'warning' | 'error'; children: string }) {
  return (
    <p
      role={tone === 'error' ? 'alert' : 'status'}
      className={cn(
        'm-0 rounded-lg px-4 py-3 text-xs',
        tone === 'error' ? 'bg-destructive/10 text-destructive' : 'bg-warning/10',
      )}
    >
      {children}
    </p>
  );
}

/**
 * Shown when the running model predates the connectors being switched on.
 *
 * Carries a real button rather than an instruction: telling the user to go
 * find the model picker and cycle it themselves is asking them to do the
 * app's job.
 */
function RestartBanner({ busy, onRestart }: { busy: boolean; onRestart(): void }) {
  return (
    <div className="bg-warning/10 flex items-start gap-3 rounded-lg px-4 py-3">
      <RotateCw className="text-warning mt-0.5 size-4 shrink-0" aria-hidden="true" />
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <span className="text-sm font-medium">
          Restart the model to finish turning connectors on.
        </span>
        <span className="text-muted-foreground text-xs">
          The running model started before connectors were enabled, so it isn't loading them yet.
          Restarting takes a moment and keeps your conversation.
        </span>
      </div>
      <Button variant="outline" size="sm" disabled={busy} onClick={onRestart}>
        {busy ? 'Restarting…' : 'Restart'}
      </Button>
    </div>
  );
}

function ServerRow({
  server,
  status,
  state,
  onToggle,
  onEdit,
  onRemove,
}: {
  server: ConnectorServer;
  status: EngineServer | null;
  state: ConnectorState;
  onToggle(enabled: boolean): void;
  onEdit(): void;
  onRemove(): void;
}) {
  return (
    <div className="flex items-start gap-3">
      <span
        className={cn('mt-1.5 size-2 shrink-0 rounded-full', dotClass(server, status))}
        aria-hidden="true"
      />
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="font-mono text-sm">{server.name}</span>
        <span className="text-muted-foreground text-xs [overflow-wrap:anywhere]">
          {displaySafe(server.summary)}
        </span>
        <span className={cn('text-xs', status?.error ? 'text-warning' : 'text-muted-foreground')}>
          {statusLine(server, status, state)}
        </span>
      </div>
      <Switch label={`Enable ${server.name}`} checked={server.enabled} onChange={onToggle} />
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="text-muted-foreground size-8 shrink-0"
            aria-label={`Actions for ${server.name}`}
          >
            <MoreHorizontal />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onSelect={onEdit}>Edit…</DropdownMenuItem>
          <DropdownMenuItem variant="destructive" onSelect={onRemove}>
            Remove
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}

/** The row's state as one dot: idle, connected, or something went wrong. */
function dotClass(server: ConnectorServer, status: EngineServer | null): string {
  if (!server.enabled || status === null) return 'bg-muted-foreground/40';
  if (status.error !== null || status.state === 'error') return 'bg-warning';
  return status.state === 'connected' ? 'bg-success' : 'bg-muted-foreground/40';
}

/** One line saying what this server is doing right now — the question the
 *  panel exists to answer. */
function statusLine(
  server: ConnectorServer,
  status: EngineServer | null,
  state: ConnectorState,
): string {
  if (!server.enabled) return 'Turned off';
  // A connector's error string can carry its own stderr, so it is scrubbed
  // like every other server-supplied string here.
  if (status?.error) return displaySafe(status.error);
  if (status !== null) {
    if (status.state === 'connected') {
      return `Connected · ${status.tools_count} ${status.tools_count === 1 ? 'tool' : 'tools'}`;
    }
    return status.state.charAt(0).toUpperCase() + status.state.slice(1);
  }
  if (!state.engine_running) return 'Start a model to connect';
  if (!state.engine_reachable) return "Couldn't check — the local server didn't answer";
  return state.needs_restart ? 'Not applied yet' : 'Not connected';
}

function ToolRow({
  name,
  description,
  server,
  granted,
  checked,
  onChange,
}: {
  name: string;
  description: string;
  server: string;
  granted: boolean;
  checked: boolean;
  onChange(next: boolean): void;
}) {
  return (
    <div className="flex items-start gap-2.5">
      <Wrench className="text-muted-foreground mt-0.5 size-4 shrink-0" aria-hidden="true" />
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-sm">{displaySafe(shortToolName(name))}</span>
          {server ? (
            <span className="bg-muted text-muted-foreground rounded-full px-2 py-0.5 text-[11px]">
              {displaySafe(server)}
            </span>
          ) : null}
          {granted ? (
            <span className="text-muted-foreground/70 text-[11px]">always allowed</span>
          ) : null}
        </div>
        <span className="text-muted-foreground text-xs">{displaySafe(description)}</span>
      </div>
      <Switch label={name} checked={checked} onChange={onChange} />
    </div>
  );
}
