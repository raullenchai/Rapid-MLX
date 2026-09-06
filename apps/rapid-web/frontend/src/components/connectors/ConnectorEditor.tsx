import { useState } from 'react';
import { X } from 'lucide-react';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import type { ConnectorServer } from '@/api/connectors';
import { Button } from '@/components/ui/button';
import { DialogOverlay, DialogPortal } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { cn } from '@/lib/utils';
import { Segmented } from '../common/Segmented';

/**
 * Add or edit one connector.
 *
 * Validation is repeated here rather than left to the server, because the
 * point is to tell the user in the field they are typing in. The server's
 * copy is the one that decides — this only decides whether Save is worth
 * pressing.
 */

/** Mirrors the server's `MAX_NAME_LENGTH`, and for the same reason: the name
 *  becomes the namespace half of every `server__tool`. */
const MAX_NAME_LENGTH = 32;

type Draft = Omit<ConnectorServer, 'summary'>;

const BLANK: Draft = {
  name: '',
  transport: 'stdio',
  command: '',
  args: [],
  env: {},
  url: '',
  enabled: true,
  timeout: 30,
};

/** Why this draft cannot be saved, or null. Mirrors `connectors.py`'s
 *  `validation_error` so the two say the same thing. */
export function draftError(draft: Draft, existingNames: string[]): string | null {
  const name = draft.name.trim();
  if (name === '') return 'Give this connector a name.';
  if (name.length > MAX_NAME_LENGTH || name.includes('__') || !/^[A-Za-z0-9_-]+$/.test(name)) {
    return `Use up to ${MAX_NAME_LENGTH} letters, numbers, dashes or underscores — the name becomes part of every tool name.`;
  }
  if (existingNames.includes(name)) return `A connector named “${name}” already exists.`;
  if (draft.transport === 'stdio') {
    if ((draft.command ?? '').trim() === '') return 'A command connector needs a command to run.';
  } else {
    const raw = (draft.url ?? '').trim();
    if (raw === '') return 'A URL connector needs a URL.';
    try {
      const parsed = new URL(raw);
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        return 'Enter an http:// or https:// URL.';
      }
    } catch {
      return 'Enter an http:// or https:// URL.';
    }
  }
  if (!Number.isFinite(draft.timeout) || draft.timeout <= 0) {
    return 'Timeout must be greater than zero.';
  }
  return null;
}

export function ConnectorEditor({
  original,
  existingNames,
  onSave,
  onCancel,
}: {
  /** null when adding. */
  original: ConnectorServer | null;
  existingNames: string[];
  onSave(server: Draft): void;
  onCancel(): void;
}) {
  const [draft, setDraft] = useState<Draft>(() =>
    original === null
      ? BLANK
      : {
          name: original.name,
          transport: original.transport,
          command: original.command ?? '',
          args: original.args,
          env: original.env,
          url: original.url ?? '',
          enabled: original.enabled,
          timeout: original.timeout,
        },
  );
  // One line per argument. Args are a list on the wire, but a user types a
  // command line — and a single space-separated field would split a path
  // containing a space into two arguments.
  const [argsText, setArgsText] = useState(() => (original?.args ?? []).join('\n'));
  const [envText, setEnvText] = useState(() =>
    Object.entries(original?.env ?? {})
      .map(([key, value]) => `${key}=${value}`)
      .join('\n'),
  );

  const args = argsText
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line !== '');
  const env = Object.fromEntries(
    envText
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.includes('='))
      .map((line) => {
        const index = line.indexOf('=');
        return [line.slice(0, index).trim(), line.slice(index + 1).trim()];
      }),
  );

  const candidate: Draft = {
    ...draft,
    name: draft.name.trim(),
    command: (draft.command ?? '').trim(),
    url: (draft.url ?? '').trim(),
    args,
    env,
  };
  // A rename is checked against every OTHER name, never its own old one.
  const error = draftError(
    candidate,
    existingNames.filter((name) => name !== original?.name),
  );

  return (
    <DialogPrimitive.Root open onOpenChange={(next) => !next && onCancel()}>
      <DialogPortal>
        {/* Above the settings window it is raised from. */}
        <DialogOverlay className="z-30" />
        <DialogPrimitive.Content
          className={cn(
            'bg-background fixed inset-x-0 bottom-0 z-30 flex max-h-[88dvh] flex-col rounded-t-xl border-t shadow-lg',
            'sm:inset-auto sm:top-1/2 sm:left-1/2 sm:h-auto sm:w-[min(480px,100%-48px)] sm:-translate-x-1/2 sm:-translate-y-1/2 sm:rounded-xl sm:border',
          )}
          aria-label={original === null ? 'Add connector' : `Edit ${original.name}`}
        >
          <header className="flex shrink-0 items-center gap-2 border-b py-3 pr-3 pl-4">
            <DialogPrimitive.Title className="m-0 min-w-0 flex-1 text-base leading-none font-semibold">
              {original === null ? 'Add connector' : 'Edit connector'}
            </DialogPrimitive.Title>
            <DialogPrimitive.Close asChild>
              <Button
                variant="ghost"
                size="icon"
                className="text-muted-foreground size-8"
                aria-label="Close"
                title="Close"
              >
                <X />
              </Button>
            </DialogPrimitive.Close>
          </header>

          <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-4">
            <Field
              id="connector-name"
              label="Name"
              hint="Becomes the prefix of every tool this server exposes."
            >
              <Input
                id="connector-name"
                value={draft.name}
                autoCapitalize="off"
                autoCorrect="off"
                spellCheck={false}
                onChange={(event) => setDraft({ ...draft, name: event.target.value })}
              />
            </Field>

            <Field id="connector-transport" label="Type">
              <Segmented<'stdio' | 'sse'>
                label="connector type"
                value={draft.transport}
                options={[
                  { value: 'stdio', label: 'Command' },
                  { value: 'sse', label: 'URL' },
                ]}
                onChange={(transport) => setDraft({ ...draft, transport })}
              />
            </Field>

            {draft.transport === 'stdio' ? (
              <>
                <Field
                  id="connector-command"
                  label="Command"
                  hint="Must be one the engine allows — npx, uvx, python and the official MCP servers."
                >
                  <Input
                    id="connector-command"
                    value={draft.command ?? ''}
                    placeholder="npx"
                    autoCapitalize="off"
                    autoCorrect="off"
                    spellCheck={false}
                    onChange={(event) => setDraft({ ...draft, command: event.target.value })}
                  />
                </Field>
                <Field id="connector-args" label="Arguments" hint="One per line.">
                  <textarea
                    id="connector-args"
                    className="border-input bg-transparent focus-visible:ring-ring/50 min-h-24 rounded-md border px-3 py-2 font-mono text-base focus-visible:ring-[3px] focus-visible:outline-none sm:text-sm"
                    value={argsText}
                    spellCheck={false}
                    onChange={(event) => setArgsText(event.target.value)}
                  />
                </Field>
                <Field
                  id="connector-env"
                  label="Environment"
                  hint="One KEY=value per line. Optional."
                >
                  <textarea
                    id="connector-env"
                    className="border-input bg-transparent focus-visible:ring-ring/50 min-h-16 rounded-md border px-3 py-2 font-mono text-base focus-visible:ring-[3px] focus-visible:outline-none sm:text-sm"
                    value={envText}
                    spellCheck={false}
                    onChange={(event) => setEnvText(event.target.value)}
                  />
                </Field>
              </>
            ) : (
              <Field id="connector-url" label="URL">
                <Input
                  id="connector-url"
                  value={draft.url ?? ''}
                  placeholder="http://localhost:3001/sse"
                  inputMode="url"
                  autoCapitalize="off"
                  autoCorrect="off"
                  spellCheck={false}
                  onChange={(event) => setDraft({ ...draft, url: event.target.value })}
                />
              </Field>
            )}

            <Field id="connector-timeout" label="Timeout" hint="Seconds a single call may take.">
              <Input
                id="connector-timeout"
                type="number"
                inputMode="numeric"
                min={1}
                value={String(draft.timeout)}
                onChange={(event) =>
                  setDraft({ ...draft, timeout: Number(event.target.value) })
                }
              />
            </Field>

            {/* Only once something has been typed: an empty form telling the
                user it is empty is noise. */}
            {error !== null && draft.name.trim() !== '' ? (
              <p className="text-destructive m-0 text-xs">{error}</p>
            ) : null}
          </div>

          <footer className="flex shrink-0 justify-end gap-2 border-t p-3 pb-[calc(env(safe-area-inset-bottom)+12px)]">
            <Button variant="ghost" onClick={onCancel}>
              Cancel
            </Button>
            <Button disabled={error !== null} onClick={() => onSave(candidate)}>
              Save
            </Button>
          </footer>
        </DialogPrimitive.Content>
      </DialogPortal>
    </DialogPrimitive.Root>
  );
}

function Field({
  id,
  label,
  hint,
  children,
}: {
  id: string;
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label htmlFor={id}>{label}</Label>
      {children}
      {hint ? <span className="text-muted-foreground text-xs">{hint}</span> : null}
    </div>
  );
}
