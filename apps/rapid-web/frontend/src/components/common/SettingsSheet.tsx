import { useEffect, useState, type ReactNode } from 'react';
import {
  CloudSun,
  Globe,
  HardDrive,
  MessageSquareText,
  Palette,
  Plug,
  Search,
  Server,
  SlidersHorizontal,
  Trash2,
  Wrench,
} from 'lucide-react';
import { useStore } from '@/state/store';
import type { Settings } from '@/state/types';
import { loadTools } from '@/chat/tools';
import { EffectiveSystemPrompt } from '@/components/chat/ConversationInstructions';
import type { ToolDefinition } from '@/api/chat';
import { SHEET_DESKTOP_SIZE } from './Sheet';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import { X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { DialogOverlay, DialogPortal } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Segmented } from './Segmented';
import { Switch } from './Switch';
import { InstructionEditor } from './InstructionEditor';
import {
  PageHeader,
  PanelBody,
  SettingsRow,
  SettingsRowDivider,
  SettingsSection,
} from './SettingsSection';
import { ConnectorsPanel } from '@/components/connectors/ConnectorsPanel';
import { ModelManagement } from '@/components/models/ModelManagement';

/**
 * The settings window: a category rail on the left, one panel on the right.
 *
 * The rule that makes it hold together: the panel area is ONE scroll
 * container whose content is swapped, so switching category cannot make the
 * window resize.
 *
 * Below `sm:` the rail becomes a horizontal strip above the panel: a 184px
 * rail beside a phone-width panel leaves nothing for the panel.
 */

export const CATEGORIES = [
  'models',
  'instructions',
  'chat',
  'tools',
  'connectors',
  'appearance',
  'app',
] as const;
export type SettingsCategory = (typeof CATEGORIES)[number];

const CATEGORY_META: Record<
  SettingsCategory,
  { title: string; icon: ReactNode }
> = {
  models: { title: 'Models', icon: <HardDrive /> },
  instructions: { title: 'System Prompt', icon: <MessageSquareText /> },
  chat: { title: 'Chat', icon: <SlidersHorizontal /> },
  tools: { title: 'Tools', icon: <Wrench /> },
  connectors: { title: 'Connectors', icon: <Plug /> },
  appearance: { title: 'Appearance', icon: <Palette /> },
  app: { title: 'Engine', icon: <Server /> },
};

export function SettingsSheet({
  open,
  onClose,
  initialCategory = 'models',
}: {
  open: boolean;
  onClose(): void;
  initialCategory?: SettingsCategory;
}) {
  const [category, setCategory] = useState<SettingsCategory>(initialCategory);

  return (
    <DialogPrimitive.Root
      open={open}
      onOpenChange={(next) => {
        if (!next) onClose();
        // Reopening lands on whichever category the caller asked for, not
        // wherever the last visit ended: the model button and the footer
        // button mean different things.
        else setCategory(initialCategory);
      }}
    >
      <DialogPortal>
        <DialogOverlay className="z-20" />
        <DialogPrimitive.Content
          className={cn(
            'bg-background data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 fixed inset-x-0 bottom-0 z-20 flex max-h-[88dvh] flex-col rounded-t-xl border-t shadow-lg duration-200',
            'data-[state=closed]:slide-out-to-bottom data-[state=open]:slide-in-from-bottom',
            'sm:data-[state=closed]:zoom-out-95 sm:data-[state=open]:zoom-in-95 sm:data-[state=closed]:slide-out-to-bottom-0 sm:data-[state=open]:slide-in-from-bottom-0',
            SHEET_DESKTOP_SIZE,
            'sm:inset-auto sm:top-1/2 sm:left-1/2 sm:-translate-x-1/2 sm:-translate-y-1/2 sm:rounded-xl sm:border',
          )}
          aria-label="Settings"
        >
          <header className="flex shrink-0 items-center gap-2 border-b py-3 pr-3 pl-4">
            <DialogPrimitive.Title className="m-0 min-w-0 flex-1 text-base leading-none font-semibold">
              Settings
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

          <div className="flex min-h-0 flex-1 flex-col sm:flex-row">
            <CategoryRail value={category} onChange={setCategory} />
            {/* One scroll container per window, not per panel: a panel that
                owns its own scroller re-anchors to the top on every switch
                and loses the position of the one being returned to. */}
            <div className="bg-muted/20 min-h-0 flex-1 overflow-y-auto pb-[env(safe-area-inset-bottom)]">
              {category === 'models' ? (
                <ModelManagement open={open} onClose={onClose} />
              ) : category === 'instructions' ? (
                <InstructionsPanel />
              ) : category === 'chat' ? (
                <ChatPanel />
              ) : category === 'tools' ? (
                <ToolsPanel />
              ) : category === 'connectors' ? (
                <ConnectorsPanel />
              ) : category === 'appearance' ? (
                <AppearancePanel />
              ) : (
                <EnginePanel />
              )}
            </div>
          </div>
        </DialogPrimitive.Content>
      </DialogPortal>
    </DialogPrimitive.Root>
  );
}

function CategoryRail({
  value,
  onChange,
}: {
  value: SettingsCategory;
  onChange(next: SettingsCategory): void;
}) {
  return (
    <nav
      className="bg-muted/40 flex shrink-0 gap-1 overflow-x-auto border-b p-2 sm:w-[184px] sm:flex-col sm:overflow-x-visible sm:overflow-y-auto sm:border-r sm:border-b-0"
      aria-label="Settings categories"
    >
      {CATEGORIES.map((category) => {
        const meta = CATEGORY_META[category];
        const selected = category === value;
        return (
          <button
            key={category}
            type="button"
            // `aria-current`, not `aria-selected`: these are navigation
            // buttons, and `aria-selected` is only meaningful inside a
            // tablist/listbox role this deliberately does not claim.
            aria-current={selected ? 'page' : undefined}
            className={cn(
              'flex shrink-0 items-center gap-2 rounded-md px-3 py-2 text-left text-sm whitespace-nowrap transition-colors outline-none',
              'focus-visible:ring-ring/50 focus-visible:ring-[3px]',
              selected
                ? 'bg-background text-foreground font-semibold shadow-xs'
                : 'text-muted-foreground hover:bg-accent hover:text-foreground font-medium',
              'sm:w-full',
            )}
            onClick={() => onChange(category)}
          >
            <span className="[&_svg]:size-4 [&_svg]:shrink-0">{meta.icon}</span>
            {meta.title}
          </button>
        );
      })}
    </nav>
  );
}

/**
 * Mirrors `rapid-mac`'s Settings → System Prompt, including its one deviation
 * from the house style: the instruction section draws NO grouped card,
 * because the editor already defines its own surface and a box around a box
 * adds weight without structure. Clear sits on the heading's trailing edge for
 * the same reason it does there — an action on the section, not a control
 * adrift in the whitespace beneath it.
 *
 * The GLOBAL layer only. A conversation's own prompt is a property of the
 * chat, so it lives on the composer — see chat/ConversationInstructions.tsx.
 */
function InstructionsPanel() {
  const system = useStore((state) => state.settings.system);
  const update = useStore((state) => state.updateSettings);

  return (
    <PanelBody>
      <PageHeader
        title="System Prompt"
        subtitle="Sent as a system message with every conversation. A conversation can add its own from the composer, which wins where the two conflict."
      />
      <SettingsSection
        flat
        title="Global default"
        subtitle="Used by every conversation on this device. Stored only in this browser."
        accessory={
          <Button
            variant="ghost"
            size="icon"
            className="text-muted-foreground size-8"
            disabled={system.trim() === ''}
            onClick={() => update({ system: '' })}
            aria-label="Clear system prompt"
            title="Clear"
          >
            <Trash2 />
          </Button>
        }
      >
        <InstructionEditor
          id="system"
          label="System prompt"
          className="min-h-44"
          value={system}
          onChange={(next) => update({ system: next })}
          placeholder="For example: Answer concisely, use plain language, and include code examples when useful."
        />
      </SettingsSection>

      <EffectiveSystemPrompt global={system} conversation="" />
    </PanelBody>
  );
}

function ChatPanel() {
  const settings = useStore((state) => state.settings);
  const update = useStore((state) => state.updateSettings);

  return (
    <PanelBody>
      <PageHeader
        title="Chat"
        subtitle="How replies are sampled. These shape what the model says, not how fast it says it."
      />
      <SettingsSection title="Sampling">
        <SliderField
          id="temperature"
          label="Temperature"
          description="Lower is more deterministic; higher is more varied."
          value={settings.temperature}
          min={0}
          max={2}
          step={0.05}
          format={(value) => value.toFixed(2)}
          onChange={(temperature) => update({ temperature })}
        />

        <SettingsRowDivider />

        <SliderField
          id="top-p"
          label="Top P"
          description="Nucleus sampling cutoff."
          value={settings.topP}
          min={0.05}
          max={1}
          step={0.05}
          format={(value) => value.toFixed(2)}
          onChange={(topP) => update({ topP })}
        />

        <SettingsRowDivider />

        <SliderField
          id="max-tokens"
          label="Max tokens"
          description="Upper bound on the length of a single reply."
          value={settings.maxTokens}
          min={256}
          max={16384}
          step={256}
          format={(value) => String(value)}
          onChange={(maxTokens) => update({ maxTokens })}
        />
      </SettingsSection>
    </PanelBody>
  );
}

/**
 * Presentation for a tool's identity. The wire identifiers are what the
 * request body, the enabled set and the dispatch gate all use; none of that
 * is touched by anything here.
 *
 * A tool with no entry falls back to its own name and the engine-facing
 * description, so one added on the server still shows something true.
 */
const TOOL_DISPLAY: Record<string, { title: string; summary: string; icon: ReactNode }> = {
  web_search: {
    title: 'Web Search',
    summary: 'Looks up current information on the web when a question needs it.',
    icon: <Search />,
  },
  browse: {
    title: 'Browse Web Page',
    summary: 'Opens a web page you or the model names and reads it. You approve each page.',
    icon: <Globe />,
  },
  weather: {
    title: 'Weather',
    summary: 'Gets the current weather for a place you name.',
    icon: <CloudSun />,
  },
};

/**
 * Settings → Tools.
 *
 * The row leads with a HUMAN name, not the wire identifier: `web_search` in a
 * monospaced face is an implementation detail presented as a title, and the
 * description under it is written FOR THE MODEL — it carries calling
 * conventions and pagination offsets, so it reads as documentation rather
 * than as a setting. It moves behind a disclosure, with the identifier beside
 * it, because someone debugging a prompt needs both.
 *
 * The list itself comes from the server: it owns the tools, and a second copy
 * here would drift.
 */
function ToolsPanel() {
  const enabled = useStore((state) => state.settings.enabledTools);
  const autoApprove = useStore((state) => state.settings.autoApproveBrowsing);
  const update = useStore((state) => state.updateSettings);
  const [tools, setTools] = useState<ToolDefinition[] | null>(null);
  const [approvalRequired, setApprovalRequired] = useState<Set<string>>(new Set());

  useEffect(() => {
    let live = true;
    loadTools()
      .then((catalogue) => {
        if (!live) return;
        setTools(catalogue.tools);
        setApprovalRequired(catalogue.approvalRequired);
      })
      .catch(() => live && setTools([]));
    return () => {
      live = false;
    };
  }, []);

  const toggle = (name: string, on: boolean) => {
    const next = enabled.filter((item) => item !== name);
    update({ enabledTools: on ? [...next, name] : next });
  };

  const gated = tools?.filter((tool) => approvalRequired.has(tool.function.name)) ?? [];

  return (
    <PanelBody>
      <PageHeader
        title="Tools"
        subtitle="Tools the model can call during a chat. Turn one off and it is never offered — and never runs, even if the model asks for it by name."
      />
      {tools === null ? (
        <p className="text-muted-foreground text-sm">Loading…</p>
      ) : tools.length === 0 ? (
        <p className="text-muted-foreground text-sm">This server exposes no tools.</p>
      ) : (
        <>
          <SettingsSection title="Available tools">
            {tools.map((tool, index) => (
              <div key={tool.function.name}>
                {index > 0 ? <SettingsRowDivider /> : null}
                <ToolRow
                  definition={tool}
                  checked={enabled.includes(tool.function.name)}
                  onChange={(on) => toggle(tool.function.name, on)}
                />
              </div>
            ))}
          </SettingsSection>

          {gated.length > 0 ? (
            <SettingsSection
              title="Browsing"
              subtitle="These fetch a page and hand its text to the model. The model picks the URL, so by default you approve each destination first."
            >
              <SettingsRow
                title="Approve every page automatically"
                description="Skips the confirmation for unattended use. Private and local addresses stay blocked either way."
                control={
                  <Switch
                    label="Approve every page automatically"
                    checked={autoApprove}
                    onChange={(on) => update({ autoApproveBrowsing: on })}
                  />
                }
              />
              <SettingsRowDivider />
              <p className="text-muted-foreground m-0 text-xs">
                An approval covers that host for the rest of the answer, so reading several pages
                on one site asks once.
              </p>
            </SettingsSection>
          ) : null}

          <SettingsSection
            title="Where they run"
            subtitle="On the Mac serving this page, not in this browser — a browser cannot reach these providers directly."
          >
            <p className="text-muted-foreground m-0 text-xs">
              At most 3 calls answer one message. After that the model has to reply from what it
              already has.
            </p>
          </SettingsSection>
        </>
      )}
    </PanelBody>
  );
}

/** One tool: switch, human summary, and the engine-facing text on request. */
function ToolRow({
  definition,
  checked,
  onChange,
}: {
  definition: ToolDefinition;
  checked: boolean;
  onChange(next: boolean): void;
}) {
  const name = definition.function.name;
  const display = TOOL_DISPLAY[name];

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-start justify-between gap-4">
        <div className="flex min-w-0 flex-1 items-start gap-2.5">
          <span className="text-muted-foreground mt-px shrink-0 [&_svg]:size-4" aria-hidden="true">
            {display?.icon ?? <Wrench />}
          </span>
          <div className="flex min-w-0 flex-col gap-1">
            <span className="text-sm leading-none font-medium">{display?.title ?? name}</span>
            <span className="text-muted-foreground text-xs">
              {display?.summary ?? definition.function.description}
            </span>
          </div>
        </div>
        <Switch label={name} checked={checked} onChange={onChange} />
      </div>

      {/* Indented to the text column so it lines up under the summary it
          expands, not under the glyph. */}
      <details className="pl-[26px]">
        <summary className="text-muted-foreground w-fit cursor-pointer text-xs">Details</summary>
        <div className="bg-muted/50 mt-1.5 flex flex-col gap-1.5 rounded-md p-3">
          {/* Exactly the string the model receives. */}
          <p className="text-muted-foreground m-0 text-xs">{definition.function.description}</p>
          <code className="text-muted-foreground/70 text-[11px]">{name}</code>
        </div>
      </details>
    </div>
  );
}

function AppearancePanel() {
  const settings = useStore((state) => state.settings);
  const update = useStore((state) => state.updateSettings);

  return (
    <PanelBody>
      <PageHeader
        title="Appearance"
        subtitle="Override the system theme. Auto follows your device's light or dark setting; Light and Dark stay put regardless of it."
      />
      <SettingsSection>
        <SettingsRow
          title="Theme"
          description="Auto follows your device's light or dark setting."
          control={
            <Segmented<Settings['theme']>
              label="theme"
              value={settings.theme}
              options={[
                { value: 'auto', label: 'Auto' },
                { value: 'light', label: 'Light' },
                { value: 'dark', label: 'Dark' },
              ]}
              onChange={(theme) => update({ theme })}
            />
          }
        />

        <SettingsRowDivider />

        <SettingsRow
          title="Maths"
          description="Switch to source if formulas render as run-together text."
          control={
            <Segmented<Settings['mathRendering']>
              label="math"
              value={settings.mathRendering}
              options={[
                { value: 'mathml', label: 'Typeset' },
                { value: 'source', label: 'Source' },
              ]}
              onChange={(mathRendering) => update({ mathRendering })}
            />
          }
        />
      </SettingsSection>
    </PanelBody>
  );
}

/**
 * What the server is running. `rapid-mac`'s equivalent panel is about the
 * .app's own self-update, which a browser has no counterpart to — what is
 * left, and what a phone actually needs, is the state of the engine it is
 * talking to.
 */
function EnginePanel() {
  const status = useStore((state) => state.status);
  const canSwitch = useStore((state) => state.canSwitch);
  const allowDownloads = useStore((state) => state.allowDownloads);

  return (
    <PanelBody>
      <PageHeader
        title="Engine"
        subtitle="The Rapid-MLX server this page is talking to. It runs on the Mac, not in this browser."
      />
      <SettingsSection title="Status">
        <ValueRow label="Model" value={status?.model ?? 'no model'} />
        <SettingsRowDivider />
        <ValueRow label="State" value={status?.state ?? 'unreachable'} />
        <SettingsRowDivider />
        <ValueRow label="Port" value={status?.port === null ? '—' : String(status?.port ?? '—')} />
        {status?.detail ? (
          <>
            <SettingsRowDivider />
            <ValueRow label="Detail" value={status.detail} />
          </>
        ) : null}
      </SettingsSection>

      <SettingsSection
        title="This server"
        subtitle="Set on the Mac when the server was started; not changeable from here."
      >
        <ValueRow label="Model switching" value={canSwitch ? 'allowed' : 'off'} />
        <SettingsRowDivider />
        <ValueRow label="Downloads" value={allowDownloads ? 'allowed' : 'off'} />
      </SettingsSection>
    </PanelBody>
  );
}

function ValueRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-4">
      <span className="text-muted-foreground w-28 shrink-0 text-xs">{label}</span>
      <span className="min-w-0 flex-1 font-mono text-xs [overflow-wrap:anywhere]">{value}</span>
    </div>
  );
}

function SliderField({
  id,
  label,
  description,
  value,
  min,
  max,
  step,
  format,
  onChange,
}: {
  id: string;
  label: string;
  description: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format(value: number): string;
  onChange(value: number): void;
}) {
  return (
    <div className="flex flex-col gap-2">
      <Label htmlFor={id}>
        {label}
        <span className="text-muted-foreground font-mono text-xs font-normal">{format(value)}</span>
      </Label>
      <Slider
        id={id}
        aria-label={label}
        className="py-2"
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={([next]) => next !== undefined && onChange(next)}
      />
      <span className="text-muted-foreground text-xs">{description}</span>
    </div>
  );
}
