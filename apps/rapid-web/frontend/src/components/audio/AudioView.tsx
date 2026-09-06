import { useMemo, useState } from 'react';
import { Check, ChevronsUpDown, CircleCheck, CircleDashed, Loader2 } from 'lucide-react';
import type { ModelEntry } from '@/api/types';
import { asApiError } from '@/api/errors';
import { useStore } from '@/state/store';
import { startModel } from '@/state/startModel';
import { audioModels, modelDetails, preferredAlias } from '@/audio/models';
import { Segmented } from '@/components/common/Segmented';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { READING_COLUMN } from '@/lib/layout';
import { cn } from '@/lib/utils';
import { SpeechPanel } from './SpeechPanel';
import { DictationPanel } from './DictationPanel';

/**
 * Speech, in two modes — mirroring `rapid-mac`'s `AudioView`.
 *
 * Readiness follows `ServerManager.ensureVoiceLane`, and the order matters:
 *
 *  1. Something is already serving → the lane rides on it. The child is
 *     spawned with `--enable-audio` and the engine's gate short-circuits on
 *     that flag before it looks at the model, so speech works from a CHAT
 *     model and nothing needs to be switched.
 *  2. Nothing is serving → the audio model is started as the served model.
 *     `rapid-mlx serve <audio-alias>` has a dedicated fork in the CLI, so
 *     this is a real path, not a workaround.
 *
 * So the page NEVER dead-ends on "start something else first" — the setup
 * card's Model row carries the action, exactly as the Mac's readiness banner
 * does. Switching is offered only when the engine is idle: tearing down a
 * live chat model to run a 400 MB voice model is the one thing this must not
 * do on its own.
 */

type Mode = 'speech' | 'dictation';

export function AudioView() {
  const [mode, setMode] = useState<Mode>('speech');

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className={cn(READING_COLUMN, 'w-full px-4 pt-3')}>
        <Segmented<Mode>
          label="Audio mode"
          className="w-full"
          value={mode}
          options={[
            { value: 'speech', label: 'Text to Speech' },
            { value: 'dictation', label: 'Speech to Text' },
          ]}
          onChange={setMode}
        />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className={cn(READING_COLUMN, 'w-full p-4')}>
          {mode === 'speech' ? <SpeechPanel /> : <DictationPanel />}
        </div>
      </div>
    </div>
  );
}

/**
 * Whether the lane can take a request, and if not, what to do about it.
 *
 * The two-branch shape is `ServerManager.ensureVoiceLane`: a serving engine
 * co-loads speech, an idle one has to be given the voice model. `blocked` is
 * the one case with no action — a live chat model is not something this page
 * may tear down by itself.
 */
export type LaneState =
  | { kind: 'live'; detail: string }
  | { kind: 'starting'; detail: string }
  | { kind: 'idle'; detail: string }
  | { kind: 'missing'; detail: string }
  | { kind: 'blocked'; detail: string };

export function laneState(
  status: { state: string; model: string | null } | null,
  entry: ModelEntry | null,
  catalogLoaded: boolean,
): LaneState {
  if (status?.state === 'starting') {
    return {
      kind: 'starting',
      detail: `${status.model ?? 'A model'} is loading. Speech will work once it is ready.`,
    };
  }

  // Rule 1: ANY ready model mounts `/v1/audio/*`, so the lane is live —
  // including from a chat model, which is the common case and the whole
  // reason audio is not a model switch.
  if (status?.state === 'ready' && status.model !== null) {
    if (entry !== null && status.model === entry.alias) {
      return { kind: 'live', detail: 'Running as the served model.' };
    }
    return { kind: 'blocked', detail: `Speech is running on ${status.model}.` };
  }

  // Rule 2: nothing is serving, so this model can be started as one.
  if (entry === null) {
    return {
      kind: 'idle',
      detail: catalogLoaded
        ? 'No audio model in the catalog. Pull one on the Mac.'
        : 'Reading the catalog…',
    };
  }
  if (!entry.cached) {
    return {
      kind: 'missing',
      detail: `Not downloaded. Pull it on the Mac with \`rapid-mlx pull ${entry.hf_path}\`.`,
    };
  }
  return { kind: 'idle', detail: 'The engine is idle — start this model to use speech.' };
}

/** Whether the lane will actually answer a request right now. */
export function laneIsLive(state: LaneState): boolean {
  return state.kind === 'live' || state.kind === 'blocked';
}

/**
 * The setup card, shared by both tabs.
 *
 * `rapid-mac`'s grammar: one bordered card, hairline-separated rows, each row
 * a readiness circle + label + caption on the left and its control on a shared
 * trailing line. One design language across the two tabs, so nothing on this
 * page invents its own alignment.
 */
export function SetupCard({ children }: { children: React.ReactNode }) {
  return <div className="divide-y rounded-xl border">{children}</div>;
}

export function SetupRow({
  label,
  caption,
  done,
  control,
}: {
  label: string;
  caption: React.ReactNode;
  /** Drives the circle. `undefined` means the row is a setting, not a step. */
  done?: boolean | undefined;
  control: React.ReactNode;
}) {
  return (
    <div className="flex items-start gap-3 p-3.5">
      {done === undefined ? null : done ? (
        <CircleCheck className="text-success mt-0.5 size-4 shrink-0" aria-hidden="true" />
      ) : (
        <CircleDashed className="text-muted-foreground mt-0.5 size-4 shrink-0" aria-hidden="true" />
      )}
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <span className="text-sm leading-none font-medium">{label}</span>
        <span className="text-muted-foreground text-xs leading-relaxed">{caption}</span>
      </div>
      <div className="flex shrink-0 items-center gap-1.5">{control}</div>
    </div>
  );
}

/**
 * The audio model picker.
 *
 * Lists every alias of its kind, downloaded or not, unlike the composer's
 * picker — a voice model that is absent is the thing the user most needs to
 * SEE, since this surface cannot pull it. Each row carries the product name
 * and a one-line summary rather than the bare registry alias.
 */
export function AudioModelPicker({
  label,
  models,
  selected,
  onSelect,
  disabled,
}: {
  label: string;
  models: ModelEntry[];
  selected: ModelEntry | null;
  onSelect(alias: string): void;
  disabled?: boolean;
}) {
  const title = selected ? modelDetails(selected).displayName : 'Choose…';

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          disabled={disabled || models.length === 0}
          // Carries the VALUE, matching `ComposerModelPicker`'s
          // `Model: <alias>`: a bare "Model" overrides the visible text, so a
          // screen reader would announce the control without its selection.
          aria-label={`${label}: ${title}`}
          className="hover:bg-accent flex w-[min(15rem,45vw)] items-center justify-between gap-1.5 rounded-md border px-2.5 py-1.5 text-sm transition-colors outline-none focus-visible:ring-ring/50 focus-visible:ring-[3px] disabled:opacity-50"
        >
          <span className="min-w-0 truncate">{title}</span>
          <ChevronsUpDown className="text-muted-foreground size-3.5 shrink-0" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="max-h-[50dvh] w-[min(26rem,90vw)] overflow-y-auto">
        {models.map((model) => {
          const details = modelDetails(model);
          return (
            <DropdownMenuItem
              key={model.alias}
              className="items-start gap-2"
              onSelect={() => onSelect(model.alias)}
            >
              <Check
                className={cn(
                  'mt-0.5 size-4 shrink-0',
                  model.alias !== selected?.alias && 'opacity-0',
                )}
              />
              <span className="flex min-w-0 flex-1 flex-col gap-1">
                <span className="flex flex-wrap items-center gap-1.5">
                  <span className="truncate text-sm">{details.displayName}</span>
                  <Badge variant="secondary">{details.badge}</Badge>
                  <Badge variant={model.cached ? 'default' : 'outline'} className="rounded-full">
                    {model.cached ? 'on disk' : 'remote'}
                  </Badge>
                </span>
                <span className="text-muted-foreground text-xs whitespace-normal">
                  {details.summary}
                </span>
              </span>
            </DropdownMenuItem>
          );
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/**
 * The Model row, shared by both tabs.
 *
 * Carries the action as well as the choice — this is the Mac's readiness
 * banner folded into the row it describes, so the page never sends the user
 * somewhere else to make speech work.
 */
export function ModelRow({
  model,
  disabled,
  onError,
}: {
  model: ReturnType<typeof useAudioModel>;
  disabled?: boolean;
  onError(message: string): void;
}) {
  return (
    <SetupRow
      label="Model"
      caption={modelCaption(model.lane)}
      done={model.live}
      control={
        <>
          <AudioModelPicker
            label="Model"
            models={model.entries}
            selected={model.entry}
            onSelect={model.select}
            disabled={disabled || model.starting}
          />
          {model.canStart ? (
            <Button size="sm" onClick={() => void model.start(onError)}>
              Start
            </Button>
          ) : model.starting || model.lane.kind === 'starting' ? (
            <Loader2 className="text-muted-foreground size-4 animate-spin" aria-hidden="true" />
          ) : null}
        </>
      }
    />
  );
}

/**
 * The selected model of one audio kind, and what to say about it.
 *
 * Selection lives here rather than in the store because it is not a served
 * model — nothing outside this surface acts on it. It resolves once the
 * catalog arrives, preferring something already on disk over any built-in
 * preference: a default that is not downloaded opens the panel onto a model
 * the user cannot run.
 */
export function useAudioModel(kind: 'tts' | 'stt', preferred: string[], fallback: string) {
  const models = useStore((state) => state.models);
  const catalogLoaded = useStore((state) => state.catalogLoaded);
  const status = useStore((state) => state.status);
  const canSwitch = useStore((state) => state.canSwitch);
  const entries = useMemo(() => audioModels(models, kind), [models, kind]);
  const [chosen, setChosen] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  // Derived, not resolved in an effect: the default follows the catalog as it
  // arrives, and a choice the user made survives every later refresh.
  const alias = chosen ?? preferredAlias(entries, preferred);
  const entry = entries.find((candidate) => candidate.alias === alias) ?? null;
  const lane = laneState(status, entry, catalogLoaded);

  return {
    entries,
    entry,
    /** What to send. Falls back to a known alias when the catalog has no audio
     *  rows at all — an `--attach` server has no catalog, and the lane still
     *  works there. */
    alias: alias ?? fallback,
    select: setChosen,
    lane,
    /** Whether the panel's controls should be usable at all. */
    live: laneIsLive(lane),
    /** Offered only when the engine is IDLE. With a chat model up the lane
     *  already works, and switching would tear that model down for nothing. */
    canStart: lane.kind === 'idle' && entry !== null && canSwitch && !starting,
    starting,
    async start(onError: (message: string) => void) {
      if (entry === null) return;
      setStarting(true);
      try {
        // Select first, so the status poll that follows is read against the
        // alias being started rather than the previous one.
        useStore.getState().selectAlias('audio', entry.alias);
        await startModel(entry.alias);
      } catch (cause) {
        onError(asApiError(cause).message);
      } finally {
        setStarting(false);
      }
    },
  };
}

/**
 * The Model row's caption.
 *
 * States the LANE, not the model: "what will happen if I press Generate" is
 * the question here, and the model's own summary does not answer it. The
 * summary is in the picker, on every row, where it belongs.
 */
export function modelCaption(lane: LaneState): string {
  return lane.detail;
}
