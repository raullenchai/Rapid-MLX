import { useCallback, useEffect, useRef, useState } from 'react';
import { Loader2, Mic, Plus, Square, X } from 'lucide-react';
import { transcribe } from '@/api/audio';
import { asApiError } from '@/api/errors';
import { useStore } from '@/state/store';
import { noticeFor } from '@/state/notices';
import { PREFERRED_STT } from '@/audio/models';
import { Recorder, RecorderError, formatDuration } from '@/audio/recorder';
import {
  ACTIVE_LIMIT,
  activeCount,
  addHistory,
  addTerm,
  loadHistory,
  loadVocabulary,
  removeTerm,
  saveHistory,
  saveVocabulary,
  setTermActive,
  vocabularyContext,
  type HistoryEntry,
  type Term,
} from '@/audio/dictation';
import { newId } from '@/lib/ids';
import { CopyButton } from '@/components/common/CopyButton';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { ModelRow, SetupCard, useAudioModel } from './AudioView';

/**
 * Speech to Text.
 *
 * `rapid-mac`'s Dictation page minus what a browser cannot have: no global
 * hotkey, no Accessibility permission, no typing at the cursor in another app.
 * What DOES transfer is the setup card, the vocabulary that keeps proper nouns
 * right, and the history that turns a mistake into a vocabulary entry — so
 * recording happens here, on the page, rather than anywhere on the system.
 */

export function DictationPanel() {
  const pushNotice = useStore((state) => state.pushNotice);
  const model = useAudioModel('stt', PREFERRED_STT, 'whisper-large-v3-turbo');

  const recorder = useRef<Recorder | null>(null);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [busy, setBusy] = useState(false);
  const [terms, setTerms] = useState<Term[]>(loadVocabulary);
  const [history, setHistory] = useState<HistoryEntry[]>(loadHistory);

  const updateTerms = useCallback((next: Term[]) => {
    setTerms(next);
    saveVocabulary(next);
  }, []);

  useEffect(() => {
    if (!recording) return;
    const started = Date.now();
    const timer = setInterval(() => setElapsed(Date.now() - started), 200);
    return () => clearInterval(timer);
  }, [recording]);

  // The microphone must not stay live if the surface goes away mid-take.
  useEffect(() => () => recorder.current?.cancel(), []);

  const finish = useCallback(async () => {
    const instance = recorder.current;
    if (!instance) return;
    setRecording(false);
    let blob: Blob;
    let durationMs: number;
    try {
      ({ blob, durationMs } = await instance.stop());
    } catch (cause) {
      // A too-short take is a mis-tap, not a failure worth a red banner.
      if (cause instanceof RecorderError && cause.kind === 'empty') return;
      pushNotice({
        tone: 'error',
        title: cause instanceof RecorderError ? cause.message : 'Recording failed.',
      });
      return;
    }

    setBusy(true);
    const startedAt = Date.now();
    try {
      const context = vocabularyContext(terms);
      const result = await transcribe({
        audio: blob,
        model: model.alias,
        ...(context ? { context } : {}),
      });
      const text = result.text.trim();
      // A silent take produces no entry: an empty row in the history is not
      // a dictation the user would ever want to copy or correct.
      if (text === '') {
        pushNotice({ tone: 'info', title: 'Nothing was recognised.' });
        return;
      }
      setHistory((current) => {
        const next = addHistory(current, {
          id: newId(),
          text,
          at: Date.now(),
          durationMs,
          latencyMs: Date.now() - startedAt,
        });
        saveHistory(next);
        return next;
      });
    } catch (cause) {
      pushNotice(noticeFor(asApiError(cause)));
    } finally {
      setBusy(false);
    }
  }, [pushNotice, terms, model.alias]);

  const begin = useCallback(async () => {
    const instance = recorder.current ?? new Recorder();
    recorder.current = instance;
    try {
      await instance.start(() => void finish());
      setElapsed(0);
      setRecording(true);
    } catch (cause) {
      pushNotice({
        tone: 'error',
        title: cause instanceof RecorderError ? cause.message : 'Recording could not start.',
      });
    }
  }, [finish, pushNotice]);

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-1">
        <h2 className="m-0 text-base font-semibold">Speech to Text</h2>
        <p className="text-muted-foreground m-0 text-sm">
          Record here and the audio is transcribed by the engine on your Mac — it never leaves it.
        </p>
      </div>

      <SetupCard>
        <ModelRow
          model={model}
          disabled={recording || busy}
          onError={(title) => pushNotice({ tone: 'error', title })}
        />
      </SetupCard>

      <div className="flex flex-col items-center gap-4 py-4">
        <button
          type="button"
          aria-label={recording ? 'Stop recording' : 'Start recording'}
          className={cn(
            'flex size-24 items-center justify-center rounded-full border-2 transition-colors outline-none',
            'focus-visible:ring-ring/50 focus-visible:ring-[3px]',
            recording
              ? 'border-destructive bg-destructive/10 text-destructive'
              : 'hover:bg-accent border-border',
            busy && 'pointer-events-none opacity-60',
          )}
          onClick={() => void (recording ? finish() : begin())}
          disabled={busy || !model.live}
        >
          {busy ? (
            <Loader2 className="size-8 animate-spin" />
          ) : recording ? (
            <Square className="size-7 fill-current" />
          ) : (
            <Mic className="size-8" />
          )}
        </button>

        <p className="text-muted-foreground m-0 text-sm">
          {busy
            ? 'Transcribing…'
            : recording
              ? formatDuration(elapsed)
              : model.live
                ? 'Tap to record, tap again to stop.'
                : 'Start the model above to record.'}
        </p>
      </div>

      <Vocabulary terms={terms} onChange={updateTerms} />
      <Recent
        entries={history}
        onClear={() => {
          setHistory([]);
          saveHistory([]);
        }}
        onRemember={(text) => updateTerms(addTerm(terms, text))}
      />
    </div>
  );
}

/**
 * The hint terms sent with every transcription.
 *
 * The cap is the whole design constraint, not a nicety: measured accuracy
 * falls off past ~20 terms, so a term added over budget is PARKED rather than
 * dropped — the list is still the user's, it is just not all being sent.
 */
function Vocabulary({ terms, onChange }: { terms: Term[]; onChange(next: Term[]): void }) {
  const [draft, setDraft] = useState('');
  const active = activeCount(terms);
  const overBudget = active >= ACTIVE_LIMIT;

  const add = () => {
    if (draft.trim() === '') return;
    onChange(addTerm(terms, draft));
    setDraft('');
  };

  return (
    <section className="flex flex-col gap-2">
      <div className="flex items-center gap-3">
        <h3 className="m-0 text-sm font-semibold">Vocabulary</h3>
        <div className="bg-muted h-[5px] w-20 overflow-hidden rounded-full">
          <div
            className={cn('h-full rounded-full', overBudget ? 'bg-warning' : 'bg-success')}
            style={{ width: `${Math.min(100, (active / ACTIVE_LIMIT) * 100)}%` }}
          />
        </div>
        <span className="text-muted-foreground font-mono text-xs tabular-nums">
          {active} of {ACTIVE_LIMIT} active
        </span>
      </div>

      <div className="flex flex-col gap-3 rounded-xl border p-3.5">
        {terms.length === 0 ? (
          <p className="text-muted-foreground m-0 text-xs">
            No terms yet. Add the names the model keeps getting wrong — project names, people,
            product names.
          </p>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {terms.map((term) => (
              <TermChip
                key={term.text}
                term={term}
                onToggle={() => onChange(setTermActive(terms, term.text, !term.active))}
                onRemove={() => onChange(removeTerm(terms, term.text))}
              />
            ))}
          </div>
        )}

        <p className="text-muted-foreground m-0 text-xs">
          Accuracy drops when more than {ACTIVE_LIMIT} terms are sent at once, so keep this to the
          names that actually get missed. Tap a term to park it.
        </p>

        <div className="flex gap-2">
          <label htmlFor="vocabulary-term" className="sr-only">
            Add a name
          </label>
          <Input
            id="vocabulary-term"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault();
                add();
              }
            }}
            placeholder="Add a name…"
            autoCapitalize="off"
            autoCorrect="off"
          />
          <Button variant="secondary" onClick={add} disabled={draft.trim() === ''}>
            <Plus />
            Add
          </Button>
        </div>
      </div>
    </section>
  );
}

function TermChip({
  term,
  onToggle,
  onRemove,
}: {
  term: Term;
  onToggle(): void;
  onRemove(): void;
}) {
  return (
    // A span wrapping two buttons: the chip body toggles, the × removes, and
    // a button inside a button is invalid and unclickable in Safari.
    <span
      className={cn(
        'flex items-center gap-1 rounded-md border py-0.5 pr-1 pl-2',
        term.active ? 'bg-accent' : 'text-muted-foreground',
      )}
    >
      <button
        type="button"
        className="font-mono text-xs outline-none"
        title={term.active ? 'Sent with each dictation. Tap to park.' : 'Parked. Tap to activate.'}
        onClick={onToggle}
      >
        {term.text}
      </button>
      <button
        type="button"
        aria-label={`Remove ${term.text}`}
        className="text-muted-foreground hover:text-foreground flex size-4 items-center justify-center rounded-sm outline-none"
        onClick={onRemove}
      >
        <X className="size-3" />
      </button>
    </span>
  );
}

/**
 * Recent dictations.
 *
 * Duration and latency are shown because they are the two numbers that explain
 * a disappointing result — a 0.4 s take that transcribed badly is a recording
 * problem, a 40 s latency is a model-size problem. "Remember" is how a term
 * gets into the vocabulary: the correction is the moment the user knows a name
 * was missed.
 */
function Recent({
  entries,
  onClear,
  onRemember,
}: {
  entries: HistoryEntry[];
  onClear(): void;
  onRemember(text: string): void;
}) {
  return (
    <section className="flex flex-col gap-2">
      <div className="flex items-center gap-3">
        <h3 className="m-0 flex-1 text-sm font-semibold">Recent</h3>
        {entries.length > 0 ? (
          <Button variant="ghost" size="sm" onClick={onClear}>
            Clear
          </Button>
        ) : null}
      </div>

      {entries.length === 0 ? (
        <p className="text-muted-foreground m-0 rounded-xl border p-3.5 text-xs">
          Dictations you make will show up here.
        </p>
      ) : (
        <div className="divide-y rounded-xl border">
          {entries.map((entry) => (
            <HistoryRow key={entry.id} entry={entry} onRemember={onRemember} />
          ))}
        </div>
      )}
    </section>
  );
}

function HistoryRow({
  entry,
  onRemember,
}: {
  entry: HistoryEntry;
  onRemember(text: string): void;
}) {
  const [selection, setSelection] = useState('');

  return (
    <div className="flex flex-col gap-2 p-3.5">
      <p
        className="m-0 text-sm leading-relaxed [overflow-wrap:anywhere]"
        // The selected words ARE the correction: a user highlights the name
        // that came out wrong, which is exactly what should be remembered.
        onMouseUp={() => setSelection(window.getSelection()?.toString().trim() ?? '')}
        onTouchEnd={() => setSelection(window.getSelection()?.toString().trim() ?? '')}
      >
        {entry.text}
      </p>
      <div className="text-muted-foreground flex items-center gap-3 font-mono text-xs tabular-nums">
        <span>{new Date(entry.at).toLocaleTimeString()}</span>
        <span>{(entry.durationMs / 1000).toFixed(1)}s</span>
        <span>{(entry.latencyMs / 1000).toFixed(2)}s</span>
        <span className="flex-1" />
        {selection !== '' ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              onRemember(selection);
              setSelection('');
            }}
          >
            <Plus />
            Remember “{selection.length > 18 ? `${selection.slice(0, 18)}…` : selection}”
          </Button>
        ) : null}
        <CopyButton text={entry.text} />
      </div>
    </div>
  );
}
