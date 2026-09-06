import { useCallback, useEffect, useRef, useState } from 'react';
import { Download, Loader2, Play, Square } from 'lucide-react';
import { fetchVoices, synthesize } from '@/api/audio';
import { asApiError } from '@/api/errors';
import { useStore } from '@/state/store';
import { noticeFor } from '@/state/notices';
import { PREFERRED_TTS, previewText, voiceDetails } from '@/audio/models';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { cn } from '@/lib/utils';
import { ModelRow, SetupCard, SetupRow, useAudioModel } from './AudioView';

/**
 * Text to Speech.
 *
 * Layout follows `rapid-mac`'s `speechSurface`: the text first, the setup card
 * under it (Model / Voice / Speed on one trailing line), then the action, then
 * the result. Voices come from the engine and are re-fetched when the model
 * changes — each family ships its own set, and Kokoro's 54 have nothing to do
 * with Qwen3's named speakers.
 */

export function SpeechPanel() {
  const pushNotice = useStore((state) => state.pushNotice);
  const model = useAudioModel('tts', PREFERRED_TTS, 'kokoro');

  const [text, setText] = useState('');
  // Keyed by the alias they came from, so a model change discards them by
  // DERIVATION rather than by clearing state in an effect — the panel never
  // renders one model's voices under another's name.
  const [loaded, setLoaded] = useState<{ alias: string; voices: string[] } | null>(null);
  const [failure, setFailure] = useState<{ alias: string; message: string } | null>(null);
  const [chosen, setChosen] = useState<string | null>(null);
  const [speed, setSpeed] = useState(1);
  const [busy, setBusy] = useState(false);
  const [audio, setAudio] = useState<{ url: string; bytes: number } | null>(null);
  const [previewing, setPreviewing] = useState<{ voice: string; loading: boolean } | null>(null);
  const preview = useRef<HTMLAudioElement | null>(null);
  // Bumped by every stop, so a synthesis that lands after the user has moved
  // on cannot start playing over whatever replaced it.
  const previewToken = useRef(0);

  const alias = model.alias;
  const live = model.live;
  const voices = loaded?.alias === alias ? loaded.voices : [];
  const voiceError = failure?.alias === alias ? failure.message : null;
  // The chosen voice only holds while the list still offers it.
  const voice = chosen !== null && voices.includes(chosen) ? chosen : (voices[0] ?? null);
  // A dead lane is not a voice problem, and saying "Unavailable" here would
  // point at the wrong control — the Model row owns that state.
  const voicePlaceholder = !live
    ? 'Start the model to list voices'
    : voiceError
      ? 'Unavailable'
      : 'Loading voices…';

  // Revoked on replacement AND unmount: each result is an object URL, and a
  // page left open through a dozen takes would hold every one of them.
  useEffect(() => {
    return () => {
      if (audio) URL.revokeObjectURL(audio.url);
    };
  }, [audio]);

  // A preview outlives neither a model change nor the panel.
  const stopPreview = useCallback(() => {
    previewToken.current += 1;
    preview.current?.pause();
    preview.current = null;
    setPreviewing(null);
  }, []);
  useEffect(() => stopPreview, [stopPreview]);

  useEffect(() => {
    // Only once the lane can actually answer: asking a stopped engine
    // produces a 503 that reads as "this model has no voices".
    if (!live) return;
    let cancelled = false;
    void fetchVoices(alias)
      .then((response) => {
        if (cancelled) return;
        setLoaded({ alias, voices: response.voices });
      })
      .catch((cause) => {
        if (cancelled) return;
        // Shown inline rather than as a notice: without voices this panel
        // cannot be used at all, so the message belongs where the control is.
        setFailure({ alias, message: asApiError(cause).message });
      });
    return () => {
      cancelled = true;
    };
    // `live` too: the fetch is skipped while the engine is down, so it has to
    // re-run when it comes up rather than leaving the list permanently empty.
  }, [alias, live]);

  const speak = useCallback(async () => {
    const input = text.trim();
    if (input === '' || busy || !voice) return;
    stopPreview();
    setBusy(true);
    try {
      const blob = await synthesize({ input, model: alias, voice, speed });
      setAudio((previous) => {
        if (previous) URL.revokeObjectURL(previous.url);
        return { url: URL.createObjectURL(blob), bytes: blob.size };
      });
    } catch (cause) {
      pushNotice(noticeFor(asApiError(cause)));
    } finally {
      setBusy(false);
    }
  }, [text, busy, voice, speed, alias, pushNotice, stopPreview]);

  const playPreview = useCallback(
    async (candidate: string) => {
      if (previewing?.voice === candidate) {
        stopPreview();
        return;
      }
      stopPreview();
      const token = previewToken.current;
      setPreviewing({ voice: candidate, loading: true });
      try {
        const blob = await synthesize({
          input: previewText(candidate),
          model: alias,
          voice: candidate,
          speed,
        });
        if (previewToken.current !== token) return;
        const element = new Audio(URL.createObjectURL(blob));
        // Revoked on `ended` rather than after `play()`: revoking a URL the
        // element is still reading from stops playback in WebKit.
        element.addEventListener('ended', () => {
          URL.revokeObjectURL(element.src);
          setPreviewing((current) => (current?.voice === candidate ? null : current));
        });
        preview.current = element;
        setPreviewing({ voice: candidate, loading: false });
        await element.play();
      } catch (cause) {
        if (previewToken.current !== token) return;
        setPreviewing(null);
        pushNotice(noticeFor(asApiError(cause)));
      }
    },
    [previewing, alias, speed, pushNotice, stopPreview],
  );

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-2">
        <Label htmlFor="speech-text">Text</Label>
        <Textarea
          id="speech-text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Type something to hear it spoken…"
          rows={5}
        />
      </div>

      <SetupCard>
        <ModelRow
          model={model}
          disabled={busy}
          onError={(title) => pushNotice({ tone: 'error', title })}
        />

        <SetupRow
          label="Voice"
          caption={voiceError ?? 'Each model ships its own set — preview before choosing.'}
          done={voice !== null}
          control={
            <VoicePicker
              voices={voices}
              value={voice}
              placeholder={voicePlaceholder}
              previewing={previewing}
              onSelect={setChosen}
              onPreview={(candidate) => void playPreview(candidate)}
            />
          }
        />

        <SetupRow
          label="Speed"
          caption="Applied when the audio is generated."
          control={
            <div className="flex w-[min(15rem,45vw)] items-center gap-3">
              <Slider
                aria-label="Speed"
                className="flex-1 py-2"
                value={[speed]}
                min={0.5}
                max={2}
                step={0.05}
                onValueChange={([next]) => next !== undefined && setSpeed(next)}
              />
              <span className="text-muted-foreground w-11 text-right font-mono text-xs tabular-nums">
                {speed.toFixed(2)}x
              </span>
            </div>
          }
        />
      </SetupCard>

      <Button
        onClick={() => void speak()}
        disabled={busy || text.trim() === '' || !voice || !live}
      >
        {busy ? <Loader2 className="animate-spin" /> : <Play />}
        {busy ? 'Generating…' : 'Generate speech'}
      </Button>

      {audio ? (
        <div className="flex flex-col gap-3 rounded-lg border p-3.5">
          <div className="flex items-center gap-2">
            <span className="flex-1 text-sm font-medium">Speech ready</span>
            <span className="text-muted-foreground font-mono text-xs">
              {(audio.bytes / 1024).toFixed(0)} KB
            </span>
          </div>
          {/* Save sits on the player's own row: it acts on the audio, so a
              line of its own read as a separate step. `min-w-0` because the
              native player has a wide intrinsic width and would otherwise
              push the button off a phone. */}
          <div className="flex items-center gap-2">
            {/* The native player: it already has play/pause, a scrubber and a
                volume control, all keyboard-accessible. */}
            <audio src={audio.url} controls className="min-w-0 flex-1" />
            <Button variant="outline" size="icon" className="shrink-0" asChild>
              <a href={audio.url} download="rapid-speech.wav" aria-label="Save" title="Save">
                <Download />
              </a>
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

/**
 * The voice list, with a preview on every row.
 *
 * Ported from `rapid-mac`'s `VoiceOptionRow`. A voice id means nothing on its
 * own — `af_heart` is American English, female — so each row carries the
 * decoded detail and a play button that synthesises a sample in that voice's
 * own language.
 *
 * Previewing is comparison, so the preview button must not close the menu:
 * a menu that closes on the first sample makes comparing two voices six
 * clicks instead of two. Synthesis is not instant either, so the button
 * spins while the sample is being generated — otherwise a tap looks ignored.
 */
function VoicePicker({
  voices,
  value,
  placeholder,
  previewing,
  onSelect,
  onPreview,
}: {
  voices: string[];
  value: string | null;
  placeholder: string;
  previewing: { voice: string; loading: boolean } | null;
  onSelect(voice: string): void;
  onPreview(voice: string): void;
}) {
  // Set by the preview button, read by the row it sits in. The button's click
  // bubbles into the row, so the flag is always set before the row reads it.
  const fromPreview = useRef(false);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          disabled={voices.length === 0}
          // Carries the VALUE: a bare "Voice" would override the visible text
          // and announce the control without its selection.
          aria-label={`Voice: ${value ?? placeholder}`}
          className="hover:bg-accent flex w-[min(15rem,45vw)] items-center justify-between gap-1.5 rounded-md border px-2.5 py-1.5 text-sm transition-colors outline-none focus-visible:ring-ring/50 focus-visible:ring-[3px] disabled:opacity-50"
        >
          <span className="min-w-0 truncate">{value ?? placeholder}</span>
          {value ? (
            <span className="text-muted-foreground shrink-0 text-xs">{voiceDetails(value)}</span>
          ) : null}
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="max-h-[50dvh] w-[min(22rem,90vw)] overflow-y-auto">
        {voices.map((candidate) => {
          const active = previewing?.voice === candidate ? previewing : null;
          return (
            <DropdownMenuItem
              key={candidate}
              className="gap-2 pr-1"
              onSelect={(event) => {
                // Radix keeps the menu open only if THIS event is defaulted.
                // Stopping propagation on the button is not enough: on
                // `pointerup` the item re-fires a synthetic `click()` whenever
                // it never saw the matching `pointerdown`.
                if (fromPreview.current) {
                  fromPreview.current = false;
                  event.preventDefault();
                  return;
                }
                onSelect(candidate);
              }}
            >
              <span className="min-w-0 flex-1 truncate text-sm">{candidate}</span>
              <span className="text-muted-foreground shrink-0 text-xs">
                {voiceDetails(candidate)}
              </span>
              <button
                type="button"
                aria-label={
                  active?.loading
                    ? `Generating ${candidate} preview`
                    : active
                      ? `Stop ${candidate} preview`
                      : `Preview ${candidate}`
                }
                className={cn(
                  'hover:bg-accent flex size-7 shrink-0 items-center justify-center rounded-md transition-colors outline-none',
                  'focus-visible:ring-ring/50 focus-visible:ring-[3px]',
                )}
                onClick={() => {
                  fromPreview.current = true;
                  onPreview(candidate);
                }}
              >
                {active?.loading ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : active ? (
                  <Square className="size-3 fill-current" />
                ) : (
                  <Play className="size-3.5" />
                )}
              </button>
            </DropdownMenuItem>
          );
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
