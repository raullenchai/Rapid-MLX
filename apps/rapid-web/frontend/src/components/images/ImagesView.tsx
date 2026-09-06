import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  ArrowUp,
  ChevronsUpDown,
  Download,
  ImageIcon,
  ImagePlus,
  Pencil,
  Ruler,
  X,
} from 'lucide-react';
import { cancelImage, fetchImageJob, startImageJob } from '@/api/images';
import { ApiError, asApiError } from '@/api/errors';
import { supportsEditing, supportsGeneration } from '@/api/types';
import { useStore } from '@/state/store';
import { noticeFor } from '@/state/notices';
import { percent } from '@/components/models/LifecycleBand';
import { ComposerModelPicker } from '@/components/models/ComposerModelPicker';
import { StatusDot } from '@/components/common/StatusDot';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { cn } from '@/lib/utils';
import { READING_COLUMN } from '@/lib/layout';
import {
  ASPECT_LABELS,
  ASPECTS,
  RESOLUTIONS,
  outputSize,
  type Aspect,
  type Resolution,
} from '@/images/size';
import {
  ImageSourceError,
  readImageSource,
  sourceFromBase64,
  type ImageSource,
} from '@/images/source';

/**
 * Text-to-image, and instruction editing of an image.
 *
 * The engine serves ONE model at a time, so rendering requires the loaded
 * model to be an image model — this server owns a single `serve` child and a
 * switch restarts it. That is stated rather than worked around: silently
 * restarting the engine would discard a chat the user is in the middle of.
 *
 * Editing is a MODE of this surface rather than a separate page, matching
 * `rapid-mac`'s `ImagesView`: it is entered by supplying a source image (the
 * last result, or a file) and everything the edit backends ignore — aspect,
 * long edge — disappears while it is on.
 */

/** How often to ask a running job for its denoise progress. */
const PROGRESS_POLL_MS = 400;

/**
 * Consecutive failed polls before a render is given up on.
 *
 * A single dropped poll is not a failure — the tunnel or the radio can lose
 * one while the render is fine — but a server that has gone away must not be
 * polled forever. Same cap as the download feed.
 */
const MAX_POLL_FAILURES = 5;

/**
 * Shown on an empty canvas. Concrete enough to produce a good first image —
 * the failure mode of an empty prompt box is a one-word prompt and a
 * disappointing result blamed on the model.
 */
const SUGGESTIONS = [
  'A cozy ramen shop at night in the rain, neon, steam, 35mm',
  'Studio portrait of an elderly fisherman, dramatic side light',
  'A minimalist product shot of a ceramic mug on linen',
  'A whale drifting through clouds above a city at dusk',
];

export function ImagesView({
  onChooseModel,
  onSelectModel,
  blockedPlaceholder,
  band,
}: {
  onChooseModel(): void;
  onSelectModel(alias: string): void;
  /** The current image model's lifecycle step, shared with the status band. */
  blockedPlaceholder: string;
  /** The shared readiness surface — downloading, starting, failed. */
  band?: ReactNode;
}) {
  const status = useStore((state) => state.status);
  const models = useStore((state) => state.models);
  const selectedImage = useStore((state) => state.selectedByKind.image);
  const pushNotice = useStore((state) => state.pushNotice);

  const [prompt, setPrompt] = useState('');
  const [aspect, setAspect] = useState<Aspect>('square');
  const [resolution, setResolution] = useState<Resolution>(512);
  /** The render in flight, or null. Starting one is itself a request, so
   *  `starting` covers the gap before an id exists. */
  const [jobId, setJobId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [progress, setProgress] = useState<{ step: number; total: number } | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [image, setImage] = useState<string | null>(null);
  const [caption, setCaption] = useState('');
  /** Non-null puts the surface in edit mode; it is the image being edited. */
  const [source, setSource] = useState<ImageSource | null>(null);
  /** The prompt this render was started with, so editing the box afterwards
   *  cannot relabel an image produced from something else. */
  const submitted = useRef('');
  const controller = useRef<AbortController | null>(null);
  const filePicker = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const url = source?.url;
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [source]);

  const rendering = starting || jobId !== null;

  const loaded = status?.model ?? null;
  // Any image model the engine is HOLDING, not just the primary. A hot load
  // leaves a chat model as the primary while an image model is resident
  // beside it, so keying on `status.model` alone reported a loaded image
  // model as absent and hid the whole canvas.
  const imageAlias = useMemo(() => {
    const resident = status?.resident ?? (loaded ? [loaded] : []);
    const isImage = (alias: string) =>
      models.some((model) => model.alias === alias && model.kind === 'image');
    // Prefer this surface's own pick, so two resident image models resolve
    // to the one the user chose.
    if (selectedImage && resident.includes(selectedImage) && isImage(selectedImage)) {
      return selectedImage;
    }
    return resident.find(isImage) ?? null;
  }, [status, loaded, models, selectedImage]);

  const imageModel = useMemo(
    () => models.find((model) => model.alias === imageAlias) ?? null,
    [models, imageAlias],
  );
  const editing = source !== null;
  // The loaded model must accept the request shape about to be sent. An
  // edit-only checkpoint 409s a generation and vice versa, so the send
  // control refuses rather than surfacing the engine's answer as a failure.
  const modelFits =
    imageModel !== null &&
    (editing ? supportsEditing(imageModel) : supportsGeneration(imageModel));
  const ready = status?.state === 'ready' && modelFits;
  // Only worth saying when a model IS loaded — otherwise the readiness band
  // already owns the explanation.
  const wrongCapability = status?.state === 'ready' && imageModel !== null && !modelFits;

  // Poll the running job. One request carries both the denoise counter and
  // the finished image, so a render occupies a single connection — the POST
  // that started it returned immediately rather than being held open, which
  // is what a tunnel cuts at 100 s with a 524.
  useEffect(() => {
    if (jobId === null) return;
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;

    const finish = () => {
      setJobId(null);
      setProgress(null);
    };

    const tick = async () => {
      try {
        const snapshot = await fetchImageJob(jobId, abort.signal);
        if (abort.signal.aborted) return;
        failures = 0;

        if (snapshot.state === 'running') {
          setProgress({ step: snapshot.step ?? 0, total: snapshot.total ?? 0 });
        } else if (snapshot.state === 'failed') {
          const failure = snapshot.error;
          pushNotice(
            noticeFor(
              new ApiError(
                failure?.status ?? 500,
                failure?.type ?? 'image_job_failed',
                failure?.message ?? 'the render failed',
              ),
            ),
          );
          finish();
          return;
        } else if (snapshot.b64_json) {
          const rendered = snapshot.b64_json;
          setImage(rendered);
          setCaption(submitted.current);
          // Chain: the next edit acts on what was just produced. Functional,
          // so this does not have to depend on `source` and restart the poll.
          setSource((previous) =>
            previous
              ? sourceFromBase64(rendered, submitted.current)
              : previous,
          );
          finish();
          return;
        } else {
          // Cancelled mid-batch keeps whatever finished, which can be
          // nothing — a success with no image, not a failure.
          if (!snapshot.cancelled) {
            pushNotice({ tone: 'warning', title: 'The engine returned no image.' });
          }
          finish();
          return;
        }
      } catch (cause) {
        if (abort.signal.aborted) return;
        // A 404 means the job is gone — only the last one is kept — so there
        // is nothing left to wait for.
        const error = asApiError(cause);
        failures += 1;
        if (error.status === 404 || failures >= MAX_POLL_FAILURES) {
          pushNotice(noticeFor(error));
          finish();
          return;
        }
      }
      // Re-scheduled AFTER the response, never on an interval: a slow poll
      // must not stack overlapping requests behind the render.
      if (!abort.signal.aborted) timer = setTimeout(() => void tick(), PROGRESS_POLL_MS);
    };

    void tick();
    return () => {
      abort.abort();
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [jobId, pushNotice]);

  // The elapsed clock, which is the only thing moving during the warm-up: a
  // cold mflux load reports no steps for tens of seconds, and a HUD with a
  // frozen readout reads as a hang. `rapid-mac`'s HUD ticks the same way.
  useEffect(() => {
    if (!rendering) return;
    const startedAt = Date.now();
    const timer = setInterval(() => setElapsedMs(Date.now() - startedAt), 100);
    return () => clearInterval(timer);
  }, [rendering]);

  const run = useCallback(async () => {
    const text = prompt.trim();
    if (text === '' || rendering || !ready) return;

    const abort = new AbortController();
    controller.current = abort;
    setStarting(true);
    setProgress(null);
    setElapsedMs(0);
    submitted.current = text;

    try {
      // Named explicitly: the engine can hold several models, and an omitted
      // `model` resolves to the primary — which is the CHAT model whenever
      // the image model was hot-loaded beside it.
      const job = await startImageJob({
        prompt: text,
        ...(source ? { image: source.blob } : { size: outputSize(aspect, resolution) }),
        ...(imageAlias ? { model: imageAlias } : {}),
        signal: abort.signal,
      });
      // Cleared here rather than on the result: the instruction has been
      // accepted, and leaving it in the box invites re-applying it to its
      // own output.
      if (source) setPrompt('');
      setJobId(job.id);
    } catch (cause) {
      // An abort is the user pressing Stop, not a failure to report.
      if (!abort.signal.aborted) pushNotice(noticeFor(asApiError(cause)));
    } finally {
      controller.current = null;
      setStarting(false);
    }
  }, [prompt, aspect, resolution, rendering, ready, pushNotice, imageAlias, source]);

  const stop = useCallback(() => {
    // Both: the engine stops at its next denoise step, and dropping the job
    // releases the page rather than polling out the render it just cancelled.
    // Named, for the same reason the render is: an omitted model cancels
    // against the primary, which may not be the model rendering.
    void cancelImage(imageAlias ?? undefined).catch(() => undefined);
    controller.current?.abort();
    setJobId(null);
    setProgress(null);
  }, [imageAlias]);

  /** Edit the image on the canvas — the common case, and no file dialog. */
  const editResult = useCallback(() => {
    if (image === null) return;
    setSource(sourceFromBase64(image, caption || 'Rendered image'));
    setPrompt('');
  }, [image, caption]);

  const importSource = useCallback(
    async (file: File | undefined) => {
      if (!file) return;
      try {
        setSource(await readImageSource(file));
        setPrompt('');
      } catch (cause) {
        pushNotice({
          tone: 'warning',
          title: "Couldn't import the image",
          body: cause instanceof ImageSourceError ? cause.message : undefined,
        });
      }
    },
    [pushNotice],
  );

  const size = outputSize(aspect, resolution);

  // What the stage shows. The SOURCE wins in edit mode: an imported file has
  // no render behind it yet, and the thing being edited is what the user must
  // be looking at while they describe the change.
  const canvas = source
    ? { url: source.url, alt: source.label }
    : image
      ? { url: `data:image/png;base64,${image}`, alt: caption }
      : null;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* The canvas is the subject, so progress is a HUD over it rather than a
          strip above it — a strip pushes the stage down and shrinks the one
          thing the user is watching. Same reasoning as `rapid-mac`'s
          `progressHUD` (ImagesView.swift:318). */}
      <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-y-auto p-4">
        {canvas ? (
          <figure className="m-0 flex max-h-full flex-col items-center gap-3">
            <img
              src={canvas.url}
              alt={canvas.alt}
              className="max-h-[52dvh] rounded-lg border object-contain shadow-sm"
            />
            <div className="flex items-center gap-2">
              {image ? <SaveButton data={image} /> : null}
              {/* Hidden once editing: the source strip already shows what is
                  being edited, and a second entry point onto the same image
                  would only restart the chain. */}
              {image !== null && !editing && !rendering ? (
                <Button variant="outline" size="sm" onClick={editResult}>
                  <Pencil />
                  Edit image
                </Button>
              ) : null}
            </div>
          </figure>
        ) : rendering ? null : (
          <Placeholder ready={ready} size={size} />
        )}

        {rendering ? (
          <ProgressHUD
            step={progress?.step ?? 0}
            total={progress?.total ?? 0}
            elapsedMs={elapsedMs}
            onCancel={stop}
          />
        ) : null}
      </div>

      <div
        className={cn(
          READING_COLUMN,
          'flex flex-col gap-2.5 p-3 pb-[calc(env(safe-area-inset-bottom)+12px)]',
        )}
      >
        {/* Above the composer, matching the chat: downloading, starting and
            failed all report themselves here rather than in the canvas. */}
        {band}

        {/* Only on an empty canvas: once there is a result, these would
            compete with it for the eye. Never while editing — the source
            image is the subject, and a starter would replace the
            instruction with an unrelated scene description. */}
        {canvas === null && !rendering ? (
          <div className="grid gap-2 sm:grid-cols-2">
            {SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                className="hover:bg-accent rounded-xl border px-3.5 py-2.5 text-left text-sm transition-colors outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
                onClick={() => setPrompt(suggestion)}
              >
                {suggestion}
              </button>
            ))}
          </div>
        ) : null}

        {/* A capability mismatch is not a lifecycle failure, so the band
            cannot say it: the model IS loaded and running, it just takes the
            other request shape. */}
        {wrongCapability ? (
          <p className="text-muted-foreground m-0 px-1 text-xs">
            {editing
              ? `${imageAlias} is text-to-image only. Choose an edit-capable model, or exit editing.`
              : `${imageAlias} only edits images. Import one to edit, or choose a text-to-image model.`}
          </p>
        ) : null}

        {/* One bordered box holding the prompt and its controls, so the
            settings visibly belong to the thing being submitted. */}
        <div className="focus-within:border-ring/60 rounded-2xl border px-3.5 pt-3 pb-2.5 transition-colors">
          {source ? (
            <EditSourceStrip
              source={source}
              disabled={rendering}
              onExit={() => setSource(null)}
            />
          ) : null}

          <label htmlFor="image-prompt" className="sr-only">
            Prompt
          </label>
          <textarea
            id="image-prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder={
              !ready
                ? wrongCapability
                  ? editing
                    ? 'Choose an edit-capable model first'
                    : 'Choose a text-to-image model first'
                  : blockedPlaceholder
                : editing
                  ? 'Describe what you want to change…'
                  : 'Describe the image…'
            }
            rows={2}
            className="placeholder:text-muted-foreground w-full resize-none border-0 bg-transparent p-0 text-base outline-none"
          />

          <div className="mt-1.5 flex items-center gap-1">
            {/* Removed rather than disabled in edit mode: the engine derives
                the output canvas from the source image and discards `size`
                entirely, so a control that still looked live would lie. */}
            {editing ? null : (
              <>
                <AspectPicker value={aspect} onChange={setAspect} />
                <SizePicker
                  value={resolution}
                  aspect={aspect}
                  size={size}
                  onChange={setResolution}
                />
              </>
            )}

            <input
              ref={filePicker}
              type="file"
              accept="image/png,image/jpeg"
              className="hidden"
              onChange={(event) => {
                void importSource(event.target.files?.[0]);
                // Cleared so picking the SAME file twice fires `change` again.
                event.target.value = '';
              }}
            />
            <Button
              variant="ghost"
              size="icon"
              className="text-muted-foreground size-9 [&_svg:not([class*=size-])]:size-4"
              onClick={() => filePicker.current?.click()}
              disabled={rendering}
              aria-label={editing ? 'Replace source image' : 'Import image to edit'}
              title={editing ? 'Replace source image' : 'Import image to edit'}
            >
              <ImagePlus />
            </Button>

            <span className="flex-1" />

            <ComposerModelPicker
              kind="image"
              // Only models that accept the shape about to be sent — offering
              // one that would 409 is a dead end reached through a selector.
              filter={editing ? supportsEditing : supportsGeneration}
              onManage={onChooseModel}
              onSelect={(model) => onSelectModel(model.alias)}
            />

            {rendering ? (
              <Button
                variant="outline"
                size="icon"
                className="size-9 rounded-full"
                onClick={stop}
                aria-label="Stop"
                title="Stop"
              >
                <span className="bg-foreground size-2.5 rounded-[2px]" />
              </Button>
            ) : (
              <Button
                size="icon"
                className="size-9 rounded-full"
                onClick={() => void run()}
                disabled={!ready || prompt.trim() === ''}
                aria-label={editing ? 'Edit image' : 'Generate'}
                title={editing ? 'Edit image' : 'Generate'}
              >
                <ArrowUp />
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * What is being edited, above the instruction box.
 *
 * Edit mode is otherwise only visible as an absence — the size controls are
 * gone and the placeholder reads differently — which is not enough to explain
 * why a prompt is producing a variation of an earlier image. Ported from
 * `rapid-mac`'s `editSourceBar`.
 */
function EditSourceStrip({
  source,
  disabled,
  onExit,
}: {
  source: ImageSource;
  disabled: boolean;
  onExit(): void;
}) {
  return (
    <div className="mb-2.5 flex items-center gap-2.5 border-b pb-2.5">
      <img
        src={source.url}
        alt=""
        className="size-9 shrink-0 rounded-md border object-cover"
      />
      <span className="flex min-w-0 flex-1 flex-col">
        <span className="text-sm font-medium">Editing image</span>
        <span className="text-muted-foreground truncate text-xs">{source.label}</span>
      </span>
      <Button
        variant="ghost"
        size="icon"
        className="text-muted-foreground size-8 [&_svg:not([class*=size-])]:size-4"
        onClick={onExit}
        disabled={disabled}
        aria-label="Exit image editing"
        title="Exit image editing"
      >
        <X />
      </Button>
    </div>
  );
}

/**
 * The wait, over its own subject.
 *
 * Diffusion reports a real step count, but only once the weights are in — a
 * cold mflux load is silent for tens of seconds. So the bar is determinate
 * when there are steps and an indeterminate shuttle before that, rather than
 * a determinate bar pinned at 0% pretending to know.
 */
function ProgressHUD({
  step,
  total,
  elapsedMs,
  onCancel,
}: {
  step: number;
  total: number;
  elapsedMs: number;
  onCancel(): void;
}) {
  const denoising = total > 0;
  const fraction = denoising ? Math.min(1, step / total) : 0;

  return (
    <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-b from-black/10 to-black/40 p-4">
      <div className="bg-card text-card-foreground animate-in fade-in-0 zoom-in-95 w-[min(340px,100%)] rounded-2xl border p-4 shadow-lg">
        <div className="flex items-center gap-2.5">
          <StatusDot role="working" pulse />
          <span className="flex-1 text-sm font-semibold">
            {denoising ? 'Generating' : 'Warming up'}
          </span>
          {denoising ? (
            <span className="text-muted-foreground font-mono text-xs font-semibold tabular-nums">
              {step} / {total}
            </span>
          ) : null}
          <Button
            variant="secondary"
            size="icon"
            className="size-6 rounded-full"
            onClick={onCancel}
            aria-label="Cancel"
            title="Cancel"
          >
            <X className="size-3" />
          </Button>
        </div>

        <div className="bg-muted relative mt-3.5 h-2.5 overflow-hidden rounded-full">
          {denoising ? (
            <div
              className="bg-primary h-full rounded-full transition-[width] duration-300 ease-out"
              style={{ width: percent(fraction) }}
            />
          ) : (
            // A shuttle that enters and leaves the track, matching
            // `ShimmerProgressBar`'s indeterminate sweep. Clipped by the
            // parent so the overhang never paints over the card's padding.
            <div className="bg-primary absolute inset-y-0 w-1/3 rounded-full animate-[hud-sweep_1.6s_ease-in-out_infinite]" />
          )}
        </div>

        <div className="text-muted-foreground mt-2.5 flex items-center justify-between text-xs font-medium">
          <span className="font-mono tabular-nums">{(elapsedMs / 1000).toFixed(1)}s</span>
          <span>{denoising ? etaText(step, total, elapsedMs) : 'First run — only happens once'}</span>
        </div>
      </div>
    </div>
  );
}

/** Extrapolated from the steps already done — the only estimate available, and
 *  honest about having none until at least one step has landed. */
function etaText(step: number, total: number, elapsedMs: number): string {
  if (step <= 0 || elapsedMs <= 0) return 'Estimating…';
  const remaining = ((total - step) * elapsedMs) / step / 1000;
  if (remaining <= 0) return 'Almost there…';
  return `~${Math.max(1, Math.round(remaining))}s left`;
}

/** 1:1 / 3:4 / 4:3, as inline segments rather than a labelled control: the
 *  ratios are self-describing and a caption would cost a row. */
function AspectPicker({
  value,
  onChange,
}: {
  value: Aspect;
  onChange(next: Aspect): void;
}) {
  return (
    <div className="flex items-center" role="radiogroup" aria-label="Aspect ratio">
      {ASPECTS.map((candidate) => (
        <button
          key={candidate}
          type="button"
          role="radio"
          aria-checked={value === candidate}
          aria-label={ASPECT_LABELS[candidate]}
          className={cn(
            'rounded-md px-2 py-1 text-sm transition-colors outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50',
            value === candidate
              ? 'bg-accent text-foreground font-medium'
              : 'text-muted-foreground hover:text-foreground',
          )}
          onClick={() => onChange(candidate)}
        >
          {ASPECT_LABELS[candidate]}
        </button>
      ))}
    </div>
  );
}

/** The long edge, labelled with the dimensions it actually produces — the
 *  number the engine is sent is what the user should see. */
function SizePicker({
  value,
  aspect,
  size,
  onChange,
}: {
  value: Resolution;
  aspect: Aspect;
  size: string;
  onChange(next: Resolution): void;
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          aria-label="Long edge"
          className="text-muted-foreground hover:text-foreground flex items-center gap-1.5 rounded-md px-2 py-1 text-sm transition-colors outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
        >
          <Ruler className="size-4 shrink-0" />
          {size.replace('x', ' × ')}
          <ChevronsUpDown className="size-3.5 shrink-0" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        {/* Full dimensions, not the long edge alone: "1024 px" does not say
            what a 4:3 render will actually come out as. */}
        {RESOLUTIONS.map((candidate) => (
          <DropdownMenuItem key={candidate} onSelect={() => onChange(candidate)}>
            {outputSize(aspect, candidate).replace('x', ' × ')} px
            {candidate === value ? <span className="text-muted-foreground ml-auto">✓</span> : null}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/**
 * The empty canvas.
 *
 * The dashed frame is sized to the chosen aspect and labelled with the exact
 * dimensions, so the size controls have something to act on before the first
 * render — otherwise changing them appears to do nothing at all.
 *
 * It says nothing about the engine's state: the readiness band above the
 * composer owns that, and duplicating it here is how the two disagree.
 */
function Placeholder({ ready, size }: { ready: boolean; size: string }) {
  const [width, height] = size.split('x').map(Number);
  // Scaled to a constant area rather than to the long edge, so a portrait and
  // a landscape frame look equally weighty instead of the taller one
  // dominating.
  const scale = 190 / Math.sqrt((width ?? 1) * (height ?? 1));

  return (
    <div className="flex flex-col items-center gap-5 text-center">
      <div
        className="text-muted-foreground/60 flex items-center justify-center rounded-2xl border-2 border-dashed"
        style={{
          width: Math.round((width ?? 1) * scale),
          height: Math.round((height ?? 1) * scale),
        }}
      >
        <div className="flex flex-col items-center gap-2">
          <ImageIcon className="size-10" strokeWidth={1.5} />
          <span className="font-mono text-sm">{size}</span>
        </div>
      </div>

      <div className="flex flex-col gap-1.5">
        <h2 className="m-0 text-2xl font-semibold tracking-tight">Draw anything</h2>
        <p className="text-muted-foreground m-0 max-w-[40ch] text-sm">
          Create images locally, then keep generating offline.
        </p>
        <p className="text-muted-foreground/70 m-0 max-w-[40ch] text-sm">
          {ready
            ? 'Pick a starter below, or describe your own.'
            : 'Choose an image model below. The engine serves one model at a time, so switching to it will stop the chat model.'}
        </p>
      </div>
    </div>
  );
}

function SaveButton({ data }: { data: string }) {
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={() => {
        // An object URL rather than the data URI directly: Safari refuses to
        // download a multi-megabyte `data:` href, and silently does nothing.
        const bytes = Uint8Array.from(atob(data), (character) => character.charCodeAt(0));
        const url = URL.createObjectURL(new Blob([bytes], { type: 'image/png' }));
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = 'rapid-image.png';
        anchor.click();
        URL.revokeObjectURL(url);
      }}
    >
      <Download />
      Save
    </Button>
  );
}
