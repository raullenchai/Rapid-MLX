/**
 * Wire types, mirroring the server exactly.
 *
 * These are transcriptions of Python dataclasses, so each one cites its
 * source. When the server changes, this file changes with it — nothing else
 * in the app should be reading raw JSON.
 */

/** ``supervisor.ChildState`` — supervisor.py:54-60. */
export type EngineState = 'stopped' | 'starting' | 'ready' | 'failed';

/** ``supervisor.ChildStatus.to_dict`` — supervisor.py:82-88, plus the two
 *  fields ``app.py`` grafts on at the route (app.py:283-288). */
export interface StatusResponse {
  state: EngineState;
  model: string | null;
  port: number | null;
  detail: string | null;
  can_switch: boolean;
  /**
   * Every alias loaded into the running engine, including `model`.
   *
   * More than one once a hot load succeeds: the engine keeps text/vision in
   * a single-slot group and gives each media modality its own, so a chat
   * model and an image model are usable at the same time. `model` remains
   * the PRIMARY — what an unrouted request falls back to.
   */
  resident?: string[];
  /** Only present on a failure; the engine's recent log tail. */
  recent_output?: string[];
}

/** ``catalog.KINDS`` — catalog.py. Video is deliberately absent: the engine's
 *  video lane needs extras a plain install does not ship. */
export type ModelKind = 'text' | 'image' | 'audio';

/** ``catalog.ModelEntry.to_dict`` — catalog.py. */
export interface ModelEntry {
  alias: string;
  hf_path: string;
  size_bytes: number | null;
  cached: boolean;
  kind: ModelKind;
  /** Whether `rapid-mlx serve` accepts this alias. Audio IS loadable — the
   *  CLI has a dedicated audio-serve fork — but only as a last resort, since
   *  the lane rides on whatever model is already running. */
  loadable: boolean;
  cached_bytes: number | null;
  tool_call_parser: string | null;
  reasoning_parser: string | null;
  /**
   * NOT a trustworthy capability signal, and deliberately unused for one.
   * The catalog has claimed vision for checkpoints that 400 every image; only
   * a loaded engine knows, via ``/v1/models/{alias}``'s ``capabilities``. It
   * is carried here because the server sends it, not because anything should
   * branch on it.
   */
  is_text_only: boolean;
  /** Audio only: ``tts`` or ``stt``. */
  audio_kind: string | null;
  /** Audio only: the backend family, e.g. ``whisper``. */
  family: string | null;
  /**
   * Image only: which request shape this checkpoint accepts. Derived from the
   * repo id by the same rule the CLI uses for its `[image:gen]` /
   * `[image:edit]` / `[image:both]` tag — the JSON catalog carries no
   * capability field of its own.
   */
  image_capability: ImageCapability | null;
}

/** ``catalog._image_capability`` — catalog.py. */
export type ImageCapability = 'generation' | 'editing' | 'both';

export function supportsGeneration(model: ModelEntry): boolean {
  return model.image_capability !== 'editing';
}

export function supportsEditing(model: ModelEntry): boolean {
  return model.image_capability === 'editing' || model.image_capability === 'both';
}

export interface ModelsResponse {
  models: ModelEntry[];
  loaded: string | null;
  state: EngineState;
  can_switch: boolean;
  allow_downloads: boolean;
}

/** ``POST /api/models/remove``. ``freed_bytes`` is null when the cached scan
 *  had no size for the snapshot — "unknown", never zero. */
export interface RemovalResult {
  ok: true;
  model: string;
  freed_bytes: number | null;
}

/** ``downloads.DownloadState`` — downloads.py:62-66, plus the synthetic
 *  ``idle`` the SSE generator emits when no job exists (app.py:558). */
export type DownloadState = 'idle' | 'running' | 'done' | 'failed' | 'cancelled';

/** ``downloads.DownloadJob.to_dict`` — downloads.py:84-95. */
export interface DownloadJob {
  state: DownloadState;
  alias?: string;
  done_bytes?: number;
  total_bytes?: number | null;
  detail?: string | null;
}

/** ``GET /api/config`` — app.py:250-259. The only unauthenticated JSON
 *  endpoint, and it answers exactly one question. */
export interface ConfigResponse {
  auth_required: boolean;
}

/** ``POST /api/auth`` — app.py:261-276. */
export interface AuthResponse {
  ok: true;
  can_switch: boolean;
  allow_downloads: boolean;
}

/** The uniform error envelope — app.py:108-117. Matches the engine's own
 *  shape, so proxied and locally-generated failures have one error path. */
export interface ErrorEnvelope {
  error: { message: string; type: string };
}

/** ``images.ImageJobState``. A cancelled render is ``done``, not a state of
 *  its own: the engine stops at the next denoise step and keeps whatever
 *  finished, so an empty result with ``cancelled`` set is a success. */
export type ImageJobState = 'running' | 'done' | 'failed';

/** ``images.ImageJob.to_dict`` — what ``POST /api/images/jobs`` answers. */
export interface ImageJob {
  id: string;
  state: ImageJobState;
  b64_json: string | null;
  cancelled: boolean;
  error: { message: string; type: string; status: number } | null;
}

/** ``GET /api/images/jobs/{id}``. The job plus the engine's denoise counter
 *  while it runs — one poll answers both, so a render occupies a single
 *  connection. Diffusion has a fixed step count, so ``step / total`` is a
 *  true fraction rather than an estimate. */
export interface ImageJobSnapshot extends ImageJob {
  step?: number;
  total?: number;
}

/** One resident model — ``resident_models.ResidentModelManager.snapshot``. */
export interface ResidentModel {
  id: string;
  aliases: string[];
  state: string;
  pinned: boolean;
  estimated_bytes: number;
  /**
   * The load-time process delta, or null. A lazy engine (mflux) faults its
   * weight pages in on the FIRST request, so this can read far smaller than
   * the reservation the admission check charged — never present it as the
   * model's size on its own.
   */
  measured_bytes: number | null;
}

/** ``GET /api/residency`` — proxied from the engine's ``/v1/models/residency``.
 *  ``memory_limit_bytes`` is 0 when the engine runs without a ceiling, which
 *  is also what the route answers when the engine is unreachable. */
export interface ResidencySnapshot {
  memory_limit_bytes: number;
  memory_used_bytes: number;
  models: ResidentModel[];
}
