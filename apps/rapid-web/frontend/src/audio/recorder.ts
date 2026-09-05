/**
 * Microphone capture, as a small state machine over `MediaRecorder`.
 *
 * The take is transcoded to 16 kHz mono WAV before it leaves (`toWav`).
 * `MediaRecorder` cannot be asked for WAV — Safari records mp4, Chrome and
 * Firefox record webm — and the engine decodes with libsndfile, which
 * supports NEITHER. Sending the native output failed on every browser with
 * `could not decode audio file`. `rapid-mac`'s `DictationRecorder` writes the
 * same 16 kHz mono WAV for the same reason.
 */

import { toWav } from './wav';

/** Matches the engine's own ceiling, so a take that cannot be sent is not made. */
export const MAX_RECORDING_BYTES = 25 * 1024 * 1024;

/** Keep enough headroom below the byte cap for timer throttling and WAV framing. */
export const MAX_RECORDING_MS = 12 * 60 * 1000;

/** A take under this is a mis-tap, not speech. */
export const MIN_RECORDING_MS = 250;

/**
 * The first supported type wins.
 *
 * Safari produces mp4 and refuses webm; Chrome and Firefox do the reverse.
 * An unsupported `mimeType` makes the constructor throw, so this is asked
 * rather than assumed.
 */
const PREFERRED_TYPES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/mp4',
  'audio/ogg;codecs=opus',
];

export function pickMimeType(
  isSupported: (type: string) => boolean = (type) =>
    typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(type),
): string | undefined {
  return PREFERRED_TYPES.find(isSupported);
}

export class RecorderError extends Error {
  constructor(
    message: string,
    readonly kind: 'denied' | 'unsupported' | 'empty' | 'tooLarge' | 'undecodable',
  ) {
    super(message);
    this.name = 'RecorderError';
  }
}

export interface Recording {
  blob: Blob;
  durationMs: number;
}

/**
 * One recording session.
 *
 * `stop()` resolves with the audio. The stream's tracks are stopped on every
 * exit path — leaving them live keeps the browser's recording indicator on
 * and holds the microphone against other apps.
 */
export class Recorder {
  private recorder: MediaRecorder | null = null;
  private stream: MediaStream | null = null;
  private chunks: Blob[] = [];
  private startedAt = 0;
  private limitTimer: ReturnType<typeof setTimeout> | null = null;

  get recording(): boolean {
    return this.recorder !== null;
  }

  async start(onLimit?: () => void): Promise<void> {
    if (this.recorder) return;

    if (typeof MediaRecorder === 'undefined' || !navigator.mediaDevices) {
      throw new RecorderError('This browser cannot record audio.', 'unsupported');
    }

    let stream: MediaStream;
    try {
      // 16 kHz mono is what the STT lane wants; these are hints, and a
      // browser that ignores them still produces something decodable.
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true },
      });
    } catch {
      throw new RecorderError(
        'Rapid needs microphone access to record. Allow it in your browser’s site settings and try again.',
        'denied',
      );
    }

    const mimeType = pickMimeType();
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    this.chunks = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) this.chunks.push(event.data);
    };

    this.stream = stream;
    this.recorder = recorder;
    this.startedAt = Date.now();
    recorder.start();
    if (onLimit) {
      this.limitTimer = setTimeout(onLimit, MAX_RECORDING_MS);
    }
  }

  /** Stop and resolve with the take, transcoded to WAV. Throws if it was too
   *  short, too big, or could not be decoded. */
  async stop(): Promise<Recording> {
    const recorder = this.recorder;
    if (!recorder) throw new RecorderError('Nothing was being recorded.', 'empty');

    const durationMs = Date.now() - this.startedAt;
    if (this.limitTimer !== null) {
      clearTimeout(this.limitTimer);
      this.limitTimer = null;
    }
    const recorded = await new Promise<Blob>((resolve) => {
      recorder.onstop = () => resolve(new Blob(this.chunks, { type: recorder.mimeType }));
      recorder.stop();
    });
    this.release();

    if (durationMs < MIN_RECORDING_MS || recorded.size === 0) {
      throw new RecorderError('That recording was too short.', 'empty');
    }
    // Check time before WebAudio expands the compressed take into a full
    // decoded buffer. Background-tab timer throttling can delay auto-stop.
    if (durationMs > MAX_RECORDING_MS + 1000) {
      throw new RecorderError('Recordings are limited to 12 minutes.', 'tooLarge');
    }

    // Transcoded HERE, not sent as recorded: the engine decodes with
    // libsndfile, which supports neither mp4 (Safari) nor webm (Chrome,
    // Firefox), so the raw take fails with `could not decode audio file` on
    // every browser. See audio/wav.ts.
    let blob: Blob;
    try {
      blob = await toWav(recorded);
    } catch {
      throw new RecorderError('That recording could not be processed.', 'undecodable');
    }

    // Checked AFTER the transcode, because that is what gets uploaded — and
    // 16 kHz mono PCM is a different size from a compressed take.
    if (blob.size > MAX_RECORDING_BYTES) {
      throw new RecorderError(
        `That recording is larger than ${MAX_RECORDING_BYTES / (1024 * 1024)} MB.`,
        'tooLarge',
      );
    }
    return { blob, durationMs };
  }

  /** Abandon the take. Safe to call when not recording. */
  cancel(): void {
    const recorder = this.recorder;
    if (recorder) {
      recorder.onstop = null;
      // `stop()` on an already-inactive recorder throws.
      if (recorder.state !== 'inactive') recorder.stop();
    }
    this.release();
  }

  private release(): void {
    if (this.limitTimer !== null) {
      clearTimeout(this.limitTimer);
      this.limitTimer = null;
    }
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.recorder = null;
    this.chunks = [];
  }
}

/** `1:04`, for the elapsed readout. */
export function formatDuration(ms: number): string {
  const total = Math.floor(ms / 1000);
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}
