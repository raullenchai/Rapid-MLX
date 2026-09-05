import { request, requestJson, uploadJson } from './client';

/**
 * The audio lane.
 *
 * It rides on WHATEVER model the engine is serving — the child is spawned
 * with `--enable-audio` — so nothing here switches models. Speech works while
 * a chat model is loaded, which is the whole reason audio is not in the model
 * picker.
 */

export interface Transcription {
  text: string;
  language?: string | null;
  duration?: number | null;
}

export function fetchVoices(model?: string): Promise<{ voices: string[] }> {
  const query = model ? `?model=${encodeURIComponent(model)}` : '';
  return requestJson<{ voices: string[] }>(`/api/audio/voices${query}`);
}

export interface SpeechOptions {
  input: string;
  model: string;
  voice: string;
  speed: number;
  signal?: AbortSignal | undefined;
}

/**
 * Synthesise speech. Resolves to the audio itself, not JSON.
 *
 * `response_format` is pinned to wav: it is the one format every family
 * emits, and `<audio>` plays it everywhere. mp3/opus are engine-side options
 * that would only add a control with no audible benefit here.
 */
export async function synthesize({
  input,
  model,
  voice,
  speed,
  signal,
}: SpeechOptions): Promise<Blob> {
  const response = await request('/api/audio/speech', {
    method: 'POST',
    body: { input, model, voice, speed, response_format: 'wav' },
    ...(signal ? { signal } : {}),
  });
  return await response.blob();
}

export interface TranscribeOptions {
  audio: Blob;
  model?: string | undefined;
  /** Proper nouns to bias the decoder toward. */
  context?: string | undefined;
  signal?: AbortSignal | undefined;
}

/**
 * Transcribe a recording.
 *
 * The custom upload header makes multipart non-simple for CORS, preserving the
 * server's CSRF boundary without Base64's wire and memory expansion.
 */
export async function transcribe({
  audio,
  model,
  context,
  signal,
}: TranscribeOptions): Promise<Transcription> {
  const form = new FormData();
  form.append('file', audio, filenameFor(audio.type));
  if (model) form.append('model', model);
  if (context) form.append('context', context);
  return uploadJson<Transcription>('/api/audio/transcriptions', form, signal);
}

/**
 * Advisory only — the engine spools every upload to a `.wav` temp file and
 * decodes the CONTAINER, not the extension. Recordings are already
 * transcoded to WAV by `audio/recorder.ts`, so this reports what the bytes
 * actually are rather than renaming them into something they are not.
 */
function filenameFor(mime: string): string {
  if (mime.includes('webm')) return 'recording.webm';
  if (mime.includes('mp4') || mime.includes('m4a')) return 'recording.mp4';
  if (mime.includes('ogg')) return 'recording.ogg';
  return 'recording.wav';
}
