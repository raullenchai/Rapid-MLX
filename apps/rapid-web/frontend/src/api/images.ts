import { request, requestJson, uploadJson } from './client';
import type { ImageJob, ImageJobSnapshot } from './types';

export interface StartOptions {
  prompt: string;
  /** Generation only — the edit backends derive their canvas from the source. */
  size?: string | undefined;
  /** PNG/JPEG bytes. Present means edit. */
  image?: Blob | undefined;
  model?: string | undefined;
  signal?: AbortSignal | undefined;
}

/**
 * Start a render or an edit, and get back an id rather than an image.
 *
 * The engine answers only once the whole image is finished, so relaying it
 * inline held a connection open with no bytes flowing for minutes and
 * Cloudflare cut it at 100 s with a 524. This POST returns immediately and
 * the result is collected by polling.
 *
 * Edits use a custom-header multipart upload, which forces a cross-origin
 * browser to preflight without expanding the image into Base64.
 */
export function startImageJob({
  prompt,
  size,
  image,
  model,
  signal,
}: StartOptions): Promise<ImageJob> {
  if (image) {
    const form = new FormData();
    form.append('image', image, image.type === 'image/jpeg' ? 'input.jpg' : 'input.png');
    form.append('prompt', prompt);
    if (model) form.append('model', model);
    return uploadJson<ImageJob>('/api/images/jobs', form, signal);
  }
  return requestJson<ImageJob>('/api/images/jobs', {
    method: 'POST',
    body: {
      mode: 'generation',
      prompt,
      ...(size ? { size } : {}),
      ...(model ? { model } : {}),
    },
    ...(signal ? { signal } : {}),
  });
}

/**
 * One poll: denoise progress while it runs, the image when it ends.
 *
 * Polled, NOT streamed, for the same reason the download feed is — measured
 * through a real trycloudflare tunnel, a sparse SSE body delivered headers in
 * 1.8 s and then no byte in 65 s.
 */
export function fetchImageJob(id: string, signal?: AbortSignal): Promise<ImageJobSnapshot> {
  return requestJson<ImageJobSnapshot>(
    `/api/images/jobs/${encodeURIComponent(id)}`,
    signal ? { signal } : {},
  );
}

/** Asks the engine to stop at the next denoise step. */
export async function cancelImage(model?: string): Promise<void> {
  await request('/api/images/cancel', { method: 'POST', body: model ? { model } : {} });
}
