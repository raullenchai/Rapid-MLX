import { request, requestJson } from './client';
import type {
  DownloadJob,
  ModelsResponse,
  RemovalResult,
  ResidencySnapshot,
  StatusResponse,
} from './types';

export function fetchStatus(signal?: AbortSignal): Promise<StatusResponse> {
  return requestJson<StatusResponse>('/api/status', signal ? { signal } : {});
}

export function fetchModels(refresh = false): Promise<ModelsResponse> {
  return requestJson<ModelsResponse>(`/api/models${refresh ? '?refresh=true' : ''}`);
}

/**
 * Switch the loaded model.
 *
 * Kills the engine child, so the server refuses with 409 while a chat stream
 * is relaying (``busy_streaming``), while another load is in flight
 * (``busy_loading``), or in --attach mode (``switch_unavailable``). All arrive
 * as an ``ApiError`` carrying the code.
 */
export function loadModel(alias: string): Promise<{ ok: true; model: string; state: string }> {
  return requestJson('/api/models/load', {
    method: 'POST',
    body: { model: alias },
  });
}

export function pullModel(alias: string): Promise<DownloadJob> {
  return requestJson<DownloadJob>('/api/models/pull', {
    method: 'POST',
    body: { model: alias },
  });
}

/**
 * Delete a model's weights from the Mac's HuggingFace cache.
 *
 * POST rather than DELETE, matching the server: the CSRF content-type check
 * runs on POST/PUT/PATCH. Refused with 409 ``model_in_use`` when the alias is
 * running or mid-download.
 */
export function removeModel(alias: string): Promise<RemovalResult> {
  return requestJson<RemovalResult>('/api/models/remove', {
    method: 'POST',
    body: { model: alias },
  });
}

export async function cancelDownload(): Promise<void> {
  await request('/api/downloads/cancel', { method: 'POST', body: {} });
}

/**
 * The current download job, or ``{ state: 'idle' }`` when there is none.
 *
 * Polled, NOT streamed. Measured against a real ``trycloudflare`` tunnel, the
 * SSE feed this replaced delivered headers in 1.8 s and then no body byte in
 * 65 s (loopback: 0.0 s). Cloudflare strips ``X-Accel-Buffering`` and padding
 * the first frame did not help. Chat streaming survives the same tunnel
 * because it emits tokens continuously — sparseness is the variable.
 */
export function fetchDownload(signal?: AbortSignal): Promise<DownloadJob> {
  return requestJson<DownloadJob>('/api/downloads/status', signal ? { signal } : {});
}

/**
 * What the engine is holding in memory, and against what ceiling.
 *
 * The route answers an empty snapshot rather than an error when the engine
 * is unreachable, so a switch does not need special handling here.
 */
export function fetchResidency(signal?: AbortSignal): Promise<ResidencySnapshot> {
  return requestJson<ResidencySnapshot>('/api/residency', signal ? { signal } : {});
}
