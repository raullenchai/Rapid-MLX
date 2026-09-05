import type { ErrorEnvelope } from './types';

/**
 * A failure carrying the server's machine-readable ``type`` code.
 *
 * The code is the server's most useful output: ``busy_streaming`` wants "stop
 * and switch", ``insufficient_storage`` wants the message verbatim, and
 * ``unauthorized`` wants the token cleared and the gate shown.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly type: string;

  constructor(status: number, type: string, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.type = type;
  }
}

/**
 * Every ``error.type`` the server can emit, as a closed set.
 *
 * Open-ended via ``(string & {})`` because a proxied engine error carries the
 * engine's own vocabulary — the named members still autocomplete and
 * typo-check.
 */
export type ErrorCode =
  // app.py — the middleware guard
  | 'origin_refused'
  | 'unsupported_media_type'
  | 'unauthorized'
  // app.py — model switching
  | 'busy_streaming'
  | 'busy_loading'
  | 'switch_unavailable'
  | 'unknown_model'
  // app.py — catalog
  | 'catalog_unavailable'
  | 'catalog_error'
  // app.py — downloads
  | 'downloads_disabled'
  | 'insufficient_storage'
  | 'download_busy'
  // app.py — the chat proxy
  | 'engine_unavailable'
  | 'engine_transport'
  // client-side, never from the wire
  | 'network'
  | (string & {});

/**
 * Pull an ``ApiError`` out of a non-ok response.
 *
 * A body that is not the expected envelope is not an error in itself — a
 * proxy or a tunnel can return its own HTML error page — so the status code
 * is always a usable fallback.
 */
export async function errorFromResponse(response: Response): Promise<ApiError> {
  let type = `http_${response.status}`;
  let message = `request failed (${response.status})`;
  try {
    const body = (await response.json()) as Partial<ErrorEnvelope>;
    if (body.error?.message) message = body.error.message;
    if (body.error?.type) type = body.error.type;
  } catch {
    // Not JSON. The status-derived defaults above stand.
  }
  return new ApiError(response.status, type, message);
}

/** Narrow an unknown catch binding, which TypeScript types as ``unknown``. */
export function asApiError(cause: unknown): ApiError {
  if (cause instanceof ApiError) return cause;
  // A fetch rejection means the request never reached the server: the tunnel
  // dropped, the Mac slept, or the radio died. That is a materially different
  // event from any HTTP status, so it gets its own code rather than a 0.
  const message = cause instanceof Error ? cause.message : String(cause);
  return new ApiError(0, 'network', message);
}
