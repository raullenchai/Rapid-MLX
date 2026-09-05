import { ApiError } from '@/api/errors';
import type { Notice } from './store';

/** `action` is explicitly optional-or-undefined: a notice for a state the
 *  user cannot change has no recovery, and a button that does nothing useful
 *  is worse than no button. */
export type NoticeSpec = Omit<Notice, 'id' | 'action'> & {
  action?: Notice['action'] | undefined;
};

/**
 * Turn a server failure into a notice with a recovery.
 *
 * The codes are the server's most useful output — a transient "wait a moment"
 * and a hard "this server cannot do that" need different copy. `recover` is
 * supplied by the caller because the right recovery depends on what was being
 * attempted: retrying a model switch is not retrying a chat turn.
 */
export function noticeFor(error: ApiError, recover?: () => void): NoticeSpec {
  const withRecovery = (label: string) => (recover ? { label, run: recover } : undefined);

  switch (error.type) {
    // ---- 409s from /api/models/load
    case 'busy_streaming':
      return {
        tone: 'warning',
        title: 'A reply is still streaming',
        body: 'Switching models would end it mid-sentence. Stop it first, or wait for it to finish.',
      };

    case 'busy_loading':
      return {
        tone: 'info',
        title: 'Another model is still loading',
        body: 'Try again once it has finished starting.',
      };

    case 'switch_unavailable':
      return {
        tone: 'info',
        title: "This server can't switch models",
        // Says why, because there is nothing the user can do here and an
        // unexplained refusal reads as a bug.
        body: "It's attached to an engine it doesn't own, so only the terminal that started it can change the model.",
      };

    // ---- catalog
    case 'unknown_model':
      return {
        tone: 'error',
        title: 'That model is not in the catalog',
        body: 'Refresh the list — it may have been added or renamed since this page loaded.',
        action: withRecovery('Refresh'),
      };

    case 'catalog_unavailable':
    case 'catalog_error':
      return {
        tone: 'warning',
        title: "Couldn't read the model catalog",
        body: error.message,
        action: withRecovery('Retry'),
      };

    // ---- downloads
    case 'downloads_disabled':
      return {
        tone: 'info',
        title: 'Downloads are off on this server',
        body: 'Pull the model from the Mac, then it will appear here as ready to start.',
      };

    case 'insufficient_storage':
      return {
        // Verbatim: the server's message names the actual shortfall, and any
        // paraphrase would lose the number that makes it actionable.
        tone: 'error',
        title: 'Not enough disk space',
        body: error.message,
      };

    case 'download_busy':
      return {
        tone: 'info',
        title: 'A download is already running',
        body: 'Only one download at a time. Wait for it, or cancel it first.',
      };

    // ---- deletion
    case 'model_in_use':
      return {
        tone: 'warning',
        // Verbatim: the server names which of the two holds it open — the
        // engine or an in-flight download — and the fix differs.
        title: 'That model is in use',
        body: error.message,
      };

    case 'removal_failed':
      return {
        tone: 'error',
        title: "Couldn't delete that model",
        body: error.message,
        action: withRecovery('Refresh'),
      };

    // ---- the chat proxy
    case 'engine_unavailable':
      return {
        tone: 'info',
        title: 'The model is still loading',
        body: error.message,
        action: withRecovery('Retry'),
      };

    case 'engine_transport':
      return {
        tone: 'error',
        title: 'Lost contact with the engine',
        body: error.message,
        action: withRecovery('Retry'),
      };

    // ---- auth and the middleware guard
    case 'unauthorized':
      return {
        tone: 'error',
        title: 'That token is no longer accepted',
        body: 'It may have been rotated. Open the link from the Mac again.',
      };

    case 'origin_refused':
    case 'unsupported_media_type':
      // Developer-facing: a user cannot cause either from the app itself, so
      // seeing one means a proxy is rewriting headers or the page is being
      // driven from somewhere unexpected.
      return {
        tone: 'error',
        title: 'The server refused the request',
        body: `${error.message} (${error.type})`,
      };

    case 'network':
      return {
        tone: 'error',
        title: "Can't reach your Mac",
        body: 'The connection dropped. Rapid-MLX is probably still running.',
        action: withRecovery('Retry'),
      };

    default:
      return {
        tone: 'error',
        title: 'Something went wrong',
        body: error.message,
        action: withRecovery('Retry'),
      };
  }
}
