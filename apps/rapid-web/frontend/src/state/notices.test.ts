import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '@/api/errors';
import { noticeFor } from './notices';

const error = (type: string, message = 'server message') => new ApiError(409, type, message);

describe('noticeFor', () => {
  it('distinguishes the three 409s that model switching can return', () => {
    // The old page routed all three into the same window.alert, so "wait a
    // moment" and "this server cannot do that at all" looked identical.
    const streaming = noticeFor(error('busy_streaming'));
    const loading = noticeFor(error('busy_loading'));
    const unavailable = noticeFor(error('switch_unavailable'));

    expect(new Set([streaming.title, loading.title, unavailable.title]).size).toBe(3);
    expect(streaming.tone).toBe('warning');
    expect(loading.tone).toBe('info');
    expect(unavailable.tone).toBe('info');
  });

  it('shows the storage message verbatim', () => {
    // The server's message names the actual shortfall; a paraphrase would
    // lose the number that makes it actionable.
    const notice = noticeFor(error('insufficient_storage', 'needs 8.2 GB, 3.1 GB free'));
    expect(notice.body).toBe('needs 8.2 GB, 3.1 GB free');
    expect(notice.tone).toBe('error');
  });

  it('shows the in-use message verbatim', () => {
    // The server names which of the two holds the model open — the engine or
    // an in-flight download — and the fix differs.
    const notice = noticeFor(error('model_in_use', 'qwen3-4b is still downloading.'));
    expect(notice.body).toBe('qwen3-4b is still downloading.');
    expect(notice.tone).toBe('warning');
  });

  it('offers no recovery for a state the user cannot change', () => {
    // A button that does nothing useful is worse than no button.
    const recover = vi.fn();
    expect(noticeFor(error('switch_unavailable'), recover).action).toBeUndefined();
    expect(noticeFor(error('downloads_disabled'), recover).action).toBeUndefined();
    expect(noticeFor(error('busy_streaming'), recover).action).toBeUndefined();
    expect(noticeFor(error('insufficient_storage'), recover).action).toBeUndefined();
    // The model is held open by something the user has to change first;
    // retrying the delete would only fail again.
    expect(noticeFor(error('model_in_use'), recover).action).toBeUndefined();
  });

  it('offers a recovery where one exists', () => {
    const recover = vi.fn();
    for (const type of [
      'unknown_model',
      'catalog_error',
      'engine_unavailable',
      'engine_transport',
      'network',
      'removal_failed',
    ]) {
      const notice = noticeFor(error(type), recover);
      expect(notice.action, type).toBeDefined();
    }
  });

  it('runs the supplied recovery', () => {
    const recover = vi.fn();
    noticeFor(error('engine_unavailable'), recover).action?.run();
    expect(recover).toHaveBeenCalledOnce();
  });

  it('omits the action entirely when no recovery was supplied', () => {
    expect(noticeFor(error('engine_unavailable')).action).toBeUndefined();
  });

  it('treats a cold start as informational, not as an error', () => {
    // A model still loading is the single most common failure on this
    // surface. Painting it red would train the user to ignore red.
    expect(noticeFor(error('engine_unavailable')).tone).toBe('info');
  });

  it('names the code for a developer-facing refusal', () => {
    // A user cannot cause either of these from the app, so seeing one means
    // a proxy is rewriting headers.
    expect(noticeFor(error('origin_refused', 'refused')).body).toContain('origin_refused');
  });

  it('falls back for an unrecognised code', () => {
    const notice = noticeFor(error('something_new', 'a novel failure'));
    expect(notice.tone).toBe('error');
    expect(notice.body).toBe('a novel failure');
  });

  it('gives every code a non-empty title', () => {
    for (const type of [
      'busy_streaming',
      'busy_loading',
      'switch_unavailable',
      'unknown_model',
      'catalog_unavailable',
      'catalog_error',
      'downloads_disabled',
      'insufficient_storage',
      'download_busy',
      'model_in_use',
      'removal_failed',
      'engine_unavailable',
      'engine_transport',
      'unauthorized',
      'origin_refused',
      'unsupported_media_type',
      'network',
      'unrecognised',
    ]) {
      expect(noticeFor(error(type)).title, type).not.toBe('');
    }
  });
});
