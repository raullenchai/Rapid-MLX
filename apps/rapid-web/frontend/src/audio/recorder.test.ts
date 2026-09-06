import { afterEach, describe, expect, it, vi } from 'vitest';
import { MAX_RECORDING_MS, Recorder, formatDuration, pickMimeType } from './recorder';

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

/**
 * The recorded container is NOT what gets uploaded.
 *
 * `MediaRecorder` cannot produce WAV, and the engine decodes with libsndfile,
 * which supports neither mp4 (Safari) nor webm (Chrome/Firefox) — so the take
 * is transcoded before it leaves. These specs pin the CHOICE of container;
 * the transcode itself is covered by `wav.test.ts` and by an e2e that drives
 * a real browser.
 */
describe('pickMimeType', () => {
  it('prefers opus-in-webm where it is supported', () => {
    expect(pickMimeType(() => true)).toBe('audio/webm;codecs=opus');
  });

  it('falls back to mp4 on Safari, which refuses webm', () => {
    // Safari produces mp4 and Chrome/Firefox produce webm, so the type is
    // asked for rather than assumed — an unsupported `mimeType` makes the
    // MediaRecorder constructor throw.
    expect(pickMimeType((type) => type === 'audio/mp4')).toBe('audio/mp4');
  });

  it('returns undefined when nothing is supported, so the default is used', () => {
    expect(pickMimeType(() => false)).toBeUndefined();
  });
});

describe('formatDuration', () => {
  it('pads the seconds', () => {
    expect(formatDuration(0)).toBe('0:00');
    expect(formatDuration(4_000)).toBe('0:04');
    expect(formatDuration(64_000)).toBe('1:04');
    expect(formatDuration(600_000)).toBe('10:00');
  });

  it('floors rather than rounding, so a timer never shows a second early', () => {
    expect(formatDuration(1_999)).toBe('0:01');
  });
});

describe('recording limit', () => {
  it('automatically asks the UI to stop before a long take is transcoded', async () => {
    vi.useFakeTimers();
    const stopTrack = vi.fn();
    vi.stubGlobal('navigator', {
      mediaDevices: {
        getUserMedia: vi.fn(async () => ({ getTracks: () => [{ stop: stopTrack }] })),
      },
    });
    class FakeMediaRecorder {
      static isTypeSupported = () => true;
      state = 'inactive';
      mimeType = 'audio/webm';
      ondataavailable: ((event: { data: Blob }) => void) | null = null;
      onstop: (() => void) | null = null;
      start() {
        this.state = 'recording';
      }
      stop() {
        this.state = 'inactive';
        this.onstop?.();
      }
    }
    vi.stubGlobal('MediaRecorder', FakeMediaRecorder);

    const reachedLimit = vi.fn();
    const recorder = new Recorder();
    await recorder.start(reachedLimit);
    await vi.advanceTimersByTimeAsync(MAX_RECORDING_MS);

    expect(reachedLimit).toHaveBeenCalledOnce();
    recorder.cancel();
    expect(stopTrack).toHaveBeenCalledOnce();
  });
});
