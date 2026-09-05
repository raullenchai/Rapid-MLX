import { describe, expect, it } from 'vitest';
import type { ModelEntry } from '@/api/types';
import { laneIsLive, laneState } from './AudioView';

function model(overrides: Partial<ModelEntry> & { alias: string }): ModelEntry {
  return {
    hf_path: `mlx-community/${overrides.alias}`,
    size_bytes: null,
    cached: true,
    kind: 'audio',
    loadable: true,
    cached_bytes: 1000,
    tool_call_parser: null,
    reasoning_parser: null,
    is_text_only: false,
    audio_kind: 'tts',
    family: 'kokoro',
    image_capability: null,
    ...overrides,
  };
}

const kokoro = model({ alias: 'kokoro' });

/**
 * The two-branch shape is `ServerManager.ensureVoiceLane`: a serving engine
 * co-loads speech, an idle one has to be given the voice model.
 */
describe('laneState', () => {
  it('is live on ANY ready model — the lane rides on a chat model', () => {
    // The whole reason audio is not a model switch: `--enable-audio` mounts
    // `/v1/audio/*` on whatever is serving.
    const state = laneState({ state: 'ready', model: 'qwen3-4b' }, kokoro, true);
    expect(state.kind).toBe('blocked');
    expect(laneIsLive(state)).toBe(true);
    expect(state.detail).toContain('qwen3-4b');
  });

  it('is live when the audio model IS the served model', () => {
    const state = laneState({ state: 'ready', model: 'kokoro' }, kokoro, true);
    expect(state.kind).toBe('live');
    expect(laneIsLive(state)).toBe(true);
  });

  it('offers a start when the engine is idle', () => {
    // Nothing to co-load onto, so `serve <audio-alias>` is the path — the
    // CLI has a dedicated audio-serve fork for exactly this.
    const state = laneState({ state: 'stopped', model: null }, kokoro, true);
    expect(state.kind).toBe('idle');
    expect(laneIsLive(state)).toBe(false);
  });

  it('never offers to start a model that is not downloaded', () => {
    const state = laneState(
      { state: 'stopped', model: null },
      model({ alias: 'kokoro', cached: false }),
      true,
    );
    expect(state.kind).toBe('missing');
    expect(state.detail).toContain('rapid-mlx pull');
  });

  it('reports a load in progress rather than offering a second start', () => {
    const state = laneState({ state: 'starting', model: 'qwen3-4b' }, kokoro, true);
    expect(state.kind).toBe('starting');
    expect(laneIsLive(state)).toBe(false);
  });

  it('does not claim an empty catalog until the catalog has loaded', () => {
    // "no audio model" and "not asked yet" are different, and saying the
    // former during boot sends the user off to pull something they have.
    expect(laneState(null, null, false).detail).toContain('Reading the catalog');
    expect(laneState(null, null, true).detail).toContain('No audio model');
  });

  it('treats a null status as idle rather than as ready', () => {
    // `selectAlias` nulls the status on a switch; failing toward "not
    // running" is the honest direction.
    expect(laneState(null, kokoro, true).kind).toBe('idle');
  });
});
