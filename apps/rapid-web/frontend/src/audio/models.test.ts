import { describe, expect, it } from 'vitest';
import type { ModelEntry } from '@/api/types';
import {
  audioModels,
  modelDetails,
  preferredAlias,
  previewText,
  voiceDetails,
} from './models';

function entry(overrides: Partial<ModelEntry> & { alias: string }): ModelEntry {
  return {
    hf_path: `mlx-community/${overrides.alias}`,
    size_bytes: null,
    cached: false,
    kind: 'audio',
    loadable: false,
    cached_bytes: null,
    tool_call_parser: null,
    reasoning_parser: null,
    is_text_only: false,
    audio_kind: 'stt',
    family: 'whisper',
    image_capability: null,
    ...overrides,
  };
}

describe('modelDetails', () => {
  it('names a model rather than echoing its alias', () => {
    const details = modelDetails(entry({ alias: 'whisper-large-v3-turbo' }));
    expect(details.displayName).toBe('Whisper Large v3 Turbo');
    expect(details.recommended).toBe(true);
  });

  it('resolves quantisation spellings to the same entry', () => {
    // The registry publishes `kokoro-4bit` AND `kokoro-82m-4bit` for one
    // checkpoint; tabulating both is how one gets missed.
    const short = modelDetails(entry({ alias: 'kokoro-4bit', audio_kind: 'tts' }));
    const long = modelDetails(entry({ alias: 'kokoro-82m-4bit', audio_kind: 'tts' }));
    expect(long).toEqual(short);
  });

  it('describes an unknown alias from its family rather than leaving it bare', () => {
    const details = modelDetails(
      entry({ alias: 'some-new-model', audio_kind: 'tts', family: 'new_family' }),
    );
    expect(details.displayName).toBe('some-new-model');
    expect(details.badge).toBe('new family');
    expect(details.summary).toContain('text-to-speech');
  });

  it('describes the same alias differently in each direction', () => {
    // `audio_kind` picks the table: an alias present in both must not be
    // described as the wrong kind of model.
    const stt = modelDetails(entry({ alias: 'unknown', audio_kind: 'stt', family: 'x' }));
    const tts = modelDetails(entry({ alias: 'unknown', audio_kind: 'tts', family: 'x' }));
    expect(stt.summary).toContain('speech-to-text');
    expect(tts.summary).toContain('text-to-speech');
  });
});

describe('audioModels', () => {
  it('shows one row per checkpoint, keeping the recognisable alias', () => {
    const rows = audioModels(
      [
        entry({ alias: 'whisper-1', hf_path: 'mlx-community/whisper-large-v3-mlx' }),
        entry({ alias: 'whisper-large-v3', hf_path: 'mlx-community/whisper-large-v3-mlx' }),
        entry({ alias: 'whisper', hf_path: 'mlx-community/whisper-large-v3-mlx' }),
      ],
      'stt',
    );
    expect(rows.map((row) => row.alias)).toEqual(['whisper-large-v3']);
  });

  it('separates the two directions', () => {
    const models = [
      entry({ alias: 'whisper-small' }),
      entry({ alias: 'kokoro', audio_kind: 'tts', family: 'kokoro' }),
    ];
    expect(audioModels(models, 'stt').map((row) => row.alias)).toEqual(['whisper-small']);
    expect(audioModels(models, 'tts').map((row) => row.alias)).toEqual(['kokoro']);
  });

  it('excludes non-audio rows', () => {
    const rows = audioModels(
      [entry({ alias: 'qwen3-4b', kind: 'text', audio_kind: null, family: null })],
      'stt',
    );
    expect(rows).toEqual([]);
  });

  it('ranks recommended first, then downloaded', () => {
    const rows = audioModels(
      [
        entry({ alias: 'whisper-base', hf_path: 'a/base' }),
        entry({ alias: 'whisper-small', hf_path: 'a/small', cached: true }),
        entry({ alias: 'whisper-large-v3-turbo', hf_path: 'a/turbo' }),
      ],
      'stt',
    );
    expect(rows.map((row) => row.alias)).toEqual([
      'whisper-large-v3-turbo',
      'whisper-small',
      'whisper-base',
    ]);
  });
});

describe('preferredAlias', () => {
  it('prefers something already on disk over any built-in preference', () => {
    // A default that is not downloaded opens the panel onto a model the lane
    // cannot run.
    const chosen = preferredAlias(
      [entry({ alias: 'whisper-base', cached: true }), entry({ alias: 'whisper-small' })],
      ['whisper-small'],
    );
    expect(chosen).toBe('whisper-base');
  });

  it('falls back to the preference list when nothing is cached', () => {
    const chosen = preferredAlias(
      [entry({ alias: 'whisper-base' }), entry({ alias: 'whisper-small' })],
      ['whisper-small'],
    );
    expect(chosen).toBe('whisper-small');
  });

  it('is null when there is nothing to choose from', () => {
    expect(preferredAlias([], ['whisper-small'])).toBeNull();
  });
});

describe('voiceDetails', () => {
  it('decodes the Kokoro id scheme rather than tabulating 54 names', () => {
    expect(voiceDetails('af_heart')).toBe('American English · Female');
    expect(voiceDetails('bm_george')).toBe('British English · Male');
    expect(voiceDetails('jf_alpha')).toBe('Japanese · Female');
  });

  it('names Qwen3 speakers, whose ids are not systematic', () => {
    expect(voiceDetails('uncle_fu')).toBe('Chinese · Male');
  });

  it('does not guess at an id it cannot decode', () => {
    expect(voiceDetails('something-else')).toBe('Multilingual');
  });
});

describe('previewText', () => {
  it('speaks the voice’s own language — an English sample proves nothing', () => {
    expect(previewText('zf_xiaobei')).toContain('你好');
    expect(previewText('jf_alpha')).toContain('こんにちは');
    expect(previewText('sohee')).toContain('안녕하세요');
    expect(previewText('af_heart')).toContain('Hello');
  });
});
