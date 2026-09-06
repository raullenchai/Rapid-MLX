import { describe, expect, it } from 'vitest';
import { displayBytes, displayName, memorySummary } from './ResidencyPanel';
import type { ResidentModel } from '@/api/types';

function model(overrides: Partial<ResidentModel> = {}): ResidentModel {
  return {
    id: 'org/qwen3-4b',
    aliases: ['qwen3-4b'],
    state: 'resident',
    pinned: false,
    estimated_bytes: 4_000_000_000,
    measured_bytes: null,
    ...overrides,
  };
}

describe('displayBytes', () => {
  it('keeps the reservation when the measured delta is smaller', () => {
    // A lazy engine's load-time delta covers metadata only; its first
    // request materialises the weights.
    expect(displayBytes(model({ estimated_bytes: 4e9, measured_bytes: 2e8 }))).toBe(4e9);
  });

  it('prefers the measurement once it exceeds the reservation', () => {
    expect(displayBytes(model({ estimated_bytes: 4e9, measured_bytes: 5e9 }))).toBe(5e9);
  });

  it('falls back to the reservation when nothing was measured', () => {
    expect(displayBytes(model({ measured_bytes: null }))).toBe(4_000_000_000);
  });
});

describe('displayName', () => {
  it('uses the served alias when it names this model', () => {
    expect(displayName(model({ aliases: ['qwen3-4b', 'qwen'] }), 'qwen')).toBe('qwen');
  });

  it('ignores a served alias belonging to another model', () => {
    expect(displayName(model(), 'flux2-klein-4b')).toBe('qwen3-4b');
  });

  it('prefers the shortest alias over the resolved repo id', () => {
    expect(displayName(model({ aliases: ['qwen3-4b-instruct', 'qwen3-4b'] }), null)).toBe(
      'qwen3-4b',
    );
  });

  it('falls back to the id when there are no aliases', () => {
    expect(displayName(model({ aliases: [] }), null)).toBe('org/qwen3-4b');
  });
});

describe('memorySummary', () => {
  it('reads as used over the ceiling', () => {
    expect(
      memorySummary({ memory_limit_bytes: 25 * 1024 ** 3, memory_used_bytes: 9.75e9, models: [] }),
    ).toBe('9.1 GB / 25 GB');
  });

  it('omits the denominator when the engine has no ceiling', () => {
    expect(memorySummary({ memory_limit_bytes: 0, memory_used_bytes: 9.75e9, models: [] })).toBe(
      '9.1 GB',
    );
  });
});
