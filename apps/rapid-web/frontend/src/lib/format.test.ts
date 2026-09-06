import { describe, expect, it } from 'vitest';
import {
  dateGroupOf,
  formatBytes,
  formatDuration,
  formatRelativeTime,
  formatTokensPerSecond,
  msUntilMidnight,
} from './format';

describe('formatBytes', () => {
  it('scales through the binary units', () => {
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(2048)).toBe('2.0 KB');
    expect(formatBytes(5 * 1024 * 1024)).toBe('5.0 MB');
    expect(formatBytes(8.2 * 1024 ** 3)).toBe('8.2 GB');
  });

  it('drops the decimal at and above 10', () => {
    // 462.3 MB is not more useful than 462 MB, but 4.7 GB is more useful
    // than 5 GB.
    expect(formatBytes(462 * 1024 * 1024)).toBe('462 MB');
    expect(formatBytes(4.7 * 1024 ** 3)).toBe('4.7 GB');
  });

  it('returns null for an unknown size', () => {
    // `size_bytes` is genuinely null for models the catalog cannot size, and
    // "0 B" would be a claim rather than an absence.
    expect(formatBytes(null)).toBeNull();
    expect(formatBytes(undefined)).toBeNull();
    expect(formatBytes(0)).toBeNull();
    expect(formatBytes(-5)).toBeNull();
  });
});

describe('formatTokensPerSecond', () => {
  it('keeps one decimal below 10', () => {
    expect(formatTokensPerSecond(9.24)).toBe('9.2 tok/s');
  });

  it('rounds at and above 10', () => {
    expect(formatTokensPerSecond(122.4)).toBe('122 tok/s');
  });

  it('returns null when there is nothing to report', () => {
    expect(formatTokensPerSecond(null)).toBeNull();
    expect(formatTokensPerSecond(0)).toBeNull();
    expect(formatTokensPerSecond(Number.NaN)).toBeNull();
    expect(formatTokensPerSecond(Number.POSITIVE_INFINITY)).toBeNull();
  });
});

describe('formatDuration', () => {
  it('uses the largest readable unit', () => {
    expect(formatDuration(420)).toBe('420 ms');
    expect(formatDuration(1500)).toBe('1.5 s');
    expect(formatDuration(95_000)).toBe('1m 35s');
  });

  it('returns null for nothing measurable', () => {
    expect(formatDuration(null)).toBeNull();
    expect(formatDuration(-1)).toBeNull();
  });
});

describe('formatRelativeTime', () => {
  const now = Date.parse('2026-08-27T12:00:00Z');

  it('is coarse on purpose', () => {
    expect(formatRelativeTime(now - 30_000, now)).toBe('just now');
    expect(formatRelativeTime(now - 11 * 60_000, now)).toBe('11m ago');
    expect(formatRelativeTime(now - 5 * 3_600_000, now)).toBe('5h ago');
    expect(formatRelativeTime(now - 3 * 86_400_000, now)).toBe('3d ago');
  });
});

describe('dateGroupOf', () => {
  // 00:30 local, so "yesterday at 23:55" is only 35 minutes ago in elapsed
  // terms. This is the case an elapsed-time bucket gets wrong.
  const justAfterMidnight = new Date(2026, 7, 27, 0, 30).getTime();

  it('puts something from late last night in Yesterday, not Today', () => {
    // An elapsed-milliseconds test (`now - then < 86400000`) says Today for
    // the next 23 hours — and says it about the most recent conversation,
    // which is the one the user is most likely looking for.
    const lateLastNight = new Date(2026, 7, 26, 23, 55).getTime();
    expect(dateGroupOf(lateLastNight, justAfterMidnight)).toBe('Yesterday');
  });

  it('puts something from earlier the same calendar day in Today', () => {
    const earlierToday = new Date(2026, 7, 27, 0, 5).getTime();
    expect(dateGroupOf(earlierToday, justAfterMidnight)).toBe('Today');
  });

  it('buckets the wider ranges', () => {
    const now = new Date(2026, 7, 27, 12, 0).getTime();
    expect(dateGroupOf(new Date(2026, 7, 24, 9, 0).getTime(), now)).toBe('Previous 7 days');
    expect(dateGroupOf(new Date(2026, 7, 10, 9, 0).getTime(), now)).toBe('Previous 30 days');
    expect(dateGroupOf(new Date(2026, 3, 1, 9, 0).getTime(), now)).toBe('Older');
  });
});

describe('msUntilMidnight', () => {
  it('counts to the next local midnight', () => {
    const now = new Date(2026, 7, 27, 23, 30, 0).getTime();
    expect(msUntilMidnight(now)).toBe(30 * 60_000);
  });

  it('is a full day just after midnight, never zero or negative', () => {
    // A zero here would spin the sidebar's ticker in a tight loop.
    const now = new Date(2026, 7, 27, 0, 0, 0).getTime();
    expect(msUntilMidnight(now)).toBe(86_400_000);
    expect(msUntilMidnight(now)).toBeGreaterThan(0);
  });
});
