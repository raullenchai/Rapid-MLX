/**
 * Formatting shared across the UI. All pure, all with an injectable clock —
 * a helper that reads `Date.now()` internally cannot be tested for the
 * boundary that matters.
 */

/**
 * Binary byte sizes.
 *
 * Returns null rather than "0 B" for a missing size, so a caller can omit the
 * element entirely — `size_bytes` is genuinely null for models the catalog
 * cannot size.
 */
export function formatBytes(bytes: number | null | undefined): string | null {
  if (bytes === null || bytes === undefined || bytes <= 0) return null;

  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }

  // One decimal below 10 (4.7 GB reads better than 5 GB), none above (462 MB
  // does not benefit from 462.3 MB).
  return `${value < 10 && unit > 0 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

/** Throughput caption, matching the Mac app's AssistantStatsFormatter. */
export function formatTokensPerSecond(tps: number | null): string | null {
  if (tps === null || !Number.isFinite(tps) || tps <= 0) return null;
  // One decimal under 10, integer at or above: the difference between 9.2 and
  // 9.7 tok/s is meaningful, the difference between 122 and 122.4 is not.
  return `${tps < 10 ? tps.toFixed(1) : Math.round(tps)} tok/s`;
}

/** Elapsed time, in the largest unit that stays readable. */
export function formatDuration(ms: number | null): string | null {
  if (ms === null || !Number.isFinite(ms) || ms < 0) return null;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

/**
 * Relative time for the conversation list.
 *
 * Deliberately coarse. A transcript from eleven minutes ago and one from
 * fourteen are the same thing to a user scanning a list.
 */
export function formatRelativeTime(timestamp: number, now: number): string {
  const elapsed = now - timestamp;
  if (elapsed < 60_000) return 'just now';
  if (elapsed < 3_600_000) return `${Math.floor(elapsed / 60_000)}m ago`;
  if (elapsed < 86_400_000) return `${Math.floor(elapsed / 3_600_000)}h ago`;
  return `${Math.floor(elapsed / 86_400_000)}d ago`;
}

export type DateGroup = 'Today' | 'Yesterday' | 'Previous 7 days' | 'Previous 30 days' | 'Older';

/**
 * Which date bucket a conversation belongs to.
 *
 * Bucketed on CALENDAR days, not elapsed milliseconds: something sent at 23:55
 * belongs to "Yesterday" once the clock passes midnight, which a
 * `now - then < 86400000` test gets wrong for the next 23 hours — and gets
 * wrong for the most recent conversation specifically.
 */
export function dateGroupOf(timestamp: number, now: number): DateGroup {
  const startOfToday = new Date(now);
  startOfToday.setHours(0, 0, 0, 0);
  const today = startOfToday.getTime();

  if (timestamp >= today) return 'Today';
  const day = 86_400_000;
  if (timestamp >= today - day) return 'Yesterday';
  if (timestamp >= today - 7 * day) return 'Previous 7 days';
  if (timestamp >= today - 30 * day) return 'Previous 30 days';
  return 'Older';
}

/** Milliseconds until the next local midnight, for the sidebar's ticker. */
export function msUntilMidnight(now: number): number {
  const next = new Date(now);
  next.setHours(24, 0, 0, 0);
  return next.getTime() - now;
}
