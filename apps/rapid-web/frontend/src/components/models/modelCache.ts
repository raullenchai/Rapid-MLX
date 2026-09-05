import type { ModelEntry } from '@/api/types';
import { formatBytes } from '@/lib/format';

/**
 * The filter / sort / aggregate helpers behind the Models panel, ported from
 * `rapid-mac`'s `Services/ModelCacheActions.swift`.
 *
 * Pure and separate from the view for the same reason they are there: these
 * are truth tables, and a truth table should be checkable without rendering
 * anything.
 */

export type FilterMode = 'all' | 'cached' | 'notCached';

export const FILTER_LABELS: Record<FilterMode, string> = {
  all: 'All',
  cached: 'Cached',
  notCached: 'Not cached',
};

/**
 * How the panel orders the filtered list.
 *
 * rapid-mac's default — family, then size within family — is deliberately
 * absent. It reads the family off `ModelInfoCatalog`, and this catalog carries
 * `family` for audio rows only; deriving one by chopping up the alias string
 * would group models by a guess.
 */
export type SortOrder = 'name' | 'sizeDescending';

export const SORT_LABELS: Record<SortOrder, string> = {
  name: 'Name',
  sizeDescending: 'Size (largest first)',
};

/** The bytes a row is sized by: measured when on disk, the manifest otherwise. */
export function entryBytes(entry: ModelEntry): number | null {
  return (entry.cached ? entry.cached_bytes : entry.size_bytes) ?? null;
}

export function filterEntries(
  entries: ModelEntry[],
  mode: FilterMode,
  query: string,
): ModelEntry[] {
  const term = query.trim().toLowerCase();
  return entries.filter((entry) => {
    if (mode === 'cached' && !entry.cached) return false;
    if (mode === 'notCached' && entry.cached) return false;
    return term === '' || entry.alias.toLowerCase().includes(term);
  });
}

/** Stable, so the render order does not shuffle under a mid-delete re-render. */
export function sortEntries(entries: ModelEntry[], order: SortOrder): ModelEntry[] {
  const byName = (left: ModelEntry, right: ModelEntry) =>
    left.alias.localeCompare(right.alias, undefined, { numeric: true });

  if (order === 'name') return [...entries].sort(byName);

  return [...entries].sort((left, right) => {
    // -1, not 0: an unsized row sorts below every sized one rather than
    // colliding with a genuinely empty model.
    const difference = (entryBytes(right) ?? -1) - (entryBytes(left) ?? -1);
    return difference !== 0 ? difference : byName(left, right);
  });
}

export interface DiskUsage {
  cachedCount: number;
  /** Null when NO cached row could be sized — "unknown", never zero. */
  totalBytes: number | null;
  /** Cached rows the catalog had no byte count for. */
  missingSizeCount: number;
}

export function aggregateOnDiskBytes(entries: ModelEntry[]): DiskUsage {
  let total = 0;
  let anyMeasured = false;
  let cachedCount = 0;
  let missing = 0;

  for (const entry of entries) {
    if (!entry.cached) continue;
    cachedCount += 1;
    const bytes = entry.cached_bytes;
    if (bytes !== null && bytes > 0) {
      total += bytes;
      anyMeasured = true;
    } else {
      missing += 1;
    }
  }

  return { cachedCount, totalBytes: anyMeasured ? total : null, missingSizeCount: missing };
}

/** The biggest thing on disk — the row that answers "what do I delete?". */
export function largestCachedEntry(entries: ModelEntry[]): ModelEntry | null {
  let best: ModelEntry | null = null;
  for (const entry of entries) {
    if (!entry.cached || entry.cached_bytes === null) continue;
    if (best === null || entry.cached_bytes > (best.cached_bytes ?? 0)) best = entry;
  }
  return best;
}

/** "12.4 GB · 3 models" for the Disk overview card. */
export function storageSummary(usage: DiskUsage): string {
  const used = usage.totalBytes === null ? 'size unknown' : (formatBytes(usage.totalBytes) ?? '0 B');
  return `${used} · ${usage.cachedCount} model${usage.cachedCount === 1 ? '' : 's'}`;
}

export interface ListHeading {
  title: string;
  countText: string;
}

/**
 * The list's own heading. It names the segment AND how much of the catalog is
 * showing, so a search that hides most of the list cannot read as a catalog
 * that lost most of its models.
 *
 * The query is deliberately not echoed: it is legible in the field just above.
 */
export function listHeading(
  mode: FilterMode,
  query: string,
  visibleCount: number,
  totalCount: number,
): ListHeading {
  const narrowed = query.trim() !== '' || mode !== 'all' || visibleCount !== totalCount;
  return {
    title: mode === 'all' ? 'All models' : FILTER_LABELS[mode],
    countText: narrowed ? `${visibleCount} of ${totalCount}` : String(totalCount),
  };
}

/**
 * "Total: 12.4 GB across 3 models". Null when nothing is cached — a
 * "Total: 0 across 0 models" line reads as an error rather than an empty disk.
 *
 * When some cached rows have a size and others do not, the `(+N unmeasured)`
 * suffix says the sum covers a subset. A partial sum dressed as a complete one
 * is worse than the truth.
 */
export function diskUsageFooter(usage: DiskUsage): string | null {
  if (usage.cachedCount === 0) return null;
  const models = `${usage.cachedCount} model${usage.cachedCount === 1 ? '' : 's'}`;
  const unmeasured = usage.missingSizeCount > 0 ? ` (+${usage.missingSizeCount} unmeasured)` : '';
  const size = usage.totalBytes === null ? null : formatBytes(usage.totalBytes);
  return size === null
    ? `Total: ${models}${unmeasured}`
    : `Total: ${size} across ${models}${unmeasured}`;
}

/** What the list says when it has no rows to show. */
export function noMatchesCopy(mode: FilterMode, query: string): string {
  if (query.trim() !== '') return `No matches for "${query.trim()}".`;
  switch (mode) {
    case 'cached':
      return 'Nothing cached on disk yet. Pick a row from "Not cached" and select it to download.';
    case 'notCached':
      return 'Every model in the catalog is already downloaded.';
    default:
      return 'No models found.';
  }
}
