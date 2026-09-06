import { describe, expect, it } from 'vitest';
import {
  aggregateOnDiskBytes,
  diskUsageFooter,
  filterEntries,
  largestCachedEntry,
  listHeading,
  noMatchesCopy,
  sortEntries,
  storageSummary,
} from './modelCache';
import type { ModelEntry } from '@/api/types';

function model(overrides: Partial<ModelEntry> = {}): ModelEntry {
  return {
    alias: 'qwen3-4b',
    hf_path: 'org/qwen3-4b',
    size_bytes: 4_000_000_000,
    cached: false,
    kind: 'text',
    loadable: true,
    cached_bytes: null,
    tool_call_parser: null,
    reasoning_parser: null,
    is_text_only: true,
    audio_kind: null,
    family: null,
    image_capability: null,
    ...overrides,
  };
}

describe('filterEntries', () => {
  const entries = [
    model({ alias: 'qwen3-4b', cached: true }),
    model({ alias: 'gemma-27b' }),
    model({ alias: 'qwen3-8b' }),
  ];

  it('keeps everything under "all"', () => {
    expect(filterEntries(entries, 'all', '').map((entry) => entry.alias)).toEqual([
      'qwen3-4b',
      'gemma-27b',
      'qwen3-8b',
    ]);
  });

  it('splits the catalog on the cached flag', () => {
    expect(filterEntries(entries, 'cached', '').map((entry) => entry.alias)).toEqual(['qwen3-4b']);
    expect(filterEntries(entries, 'notCached', '').map((entry) => entry.alias)).toEqual([
      'gemma-27b',
      'qwen3-8b',
    ]);
  });

  it('applies the query on top of the filter, case-insensitively', () => {
    expect(filterEntries(entries, 'notCached', 'QWEN').map((entry) => entry.alias)).toEqual([
      'qwen3-8b',
    ]);
  });

  it('ignores a query that is only whitespace', () => {
    expect(filterEntries(entries, 'all', '   ')).toHaveLength(3);
  });
});

describe('sortEntries', () => {
  it('orders by name numerically, so 8b does not precede 27b lexically', () => {
    const entries = [
      model({ alias: 'qwen3-27b' }),
      model({ alias: 'qwen3-8b' }),
      model({ alias: 'gemma-2b' }),
    ];
    expect(sortEntries(entries, 'name').map((entry) => entry.alias)).toEqual([
      'gemma-2b',
      'qwen3-8b',
      'qwen3-27b',
    ]);
  });

  it('sizes a cached row by its measurement and an uncached one by the manifest', () => {
    const entries = [
      model({ alias: 'small', size_bytes: 1e9 }),
      model({ alias: 'big', cached: true, cached_bytes: 9e9, size_bytes: 1e8 }),
    ];
    expect(sortEntries(entries, 'sizeDescending').map((entry) => entry.alias)).toEqual([
      'big',
      'small',
    ]);
  });

  it('sinks an unsized row below every sized one rather than treating it as empty', () => {
    const entries = [
      model({ alias: 'unsized', size_bytes: null }),
      model({ alias: 'tiny', size_bytes: 1 }),
    ];
    expect(sortEntries(entries, 'sizeDescending').map((entry) => entry.alias)).toEqual([
      'tiny',
      'unsized',
    ]);
  });

  it('breaks a size tie by name, so the order is stable across renders', () => {
    const entries = [model({ alias: 'b', size_bytes: 5 }), model({ alias: 'a', size_bytes: 5 })];
    expect(sortEntries(entries, 'sizeDescending').map((entry) => entry.alias)).toEqual(['a', 'b']);
  });

  it('does not mutate its input', () => {
    const entries = [model({ alias: 'b' }), model({ alias: 'a' })];
    sortEntries(entries, 'name');
    expect(entries.map((entry) => entry.alias)).toEqual(['b', 'a']);
  });
});

describe('aggregateOnDiskBytes', () => {
  it('counts only cached rows', () => {
    const usage = aggregateOnDiskBytes([
      model({ cached: true, cached_bytes: 2e9 }),
      model({ alias: 'remote', size_bytes: 9e9 }),
    ]);
    expect(usage).toEqual({ cachedCount: 1, totalBytes: 2e9, missingSizeCount: 0 });
  });

  it('reports an unmeasured cached row rather than folding it in as zero', () => {
    const usage = aggregateOnDiskBytes([
      model({ alias: 'measured', cached: true, cached_bytes: 2e9 }),
      model({ alias: 'unmeasured', cached: true, cached_bytes: null }),
    ]);
    expect(usage).toEqual({ cachedCount: 2, totalBytes: 2e9, missingSizeCount: 1 });
  });

  it('answers null, never zero, when nothing cached could be sized', () => {
    const usage = aggregateOnDiskBytes([model({ cached: true, cached_bytes: null })]);
    expect(usage.totalBytes).toBeNull();
  });
});

describe('largestCachedEntry', () => {
  it('ignores uncached rows however big the manifest says they are', () => {
    const largest = largestCachedEntry([
      model({ alias: 'huge-remote', size_bytes: 90e9 }),
      model({ alias: 'on-disk', cached: true, cached_bytes: 2e9 }),
    ]);
    expect(largest?.alias).toBe('on-disk');
  });

  it('is null when nothing on disk has a size', () => {
    expect(largestCachedEntry([model({ cached: true, cached_bytes: null })])).toBeNull();
  });
});

describe('storageSummary', () => {
  it('pairs the total with the model count', () => {
    expect(storageSummary({ cachedCount: 3, totalBytes: 2e9, missingSizeCount: 0 })).toBe(
      '1.9 GB · 3 models',
    );
  });

  it('singularises one model', () => {
    expect(storageSummary({ cachedCount: 1, totalBytes: 2e9, missingSizeCount: 0 })).toBe(
      '1.9 GB · 1 model',
    );
  });

  it('says the size is unknown rather than printing 0 B', () => {
    expect(storageSummary({ cachedCount: 2, totalBytes: null, missingSizeCount: 2 })).toBe(
      'size unknown · 2 models',
    );
  });
});

describe('listHeading', () => {
  it('shows a bare total when nothing narrows the list', () => {
    expect(listHeading('all', '', 8, 8)).toEqual({ title: 'All models', countText: '8' });
  });

  it('shows "n of m" once a search narrows it', () => {
    expect(listHeading('all', 'qwen', 2, 8)).toEqual({ title: 'All models', countText: '2 of 8' });
  });

  it('names the segment and stays narrowed even when a filter matches everything', () => {
    // Every row cached is still a statement about the CACHED segment, not
    // about the catalog, so the denominator has to stay visible.
    expect(listHeading('cached', '', 8, 8)).toEqual({ title: 'Cached', countText: '8 of 8' });
  });
});

describe('diskUsageFooter', () => {
  it('is null when nothing is cached, rather than "0 across 0 models"', () => {
    expect(diskUsageFooter({ cachedCount: 0, totalBytes: null, missingSizeCount: 0 })).toBeNull();
  });

  it('states the total across the cached count', () => {
    expect(diskUsageFooter({ cachedCount: 3, totalBytes: 2e9, missingSizeCount: 0 })).toBe(
      'Total: 1.9 GB across 3 models',
    );
  });

  it('marks a partial sum so it is not read as covering everything', () => {
    expect(diskUsageFooter({ cachedCount: 3, totalBytes: 2e9, missingSizeCount: 1 })).toBe(
      'Total: 1.9 GB across 3 models (+1 unmeasured)',
    );
  });

  it('drops the size entirely when none of the cached rows could be measured', () => {
    expect(diskUsageFooter({ cachedCount: 2, totalBytes: null, missingSizeCount: 2 })).toBe(
      'Total: 2 models (+2 unmeasured)',
    );
  });
});

describe('noMatchesCopy', () => {
  it('quotes the trimmed query when there is one', () => {
    expect(noMatchesCopy('all', '  qwen  ')).toBe('No matches for "qwen".');
  });

  it('explains each empty filter rather than repeating one generic line', () => {
    expect(noMatchesCopy('cached', '')).toContain('Nothing cached on disk yet');
    expect(noMatchesCopy('notCached', '')).toContain('already downloaded');
    expect(noMatchesCopy('all', '')).toBe('No models found.');
  });
});
