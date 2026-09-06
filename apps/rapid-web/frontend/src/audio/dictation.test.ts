import { beforeEach, describe, expect, it } from 'vitest';
import {
  ACTIVE_LIMIT,
  HISTORY_LIMIT,
  activeCount,
  addHistory,
  addTerm,
  loadHistory,
  loadVocabulary,
  removeTerm,
  saveVocabulary,
  setTermActive,
  vocabularyContext,
  type HistoryEntry,
  type Term,
} from './dictation';

function terms(...texts: string[]): Term[] {
  return texts.map((text) => ({ text, active: true }));
}

function fill(count: number): Term[] {
  return terms(...Array.from({ length: count }, (_, index) => `term-${index}`));
}

describe('addTerm', () => {
  it('ignores blank input', () => {
    expect(addTerm([], '   ')).toEqual([]);
  });

  it('trims and deduplicates case-insensitively', () => {
    // "Rapid" and "rapid" are one hint, and sending both spends budget twice.
    const result = addTerm(terms('Rapid'), '  rapid ');
    expect(result).toHaveLength(1);
  });

  it('parks a term added over budget instead of dropping it', () => {
    // The list is still the user's — it just is not all being sent.
    const result = addTerm(fill(ACTIVE_LIMIT), 'one-more');
    expect(result).toHaveLength(ACTIVE_LIMIT + 1);
    expect(result.at(-1)?.active).toBe(false);
    expect(activeCount(result)).toBe(ACTIVE_LIMIT);
  });
});

describe('setTermActive', () => {
  it('refuses to activate past the cap rather than silently parking another', () => {
    // Which term goes is the user's call, not the store's.
    const parked = addTerm(fill(ACTIVE_LIMIT), 'extra');
    expect(setTermActive(parked, 'extra', true)).toEqual(parked);
  });

  it('always allows parking', () => {
    const result = setTermActive(fill(ACTIVE_LIMIT), 'term-0', false);
    expect(activeCount(result)).toBe(ACTIVE_LIMIT - 1);
  });
});

describe('removeTerm', () => {
  it('drops exactly the named term', () => {
    expect(removeTerm(terms('a', 'b'), 'a').map((term) => term.text)).toEqual(['b']);
  });
});

describe('vocabularyContext', () => {
  it('is null when nothing is active — an empty field is one the engine need not parse', () => {
    expect(vocabularyContext([])).toBeNull();
    expect(vocabularyContext([{ text: 'a', active: false }])).toBeNull();
  });

  it('sends only the active terms', () => {
    expect(
      vocabularyContext([
        { text: 'Rapid', active: true },
        { text: 'parked', active: false },
        { text: 'MLX', active: true },
      ]),
    ).toBe('Rapid, MLX');
  });
});

describe('addHistory', () => {
  const entry = (id: string): HistoryEntry => ({
    id,
    text: id,
    at: 0,
    durationMs: 1000,
    latencyMs: 500,
  });

  it('puts the newest first', () => {
    const result = addHistory([entry('old')], entry('new'));
    expect(result.map((item) => item.id)).toEqual(['new', 'old']);
  });

  it('caps on write, so the stored value can never grow past the limit', () => {
    let entries: HistoryEntry[] = [];
    for (let index = 0; index < HISTORY_LIMIT + 5; index += 1) {
      entries = addHistory(entries, entry(`e${index}`));
    }
    expect(entries).toHaveLength(HISTORY_LIMIT);
  });
});

describe('persistence', () => {
  beforeEach(() => localStorage.clear());

  it('round-trips a vocabulary', () => {
    saveVocabulary([{ text: 'Rapid', active: false }]);
    expect(loadVocabulary()).toEqual([{ text: 'Rapid', active: false }]);
  });

  it('survives a corrupt value rather than taking the panel down', () => {
    localStorage.setItem('rapid-mlx-web.dictation.vocabulary', '{not json');
    localStorage.setItem('rapid-mlx-web.dictation.history', '"a string"');
    expect(loadVocabulary()).toEqual([]);
    expect(loadHistory()).toEqual([]);
  });

  it('drops malformed rows rather than the whole list', () => {
    localStorage.setItem(
      'rapid-mlx-web.dictation.vocabulary',
      JSON.stringify([{ text: 'kept' }, { text: '  ' }, null, 7]),
    );
    // A row with no `active` defaults to sent — an older build wrote them
    // without the flag.
    expect(loadVocabulary()).toEqual([{ text: 'kept', active: true }]);
  });
});
