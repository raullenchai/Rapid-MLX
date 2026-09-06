/**
 * Dictation vocabulary and history.
 *
 * Both are localStorage-backed and deliberately separate from the app store:
 * neither is part of a conversation, and the store's persist path serialises
 * the whole conversation set on every write.
 *
 * The vocabulary's cap is the design constraint, not a nicety — accuracy
 * measurably falls off past ~20 hint terms, so terms beyond the limit are
 * PARKED rather than dropped, and the parked ones are still there to swap in.
 */

const VOCABULARY_KEY = 'rapid-mlx-web.dictation.vocabulary';
const HISTORY_KEY = 'rapid-mlx-web.dictation.history';

/** Past this many hints, accuracy drops rather than improves. */
export const ACTIVE_LIMIT = 20;

/** Enough to correct a run of mistakes; not enough to become a transcript store. */
export const HISTORY_LIMIT = 30;

export interface Term {
  text: string;
  active: boolean;
}

export interface HistoryEntry {
  id: string;
  text: string;
  at: number;
  /** How long the take ran, in ms. */
  durationMs: number;
  /** How long the engine took to answer, in ms. */
  latencyMs: number;
}

function read<T>(key: string, parse: (raw: unknown) => T | null, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    return parse(JSON.parse(raw)) ?? fallback;
  } catch {
    // Safari private browsing throws on ACCESS, not only on write, and a
    // corrupt value must not take the panel down with it.
    return fallback;
  }
}

function write(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Losing a vocabulary is survivable; failing the dictation is not.
  }
}

export function loadVocabulary(): Term[] {
  return read(
    VOCABULARY_KEY,
    (raw) =>
      Array.isArray(raw)
        ? raw
            .filter(
              (item): item is Term =>
                typeof item === 'object' &&
                item !== null &&
                typeof (item as Term).text === 'string' &&
                (item as Term).text.trim() !== '',
            )
            .map((item) => ({ text: item.text, active: item.active !== false }))
        : null,
    [],
  );
}

export function saveVocabulary(terms: Term[]): void {
  write(VOCABULARY_KEY, terms);
}

/**
 * Add a term.
 *
 * Appended, and active only while there is budget: silently activating the
 * 21st would degrade every subsequent dictation to make room for one name.
 * Case-insensitively deduplicated — "Rapid" and "rapid" are one hint.
 */
export function addTerm(terms: Term[], text: string): Term[] {
  const trimmed = text.trim();
  if (trimmed === '') return terms;
  if (terms.some((term) => term.text.toLowerCase() === trimmed.toLowerCase())) return terms;
  return [...terms, { text: trimmed, active: activeCount(terms) < ACTIVE_LIMIT }];
}

export function removeTerm(terms: Term[], text: string): Term[] {
  return terms.filter((term) => term.text !== text);
}

/** Toggle a term. Activating past the cap is refused rather than silently
 *  parking someone else's term — the user should choose which one goes. */
export function setTermActive(terms: Term[], text: string, active: boolean): Term[] {
  if (active && activeCount(terms) >= ACTIVE_LIMIT) return terms;
  return terms.map((term) => (term.text === text ? { ...term, active } : term));
}

export function activeCount(terms: Term[]): number {
  return terms.filter((term) => term.active).length;
}

/** The hint string sent with a transcription, or null when there is nothing
 *  to bias toward — an empty `context` is a field the engine need not parse. */
export function vocabularyContext(terms: Term[]): string | null {
  const active = terms.filter((term) => term.active).map((term) => term.text);
  return active.length === 0 ? null : active.join(', ');
}

export function loadHistory(): HistoryEntry[] {
  return read(
    HISTORY_KEY,
    (raw) =>
      Array.isArray(raw)
        ? raw
            .filter(
              (item): item is HistoryEntry =>
                typeof item === 'object' &&
                item !== null &&
                typeof (item as HistoryEntry).id === 'string' &&
                typeof (item as HistoryEntry).text === 'string',
            )
            .slice(0, HISTORY_LIMIT)
        : null,
    [],
  );
}

export function saveHistory(entries: HistoryEntry[]): void {
  write(HISTORY_KEY, entries);
}

/** Newest first, capped. The cap is applied on write so the stored value can
 *  never grow past it, whatever an older build left behind. */
export function addHistory(entries: HistoryEntry[], entry: HistoryEntry): HistoryEntry[] {
  return [entry, ...entries].slice(0, HISTORY_LIMIT);
}
