import { advance, emptyLexState, type LexState } from '@/markdown/lex';

/**
 * The live text of the answer currently streaming.
 *
 * The hot path, and the reason the rewrite exists: the old page re-parsed the
 * entire accumulated buffer into `innerHTML` once per animation frame — O(n²)
 * over a turn, destroying every open `<details>`, the selection and each
 * `<pre>`'s scroll position sixty times a second.
 *
 * Four things keep it cheap:
 *
 * 1. The accumulator is a plain field, NOT React state, so the SSE reader
 *    does zero React work per token.
 * 2. Commits coalesce on a timer, not `requestAnimationFrame` — rAF fires 6-12x
 *    more often than needed, and iOS Safari throttles it during momentum
 *    scrolling and pauses it in background tabs.
 * 3. Lexing is incremental, so a commit costs the current block rather than
 *    the whole buffer (see markdown/lex.ts).
 * 4. Frozen tokens keep their identity, so memoised blocks skip reconciliation.
 */

/**
 * The coalescing interval.
 *
 * The Mac app measured a full markdown compile of a 24 000-character buffer at
 * 15 ms and settled on 100 ms (StreamingMarkdownStore.swift:42-47). 90 ms is
 * inside the same perceptual beat — block structure appearing within a tenth
 * of a second is not perceived as lag — while doing roughly a tenth of the
 * work a frame-rate commit would.
 *
 * Change this with a measurement, not by feel.
 */
export const COMMIT_INTERVAL_MS = 90;

/** A commit slower than this backs the interval off for the rest of the turn. */
const SLOW_COMMIT_MS = 40;
const BACKOFF_INTERVAL_MS = 180;

export interface StreamSnapshot {
  content: string;
  reasoning: string;
  lex: LexState;
}

const EMPTY: StreamSnapshot = {
  content: '',
  reasoning: '',
  lex: emptyLexState(),
};

type Listener = () => void;

export class StreamingStore {
  #content = '';
  #reasoning = '';
  #lex = emptyLexState();
  #snapshot: StreamSnapshot = EMPTY;
  #listeners = new Set<Listener>();
  #timer: ReturnType<typeof setTimeout> | undefined;
  #interval = COMMIT_INTERVAL_MS;
  #now: () => number;

  /** `now` is injected so tests can drive it without a real clock. */
  constructor(now: () => number = () => performance.now()) {
    this.#now = now;
  }

  subscribe = (listener: Listener): (() => void) => {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  };

  /**
   * The current committed snapshot.
   *
   * Referentially stable between commits — `useSyncExternalStore` compares by
   * identity and would loop forever on a fresh object each call.
   */
  getSnapshot = (): StreamSnapshot => this.#snapshot;

  /** Begin a turn. Clears everything the previous one left behind. */
  start(): void {
    this.#cancelTimer();
    this.#content = '';
    this.#reasoning = '';
    this.#lex = emptyLexState();
    this.#interval = COMMIT_INTERVAL_MS;
    this.#snapshot = EMPTY;
    this.#notify();
  }

  appendContent(text: string): void {
    this.#content += text;
    this.#schedule();
  }

  appendReasoning(text: string): void {
    this.#reasoning += text;
    this.#schedule();
  }

  /**
   * Commit immediately.
   *
   * Called at stream end: the last few tokens must not sit waiting on a timer
   * while the user looks at an answer that appears to have stopped short.
   */
  flush(): void {
    this.#cancelTimer();
    this.#commit();
  }

  /** The raw text, for the store commit at stream end. */
  current(): { content: string; reasoning: string } {
    return { content: this.#content, reasoning: this.#reasoning };
  }

  #schedule(): void {
    if (this.#timer !== undefined) return;
    this.#timer = setTimeout(() => {
      this.#timer = undefined;
      this.#commit();
    }, this.#interval);
  }

  #cancelTimer(): void {
    if (this.#timer === undefined) return;
    clearTimeout(this.#timer);
    this.#timer = undefined;
  }

  #commit(): void {
    const started = this.#now();
    this.#lex = advance(this.#lex, this.#content);
    this.#snapshot = {
      content: this.#content,
      reasoning: this.#reasoning,
      lex: this.#lex,
    };

    // A fast model on a fast Mac can burst hard enough that a commit costs
    // more than the interval, at which point commits queue behind each other
    // and the tab stops responding to touch.
    if (this.#now() - started > SLOW_COMMIT_MS) this.#interval = BACKOFF_INTERVAL_MS;

    this.#notify();
  }

  #notify(): void {
    for (const listener of this.#listeners) listener();
  }
}

/** One store per page: only one turn streams at a time. */
export const streamingStore = new StreamingStore();
