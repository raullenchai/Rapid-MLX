import { Lexer, type Token, type TokensList } from 'marked';

/**
 * Block-level markdown lexing, with incremental support for streaming.
 *
 * `marked`'s LEXER, not its HTML renderer, for two reasons:
 *
 * 1. Every token carries `raw`, the exact source that produced it — without a
 *    per-token source length the frozen-block arithmetic below is impossible.
 * 2. We walk tokens and emit React elements, so there is no HTML string and
 *    no sanitizer anywhere in the pipeline; React escapes text by construction.
 *
 * `react-markdown` + `remark-gfm` was rejected: ~250 KB for the same features
 * against marked's ~43 KB, and its AST exposes no per-block source offsets.
 */

const LEXER_OPTIONS = {
  gfm: true,
  // A single newline is NOT a hard break. Models emit wrapped prose and
  // `breaks: true` would render every wrap point as a visible line break.
  breaks: false,
} as const;

export function parseMarkdown(source: string): TokensList {
  return new Lexer(LEXER_OPTIONS).lex(source);
}

/**
 * Incremental lexing state for one streaming message.
 *
 * `frozen` holds blocks that can never change again; `frozenChars` is how much
 * of the buffer they account for. Each flush re-lexes only
 * `buffer.slice(frozenChars)`, so cost is proportional to the current block
 * rather than the whole answer. This is what removes the O(n²).
 */
export interface LexState {
  /** Referentially stable across flushes, so `React.memo` can skip them. */
  frozen: Token[];
  frozenChars: number;
  /** Re-lexed every flush. */
  tail: Token[];
}

export function emptyLexState(): LexState {
  return { frozen: [], frozenChars: 0, tail: [] };
}

/**
 * May the tokens before the last one in `tail` be frozen?
 *
 * A block that LOOKS complete can still be rewritten by text that has not
 * arrived, and freezing one that later changes makes the transcript flicker
 * between two renderings.
 *
 * The vetoes, each a construct that mutates retroactively:
 *
 *   * An open code fence — otherwise every streaming reply containing code
 *     flashes literal backticks until its closing fence arrives.
 *   * An open `$$` or `\[`. Same problem, for math.
 *   * A trailing paragraph: `Title\n=====` turns a settled paragraph into an
 *     `<h1>`, since a setext heading is only visible from the NEXT line.
 *   * A trailing table: its alignment row arrives after the header row, so a
 *     header alone lexes as a paragraph and is re-read one line later.
 *   * A trailing list: a blank line before the next item turns a tight list
 *     loose, re-wrapping every item in a `<p>`.
 */
export function canFreeze(tailSource: string, tail: Token[]): boolean {
  // An odd number of fences means one is still open.
  const fences = tailSource.match(/^ {0,3}(?:```|~~~)/gm);
  if (fences && fences.length % 2 === 1) return false;

  const displayOpeners = (tailSource.match(/(?<!\\)\$\$/g) ?? []).length;
  if (displayOpeners % 2 === 1) return false;
  const bracketOpen = (tailSource.match(/(?<!\\)\\\[/g) ?? []).length;
  const bracketClose = (tailSource.match(/(?<!\\)\\\]/g) ?? []).length;
  if (bracketOpen !== bracketClose) return false;

  const last = tail[tail.length - 1];
  if (!last) return false;
  if (last.type === 'paragraph' || last.type === 'table' || last.type === 'list') return false;

  return true;
}

/**
 * Advance the lex state to cover `buffer`.
 *
 * Returns a NEW state; `frozen` keeps the identity of every token that was
 * already frozen, which is what lets a memoised block component skip
 * reconciliation for everything above the tail.
 */
export function advance(state: LexState, buffer: string): LexState {
  const tailSource = buffer.slice(state.frozenChars);
  const tail = [...parseMarkdown(tailSource)];

  // Fewer than two tokens means there is nothing that could be complete: the
  // only token present is still being written.
  if (tail.length < 2 || !canFreeze(tailSource, tail)) {
    return { frozen: state.frozen, frozenChars: state.frozenChars, tail };
  }

  // Every token except the last is complete by construction — a new block
  // started after it, which can only happen once the previous one ended.
  const newlyFrozen = tail.slice(0, -1);
  const consumed = newlyFrozen.reduce((total, token) => total + token.raw.length, 0);

  return {
    frozen: [...state.frozen, ...newlyFrozen],
    frozenChars: state.frozenChars + consumed,
    tail: tail.slice(-1),
  };
}

/** Everything to render right now. */
export function tokensOf(state: LexState): Token[] {
  return [...state.frozen, ...state.tail];
}
