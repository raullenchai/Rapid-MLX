/**
 * Split assistant text into alternating markdown and LaTeX runs.
 *
 * Ported from `apps/rapid-mac/Sources/Rapid/UI/Markdown/LaTeXSegmenter.swift`.
 *
 * DELIMITERS — the KaTeX/MathJax defaults every model in the wild emits:
 *
 *   $$ … $$   display math, may span lines
 *   $  …  $   inline math, SINGLE LINE ONLY, so a stray dollar in prose
 *             cannot swallow the rest of the reply into "math"
 *   \[ … \]   display math, bracket form
 *   \( … \)   inline math, bracket form
 *
 * The bracket forms are NOT optional extras: instruction-tuned models emit
 * them exclusively for plain word problems, and CommonMark's escape rule then
 * eats them. `\(`, `\)`, `\[` and `\]` all wrap ASCII punctuation so they
 * collapse to bare parens, while `\frac` and `\times` survive verbatim —
 * leaving the reader `( P = \frac{47}{0.85} \approx 55.29 )`.
 *
 * Delimiter STYLE is not preserved: `\(x\)` and `$x$` both produce
 * `{ latex: 'x', display: false }`.
 *
 * ANTI-CASES. Math inside fenced, indented or inline code is STRUCTURALLY
 * unreachable — this runs after `marked.lexer()` on text-bearing tokens only,
 * and those are `code`/`codespan` tokens. The rest are implemented here:
 *
 *   \$          an escaped dollar stays literal and never opens a run
 *   \\( and \\[ an escaped backslash plus a literal bracket does NOT open math
 *   unclosed    `Use \( to group` keeps the opener literal rather than
 *               swallowing the reply. Also the streaming case, mid-answer.
 *   bare $      `it costs $20 today` — one dollar, no closer
 *   currency    `$20 to $30` has two dollars on one line, so the bare-dollar
 *               guard misses it. Any `$` followed by a DIGIT is rejected,
 *               mirroring MathJax: a real opener starts with a control
 *               sequence, a variable letter or a bracket.
 */

export type LaTeXSegment =
  { kind: 'markdown'; text: string } | { kind: 'math'; latex: string; display: boolean };

interface Opener {
  /** The literal that opens the run. */
  open: string;
  /** The literal that closes it. */
  close: string;
  display: boolean;
  /** Inline runs die at a newline; display runs may span lines. */
  singleLine: boolean;
}

/**
 * Order matters: `$$` must be tested before `$`, or every display opener is
 * mis-read as an empty inline run.
 */
const OPENERS: Opener[] = [
  { open: '$$', close: '$$', display: true, singleLine: false },
  { open: '\\[', close: '\\]', display: true, singleLine: false },
  { open: '\\(', close: '\\)', display: false, singleLine: true },
  { open: '$', close: '$', display: false, singleLine: true },
];

/**
 * Is the character at `index` escaped by an odd number of backslashes?
 *
 * Counting the run distinguishes `\$` (literal) from `\\$` (escaped
 * backslash, then a real opener). Wrong either way is visible: a literal
 * dollar opens math, or real math renders as source.
 */
function isEscaped(text: string, index: number): boolean {
  let backslashes = 0;
  for (let cursor = index - 1; cursor >= 0 && text[cursor] === '\\'; cursor -= 1) backslashes += 1;
  return backslashes % 2 === 1;
}

/**
 * Would a bare `$` here be currency rather than a math opener?
 *
 * `$20 to $30` is the case the single-dollar guard cannot catch, because
 * there genuinely are two dollars on the line.
 */
function looksLikeCurrency(text: string, index: number): boolean {
  const next = text[index + 1];
  return next !== undefined && next >= '0' && next <= '9';
}

/**
 * Skip a backtick-delimited inline code run starting at `index`.
 *
 * Returns the index just past the closing run, or -1 if this is not a code
 * span. Per CommonMark a run of N backticks closes on the next run of exactly
 * N, so the length must be counted.
 *
 * This anti-case is NOT structural, contrary to first expectations.
 * Segmentation has to happen on the block's RAW source, because marked's
 * inline lexer shreds `\[` and `\,` into `escape` tokens and destroys the
 * formula before it can be recognised. Running before inline lexing puts
 * inline code back in scope. Fenced and indented code stay structural.
 */
function skipCodeSpan(text: string, index: number): number {
  let openLength = 0;
  while (text[index + openLength] === '`') openLength += 1;
  if (openLength === 0) return -1;

  let cursor = index + openLength;
  while (cursor < text.length) {
    if (text[cursor] !== '`') {
      cursor += 1;
      continue;
    }
    let closeLength = 0;
    while (text[cursor + closeLength] === '`') closeLength += 1;
    if (closeLength === openLength) return cursor + closeLength;
    cursor += closeLength;
  }

  // Unclosed. Not a code span at all, so scanning resumes normally.
  return -1;
}

function findOpener(text: string, from: number): { at: number; opener: Opener } | null {
  let cursor = from;
  while (cursor < text.length) {
    const char = text[cursor];

    if (char === '`') {
      const past = skipCodeSpan(text, cursor);
      if (past !== -1) {
        cursor = past;
        continue;
      }
    }

    if (char !== '$' && char !== '\\') {
      cursor += 1;
      continue;
    }

    for (const opener of OPENERS) {
      if (!text.startsWith(opener.open, cursor)) continue;
      // `\[` and `\(` ARE backslash sequences, so "escaped" for them means a
      // preceding backslash — `\\[` is an escaped backslash plus a literal
      // bracket. `isEscaped` counts the run before the opener's first
      // character and answers both cases correctly.
      if (isEscaped(text, cursor)) continue;
      if (opener.open === '$' && looksLikeCurrency(text, cursor)) continue;
      return { at: cursor, opener };
    }

    cursor += 1;
  }
  return null;
}

/** The index of the closing delimiter, or -1 if the run never closes. */
function findClose(text: string, from: number, opener: Opener): number {
  for (let cursor = from; cursor < text.length; cursor += 1) {
    const char = text[cursor];

    // An inline run dies at a newline. Without this a single stray dollar
    // turns everything after it into one enormous formula.
    if (opener.singleLine && char === '\n') return -1;

    if (char !== '$' && char !== '\\') continue;
    if (!text.startsWith(opener.close, cursor)) continue;
    if (isEscaped(text, cursor)) continue;
    return cursor;
  }
  return -1;
}

/**
 * Split `input` into alternating segments.
 *
 * Concatenating the markdown bodies and re-wrapping the math bodies in their
 * delimiters reconstructs the input, up to the bracket-to-dollar
 * normalisation. That round-trip is the single most valuable property in this
 * module and is tested directly.
 */
export function segmentLaTeX(input: string): LaTeXSegment[] {
  if (input === '') return [];

  const segments: LaTeXSegment[] = [];
  let plainStart = 0;
  let cursor = 0;

  while (cursor < input.length) {
    const found = findOpener(input, cursor);
    if (!found) break;

    const bodyStart = found.at + found.opener.open.length;
    const closeAt = findClose(input, bodyStart, found.opener);

    if (closeAt === -1) {
      // Unclosed. The opener stays literal — this is the streaming case, on
      // every formula, for as long as the closer has not arrived. Resume
      // scanning after it so a later, complete run in the same text is still
      // found.
      cursor = bodyStart;
      continue;
    }

    const latex = input.slice(bodyStart, closeAt).trim();
    if (latex === '') {
      // `$$` with nothing between it is not math. Skip past the whole thing
      // rather than treating the closer as a fresh opener.
      cursor = closeAt + found.opener.close.length;
      continue;
    }

    if (found.at > plainStart) {
      segments.push({
        kind: 'markdown',
        text: input.slice(plainStart, found.at),
      });
    }
    segments.push({ kind: 'math', latex, display: found.opener.display });

    cursor = closeAt + found.opener.close.length;
    plainStart = cursor;
  }

  if (plainStart < input.length) {
    segments.push({ kind: 'markdown', text: input.slice(plainStart) });
  }

  return segments;
}

/** True when `input` contains no math at all — the common case, worth a
 *  cheap answer so the caller can skip building segments entirely. */
export function hasMath(input: string): boolean {
  return segmentLaTeX(input).some((segment) => segment.kind === 'math');
}
