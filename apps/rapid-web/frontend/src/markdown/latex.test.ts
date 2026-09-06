import { describe, expect, it } from 'vitest';
import { hasMath, segmentLaTeX, type LaTeXSegment } from './latex';

/** Rebuild the input from its segments, re-wrapping math in `$`/`$$`.
 *  Delimiter style is deliberately not preserved, so a bracket-form input
 *  round-trips to the dollar form — that normalisation is the only licensed
 *  difference. */
function rebuild(segments: LaTeXSegment[]): string {
  return segments
    .map((segment) =>
      segment.kind === 'markdown'
        ? segment.text
        : segment.display
          ? `$$${segment.latex}$$`
          : `$${segment.latex}$`,
    )
    .join('');
}

const math = (segments: LaTeXSegment[]) => segments.filter((s) => s.kind === 'math');

describe('the four delimiter forms', () => {
  it('reads $$ … $$ as display math', () => {
    expect(math(segmentLaTeX('before $$x^2$$ after'))).toEqual([
      { kind: 'math', latex: 'x^2', display: true },
    ]);
  });

  it('reads $ … $ as inline math', () => {
    expect(math(segmentLaTeX('before $x^2$ after'))).toEqual([
      { kind: 'math', latex: 'x^2', display: false },
    ]);
  });

  it('reads \\[ … \\] as display math', () => {
    // Not an optional extra: a dogfood run had a model emit ONLY bracket
    // delimiters for a plain word problem, and CommonMark's escape rule then
    // strips them, leaving the LaTeX visible as source.
    expect(math(segmentLaTeX('So: \\[ 0.85P = 47 \\] done'))).toEqual([
      { kind: 'math', latex: '0.85P = 47', display: true },
    ]);
  });

  it('reads \\( … \\) as inline math', () => {
    expect(math(segmentLaTeX('where \\( P = 55 \\) exactly'))).toEqual([
      { kind: 'math', latex: 'P = 55', display: false },
    ]);
  });

  it('does not mistake $$ for an empty inline run', () => {
    // If `$` were tested before `$$`, every display opener would parse as an
    // empty inline run and the body would leak into the markdown.
    expect(math(segmentLaTeX('$$a$$'))).toEqual([{ kind: 'math', latex: 'a', display: true }]);
  });

  it('lets display math span lines', () => {
    expect(math(segmentLaTeX('$$\n\\int_0^1 x\\,dx\n$$'))).toEqual([
      { kind: 'math', latex: '\\int_0^1 x\\,dx', display: true },
    ]);
  });

  it('finds several runs in one body', () => {
    expect(math(segmentLaTeX('$a$ and $b$ and \\[c\\]')).map((s) => s.latex)).toEqual([
      'a',
      'b',
      'c',
    ]);
  });
});

describe('anti-cases', () => {
  it('leaves an escaped dollar literal', () => {
    const text = 'costs \\$5 and \\$6';
    expect(math(segmentLaTeX(text))).toEqual([]);
    expect(rebuild(segmentLaTeX(text))).toBe(text);
  });

  it('does NOT open math on \\\\( — an escaped backslash then a paren', () => {
    // The distinction from `\(` is one backslash. Getting it wrong one way
    // renders real math as source; the other way, literal text becomes math.
    const text = 'a \\\\(b\\\\) c';
    expect(math(segmentLaTeX(text))).toEqual([]);
  });

  it('does NOT open math on \\\\[', () => {
    expect(math(segmentLaTeX('a \\\\[b\\\\] c'))).toEqual([]);
  });

  it('leaves an unclosed \\( opener literal', () => {
    // `Use \( to group` never closes. Swallowing the rest of the reply into a
    // formula is the worst possible reading.
    const text = 'Use \\( to group things';
    expect(math(segmentLaTeX(text))).toEqual([]);
    expect(rebuild(segmentLaTeX(text))).toBe(text);
  });

  it('leaves an unclosed $$ literal', () => {
    // This is the STREAMING case, on every display formula, for as long as
    // the closing delimiter has not arrived yet.
    const text = 'here it comes $$\\int_0^1';
    expect(math(segmentLaTeX(text))).toEqual([]);
    expect(rebuild(segmentLaTeX(text))).toBe(text);
  });

  it('still finds a later complete run after an unclosed opener', () => {
    expect(math(segmentLaTeX('\\( unclosed and then $x$')).map((s) => s.latex)).toEqual(['x']);
  });

  it('leaves a bare dollar in prose alone', () => {
    const text = 'it costs $20 today';
    expect(math(segmentLaTeX(text))).toEqual([]);
    expect(rebuild(segmentLaTeX(text))).toBe(text);
  });

  it('leaves a currency PAIR alone', () => {
    // Two dollars on one line, so the single-dollar guard cannot catch it.
    // Rejecting `$` followed by a digit is what does, mirroring MathJax.
    const text = 'between $20 to $30 per unit';
    expect(math(segmentLaTeX(text))).toEqual([]);
    expect(rebuild(segmentLaTeX(text))).toBe(text);
  });

  it.each(['$0', '$5', '$9'])('rejects %s as an opener', (opener) => {
    expect(math(segmentLaTeX(`${opener}x${opener}`))).toEqual([]);
  });

  it('still accepts a real opener that starts with a control sequence', () => {
    expect(math(segmentLaTeX('$\\frac{47}{0.85}$'))).toEqual([
      { kind: 'math', latex: '\\frac{47}{0.85}', display: false },
    ]);
  });

  it('does not let an inline run cross a newline', () => {
    // Without the single-line rule one stray dollar turns everything after it
    // into a single enormous formula.
    const text = 'a $ dollar here\nand $ another there';
    expect(math(segmentLaTeX(text))).toEqual([]);
    expect(rebuild(segmentLaTeX(text))).toBe(text);
  });

  it('treats empty delimiters as not-math', () => {
    expect(math(segmentLaTeX('an empty $$ $$ pair'))).toEqual([]);
  });
});

describe('anti-cases that are structural, not defended', () => {
  // These three are unreachable by construction: segmentation runs AFTER
  // marked.lexer(), on the inline content of text-bearing tokens only, so
  // fenced code, indented code and inline code are `code`/`codespan` tokens
  // the segmenter never receives. They are asserted anyway, so a future
  // refactor that moves segmentation ahead of lexing fails loudly here rather
  // than silently turning shell scripts into equations.

  it('never receives fenced code — asserted at the pipeline level', async () => {
    const { parseMarkdown } = await import('./lex');
    const tokens = parseMarkdown('```bash\necho "$x$"\n```');
    expect(tokens[0]?.type).toBe('code');
  });

  it('never receives indented code', async () => {
    const { parseMarkdown } = await import('./lex');
    const tokens = parseMarkdown('    echo "$x$"');
    expect(tokens[0]?.type).toBe('code');
  });

  it('never receives inline code', async () => {
    const { parseMarkdown } = await import('./lex');
    const tokens = parseMarkdown('the price is `$5.00` exactly');
    const paragraph = tokens[0] as { tokens?: Array<{ type: string }> };
    expect(paragraph.tokens?.some((t) => t.type === 'codespan')).toBe(true);
  });
});

describe('the round-trip property', () => {
  // The single most valuable property here: concatenating the markdown bodies
  // and re-wrapping the math bodies reconstructs the input. Any segment that
  // silently drops or duplicates text fails this, whatever else it gets right.
  it.each([
    'plain prose with no math at all',
    'inline $x$ math',
    'display $$x$$ math',
    'both $a$ and $$b$$ together',
    'leading $a$',
    '$a$ trailing',
    '$a$',
    'costs \\$5',
    'it costs $20 today',
    'unclosed \\( opener',
    '',
    'multi\nline\nprose',
  ])('round-trips %o', (input) => {
    expect(rebuild(segmentLaTeX(input))).toBe(input);
  });

  it('round-trips bracket forms up to delimiter normalisation', () => {
    // The one licensed difference: `\(x\)` and `$x$` mean the same thing, and
    // the choice carries nothing downstream.
    expect(rebuild(segmentLaTeX('a \\(x\\) b'))).toBe('a $x$ b');
    expect(rebuild(segmentLaTeX('a \\[x\\] b'))).toBe('a $$x$$ b');
  });

  it('preserves surrounding whitespace exactly', () => {
    expect(rebuild(segmentLaTeX('  spaced  $x$  out  '))).toBe('  spaced  $x$  out  ');
  });
});

describe('hasMath', () => {
  it('is false for prose', () => {
    expect(hasMath('no math here, and $20 is currency')).toBe(false);
  });

  it('is true when a run closes', () => {
    expect(hasMath('an $x$ appears')).toBe(true);
  });

  it('is false for an unclosed opener', () => {
    expect(hasMath('still waiting for $$')).toBe(false);
  });
});
