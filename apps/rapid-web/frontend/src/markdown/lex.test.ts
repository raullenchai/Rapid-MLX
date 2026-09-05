import { describe, expect, it } from 'vitest';
import { advance, canFreeze, emptyLexState, parseMarkdown, tokensOf } from './lex';

/** Feed `source` one character at a time, as a stream would. */
function stream(source: string) {
  let state = emptyLexState();
  for (let index = 1; index <= source.length; index += 1) {
    state = advance(state, source.slice(0, index));
  }
  return state;
}

const shapes = (source: string) => parseMarkdown(source).map((token) => token.type);

describe('parseMarkdown', () => {
  it('lexes the block constructs the renderer supports', () => {
    expect(shapes('# heading')).toEqual(['heading']);
    expect(shapes('- one\n- two')).toEqual(['list']);
    expect(shapes('> quoted')).toEqual(['blockquote']);
    expect(shapes('```js\ncode\n```')).toEqual(['code']);
    expect(shapes('| a | b |\n| - | - |\n| 1 | 2 |')).toEqual(['table']);
    expect(shapes('---')).toEqual(['hr']);
  });

  it('does not treat a single newline as a hard break', () => {
    // Models emit wrapped prose; `breaks: true` would render every wrap point
    // as a visible line break.
    const paragraph = parseMarkdown('one\ntwo')[0] as {
      tokens?: Array<{ type: string }>;
    };
    expect(paragraph.tokens?.some((token) => token.type === 'br')).toBe(false);
  });

  it('gives every token its exact source in `raw`', () => {
    // The frozen-block arithmetic depends entirely on this.
    const source = '# heading\n\nparagraph\n\n```\ncode\n```\n';
    const rebuilt = parseMarkdown(source)
      .map((token) => token.raw)
      .join('');
    expect(rebuilt).toBe(source);
  });
});

describe('canFreeze vetoes', () => {
  it('refuses while a code fence is open', () => {
    // The generalisation of the old page's special case: without it, every
    // streaming reply containing code flashes literal backticks until its
    // closing fence arrives.
    const source = 'text\n\n```js\nconst a = 1;';
    expect(canFreeze(source, [...parseMarkdown(source)])).toBe(false);
  });

  it('allows once the fence closes', () => {
    const source = 'text\n\n```js\nconst a = 1;\n```\n\nmore\n\n';
    expect(canFreeze(source, [...parseMarkdown(source)])).toBe(true);
  });

  it('refuses while $$ is open', () => {
    const source = 'text\n\n$$\\int_0^1';
    expect(canFreeze(source, [...parseMarkdown(source)])).toBe(false);
  });

  it('refuses while \\[ is unmatched', () => {
    const source = 'text\n\n\\[ 0.85P = 47';
    expect(canFreeze(source, [...parseMarkdown(source)])).toBe(false);
  });

  it('refuses to freeze a trailing paragraph', () => {
    // `Title\n=====` turns a settled paragraph into an <h1> retroactively —
    // a setext heading is only recognisable from the line AFTER it.
    const source = '# done\n\nTitle';
    expect(canFreeze(source, [...parseMarkdown(source)])).toBe(false);
  });

  it('refuses to freeze a trailing table', () => {
    // The alignment row arrives after the header row.
    const source = '# done\n\n| a | b |';
    expect(canFreeze(source, [...parseMarkdown(source)])).toBe(false);
  });

  it('refuses to freeze a trailing list', () => {
    // A blank line before the next item turns a tight list loose, re-wrapping
    // every item in a <p>.
    const source = '# done\n\n- one\n- two';
    expect(canFreeze(source, [...parseMarkdown(source)])).toBe(false);
  });

  it('refuses on an empty tail', () => {
    expect(canFreeze('', [])).toBe(false);
  });
});

describe('advance', () => {
  it('produces the same tokens as a whole-buffer lex, fed one char at a time', () => {
    // The core correctness property. If incremental lexing ever disagrees
    // with a cold parse, the reader sees one thing mid-stream and a different
    // thing the moment the answer settles.
    const source = [
      '# Heading',
      '',
      'A paragraph with **bold** and `code`.',
      '',
      '- one',
      '- two',
      '',
      '```python',
      'print("hi")',
      '```',
      '',
      '| a | b |',
      '| - | - |',
      '| 1 | 2 |',
      '',
      'Trailing prose.',
      '',
    ].join('\n');

    const streamed = tokensOf(stream(source)).map((token) => token.raw);
    const cold = parseMarkdown(source).map((token) => token.raw);
    expect(streamed.join('')).toBe(cold.join(''));
    expect(streamed).toEqual(cold);
  });

  it('keeps frozen tokens referentially identical across flushes', () => {
    // This is what lets React.memo skip everything above the tail. Without
    // it the whole transcript reconciles on every commit and the incremental
    // lex buys nothing.
    const source = '# One\n\nSettled paragraph.\n\n';
    let state = advance(emptyLexState(), source);
    const firstFrozen = state.frozen[0];
    expect(firstFrozen).toBeDefined();

    state = advance(state, `${source}Still being written`);
    state = advance(state, `${source}Still being written more`);

    expect(state.frozen[0]).toBe(firstFrozen);
  });

  it('never freezes past the end of the buffer', () => {
    const source = '# One\n\n# Two\n\n# Three\n\n';
    const state = advance(emptyLexState(), source);
    expect(state.frozenChars).toBeLessThanOrEqual(source.length);
    // frozenChars must account for exactly the frozen tokens, or the next
    // slice starts in the wrong place and duplicates or drops text.
    const accounted = state.frozen.reduce((total, token) => total + token.raw.length, 0);
    expect(accounted).toBe(state.frozenChars);
  });

  it('never freezes any part of an open fence, however long the buffer grows', () => {
    // A paragraph BEFORE the fence may legitimately freeze — a blank line
    // terminated it and no later text can reinterpret it. What must not
    // happen is the fence's own content freezing, because everything inside
    // it is still one unterminated block.
    const state = stream('intro\n\n```js\nline one\nline two\nline three');
    const frozenSource = state.frozen.map((token) => token.raw).join('');
    expect(frozenSource).not.toContain('```');
    expect(frozenSource).not.toContain('line one');
    // The fence and everything in it is still hot, so it re-lexes each flush
    // and reads as code rather than as literal backticks.
    expect(state.tail.some((token) => token.type === 'code')).toBe(true);
  });

  it('renders an unterminated fence as code, not as literal backticks', () => {
    // Caught in a screenshot rather than by an assertion, the first time.
    const tokens = tokensOf(stream('```python\nprint("partial"'));
    expect(tokens[0]?.type).toBe('code');
  });

  it('does freeze once enough settled blocks accumulate', () => {
    const state = stream('# One\n\n# Two\n\n# Three\n\nstill writing');
    expect(state.frozen.length).toBeGreaterThan(0);
  });

  it('handles an empty buffer', () => {
    const state = advance(emptyLexState(), '');
    expect(tokensOf(state)).toEqual([]);
  });

  it('reconstructs the source exactly from all tokens', () => {
    const source = '# H\n\ntext\n\n- a\n- b\n\nmore text\n';
    expect(
      tokensOf(stream(source))
        .map((token) => token.raw)
        .join(''),
    ).toBe(source);
  });
});
