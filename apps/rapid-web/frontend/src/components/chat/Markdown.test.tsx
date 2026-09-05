import { cleanup, render, within } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { Markdown } from './Markdown';
import { parseMarkdown } from '@/markdown/lex';

// Renders accumulate in the same document otherwise, so a `screen` query that
// should find one control finds one per test that has run so far.
afterEach(cleanup);

function draw(source: string, options: { streaming?: boolean; math?: 'mathml' | 'source' } = {}) {
  const result = render(
    <Markdown
      tokens={[...parseMarkdown(source)]}
      streaming={options.streaming ?? false}
      mathRendering={options.math ?? 'mathml'}
    />,
  );
  return result.container;
}

/**
 * Find a MathML root.
 *
 * `querySelector('math')` does NOT match: Temml's output is injected as a
 * string, so jsdom parses it into the HTML namespace where `math` is an
 * unknown element. Comparing tag names sidesteps the namespace.
 */
function mathRoot(container: HTMLElement): Element | null {
  return (
    Array.from(container.querySelectorAll('*')).find(
      (element) => element.tagName.toLowerCase() === 'math',
    ) ?? null
  );
}

describe('block rendering', () => {
  it('renders the constructs the old page supported', () => {
    const container = draw('# Title\n\nSome **bold** prose.\n\n- one\n- two\n\n> quoted\n\n---');
    expect(container.querySelector('h1')).toHaveTextContent('Title');
    expect(container.querySelector('strong')).toHaveTextContent('bold');
    expect(container.querySelectorAll('li')).toHaveLength(2);
    expect(container.querySelector('blockquote')).toHaveTextContent('quoted');
    expect(container.querySelector('hr')).toBeInTheDocument();
  });

  it('renders a GFM table with its alignment', () => {
    const container = draw('| a | b |\n| :- | -: |\n| 1 | 2 |');
    expect(container.querySelectorAll('th')).toHaveLength(2);
    expect(container.querySelectorAll('td')).toHaveLength(2);
    expect(container.querySelector('th')).toHaveStyle({ textAlign: 'left' });
    expect(container.querySelectorAll('th')[1]).toHaveStyle({
      textAlign: 'right',
    });
  });

  it('makes the table scroll region reachable by keyboard', () => {
    // A bare overflow container cannot be scrolled without a pointer.
    const container = draw('| a |\n| - |\n| 1 |');
    const region = container.querySelector('[role="region"]');
    expect(region).toHaveAttribute('tabindex', '0');
    expect(region).toHaveAttribute('aria-label');
  });

  it('renders strikethrough', () => {
    expect(draw('~~gone~~').querySelector('del')).toHaveTextContent('gone');
  });

  it('renders a task list read-only', () => {
    // A checkbox in a transcript reports what the model wrote; clicking it
    // would be editing the answer.
    const box = draw('- [x] done').querySelector('input[type="checkbox"]');
    expect(box).toBeDisabled();
    expect(box).toBeChecked();
  });

  it('preserves an ordered list start', () => {
    expect(draw('5. five\n6. six').querySelector('ol')).toHaveAttribute('start', '5');
  });
});

describe('code blocks', () => {
  it('highlights a known language when settled', () => {
    const container = draw('```python\ndef f(): pass\n```');
    expect(container.querySelector('.hljs-keyword')).toBeInTheDocument();
  });

  it('renders an unknown language as plain text', () => {
    // Never highlightAuto: a wrong guess is worse than no colour, and it runs
    // every registered grammar over the body.
    const container = draw('```notalanguage\nsome text\n```');
    expect(container.querySelector('.hljs-keyword')).not.toBeInTheDocument();
    expect(container.querySelector('code')).toHaveTextContent('some text');
  });

  it('offers a copy button when settled', () => {
    const container = draw('```js\nconst a = 1;\n```');
    expect(within(container).getByRole('button', { name: 'Copy' })).toBeInTheDocument();
  });

  it('skips highlighting and the copy button while streaming', () => {
    // The body changes on every commit, so the work would be thrown away —
    // and a partial line highlights as broken syntax anyway.
    const container = draw('```python\ndef f(): pa', { streaming: true });
    expect(container.querySelector('.hljs-keyword')).not.toBeInTheDocument();
    expect(within(container).queryByRole('button', { name: 'Copy' })).not.toBeInTheDocument();
  });

  it('renders an unterminated fence as code', () => {
    // Every streaming reply containing code flashed literal backticks in the
    // old page until its closing fence arrived.
    const container = draw('```python\nprint("partial"', { streaming: true });
    expect(container.querySelector('pre code')).toHaveTextContent('print("partial"');
  });
});

describe('link safety', () => {
  it('renders an allowed link as an anchor that cannot reach the opener', () => {
    const anchor = draw('[docs](https://example.com/x)').querySelector('a');
    expect(anchor).toHaveAttribute('href', 'https://example.com/x');
    expect(anchor).toHaveAttribute('rel', 'noopener noreferrer');
  });

  it.each([
    ['javascript:', '[click](javascript:alert(1))'],
    ['file:', '[open](file:///etc/passwd)'],
    ['vscode:', '[open](vscode://file/x)'],
  ])('renders a %s link inert, keeping the label', (_scheme, source) => {
    const container = draw(source);
    // No anchor at all — not an anchor with a missing href, which would still
    // be focusable and would still look clickable.
    expect(container.querySelector('a')).not.toBeInTheDocument();
    expect(container.textContent).toContain(source.slice(1, source.indexOf(']')));
  });

  it('never emits an href attribute for a refused scheme', () => {
    const container = draw('[x](javascript:alert(1))');
    expect(container.innerHTML).not.toContain('javascript:');
  });

  it('renders a blocked image as its alt text', () => {
    // The CSP is `img-src 'self' data:`, so a remote image would render as a
    // broken-image glyph — worse than the description the model wrote.
    const container = draw('![a diagram](https://cdn.example/pic.png)');
    expect(container.querySelector('img')).not.toBeInTheDocument();
    expect(container).toHaveTextContent('a diagram');
  });
});

describe('inline HTML', () => {
  // Models routinely write `<b>` and `<br>` in prose, and showing the literal
  // tag reads as a rendering failure. Only the allow-list in markdown/html.ts
  // becomes markup, and only in bare form.

  it.each([
    ['<b>bold</b>', 'b', 'bold'],
    ['<strong>strong</strong>', 'strong', 'strong'],
    ['<i>italic</i>', 'i', 'italic'],
    ['<u>under</u>', 'u', 'under'],
    ['<sub>sub</sub>', 'sub', 'sub'],
    ['<sup>sup</sup>', 'sup', 'sup'],
    ['<mark>mark</mark>', 'mark', 'mark'],
  ])('renders %s as real markup', (source, tag, content) => {
    const container = draw(source);
    const element = container.querySelector(tag);
    expect(element).toBeInTheDocument();
    expect(element).toHaveTextContent(content);
  });

  it('renders <br> as a line break', () => {
    expect(draw('one<br>two').querySelector('br')).toBeInTheDocument();
  });

  it('maps <strike> onto <s>, which React will accept', () => {
    // React refuses the deprecated element name outright.
    const container = draw('<strike>gone</strike>');
    expect(container.querySelector('s')).toHaveTextContent('gone');
  });

  it('renders markdown INSIDE an html tag', () => {
    // `<b>` and `</b>` are separate tokens with the run between them, so the
    // nesting is rebuilt rather than read off the token tree.
    const container = draw('<b>bold with `code`</b>');
    expect(container.querySelector('b')?.querySelector('code')).toHaveTextContent('code');
  });

  it('still shows a script tag as text, beside a rendered one', () => {
    // The exact line from the bug report: the safe tag renders, the two
    // dangerous ones do not.
    const container = draw('<b>HTML粗体</b> <script>alert(1)</script> <img src=x onerror=alert(2)>');

    expect(container.querySelector('b')).toHaveTextContent('HTML粗体');
    expect(container.querySelector('script')).not.toBeInTheDocument();
    expect(container.querySelector('img')).not.toBeInTheDocument();
    expect(container).toHaveTextContent('<script>alert(1)</script>');
    expect(container).toHaveTextContent('<img src=x onerror=alert(2)>');
  });

  it('refuses an allow-listed tag that carries an attribute', () => {
    // No attribute ever reaches an element, so there is nothing to sanitise.
    const container = draw('<b onclick="alert(1)">x</b>');
    expect(container.querySelector('b')).not.toBeInTheDocument();
    expect(container).toHaveTextContent('<b onclick="alert(1)">x');
  });

  it('does not let an unclosed tag swallow the rest of the transcript', () => {
    expect(draw('<b>open').querySelector('b')).toHaveTextContent('open');
  });
});

describe('XSS fixtures', () => {
  // Everything in this pipeline is a React element built from a token, so
  // markup cannot be injected — except through Temml's MathML, which is the
  // app's only dangerouslySetInnerHTML. These fixtures are the guard on that,
  // and a Temml version bump must re-run them.

  it.each([
    '<img src=x onerror=alert(1)>',
    '<script>alert(1)</script>',
    '</script><script>alert(1)</script>',
    '<svg/onload=alert(1)>',
    '<iframe src="javascript:alert(1)"></iframe>',
  ])('renders %o as text, not as markup', (payload) => {
    const container = draw(payload);
    expect(container.querySelector('script')).not.toBeInTheDocument();
    expect(container.querySelector('img')).not.toBeInTheDocument();
    expect(container.querySelector('iframe')).not.toBeInTheDocument();
    expect(container.querySelector('svg')).not.toBeInTheDocument();
  });

  it('does not execute markup smuggled through \\text{}', () => {
    // The interesting case: this reaches Temml, which produces the string we
    // then inject.
    const container = draw('$\\text{<img src=x onerror=alert(1)>}$');
    expect(container.querySelector('img')).not.toBeInTheDocument();
    expect(container.querySelector('script')).not.toBeInTheDocument();
  });

  it('does not emit an anchor from \\href, because trust is disabled', () => {
    // With trust enabled a model could put a link inside a formula and bypass
    // the scheme allow-list that governs every other link on the page. Temml
    // instead renders `\href` itself as a rejected command and typesets the
    // argument as individual glyphs — so the characters j-a-v-a... do appear
    // in the markup, but only as <mi> elements. Asserting on the absence of
    // the substring would therefore pass for the wrong reason; the real
    // question is whether anything CLICKABLE was produced.
    const container = draw('$\\href{javascript:alert(1)}{click}$');
    expect(container.querySelector('a')).not.toBeInTheDocument();
    expect(container.querySelector('[href]')).not.toBeInTheDocument();
    expect(container.querySelector('[onclick]')).not.toBeInTheDocument();
  });

  it('renders a malformed formula without taking out the block around it', () => {
    // throwOnError is off, so the reader is never left with a hole where an
    // answer was.
    const container = draw('before $\\nosuchcommand{x}$ after');
    expect(container).toHaveTextContent('before');
    expect(container).toHaveTextContent('after');
  });

  it('renders raw HTML blocks as text', () => {
    const container = draw('<div onclick="alert(1)">hello</div>');
    expect(container.querySelector('div[onclick]')).not.toBeInTheDocument();
    expect(container).toHaveTextContent('hello');
  });
});

describe('math', () => {
  it('typesets an inline formula as MathML', () => {
    expect(mathRoot(draw('the value $x^2$ here'))).not.toBeNull();
  });

  it('typesets a display formula', () => {
    expect(mathRoot(draw('$$\\int_0^1 x\\,dx$$'))).not.toBeNull();
  });

  it('renders bracket-form math, which CommonMark would otherwise eat', () => {
    expect(mathRoot(draw('So: \\[ 0.85P = 47 \\]'))).not.toBeNull();
  });

  it('survives marked shredding backslash escapes in the inline lexer', () => {
    // The regression this pins, and the reason math is segmented from the
    // block's RAW source rather than from a token tree: marked's inline lexer
    // applies CommonMark's escape rule, turning `\[`, `\]` and `\,` into
    // `escape` tokens. Segmenting after that point sees
    // escape/text/escape and the formula is already unrecoverable — which is
    // precisely the "delimiters silently stripped, LaTeX left as source"
    // failure the Mac app documented.
    for (const source of [
      'So: \\[ 0.85P = 47 \\]',
      '$$\\int_0^1 x\\,dx$$',
      'where \\( P = \\frac{47}{0.85} \\) exactly',
    ]) {
      const container = draw(source);
      expect(mathRoot(container), source).not.toBeNull();
      // And the delimiters must not survive as visible text.
      expect(container.textContent, source).not.toContain('\\[');
      expect(container.textContent, source).not.toContain('\\frac');
    }
  });

  it('still renders prose around a formula', () => {
    const container = draw('before \\( x \\) after');
    expect(container).toHaveTextContent('before');
    expect(container).toHaveTextContent('after');
  });

  it('keeps inline formatting in the prose around a formula', () => {
    // The prose fragments between formulas are inline-lexed separately, so
    // this checks the re-lex actually happens rather than being emitted raw.
    const container = draw('**bold** then $x$ then `code`');
    expect(container.querySelector('strong')).toHaveTextContent('bold');
    expect(container.querySelector('code')).toHaveTextContent('code');
  });

  it('shows the source when math rendering is set to source', () => {
    // The escape hatch for a browser whose MathML support is poor enough to
    // flatten a fraction into "12".
    const container = draw('the value $x^2$ here', { math: 'source' });
    expect(mathRoot(container)).toBeNull();
    expect(container).toHaveTextContent('x^2');
  });

  it('shows source rather than typesetting while streaming', () => {
    const container = draw('the value $x^2$ here', { streaming: true });
    expect(mathRoot(container)).toBeNull();
  });

  it('leaves currency alone', () => {
    const container = draw('between $20 to $30 per unit');
    expect(mathRoot(container)).toBeNull();
    expect(container).toHaveTextContent('between $20 to $30 per unit');
  });

  it('does not treat a dollar inside inline code as math', () => {
    const container = draw('the price is `$5.00` exactly');
    expect(mathRoot(container)).toBeNull();
    expect(container.querySelector('code')).toHaveTextContent('$5.00');
  });

  it('does not treat a dollar inside a fence as math', () => {
    const container = draw('```bash\necho "$x$"\n```');
    expect(mathRoot(container)).toBeNull();
  });
});
