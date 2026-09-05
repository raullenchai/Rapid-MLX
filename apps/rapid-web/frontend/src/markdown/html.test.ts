import { describe, expect, it } from 'vitest';
import { foldHtml, scanHtml, type HtmlEvent } from './html';

/**
 * The allow-list, tested as a truth table.
 *
 * The security property is that a tag is recognised ONLY in its bare form, so
 * there is no attribute to sanitise. Everything below either proves a safe tag
 * becomes an element, or proves a dangerous one stays text.
 */

const text = (value: string): HtmlEvent => ({ kind: 'text', value });

describe('scanHtml — what becomes an element', () => {
  it.each(['b', 'strong', 'i', 'em', 'u', 's', 'mark', 'sub', 'sup', 'code', 'kbd'])(
    'accepts <%s>',
    (tag) => {
      expect(scanHtml(`<${tag}>x</${tag}>`)).toEqual([
        { kind: 'open', tag },
        text('x'),
        { kind: 'close', tag },
      ]);
    },
  );

  it('treats a void tag as self-closing in every spelling', () => {
    expect(scanHtml('<br>')).toEqual([{ kind: 'void', tag: 'br' }]);
    expect(scanHtml('<br/>')).toEqual([{ kind: 'void', tag: 'br' }]);
    expect(scanHtml('<br />')).toEqual([{ kind: 'void', tag: 'br' }]);
  });

  it('is case-insensitive', () => {
    expect(scanHtml('<B>x</B>')).toEqual([
      { kind: 'open', tag: 'b' },
      text('x'),
      { kind: 'close', tag: 'b' },
    ]);
  });

  it('keeps the text around a tag', () => {
    expect(scanHtml('before <b>bold</b> after')).toEqual([
      text('before '),
      { kind: 'open', tag: 'b' },
      text('bold'),
      { kind: 'close', tag: 'b' },
      text(' after'),
    ]);
  });
});

describe('scanHtml — what stays text', () => {
  // The whole point. Each of these must survive as its literal source.
  it.each([
    '<script>alert(1)</script>',
    '<img src=x onerror=alert(1)>',
    '<iframe src="javascript:alert(1)"></iframe>',
    '<svg/onload=alert(1)>',
    '<object data="x"></object>',
    '<style>body{display:none}</style>',
    '<form action="/x"></form>',
    '<a href="javascript:alert(1)">x</a>',
    '<link rel="stylesheet" href="x">',
    '<meta http-equiv="refresh" content="0">',
  ])('refuses %s', (payload) => {
    const events = scanHtml(payload);
    expect(events.every((event) => event.kind === 'text')).toBe(true);
    expect(events.map((e) => (e.kind === 'text' ? e.value : '')).join('')).toBe(payload);
  });

  it('refuses an ALLOWED tag that carries an attribute', () => {
    // This is what makes "no attribute sanitiser" sound rather than an
    // omission: an attribute means the tag is not recognised at all.
    for (const payload of [
      '<b onclick="alert(1)">x</b>',
      '<b style="position:fixed">x</b>',
      '<b id="x">y</b>',
      '<code class="hljs">x</code>',
    ]) {
      const events = scanHtml(payload);
      expect(events[0]).toEqual(text(payload.slice(0, payload.indexOf('>') + 1)));
    }
  });

  it('does not let a tag-like fragment be partly interpreted', () => {
    // Consumed whole, so `<img src=x onerror=...>` cannot leave a stray `<b>`
    // behind from inside its attribute text.
    const events = scanHtml('<img src=x onerror="<b>">');
    expect(events.every((event) => event.kind === 'text')).toBe(true);
  });

  it('leaves a bare comparison alone', () => {
    // `a < b` is prose, not a tag.
    expect(scanHtml('a < b and c > d')).toEqual([text('a < b and c > d')]);
  });
});

describe('foldHtml', () => {
  const payload = (value: string) => ({ kind: 'payload' as const, value });

  it('nests a run between its open and close', () => {
    expect(foldHtml([{ kind: 'open', tag: 'b' }, payload('x'), { kind: 'close', tag: 'b' }])).toEqual([
      { kind: 'element', tag: 'b', children: [{ kind: 'payload', value: 'x' }] },
    ]);
  });

  it('closes an unclosed tag at the end', () => {
    // Mid-stream `<b>` has not met its `</b>` yet; rendering bold now and
    // bold later beats flickering between the two.
    expect(foldHtml([{ kind: 'open', tag: 'b' }, payload('x')])).toEqual([
      { kind: 'element', tag: 'b', children: [{ kind: 'payload', value: 'x' }] },
    ]);
  });

  it('renders a stray closing tag as text', () => {
    expect(foldHtml([{ kind: 'close', tag: 'b' }])).toEqual([
      { kind: 'text', value: '</b>' },
    ]);
  });

  it('recovers from misnesting without stranding a frame', () => {
    const tree = foldHtml([
      { kind: 'open', tag: 'b' },
      { kind: 'open', tag: 'i' },
      payload('x'),
      { kind: 'close', tag: 'b' },
      payload('after'),
    ]);
    // `</b>` closes the `<i>` inside it too, and `after` lands at the root.
    expect(tree).toEqual([
      {
        kind: 'element',
        tag: 'b',
        children: [{ kind: 'element', tag: 'i', children: [{ kind: 'payload', value: 'x' }] }],
      },
      { kind: 'payload', value: 'after' },
    ]);
  });

  it('keeps a void tag childless', () => {
    expect(foldHtml([{ kind: 'void', tag: 'br' }])).toEqual([
      { kind: 'element', tag: 'br', children: [] },
    ]);
  });
});
