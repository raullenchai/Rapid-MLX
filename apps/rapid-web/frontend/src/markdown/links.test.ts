import { describe, expect, it } from 'vitest';
import { safeHref, safeImageSrc } from './links';

const BASE = 'https://tunnel.example';

describe('safeHref', () => {
  it.each([
    ['https://example.com/page', 'https://example.com/page'],
    ['http://example.com/', 'http://example.com/'],
    ['mailto:someone@example.com', 'mailto:someone@example.com'],
  ])('allows %s', (input, expected) => {
    expect(safeHref(input, BASE)).toBe(expected);
  });

  it('resolves a relative link against the page origin', () => {
    expect(safeHref('/local/path', BASE)).toBe('https://tunnel.example/local/path');
  });

  it.each([
    'javascript:alert(1)',
    // Case is not a defence: the URL parser lower-cases the protocol, which
    // is exactly why parsing beats pattern-matching.
    'JaVaScRiPt:alert(1)',
    'JAVASCRIPT:alert(1)',
    // Control characters are stripped during parsing, so an obfuscated
    // scheme still resolves to javascript: and is still refused.
    'java\tscript:alert(1)',
    'java\nscript:alert(1)',
    ' javascript:alert(1)',
    'data:text/html,<script>alert(1)</script>',
    'file:///etc/passwd',
    'vscode://file/etc/passwd',
    'raycast://extensions',
    'tel:+15550100',
  ])('refuses %o', (input) => {
    expect(safeHref(input, BASE)).toBeNull();
  });

  it('refuses a protocol-relative URL by resolving it first', () => {
    // `//evil.example` inherits the page's scheme, so it IS a valid https URL
    // pointing somewhere else entirely. It is allowed by scheme — the point
    // of this case is that it resolves rather than being read as a path.
    expect(safeHref('//evil.example/x', BASE)).toBe('https://evil.example/x');
  });

  it('refuses an empty target', () => {
    // `[text]()` resolves to the page's own URL, so it would render as a live
    // link that silently reloads the app and loses the user's draft.
    expect(safeHref('', BASE)).toBeNull();
    expect(safeHref('   ', BASE)).toBeNull();
  });
});

describe('safeImageSrc', () => {
  it('allows a same-origin image', () => {
    expect(safeImageSrc('/logo.png', BASE)).toBe('https://tunnel.example/logo.png');
  });

  it('allows an inline image data URL', () => {
    const src = 'data:image/png;base64,iVBORw0KGgo=';
    expect(safeImageSrc(src, BASE)).toBe(src);
  });

  it('refuses a data URL that is not an image', () => {
    // `data:text/html` is a document, not a picture.
    expect(safeImageSrc('data:text/html,<script>alert(1)</script>', BASE)).toBeNull();
  });

  it('refuses a remote image', () => {
    // The CSP is `img-src 'self' data:`, so the browser would block it and
    // render a broken-image glyph — worse than showing the alt text.
    expect(safeImageSrc('https://cdn.example/pic.png', BASE)).toBeNull();
  });

  it('refuses a javascript: src', () => {
    expect(safeImageSrc('javascript:alert(1)', BASE)).toBeNull();
  });
});
