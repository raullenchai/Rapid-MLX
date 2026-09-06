/**
 * Link scheme allow-list.
 *
 * Ported from `ChatLinkSafetyFilter.swift`: default-deny, with exactly `http`,
 * `https` and `mailto` permitted. Everything else — `file:`, `javascript:`,
 * `data:`, `vscode:` — renders as text with a dead click.
 *
 * `javascript:` is the case worth naming: HTML escaping does not touch a URL
 * scheme, so it survives escaping intact and lands in an `href` unchanged.
 * Here it cannot reach an `href` at all — a rejected scheme computes null.
 */

const ALLOWED_SCHEMES = new Set(['http:', 'https:', 'mailto:']);

/**
 * The href to use, or null if the link must be rendered inert.
 *
 * Parsing through `URL` rather than pattern-matching is what makes this
 * robust: the constructor normalises first, so obfuscated schemes
 * (`java\tscript:`), case variants (`JaVaScRiPt:`) and protocol-relative URLs
 * all resolve to their real protocol before the comparison.
 */
export function safeHref(raw: string, base: string = window.location.origin): string | null {
  // An empty target — `[text]()` — resolves to the page's own URL, rendering
  // as a live link that silently reloads the app and loses the draft.
  if (raw.trim() === '') return null;

  let url: URL;
  try {
    url = new URL(raw, base);
  } catch {
    return null;
  }
  return ALLOWED_SCHEMES.has(url.protocol) ? url.href : null;
}

/**
 * Images are narrower still: the CSP is `img-src 'self' data:`, so a remote
 * host would be blocked by the browser anyway and would render as a broken
 * image rather than as the alt text. Only same-origin and inline data URLs
 * can actually load, and a `data:` URL must be an image — `data:text/html`
 * is a document, not a picture.
 */
export function safeImageSrc(raw: string, base: string = window.location.origin): string | null {
  let url: URL;
  try {
    url = new URL(raw, base);
  } catch {
    return null;
  }

  if (url.protocol === 'data:') {
    return url.href.startsWith('data:image/') ? url.href : null;
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
  // Same-origin only, per the CSP. A remote URL that the browser will refuse
  // is worse than no image: it renders as a broken-image glyph instead of the
  // alt text that describes what the model meant.
  return url.origin === base ? url.href : null;
}
