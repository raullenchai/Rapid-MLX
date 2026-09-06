/**
 * A strict allow-list for inline HTML a model emits in its prose.
 *
 * Models routinely write `<b>`, `<br>`, `<sub>` and friends, and rendering
 * them as literal tag soup reads as a rendering failure. Rendering them by
 * handing a string to `innerHTML` would be the other, worse failure: this app
 * feeds the model page content through `browse` and MCP connectors, so "the
 * model wrote it" is not a trust signal.
 *
 * Two rules make this safe by construction, and both are load-bearing:
 *
 * 1. **Only the tags below.** `script`, `img`, `iframe`, `svg`, `object`,
 *    `style`, `form` and everything else render as TEXT, exactly as before.
 * 2. **No attributes, ever.** A tag is recognised only in its bare form —
 *    `<b>`, `</b>`, `<br/>`. `<b onclick=…>` does not match and renders as
 *    text. So there is no attribute to sanitise, and `onerror`/`onload`/
 *    `style`/`href` have no way in.
 *
 * The output is a tree the caller turns into React elements. No HTML string
 * is ever produced, so there is nothing for a sanitiser to miss.
 */

/** Tags that carry meaning and cannot execute or load anything. */
export const ALLOWED_TAGS = new Set([
  'b',
  'strong',
  'i',
  'em',
  'u',
  's',
  'strike',
  'del',
  'ins',
  'mark',
  'small',
  'sub',
  'sup',
  'code',
  'kbd',
  'samp',
  'var',
  'q',
  'cite',
  'abbr',
  'br',
  'wbr',
]);

/** Tags with no closing form. */
export const VOID_TAGS = new Set(['br', 'wbr']);

/** Deprecated spellings React refuses, mapped to their modern element. */
const TAG_ALIASES: Record<string, string> = { strike: 's' };

export function elementFor(tag: string): string {
  return TAG_ALIASES[tag] ?? tag;
}

/**
 * A bare tag and nothing else.
 *
 * The absence of an attribute clause is the security property — not an
 * omission. Widening this to accept attributes would require an attribute
 * allow-list, a URL scheme check and an `on*` filter, none of which exist
 * here.
 */
const BARE_TAG = /^<(\/?)([a-zA-Z][a-zA-Z0-9]*)\s*(\/?)>$/;

/** Anything that LOOKS like a tag, so a rejected one is consumed whole and
 *  emitted as text rather than being partly interpreted. */
const TAG_LIKE = /<\/?[a-zA-Z][^>]*>/g;

export type HtmlEvent =
  | { kind: 'open'; tag: string }
  | { kind: 'close'; tag: string }
  | { kind: 'void'; tag: string }
  | { kind: 'text'; value: string };

/** Split a raw fragment into tag events and literal text. */
export function scanHtml(raw: string): HtmlEvent[] {
  const events: HtmlEvent[] = [];
  let cursor = 0;

  TAG_LIKE.lastIndex = 0;
  for (let match = TAG_LIKE.exec(raw); match !== null; match = TAG_LIKE.exec(raw)) {
    if (match.index > cursor) {
      events.push({ kind: 'text', value: raw.slice(cursor, match.index) });
    }
    cursor = match.index + match[0].length;

    const parsed = BARE_TAG.exec(match[0]);
    const tag = parsed?.[2]?.toLowerCase();
    if (parsed === null || tag === undefined || !ALLOWED_TAGS.has(tag)) {
      // Not on the list, or carries attributes: the literal source is what
      // the reader sees, which is what this did for every tag before.
      events.push({ kind: 'text', value: match[0] });
      continue;
    }

    if (VOID_TAGS.has(tag)) events.push({ kind: 'void', tag });
    else if (parsed[1] === '/') events.push({ kind: 'close', tag });
    else if (parsed[3] === '/') events.push({ kind: 'void', tag });
    else events.push({ kind: 'open', tag });
  }

  if (cursor < raw.length) events.push({ kind: 'text', value: raw.slice(cursor) });
  return events;
}

export type HtmlNode<T> =
  | { kind: 'text'; value: string }
  | { kind: 'payload'; value: T }
  | { kind: 'element'; tag: string; children: HtmlNode<T>[] };

/**
 * Fold a flat event stream into a tree.
 *
 * `payload` items pass through untouched, which is what lets the inline
 * caller interleave marked's own tokens with tag events — `<b>` and `</b>`
 * arrive as SEPARATE tokens with the emphasised text between them, so the
 * nesting has to be rebuilt here rather than read off the token tree.
 *
 * Total by construction: an unclosed tag closes at the end (so a half-arrived
 * `<b>` renders bold mid-stream instead of flickering), and a stray closing
 * tag with nothing open renders as text.
 */
export function foldHtml<T>(
  items: Array<HtmlEvent | { kind: 'payload'; value: T }>,
): HtmlNode<T>[] {
  const root: HtmlNode<T>[] = [];
  const stack: Array<{ tag: string; children: HtmlNode<T>[] }> = [];
  const top = () => stack[stack.length - 1]?.children ?? root;

  for (const item of items) {
    switch (item.kind) {
      case 'open':
        stack.push({ tag: item.tag, children: [] });
        break;
      case 'close': {
        // Searched from the top down rather than with `findLastIndex`, which
        // needs a newer lib target than this project sets.
        let index = -1;
        for (let at = stack.length - 1; at >= 0; at -= 1) {
          if (stack[at]?.tag === item.tag) {
            index = at;
            break;
          }
        }
        if (index === -1) {
          // Nothing to close. Showing the source is honest; silently dropping
          // it would hide what the model actually wrote.
          top().push({ kind: 'text', value: `</${item.tag}>` });
          break;
        }
        // Everything opened inside it closes with it, so misnesting cannot
        // leave a frame stranded on the stack.
        while (stack.length > index) {
          const frame = stack.pop();
          if (frame === undefined) break;
          (stack[stack.length - 1]?.children ?? root).push({
            kind: 'element',
            tag: frame.tag,
            children: frame.children,
          });
        }
        break;
      }
      case 'void':
        top().push({ kind: 'element', tag: item.tag, children: [] });
        break;
      case 'text':
        top().push({ kind: 'text', value: item.value });
        break;
      case 'payload':
        top().push({ kind: 'payload', value: item.value });
        break;
    }
  }

  // Auto-close what the model left open.
  while (stack.length > 0) {
    const frame = stack.pop();
    if (frame === undefined) break;
    (stack[stack.length - 1]?.children ?? root).push({
      kind: 'element',
      tag: frame.tag,
      children: frame.children,
    });
  }

  return root;
}
