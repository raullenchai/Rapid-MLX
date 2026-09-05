/**
 * Repair the compact directory-list syntax small models sometimes produce.
 *
 * The filesystem MCP server returns one `[FILE]` or `[DIR]` entry per line,
 * but a small model can join them into `-[FILE] a -[DIR] b`. Once that text
 * reaches Markdown it is one enormous list item. Turn only repeated markers
 * into real Markdown bullets; a single marker may be ordinary prose.
 */
export function normalizeDirectoryListing(text: string): string {
  // Do not rewrite examples or literal output inside fenced code.
  if (text.includes('```') || text.includes('~~~')) return text;

  const pattern = /(^|[ \t]+)-?[ \t]*\[(FILE|DIR)\][ \t]+/gm;
  if ((text.match(pattern) ?? []).length < 2) return text;

  return text.replace(pattern, (_match, boundary: string, kind: string) => {
    const lineBreak = boundary === '' ? '' : '\n';
    return `${lineBreak}- \`[${kind}]\` `;
  });
}
