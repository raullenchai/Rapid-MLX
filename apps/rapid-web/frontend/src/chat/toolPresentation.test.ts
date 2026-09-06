import { describe, expect, it } from 'vitest';
import { parseMarkdown } from '@/markdown/lex';
import { normalizeDirectoryListing } from './toolPresentation';

describe('normalizeDirectoryListing', () => {
  it('turns a flattened filesystem result into one Markdown item per entry', () => {
    const normalized = normalizeDirectoryListing(
      'Files:\n-[FILE] a.txt -[DIR] Notes -[FILE] b.txt',
    );
    expect(normalized).toBe(
      'Files:\n- `[FILE]` a.txt\n- `[DIR]` Notes\n- `[FILE]` b.txt',
    );
    const list = parseMarkdown(normalized).find((token) => token.type === 'list');
    expect(list?.type === 'list' ? list.items : []).toHaveLength(3);
  });

  it('normalizes entries that already have line breaks', () => {
    expect(normalizeDirectoryListing('[FILE] a.txt\n[DIR] Notes')).toBe(
      '- `[FILE]` a.txt\n- `[DIR]` Notes',
    );
  });

  it('leaves a single marker in ordinary prose alone', () => {
    expect(normalizeDirectoryListing('The tool labels files as [FILE] name.')).toBe(
      'The tool labels files as [FILE] name.',
    );
  });

  it('does not rewrite fenced examples', () => {
    const example = '```text\n-[FILE] a -[DIR] b\n```';
    expect(normalizeDirectoryListing(example)).toBe(example);
  });
});
