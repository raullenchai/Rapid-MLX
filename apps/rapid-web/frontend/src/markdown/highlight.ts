import { createLowlight } from 'lowlight';
import bash from 'highlight.js/lib/languages/bash';
import c from 'highlight.js/lib/languages/c';
import cpp from 'highlight.js/lib/languages/cpp';
import diff from 'highlight.js/lib/languages/diff';
import go from 'highlight.js/lib/languages/go';
import javascript from 'highlight.js/lib/languages/javascript';
import json from 'highlight.js/lib/languages/json';
import markdown from 'highlight.js/lib/languages/markdown';
import python from 'highlight.js/lib/languages/python';
import rust from 'highlight.js/lib/languages/rust';
import sql from 'highlight.js/lib/languages/sql';
import swift from 'highlight.js/lib/languages/swift';
import typescript from 'highlight.js/lib/languages/typescript';
import yaml from 'highlight.js/lib/languages/yaml';

/**
 * Syntax highlighting: `lowlight` over `highlight.js/lib/core`, with an
 * explicit language list.
 *
 * lowlight emits a HAST TREE rather than an HTML string, so it maps onto React
 * elements and highlighting cannot introduce markup — no
 * `dangerouslySetInnerHTML` on this path.
 *
 * `lib/core` plus explicit registration because `lib/common` drags in ~180
 * languages, ~120 KB more than this list. Measured at 79 KB minified for this
 * set; see size-budget.json.
 */

const lowlight = createLowlight({
  bash,
  c,
  cpp,
  diff,
  go,
  javascript,
  json,
  markdown,
  python,
  rust,
  sql,
  swift,
  typescript,
  yaml,
});

/**
 * Aliases models actually emit. `lowlight` knows highlight.js's own aliases
 * for registered languages, so this only covers names it would otherwise
 * reject outright.
 */
const ALIASES: Record<string, string> = {
  sh: 'bash',
  shell: 'bash',
  zsh: 'bash',
  console: 'bash',
  js: 'javascript',
  jsx: 'javascript',
  ts: 'typescript',
  tsx: 'typescript',
  py: 'python',
  python3: 'python',
  rs: 'rust',
  golang: 'go',
  yml: 'yaml',
  md: 'markdown',
  'c++': 'cpp',
  patch: 'diff',
};

export function resolveLanguage(lang: string | undefined): string | null {
  if (!lang) return null;
  // A fence can carry more than a language: ```python title=foo.py
  const first = lang.trim().split(/\s+/)[0]?.toLowerCase();
  if (!first) return null;
  const resolved = ALIASES[first] ?? first;
  return lowlight.registered(resolved) ? resolved : null;
}

export type HastNode =
  | { type: 'text'; value: string }
  | {
      type: 'element';
      tagName: string;
      properties?: { className?: string[] };
      children: HastNode[];
    }
  | { type: 'root'; children: HastNode[] };

/**
 * Highlight `code`, or return null when the language is unknown.
 *
 * Never `highlightAuto`: it runs every registered grammar over the body and is
 * by far the most expensive call in the library. An unknown language renders
 * as plain preformatted text, which is correct — a wrong guess is worse than
 * no colour.
 */
export function highlight(code: string, lang: string | undefined): HastNode | null {
  const language = resolveLanguage(lang);
  if (!language) return null;
  try {
    return lowlight.highlight(language, code) as HastNode;
  } catch {
    // A grammar can throw on pathological input. Plain text is a fine answer.
    return null;
  }
}
