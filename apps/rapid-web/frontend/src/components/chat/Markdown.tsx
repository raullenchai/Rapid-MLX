import { memo, useMemo, useState } from 'react';
import type { Token, Tokens } from 'marked';
import { highlight, type HastNode } from '@/markdown/highlight';
import {
  elementFor,
  foldHtml,
  scanHtml,
  VOID_TAGS,
  type HtmlEvent,
  type HtmlNode,
} from '@/markdown/html';
import { parseMarkdown } from '@/markdown/lex';
import { safeHref, safeImageSrc } from '@/markdown/links';
import { segmentLaTeX } from '@/markdown/latex';
import { renderMath } from '@/markdown/math';
import { copyText } from '@/lib/clipboard';

/**
 * Markdown tokens to React elements.
 *
 * Tokens, never an HTML string: React escapes every text node by construction,
 * so there is nothing to sanitize. The single exception is Temml's MathML,
 * the app's only `dangerouslySetInnerHTML`, which has its own fixtures.
 *
 * Two modes, selected by the `streaming` prop:
 *
 *   settled     highlighting, copy buttons, math
 *   streaming   the cheap one — plain code, closed math only
 *
 * The split keys off `message.status === 'streaming'`, never a global "is
 * anything streaming" flag: the latter is true for the whole transcript while
 * the last answer arrives, so it would swap every earlier message's renderer
 * the moment a new turn starts.
 */

export interface MarkdownProps {
  tokens: Token[];
  /** Cheap mode: no highlighting, no copy buttons, math only when closed. */
  streaming?: boolean;
  mathRendering: 'mathml' | 'source';
}

export const Markdown = memo(function Markdown({ tokens, streaming, mathRendering }: MarkdownProps) {
  return (
    <>
      {tokens.map((token, index) => (
        <Block
          // Index is a stable key here BECAUSE frozen tokens never reorder:
          // blocks are only ever appended during a stream, and a settled
          // message is immutable.
          key={index}
          token={token}
          streaming={streaming ?? false}
          mathRendering={mathRendering}
        />
      ))}
    </>
  );
});

interface BlockProps {
  token: Token;
  streaming: boolean;
  mathRendering: 'mathml' | 'source';
}

/**
 * Memoised on the token's identity.
 *
 * This is what makes incremental lexing pay off: `advance` keeps frozen
 * tokens referentially identical across commits, so every block above the
 * streaming tail skips reconciliation entirely. Their DOM is never touched,
 * which is why an open `<details>`, a text selection and a `<pre>`'s scroll
 * position all survive a commit — none of which was true of the old page.
 */
const Block = memo(function Block({ token, streaming, mathRendering }: BlockProps) {
  switch (token.type) {
    case 'heading': {
      const heading = token as Tokens.Heading;
      const Tag = `h${Math.min(heading.depth, 6)}` as 'h1';
      return (
        <Tag className={'md-h'}>
          <InlineSource
            source={heading.text}
            fallback={heading.tokens}
            mathRendering={mathRendering}
            streaming={streaming}
          />
        </Tag>
      );
    }

    case 'paragraph': {
      const paragraph = token as Tokens.Paragraph;
      return (
        <p className={'md-p'}>
          <InlineSource
            source={paragraph.text}
            fallback={paragraph.tokens}
            mathRendering={mathRendering}
            streaming={streaming}
          />
        </p>
      );
    }

    case 'code':
      return <CodeBlock token={token as Tokens.Code} streaming={streaming} />;

    case 'blockquote': {
      const quote = token as Tokens.Blockquote;
      return (
        <blockquote className={'md-quote'}>
          <Markdown tokens={quote.tokens} streaming={streaming} mathRendering={mathRendering} />
        </blockquote>
      );
    }

    case 'list': {
      const list = token as Tokens.List;
      const Tag = list.ordered ? 'ol' : 'ul';
      const startsAtOne = list.start === '' || list.start === 1;
      return (
        <Tag
          className={'md-list'}
          {...(list.ordered && !startsAtOne ? { start: Number(list.start) } : {})}
        >
          {list.items.map((item, index) => (
            <li key={index} className={'md-li'}>
              {item.task ? (
                // Rendered disabled: a checkbox in a transcript reports what
                // the model wrote, and clicking it would edit the answer.
                <input
                  type="checkbox"
                  checked={item.checked ?? false}
                  disabled
                  readOnly
                  className={"mr-2 align-middle"}
                />
              ) : null}
              <Markdown tokens={item.tokens} streaming={streaming} mathRendering={mathRendering} />
            </li>
          ))}
        </Tag>
      );
    }

    case 'table':
      return (
        <Table token={token as Tokens.Table} streaming={streaming} mathRendering={mathRendering} />
      );

    case 'hr':
      return <hr className={'md-rule'} />;

    case 'space':
      return null;

    case 'html':
      // A whole block of raw HTML. Only the allow-listed inline tags become
      // elements; a `<script>`, an `<img>` or anything carrying an attribute
      // renders as its literal source, exactly as this did before.
      return (
        <p className={'md-p'}>
          <Inline
            tokens={[token]}
            mathRendering={mathRendering}
            streaming={streaming}
          />
        </p>
      );

    default: {
      const fallback = token as { tokens?: Token[]; raw?: string };
      if (fallback.tokens) {
        return (
          <p className={'md-p'}>
            <Inline tokens={fallback.tokens} mathRendering={mathRendering} streaming={streaming} />
          </p>
        );
      }
      return fallback.raw ? <p className={'md-p'}>{fallback.raw}</p> : null;
    }
  }
});

function Table({
  token,
  streaming,
  mathRendering,
}: { token: Tokens.Table } & Omit<BlockProps, 'token'>) {
  return (
    // A table is the one block that can exceed the reading measure, so it
    // scrolls horizontally on its own. `tabIndex` and the role make that
    // scroll region reachable by keyboard, which a bare overflow container
    // is not.
    <div className={'md-table-wrap'} role="region" aria-label="Table" tabIndex={0}>
      <table className={'md-table'}>
        <thead>
          <tr>
            {token.header.map((cell, index) => (
              <th key={index} style={alignmentOf(token.align[index])}>
                <InlineSource
                  source={cell.text}
                  fallback={cell.tokens}
                  mathRendering={mathRendering}
                  streaming={streaming}
                />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {token.rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell, cellIndex) => (
                <td key={cellIndex} style={alignmentOf(token.align[cellIndex])}>
                  <InlineSource
                    source={cell.text}
                    fallback={cell.tokens}
                    mathRendering={mathRendering}
                    streaming={streaming}
                  />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function alignmentOf(align: 'center' | 'left' | 'right' | null | undefined) {
  return align ? { textAlign: align } : undefined;
}

function CodeBlock({ token, streaming }: { token: Tokens.Code; streaming: boolean }) {
  const [copied, setCopied] = useState(false);

  // Highlighting is skipped entirely while streaming: the body changes on
  // every commit, so the work would be thrown away each time, and a partial
  // line highlights as broken syntax anyway.
  const tree = useMemo(
    () => (streaming ? null : highlight(token.text, token.lang)),
    [streaming, token.text, token.lang],
  );

  const onCopy = () => {
    void copyText(token.text).then((ok) => {
      if (!ok) return;
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    });
  };

  return (
    <div className={'md-code-wrap'}>
      {!streaming ? (
        <div className={"text-muted-foreground flex items-center justify-between gap-2 border-b py-1 pr-2 pl-3"}>
          <span className={"font-mono text-[10.5px] tracking-[0.08em] uppercase"}>{token.lang?.trim().split(/\s+/)[0] ?? ''}</span>
          <button type="button" className={"hover:bg-accent hover:text-accent-foreground rounded-sm px-2 py-[3px] text-xs"} onClick={onCopy}>
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
      ) : null}
      <pre className={'md-pre'}>
        <code>{tree ? <Hast node={tree} /> : token.text}</code>
      </pre>
    </div>
  );
}

/** lowlight's hast tree, as React elements — so highlighting introduces no
 *  markup of its own. */
function Hast({ node }: { node: HastNode }): React.ReactNode {
  if (node.type === 'text') return node.value;
  if (node.type === 'root') {
    return node.children.map((child, index) => <Hast key={index} node={child} />);
  }
  return (
    <span className={node.properties?.className?.join(' ')}>
      {node.children.map((child, index) => (
        <Hast key={index} node={child} />
      ))}
    </span>
  );
}

interface InlineProps {
  tokens: Token[] | undefined;
  mathRendering: 'mathml' | 'source';
  streaming: boolean;
}

/**
 * Inline content, segmented for math BEFORE it is inline-lexed.
 *
 * The ordering is forced and is the whole reason this component exists.
 * `marked`'s inline lexer applies CommonMark's backslash-escape rule, turning
 * `\[`, `\]`, `\(`, `\)` and `\,` into `escape` tokens — so by the time a
 * token tree exists, `\[ 0.85P = 47 \]` is shredded and unrecoverable.
 *
 * So math is found on the block's RAW inline source and only the prose between
 * formulas reaches the inline lexer. The cost: inline code is no longer
 * structurally excluded, so `segmentLaTeX` skips backtick runs itself.
 */
function InlineSource({
  source,
  fallback,
  mathRendering,
  streaming,
}: {
  source: string;
  fallback: Token[] | undefined;
  mathRendering: 'mathml' | 'source';
  streaming: boolean;
}): React.ReactNode {
  const segments = useMemo(() => segmentLaTeX(source), [source]);

  // No math: use the tokens marked already produced rather than re-lexing.
  const hasMath = segments.some((segment) => segment.kind === 'math');
  if (!hasMath) {
    return <Inline tokens={fallback} mathRendering={mathRendering} streaming={streaming} />;
  }

  return segments.map((segment, index) => {
    if (segment.kind === 'markdown') {
      return (
        <Inline
          key={index}
          tokens={inlineTokensOf(segment.text)}
          mathRendering={mathRendering}
          streaming={streaming}
        />
      );
    }
    // Mid-stream a formula may be complete while the block around it is not;
    // `segmentLaTeX` only ever yields closed runs, so this is safe to typeset
    // even while streaming. It is still deferred, because the surrounding
    // block re-lexes on every commit and typesetting would be thrown away.
    if (mathRendering === 'source' || streaming) {
      return (
        <code key={index} className={"font-mono text-muted-foreground text-[0.88em]"}>
          {segment.latex}
        </code>
      );
    }
    return <MathRun key={index} latex={segment.latex} display={segment.display} />;
  });
}

/** Inline-lex a prose fragment. */
function inlineTokensOf(source: string): Token[] {
  const blocks = parseMarkdown(source);
  const first = blocks[0] as { tokens?: Token[] } | undefined;
  // A fragment between two formulas can be whitespace, which lexes to nothing.
  return first?.tokens ?? [{ type: 'text', raw: source, text: source }];
}

function Inline({ tokens, mathRendering, streaming }: InlineProps): React.ReactNode {
  if (!tokens) return null;

  // `<b>` and `</b>` arrive as two SEPARATE html tokens with the emphasised
  // run between them, so the nesting has to be rebuilt from the flat list —
  // it is not in the token tree. Everything not on the allow-list becomes a
  // text event and renders exactly as it did before.
  type Payload = { token: Token; index: number };
  const items: Array<HtmlEvent | { kind: 'payload'; value: Payload }> = [];
  tokens.forEach((token, index) => {
    if (token.type === 'html') items.push(...scanHtml((token as Tokens.HTML).raw));
    else items.push({ kind: 'payload', value: { token, index } });
  });

  return (
    <HtmlTree
      nodes={foldHtml(items)}
      mathRendering={mathRendering}
      streaming={streaming}
    />
  );
}

/** The folded tree as elements. Every tag came from the allow-list and every
 *  one is written here as JSX — no HTML string is ever constructed. */
function HtmlTree({
  nodes,
  mathRendering,
  streaming,
}: {
  nodes: HtmlNode<{ token: Token; index: number }>[];
} & Omit<InlineProps, 'tokens'>): React.ReactNode {
  return nodes.map((node, index) => {
    if (node.kind === 'text') return <span key={index}>{node.value}</span>;
    if (node.kind === 'payload') {
      return (
        <InlineToken
          key={index}
          token={node.value.token}
          mathRendering={mathRendering}
          streaming={streaming}
        />
      );
    }
    const Tag = elementFor(node.tag) as 'b';
    if (VOID_TAGS.has(node.tag)) return <Tag key={index} />;
    return (
      <Tag key={index}>
        <HtmlTree nodes={node.children} mathRendering={mathRendering} streaming={streaming} />
      </Tag>
    );
  });
}

function InlineToken({
  token,
  mathRendering,
  streaming,
}: { token: Token } & Omit<InlineProps, 'tokens'>): React.ReactNode {
  switch (token.type) {
    case 'text': {
      const text = token as Tokens.Text;
      // A `text` token can itself carry inline children — a link inside
      // emphasis, say. Math is NOT looked for here: InlineSource already
      // segmented it out of the raw source, before this token existed.
      if (text.tokens)
        return <Inline tokens={text.tokens} mathRendering={mathRendering} streaming={streaming} />;
      return text.text;
    }

    case 'escape':
      return (token as Tokens.Escape).text;

    case 'strong':
      return (
        <strong>
          <Inline
            tokens={(token as Tokens.Strong).tokens}
            mathRendering={mathRendering}
            streaming={streaming}
          />
        </strong>
      );

    case 'em':
      return (
        <em>
          <Inline
            tokens={(token as Tokens.Em).tokens}
            mathRendering={mathRendering}
            streaming={streaming}
          />
        </em>
      );

    case 'del':
      return (
        <del>
          <Inline
            tokens={(token as Tokens.Del).tokens}
            mathRendering={mathRendering}
            streaming={streaming}
          />
        </del>
      );

    case 'codespan':
      // Never scanned for math: `$5.00` in backticks is shell prose.
      return <code className={'md-codespan'}>{(token as Tokens.Codespan).text}</code>;

    case 'br':
      return <br />;

    case 'link': {
      const link = token as Tokens.Link;
      const href = safeHref(link.href);
      const label = (
        <Inline tokens={link.tokens} mathRendering={mathRendering} streaming={streaming} />
      );
      // A refused scheme renders as a SPAN carrying the label, not as an
      // anchor without an href: the text stays visible, selectable and
      // copyable, and the click is simply dead.
      if (href === null) return <span className={'md-dead-link'}>{label}</span>;
      return (
        <a className={'md-link'} href={href} target="_blank" rel="noopener noreferrer">
          {label}
        </a>
      );
    }

    case 'image': {
      const image = token as Tokens.Image;
      const src = safeImageSrc(image.href);
      // A blocked image renders its alt text, which describes what the model
      // meant — better than a broken-image glyph.
      if (src === null) return <span className={"text-muted-foreground italic"}>{image.text}</span>;
      return <img src={src} alt={image.text} loading="lazy" className={"h-auto max-w-full rounded-sm"} />;
    }

    case 'html':
      // Unreachable in practice: `Inline` turns every html token into tag
      // events before this sees it. Kept as the literal source so a token
      // arriving by some other path still shows what the model wrote.
      return (token as Tokens.HTML).raw;

    default:
      return (token as { raw?: string }).raw ?? null;
  }
}

/** Named MathRun, not Math: a component called `Math` shadows the global
 *  `Math` object for the whole module, and `Math.min` in the heading case
 *  then throws at render time. */
function MathRun({ latex, display }: { latex: string; display: boolean }) {
  const markup = useMemo(() => renderMath(latex, display), [latex, display]);

  // Falling back to the source rather than rendering nothing: the reader is
  // never left with a hole where a formula was.
  if (markup === '') {
    return <code className={"font-mono text-muted-foreground text-[0.88em]"}>{latex}</code>;
  }

  return (
    <span
      className={display ? 'md-math-display' : 'md-math-inline'}
      // THE ONLY dangerouslySetInnerHTML IN THE APP. Temml emits MathML as a
      // string. `trust: false` disables \href and \includegraphics; the XSS
      // fixtures in Markdown.test.tsx cover the rest, and a Temml version bump
      // must re-run them.
      dangerouslySetInnerHTML={{ __html: markup }}
      // The source stays available to a screen reader and to a reader whose
      // browser renders MathML poorly.
      title={latex}
    />
  );
}
