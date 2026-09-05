import { memo, useRef, useState, useSyncExternalStore } from 'react';
import {
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  CircleCheck,
  Loader2,
  Pencil,
  RotateCw,
  Trash2,
} from 'lucide-react';
import { Markdown } from '@/components/chat/Markdown';
import { parseMarkdown, tokensOf } from '@/markdown/lex';
import { streamingStore } from '@/chat/StreamingStore';
import { withoutToolCallEcho } from '@/chat/toolEcho';
import { normalizeDirectoryListing } from '@/chat/toolPresentation';
import { CopyButton } from '@/components/common/CopyButton';
import { cn } from '@/lib/utils';
import { formatDuration, formatTokensPerSecond } from '@/lib/format';
import type { ToolCall } from '@/api/chat';
import type { MessageNode } from '@/state/types';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';

export interface MessageRowProps {
  node: MessageNode;
  mathRendering: 'mathml' | 'source';
  /** Position within its sibling group, when there is more than one. */
  branch: { index: number; total: number } | null;
  /** Results for this node's tool calls, keyed by call id. Absent means the
   *  call has not come back yet. */
  toolResults?: Map<string, MessageNode>;
  onBranch(direction: -1 | 1): void;
  onRetry(): void;
  onEdit(content: string): void;
  onDelete(): void;
  /** Blocks every action: the in-flight turn writes to a specific node id,
   *  and swapping the tree under it lands tokens in the wrong branch. */
  busy: boolean;
}

export const MessageRow = memo(function MessageRow(props: MessageRowProps) {
  const { node } = props;
  // A tool result is not a row. It is rendered inside the chip belonging to
  // the call that produced it.
  if (node.role === 'tool') return null;
  if (node.role === 'user') return <UserRow {...props} />;
  return <AssistantRow {...props} />;
});

function UserRow({ node, branch, onBranch, onEdit, onDelete, busy }: MessageRowProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(node.content);

  if (editing) {
    return (
      <div className="animate-in fade-in-0 slide-in-from-bottom-2 group flex flex-col items-end">
        <div className="flex w-full max-w-[84%] flex-col gap-2">
          <Textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            rows={Math.min(10, draft.split('\n').length + 1)}
            aria-label="Edit message"
            autoFocus
          />
          <div className="flex justify-end gap-2">
            <Button variant="ghost" size="sm" onClick={() => setEditing(false)}>
              Cancel
            </Button>
            <Button
              size="sm"
              disabled={draft.trim() === '' || draft === node.content}
              onClick={() => {
                setEditing(false);
                onEdit(draft.trim());
              }}
            >
              Send
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="animate-in fade-in-0 slide-in-from-bottom-2 group flex flex-col items-end">
      {/* Plain text, never markdown: this is what the user typed, and
          rendering it would change what they see from what they wrote.

          `secondary`, NOT `primary`: shadcn's primary is near-black in light
          and near-white in dark, so the bubble INVERTED the theme — a slab of
          black on a white page, and a white slab at night. Secondary is a
          tint of the surface in both, so the prompt still reads as distinct
          from the answer without fighting the page around it. */}
      <div className="bg-secondary text-secondary-foreground max-w-[84%] rounded-lg rounded-br-sm px-4 py-2.5 shadow-xs [overflow-wrap:anywhere] whitespace-pre-wrap">
        {node.content}
      </div>
      <MessageActions>
        <BranchSwitcher branch={branch} onBranch={onBranch} noun="version" busy={busy} />
        <CopyButton text={node.content} />
        <ActionButton
          label="Edit"
          icon={<Pencil />}
          onClick={() => {
            setDraft(node.content);
            setEditing(true);
          }}
          disabled={busy}
        />
        <ActionButton label="Delete" icon={<Trash2 />} onClick={onDelete} disabled={busy} />
      </MessageActions>
    </div>
  );
}

function AssistantRow({
  node,
  mathRendering,
  branch,
  toolResults,
  onBranch,
  onRetry,
  onDelete,
  busy,
}: MessageRowProps) {
  const streaming = node.status === 'streaming';
  // A turn that dispatched tools is a step, not an answer: Copy would copy a
  // fragment, Retry would re-run the dispatch rather than the reply, and its
  // throughput belongs to the answer that follows and prints its own.
  //
  // Keyed on the calls alone, NOT on empty content: a model that also narrates
  // the call — or leaks the raw JSON into its prose, which small ones do — is
  // still dispatching, and testing for emptiness put a full stats row between
  // the call and its result.
  const dispatchOnly = (node.toolCalls?.length ?? 0) > 0;

  return (
    <div className="animate-in fade-in-0 slide-in-from-bottom-2 group flex flex-col items-start">
      {/* `w-full`, not just `max-w-full`: the column is `items-start`, so
          without it this box shrink-wraps its content and a code block or a
          table ends up as wide as its longest line rather than as wide as the
          transcript. */}
      <div className="w-full max-w-full leading-relaxed">
        {streaming ? (
          <StreamingBody mathRendering={mathRendering} />
        ) : (
          <SettledBody node={node} mathRendering={mathRendering} />
        )}
      </div>

      {node.toolCalls?.length ? (
        <div className="mt-2 flex w-full flex-col gap-1.5">
          {node.toolCalls.map((call) => (
            <ToolCallChip key={call.id} call={call} result={toolResults?.get(call.id)} />
          ))}
        </div>
      ) : null}

      {node.status === 'failed' && node.error ? (
        <div
          className="border-destructive/40 text-destructive mt-2 rounded-md border px-3 py-2 text-sm"
          role="alert"
        >
          {node.error.message}
        </div>
      ) : null}

      {!streaming && !dispatchOnly ? (
        // The actions and the stats share ONE row: actions left, stats
        // right. `w-full` is what lets `ml-auto` push the stats to the far
        // edge — without it the row shrink-wraps the buttons and there is no
        // space to push into.
        <MessageActions className="w-full">
          <BranchSwitcher branch={branch} onBranch={onBranch} noun="response" busy={busy} />
          <CopyButton text={normalizeDirectoryListing(node.content)} />
          <ActionButton label="Retry" icon={<RotateCw />} onClick={onRetry} disabled={busy} />
          <ActionButton label="Delete" icon={<Trash2 />} onClick={onDelete} disabled={busy} />
          {node.stats ? <Stats stats={node.stats} className="ml-auto pl-3" /> : null}
        </MessageActions>
      ) : null}
    </div>
  );
}

/**
 * One tool call, with its result folded underneath.
 *
 * The phase is derived from the result rather than stored: a call is running
 * until one arrives, and then it either succeeded or it did not. A separate
 * state field could disagree with the transcript it describes.
 */
function ToolCallChip({ call, result }: { call: ToolCall; result: MessageNode | undefined }) {
  const running = result === undefined;
  const failed = result?.status === 'failed';

  return (
    <details
      className="border-l-2 pl-2.5 text-sm"
      // Open while it runs and when it failed; a success is the ordinary case
      // and does not need its payload on screen.
      open={running || failed}
    >
      <summary className="text-muted-foreground flex cursor-pointer items-center gap-1.5 py-0.5 text-xs">
        {running ? (
          <Loader2 className="size-3.5 animate-spin" />
        ) : failed ? (
          <CircleAlert className="text-destructive size-3.5" />
        ) : (
          <CircleCheck className="text-success size-3.5" />
        )}
        <span className="font-mono">{call.function.name}</span>
        {running ? <span>calling…</span> : null}
      </summary>
      <div className="text-muted-foreground pb-1 text-[13px] leading-normal">
        <pre className="mt-1 overflow-x-auto font-mono text-[12px] whitespace-pre-wrap">
          {prettyArguments(call.function.arguments)}
        </pre>
        {result ? (
          <pre className="mt-1.5 max-h-56 overflow-auto font-mono text-[12px] whitespace-pre-wrap">
            {result.content}
          </pre>
        ) : null}
      </div>
    </details>
  );
}

/** Re-indent the model's argument JSON, or show it verbatim if it is not JSON. */
function prettyArguments(raw: string): string {
  try {
    return JSON.stringify(JSON.parse(raw || '{}'), null, 2);
  } catch {
    return raw;
  }
}

/**
 * The `‹ 2/3 ›` control, shared by both roles.
 *
 * On a user row the alternatives are edits of the prompt, on an assistant row
 * they are retries of the answer — so the noun differs, but nothing else does.
 * Rendered only when there IS an alternative: a permanent `1/1` with two dead
 * arrows on every row is noise, and the control appearing is itself the signal
 * that the previous version is still reachable.
 */
function BranchSwitcher({
  branch,
  onBranch,
  noun,
  busy,
}: {
  branch: MessageRowProps['branch'];
  onBranch: MessageRowProps['onBranch'];
  noun: 'version' | 'response';
  busy: boolean;
}) {
  if (!branch || branch.total <= 1) return null;
  const Noun = noun === 'version' ? 'Version' : 'Response';

  return (
    <span className="mr-1 inline-flex items-center gap-px">
      <Button
        variant="ghost"
        size="icon"
        className="text-muted-foreground size-7 [&_svg:not([class*=size-])]:size-3.5"
        // Bounded, not wrapping: the disabled state then matches what
        // the control actually does at each end.
        disabled={branch.index === 0 || busy}
        onClick={() => onBranch(-1)}
        aria-label={`Previous ${noun}`}
        title={`Previous ${noun}`}
      >
        <ChevronLeft />
      </Button>
      <span
        className="text-muted-foreground min-w-[26px] text-center font-mono text-[11px]"
        aria-label={`${Noun} ${branch.index + 1} of ${branch.total}`}
      >
        {branch.index + 1}/{branch.total}
      </span>
      <Button
        variant="ghost"
        size="icon"
        className="text-muted-foreground size-7 [&_svg:not([class*=size-])]:size-3.5"
        disabled={branch.index === branch.total - 1 || busy}
        onClick={() => onBranch(1)}
        aria-label={`Next ${noun}`}
        title={`Next ${noun}`}
      >
        <ChevronRight />
      </Button>
    </span>
  );
}

/** The settled renderer. Tokens are parsed once, on the final content. */
function SettledBody({
  node,
  mathRendering,
}: {
  node: MessageNode;
  mathRendering: 'mathml' | 'source';
}) {
  // Rendered, not stored: the transcript keeps what the model actually said,
  // so the echo can stop being hidden without the history having been edited.
  const content = normalizeDirectoryListing(withoutToolCallEcho(node.content, node.toolCalls));
  const tokens = useMemoTokens(content);

  return (
    <>
      {node.reasoning ? <Reasoning text={node.reasoning} /> : null}
      {content === '' ? (
        // A turn that only asked for tools is not an empty answer — the chips
        // below it are what it produced.
        node.status !== 'failed' && !node.toolCalls?.length ? (
          <p className="text-muted-foreground italic">(empty response)</p>
        ) : null
      ) : (
        <Markdown tokens={tokens} mathRendering={mathRendering} />
      )}
    </>
  );
}

/**
 * The streaming renderer.
 *
 * Subscribes to StreamingStore rather than to the app store, so a commit
 * re-renders this one row and nothing else. Tokens come from the incremental
 * lexer, so the blocks above the tail keep their identity and their DOM.
 */
function StreamingBody({ mathRendering }: { mathRendering: 'mathml' | 'source' }) {
  const snapshot = useSyncExternalStore(streamingStore.subscribe, streamingStore.getSnapshot);
  const tokens = tokensOf(snapshot.lex);

  return (
    <>
      {snapshot.reasoning ? <Reasoning text={snapshot.reasoning} defaultOpen /> : null}
      <Markdown tokens={tokens} streaming mathRendering={mathRendering} />
      <span
        className="bg-foreground ml-px inline-block h-[1.05em] w-0.5 animate-[caret-blink_1s_steps(2,start)_infinite] align-text-bottom"
        aria-hidden="true"
      />
    </>
  );
}

/** A reasoning model's scratchpad. Collapsed by default once settled. */
function Reasoning({ text, defaultOpen }: { text: string; defaultOpen?: boolean }) {
  return (
    <details className="mb-2.5 border-l-2 pl-2.5" open={defaultOpen}>
      <summary className="reasoning-summary text-muted-foreground py-0.5 text-xs">Thinking</summary>
      {/* Plain text, not markdown: a scratchpad is not prose, and parsing it
          would spend the budget on something nobody reads twice. */}
      <div className="text-muted-foreground pb-1 text-[13.5px] leading-normal whitespace-pre-wrap">
        {text}
      </div>
    </details>
  );
}

function Stats({
  stats,
  className,
}: {
  stats: NonNullable<MessageNode['stats']>;
  className?: string;
}) {
  const tps = formatTokensPerSecond(stats.tps);
  const ttft = formatDuration(stats.ttftMs);

  return (
    <div
      className={cn(
        // No wrap: this sits on one line beside the actions now, and a
        // wrapped metric would push the row to two lines on a phone.
        'text-muted-foreground flex shrink-0 items-center gap-2.5 font-mono text-[11px] whitespace-nowrap',
        className,
      )}
    >
      {tps ? <span className="text-success">{tps}</span> : null}
      <span>
        {stats.tokens} tokens
        {/* Marked, because a count derived from content.length/4 is off by a
            wide and model-dependent margin and must not read as measured. */}
        {stats.tokensEstimated ? ' (est.)' : ''}
      </span>
      {ttft ? <span>{ttft} to first token</span> : null}
    </div>
  );
}

/**
 * The row under a message: actions on the left, stats on the right.
 *
 * Always visible. These used to fade in on hover, which put Copy and Retry
 * behind a gesture that does not exist on touch and left the throughput
 * numbers invisible until the pointer happened to land on the message.
 */
function MessageActions({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn('mt-[3px] flex items-center gap-0.5', className)}>{children}</div>
  );
}

/**
 * One message action, as an icon.
 *
 * The label does not disappear with the text — it moves to `aria-label` and
 * `title`, so a screen reader still announces "Retry" and a pointer user
 * still gets a tooltip. An icon-only control with neither is unusable.
 */
function ActionButton({
  label,
  icon,
  onClick,
  disabled,
}: {
  label: string;
  icon: React.ReactNode;
  onClick(): void;
  disabled?: boolean;
}) {
  return (
    <Button
      variant="ghost"
      size="icon"
      className="text-muted-foreground size-7 [&_svg:not([class*=size-])]:size-3.5"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
    >
      {icon}
    </Button>
  );
}

/** Parse once per distinct content string. */
function useMemoTokens(content: string) {
  const cache = useRef<{
    content: string;
    tokens: ReturnType<typeof parseMarkdown>;
  } | null>(null);
  if (cache.current?.content !== content) {
    cache.current = { content, tokens: parseMarkdown(content) };
  }
  return [...cache.current.tokens];
}
