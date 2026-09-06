import { useMemo, useState } from 'react';
import { Archive, ChevronRight, MoreHorizontal, Pencil, Pin, Trash2 } from 'lucide-react';
import { groupConversations } from '@/chat/ConversationSearch';
import { useMidnightTick } from '@/lib/useMidnightTick';
import { useStore } from '@/state/store';
import type { Conversation } from '@/state/types';
import { ConfirmDialog } from '@/components/common/ConfirmDialog';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';

/**
 * The conversation list.
 *
 * Rendered in two places and deliberately ONE implementation: the persistent
 * sidebar on a wide screen, and the slide-out drawer below the breakpoint. Two
 * copies would drift and the narrow build would quietly become the worse one.
 *
 * `onNavigate` is what differs: the drawer closes after a selection.
 */
export function ConversationList({ onNavigate }: { onNavigate?: () => void }) {
  const conversations = useStore((state) => state.conversations);
  const activeId = useStore((state) => state.activeId);
  const setActive = useStore((state) => state.setActiveConversation);
  const remove = useStore((state) => state.deleteConversation);

  const [pendingDelete, setPendingDelete] = useState<Conversation | null>(null);
  // Collapsed by default: archiving is how a conversation is put out of the
  // way, so expanding the archive on every visit undoes what it is for.
  const [archiveOpen, setArchiveOpen] = useState(false);
  const now = useMidnightTick();

  const archived = useMemo(
    () => conversations.filter((conversation) => conversation.isArchived),
    [conversations],
  );

  // Archived conversations are hidden from the grouped list but NOT from
  // search — that is what lets the archive exist without a mode toggle.
  const grouped = useMemo(
    () =>
      groupConversations(
        conversations.filter((conversation) => !conversation.isArchived),
        now,
      ),
    [conversations, now],
  );

  const renderRow = (conversation: Conversation) => (
    <Row
      key={conversation.id}
      conversation={conversation}
      current={conversation.id === activeId}
      onOpen={() => {
        setActive(conversation.id);
        onNavigate?.();
      }}
      onDelete={() => setPendingDelete(conversation)}
    />
  );

  return (
    <>
      <div className="px-2.5 pt-1 pb-3">
        {grouped.length === 0 && archived.length === 0 ? (
          <p className="text-muted-foreground m-0 px-3.5 py-6 text-center text-sm">
            No conversations yet.
          </p>
        ) : (
          grouped.map((section) => (
            <section key={section.label}>
              <h3 className="text-muted-foreground mt-3 mb-1 px-1.5 text-[11px] font-medium tracking-wide uppercase">
                {section.label}
              </h3>
              {/* A gap between rows, not padding inside them: the selected
                  row's fill should stay tight to its own text rather than
                  growing to meet its neighbours. */}
              <div className="flex flex-col gap-1">{section.rows.map(renderRow)}</div>
            </section>
          ))
        )}

        {archived.length > 0 ? (
          <section>
            <h3 className="mt-3 mb-1">
              <button
                type="button"
                className="text-muted-foreground hover:text-foreground flex w-full items-center gap-1 rounded-md px-1.5 py-1 text-[11px] font-medium tracking-wide uppercase transition-colors outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
                onClick={() => setArchiveOpen((open) => !open)}
                aria-expanded={archiveOpen}
              >
                <ChevronRight
                  className={cn('size-3 shrink-0 transition-transform', archiveOpen && 'rotate-90')}
                />
                Archived ({archived.length})
              </button>
            </h3>
            {archiveOpen ? (
              <div className="flex flex-col gap-1">{archived.map(renderRow)}</div>
            ) : null}
          </section>
        ) : null}
      </div>

      <ConfirmDialog
        open={pendingDelete !== null}
        title={`Delete "${titleOf(pendingDelete)}"?`}
        body="This cannot be undone."
        confirmLabel="Delete"
        destructive
        onCancel={() => setPendingDelete(null)}
        onConfirm={() => {
          if (pendingDelete) remove(pendingDelete.id);
          setPendingDelete(null);
        }}
      />
    </>
  );
}

function Row({
  conversation,
  current,
  onOpen,
  onDelete,
}: {
  conversation: Conversation;
  current: boolean;
  onOpen(): void;
  onDelete(): void;
}) {
  const update = useStore((state) => state.updateConversation);
  const [renaming, setRenaming] = useState(false);
  const [draft, setDraft] = useState(conversation.title);

  if (renaming) {
    return (
      <div className="flex items-center gap-1 rounded-md py-1 pr-1.5 pl-2.5">
        <Input
          className="h-8 min-w-0 flex-1 text-sm"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              // A rename is explicit, so it is marked custom and the
              // auto-derivation can never stomp it later.
              update(conversation.id, {
                title: draft.trim(),
                hasCustomTitle: draft.trim() !== '',
              });
              setRenaming(false);
            }
            if (event.key === 'Escape') {
              event.stopPropagation();
              setDraft(conversation.title);
              setRenaming(false);
            }
          }}
          onBlur={() => setRenaming(false)}
          aria-label="Conversation name"
          autoFocus
        />
      </div>
    );
  }

  return (
    // The selected row is marked by its fill alone. `bg-accent` is the same
    // token the hover state uses, so it reads as "this one" without
    // introducing a second selection language.
    <div
      className={cn(
        'group hover:bg-accent flex items-center gap-1 rounded-md py-1 pr-1.5 pl-2.5',
        current && 'bg-accent',
      )}
    >
      <button
        type="button"
        className="flex min-w-0 flex-1 py-2 text-left"
        onClick={onOpen}
      >
        <span className="truncate text-sm">{titleOf(conversation)}</span>
      </button>

      {/* Hidden until hover on a pointer device; always shown on touch,
          where there is no hover and they would be unreachable.
          `has-[[data-state=open]]` keeps the pair visible while the menu is
          open — otherwise moving the pointer onto the popup takes it off the
          row, the row un-hovers, and the trigger vanishes from under it. */}
      <div className="flex shrink-0 items-center gap-px opacity-0 transition-opacity duration-150 group-hover:opacity-100 group-focus-within:opacity-100 has-[[data-state=open]]:opacity-100 [@media(hover:none)]:opacity-100">
        {/* Pin stays on the row rather than moving into the menu: it is the
            one action worth a single click, and it is also the only one whose
            state is worth seeing at a glance. Same split as the Mac app's
            sidebar (SidebarView.swift `rowControls`). */}
        <IconAction
          label={conversation.isPinned ? 'Unpin' : 'Pin'}
          active={conversation.isPinned}
          onClick={() => update(conversation.id, { isPinned: !conversation.isPinned })}
        >
          <Pin className={cn(conversation.isPinned && 'fill-current')} />
        </IconAction>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="text-muted-foreground size-[30px] [&_svg:not([class*=size-])]:size-4"
              aria-label="Conversation actions"
              title="More"
              // The row behind this is a click target too; without this the
              // menu opens AND the conversation switches.
              onClick={(event) => event.stopPropagation()}
            >
              <MoreHorizontal />
            </Button>
          </DropdownMenuTrigger>

          <DropdownMenuContent align="end" className="w-44">
            <DropdownMenuItem
              onSelect={() => {
                setDraft(conversation.title);
                setRenaming(true);
              }}
            >
              <Pencil />
              Rename
            </DropdownMenuItem>

            <DropdownMenuSeparator />

            <DropdownMenuItem
              onSelect={() => update(conversation.id, { isPinned: !conversation.isPinned })}
            >
              <Pin />
              {conversation.isPinned ? 'Unpin' : 'Pin'}
            </DropdownMenuItem>
            <DropdownMenuItem
              onSelect={() => update(conversation.id, { isArchived: !conversation.isArchived })}
            >
              <Archive />
              {conversation.isArchived ? 'Unarchive' : 'Archive'}
            </DropdownMenuItem>

            <DropdownMenuSeparator />

            {/* Last, separated, and the only red item — deleting a
                conversation is the one thing here that cannot be undone. */}
            <DropdownMenuItem variant="destructive" onSelect={onDelete}>
              <Trash2 />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
}

function IconAction({
  label,
  active,
  onClick,
  children,
}: {
  label: string;
  active?: boolean;
  onClick(): void;
  children: React.ReactNode;
}) {
  return (
    <Button
      variant="ghost"
      size="icon"
      // 30px: below this a thumb misses between two adjacent actions.
      className={cn(
        'size-[30px] [&_svg:not([class*=size-])]:size-[15px]',
        active ? 'text-foreground' : 'text-muted-foreground',
      )}
      onClick={onClick}
      aria-label={label}
      title={label}
    >
      {children}
    </Button>
  );
}

function titleOf(conversation: Conversation | null): string {
  if (!conversation) return '';
  return conversation.title.trim() === '' ? 'New chat' : conversation.title;
}
