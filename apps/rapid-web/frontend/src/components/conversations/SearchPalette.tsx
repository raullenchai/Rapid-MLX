import { useEffect, useMemo, useState } from 'react';
import { Archive, MessageSquare, Pin, SquarePen } from 'lucide-react';
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command';
import { groupConversations, searchConversations } from '@/chat/ConversationSearch';
import { formatRelativeTime } from '@/lib/format';
import { SHEET_DESKTOP_SIZE } from '@/components/common/Sheet';
import { cn } from '@/lib/utils';
import { useStore } from '@/state/store';
import { useMidnightTick } from '@/lib/useMidnightTick';

/**
 * Global conversation search, opened with the sidebar's magnifier or ⌘K.
 *
 * Replaces an always-on search field and an Archived toggle — two permanent
 * controls in a 260px rail, one of which was a mode that silently hid active
 * conversations.
 *
 * Two behaviours are why the toggle is not needed: archived conversations are
 * **included** (marked with a box icon; search is their recovery path), and
 * **every branch is searched**, not just the visible path.
 */
export function SearchPalette({
  open,
  onOpenChange,
  onNewChat,
}: {
  open: boolean;
  onOpenChange(open: boolean): void;
  onNewChat(): void;
}) {
  const conversations = useStore((state) => state.conversations);
  const setActive = useStore((state) => state.setActiveConversation);
  const [query, setQuery] = useState('');
  const now = useMidnightTick();

  const sections = useMemo(
    () => groupConversations(searchConversations(conversations, query), now),
    [conversations, query, now],
  );

  return (
    <CommandDialog
      open={open}
      // Cleared here rather than in an effect on `open`: a stale query would
      // make the palette reopen onto the previous search's results, and doing
      // it in the handler avoids a second render pass to undo the first.
      onOpenChange={(next) => {
        if (!next) setQuery('');
        onOpenChange(next);
      }}
      title="Search conversations"
      description="Find a conversation by title or by anything said in it."
      // The same footprint as the Model and Settings sheets. This one is built
      // on `CommandDialog` rather than on `Sheet`, so it has to be told — the
      // shared constant is what stops the two drifting apart.
      //
      // `flex flex-col` because `DialogContent` is a `grid` by default, and a
      // grid child does not inherit the fixed height: the list would keep its
      // own 420px cap and leave the rest of the box empty. `p-0` overrides the
      // primitive's `p-6`, which would otherwise inset the search field.
      className={cn(SHEET_DESKTOP_SIZE, 'flex flex-col gap-0 overflow-hidden p-0 sm:max-w-none')}
      showCloseButton={false}
      // cmdk filters by item text by default. Ours is already filtered — and
      // by message bodies, which are not in the item — so its filter would
      // throw away rows that matched on content.
      shouldFilter={false}
    >
      <CommandInput
        placeholder="Search conversations"
        value={query}
        onValueChange={setQuery}
      />
      {/* Fills the dialog rather than capping at its own height: the box is a
          fixed size now, so a 420px cap would leave dead space below the last
          result. `min-h-0` is what lets a flex child actually scroll. */}
      <CommandList className="max-h-none min-h-0 flex-1">
        <CommandEmpty>
          {conversations.length === 0 ? 'No conversations yet.' : 'Nothing matches.'}
        </CommandEmpty>

        {/* Offered first, and only with no query: a blank palette is often
            opened to start something rather than to find something. */}
        {query.trim() === '' ? (
          <CommandGroup>
            <CommandItem
              value="__new-chat"
              onSelect={() => {
                onNewChat();
                onOpenChange(false);
              }}
            >
              <SquarePen />
              New chat
            </CommandItem>
          </CommandGroup>
        ) : null}

        {sections.map((section) => (
          <CommandGroup key={section.label} heading={section.label}>
            {section.rows.map((conversation) => (
              <CommandItem
                key={conversation.id}
                value={conversation.id}
                onSelect={() => {
                  setActive(conversation.id);
                  onOpenChange(false);
                }}
              >
                {conversation.isArchived ? <Archive /> : <MessageSquare />}
                <span className="min-w-0 flex-1 truncate">
                  {conversation.title.trim() === '' ? 'New chat' : conversation.title}
                </span>
                {conversation.isPinned ? (
                  <Pin className="size-3 shrink-0" aria-label="Pinned" />
                ) : null}
                <span className="text-muted-foreground shrink-0 text-xs">
                  {formatRelativeTime(conversation.updatedAt, now)}
                </span>
              </CommandItem>
            ))}
          </CommandGroup>
        ))}
      </CommandList>
    </CommandDialog>
  );
}

/** Bind ⌘K / Ctrl+K to open the palette. */
export function useSearchShortcut(onOpen: () => void) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'k' || !(event.metaKey || event.ctrlKey)) return;
      event.preventDefault();
      onOpen();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onOpen]);
}
