import { Menu, Plus } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';

/**
 * The strip above the transcript.
 *
 * Near-empty on purpose: the rail already carries the wordmark, model
 * selector, New chat and Settings. Below the layout breakpoint the rail is
 * off screen, so the control that opens it lives here.
 */

export interface ChatBarProps {
  title: string;
  /** Null on a wide screen: the rail is already visible. */
  onOpenSidebar: (() => void) | null;
  /** Null on the images surface, which has no conversations. */
  onNewChat: (() => void) | null;
}

export function ChatBar({ title, onOpenSidebar, onNewChat }: ChatBarProps) {
  return (
    <header className="bg-background/85 relative z-2 flex shrink-0 items-center gap-1.5 border-b px-3 pt-[calc(env(safe-area-inset-top)+10px)] pb-2.5 backdrop-blur-xl backdrop-saturate-150">
      {onOpenSidebar ? (
        <Button
          variant="ghost"
          size="icon"
          className="size-8"
          onClick={onOpenSidebar}
          aria-label="Open sidebar"
          title="Conversations"
        >
          <Menu />
        </Button>
      ) : null}

      {/* Centred when flanked by buttons, left-aligned when alone. */}
      <h1
        className={cn(
          'text-muted-foreground m-0 min-w-0 flex-1 truncate text-sm font-medium',
          onOpenSidebar ? 'text-center' : 'text-left',
        )}
      >
        {title}
      </h1>

      {onOpenSidebar && onNewChat ? (
        <Button
          variant="ghost"
          size="icon"
          className="size-8"
          onClick={onNewChat}
          aria-label="New chat"
          title="New chat"
        >
          <Plus />
        </Button>
      ) : null}
    </header>
  );
}
