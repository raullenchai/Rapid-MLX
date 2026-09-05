import * as DialogPrimitive from '@radix-ui/react-dialog';
import { useEffect, useState, type ReactNode } from 'react';
import {
  AudioLines,
  Image as ImageIcon,
  PanelLeft,
  Search,
  Settings as SettingsIcon,
  SquarePen,
  X,
} from 'lucide-react';
import { ConversationList } from './ConversationList';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { DialogOverlay, DialogPortal } from '@/components/ui/dialog';
import { ResidencyPanel } from '@/components/models/ResidencyPanel';
import { Wordmark } from '@/components/common/Wordmark';

/**
 * The persistent left rail and its narrow-screen counterpart.
 *
 * They are separate elements, not one repositioned: above the breakpoint the
 * rail sits BESIDE the transcript and must be a landmark (no focus trap, no
 * Escape, content beside it reachable). Below it, the drawer overlays the
 * transcript and must be a real modal.
 */

export type Surface = 'chat' | 'images' | 'audio';

export interface SidebarProps {
  onNewChat(): void;
  onOpenSettings(): void;
  onSearch(): void;
  surface: Surface;
  onSelectSurface(next: Surface): void;
  collapsed: boolean;
  onToggleCollapsed(): void;
}

export function Sidebar({
  onNewChat,
  onOpenSettings,
  onSearch,
  surface,
  onSelectSurface,
  collapsed,
  onToggleCollapsed,
}: SidebarProps) {
  return (
    <aside
      className={cn(
        'bg-sidebar text-sidebar-foreground border-sidebar-border flex h-dvh shrink-0 flex-col overflow-hidden border-r transition-[width] duration-200',
        collapsed ? 'w-0 border-r-0' : 'w-[260px]',
      )}
      // A landmark, not a dialog: a screen reader user can jump here and back
      // out without anything being trapped.
      aria-label="Conversations"
    >
      <SidebarTop
        onDismiss={onToggleCollapsed}
        dismissLabel="Collapse sidebar"
        icon={<PanelLeft />}
        onSearch={onSearch}
      />
      <SurfaceNav
        surface={surface}
        onNewChat={onNewChat}
        onSelectSurface={onSelectSurface}
      />
      <ListRegion>
        {surface === 'chat' ? <ConversationList /> : <SurfaceNote surface={surface} />}
      </ListRegion>
      <SidebarFooter onOpenSettings={onOpenSettings} />
    </aside>
  );
}

export function SidebarDrawer({
  open,
  onClose,
  onNewChat,
  onOpenSettings,
  onSearch,
  surface,
  onSelectSurface,
}: {
  open: boolean;
  onClose(): void;
  onNewChat(): void;
  onOpenSettings(): void;
  onSearch(): void;
  surface: Surface;
  onSelectSurface(next: Surface): void;
}) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={(next) => !next && onClose()}>
      <DialogPortal>
        <DialogOverlay className="z-20" />
        <DialogPrimitive.Content
          className="bg-sidebar text-sidebar-foreground border-sidebar-border data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:slide-out-to-left data-[state=open]:slide-in-from-left fixed inset-y-0 left-0 z-20 flex h-dvh w-[min(300px,86vw)] flex-col border-r shadow-lg transition ease-in-out data-[state=closed]:duration-300 data-[state=open]:duration-500"
          aria-label="Conversations"
        >
          <DialogPrimitive.Title className="sr-only">Conversations</DialogPrimitive.Title>
          <SidebarTop
            onDismiss={onClose}
            dismissLabel="Close sidebar"
            icon={<X />}
            // Sequenced, not simultaneous. Both are Radix modals, and
            // dismissing this one in the same tick as opening the palette
            // makes the drawer's exit reclaim focus and close it again —
            // the palette flashes and vanishes. Waiting for the close
            // animation to finish lets the palette take focus cleanly.
            onSearch={() => {
              onClose();
              setTimeout(onSearch, 320);
            }}
          />
          <SurfaceNav
            surface={surface}
            onNewChat={() => {
              onNewChat();
              onClose();
            }}
            onSelectSurface={(next) => {
              onSelectSurface(next);
              onClose();
            }}
          />
          <ListRegion>
            {surface === 'chat' ? (
              <ConversationList onNavigate={onClose} />
            ) : (
              <SurfaceNote surface={surface} />
            )}
          </ListRegion>
          <SidebarFooter
            onOpenSettings={() => {
              onOpenSettings();
              onClose();
            }}
          />
        </DialogPrimitive.Content>
      </DialogPortal>
    </DialogPrimitive.Root>
  );
}

/**
 * The rail's primary navigation: one row per destination.
 *
 * "New Chat" is both the chat destination and the action, as its label says —
 * returning to an EXISTING conversation is what the list below is for, so a
 * separate "Chat" row would be a second way to do nothing.
 */
function SurfaceNav({
  surface,
  onNewChat,
  onSelectSurface,
}: {
  surface: Surface;
  onNewChat(): void;
  onSelectSurface(next: Surface): void;
}) {
  return (
    <nav className="flex shrink-0 flex-col gap-0.5 px-3 pb-2" aria-label="Views">
      <NavRow icon={<SquarePen />} label="New Chat" onClick={onNewChat} />
      <NavRow
        icon={<ImageIcon />}
        label="Images"
        current={surface === 'images'}
        onClick={() => onSelectSurface('images')}
      />
      <NavRow
        icon={<AudioLines />}
        label="Audio"
        current={surface === 'audio'}
        onClick={() => onSelectSurface('audio')}
      />
    </nav>
  );
}

function NavRow({
  icon,
  label,
  current,
  disabled,
  onClick,
}: {
  icon: ReactNode;
  label: string;
  current?: boolean;
  disabled?: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      // `aria-current="page"`, not `aria-selected`: this is navigation, not a
      // tablist/listbox role the component does not claim.
      aria-current={current ? 'page' : undefined}
      disabled={disabled}
      title={disabled ? `${label} is not available yet` : undefined}
      className={cn(
        'flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-sm font-medium transition-colors outline-none',
        'focus-visible:ring-ring/50 focus-visible:ring-[3px]',
        'disabled:pointer-events-none disabled:opacity-40',
        current
          ? 'bg-background text-foreground shadow-xs'
          : 'text-foreground hover:bg-accent',
        '[&_svg]:text-muted-foreground [&_svg]:size-4 [&_svg]:shrink-0',
      )}
      onClick={onClick}
    >
      {icon}
      {label}
    </button>
  );
}

/** Neither non-chat surface keeps a history, so the list area explains why. */
function SurfaceNote({ surface }: { surface: Surface }) {
  return (
    <p className="text-muted-foreground m-0 px-4 py-3 text-xs leading-relaxed">
      {surface === 'images'
        ? 'Images are not saved between visits. Use Save to keep one.'
        : 'Speech runs on whichever model is loaded — no separate model to start.'}
    </p>
  );
}

function SidebarTop({
  onDismiss,
  dismissLabel,
  icon,
  onSearch,
}: {
  onDismiss(): void;
  dismissLabel: string;
  icon: ReactNode;
  onSearch(): void;
}) {
  return (
    <div className="flex shrink-0 items-center gap-1 pt-[calc(env(safe-area-inset-top)+12px)] pr-2.5 pb-2 pl-4">
      <Wordmark className="min-w-0 flex-1 text-lg" />
      {/* Search lives here rather than as a field in the list: a permanent
          input costs a row of a 260px rail for something used occasionally,
          and the Archived toggle beside it was a mode the list got stuck in.
          Same placement as the Mac app's toolbar magnifier. */}
      <Button
        variant="ghost"
        size="icon"
        className="size-8"
        onClick={onSearch}
        aria-label="Search conversations"
        title="Search conversations — ⌘K"
      >
        <Search />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        className="size-8"
        onClick={onDismiss}
        aria-label={dismissLabel}
        title={dismissLabel}
      >
        {icon}
      </Button>
    </div>
  );
}

/** The only thing that scrolls, so the header and footer stay put. */
function ListRegion({ children }: { children: ReactNode }) {
  return <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>;
}

function SidebarFooter({ onOpenSettings }: { onOpenSettings(): void }) {
  return (
    <>
      <ResidencyPanel />
      <div className="border-sidebar-border shrink-0 border-t px-3 pt-2 pb-[calc(env(safe-area-inset-bottom)+10px)]">
        <Button
          variant="ghost"
          className="text-muted-foreground hover:text-foreground w-full justify-start"
          onClick={onOpenSettings}
        >
          <SettingsIcon />
          Settings
        </Button>
      </div>
    </>
  );
}

/**
 * Is the viewport wide enough for the rail to sit beside the transcript?
 *
 * 900px: the rail costs 260px and the transcript's reading measure is 720px.
 * `matchMedia` rather than a resize listener, so it fires once per crossing.
 */
export function useWideLayout(): boolean {
  const [wide, setWide] = useState(
    () => typeof window !== 'undefined' && window.matchMedia('(min-width: 900px)').matches,
  );

  useEffect(() => {
    const query = window.matchMedia('(min-width: 900px)');
    const onChange = (event: MediaQueryListEvent) => setWide(event.matches);
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, []);

  return wide;
}
