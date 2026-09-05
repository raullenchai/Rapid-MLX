import * as DialogPrimitive from '@radix-ui/react-dialog';
import type { ReactNode } from 'react';
import { X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { DialogOverlay, DialogPortal } from '@/components/ui/dialog';

/**
 * A modal sheet. Bottom sheet on a phone (thumb reach), centred dialog above
 * the 640px breakpoint.
 *
 * Built on the dialog primitives rather than components/ui/sheet.tsx, which
 * only knows the four fixed edges and cannot change side at a breakpoint.
 * Radix supplies the focus trap and Escape handling.
 */

/**
 * The desktop footprint shared by every modal window.
 *
 * Exported because the search palette is built on `CommandDialog`, not on this
 * component, and a literal copied into both is how the two would drift.
 *
 * A DEFINITE size, not a ceiling (`h`/`w`, never `max-h`): with a ceiling each
 * window sized to its own content — Settings 630px, the model picker 259px —
 * so opening one after another made the dialog jump.
 *
 * 960 rather than 760: Settings spends 184 of it on the category rail, which
 * left its panels narrower than the model rows and the two-column settings
 * rows wanted. The `100%-48px` clamp is what keeps this honest on a window
 * narrower than the figure.
 *
 * `sm:`-scoped throughout. On a phone these stay content-sized bottom sheets;
 * a fixed height there would push a two-row picker over most of the screen.
 */
export const SHEET_DESKTOP_SIZE = 'sm:h-[min(80dvh,760px)] sm:w-[min(960px,100%-48px)]';

export interface SheetProps {
  open: boolean;
  onClose(): void;
  title: string;
  /** Rendered in the header, right of the title. */
  actions?: ReactNode;
  children: ReactNode;
}

export function Sheet({ open, onClose, title, actions, children }: SheetProps) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={(next) => !next && onClose()}>
      <DialogPortal>
        <DialogOverlay className="z-20" />
        <DialogPrimitive.Content
          className={cn(
            'bg-background data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 fixed inset-x-0 bottom-0 z-20 flex max-h-[88dvh] flex-col rounded-t-xl border-t shadow-lg duration-200',
            'data-[state=closed]:slide-out-to-bottom data-[state=open]:slide-in-from-bottom',
            'sm:data-[state=closed]:zoom-out-95 sm:data-[state=open]:zoom-in-95 sm:data-[state=closed]:slide-out-to-bottom-0 sm:data-[state=open]:slide-in-from-bottom-0',
            SHEET_DESKTOP_SIZE,
            'sm:inset-auto sm:top-1/2 sm:left-1/2 sm:-translate-x-1/2 sm:-translate-y-1/2 sm:rounded-xl sm:border',
          )}
          aria-label={title}
        >
          <header className="flex shrink-0 items-center gap-2 border-b py-3 pr-3 pl-4">
            <DialogPrimitive.Title className="m-0 min-w-0 flex-1 text-base leading-none font-semibold">
              {title}
            </DialogPrimitive.Title>
            <div className="flex shrink-0 items-center gap-1">
              {actions}
              {/* An icon, matching the sidebar's own close control, rather
                  than the word "Done". "Done" implies the sheet is committing
                  something — but every sheet here writes as the user acts, so
                  there is nothing to confirm and no way to cancel. The label
                  stays in `aria-label`/`title`, which is also what keeps
                  `getByRole('button', { name: 'Close' })` working. */}
              <DialogPrimitive.Close asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-muted-foreground size-8"
                  aria-label="Close"
                  title="Close"
                >
                  <X />
                </Button>
              </DialogPrimitive.Close>
            </div>
          </header>
          <div className="min-h-0 flex-1 overflow-y-auto pb-[env(safe-area-inset-bottom)]">
            {children}
          </div>
        </DialogPrimitive.Content>
      </DialogPortal>
    </DialogPrimitive.Root>
  );
}
