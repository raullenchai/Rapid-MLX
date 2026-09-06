import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { buttonVariants } from '@/components/ui/button';
import { cn } from '@/lib/utils';

/**
 * Replaces `window.confirm`.
 *
 * The native dialog cannot say what the action will cost, and for deletion
 * here that is the point: a subtree spans branches not visible on screen.
 */
export interface ConfirmDialogProps {
  open: boolean;
  title: string;
  body?: string | undefined;
  confirmLabel: string;
  destructive?: boolean;
  onConfirm(): void;
  onCancel(): void;
}

export function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel,
  destructive,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  return (
    <AlertDialog open={open} onOpenChange={(next) => !next && onCancel()}>
      {/* z-50 by default, above the z-20 sheet it can be raised from. */}
      <AlertDialogContent className="sm:max-w-sm">
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          {body ? <AlertDialogDescription>{body}</AlertDialogDescription> : null}
        </AlertDialogHeader>
        <AlertDialogFooter>
          {/* Cancel is focused first: a stray Return must not delete. */}
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            className={cn(destructive && buttonVariants({ variant: 'destructive' }))}
            onClick={onConfirm}
          >
            {confirmLabel}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
