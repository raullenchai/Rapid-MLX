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
import { Button } from '@/components/ui/button';
import { useStore } from '@/state/store';

/**
 * Approve one action the model chose for itself.
 *
 * Two shapes, one dialog. A `browse` is approved by HOST — the model picks
 * the URL, so an unapproved fetch is an exfiltration primitive no SSRF check
 * can close, and the exact address is the whole defence. A connector call is
 * approved by TOOL, and names the server: "run read_file?" is unanswerable
 * without knowing whose.
 *
 * Only the connector prompt offers "Always allow", and only per tool. A
 * browse grant covers the host for the rest of the answer and no longer,
 * because the next answer's URLs are chosen by the model too.
 */
export function ApprovalDialog() {
  const pending = useStore((state) => state.pendingApproval);
  const answer = useStore((state) => state.answerApproval);

  const isTool = pending?.kind === 'tool';

  return (
    <AlertDialog open={pending !== null} onOpenChange={(next) => !next && answer('declined')}>
      <AlertDialogContent className="sm:max-w-md">
        <AlertDialogHeader>
          <AlertDialogTitle>
            {pending?.kind === 'tool'
              ? `Run ${pending.short}?`
              : `Let the model fetch ${pending?.host}?`}
          </AlertDialogTitle>
          <AlertDialogDescription>
            {pending?.kind === 'tool'
              ? `The model asked to run this tool from the “${pending.server}” connector — a program on the Mac serving this page.`
              : 'The model asked to read this page. Approving also covers later pages on the same host for this answer.'}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <div className="bg-muted/50 max-h-32 overflow-auto rounded-md border px-3 py-2 font-mono text-xs [overflow-wrap:anywhere] whitespace-pre-wrap">
          {pending?.kind === 'tool' ? pending.args : pending?.url}
        </div>
        <AlertDialogFooter>
          {/* Declining is focused first: a stray Return must not run a
              program, or send a request to a host the user has not read. */}
          <AlertDialogCancel onClick={() => answer('declined')}>Don't allow</AlertDialogCancel>
          {isTool ? (
            <Button variant="outline" onClick={() => answer('always')}>
              Always allow
            </Button>
          ) : null}
          <AlertDialogAction onClick={() => answer('allowed')}>
            {isTool ? 'Allow once' : 'Allow'}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
