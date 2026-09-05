import { useState } from 'react';
import { MessageSquareText, Trash2 } from 'lucide-react';
import { useActiveConversation, useStore } from '@/state/store';
import { composeSystemPrompt, normalizeInstruction } from '@/chat/instructions';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { InstructionEditor } from '@/components/common/InstructionEditor';

/**
 * The open conversation's own system prompt, as a composer control.
 *
 * Ported from `rapid-mac`'s `ConversationInstructionsPopover`, and it belongs
 * HERE rather than in Settings for the reason the model picker does: it is a
 * property of the message about to be sent, not a device preference. Settings
 * owns only the global default.
 *
 * Unlike everything else in this app, the editor is a DRAFT with Save and
 * Cancel. Every other surface writes as the user types, but a system prompt
 * silently changing what the next turn does — while they are still composing
 * it — is exactly the case that wants a commit step.
 */
export function ConversationInstructionsButton({ disabled }: { disabled: boolean }) {
  const conversation = useActiveConversation();
  const save = useStore((state) => state.setConversationInstructions);
  const global = useStore((state) => state.settings.system);
  const stored = conversation?.customInstructions ?? '';

  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(stored);

  const active = normalizeInstruction(stored) !== null;

  return (
    <>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={cn('size-8 shrink-0 rounded-full', active ? 'text-primary' : 'text-muted-foreground')}
        disabled={disabled}
        onClick={() => {
          // Seeded at the click, not in an effect: the stored value may have
          // moved since the last visit, and switching conversation must not
          // carry the old draft over.
          setDraft(stored);
          setOpen(true);
        }}
        aria-label="Conversation system prompt"
        title="Conversation system prompt"
      >
        <MessageSquareText />
      </Button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Conversation system prompt</DialogTitle>
            <DialogDescription>
              Sent only with this conversation. Where it conflicts with the global default, this
              one wins.
            </DialogDescription>
          </DialogHeader>

          <InstructionEditor
            id="conversation-system"
            label="Conversation system prompt"
            className="min-h-40"
            value={draft}
            autoFocus
            onChange={setDraft}
            placeholder="For example: This chat is about a Rust codebase; assume I know the borrow checker."
          />

          <EffectiveSystemPrompt global={global} conversation={draft} />

          <DialogFooter>
            <Button
              variant="ghost"
              size="icon"
              className="text-muted-foreground mr-auto size-8"
              disabled={draft.trim() === ''}
              onClick={() => setDraft('')}
              aria-label="Clear conversation system prompt"
              title="Clear"
            >
              <Trash2 />
            </Button>
            <Button variant="ghost" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => {
                save(draft);
                setOpen(false);
              }}
            >
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

/**
 * What will actually be sent, assembled by the same function the wire path
 * uses so the displayed precedence cannot drift from the real one. Collapsed:
 * it answers a question most visits do not ask.
 */
export function EffectiveSystemPrompt({
  global,
  conversation,
}: {
  global: string;
  conversation: string;
}) {
  const prompt = composeSystemPrompt({ global, conversation });

  return (
    <details>
      <summary className="text-muted-foreground w-fit cursor-pointer text-xs">
        Effective system prompt
      </summary>
      <div className="bg-muted/50 mt-1.5 flex flex-col gap-1.5 rounded-md p-3">
        <p className="text-muted-foreground m-0 text-xs">
          Tool guidance is added on top of this once a tool has returned a result.
        </p>
        <pre className="m-0 max-h-40 overflow-auto font-mono text-[11px] whitespace-pre-wrap">
          {prompt === '' ? 'No system message is sent.' : prompt}
        </pre>
      </div>
    </details>
  );
}
