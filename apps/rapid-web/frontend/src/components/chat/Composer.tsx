import { useEffect, useRef, useState, type ReactNode } from 'react';
import { ArrowUp, Square } from 'lucide-react';
import { READING_COLUMN } from '@/lib/layout';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { ConversationInstructionsButton } from './ConversationInstructions';

export interface ComposerProps {
  placeholder: string;
  sendTooltip: string;
  canSend: boolean;
  streaming: boolean;
  onSend(text: string): void;
  onStop(): void;
  /** Called when Return is pressed while sending is gated. */
  onBlocked(): void;
  /** The model picker, rendered on the control row beside Send. */
  picker?: ReactNode;
}

export function Composer({
  placeholder,
  sendTooltip,
  canSend,
  streaming,
  onSend,
  onStop,
  onBlocked,
  picker,
}: ComposerProps) {
  const [draft, setDraft] = useState('');
  const field = useRef<HTMLTextAreaElement>(null);

  // Recomputed rather than tracked: a paste can add many lines at once.
  useEffect(() => {
    const element = field.current;
    if (!element) return;
    element.style.height = 'auto';
    element.style.height = `${Math.min(element.scrollHeight, window.innerHeight * 0.34)}px`;
  }, [draft]);

  const submit = () => {
    const text = draft.trim();
    if (text === '') return;

    if (!canSend) {
      // The draft is NOT consumed — clearing the field would throw away what
      // the user typed for a condition they cannot see.
      onBlocked();
      return;
    }

    setDraft('');
    onSend(text);
  };

  const idle = !streaming && draft.trim() === '';


  return (
    // Transparent: no fill, no top border, no backdrop blur. Each of those
    // draws a full-width rectangle behind the input, which is the band this
    // used to show. The input box carries its own opaque background, so it
    // stays readable over whatever scrolls beneath it.
    <footer className="relative z-2 shrink-0 px-3 pt-2 pb-[calc(env(safe-area-inset-bottom)+10px)]">
      <div
        className={cn(
          READING_COLUMN,
          'bg-background focus-within:border-ring flex flex-col rounded-xl border px-3.5 pt-2.5 pb-2 shadow-xs transition-[color,box-shadow]',
        )}
      >
        <textarea
          ref={field}
          className="placeholder:text-muted-foreground max-h-[34vh] w-full resize-none border-none bg-transparent leading-[1.45] outline-none"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key !== 'Enter') return;

            // An IME is mid-composition: this Return is committing a
            // candidate, not ending the message. Sending here truncates
            // whatever was being composed — for Chinese, Japanese and Korean
            // input that is every single sentence. `isComposing` is the
            // standard signal; keyCode 229 is the older one some IMEs still
            // report, and both are cheap to check.
            if (event.nativeEvent.isComposing || event.keyCode === 229) return;

            // Shift+Return is the newline everywhere, which is what makes
            // plain Return safe to bind to send.
            if (event.shiftKey) return;

            // On touch there is no Shift, so a bare Return has to stay a
            // newline or a phone loses the only way to type one. The send
            // button is already under the thumb there.
            if (!sendsOnEnter() && !event.metaKey && !event.ctrlKey) return;

            event.preventDefault();
            submit();
          }}
          enterKeyHint={sendsOnEnter() ? 'send' : 'enter'}
          placeholder={placeholder}
          rows={1}
          autoCapitalize="sentences"
          aria-label="Message"
        />

        {/* The control row. The model and the conversation's system prompt
            belong here rather than in the sidebar or Settings: both are
            properties of the message about to be sent. */}
        <div className="mt-1 flex items-center gap-1">
          <ConversationInstructionsButton disabled={streaming} />
          <span className="min-w-0 flex-1" />
          {picker}
          <Button
            type="button"
            size="icon"
            variant={streaming ? 'secondary' : 'default'}
            className={cn(
              'size-8 shrink-0 rounded-full',
              // An outline, never a dead grey fill: "nothing to send yet" must
              // not read as "broken".
              'disabled:border disabled:bg-transparent disabled:text-muted-foreground disabled:opacity-100',
            )}
            onClick={streaming ? onStop : submit}
            // Never disabled while gated: a disabled button cannot explain
            // itself. Disabled only with nothing to send.
            disabled={idle}
            title={streaming ? 'Stop' : sendTooltip}
            aria-label={streaming ? 'Stop generating' : 'Send'}
          >
            {streaming ? <Square className="fill-current" /> : <ArrowUp />}
          </Button>
        </div>
      </div>
    </footer>
  );
}

/**
 * Does a bare Return send, or insert a newline?
 *
 * It sends wherever there is a physical keyboard, since Shift+Return is then
 * available for the newline. On touch there is no Shift, so Return must stay a
 * newline or the phone loses the only way to type one.
 *
 * Keyed on `(hover: none) and (pointer: coarse)`, not a width query: a narrow
 * desktop window still has a keyboard, and an iPad without one does not.
 */
function sendsOnEnter(): boolean {
  if (typeof window === 'undefined' || !window.matchMedia) return true;
  return !window.matchMedia('(hover: none) and (pointer: coarse)').matches;
}
