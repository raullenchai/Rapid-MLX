import { MAX_INSTRUCTION_LENGTH } from '@/chat/instructions';
import { cn } from '@/lib/utils';
import { Textarea } from '@/components/ui/textarea';

/**
 * The shared editor for both instruction layers, ported from `rapid-mac`'s
 * `InstructionTextEditor`: one field, its own surface, and a live count
 * against the cap so a truncation is never a surprise.
 */
export function InstructionEditor({
  id,
  label,
  value,
  placeholder,
  className,
  autoFocus,
  onChange,
}: {
  id: string;
  label: string;
  value: string;
  placeholder: string;
  className?: string;
  autoFocus?: boolean;
  onChange(next: string): void;
}) {
  return (
    <div className="relative">
      <Textarea
        id={id}
        aria-label={label}
        // `pb-8` reserves the counter's line, so a prompt long enough to fill
        // the box scrolls under it rather than behind it.
        className={cn('resize-none pb-8', className)}
        value={value}
        autoFocus={autoFocus}
        maxLength={MAX_INSTRUCTION_LENGTH}
        onChange={(event) => onChange(event.target.value.slice(0, MAX_INSTRUCTION_LENGTH))}
        placeholder={placeholder}
      />
      <span className="text-muted-foreground pointer-events-none absolute right-3 bottom-2 text-xs">
        {value.length} of {MAX_INSTRUCTION_LENGTH} characters
      </span>
    </div>
  );
}
