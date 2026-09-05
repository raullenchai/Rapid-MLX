import { cn } from '@/lib/utils';

/**
 * A settings toggle.
 *
 * `role="switch"` on a button rather than a checkbox input: nothing here
 * submits, and a native checkbox cannot be styled into this shape without
 * being replaced by exactly this markup anyway.
 *
 * `label` becomes the accessible name, so an icon-free row still announces
 * what it controls.
 */
export function Switch({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange(next: boolean): void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={cn(
        'focus-visible:ring-ring inline-flex h-6 w-10 shrink-0 items-center rounded-full border transition-colors focus-visible:ring-2 focus-visible:outline-none',
        checked ? 'bg-primary border-primary' : 'bg-input border-input',
      )}
    >
      <span
        className={cn(
          'bg-background block size-4 rounded-full shadow-sm transition-transform',
          checked ? 'translate-x-[1.15rem]' : 'translate-x-[0.15rem]',
        )}
      />
    </button>
  );
}
