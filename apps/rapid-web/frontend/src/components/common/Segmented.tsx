import * as RadioGroup from '@radix-ui/react-radio-group';
import { cn } from '@/lib/utils';

/**
 * A radio group styled as shadcn's TabsList.
 *
 * Radios rather than Radix Tabs or ToggleGroup: arrow keys move between
 * options and a screen reader announces "2 of 3", which neither of the other
 * two gives for a setting that is a choice rather than a view.
 */
export function Segmented<T extends string>({
  label,
  value,
  options,
  className,
  onChange,
}: {
  label: string;
  value: T;
  options: Array<{ value: T; label: string }>;
  className?: string;
  onChange(value: T): void;
}) {
  return (
    <RadioGroup.Root
      className={cn(
        'bg-muted text-muted-foreground inline-flex h-9 w-fit items-center justify-center rounded-lg p-[3px]',
        className,
      )}
      value={value}
      onValueChange={(next) => onChange(next as T)}
      aria-label={label}
    >
      {options.map((option) => (
        <RadioGroup.Item
          key={option.value}
          value={option.value}
          className="data-[state=checked]:bg-background dark:data-[state=checked]:text-foreground focus-visible:border-ring focus-visible:ring-ring/50 dark:data-[state=checked]:border-input dark:data-[state=checked]:bg-input/30 text-foreground inline-flex h-[calc(100%-1px)] flex-1 items-center justify-center gap-1.5 rounded-md border border-transparent px-2 py-1 text-sm font-medium whitespace-nowrap transition-[color,box-shadow] focus-visible:ring-[3px] focus-visible:outline-1 data-[state=checked]:shadow-sm"
        >
          {option.label}
        </RadioGroup.Item>
      ))}
    </RadioGroup.Root>
  );
}
