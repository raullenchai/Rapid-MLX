import type { StatusRole } from '@/readiness/ModelReadiness';
import { cn } from '@/lib/utils';

const ROLE_FILL: Record<StatusRole, string> = {
  idle: 'bg-muted-foreground',
  ready: 'bg-success',
  working: 'bg-warning',
  error: 'bg-destructive',
};

/** The engine status dot. Pulses only while something is actually happening. */
export function StatusDot({
  role,
  pulse,
  className,
}: {
  role: StatusRole;
  pulse?: boolean;
  className?: string;
}) {
  return (
    <span
      className={cn(
        'size-2 shrink-0 rounded-full',
        ROLE_FILL[role],
        pulse && 'animate-[status-pulse_1.4s_ease-in-out_infinite]',
        className,
      )}
      aria-hidden="true"
    />
  );
}
