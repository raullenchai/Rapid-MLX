import {
  accessibilityLabel,
  actionIsRenderable,
  actionTitle,
  detail,
  headline,
  isWorking,
  progressFraction,
  readinessAction,
  statusRole,
  type ModelReadiness,
  type ReadinessAction,
} from '@/readiness/ModelReadiness';
import { cn } from '@/lib/utils';
import { READING_COLUMN } from '@/lib/layout';
import { Button } from '@/components/ui/button';
import { StatusDot } from '@/components/common/StatusDot';

/**
 * The readiness surface above the composer.
 *
 * Every string comes from ModelReadiness — this component invents no copy of
 * its own.
 */

export interface LifecycleBandProps {
  readiness: ModelReadiness;
  /** Increments when a gated send is attempted. Flashes the band. */
  attentionToken: number;
  onAction(action: ReadinessAction): void;
}

export function LifecycleBand({ readiness, attentionToken, onAction }: LifecycleBandProps) {
  // Ready is the quiet state: nothing to say, nothing to do, no chrome.
  if (readiness.kind === 'ready') return null;

  const action = readinessAction(readiness);
  const fraction = progressFraction(readiness);
  const working = isWorking(readiness);
  const role = statusRole(readiness);
  const detailText = detail(readiness);

  return (
    // The padding matches the composer's footer, so the band's edges line up
    // with the input box below it rather than sitting 12px wider.
    <div className="mb-2 px-3">
      <div
        className={cn(
          READING_COLUMN,
          'bg-card text-card-foreground animate-in fade-in-0 zoom-in-95 relative flex items-center gap-2.5 overflow-hidden rounded-lg border py-2.5 pr-2.5 pl-3 text-sm shadow-xs',
          role === 'error' && 'border-destructive/40',
        )}
        // Re-keying restarts the entrance animation, which is what makes a
        // SECOND blocked send visible.
        key={attentionToken}
        role="status"
        aria-label={accessibilityLabel(readiness)}
      >
        <StatusDot role={role} pulse={working} />

        <div className="flex min-w-0 flex-1 flex-col gap-px">
          {/* Not truncated: the model name lives here. */}
          <span className="font-medium [overflow-wrap:anywhere]">{headline(readiness)}</span>
          {/* Dropped first on a narrow screen — the composer placeholder already
              paraphrases it, whereas dropping the headline loses the model. */}
          {detailText ? (
            <span className="text-muted-foreground truncate text-xs max-[380px]:hidden">
              {detailText}
            </span>
          ) : null}
        </div>

        {action && actionIsRenderable(action) ? (
          <Button size="sm" onClick={() => onAction(action)}>
            {actionTitle(action)}
          </Button>
        ) : null}

        {/* Determinate ONLY when a real fraction exists: an indeterminate bar
            would imply a precision the byte monitor does not have. */}
        {fraction !== null ? (
          <div className="bg-muted absolute inset-x-0 bottom-0 h-0.5">
            <div
              className="bg-primary h-full transition-[width] duration-300"
              style={{ width: percent(fraction) }}
            />
          </div>
        ) : null}
      </div>
    </div>
  );
}

/**
 * Clamp BEFORE rounding: the byte monitor overshoots on the final chunk, so
 * an unclamped fraction renders as "101%".
 */
export function percent(fraction: number): string {
  const clamped = Math.min(1, Math.max(0, fraction));
  return `${Math.round(clamped * 100)}%`;
}
