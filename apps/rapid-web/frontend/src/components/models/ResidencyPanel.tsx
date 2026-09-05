import { useEffect, useState } from 'react';
import { Cpu, Lock } from 'lucide-react';
import { fetchResidency } from '@/api/models';
import type { ResidencySnapshot, ResidentModel } from '@/api/types';
import { formatBytes } from '@/lib/format';
import { useStore } from '@/state/store';

/**
 * What the engine is holding in memory, at the foot of the sidebar.
 *
 * Ported from `rapid-mac`'s `SidebarView.residencyFooter`. Read-only: the
 * lock marks the pinned primary, it is not a control. The engine does expose
 * pin/unpin and unload, but a phone is the wrong place to evict the model
 * whoever is at the Mac is generating with.
 */

/** How often the snapshot is re-read. Matches the Mac app's loop. */
const POLL_MS = 5000;

/** Consecutive failures before the panel gives up on a dead server. */
const MAX_FAILURES = 5;

/**
 * The bytes to show for one model.
 *
 * A lazy engine (mflux) faults its weight pages in on the FIRST request, so
 * the load-time delta can read far smaller than the reservation admission
 * charged. Never present that partial delta as the model's size.
 */
export function displayBytes(model: ResidentModel): number {
  return Math.max(model.estimated_bytes, model.measured_bytes ?? 0);
}

/**
 * The name to show for one model.
 *
 * Startup entries are keyed by the resolved HF repo and keep the catalog
 * alias alongside, so the shortest alias is the one that matches what the
 * picker calls it. `served` wins when it names this model, so the row and
 * the composer agree.
 */
export function displayName(model: ResidentModel, served: string | null): string {
  if (served !== null && (model.id === served || model.aliases.includes(served))) {
    return served;
  }
  const shortest = [...model.aliases].sort(
    (left, right) => left.length - right.length || left.localeCompare(right),
  )[0];
  return shortest ?? model.id;
}

/** "9.1 GB / 25 GB", or just the numerator when the engine has no ceiling. */
export function memorySummary(snapshot: ResidencySnapshot): string {
  const used = formatBytes(snapshot.memory_used_bytes) ?? '0 B';
  const limit = formatBytes(snapshot.memory_limit_bytes);
  return limit === null ? used : `${used} / ${limit}`;
}

export function ResidencyPanel() {
  const snapshot = useResidency();
  const served = useStore((state) => state.status?.model ?? null);

  if (snapshot === null || snapshot.models.length === 0) return null;

  const fraction =
    snapshot.memory_limit_bytes > 0
      ? Math.min(1, snapshot.memory_used_bytes / snapshot.memory_limit_bytes)
      : null;

  return (
    // A rule, not a gap: this block describes the MACHINE rather than the
    // app's navigation, and whitespace alone left it reading as another
    // oddly-formatted nav group.
    <div
      className="border-sidebar-border shrink-0 border-t px-3 py-2.5"
      aria-label="Resident models"
    >
      <div className="text-muted-foreground flex items-center gap-1.5 text-xs">
        <Cpu className="size-3.5 shrink-0" />
        <span className="flex-1">Resident</span>
        <span className="font-mono text-[11px]">{memorySummary(snapshot)}</span>
      </div>

      {fraction !== null ? (
        <div className="bg-background mt-1.5 h-1.5 overflow-hidden rounded-full">
          <div
            className="bg-primary h-full transition-[width] duration-500"
            style={{ width: `${Math.round(fraction * 100)}%` }}
          />
        </div>
      ) : null}

      <ul className="m-0 mt-1.5 flex list-none flex-col gap-1 p-0">
        {/* Capped: the rail is 260px of vertical space shared with the
            conversation list, and a residency block that grows without a
            bound pushes the history off the screen. */}
        {snapshot.models.slice(0, 4).map((model) => (
          <li key={model.id} className="flex items-center gap-1.5 text-xs">
            {model.pinned ? (
              <Lock className="text-primary size-3 shrink-0" aria-label="Pinned" />
            ) : (
              <span className="bg-muted-foreground size-1.5 shrink-0 rounded-full" />
            )}
            <span className="min-w-0 flex-1 truncate">{displayName(model, served)}</span>
            <span className="text-muted-foreground font-mono text-[11px]">
              {formatBytes(displayBytes(model)) ?? '—'}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * Poll the residency snapshot for as long as this panel is mounted.
 *
 * Mounting is the gate: on a phone the sidebar is a drawer, so a closed
 * drawer polls nothing. Re-scheduled AFTER each response so a slow request
 * cannot stack overlapping polls, same rule as the download feed.
 */
function useResidency(): ResidencySnapshot | null {
  const [snapshot, setSnapshot] = useState<ResidencySnapshot | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;

    const tick = async () => {
      let again = true;
      try {
        const next = await fetchResidency(controller.signal);
        if (controller.signal.aborted) return;
        failures = 0;
        setSnapshot(next);
      } catch {
        failures += 1;
        again = failures < MAX_FAILURES;
      }
      if (again && !controller.signal.aborted) timer = setTimeout(() => void tick(), POLL_MS);
    };

    void tick();

    return () => {
      controller.abort();
      if (timer !== undefined) clearTimeout(timer);
    };
  }, []);

  return snapshot;
}
