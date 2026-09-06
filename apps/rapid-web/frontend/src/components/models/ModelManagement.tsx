import { useEffect, useMemo, useState } from 'react';
import { ArrowUpDown, Check, HardDrive, Search, Trash2, X } from 'lucide-react';
import {
  cancelDownload,
  fetchDownload,
  fetchModels,
  pullModel,
  removeModel,
} from '@/api/models';
import { asApiError } from '@/api/errors';
import type { DownloadJob, ModelEntry, ModelKind } from '@/api/types';
import { formatBytes } from '@/lib/format';
import { useStore } from '@/state/store';
import { startModel } from '@/state/startModel';
import { noticeFor } from '@/state/notices';
import { percent } from '@/components/models/LifecycleBand';
import {
  aggregateOnDiskBytes,
  diskUsageFooter,
  entryBytes,
  filterEntries,
  largestCachedEntry,
  listHeading,
  noMatchesCopy,
  sortEntries,
  storageSummary,
  FILTER_LABELS,
  SORT_LABELS,
  type FilterMode,
  type SortOrder,
} from '@/components/models/modelCache';
import { ConfirmDialog } from '@/components/common/ConfirmDialog';
import { Segmented } from '@/components/common/Segmented';
import { PageHeader, SettingsRowDivider, SettingsSection } from '@/components/common/SettingsSection';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

/**
 * Model management, as the Settings window's first panel.
 *
 * Ported from `rapid-mac`'s `SettingsModelManagementPanel`, same layout top to
 * bottom: a Disk overview card, the capability tabs, the search + filter +
 * sort controls, a counted list heading, one card of rows, and the
 * "Total: X across N models" footer.
 *
 * A kind with no entries is HIDDEN rather than shown empty: audio aliases only
 * exist once the engine ships the audio registry, and an always-present empty
 * tab reads as a broken install.
 */

const KIND_LABELS: Record<ModelKind, string> = {
  text: 'Text',
  image: 'Image',
  audio: 'Audio',
};

const KIND_ORDER: ModelKind[] = ['text', 'image', 'audio'];

const FILTER_ORDER: FilterMode[] = ['all', 'cached', 'notCached'];
const SORT_ORDER: SortOrder[] = ['name', 'sizeDescending'];

export function ModelManagement({ open, onClose }: { open: boolean; onClose(): void }) {
  const models = useStore((state) => state.models);
  const selectedByKind = useStore((state) => state.selectedByKind);
  const allowDownloads = useStore((state) => state.allowDownloads);
  const download = useStore((state) => state.download);
  // The STATE alone, not the job: the poll effect keys on this, and
  // subscribing to the whole job would tear it down and rebuild it on every
  // byte of progress.
  const downloadState = useStore((state) => state.download?.state ?? null);
  const setModels = useStore((state) => state.setModels);
  const setDownload = useStore((state) => state.setDownload);
  const selectAlias = useStore((state) => state.selectAlias);
  const pushNotice = useStore((state) => state.pushNotice);

  const [kind, setKind] = useState<ModelKind>('text');
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState<FilterMode>('all');
  const [sort, setSort] = useState<SortOrder>('name');
  const [loading, setLoading] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<ModelEntry | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  const refresh = useMemo(
    () => async (force: boolean) => {
      setLoading(true);
      setFailure(null);
      try {
        const response = await fetchModels(force);
        setModels(response.models);
        useStore.getState().setCapabilities(response.can_switch, response.allow_downloads);
      } catch (cause) {
        const error = asApiError(cause);
        setFailure(error.message);
      } finally {
        setLoading(false);
      }
    },
    [setModels],
  );

  useEffect(() => {
    if (!open) return;
    // `force`: opening the panel is a deliberate act, not a poll, so it
    // should not be served a disk scan up to the server's TTL old.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh(true);
  }, [open, refresh]);

  // Poll the current download, re-attaching to a job already running so
  // reopening mid-download shows real progress. A poll rather than a stream
  // because trycloudflare buffers the sparse SSE feed this replaced.
  //
  // Runs in exactly two situations: `null` (discovery on open, which
  // re-attaches to a pull started before this page loaded) and `running`. A
  // terminal job is NOT polled — the server retains the last finished job
  // forever, so asking again returns the same answer.
  useEffect(() => {
    if (!open || !allowDownloads) return;
    if (downloadState !== null && downloadState !== 'running') return;

    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    // Only a completion we actually WATCHED means the disk changed.
    const watching = downloadState === 'running';
    let failures = 0;

    const tick = async () => {
      let again: boolean;
      try {
        const job = await fetchDownload(controller.signal);
        if (controller.signal.aborted) return;
        failures = 0;
        setDownload(job.state === 'idle' ? null : job);
        if (job.state === 'done' && watching) void refresh(true);
        again = job.state === 'running';
      } catch {
        // A dropped request is not a reason to abandon a live download, but a
        // server that is simply gone must not be polled forever.
        failures += 1;
        again = failures < 5;
      }
      // Scheduled AFTER the response, not on an interval: a slow request must
      // not stack up a queue of overlapping polls.
      if (again && !controller.signal.aborted) timer = setTimeout(() => void tick(), 1000);
    };

    void tick();

    return () => {
      controller.abort();
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [open, allowDownloads, downloadState, setDownload, refresh]);

  // Only kinds that actually have rows, matching rapid-mac's `availableKinds`.
  const availableKinds = useMemo(
    () => KIND_ORDER.filter((candidate) => models.some((model) => model.kind === candidate)),
    [models],
  );

  // A kind can disappear between renders (a catalog refresh, an --attach
  // server); leaving the tab pointed at it would show a permanently empty
  // list with no way back.
  const activeKind = availableKinds.includes(kind) ? kind : (availableKinds[0] ?? 'text');

  const kindEntries = useMemo(
    () => models.filter((model) => model.kind === activeKind),
    [models, activeKind],
  );

  const visible = useMemo(
    () => sortEntries(filterEntries(kindEntries, filter, query), sort),
    [kindEntries, filter, query, sort],
  );

  const usage = useMemo(() => aggregateOnDiskBytes(kindEntries), [kindEntries]);
  const largest = useMemo(() => largestCachedEntry(kindEntries), [kindEntries]);
  const heading = listHeading(filter, query, visible.length, kindEntries.length);
  const footer = diskUsageFooter(usage);

  const choose = async (model: ModelEntry) => {
    if (!model.loadable) {
      pushNotice({
        tone: 'info',
        title: `${model.alias} cannot be started here`,
        body: 'This kind of model needs extras a plain install does not ship.',
      });
      return;
    }

    if (model.alias === selectedByKind[model.kind]) {
      onClose();
      return;
    }

    if (!model.cached) {
      if (!allowDownloads) {
        pushNotice({
          tone: 'info',
          title: 'Downloads are off on this server',
          body: `Pull ${model.alias} from the Mac, then it will appear here as ready to start.`,
        });
        return;
      }
      try {
        const job = await pullModel(model.alias);
        setDownload(job);
        selectAlias(model.kind, model.alias);
      } catch (cause) {
        pushNotice(noticeFor(asApiError(cause), () => void refresh(true)));
      }
      return;
    }

    selectAlias(model.kind, model.alias);
    try {
      // Adopts the server's OWN account of what it is now doing, rather than
      // waiting up to 15 s for the next poll — see state/startModel.ts.
      await startModel(model.alias);
      onClose();
    } catch (cause) {
      pushNotice(noticeFor(asApiError(cause)));
    }
  };

  const remove = async (model: ModelEntry) => {
    setDeleting(model.alias);
    try {
      const result = await removeModel(model.alias);
      const freed = formatBytes(result.freed_bytes);
      pushNotice({
        tone: 'info',
        title: freed ? `Deleted ${model.alias} — freed ${freed}` : `Deleted ${model.alias}`,
        body: 'You can download it again by selecting it.',
      });
      // The row still says "on disk" until the catalog is re-scanned.
      await refresh(true);
    } catch (cause) {
      pushNotice(noticeFor(asApiError(cause), () => void refresh(true)));
    } finally {
      setDeleting(null);
    }
  };

  return (
    <div className="flex min-h-full flex-col gap-6 px-5 pt-5 pb-[calc(env(safe-area-inset-bottom)+16px)]">
      <PageHeader
        title="Models"
        subtitle="Manage the on-disk model cache. Download what you need; delete what you don't to reclaim space."
      />

      <SettingsSection title="Disk overview">
        <div className="flex items-baseline gap-4">
          <span className="flex items-center gap-2 text-sm font-medium">
            <HardDrive className="text-muted-foreground size-4 shrink-0" aria-hidden="true" />
            Models
          </span>
          <span className="text-muted-foreground flex-1 text-right font-mono text-xs">
            {storageSummary(usage)}
          </span>
        </div>
        {largest ? (
          <>
            <SettingsRowDivider />
            <div className="flex items-baseline gap-3">
              <span className="text-muted-foreground shrink-0 text-xs">Largest</span>
              <span className="min-w-0 flex-1 truncate text-sm font-medium">{largest.alias}</span>
              <span className="text-muted-foreground shrink-0 font-mono text-xs">
                {formatBytes(largest.cached_bytes)}
              </span>
            </div>
          </>
        ) : null}
      </SettingsSection>

      <div className="flex flex-col gap-3">
        {availableKinds.length > 1 ? (
          <Segmented<ModelKind>
            label="Model type"
            className="w-full"
            value={activeKind}
            options={availableKinds.map((candidate) => ({
              value: candidate,
              label: KIND_LABELS[candidate],
            }))}
            onChange={setKind}
          />
        ) : null}

        <div className="flex items-center gap-2">
          <div className="relative min-w-0 flex-1">
            <label htmlFor="model-search" className="sr-only">
              Search models
            </label>
            <Search
              className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2"
              aria-hidden="true"
            />
            <Input
              id="model-search"
              type="search"
              className="h-10 px-9"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search models"
              autoCapitalize="off"
              autoCorrect="off"
            />
            {query !== '' ? (
              <Button
                variant="ghost"
                size="icon"
                className="text-muted-foreground absolute top-1/2 right-1 size-8 -translate-y-1/2"
                aria-label="Clear search"
                onClick={() => setQuery('')}
              >
                <X />
              </Button>
            ) : null}
          </div>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" className="h-10 shrink-0">
                <ArrowUpDown />
                Sort
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {SORT_ORDER.map((order) => (
                <DropdownMenuItem key={order} onSelect={() => setSort(order)}>
                  {/* A reserved slot, not a conditional icon: rendering the
                      check only on the active row indents the labels
                      differently from one another. */}
                  <span className="flex size-4 shrink-0 items-center justify-center">
                    {sort === order ? <Check className="size-4" /> : null}
                  </span>
                  {SORT_LABELS[order]}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        <Segmented<FilterMode>
          label="Filter"
          className="w-full"
          value={filter}
          options={FILTER_ORDER.map((mode) => ({ value: mode, label: FILTER_LABELS[mode] }))}
          onChange={setFilter}
        />
      </div>

      <section className="flex flex-col gap-2">
        <div className="flex items-baseline gap-2">
          <h3 className="m-0 text-[15px] leading-none font-semibold">{heading.title}</h3>
          <span className="text-muted-foreground font-mono text-xs">{heading.countText}</span>
        </div>

        {failure ? (
          <p className="text-destructive m-0 py-6 text-center text-sm">{failure}</p>
        ) : visible.length === 0 ? (
          <p className="text-muted-foreground m-0 py-6 text-sm">
            {loading ? 'Loading…' : noMatchesCopy(filter, query)}
          </p>
        ) : (
          // One card holding every row, split by hairlines — rapid-mac's
          // `listSection`. A card per row reads as a stack of stripes rather
          // than one table. `p-1.5` insets the rows so a selected row's fill
          // sits INSIDE the card rather than colliding with its rounded edge.
          <div className="bg-card flex flex-col rounded-lg border p-1.5">
            {visible.map((model, index) => (
              <div key={model.alias}>
                {index > 0 ? <div aria-hidden="true" className="bg-border mx-1.5 h-px" /> : null}
                <ModelRow
                  model={model}
                  current={model.alias === selectedByKind[model.kind]}
                  deleting={deleting === model.alias}
                  onChoose={() => void choose(model)}
                  onDelete={() => setPendingDelete(model)}
                />
              </div>
            ))}
          </div>
        )}

        {footer ? <p className="text-muted-foreground m-0 text-xs">{footer}</p> : null}
      </section>

      {download ? <DownloadStrip job={download} onCancel={() => void cancelDownload()} /> : null}

      <ConfirmDialog
        open={pendingDelete !== null}
        title={`Delete "${pendingDelete?.alias ?? ''}"?`}
        body={deleteBody(pendingDelete)}
        confirmLabel="Delete"
        destructive
        onCancel={() => setPendingDelete(null)}
        onConfirm={() => {
          if (pendingDelete) void remove(pendingDelete);
          setPendingDelete(null);
        }}
      />
    </div>
  );
}

function deleteBody(model: ModelEntry | null): string {
  const size = formatBytes(model?.cached_bytes);
  const freed = size ? ` Frees ${size}.` : '';
  return `Removes the weights from your Mac. You can download it again later by selecting it.${freed}`;
}

function ModelRow({
  model,
  current,
  deleting,
  onChoose,
  onDelete,
}: {
  model: ModelEntry;
  current: boolean;
  deleting: boolean;
  onChoose(): void;
  onDelete(): void;
}) {
  const size = formatBytes(entryBytes(model));

  return (
    // A div, not the button it used to be: the trash is a second control, and
    // a button inside a button is invalid and unclickable in Safari.
    <div
      className={cn(
        'group hover:bg-accent flex w-full items-center gap-1 rounded-md pr-1.5 pl-3',
        current && 'bg-accent',
      )}
    >
      {/* `text-left` belongs on the button, NOT on the row around it: a
          <button> gets `text-align: center` from the UA stylesheet, which
          beats an inherited value. */}
      <button
        type="button"
        className="flex min-w-0 flex-1 items-center gap-2.5 py-2.5 text-left outline-none"
        onClick={onChoose}
      >
        <span className="flex min-w-0 flex-1 flex-col gap-1">
          <span className="text-sm font-medium [overflow-wrap:anywhere]">{model.alias}</span>
          <span className="text-muted-foreground flex items-center gap-1.5 text-xs">
            {size ?? 'size unknown'}
            {model.reasoning_parser ? <Badge variant="secondary">thinks</Badge> : null}
            {model.tool_call_parser ? <Badge variant="secondary">tools</Badge> : null}
            {model.audio_kind ? <Badge variant="secondary">{model.audio_kind}</Badge> : null}
          </span>
        </span>
        <Badge variant={model.cached ? 'default' : 'outline'} className="shrink-0 rounded-full">
          {model.cached ? 'on disk' : 'remote'}
        </Badge>
      </button>

      {/* A fixed slot, always present, holding the trash only where there is
          something to delete. Rendering it conditionally instead let the row's
          width change, so the badges sat in two different columns. */}
      <span className="flex size-9 shrink-0 items-center justify-center">
        {model.cached ? (
          <Button
            variant="ghost"
            size="icon"
            // Revealed on hover on a pointer device, always shown on touch,
            // where there is no hover and it would be unreachable.
            className="text-muted-foreground size-9 opacity-0 transition-opacity duration-150 group-hover:opacity-100 group-focus-within:opacity-100 [&_svg:not([class*=size-])]:size-4 [@media(hover:none)]:opacity-100"
            onClick={onDelete}
            disabled={deleting}
            aria-label={`Delete ${model.alias}`}
            title="Delete"
          >
            <Trash2 />
          </Button>
        ) : null}
      </span>
    </div>
  );
}

function DownloadStrip({ job, onCancel }: { job: DownloadJob; onCancel(): void }) {
  const failed = job.state === 'failed';
  const done = job.done_bytes ?? 0;
  const total = job.total_bytes ?? null;
  const fraction = total && total > 0 ? done / total : null;

  const label =
    job.state === 'failed'
      ? 'failed'
      : job.state === 'cancelled'
        ? 'cancelled'
        : job.state === 'done'
          ? 'done'
          : fraction !== null
            ? `${percent(fraction)} of ${formatBytes(total) ?? ''}`
            : (formatBytes(done) ?? 'starting…');

  return (
    <div
      className={cn(
        'bg-muted sticky bottom-0 -mx-5 border-t px-5 pt-2.5 pb-[calc(env(safe-area-inset-bottom)+10px)]',
        failed && 'border-destructive/40',
      )}
    >
      <div className="mb-2 flex items-center gap-2 text-[13px]">
        <span className="min-w-0 flex-1 truncate font-medium">{job.alias ?? 'Downloading'}</span>
        <span className={cn('font-mono text-xs', failed ? 'text-destructive' : 'text-muted-foreground')}>
          {label}
        </span>
        {job.state === 'running' ? (
          <Button variant="ghost" size="sm" onClick={onCancel}>
            Cancel
          </Button>
        ) : null}
      </div>
      <div className="bg-background h-[3px] overflow-hidden rounded-full">
        <div
          className={cn(
            'h-full transition-[width] duration-300',
            failed ? 'bg-destructive' : 'bg-primary',
          )}
          style={{
            width:
              job.state === 'done'
                ? '100%'
                : job.state === 'cancelled'
                  ? '0%'
                  : fraction !== null
                    ? percent(fraction)
                    : '0%',
          }}
        />
      </div>
      {job.detail ? <p className="text-muted-foreground m-0 mt-1.5 text-xs">{job.detail}</p> : null}
    </div>
  );
}
