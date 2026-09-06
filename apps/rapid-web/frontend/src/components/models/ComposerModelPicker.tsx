import { Check, ChevronsUpDown, Settings2 } from 'lucide-react';
import type { ModelEntry, ModelKind } from '@/api/types';
import { useStore } from '@/state/store';
import { formatBytes } from '@/lib/format';
import { cn } from '@/lib/utils';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

/**
 * The model in play, as a control inside the composer.
 *
 * Replaces the "Choose a model" row that used to sit at the top of the
 * sidebar: the model is a property of the message about to be sent, so it
 * belongs beside the send button rather than in the navigation. Matches
 * `rapid-mac`'s `ModelPickerBar`, which sits in the same place.
 *
 * Lists only DOWNLOADED models of the right kind — switching restarts the
 * engine, and starting a download from here would be a multi-GB action
 * triggered from a menu that looks like a selector. Anything else goes
 * through the model manager, which is what the last item opens.
 */
export function ComposerModelPicker({
  kind,
  filter,
  onManage,
  onSelect,
}: {
  kind: ModelKind;
  /** Narrows the list further, e.g. to the image models that can edit. */
  filter?: ((model: ModelEntry) => boolean) | undefined;
  onManage(): void;
  onSelect(model: ModelEntry): void;
}) {
  const models = useStore((state) => state.models);
  const status = useStore((state) => state.status);
  const canSwitch = useStore((state) => state.canSwitch);
  const selected = useStore((state) => state.selectedByKind[kind]);

  const options = models.filter(
    (model) =>
      model.kind === kind && model.cached && model.loadable && (!filter || filter(model)),
  );
  // What this SURFACE is set to, not what the engine happens to be serving:
  // the engine is shared, so on the images page `status.model` is the chat's
  // model until an image model is actually loaded.
  const label = selected ?? 'No model';
  // The tick means "this is what the engine is running", which is only true
  // when the served model is also this surface's selection.
  const live = selected !== null && status?.model === selected && status.state === 'ready';

  if (!canSwitch) {
    return (
      <span
        className="text-muted-foreground flex items-center gap-1.5 px-2 py-1 text-sm"
        title="This server does not own the engine, so the model cannot be changed here."
      >
        <span className="max-w-[18ch] truncate">{label}</span>
      </span>
    );
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className="text-muted-foreground hover:text-foreground flex shrink-0 items-center gap-1.5 rounded-md px-2 py-1 text-sm transition-colors outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
          aria-label={`Model: ${label}`}
          title="Change model"
        >
          {live ? <Check className="size-4 shrink-0" /> : null}
          <span className="max-w-[18ch] truncate">{label}</span>
          <ChevronsUpDown className="size-3.5 shrink-0" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="max-h-[50dvh] w-64 overflow-y-auto">
        <DropdownMenuLabel>Downloaded</DropdownMenuLabel>
        {options.length === 0 ? (
          <DropdownMenuItem disabled>Nothing downloaded yet</DropdownMenuItem>
        ) : (
          options.map((model) => (
            <DropdownMenuItem key={model.alias} onSelect={() => onSelect(model)}>
              <Check
                className={cn('size-4 shrink-0', model.alias !== selected && 'opacity-0')}
              />
              <span className="min-w-0 flex-1 truncate">{model.alias}</span>
              <span className="text-muted-foreground text-xs">
                {formatBytes(model.cached_bytes) ?? ''}
              </span>
            </DropdownMenuItem>
          ))
        )}
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={onManage}>
          <Settings2 className="size-4 shrink-0" />
          Manage models…
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
