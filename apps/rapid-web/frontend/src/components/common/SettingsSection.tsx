import type { ReactNode } from 'react';

/**
 * The grouped-section primitives every Settings panel is built from, ported
 * from `rapid-mac`'s `Components/SettingsSection.swift`. The rules they encode
 * are the same three:
 *
 * * One card per section, and never a card inside a card.
 * * Section headings live OUTSIDE the card, so the card holds controls only.
 * * Rows are separated by a hairline inset to the content's leading edge.
 */

/** A panel's content column, filling the panel's width. */
export function PanelBody({ children }: { children: ReactNode }) {
  return (
    <div className="flex flex-col gap-6 px-5 pt-5 pb-[calc(env(safe-area-inset-bottom)+20px)]">
      {children}
    </div>
  );
}

/** The one title on a panel, with its supporting line. */
export function PageHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="flex flex-col gap-1">
      <h2 className="m-0 text-xl leading-tight font-semibold">{title}</h2>
      {subtitle ? <p className="text-muted-foreground m-0 text-sm">{subtitle}</p> : null}
    </div>
  );
}

/**
 * A titled group of settings rows on one card.
 *
 * `flat` drops the card, for the one case rapid-mac's
 * `InstructionEditorSection` also special-cases: content that already defines
 * its own bordered surface (a textarea, a code block). Nesting that in another
 * box adds visual weight without adding structure.
 *
 * `accessory` is a control on the heading's trailing edge — where a "Clear"
 * belongs, rather than floating in the whitespace under the thing it clears.
 */
export function SettingsSection({
  title,
  subtitle,
  accessory,
  flat = false,
  children,
}: {
  title?: string;
  subtitle?: string;
  accessory?: ReactNode;
  flat?: boolean;
  children: ReactNode;
}) {
  return (
    <section className="flex flex-col gap-2">
      {title ? (
        <div className="flex items-start gap-3">
          <div className="flex min-w-0 flex-1 flex-col gap-1">
            <h3 className="m-0 text-[15px] leading-none font-semibold">{title}</h3>
            {subtitle ? <p className="text-muted-foreground m-0 text-xs">{subtitle}</p> : null}
          </div>
          {accessory ? <div className="shrink-0">{accessory}</div> : null}
        </div>
      ) : null}
      {flat ? (
        children
      ) : (
        <div className="bg-card flex flex-col rounded-lg border p-4">{children}</div>
      )}
    </section>
  );
}

/**
 * One settings row: a label (plus optional explanation) leading, a control
 * trailing. The control keeps its intrinsic width and the label takes the
 * rest, which is what makes the row survive a narrow panel — below `sm:` the
 * control drops beneath the label instead of squeezing it to nothing.
 */
export function SettingsRow({
  title,
  description,
  control,
}: {
  title: string;
  description?: string;
  control?: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between sm:gap-4">
      <div className="flex min-w-0 flex-col gap-1">
        <span className="text-sm leading-none font-medium">{title}</span>
        {description ? <span className="text-muted-foreground text-xs">{description}</span> : null}
      </div>
      {control ? <div className="shrink-0">{control}</div> : null}
    </div>
  );
}

/** The separator between rows inside one card. */
export function SettingsRowDivider() {
  return <div aria-hidden="true" className="bg-border my-4 h-px" />;
}
