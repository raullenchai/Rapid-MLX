import { useEffect } from 'react';
import { X } from 'lucide-react';
import { useStore } from '@/state/store';
import { Button } from '@/components/ui/button';

/**
 * In-UI notices, replacing every `window.alert` in the old page.
 *
 * An alert steals focus, cannot be dismissed by tapping away, and on iOS can
 * arrive behind the keyboard.
 */
export function NoticeStack() {
  const notices = useStore((state) => state.notices);

  if (notices.length === 0) return null;

  return (
    // Below the header, not above the composer: the iOS keyboard would cover
    // a bottom notice exactly when most of these fire.
    //
    // Capped and right-aligned rather than full-bleed. On a phone the cap is
    // wider than the viewport so this stays edge-to-edge, but on a desktop
    // window a notice stretched across ~2000px puts its dismiss button a
    // screen away from the text it belongs to.
    <div className="pointer-events-none fixed inset-x-0 top-[calc(env(safe-area-inset-top)+58px)] z-25 flex flex-col items-end gap-2 px-3">
      {notices.map((notice) => (
        <NoticeRow key={notice.id} id={notice.id} />
      ))}
    </div>
  );
}

function NoticeRow({ id }: { id: string }) {
  const notice = useStore((state) => state.notices.find((candidate) => candidate.id === id));
  const dismiss = useStore((state) => state.dismissNotice);

  useEffect(() => {
    // Only info expires on its own. A warning or error names something the
    // user has to act on; clearing it would hide a failure they never saw.
    if (notice?.tone !== 'info') return;
    const timer = setTimeout(() => dismiss(id), 6000);
    return () => clearTimeout(timer);
  }, [notice?.tone, dismiss, id]);

  if (!notice) return null;

  return (
    // No tone-coloured left border: `role` and the wording already carry the
    // severity, and the stripe read as decoration rather than as meaning.
    <div
      className="bg-popover text-popover-foreground animate-in fade-in-0 slide-in-from-top-2 pointer-events-auto flex w-full max-w-md items-center gap-2.5 rounded-lg border py-2.5 pr-2 pl-3 text-sm shadow-md"
      // `alert` interrupts, `status` waits for a pause.
      role={notice.tone === 'error' ? 'alert' : 'status'}
    >
      <div className="flex min-w-0 flex-1 flex-col gap-px">
        <span className="font-medium">{notice.title}</span>
        {notice.body ? (
          <span className="text-muted-foreground text-[13px]">{notice.body}</span>
        ) : null}
      </div>
      {notice.action ? (
        <Button
          variant="secondary"
          size="sm"
          onClick={() => {
            notice.action?.run();
            dismiss(notice.id);
          }}
        >
          {notice.action.label}
        </Button>
      ) : null}
      {/* 32px, not the 15px the glyph needs: a smaller dismiss target is a
          coin-toss with a thumb. */}
      <Button
        variant="ghost"
        size="icon"
        className="text-muted-foreground size-8"
        aria-label="Dismiss"
        onClick={() => dismiss(notice.id)}
      >
        <X />
      </Button>
    </div>
  );
}
