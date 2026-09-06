import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { ArrowDown } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { READING_COLUMN } from '@/lib/layout';
import { cn } from '@/lib/utils';

/**
 * Is the transcript scrolled to (or near) its end?
 *
 * 64px of slack, since a strict test would drop follow on a one-pixel
 * overscroll. A transcript that does not overflow is trivially at the bottom,
 * which keeps the jump button off an empty conversation.
 */
function isAtBottom(element: HTMLElement): boolean {
  return element.scrollHeight - element.scrollTop - element.clientHeight < 64;
}

export interface TranscriptProps {
  children: React.ReactNode;
  /** Bumped whenever new content lands, to drive the follow behaviour. */
  revision: number;
  streaming: boolean;
}

/**
 * The scrolling transcript. Follow is conditional on the user being at the
 * bottom — scrolling back during a stream must not yank the view down.
 */
export function Transcript({ children, revision, streaming }: TranscriptProps) {
  const ref = useRef<HTMLDivElement>(null);
  const inner = useRef<HTMLDivElement>(null);
  const [showJump, setShowJump] = useState(false);

  const onScroll = useCallback(() => {
    const element = ref.current;
    if (!element) return;
    setShowJump(!isAtBottom(element));
  }, []);

  // The transcript also resizes without a commit of its own: the readiness
  // band and the notice stack are siblings, so one appearing shortens the
  // scroller and one leaving lengthens it. Neither fires `scroll` nor bumps
  // `revision`, so a stale `true` outlived the overflow that justified it —
  // that is how the button came to hover over a transcript with nothing
  // below the fold.
  useEffect(() => {
    const element = ref.current;
    const content = inner.current;
    if (!element || !content) return;

    const observer = new ResizeObserver(() => setShowJump(!isAtBottom(element)));
    observer.observe(element);
    observer.observe(content);
    return () => observer.disconnect();
  }, []);

  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;

    // Recomputed from the DOM on every commit, not left to the scroll
    // handler. `scroll` only fires when the user scrolls, so switching to a
    // shorter conversation — or to an empty one — leaves whatever the last
    // one decided. That is how the button came to sit on a brand new chat
    // with nothing above it to jump past.
    const bottom = isAtBottom(element);
    setShowJump(!bottom);

    if (!bottom) return;
    // Direct assignment, not scrollIntoView and not smooth: this fires on
    // every commit, and a queued animation per commit is what made the old
    // page unusable mid-answer.
    element.scrollTop = element.scrollHeight;
  }, [revision]);

  const jump = useCallback(() => {
    const element = ref.current;
    if (!element) return;
    element.scrollTo({ top: element.scrollHeight, behavior: 'smooth' });
  }, []);

  return (
    <div className="relative z-1 min-h-0 flex-1">
      <div
        ref={ref}
        className="h-full overflow-y-auto px-4 pt-5 pb-2"
        onScroll={onScroll}
        // `log` with live announcements OFF. A live transcript would have a
        // screen reader read every 90 ms commit; state changes are announced
        // through the dedicated region instead.
        role="log"
        aria-live="off"
        aria-label="Conversation"
      >
        {/* The SCROLL container stays full-width so the scrollbar sits at the
            window edge; only the column inside it is centred. */}
        <div ref={inner} className={cn(READING_COLUMN, 'flex min-h-full flex-col gap-5')}>
          {children}
        </div>
      </div>

      {showJump ? (
        <Button
          variant="outline"
          size="icon"
          className="animate-in fade-in-0 zoom-in-95 absolute bottom-3.5 left-1/2 size-9 -translate-x-1/2 rounded-full shadow-md"
          onClick={jump}
          aria-label="Jump to latest"
        >
          <ArrowDown />
          {/* The arrow says there is content below; the ring says it is still
              arriving. */}
          {streaming ? (
            <span
              className="border-primary absolute -inset-[3px] animate-spin rounded-full border-2 border-t-transparent"
              aria-hidden="true"
            />
          ) : null}
        </Button>
      ) : null}
    </div>
  );
}

/** Announces state changes only — never streamed tokens. */
export function LiveRegion({ message }: { message: string }) {
  const [announced, setAnnounced] = useState('');

  useEffect(() => {
    // A one-frame clear before re-announcing: an identical string assigned
    // twice is not re-read by most screen readers. The clear IS the point.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setAnnounced('');
    const timer = setTimeout(() => setAnnounced(message), 60);
    return () => clearTimeout(timer);
  }, [message]);

  return (
    <div className="sr-only" role="status" aria-live="polite">
      {announced}
    </div>
  );
}
