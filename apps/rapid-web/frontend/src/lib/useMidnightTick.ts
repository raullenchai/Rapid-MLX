import { useEffect, useState } from 'react';

import { msUntilMidnight } from '@/lib/format';

/**
 * Re-render when the local day rolls over.
 *
 * Without this an open tab keeps saying "Today" about yesterday's
 * conversations until something else happens to re-render — and the phone
 * that is left open overnight is the common case, not the rare one. The Mac
 * app's sidebar carries the same ticker.
 */
export function useMidnightTick(): number {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    // A one-second cushion, so the timer cannot fire a hair BEFORE midnight
    // and reschedule itself for a few milliseconds later in a tight loop.
    const timer = setTimeout(() => setNow(Date.now()), msUntilMidnight(now) + 1000);
    return () => clearTimeout(timer);
  }, [now]);

  return now;
}
