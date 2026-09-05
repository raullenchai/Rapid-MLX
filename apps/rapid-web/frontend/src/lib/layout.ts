/**
 * The transcript's reading measure.
 *
 * Shared rather than repeated: the composer and the lifecycle band sit under
 * the transcript, so drift misaligns the column edges. 720px is what
 * `Sidebar.tsx` already assumes for its 900px breakpoint (260 + 720).
 *
 * Only the CONTENT is constrained — the elements stay full-width, so the
 * composer's border and the scrollbar still run to the window edge.
 */
export const READING_COLUMN = 'mx-auto w-full max-w-[720px]';
