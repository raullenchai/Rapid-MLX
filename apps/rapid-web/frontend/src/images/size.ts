/**
 * Output size, composed from an aspect and a long edge.
 *
 * Two independent choices rather than a flat list of dimensions, following
 * `rapid-mac`'s `Aspect` / `Resolution`: a flat list of every combination is
 * eighteen entries, and users pick shape and detail separately.
 *
 * The engine validates `256..2048` on each side and requires a multiple of
 * 16 (`parse_image_size`), so every pair below is rounded to 16 — an
 * un-rounded value is a 400 the user cannot act on.
 */

export const ASPECTS = ['square', 'portrait', 'landscape'] as const;
export type Aspect = (typeof ASPECTS)[number];

export const RESOLUTIONS = [512, 768, 1024, 1280, 1536, 2048] as const;
export type Resolution = (typeof RESOLUTIONS)[number];

export const ASPECT_LABELS: Record<Aspect, string> = {
  square: '1:1',
  portrait: '3:4',
  landscape: '4:3',
};

/** The engine's floor, ceiling and step. */
const MIN_DIM = 256;
const MAX_DIM = 2048;
const STEP = 16;

function clampToStep(value: number): number {
  const stepped = Math.round(value / STEP) * STEP;
  return Math.min(MAX_DIM, Math.max(MIN_DIM, stepped));
}

/** ``"WxH"`` for the engine's ``size`` field. */
export function outputSize(aspect: Aspect, longEdge: Resolution): string {
  const short = clampToStep((longEdge * 3) / 4);
  const long = clampToStep(longEdge);
  switch (aspect) {
    case 'square':
      return `${long}x${long}`;
    case 'portrait':
      return `${short}x${long}`;
    case 'landscape':
      return `${long}x${short}`;
  }
}
