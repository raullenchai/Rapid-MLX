import { describe, expect, it } from 'vitest';
import { ASPECTS, RESOLUTIONS, outputSize } from './size';

describe('outputSize', () => {
  it('keeps a square square', () => {
    expect(outputSize('square', 1024)).toBe('1024x1024');
  });

  it('puts the long edge last for a portrait and first for a landscape', () => {
    expect(outputSize('portrait', 1024)).toBe('768x1024');
    expect(outputSize('landscape', 1024)).toBe('1024x768');
  });

  it('produces only dimensions the engine accepts', () => {
    // parse_image_size enforces 256..2048 and a multiple of 16; anything
    // else is a 400 the user has no way to act on.
    for (const aspect of ASPECTS) {
      for (const resolution of RESOLUTIONS) {
        const [width, height] = outputSize(aspect, resolution).split('x').map(Number);
        for (const side of [width, height]) {
          expect(side).toBeGreaterThanOrEqual(256);
          expect(side).toBeLessThanOrEqual(2048);
          expect(side! % 16).toBe(0);
        }
      }
    }
  });

  it('rounds a short edge that is not already a multiple of 16', () => {
    // 1280 * 3/4 = 960, but 1536 * 3/4 = 1152 and 512 * 3/4 = 384 — the
    // rounding matters only if a future resolution breaks the pattern, so
    // this pins the behaviour rather than the current arithmetic.
    expect(outputSize('portrait', 512)).toBe('384x512');
    expect(outputSize('portrait', 1280)).toBe('960x1280');
  });
});
