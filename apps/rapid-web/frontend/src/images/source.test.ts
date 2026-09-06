import { afterEach, describe, expect, it, vi } from 'vitest';
import { ImageSourceError, readImageSource } from './source';

/**
 * The probe's URL scheme is the point of these specs.
 *
 * Imported files stay binary and use a revocable object URL for measurement
 * and preview, so no multi-megabyte Base64 copies are created.
 */

/** Captures what the size probe was pointed at, and reports `dimensions`. */
function stubImage(dimensions: { width: number; height: number } | null): { src?: string } {
  const seen: { src?: string } = {};
  class FakeImage {
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    naturalWidth = dimensions?.width ?? 0;
    naturalHeight = dimensions?.height ?? 0;
    set src(value: string) {
      seen.src = value;
      queueMicrotask(() => (dimensions ? this.onload?.() : this.onerror?.()));
    }
  }
  vi.stubGlobal('Image', FakeImage);
  return seen;
}

function pngFile(name = 'photo.png'): File {
  // Only the magic bytes matter — the format is sniffed, not read from
  // `File.type`, which is empty from some Android pickers.
  return new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47, 1, 2, 3])], name, {
    type: 'image/png',
  });
}

function stubObjectUrls() {
  const createObjectURL = vi.fn(() => 'blob:source-image');
  const revokeObjectURL = vi.fn();
  vi.stubGlobal('URL', { createObjectURL, revokeObjectURL });
  return { createObjectURL, revokeObjectURL };
}

afterEach(() => vi.unstubAllGlobals());

describe('readImageSource', () => {
  it('measures and retains a revocable blob URL', async () => {
    const urls = stubObjectUrls();
    const probe = stubImage({ width: 512, height: 512 });
    const source = await readImageSource(pngFile());

    expect(probe.src).toBe('blob:source-image');
    expect(source.url).toBe('blob:source-image');
    expect(source.blob).toBeInstanceOf(File);
    expect(urls.createObjectURL).toHaveBeenCalledOnce();
    expect(urls.revokeObjectURL).not.toHaveBeenCalled();
  });

  it('sniffs the format rather than trusting the file type', async () => {
    stubObjectUrls();
    stubImage({ width: 8, height: 8 });
    const mislabelled = new File([new Uint8Array([0, 1, 2, 3])], 'x.png', {
      type: 'image/png',
    });

    await expect(readImageSource(mislabelled)).rejects.toBeInstanceOf(ImageSourceError);
  });

  it("refuses an image past the engine's edge ceiling", async () => {
    const urls = stubObjectUrls();
    stubImage({ width: 9000, height: 9000 });
    await expect(readImageSource(pngFile())).rejects.toThrow(/8192 px/);
    expect(urls.revokeObjectURL).toHaveBeenCalledWith('blob:source-image');
  });

  it('uses a phone-safe pixel cap below the engine ceiling', async () => {
    stubObjectUrls();
    stubImage({ width: 5000, height: 5000 });
    await expect(readImageSource(pngFile())).rejects.toThrow(/20 megapixels/);
  });

  it('names the source by its filename, without the extension', async () => {
    stubObjectUrls();
    stubImage({ width: 64, height: 64 });
    const source = await readImageSource(pngFile('beach sunset.png'));

    expect(source.label).toBe('beach sunset');
    expect(source.mediaType).toBe('image/png');
  });
});
