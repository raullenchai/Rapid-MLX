/**
 * Reading a source image for an edit.
 *
 * The byte/edge limits mirror the engine. The browser pixel cap is stricter:
 * 20 MP expands to roughly 76 MiB of RGBA, already substantial on a phone.
 */

/** `_MAX_EDIT_IMAGE_BYTES`. */
export const MAX_SOURCE_BYTES = 25 * 1024 * 1024;
const MAX_EDGE = 8192;
const MAX_PIXELS = 20_000_000;

export interface ImageSource {
  /** Original bytes, sent directly as multipart. */
  blob: Blob;
  /** Object URL used by previews and revoked when the source changes. */
  url: string;
  /** For the `<img>` src and the caption. */
  mediaType: 'image/png' | 'image/jpeg';
  label: string;
}

export class ImageSourceError extends Error {}

/**
 * Sniffs the format from the bytes rather than trusting `File.type`, which is
 * derived from the extension and is empty for a file picked from some Android
 * pickers.
 */
function sniff(bytes: Uint8Array): ImageSource['mediaType'] | null {
  if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) {
    return 'image/png';
  }
  if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return 'image/jpeg';
  return null;
}

function measureUrl(url: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const probe = new Image();
    probe.onload = () => resolve({ width: probe.naturalWidth, height: probe.naturalHeight });
    probe.onerror = () => reject(new ImageSourceError("That file isn't a readable image."));
    probe.src = url;
  });
}

async function measure(blob: Blob, url: string): Promise<{ width: number; height: number }> {
  if (typeof createImageBitmap === 'function') {
    try {
      const bitmap = await createImageBitmap(blob);
      const dimensions = { width: bitmap.width, height: bitmap.height };
      bitmap.close();
      return dimensions;
    } catch {
      // Older Safari builds expose createImageBitmap but reject some valid
      // JPEG profiles. The object URL path remains the compatibility fallback.
    }
  }
  return await measureUrl(url);
}

/** Validate and encode a picked file. Throws `ImageSourceError` with copy the
 *  user can act on. */
export async function readImageSource(file: File): Promise<ImageSource> {
  if (file.size > MAX_SOURCE_BYTES) {
    throw new ImageSourceError(
      `Choose an image smaller than ${MAX_SOURCE_BYTES / (1024 * 1024)} MB.`,
    );
  }

  const bytes = new Uint8Array(await file.slice(0, 4).arrayBuffer());
  const mediaType = sniff(bytes);
  if (mediaType === null) throw new ImageSourceError('Choose a PNG or JPEG image.');

  const url = URL.createObjectURL(file);
  try {
    const { width, height } = await measure(file, url);
    if (width > MAX_EDGE || height > MAX_EDGE || width * height > MAX_PIXELS) {
      throw new ImageSourceError('Choose an image no larger than 8192 px or 20 megapixels.');
    }
  } catch (cause) {
    URL.revokeObjectURL(url);
    throw cause;
  }

  return {
    blob: file,
    url,
    mediaType,
    // The filename without its extension, matching rapid-mac — it is what the
    // source strip shows under "Editing image".
    label: file.name.replace(/\.[^.]+$/, '') || 'Imported image',
  };
}

/** Turn an engine result into the same binary source shape used by imports. */
export function sourceFromBase64(data: string, label: string): ImageSource {
  const parts: ArrayBuffer[] = [];
  // Multiple of four, so every slice is independently valid Base64. Keeping
  // the decoded string chunk-sized avoids one additional full-image copy.
  const chunkSize = 32 * 1024;
  for (let offset = 0; offset < data.length; offset += chunkSize) {
    const binary = atob(data.slice(offset, offset + chunkSize));
    const buffer = new ArrayBuffer(binary.length);
    const bytes = new Uint8Array(buffer);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    parts.push(buffer);
  }
  const blob = new Blob(parts, { type: 'image/png' });
  return { blob, url: URL.createObjectURL(blob), mediaType: 'image/png', label };
}
