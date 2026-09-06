/**
 * 16 kHz mono 16-bit PCM WAV, written in the browser.
 *
 * This exists because **the engine cannot decode what `MediaRecorder`
 * produces**. It spools every upload to a `.wav`-suffixed temp file and hands
 * it to libsndfile (via `soundfile`/`mlx_audio.stt.utils.load_audio`), and
 * libsndfile supports neither MP4 nor WebM — the only two containers browsers
 * record. Safari emits mp4, Chrome and Firefox emit webm, so *every* browser
 * failed with `could not decode audio file` until the bytes were transcoded
 * here. Renaming the file does not help: the decoder reads the container, not
 * the extension.
 *
 * `rapid-mac`'s `DictationRecorder` hand-writes the same 16 kHz mono WAV for
 * exactly this reason. Matching it is the fix, not a duplication.
 *
 * WebAudio does the decoding: `decodeAudioData` goes through the platform's
 * own codecs, so it reads whatever the platform just recorded.
 */

/** What the STT lane wants. Whisper resamples to this internally anyway, so
 *  sending more is wasted bytes on a phone uplink. */
export const TARGET_SAMPLE_RATE = 16_000;

/**
 * Decode a recording and re-encode it as WAV.
 *
 * Mixed to mono and resampled by `OfflineAudioContext`, which is the one
 * resampler guaranteed to be present and is implemented natively.
 */
export async function toWav(input: Blob): Promise<Blob> {
  const bytes = await input.arrayBuffer();

  // A plain `AudioContext` would be left running; this one is offline and its
  // length is irrelevant, it exists only to own `decodeAudioData`.
  const decoder = new OfflineAudioContext(1, 1, TARGET_SAMPLE_RATE);
  // Safari's older implementation is callback-only and returns undefined, so
  // the promise form cannot be assumed.
  const decoded = await new Promise<AudioBuffer>((resolve, reject) => {
    const pending = decoder.decodeAudioData(bytes, resolve, reject);
    if (pending instanceof Promise) pending.then(resolve, reject);
  });

  const frames = Math.ceil(decoded.duration * TARGET_SAMPLE_RATE);
  const rendering = new OfflineAudioContext(1, Math.max(1, frames), TARGET_SAMPLE_RATE);
  const source = rendering.createBufferSource();
  source.buffer = decoded;
  source.connect(rendering.destination);
  source.start();
  const rendered = await rendering.startRendering();

  return encodeWav(rendered.getChannelData(0), TARGET_SAMPLE_RATE);
}

/** Float samples in [-1, 1] to a 16-bit PCM WAV. */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  const ascii = (offset: number, text: string) => {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index));
    }
  };

  ascii(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM header size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true); // byte rate
  view.setUint16(32, bytesPerSample, true); // block align
  view.setUint16(34, 8 * bytesPerSample, true);
  ascii(36, 'data');
  view.setUint32(40, samples.length * bytesPerSample, true);

  for (let index = 0; index < samples.length; index += 1) {
    // Clamped before scaling: WebAudio can render slightly outside [-1, 1]
    // and the wrap would land as a loud click rather than a clip.
    const sample = Math.max(-1, Math.min(1, samples[index] ?? 0));
    view.setInt16(44 + index * bytesPerSample, Math.round(sample * 32767), true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}
