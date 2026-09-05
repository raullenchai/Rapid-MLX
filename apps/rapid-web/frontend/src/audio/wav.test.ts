import { describe, expect, it } from 'vitest';
import { TARGET_SAMPLE_RATE, encodeWav } from './wav';

/**
 * The engine decodes with libsndfile, which supports neither mp4 (Safari) nor
 * webm (Chrome/Firefox) — the only containers `MediaRecorder` produces. So the
 * bytes leaving the browser must already be a WAV libsndfile can read.
 */

function parse(blob: ArrayBuffer) {
  const view = new DataView(blob);
  const tag = (offset: number) =>
    String.fromCharCode(...new Uint8Array(blob, offset, 4));
  return {
    riff: tag(0),
    riffSize: view.getUint32(4, true),
    wave: tag(8),
    fmt: tag(12),
    fmtSize: view.getUint32(16, true),
    format: view.getUint16(20, true),
    channels: view.getUint16(22, true),
    sampleRate: view.getUint32(24, true),
    byteRate: view.getUint32(28, true),
    blockAlign: view.getUint16(32, true),
    bits: view.getUint16(34, true),
    data: tag(36),
    dataSize: view.getUint32(40, true),
    sampleAt: (index: number) => view.getInt16(44 + index * 2, true),
  };
}

async function encoded(samples: Float32Array, rate = TARGET_SAMPLE_RATE) {
  return parse(await encodeWav(samples, rate).arrayBuffer());
}

describe('encodeWav', () => {
  it('writes a header libsndfile will accept', async () => {
    const wav = await encoded(new Float32Array(8));

    expect(wav.riff).toBe('RIFF');
    expect(wav.wave).toBe('WAVE');
    expect(wav.fmt).toBe('fmt ');
    expect(wav.data).toBe('data');
    expect(wav.fmtSize).toBe(16);
    expect(wav.format).toBe(1); // PCM
    expect(wav.bits).toBe(16);
    expect(wav.channels).toBe(1);
    expect(wav.sampleRate).toBe(TARGET_SAMPLE_RATE);
  });

  it('declares sizes that match the bytes actually present', async () => {
    // A size field that disagrees with the payload is exactly what makes a
    // decoder reject an otherwise valid file.
    const samples = new Float32Array(100);
    const blob = encodeWav(samples, TARGET_SAMPLE_RATE);
    const wav = parse(await blob.arrayBuffer());

    expect(blob.size).toBe(44 + 200);
    expect(wav.riffSize).toBe(blob.size - 8);
    expect(wav.dataSize).toBe(200);
    expect(wav.byteRate).toBe(TARGET_SAMPLE_RATE * 2);
    expect(wav.blockAlign).toBe(2);
  });

  it('is labelled audio/wav, which is what the upload reports', () => {
    expect(encodeWav(new Float32Array(4), TARGET_SAMPLE_RATE).type).toBe('audio/wav');
  });

  it('scales full-scale samples without wrapping', async () => {
    const wav = await encoded(Float32Array.from([0, 1, -1]));

    expect(wav.sampleAt(0)).toBe(0);
    expect(wav.sampleAt(1)).toBe(32767);
    expect(wav.sampleAt(2)).toBe(-32767);
  });

  it('clamps out-of-range samples rather than letting them wrap', async () => {
    // WebAudio can render slightly outside [-1, 1]; an unclamped wrap turns
    // the loudest moment of a take into a click at the opposite polarity.
    const wav = await encoded(Float32Array.from([1.5, -1.5]));

    expect(wav.sampleAt(0)).toBe(32767);
    expect(wav.sampleAt(1)).toBe(-32767);
  });

  it('handles an empty take without producing a malformed file', async () => {
    const wav = await encoded(new Float32Array(0));

    expect(wav.dataSize).toBe(0);
    expect(wav.riffSize).toBe(36);
  });
});
