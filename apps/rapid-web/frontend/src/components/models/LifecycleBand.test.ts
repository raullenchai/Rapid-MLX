import { describe, expect, it } from 'vitest';
import { percent } from './LifecycleBand';

describe('percent', () => {
  it('formats a normal fraction', () => {
    expect(percent(0)).toBe('0%');
    expect(percent(0.463)).toBe('46%');
    expect(percent(1)).toBe('100%');
  });

  it('clamps an overshoot to 100%', () => {
    // The byte monitor overshoots on the final chunk, so an unclamped
    // fraction renders as "101%" — which reads as a bug in the download.
    expect(percent(1.03)).toBe('100%');
    expect(percent(5)).toBe('100%');
  });

  it('clamps a negative to 0%', () => {
    expect(percent(-0.1)).toBe('0%');
  });

  it('clamps before rounding, not after', () => {
    // Rounding first would turn 1.004 into 100% by luck rather than by rule,
    // and 1.6 would still escape as 160%.
    expect(percent(1.6)).toBe('100%');
  });
});
