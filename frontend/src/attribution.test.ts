import { describe, it, expect } from 'vitest';
import { barWidth, signedPercent, topContributor } from '../src/attribution';

describe('attribution helpers', () => {
  it('barWidth normalizes against the max magnitude', () => {
    expect(barWidth({ feature: 'x', contribution: 1 }, 1)).toBe(100);
    expect(barWidth({ feature: 'x', contribution: 0.5 }, 1)).toBe(50);
    expect(barWidth({ feature: 'x', contribution: -0.25 }, 0.5)).toBe(50);
    expect(barWidth({ feature: 'x', contribution: 1 }, 0)).toBe(0);
  });

  it('signedPercent renders a signed integer percent', () => {
    expect(signedPercent({ feature: 'x', contribution: 1.0 })).toBe('100%');
    expect(signedPercent({ feature: 'x', contribution: 0.557 })).toBe('56%');
    expect(signedPercent({ feature: 'x', contribution: -0.447 })).toBe('-45%');
  });

  it('topContributor returns the highest-magnitude contribution', () => {
    const values = [
      { feature: 'temperature', contribution: 1.0 },
      { feature: 'pressure', contribution: 0.557 },
      { feature: 'humidity', contribution: 0.447 },
    ];
    expect(topContributor(values)?.feature).toBe('temperature');
    expect(topContributor([])).toBeNull();
  });
});
