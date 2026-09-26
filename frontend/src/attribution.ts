// Pure helpers for rendering attribution, kept testable.
import type { Contribution } from './types';

// Normalized bar width (0-100) given a contribution and the max magnitude.
export function barWidth(c: Contribution, max: number): number {
  const mag = Math.abs(c.contribution);
  if (max <= 0) return 0;
  return Math.min(100, (mag / max) * 100);
}

// Signed percentage string e.g. "+100%" / "-45%".
export function signedPercent(c: Contribution): string {
  return `${(c.contribution * 100).toFixed(0)}%`;
}

// Highest-magnitude contributor, used to highlight the driver.
export function topContributor(values: Contribution[]): Contribution | null {
  if (values.length === 0) return null;
  return values.reduce((a, b) => (Math.abs(b.contribution) > Math.abs(a.contribution) ? b : a));
}
