// Regional weather-event grouping helpers.
//
// The Dashboard should not treat every spell of weather-related anomalies on
// every station as an independent event: anomalies judged
// LIKELY_WEATHER_EVENT that occur across several stations within a short time
// window are one regional weather event. This module groups them so the UI can
// render a single explicit "one regional weather event" card per group.
import type { Anomaly } from './types';

export const REGIONAL_WINDOW_MS = 60 * 60 * 1000; // 60 minutes

export interface RegionalWeatherEvent {
  anomalies: Anomaly[];
  stationIds: string[];
  stationCount: number;
  startTs: string;
  endTs: string;
  severity: string;
}

const WINDOW_START = 'LIKELY_WEATHER_EVENT';

export function isWeatherAnomaly(a: Anomaly): boolean {
  return a.is_weather_event === true && (a.event_assessment ?? '') === WINDOW_START;
}

export function groupRegionalWeatherEvents(
  anomalies: Anomaly[],
  windowMs: number = REGIONAL_WINDOW_MS,
): RegionalWeatherEvent[] {
  const sorted = anomalies
    .filter(isWeatherAnomaly)
    .slice()
    .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

  const groups: Anomaly[][] = [];
  for (const a of sorted) {
    const last = groups[groups.length - 1];
    const prevTs = last ? new Date(last[last.length - 1].timestamp).getTime() : undefined;
    const cur = new Date(a.timestamp).getTime();
    if (last && prevTs !== undefined && cur - prevTs <= windowMs) {
      last.push(a);
    } else {
      groups.push([a]);
    }
  }

  const sevRank: Record<string, number> = { HIGH: 3, MEDIUM: 2, LOW: 1 };

  return groups
    .filter((g) => new Set(g.map((a) => a.station_id)).size >= 2)
    .map((g) => {
      const stationIds = Array.from(new Set(g.map((a) => a.station_id))).sort();
      const severity = g
        .map((a) => a.severity ?? 'LOW')
        .sort((a, b) => (sevRank[b] ?? 0) - (sevRank[a] ?? 0))[0];
      return {
        anomalies: g,
        stationIds,
        stationCount: stationIds.length,
        startTs: g[0].timestamp,
        endTs: g[g.length - 1].timestamp,
        severity,
      };
    });
}