import { describe, it, expect } from 'vitest';
import {
  isWeatherAnomaly,
  groupRegionalWeatherEvents,
  REGIONAL_WINDOW_MS,
} from './weatherEvents';
import type { Anomaly } from './types';

function anomaly(overrides: Partial<Anomaly> & { id: number; station_id: string; timestamp: string }): Anomaly {
  return {
    anomaly_type: 'DRIFT',
    confidence: 0.8,
    severity: 'HIGH',
    is_weather_event: true,
    event_assessment: 'LIKELY_WEATHER_EVENT',
    ...overrides,
  };
}

describe('weather event grouping helpers', () => {
  it('isWeatherAnomaly requires weather-side assessment', () => {
    const weather = anomaly({ id: 1, station_id: 'AWS-001', timestamp: '2026-09-15T02:10:00Z' });
    expect(isWeatherAnomaly(weather)).toBe(true);

    const sensorFault = anomaly({
      id: 2,
      station_id: 'AWS-001',
      timestamp: '2026-09-15T02:10:00Z',
      is_weather_event: false,
      event_assessment: 'LIKELY_SENSOR_FAULT',
    });
    expect(isWeatherAnomaly(sensorFault)).toBe(false);

    const unassessed = anomaly({
      id: 3,
      station_id: 'AWS-001',
      timestamp: '2026-09-15T02:10:00Z',
      event_assessment: null,
    });
    expect(isWeatherAnomaly(unassessed)).toBe(false);
  });

  it('groups multiple stations within the window into ONE regional event', () => {
    const groups = groupRegionalWeatherEvents([
      anomaly({ id: 1, station_id: 'AWS-023', timestamp: '2026-09-15T02:10:00Z' }),
      anomaly({ id: 2, station_id: 'AWS-001', timestamp: '2026-09-15T02:25:00Z' }),
      anomaly({ id: 3, station_id: 'AWS-002', timestamp: '2026-09-15T02:45:00Z', severity: 'MEDIUM' }),
    ]);
    expect(groups).toHaveLength(1);
    const g = groups[0];
    // Explicit "one regional weather event" semantics: exactly one group
    // containing all three anomalies across three stations.
    expect(g.anomalies).toHaveLength(3);
    expect(g.stationCount).toBe(3);
    expect(g.stationIds).toEqual(['AWS-001', 'AWS-002', 'AWS-023']);
    expect(g.startTs).toBe('2026-09-15T02:10:00Z');
    expect(g.endTs).toBe('2026-09-15T02:45:00Z');
    expect(g.severity).toBe('HIGH');
  });

  it('splits clusters separated by more than the window', () => {
    const groups = groupRegionalWeatherEvents([
      anomaly({ id: 1, station_id: 'AWS-023', timestamp: '2026-09-15T02:10:00Z' }),
      anomaly({ id: 2, station_id: 'AWS-001', timestamp: '2026-09-15T02:20:00Z' }),
      anomaly({ id: 3, station_id: 'AWS-005', timestamp: '2026-09-15T06:00:00Z' }),
      anomaly({ id: 4, station_id: 'AWS-006', timestamp: '2026-09-15T06:05:00Z' }),
    ]);
    expect(groups).toHaveLength(2);
    expect(groups[0].stationIds).toEqual(['AWS-001', 'AWS-023']);
    expect(groups[1].stationIds).toEqual(['AWS-005', 'AWS-006']);
  });

  it('keeps single-station clusters out of regional grouping', () => {
    const groups = groupRegionalWeatherEvents([
      anomaly({ id: 1, station_id: 'AWS-023', timestamp: '2026-09-15T02:10:00Z' }),
      anomaly({ id: 2, station_id: 'AWS-023', timestamp: '2026-09-15T02:20:00Z' }),
    ]);
    expect(groups).toHaveLength(0);
  });

  it('honours the window boundary (<= window stays grouped)', () => {
    const groups = groupRegionalWeatherEvents(
      [
        anomaly({ id: 1, station_id: 'AWS-001', timestamp: '2026-09-15T02:00:00Z' }),
        anomaly({ id: 2, station_id: 'AWS-002', timestamp: '2026-09-15T03:00:00Z' }),
      ],
      REGIONAL_WINDOW_MS,
    );
    expect(groups).toHaveLength(1);
    expect(groups[0].stationCount).toBe(2);
  });

  it('returns an empty list when nothing is a weather event', () => {
    const groups = groupRegionalWeatherEvents([
      anomaly({
        id: 1,
        station_id: 'AWS-023',
        timestamp: '2026-09-15T02:10:00Z',
        is_weather_event: false,
        event_assessment: 'LIKELY_SENSOR_FAULT',
      }),
    ]);
    expect(groups).toHaveLength(0);
  });
});