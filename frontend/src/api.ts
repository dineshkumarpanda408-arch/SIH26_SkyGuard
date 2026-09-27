// API client for the SkyGuard backend.
import type {
  Station,
  StationPrediction,
  ProductionAnalytics,
  ModelMetadata,
  Anomaly,
  AnomalyTimeline,
  SensorHealth,
  Explanation,
  ModelInfo,
  ModelEvaluation,
  SimulationResult,
  DeepDiveStation,
  DeepDiveTelemetryResponse,
  FlaggedLogResponse,
  DeepDiveExplainResponse,
  AuthPatternMeta,
  AuthSession,
  AuthMe,
  AuthError,
} from './types';

function getBaseUrl(): string {
  const envUrl = import.meta.env.VITE_API_URL;
  if (!envUrl || envUrl === '/api') return '/api';
  const clean = envUrl.replace(/\/$/, '');
  return clean.endsWith('/api') ? clean : `${clean}/api`;
}

const BASE = getBaseUrl();

const cache = new Map<string, { data: unknown; ts: number }>();
const CACHE_TTL = 60_000; // 60 seconds

const NO_CACHE = ['/simulation/inject', '/health'];

// WeatherLock session token attached to every authenticated request.
let authToken: string | null = null;
let authFailureHandler: (() => void) | null = null;

export function setAuthToken(token: string | null) {
  authToken = token;
}

export function setAuthFailureHandler(handler: (() => void) | null) {
  authFailureHandler = handler;
}

export function clearCache() {
  cache.clear();
}

function authHeaders(): Record<string, string> {
  return authToken ? { Authorization: `Bearer ${authToken}` } : {};
}

function isAuthPath(path: string) {
  return path.startsWith('/auth/');
}

async function get<T>(path: string): Promise<T> {
  const skip = NO_CACHE.some((p) => path.startsWith(p));
  if (!skip) {
    const hit = cache.get(path);
    if (hit && Date.now() - hit.ts < CACHE_TTL) return hit.data as T;
  }
  const res = await fetch(`${BASE}${path}`, { headers: authHeaders() });
  if (!res.ok) {
    if (res.status === 401 && !isAuthPath(path)) authFailureHandler?.();
    throw new Error(`API ${res.status}: ${res.statusText}`);
  }
  const data = (await res.json()) as T;
  if (!skip) cache.set(path, { data, ts: Date.now() });
  return data;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    if (res.status === 401 && !isAuthPath(path)) authFailureHandler?.();
    throw new Error(`API ${res.status}: ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export const api = {
  stations: () => get<Station[]>('/stations'),
  station: (id: string) => get<Station>(`/stations/${id}`),
  predictions: () => get<StationPrediction[]>('/stations/predictions'),
  predict: (id: string) => get<StationPrediction>(`/stations/${id}/predict`),
  anomalies: (params?: { station_id?: string; severity?: string; anomaly_type?: string }) => {
    const q = new URLSearchParams();
    if (params?.station_id) q.set('station_id', params.station_id);
    if (params?.severity) q.set('severity', params.severity);
    if (params?.anomaly_type) q.set('anomaly_type', params.anomaly_type);
    const qs = q.toString();
    return get<Anomaly[]>(`/anomalies${qs ? `?${qs}` : ''}`);
  },
  anomaly: (id: number) => get<Anomaly>(`/anomalies/${id}`),
  anomalyTimeline: (id: number) => get<AnomalyTimeline>(`/anomalies/${id}/timeline`),
  explanation: (id: number) => get<Explanation>(`/explanations/${id}`),
  sensorHealth: (id: string) => get<SensorHealth>(`/sensor-health/${id}`),
  analytics: (params?: { source?: 'LIVE' | 'HISTORICAL' }) => {
    const q = new URLSearchParams();
    if (params?.source) q.set('source', params.source);
    const qs = q.toString();
    return get<ProductionAnalytics>(`/analytics${qs ? `?${qs}` : ''}`);
  },
  modelInfo: () => get<ModelInfo>('/model/info'),
  modelMetadata: () => get<ModelMetadata>('/model/metadata'),
  modelEvaluation: () => get<ModelEvaluation>('/model/evaluation'),
  simulate: (payload: {
    station_id: string;
    variable: string;
    anomaly_type: string;
    magnitude?: number;
    duration?: number;
  }) => post<SimulationResult>('/simulation/inject', payload),
  health: () => get<{ status: string; system: string; model_trained: boolean }>('/health'),

  // ------------------------------ WeatherLock auth ---------------------------
  authPattern: () => get<AuthPatternMeta>('/auth/pattern'),
  authMe: () => get<AuthMe>('/auth/me'),
  authLogin: async (payload: {
    username: string;
    pattern: string[];
    pin: string;
  }): Promise<AuthSession> => {
    const res = await fetch(`${BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    let data: unknown = null;
    try {
      data = await res.json();
    } catch {
      data = null;
    }
    if (!res.ok) {
      const detail = (data as { detail?: unknown })?.detail;
      const info: AuthError =
        detail && typeof detail === 'object'
          ? (detail as AuthError)
          : { message: typeof detail === 'string' ? detail : `Login failed (${res.status})` };
      const err = new Error(info.message) as Error & { auth?: AuthError; status?: number };
      err.auth = info;
      err.status = res.status;
      throw err;
    }
    return data as AuthSession;
  },
  authLogout: () => post<{ status: string }>('/auth/logout', {}),

  // Deep-Dive XAI & PyOD telemetry API methods
  deepDiveStations: () => get<DeepDiveStation[]>('/deep-dive/stations'),
  deepDiveTelemetry: (params: {
    station_id?: string;
    days?: number;
    contamination?: number;
    seed?: number;
    start_date?: string;
    end_date?: string;
  }) => {
    const q = new URLSearchParams();
    if (params.station_id) q.set('station_id', params.station_id);
    if (params.days != null) q.set('days', String(params.days));
    if (params.contamination != null) q.set('contamination', String(params.contamination));
    if (params.seed != null) q.set('seed', String(params.seed));
    if (params.start_date) q.set('start_date', params.start_date);
    if (params.end_date) q.set('end_date', params.end_date);
    const qs = q.toString();
    return get<DeepDiveTelemetryResponse>(`/deep-dive/telemetry${qs ? `?${qs}` : ''}`);
  },
  deepDivePlotlyChart: (params: {
    station_id?: string;
    days?: number;
    contamination?: number;
    seed?: number;
    start_date?: string;
    end_date?: string;
  }) => {
    const q = new URLSearchParams();
    if (params.station_id) q.set('station_id', params.station_id);
    if (params.days != null) q.set('days', String(params.days));
    if (params.contamination != null) q.set('contamination', String(params.contamination));
    if (params.seed != null) q.set('seed', String(params.seed));
    if (params.start_date) q.set('start_date', params.start_date);
    if (params.end_date) q.set('end_date', params.end_date);
    const qs = q.toString();
    return get<{ data: any[]; layout: any; min_date?: string; max_date?: string; start_date?: string; end_date?: string }>(`/deep-dive/plotly-chart${qs ? `?${qs}` : ''}`);
  },
  deepDiveFlaggedLog: (params: {
    station_id?: string;
    days?: number;
    contamination?: number;
    seed?: number;
    anomaly_types?: string;
    min_confidence?: number;
    only_anomalies?: boolean;
  }) => {
    const q = new URLSearchParams();
    if (params.station_id) q.set('station_id', params.station_id);
    if (params.days != null) q.set('days', String(params.days));
    if (params.contamination != null) q.set('contamination', String(params.contamination));
    if (params.seed != null) q.set('seed', String(params.seed));
    if (params.anomaly_types) q.set('anomaly_types', params.anomaly_types);
    if (params.min_confidence != null) q.set('min_confidence', String(params.min_confidence));
    if (params.only_anomalies != null) q.set('only_anomalies', String(params.only_anomalies));
    const qs = q.toString();
    return get<FlaggedLogResponse>(`/deep-dive/flagged-log${qs ? `?${qs}` : ''}`);
  },
  deepDiveExplain: (params: {
    station_id?: string;
    days?: number;
    contamination?: number;
    seed?: number;
    index?: number;
  }) => {
    const q = new URLSearchParams();
    if (params.station_id) q.set('station_id', params.station_id);
    if (params.days != null) q.set('days', String(params.days));
    if (params.contamination != null) q.set('contamination', String(params.contamination));
    if (params.seed != null) q.set('seed', String(params.seed));
    if (params.index != null) q.set('index', String(params.index));
    const qs = q.toString();
    return get<DeepDiveExplainResponse>(`/deep-dive/explain${qs ? `?${qs}` : ''}`);
  },
  deepDiveReSimulate: (payload: {
    station_id: string;
    days: number;
    contamination: number;
    seed: number;
  }) => post<{ status: string; station_id: string; total_readings: number; detected_anomalies: number }>('/deep-dive/re-simulate', payload),
  deepDiveCsv: async (params: {
    station_id?: string;
    days?: number;
    contamination?: number;
    seed?: number;
    anomaly_types?: string;
    min_confidence?: number;
    only_anomalies?: boolean;
  }): Promise<Blob> => {
    const q = new URLSearchParams();
    if (params.station_id) q.set('station_id', params.station_id);
    if (params.days != null) q.set('days', String(params.days));
    if (params.contamination != null) q.set('contamination', String(params.contamination));
    if (params.seed != null) q.set('seed', String(params.seed));
    if (params.anomaly_types) q.set('anomaly_types', params.anomaly_types);
    if (params.min_confidence != null) q.set('min_confidence', String(params.min_confidence));
    if (params.only_anomalies != null) q.set('only_anomalies', String(params.only_anomalies));
    q.set('export_csv', 'true');
    const res = await fetch(`${BASE}/deep-dive/flagged-log?${q.toString()}`, {
      headers: authHeaders(),
    });
    if (!res.ok) {
      if (res.status === 401) authFailureHandler?.();
      throw new Error(`Export failed (${res.status})`);
    }
    return res.blob();
  },
};

