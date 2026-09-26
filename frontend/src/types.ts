// Type definitions matching the backend Pydantic schemas.

export interface Station {
  station_id: string;
  name?: string | null;
  latitude?: number | null;
  longitude?: number | null;
  active: boolean;
}

export interface StationPrediction {
  station_id: string;
  data_source: string;
  model_version: string;
  timestamp: string | null;
  status: 'NORMAL' | 'ANOMALY' | 'STALE' | 'UNAVAILABLE' | null;
  stale: boolean;
  is_anomaly: boolean | null;
  anomaly_score: number | null;
  anomaly_type: string | null;
  severity: string | null;
  confidence: number | null;
  feature: string | null;
  primary_feature: string | null;
  root_cause: string | null;
  is_weather_event: boolean;
  sensor_fault_likelihood: number | null;
  corrected_value: number | null;
  correction_method: string | null;
  correction_confidence: number | null;
  component_scores: Record<string, number> | null;
  rule_flags: Record<string, boolean> | null;
  anomaly_probability: number | null;
  calibration_status: string | null;
  computed_by: Record<string, string | null> | null;
  dataset_version: string | null;
  dataset_sha256: string | null;
  explanation: string | null;
  readings: { temperature: number | null; pressure: number | null; humidity: number | null } | null;
  sensor_health: string | null;
  health_score: number | null;
  n_readings: number;
}

export interface ProductionStationRow {
  station_id: string;
  status: string | null;
  anomaly_type: string | null;
  severity: string | null;
  sensor_health: string | null;
  anomaly_score: number | null;
  timestamp: string | null;
  data_source: string | null;
}

export interface ProductionAnalytics {
  data_source: string;
  model_version: string;
  total_stations: number;
  status_distribution: Record<string, number>;
  severity_distribution: Record<string, number>;
  anomaly_type_distribution: Record<string, number>;
  sensor_health_distribution: Record<string, number>;
  stations: ProductionStationRow[];
}

export interface ModelMetadata {
  model_name: string | null;
  model_version: string | null;
  trained_at_utc: string | null;
  algorithm: string | null;
  threshold: number | null;
  data_source: string | null;
  dataset_version: string | null;
  dataset_sha256: string | null;
  synthetic: boolean | null;
  training_stations: string[] | null;
  n_training_stations: number | null;
  stations_covered: string[] | null;
  n_stations: number | null;
  variables_covered: string[] | null;
  n_variables: number | null;
  n_observations: number | null;
  n_training_obs: number | null;
  n_validation_obs: number | null;
  n_test_obs: number | null;
  split: { method?: string; train?: string; validation?: string; test?: string } | null;
  n_features: number | null;
  feature_count: number | null;
  features: string[] | null;
  calibration: { method?: string | null; status?: string | null; fitted?: boolean | null } | null;
  dataset_file: string | null;
  notes: string | null;
}

export interface Reading {
  id: number;
  station_id: string;
  timestamp: string;
  temperature?: number | null;
  pressure?: number | null;
  humidity?: number | null;
  is_injected?: boolean;
  ground_truth?: boolean;
  is_anomaly?: boolean;
  anomaly_score?: number | null;
}

export interface AnomalyEvidence {
  signal: string;
  label: string;
  status: string;
  detail: string;
  computed_by?: string | null;
  kind?: string | null;
}

export interface Anomaly {
  id: number;
  station_id: string;
  timestamp: string;
  anomaly_type: string;
  anomaly_type_source?: string | null;
  computed_by?: Record<string, string | null> | null;
  confidence: number;
  severity: string;
  cause?: string | null;
  is_weather_event: boolean;
  event_assessment?: string | null;
  corrected_value?: number | null;
  correction_method?: string | null;
  correction_confidence?: number | null;
  feature?: string | null;
  raw_value?: number | null;
  expected_value?: number | null;
  score?: number | null;
  sensor_fault_likelihood?: number | null;
  event_evidence?: string | null;
  evidence?: AnomalyEvidence[] | null;
  anomaly_probability?: number | null;
  calibration_status?: string | null;
}

export interface TimelinePoint {
  label: string;
  timestamp: string;
  value: number | null;
  expected_value?: number | null;
  is_anomaly: boolean;
}

export interface AnomalyTimeline {
  anomaly_id: number | null;
  station_id: string | null;
  variable: string | null;
  points: TimelinePoint[];
}

export interface Contribution {
  feature: string;
  contribution: number;
}

export interface Explanation {
  anomaly_id: number;
  values: Contribution[];
  narrative: string;
  method?: string | null;
}

export interface SensorHealth {
  station_id: string;
  health_score: number;
  health_status: string;
  data_quality: number;
  anomaly_frequency: number;
  communication: number;
  stability: number;
  drift: number;
  trend: string;
  degradation?: {
    degradation_status: string;
    trend: string;
    trend_label?: string;
    risk_level: string;
    evidence: string[];
    recommended_action?: string;
  };
}

export interface EvalMetrics {
  n: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
  false_positive_rate: number;
  false_negative_rate: number;
  roc_auc: number | null;
  confusion_matrix: {
    tn: number;
    fp: number;
    fn: number;
    tp: number;
    grid: number[][];
  };
}

export interface EvalDataset {
  observations: number;
  ground_truth_anomalies: number;
  anomaly_rate: number;
  injections: number;
}

export interface PerCategoryMetrics {
  injected: number;
  detected: number;
  recall: number;
}

export interface MulticlassMatrix {
  labels: string[];
  grid: Record<string, Record<string, number>>;
  matrix: number[][];
}

export interface ModelInfo {
  name: string;
  pipeline: string[];
  variables: string[];
  note: string;
}

export interface ModelComparisonRow extends EvalMetrics {
  name: string;
}

export interface ModelVersionComparison {
  label: string;
  artifact: string;
  n: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number | null;
}

export interface ModelEvaluation {
  performance: EvalMetrics;
  by_type: Record<string, { precision: number; recall: number; f1: number; count: number }>;
  comparison: ModelComparisonRow[];
  selected_model: string;
  detailed: EvalMetrics;
  dataset: EvalDataset;
  per_category: Record<string, PerCategoryMetrics>;
  multiclass_confusion_matrix: MulticlassMatrix;
  model_versions?: {
    note?: string | null;
    comparison?: ModelVersionComparison[] | null;
  } | null;
  comparison_provenance?: Record<string, unknown> | null;
}

export interface SimulationGroundTruth {
  requested_type?: string | null;
  canonical_type?: string | null;
  variable?: string | null;
  magnitude_sigma?: number | null;
  duration?: number | null;
  start_idx?: number | null;
  affected_indices?: number[] | null;
  true_value?: number | null;
}

export interface SimulationResult {
  message: string;
  station_id: string;
  variable: string;
  anomaly_type: string;
  injected_readings: number;
  detected: boolean;
  confidence?: number | null;
  severity?: string | null;
  detected_type?: string | null;
  latency_ms?: number | null;
  anomaly_id?: number | null;
  result?: Anomaly | null;
  computed_by?: Record<string, string | null> | null;
  ground_truth?: SimulationGroundTruth | null;
  model?: {
    model_version: string;
    dataset_version: string;
    threshold: number;
    calibration_status: string;
  } | null;
  calibration_status?: string | null;
  anomaly_probability?: number | null;
}

export interface DeepDiveStation {
  id: string;
  name: string;
  elevation: string;
  climate: string;
  description: string;
  is_default: boolean;
}

export interface DeepDiveTelemetryPoint {
  index: number;
  timestamp: string;
  temperature: number | null;
  pressure: number | null;
  humidity: number | null;
  sensor_health: number;
  is_anomaly: boolean;
  is_temp_anomaly?: boolean;
  is_pressure_anomaly?: boolean;
  is_humidity_anomaly?: boolean;
  affected_feature?: string;
  anomaly_type: string;
  anomaly_score: number;
  confidence: number;
  ground_truth: boolean;
  ground_truth_type: string;
}

export interface DeepDiveAnomalyEvent {
  index: number;
  timestamp: string;
  station_id: string;
  temperature: number | null;
  pressure: number | null;
  humidity: number | null;
  pred_anomaly_type: string;
  is_temp_anomaly?: boolean;
  is_pressure_anomaly?: boolean;
  is_humidity_anomaly?: boolean;
  affected_feature?: string;
  confidence: number;
  anomaly_score: number;
  sensor_health: number;
  ground_truth: boolean;
  ground_truth_type: string;
}

export interface DeepDiveKPI {
  total_readings: number;
  total_anomalies: number;
  anomaly_rate: number;
  latest_health: number;
  avg_health: number;
  health_status: 'OPTIMAL' | 'DEGRADED' | 'CRITICAL';
}

export interface DeepDiveFailureDistribution {
  name: string;
  count: number;
  pct: string;
}

export interface DeepDiveTelemetryResponse {
  station_id: string;
  days: number;
  contamination: number;
  seed: number;
  min_date: string | null;
  max_date: string | null;
  kpi: DeepDiveKPI;
  failure_distribution: DeepDiveFailureDistribution[];
  time_series: DeepDiveTelemetryPoint[];
  anomalies: DeepDiveAnomalyEvent[];
}

export interface FlaggedLogRow {
  index: number;
  timestamp: string;
  station_id: string;
  temperature: number | null;
  pressure: number | null;
  humidity: number | null;
  pred_anomaly_type: string;
  ground_truth: string;
  confidence: number;
  anomaly_score: number;
  sensor_health: number;
  is_anomaly: boolean;
}

export interface FlaggedLogResponse {
  station_id: string;
  total_filtered: number;
  available_types: string[];
  rows: FlaggedLogRow[];
}

export interface ShapFeatureContribution {
  feature: string;
  label: string;
  shap_value: number;
  abs_magnitude: number;
  color_type: 'primary' | 'deviation' | 'baseline';
}

export interface ShapExplanationDetails {
  features: ShapFeatureContribution[];
  top_contributor: string | null;
  top_contributor_label: string | null;
  recommended_action: string;
  predicted_anomaly_type: string;
}

export interface DeepDiveExplainResponse {
  event: FlaggedLogRow;
  explanation: ShapExplanationDetails;
}

// ------------------------------- WeatherLock auth -----------------------------

export interface WeatherIcon {
  id: string;
  emoji: string;
  label: string;
}

export interface AuthPatternMeta {
  icons: WeatherIcon[];
  pattern_length: number;
  pin_length: number;
  default_username: string;
  max_attempts: number;
  lockout_seconds: number;
}

export interface AuthSession {
  token: string;
  username: string;
  expires_at: string | null;
  idle_timeout_seconds: number;
}

export interface AuthMe {
  username: string;
  authenticated: boolean;
  expires_at: string | null;
  idle_timeout_seconds: number;
  pattern_length: number;
}

export interface AuthError {
  message: string;
  reason?: string;
  attempts_left?: number;
  retry_after?: number;
}

