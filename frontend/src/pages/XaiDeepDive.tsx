import { useState, useEffect, useMemo, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import {
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  BarChart,
  Bar,
  Cell,
  PieChart,
  Pie,
  Legend,
} from 'recharts';
import { api } from '../api';
import type {
  DeepDiveStation,
  DeepDiveTelemetryResponse,
  FlaggedLogRow,
  DeepDiveExplainResponse,
} from '../types';
import {
  PageHeader,
  Card,
  StatCard,
  Badge,
  Spinner,
  ErrorState,
  Table,
  Th,
  Td,
  Button,
  ProgressBar,
  fmt,
} from '../components/ui';
import {
  IconActivity,
  IconAlert,
  IconChart,
  IconCpu,
  IconDownload,
  IconFilter,
  IconFlask,
  IconGauge,
  IconHeart,
  IconPin,
  IconRefresh,
  IconSettings,
} from '../components/icons';

const ANOMALY_COLORS: Record<string, string> = {
  Spike: '#ef4444',
  Frozen: '#3b82f6',
  Drift: '#f59e0b',
  'Communication gap': '#8b5cf6',
  'Multivariate Outlier': '#ec4899',
  Normal: '#10b981',
};

export default function XaiDeepDive() {
  // Station & Simulation State
  const [stations, setStations] = useState<DeepDiveStation[]>([]);
  const [stationId, setStationId] = useState<string>('AWS-1');
  const [days, setDays] = useState<number>(30);
  const [contamination, setContamination] = useState<number>(0.055);
  const [seed, setSeed] = useState<number>(42);
  const [showGroundTruth, setShowGroundTruth] = useState<boolean>(true);

  // Active Screen Mode: 'overview' | 'detail' | 'inspector'
  const [activeTab, setActiveTab] = useState<'detail' | 'overview' | 'inspector'>('detail');

  // Telemetry & Deep-Dive Data
  const [telemetryData, setTelemetryData] = useState<DeepDiveTelemetryResponse | null>(null);
  const [loadingTelemetry, setLoadingTelemetry] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [isRetraining, setIsRetraining] = useState<boolean>(false);

  // Plotly chart state
  const plotlyContainerRef = useRef<HTMLDivElement>(null);
  const [plotlyChartData, setPlotlyChartData] = useState<{ data: any[]; layout: any; start_date?: string; end_date?: string } | null>(null);
  const [loadingPlotly, setLoadingPlotly] = useState<boolean>(true);

  // Date Range Filter
  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');

  // Flagged Log Filters
  const [selectedAnomalyTypes, setSelectedAnomalyTypes] = useState<string[]>([]);
  const [minConfidence, setMinConfidence] = useState<number>(0.0);
  const [onlyAnomalies, setOnlyAnomalies] = useState<boolean>(true);

  // SHAP Explainability Inspector State
  const [selectedAnomalyIndex, setSelectedAnomalyIndex] = useState<number | null>(null);
  const [explanationData, setExplanationData] = useState<DeepDiveExplainResponse | null>(null);
  const [loadingExplanation, setLoadingExplanation] = useState<boolean>(false);

  // Load Station Metadata
  useEffect(() => {
    api
      .deepDiveStations()
      .then((stns) => {
        setStations(stns);
        const def = stns.find((s) => s.is_default);
        if (def) setStationId(def.id);
      })
      .catch((e) => setError(e.message));
  }, []);

  // Fetch Telemetry Data on parameter changes
  const fetchTelemetry = (resetExplain: boolean = false) => {
    setLoadingTelemetry(true);
    setError(null);
    api
      .deepDiveTelemetry({
        station_id: stationId,
        days,
        contamination,
        seed,
      })
      .then((res) => {
        setTelemetryData(res);
        setLoadingTelemetry(false);
        if (res.min_date && !startDate) setStartDate(res.min_date);
        if (res.max_date && !endDate) setEndDate(res.max_date);

        // Auto select first anomaly for SHAP inspector
        if (res.anomalies.length > 0 && (selectedAnomalyIndex === null || resetExplain)) {
          setSelectedAnomalyIndex(res.anomalies[0].index);
        }
      })
      .catch((e) => {
        setError(e.message);
        setLoadingTelemetry(false);
      });
  };

  useEffect(() => {
    setLoadingTelemetry(true);
    const timer = setTimeout(() => fetchTelemetry(true), 350);
    return () => clearTimeout(timer);
  }, [stationId, days, contamination, seed]);

  // Fetch Plotly Time Series Data (Exact 4-channel subplots from weather_anomaly_detection)
  useEffect(() => {
    if (activeTab !== 'detail') {
      setLoadingPlotly(false);
      return;
    }
    setLoadingPlotly(true);
    const timer = setTimeout(() => {
      api
        .deepDivePlotlyChart({
          station_id: stationId,
          days,
          contamination,
          seed,
          start_date: startDate || undefined,
          end_date: endDate || undefined,
        })
        .then((res) => {
          setPlotlyChartData(res);
          setLoadingPlotly(false);
          if (res.start_date && !startDate) setStartDate(res.start_date);
          if (res.end_date && !endDate) setEndDate(res.end_date);
        })
        .catch((e) => {
          console.error('Failed to load Plotly chart:', e);
          setLoadingPlotly(false);
        });
    }, 350);
    return () => clearTimeout(timer);
  }, [stationId, days, contamination, seed, startDate, endDate, activeTab]);

  // Mount/Update Plotly chart on DOM node
  useEffect(() => {
    if (!plotlyContainerRef.current || !plotlyChartData) return;
    const layout = {
      ...plotlyChartData.layout,
      autosize: true,
      margin: { l: 50, r: 30, t: 40, b: 40 },
    };
    Plotly.react(plotlyContainerRef.current, plotlyChartData.data, layout, {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
    });
  }, [plotlyChartData, activeTab]);

  // Fetch SHAP Explanation when selected anomaly changes
  useEffect(() => {
    if (selectedAnomalyIndex == null) return;
    setLoadingExplanation(true);
    api
      .deepDiveExplain({
        station_id: stationId,
        days,
        contamination,
        seed,
        index: selectedAnomalyIndex,
      })
      .then((res) => {
        setExplanationData(res);
        setLoadingExplanation(false);
      })
      .catch((e) => {
        console.error('Failed to load SHAP explanation:', e);
        setLoadingExplanation(false);
      });
  }, [selectedAnomalyIndex, stationId, days, contamination, seed]);

  // Re-run simulation
  const handleReRunSimulation = async () => {
    setIsRetraining(true);
    try {
      await api.deepDiveReSimulate({
        station_id: stationId,
        days,
        contamination,
        seed,
      });
      fetchTelemetry(true);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setIsRetraining(false);
    }
  };



  // Filtered Flagged Anomalies Log
  const filteredLogRows = useMemo(() => {
    if (!telemetryData) return [];
    let list: FlaggedLogRow[] = telemetryData.time_series.map((pt) => ({
      index: pt.index,
      timestamp: pt.timestamp,
      station_id: telemetryData.station_id,
      temperature: pt.temperature,
      pressure: pt.pressure,
      humidity: pt.humidity,
      pred_anomaly_type: pt.anomaly_type,
      ground_truth: pt.ground_truth_type,
      confidence: pt.confidence,
      anomaly_score: pt.anomaly_score,
      sensor_health: pt.sensor_health,
      is_anomaly: pt.is_anomaly,
    }));

    if (onlyAnomalies) {
      list = list.filter((r) => r.is_anomaly);
    }
    if (selectedAnomalyTypes.length > 0) {
      list = list.filter((r) => selectedAnomalyTypes.includes(r.pred_anomaly_type));
    }
    if (minConfidence > 0) {
      list = list.filter((r) => r.confidence >= minConfidence);
    }
    return list;
  }, [telemetryData, onlyAnomalies, selectedAnomalyTypes, minConfidence]);

  // CSV export via authenticated blob download
  const [exporting, setExporting] = useState<boolean>(false    );
  const handleCsvExport = async () => {
    try {
      setExporting(true);
      const blob = await api.deepDiveCsv({
        station_id: stationId,
        days,
        contamination,
        seed,
        anomaly_types: selectedAnomalyTypes.length > 0 ? selectedAnomalyTypes.join(',') : undefined,
        min_confidence: minConfidence > 0 ? minConfidence : undefined,
        only_anomalies: onlyAnomalies,
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `weather_anomalies_${stationId}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'CSV export failed.');
    } finally {
      setExporting(false);
    }
  };

  if (error && !telemetryData) {
    return <ErrorState message={error} />;
  }

  const kpi = telemetryData?.kpi;
  const currentStationMeta = stations.find((s) => s.id === stationId);

  return (
    <div className="space-y-7 animate-fade-up">
      {/* Top Header */}
      <PageHeader
        icon={<IconCpu size={22} />}
        title={`Multivariate Telemetry & XAI Deep-Dive: ${stationId}`}
        subtitle="Full-spectrum PyOD Isolation Forest, 9-trend feature engineering, and interactive SHAP TreeExplainer attribution"
        actions={
          <div className="flex flex-wrap items-center gap-3">
            {/* Screen View Mode Switcher */}
            <div className="inline-flex rounded-xl p-1 bg-panel3/70 border border-borderline/80">
              <button
                type="button"
                onClick={() => setActiveTab('detail')}
                className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all flex items-center gap-1.5 ${
                  activeTab === 'detail'
                    ? 'bg-sky-500 text-slate-900 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <IconActivity size={14} />
                <span>4-Channel Graphs</span>
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('inspector')}
                className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all flex items-center gap-1.5 ${
                  activeTab === 'inspector'
                    ? 'bg-indigo-500 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <IconFlask size={14} />
                <span>SHAP Inspector</span>
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('overview')}
                className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all flex items-center gap-1.5 ${
                  activeTab === 'overview'
                    ? 'bg-emerald-500 text-slate-900 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <IconFilter size={14} />
                <span>Flagged Log</span>
              </button>
            </div>

            {/* Re-simulate / Retrain Button */}
            <Button
              variant="outline"
              onClick={handleReRunSimulation}
              disabled={isRetraining || loadingTelemetry}
              className="!px-3 !py-1.5 !text-xs flex items-center gap-1.5"
            >
              <IconRefresh size={14} className={isRetraining ? 'animate-spin text-sky-400' : ''} />
              <span>{isRetraining ? 'Retraining PyOD…' : 'Re-run Simulation'}</span>
            </Button>
          </div>
        }
      />

      {/* Legacy / Research notice */}
      <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 flex items-start gap-3">
        <IconFlask size={18} className="text-amber-400 mt-0.5 shrink-0" />
        <div className="text-xs leading-relaxed">
          <div className="font-bold text-amber-300 uppercase tracking-wider mb-1">
            Legacy / Research Deep-Dive
          </div>
          <div className="text-amber-200/90">
            This page demonstrates the original 9-feature research pipeline and is NOT the current production detector.
            Production anomaly detection and SHAP explanations use the current SkyGuard v3 pipeline.
          </div>
        </div>
      </div>

      {/* Control Panel: Station & Hyperparameters */}
      <Card>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 items-center">
          {/* Station Selector */}
          <div>
            <label className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
              Weather Station
            </label>
            <select
              value={stationId}
              onChange={(e) => setStationId(e.target.value)}
              className="select w-full !py-1.5 text-xs font-medium"
            >
              {stations.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>

          {/* Observation Period Days */}
          <div>
            <div className="flex justify-between items-center text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
              <span>Observation Period</span>
              <span className="text-sky-300 font-mono">{days} Days</span>
            </div>
            <input
              type="range"
              min={7}
              max={30}
              step={1}
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              className="w-full accent-sky-400 cursor-pointer h-1.5 bg-slate-700 rounded-lg"
            />
          </div>

          {/* Model Contamination Slider */}
          <div>
            <div className="flex justify-between items-center text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
              <span>Sensitivity (Contamination)</span>
              <span className="text-sky-300 font-mono">{(contamination * 100).toFixed(1)}%</span>
            </div>
            <input
              type="range"
              min={0.01}
              max={0.1}
              step={0.005}
              value={contamination}
              onChange={(e) => setContamination(Number(e.target.value))}
              className="w-full accent-sky-400 cursor-pointer h-1.5 bg-slate-700 rounded-lg"
            />
          </div>

          {/* Random Seed */}
          <div>
            <label className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
              Simulation Seed
            </label>
            <input
              type="number"
              min={1}
              max={9999}
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="w-full rounded-lg border border-borderline/80 bg-panel3/60 px-3 py-1.5 text-xs font-mono text-slate-100 focus:border-sky-500 focus:outline-none"
            />
          </div>

          {/* Demo Ground Truth Toggle */}
          <div className="flex items-center gap-2 pt-2 md:pt-4">
            <label className="flex items-center gap-2 cursor-pointer text-xs font-medium text-slate-300 select-none">
              <input
                type="checkbox"
                checked={showGroundTruth}
                onChange={(e) => setShowGroundTruth(e.target.checked)}
                className="rounded border-slate-600 bg-slate-800 text-sky-500 focus:ring-sky-500 h-4 w-4"
              />
              <span>🎯 Ground-Truth Labels</span>
            </label>
          </div>
        </div>

        {currentStationMeta && (
          <div className="mt-3 pt-3 border-t border-borderline/50 flex flex-wrap items-center gap-4 text-xs text-slate-400">
            <span>
              Elevation: <strong className="text-slate-200">{currentStationMeta.elevation}</strong>
            </span>
            <span>·</span>
            <span>
              Climate: <strong className="text-slate-200">{currentStationMeta.climate}</strong>
            </span>
            <span>·</span>
            <span className="text-slate-400">{currentStationMeta.description}</span>
          </div>
        )}
      </Card>

      {/* KPI Cards */}
      {kpi && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            label="Total Telemetry Points"
            value={kpi.total_readings}
            icon={<IconPin size={20} />}
            tone="sky"
            sub={`${days} Days @ 5-min intervals`}
          />
          <StatCard
            label="Detected Anomalies"
            value={kpi.total_anomalies}
            icon={<IconAlert size={20} />}
            tone="red"
            sub="Multivariate PyOD IForest"
          />
          <StatCard
            label="Anomaly Rate"
            value={`${kpi.anomaly_rate}%`}
            icon={<IconGauge size={20} />}
            tone="amber"
            sub="Sensitivity threshold"
          />
          <StatCard
            label="Live Sensor Health"
            value={`${kpi.latest_health}%`}
            icon={<IconHeart size={20} />}
            tone={kpi.latest_health >= 85 ? 'emerald' : kpi.latest_health >= 60 ? 'amber' : 'red'}
            sub={`${kpi.health_status} (Avg: ${kpi.avg_health}%)`}
          />
        </div>
      )}

      {loadingTelemetry ? (
        <Spinner label="Processing multivariate weather telemetry and fitting PyOD IForest models…" />
      ) : (
        <>
          {/* ========================================================================= */}
          {/* SCREEN 2 / TAB 1: MULTIVARIATE TELEMETRY & XAI DEEP-DIVE: AWS-3           */}
          {/* ========================================================================= */}
          {activeTab === 'detail' && (
            <div className="space-y-6">
              {/* Date Range Selector */}
              <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl border border-borderline/70 bg-panel2/60">
                <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
                  <IconChart size={16} className="text-sky-400" />
                  <span>Time Series Filter Range:</span>
                </div>
                <div className="flex flex-wrap items-center gap-3">
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-slate-400">Start Date:</span>
                    <input
                      type="date"
                      value={startDate}
                      onChange={(e) => setStartDate(e.target.value)}
                      className="rounded-lg border border-borderline/80 bg-panel3/70 px-2.5 py-1 text-xs font-mono text-slate-100"
                    />
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-slate-400">End Date:</span>
                    <input
                      type="date"
                      value={endDate}
                      onChange={(e) => setEndDate(e.target.value)}
                      className="rounded-lg border border-borderline/80 bg-panel3/70 px-2.5 py-1 text-xs font-mono text-slate-100"
                    />
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Button
                      variant="ghost"
                      onClick={() => {
                        if (telemetryData?.min_date) {
                          setStartDate(telemetryData.min_date);
                          const d = new Date(telemetryData.min_date);
                          d.setDate(d.getDate() + 7);
                          setEndDate(d.toISOString().split('T')[0]);
                        }
                      }}
                      className="!px-2 !py-1 !text-xs text-sky-300"
                    >
                      7-Day Slice (Reference)
                    </Button>
                    <Button
                      variant="ghost"
                      onClick={() => {
                        if (telemetryData?.min_date) {
                          setStartDate(telemetryData.min_date);
                          const d = new Date(telemetryData.min_date);
                          d.setDate(d.getDate() + 14);
                          setEndDate(d.toISOString().split('T')[0]);
                        }
                      }}
                      className="!px-2 !py-1 !text-xs text-slate-300"
                    >
                      14 Days
                    </Button>
                    <Button
                      variant="ghost"
                      onClick={() => {
                        if (telemetryData?.min_date) setStartDate(telemetryData.min_date);
                        if (telemetryData?.max_date) setEndDate(telemetryData.max_date);
                      }}
                      className="!px-2 !py-1 !text-xs text-slate-400"
                    >
                      Full 30 Days
                    </Button>
                  </div>
                </div>
              </div>

              {/* 4-Row Synchronized Plotly Subplots (Exact match to weather_anomaly_detection/dashboard.py) */}
              <Card
                title={`Multivariate Telemetry & XAI Deep-Dive: ${stationId}`}
                subtitle="Interactive multi-parameter time-series, red anomaly flags, synchronized crosshair tooltips, and root-cause diagnostics"
                right={
                  <div className="flex flex-wrap items-center gap-3 text-xs font-mono">
                    <span className="flex items-center gap-1.5 text-orange-400">
                      <span className="w-2.5 h-0.5 bg-orange-400" /> Temperature (°C)
                    </span>
                    <span className="flex items-center gap-1.5 text-blue-400">
                      <span className="w-2.5 h-0.5 bg-blue-500" /> Pressure (hPa)
                    </span>
                    <span className="flex items-center gap-1.5 text-cyan-400">
                      <span className="w-2.5 h-0.5 bg-cyan-400" /> Humidity (%)
                    </span>
                    <span className="flex items-center gap-1.5 text-emerald-400">
                      <span className="w-2.5 h-0.5 bg-emerald-500" /> Health (%)
                    </span>
                    <span className="flex items-center gap-1.5 text-red-400">
                      <span className="w-2 h-2 rounded-full bg-red-600 border border-red-900" /> Flagged Outliers
                    </span>
                  </div>
                }
              >
                <div className="w-full bg-white rounded-xl p-2 shadow-inner overflow-hidden min-h-[730px]">
                  {loadingPlotly ? (
                    <div className="h-[720px] flex items-center justify-center text-slate-500">
                      <Spinner label="Rendering synchronized 4-channel Plotly subplots..." />
                    </div>
                  ) : (
                    <div ref={plotlyContainerRef} className="w-full h-[720px]" />
                  )}
                </div>
              </Card>
            </div>
          )}

          {/* ========================================================================= */}
          {/* SCREEN 3 / TAB 2: INTERACTIVE SHAP EXPLAINABILITY INSPECTOR               */}
          {/* ========================================================================= */}
          {activeTab === 'inspector' && (
            <div className="space-y-6">
              {/* Anomaly Selector Dropdown */}
              <Card
                title="🔍 Anomaly Diagnostic Instance Selector"
                subtitle="Select any detected outlier event to inspect its 9-dimensional SHAP attribution and mechanical fault mode"
              >
                {telemetryData && telemetryData.anomalies.length > 0 ? (
                  <div className="flex flex-col sm:flex-row items-center gap-3">
                    <select
                      value={selectedAnomalyIndex ?? ''}
                      onChange={(e) => setSelectedAnomalyIndex(Number(e.target.value))}
                      className="select flex-1 font-mono text-xs !py-2"
                    >
                      {telemetryData.anomalies.map((anom) => (
                        <option key={anom.index} value={anom.index}>
                          [{anom.timestamp}] — Type: {anom.pred_anomaly_type} | Anomaly Score: {fmt(anom.anomaly_score, 3)} | Legacy Uncalib Score: {(anom.confidence * 100).toFixed(0)}% | Index: {anom.index}
                        </option>
                      ))}
                    </select>

                    <Button
                      variant="outline"
                      onClick={() => {
                        const anoms = telemetryData.anomalies;
                        const currIdx = anoms.findIndex((a) => a.index === selectedAnomalyIndex);
                        const nextIdx = (currIdx + 1) % anoms.length;
                        setSelectedAnomalyIndex(anoms[nextIdx].index);
                      }}
                      className="!text-xs shrink-0"
                    >
                      Next Anomaly →
                    </Button>
                  </div>
                ) : (
                  <div className="text-sm text-slate-500 py-6 text-center">
                    No anomalies detected in the dataset with the current sensitivity threshold.
                  </div>
                )}
              </Card>

              {loadingExplanation ? (
                <Spinner label="Computing SHAP TreeExplainer decomposition across 9 trend features…" />
              ) : explanationData && explanationData.explanation ? (
                <div className="grid lg:grid-cols-12 gap-6">
                  {/* Left Column (7 cols): SHAP Feature Attribution Chart */}
                  <div className="lg:col-span-7">
                    <Card
                      title="📊 SHAP Feature Attribution (TreeExplainer)"
                      subtitle={`Local Shapley decomposition across raw telemetry, rolling means (20), and trend deviations`}
                      right={
                        <Badge status={explanationData.explanation.predicted_anomaly_type}>
                          {explanationData.explanation.predicted_anomaly_type}
                        </Badge>
                      }
                    >
                      <div className="mb-3 flex items-center gap-3 text-[11px] text-slate-400">
                        <span className="flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded bg-red-500" /> Primary Driver
                        </span>
                        <span className="flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded bg-amber-500" /> Trend Deviation
                        </span>
                        <span className="flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded bg-sky-500" /> Baseline Metric
                        </span>
                      </div>

                      <ResponsiveContainer width="100%" height={360}>
                        <BarChart
                          layout="vertical"
                          data={(explanationData.explanation.features || []).slice().reverse()}
                          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                        >
                          <XAxis
                            type="number"
                            stroke="#64748b"
                            tickLine={false}
                            axisLine={false}
                            tick={{ fontSize: 11, fill: '#94a3b8' }}
                          />
                          <YAxis
                            type="category"
                            dataKey="label"
                            stroke="#94a3b8"
                            tickLine={false}
                            axisLine={false}
                            width={130}
                            tick={{ fontSize: 11, fill: '#cbd5e1', fontWeight: 500 }}
                          />
                          <Tooltip
                            content={({ active, payload }) => {
                              if (active && payload && payload.length) {
                                const item = payload[0].payload;
                                return (
                                  <div className="rounded-xl border border-borderline bg-panel2 px-3 py-2 text-xs shadow-xl">
                                    <div className="font-semibold text-slate-200">{item.label}</div>
                                    <div className="text-sky-400 mt-0.5 font-mono">
                                      SHAP Value: {item.shap_value >= 0 ? `+${item.shap_value}` : item.shap_value}
                                    </div>
                                    <div className="text-slate-400 font-mono text-[11px]">
                                      Magnitude |SHAP|: {item.abs_magnitude}
                                    </div>
                                  </div>
                                );
                              }
                              return null;
                            }}
                          />
                          <Bar dataKey="abs_magnitude" name="|SHAP|" radius={[0, 4, 4, 0]} barSize={16}>
                            {(explanationData.explanation.features || [])
                              .slice()
                              .reverse()
                              .map((f, i) => {
                                const col =
                                  f.color_type === 'primary'
                                    ? '#ef4444'
                                    : f.color_type === 'deviation'
                                    ? '#f59e0b'
                                    : '#38bdf8';
                                return <Cell key={`shap-cell-${i}`} fill={col} />;
                              })}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>

                      {explanationData.explanation.top_contributor_label && (
                        <div className="mt-4 pt-3 border-t border-borderline/60 flex items-center justify-between text-xs">
                          <span className="text-slate-400">Primary Driving Feature:</span>
                          <span className="font-bold text-red-400">
                            {explanationData.explanation.top_contributor_label}
                          </span>
                        </div>
                      )}
                    </Card>
                  </div>

                  {/* Right Column (5 cols): Diagnostic Root-Cause Report */}
                  <div className="lg:col-span-5">
                    <Card
                      title="🛠️ Diagnostic Root-Cause Report"
                      subtitle="Mechanical fault diagnosis and recommended maintenance action"
                    >
                      <div className="space-y-4">
                        <div className="p-3.5 rounded-xl bg-panel3/50 border border-borderline/70 space-y-2.5 text-xs">
                          <div className="flex justify-between items-center">
                            <span className="text-slate-400">Event Timestamp:</span>
                            <span className="font-mono text-slate-200 font-semibold">
                              {explanationData.event.timestamp}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-slate-400">Station Identifier:</span>
                            <span className="font-bold text-sky-400">{explanationData.event.station_id}</span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-slate-400">PyOD Anomaly Score:</span>
                            <span className="font-mono text-red-400 font-semibold">
                              {fmt(explanationData.event.anomaly_score, 4)}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-slate-400">Legacy Uncalibrated Score:</span>
                            <span className="font-mono text-emerald-400 font-semibold">
                              {(explanationData.event.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-slate-400">Sensor Health Score:</span>
                            <span className="font-mono text-slate-200 font-semibold">
                              {explanationData.event.sensor_health}%
                            </span>
                          </div>
                        </div>

                        {/* Observed Telemetry Values */}
                        <div className="p-3.5 rounded-xl bg-panel3/30 border border-borderline/50">
                          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
                            Observed Telemetry Channel Readings
                          </div>
                          <div className="grid grid-cols-3 gap-2 text-center">
                            <div className="p-2 rounded-lg bg-soft border border-borderline/40">
                              <div className="text-[10px] text-slate-400">Temp</div>
                              <div className="text-sm font-bold text-orange-300 mt-0.5">
                                {explanationData.event.temperature ?? 'NaN'} °C
                              </div>
                            </div>
                            <div className="p-2 rounded-lg bg-soft border border-borderline/40">
                              <div className="text-[10px] text-slate-400">Pressure</div>
                              <div className="text-sm font-bold text-sky-300 mt-0.5">
                                {explanationData.event.pressure ?? 'NaN'} hPa
                              </div>
                            </div>
                            <div className="p-2 rounded-lg bg-soft border border-borderline/40">
                              <div className="text-[10px] text-slate-400">Humidity</div>
                              <div className="text-sm font-bold text-teal-300 mt-0.5">
                                {explanationData.event.humidity ?? 'NaN'} %
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Predicted Failure Mode */}
                        <div className="p-4 rounded-xl border border-red-500/20 bg-red-500/[0.06]">
                          <div className="text-[10px] uppercase font-bold text-red-400 tracking-wider mb-1">
                            Predicted Failure Mode
                          </div>
                          <div className="text-xl font-black text-red-400">
                            {explanationData.explanation.predicted_anomaly_type}
                          </div>
                        </div>

                        {/* Recommended Maintenance Action */}
                        <div className="p-4 rounded-xl border border-sky-500/20 bg-sky-500/[0.06]">
                          <div className="flex items-center gap-1.5 text-[11px] font-bold text-sky-400 uppercase tracking-wider mb-1.5">
                            <IconSettings size={14} />
                            <span>Recommended Maintenance Action</span>
                          </div>
                          <p className="text-xs text-slate-200 leading-relaxed">
                            {explanationData.explanation.recommended_action}
                          </p>
                        </div>
                      </div>
                    </Card>
                  </div>
                </div>
              ) : null}
            </div>
          )}

          {/* ========================================================================= */}
          {/* SCREEN 1 / TAB 3: OVERVIEW & FLAGGED TELEMETRY ANOMALIES LOG             */}
          {/* ========================================================================= */}
          {activeTab === 'overview' && (
            <div className="space-y-6">
              {/* Overview Visual Charts */}
              <div className="grid lg:grid-cols-2 gap-6">
                {/* Failure Modes & Root-Cause Distribution */}
                <Card
                  title="📌 Detected Failure Modes & Root-Cause Distribution"
                  subtitle="Categorization of detected outliers across the observation period"
                >
                  {telemetryData?.failure_distribution && telemetryData.failure_distribution.length > 0 ? (
                    <div className="space-y-4">
                      <ResponsiveContainer width="100%" height={240}>
                        <PieChart>
                          <Pie
                            data={telemetryData.failure_distribution}
                            dataKey="count"
                            nameKey="name"
                            innerRadius={55}
                            outerRadius={90}
                            paddingAngle={4}
                            stroke="none"
                          >
                            {telemetryData.failure_distribution.map((d) => (
                              <Cell
                                key={d.name}
                                fill={ANOMALY_COLORS[d.name] || '#64748b'}
                              />
                            ))}
                          </Pie>
                          <Tooltip
                            content={({ active, payload }) => {
                              if (active && payload && payload.length) {
                                const item = payload[0].payload;
                                return (
                                  <div className="rounded-xl border border-borderline bg-panel2 px-3 py-2 text-xs shadow-xl">
                                    <div className="font-semibold text-slate-200">{item.name}</div>
                                    <div className="text-slate-400 mt-0.5">
                                      {item.count} events ({item.pct})
                                    </div>
                                  </div>
                                );
                              }
                              return null;
                            }}
                          />
                          <Legend
                            verticalAlign="bottom"
                            formatter={(val) => <span className="text-xs text-slate-300 ml-1">{val}</span>}
                          />
                        </PieChart>
                      </ResponsiveContainer>

                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 pt-2 border-t border-borderline/50 text-center">
                        {telemetryData.failure_distribution.map((s) => (
                          <div key={s.name} className="p-2 rounded-lg bg-panel3/30 border border-borderline/40">
                            <div className="text-[11px] text-slate-400 truncate">{s.name}</div>
                            <div className="text-lg font-bold text-slate-100 mt-0.5">{s.count}</div>
                            <div className="text-[10px] font-mono text-slate-500">{s.pct}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="text-sm text-slate-500 py-12 text-center">
                      No anomalies detected in the dataset.
                    </div>
                  )}
                </Card>

                {/* Station Health Index Gauge */}
                <Card
                  title="🛡️ Station Health Index Gauge"
                  subtitle="Real-time composite telemetry reliability index (0–100)"
                >
                  <div className="flex flex-col items-center justify-center py-6 space-y-4">
                    <div className="relative flex items-center justify-center">
                      <div className="text-5xl font-black tabular-nums text-slate-50">
                        {kpi?.latest_health}
                        <span className="text-2xl text-slate-500 font-normal">%</span>
                      </div>
                    </div>

                    <div className="w-full max-w-sm space-y-2">
                      <div className="flex justify-between text-xs text-slate-400">
                        <span>Degraded (&lt;60%)</span>
                        <span>Optimal (&ge;85%)</span>
                      </div>
                      <ProgressBar
                        value={kpi?.latest_health ?? 100}
                        color={
                          (kpi?.latest_health ?? 100) >= 85
                            ? 'bg-emerald-500'
                            : (kpi?.latest_health ?? 100) >= 60
                            ? 'bg-amber-500'
                            : 'bg-red-500'
                        }
                      />
                    </div>

                    <div className="flex items-center gap-2 pt-2">
                      <Badge
                        status={
                          (kpi?.latest_health ?? 100) >= 85
                            ? 'healthy'
                            : (kpi?.latest_health ?? 100) >= 60
                            ? 'warning'
                            : 'critical'
                        }
                      >
                        {kpi?.health_status ?? 'OPTIMAL'}
                      </Badge>
                      <span className="text-xs text-slate-400">
                        Average observation health: <strong>{kpi?.avg_health}%</strong>
                      </span>
                    </div>
                  </div>
                </Card>
              </div>

              {/* Summary Table: Flagged Telemetry Anomalies Log */}
              <Card
                title={`📋 Flagged Telemetry Anomalies Log (${filteredLogRows.length})`}
                subtitle="Filterable telemetry log with predicted failure modes, legacy uncalibrated scores, and instant CSV export"
                right={
                  <button
                    type="button"
                    onClick={handleCsvExport}
                    disabled={exporting}
                    className="inline-flex items-center gap-1.5 rounded-xl border border-sky-500/30 bg-sky-500/10 hover:bg-sky-500/20 px-3 py-1.5 text-xs font-semibold text-sky-300 transition-colors shadow-sm disabled:opacity-60 disabled:pointer-events-none"
                  >
                    <IconDownload size={14} />
                    <span>{exporting ? '⏳ Exporting…' : '📥 Export Report (CSV)'}</span>
                  </button>
                }
              >
                {/* Table Filter Controls */}
                <div className="p-3.5 rounded-xl bg-panel3/40 border border-borderline/60 mb-4 grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
                  {/* Anomaly Type Multiselect */}
                  <div>
                    <label className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
                      Filter by Anomaly Type:
                    </label>
                    <div className="flex flex-wrap gap-1.5">
                      {['Spike', 'Frozen', 'Drift', 'Communication gap', 'Multivariate Outlier'].map((type) => {
                        const active = selectedAnomalyTypes.includes(type);
                        return (
                          <button
                            key={type}
                            type="button"
                            onClick={() => {
                              setSelectedAnomalyTypes((prev) =>
                                active ? prev.filter((t) => t !== type) : [...prev, type]
                              );
                            }}
                            className={`px-2 py-0.5 text-[11px] font-medium rounded-md transition-all ${
                              active
                                ? 'bg-sky-500 text-slate-900 font-bold'
                                : 'bg-panel3 border border-borderline/80 text-slate-400 hover:text-slate-200'
                            }`}
                          >
                            {type}
                          </button>
                        );
                      })}
                      {selectedAnomalyTypes.length > 0 && (
                        <button
                          type="button"
                          onClick={() => setSelectedAnomalyTypes([])}
                          className="px-1.5 py-0.5 text-[10px] text-slate-500 hover:text-slate-300 underline"
                        >
                          Clear
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Min Confidence Slider */}
                  <div>
                    <div className="flex justify-between items-center text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
                      <span>Minimum Legacy Uncalibrated Score</span>
                      <span className="text-sky-300 font-mono">{(minConfidence * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={minConfidence}
                      onChange={(e) => setMinConfidence(Number(e.target.value))}
                      className="w-full accent-sky-400 cursor-pointer h-1.5 bg-slate-700 rounded-lg"
                    />
                  </div>

                  {/* Anomaly Only Checkbox */}
                  <div className="flex items-center gap-2 pt-2 md:pt-4 md:justify-end">
                    <label className="flex items-center gap-2 cursor-pointer text-xs font-medium text-slate-300 select-none">
                      <input
                        type="checkbox"
                        checked={onlyAnomalies}
                        onChange={(e) => setOnlyAnomalies(e.target.checked)}
                        className="rounded border-slate-600 bg-slate-800 text-sky-500 focus:ring-sky-500 h-4 w-4"
                      />
                      <span>Show Only Detected Anomalies</span>
                    </label>
                  </div>
                </div>

                {/* Table */}
                {filteredLogRows.length === 0 ? (
                  <div className="text-sm text-slate-500 py-12 text-center">
                    No telemetry rows match the selected filter criteria.
                  </div>
                ) : (
                  <div className="max-h-[500px] overflow-auto">
                    <Table>
                      <thead>
                        <tr className="border-b border-borderline/70 sticky top-0 bg-panel2/95 backdrop-blur-sm z-10">
                          <Th>Timestamp</Th>
                          <Th>Station</Th>
                          <Th right>Temp (°C)</Th>
                          <Th right>Pressure (hPa)</Th>
                          <Th right>Humidity (%)</Th>
                          <Th>Predicted Diagnosis</Th>
                          {showGroundTruth && <Th>Ground Truth</Th>}
                          <Th right>Legacy Uncalibrated Score</Th>
                          <Th right>Score</Th>
                          <Th right>Health (%)</Th>
                          <Th right>Action</Th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-borderline/50">
                        {filteredLogRows.slice(0, 150).map((row) => (
                          <tr
                            key={row.index}
                            className={`hover:bg-soft transition-colors group ${
                              row.is_anomaly ? 'bg-red-500/[0.02]' : ''
                            }`}
                          >
                            <Td className="font-mono text-slate-400 whitespace-nowrap">{row.timestamp}</Td>
                            <Td className="font-semibold text-sky-300">{row.station_id}</Td>
                            <Td right mono className="text-orange-300">
                              {row.temperature != null ? fmt(row.temperature, 2) : 'NaN'}
                            </Td>
                            <Td right mono className="text-sky-300">
                              {row.pressure != null ? fmt(row.pressure, 2) : 'NaN'}
                            </Td>
                            <Td right mono className="text-teal-300">
                              {row.humidity != null ? fmt(row.humidity, 2) : 'NaN'}
                            </Td>
                            <Td>
                              <Badge status={row.pred_anomaly_type}>{row.pred_anomaly_type}</Badge>
                            </Td>
                            {showGroundTruth && (
                              <Td>
                                <span
                                  className={`text-xs font-medium ${
                                    row.ground_truth !== 'Normal' ? 'text-amber-400' : 'text-slate-500'
                                  }`}
                                >
                                  {row.ground_truth}
                                </span>
                              </Td>
                            )}
                            <Td right mono className="text-slate-300">
                              {fmt(row.confidence, 3)}
                            </Td>
                            <Td right mono className="text-red-400 font-semibold">
                              {fmt(row.anomaly_score, 4)}
                            </Td>
                            <Td right mono className="text-emerald-300">
                              {fmt(row.sensor_health, 1)}%
                            </Td>
                            <Td right>
                              <button
                                type="button"
                                onClick={() => {
                                  setSelectedAnomalyIndex(row.index);
                                  setActiveTab('inspector');
                                }}
                                className="text-xs font-semibold text-sky-400 hover:text-sky-300 px-2 py-1 rounded bg-sky-500/10 hover:bg-sky-500/20 border border-sky-500/20 transition-all"
                              >
                                Inspect SHAP →
                              </button>
                            </Td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  </div>
                )}
              </Card>
            </div>
          )}
        </>
      )}
    </div>
  );
}
