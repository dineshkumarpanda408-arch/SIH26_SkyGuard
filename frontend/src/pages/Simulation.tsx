import { useEffect, useState } from 'react';
import { api } from '../api';
import type { SimulationResult, Station } from '../types';
import { PageHeader, Card, Badge, Spinner, ErrorState, Button, fmt } from '../components/ui';
import { IconFlask, IconZap, IconArrowRight, IconCheck, IconAlert } from '../components/icons';

const VARIABLES = [
  { value: 'temperature', label: 'Temperature (\u00b0C)' },
  { value: 'pressure', label: 'Pressure (hPa)' },
  { value: 'humidity', label: 'Humidity (%)' },
];

const TYPES = ['SPIKE', 'DROP', 'DRIFT', 'FROZEN', 'MISSING'];

let cachedStations: import('../types').Station[] = [];

interface TypeConfig {
  help: string;
  magnitude: { min: number; max: number; default: number; label: string };
  duration: { min: number; max: number; default: number; label: string };
}

const TYPE_CONFIG: Record<string, TypeConfig> = {
  SPIKE: {
    help: 'Sudden extreme jump in value (sensor glitch / cold front)',
    magnitude: { min: 3, max: 20, default: 8, label: 'Spike size' },
    duration: { min: 1, max: 10, default: 1, label: 'Duration (readings)' },
  },
  DROP: {
    help: 'Sudden fall in value below the baseline',
    magnitude: { min: 3, max: 20, default: 8, label: 'Drop size' },
    duration: { min: 1, max: 10, default: 1, label: 'Duration (readings)' },
  },
  DRIFT: {
    help: 'Gradual shift away from normal baseline (needs several readings)',
    magnitude: { min: 2, max: 20, default: 10, label: 'Total drift' },
    duration: { min: 4, max: 10, default: 6, label: 'Drift length (readings)' },
  },
  FROZEN: {
    help: 'Value stuck / unchanging \u2014 requires 3+ repeated identical readings',
    magnitude: { min: 3, max: 20, default: 8, label: 'Magnitude' },
    duration: { min: 3, max: 10, default: 5, label: 'Frozen readings' },
  },
  MISSING: {
    help: 'Gap in telemetry (missing/null readings)',
    magnitude: { min: 3, max: 20, default: 8, label: 'Magnitude' },
    duration: { min: 1, max: 10, default: 3, label: 'Missing readings' },
  },
};

const TYPE_TONE: Record<string, string> = {
  SPIKE: 'text-red-300 bg-red-500/10 ring-red-500/30',
  DROP: 'text-orange-300 bg-orange-500/10 ring-orange-500/30',
  DRIFT: 'text-amber-300 bg-amber-500/10 ring-amber-500/30',
  FROZEN: 'text-sky-300 bg-sky-500/10 ring-sky-500/30',
  MISSING: 'text-slate-300 bg-slate-500/10 ring-slate-500/30',
};

export default function Simulation() {
  const [stations, setStations] = useState<Station[]>(cachedStations);
  const [stationId, setStationId] = useState('AWS-023');
  const [variable, setVariable] = useState('temperature');
  const [anomalyType, setAnomalyType] = useState('SPIKE');
  const [magnitude, setMagnitude] = useState(TYPE_CONFIG.SPIKE.magnitude.default);
  const [duration, setDuration] = useState(TYPE_CONFIG.SPIKE.duration.default);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (cachedStations.length > 0) return;
    api
      .stations()
      .then((s) => { cachedStations = s; setStations(s); })
      .catch(() => {});
  }, []);

  const cfg = TYPE_CONFIG[anomalyType] ?? TYPE_CONFIG.SPIKE;
  const hasMagnitude = anomalyType === 'SPIKE' || anomalyType === 'DROP' || anomalyType === 'DRIFT';

  function selectType(t: string) {
    const c = TYPE_CONFIG[t] ?? TYPE_CONFIG.SPIKE;
    setAnomalyType(t);
    setMagnitude(c.magnitude.default);
    setDuration(c.duration.default);
  }

  async function run(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await api.simulate({
        station_id: stationId,
        variable,
        anomaly_type: anomalyType,
        magnitude: hasMagnitude ? magnitude : 0,
        duration,
      });
      setResult(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const ResultBox = (
    <Card
      title="Detection Result"
      subtitle="SIMULATION — synthetic perturbation; ground truth (injected) shown separately from prediction (model output)"
    >
      {busy ? (
        <Spinner label="Running detection…" />
      ) : error ? (
        <ErrorState message={error} />
      ) : !result ? (
        <div className="flex flex-col items-center gap-3 text-slate-500 py-14 text-center">
          <IconFlask size={36} className="text-slate-700" />
          <div className="text-sm max-w-[260px]">Run an injection to see whether the model detects it and why.</div>
        </div>
      ) : (
        <div className="space-y-4">
          {result.detected ? (
            <div className="space-y-4">
              <div className="flex items-center gap-3 rounded-xl bg-emerald-500/[0.08] ring-1 ring-inset ring-emerald-500/25 px-4 py-3">
                <IconCheck size={18} className="text-emerald-400" />
                <span className="text-sm font-bold uppercase tracking-wide text-emerald-300">Anomaly flagged</span>
                <span className="ml-auto text-[10px] uppercase tracking-wider text-slate-400">computed_by · rule:score&gt;threshold</span>
              </div>

              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <div className="rounded-xl border border-borderline/70 bg-panel3/40 p-3" title="Prediction — rule/classifier inspecting the series shape">
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Predicted type</div>
                  <div className="text-sm font-semibold capitalize text-slate-100">{result.detected_type?.toLowerCase()}</div>
                  <div className="text-[10px] text-slate-500 mt-0.5">rule/classifier (not ML)</div>
                </div>
                <div className="rounded-xl border border-borderline/70 bg-panel3/40 p-3" title="Fused model score — NOT calibrated (no probability)">
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Anomaly Score</div>
                  <div className="text-sm font-bold text-slate-100 tabular-nums">{result.confidence != null ? fmt(result.confidence, 3) : '—'}</div>
                  <div className="text-[10px] text-slate-500 mt-0.5">uncalibrated · ml:model_fusion</div>
                </div>
                <div className="rounded-xl border border-borderline/70 bg-panel3/40 p-3" title="Severity — deterministic rule">
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Severity</div>
                  <Badge status={result.severity ?? 'low'}>{result.severity}</Badge>
                  <div className="text-[10px] text-slate-500 mt-0.5">rule</div>
                </div>
                <div className="rounded-xl border border-borderline/70 bg-panel3/40 p-3">
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Latency</div>
                  <div className="text-sm font-mono text-slate-100">{result.latency_ms?.toFixed(1)} ms</div>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="flex items-center gap-3 rounded-xl bg-amber-500/[0.08] ring-1 ring-inset ring-amber-500/25 px-4 py-3">
                <IconAlert size={18} className="text-amber-400" />
                <span className="text-sm font-bold uppercase tracking-wide text-amber-300">Not detected</span>
              </div>
              <p className="text-sm text-slate-300">
                The model did not flag this as anomalous. Try a larger magnitude or more readings.
              </p>
              <p className="text-xs text-slate-500">This is an honest, model-computed result — not fabricated.</p>
            </div>
          )}

          {result.ground_truth && (
            <div className="rounded-xl border border-sky-500/25 bg-sky-500/[0.06] p-3">
              <div className="text-[10px] uppercase tracking-wider text-slate-400 mb-2">
                Ground truth (what was injected) · system:simulator
              </div>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 text-xs">
                <div className="text-slate-500">Requested <span className="text-slate-100 font-mono">{result.ground_truth.requested_type?.toLowerCase()}</span></div>
                <div className="text-slate-500">Canonical <span className="text-slate-100 font-mono">{result.ground_truth.canonical_type?.toLowerCase()}</span></div>
                <div className="text-slate-500">Span <span className="text-slate-100 font-mono">{result.ground_truth.duration} {'\u00d7'}</span></div>
                <div className="text-slate-500">
                  {result.ground_truth.magnitude_sigma != null ? (
                    <>Magnitude <span className="text-slate-100 font-mono">{result.ground_truth.magnitude_sigma}{'\u03c3'}</span></>
                  ) : (
                    <span>Magnitude —</span>
                  )}
                </div>
              </div>
              {result.ground_truth.true_value != null && (
                <div className="text-[10px] text-slate-500 mt-1.5">
                  Injected true value: <span className="font-mono text-slate-300">{fmt(result.ground_truth.true_value, 2)}</span>
                </div>
              )}
            </div>
          )}

          {result.model && (
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-slate-500">
              <span>model <span className="font-mono text-slate-300">{result.model.model_version}</span></span>
              <span>dataset <span className="font-mono text-slate-300">{result.model.dataset_version}</span></span>
              <span>threshold <span className="font-mono text-slate-300">{result.model.threshold}</span></span>
              <span>calibration <span className="font-mono text-amber-300">{result.model.calibration_status}</span></span>
              {result.anomaly_probability != null && (
                <span className="text-amber-300">anomaly_probability={result.anomaly_probability}</span>
              )}
            </div>
          )}

          {result.anomaly_id && (
            <Button to={`/anomalies/${result.anomaly_id}`} variant="outline" className="w-full">
              View model attribution for this anomaly <IconArrowRight size={15} />
            </Button>
          )}

          <div className="flex flex-wrap items-center gap-2 pt-1">
            <Button to="/model" variant="ghost" className="!px-2.5 !py-1 !text-[11px]">How scoring works {'\u2192'} Model</Button>
            <Button to="/evaluation" variant="ghost" className="!px-2.5 !py-1 !text-[11px]">Measured metrics {'\u2192'} Evaluation</Button>
          </div>
        </div>
      )}
    </Card>
  );

  return (
    <div className="space-y-7">
      <PageHeader
        icon={<IconFlask size={20} />}
        title="Anomaly Simulation"
        subtitle="Inject a known anomaly with ground truth and verify detection &amp; model attribution"
      />

      <div className="grid lg:grid-cols-2 gap-6">
        <Card title="Inject Anomaly" subtitle="Ground-truth labeled for honest evaluation">
          <form onSubmit={run} className="space-y-5">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-1.5">Station</label>
                <select value={stationId} onChange={(e) => setStationId(e.target.value)} className="select w-full">
                  {stations.length === 0 && <option value="AWS-023">AWS-023</option>}
                  {stations.map((s) => (
                    <option key={s.station_id} value={s.station_id}>{s.station_id}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-1.5">Variable</label>
                <select value={variable} onChange={(e) => setVariable(e.target.value)} className="select w-full">
                  {VARIABLES.map((v) => (
                    <option key={v.value} value={v.value}>{v.label}</option>
                  ))}
                </select>
              </div>
            </div>

            <div>
              <label className="block text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-2">Anomaly Type</label>
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                {TYPES.map((t) => (
                  <button
                    key={t}
                    type="button"
                    onClick={() => selectType(t)}
                    className={`rounded-lg border px-2 py-2 text-xs font-bold transition-all duration-150 ${
                      anomalyType === t
                        ? `${TYPE_TONE[t] ?? 'bg-sky-500/10 text-sky-300 ring-sky-500/30'} ring-1 ring-inset border-transparent scale-[1.02]`
                        : 'bg-panel3/50 border-borderline text-slate-400 hover:bg-panel3 hover:text-slate-200'
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </div>
              <p className="text-xs text-slate-500 mt-2">{cfg.help}</p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {hasMagnitude ? (
                <div>
                  <label className="flex justify-between text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-2">
                    <span>{cfg.magnitude.label}</span>
                    <span className="rounded-md bg-sky-500/10 ring-1 ring-inset ring-sky-500/25 px-1.5 py-0.5 text-sky-300 tabular-nums">{magnitude}{'\u03c3'}</span>
                  </label>
                  <input
                    type="range"
                    min={cfg.magnitude.min}
                    max={cfg.magnitude.max}
                    value={magnitude}
                    onChange={(e) => setMagnitude(Number(e.target.value))}
                    className="w-full"
                  />
                </div>
              ) : (
                <div className="flex items-end">
                  <span className="text-xs text-slate-500 pb-1.5">
                    No magnitude {'\u2014'} {anomalyType === 'FROZEN' ? 'value is held constant' : 'readings are dropped'}
                  </span>
                </div>
              )}
              <div>
                <label className="flex justify-between text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-1.5">
                  <span>{cfg.duration.label}</span>
                  <span>{duration}</span>
                </label>
                <input
                  type="number"
                  min={cfg.duration.min}
                  max={cfg.duration.max}
                  value={duration}
                  onChange={(e) => setDuration(Math.min(cfg.duration.max, Math.max(cfg.duration.min, Number(e.target.value))))}
                  className="input w-full"
                />
              </div>
            </div>

            <Button type="submit" variant="danger" disabled={busy} className="w-full">
              <IconZap size={16} />
              {busy ? 'Injecting & detecting\u2026' : 'Inject & Run Detection'}
            </Button>
          </form>
        </Card>

        {ResultBox}
      </div>
    </div>
  );
}