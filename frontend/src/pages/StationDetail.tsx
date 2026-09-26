import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { api } from '../api';
import type { Anomaly, SensorHealth, StationPrediction } from '../types';
import {
  PageHeader,
  Card,
  Badge,
  Spinner,
  ErrorState,
  StatusDot,
  ProgressBar,
  Button,
  fmt,
} from '../components/ui';
import { IconPin, IconArrowRight, IconCpu } from '../components/icons';

export default function StationDetail() {
  const { id = '' } = useParams();
  const [health, setHealth] = useState<SensorHealth | null>(null);
  const [anomalies, setAnomalies] = useState<Anomaly[] | null>(null);
  const [prediction, setPrediction] = useState<StationPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setError(null);
    Promise.all([api.sensorHealth(id), api.anomalies({ station_id: id }), api.predict(id)])
      .then(([h, a, p]) => {
        if (alive) {
          setHealth(h);
          setAnomalies(a);
          setPrediction(p);
        }
      })
      .catch((e) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, [id]);

  if (error) return <ErrorState message={error} />;
  if (!health || !anomalies) return <Spinner label="Loading station…" />;

  const comps = [
    { label: 'Data Quality', value: health.data_quality, color: 'bg-sky-500' },
    { label: 'Anomaly-Free Score', value: health.anomaly_frequency, color: 'bg-red-500' },
    { label: 'Communication', value: health.communication, color: 'bg-emerald-500' },
    { label: 'Stability', value: health.stability, color: 'bg-teal-500' },
    { label: 'Drift Health', value: health.drift, color: 'bg-amber-500' },
  ];
  const healthTone = health.health_score > 70 ? 'bg-green-500 from-green-400 to-emerald-500' : health.health_score > 40 ? 'bg-yellow-500' : 'bg-red-500 from-red-400 to-rose-500';

  return (
    <div className="space-y-7">
      <PageHeader
        icon={<IconPin size={20} />}
        title={id}
        subtitle={`${prediction?.status ?? '—'}${prediction?.data_source ? ` · ${prediction.data_source}` : ''}${prediction?.model_version ? ` · ${prediction.model_version}` : ''}`}
        actions={
          <div className="flex items-center gap-3">
            <Button to="/deep-dive" variant="outline" className="!px-3 !py-1.5 !text-xs flex items-center gap-1.5">
              <IconCpu size={14} className="text-sky-400" />
              <span>XAI Deep-Dive</span>
            </Button>
            <StatusDot status={prediction?.status ?? 'online'} />
            <Badge status={prediction?.status ?? 'UNAVAILABLE'}>{prediction?.status ?? 'UNAVAILABLE'}</Badge>
          </div>
        }
      />

      <div className="grid lg:grid-cols-3 gap-6">
        <Card title="Sensor Health" subtitle={`Health score ${fmt(health.health_score)} / 100`}>
          <div className="mb-4">
            <div className="flex items-end justify-between mb-2">
              <span className="text-3xl font-extrabold tabular-nums text-slate-50">{fmt(health.health_score, 0)}</span>
              <Badge status={health.health_status}>{health.health_status}</Badge>
            </div>
            <ProgressBar value={health.health_score} color={healthTone} />
          </div>
          <div className="space-y-3">
            {comps.map((c) => (
              <div key={c.label}>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">{c.label}</span>
                  <span className="text-slate-200 tabular-nums">{fmt(c.value, 0)}/100</span>
                </div>
                <ProgressBar value={c.value} color={c.color} />
              </div>
            ))}
          </div>
          {health.degradation && (
            <div className="mt-4 border-t border-borderline/60 pt-4">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1.5">Degradation</div>
              <div className="flex flex-wrap items-center gap-2">
                <Badge status={health.degradation.risk_level}>{health.degradation.degradation_status}</Badge>
                <span className="text-xs text-slate-300">
                  Trend <span className="font-semibold text-slate-100">{health.degradation.trend_label ?? health.degradation.trend}</span> · Risk{' '}
                  <span className="font-semibold text-slate-100">{health.degradation.risk_level}</span>
                </span>
              </div>
              {health.degradation.recommended_action && (
                <div className="mt-2 rounded-lg bg-sky-500/10 ring-1 ring-inset ring-sky-500/25 px-3 py-2 text-xs text-sky-200 leading-relaxed">
                  {health.degradation.recommended_action}
                </div>
              )}
              <div className="text-xs text-slate-400 mt-2 leading-relaxed">{health.degradation.evidence?.join(' ')}</div>
            </div>
          )}
        </Card>

        <Card
          title={`Anomalies \u2014 ${id}`}
          className="lg:col-span-2"
          right={<Button to="/anomalies" variant="ghost" className="!px-2.5 !py-1 !text-xs">All anomalies <IconArrowRight size={14} /></Button>}
        >
          {anomalies.length === 0 ? (
            <div className="text-sm text-slate-500 py-12 text-center">No anomalies recorded for this station.</div>
          ) : (
            <ul className="divide-y divide-borderline/60 max-h-[520px] overflow-auto">
              {anomalies.slice(0, 20).map((a) => (
                <li key={a.id} className="py-3 -mx-2 px-2 rounded-lg hover:bg-soft transition-colors">
                  <Link to={`/anomalies/${a.id}`} className="flex items-center justify-between gap-3 group">
                    <div className="min-w-0 flex items-center gap-3">
                      <span className="w-1.5 h-6 rounded-full bg-gradient-to-b from-sky-500/60 to-indigo-500/60 shrink-0" />
                      <div className="min-w-0">
                        <div className="text-xs text-slate-500">{new Date(a.timestamp).toLocaleString()}</div>
                        <div className="text-sm text-slate-200 group-hover:text-sky-300 transition-colors capitalize truncate">
                          {a.anomaly_type?.toLowerCase()} on {a.feature} (raw {fmt(a.raw_value)})
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <span
                        className="text-xs text-slate-400 tabular-nums"
                        title="Fused model score (uncalibrated \u2014 not a probability)"
                      >
                        {fmt(a.score ?? a.confidence, 3)}
                        <span className="ml-1.5 text-[9px] uppercase tracking-wider text-slate-500">score</span>
                      </span>
                      <Badge status={a.severity}>{a.severity}</Badge>
                    </div>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    </div>
  );
}