import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { api } from '../api';
import type { Anomaly, AnomalyTimeline } from '../types';
import ExplanationPanel from '../components/ExplanationPanel';
import ModelVersionTag from '../components/ModelVersionTag';
import { PageHeader, Card, Badge, Spinner, ErrorState, fmt } from '../components/ui';
import { IconArrowLeft, IconAlert, IconCheck, IconInfo } from '../components/icons';

function Row({ label, value, mono }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-4 py-2.5 border-b border-borderline/50 last:border-0">
      <span className="text-[11px] text-slate-500 uppercase tracking-wider">{label}</span>
      <span className={`text-sm text-slate-100 ${mono ? 'font-mono text-[13px]' : ''}`}>{value}</span>
    </div>
  );
}

export default function AnomalyDetail() {
  const { id = '' } = useParams();
  const [anomaly, setAnomaly] = useState<Anomaly | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [timeline, setTimeline] = useState<AnomalyTimeline | null>(null);
  const [timelineError, setTimelineError] = useState<string | null>(null);

  useEffect(() => {
    setAnomaly(null);
    setError(null);
    setTimeline(null);
    setTimelineError(null);
    api
      .anomaly(Number(id))
      .then(setAnomaly)
      .catch((e) => setError(e.message));
    api
      .anomalyTimeline(Number(id))
      .then(setTimeline)
      .catch((e) => setTimelineError(e.message));
  }, [id]);

  if (error) return <ErrorState message={error} />;
  if (!anomaly) return <Spinner label="Loading anomaly\u2026" />;

  return (
    <div className="space-y-7">
      <PageHeader
        title={`Anomaly #${anomaly.id}`}
        icon={<IconAlert size={20} />}
        subtitle={<Link to="/anomalies" className="inline-flex items-center gap-1.5 text-xs text-sky-400 hover:text-sky-300 font-medium transition-colors"><IconArrowLeft size={14} /> Back to anomalies</Link>}
        actions={
          <div className="flex items-center gap-2">
            <ModelVersionTag />
            <Badge status={anomaly.severity}>
              <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse-soft" /> {anomaly.severity} severity
            </Badge>
          </div>
        }
      />

      <div className="grid lg:grid-cols-2 gap-6">
        <Card title="Detection Summary" subtitle={new Date(anomaly.timestamp).toLocaleString()}>
          <Row label="Station" value={<span className="font-semibold text-sky-300">{anomaly.station_id}</span>} />
          <Row label="Anomaly Type" value={<span className="capitalize">{anomaly.anomaly_type?.toLowerCase()}</span>} />
          <Row label="Affected Variable" value={anomaly.feature ?? '\u2014'} />
          <Row label="Observed Value" value={<span className="text-red-400 font-semibold">{fmt(anomaly.raw_value)}</span>} mono />
          <Row label="Expected Value" value={fmt(anomaly.expected_value)} mono />
          <Row
            label="Anomaly Score"
            value={
              <span title="Fused model score — not a calibrated probability">
                {anomaly.score != null ? fmt(anomaly.score, 3) : '\u2014'}
                <span className="ml-1.5 text-[10px] font-normal text-slate-500 uppercase">uncalibrated</span>
              </span>
            }
            mono
          />
          <Row
            label="Type Source"
            value={<span className="text-[11px] font-normal text-slate-400">{anomaly.anomaly_type_source ?? 'rule/classifier'}</span>}
          />
          <Row
            label="Assessment"
            value={
              anomaly.is_weather_event ? (
                <Badge status="healthy">Genuine weather</Badge>
              ) : (
                <span className="text-red-400 font-medium">LIKELY SENSOR FAULT</span>
              )
            }
          />
        </Card>

        <Card title="Corrected Value" subtitle="Statistical imputation (diurnal estimate / fallback) \u2014 not ML">
          {anomaly.corrected_value !== null && anomaly.corrected_value !== undefined ? (
            <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/[0.06] p-4">
              <div className="flex items-center gap-2 text-emerald-300 mb-1">
                <IconCheck size={16} />
                <span className="text-[11px] font-semibold uppercase tracking-wider">Imputed reading</span>
              </div>
              <div className="text-4xl font-extrabold text-emerald-300 mb-1 tabular-nums">
                {fmt(anomaly.corrected_value)}
                {anomaly.feature === 'pressure' ? ' hPa' : anomaly.feature === 'humidity' ? ' %' : ' \u00b0C'}
              </div>
              <div className="text-xs text-slate-400 mb-4">Method: {anomaly.correction_method ?? '\u2014'}</div>
              <Row label="Correction Confidence" value={`${fmt((anomaly.correction_confidence ?? 0) * 100, 1)}%`} mono />
              <div className="mt-3 text-xs text-slate-500">
                Difference from observed: {fmt((anomaly.corrected_value ?? 0) - (anomaly.raw_value ?? 0), 1)}
              </div>
            </div>
          ) : (
            <div className="text-sm text-slate-500 py-12 text-center">No corrected value computed for this anomaly.</div>
          )}
        </Card>
      </div>

      {anomaly.event_assessment && (
        <Card title="Weather vs Sensor-Fault Assessment" subtitle="Spatial consistency across neighboring stations">
          <div className="flex flex-wrap items-center gap-4 mb-4">
            <Badge status={anomaly.is_weather_event ? 'normal' : 'critical'}>
              {anomaly.event_assessment.toLowerCase().replace(/_/g, ' ')}
            </Badge>
            {anomaly.event_assessment && (
              <span
                className="text-xs text-slate-400"
                title="Rule-based spatial heuristic, not a trained probability"
              >
                sensor-fault likelihood (rule heuristic) = {fmt(anomaly.sensor_fault_likelihood ?? 0, 2)}
              </span>
            )}
          </div>
          {anomaly.event_evidence ? (
            <ul className="text-xs text-slate-300 space-y-1.5 list-none">
              {anomaly.event_evidence.split('\n').filter(Boolean).map((line, i) => (
                <li key={i} className="flex items-start gap-2">
                  <IconInfo size={13} className="mt-0.5 shrink-0 text-slate-500" />
                  <span className="leading-relaxed">{line}</span>
                </li>
              ))}
            </ul>
          ) : (
            <div className="text-xs text-slate-500">No detailed evidence recorded.</div>
          )}
        </Card>
      )}

      <Card
        title="Evidence & Provenance"
        subtitle="Signals that support this detection — each tagged with what computed it (ML, heuristic, or rule)"
      >
        {anomaly.evidence?.length ? (
          <ul className="space-y-3 list-none">
            {anomaly.evidence.map((e, i) => {
              const ok = e.status === 'consistent' || e.status === 'weather' || e.status === 'normal';
              const warn = e.status === 'flagged' || e.status === 'elevated' || e.status === 'inconsistent' || e.status === 'sensor_fault';
              const tone = warn
                ? { icon: <IconAlert size={14} className="mt-0.5 shrink-0 text-amber-400" />, label: 'text-amber-300' }
                : ok
                  ? { icon: <IconCheck size={14} className="mt-0.5 shrink-0 text-emerald-400" />, label: 'text-emerald-300' }
                  : { icon: <IconInfo size={14} className="mt-0.5 shrink-0 text-slate-500" />, label: 'text-slate-400' };
              return (
                <li key={i} className="rounded-xl border border-borderline/60 bg-panel3/40 p-3.5">
                  <div className="flex items-start gap-2.5">
                    {tone.icon}
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className={`text-[13px] font-semibold ${tone.label}`}>{e.label}</span>
                        <span className="text-[10px] uppercase tracking-wider text-slate-500 px-1.5 py-0.5 rounded bg-slate-500/10 border border-borderline/50">
                          {e.kind ?? e.status}
                        </span>
                        <span className="text-[9px] font-mono text-slate-600 break-all">{e.computed_by}</span>
                      </div>
                      <p className="text-xs text-slate-300 mt-1 leading-relaxed">{e.detail}</p>
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        ) : (
          <div className="text-xs text-slate-500 py-8 text-center">
            No structured evidence recorded for this anomaly.
          </div>
        )}
      </Card>

      <Card
        title="Timeline"
        subtitle="Same station & variable — normal readings before, the flagged reading, normal readings after"
      >
        {timelineError ? (
          <div className="text-xs text-slate-500 py-8 text-center">Could not load timeline: {timelineError}</div>
        ) : !timeline ? (
          <Spinner label="Loading timeline\u2026" />
        ) : (
          <div className="flex gap-2 overflow-x-auto pb-1">
            {timeline.points.map((p, i) => {
              const unit = timeline.variable === 'pressure' ? ' hPa' : timeline.variable === 'humidity' ? ' %' : ' \u00b0C';
              const val = p.value != null ? fmt(p.value, 2) : '\u2014';
              return (
                <div key={i} className="flex items-center gap-2">
                  {i > 0 && <div className="h-px w-4 bg-borderline" />}
                  <div
                    className={`rounded-xl border px-3 py-2.5 min-w-[104px] ${
                      p.is_anomaly ? 'border-red-500/40 bg-red-500/[0.08]' : 'border-borderline/70 bg-panel3/40'
                    }`}
                  >
                    <div
                      className={`text-[10px] font-bold uppercase tracking-wider ${p.is_anomaly ? 'text-red-400' : 'text-slate-500'}`}
                    >
                      {p.is_anomaly ? 'Anomaly' : 'Normal'}
                    </div>
                    <div className={`text-lg font-extrabold tabular-nums mt-0.5 ${p.is_anomaly ? 'text-red-400' : 'text-slate-100'}`}>
                      {val}
                      <span className="text-[10px] font-normal text-slate-500">{unit}</span>
                    </div>
                    <div className="text-[10px] text-slate-500 mt-0.5 whitespace-nowrap">
                      {new Date(p.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                    {p.is_anomaly && p.expected_value != null && (
                      <div className="text-[10px] text-slate-500 whitespace-nowrap">expected {fmt(p.expected_value, 2)}</div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
        {timeline && (
          <div className="text-[11px] text-slate-500 mt-3">
            {timeline.variable ? <>Variable: {timeline.variable} {'\u00b7'} </> : 'variable unknown \u00b7 '}
            the flagged point is the persisted anomaly record (its raw value); context readings are real station history.
            Fewer context points simply mean the flag sat at the edge of available data.
          </div>
        )}
      </Card>

      <ExplanationPanel anomalyId={anomaly.id} />
    </div>
  );
}