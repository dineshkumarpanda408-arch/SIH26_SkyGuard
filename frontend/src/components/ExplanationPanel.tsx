import { useEffect, useState } from 'react';
import { api } from '../api';
import type { Explanation, Contribution } from '../types';
import { Card, Spinner, ProgressBar } from './ui';
import { barWidth, signedPercent, topContributor } from '../attribution';
import { IconAlert, IconInfo } from './icons';

const FEATURES: Record<string, { label: string; color: string }> = {
  temperature: { label: 'Temperature', color: 'bg-red-500' },
  pressure: { label: 'Pressure', color: 'bg-sky-500' },
  humidity: { label: 'Humidity', color: 'bg-emerald-500' },
};

function Bar({ c, max, top }: { c: Contribution; max: number; top: boolean }) {
  const meta = FEATURES[c.feature] ?? {
    label: c.feature,
    color: 'bg-slate-500',
  };
  const width = barWidth(c, max);
  const pct = signedPercent(c);
  return (
    <div className={`rounded-xl p-3 ${top ? 'bg-sky-500/[0.08] ring-1 ring-inset ring-sky-500/25' : 'bg-soft'}`}>
      <div className="flex justify-between items-center text-xs mb-1.5">
        <span className={`capitalize font-medium ${top ? 'text-sky-200' : 'text-slate-300'}`}>
          {meta.label}
          {top && (
            <span className="ml-2 rounded-full bg-sky-500/20 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-sky-300 ring-1 ring-inset ring-sky-500/30">
              top driver
            </span>
          )}
        </span>
        <span className={`tabular-nums font-semibold ${c.contribution >= 0 ? 'text-red-400' : 'text-sky-400'}`}>
          {pct}
          {c.contribution >= 0 ? ' \u2191' : ' \u2193'}
        </span>
      </div>
      <ProgressBar value={width} color={meta.color} />
    </div>
  );
}

export default function ExplanationPanel({ anomalyId }: { anomalyId: number }) {
  const [data, setData] = useState<Explanation | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setData(null);
    setError(null);
    api
      .explanation(anomalyId)
      .then((e) => alive && setData(e))
      .catch((e) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, [anomalyId]);

  if (error)
    return (
      <div className="flex items-center gap-2 rounded-xl border border-borderline/70 bg-panel2/60 px-4 py-3 text-xs text-slate-400">
        <IconInfo size={14} className="shrink-0 text-slate-500" /> Could not load explanation: {error}
      </div>
    );
  if (!data) return <Spinner label="Computing model attribution\u2026" />;

  const max = Math.max(...data.values.map((v) => Math.abs(v.contribution)), 1e-6);
  const top = topContributor(data.values);
  const driver = top ? FEATURES[top.feature]?.label ?? top.feature : null;
  const isStatistical = data.method === 'statistical';
  const isShap = data.method === 'shap';
  const title = isStatistical
    ? 'Statistical Attribution \u2014 Why was this flagged?'
    : isShap
      ? 'SHAP Attribution \u2014 Why did the model flag this?'
      : 'Model Attribution \u2014 Why did the model flag this?';
  const subtitle = isStatistical
    ? 'Statistical fallback: per-variable deviation from recent history (z-score), not model output'
    : isShap
      ? 'SHAP values from the trained detector: per-feature SHAP contribution to the flagged reading (real model output)'
      : 'SHAP attribution from the trained detector: per-feature SHAP contribution to the flagged reading (real model output)';
  return (
    <Card title={title} subtitle={subtitle}>
      <div className="space-y-2.5 mb-5">
        {data.values.map((c) => (
          <Bar key={c.feature} c={c} max={max} top={top?.feature === c.feature} />
        ))}
      </div>
      <div className="border-t border-borderline/60 pt-4">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1.5">
          Model Narrative
          {driver && <span className="normal-case tracking-normal text-sky-400/80"> {'\u00b7'} primary driver: {driver}</span>}
        </div>
        {data.narrative ? (
          <p className="text-sm text-slate-200 leading-relaxed">{data.narrative}</p>
        ) : (
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <IconAlert size={14} /> No narrative generated for this anomaly.
          </div>
        )}
      </div>
    </Card>
  );
}