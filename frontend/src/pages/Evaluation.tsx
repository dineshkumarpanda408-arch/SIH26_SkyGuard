import { useEffect, useState } from 'react';
import { api } from '../api';
import type { ModelEvaluation } from '../types';
import { PageHeader, Card, Badge, Spinner, ErrorState, StatCard, Table, Th, Td, fmt } from '../components/ui';
import { IconGauge, IconCheck, IconAlert, IconRadar } from '../components/icons';
import ModelVersionTag from '../components/ModelVersionTag';

function ConfusionMatrix({ grid }: { grid: number[][]; labels?: string[] }) {
  const [tn, fp, fn, tp] = [grid[0][0], grid[0][1], grid[1][0], grid[1][1]];
  const cells = [
    { label: 'TP', value: tp, cls: 'bg-emerald-500/[0.12] text-emerald-300', tooltip: 'Correctly detected anomalies' },
    { label: 'FP', value: fp, cls: 'bg-red-500/[0.1] text-red-300', tooltip: 'Normal readings flagged as anomalies' },
    { label: 'FN', value: fn, cls: 'bg-amber-500/[0.1] text-amber-300', tooltip: 'Missed anomalies' },
    { label: 'TN', value: tn, cls: 'bg-sky-500/[0.1] text-sky-300', tooltip: 'Correctly normal readings' },
  ];
  return (
    <div className="grid grid-cols-2 gap-3">
      {cells.map((c) => (
        <div key={c.label} title={c.tooltip} className={`rounded-xl px-4 py-3 flex items-center justify-between ring-1 ring-inset ${c.cls}`}>
          <span className="text-lg font-extrabold tabular-nums">{c.value}</span>
          <span className="text-[10px] font-bold uppercase tracking-wider opacity-80">{c.label}</span>
        </div>
      ))}
    </div>
  );
}

function MultiClassMatrix({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  return (
    <Table>
      <thead>
        <tr className="border-b border-borderline/70">
          <Th>True \\ Pred</Th>
          {labels.map((l) => (
            <Th key={l} className="text-center">{l.toLowerCase().replace(/_/g, ' ')}</Th>
          ))}
        </tr>
      </thead>
      <tbody className="divide-y divide-borderline/40">
        {matrix.map((row, r) => (
          <tr key={labels[r]}>
            <Td className="font-medium capitalize">{labels[r].toLowerCase().replace(/_/g, ' ')}</Td>
            {row.map((cell, c) => {
              const diag = r === c;
              return (
                <td
                  key={c}
                  className={`px-4 py-2.5 text-center text-sm tabular-nums ${diag ? 'text-emerald-400 font-extrabold' : 'text-slate-400'}`}
                >
                  {cell}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </Table>
  );
}

export default function Evaluation() {
  const [data, setData] = useState<ModelEvaluation | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .modelEvaluation()
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <ErrorState message={error} />;
  if (!data) return <Spinner label="Running evaluation\u2026" />;

  const d = data.detailed ?? data.performance;
  const matrix = d.confusion_matrix?.grid ?? [[d.tn, d.fp], [d.fn, d.tp]];
  const perCategory = data.per_category ?? {};
  const dataset = data.dataset;

  return (
    <div className="space-y-7">
      <PageHeader
        icon={<IconGauge size={20} />}
        title="Controlled Ground-Truth Evaluation"
        subtitle="Metrics computed on a synthetic injected evaluation frame built from the same 24 AWS stations (not a separate live network)"
        actions={<ModelVersionTag />}
      />

      <div className="flex flex-col gap-3 rounded-2xl border border-borderline/70 bg-panel2/50 p-4 text-xs text-slate-400 leading-relaxed animate-fade-up">
        <p>
          This page shows <strong className="text-slate-300">three different, deliberately separate</strong> measurements and they are{' '}
          <strong className="text-slate-300">not directly comparable</strong>:
          <span className="text-slate-500 ml-1">(1) the ablation / research comparison scores each standalone analyser on its own feature
          set; (2) the detailed production pipeline scores the fused v3 production model; (3) the version comparison scores the
          persisted artifacts on the identical frame. These are different scoring paths — all numbers below are computed live
          from the backend evaluation, never hardcoded.</span>
        </p>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <span className="text-slate-400">Badge below = ablation winner (standalone analysers).</span>
          <span className="text-slate-400">Stat cards below = fused production pipeline (v3), same frame.</span>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3 animate-fade-up">
        <Badge status="info">Selected: {data.selected_model}</Badge>
        {dataset && (
          <Badge status="default">
            {dataset.observations.toLocaleString()} pts {'\u00b7'} {dataset.ground_truth_anomalies} anomalies (
            {fmt(dataset.anomaly_rate * 100, 1)}%) {'\u00b7'} {dataset.injections} injections
          </Badge>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 animate-fade-up">
        <StatCard label="Recall" value={`${fmt((d.recall as number) * 100, 1)}%`} icon={<IconCheck size={20} />} tone="emerald" />
        <StatCard label="Precision" value={`${fmt((d.precision as number) * 100, 1)}%`} icon={<IconGauge size={20} />} tone="sky" />
        <StatCard label="F1 Score" value={`${fmt((d.f1 as number) * 100, 1)}%`} icon={<IconAlert size={20} />} tone="violet" />
        <StatCard
          label="ROC-AUC"
          value={d.roc_auc != null ? `${fmt((d.roc_auc as number) * 100, 1)}%` : 'N/A'}
          icon={<IconRadar size={20} />}
          tone="indigo"
        />
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <Card
          title="Detailed Binary Confusion Matrix — Production Pipeline (v3)"
          subtitle={`n = ${d.n} \u00b7 Accuracy ${fmt(d.accuracy * 100, 1)}% \u00b7 fused production scores on the same frame as the version comparison`}
        >
          <ConfusionMatrix grid={matrix} />
          <div className="mt-4 pt-4 border-t border-borderline/60 grid grid-cols-2 gap-3">
            <div className="flex items-center justify-between rounded-xl border border-borderline/70 bg-panel3/40 px-4 py-2.5 text-sm">
              <span className="text-xs text-slate-500">Anomaly rate</span>
              <span className="font-semibold tabular-nums text-slate-100">{fmt(((matrix[0][1] + matrix[1][1]) / d.n) * 100, 1)}%</span>
            </div>
            <div className="flex items-center justify-between rounded-xl border border-borderline/70 bg-panel3/40 px-4 py-2.5 text-sm">
              <span className="text-xs text-slate-500">Overall accuracy</span>
              <span className="font-semibold tabular-nums text-emerald-300">{fmt(d.accuracy * 100, 1)}%</span>
            </div>
          </div>
        </Card>

        <Card title="Per-Category Detection Recall" subtitle="Ground-truth injected anomaly recovery by type">
          <div className="overflow-x-auto">
            <table className="w-full text-sm min-w-[360px]">
              <thead>
                <tr className="text-left text-[11px] uppercase tracking-wider text-slate-500 border-b border-borderline/70">
                  <th className="py-3 text-left">Type</th>
                  <th className="py-3 text-right">Injected</th>
                  <th className="py-3 text-right">Detected</th>
                  <th className="py-3 text-right">Recall</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-borderline/50">
                {Object.entries(perCategory)
                  .filter(([k]) => k !== 'NORMAL')
                  .map(([k, v]) => {
                  const pct = v.recall * 100;
                  return (
                    <tr key={k} className="hover:bg-soft transition-colors">
                      <Td className="capitalize">{k.toLowerCase().replace(/_/g, ' ')}</Td>
                      <Td right mono>{v.injected}</Td>
                      <Td right mono>{v.detected}</Td>
                      <td className="px-4 py-3 text-right">
                        <span
                          className={`inline-block rounded-lg px-2 py-0.5 text-xs font-bold tabular-nums ring-1 ring-inset ${
                            pct >= 80
                              ? 'text-emerald-300 bg-emerald-500/10 ring-emerald-500/25'
                              : pct >= 40
                                ? 'text-amber-300 bg-amber-500/10 ring-amber-500/25'
                                : 'text-red-300 bg-red-500/10 ring-red-500/25'
                          }`}
                        >
                          {fmt(v.recall * 100, 1)}%
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      <Card
        title="Ablation / Research Comparison"
        subtitle="Same injected frame, each standalone analyser scored on its own feature set (research path — not the fused production score)"
      >
        <Table>
          <thead>
            <tr className="border-b border-borderline/70">
              <Th>Analysis</Th>
              <Th right>Precision</Th>
              <Th right>Recall</Th>
              <Th right>F1</Th>
              <Th right>ROC-AUC</Th>
            </tr>
          </thead>
          <tbody className="divide-y divide-borderline/50">
            {data.comparison.map((r) => (
              <tr key={r.name}>
                <Td className="font-medium text-slate-100">{r.name}</Td>
                <Td right mono>{fmt(r.precision * 100, 1)}%</Td>
                <Td right mono>{fmt(r.recall * 100, 1)}%</Td>
                <Td right mono>{fmt(r.f1 * 100, 1)}%</Td>
                <Td right mono>{r.roc_auc != null ? fmt(r.roc_auc * 100, 1) + '%' : 'N/A'}</Td>
              </tr>
            ))}
          </tbody>
        </Table>
        <p className="text-[11px] text-slate-500 mt-3">
          Standalone analysers measure a different thing from the fused production pipeline (their ablation
          numbers differ from the production cards by design). Do not read these rows as production metrics.
        </p>
      </Card>

      <Card
        title="Root-Cause Diagnosis Performance"
        subtitle="True anomaly type vs root-cause diagnosis (diagonal = correct diagnosis)"
      >
        <p className="text-xs text-slate-500 mb-4">
          This validates root-cause classification, not just binary detection. Diagnoses are computed by the
          SkyGuard classifier for each anomalous reading.
        </p>
        <MultiClassMatrix labels={data.multiclass_confusion_matrix.labels} matrix={data.multiclass_confusion_matrix.matrix} />
      </Card>

      {(data.model_versions?.comparison && data.model_versions.comparison.length > 0) && (
        <Card
          title="Version Comparison (measured \u2014 persisted artifacts, identical frame)"
          subtitle="Each persisted version scored on the identical hard injected evaluation frame at its own artifact threshold \u2014 real detector outputs from the persisted artifacts"
        >
          {data.model_versions.note && <p className="text-xs text-slate-500 mb-3">{data.model_versions.note}</p>}
          <Table>
            <thead>
              <tr className="border-b border-borderline/70">
                <Th>Version</Th>
                <Th right>Precision</Th>
                <Th right>Recall</Th>
                <Th right>F1</Th>
                <Th right>ROC-AUC</Th>
              </tr>
            </thead>
            <tbody className="divide-y divide-borderline/50">
              {data.model_versions.comparison.map((r) => (
                <tr key={r.label}>
                  <Td className="font-medium text-slate-100">{r.label}</Td>
                  <Td right mono>{fmt(r.precision * 100, 1)}%</Td>
                  <Td right mono>{fmt(r.recall * 100, 1)}%</Td>
                  <Td right mono>{fmt(r.f1 * 100, 1)}%</Td>
                  <Td right mono>{r.roc_auc != null ? fmt(r.roc_auc * 100, 1) + '%' : 'N/A'}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
          {data.comparison_provenance && (
            <p className="text-[11px] text-slate-500 mt-3">
              provenance: {JSON.stringify(data.comparison_provenance).slice(0, 240)}
            </p>
          )}
        </Card>
      )}
    </div>
  );
}