import { useEffect, useState } from 'react';
import { api } from '../api';
import type { ModelMetadata } from '../types';
import { PageHeader, Card, Spinner, ErrorState, Badge } from '../components/ui';
import { IconCpu, IconTag } from '../components/icons';

export default function Model() {
  const [meta, setMeta] = useState<ModelMetadata | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .modelMetadata()
      .then(setMeta)
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <ErrorState message={error} />;
  if (!meta) return <Spinner label="Loading model metadata\u2026" />;

  const rows: { label: string; value?: string | number | null; badge?: boolean }[] = [
    { label: 'Model Version', value: meta.model_version },
    { label: 'Algorithm', value: meta.algorithm },
    { label: 'Production Threshold', value: meta.threshold },
    { label: 'Data Source', value: meta.data_source, badge: true },
    { label: 'Training Dataset', value: meta.dataset_file },
    { label: 'Dataset Version', value: meta.dataset_version },
    {
      label: 'Dataset SHA-256',
      value: meta.dataset_sha256 ? `${String(meta.dataset_sha256).slice(0, 16)}\u2026` : null,
    },
    { label: 'Calibration', value: (meta.calibration?.status ?? 'uncalibrated'), badge: true },
    { label: 'Training Timestamp', value: meta.trained_at_utc ? new Date(meta.trained_at_utc).toLocaleString() : null },
    { label: 'Training Stations', value: meta.n_training_stations },
    { label: 'Stations Covered', value: meta.n_stations },
    { label: 'Variables Covered', value: meta.n_variables },
    { label: 'Total Observations', value: meta.n_observations },
    { label: 'Training Observations', value: meta.n_training_obs },
    { label: 'Validation Observations', value: meta.n_validation_obs },
    { label: 'Test Observations', value: meta.n_test_obs },
    { label: 'Feature Count', value: meta.n_features },
    { label: 'Synthetic Telemetry', value: meta.synthetic ? 'yes' : 'no' },
    {
      label: 'Train / Val / Test Split',
      value: meta.split ? `${meta.split.train} / ${meta.split.validation} / ${meta.split.test}` : null,
    },
  ];

  return (
    <div className="space-y-7">
      <PageHeader
        icon={<IconCpu size={20} />}
        title="Model Information"
        subtitle={
          meta.data_source === 'HISTORICAL'
            ? 'Historical dataset used for model development (not live AWS).'
            : `Data source: ${meta.data_source}`
        }
      />

      <Card title="Production Model Metadata" subtitle="From the persisted trained artifact">
        <dl className="grid md:grid-cols-2 gap-x-10 gap-y-2">
          {rows.map((r) => (
            <div key={r.label} className="flex items-center justify-between gap-4 border-b border-borderline/40 py-2.5">
              <dt className="text-sm text-slate-400">{r.label}</dt>
              <dd className="text-right text-sm text-slate-100 font-mono">
                {r.badge && r.value ? <Badge status={String(r.value).toLowerCase()}>{r.value}</Badge> : (r.value ?? '\u2014')}
              </dd>
            </div>
          ))}
        </dl>
      </Card>

      <Card title="Feature Set" subtitle={`${meta.n_features ?? 0} engineered features`}>
        {meta.features && meta.features.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {meta.features.map((f) => (
              <span
                key={f}
                className="inline-flex items-center gap-1.5 rounded-lg border border-borderline/70 bg-panel3/50 px-2.5 py-1 text-xs font-medium text-slate-300"
              >
                <IconTag size={13} className="text-sky-400/70" />
                {f}
              </span>
            ))}
          </div>
        ) : (
          <p className="text-xs text-slate-500">No feature list recorded in metadata.</p>
        )}
      </Card>

      {meta.notes && <p className="text-xs text-slate-500 px-1">{meta.notes}</p>}
      {(meta.calibration?.status ?? 'uncalibrated') !== 'calibrated' && (
        <p className="text-xs text-slate-500 px-1">
          Scores are uncalibrated: the anomaly score is a fused model score, not a probability.
        </p>
      )}
    </div>
  );
}