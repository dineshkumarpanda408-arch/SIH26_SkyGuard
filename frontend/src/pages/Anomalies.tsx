import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';
import type { Anomaly } from '../types';
import ModelVersionTag from '../components/ModelVersionTag';
import { PageHeader, Card, Badge, Spinner, ErrorState, Table, Th, Td, fmt } from '../components/ui';
import { IconAlert, IconFlask, IconCpu } from '../components/icons';
import { Button } from '../components/ui';

export default function Anomalies() {
  const [anomalies, setAnomalies] = useState<Anomaly[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [severity, setSeverity] = useState('');
  const [anomalyType, setAnomalyType] = useState('');

  useEffect(() => {
    api
      .anomalies({ severity: severity || undefined, anomaly_type: anomalyType || undefined })
      .then(setAnomalies)
      .catch((e) => setError(e.message));
  }, [severity, anomalyType]);

  if (error) return <ErrorState message={error} />;

  return (
    <div className="space-y-7">
      <PageHeader
        icon={<IconAlert size={20} />}
        title="Anomaly Investigation"
        subtitle="Detected anomalies across the AWS network"
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <ModelVersionTag />
            <Button to="/deep-dive" variant="outline" className="!px-3 !py-1.5 !text-xs flex items-center gap-1.5">
              <IconCpu size={14} className="text-sky-400" />
              <span>XAI Deep-Dive</span>
            </Button>
            <select value={severity} onChange={(e) => setSeverity(e.target.value)} className="select !py-1.5">
              <option value="">All Severity</option>
              <option value="HIGH">HIGH</option>
              <option value="MEDIUM">MEDIUM</option>
              <option value="LOW">LOW</option>
            </select>
            <select value={anomalyType} onChange={(e) => setAnomalyType(e.target.value)} className="select !py-1.5">
              <option value="">All Types</option>
              <option value="SPIKE">SPIKE</option>
              <option value="DROP">DROP</option>
              <option value="DRIFT">DRIFT</option>
              <option value="FROZEN_SENSOR">FROZEN_SENSOR</option>
              <option value="MISSING_DATA">MISSING_DATA</option>
            </select>
          </div>
        }
      />

      {!anomalies ? (
        <Spinner label="Loading anomalies\u2026" />
      ) : (
        <Card
          title={`Anomalies (${anomalies.length})`}
          subtitle="Click a row to investigate its model attribution"
        >
          {anomalies.length === 0 ? (
            <div className="text-slate-500 py-14 text-center text-sm">No anomalies match the current filters.</div>
          ) : (
            <Table>
              <thead>
                <tr className="border-b border-borderline/70">
                  <Th>ID</Th>
                  <Th>Time</Th>
                  <Th>Station</Th>
                  <Th>Type</Th>
                  <Th>Feature</Th>
                  <Th>Raw</Th>
                  <Th right>
                    <span title="Fused model score (uncalibrated \u2014 not a probability)">Anomaly Score</span>
                  </Th>
                  <Th right>Severity</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-borderline/50">
                {anomalies.map((a) => (
                  <tr
                    key={a.id}
                    onClick={() => (window.location.href = `/anomalies/${a.id}`)}
                    className="group cursor-pointer transition-colors hover:bg-sky-500/[0.05]"
                  >
                    <td className="px-4 py-3 font-mono text-[13px] text-slate-400 group-hover:text-sky-300 transition-colors">{a.id}</td>
                    <Td className="text-slate-400 whitespace-nowrap">{new Date(a.timestamp).toLocaleString()}</Td>
                    <Td className="font-semibold text-sky-300">{a.station_id}</Td>
                    <Td className="capitalize">{a.anomaly_type?.toLowerCase()}</Td>
                    <Td className="text-slate-400">{a.feature ?? '\u2014'}</Td>
                    <Td mono>{fmt(a.raw_value)}</Td>
                    <Td right mono>{fmt(a.score ?? a.confidence, 2)}</Td>
                    <Td right><Badge status={a.severity}>{a.severity}</Badge></Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </Card>
      )}

      <div className="flex items-center gap-2 text-xs text-slate-500">
        <IconFlask size={14} className="text-sky-500/60" />
        <span>
          Tip: Use <Link to="/simulation" className="text-sky-400 hover:text-sky-300 font-medium hover:underline">Simulation</Link> to inject known anomaly types and verify detection + attribution.
        </span>
      </div>
    </div>
  );
}