import { useEffect, useRef, useState } from 'react';
import { PageHeader, StatCard, Card, Badge, Spinner, ErrorState, Table, Th, Td } from '../components/ui';
import { IconActivity, IconZap, IconAlert } from '../components/icons';
import ModelVersionTag from '../components/ModelVersionTag';

interface LiveReading {
  type: 'reading' | 'snapshot';
  station_id?: string;
  timestamp?: string;
  reading_number?: number;
  temperature?: number | null;
  pressure?: number | null;
  humidity?: number | null;
  is_anomaly?: boolean | null;
  score?: number | null;
  anomaly_type?: string | null;
  severity?: string | null;
  feature?: string | null;
  demo?: boolean;
  stations?: unknown[];
}

function LivePill({ connected, error }: { connected: boolean; error: string | null }) {
  return (
    <div
      className={`flex items-center gap-2 rounded-xl px-3.5 py-1.5 text-xs font-bold uppercase tracking-wider ring-1 ring-inset ${
        connected
          ? 'text-green-300 bg-green-500/10 ring-green-500/25'
          : error
            ? 'text-red-300 bg-red-500/10 ring-red-500/25'
            : 'text-slate-300 bg-slate-500/10 ring-slate-500/25'
      }`}
    >
      <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400 animate-pulse-soft' : error ? 'bg-red-400' : 'bg-slate-400'}`} />
      {connected ? 'Live' : error ? 'Error' : 'Connecting\u2026'}
    </div>
  );
}

export default function Live() {
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [readings, setReadings] = useState<LiveReading[]>([]);
  const [counts, setCounts] = useState({ total: 0, anomalies: 0 });
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let alive = true;
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';

    function connect() {
      const defaultWs = `${proto}://${window.location.host}/api/live`;
      const targetWs = import.meta.env.VITE_WS_URL || defaultWs;
      const ws = new WebSocket(targetWs);
      wsRef.current = ws;
      ws.onopen = () => {
        if (alive) setConnected(true);
      };
      ws.onmessage = (ev) => {
        if (!alive) return;
        try {
          const msg: LiveReading = JSON.parse(ev.data);
          if (msg.type === 'reading') {
            setReadings((r) => [msg, ...r].slice(0, 60));
            setCounts((c) => ({
              total: c.total + 1,
              anomalies: c.anomalies + (msg.is_anomaly ? 1 : 0),
            }));
          }
        } catch {
          /* ignore malformed frames */
        }
      };
      ws.onerror = () => {
        if (alive) setError('WebSocket connection error.');
      };
      ws.onclose = () => {
        if (alive) {
          setConnected(false);
          setTimeout(connect, 3000);
        }
      };
    }
    connect();
    return () => {
      alive = false;
      wsRef.current?.close();
    };
  }, []);

  return (
    <div className="space-y-7">
      <PageHeader
        icon={<IconActivity size={20} />}
        title="Live Monitor"
        subtitle="Real-time AWS-023 telemetry streaming via WebSocket"
        actions={
          <div className="flex items-center gap-2">
            <ModelVersionTag />
            <LivePill connected={connected} error={error} />
          </div>
        }
      />

      <div className="grid grid-cols-3 gap-4 animate-fade-up">
        <StatCard label="Readings Received" value={counts.total} icon={<IconZap size={20} />} tone="sky" />
        <StatCard label="Anomalies Flagged" value={counts.anomalies} icon={<IconAlert size={20} />} tone="red" />
        <StatCard
          label="Healthy Readings"
          value={counts.total ? `${((1 - counts.anomalies / counts.total) * 100).toFixed(0)}%` : '\u2014'}
          icon={<IconActivity size={20} />}
          tone="emerald"
        />
      </div>

      <Card title="Live Telemetry Stream" subtitle="AWS-023 \u2014 latest values shown first">
        {error ? (
          <ErrorState message={error} />
        ) : readings.length === 0 ? (
          <Spinner label="Waiting for live data\u2026" />
        ) : (
          <Table>
            <thead>
              <tr className="border-b border-borderline/70">
                <Th>#</Th>
                <Th>Time</Th>
                <Th>Temp ({'\u00b0'}C)</Th>
                <Th>Pressure (hPa)</Th>
                <Th>Humidity (%)</Th>
                <Th>
                    <span title="Fused model score (uncalibrated \u2014 not a probability)">Anomaly Score</span>
                  </Th>
                <Th>Type</Th>
                <Th right>Status</Th>
              </tr>
            </thead>
            <tbody className="divide-y divide-borderline/50">
              {readings.map((r, i) => (
                <tr key={i} className={`transition-colors ${r.is_anomaly ? 'bg-red-500/[0.06] hover:bg-red-500/[0.1]' : 'hover:bg-soft'}`}>
                  <Td mono className={r.is_anomaly ? '!text-red-400 font-semibold' : 'text-slate-500'}>
                    {r.reading_number ?? '\u2014'}
                  </Td>
                  <Td className="text-slate-400 whitespace-nowrap">
                    {r.timestamp ? new Date(r.timestamp).toLocaleTimeString() : '\u2014'}
                  </Td>
                  <Td mono className={r.is_anomaly ? '!text-red-400 font-semibold' : ''}>
                    {r.temperature?.toFixed(1) ?? '\u2014'}
                  </Td>
                  <Td mono>{r.pressure?.toFixed(1) ?? '\u2014'}</Td>
                  <Td mono>{r.humidity?.toFixed(1) ?? '\u2014'}</Td>
                  <Td mono className="text-slate-400">{r.score?.toFixed(3) ?? '\u2014'}</Td>
                  <Td className="text-slate-300">
                    {r.is_anomaly ? (
                      <span className="flex items-center gap-1.5 font-semibold text-red-400">
                        {r.anomaly_type ?? 'Anomaly'}
                        {r.demo && (
                          <span className="rounded-full bg-slate-500/20 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-slate-400 ring-1 ring-inset ring-slate-500/30">
                            Demo
                          </span>
                        )}
                      </span>
                    ) : (
                      '\u2014'
                    )}
                  </Td>
                  <Td right>
                    {r.is_anomaly ? (
                      <Badge status="critical">{'\u{1F6A8} ANOMALY DETECTED'}</Badge>
                    ) : (
                      <Badge status="healthy">Normal</Badge>
                    )}
                  </Td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Card>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500">
        <span className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-green-400" /> System is pushing demo readings every few seconds.</span>
        <span>Score is the fused model score (uncalibrated \u2014 not a probability).</span>
        <span>Demo anomalies match an Anomalies-page record by station + type (timestamp is type-based, not exact).</span>
      </div>
    </div>
  );
}