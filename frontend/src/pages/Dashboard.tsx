import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';
import type { Station, Anomaly } from '../types';
import { groupRegionalWeatherEvents } from '../weatherEvents';
import ModelVersionTag from '../components/ModelVersionTag';
import {
  PageHeader,
  StatCard,
  Card,
  Badge,
  Spinner,
  StatusDot,
  Button,
  fmt,
} from '../components/ui';
import { IconGrid, IconPin, IconAlert, IconChart, IconCloud, IconArrowRight, IconChevronRight } from '../components/icons';

export default function Dashboard() {
  const [stations, setStations] = useState<Station[] | null>(null);
  const [anomalies, setAnomalies] = useState<Anomaly[] | null>(null);
  const [health, setHealth] = useState<{ status: string; system: string; model_trained: boolean } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [takingLonger, setTakingLonger] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setTakingLonger(true), 4000);
    Promise.all([api.stations(), api.anomalies(), api.health()])
      .then(([s, a, h]) => {
        setStations(s);
        setAnomalies(a);
        setHealth(h);
      })
      .catch((e) => setError(e.message))
      .finally(() => clearTimeout(timer));
    return () => clearTimeout(timer);
  }, []);

  if (error) return (
    <div className="space-y-4 max-w-xl mx-auto mt-12 p-6 rounded-2xl bg-panel2 border border-red-500/30 text-center">
      <div className="text-red-400 font-bold text-lg">⚠️ Backend Connection Error</div>
      <div className="text-slate-300 text-sm">{error}</div>
      <div className="text-slate-400 text-xs text-left bg-panel3 p-4 rounded-xl space-y-2 font-mono">
        <div>Target API URL: <span className="text-sky-400 font-bold">{import.meta.env.VITE_API_URL || '/api (default)'}</span></div>
        <div className="text-slate-500">Tip: If your backend is sleeping or spinning up on Render free tier, please wait 30 seconds and refresh.</div>
      </div>
      <button 
        onClick={() => window.location.reload()} 
        className="px-4 py-2 text-xs font-semibold rounded-xl bg-sky-500 hover:bg-sky-400 text-slate-950 transition-colors"
      >
        Retry Connection
      </button>
    </div>
  );

  if (!stations || !anomalies) return (
    <div className="flex flex-col items-center justify-center py-20 space-y-4">
      <Spinner label="Loading dashboard…" />
      {takingLonger && (
        <div className="max-w-md text-center p-4 rounded-xl bg-panel2/80 border border-amber-500/30 text-xs text-amber-200/90 space-y-1.5 animate-pulse">
          <div className="font-semibold text-amber-300">⏳ Waking up backend service...</div>
          <div>Render free tier web services spin down during inactivity. Initial cold-start takes 45–60 seconds.</div>
        </div>
      )}
    </div>
  );

  const high = anomalies.filter((a) => a.severity === 'HIGH').length;
  const medium = anomalies.filter((a) => a.severity === 'MEDIUM').length;
  const weather = anomalies.filter((a) => a.is_weather_event).length;
  const active = stations.filter((s) => s.active).length;

  const regionalGroups = groupRegionalWeatherEvents(anomalies);

  return (
    <div className="space-y-7">
      <PageHeader
        icon={<IconGrid size={20} />}
        title="Command Center"
        subtitle={`${health?.system ?? 'AI/ML'} anomaly detection across ${stations.length} automatic weather stations`}
        actions={
          <div className="flex items-center gap-3">
            <ModelVersionTag />
            <Button to="/simulation" variant="outline" className="!px-3 !py-1.5 !text-xs">
              Test the detector <IconArrowRight size={13} />
            </Button>
            <div className="flex items-center gap-2 rounded-xl px-3 py-1.5 text-xs font-semibold text-green-300 bg-green-500/10 ring-1 ring-inset ring-green-500/25">
              <StatusDot status={health?.model_trained ? 'online' : 'offline'} />
              {health?.model_trained ? 'Model Trained ' : 'Model Not Trained '}
              {'\u2014 '}
              {health?.status === 'online' ? 'Online' : 'Offline'}
            </div>
          </div>
        }
      />

      <div className="flex flex-col gap-3 rounded-2xl border border-borderline/70 bg-panel2/50 p-4 sm:flex-row sm:items-center sm:justify-between sm:gap-6 animate-fade-up">
        <p className="text-sm text-slate-300 leading-relaxed max-w-2xl">
          SkyGuard fuses trained isolation-forest detectors to score every reading. Deterministic rules then classify the
          anomaly type, severity and threshold decision, and statistics judge weather-vs-sensor cause and compute a corrected value.
        </p>
        <div className="flex flex-wrap items-center gap-2 text-[11px] shrink-0">
          <span className="rounded-full px-2.5 py-1 font-semibold text-violet-300 bg-violet-500/10 ring-1 ring-inset ring-violet-500/25">ML {'\u00b7'} detector fusion (scores)</span>
          <span className="rounded-full px-2.5 py-1 font-semibold text-sky-300 bg-sky-500/10 ring-1 ring-inset ring-sky-500/25">Rule {'\u00b7'} type, severity, threshold</span>
          <span className="rounded-full px-2.5 py-1 font-semibold text-emerald-300 bg-emerald-500/10 ring-1 ring-inset ring-emerald-500/25">Statistical {'\u00b7'} cause, correction</span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-4 animate-fade-up">
        <StatCard label="Active Stations" value={active} icon={<IconPin size={20} />} tone="sky" sub={`of ${stations.length} total`} />
        <StatCard label="Anomalies Detected" value={anomalies.length} icon={<IconAlert size={20} />} tone="red" sub="network-wide" />
        <StatCard label="High Severity" value={high} icon={<IconChart size={20} />} tone="violet" sub="needs review" />
        <StatCard label="Medium Severity" value={medium} icon={<IconAlert size={20} />} tone="amber" sub="monitor closely" />
        <StatCard label="Weather Events" value={weather} icon={<IconCloud size={20} />} tone="emerald" sub="of anomalies judged weather-related (statistical)" />
      </div>

      {regionalGroups.length > 0 && (
        <section className="space-y-3 animate-fade-up">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h3 className="text-[13px] font-semibold uppercase tracking-wider text-slate-300">Regional Weather Events</h3>
            <span className="text-[11px] text-slate-500">
              weather-side anomalies (LIKELY_WEATHER_EVENT) clustered in a short time window {'\u2014'} one card per regional event
            </span>
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            {regionalGroups.map((g) => (
              <div key={g.startTs} className="rounded-2xl border border-emerald-500/25 bg-emerald-500/[0.05] p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-xs font-bold uppercase tracking-wider text-emerald-300">
                      ONE regional weather event
                    </div>
                    <div className="text-[11px] text-slate-400 mt-1">
                      {new Date(g.startTs).toLocaleString()} {'\u2192'} {new Date(g.endTs).toLocaleString()}
                    </div>
                    <div className="text-[11px] text-slate-500 mt-0.5">
                      {g.anomalies.length} weather-related anomaly record{(g.anomalies.length === 1 ? '' : 's')} clustered
                      within a {Math.round((new Date(g.endTs).getTime() - new Date(g.startTs).getTime()) / 60000)} min window
                    </div>
                  </div>
                  <Badge status={g.severity === 'HIGH' ? 'high' : g.severity === 'MEDIUM' ? 'medium' : 'low'}>{g.severity}</Badge>
                </div>
                <div className="mt-3 pt-2 border-t border-borderline/40">
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1.5">
                    Affected stations ({g.stationCount})
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {g.stationIds.map((sid) => (
                      <Link
                        key={sid}
                        to={`/stations/${sid}`}
                        className="rounded-lg bg-panel3/70 ring-1 ring-inset ring-borderline px-2 py-1 text-xs font-semibold text-sky-300 transition-colors hover:text-sky-200 hover:ring-sky-500/30"
                      >
                        {sid}
                      </Link>
                    ))}
                  </div>
                  <div className="mt-3 pt-2 border-t border-borderline/40 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-500">
                    {g.anomalies.map((a) => (
                      <Link key={a.id} to={`/anomalies/${a.id}`} className="transition-colors hover:text-sky-300">
                        {a.station_id} {'\u00b7'} {(a.anomaly_type ?? 'anomaly').toLowerCase()}
                        {a.feature ? ` \u00b7 ${a.feature}` : ''}
                      </Link>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <div className="grid lg:grid-cols-2 gap-6">
        <Card
          title="Network Map"
          subtitle="Automatic Weather Stations"
          right={<StatusDot status={active ? 'online' : 'offline'} />}
        >
          <div className="grid grid-cols-3 md:grid-cols-4 gap-2.5 max-h-80 overflow-auto pr-1">
            {stations.map((s) => (
              <Link
                key={s.station_id}
                to={`/stations/${s.station_id}`}
                className="group rounded-xl border border-borderline/70 bg-panel3/40 hover:bg-panel3 hover:border-sky-500/30 p-3 text-center transition-all duration-150"
              >
                <div className="text-sm font-bold text-sky-300 group-hover:text-sky-200">{s.station_id}</div>
                <div className="text-[11px] text-slate-500 truncate mt-0.5">{s.name ?? '\u2014'}</div>
                <div className="mt-2 flex justify-center">
                  <StatusDot status={s.active ? 'online' : 'offline'} pulse={false} />
                </div>
              </Link>
            ))}
          </div>
        </Card>

        <Card
          title="Recent Anomalies"
          subtitle="Latest detections across the network"
          right={<Button to="/anomalies" variant="ghost" className="!px-2.5 !py-1 !text-xs">View all <IconArrowRight size={14} /></Button>}
        >
          {anomalies.length === 0 ? (
            <div className="text-sm text-slate-500 py-10 text-center">No anomalies recorded yet.</div>
          ) : (
            <ul className="divide-y divide-borderline/60 max-h-80 overflow-auto">
              {anomalies.slice(0, 12).map((a) => (
                <li key={a.id} className="py-2.5 -mx-2 px-2 rounded-lg hover:bg-soft transition-colors">
                  <Link to={`/anomalies/${a.id}`} className="flex items-center justify-between gap-3 group">
                    <div className="min-w-0 flex items-center gap-3">
                      <span className="w-1.5 h-6 rounded-full bg-gradient-to-b from-red-500/70 to-rose-500/70 shrink-0" />
                      <div className="min-w-0">
                        <div className="text-xs text-slate-500">
                          {new Date(a.timestamp).toLocaleString()}{' \u00b7 '}
                          <span className="text-slate-300">{a.station_id}</span>
                        </div>
                        <div className="text-sm font-medium capitalize text-sky-300 group-hover:text-sky-200 truncate">
                          {a.anomaly_type?.toLowerCase()} {a.feature ? `on ${a.feature}` : ''}
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

      <div className="flex items-center justify-center gap-1 text-xs text-slate-500">
        <IconChevronRight size={14} className="text-sky-500/60" />
        Explore a station or anomaly to see model attribution behind every flag.
      </div>
    </div>
  );
}