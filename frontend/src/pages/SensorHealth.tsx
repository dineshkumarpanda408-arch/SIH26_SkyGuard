import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';
import type { Station, SensorHealth } from '../types';
import { PageHeader, Badge, Spinner, ErrorState, StatusDot, ProgressBar, fmt } from '../components/ui';
import { IconHeart, IconChevronRight } from '../components/icons';

export default function SensorHealth() {
  const [stations, setStations] = useState<Station[] | null>(null);
  const [health, setHealth] = useState<Record<string, SensorHealth>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api
      .stations()
      .then(async (s) => {
        if (!alive) return;
        setStations(s);
        const entries = await Promise.all(
          s.map(async (st) => {
            try {
              return [st.station_id, await api.sensorHealth(st.station_id)] as const;
            } catch {
              return null;
            }
          }),
        );
        if (alive) setHealth(Object.fromEntries(entries.filter((e): e is readonly [string, SensorHealth] => e !== null)));
      })
      .catch((e) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, []);

  if (error) return <ErrorState message={error} />;
  if (!stations) return <Spinner label="Loading sensor health\u2026" />;

  const sorted = [...stations].sort((a, b) => {
    const ha = health[a.station_id]?.health_score ?? 1;
    const hb = health[b.station_id]?.health_score ?? 1;
    return ha - hb;
  });

  const avg = sorted.length ? Math.round(sorted.reduce((acc, s) => acc + (health[s.station_id]?.health_score ?? 0), 0) / sorted.length) : 0;
  const worst = sorted.filter((s) => (health[s.station_id]?.health_score ?? 100) < 60).length;

  return (
    <div className="space-y-7">
      <PageHeader
        icon={<IconHeart size={20} />}
        title="Sensor Health"
        subtitle="Real health scores computed from each station's own data (worst first). Sub-scores are /100 where 100 = best."
      />

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 animate-fade-up">
        <div className="rounded-2xl border border-borderline/80 bg-gradient-to-b from-panel2 to-panel shadow-card p-4 sm:p-5">
          <div className="text-2xl sm:text-3xl font-extrabold tracking-tight tabular-nums text-sky-300">{sorted.length}</div>
          <div className="text-xs text-slate-400 mt-0.5">Stations monitored</div>
        </div>
        <div className="rounded-2xl border border-borderline/80 bg-gradient-to-b from-panel2 to-panel shadow-card p-4 sm:p-5">
          <div className="text-2xl sm:text-3xl font-extrabold tracking-tight tabular-nums text-emerald-300">{avg}</div>
          <div className="text-xs text-slate-400 mt-0.5">Avg health score</div>
        </div>
        <div className="rounded-2xl border border-borderline/80 bg-gradient-to-b from-panel2 to-panel shadow-card p-4 sm:p-5">
          <div className="text-2xl sm:text-3xl font-extrabold tracking-tight tabular-nums text-amber-300">{worst}</div>
          <div className="text-xs text-slate-400 mt-0.5">Stations need attention</div>
        </div>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        {sorted.map((s) => {
          const h = health[s.station_id];
          const tone = h ? (h.health_score > 70 ? 'from-green-400 to-emerald-500' : h.health_score > 40 ? 'from-yellow-400 to-amber-500' : 'from-red-400 to-rose-500') : 'bg-sky-500';
          return (
            <Link key={s.station_id} to={`/stations/${s.station_id}`} className="group block">
              <div className="rounded-2xl border border-borderline/80 bg-gradient-to-b from-panel2 to-panel shadow-card p-4 transition-all duration-200 group-hover:border-sky-500/30 group-hover:shadow-cardHover">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2.5 min-w-0">
                    <span className="w-8 h-8 shrink-0 rounded-lg bg-sky-500/10 ring-1 ring-inset ring-sky-500/25 text-sky-300 flex items-center justify-center group-hover:bg-sky-500/15 transition-colors">
                      <IconHeart size={15} />
                    </span>
                    <span className="font-bold text-slate-100 group-hover:text-sky-300 transition-colors truncate">{s.station_id}</span>
                  </div>
                  <StatusDot status={h?.health_status ?? 'online'} pulse={false} />
                </div>
                <div className="text-xs text-slate-500 mb-3 truncate">{s.name ?? '\u2014'}</div>
                {h ? (
                  <>
                    <ProgressBar value={h.health_score} color={`bg-gradient-to-r ${tone}`} />
                    <div className="flex items-center justify-between mt-2">
                      <span className="text-lg font-extrabold tabular-nums text-slate-50">{fmt(h.health_score, 0)}<span className="text-xs font-normal text-slate-500">/100</span></span>
                      <Badge status={h.health_status}>{h.health_status}</Badge>
                    </div>
                    <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs pt-3 border-t border-borderline/60">
                      <span className="text-slate-500">Anomaly-Free Score</span>
                      <span className="text-right tabular-nums text-slate-300">{fmt(h.anomaly_frequency, 0)}/100</span>
                      <span className="text-slate-500">Drift Health</span>
                      <span className="text-right tabular-nums text-slate-300">{fmt(h.drift, 0)}/100</span>
                    </div>
                    <div className="flex items-center justify-end gap-1 mt-2 text-sky-400/0 group-hover:text-sky-400 transition-colors text-[11px] font-medium">
                      Inspect station <IconChevronRight size={12} />
                    </div>
                  </>
                ) : (
                  <div className="text-xs text-slate-500 py-3">No health metric available.</div>
                )}
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}