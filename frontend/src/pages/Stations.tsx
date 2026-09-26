import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import { api } from '../api';
import type { Station, StationPrediction } from '../types';
import { PageHeader, Card, Spinner, ErrorState, StatusDot } from '../components/ui';
import { IconPin, IconChevronRight } from '../components/icons';

const icon = L.icon({
  iconUrl:
    'data:image/svg+xml;base64,' +
    btoa(
      '<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30"><path d="M15 0C7 0 1 6 1 14c0 10 14 20 14 20s14-10 14-20C29 6 23 0 15 0zm0 19a5 5 0 110-10 5 5 0 010 10z" fill="#38bdf8" stroke="#0369a1"/></svg>',
    ),
  iconSize: [30, 30],
  iconAnchor: [15, 30],
});

export default function Stations() {
  const [stations, setStations] = useState<Station[] | null>(null);
  const [predictions, setPredictions] = useState<Record<string, StationPrediction>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .stations()
      .then(async (s) => {
        setStations(s);
        setPredictions(Object.fromEntries((await api.predictions()).map((p) => [p.station_id, p])));
      })
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <ErrorState message={error} />;
  if (!stations) return <Spinner label="Loading stations\u2026" />;

  const statusOf = (sid: string): string => predictions[sid]?.status ?? 'UNAVAILABLE';
  const mapped = stations.filter((s) => s.latitude !== null && s.longitude !== null);

  return (
    <div className="space-y-7">
      <PageHeader
        icon={<IconPin size={20} />}
        title="Weather Station Network"
        subtitle={`${stations.length} automatic weather stations across Odisha`}
      />

      <Card title="Station Map" subtitle={`${mapped.length} stations with coordinates`}>
        <div className="h-[420px] rounded-xl overflow-hidden shadow-card">
          <MapContainer
            center={[20.2, 84.5]}
            zoom={6}
            style={{ height: '100%', width: '100%', background: '#0b1220' }}
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; OpenStreetMap contributors'
            />
            {mapped.map((s) => (
              <Marker
                key={s.station_id}
                position={[s.latitude as number, s.longitude as number]}
                icon={icon}
              >
                <Popup>
                  <div className="font-bold">{s.station_id}</div>
                  <div className="text-xs text-slate-400">{s.name}</div>
                  <Link to={`/stations/${s.station_id}`} className="text-xs text-sky-400 underline mt-1 inline-block">
                    View details {'\u2192'}
                  </Link>
                </Popup>
              </Marker>
            ))}
          </MapContainer>
        </div>
      </Card>

      <Card title="Station List" subtitle="Click a station to inspect health and anomalies">
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
          {stations.map((s) => (
            <Link
              key={s.station_id}
              to={`/stations/${s.station_id}`}
              className="group flex items-center justify-between gap-3 rounded-xl border border-borderline/70 bg-panel3/40 px-4 py-3.5 hover:border-sky-500/30 hover:bg-panel3 transition-all duration-150"
            >
              <div className="min-w-0 flex items-center gap-3">
                <span className="w-8 h-8 shrink-0 rounded-lg bg-sky-500/10 ring-1 ring-inset ring-sky-500/25 text-sky-300 flex items-center justify-center">
                  <IconPin size={15} />
                </span>
                <div className="min-w-0">
                  <div className="font-bold text-slate-100 group-hover:text-sky-300 transition-colors">{s.station_id}</div>
                  <div className="text-xs text-slate-500 truncate">{s.name ?? '\u2014'}</div>
                </div>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <StatusDot status={statusOf(s.station_id)} pulse={false} />
                <IconChevronRight size={16} className="text-slate-600 group-hover:text-sky-400 transition-colors" />
              </div>
            </Link>
          ))}
        </div>
      </Card>
    </div>
  );
}