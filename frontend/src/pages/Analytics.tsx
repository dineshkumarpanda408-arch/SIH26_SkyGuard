import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
  Legend,
} from 'recharts';
import { Link } from 'react-router-dom';
import { useAnalytics } from '../hooks/useAnalytics';
import type { ProductionAnalytics, ProductionStationRow } from '../types';
import {
  PageHeader,
  Card,
  Spinner,
  ErrorState,
  StatCard,
  StatusDot,
  Badge,
  Table,
  Th,
  Td,
  Button,
  fmt,
} from '../components/ui';
import {
  IconChart,
  IconPin,
  IconAlert,
  IconHeart,
  IconSettings,
  IconRefresh,
  IconActivity,
  IconArrowRight,
} from '../components/icons';

const STATUS_CONFIG: Record<string, { label: string; color: string }> = {
  NORMAL: { label: 'Normal', color: '#22c55e' },
  ANOMALY: { label: 'Anomalous', color: '#ef4444' },
  STALE: { label: 'Stale', color: '#eab308' },
  UNAVAILABLE: { label: 'Unavailable', color: '#64748b' },
};

const SEVERITY_COLOR: Record<string, { bg: string; border: string; text: string }> = {
  HIGH: { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-300' },
  MEDIUM: { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-300' },
  LOW: { bg: 'bg-blue-500/10', border: 'border-blue-500/30', text: 'text-blue-300' },
};

// --- Child Components ---

interface AnalyticsHeaderProps {
  totalStations: number;
  dataSource: 'LIVE' | 'HISTORICAL';
  isRefreshing: boolean;
  onSelectSource: (source: 'LIVE' | 'HISTORICAL') => void;
  onRefresh: () => void;
}

function AnalyticsHeader({
  totalStations,
  dataSource,
  isRefreshing,
  onSelectSource,
  onRefresh,
}: AnalyticsHeaderProps) {
  return (
    <PageHeader
      icon={<IconChart size={20} />}
      title="Analytics"
      subtitle={`Production network aggregates from ${totalStations} AWS stations (data source: ${dataSource})`}
      actions={
        <div className="flex items-center gap-3">
          {/* Data Source Selector */}
          <div className="inline-flex rounded-xl p-1 bg-panel3/70 border border-borderline/80">
            <button
              type="button"
              onClick={() => onSelectSource('HISTORICAL')}
              className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
                dataSource === 'HISTORICAL'
                  ? 'bg-sky-500 text-slate-900 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              HISTORICAL
            </button>
            <button
              type="button"
              onClick={() => onSelectSource('LIVE')}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
                dataSource === 'LIVE'
                  ? 'bg-emerald-500 text-slate-900 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <span className="w-2 h-2 rounded-full bg-slate-900 animate-pulse-soft" />
              LIVE
            </button>
          </div>

          {/* Refresh Button */}
          <Button
            variant="outline"
            onClick={onRefresh}
            disabled={isRefreshing}
            className="!px-3 !py-1.5 !text-xs flex items-center gap-1.5"
          >
            <IconRefresh size={14} className={isRefreshing ? 'animate-spin text-sky-400' : ''} />
            <span>{isRefreshing ? 'Updating\u2026' : 'Refresh'}</span>
          </Button>
        </div>
      }
    />
  );
}

interface KpiCardsProps {
  totalStations: number;
  statusDist: Record<string, number>;
}

function KpiCards({ totalStations, statusDist }: KpiCardsProps) {
  const normal = statusDist.NORMAL ?? 0;
  const anomalous = statusDist.ANOMALY ?? 0;
  const stale = statusDist.STALE ?? 0;
  const unavailable = statusDist.UNAVAILABLE ?? 0;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 animate-fade-up">
      <StatCard
        label="Total Stations"
        value={totalStations}
        icon={<IconPin size={20} />}
        tone="sky"
        sub={`${normal} healthy normal`}
      />
      <StatCard
        label="Anomalous"
        value={anomalous}
        icon={<IconAlert size={20} />}
        tone="red"
        sub={anomalous > 0 ? 'requires investigation' : 'no active alerts'}
      />
      <StatCard
        label="Stale"
        value={stale}
        icon={<IconSettings size={20} />}
        tone="amber"
        sub={stale > 0 ? 'delayed telemetry' : 'all streams current'}
      />
      <StatCard
        label="Unavailable"
        value={unavailable}
        icon={<IconActivity size={20} />}
        tone="slate"
        sub={unavailable > 0 ? 'offline stations' : 'all stations active'}
      />
    </div>
  );
}

interface StatusPieCardProps {
  statusDist: Record<string, number>;
}

function StatusPieCard({ statusDist }: StatusPieCardProps) {
  const total = Object.values(statusDist).reduce((sum, v) => sum + v, 0);

  const statusPie = Object.entries(statusDist)
    .filter(([, v]) => v > 0)
    .map(([key, value]) => {
      const cfg = STATUS_CONFIG[key] || { label: key, color: '#38bdf8' };
      const pct = total > 0 ? ((value / total) * 100).toFixed(1) : '0';
      return {
        name: cfg.label,
        key,
        value,
        color: cfg.color,
        pct: `${pct}%`,
      };
    });

  return (
    <Card title="Station Status Distribution" subtitle="Proportion of stations across the network">
      {statusPie.length === 0 ? (
        <div className="text-sm text-slate-500 py-10 text-center">No station status data available.</div>
      ) : (
        <div className="space-y-4">
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie
                data={statusPie}
                dataKey="value"
                nameKey="name"
                innerRadius={50}
                outerRadius={85}
                paddingAngle={3}
                stroke="none"
              >
                {statusPie.map((d) => (
                  <Cell key={d.key} fill={d.color} />
                ))}
              </Pie>
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const item = payload[0].payload;
                    return (
                      <div className="rounded-xl border border-borderline bg-panel2 px-3 py-2 text-xs shadow-xl">
                        <div className="font-semibold text-slate-200">{item.name}</div>
                        <div className="text-slate-400 mt-0.5">
                          {item.value} stations ({item.pct})
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend
                verticalAlign="bottom"
                formatter={(val) => <span className="text-xs text-slate-300 ml-1">{val}</span>}
              />
            </PieChart>
          </ResponsiveContainer>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-borderline/50 text-center">
            {statusPie.map((s) => (
              <div key={s.key} className="p-2 rounded-lg bg-panel3/30 border border-borderline/40">
                <div className="text-xs text-slate-400">{s.name}</div>
                <div className="text-lg font-bold text-slate-100 mt-0.5">{s.value}</div>
                <div className="text-[11px] font-mono text-slate-500">{s.pct}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

interface AnomalyTypeBarCardProps {
  typeDist: Record<string, number>;
}

function formatTypeName(raw: string): string {
  return raw
    .replace(/_/g, ' ')
    .toLowerCase()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function AnomalyTypeBarCard({ typeDist }: AnomalyTypeBarCardProps) {
  const total = Object.entries(typeDist)
    .filter(([k]) => k !== 'NORMAL' && k !== 'NONE')
    .reduce((sum, [, v]) => sum + v, 0);

  const typeData = Object.entries(typeDist)
    .filter(([k]) => k !== 'NORMAL' && k !== 'NONE')
    .map(([k, v]) => ({
      name: formatTypeName(k),
      count: v,
      rawKey: k,
      pct: total > 0 ? ((v / total) * 100).toFixed(1) + '%' : '0%',
    }))
    .sort((a, b) => b.count - a.count);

  return (
    <Card title="Detected Anomalies by Type" subtitle="Distribution of detected events across the AWS network">
      {typeData.length === 0 ? (
        <div className="text-sm text-slate-500 py-10 text-center">
          No production anomalies detected across the network.
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            layout="vertical"
            data={typeData}
            margin={{ top: 5, right: 25, left: 10, bottom: 5 }}
          >
            <XAxis
              type="number"
              stroke="#64748b"
              allowDecimals={false}
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 11, fill: '#64748b' }}
            />
            <YAxis
              type="category"
              dataKey="name"
              stroke="#94a3b8"
              tickLine={false}
              axisLine={false}
              width={105}
              tick={{ fontSize: 12, fill: '#cbd5e1', fontWeight: 500 }}
            />
            <Tooltip
              cursor={{ fill: 'rgba(56,189,248,0.06)' }}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const item = payload[0].payload;
                  return (
                    <div className="rounded-xl border border-borderline bg-panel2 px-3 py-2 text-xs shadow-xl">
                      <div className="font-semibold text-sky-300">{item.name}</div>
                      <div className="text-slate-300 mt-0.5 font-mono">
                        {item.count} detections <span className="text-slate-500">({item.pct})</span>
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar dataKey="count" name="Count" fill="#38bdf8" radius={[0, 6, 6, 0]} barSize={16}>
              {typeData.map((_, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={index % 2 === 0 ? '#38bdf8' : '#818cf8'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </Card>
  );
}

interface SeverityDistributionCardProps {
  sevDist: Record<string, number>;
}

function SeverityDistributionCard({ sevDist }: SeverityDistributionCardProps) {
  const sevData = Object.entries(sevDist)
    .filter(([k]) => k !== 'NONE' && k !== 'NORMAL')
    .map(([k, v]) => ({ name: k.toUpperCase(), count: v }))
    .sort((a, b) => {
      const order: Record<string, number> = { HIGH: 1, MEDIUM: 2, LOW: 3 };
      return (order[a.name] || 99) - (order[b.name] || 99);
    });

  const total = sevData.reduce((sum, s) => sum + s.count, 0);

  return (
    <Card title="Severity Distribution" subtitle="Breakdown of detected anomaly severities">
      {sevData.length === 0 ? (
        <div className="text-sm text-slate-500 py-10 text-center">No anomalous severities recorded.</div>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {sevData.map((s) => {
              const style = SEVERITY_COLOR[s.name] || {
                bg: 'bg-panel3/40',
                border: 'border-borderline/70',
                text: 'text-slate-100',
              };
              const pct = total > 0 ? ((s.count / total) * 100).toFixed(0) : '0';
              return (
                <div
                  key={s.name}
                  className={`rounded-xl border ${style.border} ${style.bg} p-4 flex flex-col justify-between`}
                >
                  <div className="flex items-center justify-between">
                    <Badge status={s.name} dot={false}>
                      {s.name}
                    </Badge>
                    <span className="text-xs text-slate-400 font-mono">{pct}%</span>
                  </div>
                  <div className="mt-3">
                    <div className="text-2xl font-extrabold tabular-nums text-slate-100">{s.count}</div>
                    <div className="text-[11px] text-slate-400 mt-0.5">detected events</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </Card>
  );
}

interface SensorHealthCardProps {
  healthDist: Record<string, number>;
}

function SensorHealthCard({ healthDist }: SensorHealthCardProps) {
  const entries = Object.entries(healthDist);

  return (
    <Card title="Sensor Health Distribution" subtitle="Operational telemetry status across all AWS stations">
      {entries.length === 0 ? (
        <div className="text-sm text-slate-500 py-10 text-center">No sensor health data available.</div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {entries.map(([status, count]) => (
            <div
              key={status}
              className="rounded-xl border border-borderline/70 bg-panel3/40 p-4 flex items-center gap-3.5 hover:border-sky-500/20 transition-all"
            >
              <StatusDot status={status} />
              <div>
                <div className="text-2xl font-extrabold tabular-nums text-slate-100">{count}</div>
                <div className="text-xs text-slate-400 capitalize">{status.toLowerCase()}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

interface AnomalousStationsTableProps {
  stations: ProductionStationRow[];
}

function AnomalousStationsTable({ stations }: AnomalousStationsTableProps) {
  const abnormal = stations.filter((s) => s.status === 'ANOMALY');

  return (
    <Card
      title={`Anomalous Stations (${abnormal.length})`}
      subtitle="Stations currently flagged with detected anomalies in the network"
      right={
        <Button to="/anomalies" variant="ghost" className="!px-2.5 !py-1 !text-xs">
          View all anomalies <IconArrowRight size={14} />
        </Button>
      }
    >
      {abnormal.length === 0 ? (
        <div className="text-sm text-slate-500 py-10 text-center">
          No anomalous stations currently flagged across the network.
        </div>
      ) : (
        <Table>
          <thead>
            <tr className="border-b border-borderline/70">
              <Th>Station</Th>
              <Th>Anomaly Type</Th>
              <Th>Severity</Th>
              <Th right>
                <span title="Fused model score (uncalibrated \u2014 not a probability)">Anomaly Score</span>
              </Th>
              <Th right>Last Reading</Th>
              <Th right>Action</Th>
            </tr>
          </thead>
          <tbody className="divide-y divide-borderline/50">
            {abnormal.map((s) => (
              <tr key={s.station_id} className="hover:bg-soft transition-colors group">
                <Td className="font-semibold text-sky-300">
                  <Link
                    to={`/stations/${s.station_id}`}
                    className="hover:underline flex items-center gap-1.5"
                  >
                    {s.station_id}
                  </Link>
                </Td>
                <Td className="capitalize">
                  <span className="font-medium text-slate-200">
                    {s.anomaly_type?.toLowerCase().replace(/_/g, ' ') ?? '\u2014'}
                  </span>
                </Td>
                <Td>
                  <Badge status={s.severity ?? ''}>{s.severity ?? '\u2014'}</Badge>
                </Td>
                <Td right mono className="font-semibold text-slate-200">
                  {s.anomaly_score != null ? fmt(s.anomaly_score, 3) : '\u2014'}
                </Td>
                <Td right className="text-slate-400 whitespace-nowrap text-xs">
                  {s.timestamp ? new Date(s.timestamp).toLocaleString() : '\u2014'}
                </Td>
                <Td right>
                  <Link
                    to={`/stations/${s.station_id}`}
                    className="text-xs text-sky-400 hover:text-sky-300 font-medium inline-flex items-center gap-1"
                  >
                    Details <IconArrowRight size={12} />
                  </Link>
                </Td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}
    </Card>
  );
}

// --- Main Analytics Page ---

export default function Analytics() {
  const {
    data,
    loading,
    isRefreshing,
    error,
    dataSource,
    setDataSource,
    refetch,
  } = useAnalytics({
    initialSource: 'HISTORICAL',
    autoPoll: true,
    pollingIntervalMs: 8000,
  });

  if (error && !data) {
    return <ErrorState message={error} />;
  }

  if (loading && !data) {
    return <Spinner label="Aggregating dynamic production analytics\u2026" />;
  }

  const analyticsData = data as ProductionAnalytics;
  const statusDist = analyticsData.status_distribution ?? {};
  const sevDist = analyticsData.severity_distribution ?? {};
  const typeDist = analyticsData.anomaly_type_distribution ?? {};
  const healthDist = analyticsData.sensor_health_distribution ?? {};
  const stations = analyticsData.stations ?? [];

  return (
    <div className="space-y-7">
      <AnalyticsHeader
        totalStations={analyticsData.total_stations}
        dataSource={dataSource}
        isRefreshing={isRefreshing}
        onSelectSource={setDataSource}
        onRefresh={refetch}
      />

      {/* Dynamic KPI Cards */}
      <KpiCards
        totalStations={analyticsData.total_stations}
        statusDist={statusDist}
      />

      {/* Station Status & Anomaly Type Charts */}
      <div className="grid lg:grid-cols-2 gap-6">
        <StatusPieCard statusDist={statusDist} />
        <AnomalyTypeBarCard typeDist={typeDist} />
      </div>

      {/* Severity & Sensor Health Breakdown */}
      <div className="grid lg:grid-cols-2 gap-6">
        <SeverityDistributionCard sevDist={sevDist} />
        <SensorHealthCard healthDist={healthDist} />
      </div>

      {/* Anomalous Stations Table */}
      <AnomalousStationsTable stations={stations} />

      {/* Model Version & Provenance Footer */}
      <div className="flex items-center gap-2 text-xs text-slate-500">
        <IconHeart size={14} className="text-sky-500/60" />
        <span>
          Model version: {analyticsData.model_version}
          {' \u00b7 '}
          Live aggregates computed dynamically from detected network events. Controlled ground-truth evaluation metrics are on the{' '}
          <Link to="/evaluation" className="text-sky-400 hover:underline">
            Evaluation page
          </Link>.
        </span>
      </div>
    </div>
  );
}