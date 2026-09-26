import { PageHeader, Card } from '../components/ui';
import { IconSettings, IconBraces, IconCloud, IconCpu, IconZap, IconInfo, IconTag } from '../components/icons';

function BulletList({ items, mono = false }: { items: string[]; mono?: boolean }) {
  return (
    <ul className="text-sm space-y-2">
      {items.map((item, i) => (
        <li key={i} className="flex items-start gap-2.5 text-slate-300">
          <span className={`mt-1.5 w-1.5 h-1.5 shrink-0 rounded-full bg-sky-400/70 ${mono ? '' : ''}`} />
          <span className={`${mono ? 'font-mono text-[12.5px] text-slate-400' : ''}`}>{item}</span>
        </li>
      ))}
    </ul>
  );
}

export default function Settings() {
  const apiEndpoints = [
    'GET  /api/health',
    'GET  /api/stations',
    'GET  /api/stations/:id',
    'GET  /api/readings',
    'GET  /api/anomalies',
    'GET  /api/anomalies/:id',
    'GET  /api/explanations/:id',
    'GET  /api/sensor-health/:id',
    'GET  /api/analytics',
    'GET  /api/model/info',
    'GET  /api/model/evaluation',
    'POST /api/simulation/inject',
    'WS   /api/live',
  ];

  return (
    <div className="space-y-7">
      <PageHeader
        icon={<IconSettings size={20} />}
        title="Settings"
        subtitle="System configuration reference"
      />

      <div className="grid lg:grid-cols-2 gap-6">
        <Card title="Detection Pipeline" subtitle="Applied to every incoming reading">
          <BulletList
            items={[
              'Validation & imputation of raw telemetry',
              'Temporal + multivariate feature extraction',
              'Baseline Isolation Forest (3 raw variables)',
              'Temporal isolation forest',
              'Multivariate isolation forest',
              'Anomaly score fusion & decision threshold',
            ]}
          />
        </Card>

        <Card title="Decision & Explanation Settings" subtitle="Tunable via backend config (app/config.py)">
          <BulletList
            items={[
              'Anomaly score threshold for flagging',
              'Severity mapping',
              'Weather-event vs sensor-fault assessment',
              'Corrected-value imputation toggle',
              'Attribution via SHAP (shap.TreeExplainer over the trained detector; statistical z-score fallback)',
            ]}
          />
        </Card>

        <Card title="API Reference" subtitle="REST + WebSocket endpoints">
          <BulletList items={apiEndpoints} mono />
        </Card>

        <Card title="About" subtitle="SkyGuard AI">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 shrink-0 rounded-xl bg-gradient-to-br from-sky-400 to-indigo-600 p-[1px]">
              <div className="w-full h-full rounded-[11px] bg-panel flex items-center justify-center text-sky-300">
                <IconCloud size={19} />
              </div>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed">
              AI/ML-based intelligent anomaly detection for Automatic Weather Stations
              (AWS). Anomaly scores and model attributions come from the actual trained
              isolation-forest detectors; anomaly type, severity and cause are
              deterministic rule/statistical decisions. Nothing is fabricated.
            </p>
          </div>
          <div className="mt-5 pt-4 border-t border-borderline/60 grid grid-cols-2 gap-3">
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <IconCpu size={14} className="text-sky-500/60" /> Isolation Forest ensemble
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <IconBraces size={14} className="text-sky-500/60" /> REST + WebSocket API
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <IconZap size={14} className="text-sky-500/60" /> Real-time inference
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <IconTag size={14} className="text-sky-500/60" /> Ground-truth evaluation
            </div>
          </div>
        </Card>
      </div>

      <div className="flex items-center gap-2 text-xs text-slate-500">
        <IconInfo size={14} className="text-sky-500/60" />
        Settings are read from the backend configuration; this page is informational.
      </div>
    </div>
  );
}