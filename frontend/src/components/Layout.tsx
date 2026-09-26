import { NavLink, Link } from 'react-router-dom';
import type { ReactNode } from 'react';
import { useEffect, useState } from 'react';
import { StatusDot } from './ui';
import { useTheme } from '../theme';
import { api } from '../api';
import {
  IconGrid,
  IconActivity,
  IconPin,
  IconAlert,
  IconChart,
  IconHeart,
  IconFlask,
  IconCpu,
  IconGauge,
  IconSettings,
  IconClock,
  IconRadar,
  IconChevronRight,
  IconSun,
  IconMoon,
} from './icons';
import type { IconProps } from './icons';

type NavItem = { to: string; label: string; icon: (p: IconProps) => ReactNode; end?: boolean };

const NAV_GROUPS: { label: string; items: NavItem[] }[] = [
  {
    label: 'Monitor',
    items: [
      { to: '/', label: 'Dashboard', icon: IconGrid, end: true },
      { to: '/live', label: 'Live Monitor', icon: IconActivity },
    ],
  },
  {
    label: 'Network',
    items: [
      { to: '/stations', label: 'Stations', icon: IconPin },
      { to: '/anomalies', label: 'Anomalies', icon: IconAlert },
    ],
  },
  {
    label: 'Insights',
    items: [
      { to: '/deep-dive', label: 'XAI Deep-Dive', icon: IconCpu },
      { to: '/analytics', label: 'Analytics', icon: IconChart },
      { to: '/sensor-health', label: 'Sensor Health', icon: IconHeart },
    ],
  },
  {
    label: 'Tools',
    items: [
      { to: '/simulation', label: 'Simulation', icon: IconFlask },
      { to: '/model', label: 'Model', icon: IconCpu },
      { to: '/evaluation', label: 'Evaluation', icon: IconGauge },
    ],
  },
  {
    label: 'System',
    items: [{ to: '/settings', label: 'Settings', icon: IconSettings }],
  },
];

const FLAT_NAV: NavItem[] = NAV_GROUPS.flatMap((g) => g.items);

function navClass({ isActive }: { isActive: boolean }) {
  return `group relative flex items-center gap-2.5 rounded-xl px-3 py-2 text-sm font-medium transition-all duration-150 ${
    isActive
      ? 'text-sky-300 bg-gradient-to-r from-sky-500/20 to-indigo-500/10 ring-1 ring-inset ring-sky-500/25 shadow-[0_4px_16px_-6px_rgba(56,189,248,0.4)]'
      : 'text-slate-400 hover:text-slate-100 hover:bg-soft'
  }`;
}

export default function Layout({ children }: { children: ReactNode }) {
  const { theme, toggle } = useTheme();
  const [systemOnline, setSystemOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let active = true;
    api
      .health()
      .then((h) => {
        if (active) setSystemOnline(h.status === 'online' && !!h.model_trained);
      })
      .catch(() => {
        if (active) setSystemOnline(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const online = systemOnline === null ? false : systemOnline;
  return (
    <div className="min-h-screen flex flex-col bg-ink">
      {/* background glow */}
      <div className="pointer-events-none fixed inset-0 bg-hero" aria-hidden="true" />

      {/* header */}
      <header className="relative z-30 border-b border-borderline/70 bg-panel/70 backdrop-blur-xl px-4 sm:px-6 py-3 flex items-center justify-between sticky top-0">
        <Link to="/" className="flex items-center gap-3 group">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-400 to-indigo-600 p-[1px] shadow-glow">
            <div className="w-full h-full rounded-[11px] bg-panel flex items-center justify-center text-sky-300 group-hover:text-sky-200 transition-colors">
              <IconRadar size={20} />
            </div>
          </div>
          <div>
            <div className="text-base font-extrabold tracking-wide leading-none text-slate-50">
              SKYGUARD <span className="bg-gradient-to-r from-sky-400 to-indigo-300 bg-clip-text text-transparent">AI</span>
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5">Intelligent AWS Anomaly Detection</div>
          </div>
        </Link>

        <div className="flex items-center gap-3 sm:gap-5">
          <div className="hidden md:flex items-center gap-2 text-xs text-slate-500">
            <IconClock size={14} />
            <span>
              Updated{' '}
              {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          </div>
          <span className={`flex items-center gap-2 rounded-full py-1.5 pl-2.5 pr-3 text-xs font-semibold ring-1 ring-inset ${
            online
              ? 'text-green-300 bg-green-500/10 ring-green-500/25'
              : 'text-red-300 bg-red-500/10 ring-red-500/25'
          }`}>
            <StatusDot status={online ? 'online' : 'offline'} />
            {systemOnline === null ? 'CONNECTING' : online ? 'SYSTEM ONLINE' : 'SYSTEM OFFLINE'}
          </span>
          <button
            type="button"
            onClick={toggle}
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            className="flex items-center justify-center w-9 h-9 rounded-xl bg-panel3/70 border border-borderline text-slate-400 hover:text-sky-300 hover:border-sky-500/30 transition-colors"
          >
            {theme === 'dark' ? <IconSun size={18} /> : <IconMoon size={18} />}
          </button>
        </div>
      </header>

      <div className="relative z-10 flex flex-1 bg-grid">
        {/* sidebar */}
        <aside className="hidden lg:flex w-60 shrink-0 flex-col border-r border-borderline/60 bg-panel/40 backdrop-blur-sm p-4 gap-1">
          {NAV_GROUPS.map((group) => (
            <div key={group.label} className="mb-3 last:mb-0">
              <div className="px-3 pb-1.5 text-[10px] font-bold uppercase tracking-[0.14em] text-slate-600">
                {group.label}
              </div>
              <div className="space-y-1">
                {group.items.map((n) => (
                  <NavLink key={n.to} to={n.to} end={n.end} className={navClass}>
                    {({ isActive }) => (
                      <>
                        {isActive && (
                          <span className="absolute left-0 inset-y-0 w-0.5 rounded-full bg-gradient-to-b from-sky-400 to-indigo-500" />
                        )}
                        <span className={isActive ? 'text-sky-300' : 'text-slate-500 group-hover:text-slate-300 transition-colors'}>
                          {n.icon({ size: 17 })}
                        </span>
                        {n.label}
                      </>
                    )}
                  </NavLink>
                ))}
              </div>
            </div>
          ))}

          <div className="mt-auto pt-3">
            <Link
              to="/stations/AWS-023"
              className="flex items-center justify-between rounded-xl border border-borderline/70 bg-panel3/50 hover:border-sky-500/30 px-3 py-2.5 text-xs text-slate-400 hover:text-slate-200 transition-colors"
            >
              <span className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse-soft" />
                Demo: AWS-023
              </span>
              <IconChevronRight size={14} />
            </Link>
          </div>
        </aside>

        {/* main */}
        <main className="flex-1 min-w-0 px-4 sm:px-6 lg:px-8 py-6 lg:py-8 pb-24 lg:pb-8">{children}</main>
      </div>

      {/* mobile bottom nav */}
      <nav className="lg:hidden fixed bottom-0 inset-x-0 z-30 border-t border-borderline/70 bg-panel/90 backdrop-blur-xl">
        <div className="flex overflow-x-auto">
          {FLAT_NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              className={({ isActive }) =>
                `flex flex-1 min-w-16 flex-col items-center gap-1 px-2 py-2.5 text-[10px] font-medium transition-colors ${
                  isActive ? 'text-sky-300' : 'text-slate-500'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <span className={`rounded-lg px-2.5 py-1 transition-colors ${isActive ? 'bg-sky-500/15 ring-1 ring-inset ring-sky-500/30' : ''}`}>
                    {n.icon({ size: 18 })}
                  </span>
                  {n.label}
                </>
              )}
            </NavLink>
          ))}
        </div>
      </nav>
    </div>
  );
}