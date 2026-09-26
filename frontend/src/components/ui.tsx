import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { IconAlert } from './icons';

/* ---------------------------------- Card ---------------------------------- */

export function Card({
  title,
  subtitle,
  children,
  className = '',
  right,
  hover,
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
  right?: ReactNode;
  hover?: boolean;
}) {
  return (
    <div
      className={`rounded-2xl border border-borderline/80 bg-gradient-to-b from-panel2 to-panel shadow-card ${
        hover ? 'transition-all duration-200 hover:-translate-y-0.5 hover:shadow-cardHover hover:border-sky-500/25' : ''
      } ${className}`}
    >
      {(title || right) && (
        <div className="flex items-center justify-between gap-3 px-5 pt-4 pb-3 border-b border-borderline/70">
          <div className="min-w-0">
            {title && (
              <h3 className="text-[13px] font-semibold uppercase tracking-wider text-slate-300">{title}</h3>
            )}
            {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
          </div>
          {right && <div className="shrink-0">{right}</div>}
        </div>
      )}
      <div className="p-5">{children}</div>
    </div>
  );
}

/* -------------------------------- StatCard --------------------------------- */

const STAT_TONES = {
  sky: 'text-sky-300 bg-sky-500/12 ring-sky-500/25',
  indigo: 'text-indigo-300 bg-indigo-500/12 ring-indigo-500/25',
  emerald: 'text-emerald-300 bg-emerald-500/12 ring-emerald-500/25',
  amber: 'text-amber-300 bg-amber-500/12 ring-amber-500/25',
  red: 'text-red-300 bg-red-500/12 ring-red-500/25',
  violet: 'text-violet-300 bg-violet-500/12 ring-violet-500/25',
  slate: 'text-slate-300 bg-slate-500/12 ring-slate-500/25',
} as const;

export type StatTone = keyof typeof STAT_TONES;

export function StatCard({
  label,
  value,
  icon,
  tone = 'sky',
  sub,
  className = '',
}: {
  label: string;
  value: ReactNode;
  icon?: ReactNode;
  tone?: StatTone;
  sub?: ReactNode;
  className?: string;
}) {
  const tones = STAT_TONES[tone];
  return (
    <div className={`relative rounded-2xl border border-borderline/80 bg-gradient-to-b from-panel2 to-panel shadow-card p-4 sm:p-5 ${className}`}>
      <div className="flex items-center gap-3">
        {icon && (
          <div className={`w-10 h-10 shrink-0 rounded-xl ring-1 ring-inset flex items-center justify-center ${tones}`}>
            {icon}
          </div>
        )}
        <div className="min-w-0 flex-1">
          <div className={`text-2xl sm:text-3xl font-extrabold tracking-tight tabular-nums ${tone === 'slate' ? 'text-slate-200' : tones.split(' ')[0]}`}>
            {value}
          </div>
          <div className="text-xs text-slate-400 mt-0.5 truncate">{label}</div>
        </div>
      </div>
      {sub && <div className="mt-3 pt-3 border-t border-borderline/60 text-xs text-slate-500">{sub}</div>}
    </div>
  );
}

/* ---------------------------------- Badge ---------------------------------- */

const BADGE_MAP: Record<string, string> = {
  healthy: 'text-green-300 bg-green-500/10 ring-green-500/25',
  normal: 'text-green-300 bg-green-500/10 ring-green-500/25',
  online: 'text-green-300 bg-green-500/10 ring-green-500/25',
  warning: 'text-yellow-300 bg-yellow-500/10 ring-yellow-500/25',
  medium: 'text-yellow-300 bg-yellow-500/10 ring-yellow-500/25',
  stale: 'text-yellow-300 bg-yellow-500/10 ring-yellow-500/25',
  degraded: 'text-orange-300 bg-orange-500/10 ring-orange-500/25',
  high: 'text-orange-300 bg-orange-500/10 ring-orange-500/25',
  critical: 'text-red-300 bg-red-500/10 ring-red-500/30',
  anomaly: 'text-red-300 bg-red-500/10 ring-red-500/30',
  active: 'text-red-300 bg-red-500/10 ring-red-500/30',
  confirmed: 'text-red-300 bg-red-500/10 ring-red-500/30',
  low: 'text-sky-300 bg-sky-500/10 ring-sky-500/25',
  moderate: 'text-sky-300 bg-sky-500/10 ring-sky-500/25',
  info: 'text-blue-300 bg-blue-500/10 ring-blue-500/25',
  default: 'text-slate-300 bg-slate-500/10 ring-slate-500/25',
  offline: 'text-slate-300 bg-slate-500/10 ring-slate-500/25',
  unavailable: 'text-slate-300 bg-slate-500/10 ring-slate-500/25',
};

export function Badge({
  status,
  children,
  dot = true,
}: {
  status: string;
  children: ReactNode;
  dot?: boolean;
}) {
  const cls =
    BADGE_MAP[status.toLowerCase()] ?? BADGE_MAP.default;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[11px] font-semibold uppercase tracking-wider ring-1 ring-inset ${cls}`}
    >
      {dot && <span className="w-1.5 h-1.5 rounded-full bg-current" />}
      {children}
    </span>
  );
}

/* --------------------------------- StatusDot ------------------------------- */

const DOT_COLORS: Record<string, { bg: string; glow: string }> = {
  green: { bg: 'bg-green-400', glow: 'var(--dot-good)' },
  red: { bg: 'bg-red-400', glow: 'var(--dot-red)' },
  yellow: { bg: 'bg-yellow-400', glow: 'var(--dot-yellow)' },
  orange: { bg: 'bg-orange-400', glow: 'var(--dot-orange)' },
  slate: { bg: 'bg-slate-500', glow: 'var(--dot-slate)' },
};

export function StatusDot({ status, pulse = true }: { status: string; pulse?: boolean }) {
  const s = status.toLowerCase();
  let color = DOT_COLORS.red;
  if (['healthy', 'normal', 'online'].includes(s)) color = DOT_COLORS.green;
  else if (['stale', 'warning', 'medium', 'watch'].includes(s)) color = DOT_COLORS.yellow;
  else if (['unavailable', 'offline'].includes(s)) color = DOT_COLORS.slate;
  else if (['degraded', 'high'].includes(s)) color = DOT_COLORS.orange;
  return (
    <span
      className={`inline-block w-2.5 h-2.5 rounded-full ${color.bg} ${pulse ? 'animate-pulse-soft' : ''}`}
      style={{ boxShadow: color.glow }}
    />
  );
}

/* --------------------------------- Spinner --------------------------------- */

export function Spinner({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-slate-400 py-10 justify-center">
      <svg className="w-5 h-5 animate-spin text-sky-400" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.15" strokeWidth="3" />
        <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      </svg>
      <span className="text-sm">{label}</span>
    </div>
  );
}

/* -------------------------------- ErrorState ------------------------------- */

export function ErrorState({ message, title = 'Something went wrong' }: { message: string; title?: string }) {
  return (
    <div className="rounded-2xl border border-red-500/25 bg-red-500/[0.07] p-5 flex items-start gap-3">
      <div className="mt-0.5 w-9 h-9 shrink-0 rounded-xl bg-red-500/15 text-red-400 ring-1 ring-inset ring-red-500/25 flex items-center justify-center">
        <IconAlert size={18} />
      </div>
      <div className="min-w-0">
        <div className="text-sm font-semibold text-red-300">{title}</div>
        <div className="text-xs text-red-300/80 mt-0.5 leading-relaxed">{message}</div>
      </div>
    </div>
  );
}

/* -------------------------------- ProgressBar ------------------------------ */

export function ProgressBar({ value, color = 'bg-sky-500' }: { value: number; color?: string }) {
  const v = Math.max(0, Math.min(100, value));
  return (
    <div className="w-full h-1.5 bg-soft rounded-full overflow-hidden">
      <div
        className={`h-full ${color} rounded-full transition-all duration-500 ${v > 0 ? 'shadow-[0_0_8px_rgba(56,189,248,0.35)]' : ''}`}
        style={{ width: `${v}%` }}
      />
    </div>
  );
}

/* -------------------------------- PageHeader ------------------------------- */

export function PageHeader({
  title,
  subtitle,
  actions,
  icon,
}: {
  title: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4 animate-fade-up">
      <div className="flex items-center gap-3.5 min-w-0">
        {icon && (
          <div className="w-11 h-11 shrink-0 rounded-xl bg-gradient-to-br from-sky-500/20 to-indigo-500/10 ring-1 ring-inset ring-sky-500/25 text-sky-300 flex items-center justify-center shadow-card">
            {icon}
          </div>
        )}
        <div className="min-w-0">
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-50">{title}</h1>
          {subtitle && <p className="text-xs sm:text-sm text-slate-400 mt-0.5">{subtitle}</p>}
        </div>
      </div>
      {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
    </div>
  );
}

/* ---------------------------------- Button --------------------------------- */

type ButtonProps = {
  children: ReactNode;
  onClick?: () => void;
  to?: string;
  disabled?: boolean;
  type?: 'button' | 'submit';
  variant?: 'primary' | 'danger' | 'ghost' | 'outline';
  className?: string;
};

export function Button({
  children,
  onClick,
  to,
  disabled,
  type = 'button',
  variant = 'primary',
  className = '',
}: ButtonProps) {
  const base =
    'inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-sky-500/30 disabled:opacity-50 disabled:pointer-events-none';
  const variants: Record<string, string> = {
    primary:
      'bg-gradient-to-r from-sky-500 to-indigo-500 text-white shadow-[0_8px_24px_-8px_rgba(59,130,246,0.6)] hover:brightness-110',
    danger:
      'bg-gradient-to-r from-red-500 to-rose-500 text-white shadow-[0_8px_24px_-8px_rgba(239,68,68,0.5)] hover:brightness-110',
    ghost: 'bg-panel3/70 text-slate-200 border border-borderline hover:bg-panel3 hover:border-sky-500/30',
    outline: 'border border-sky-500/40 text-sky-300 hover:bg-sky-500/10',
  };
  const cls = `${base} ${variants[variant]} ${className}`;
  if (to) {
    return (
      <Link to={to} className={cls}>
        {children}
      </Link>
    );
  }
  return (
    <button type={type} onClick={onClick} disabled={disabled} className={cls}>
      {children}
    </button>
  );
}

/* ------------------------------ Table helpers ------------------------------ */

export function Th({
  children,
  right,
  className = '',
}: {
  children?: ReactNode;
  right?: boolean;
  className?: string;
}) {
  return (
    <th
      className={`px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-slate-500 whitespace-nowrap ${
        right ? 'text-right' : 'text-left'
      } ${className}`}
    >
      {children}
    </th>
  );
}

export function Td({
  children,
  className = '',
  mono,
  right,
}: {
  children?: ReactNode;
  className?: string;
  mono?: boolean;
  right?: boolean;
}) {
  return (
    <td className={`px-4 py-3 text-sm text-slate-300 ${mono ? 'font-mono text-[13px]' : ''} ${right ? 'text-right' : ''} ${className}`}>
      {children}
    </td>
  );
}

export function Table({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <div className="overflow-x-auto">
      <table className={`w-full text-sm min-w-[640px] ${className}`}>{children}</table>
    </div>
  );
}

/* ---------------------------------- misc ----------------------------------- */

export function fmt(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || Number.isNaN(n)) return '\u2014';
  return n.toFixed(digits);
}

export function InfoTip({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-center gap-2 text-xs text-slate-500">{children}</div>
  );
}