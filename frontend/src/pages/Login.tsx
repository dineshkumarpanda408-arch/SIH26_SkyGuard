import { useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import { api } from '../api';
import { useAuth } from '../auth/AuthContext';
import type { AuthPatternMeta, WeatherIcon } from '../types';
import { IconAlert, IconRadar, IconRefresh } from '../components/icons';

const STEPS = 4;

function shuffle<T>(array: T[]): T[] {
  const a = [...array];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export default function Login() {
  const { login } = useAuth();
  const [meta, setMeta] = useState<AuthPatternMeta | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);

  const [grid, setGrid] = useState<WeatherIcon[]>([]);
  const [pattern, setPattern] = useState<string[]>([]);
  const [username, setUsername] = useState('');
  const [pin, setPin] = useState('');
  const [showPin, setShowPin] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [attemptsLeft, setAttemptsLeft] = useState<number | null>(null);
  const [lockUntil, setLockUntil] = useState<number | null>(null);
  const [lockSeconds, setLockSeconds] = useState(0);

  const patternLength = meta?.pattern_length ?? STEPS;
  const pinLength = meta?.pin_length ?? 6;

  // Load the icon catalogue + secret shape once; shuffle the grid on start.
  useEffect(() => {
    let active = true;
    api
      .authPattern()
      .then((m) => {
        if (!active) return;
        setMeta(m);
        setUsername(m.default_username);
        setGrid(shuffle(m.icons));
      })
      .catch((e: Error) => {
        if (active) setMetaError(`Unable to reach SkyGuard backend: ${e.message}`);
      });
    return () => {
      active = false;
    };
  }, []);

  // Countdown for the temporary lockout (423).
  useEffect(() => {
    if (lockUntil == null) return;
    const tick = () => {
      const remaining = Math.max(0, Math.ceil((lockUntil - Date.now()) / 1000));
      setLockSeconds(remaining);
      if (remaining <= 0) {
        setLockUntil(null);
        setError(null);
        setAttemptsLeft(null);
      }
    };
    tick();
    const id = setInterval(tick, 500);
    return () => clearInterval(id);
  }, [lockUntil]);

  const locked = lockUntil != null && lockSeconds > 0;
  const patternFull = pattern.length >= patternLength;
  const pinFull = pin.replace(/\D/g, '').length >= pinLength;

  const canSubmit = useMemo(
    () => username.trim().length > 0 && patternFull && pinFull && !loading && !locked,
    [username, patternFull, pinFull, loading, locked],
  );

  const toggleIcon = (iconId: string) => {
    if (locked) return;
    setError(null);
    setPattern((prev) => {
      const idx = prev.indexOf(iconId);
      if (idx >= 0) return prev.filter((x) => x !== iconId);
      if (prev.length >= patternLength) return prev;
      return [...prev, iconId];
    });
  };

  const resetPattern = () => {
    setPattern([]);
    setError(null);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    setAttemptsLeft(null);
    try {
      await login(username.trim(), pattern, pin);
      // On success, App re-renders and swaps login for the dashboard.
    } catch (err) {
      const authErr = (err as Error & {
        status?: number;
        auth?: { message: string; attempts_left?: number; retry_after?: number };
      }).auth;
      // Positions change after every attempt, so the pattern can't be memorised.
      setGrid((g) => shuffle(g));
      setPattern([]);
      setPin('');
      if (err instanceof Error && authErr) {
        if (authErr.retry_after != null) {
          setLockUntil(Date.now() + authErr.retry_after * 1000);
          const msg = authErr.message || `Too many attempts. Try again in ${authErr.retry_after}s.`;
          setError(msg);
        } else {
          setError(authErr.message || 'Authentication failed.');
          if (authErr.attempts_left != null) setAttemptsLeft(authErr.attempts_left);
        }
      } else {
        setError(err instanceof Error ? err.message : 'Authentication failed. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handlePinChange = (v: string) => {
    setPin(v.replace(/\D/g, '').slice(0, pinLength));
  };

  if (metaError) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-ink px-4">
        <div className="pointer-events-none fixed inset-0 bg-hero" aria-hidden="true" />
        <div className="relative z-10 w-full max-w-md rounded-3xl border border-red-500/30 bg-gradient-to-b from-panel2 to-panel shadow-card p-8 text-center">
          <IconAlert size={28} className="mx-auto text-red-400" />
          <h1 className="mt-4 text-lg font-bold text-slate-100">SkyGuard Backend Offline</h1>
          <p className="mt-2 text-sm text-slate-400">{metaError}</p>
          <p className="mt-4 text-xs text-slate-500">
            Start the FastAPI server (<code className="text-sky-300">uvicorn app.main:app --reload --port 8000</code>) then refresh.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-ink px-4 py-8 relative">
      {/* background */}
      <div className="pointer-events-none fixed inset-0 bg-hero" aria-hidden="true" />
      <div className="pointer-events-none fixed inset-0 bg-grid opacity-60" aria-hidden="true" />

      <div className="relative z-10 w-full max-w-md animate-fade-up">
        {/* Brand */}
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-sky-400 to-indigo-600 p-[1.5px] shadow-glow">
            <div className="w-full h-full rounded-[14px] bg-panel flex items-center justify-center text-sky-300">
              <IconRadar size={30} />
            </div>
          </div>
          <h1 className="mt-4 text-2xl font-black tracking-wide text-slate-50">
            SKYGUARD <span className="bg-gradient-to-r from-sky-400 to-indigo-300 bg-clip-text text-transparent">AI</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1 uppercase tracking-[0.2em]">Intelligent Weather Monitor</p>
        </div>

        {/* WeatherLock card */}
        <form
          onSubmit={handleSubmit}
          className="rounded-3xl border border-borderline/80 bg-gradient-to-b from-panel2 to-panel shadow-card p-6 sm:p-7 space-y-5"
        >
          <div className="text-center">
            <div className="inline-flex items-center gap-2 rounded-full py-1 pl-1 pr-3 text-[11px] font-bold uppercase tracking-widest ring-1 ring-inset ring-sky-500/30 bg-sky-500/10 text-sky-300">
              <span className="flex items-center justify-center w-5 h-5 rounded-full bg-sky-500/20">🔐</span>
              WeatherLock
            </div>
            <p className="mt-2 text-xs text-slate-400 leading-relaxed">
              Visual security pattern + PIN for the single SkyGuard operator account.
            </p>
          </div>

          {/* Username */}
          <div>
            <label className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
              Operator Username
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoCapitalize="none"
              autoCorrect="off"
              spellCheck={false}
              disabled={loading || locked}
              className="input w-full !py-2.5"
              placeholder="skyguard"
            />
          </div>

          {/* Visual pattern grid */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                Select your security pattern <span className="text-sky-400 normal-case font-mono">({pattern.length}/{patternLength})</span>
              </label>
              <button
                type="button"
                onClick={resetPattern}
                disabled={pattern.length === 0 || locked}
                className="flex items-center gap-1 text-[11px] font-medium text-slate-500 hover:text-sky-300 disabled:opacity-40 transition-colors"
              >
                <IconRefresh size={12} />
                Clear
              </button>
            </div>

            {/* Selected sequence */}
            <div className="mb-3 flex items-center justify-center gap-1.5 min-h-[46px] rounded-xl border border-borderline/60 bg-panel3/40 px-2 py-2 flex-wrap">
              {pattern.length === 0 ? (
                <span className="text-[11px] text-slate-500">Choose {patternLength} icons, in order…</span>
              ) : (
                Array.from({ length: patternLength }).map((_, i) => {
                  const icon = meta?.icons.find((ic) => ic.id === pattern[i]);
                  return (
                    <div key={i} className="flex items-center gap-1.5">
                      <div
                        className={`flex items-center gap-1.5 rounded-xl px-2.5 py-1.5 text-sm ring-1 ring-inset ${
                          icon
                            ? 'bg-sky-500/12 ring-sky-500/30'
                            : 'bg-panel3/60 ring-borderline/70 text-slate-600'
                        }`}
                      >
                        <span className="text-[10px] font-bold text-sky-400">{i + 1}</span>
                        <span>{icon ? icon.emoji : '\u2022'}</span>
                      </div>
                      {i < patternLength - 1 && <span className="text-slate-600">→</span>}
                    </div>
                  );
                })
              )}
            </div>

            {/* 3x3 shuffled grid */}
            <div className="grid grid-cols-3 gap-2">
              {grid.map((icon) => {
                const selected = pattern.indexOf(icon.id) >= 0;
                return (
                  <button
                    key={`${icon.id}-${icon.id}`}
                    type="button"
                    onClick={() => toggleIcon(icon.id)}
                    disabled={loading || locked}
                    className={`group flex flex-col items-center justify-center gap-1 rounded-2xl border px-2 py-3.5 transition-all duration-150 disabled:opacity-60 ${
                      selected
                        ? 'border-sky-500/50 bg-sky-500/15 text-sky-200 shadow-[0_0_18px_-4px_rgba(56,189,248,0.5)]'
                        : 'border-borderline/70 bg-panel3/50 text-slate-300 hover:border-sky-500/35 hover:bg-panel3 hover:text-slate-100'
                    }`}
                    title={icon.label}
                  >
                    <span className="text-2xl leading-none drop-shadow-sm">{icon.emoji}</span>
                    <span className={`text-[10px] font-medium truncate w-full text-center ${selected ? 'text-sky-300' : 'text-slate-500 group-hover:text-slate-400'}`}>
                      {icon.label}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* PIN */}
          <div>
            <label className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
              Security PIN <span className="text-sky-400 normal-case font-mono">({pin.length}/{pinLength} digits)</span>
            </label>
            <div className="relative">
              <input
                type={showPin ? 'text' : 'password'}
                inputMode="numeric"
                value={pin}
                onChange={(e) => handlePinChange(e.target.value)}
                placeholder="\u2022\u2022\u2022\u2022\u2022\u2022"
                disabled={loading || locked}
                autoComplete="off"
                className="input w-full !py-2.5 !tracking-widest font-mono !text-base"
              />
              <button
                type="button"
                onClick={() => setShowPin((s) => !s)}
                tabIndex={-1}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-xl leading-none text-slate-400 hover:text-sky-300 transition-colors"
                title={showPin ? 'Hide PIN' : 'Show PIN'}
              >
                {showPin ? '🙈' : '👁️'}
              </button>
            </div>
          </div>

          {/* Error / lockout */}
          {error && (
            <div
              className={`flex items-start gap-2.5 rounded-xl border px-3 py-2.5 text-xs leading-relaxed ${
                locked
                  ? 'border-red-500/30 bg-red-500/[0.08] text-red-300'
                  : 'border-red-500/25 bg-red-500/[0.06] text-red-300/90'
              } animate-fade-up`}
            >
              <IconAlert size={15} className="mt-0.5 shrink-0" />
              <div>
                <div className="font-semibold">{locked ? 'Too many attempts' : error}</div>
                {locked ? (
                  <div className="text-red-300/80">Try again in {lockSeconds}s.</div>
                ) : attemptsLeft != null ? (
                  <div className="text-red-300/70">
                    {attemptsLeft} attempt{attemptsLeft === 1 ? '' : 's'} remaining before a {meta?.lockout_seconds ?? 0}s lockout.
                  </div>
                ) : null}
              </div>
            </div>
          )}

          <button
            type="submit"
            disabled={!canSubmit}
            className="w-full inline-flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-bold transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-sky-500/40 disabled:opacity-50 disabled:pointer-events-none bg-gradient-to-r from-sky-500 to-indigo-500 text-white shadow-[0_10px_28px_-8px_rgba(59,130,246,0.65)] hover:brightness-110"
          >
            {loading ? (
              <>
                <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
                  <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                </svg>
                Verifying…
              </>
            ) : locked ? (
              <>
                <IconAlert size={16} />
                Locked — {lockSeconds}s
              </>
            ) : (
              <>🔐 Unlock Dashboard</>
            )}
          </button>

          <p className="text-center text-[11px] text-slate-500 leading-relaxed">
            Session stays active for 30 minutes of use. Positions shuffle after every attempt.
          </p>
        </form>

        {/* Demo credential hint */}
        <div className="mt-5 rounded-2xl border border-sky-500/15 bg-sky-500/[0.05] px-4 py-3 text-center">
          <div className="text-[10px] font-bold uppercase tracking-widest text-sky-400/80 mb-1.5">
            Prototype Operator Credentials
          </div>
          <div className="text-xs text-slate-300 leading-relaxed">
            Username <code className="text-sky-300 font-mono">skyguard</code> · Pattern{' '}
            <span className="text-slate-100">🌡️ → 💧 → 📊 → ⚡</span> · PIN{' '}
            <code className="text-sky-300 font-mono">739214</code>
          </div>
        </div>
      </div>
    </div>
  );
}