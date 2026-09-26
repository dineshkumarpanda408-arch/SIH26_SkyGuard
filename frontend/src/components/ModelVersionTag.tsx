import { useEffect, useState } from 'react';
import { api } from '../api';

export default function ModelVersionTag() {
  const [version, setVersion] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    api
      .modelMetadata()
      .then((m) => alive && setVersion(m.model_version ?? null))
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, []);

  if (failed) return null;

  return (
    <span
      title="Persisted production model version (single source of truth)"
      className="inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-semibold text-violet-300 bg-violet-500/10 ring-1 ring-inset ring-violet-500/25"
    >
      <span className="w-1.5 h-1.5 rounded-full bg-violet-400" />
      Production model {'\u00b7'} {version ?? '\u2026'}
    </span>
  );
}