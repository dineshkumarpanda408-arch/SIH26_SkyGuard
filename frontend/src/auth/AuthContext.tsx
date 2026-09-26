import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { api, setAuthToken, setAuthFailureHandler, clearCache } from '../api';

const STORAGE_KEY = 'skyguard_weatherlock_session';

interface StoredSession {
  token: string;
  username: string;
}

interface AuthContextValue {
  ready: boolean;
  token: string | null;
  username: string | null;
  login: (username: string, pattern: string[], pin: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function readStored(): StoredSession | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as StoredSession;
    if (parsed && parsed.token) return parsed;
    return null;
  } catch {
    return null;
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [token, setToken] = useState<string | null>(null);
  const [username, setUsername] = useState<string | null>(null);

  const logout = useCallback(() => {
    const hadToken = token != null;
    setToken(null);
    setUsername(null);
    setAuthToken(null);
    clearCache();
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* ignore */
    }
    if (hadToken) {
      // Best-effort server-side invalidation; never block the UI on it.
      api.authLogout().catch(() => undefined);
    }
  }, [token]);

  // Restore a previously stored session (and validate it against the backend).
  useEffect(() => {
    const stored = readStored();
    if (!stored) {
      setReady(true);
      return;
    }
    setToken(stored.token);
    setAuthToken(stored.token);
    api
      .authMe()
      .then((me) => {
        setUsername(me.username);
        setReady(true);
      })
      .catch(() => {
        // Token expired / invalidated server-side -> drop back to login.
        setToken(null);
        setUsername(null);
        setAuthToken(null);
        clearCache();
        try {
          localStorage.removeItem(STORAGE_KEY);
        } catch {
          /* ignore */
        }
        setReady(true);
      });
  }, []);

  // Anything returning 401 (expired/invalidated session, idle timeout) kicks
  // the operator back to the WeatherLock screen.
  useEffect(() => {
    setAuthFailureHandler(() => logout);
    return () => setAuthFailureHandler(null);
  }, [logout]);

  const login = useCallback(async (uname: string, pattern: string[], pin: string) => {
    const session = await api.authLogin({ username: uname, pattern, pin });
    setToken(session.token);
    setUsername(session.username);
    setAuthToken(session.token);
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ token: session.token, username: session.username } as StoredSession),
      );
    } catch {
      /* ignore */
    }
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({ ready, token, username, login, logout }),
    [ready, token, username, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>');
  return ctx;
}