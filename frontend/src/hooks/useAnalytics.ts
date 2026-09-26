import { useEffect, useState, useCallback, useRef } from 'react';
import { api } from '../api';
import type { ProductionAnalytics } from '../types';

// Persists across page navigations so re-mounting shows cached data instantly
const globalCache: { data: ProductionAnalytics | null } = { data: null };

export interface UseAnalyticsOptions {
  initialSource?: 'LIVE' | 'HISTORICAL';
  pollingIntervalMs?: number;
  autoPoll?: boolean;
}

export interface UseAnalyticsReturn {
  data: ProductionAnalytics | null;
  loading: boolean;
  isRefreshing: boolean;
  error: string | null;
  dataSource: 'LIVE' | 'HISTORICAL';
  setDataSource: (source: 'LIVE' | 'HISTORICAL') => void;
  refetch: () => Promise<void>;
}

export function useAnalytics(options: UseAnalyticsOptions = {}): UseAnalyticsReturn {
  const {
    initialSource = 'HISTORICAL',
    pollingIntervalMs = 10000,
    autoPoll = true,
  } = options;

  const [dataSource, setDataSourceState] = useState<'LIVE' | 'HISTORICAL'>(initialSource);
  const [data, setData] = useState<ProductionAnalytics | null>(globalCache.data);
  const [loading, setLoading] = useState<boolean>(globalCache.data === null);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const isMountedRef = useRef(true);

  const fetchAnalytics = useCallback(
    async (source: 'LIVE' | 'HISTORICAL', isBackground = false) => {
      if (!isBackground) {
        setLoading(true);
      } else {
        setIsRefreshing(true);
      }
      setError(null);

      try {
        const result = await api.analytics({ source });
        if (isMountedRef.current) {
          globalCache.data = result;
          setData(result);
        }
      } catch (err) {
        if (isMountedRef.current) {
          const msg = err instanceof Error ? err.message : 'Failed to load analytics data';
          setError(msg);
        }
      } finally {
        if (isMountedRef.current) {
          setLoading(false);
          setIsRefreshing(false);
        }
      }
    },
    [],
  );

  const setDataSource = useCallback(
    (source: 'LIVE' | 'HISTORICAL') => {
      setDataSourceState(source);
      fetchAnalytics(source, false);
    },
    [fetchAnalytics],
  );

  const refetch = useCallback(async () => {
    await fetchAnalytics(dataSource, true);
  }, [fetchAnalytics, dataSource]);

  useEffect(() => {
    isMountedRef.current = true;
    fetchAnalytics(dataSource, false);

    let intervalId: ReturnType<typeof setInterval> | null = null;
    if (autoPoll && pollingIntervalMs > 0) {
      intervalId = setInterval(() => {
        fetchAnalytics(dataSource, true);
      }, pollingIntervalMs);
    }

    return () => {
      isMountedRef.current = false;
      if (intervalId) clearInterval(intervalId);
    };
  }, [dataSource, autoPoll, pollingIntervalMs, fetchAnalytics]);

  return {
    data,
    loading,
    isRefreshing,
    error,
    dataSource,
    setDataSource,
    refetch,
  };
}
