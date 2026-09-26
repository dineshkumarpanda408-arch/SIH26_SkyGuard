import { useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import { api } from './api';
import { useAuth } from './auth/AuthContext';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Live from './pages/Live';
import Stations from './pages/Stations';
import StationDetail from './pages/StationDetail';
import Anomalies from './pages/Anomalies';
import AnomalyDetail from './pages/AnomalyDetail';
import Analytics from './pages/Analytics';
import SensorHealth from './pages/SensorHealth';
import XaiDeepDive from './pages/XaiDeepDive';
import Simulation from './pages/Simulation';
import Model from './pages/Model';
import Evaluation from './pages/Evaluation';
import Settings from './pages/Settings';

export default function App() {
  const { ready, token } = useAuth();

  useEffect(() => {
    if (!ready || !token) return;
    // Prefetch heavy pages in background so they load instantly on first visit
    api.stations();
    api.anomalies();
    api.health();
    api.modelMetadata();
    api.modelEvaluation();
  }, [ready, token]);

  if (!ready) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-ink">
        <div className="pointer-events-none fixed inset-0 bg-hero" aria-hidden="true" />
        <div className="relative z-10 flex items-center gap-3 text-slate-400">
          <svg className="w-5 h-5 animate-spin text-sky-400" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.15" strokeWidth="3" />
            <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
          </svg>
          <span className="text-sm">Restoring WeatherLock session…</span>
        </div>
      </div>
    );
  }

  if (!token) {
    return <Login />;
  }

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/live" element={<Live />} />
        <Route path="/stations" element={<Stations />} />
        <Route path="/stations/:id" element={<StationDetail />} />
        <Route path="/anomalies" element={<Anomalies />} />
        <Route path="/anomalies/:id" element={<AnomalyDetail />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/deep-dive" element={<XaiDeepDive />} />
        <Route path="/sensor-health" element={<SensorHealth />} />
        <Route path="/simulation" element={<Simulation />} />
        <Route path="/model" element={<Model />} />
        <Route path="/evaluation" element={<Evaluation />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Layout>
  );
}
