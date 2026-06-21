import { useState, useEffect } from 'react';
import { DashboardState } from '../types';
import { orchestrator } from '../api/orchestrator';

interface HeaderProps {
  health: DashboardState;
  tradesToday: number;
}

export default function Header({ health, tradesToday }: HeaderProps) {
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setConnected(orchestrator.isConnected());
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const pnlColor = health.dailyPnL >= 0 ? '#00c853' : '#ff1744';

  return (
    <header className="header">
      <div className="header-left">
        <span className="header-logo">🐙</span>
        <div>
          <h1 className="header-title">Octopus</h1>
          <span className="header-subtitle">Trading Dashboard</span>
        </div>
      </div>

      <div className="header-center">
        <div className="header-stat">
          <span className="header-stat-label">Connection</span>
          <span className={`header-status-dot ${connected ? 'connected' : 'disconnected'}`} />
          <span className="header-stat-value">{connected ? 'Live' : 'Simulated'}</span>
        </div>
        <div className="header-stat">
          <span className="header-stat-label">Trades Today</span>
          <span className="header-stat-value">{tradesToday}</span>
        </div>
        <div className="header-stat">
          <span className="header-stat-label">Daily P&amp;L</span>
          <span className="header-stat-value" style={{ color: pnlColor }}>
            {health.dailyPnL >= 0 ? '+' : ''}${health.dailyPnL.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>
      </div>

      <div className="header-right">
        <div className="header-ftmo">
          <span className="header-stat-label">FTMO</span>
          <span className="header-ftmo-phase">{health.ftmoPhase}</span>
          <span className={`header-ftmo-status ${health.ftmoStatus === 'Passed' ? 'passed' : health.ftmoStatus === 'At Risk' ? 'at-risk' : ''}`}>
            {health.ftmoStatus}
          </span>
        </div>
      </div>
    </header>
  );
}