import { useState, useEffect } from 'react';
import { DashboardState } from '../types';
import { orchestrator } from '../api/orchestrator';
import Header from './Header';
import SystemStatus from './SystemStatus';
import EquityChart from './EquityChart';
import RiskMetrics from './RiskMetrics';
import Positions from './Positions';
import LiveTrader from './LiveTrader';
import TradeHistory from './TradeHistory';

export default function Dashboard() {
  const [health, setHealth] = useState<DashboardState>(orchestrator.fetchHealth());
  const [tradesToday, setTradesToday] = useState(Math.floor(Math.random() * 25));

  useEffect(() => {
    const unsub = orchestrator.on('health', (data: DashboardState) => {
      setHealth(data);
    });

    const interval = setInterval(() => {
      setHealth(orchestrator.fetchHealth());
      setTradesToday((prev) => prev + (Math.random() > 0.85 ? 1 : 0));
    }, 5000);

    return () => {
      unsub();
      clearInterval(interval);
    };
  }, []);

  // Connect WebSocket on mount
  useEffect(() => {
    orchestrator.connect();
    return () => orchestrator.disconnect();
  }, []);

  return (
    <div className="dashboard">
      <Header health={health} tradesToday={tradesToday} />
      <SystemStatus />
      <div className="dashboard-grid">
        <div className="dashboard-grid-main">
          <EquityChart />
        </div>
        <div className="dashboard-grid-side">
          <RiskMetrics />
        </div>
      </div>
      <div className="dashboard-grid-2col">
        <Positions />
        <LiveTrader />
      </div>
      <TradeHistory />
    </div>
  );
}