import { useState, useEffect } from 'react';
import { Trade } from '../types';
import { orchestrator } from '../api/orchestrator';

export default function TradeHistory() {
  const [trades, setTrades] = useState<Trade[]>([]);

  useEffect(() => {
    setTrades(orchestrator.fetchTrades(50));

    const interval = setInterval(() => {
      setTrades(orchestrator.fetchTrades(50));
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const formatDuration = (entry: string, exit: string): string => {
    const diff = new Date(exit).getTime() - new Date(entry).getTime();
    const hours = Math.floor(diff / 3600000);
    const minutes = Math.floor((diff % 3600000) / 60000);
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m`;
    return '<1m';
  };

  const sorted = [...trades].sort(
    (a, b) => new Date(b.exitTime).getTime() - new Date(a.exitTime).getTime()
  );

  return (
    <div className="trade-history">
      <h3 className="section-title">Trade History</h3>
      <div className="trade-history-wrapper">
        <table className="trade-history-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Symbol</th>
              <th>Direction</th>
              <th>Entry</th>
              <th>Exit</th>
              <th>Lots</th>
              <th>P&amp;L</th>
              <th>Duration</th>
            </tr>
          </thead>
          <tbody>
            {sorted.slice(0, 50).map((trade) => (
              <tr key={trade.id}>
                <td className="trade-id">{trade.id}</td>
                <td className="trade-symbol">{trade.symbol}</td>
                <td>
                  <span className={`direction-badge ${trade.direction.toLowerCase()}`}>
                    {trade.direction}
                  </span>
                </td>
                <td>{trade.entryPrice.toFixed(4)}</td>
                <td>{trade.exitPrice.toFixed(4)}</td>
                <td>{trade.lots.toFixed(2)}</td>
                <td className={trade.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                  {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                </td>
                <td className="trade-duration">
                  {formatDuration(trade.entryTime, trade.exitTime)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}