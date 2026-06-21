import { useState, useEffect } from 'react';
import { Position } from '../types';
import { orchestrator } from '../api/orchestrator';

export default function Positions() {
  const [positions, setPositions] = useState<Position[]>([]);

  useEffect(() => {
    setPositions(orchestrator.fetchPositions());

    const unsub = orchestrator.on('health', () => {
      setPositions(orchestrator.fetchPositions());
    });

    const interval = setInterval(() => {
      setPositions((prev) =>
        prev.map((p) => {
          const change = (Math.random() - 0.5) * 0.002;
          const currentPrice = Math.round((p.currentPrice + change) * 10000) / 10000;
          const pnl =
            Math.round(
              (currentPrice - p.entryPrice) * p.lots * (p.direction === 'LONG' ? 1 : -1) * 100000 * 100
            ) / 100;
          return { ...p, currentPrice, unrealizedPnL: pnl };
        })
      );
    }, 3000);

    return () => {
      unsub();
      clearInterval(interval);
    };
  }, []);

  const handleClose = (symbol: string) => {
    setPositions((prev) => prev.filter((p) => p.symbol !== symbol));
  };

  if (positions.length === 0) {
    return (
      <div className="positions">
        <h3 className="section-title">Open Positions</h3>
        <div className="positions-empty">No open positions</div>
      </div>
    );
  }

  return (
    <div className="positions">
      <h3 className="section-title">Open Positions</h3>
      <div className="positions-table-wrapper">
        <table className="positions-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Direction</th>
              <th>Entry</th>
              <th>Price</th>
              <th>Lots</th>
              <th>P&amp;L</th>
              <th>SLBE</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {positions.map((pos) => (
              <tr key={pos.symbol} className={`position-row ${pos.direction.toLowerCase()}`}>
                <td className="position-symbol">{pos.symbol}</td>
                <td>
                  <span className={`direction-badge ${pos.direction.toLowerCase()}`}>
                    {pos.direction}
                  </span>
                </td>
                <td>{pos.entryPrice.toFixed(4)}</td>
                <td className={pos.currentPrice >= pos.entryPrice ? 'price-up' : 'price-down'}>
                  {pos.currentPrice.toFixed(4)}
                </td>
                <td>{pos.lots.toFixed(2)}</td>
                <td className={pos.unrealizedPnL >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                  {pos.unrealizedPnL >= 0 ? '+' : ''}${pos.unrealizedPnL.toFixed(2)}
                </td>
                <td>
                  <span className={`slbe-badge ${pos.slbeActive ? 'active' : 'inactive'}`}>
                    {pos.slbeActive ? 'Active' : 'Inactive'}
                  </span>
                </td>
                <td>
                  <button className="close-btn" onClick={() => handleClose(pos.symbol)}>
                    Close
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}