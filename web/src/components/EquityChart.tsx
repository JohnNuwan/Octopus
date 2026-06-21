import { useState, useEffect, useMemo } from 'react';
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts';
import { orchestrator } from '../api/orchestrator';

type Period = '1H' | '1D' | '1W' | '1M';

interface EquityPoint {
  timestamp: number;
  equity: number;
  volume: number;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const data = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip-time">
        {new Date(data.timestamp).toLocaleTimeString()}
      </p>
      <p className="chart-tooltip-equity">
        Equity: <strong>${data.equity.toLocaleString(undefined, { minimumFractionDigits: 2 })}</strong>
      </p>
      <p className="chart-tooltip-volume">
        Volume: {data.volume.toLocaleString()}
      </p>
    </div>
  );
};

export default function EquityChart() {
  const [period, setPeriod] = useState<Period>('1D');
  const [data, setData] = useState<EquityPoint[]>([]);

  useEffect(() => {
    setData(orchestrator.fetchEquityHistory(100));

    const unsub = orchestrator.on('health', () => {
      setData(orchestrator.fetchEquityHistory(100));
    });

    const interval = setInterval(() => {
      setData((prev) => {
        const last = prev[prev.length - 1];
        if (!last) return orchestrator.fetchEquityHistory(100);
        const change = (Math.random() - 0.45) * 300;
        const newEquity = Math.round((last.equity + change) * 100) / 100;
        const point = {
          timestamp: Date.now(),
          equity: newEquity,
          volume: Math.round(Math.random() * 500 + 10),
        };
        return [...prev.slice(-100), point];
      });
    }, 5000);

    return () => {
      unsub();
      clearInterval(interval);
    };
  }, []);

  const periods: Period[] = ['1H', '1D', '1W', '1M'];

  return (
    <div className="equity-chart">
      <div className="equity-chart-header">
        <h3>Equity Curve</h3>
        <div className="period-selector">
          {periods.map((p) => (
            <button
              key={p}
              className={`period-btn ${period === p ? 'active' : ''}`}
              onClick={() => setPeriod(p)}
            >
              {p}
            </button>
          ))}
        </div>
      </div>
      <div className="equity-chart-body">
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00c853" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#00c853" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3e" />
            <XAxis
              dataKey="timestamp"
              tick={{ fill: '#888', fontSize: 11 }}
              tickFormatter={(v) => {
                const d = new Date(v);
                return period === '1H'
                  ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                  : d.toLocaleDateString([], { month: 'short', day: 'numeric' });
              }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: '#888', fontSize: 11 }}
              tickFormatter={(v) => `$${(v / 1000).toFixed(1)}k`}
              axisLine={false}
              tickLine={false}
              domain={['auto', 'auto']}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="equity"
              stroke="#00c853"
              strokeWidth={2}
              fill="url(#equityGradient)"
              dot={false}
              activeDot={{ r: 4, fill: '#00c853', stroke: '#0a0e17' }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="equity-chart-volume">
        <ResponsiveContainer width="100%" height={60}>
          <BarChart data={data.slice(-50)} margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3e" vertical={false} />
            <Bar dataKey="volume" fill="#1a3a2e" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}