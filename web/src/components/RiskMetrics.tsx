import { useState, useEffect } from 'react';

interface RiskData {
  kellySize: number;
  var95: number;
  maxDrawdown: number;
  sharpeRatio: number;
}

function generateRiskData(): RiskData {
  return {
    kellySize: Math.round(Math.random() * 20 * 100) / 100,
    var95: Math.round(Math.random() * 5 * 100) / 100,
    maxDrawdown: Math.round(Math.random() * 25 * 100) / 100,
    sharpeRatio: Math.round((Math.random() * 3 + 0.5) * 100) / 100,
  };
}

function getRiskColor(value: number, metric: keyof RiskData): string {
  switch (metric) {
    case 'kellySize':
      return value > 10 ? '#ff1744' : value > 5 ? '#ffc107' : '#00c853';
    case 'var95':
      return value > 3 ? '#ff1744' : value > 1.5 ? '#ffc107' : '#00c853';
    case 'maxDrawdown':
      return value > 15 ? '#ff1744' : value > 8 ? '#ffc107' : '#00c853';
    case 'sharpeRatio':
      return value > 2 ? '#00c853' : value > 1 ? '#ffc107' : '#ff1744';
    default:
      return '#888';
  }
}

const metricLabels: Record<keyof RiskData, string> = {
  kellySize: 'Kelly Size',
  var95: 'VaR (95%)',
  maxDrawdown: 'Max Drawdown',
  sharpeRatio: 'Sharpe Ratio',
};

const metricFormats: Record<keyof RiskData, string> = {
  kellySize: '%',
  var95: '%',
  maxDrawdown: '%',
  sharpeRatio: '',
};

export default function RiskMetrics() {
  const [risk, setRisk] = useState<RiskData>(generateRiskData());

  useEffect(() => {
    const interval = setInterval(() => {
      setRisk(generateRiskData());
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  const metrics = Object.keys(risk) as (keyof RiskData)[];

  return (
    <div className="risk-metrics">
      <h3 className="section-title">Risk Metrics</h3>
      <div className="risk-grid">
        {metrics.map((metric) => {
          const value = risk[metric];
          const color = getRiskColor(value, metric);
          return (
            <div key={metric} className="risk-card" style={{ borderLeftColor: color }}>
              <div className="risk-card-label">{metricLabels[metric]}</div>
              <div className="risk-card-value" style={{ color }}>
                {value.toFixed(2)}{metricFormats[metric]}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}