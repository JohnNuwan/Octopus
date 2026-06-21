import { useState, useEffect } from 'react';
import { TradingDecision } from '../types';
import { orchestrator } from '../api/orchestrator';

export default function LiveTrader() {
  const [lastDecision, setLastDecision] = useState<TradingDecision | null>(null);
  const [stepsSinceTrade, setStepsSinceTrade] = useState(0);
  const [decisionInterval] = useState(300); // ms
  const [onCooldown, setOnCooldown] = useState(false);
  const [kickProgress, setKickProgress] = useState(100);

  useEffect(() => {
    const unsub = orchestrator.on('trade_decision', (data: TradingDecision) => {
      setLastDecision(data);
      setStepsSinceTrade(0);
      setOnCooldown(true);
      setKickProgress(100);
      setTimeout(() => setOnCooldown(false), 5000);
    });

    const stepInterval = setInterval(() => {
      setStepsSinceTrade((prev) => prev + 1);
    }, decisionInterval);

    const kickInterval = setInterval(() => {
      setKickProgress((prev) => {
        if (onCooldown) return prev;
        const decay = Math.random() * 8;
        return Math.max(0, prev - decay);
      });
    }, 2000);

    return () => {
      unsub();
      clearInterval(stepInterval);
      clearInterval(kickInterval);
    };
  }, [decisionInterval, onCooldown]);

  const getKickColor = () => {
    if (kickProgress > 60) return '#00c853';
    if (kickProgress > 30) return '#ffc107';
    return '#ff1744';
  };

  return (
    <div className="live-trader">
      <h3 className="section-title">Live Trader</h3>

      {/* KICK bar */}
      <div className="kick-container">
        <div className="kick-label">
          <span>KICK Mechanism</span>
          <span style={{ color: getKickColor(), fontWeight: 600 }}>
            {kickProgress.toFixed(0)}%
          </span>
        </div>
        <div className="kick-bar-bg">
          <div
            className="kick-bar-fill"
            style={{
              width: `${kickProgress}%`,
              backgroundColor: getKickColor(),
              transition: 'width 0.3s ease, background-color 0.3s ease',
            }}
          />
        </div>
      </div>

      {/* Decision info */}
      <div className="trader-info">
        <div className="trader-info-row">
          <span className="trader-info-label">Last Action</span>
          <span className="trader-info-value">
            {lastDecision ? (
              <>
                <span className={`direction-badge ${lastDecision.action === 'BUY' ? 'long' : 'short'}`}>
                  {lastDecision.action}
                </span>
                <span style={{ marginLeft: 8 }}>{lastDecision.symbol}</span>
                <span style={{ marginLeft: 8, color: '#888' }}>
                  ({(lastDecision.confidence).toFixed(0)}% confidence)
                </span>
              </>
            ) : (
              <span style={{ color: '#888' }}>Waiting...</span>
            )}
          </span>
        </div>
        <div className="trader-info-row">
          <span className="trader-info-label">Steps Since Trade</span>
          <span className="trader-info-value">{stepsSinceTrade}</span>
        </div>
        <div className="trader-info-row">
          <span className="trader-info-label">Decision Interval</span>
          <span className="trader-info-value">{decisionInterval}ms</span>
        </div>
        <div className="trader-info-row">
          <span className="trader-info-label">Cooldown</span>
          <span className={`trader-info-value ${onCooldown ? 'cooldown-active' : 'cooldown-inactive'}`}>
            {onCooldown ? 'Active' : 'Inactive'}
          </span>
        </div>
      </div>

      {/* Last decision details */}
      {lastDecision && (
        <div className="trader-decision-detail">
          <div className="detail-row">
            <span>Agent</span>
            <span className="detail-value">{lastDecision.agentId}</span>
          </div>
          <div className="detail-row">
            <span>Lots</span>
            <span className="detail-value">{lastDecision.lots.toFixed(2)}</span>
          </div>
          <div className="detail-row">
            <span>Stop Loss</span>
            <span className="detail-value">{lastDecision.stopLoss.toFixed(4)}</span>
          </div>
          <div className="detail-row">
            <span>Take Profit</span>
            <span className="detail-value">{lastDecision.takeProfit.toFixed(4)}</span>
          </div>
        </div>
      )}
    </div>
  );
}