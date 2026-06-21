export interface TradingDecision {
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  lots: number;
  stopLoss: number;
  takeProfit: number;
  confidence: number;
  agentId: string;
  timestamp: string;
}

export interface OHLCV {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp: number;
}

export interface MarketData {
  symbol: string;
  candles: OHLCV[];
}

export interface Position {
  symbol: string;
  direction: 'LONG' | 'SHORT';
  entryPrice: number;
  currentPrice: number;
  lots: number;
  unrealizedPnL: number;
  slbeActive: boolean;
}

export interface Trade {
  id: string;
  symbol: string;
  direction: 'LONG' | 'SHORT';
  entryPrice: number;
  exitPrice: number;
  lots: number;
  pnl: number;
  entryTime: string;
  exitTime: string;
}

export interface ServiceStatus {
  id: string;
  type: string;
  connectedAt: string;
  lastSeen: string;
}

export interface DashboardState {
  equity: number;
  balance: number;
  dailyPnL: number;
  totalReturn: number;
  ftmoPhase: string;
  ftmoStatus: string;
}