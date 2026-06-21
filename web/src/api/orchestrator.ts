import { TradingDecision, Position, Trade, ServiceStatus, MarketData, DashboardState } from '../types';

// ── Synthetic data generators ──────────────────────────────────────

function randomBetween(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

const SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'BTCUSD', 'ETHUSD', 'XAUUSD'];

let equitySeed = 100000;
const balanceSeed = 100000;

function nextEquityPoint(): { equity: number; balance: number; dailyPnL: number } {
  const change = randomBetween(-800, 1200);
  equitySeed += change;
  const dailyPnL = change;
  return { equity: Math.round(equitySeed * 100) / 100, balance: balanceSeed, dailyPnL: Math.round(dailyPnL * 100) / 100 };
}

function generateEquityHistory(points: number): { timestamp: number; equity: number; volume: number }[] {
  const history: { timestamp: number; equity: number; volume: number }[] = [];
  let eq = 100000;
  const now = Date.now();
  for (let i = points; i >= 0; i--) {
    eq += randomBetween(-400, 600);
    history.push({
      timestamp: now - i * 60000,
      equity: Math.round(eq * 100) / 100,
      volume: Math.round(randomBetween(10, 500)),
    });
  }
  return history;
}

function generatePositions(): Position[] {
  const count = Math.floor(randomBetween(1, 5));
  const positions: Position[] = [];
  const used = new Set<string>();
  for (let i = 0; i < count; i++) {
    const symbol = pick(SYMBOLS.filter(s => !used.has(s)));
    if (!symbol) break;
    used.add(symbol);
    const direction = Math.random() > 0.5 ? 'LONG' : 'SHORT';
    const entryPrice = Math.round(randomBetween(1.05, 1.25) * 10000) / 10000;
    const change = randomBetween(-0.03, 0.03);
    const currentPrice = Math.round((entryPrice + (direction === 'LONG' ? change : -change)) * 10000) / 10000;
    const lots = Math.round(randomBetween(0.01, 2) * 100) / 100;
    const pnl = Math.round((currentPrice - entryPrice) * lots * (direction === 'LONG' ? 1 : -1) * 100000 * 100) / 100;
    positions.push({
      symbol,
      direction: direction as 'LONG' | 'SHORT',
      entryPrice,
      currentPrice,
      lots,
      unrealizedPnL: pnl,
      slbeActive: Math.random() > 0.4,
    });
  }
  return positions;
}

function generateTrades(count: number): Trade[] {
  const trades: Trade[] = [];
  const now = Date.now();
  for (let i = 0; i < count; i++) {
    const symbol = pick(SYMBOLS);
    const direction = Math.random() > 0.5 ? 'LONG' : 'SHORT';
    const entryPrice = Math.round(randomBetween(1.05, 1.25) * 10000) / 10000;
    const exitPrice = Math.round((entryPrice + randomBetween(-0.05, 0.05)) * 10000) / 10000;
    const lots = Math.round(randomBetween(0.01, 1.5) * 100) / 100;
    const pnl = Math.round((exitPrice - entryPrice) * lots * (direction === 'LONG' ? 1 : -1) * 100000 * 100) / 100;
    const entryTime = new Date(now - i * randomBetween(300000, 3600000)).toISOString();
    const exitTime = new Date(new Date(entryTime).getTime() + randomBetween(60000, 7200000)).toISOString();
    trades.push({
      id: `T-${String(1000 + i).padStart(4, '0')}`,
      symbol,
      direction: direction as 'LONG' | 'SHORT',
      entryPrice,
      exitPrice,
      lots,
      pnl,
      entryTime,
      exitTime,
    });
  }
  return trades;
}

function generateHealth(): DashboardState {
  const ep = nextEquityPoint();
  return {
    equity: ep.equity,
    balance: ep.balance,
    dailyPnL: ep.dailyPnL,
    totalReturn: Math.round(((ep.equity - 100000) / 100000) * 10000) / 100,
    ftmoPhase: pick(['Phase 1', 'Phase 2', 'Funded', 'Evaluation']),
    ftmoStatus: pick(['On Track', 'At Risk', 'Passed', 'In Progress']),
  };
}

function generateServiceStatus(): ServiceStatus[] {
  const now = new Date().toISOString();
  const services = ['Orchestrator', 'Engine', 'Execution', 'Quant', 'Web', 'Invest'];
  return services.map((name) => ({
    id: name.toLowerCase(),
    type: name,
    connectedAt: new Date(Date.now() - randomBetween(3600000, 86400000)).toISOString(),
    lastSeen: now,
  }));
}

// ── Synthetic data topics ──────────────────────────────────────────

let syntheticInterval: ReturnType<typeof setInterval> | null = null;
const listeners: Record<string, Set<(data: any) => void>> = {};

function emit(event: string, data: any) {
  if (listeners[event]) {
    listeners[event].forEach((fn) => fn(data));
  }
}

function startSyntheticData() {
  if (syntheticInterval) return;
  // Emit health every 5s
  syntheticInterval = setInterval(() => {
    emit('health', generateHealth());
    emit('service_status', generateServiceStatus());
    emit('market_data', {
      symbol: pick(SYMBOLS),
      candles: Array.from({ length: 50 }, (_, i) => ({
        open: randomBetween(1.05, 1.15),
        high: randomBetween(1.08, 1.18),
        low: randomBetween(1.02, 1.12),
        close: randomBetween(1.05, 1.15),
        volume: randomBetween(100, 1000),
        timestamp: Date.now() - (50 - i) * 60000,
      })),
    });

    // Occasionally emit a trade decision
    if (Math.random() > 0.7) {
      emit('trade_decision', {
        symbol: pick(SYMBOLS),
        action: pick(['BUY', 'SELL']),
        lots: Math.round(randomBetween(0.1, 2) * 100) / 100,
        stopLoss: Math.round(randomBetween(1.00, 1.10) * 10000) / 10000,
        takeProfit: Math.round(randomBetween(1.15, 1.30) * 10000) / 10000,
        confidence: Math.round(randomBetween(55, 95)),
        agentId: pick(['alpha', 'beta', 'gamma', 'delta']),
        timestamp: new Date().toISOString(),
      });
    }
  }, 5000);
}

// ── Public API ─────────────────────────────────────────────────────

export type EventCallback = (data: any) => void;

class OrchestratorAPI {
  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private baseDelay = 1000;
  private syntheticMode = false;
  private eventListeners: Record<string, Set<EventCallback>> = {};

  connect(url = 'ws://localhost:8080/ws'): void {
    if (this.ws?.readyState === WebSocket.OPEN || this.ws?.readyState === WebSocket.CONNECTING) return;

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        console.log('[Orchestrator WS] Connected');
        this.reconnectAttempts = 0;
        this.syntheticMode = false;
        if (syntheticInterval) {
          clearInterval(syntheticInterval);
          syntheticInterval = null;
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type && this.eventListeners[msg.type]) {
            this.eventListeners[msg.type].forEach((fn) => fn(msg.data || msg.payload));
          }
          // Also dispatch to generic listeners
          if (this.eventListeners['message']) {
            this.eventListeners['message'].forEach((fn) => fn(msg));
          }
        } catch {
          // ignore malformed messages
        }
      };

      this.ws.onclose = () => {
        console.log('[Orchestrator WS] Disconnected');
        this.ws = null;
        this.scheduleReconnect(url);
        this.fallbackToSynthetic();
      };

      this.ws.onerror = () => {
        console.log('[Orchestrator WS] Error');
        this.ws?.close();
      };
    } catch {
      console.log('[Orchestrator WS] Failed to connect, using synthetic data');
      this.fallbackToSynthetic();
    }
  }

  private scheduleReconnect(url: string): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('[Orchestrator WS] Max reconnect attempts reached');
      return;
    }
    const delay = Math.min(this.baseDelay * Math.pow(2, this.reconnectAttempts), 30000);
    this.reconnectAttempts++;
    console.log(`[Orchestrator WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    this.reconnectTimer = setTimeout(() => this.connect(url), delay);
  }

  private fallbackToSynthetic(): void {
    if (this.syntheticMode) return;
    this.syntheticMode = true;
    console.log('[Orchestrator] Switching to synthetic data mode');
    startSyntheticData();
  }

  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (syntheticInterval) {
      clearInterval(syntheticInterval);
      syntheticInterval = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  on(event: string, callback: EventCallback): () => void {
    if (!this.eventListeners[event]) {
      this.eventListeners[event] = new Set();
    }
    this.eventListeners[event].add(callback);
    // Also register in the global listeners for synthetic data
    if (!listeners[event]) {
      listeners[event] = new Set();
    }
    listeners[event].add(callback);

    return () => {
      this.eventListeners[event]?.delete(callback);
      listeners[event]?.delete(callback);
    };
  }

  off(event: string, callback: EventCallback): void {
    this.eventListeners[event]?.delete(callback);
    listeners[event]?.delete(callback);
  }

  // ── Sync data fetchers (for React state initialisation) ─────────

  fetchHealth(): DashboardState {
    return generateHealth();
  }

  fetchPositions(): Position[] {
    return generatePositions();
  }

  fetchTrades(count = 50): Trade[] {
    return generateTrades(count);
  }

  fetchEquityHistory(points = 100): { timestamp: number; equity: number; volume: number }[] {
    return generateEquityHistory(points);
  }

  fetchServiceStatus(): ServiceStatus[] {
    return generateServiceStatus();
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN || false;
  }
}

export const orchestrator = new OrchestratorAPI();
export default orchestrator;
