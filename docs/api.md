# API Reference

## gRPC Protocol

Défini dans `orchestrator/proto/octopus.proto` — package `octopus`.

### Service `Orchestrator`

```protobuf
service Orchestrator {
    rpc SendTradingDecision(TradingDecision) returns (DecisionAck);
    rpc GetMarketData(MarketDataRequest) returns (MarketDataResponse);
    rpc ConfirmOrderExecution(OrderConfirmation) returns (ConfirmationAck);
    rpc HealthCheck(HealthRequest) returns (HealthResponse);
}
```

#### SendTradingDecision
Engine → Orchestrator : envoie une décision de trading.

```protobuf
message TradingDecision {
    string symbol = 1;          // XAUUSD
    int32 action = 2;           // 0=Hold, 1=Buy, 2=Sell, 3=Split, 4=Close
    double lots = 3;            // Taille de position
    double stop_loss = 4;       // Prix du stop loss
    double take_profit = 5;     // Prix du take profit
    double confidence = 6;      // Confiance de l'agent (0-1)
    string agent_id = 7;        // ID de l'agent
    int64 timestamp = 8;        // Timestamp Unix
}

message DecisionAck {
    bool accepted = 1;
    string reason = 2;           // Raison du refus si applicable
}
```

#### GetMarketData
Engine → Orchestrator : demande des données de marché.

```protobuf
message MarketDataRequest {
    string symbol = 1;
    int32 num_candles = 2;       // Nombre de bougies
    string timeframe = 3;        // M1, M5, M15, H1, D1
}

message OHLCV {
    double open = 1;
    double high = 2;
    double low = 3;
    double close = 4;
    double volume = 5;
    int64 timestamp = 6;
}

message MarketDataResponse {
    repeated OHLCV candles = 1;
    string symbol = 2;
}
```

#### ConfirmOrderExecution
Execution → Orchestrator : confirme l'exécution d'un ordre.

```protobuf
message OrderConfirmation {
    string order_id = 1;
    string symbol = 2;
    int32 action = 3;
    double lots = 4;
    double executed_price = 5;
    double pnl = 6;
    int64 timestamp = 7;
}

message ConfirmationAck {
    bool received = 1;
}
```

#### HealthCheck
```protobuf
message HealthRequest {}
message HealthResponse {
    bool healthy = 1;
    string version = 2;
    int64 uptime_seconds = 3;
}
```

## WebSocket

Le dashboard web se connecte à `ws://orchestrator:8080/ws` et reçoit :

| Événement | Payload | Source |
|-----------|---------|--------|
| `trade_decision` | TradingDecision | Engine |
| `order_confirmation` | OrderConfirmation | Execution |
| `market_data` | MarketDataResponse | Data Provider |
| `health` | HealthResponse | Auto |
| `service_status` | ServiceState | Auto |

## Variables d'environnement

| Variable | Service | Défaut | Description |
|----------|---------|--------|-------------|
| `ORCHESTRATOR_HOST` | Tous | orchestrator | Adresse du serveur gRPC |
| `ORCHESTRATOR_PORT` | Tous | 9091 | Port gRPC |
| `LOG_LEVEL` | Orchestrator | info | Niveau de log |
| `MT5_LOGIN` | Execution | — | Login MT5 |
| `MT5_PASSWORD` | Execution | — | Password MT5 |
| `MT5_SERVER` | Execution | — | Serveur MT5 |
| `TR_PHONE` | Invest | — | Téléphone Trade Republic |
| `TR_PIN` | Invest | — | PIN Trade Republic |
| `INVEST_STRATEGY` | Invest | core_satellite | Stratégie d'investissement |
| `MONTHLY_BUDGET` | Invest | 300 | Budget mensuel |
| `JAX_PLATFORM_NAME` | Engine | gpu | Plateforme JAX |
| `JULIA_NUM_THREADS` | Quant | 32 | Threads Julia |

## Port mapping

| Port | Service | Protocole | Usage |
|------|---------|-----------|-------|
| 9091 | Orchestrator | gRPC | Communication interne |
| 8080 | Orchestrator | HTTP/WS | Health check + WebSocket |
| 3000 | Web | HTTP | Dashboard React |