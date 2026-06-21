# Architecture Octopus

## Vue d'ensemble

Octopus est une architecture multi-agent pour le trading algorithmique XAUUSD, optimisée pour les challenges FTMO 10K. Elle combine l'apprentissage par renforcement profond (MuZero), les modèles du monde (World Models), l'encodage auto-supervisé (JEPA/VICReg), et l'exécution basse latence (Rust).

```
                    ┌─────────────────────────────────────┐
                    │           DONNÉES MARCHÉ            │
                    │     XAUUSD (M15/M1) • MT5 • API     │
                    └────────┬────────────────────┬───────┘
                             │                    │
                    ┌────────▼────────┐   ┌───────▼────────┐
                    │    ENGINE       │   │     QUANT      │
                    │  Python/PyTorch │   │    Julia       │
                    │  ┌────────────┐ │   │  ┌──────────┐ │
                    │  │ TS-JEPA    │ │   │  │ Kelly    │ │
                    │  │ (VICReg)   │ │   │  │ VaR/CVaR │ │
                    │  └─────┬──────┘ │   │  │ Markowitz│ │
                    │  ┌─────▼──────┐ │   │  └──────────┘ │
                    │  │ RSSM WM    │ │   └───────┬────────┘
                    │  └─────┬──────┘ │           │
                    │  ┌─────▼──────┐ │           │
                    │  │ Actor-Critic│ │           │
                    │  │ + MCTS     │ │           │
                    │  └─────┬──────┘ │           │
                    └────────┼────────┘           │
                             │                    │
                    ┌────────▼────────────────────▼────────┐
                    │           ORCHESTRATOR (Go)            │
                    │  gRPC • WebSocket • Monitoring • Routing│
                    └────────┬────────────────────┬──────────┘
                             │                    │
                    ┌────────▼────────┐   ┌───────▼────────┐
                    │   EXECUTION     │   │      WEB       │
                    │   Rust          │   │  TypeScript    │
                    │  MT5 Orders     │   │  Dashboard     │
                    │  IOC/FOK        │   │  Métriques     │
                    │  < 1ms          │   │  Configuration │
                    └────────┬────────┘   └────────────────┘
                             │
                    ┌────────▼────────┐
                    │      MT5        │
                    │  Compte FTMO    │
                    └─────────────────┘
```

## Services

| Service | Langage | Technologie | Rôle |
|---------|---------|-------------|------|
| **Orchestrator** | Go | gRPC, WebSocket | Cerveau central, routage, monitoring |
| **Engine** | Python | PyTorch, JAX, MuZero | RL, encodage JEPA, World Model, décisions |
| **Execution** | Rust | Tonic, MT5 API | Ordres latence critique, IOC/FOK |
| **Quant** | Julia | LinearAlgebra | Kelly, VaR, optimisation portefeuille |
| **Invest** | Python | Trade Republic API | Investissement long terme, DCA |
| **Web** | TypeScript | React, Recharts | Dashboard monitoring temps réel |

## Flux de données

### Mode FTMO Challenge (court-terme)

```
1. Market Data (MT5) ──► Engine (observation OHLCV)
2. Engine ──► JEPA Encoder ──► Embedding latent
3. Embedding ──► RSSM World Model ──► État latent
4. État latent ──► MCTS (150 simulations) ──► Distribution d'actions
5. Action choisie ──► gRPC ──► Orchestrator
6. Orchestrator ──► Execution (Rust) ──► MT5
7. Confirmation exécution ──► Orchestrator ──► Engine (Replay Buffer)
8. Web Dashboard ← WebSocket ← Orchestrator
```

### Mode Investissement long terme

```
1. Trade Republic API ← Invest Manager
2. Portfolio Optimizer (Markowitz / Risk Parity)
3. DCA Manager (Sparplan, rebalance mensuel)
4. Projection 20 ans (intérêts composés)
5. Exécution via API Trade Republic
```

## Communication inter-services

- **gRPC** : communication synchrone entre Engine ↔ Orchestrator ↔ Execution
- **WebSocket** : streaming temps réel Orchestrator → Web Dashboard
- **Volumes Docker** : partage de données (weights, results)
- **Variables d'environnement** : configuration partagée via docker-compose

## Workflow FTMO Challenge

### Phase 1 (Simulation)
- Capital initial : $10,000
- Objectif : +10% ($1,000)
- Perte max quotidienne : 5% ($500)
- Perte max totale : 10% ($1,000)
- Jours min de trading : 10

### Phase 2 (Simulation)
- Capital : solde de fin de phase 1
- Objectif : +10%
- Perte max quotidienne : 5%
- Perte max totale : 10%

### Funded Account
- Capital réel fourni par FTMO
- Partage des profits : 80% trader / 20% FTMO
- Règles identiques (daily/total drawdown)