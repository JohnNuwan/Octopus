# 🐙 Octopus — Multi-Agent Trading Engine

> Architecture JEPA + World Model pour le trading algorithmique XAUUSD.
> Optimisé pour les challenges FTMO 10K.

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://python.org)
[![Rust](https://img.shields.io/badge/Rust-2021-orange?logo=rust)](https://rust-lang.org)
[![Go](https://img.shields.io/badge/Go-1.23-blue?logo=go)](https://go.dev)
[![Julia](https://img.shields.io/badge/Julia-1.11-purple?logo=julia)](https://julialang.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5-blue?logo=typescript)](https://typescriptlang.org)
[![FastAPI](https://img.shields.io/badge/Agent-FastAPI-teal?logo=fastapi)](https://fastapi.tiangolo.com)
[![CI](https://github.com/JohnNuwan/Octopus/actions/workflows/ci.yml/badge.svg)](https://github.com/JohnNuwan/Octopus/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-130-green)](docs/testing.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Features

### 🔴 Mode FTMO Challenge (Court-terme)
- **TS-JEPA** : encodeur auto-supervisé basé sur VICReg (Bardes, LeCun, 2022)
- **RSSM World Model** : prédit les transitions du marché dans l'espace latent (DreamerV3)
- **MuZero-style MCTS** : 150 simulations par décision dans l'espace latent
- **Actor-Critic** : entraîné exclusivement sur des trajectoires imaginaires
- **20 features** : 5 techniques + 15 macro (calendrier, sentiment, taux CB)
- **FTMO Rules** : Daily/Total Drawdown, Profit Target, 2 phases
- **SLBE** : Stop Loss Break Even — verrouille les gains à +0.5%
- **KICK** : mécanisme anti-overtrading

### 🟢 Mode Investissement Long-terme
- **Markowitz / Risk Parity / Lazy** : optimisation de portefeuille
- **DCA Manager** : Dollar Cost Averaging + Sparplan
- **Trade Republic API** : exécution réelle
- **Projection 20 ans** : intérêts composés

### 🤖 Agent Manager System
- **5 agents spécialisés** : Training, Research, Macro, Trading, Strategy
- **AgentManager** : orchestrateur central avec ordonnancement
- **Dashboard** : monitoring temps réel (FastAPI, port 9093)
- **Intégration Hermes** : cron jobs + skills pour la production

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │         AGENT MANAGER         │
                    │   Training • Research • Macro │
                    │   Trading • Strategy          │
                    └──────┬──────────┬─────────────┘
                           │          │
           ┌───────────────┼──────────┼───────────────┐
           │               │          │               │
    ┌──────▼──────┐ ┌─────▼─────┐ ┌──▼───────┐ ┌─────▼──────┐
    │    ENGINE    │ │   MACRO   │ │  QUANT   │ │   INVEST   │
    │ Python/JAX   │ │  Python   │ │  Julia   │ │  Python    │
    │ JEPA + WM    │ │ Calendar  │ │ Kelly    │ │ Trade Rep. │
    │ Actor-Critic │ │ Sentiment │ │ VaR/CVaR │ │ DCA/Portf. │
    │ MCTS 150 sim │ │ Banks CB  │ │ Markow.  │ │ Projection │
    └──────┬───────┘ └───────────┘ └──────────┘ └────────────┘
           │
    ┌──────▼───────────────────────────────────────┐
    │           ORCHESTRATOR (Go)                    │
    │    gRPC • WebSocket • Monitoring • Routing     │
    └──────┬────────────────────────────────────────┘
           │
    ┌──────▼────────┐          ┌───────────────────┐
    │   EXECUTION   │          │   WEB DASHBOARD    │
    │   Rust/Tonic  │          │  TypeScript/React  │
    │ MT5 Orders    │          │  Equity • P&L      │
    │ IOC/FOK <1ms  │          │  Positions • Risk  │
    └───────────────┘          └───────────────────┘
```

## Modules

| Service | Langage | Stack | Rôle |
|---------|---------|-------|------|
| **Engine** | Python | PyTorch, JEPA, WM, AC, MCTS | RL, encodage, décisions |
| **Macro** | Python | investpy, FinBERT, RSS | Calendrier, sentiment, taux CB |
| **Quant** | Julia | Optimisation | Kelly, VaR, Markowitz |
| **Orchestrator** | Go | gRPC, WebSocket | Routage central |
| **Execution** | Rust | Tonic, MT5 API | Ordres < 1ms |
| **Invest** | Python | Trade Republic | DCA, Portfolio |
| **Web** | TypeScript | React, Recharts | Dashboard temps réel |
| **Agent** | Python | FastAPI, asyncio | Agents intelligents |

## Quick Start

```bash
# Cloner
git clone https://github.com/JohnNuwan/Octopus.git
cd Octopus

# Lancer tous les services
docker compose up -d

# Dashboards
# → http://localhost:3000  (Web)
# → http://localhost:9093  (Agent Manager)

# Logs
docker compose logs -f orchestrator
```

## Tests

```bash
# Engine (77 tests)
cd engine && source /home/kadeva/trading-env/bin/activate && python -m pytest tests/ -v

# Macro (27 tests)
cd ../macro && python -m pytest tests/ -v

# Agent (26 tests)
cd ../agent && python -m pytest tests/ -v
```

**130 tests — tous passent.** Modules : Engine, JEPA, World Model, Actor-Critic, MCTS, Replay, Trader, Macro, Agent Manager.

## Documentation complète

| Doc | Description |
|-----|-------------|
| [Architecture](docs/architecture.md) | 8 services, flux de données, protocoles |
| [Engine](docs/engine.md) | JEPA, World Model, Actor-Critic, MCTS, Training |
| [Macro](docs/macro.md) | Calendrier, sentiment, banques centrales, features |
| [API](docs/api.md) | gRPC, WebSocket, ports, variables d'env |
| [Déploiement](docs/deployment.md) | Docker Compose, GPU, configuration |
| [Tests](docs/testing.md) | 130 tests, coverage, ajout de tests |
| [Contribution](docs/contributing.md) | Structure, workflow, style de code |
| [Hermes Config](hermes-config/README.md) | Pipeline productif via Hermes Agent |

## Papiers Clés

| Papier | Contribution |
|--------|-------------|
| [MuZero](https://arxiv.org/abs/1911.08265) (Schrittwieser, 2019) | Model-based RL avec MCTS latent |
| [DreamerV3](https://arxiv.org/abs/2301.04104) (Hafner, 2023) | World Model, RSSM, Symlog |
| [VICReg](https://arxiv.org/abs/2105.04906) (Bardes, LeCun, 2022) | Self-supervised, variance-invariance-covariance |
| [TS-JEPA](https://arxiv.org/abs/2509.25449) (2025) | Time-Series JEPA |
| [FLAG-Trader](https://arxiv.org/abs/2503.20533) (2025) | LLM + PPO hybride |
| [CausalFormer](https://arxiv.org/abs/2410.14091) (2024) | Découverte causale temporelle |

## Licence

MIT — Recherche et usage personnel uniquement. Le trading comporte des risques financiers.