# 🐙 Octopus — Multi-Agent Trading Engine

> Architecture JEPA + World Model pour le trading algorithmique XAUUSD.
> Optimisé pour les challenges FTMO 10K.

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://python.org)
[![Rust](https://img.shields.io/badge/Rust-2021-orange?logo=rust)](https://rust-lang.org)
[![Go](https://img.shields.io/badge/Go-1.23-blue?logo=go)](https://go.dev)
[![Julia](https://img.shields.io/badge/Julia-1.11-purple?logo=julia)](https://julialang.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5-blue?logo=typescript)](https://typescriptlang.org)
[![CI](https://github.com/JohnNuwan/Octopus/actions/workflows/ci.yml/badge.svg)](https://github.com/JohnNuwan/Octopus/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Features

### 🔴 Mode FTMO Challenge (Court-terme)
- **TS-JEPA** : encodeur auto-supervisé basé sur VICReg (Bardes, LeCun, 2022)
- **RSSM World Model** : prédit les transitions du marché dans l'espace latent (DreamerV3)
- **MuZero-style MCTS** : 150 simulations par décision dans l'espace latent
- **Actor-Critic** : entraîné exclusivement sur des trajectoires imaginaires
- **FTMO Rules** : Daily/Total Drawdown, Profit Target, 2 phases
- **SLBE** : Stop Loss Break Even pour protéger les gains
- **KICK** : mécanisme anti-overtrading

### 🟢 Mode Investissement Long-terme
- **Markowitz / Risk Parity / Lazy** : optimisation de portefeuille
- **DCA Manager** : Dollar Cost Averaging + Sparplan
- **Trade Republic API** : exécution réelle
- **Projection 20 ans** : intérêts composés

---

## Architecture

```
                    ┌─────────────────────────┐
                    │     TS-JEPA ENCODER      │
                    │  (VICReg, Momentum)      │
                    └────────┬────────────────┘
                             │
                    ┌────────▼────────────────┐
                    │   RSSM WORLD MODEL       │
                    │  (GRU + 32×32 Latent)    │
                    └────────┬────────────────┘
                             │
                    ┌────────▼────────────────┐
                    │   ACTOR-CRITIC (PPO)    │
                    │  Entraîné sur rêves     │
                    │  (15 pas d'imagination)  │
                    └────────┬────────────────┘
                             │
                    ┌────────▼────────────────┐
                    │  RISK & EXECUTION       │
                    │  FTMO • SLBE • KICK    │
                    │  Kelly • VaR • MT5      │
                    └─────────────────────────┘
```

## Modules

| Service | Langage | Technologie | Rôle |
|---------|---------|-------------|------|
| **Engine** | Python | PyTorch, MuZero, JEPA | RL, encodage, décisions |
| **Orchestrator** | Go | gRPC, WebSocket | Routage, monitoring |
| **Execution** | Rust | Tonic, MT5 API | Ordres < 1ms |
| **Quant** | Julia | Optimisation | Kelly, VaR, Markowitz |
| **Invest** | Python | Trade Republic | DCA, Portfolio |
| **Web** | TypeScript | React, Recharts | Dashboard temps réel |

## Quick Start

```bash
# Cloner
git clone https://github.com/JohnNuwan/Octopus.git
cd Octopus

# Lancer tous les services
docker compose up -d

# Dashboard → http://localhost:3000
# Logs
docker compose logs -f orchestrator
```

## Tests

```bash
cd engine
source /home/kadeva/trading-env/bin/activate
python -m pytest tests/ -v
```

**77 tests — modules :** Environment, JEPA, World Model, Actor-Critic, MCTS, Replay Buffer, Live Trader

## Documentation complète

| Documentation | Description |
|---------------|-------------|
| [Architecture](docs/architecture.md) | Vue d'ensemble des 6 services et flux |
| [Engine](docs/engine.md) | Détail du moteur Python (JEPA, WM, AC, MCTS) |
| [API](docs/api.md) | Protocole gRPC, WebSocket, variables d'env |
| [Déploiement](docs/deployment.md) | Docker Compose, GPU, configuration |
| [Tests](docs/testing.md) | 77 tests, coverage, ajout de tests |
| [Contribution](docs/contributing.md) | Structure, workflow, style de code |

## Papiers Clés

| Papier | Contribution |
|--------|-------------|
| [MuZero](https://arxiv.org/abs/1911.08265) (Schrittwieser et al., 2019) | Model-based RL avec MCTS latent |
| [DreamerV3](https://arxiv.org/abs/2301.04104) (Hafner et al., 2023) | World Model, RSSM, Symlog |
| [VICReg](https://arxiv.org/abs/2105.04906) (Bardes, Ponce, LeCun, 2022) | Self-supervised, variance-invariance-covariance |
| [TS-JEPA](https://arxiv.org/abs/2509.25449) (2025) | Time-Series JEPA |
| [FLAG-Trader](https://arxiv.org/abs/2503.20533) (2025) | LLM + PPO hybride |
| [CausalFormer](https://arxiv.org/abs/2410.14091) (2024) | Découverte causale temporelle |

## Licence

MIT — Recherche et usage personnel uniquement. Le trading comporte des risques financiers.