# 🐙 Octopus — Multi-Agent Trading Engine

> Architecture JEPA + World Model pour le trading algorithmique XAUUSD.
> Optimisé pour les challenges FTMO 10K.
>
> *"Apprend la structure, ignore le bruit, gagne le marché."*

## Architecture

```
                    ┌─────────────────────────┐
                    │     TS-JEPA ENCODER      │
                    │  Filtre le bruit du      │
                    │  marché via VICReg       │
                    └────────┬────────────────┘
                             │
                    ┌────────▼────────────────┐
                    │   RSSM WORLD MODEL       │
                    │  Prédit les transitions  │
                    │  dans l'espace latent     │
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
                    │  Kelly • HMM • MT5      │
                    └─────────────────────────┘
```

## Modules

| Module | Langage | Technologie |
|--------|---------|-------------|
| **Engine** | Python | PyTorch + JAX, TS-JEPA, RSSM, MuZero-style MCTS |
| **Quant** | Julia | Optimisation Kelly, VaR/CVaR, calculs HPC |
| **Execution** | Rust | Ordres MT5, latence < 1ms, IOC/FOK |
| **Orchestrator** | Go | gRPC, FastAPI, Redis Pub/Sub, monitoring |
| **Web** | TypeScript | Dashboard React, TensorBoard, métriques live |

## Démarrage Rapide

```bash
# Cloner
git clone https://github.com/JohnNuwan/Octopus.git
cd Octopus

# Lancer tous les services
docker compose up -d

# Voir les logs
docker compose logs -f engine orchestrator

# Entraînement (30K steps ~ 6-10h sur 2×3090)
docker compose exec engine python -m src.training.trainer
```

## Papiers Clés

| Papier | Année | Contribution |
|--------|-------|-------------|
| MuZero (Schrittwieser et al.) | 2019 | Model-based RL avec MCTS latent |
| DreamerV3 (Hafner et al.) | 2023/2025 | World Model, RSSM, Symlog, imagination |
| VICReg (Bardes, Ponce, LeCun) | 2022 | Self-supervised, variance-invariance-covariance |
| TS-JEPA | 2025 | Time-Series JEPA, séries temporelles |
| DeePM (Wood et al.) | 2025 | Causal Sieve, EVaR robust, Macro Graph Prior |
| FLAG-Trader | 2025 | LLM + PPO hybride pour trading |
| TAFAS (AAAI 2025) | 2025 | Test-time adaptation, non-stationnarité |
| CausalFormer | 2024 | Découverte causale temporelle |

## Licence

MIT — Recherche et usage personnel uniquement. Trading à risque.