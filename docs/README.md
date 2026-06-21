# 📚 Documentation Octopus

> Multi-Agent Trading Engine — JEPA + World Model pour XAUUSD FTMO

---

## Guide de démarrage

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | Vue d'ensemble des 8 services, flux de données, communication |
| [Engine](engine.md) | Moteur Python : JEPA, World Model, MuZero, Actor-Critic, MCTS |
| [Macro](macro.md) | Calendrier économique, sentiment news, banques centrales |
| [API Reference](api.md) | gRPC Protocol, WebSocket, variables d'environnement |
| [Déploiement](deployment.md) | Docker Compose, GPU, configuration, monitoring |
| [Tests](testing.md) | 130 tests unitaires, coverage, ajout de tests |
| [Contribution](contributing.md) | Structure du repo, workflow, style de code |
| [Hermes Config](../hermes-config/README.md) | Pipeline productif via Hermes Agent |

---

## Structure du projet

```
Octopus/
├── engine/src/           Moteur Python (RL/ML)
│   ├── networks/         JEPA, World Model, Actor-Critic
│   ├── mcts/             Monte Carlo Tree Search
│   ├── environment/      Trading env + FTMO + SLBE
│   ├── training/         Trainer + Replay Buffer
│   ├── live/             Trader temps réel
│   └── config/           Hyperparamètres
├── orchestrator/         Go gRPC server
│   ├── cmd/main.go       Point d'entrée
│   └── proto/            Définitions gRPC
├── execution/            Rust execution engine
│   └── src/              Client gRPC + MT5 connector
├── quant/                Julia quant module
│   └── src/              Kelly, VaR, Markowitz
├── invest/               Long-term investment
│   └── src/              Trade Republic + Portfolio + DCA
├── web/                  Dashboard TypeScript/React
│   └── src/              Components, API client, types
├── docs/                 Documentation
└── docker-compose.yml    6 services orchestrés
```