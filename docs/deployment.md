# Déploiement Octopus

## Prérequis

### Matériel recommandé
- **GPU** : NVIDIA (CUDA 12+) avec 8 GB+ VRAM (48 GB recommandé pour entraînement)
- **RAM** : 32 GB minimum
- **Stockage** : 50 GB (dataset, checkpoints)

### Logiciel
```bash
# Docker
sudo apt install docker.io docker-compose-v2

# NVIDIA GPU (optionnel)
sudo apt install nvidia-driver-545 nvidia-container-toolkit
sudo systemctl restart docker
```

## Démarrage rapide

```bash
# Cloner
git clone https://github.com/JohnNuwan/Octopus.git
cd Octopus

# Configuration
cp .env.example .env  # éditer avec vos credentials

# Lancer tous les services
docker compose up -d

# Vérifier l'état
docker compose ps
docker compose logs -f orchestrator
```

## Configuration

### Fichier `.env`
```bash
# MT5 pour exécution réelle
MT5_LOGIN=12345678
MT5_PASSWORD=****
MT5_SERVER=ICMarkets-Live

# Trade Republic pour investissement
TR_PHONE=+33612345678
TR_PIN=1234
INVEST_STRATEGY=core_satellite
MONTHLY_BUDGET=500
```

### 🐍 Agent Manager
```bash
docker compose up -d agent
# Dashboard : http://localhost:9093
# API : curl http://localhost:9093/api/agents
```

## Services Docker

| Service | Build | Ports | Dépend de |
|---------|-------|-------|-----------|
| orchestrator | Go | 9091, 8080 | — |
| engine | Python | — | orchestrator |
| macro | Python | 9092 | orchestrator |
| execution | Rust | — | orchestrator |
| quant | Julia | — | orchestrator |
| invest | Python | — | orchestrator |
| web | TypeScript | 3000 | orchestrator |
| agent | Python | 9093 | — |

## GPU passthrough

Pour l'entraînement sur GPU :
```bash
# Installer NVIDIA container toolkit
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker

# Lancer avec GPU
docker compose up -d engine
```

Le service engine est configuré pour réserver les GPUs via `deploy.resources.reservations.devices`.

## Monitoring

### Logs
```bash
# Tous les services
docker compose logs -f

# Service spécifique
docker compose logs -f engine
docker compose logs -f orchestrator --tail=100
```

### Dashboard
Accéder au dashboard web : `http://localhost:3000`

### Health check gRPC
```bash
# Via grpcurl (si installé)
grpcurl -plaintext localhost:9091 octopus.Orchestrator/HealthCheck
```

## Mise à jour

```bash
git pull origin main
docker compose up -d --build
```

Pour mettre à jour un seul service :
```bash
docker compose up -d --build engine
```

## Dépannage

| Problème | Cause | Solution |
|----------|-------|----------|
| GPU non détecté | NVIDIA toolkit manquant | `sudo apt install nvidia-container-toolkit` |
| Engine OOM | Trop de paramètres | Réduire `batch_size` ou `hidden_dim` |
| Protoc manquant | Pour build Rust | Utiliser `PROTOC=/path/to/protoc` |
| Port déjà utilisé | Conflit | Changer `ports:` dans docker-compose.yml |
| Erreur gRPC | Service pas prêt | Attendre 10s, vérifier les logs |