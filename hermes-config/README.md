# Configuration Hermes pour Octopus — Serveur GPU

Ce répertoire contient la configuration Hermes Agent pour orchestrer
la pipeline Octopus sur le serveur GPU.

## Installation

```bash
# 1. Installer Hermes Agent
pip install hermes-agent

# 2. Configurer WireGuard (tunnel sécurisé vers le serveur GPU)
# Éditer /etc/wireguard/octopus.conf avec les clés fournies
sudo wg-quick up octopus

# 3. Copier les skills Hermes
cp -r skills/* ~/.hermes/profiles/default/skills/

# 4. Recharger Hermes
hermes reload
```

## Skills disponibles

| Skill | Fichier | Description |
|-------|---------|-------------|
| Training Pipeline | `skills/octopus-training-pipeline.md` | Entraînement RL complet |
| Research Pipeline | `skills/octopus-research-pipeline.md` | Recherche de papiers arXiv |
| Macro Pipeline | `skills/octopus-macro-pipeline.md` | Analyse macro-économique |
| Backtest Pipeline | `skills/octopus-backtest-pipeline.md` | Backtesting de stratégies |

## Cron Jobs recommandés

Ajouter dans Hermes (`~/.hermes/profiles/default/cron.yaml`) :

```yaml
- schedule: "0 8 * * 1-5"    # 8h UTC, jours ouvrés
  skill: octopus-macro-pipeline
  notify: true

- schedule: "0 9 * * 1"       # 9h UTC, lundi
  skill: octopus-research-pipeline
  notify: true

- schedule: "0 22 * * 5"      # 22h UTC, vendredi
  skill: octopus-backtest-pipeline
  notify: true
```

## Vérification

```bash
# Tester la connexion GPU
python3 -c "import torch; print('GPU disponible:', torch.cuda.is_available())"

# Lister les skills Hermes actifs
hermes skills list

# Tester un skill
hermes run octopus-training-pipeline --dry-run
```

## Architecture des agents

Le service Python `agent/` fournit le moteur d'exécution backend.
Hermes sert de planificateur et d'interface utilisateur.
Les skills Hermes appellent le service agent via l'API sur le port 9093.