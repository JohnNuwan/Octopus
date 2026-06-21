# Contribution à Octopus

## Structure du repo

```
Octopus/
├── engine/            Python (PyTorch) — cœur RL
├── macro/             Python — calendrier, sentiment, CB
├── execution/         Rust — exécution latence critique
├── orchestrator/      Go — routage gRPC central
├── quant/             Julia — calculs quantitatifs
├── invest/            Python — investissement long terme
├── agent/             Python — agents intelligents + dashboard
├── web/               TypeScript/React — dashboard
├── hermes-config/     Skills/cron pour Hermes Agent
└── docs/              Documentation
```

## Workflow Git

1. **Brancher** : `git checkout -b feature/nom-de-la-feature`
2. **Développer** avec tests
3. **Tester** : exécuter les tests avant commit
4. **Commit** : message clair en français ou anglais (préfixes : `feat:` `fix:` `docs:` `test:`)
5. **PR vers main** : revue par un pair

## Style de code

### Python
- **Docstrings** : Google Style PEP257, en français
- **Typage** : annotations de type obligatoires
- **Format** : Black (line-length=100)
- **Import order** : stdlib → tiers → local

```python
def calculate_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25
) -> float:
    \"\"\"Calcule la taille de position Kelly fractionnaire.
    
    Args:
        win_rate: Probabilité de gain.
        avg_win: Gain moyen par trade gagnant.
        avg_loss: Perte moyenne par trade perdant.
        fraction: Fraction du Kelly complet.
        
    Returns:
        Pourcentage du capital à risquer.
    \"\"\"
    b = avg_win / abs(avg_loss)
    p = win_rate
    q = 1.0 - p
    full_kelly = (p * b - q) / b
    return max(0.0, full_kelly * fraction)
```

### Rust
- **Style** : `rustfmt` par défaut
- **Erreurs** : `anyhow::Result` pour le code applicatif
- **Logging** : `tracing` (pas `println`)
- **Async** : `tokio` avec `tonic` pour gRPC

### Go
- **Style** : `gofmt` par défaut
- **Organisation** : packages dans `internal/` (router, registry, monitoring)
- **Logging** : `log` standard ou `slog`

### TypeScript
- **Style** : `prettier` par défaut
- **Composants** : React fonctionnels avec hooks
- **Types** : interfaces plutôt que types

## Tests

```bash
# Tous les tests
cd engine && source ../trading-env/bin/activate && python -m pytest

# Tests spécifiques
python -m pytest tests/test_jepa.py -v

# Avec coverage
python -m pytest --cov=src tests/
```

Toute nouvelle fonctionnalité doit inclure des tests. Voir [testing.md](testing.md).

## Skills et mémoire Hermes

Ce projet utilise Hermes Agent pour le développement. Les workflows réutilisables sont sauvegardés comme **skills** dans `~/.hermes/skills/`. Avant de commencer une tâche récurrente, vérifier les skills existants :

```bash
hermes skills list
hermes skills view trading
```