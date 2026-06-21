# Tests Octopus

> **77 tests unitaires** — lancés via pytest.

## Exécuter les tests

```bash
cd /home/kadeva/Octopus/engine
source /home/kadeva/trading-env/bin/activate

# Tous les tests
python -m pytest tests/ -v

# Avec coverage
python -m pytest --cov=src tests/ --cov-report=term

# Un module spécifique
python -m pytest tests/test_jepa.py -v

# Un test spécifique
python -m pytest tests/test_actor_critic.py::TestActorNetwork::test_forward_shapes -v

# En mode verbose avec sortie
python -m pytest tests/ -v -s
```

## Modules testés

### test_environment.py (15 tests)

| Test | Description |
|------|-------------|
| `OctopusTradingEnv` | Environnement Gymnasium |
| `test_initialization` | Vérifie state_dim, action_dim, FTMO |
| `test_reset` | Observation de bonne dimension |
| `test_step_hold` | Action 0 ne change pas la position |
| `test_step_buy` | Action 1 ouvre une position longue |
| `test_step_sell` | Action 2 ouvre une position courte |
| `test_step_close` | Action 4 ferme la position |
| `test_step_split` | Action 3 réduit de 50% |
| `FTMOEnforcer` | Règles FTMO |
| `test_daily_loss_limit` | -5% déclenche échec |
| `test_total_loss_limit` | -10% déclenche échec |
| `test_profit_target` | +10% valide phase 1 |
| `test_two_phases` | Phase 1 → Phase 2 |
| `SLBESystem` | Stop Loss Break Even |
| `test_initial_bonus` | Activation à +0.5% |

### test_jepa.py (12 tests)

| Test | Description |
|------|-------------|
| `JEPAEncoder` | Encodeur temporel |
| `test_forward_shape` | (B, S, F) → (B, embed_dim) |
| `test_deterministic` | Même entrée = même sortie en eval |
| `TSJEPA` | Architecture complète |
| `test_train_step` | Loss descendante |
| `test_target_encoder_update` | Poids bougent |
| `VICRegLoss` | Perte auto-supervisée |
| `test_perfect_match` | Perte faible quand identique |
| `PositionalEncoding` | Encodage sinusoïdal |

### test_world_model.py (10 tests)

| Test | Description |
|------|-------------|
| `RSSMTransition` | Module de transition |
| `test_forward_shapes` | Dimensions correctes |
| `test_posterior_vs_prior` | Avec/sans embedding |
| `RSSMWorldModel` | World Model complet |
| `test_imagine_shapes` | Trajectoire imaginaire |
| `test_predict_continue` | Sortie entre 0 et 1 |
| `SymlogSymexp` | Transformations |

### test_actor_critic.py (9 tests)

| Test | Description |
|------|-------------|
| `ActorNetwork` | Politique stochastique |
| `test_action_probs_sum_to_one` | Distribution valide |
| `CriticNetwork` | Estimateur de valeur |
| `ActorCritic` | Module complet |
| `test_compute_loss` | Perte non-NaN |
| `test_lambda_returns` | Forme et monotonie |

### test_mcts.py (5 tests)

| Test | Description |
|------|-------------|
| `MCTSNode` | Nœud de l'arbre |
| `test_value` | total_value / visit_count |
| `test_ucb_score` | Score d'exploration |
| `MCTS` | Recherche arborescente |
| `test_search_with_mock_network` | Distribution valide |

### test_replay_buffer.py (7 tests)

| Test | Description |
|------|-------------|
| `GameHistory` | Historique d'épisode |
| `test_save_load` | Pickle round-trip |
| `ReplayBuffer` | Tampon de rejeu |
| `test_capacity` | FIFO overflow |

### test_trader.py (5 tests)

| Test | Description |
|------|-------------|
| `LiveTrader` | Trader temps réel |
| `test_step_synthetic` | Action valide |
| `test_cooldown` | Pas de trade en cooldown |
| `test_kick_mechanism` | Force action non-Hold |

## Ajouter un nouveau test

1. Créer le fichier `tests/test_mon_module.py`
2. Importer les classes à tester
3. Utiliser les fixtures de `conftest.py`
4. Nommer les tests `test_*` et les classes `Test*`
5. Vérifier que le test passe : `python -m pytest tests/test_mon_module.py -v`

```python
"""Tests pour le module mon_module.

Références:
    Google Style PEP257.
"""

import pytest
import torch
from src.mon_module import MaClasse


class TestMaClasse:
    """Tests pour MaClasse."""
    
    def test_initialization(self):
        """Teste l'initialisation de MaClasse."""
        obj = MaClasse()
        assert obj.param == valeur_attendue
    
    def test_forward_shape(self):
        """Vérifie la forme de la sortie."""
        obj = MaClasse()
        x = torch.randn(4, 20)
        out = obj(x)
        assert out.shape == (4, 64)
```