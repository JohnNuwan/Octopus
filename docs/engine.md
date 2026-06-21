# Engine Octopus — Moteur Python

> Apprentissage par renforcement profond pour le trading XAUUSD.

## Architecture neurale

### TS-JEPA (Joint Embedding Predictive Architecture)

L'encodeur JEPA transforme les observations brutes du marché (OHLCV + indicateurs) en un **embedding latent débruité** via l'objectif VICReg.

```
x (B, seq_len, features)
│
├─► Input Projection (Linear 20→256)
├─► Positional Encoding
├─► Transformer Encoder ×4 (causal attention)
├─► Attention Pooling (une requête apprise)
├─► Output Projection (256→64)
│
└─► z (B, embedding_dim)
```

**Objectif VICReg** (3 termes) :
| Terme | Rôle | Formule |
|-------|------|---------|
| **Invariance** | Rapproche prédiction et cible | MSE(z_pred, z_target) |
| **Variance** | Empêche l'effondrement | ReLU(1 - σ(z)) |
| **Covariance** | Décorrèle les dimensions | Σ(cov_ij²) pour i≠j |

**Target encoder** : mis à jour par moyenne exponentielle (momentum=0.99) pour stabilité.

**Références :**
- Bardes, Ponce, LeCun. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.* ICLR 2022.
- TS-JEPA. *Joint Embeddings Go Temporal.* arXiv 2509.25449, 2025.

### RSSM World Model

Le World Model apprend à **prédire les transitions** du marché dans l'espace latent sans interagir avec le marché réel.

```
État latent = Stochastique (32×32 catégories) + Déterministe (256 GRU)
│
├─► Transition : (sₜ, aₜ, obsₜ) → (sₜ₊₁ prior, sₜ₊₁ posterior)
│   ├─► Prior : prédit l'état suivant sans voir l'observation
│   └─► Posterior : prédit l'état suivant avec l'observation (apprentissage)
│
├─► Reward Head : prédit la récompense depuis l'état latent
├─► Continue Head : prédit la fin d'épisode (sigmoid)
└─► Embed Predictor : reconstruit l'observation
```

**RSSM Transition** :
1. Concatène l'état stochastique + action + embedding
2. Projection linéaire → GRU → nouvel état déterministe
3. Prédiction prior (sans observation) : Linear(deter) → OneHotCategorical
4. Prédiction posterior (avec observation) : Linear(deter + embedding) → OneHotCategorical
5. Straight-through gradient estimator

**Références :**
- Hafner et al. *Mastering Diverse Domains through World Models.* Nature, 2025.
- Hafner et al. *Dream to Control: Learning Behaviors by Latent Imagination.* ICLR 2020.

### Actor-Critic

L'actor-critic est **entraîné exclusivement sur des trajectoires imaginaires** générées par le World Model (DreamerV3-style).

**Actor Network** : État latent → distribution sur 5 actions (Hold/Buy/Sell/Split/Close)
- Couches : Linear(high_dim→256) → LayerNorm → GELU → Linear(256→256) → GELU → Linear(256→5)
- Initialisation orthogonale (gain=0.01)

**Critic Network** : État latent → valeur estimée (symlog)
- Architecture identique à l'actor mais sortie 1 dimension

**Entraînement** :
1. World Model génère des trajectoires imaginaires (15 pas)
2. Actor propose des actions → évaluées par le critic
3. PPO-style : λ-returns, avantage, entropie
4. Perte totale = actor_loss - entropy + critic_loss

### MCTS (Monte Carlo Tree Search)

Planification dans l'espace latent à 150 simulations par décision.

```
1. SELECT : descendre dans l'arbre via UCB
   score = value + pb_c × prior
   pb_c = ln((N + base + 1) / base) + c_init

2. EXPAND : ajouter les enfants via recurrent_inference

3. BACKUP : remonter les valeurs (reward + discount × value)
```

## Trading Environment

### Actions (discrètes, 5)
| Action | Code | Description |
|--------|------|-------------|
| Hold | 0 | Ne rien faire |
| Buy | 1 | Ouvrir une position longue |
| Sell | 2 | Ouvrir une position courte |
| Split | 3 | Fermer 50% de la position (si profitable) |
| Close | 4 | Fermer toute la position |

### Position Sizing
```python
risk_amount = balance × 0.8%  # Risque de 0.8% par trade
stop_distance = max(ATR × 1.3, 5.0)
lots = risk_amount / (stop_distance × 100)
```

### FTMOEnforcer
- Perte quotidienne max : 5% du capital de phase
- Perte totale max : 10% du capital de phase
- Objectif de profit : 10%
- Deux phases obligatoires

### SLBESystem (Stop Loss Break Even)
- S'active à +0.5% de profit non réalisé
- Déplace le stop loss au prix d'entrée
- Bonus de récompense : +6.0 à l'activation

### Reward Shaping
| Événement | Récompense |
|-----------|------------|
| SLBE activé | +6.0 |
| SLBE touché | +1.0 |
| Split de position | +10.0 |
| Big winner (>2%) | +15.0 |
| Quality trade (>0%) | +10.0 |
| Perte non réalisée | -pnl% × 100 |
| Max drawdown (>5%) | -10.0 |
| Inactivité (>100 steps) | -1.0/step |
| Croissance finale (>10%) | +50.0 |

## Training Loop

### MuZero-style

1. **Self-play** (tous les 5 steps) :
   - JEPA encode l'observation → embedding
   - World Model produit l'état latent
   - MCTS planifie 150 simulations
   - Action selon distribution des visites
   - Stocke l'expérience dans le Replay Buffer

2. **Entraînement** (si buffer > batch_size) :
   - Échantillonne un batch du Replay Buffer
   - Encode les observations avec JEPA
   - Déroule le World Model sur n_steps
   - Calcule la perte actor-critic
   - Rétropropagation et mise à jour AdamW

3. **Hybrid Learning** (tous les 10 steps) :
   - Intègre les trades live dans le Replay Buffer
   - Apprentissage continu hors ligne

4. **Checkpoint** :
   - Sauvegarde tous les checkpoint_interval steps
   - Keep dernier 5 checkpoints

### Configuration
| Paramètre | Valeur par défaut | Description |
|-----------|-------------------|-------------|
| training_steps | 30,000 | Steps total d'entraînement |
| batch_size | 128 | Taille du mini-batch |
| learning_rate | 1e-4 | Learning rate AdamW |
| replay_buffer_size | 100,000 | Capacité max du buffer |
| num_simulations | 150 | Simulations MCTS par décision |
| discount | 0.997 | Facteur d'actualisation |