# Module Macro-économique pour Octopus

Analyse les données macro-économiques pour enrichir les features du moteur de trading XAUUSD.

## Composants

| Module | Fichier | Rôle |
|--------|---------|------|
| **Economic Calendar** | `src/calendar.py` | Calendrier économique Investing.com (investpy) |
| **News Sentiment** | `src/sentiment.py` | Analyse NLP FinBERT + règles lexicales |
| **Central Banks** | `src/central_banks.py` | Taux Fed, BCE, BOE, BOJ, SNB... |
| **Feature Engine** | `src/macro_features.py` | 15 features normalisées → engine |
| **gRPC Server** | `src/macro_server.py` | Service pour l'orchestrateur |

## Features produites (15 par pas de temps)

### Calendrier (5)
- Volatilité macro (0-1)
- Incertitude (surprises récentes)
- Surprise moyenne normalisée
- Événements Fed (count/5)
- Événements high impact (count/10)

### Sentiment (5)
- Sentiment global (-1 à +1)
- Sentiment or spécifique (-1 à +1)
- Ratio bullish / total
- Pertinence or moyenne
- Force du signal (|gold_sentiment|)

### Banques Centrales (5)
- Taux Fed (normalisé /10)
- Taux BCE (normalisé /10)
- Spread Fed-BCE
- Indice force dollar (0-1)
- Score environnement or (0-1)

## Tests

```bash
cd macro && python -m pytest tests/ -v
```

27 tests — couverture : calendrier, sentiment, CB, features, plages numériques.