# Octopus Backtest Pipeline

Lance un backtest complet de la stratégie Octopus sur XAUUSD.

## Étapes

1. **Récupérer les dernières données XAUUSD**
   ```
   cd /home/kadeva/Octopus/engine
   source /home/kadeva/trading-env/bin/activate
   python3 -c "from src.data import get_forex_data; df = get_forex_data('XAUUSD', period='6mo'); print(f'{len(df)} candles chargées')"
   ```

2. **Configurer le backtest**
   - Période : 6 derniers mois
   - Timeframe : H1 (1 heure)
   - Capital initial : 10 000 USD
   - Spread : 0.20 (compte ECN)

3. **Exécuter le backtest**
   ```
   python3 -m src.backtest --symbol XAUUSD --period 6mo --capital 10000
   ```

4. **Calculer les métriques de performance**
   - **Sharpe Ratio** (> 1.5 : bon)
   - **Sortino Ratio** (> 2.0 : excellent)
   - **Maximum Drawdown** (< 15% : acceptable)
   - **Win Rate** (> 55% : bon)
   - **Profit Factor** (> 1.5 : bon)

5. **Comparer avec les baselines**
   - Buy & Hold sur XAUUSD
   - SMA crossover (50/200)
   - RSI (30/70)

6. **Envoyer le rapport**
   - Résumé des métriques
   - Comparaison avec baselines
   - Equity curve (graphique)
   - Recommandation : déployer ou itérer

## Sortie attendue

```
📈 Backtest Report — XAUUSD H1 — 6mo

Metrics:
Sharpe:      1.82 ✅
Sortino:     2.34 ✅
Max DD:      -12.3% ✅
Win Rate:    61.2% ✅
Profit Fact: 1.89 ✅

vs Baseline:
Octopus:    +18.4%
Buy & Hold: +6.7%
SMA Cross:  -2.1%

Verdict: ✅ Prêt pour le déploiement
```