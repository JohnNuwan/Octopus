# Octopus Macro Pipeline

Analyse les conditions macro-économiques et génère des alertes.

## Étapes

1. **Exécuter le moteur macro**
   ```
   cd /home/kadeva/Octopus/macro && source /home/kadeva/trading-env/bin/activate && python3 -c "
   from src.macro_features import MacroFeatureEngine
   e = MacroFeatureEngine(use_finbert=False)
   print(e.to_dict())
   "
   ```

2. **Analyser les indicateurs clés**
   - **Volatilité** (VIX / or) : détecter les régimes de marché
   - **Sentiment** (or / crypto) : mesurer l'appétit au risque
   - **Taux CB** (Fed, BCE) : politique monétaire
   - **Or environnement** : corrélations avec les actifs

3. **Alerter si conditions extrêmes**
   - Volatilité > 0.8 → Marché en stress, réduire l'exposition
   - Sentiment or < -0.5 → Peur extrême, opportunité d'achat
   - Taux CB en baisse → Assouplissement monétaire, hausse or
   - Corrélation or/SPX hors norme → Régime de marché inhabituel

4. **Envoyer un rapport formaté**
   - Synthèse des indicateurs en une phrase
   - Tableau des métriques clés
   - Alertes actives
   - Recommandations pour le trading

## Sortie attendue

```
📊 Macro Analysis — 2026-06-21 08:00 UTC

Conditions: ⚠️ Volatile (VIX: 22.5)
Sentiment: 🟢 Neutre (Gold: +0.12)
Fed Rate: 4.50% → 4.25% (tendance baisse)

Alertes:
⚠️ Volatilité élevée (0.82 > seuil 0.80)
→ Réduire exposition recommandée

Recommandation:
Marché en stress modéré — privilégier les positions défensives.
```