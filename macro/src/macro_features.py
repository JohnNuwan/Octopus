"""Moteur de features macro-économiques pour Octopus.

Transforme les données brutes (calendrier, sentiment, taux CB)
en features normalisées qui enrichissent l'observation du JEPA encoder.

Produit 15 features macro par pas de temps :
- 5 features calendrier (volatilité, surprises, régime)
- 5 features sentiment (bullish/bearish, gold sentiment)
- 5 features central banks (taux Fed, spread, cycle)

Format de sortie : np.ndarray (15,) compatible avec observation_shape=(35, 96)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from .calendar import EconomicCalendar, EconomicEvent, EventImportance
from .sentiment import NewsSentimentAnalyzer
from .central_banks import CentralBankMonitor

logger = logging.getLogger(__name__)

# Nombre de features macro produites
NUM_MACRO_FEATURES = 15


@dataclass
class MacroFeatures:
    """Features macro-économiques pour un pas de temps.

    Attributes:
        timestamp: Horodatage.
        calendar_features: (5,) — volatilité, incertitude, surprises, fed_count, high_impact_count.
        sentiment_features: (5,) — overall, gold_sentiment, bullish_ratio, gold_relevance, signal_strength.
        central_bank_features: (5,) — fed_rate, ecb_rate, spread, dxy_strength, gold_env_score.
        raw: Vecteur complet (15,).
    """
    timestamp: datetime = field(default_factory=datetime.now)
    calendar_features: np.ndarray = field(default_factory=lambda: np.zeros(5))
    sentiment_features: np.ndarray = field(default_factory=lambda: np.zeros(5))
    central_bank_features: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @property
    def raw(self) -> np.ndarray:
        """Vecteur complet des features macro (15,)."""
        return np.concatenate([
            self.calendar_features,
            self.sentiment_features,
            self.central_bank_features,
        ]).astype(np.float32)


class MacroFeatureEngine:
    """Moteur de features macro-économiques.

    Agrège les données du calendrier, du sentiment et des banques
    centrales pour produire des features normalisées.

    Attributes:
        calendar: Calendrier économique.
        sentiment: Analyseur de sentiment.
        central_banks: Moniteur de banques centrales.
        history: Buffer d'historique des features.
    """

    def __init__(
        self,
        max_history: int = 500,
        use_finbert: bool = True,
    ) -> None:
        """Initialise le moteur de features macro.

        Args:
            max_history: Taille maximale du buffer d'historique.
            use_finbert: Utiliser FinBERT pour le sentiment.
        """
        self.calendar = EconomicCalendar()
        self.sentiment = NewsSentimentAnalyzer(use_finbert=use_finbert)
        self.central_banks = CentralBankMonitor()
        self.max_history = max_history
        self.history: List[MacroFeatures] = []

    def _extract_calendar_features(self) -> np.ndarray:
        """Extrait les features du calendrier économique (5).

        Returns:
            Vecteur (5,) : [volatilité, incertitude, surprise_moy, fed_events, high_impact].
        """
        try:
            regime = self.calendar.get_macro_regime()
            return np.array([
                regime.get("macro_volatility", 0.0),
                regime.get("uncertainty", 0.0),
                min(regime.get("avg_surprise", 0.0) / 2.0, 1.0),  # Normalisé
                min(regime.get("fed_events_count", 0) / 5.0, 1.0),  # Normalisé
                min(regime.get("high_impact_count", 0) / 10.0, 1.0),  # Normalisé
            ], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Erreur features calendrier: {e}")
            return np.zeros(5, dtype=np.float32)

    def _extract_sentiment_features(self) -> np.ndarray:
        """Extrait les features de sentiment (5).

        Returns:
            Vecteur (5,) : [overall, gold_sentiment, bullish_ratio, gold_relevance, signal].
        """
        try:
            result = self.sentiment.analyze()
            total = result.bullish_count + result.bearish_count + result.neutral_count
            bullish_ratio = (
                result.bullish_count / total if total > 0 else 0.5
            )
            gold_relevance = 0.0
            if result.top_articles:
                gold_relevance = np.mean([
                    a.gold_relevance for a in result.top_articles[:5]
                ])

            return np.array([
                max(-1.0, min(1.0, result.overall_sentiment)),  # -1 à +1
                max(-1.0, min(1.0, result.gold_sentiment)),      # -1 à +1
                bullish_ratio,                                   # 0 à 1
                gold_relevance,                                  # 0 à 1
                abs(result.gold_sentiment),                       # Force du signal 0-1
            ], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Erreur features sentiment: {e}")
            return np.array([0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)

    def _extract_central_bank_features(self) -> np.ndarray:
        """Extrait les features des banques centrales (5).

        Returns:
            Vecteur (5,) : [fed_rate, ecb_rate, spread, dxy_strength, gold_env].
        """
        try:
            fed = self.central_banks.rates.get("FED")
            ecb = self.central_banks.rates.get("ECB")
            spread = self.central_banks.get_real_rate_spread()
            dxy = self.central_banks.get_dollar_strength_index()
            gold_env, _ = self.central_banks.get_gold_environment_score()

            return np.array([
                fed.current_rate / 10.0 if fed else 0.0,     # Taux Fed normalisé (0-0.5)
                ecb.current_rate / 10.0 if ecb else 0.0,     # Taux BCE normalisé
                (spread + 5.0) / 10.0,                        # Spread normalisé (0-1)
                dxy,                                          # Force dollar 0-1
                gold_env,                                     # Environnement or 0-1
            ], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Erreur features CB: {e}")
            return np.array([0.0, 0.0, 0.5, 0.5, 0.5], dtype=np.float32)

    def get_features(self) -> MacroFeatures:
        """Génère les features macro actuelles.

        Returns:
            MacroFeatures contenant les features et l'horodatage.
        """
        features = MacroFeatures(
            timestamp=datetime.now(),
            calendar_features=self._extract_calendar_features(),
            sentiment_features=self._extract_sentiment_features(),
            central_bank_features=self._extract_central_bank_features(),
        )

        # Mettre à jour l'historique
        self.history.append(features)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return features

    def get_features_vector(self) -> np.ndarray:
        """Retourne le vecteur de features macro sous forme numpy.

        Returns:
            Vecteur (15,) de type float32.
        """
        return self.get_features().raw

    def get_features_history(self, n: int = 96) -> np.ndarray:
        """Retourne l'historique des features macro.

        Args:
            n: Nombre de pas d'historique.

        Returns:
            Matrice (n, 15) ou (max_length, 15) si historique insuffisant.
        """
        if len(self.history) == 0:
            return np.zeros((n, NUM_MACRO_FEATURES), dtype=np.float32)

        available = min(n, len(self.history))
        recent = self.history[-available:]
        features = np.array([f.raw for f in recent])

        # Pad avec des zéros si pas assez d'historique
        if available < n:
            padding = np.zeros((n - available, NUM_MACRO_FEATURES), dtype=np.float32)
            features = np.vstack([padding, features])

        return features

    def to_dict(self) -> Dict:
        """Sérialise l'état actuel en dictionnaire.

        Returns:
            Dict avec les features actuelles.
        """
        features = self.get_features()
        return {
            "timestamp": features.timestamp.isoformat(),
            "calendar": {
                "volatility": float(features.calendar_features[0]),
                "uncertainty": float(features.calendar_features[1]),
                "avg_surprise": float(features.calendar_features[2]),
                "fed_events": float(features.calendar_features[3]),
                "high_impact": float(features.calendar_features[4]),
            },
            "sentiment": {
                "overall": float(features.sentiment_features[0]),
                "gold_sentiment": float(features.sentiment_features[1]),
                "bullish_ratio": float(features.sentiment_features[2]),
                "gold_relevance": float(features.sentiment_features[3]),
                "signal_strength": float(features.sentiment_features[4]),
            },
            "central_banks": {
                "fed_rate": float(features.central_bank_features[0] * 10.0),
                "ecb_rate": float(features.central_bank_features[1] * 10.0),
                "spread": float(features.central_bank_features[2] * 10.0 - 5.0),
                "dxy_strength": float(features.central_bank_features[3]),
                "gold_environment": float(features.central_bank_features[4]),
            },
        }