"""Tests du module Macro-économique Octopus.

Teste le calendrier économique, l'analyse de sentiment,
les banques centrales, et le feature engineering.

Références:
    Google Style PEP257.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from src.calendar import EconomicCalendar, EconomicEvent, EventImportance
from src.sentiment import NewsSentimentAnalyzer, RuleBasedSentiment, NewsArticle
from src.central_banks import CentralBankMonitor, CentralBankRate, RateDirection
from src.macro_features import MacroFeatureEngine, MacroFeatures, NUM_MACRO_FEATURES


class TestRuleBasedSentiment:
    """Tests de l'analyseur de sentiment lexical."""

    def test_bullish_text(self):
        """Teste un texte bullish."""
        score = RuleBasedSentiment.score(
            "Gold surges to new highs as rally continues"
        )
        assert score > 0, f"Score devrait être positif, obtenu: {score}"

    def test_bearish_text(self):
        """Teste un texte bearish."""
        score = RuleBasedSentiment.score(
            "Gold crashes amid panic selloff"
        )
        assert score < 0, f"Score devrait être négatif, obtenu: {score}"

    def test_neutral_text(self):
        """Teste un texte neutre."""
        score = RuleBasedSentiment.score(
            "The meeting will be held on Tuesday"
        )
        assert score == 0.0, f"Score devrait être 0, obtenu: {score}"

    def test_mixed_sentiment(self):
        """Teste un texte mixte."""
        score = RuleBasedSentiment.score(
            "Strong growth but recession fears remain"
        )
        # Devrait être proche de 0
        assert -0.5 < score < 0.5


class TestNewsSentimentAnalyzer:
    """Tests de l'analyseur de sentiment complet."""

    def test_initialization(self):
        """Teste l'initialisation."""
        analyzer = NewsSentimentAnalyzer(use_finbert=False)
        assert analyzer.max_articles == 50
        assert len(analyzer.rss_sources) == 4

    def test_compute_gold_relevance(self):
        """Teste le calcul de pertinence pour l'or."""
        analyzer = NewsSentimentAnalyzer(use_finbert=False)
        score = analyzer._compute_gold_relevance(
            "FOMC decision impacts gold prices and dollar"
        )
        assert score > 0, f"Devrait être > 0, obtenu: {score}"
        assert score <= 1.0, f"Devrait être ≤ 1, obtenu: {score}"

    def test_compute_gold_relevance_no_keywords(self):
        """Teste la pertinence sans mots-clés or."""
        analyzer = NewsSentimentAnalyzer(use_finbert=False)
        score = analyzer._compute_gold_relevance(
            "Tech stocks rally on earnings"
        )
        assert score == 0.0


class TestEconomicCalendar:
    """Tests du calendrier économique."""

    def test_initialization(self):
        """Teste l'initialisation du calendrier."""
        cal = EconomicCalendar()
        assert cal.lookback_days == 7
        assert cal.lookahead_days == 14
        assert len(cal.GOLD_IMPACT_EVENTS) > 10

    def test_is_gold_event(self):
        """Teste la détection d'événements liés à l'or."""
        cal = EconomicCalendar()
        assert cal.is_gold_event("Non Farm Payrolls")
        assert cal.is_gold_event("Fed Interest Rate Decision")
        assert not cal.is_gold_event("Random Event")

    def test_is_fed_event(self):
        """Teste la détection d'événements Fed."""
        cal = EconomicCalendar()
        assert cal.is_fed_event("FOMC Statement")
        assert cal.is_fed_event("Fed Interest Rate Decision")
        assert not cal.is_fed_event("ECB Press Conference")

    def test_compute_surprise(self):
        """Teste le calcul de surprise normalisé."""
        cal = EconomicCalendar()
        surprise = cal._compute_surprise(200.0, 100.0)
        assert surprise > 0

        # Pas de surprise si données manquantes
        assert cal._compute_surprise(None, 100.0) == 0.0
        assert cal._compute_surprise(100.0, None) == 0.0

    def test_compute_volatility_score(self):
        """Teste le score de volatilité."""
        cal = EconomicCalendar()
        high = cal._compute_volatility_score("high", True, True)
        low = cal._compute_volatility_score("low", False, False)
        assert high > low

    def test_event_creation(self):
        """Teste la création d'un événement."""
        event = EconomicEvent(
            id="123",
            date="01/01/2025",
            time="14:30",
            zone="united states",
            currency="USD",
            importance=EventImportance.HIGH,
            event="Non Farm Payrolls",
            actual=200.0,
            forecast=180.0,
            previous=150.0,
        )
        assert event.importance == EventImportance.HIGH
        assert event.currency == "USD"


class TestCentralBankMonitor:
    """Tests du moniteur de banques centrales."""

    def test_initialization(self):
        """Teste l'initialisation avec toutes les banques."""
        monitor = CentralBankMonitor()
        assert len(monitor.rates) == 8
        assert "FED" in monitor.rates
        assert "ECB" in monitor.rates

    def test_fed_rate(self):
        """Teste le taux Fed."""
        monitor = CentralBankMonitor()
        fed = monitor.rates["FED"]
        assert fed.current_rate == 3.75
        assert fed.currency == "USD"

    def test_real_rate_spread(self):
        """Teste le calcul du spread Fed - BCE."""
        monitor = CentralBankMonitor()
        spread = monitor.get_real_rate_spread()
        expected = 3.75 - 2.15  # Fed - BCE
        assert spread == pytest.approx(expected)

    def test_dollar_strength_index(self):
        """Teste l'indice de force dollar."""
        monitor = CentralBankMonitor()
        dxy = monitor.get_dollar_strength_index()
        assert 0 <= dxy <= 1

    def test_gold_environment_score(self):
        """Teste le score d'environnement pour l'or."""
        monitor = CentralBankMonitor()
        score, desc = monitor.get_gold_environment_score()
        assert 0 <= score <= 1
        assert isinstance(desc, str)

    def test_hiking_cycle_status(self):
        """Teste l'analyse du cycle de taux."""
        monitor = CentralBankMonitor()
        status = monitor.get_hiking_cycle_status()
        assert "FED" in status
        assert status["FED"] == "easing"  # Fed en baisse
        assert status["BOJ"] == "tightening"  # BOJ en hausse

    def test_get_summary(self):
        """Teste le résumé."""
        monitor = CentralBankMonitor()
        summary = monitor.get_summary()
        assert "FED" in summary
        assert "change_bps" in summary["FED"]
        assert "direction" in summary["FED"]


class TestMacroFeatures:
    """Tests des features macro."""

    def test_macro_features_dataclass(self):
        """Teste le dataclass MacroFeatures."""
        features = MacroFeatures()
        assert features.calendar_features.shape == (5,)
        assert features.sentiment_features.shape == (5,)
        assert features.central_bank_features.shape == (5,)
        assert features.raw.shape == (15,)
        assert features.raw.dtype == np.float32

    def test_macro_feature_engine_initialization(self):
        """Teste l'initialisation du moteur."""
        engine = MacroFeatureEngine(use_finbert=False)
        assert engine.max_history == 500

    def test_get_features_vector(self):
        """Teste la production d'un vecteur de features."""
        engine = MacroFeatureEngine(use_finbert=False)
        vector = engine.get_features_vector()
        assert vector.shape == (15,)
        assert vector.dtype == np.float32
        assert not np.any(np.isnan(vector))

    def test_get_features_history(self):
        """Teste l'historique des features."""
        engine = MacroFeatureEngine(use_finbert=False)

        # Historique vide
        hist = engine.get_features_history(96)
        assert hist.shape == (96, 15)

        # Après avoir généré des features
        for _ in range(5):
            engine.get_features()
        hist = engine.get_features_history(96)
        assert hist.shape == (96, 15)

    def test_to_dict(self):
        """Teste la sérialisation en dictionnaire."""
        engine = MacroFeatureEngine(use_finbert=False)
        d = engine.to_dict()
        assert "timestamp" in d
        assert "calendar" in d
        assert "sentiment" in d
        assert "central_banks" in d
        assert "volatility" in d["calendar"]
        assert "gold_sentiment" in d["sentiment"]
        assert "fed_rate" in d["central_banks"]

    def test_numeric_features_in_range(self):
        """Vérifie que toutes les features sont dans des plages valides."""
        engine = MacroFeatureEngine(use_finbert=False)
        for _ in range(3):
            features = engine.get_features()
            # Calendar features [0-1]
            assert np.all((features.calendar_features >= 0.0) &
                          (features.calendar_features <= 1.0 +
                           1e-6)), f"Calendar features out of range: {features.calendar_features}"
            # Sentiment features: overall et gold_sentiment peuvent être [-1, 1]
            # Les autres [0, 1]
            assert -1.0 - 1e-6 <= features.sentiment_features[0] <= 1.0 + 1e-6
            assert -1.0 - 1e-6 <= features.sentiment_features[1] <= 1.0 + 1e-6
            # CB features [0-1]
            assert np.all((features.central_bank_features >= 0.0) &
                          (features.central_bank_features <= 1.0 +
                           1e-6)), f"CB features out of range: {features.central_bank_features}"


class TestNUM_MACRO_FEATURES:
    """Vérifie la constante NUM_MACRO_FEATURES."""

    def test_constant_value(self):
        """Vérifie que NUM_MACRO_FEATURES = 15."""
        assert NUM_MACRO_FEATURES == 15