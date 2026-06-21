"""Calendrier économique — événements macro impactant XAUUSD.

Récupère le calendrier économique depuis Investing.com via investpy.
Fournit une interface unifiée pour filtrer les événements par :
- Importance (high/medium/low)
- Pays (US, EU, UK, JP, CH, AU, NZ)
- Catégorie (Inflation, Employment, Central Banks, GDP)

Références:
    investpy — https://investpy.readthedocs.io
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EventImportance(Enum):
    """Importance d'un événement économique."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class EconomicEvent:
    """Événement du calendrier économique.

    Attributes:
        id: Identifiant unique.
        date: Date de l'événement.
        time: Heure de l'événement.
        zone: Pays / zone économique.
        currency: Devise concernée.
        importance: Importance (high/medium/low).
        event: Titre de l'événement.
        actual: Valeur réalisée.
        forecast: Valeur prévue.
        previous: Valeur précédente.
        surprise: Écart actual - forecast (normalisé).
        volatility_score: Score de volatilité estimé (0-1).
    """
    id: str
    date: str
    time: str
    zone: str
    currency: str
    importance: EventImportance
    event: str
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    surprise: float = 0.0
    volatility_score: float = 0.0


class EconomicCalendar:
    """Récupère et analyse le calendrier économique.

    Utilise investpy pour récupérer les données Investing.com.
    Maintient un cache local pour éviter les requêtes répétées.

    Attributes:
        cache: Cache des événements {date: [EconomicEvent]}.
        cache_ttl: Durée de validité du cache en heures.
    """

    # Événements à fort impact XAUUSD
    GOLD_IMPACT_EVENTS = {
        "Fed Interest Rate Decision",
        "Non Farm Payrolls",
        "CPI", "Consumer Price Index",
        "GDP", "Gross Domestic Product",
        "Initial Jobless Claims",
        "FOMC Statement",
        "ECB Interest Rate Decision",
        "Unemployment Rate",
        "Retail Sales",
        "ISM Manufacturing PMI",
        "ISM Services PMI",
        "Industrial Production",
        "Core CPI",
        "Core PCE Price Index",
        "Michigan Consumer Sentiment",
        "CB Consumer Confidence",
        "Durable Goods Orders",
        "Building Permits",
        "Housing Starts",
        "Philadelphia Fed Manufacturing Index",
        "NY Empire State Manufacturing Index",
        "Producer Price Index",
    }

    # Événements moyenne importance XAUUSD
    GOLD_MEDIUM_EVENTS = {
        "Trade Balance",
        "Current Account",
        "Factory Orders",
        "Wholesale Inventories",
        "Business Inventories",
        "Treasury Budget",
        "Existing Home Sales",
        "New Home Sales",
        "Pending Home Sales",
        "FHFA House Price Index",
        "S&P/CS Composite-20 HPI",
        "Consumer Credit",
        "Personal Income",
        "Personal Spending",
        "Chicago PMI",
        "Dallas Fed Manufacturing Index",
        "Kansas Fed Manufacturing Index",
        "Richmond Fed Manufacturing Index",
    }

    def __init__(
        self,
        cache_ttl_hours: float = 1.0,
        lookback_days: int = 7,
        lookahead_days: int = 14,
    ) -> None:
        """Initialise le calendrier économique.

        Args:
            cache_ttl_hours: Durée de validité du cache.
            lookback_days: Jours de lookback pour les événements passés.
            lookahead_days: Jours de lookahead pour les événements futurs.
        """
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.lookback_days = lookback_days
        self.lookahead_days = lookahead_days
        self._cache: Dict[str, List[EconomicEvent]] = {}
        self._cache_time: Optional[datetime] = None
        self._investpy_available = False

        # Vérifier si investpy est disponible
        try:
            import investpy  # noqa: F401
            self._investpy_available = True
        except ImportError:
            logger.warning(
                "investpy non disponible. Utilisation du mode dégradé."
            )

    def _fetch_investpy(
        self,
        from_date: str,
        to_date: str,
        countries: Optional[List[str]] = None,
        importances: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Récupère le calendrier via investpy.

        Args:
            from_date: Date début (dd/mm/yyyy).
            to_date: Date fin (dd/mm/yyyy).
            countries: Liste des pays.
            importances: Liste des importances.

        Returns:
            DataFrame des événements.
        """
        try:
            import investpy
            df = investpy.economic_calendar(
                time_zone="GMT +0:00",
                time_filter="time_only",
                countries=countries or [
                    "united states", "euro zone", "united kingdom",
                    "japan", "switzerland", "australia", "new zealand",
                    "china", "canada",
                ],
                importances=importances or ["high", "medium"],
                from_date=from_date,
                to_date=to_date,
            )
            return df
        except Exception as e:
            logger.error(f"Erreur investpy: {e}")
            return pd.DataFrame()

    def _compute_surprise(
        self,
        actual: Optional[float],
        forecast: Optional[float],
    ) -> float:
        """Calcule le score de surprise normalisé.

        Args:
            actual: Valeur réelle.
            forecast: Valeur prévue.

        Returns:
            Score de surprise (écart-type normalisé, ou 0 si non disponible).
        """
        if actual is None or forecast is None:
            return 0.0
        diff = abs(actual - forecast)
        base = max(abs(forecast), 1e-8)
        return min(diff / base, 5.0)  # Cap à 5 écarts-types

    def _compute_volatility_score(
        self,
        importance: str,
        is_gold_event: bool,
        is_fed_event: bool,
    ) -> float:
        """Estime le score de volatilité potentiel.

        Args:
            importance: 'high', 'medium', 'low'.
            is_gold_event: Lié à l'or.
            is_fed_event: Lié à la Fed.

        Returns:
            Score 0-1.
        """
        base = {"high": 0.6, "medium": 0.3, "low": 0.1}.get(importance, 0.1)
        if is_fed_event:
            base += 0.3  # La Fed impacte fortement XAUUSD
        if is_gold_event:
            base += 0.1
        return min(base, 1.0)

    def is_fed_event(self, event: str) -> bool:
        """Détecte les événements liés à la Fed.

        Args:
            event: Titre de l'événement.

        Returns:
            True si l'événement est lié à la Fed.
        """
        fed_keywords = [
            "fed", "fomc", "federal reserve", "powell",
            "interest rate", "beige book",
        ]
        return any(kw in event.lower() for kw in fed_keywords)

    def is_gold_event(self, event: str) -> bool:
        """Détecte les événements impactant l'or.

        Args:
            event: Titre de l'événement.

        Returns:
            True si l'événement impacte l'or.
        """
        for kw in self.GOLD_IMPACT_EVENTS:
            if kw.lower() in event.lower():
                return True
        return False

    def fetch_events(
        self,
        force_refresh: bool = False,
    ) -> List[EconomicEvent]:
        """Récupère les événements économiques.

        Utilise le cache si disponible et valide.

        Args:
            force_refresh: Force le rafraîchissement.

        Returns:
            Liste des événements économiques.
        """
        now = datetime.now()

        # Vérifier le cache
        if (
            not force_refresh
            and self._cache
            and self._cache_time
            and (now - self._cache_time) < self.cache_ttl
        ):
            all_events = []
            for events in self._cache.values():
                all_events.extend(events)
            return all_events

        # Dates
        from_date = (now - timedelta(days=self.lookback_days)).strftime("%d/%m/%Y")
        to_date = (now + timedelta(days=self.lookahead_days)).strftime("%d/%m/%Y")

        events: List[EconomicEvent] = []

        # Investpy
        if self._investpy_available:
            try:
                df = self._fetch_investpy(from_date, to_date)
                if not df.empty:
                    for _, row in df.iterrows():
                        imp = str(row.get("importance", "")).lower()
                        event_name = str(row.get("event", ""))
                        events.append(EconomicEvent(
                            id=str(row.get("id", "")),
                            date=str(row.get("date", "")),
                            time=str(row.get("time", "")),
                            zone=str(row.get("zone", "")),
                            currency=str(row.get("currency", "")),
                            importance=EventImportance(imp) if imp in ["high", "medium", "low"] else EventImportance.MEDIUM,
                            event=event_name,
                            actual=self._parse_float(row.get("actual")),
                            forecast=self._parse_float(row.get("forecast")),
                            previous=self._parse_float(row.get("previous")),
                        ))
            except Exception as e:
                logger.error(f"Erreur fetch investpy: {e}")

        # Calculer les métriques dérivées
        for ev in events:
            ev.surprise = self._compute_surprise(ev.actual, ev.forecast)
            ev.volatility_score = self._compute_volatility_score(
                ev.importance.value,
                self.is_gold_event(ev.event),
                self.is_fed_event(ev.event),
            )

        # Mettre en cache
        if events:
            for ev in events:
                day = ev.date
                if day not in self._cache:
                    self._cache[day] = []
                self._cache[day].append(ev)
            self._cache_time = now

        logger.info(f"📅 {len(events)} événements économiques récupérés")
        return events

    def get_upcoming_events(
        self,
        hours_ahead: int = 24,
        min_importance: str = "medium",
    ) -> List[EconomicEvent]:
        """Récupère les événements à venir dans les N heures.

        Args:
            hours_ahead: Nombre d'heures à regarder.
            min_importance: Importance minimale.

        Returns:
            Liste des événements à venir.
        """
        events = self.fetch_events()
        now = datetime.now()
        cutoff = now + timedelta(hours=hours_ahead)

        upcoming = []
        for ev in events:
            try:
                ev_time = datetime.strptime(f"{ev.date} {ev.time}", "%d/%m/%Y %H:%M")
                if now <= ev_time <= cutoff:
                    imp_score = {"high": 3, "medium": 2, "low": 1}
                    if imp_score.get(ev.importance.value, 0) >= imp_score.get(min_importance, 2):
                        upcoming.append(ev)
            except (ValueError, TypeError):
                continue

        return sorted(upcoming, key=lambda e: f"{e.date} {e.time}")

    def get_recent_surprises(
        self,
        hours_back: int = 48,
        min_surprise: float = 0.5,
    ) -> List[EconomicEvent]:
        """Récupère les événements récents avec fort surprise.

        Args:
            hours_back: Nombre d'heures en arrière.
            min_surprise: Score de surprise minimum.

        Returns:
            Liste des événements surprenants.
        """
        events = self.fetch_events()
        now = datetime.now()
        cutoff = now - timedelta(hours=hours_back)

        surprises = []
        for ev in events:
            try:
                ev_time = datetime.strptime(f"{ev.date} {ev.time}", "%d/%m/%Y %H:%M")
                if cutoff <= ev_time <= now and ev.surprise >= min_surprise:
                    surprises.append(ev)
            except (ValueError, TypeError):
                continue

        return sorted(surprises, key=lambda e: e.surprise, reverse=True)

    @staticmethod
    def _parse_float(value: any) -> Optional[float]:
        """Convertit une valeur en float.

        Args:
            value: Valeur à convertir.

        Returns:
            Valeur float ou None.
        """
        if value is None:
            return None
        try:
            # Nettoyer : enlever %, K, M, B
            cleaned = str(value).replace("%", "").replace(",", "")
            cleaned = cleaned.replace("K", "e3").replace("M", "e6").replace("B", "e9")
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def get_macro_regime(self) -> Dict[str, float]:
        """Analyse le régime macro actuel.

        Returns:
            Dictionnaire des scores de régime.
        """
        events = self.fetch_events()

        # Compter les événements par type
        fed_events = [e for e in events if self.is_fed_event(e.event)]
        high_impact = [e for e in events if e.importance == EventImportance.HIGH]
        surprises = [e for e in events if e.surprise > 1.0]

        # Score de volatilité macro (0-1)
        macro_volatility = 0.0
        if high_impact:
            recent_high = [
                e for e in high_impact
                if e.surprise > 0
            ]
            macro_volatility = len(recent_high) / max(len(high_impact), 1)

        # Score d'incertitude (surprises)
        uncertainty = min(len(surprises) / max(len(events), 1) * 5, 1.0)

        return {
            "macro_volatility": macro_volatility,
            "uncertainty": uncertainty,
            "fed_events_count": len(fed_events),
            "high_impact_count": len(high_impact),
            "total_events": len(events),
            "avg_surprise": np.mean([e.surprise for e in events]) if events else 0.0,
        }