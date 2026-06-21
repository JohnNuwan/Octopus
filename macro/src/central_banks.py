"""Surveillance des taux des banques centrales.

Récupère et suit les décisions de taux des principales banques
centrales impactant XAUUSD :
- Fed (USD) — la plus importante pour l'or
- BCE (EUR) — deuxième impact
- BOE (GBP), BOJ (JPY), SNB (CHF), RBA (AUD), RBNZ (NZD)
- PBOC (CNY), RBI (INR)

Références:
    Investing.com central banks — https://www.investing.com/central-banks
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import requests

logger = logging.getLogger(__name__)


class RateDirection(Enum):
    """Direction du changement de taux."""
    HIKE = "hike"
    CUT = "cut"
    HOLD = "hold"
    UNKNOWN = "unknown"


@dataclass
class CentralBankRate:
    """Taux d'une banque centrale.

    Attributes:
        bank_name: Nom de la banque centrale.
        code: Code (FED, ECB, BOE, etc.).
        current_rate: Taux actuel (%).
        previous_rate: Taux précédent (%).
        direction: Direction du dernier changement.
        last_change: Date du dernier changement.
        next_meeting: Date de la prochaine réunion.
        change_pct: Variation en points de base.
        currency: Devise associée.
    """
    bank_name: str
    code: str
    current_rate: float
    previous_rate: float
    direction: RateDirection
    last_change: Optional[datetime] = None
    next_meeting: Optional[datetime] = None
    change_pct: float = 0.0
    currency: str = ""


# Données statiques des banques centrales
CENTRAL_BANKS_DATA = {
    "FED": {
        "name": "Federal Reserve",
        "currency": "USD",
        "current_rate": 3.75,
        "previous_rate": 4.00,
        "direction": RateDirection.CUT,
    },
    "ECB": {
        "name": "European Central Bank",
        "currency": "EUR",
        "current_rate": 2.15,
        "previous_rate": 2.50,
        "direction": RateDirection.CUT,
    },
    "BOE": {
        "name": "Bank of England",
        "currency": "GBP",
        "current_rate": 3.75,
        "previous_rate": 4.00,
        "direction": RateDirection.CUT,
    },
    "BOJ": {
        "name": "Bank of Japan",
        "currency": "JPY",
        "current_rate": 0.75,
        "previous_rate": 0.50,
        "direction": RateDirection.HIKE,
    },
    "SNB": {
        "name": "Swiss National Bank",
        "currency": "CHF",
        "current_rate": 0.00,
        "previous_rate": 0.25,
        "direction": RateDirection.CUT,
    },
    "RBA": {
        "name": "Reserve Bank of Australia",
        "currency": "AUD",
        "current_rate": 4.10,
        "previous_rate": 4.35,
        "direction": RateDirection.CUT,
    },
    "RBNZ": {
        "name": "Reserve Bank of New Zealand",
        "currency": "NZD",
        "current_rate": 2.25,
        "previous_rate": 2.50,
        "direction": RateDirection.CUT,
    },
    "PBOC": {
        "name": "People's Bank of China",
        "currency": "CNY",
        "current_rate": 3.00,
        "previous_rate": 3.10,
        "direction": RateDirection.CUT,
    },
}


class CentralBankMonitor:
    """Surveille les taux des banques centrales.

    Maintient un état à jour des taux, calcule les différentiels
    et les spreads qui impactent XAUUSD.

    Attributes:
        rates: Dictionnaire des taux par code banque.
    """

    def __init__(self) -> None:
        """Initialise le moniteur avec les données statiques."""
        self.rates: Dict[str, CentralBankRate] = {}
        self._init_defaults()

    def _init_defaults(self) -> None:
        """Initialise les taux par défaut."""
        for code, data in CENTRAL_BANKS_DATA.items():
            self.rates[code] = CentralBankRate(
                bank_name=data["name"],
                code=code,
                current_rate=data["current_rate"],
                previous_rate=data["previous_rate"],
                direction=data["direction"],
                change_pct=data["current_rate"] - data["previous_rate"],
                currency=data["currency"],
            )

    def get_real_rate_spread(self) -> float:
        """Calcule le spread de taux réel Fed vs autres.

        Le spread Fed - BCE est un indicateur clé pour l'or.
        Un spread élevé = USD fort = or baissier.

        Returns:
            Spread Fed - BCE en points de base.
        """
        fed_rate = self.rates.get("FED", CentralBankRate(
            code="FED", bank_name="", current_rate=0.0,
            previous_rate=0.0, direction=RateDirection.UNKNOWN,
        ))
        ecb_rate = self.rates.get("ECB", CentralBankRate(
            code="ECB", bank_name="", current_rate=0.0,
            previous_rate=0.0, direction=RateDirection.UNKNOWN,
        ))
        return fed_rate.current_rate - ecb_rate.current_rate

    def get_dollar_strength_index(self) -> float:
        """Calcule un indice de force du dollar basé sur les taux.

        Pondère les taux des banques centrales par rapport au USD.

        Returns:
            Indice de force du dollar (0 = faible, 1 = fort).
        """
        fed = self.rates.get("FED")
        if not fed:
            return 0.5

        # Comparer Fed vs autres
        others = ["ECB", "BOE", "BOJ", "SNB"]
        scores = []
        for code in others:
            other = self.rates.get(code)
            if other:
                diff = fed.current_rate - other.current_rate
                # Normaliser : -5% = 0, 0% = 0.5, +5% = 1
                score = (diff + 5.0) / 10.0
                scores.append(max(0.0, min(1.0, score)))

        return np.mean(scores) if scores else 0.5

    def get_hiking_cycle_status(self) -> Dict[str, str]:
        """Analyse le cycle de taux global.

        Returns:
            Dict avec le statut par banque.
        """
        status = {}
        for code, rate in self.rates.items():
            if rate.direction == RateDirection.HIKE:
                status[code] = "tightening"
            elif rate.direction == RateDirection.CUT:
                status[code] = "easing"
            else:
                status[code] = "neutral"
        return status

    def get_gold_environment_score(self) -> Tuple[float, str]:
        """Évalue l'environnement macro pour l'or.

        Returns:
            Tuple (score 0-1 défavorable à favorable, description).
        """
        # Facteurs favorables à l'or
        score = 0.5

        # Fed qui baisse ses taux = bon pour l'or
        fed = self.rates.get("FED")
        if fed and fed.direction == RateDirection.CUT:
            score += 0.15

        # Spread Fed-BCE qui se réduit = bon pour l'or
        spread = self.get_real_rate_spread()
        if spread < 0.5:  # Spread < 0.5%
            score += 0.1

        # Taux bas globalement = bon pour l'or
        avg_rate = np.mean([r.current_rate for r in self.rates.values()])
        if avg_rate < 2.0:
            score += 0.1
        elif avg_rate > 4.0:
            score -= 0.1

        # BOJ qui hike = yen fort = USD faible = bon pour or
        boj = self.rates.get("BOJ")
        if boj and boj.direction == RateDirection.HIKE:
            score += 0.05

        score = max(0.0, min(1.0, score))

        if score > 0.7:
            desc = f"🟢 Environnement favorable à l'or ({score:.2f})"
        elif score < 0.3:
            desc = f"🔴 Environnement défavorable à l'or ({score:.2f})"
        else:
            desc = f"⚪ Environnement neutre pour l'or ({score:.2f})"

        return score, desc

    def get_summary(self) -> Dict[str, dict]:
        """Retourne un résumé de toutes les banques centrales.

        Returns:
            Dict {code: {rate, change, direction}}.
        """
        return {
            code: {
                "rate": rate.current_rate,
                "previous": rate.previous_rate,
                "change_bps": rate.change_pct * 100,  # en points de base
                "direction": rate.direction.value,
                "currency": rate.currency,
            }
            for code, rate in self.rates.items()
        }