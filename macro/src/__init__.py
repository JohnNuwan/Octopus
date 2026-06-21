"""Module Macro-économique Octopus — Calendrier, Sentiment, Taux Banques Centrales.

Analyse les données macro-économiques et le sentiment de marché
pour enrichir les features du moteur de trading XAUUSD.

Sources :
- Calendrier économique : Investing.com (via investpy)
- Sentiment news : NLP FinBERT sur flux RSS Bloomberg/Reuters
- Taux banques centrales : Fed, BCE, BOE, BOJ, SNB, RBA, RBNZ
"""

from .calendar import EconomicCalendar, EconomicEvent
from .sentiment import NewsSentimentAnalyzer, SentimentResult
from .central_banks import CentralBankMonitor, CentralBankRate
from .macro_features import MacroFeatureEngine, MacroFeatures

__all__ = [
    "EconomicCalendar", "EconomicEvent",
    "NewsSentimentAnalyzer", "SentimentResult",
    "CentralBankMonitor", "CentralBankRate",
    "MacroFeatureEngine", "MacroFeatures",
]