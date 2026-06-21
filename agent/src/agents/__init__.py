"""
Agents spécialisés Octopus.

Modules :
- training_agent : Entraînement du modèle RL
- research_agent : Recherche de papiers arXiv
- macro_agent : Analyse macro-économique
- trading_agent : Surveillance des trades live
- strategy_agent : Backtest et optimisation de stratégies
"""

from .training_agent import TrainingAgent
from .research_agent import ResearchAgent
from .macro_agent import MacroAgent
from .trading_agent import TradingAgent
from .strategy_agent import StrategyAgent

__all__ = [
    "TrainingAgent",
    "ResearchAgent",
    "MacroAgent",
    "TradingAgent",
    "StrategyAgent",
]