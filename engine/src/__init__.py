"""Module principal du moteur Octopus — Trading Algorithmique Multi-Agent.

Ce module implémente l'architecture JEPA + World Model pour le trading
algorithmique XAUUSD, optimisé pour les challenges FTMO 10K.

Références:
    - MuZero: Schrittwieser et al., 2019
    - DreamerV3: Hafner et al., 2023
    - JEPA / VICReg: Bardes et al., 2022 (LeCun)
"""

from . import config
from . import networks
from . import environment
from . import training
from . import live

__version__ = "0.1.0"
__author__ = "JohnNuwan"
__description__ = "Moteur de trading RL avec JEPA + World Model pour XAUUSD"