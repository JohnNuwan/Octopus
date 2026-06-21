"""Configuration partagée des tests unitaires du moteur Octopus.

Définit les fixtures globales utilisées par l'ensemble des tests :
graine aléatoire fixe, device CPU, configurations par défaut,
observations synthétiques et environnements factices.

Attributes:
    SEED: Graine aléatoire fixe pour la reproductibilité (42).
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent de src/ pour que les imports relatifs
# (ex: from ..networks.jepa import TSJEPA) fonctionnent.
ENGINE_DIR = Path(__file__).parent.parent  # engine/
sys.path.insert(0, str(ENGINE_DIR.resolve()))

import numpy as np
import pandas as pd
import torch
import pytest
from typing import Tuple

from src.config import JEPAConfig, MuZeroConfig
from src.environment import OctopusTradingEnv
from src.training.replay_buffer import GameHistory

SEED: int = 42


@pytest.fixture(autouse=True)
def _seed_random() -> None:
    """Fixe les graines aléatoires avant chaque test.

    Garantit la reproductibilité des tests en fixant les graines
    de NumPy, PyTorch et du module random standard.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Fixture du device de calcul — toujours CPU.

    Returns:
        Instance de torch.device pointant vers le CPU.
    """
    return torch.device("cpu")


@pytest.fixture(scope="session")
def jepa_config() -> JEPAConfig:
    """Configuration JEPA par défaut.

    Returns:
        Instance de JEPAConfig avec les valeurs par défaut.
    """
    return JEPAConfig()


@pytest.fixture(scope="session")
def muzero_config() -> MuZeroConfig:
    """Configuration MuZero par défaut.

    Returns:
        Instance de MuZeroConfig avec les valeurs par défaut.
    """
    return MuZeroConfig()


@pytest.fixture
def sample_observation() -> torch.Tensor:
    """Observation synthétique pour les tests JEPA.

    Retourne un tenseur de forme (batch_size=4, features=5, seq_len=20).

    Returns:
        Tenseur d'observation factice de dimensions (4, 5, 20).
    """
    return torch.randn(4, 5, 20)


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """DataFrame minimal de données de marché pour l'environnement.

    Contient les colonnes Open/High/Low/Close/Volume avec 500 pas
    de données synthétiques indexées temporellement.

    Returns:
        DataFrame de 500 lignes avec colonnes OHLCV.
    """
    n_steps: int = 500
    base_price: float = 100.0
    returns: np.ndarray = np.random.randn(n_steps) * 0.01
    prices: np.ndarray = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "Open": prices * (1 + np.random.randn(n_steps) * 0.001),
        "High": prices * (1 + np.abs(np.random.randn(n_steps)) * 0.005),
        "Low": prices * (1 - np.abs(np.random.randn(n_steps)) * 0.005),
        "Close": prices,
        "Volume": np.random.randint(100, 1000, size=n_steps),
    })
    df.index = pd.date_range("2024-01-01", periods=n_steps, freq="15min")
    df["atr"] = prices * 0.01
    return df


@pytest.fixture
def env(empty_df: pd.DataFrame) -> OctopusTradingEnv:
    """Environnement de trading Octopus factice.

    Args:
        empty_df: DataFrame de marché créé par la fixture empty_df.

    Returns:
        Instance d'OctopusTradingEnv initialisée.
    """
    return OctopusTradingEnv(empty_df, lookback=96, initial_capital=10000.0)


@pytest.fixture
def game_history() -> GameHistory:
    """Historique de partie factice avec 10 pas simulés.

    Returns:
        Instance de GameHistory contenant 10 steps.
    """
    history = GameHistory()
    for i in range(10):
        obs = np.random.randn(480).astype(np.float32)  # lookback * 5 features
        action = np.random.randint(0, 5)
        reward = float(np.random.randn())
        policy = np.ones(5, dtype=np.float32) / 5.0
        value = float(np.random.randn())
        done = (i == 9)
        history.add_step(obs, action, reward, policy, value, done)
    return history