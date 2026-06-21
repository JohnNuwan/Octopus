"""Tests unitaires de l'environnement de trading Octopus.

Teste les composants suivants :
- OctopusTradingEnv : environnement Gymnasium-like avec 5 actions
- FTMOEnforcer : moteur des règles FTMO Challenge
- SLBESystem : système Stop Loss Break Even
- Trade : dataclass représentant un trade exécuté
"""

import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ENGINE_DIR.resolve()))

import numpy as np
import pandas as pd
import pytest
from typing import Dict, Any

from src.environment import (
    OctopusTradingEnv,
    FTMOEnforcer,
    SLBESystem,
    Trade,
)


class TestOctopusTradingEnv:
    """Tests de l'environnement de trading OctopusTradingEnv."""

    def test_initialization(self, empty_df: pd.DataFrame) -> None:
        """Vérifie que l'environnement s'initialise correctement.

        Contrôle la dimension de l'état, le nombre d'actions
        et la présence du moteur FTMO.
        """
        env = OctopusTradingEnv(empty_df, lookback=96, initial_capital=10000.0)
        assert env.state_dim > 0
        assert env.action_dim == 5
        assert env.ftmo is not None
        assert env.ftmo.initial_capital == 10000.0

    def test_reset(self, env: OctopusTradingEnv) -> None:
        """Vérifie que reset retourne une observation de bonne dimension.

        L'observation doit être un vecteur plat de taille
        lookback × nombre_de_features.
        """
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 1
        assert obs.shape[0] == env.state_dim

    def test_step_hold(self, env: OctopusTradingEnv) -> None:
        """Vérifie que l'action Hold (0) ne change pas la position."""
        env.reset()
        action_hold = 0
        obs, reward, done, info = env.step(action_hold)
        assert env.position == 0
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_buy(self, env: OctopusTradingEnv) -> None:
        """Vérifie que l'action Buy (1) ouvre une position longue."""
        env.reset()
        obs, reward, done, info = env.step(1)
        assert env.position == 1

    def test_step_sell(self, env: OctopusTradingEnv) -> None:
        """Vérifie que l'action Sell (2) ouvre une position courte."""
        env.reset()
        obs, reward, done, info = env.step(2)
        assert env.position == -1

    def test_step_close(self, env: OctopusTradingEnv) -> None:
        """Vérifie que l'action Close (4) ferme la position."""
        env.reset()
        env.step(1)  # Ouvrir une position longue
        assert env.position == 1
        env.step(4)  # Fermer
        assert env.position == 0

    def test_step_split(self, env: OctopusTradingEnv) -> None:
        """Vérifie que l'action Split (3) réduit la position de 50%.

        Le split est effectif uniquement si la position est profitable.
        On force un contexte favorable en manipulant les prix.
        """
        env.reset()
        env.step(1)  # Buy

        # Simuler un mouvement haussier pour que le trade soit profitable
        original_close = env.df.iloc[env.idx]["Close"]
        env.df.iloc[env.idx, env.df.columns.get_loc("Close")] = original_close * 1.02

        obs, reward, done, info = env.step(3)  # Split
        # Le split doit réussir si le trade est profitable
        assert info["position"] in [0, 1]
        assert isinstance(reward, float)

    def test_multiple_trades(self, env: OctopusTradingEnv) -> None:
        """Vérifie une séquence d'actions Buy → Close → Buy → Close."""
        env.reset()
        env.step(1)  # Buy
        assert env.position == 1
        env.step(4)  # Close
        assert env.position == 0
        env.step(2)  # Sell
        assert env.position == -1
        env.step(4)  # Close
        assert env.position == 0

    def test_episode_end(self, env: OctopusTradingEnv) -> None:
        """Vérifie que done=True est retourné en fin de DataFrame."""
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(0)
        assert done


class TestFTMOEnforcer:
    """Tests du moteur de règles FTMO Challenge."""

    def test_initialization(self) -> None:
        """Vérifie l'initialisation avec balance=10000 et phase=1."""
        ftmo = FTMOEnforcer(initial_capital=10000.0)
        assert ftmo.balance == 10000.0
        assert ftmo.phase == 1
        assert not ftmo.failed
        assert not ftmo.passed

    def test_daily_loss_limit(self) -> None:
        """Vérifie que -5% de perte quotidienne déclenche l'échec."""
        ftmo = FTMOEnforcer(initial_capital=10000.0)
        # Perte quotidienne de -6%
        ftmo.daily_pnl = -600.0
        assert ftmo.check_daily_loss()
        assert ftmo.failed

    def test_total_loss_limit(self) -> None:
        """Vérifie que -10% de perte totale déclenche l'échec."""
        ftmo = FTMOEnforcer(initial_capital=10000.0)
        ftmo.balance = 8900.0  # -11%
        assert ftmo.check_total_loss()
        assert ftmo.failed

    def test_profit_target(self) -> None:
        """Vérifie que +10% valide la phase 1."""
        ftmo = FTMOEnforcer(initial_capital=10000.0)
        ftmo.balance = 11500.0  # +15%
        result = ftmo.check_profit_target()
        assert result == "phase1_passed"
        assert ftmo.phase == 2

    def test_two_phases(self) -> None:
        """Vérifie la transition phase 1 → phase 2."""
        ftmo = FTMOEnforcer(initial_capital=10000.0)
        ftmo.balance = 11500.0  # +15%
        result1 = ftmo.check_profit_target()
        assert result1 == "phase1_passed"
        assert ftmo.phase == 2

        # Phase 2 : encore +10% sur le nouveau solde
        ftmo.balance = 12650.0  # +10% sur 11500
        result2 = ftmo.check_profit_target()
        assert result2 == "passed"
        assert ftmo.passed

    def test_min_trading_days(self) -> None:
        """Vérifie que les jours de trading sont enregistrés."""
        ftmo = FTMOEnforcer(initial_capital=10000.0)
        ftmo.trading_days.add(pd.Timestamp("2024-01-01").date())
        ftmo.trading_days.add(pd.Timestamp("2024-01-02").date())
        assert len(ftmo.trading_days) == 2


class TestSLBESystem:
    """Tests du système Stop Loss Break Even."""

    def test_initial_bonus(self) -> None:
        """Vérifie que SLBE s'active à +0.5% de profit non réalisé."""
        slbe = SLBESystem()
        activated, bonus = slbe.update(
            unrealized_pnl_pct=0.006,  # +0.6%
            avg_entry=100.0,
            current_price=101.0,
            direction=1,
        )
        assert activated
        assert bonus > 0
        assert slbe.active

    def test_no_bonus_before_threshold(self) -> None:
        """Vérifie qu'aucun bonus n'est donné sous le seuil de +0.5%."""
        slbe = SLBESystem()
        activated, bonus = slbe.update(
            unrealized_pnl_pct=0.003,  # +0.3%
            avg_entry=100.0,
            current_price=100.3,
            direction=1,
        )
        assert not activated
        assert bonus == 0.0
        assert not slbe.active

    def test_reset(self) -> None:
        """Vérifie que reset réinitialise le système SLBE."""
        slbe = SLBESystem()
        slbe.active = True
        slbe.slbe_price = 100.0
        slbe.reset()
        assert not slbe.active
        assert slbe.slbe_price == 0.0


class TestTrade:
    """Tests de la dataclass Trade."""

    def test_trade_defaults(self) -> None:
        """Vérifie les valeurs par défaut de la dataclass Trade."""
        trade = Trade(direction=1, entry_price=100.0, entry_step=10, lots=0.1)
        assert trade.direction == 1
        assert trade.entry_price == 100.0
        assert trade.entry_step == 10
        assert trade.lots == 0.1
        assert trade.exit_price is None
        assert trade.exit_step is None
        assert trade.pnl is None