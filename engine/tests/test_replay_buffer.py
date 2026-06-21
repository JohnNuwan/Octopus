"""Tests unitaires du tampon de rejeu (Replay Buffer).

Teste les composants suivants :
- GameHistory : historique d'un épisode de trading
- ReplayBuffer : tampon de rejeu pour l'entraînement
"""

import sys
import os
import tempfile
from pathlib import Path

ENGINE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ENGINE_DIR.resolve()))

import numpy as np
import torch
import pytest
from typing import Tuple, List

from src.training.replay_buffer import GameHistory, ReplayBuffer


class TestGameHistory:
    """Tests de l'historique de partie."""

    def test_add_step(self) -> None:
        """Vérifie l'ajout d'un pas dans l'historique."""
        game = GameHistory()
        obs = np.random.randn(480).astype(np.float32)
        game.add_step(obs, action=1, reward=0.5, policy=np.ones(5) / 5.0, value=0.0)
        assert len(game) == 1
        assert len(game.observations) == 1
        assert len(game.actions) == 1
        assert game.actions[0] == 1

    def test_total_reward(self) -> None:
        """Vérifie que total_reward est la somme des récompenses."""
        game = GameHistory()
        for i in range(3):
            game.add_step(
                np.random.randn(480).astype(np.float32),
                action=i,
                reward=float(i),
                policy=np.ones(5) / 5.0,
                value=0.0,
            )
        assert game.total_reward == 3.0  # 0 + 1 + 2

    def test_save_load(self) -> None:
        """Vérifie le round-trip pickle avec un fichier temporaire."""
        game = GameHistory()
        for i in range(5):
            game.add_step(
                np.random.randn(480).astype(np.float32),
                action=i % 5,
                reward=float(i * 0.1),
                policy=np.ones(5) / 5.0,
                value=float(i),
                done=(i == 4),
            )

        # Sauvegarder dans un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name

        try:
            game.save(tmp_path)
            assert os.path.exists(tmp_path)

            loaded = GameHistory.load(tmp_path)
            assert len(loaded) == len(game)
            assert loaded.actions == game.actions
            assert loaded.rewards == game.rewards
            assert loaded.total_reward == game.total_reward
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_to_dict(self) -> None:
        """Vérifie la conversion en dictionnaire."""
        game = GameHistory()
        game.add_step(
            np.random.randn(480).astype(np.float32),
            action=2,
            reward=1.0,
            policy=np.ones(5) / 5.0,
            value=0.5,
        )
        data = game.to_dict()
        assert isinstance(data, dict)
        assert "observations" in data
        assert "actions" in data
        assert "rewards" in data
        assert "policies" in data
        assert "values" in data
        assert "continues" in data
        assert "total_reward" in data
        assert data["actions"] == [2]
        assert data["rewards"] == [1.0]


class TestReplayBuffer:
    """Tests du tampon de rejeu."""

    def test_add_game(self) -> None:
        """Vérifie l'ajout d'un épisode complet au tampon."""
        buffer = ReplayBuffer(capacity=10)
        game = GameHistory()
        game.add_step(
            np.random.randn(480).astype(np.float32),
            action=0,
            reward=1.0,
            policy=np.ones(5) / 5.0,
            value=0.0,
        )
        buffer.add_game(game)
        assert buffer.num_games == 1
        assert len(buffer) == 1

    def test_add_step(self) -> None:
        """Vérifie que add_step crée automatiquement un nouveau jeu."""
        buffer = ReplayBuffer(capacity=10)
        buffer.add_step(
            obs=np.random.randn(480).astype(np.float32),
            action=0,
            reward=1.0,
            policy=np.ones(5) / 5.0,
            value=0.0,
        )
        assert buffer.num_games == 1
        assert len(buffer) == 1

    def test_add_step_continue(self) -> None:
        """Vérifie que add_step continue ajoute au dernier jeu."""
        buffer = ReplayBuffer(capacity=10)
        for i in range(3):
            buffer.add_step(
                obs=np.random.randn(480).astype(np.float32),
                action=i,
                reward=float(i),
                policy=np.ones(5) / 5.0,
                value=0.0,
            )
        assert buffer.num_games == 1
        assert len(buffer) == 3

    def test_sample_batch_shapes(self) -> None:
        """Vérifie les dimensions des tenseurs échantillonnés."""
        buffer = ReplayBuffer(capacity=100)
        obs_dim = 480

        # Ajouter suffisamment de données
        for _ in range(20):
            game = GameHistory()
            for i in range(10):
                game.add_step(
                    obs=np.random.randn(obs_dim).astype(np.float32),
                    action=i % 5,
                    reward=float(i) * 0.1,
                    policy=np.ones(5) / 5.0,
                    value=float(i) * 0.01,
                    done=(i == 9),
                )
            buffer.add_game(game)

        batch_size = 4
        n_steps = 5
        (
            obs_batch,
            action_batch,
            reward_batch,
            policy_batch,
            value_batch,
            continue_batch,
        ) = buffer.sample_batch(batch_size=batch_size, n_steps=n_steps)

        assert obs_batch.shape == (batch_size, n_steps, obs_dim)
        assert action_batch.shape == (batch_size, n_steps)
        assert reward_batch.shape == (batch_size, n_steps)
        assert policy_batch.shape == (batch_size, n_steps, 5)
        assert value_batch.shape == (batch_size, n_steps)
        assert continue_batch.shape == (batch_size, n_steps)

    def test_capacity(self) -> None:
        """Vérifie le comportement FIFO quand la capacité est dépassée."""
        buffer = ReplayBuffer(capacity=3)

        for i in range(5):
            game = GameHistory()
            game.add_step(
                np.random.randn(480).astype(np.float32),
                action=i,
                reward=1.0,
                policy=np.ones(5) / 5.0,
                value=0.0,
            )
            buffer.add_game(game)

        assert buffer.num_games == 3  # FIFO : seulement les 3 derniers

    def test_len(self) -> None:
        """Vérifie que __len__ retourne le nombre total de pas."""
        buffer = ReplayBuffer(capacity=10)

        for _ in range(3):
            game = GameHistory()
            for i in range(5):
                game.add_step(
                    np.random.randn(480).astype(np.float32),
                    action=i,
                    reward=1.0,
                    policy=np.ones(5) / 5.0,
                    value=0.0,
                )
            buffer.add_game(game)

        assert len(buffer) == 15  # 3 jeux × 5 pas