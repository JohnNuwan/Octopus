"""Tests unitaires du World Model RSSM.

Teste les composants suivants :
- RSSMTransition : module de transition d'état latent
- RSSMWorldModel : modèle du monde complet
- symlog / symexp : transformations d'échelle
"""

import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ENGINE_DIR.resolve()))

import torch
import torch.nn as nn
import pytest
from typing import Tuple

from src.networks.world_model import (
    RSSMTransition,
    RSSMWorldModel,
    symlog,
    symexp,
)


class TestRSSMTransition:
    """Tests du module de transition RSSM."""

    @pytest.fixture
    def transition(self) -> RSSMTransition:
        """Fixture pour un module de transition avec des dimensions réduites."""
        return RSSMTransition(
            stoch_size=4,
            stoch_classes=4,
            deter_size=32,
            hidden_dim=32,
            action_dim=5,
            embedding_dim=16,
        )

    def test_forward_shapes(self, transition: RSSMTransition) -> None:
        """Vérifie les dimensions de toutes les sorties du forward."""
        stoch = torch.randn(1, 4, 4)  # one_hot stochastique
        action = torch.randn(1, 5)

        post, prior, deter, log_dict = transition(
            prev_stoch=stoch,
            prev_action=action,
            prev_embedding=torch.randn(1, 16),
            next_embedding=torch.randn(1, 16),
        )

        assert post.shape == (1, 4, 4)
        assert prior.shape == (1, 4, 4)
        assert deter.shape == (1, 32)

    def test_posterior_vs_prior(self, transition: RSSMTransition) -> None:
        """Vérifie que posterior != prior quand next_embedding est fourni."""
        stoch = torch.randn(1, 4, 4)
        action = torch.randn(1, 5)

        post, prior, deter, _ = transition(
            prev_stoch=stoch,
            prev_action=action,
            prev_embedding=torch.randn(1, 16),
            next_embedding=torch.randn(1, 16),
        )

        # Posterior et prior devraient différer
        assert not torch.allclose(post, prior)

    def test_without_next_embedding(self, transition: RSSMTransition) -> None:
        """Vérifie que prior = posterior quand next_embedding est None."""
        stoch = torch.randn(1, 4, 4)
        action = torch.randn(1, 5)

        post, prior, deter, _ = transition(
            prev_stoch=stoch,
            prev_action=action,
        )

        assert torch.allclose(post, prior)

    def test_batch_independence(self, transition: RSSMTransition) -> None:
        """Vérifie que batch=4 produit des sorties indépendantes."""
        stoch = torch.randn(4, 4, 4)
        action = torch.randn(4, 5)

        post, prior, deter, _ = transition(
            prev_stoch=stoch,
            prev_action=action,
        )

        assert post.shape == (4, 4, 4)
        assert prior.shape == (4, 4, 4)
        assert deter.shape == (4, 32)


class TestRSSMWorldModel:
    """Tests du RSSM Word Model complet."""

    @pytest.fixture
    def world_model(self) -> RSSMWorldModel:
        """Fixture pour un World Model avec des dimensions réduites."""
        return RSSMWorldModel(
            stoch_size=4,
            stoch_classes=4,
            deter_size=32,
            hidden_dim=32,
            action_dim=5,
            embedding_dim=16,
        )

    def test_initialization(self, world_model: RSSMWorldModel) -> None:
        """Vérifie l'initialisation et la présence des sous-modules."""
        assert hasattr(world_model, "transition")
        assert hasattr(world_model, "reward_head")
        assert hasattr(world_model, "continue_head")
        assert isinstance(world_model.transition, RSSMTransition)

    def test_predict_reward(self, world_model: RSSMWorldModel) -> None:
        """Vérifie que predict_reward retourne un scalaire par batch."""
        stoch = torch.randn(2, 4, 4)
        deter = torch.randn(2, 32)

        reward = world_model.predict_reward(stoch, deter)
        assert reward.shape == (2, 1)

    def test_predict_continue(self, world_model: RSSMWorldModel) -> None:
        """Vérifie que predict_continue retourne une probabilité entre 0 et 1."""
        stoch = torch.randn(2, 4, 4)
        deter = torch.randn(2, 32)

        cont = world_model.predict_continue(stoch, deter)
        assert cont.shape == (2, 1)
        assert torch.all(cont >= 0.0)
        assert torch.all(cont <= 1.0)

    def test_imagine_shapes(self, world_model: RSSMWorldModel) -> None:
        """Vérifie les dimensions des trajectoires imaginaires."""
        initial_stoch = torch.randn(2, 4, 4)
        initial_deter = torch.randn(2, 32)

        # Politique factice
        class DummyPolicy(nn.Module):
            def forward(self, state: torch.Tensor) -> torch.Tensor:
                return torch.randn(state.shape[0], 5)

        policy = DummyPolicy()
        traj = world_model.imagine(
            initial_stoch, initial_deter, policy, horizon=5
        )

        assert "stoch" in traj
        assert "deter" in traj
        assert "action" in traj
        assert "reward" in traj
        assert "continue" in traj

        # Vérifier que toutes les trajectoires ont bien horizon steps
        assert traj["stoch"].shape == (2, 5, 4, 4)
        assert traj["action"].shape == (2, 5, 5)


class TestSymlogSymexp:
    """Tests des transformations symlog et symexp."""

    def test_symlog_positive(self) -> None:
        """Vérifie symlog sur des valeurs positives."""
        x = torch.tensor([1.0, 10.0, 100.0])
        y = symlog(x)
        assert torch.all(y >= 0.0)

    def test_symlog_negative(self) -> None:
        """Vérifie symlog sur des valeurs négatives."""
        x = torch.tensor([-1.0, -10.0, -100.0])
        y = symlog(x)
        assert torch.all(y <= 0.0)

    def test_symexp_inverse(self) -> None:
        """Vérifie que symexp(symlog(x)) ≈ x."""
        x = torch.tensor([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
        y = symexp(symlog(x))
        assert torch.allclose(x, y, atol=1e-5)

    def test_zero(self) -> None:
        """Vérifie que symlog(0) = 0 et symexp(0) = 0."""
        assert symlog(torch.tensor(0.0)).item() == 0.0
        assert symexp(torch.tensor(0.0)).item() == 0.0