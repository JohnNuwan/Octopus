"""Tests unitaires du module Actor-Critic.

Teste les composants suivants :
- ActorNetwork : réseau de politique avec sortie discrète
- CriticNetwork : réseau de valeur avec sortie scalaire
- ActorCritic : module combiné avec calcul de perte et λ-returns
"""

import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ENGINE_DIR.resolve()))

import torch
import pytest
from typing import Tuple

from src.networks.actor_critic import (
    ActorNetwork,
    CriticNetwork,
    ActorCritic,
)


class TestActorNetwork:
    """Tests du réseau de politique (Actor)."""

    def test_forward_shapes(self) -> None:
        """Vérifie les dimensions des sorties logits et probabilités."""
        actor = ActorNetwork(state_dim=128, action_dim=5, hidden_dim=64)
        state = torch.randn(4, 128)
        logits, probs = actor(state)
        assert logits.shape == (4, 5)
        assert probs.shape == (4, 5)

    def test_action_probs_sum_to_one(self) -> None:
        """Vérifie que les probabilités d'action somment à 1."""
        actor = ActorNetwork(state_dim=128, action_dim=5, hidden_dim=64)
        state = torch.randn(4, 128)
        _, probs = actor(state)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_sample_action_deterministic(self) -> None:
        """Vérifie que le mode déterministe retourne la meilleure action."""
        actor = ActorNetwork(state_dim=128, action_dim=5, hidden_dim=64)
        state = torch.randn(2, 128)
        action, log_prob = actor.sample_action(state, deterministic=True)
        assert action.shape == (2, 5)
        assert action.dtype == torch.float32
        # Les actions one-hot doivent sommer à 1
        assert torch.allclose(action.sum(dim=-1), torch.ones(2))

    def test_sample_action_stochastic(self) -> None:
        """Vérifie que le mode stochastique produit une distribution valide."""
        actor = ActorNetwork(state_dim=128, action_dim=5, hidden_dim=64)
        state = torch.randn(2, 128)
        action, log_prob = actor.sample_action(state, deterministic=False)
        assert action.shape == (2, 5)
        assert log_prob.shape == (2,)


class TestCriticNetwork:
    """Tests du réseau de valeur (Critic)."""

    def test_forward_shape(self) -> None:
        """Vérifie que la sortie du critic est (batch, 1)."""
        critic = CriticNetwork(state_dim=128, hidden_dim=64)
        state = torch.randn(4, 128)
        value = critic(state)
        assert value.shape == (4, 1)

    def test_output_finite(self) -> None:
        """Vérifie que les valeurs sont finies (pas de NaN ou Inf)."""
        critic = CriticNetwork(state_dim=128, hidden_dim=64)
        state = torch.randn(4, 128)
        value = critic(state)
        assert torch.isfinite(value).all()


class TestActorCritic:
    """Tests du module Actor-Critic complet."""

    def test_initialization(self) -> None:
        """Vérifie l'initialisation avec les sous-modules actor et critic."""
        ac = ActorCritic(
            stoch_size=4,
            stoch_classes=4,
            deter_size=32,
            action_dim=5,
            hidden_dim=64,
        )
        assert hasattr(ac, "actor")
        assert hasattr(ac, "critic")
        assert isinstance(ac.actor, ActorNetwork)
        assert isinstance(ac.critic, CriticNetwork)

    def test_compute_loss(self) -> None:
        """Vérifie que compute_loss retourne une perte non-NaN avec des données factices."""
        ac = ActorCritic(
            stoch_size=4,
            stoch_classes=4,
            deter_size=32,
            action_dim=5,
            hidden_dim=64,
        )

        batch_size, steps = 4, 5
        stoch = torch.randn(batch_size, steps, 4, 4)
        deter = torch.randn(batch_size, steps, 32)
        actions = torch.nn.functional.one_hot(
            torch.randint(0, 5, (batch_size, steps)), num_classes=5
        ).float()
        rewards = torch.randn(batch_size, steps, 1)
        continues = torch.sigmoid(torch.randn(batch_size, steps, 1))

        loss, metrics = ac.compute_loss(stoch, deter, actions, rewards, continues)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert isinstance(metrics, dict)

    def test_lambda_returns(self) -> None:
        """Vérifie la forme et la monotonie des λ-returns.

        Les λ-returns doivent avoir la même forme que les rewards
        et décroître avec le temps (discount factor).
        """
        ac = ActorCritic(
            stoch_size=4,
            stoch_classes=4,
            deter_size=32,
            action_dim=5,
            hidden_dim=64,
            gamma=0.99,
            lambda_=0.95,
        )

        batch_size, steps = 4, 10
        rewards = torch.ones(batch_size, steps, 1) * 0.1
        values = torch.zeros(batch_size, steps, 1)
        continues = torch.ones(batch_size, steps, 1)

        lambda_returns = ac._compute_lambda_returns(rewards, values, continues)
        assert lambda_returns.shape == (batch_size, steps, 1)

        # Vérifier que les λ-returns sont décroissants (discount)
        for b in range(batch_size):
            for t in range(steps - 1):
                assert lambda_returns[b, t].item() >= lambda_returns[b, t + 1].item()

    def test_get_action(self) -> None:
        """Vérifie que get_action retourne une action one-hot valide."""
        ac = ActorCritic(
            stoch_size=4,
            stoch_classes=4,
            deter_size=32,
            action_dim=5,
            hidden_dim=64,
        )

        stoch = torch.randn(2, 4, 4)
        deter = torch.randn(2, 32)

        action = ac.get_action(stoch, deter, deterministic=True)
        assert action.shape == (2, 5)
        assert torch.allclose(action.sum(dim=-1), torch.ones(2))