"""Tests unitaires du module MCTS (Monte Carlo Tree Search).

Teste les composants suivants :
- MCTSNode : nœud de l'arbre de recherche
- MCTS : algorithme de recherche arborescente
"""

import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ENGINE_DIR.resolve()))

import math
import torch
import numpy as np
import pytest
from typing import Tuple, Dict, Any

from src.mcts import MCTS, MCTSNode


class MockMuZeroNetwork:
    """Réseau factice pour les tests MCTS.

    Simule les méthodes initial_inference et recurrent_inference
    du World Model utilisé par MCTS.
    """

    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 5,
        hidden_dim: int = 32,
    ) -> None:
        """Initialise le réseau factice.

        Args:
            state_dim: Dimension de l'état latent.
            action_dim: Nombre d'actions discrètes.
            hidden_dim: Dimension cachée factice.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def initial_inference(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inférence initiale : état → hidden, policy, value.

        Args:
            state: État d'entrée (batch, state_dim).

        Returns:
            Tuple (hidden_state, policy_logits, value).
        """
        batch = state.shape[0]
        hidden = torch.randn(batch, self.hidden_dim)
        policy_logits = torch.randn(batch, self.action_dim)
        value = torch.randn(batch, 1)
        return hidden, policy_logits, value

    def recurrent_inference(
        self, hidden_state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inférence récurrente : hidden + action → next_hidden, reward, policy, value.

        Args:
            hidden_state: État caché (batch, hidden_dim).
            action: Action one-hot (batch, action_dim).

        Returns:
            Tuple (next_hidden, reward, policy_logits, value).
        """
        batch = hidden_state.shape[0]
        next_hidden = torch.randn(batch, self.hidden_dim)
        reward = torch.randn(batch, 1)
        policy_logits = torch.randn(batch, self.action_dim)
        value = torch.randn(batch, 1)
        return next_hidden, reward, policy_logits, value


class TestMCTSNode:
    """Tests du nœud MCTS."""

    def test_initialization(self) -> None:
        """Vérifie l'initialisation d'un nœud vide."""
        node = MCTSNode()
        assert node.visit_count == 0
        assert node.total_value == 0.0
        assert node.value == 0.0
        assert not node.expanded
        assert node.children == {}

    def test_value(self) -> None:
        """Vérifie que value = total_value / visit_count."""
        node = MCTSNode()
        node.visit_count = 5
        node.total_value = 10.0
        assert node.value == 2.0

    def test_ucb_score(self) -> None:
        """Vérifie que UCB croît avec prior et décroît avec visit_count."""
        parent_visits = 10

        node_high_prior = MCTSNode(prior=0.5, visit_count=1)
        node_low_prior = MCTSNode(prior=0.1, visit_count=1)

        score_high = node_high_prior.ucb_score(parent_visits)
        score_low = node_low_prior.ucb_score(parent_visits)
        assert score_high >= score_low  # UCB devrait être plus élevé avec prior plus haut

    def test_ucb_score_visit_count(self) -> None:
        """Vérifie que UCB décroît avec le nombre de visites."""
        parent_visits = 10

        node_few_visits = MCTSNode(prior=0.3, visit_count=1)
        node_many_visits = MCTSNode(prior=0.3, visit_count=10)

        score_few = node_few_visits.ucb_score(parent_visits)
        score_many = node_many_visits.ucb_score(parent_visits)
        assert score_few > score_many


class TestMCTS:
    """Tests de l'algorithme MCTS."""

    def test_initialization(self) -> None:
        """Vérifie l'initialisation avec les paramètres par défaut."""
        mcts = MCTS()
        assert mcts.action_dim == 5
        assert mcts.num_simulations == 150
        assert mcts.discount == 0.99

    def test_search_with_mock_network(self) -> None:
        """Vérifie que search retourne une distribution de visite valide.

        La distribution doit être un vecteur de probabilités
        qui somme à 1.
        """
        mcts = MCTS(
            action_dim=5,
            num_simulations=10,  # Faible pour les tests
            discount=0.99,
        )

        mock_network = MockMuZeroNetwork(state_dim=128, action_dim=5)
        root_state = torch.randn(1, 128)

        visit_dist = mcts.search(root_state, mock_network, device=torch.device("cpu"))

        assert isinstance(visit_dist, np.ndarray)
        assert visit_dist.shape == (5,)
        assert np.allclose(visit_dist.sum(), 1.0, atol=1e-6)
        assert np.all(visit_dist >= 0.0)