"""Tests unitaires du module JEPA (Joint Embedding Predictive Architecture).

Teste les composants suivants :
- JEPAEncoder : encodeur temporel avec attention
- TSJEPA : architecture TS-JEPA complète avec target encoder momentum
- VICRegLoss : perte variance-invariance-covariance
- PositionalEncoding : encodage positionnel sinusoïdal
- JEPAPredictor : prédicteur d'embedding
"""

import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ENGINE_DIR.resolve()))

import torch
import pytest
from typing import Tuple

from src.networks.jepa import (
    JEPAEncoder,
    TSJEPA,
    VICRegLoss,
    PositionalEncoding,
    JEPAPredictor,
)


class TestJEPAEncoder:
    """Tests de l'encodeur JEPA."""

    def test_initialization(self) -> None:
        """Vérifie l'initialisation avec input_dim, seq_len, embedding_dim."""
        encoder = JEPAEncoder(input_dim=20, seq_len=96, embedding_dim=64)
        assert encoder.input_dim == 20
        assert encoder.seq_len == 96
        assert encoder.embedding_dim == 64

    def test_forward_shape(self) -> None:
        """Vérifie que forward produit la forme (B, embedding_dim)."""
        encoder = JEPAEncoder(input_dim=20, seq_len=96, embedding_dim=64)
        x = torch.randn(4, 96, 20)
        output = encoder(x)
        assert output.shape == (4, 64)

    def test_forward_different_batches(self) -> None:
        """Vérifie le fonctionnement avec batch=1 et batch=8."""
        encoder = JEPAEncoder(input_dim=20, seq_len=96, embedding_dim=64)

        x1 = torch.randn(1, 96, 20)
        out1 = encoder(x1)
        assert out1.shape == (1, 64)

        x8 = torch.randn(8, 96, 20)
        out8 = encoder(x8)
        assert out8.shape == (8, 64)

    def test_output_range(self) -> None:
        """Vérifie qu'il n'y a pas de NaN et que les valeurs sont finies."""
        encoder = JEPAEncoder(input_dim=20, seq_len=96, embedding_dim=64)
        x = torch.randn(4, 96, 20)
        output = encoder(x)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()

    def test_deterministic(self) -> None:
        """Vérifie que la même entrée produit la même sortie en eval mode."""
        encoder = JEPAEncoder(input_dim=20, seq_len=96, embedding_dim=64)
        encoder.eval()
        x = torch.randn(2, 96, 20)
        out1 = encoder(x)
        out2 = encoder(x)
        assert torch.allclose(out1, out2, atol=1e-6)


class TestTSJEPA:
    """Tests de l'architecture TS-JEPA complète."""

    def test_initialization(self) -> None:
        """Vérifie que l'architecture contient encoder + target_encoder + predictor."""
        model = TSJEPA(input_dim=20, seq_len=96, embedding_dim=64)
        assert hasattr(model, "encoder")
        assert hasattr(model, "target_encoder")
        assert hasattr(model, "predictor")
        assert isinstance(model.encoder, JEPAEncoder)
        assert isinstance(model.target_encoder, JEPAEncoder)

    def test_train_step(self) -> None:
        """Vérifie que la loss descend après quelques pas d'optimisation."""
        model = TSJEPA(input_dim=5, seq_len=20, embedding_dim=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        x = torch.randn(32, 20, 5)  # grand batch pour stabilité VICReg

        losses = []
        for _ in range(50):
            optimizer.zero_grad()
            loss, metrics = model.train_step(x, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        # La loss doit être finie et diminuer
        assert all(torch.isfinite(torch.tensor(losses)))
        assert losses[-1] < losses[0]

    def test_target_encoder_update(self) -> None:
        """Vérifie que les poids du target encoder bougent après update.

        On modifie d'abord l'encodeur online (par un pas d'optimisation)
        avant de vérifier que le target encoder suit.
        """
        model = TSJEPA(input_dim=5, seq_len=20, embedding_dim=16, momentum=0.5)
        optimizer = torch.optim.SGD(model.encoder.parameters(), lr=10.0)

        # Modifier l'encodeur online par un pas d'optimisation
        x = torch.randn(4, 20, 5)
        z_online = model.encoder(x)
        loss = z_online.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Vérifier que les poids du target encoder diffèrent maintenant
        weights_before = [
            p.clone() for p in model.target_encoder.parameters()
        ]
        model.update_target_encoder()
        weights_after = list(model.target_encoder.parameters())
        any_changed = any(
            not torch.allclose(a, b) for a, b in zip(weights_before, weights_after)
        )
        assert any_changed

    def test_encode_target(self) -> None:
        """Vérifie que encode_target retourne des tenseurs sans gradients."""
        model = TSJEPA(input_dim=5, seq_len=20, embedding_dim=16)
        x = torch.randn(2, 20, 5)
        output = model.encode_target(x)
        assert output.shape == (2, 16)
        assert not output.requires_grad

    def test_get_embedding(self) -> None:
        """Vérifie get_embedding avec les sélecteurs online et target."""
        model = TSJEPA(input_dim=5, seq_len=20, embedding_dim=16)
        x = torch.randn(2, 20, 5)

        emb_online = model.get_embedding(x, use_target=False)
        emb_target = model.get_embedding(x, use_target=True)

        assert emb_online.shape == (2, 16)
        assert emb_target.shape == (2, 16)
        assert emb_online.requires_grad
        assert not emb_target.requires_grad


class TestVICRegLoss:
    """Tests de la perte VICReg."""

    def test_forward(self) -> None:
        """Vérifie que forward retourne un tuple (loss, dict)."""
        loss_fn = VICRegLoss()
        z_pred = torch.randn(8, 16)
        z_target = torch.randn(8, 16)
        loss, metrics = loss_fn(z_pred, z_target)
        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)
        assert loss.ndim == 0  # scalaire

    def test_perfect_match(self) -> None:
        """Vérifie que z_pred ≈ z_target donne une perte faible mais non nulle.

        À cause des termes variance/covariance, la perte ne peut pas être
        exactement zéro même avec une prédiction parfaite.
        """
        loss_fn = VICRegLoss()
        z = torch.randn(8, 16)
        loss, metrics = loss_fn(z, z)
        assert loss.item() < 50.0  # Faible mais pas zéro
        assert loss.item() > 0.0

    def test_different_embeddings(self) -> None:
        """Vérifie que des embeddings très différents donnent une perte élevée."""
        loss_fn = VICRegLoss()
        z_pred = torch.randn(8, 16)
        z_target = torch.randn(8, 16) * 100  # Très différent
        loss, metrics = loss_fn(z_pred, z_target)
        assert loss.item() > 50.0


class TestPositionalEncoding:
    """Tests de l'encodage positionnel."""

    def test_forward_shape(self) -> None:
        """Vérifie que l'encodage préserve la forme d'entrée."""
        pe = PositionalEncoding(d_model=64, dropout=0.0)
        x = torch.randn(4, 20, 64)
        output = pe(x)
        assert output.shape == x.shape

    def test_pe_values(self) -> None:
        """Vérifie que les valeurs d'encodage sont dans [-1, 1]."""
        pe = PositionalEncoding(d_model=64, dropout=0.0)
        # Vérifier que les positions paires/utilisent sin/cos
        assert pe.pe[0, 0, 0] == torch.sin(torch.tensor(0.0))
        # Les valeurs du tableau PE doivent être bornées
        assert torch.all(pe.pe >= -1.0)
        assert torch.all(pe.pe <= 1.0)