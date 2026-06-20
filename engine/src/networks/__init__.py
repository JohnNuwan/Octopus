"""Module des réseaux de neurones pour le moteur Octopus.

Contient les implémentations de l'architecture JEPA + World Model
pour le trading algorithmique XAUUSD.

Références:
    - TS-JEPA: Joint Embeddings Go Temporal (arXiv 2509.25449, 2025)
    - VICReg: Bardes, Ponce, LeCun. ICLR 2022.
    - DreamerV3: Hafner et al. Nature, 2025.
"""

from . import jepa
from . import world_model
from . import actor_critic

__all__ = ["jepa", "world_model", "actor_critic"]