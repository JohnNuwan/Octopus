"""
Dashboard des agents Octopus — monitoring temps réel.

Fournit une API REST pour le suivi des agents et
une interface web pour la visualisation.
"""

from .app import create_app

__all__ = ["create_app"]