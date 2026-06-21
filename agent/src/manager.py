"""
Agent Manager — Orchestrateur central des agents Octopus.

Gère le cycle de vie des agents : planification, exécution,
monitoring, reporting. Peut être utilisé en standalone
ou déclenché par Hermes en production.

Agents gérés :
- TrainingAgent : entraînement du modèle RL
- ResearchAgent : recherche de papiers arXiv
- MacroAgent : analyse macro-économique
- TradingAgent : surveillance des trades live
- StrategyAgent : backtest et optimisation
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Résultat d'exécution d'un agent.

    Attributes:
        agent_name: Nom de l'agent exécuté.
        status: Statut de l'exécution (success, failed, running).
        started_at: Timestamp de début d'exécution.
        completed_at: Timestamp de fin d'exécution.
        summary: Résumé textuel du résultat.
        metrics: Métriques numériques produites par l'agent.
        error: Message d'erreur si l'exécution a échoué.
    """
    agent_name: str
    status: str  # success, failed, running
    started_at: datetime
    completed_at: Optional[datetime] = None
    summary: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class AgentManager:
    """Orchestrateur central des agents.

    Enregistre, exécute et monitor les agents Octopus.
    Supporte les modes standalone et Hermes.

    Attributes:
        hermes_mode: Mode d'intégration avec Hermes Agent.
        agents: Dictionnaire des agents enregistrés.
        history: Historique des exécutions.
    """

    def __init__(self, hermes_mode: bool = False):
        """Initialise le gestionnaire d'agents.

        Args:
            hermes_mode: Active le mode Hermes (notifications, logs).
        """
        self.hermes_mode = hermes_mode
        self.agents: Dict[str, Any] = {}
        self.history: List[AgentResult] = []
        self._running = False

    def register_agent(self, name: str, agent) -> None:
        """Enregistre un agent auprès du gestionnaire.

        Args:
            name: Nom unique de l'agent.
            agent: Instance de l'agent (doit avoir une méthode run()).

        Raises:
            ValueError: Si un agent avec ce nom existe déjà.
        """
        if name in self.agents:
            raise ValueError(f"Agent '{name}' déjà enregistré")
        self.agents[name] = agent
        logger.info("Agent enregistré: %s", name)

    async def run_agent(self, name: str, **kwargs) -> AgentResult:
        """Exécute un agent spécifique.

        Args:
            name: Nom de l'agent à exécuter.
            **kwargs: Paramètres passés à l'agent.

        Returns:
            AgentResult contenant le statut et les métriques.
        """
        if name not in self.agents:
            return AgentResult(
                name, "failed", datetime.now(),
                error=f"Agent '{name}' introuvable"
            )

        result = AgentResult(name, "running", datetime.now())
        try:
            agent = self.agents[name]
            if asyncio.iscoroutinefunction(agent.run):
                output = await agent.run(**kwargs)
            else:
                output = agent.run(**kwargs)

            result.status = "success"
            result.completed_at = datetime.now()
            result.summary = output.get("summary", "")
            result.metrics = output.get("metrics", {})
            logger.info("Agent '%s' terminé: success", name)
        except Exception as e:
            result.status = "failed"
            result.completed_at = datetime.now()
            result.error = str(e)
            logger.exception("Agent '%s' échoué: %s", name, e)

        self.history.append(result)
        return result

    async def run_schedule(self) -> List[AgentResult]:
        """Exécute la planification quotidienne des agents.

        Ordre : macro (matin), research (hebdo), training (si dispo).

        Returns:
            Liste des résultats d'exécution.
        """
        self._running = True
        results = []

        if "macro" in self.agents:
            r = await self.run_agent("macro")
            results.append(r)

        if "research" in self.agents:
            r = await self.run_agent("research")
            results.append(r)

        if "training" in self.agents:
            r = await self.run_agent("training")
            results.append(r)

        self._running = False
        return results

    def get_status(self) -> Dict:
        """Retourne l'état de tous les agents.

        Returns:
            Dict avec la liste des agents, statut courant,
            nombre total d'exécutions et derniers résultats.
        """
        return {
            "agents": list(self.agents.keys()),
            "running": self._running,
            "total_runs": len(self.history),
            "last_run": self.history[-1] if self.history else None,
            "recent_results": [
                {
                    "name": r.agent_name,
                    "status": r.status,
                    "time": r.started_at.isoformat()
                }
                for r in self.history[-10:]
            ],
        }

    def get_agent_report(self, name: str) -> Optional[Dict]:
        """Rapport détaillé d'un agent spécifique.

        Args:
            name: Nom de l'agent.

        Returns:
            Dict avec l'historique et les métriques, ou None si inconnu.
        """
        agent_results = [r for r in self.history if r.agent_name == name]
        if not agent_results:
            return None

        last = agent_results[-1]
        successful = sum(1 for r in agent_results if r.status == "success")
        return {
            "name": name,
            "status": last.status,
            "last_run": last.started_at.isoformat(),
            "summary": last.summary,
            "metrics": last.metrics,
            "total_runs": len(agent_results),
            "success_rate": successful / len(agent_results) if agent_results else 0.0,
        }