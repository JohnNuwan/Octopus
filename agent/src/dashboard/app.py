"""
Dashboard des agents Octopus — monitoring temps réel.

Application FastAPI légère exposant :
- /api/agents : Liste et statut des agents
- /api/agents/{name} : Détail d'un agent
- /api/history : Historique des exécutions
- /api/schedule : Déclenchement de la planification
- / : Interface web de monitoring
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger(__name__)

# Chemin vers les templates
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Singleton du manager (initialisé au démarrage)
_manager = None


def get_manager():
    """Retourne l'instance du AgentManager (singleton)."""
    global _manager
    if _manager is None:
        from agent.src.manager import AgentManager
        from agent.src.agents import (
            TrainingAgent, ResearchAgent, MacroAgent,
            TradingAgent, StrategyAgent,
        )
        _manager = AgentManager(hermes_mode=True)
        _manager.register_agent("training", TrainingAgent())
        _manager.register_agent("research", ResearchAgent())
        _manager.register_agent("macro", MacroAgent())
        _manager.register_agent("trading", TradingAgent())
        _manager.register_agent("strategy", StrategyAgent())
        logger.info("AgentManager initialisé avec 5 agents")
    return _manager


def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI.

    Returns:
        Application FastAPI configurée.
    """
    app = FastAPI(
        title="Octopus Agent Manager",
        description="Dashboard de monitoring des agents Octopus",
        version="0.1.0",
    )

    # CORS pour le dashboard web
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    router = APIRouter(prefix="/api")

    @router.get("/agents")
    async def list_agents():
        """Liste tous les agents et leur statut.

        Returns:
            Dict avec l'état global et la liste des agents.
        """
        manager = get_manager()
        return manager.get_status()

    @router.get("/agents/{name}")
    async def get_agent(name: str):
        """Détail d'un agent spécifique.

        Args:
            name: Nom de l'agent.

        Returns:
            Rapport détaillé de l'agent.

        Raises:
            HTTPException 404: Agent inconnu.
        """
        manager = get_manager()
        report = manager.get_agent_report(name)
        if report is None:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{name}' introuvable"
            )
        return report

    @router.post("/agents/{name}/run")
    async def run_agent(name: str):
        """Exécute un agent spécifique.

        Args:
            name: Nom de l'agent à exécuter.

        Returns:
            Résultat de l'exécution.
        """
        manager = get_manager()
        result = await manager.run_agent(name)
        return {
            "status": result.status,
            "summary": result.summary,
            "metrics": result.metrics,
            "error": result.error,
        }

    @router.get("/history")
    async def list_history(limit: int = 20, offset: int = 0):
        """Historique des exécutions.

        Args:
            limit: Nombre max de résultats.
            offset: Offset pour la pagination.

        Returns:
            Liste des résultats d'exécution.
        """
        manager = get_manager()
        return [
            {
                "agent_name": r.agent_name,
                "status": r.status,
                "started_at": r.started_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "summary": r.summary,
            }
            for r in manager.history[-limit - offset:len(manager.history) - offset]
        ][::-1]

    @router.post("/schedule")
    async def run_schedule():
        """Déclenche la planification quotidienne.

        Returns:
            Liste des résultats d'exécution.
        """
        manager = get_manager()
        results = await manager.run_schedule()
        return [
            {
                "agent_name": r.agent_name,
                "status": r.status,
                "summary": r.summary,
            }
            for r in results
        ]

    @router.get("/health")
    async def health():
        """Endpoint de santé du service.

        Returns:
            Statut du service et timestamp.
        """
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "agents_loaded": len(get_manager().agents),
        }

    # Route pour le dashboard web
    @app.get("/")
    async def dashboard():
        """Page principale du dashboard."""
        html_path = TEMPLATES_DIR / "index.html"
        if html_path.exists():
            return FileResponse(str(html_path))
        return JSONResponse(
            status_code=200,
            content={"message": "Dashboard non trouvé"},
        )

    app.include_router(router)

    @app.on_event("startup")
    async def startup():
        """Initialisation au démarrage."""
        logger.info("Dashboard Octopus démarré sur le port 9093")

    return app


# Point d'entrée pour uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9093)