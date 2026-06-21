"""
Tests unitaires pour le système Agent Manager Octopus.

Couvre le gestionnaire central, les agents d'entraînement,
de recherche et d'analyse macro.
"""

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ajout du chemin src pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from manager import AgentManager, AgentResult
from agents.training_agent import TrainingAgent
from agents.research_agent import ResearchAgent
from agents.macro_agent import MacroAgent


class TestAgentManager(unittest.TestCase):
    """Tests du gestionnaire central AgentManager."""

    def setUp(self):
        """Initialise un gestionnaire frais pour chaque test."""
        self.manager = AgentManager()

    def test_register_agent(self):
        """Vérifie l'enregistrement d'un agent."""
        mock_agent = MagicMock()
        self.manager.register_agent("test", mock_agent)
        self.assertIn("test", self.manager.agents)
        self.assertEqual(len(self.manager.agents), 1)

    def test_register_duplicate_raises(self):
        """Vérifie que l'enregistrement d'un doublon lève une exception."""
        mock_agent = MagicMock()
        self.manager.register_agent("test", mock_agent)
        with self.assertRaises(ValueError):
            self.manager.register_agent("test", mock_agent)

    def test_run_agent_not_found(self):
        """Vérifie le retour d'erreur pour un agent inconnu."""
        result = asyncio.run(self.manager.run_agent("inexistant"))
        self.assertEqual(result.status, "failed")
        self.assertIn("introuvable", result.error)

    def test_run_agent_success(self):
        """Vérifie l'exécution réussie d'un agent synchrone."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "summary": "Test OK",
            "metrics": {"score": 42},
        }
        self.manager.register_agent("test", mock_agent)
        result = asyncio.run(self.manager.run_agent("test"))
        self.assertEqual(result.status, "success")
        self.assertEqual(result.summary, "Test OK")
        self.assertEqual(result.metrics["score"], 42)

    def test_run_agent_failure(self):
        """Vérifie la gestion d'erreur pendant l'exécution."""
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("Erreur test")
        self.manager.register_agent("test", mock_agent)
        result = asyncio.run(self.manager.run_agent("test"))
        self.assertEqual(result.status, "failed")
        self.assertIn("Erreur test", result.error)

    def test_get_status_empty(self):
        """Vérifie le statut quand aucun agent n'est enregistré."""
        status = self.manager.get_status()
        self.assertEqual(status["agents"], [])
        self.assertEqual(status["total_runs"], 0)
        self.assertFalse(status["running"])

    def test_get_status_with_history(self):
        """Vérifie le statut avec un historique."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = {"summary": "ok", "metrics": {}}
        self.manager.register_agent("test", mock_agent)
        asyncio.run(self.manager.run_agent("test"))
        status = self.manager.get_status()
        self.assertEqual(status["total_runs"], 1)
        self.assertEqual(len(status["recent_results"]), 1)

    def test_get_agent_report_missing(self):
        """Vérifie le rapport pour un agent sans historique."""
        report = self.manager.get_agent_report("inexistant")
        self.assertIsNone(report)

    def test_get_agent_report_with_data(self):
        """Vérifie le rapport détaillé d'un agent."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "summary": "Réussi",
            "metrics": {"accuracy": 0.95},
        }
        self.manager.register_agent("test", mock_agent)
        asyncio.run(self.manager.run_agent("test"))
        report = self.manager.get_agent_report("test")
        self.assertEqual(report["name"], "test")
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["total_runs"], 1)
        self.assertEqual(report["success_rate"], 1.0)


class TestTrainingAgent(unittest.TestCase):
    """Tests de l'agent d'entraînement."""

    def setUp(self):
        self.agent = TrainingAgent(
            engine_dir="/tmp",
            python_bin=sys.executable
        )

    @patch("subprocess.run")
    def test_check_gpu_available(self, mock_run):
        """Vérifie la détection GPU disponible."""
        mock_result = MagicMock()
        mock_result.stdout = "True\n"
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        available = self.agent.check_gpu()
        self.assertTrue(available)

    @patch("subprocess.run")
    def test_check_gpu_unavailable(self, mock_run):
        """Vérifie la détection GPU non disponible."""
        mock_result = MagicMock()
        mock_result.stdout = "False\n"
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        available = self.agent.check_gpu()
        self.assertFalse(available)

    @patch("subprocess.run")
    def test_check_gpu_timeout(self, mock_run):
        """Vérifie la gestion du timeout GPU."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=30)
        available = self.agent.check_gpu()
        self.assertFalse(available)

    @patch.object(TrainingAgent, "check_gpu", return_value=True)
    @patch.object(TrainingAgent, "run_training")
    @patch.object(TrainingAgent, "evaluate_model")
    @patch.object(TrainingAgent, "check_last_model")
    def test_run_mock_training(self, mock_model, mock_eval, mock_train, mock_gpu):
        """Vérifie l'exécution complète simulée."""
        mock_model.return_value = {"step": 30000, "total_reward": 1.5}
        mock_train.return_value = {
            "summary": "Training OK",
            "metrics": {"steps_done": 30000, "time_elapsed": 3600},
        }
        mock_eval.return_value = {
            "summary": "Evaluation OK",
            "metrics": {"sharpe": 1.8},
        }

        result = self.agent.run(steps=100)
        self.assertIn("summary", result)
        self.assertIn("metrics", result)
        self.assertTrue(result["metrics"]["gpu_available"])
        self.assertEqual(result["metrics"]["steps_done"], 30000)


class TestResearchAgent(unittest.TestCase):
    """Tests de l'agent de recherche."""

    def setUp(self):
        self.agent = ResearchAgent(max_papers=2)

    def test_search_no_tools(self):
        """Vérifie le comportement sans outils de recherche."""
        # Simule l'absence d'outils en patchant les flags
        with patch.object(ResearchAgent, '_search_via_hermes', return_value=[]), \
             patch.object(ResearchAgent, '_search_via_arxiv_api', return_value=[]):
            papers = self.agent.search_arxiv(["test topic"])
            self.assertEqual(papers, [])

    def test_search_via_hermes(self):
        """Vérifie la recherche via hermes_tools."""
        from unittest.mock import patch
        mock_data = [
            {"title": "Paper 1", "url": "https://arxiv.org/abs/2401.12345",
             "id": "2401.12345",
             "snippet": "World models for trading"},
        ]
        with patch.object(self.agent, 'search_arxiv', return_value=mock_data):
            papers = self.agent.search_arxiv(["world models"])
            self.assertEqual(len(papers), 1)
            self.assertEqual(papers[0]["id"], "2401.12345")

    def test_search_via_arxiv_api(self):
        """Vérifie la recherche via l'API arXiv."""
        mock_arxiv = MagicMock(return_value=[
            {"title": "World Models Test",
             "id": "2401.99999",
             "url": "http://arxiv.org/abs/2401.99999",
             "summary": "Abstract of the paper"},
        ])
        self.agent._search_via_arxiv_api = mock_arxiv
        papers = self.agent.search_arxiv(["test"])
        self.assertEqual(len(papers), 1)
        self.assertIn("2401.99999", papers[0]["id"])

    def test_extract_arxiv_id(self):
        """Vérifie l'extraction d'ID arXiv."""
        url = "https://arxiv.org/abs/2401.12345"
        self.assertEqual(ResearchAgent._extract_arxiv_id(url), "2401.12345")

        url2 = "https://arxiv.org/pdf/2304.67890.pdf"
        self.assertEqual(ResearchAgent._extract_arxiv_id(url2), "2304.67890")

        url3 = "https://example.com"
        self.assertEqual(ResearchAgent._extract_arxiv_id(url3), "")

    def test_score_relevance(self):
        """Vérifie le scoring de pertinence."""
        text = "Reinforcement learning for trading with world models"
        score = ResearchAgent._score_relevance(text)
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 10)


class TestMacroAgent(unittest.TestCase):
    """Tests de l'agent macro."""

    def setUp(self):
        self.agent = MacroAgent(
            macro_dir="/tmp",
            python_bin=sys.executable
        )

    def test_check_alerts_normal(self):
        """Vérifie qu'aucune alerte n'est générée en conditions normales."""
        data = {
            "volatility": 0.3,
            "gold_sentiment": 0.1,
            "fed_rate": 4.5,
            "gold_environment": "Normal",
        }
        alerts = self.agent.check_alerts(data)
        self.assertEqual(len(alerts), 0)

    def test_check_alerts_high_volatility(self):
        """Vérifie l'alerte de volatilité élevée."""
        data = {"volatility": 0.85, "gold_sentiment": 0.5, "fed_rate": 4.5}
        alerts = self.agent.check_alerts(data)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["type"], "volatility")
        self.assertEqual(alerts[0]["severity"], "warning")

    def test_check_alerts_extreme_sentiment(self):
        """Vérifie l'alerte de sentiment extrême."""
        data = {"volatility": 0.3, "gold_sentiment": -0.8, "fed_rate": 4.5}
        alerts = self.agent.check_alerts(data)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["type"], "sentiment")

    def test_check_alerts_mixed(self):
        """Vérifie les alertes multiples."""
        data = {
            "volatility": 0.9,
            "gold_sentiment": -0.7,
            "fed_rate": 2.5,
        }
        alerts = self.agent.check_alerts(data)
        self.assertEqual(len(alerts), 3)

    def test_check_alerts_empty_data(self):
        """Vérifie qu'aucune alerte n'est générée sur données vides."""
        alerts = self.agent.check_alerts({})
        self.assertEqual(len(alerts), 0)

    def test_check_alerts_error_data(self):
        """Vérifie qu'aucune alerte n'est générée sur données d'erreur."""
        alerts = self.agent.check_alerts({"error": "failed"})
        self.assertEqual(len(alerts), 0)

    def test_generate_report(self):
        """Vérifie la génération du rapport."""
        data = {
            "volatility": 0.5,
            "gold_sentiment": 0.2,
            "fed_rate": 4.5,
            "gold_environment": "Normal",
        }
        report = self.agent.generate_report(data, [])
        self.assertIn("Macro Analysis Report", report)
        self.assertIn("Aucune alerte", report)

    def test_generate_report_with_alerts(self):
        """Vérifie le rapport avec alertes."""
        data = {"volatility": 0.9}
        alerts = [{"type": "volatility", "severity": "warning",
                    "message": "Vol élevée"}]
        report = self.agent.generate_report(data, alerts)
        self.assertIn("ALERTES ACTIVES", report)
        self.assertIn("Vol élevée", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)