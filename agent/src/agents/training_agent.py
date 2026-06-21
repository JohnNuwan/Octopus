"""
Agent d'entraînement du modèle RL Octopus.

Vérifie le GPU, lance l'entraînement, évalue le modèle
et génère un rapport complet des métriques.
"""

import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

ENGINE_DIR = Path("/home/kadeva/Octopus/engine")
VENV_PYTHON = "/home/kadeva/trading-env/bin/python3"


class TrainingAgent:
    """Agent responsable de l'entraînement du modèle RL.

    Attributes:
        engine_dir: Chemin vers le répertoire engine d'Octopus.
        python_bin: Chemin vers l'interpréteur Python du venv.
    """

    def __init__(self, engine_dir: Optional[str] = None,
                 python_bin: Optional[str] = None):
        """Initialise l'agent d'entraînement.

        Args:
            engine_dir: Chemin vers engine/ (sinon défaut Octopus/engine).
            python_bin: Chemin vers python du venv (sinon trading-env).
        """
        self.engine_dir = Path(engine_dir) if engine_dir else ENGINE_DIR
        self.python_bin = python_bin or VENV_PYTHON

    def check_gpu(self) -> bool:
        """Vérifie la disponibilité d'un GPU via PyTorch.

        Returns:
            True si un GPU CUDA est disponible.
        """
        try:
            result = subprocess.run(
                [self.python_bin, "-c",
                 "import torch; print(torch.cuda.is_available())"],
                capture_output=True, text=True, timeout=30,
                cwd=str(self.engine_dir)
            )
            available = result.stdout.strip() == "True"
            if not available:
                logger.warning("GPU non disponible: %s", result.stderr)
            return available
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error("Erreur vérification GPU: %s", e)
            return False

    def check_last_model(self) -> Dict[str, Any]:
        """Analyse le dernier checkpoint sauvegardé.

        Returns:
            Dict avec step, reward, loss ou message d'absence.
        """
        checkpoint_dir = self.engine_dir / "weights"
        if not checkpoint_dir.exists():
            return {"status": "no_checkpoints", "message": "Dossier weights introuvable"}

        try:
            result = subprocess.run(
                [self.python_bin, "-c", f"""
import torch, glob
ckpts = sorted(glob.glob('{checkpoint_dir}/checkpoint*.pth'))
if ckpts:
    ckpt = torch.load(ckpts[-1], map_location='cpu')
    print('STEPS:', ckpt.get('step', 'N/A'))
    print('REWARD:', ckpt.get('total_reward', 'N/A'))
    print('LOSS:', ckpt.get('loss', 'N/A'))
else:
    print('STATUS: no_checkpoints')
"""],
                capture_output=True, text=True, timeout=30,
                cwd=str(self.engine_dir)
            )
            output = result.stdout.strip()
            if "STATUS: no_checkpoints" in output:
                return {"status": "no_checkpoints"}

            metrics = {}
            for line in output.split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    metrics[key.strip().lower()] = val.strip()
            return metrics
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error("Erreur lecture checkpoint: %s", e)
            return {"status": "error", "message": str(e)}

    def run_training(self, steps: int = 30000,
                     timeout: int = 36000) -> Dict[str, Any]:
        """Lance l'entraînement du modèle.

        Args:
            steps: Nombre de steps d'entraînement.
            timeout: Timeout en secondes (défaut: 10h).

        Returns:
            Dict avec le résumé de l'exécution.
        """
        if not self.check_gpu():
            return {
                "summary": "GPU non disponible — entraînement impossible",
                "metrics": {"gpu_available": False, "steps_done": 0},
            }

        logger.info("Démarrage entraînement: %d steps", steps)
        start_time = time.time()

        try:
            result = subprocess.run(
                [self.python_bin, "-m", "src.training",
                 "--steps", str(steps)],
                capture_output=True, text=True,
                timeout=timeout, cwd=str(self.engine_dir)
            )
            elapsed = time.time() - start_time

            if result.returncode == 0:
                summary = f"Entraînement réussi: {steps} steps en {elapsed:.1f}s"
                metrics = {
                    "steps_done": steps,
                    "time_elapsed": elapsed,
                    "gpu_util": self._get_gpu_util(),
                }
                # Tenter d'extraire reward et sharpe des logs
                metrics.update(self._parse_training_output(result.stdout))
            else:
                summary = f"Entraînement échoué (code {result.returncode})"
                metrics = {"steps_done": 0, "error": result.stderr[:500]}

            return {"summary": summary, "metrics": metrics}

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            return {
                "summary": f"Entraînement interrompu (timeout {timeout}s)",
                "metrics": {"steps_done": steps, "time_elapsed": elapsed},
            }
        except Exception as e:
            logger.exception("Erreur entraînement: %s", e)
            return {"summary": f"Erreur: {e}", "metrics": {"steps_done": 0}}

    def evaluate_model(self) -> Dict[str, Any]:
        """Teste le modèle sur des données fraîches.

        Returns:
            Métriques d'évaluation (sharpe, total_reward).
        """
        try:
            result = subprocess.run(
                [self.python_bin, "-m", "src.training", "--evaluate"],
                capture_output=True, text=True, timeout=300,
                cwd=str(self.engine_dir)
            )
            metrics = self._parse_training_output(result.stdout)
            return {
                "summary": "Évaluation terminée",
                "metrics": metrics,
            }
        except Exception as e:
            logger.error("Erreur évaluation: %s", e)
            return {"summary": f"Évaluation échouée: {e}", "metrics": {}}

    def run(self, steps: int = 30000, **kwargs) -> Dict[str, Any]:
        """Point d'entrée principal (interface AgentManager).

        Args:
            steps: Nombre de steps d'entraînement.
            **kwargs: Paramètres supplémentaires.

        Returns:
            Rapport complet avec summary et metrics.
        """
        gpu_ok = self.check_gpu()
        last_model = self.check_last_model()
        training = self.run_training(steps=steps)
        evaluation = self.evaluate_model()

        metrics = {
            "gpu_available": gpu_ok,
            **training.get("metrics", {}),
            **evaluation.get("metrics", {}),
        }

        return {
            "summary": (
                f"Training Agent — {training.get('summary', 'Terminé')}\n"
                f"GPU: {'✅' if gpu_ok else '❌'} | "
                f"Dernier modèle: {last_model.get('step', 'N/A')} steps"
            ),
            "metrics": metrics,
        }

    def _get_gpu_util(self) -> float:
        """Récupère l'utilisation GPU via nvidia-smi.

        Returns:
            Pourcentage d'utilisation GPU (0.0-100.0).
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            return float(result.stdout.strip().split("\n")[0])
        except (FileNotFoundError, ValueError, IndexError):
            return 0.0

    def _parse_training_output(self, output: str) -> Dict[str, Any]:
        """Parse la sortie d'entraînement pour extraire les métriques.

        Args:
            output: Sortie texte de l'entraînement.

        Returns:
            Dict des métriques extraites.
        """
        metrics = {}
        for line in output.split("\n"):
            line_lower = line.lower()
            for key in ["sharpe", "total_reward", "reward", "loss", "step"]:
                if key in line_lower:
                    try:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if key in p.lower():
                                val = parts[i + 1] if i + 1 < len(parts) else ""
                                metrics[key] = float(val.replace(",", ""))
                    except (ValueError, IndexError):
                        pass
        return metrics