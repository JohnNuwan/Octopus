"""
Agent d'analyse macro-économique Octopus.

Exécute le MacroFeatureEngine, analyse les indicateurs clés
et génère des alertes sur les conditions extrêmes du marché.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

MACRO_DIR = Path("/home/kadeva/Octopus/macro")
VENV_PYTHON = "/home/kadeva/trading-env/bin/python3"

SEUIL_VOLATILITE = 0.8
SEUIL_SENTIMENT_OR = -0.5


class MacroAgent:
    """Agent d'analyse macro-économique.

    Attributes:
        macro_dir: Chemin vers le répertoire macro.
        python_bin: Chemin vers l'interpréteur Python du venv.
        use_finbert: Utiliser FinBERT pour l'analyse de sentiment.
    """

    def __init__(self, macro_dir: Optional[str] = None,
                 python_bin: Optional[str] = None,
                 use_finbert: bool = False):
        """Initialise l'agent macro.

        Args:
            macro_dir: Chemin vers macro/ (sinon défaut Octopus/macro).
            python_bin: Chemin vers python du venv.
            use_finbert: Activer FinBERT (nécessite transformers).
        """
        self.macro_dir = Path(macro_dir) if macro_dir else MACRO_DIR
        self.python_bin = python_bin or VENV_PYTHON
        self.use_finbert = use_finbert

    def analyze_macro(self) -> Dict[str, Any]:
        """Exécute le MacroFeatureEngine et retourne les indicateurs.

        Returns:
            Dict des indicateurs macro-économiques.
        """
        code = f"""
import json, sys
sys.path.insert(0, '{self.macro_dir}')
try:
    from src.macro_features import MacroFeatureEngine
    engine = MacroFeatureEngine(use_finbert={str(self.use_finbert).lower()})
    data = engine.to_dict() if hasattr(engine, 'to_dict') else {{}}
    print(json.dumps(data))
except Exception as e:
    print(json.dumps({{"error": str(e), "status": "failed"}}))
"""
        try:
            result = subprocess.run(
                [self.python_bin, "-c", code],
                capture_output=True, text=True, timeout=120,
                cwd=str(self.macro_dir)
            )
            if result.returncode == 0:
                data = json.loads(result.stdout.strip())
                logger.info("Analyse macro réussie: %d indicateurs",
                            len(data))
                return data
            else:
                logger.error("Erreur macro engine: %s", result.stderr)
                return {"status": "failed", "error": result.stderr[:500]}
        except (subprocess.TimeoutExpired, FileNotFoundError,
                json.JSONDecodeError) as e:
            logger.error("Erreur analyse macro: %s", e)
            return {"status": "failed", "error": str(e)}

    def check_alerts(self, macro_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Détecte les conditions macro extrêmes.

        Args:
            macro_data: Données du MacroFeatureEngine.

        Returns:
            Liste des alertes avec type, sévérité et message.
        """
        alerts = []
        if not macro_data or "error" in macro_data:
            return alerts

        volatility = macro_data.get("volatility", macro_data.get("vol", 0))
        if isinstance(volatility, (int, float)) and volatility > SEUIL_VOLATILITE:
            alerts.append({
                "type": "volatility",
                "severity": "warning",
                "message": (
                    f"Volatilité élevée ({volatility:.2f} > seuil "
                    f"{SEUIL_VOLATILITE}) — Marché en stress, "
                    f"réduire l'exposition recommandée"
                ),
            })

        sentiment = macro_data.get("gold_sentiment",
                                   macro_data.get("sentiment", 0))
        if isinstance(sentiment, (int, float)) and sentiment < SEUIL_SENTIMENT_OR:
            alerts.append({
                "type": "sentiment",
                "severity": "warning",
                "message": (
                    f"Sentiment or extrêmement négatif ({sentiment:.2f}) "
                    f"— Opportunité d'achat potentielle"
                ),
            })

        fed_rate = macro_data.get("fed_rate",
                                  macro_data.get("central_bank_rate", 0))
        if isinstance(fed_rate, (int, float)) and fed_rate < 3.0:
            alerts.append({
                "type": "central_bank",
                "severity": "info",
                "message": (
                    f"Taux CB bas ({fed_rate:.2f}%) "
                    f"— Assouplissement monétaire, hausse or attendue"
                ),
            })

        env = macro_data.get("gold_environment",
                             macro_data.get("environment", ""))
        if isinstance(env, str) and "extreme" in env.lower():
            alerts.append({
                "type": "environment",
                "severity": "warning",
                "message": f"Environnement or extrême: {env}",
            })

        return alerts

    def generate_report(self, macro_data: Dict[str, Any],
                        alerts: List[Dict[str, str]]) -> str:
        """Génère un rapport textuel formaté.

        Args:
            macro_data: Données macro.
            alerts: Alertes détectées.

        Returns:
            Rapport formaté en texte.
        """
        lines = [
            "📊 Macro Analysis Report",
            f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 50,
        ]

        if "error" in macro_data:
            lines.append(f"\n❌ Erreur: {macro_data['error']}")
            return "\n".join(lines)

        volatility = macro_data.get("volatility", macro_data.get("vol", "N/A"))
        sentiment = macro_data.get("gold_sentiment",
                                   macro_data.get("sentiment", "N/A"))
        fed_rate = macro_data.get("fed_rate",
                                  macro_data.get("central_bank_rate", "N/A"))
        env = macro_data.get("gold_environment",
                             macro_data.get("environment", "Normal"))

        lines.append(f"\n📈 Volatilité:     \t{volatility}")
        lines.append(f"💭 Sentiment Or:   \t{sentiment}")
        lines.append(f"🏦 Taux Fed:       \t{fed_rate}")
        lines.append(f"🌍 Environnement:  \t{env}")

        if alerts:
            lines.append("\n⚠️ ALERTES ACTIVES:")
            for alert in alerts:
                icon = {"warning": "⚠️", "info": "ℹ️", "critical": "🚨"}
                lines.append(f"  {icon.get(alert['severity'], '•')} "
                             f"{alert['message']}")
        else:
            lines.append("\n✅ Aucune alerte — conditions normales")

        return "\n".join(lines)

    def run(self, **kwargs) -> Dict[str, Any]:
        """Point d'entrée principal (interface AgentManager).

        Returns:
            Rapport complet avec métriques et alertes.
        """
        macro_data = self.analyze_macro()
        alerts = self.check_alerts(macro_data)
        report = self.generate_report(macro_data, alerts)

        metrics = {
            "volatility": macro_data.get("volatility",
                                         macro_data.get("vol", 0)),
            "gold_sentiment": macro_data.get("gold_sentiment",
                                              macro_data.get("sentiment", 0)),
            "fed_rate": macro_data.get("fed_rate",
                                       macro_data.get("central_bank_rate", 0)),
            "gold_environment": macro_data.get("gold_environment",
                                               macro_data.get("environment", "normal")),
        }

        return {
            "summary": report,
            "metrics": metrics,
            "alerts": alerts,
        }