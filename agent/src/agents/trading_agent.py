"""
Agent de surveillance des trades live Octopus.

Vérifie les positions ouvertes, calcule les métriques de risque
et envoie des alertes via Telegram ou Discord.
"""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Tentative d'import httpx pour les webhooks
try:
    import httpx
    HTTPX_DISPONIBLE = True
except ImportError:
    HTTPX_DISPONIBLE = False

ENGINE_DIR = Path("/home/kadeva/Octopus/engine")
VENV_PYTHON = "/home/kadeva/trading-env/bin/python3"


class TradingAgent:
    """Agent de surveillance des trades live.

    Attributes:
        engine_dir: Chemin vers le répertoire engine.
        python_bin: Chemin vers l'interpréteur Python du venv.
        telegram_token: Token API Telegram (optionnel).
        telegram_chat_id: ID du chat Telegram (optionnel).
        discord_webhook: URL du webhook Discord (optionnel).
    """

    def __init__(self, engine_dir: Optional[str] = None,
                 python_bin: Optional[str] = None,
                 telegram_token: Optional[str] = None,
                 telegram_chat_id: Optional[str] = None,
                 discord_webhook: Optional[str] = None):
        """Initialise l'agent de trading.

        Args:
            engine_dir: Chemin vers engine/.
            python_bin: Chemin vers python du venv.
            telegram_token: Token du bot Telegram.
            telegram_chat_id: ID du chat Telegram.
            discord_webhook: URL du webhook Discord.
        """
        self.engine_dir = Path(engine_dir) if engine_dir else ENGINE_DIR
        self.python_bin = python_bin or VENV_PYTHON
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.discord_webhook = discord_webhook

    def check_positions(self) -> List[Dict[str, Any]]:
        """Vérifie les positions ouvertes via le moteur d'exécution.

        Returns:
            Liste des positions ouvertes avec symbole, volume, PnL.
        """
        code = """
import json
try:
    from src.execution import TradeManager
    tm = TradeManager()
    positions = tm.get_positions()
    print(json.dumps(positions))
except Exception as e:
    print(json.dumps({"error": str(e)}))
"""
        try:
            result = subprocess.run(
                [self.python_bin, "-c", code],
                capture_output=True, text=True, timeout=30,
                cwd=str(self.engine_dir)
            )
            data = json.loads(result.stdout.strip())
            if isinstance(data, list):
                return data
            return []
        except (subprocess.TimeoutExpired, FileNotFoundError,
                json.JSONDecodeError) as e:
            logger.error("Erreur vérification positions: %s", e)
            return []

    def check_risk(self, positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calcule les métriques de risque.

        Args:
            positions: Liste des positions ouvertes.

        Returns:
            Dict avec pnl, drawdown, nombre de positions, daily_pnl.
        """
        if not positions:
            return {
                "pnl": 0.0,
                "drawdown": 0.0,
                "positions": 0,
                "daily_pnl": 0.0,
            }

        total_pnl = sum(
            p.get("profit", p.get("pnl", p.get("unrealized_pnl", 0)))
            for p in positions if isinstance(p, dict)
        )

        # Calcul approximatif du drawdown
        peak = max(
            p.get("profit", p.get("pnl", 0))
            for p in positions if isinstance(p, dict)
        ) or 1
        current = total_pnl
        drawdown = max(0.0, (peak - current) / peak * 100) if peak > 0 else 0.0

        return {
            "pnl": round(total_pnl, 2),
            "drawdown": round(drawdown, 2),
            "positions": len(positions),
            "daily_pnl": round(total_pnl, 2),  # Approximation
        }

    def send_alert(self, message: str, level: str = "info") -> bool:
        """Envoie une alerte via Telegram ou Discord.

        Args:
            message: Contenu de l'alerte.
            level: Niveau (info, warning, critical).

        Returns:
            True si l'envoi a réussi.
        """
        success = False

        # Envoi Telegram
        if self.telegram_token and self.telegram_chat_id and HTTPX_DISPONIBLE:
            try:
                url = (f"https://api.telegram.org/bot{self.telegram_token}"
                       f"/sendMessage")
                payload = {
                    "chat_id": self.telegram_chat_id,
                    "text": f"[{level.upper()}] {message}",
                    "parse_mode": "Markdown",
                }
                resp = httpx.post(url, json=payload, timeout=10)
                if resp.status_code == 200:
                    success = True
                    logger.info("Alerte Telegram envoyée")
            except Exception as e:
                logger.error("Erreur envoi Telegram: %s", e)

        # Envoi Discord
        if self.discord_webhook and HTTPX_DISPONIBLE:
            try:
                colors = {"info": 0x3498DB, "warning": 0xF1C40F,
                          "critical": 0xE74C3C}
                payload = {
                    "embeds": [{
                        "title": f"Octopus Trading - {level.upper()}",
                        "description": message,
                        "color": colors.get(level, 0x3498DB),
                        "timestamp": datetime.now().isoformat(),
                    }]
                }
                resp = httpx.post(self.discord_webhook, json=payload,
                                  timeout=10)
                if resp.status_code in (200, 204):
                    success = True
                    logger.info("Alerte Discord envoyée")
            except Exception as e:
                logger.error("Erreur envoi Discord: %s", e)

        return success

    def run(self, **kwargs) -> Dict[str, Any]:
        """Point d'entrée principal (interface AgentManager).

        Returns:
            Rapport avec métriques et alertes.
        """
        positions = self.check_positions()
        risk = self.check_risk(positions)

        alerts = []
        if risk["drawdown"] > 15:
            alerts.append({
                "type": "drawdown",
                "severity": "critical",
                "message": f"Drawdown critique: {risk['drawdown']:.1f}%",
            })
        elif risk["drawdown"] > 10:
            alerts.append({
                "type": "drawdown",
                "severity": "warning",
                "message": f"Drawdown élevé: {risk['drawdown']:.1f}%",
            })

        if risk["pnl"] < -500:
            alerts.append({
                "type": "pnl",
                "severity": "warning",
                "message": f"Perte journalière: ${risk['pnl']:.2f}",
            })

        for alert in alerts:
            self.send_alert(alert["message"], alert["severity"])

        positions_str = "\n".join(
            f"  {p.get('symbol', '?')} | Vol: {p.get('volume', 0)} | "
            f"PnL: ${p.get('profit', p.get('pnl', 0)):.2f}"
            for p in positions[:5]
        ) if positions else "  Aucune position ouverte"

        return {
            "summary": (
                f"Trading Agent — {len(positions)} position(s) ouverte(s)\n"
                f"PnL Total: ${risk['pnl']:.2f} | "
                f"Drawdown: {risk['drawdown']:.1f}%\n"
                f"Alertes: {len(alerts)} active(s)\n\n"
                f"{positions_str}"
            ),
            "metrics": risk,
            "alerts": alerts,
        }