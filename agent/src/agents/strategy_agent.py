"""
Agent de stratégie et backtesting Octopus.

Lance des backtests, optimise les hyperparamètres et compare
plusieurs configurations de stratégie.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

ENGINE_DIR = Path("/home/kadeva/Octopus/engine")
VENV_PYTHON = "/home/kadeva/trading-env/bin/python3"

CONFIG_PAR_DEFAUT = {
    "symbol": "XAUUSD",
    "period": "6mo",
    "timeframe": "H1",
    "capital": 10000,
    "spread": 0.20,
}


class StrategyAgent:
    """Agent de backtest et optimisation de stratégies.

    Attributes:
        engine_dir: Chemin vers le répertoire engine.
        python_bin: Chemin vers l'interpréteur Python du venv.
        default_config: Configuration par défaut pour les backtests.
    """

    def __init__(self, engine_dir: Optional[str] = None,
                 python_bin: Optional[str] = None,
                 default_config: Optional[Dict[str, Any]] = None):
        """Initialise l'agent de stratégie.

        Args:
            engine_dir: Chemin vers engine/.
            python_bin: Chemin vers python du venv.
            default_config: Configuration par défaut.
        """
        self.engine_dir = Path(engine_dir) if engine_dir else ENGINE_DIR
        self.python_bin = python_bin or VENV_PYTHON
        self.default_config = default_config or CONFIG_PAR_DEFAUT

    def run_backtest(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Lance un backtest avec la configuration donnée.

        Args:
            config: Configuration du backtest (fusion avec défaut).

        Returns:
            Dict avec les métriques de performance.
        """
        cfg = {**self.default_config, **(config or {})}
        code = f"""
import json
import sys
sys.path.insert(0, '{self.engine_dir}')
try:
    from src.backtest import BacktestEngine
    engine = BacktestEngine(
        symbol="{cfg['symbol']}",
        period="{cfg['period']}",
        timeframe="{cfg['timeframe']}",
        capital={cfg['capital']},
        spread={cfg['spread']}
    )
    results = engine.run()
    print(json.dumps(results))
except Exception as e:
    print(json.dumps({{"error": str(e), "status": "failed"}}))
"""
        try:
            result = subprocess.run(
                [self.python_bin, "-c", code],
                capture_output=True, text=True, timeout=600,
                cwd=str(self.engine_dir)
            )
            if result.returncode == 0:
                data = json.loads(result.stdout.strip())
                if "error" not in data:
                    logger.info("Backtest réussi: %s", cfg["symbol"])
                    return data
                else:
                    logger.error("Backtest échoué: %s", data["error"])
                    return {"error": data["error"]}
            else:
                logger.error("Erreur backtest: %s", result.stderr)
                return {"error": result.stderr[:500]}
        except subprocess.TimeoutExpired:
            return {"error": "Timeout (600s dépassé)"}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error("Erreur backtest: %s", e)
            return {"error": str(e)}

    def optimize_params(self, param_grid: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """Optimise les hyperparamètres par recherche en grille.

        Args:
            param_grid: Grille de paramètres à explorer.

        Returns:
            Dict avec les meilleurs paramètres et métriques associées.
        """
        grid = param_grid or {
            "learning_rate": [0.0001, 0.0003, 0.001],
            "gamma": [0.95, 0.99],
            "batch_size": [32, 64, 128],
        }

        code = f"""
import json, itertools
import sys
sys.path.insert(0, '{self.engine_dir}')
try:
    from src.backtest import BacktestEngine
    grid = {json.dumps(grid)}
    best_score = -float('inf')
    best_params = {{}}
    keys = list(grid.keys())
    for values in itertools.product(*grid.values()):
        params = dict(zip(keys, values))
        engine = BacktestEngine(**params)
        results = engine.run()
        score = results.get('sharpe', results.get('total_reward', 0))
        if score > best_score:
            best_score = score
            best_params = params
    print(json.dumps({{"best_params": best_params, "best_score": best_score}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
        try:
            result = subprocess.run(
                [self.python_bin, "-c", code],
                capture_output=True, text=True, timeout=1800,
                cwd=str(self.engine_dir)
            )
            data = json.loads(result.stdout.strip())
            if "error" not in data:
                logger.info("Optimisation terminée: best_score=%.4f",
                            data.get("best_score", 0))
                return data
            else:
                logger.error("Optimisation échouée: %s", data["error"])
                return {"error": data["error"]}
        except Exception as e:
            logger.error("Erreur optimisation: %s", e)
            return {"error": str(e)}

    def compare_strategies(self,
                           configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compare plusieurs configurations de stratégie.

        Args:
            configs: Liste de configurations à comparer.

        Returns:
            Liste des résultats classés par performance.
        """
        results = []
        for config in configs:
            result = self.run_backtest(config)
            if "error" not in result:
                result["config"] = config
                results.append(result)

        # Classement par Sharpe Ratio
        results.sort(
            key=lambda r: r.get("sharpe", r.get("total_reward", 0)),
            reverse=True
        )

        return results

    def run(self, **kwargs) -> Dict[str, Any]:
        """Point d'entrée principal (interface AgentManager).

        Executes backtest, compares with baselines and returns report.

        Returns:
            Rapport complet avec metrics et comparaisons.
        """
        # Backtest principal
        config = kwargs.get("config", {})
        config.setdefault("period", "6mo")
        results = self.run_backtest(config)

        # Baselines
        baselines_configs = [
            {"symbol": "XAUUSD", "period": "6mo", "strategy": "buy_hold"},
            {"symbol": "XAUUSD", "period": "6mo", "strategy": "sma_cross"},
        ]
        baseline_results = self.compare_strategies(baselines_configs)

        # Extraction métriques
        metrics = {}
        if "error" not in results:
            metrics = {
                "sharpe": results.get("sharpe", results.get("sharpe_ratio", 0)),
                "sortino": results.get("sortino", results.get("sortino_ratio", 0)),
                "max_dd": results.get("max_dd", results.get("max_drawdown", 0)),
                "win_rate": results.get("win_rate", results.get("winrate", 0)),
                "profit_factor": results.get("profit_factor", results.get("profit_factor", 0)),
            }

        summary_lines = [
            "📈 Strategy Agent — Backtest Report",
            f"Symbol: {config.get('symbol', 'XAUUSD')} | "
            f"Période: {config.get('period', '6mo')}",
            "-" * 40,
        ]

        if "error" in results:
            summary_lines.append(f"\n❌ Erreur: {results['error']}")
        else:
            for key, val in metrics.items():
                status = "✅" if val > 0 else "⚠️"
                summary_lines.append(f"  {status} {key}: {val}")

            if baseline_results:
                summary_lines.append("\n📊 Comparaison baselines:")
                best_model = results.get("sharpe", 0) or results.get("total_reward", 0)
                for br in baseline_results:
                    baseline_sharpe = br.get("sharpe", 0) or br.get("total_reward", 0)
                    strat = br.get("config", {}).get("strategy", "unknown")
                    diff = best_model - baseline_sharpe
                    summary_lines.append(
                        f"  vs {strat}: "
                        f"{'✅ supérieur' if diff > 0 else '❌ inférieur'} "
                        f"(écart: {diff:.2f})"
                    )

        return {
            "summary": "\n".join(summary_lines),
            "metrics": metrics,
            "best_params": results.get("best_params", {}),
        }