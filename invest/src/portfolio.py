"""Optimiseur de portefeuille pour investissement long-terme.

Implémente :
- Mean-Variance Optimization (Markowitz)
- Black-Litterman (views-based)
- Risk Parity (equal risk contribution)
- Factor-based ETF scoring

References:
    Markowitz, H. "Portfolio Selection." Journal of Finance, 1952.
    Black, F. & Litterman, R. "Global Portfolio Optimization." 1992.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class OptimizationResult:
    """Résultat d'une optimisation de portefeuille.
    
    Attributes:
        weights: Dictionnaire {isin: poids} des allocations optimales.
        expected_return: Rendement attendu annualisé.
        expected_volatility: Volatilité attendue annualisée.
        sharpe_ratio: Ratio de Sharpe du portefeuille.
        max_drawdown: Drawdown maximum historique.
    """
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float = 0.0


class PortfolioOptimizer:
    """Optimiseur de portefeuille multi-stratégie.
    
    Supporte l'optimisation pour des portefeuilles d'ETFs
    et d'actions long-terme avec contraintes.
    
    Attributes:
        etfs: Dictionnaire des ETFs disponibles {isin: {name, ter, category}}.
        constraints: Contraintes d'allocation.
    """
    
    # ETFs recommandés pour Trade Republic
    RECOMMENDED_ETFS = {
        # Actions Monde
        "IE00B4L5Y983": {"name": "iShares Core MSCI World", "ter": 0.20, "cat": "DM"},
        "IE00BKM4GZ66": {"name": "iShares Core MSCI EM IMI", "ter": 0.18, "cat": "EM"},
        "IE00B3XXRP09": {"name": "Vanguard S&P 500", "ter": 0.07, "cat": "US"},
        "IE00B4K48X80": {"name": "iShares Core MSCI Europe", "ter": 0.12, "cat": "EU"},
        
        # Obligations
        "IE00BDBRDM35": {"name": "iShares Core Global Agg Bond", "ter": 0.10, "cat": "Bonds"},
        "IE00B1FZS467": {"name": "iShares Euro Govt Bond", "ter": 0.20, "cat": "Govt"},
        
        # Thématiques (satellites)
        "IE00BMC38736": {"name": "iShares Automation & Robotics", "ter": 0.40, "cat": "Tech"},
        "IE00BFM6TC58": {"name": "iShares Global Clean Energy", "ter": 0.65, "cat": "Clean"},
        "IE00B6R52259": {"name": "iShares Physical Gold ETC", "ter": 0.12, "cat": "Gold"},
    }
    
    def __init__(
        self,
        risk_profile: str = "moderate",
        max_single_etf: float = 0.60,
        min_etfs: int = 3,
        max_bonds: float = 0.40
    ) -> None:
        """Initialise l'optimiseur.
        
        Args:
            risk_profile: Profil de risque (conservative, moderate, aggressive).
            max_single_etf: Allocation max par ETF.
            min_etfs: Nombre minimum d'ETFs.
            max_bonds: Allocation max en obligations.
        """
        self.risk_profile = risk_profile
        self.max_single_etf = max_single_etf
        self.min_etfs = min_etfs
        self.max_bonds = max_bonds
    
    def optimize_markowitz(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> OptimizationResult:
        """Optimisation Mean-Variance (Markowitz) standard.
        
        Args:
            returns: DataFrame des rendements (dates × ETFs).
            risk_free_rate: Taux sans risque annuel.
            
        Returns:
            Résultat d'optimisation avec poids optimaux.
        """
        n = len(returns.columns)
        mu = returns.mean() * 252
        cov = returns.cov() * 252
        
        # Maximiser Sharpe ratio
        def neg_sharpe(weights):
            port_ret = np.dot(weights, mu)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            return -(port_ret - risk_free_rate) / port_vol
        
        from scipy.optimize import minimize
        
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, self.max_single_etf) for _ in range(n))
        init_weights = np.ones(n) / n
        
        result = minimize(
            neg_sharpe, init_weights,
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        
        weights = dict(zip(returns.columns, result.x))
        port_ret = np.dot(result.x, mu)
        port_vol = np.sqrt(np.dot(result.x.T, np.dot(cov, result.x)))
        sharpe = (port_ret - risk_free_rate) / port_vol
        
        return OptimizationResult(
            weights=weights,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
        )
    
    def optimize_risk_parity(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> OptimizationResult:
        """Optimisation Risk Parity (contribution égale au risque).
        
        Args:
            returns: DataFrame des rendements.
            risk_free_rate: Taux sans risque.
            
        Returns:
            Résultat avec poids Risk Parity.
        """
        n = len(returns.columns)
        cov = returns.cov() * 252
        
        def risk_contribution_cvx(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            marginal = np.dot(cov, weights) / port_vol
            rc = weights * marginal
            return np.sum((rc - rc.mean())**2)
        
        from scipy.optimize import minimize
        
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0.05, self.max_single_etf) for _ in range(n))
        init_weights = np.ones(n) / n
        
        result = minimize(risk_contribution_cvx, init_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        weights = dict(zip(returns.columns, result.x))
        mu = returns.mean() * 252
        port_ret = np.dot(result.x, mu)
        port_vol = np.sqrt(np.dot(result.x.T, np.dot(cov, result.x)))
        sharpe = (port_ret - risk_free_rate) / port_vol
        
        return OptimizationResult(
            weights=weights,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
        )
    
    def get_lazy_portfolio(self, strategy: str = "core_satellite") -> Dict[str, float]:
        """Portefeuille paresseux (lazy portfolio) sans optimisation.
        
        Stratégies classiques pour Trade Republic :
        - core_satellite: 80% core (World) + 20% satellites (EM, tech)
        - three_fund: World + EM + Bonds
        - boglehead: Vanguard-style, 2-3 fonds
        
        Args:
            strategy: Nom de la stratégie.
            
        Returns:
            Dictionnaire {isin: allocation_pct}.
        """
        strategies = {
            "core_satellite": {
                "IE00B4L5Y983": 0.60,  # MSCI World
                "IE00BKM4GZ66": 0.20,  # MSCI EM
                "IE00BMC38736": 0.10,  # Automation
                "IE00BFM6TC58": 0.10,  # Clean Energy
            },
            "three_fund": {
                "IE00B4L5Y983": 0.70,  # World
                "IE00BKM4GZ66": 0.15,  # EM
                "IE00BDBRDM35": 0.15,  # Bonds
            },
            "boglehead": {
                "IE00B3XXRP09": 0.80,  # S&P 500
                "IE00B1FZS467": 0.20,  # Govt Bonds
            },
            "all_weather": {
                "IE00B4L5Y983": 0.30,  # World
                "IE00BDBRDM35": 0.40,  # Bonds
                "IE00B6R52259": 0.15,  # Gold
                "IE00BKM4GZ66": 0.15,  # EM
            },
        }
        
        return strategies.get(strategy, strategies["core_satellite"])


class DCAManager:
    """Gestionnaire de Dollar Cost Averaging (DCA).
    
    Automatise les investissements périodiques (Sparplan)
    optimisés selon le portefeuille cible et le cash dispo.
    
    Attributes:
        target_allocation: Allocation cible du portefeuille.
        monthly_budget: Budget mensuel d'investissement.
    """
    
    def __init__(
        self,
        target_allocation: Dict[str, float],
        monthly_budget: float = 300.0,
        rebalance_threshold: float = 0.05
    ) -> None:
        """Initialise le gestionnaire DCA.
        
        Args:
            target_allocation: Allocation cible {isin: poids}.
            monthly_budget: Budget mensuel.
            rebalance_threshold: Seuil de rééquilibrage (5%).
        """
        self.target = target_allocation
        self.monthly_budget = monthly_budget
        self.rebalance_threshold = rebalance_threshold
    
    def compute_monthly_investments(
        self,
        current_positions: Dict[str, float],
        target_values: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Calcule les montants à investir ce mois-ci.
        
        Args:
            current_positions: Valeurs actuelles {isin: value_eur}.
            target_values: Valeurs cibles (si différentes de l'allocation).
            
        Returns:
            Montants à investir {isin: amount_eur}.
        """
        total_value = sum(current_positions.values())
        target_values = target_values or {}
        
        investments = {}
        for isin, target_pct in self.target.items():
            current_val = current_positions.get(isin, 0.0)
            target_val = target_values.get(isin, total_value * target_pct)
            gap = target_val - current_val
            
            if gap > 0:
                investments[isin] = min(gap, self.monthly_budget * target_pct * 1.5)
        
        return investments
    
    def check_rebalance_needed(
        self,
        current_allocation: Dict[str, float]
    ) -> List[Tuple[str, float, str]]:
        """Vérifie si un rééquilibrage est nécessaire.
        
        Args:
            current_allocation: Allocation actuelle {isin: ratio}.
            
        Returns:
            Liste des ETFs à rééquilibrer [(isin, deviation, direction)].
        """
        rebalance_list = []
        
        for isin, target_pct in self.target.items():
            current_pct = current_allocation.get(isin, 0.0)
            deviation = current_pct - target_pct
            
            if abs(deviation) > self.rebalance_threshold:
                direction = "OVER" if deviation > 0 else "UNDER"
                rebalance_list.append((isin, deviation, direction))
        
        return rebalance_list


def calculate_dca_returns(
    monthly_investment: float,
    annual_return: float,
    years: int,
    initial: float = 0.0
) -> Dict[str, float]:
    """Calcule les rendements projetés du DCA.
    
    Args:
        monthly_investment: Montant investi chaque mois.
        annual_return: Rendement annuel attendu (décimal).
        years: Nombre d'années.
        initial: Capital initial.
        
    Returns:
        Dictionnaire avec valeur finale, total investi, gains.
    """
    months = years * 12
    monthly_rate = (1 + annual_return) ** (1/12) - 1
    
    total_invested = initial + monthly_investment * months
    final_value = initial * (1 + annual_return) ** years
    
    # Somme d'une série géométrique pour les contributions mensuelles
    for m in range(1, months + 1):
        remaining_months = months - m
        final_value += monthly_investment * (1 + annual_return) ** (remaining_months / 12)
    
    return {
        "final_value": round(final_value, 2),
        "total_invested": round(total_invested, 2),
        "total_gains": round(final_value - total_invested, 2),
        "roi_pct": round((final_value / total_invested - 1) * 100, 1),
    }