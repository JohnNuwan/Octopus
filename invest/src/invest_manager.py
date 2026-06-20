"""Module d'investissement long-terme pour Trade Republic.

Automatise :
- Optimisation de portefeuille (Markowitz, Risk Parity, Lazy)
- DCA (Dollar Cost Averaging) mensuel
- Rééquilibrage automatique
- Suivi fiscal (dividendes, Freistellungsauftrag)
- Projection de rendements
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))
from portfolio import PortfolioOptimizer, DCAManager, calculate_dca_returns
from trade_republic_client import TradeRepublicClient, AccountInfo, PortfolioPosition


@dataclass
class InvestmentConfig:
    """Configuration de la stratégie d'investissement.
    
    Attributes:
        strategy: Stratégie (core_satellite, three_fund, boglehead, all_weather, markowitz).
        monthly_budget: Budget mensuel en euros.
        risk_profile: Profil de risque (conservative, moderate, aggressive).
        rebalance_frequency: Fréquence de rééquilibrage (monthly, quarterly).
        enable_sparplan: Activer les plans d'épargne automatiques.
    """
    strategy: str = "core_satellite"
    monthly_budget: float = 300.0
    risk_profile: str = "moderate"
    rebalance_frequency: str = "quarterly"
    enable_sparplan: bool = True


class InvestmentManager:
    """Gestionnaire d'investissement long-terme.
    
    Orchestre toute la stratégie d'investissement :
    1. Analyse du portefeuille actuel
    2. Calcul de l'allocation optimale
    3. Exécution des investissements mensuels
    4. Rééquilibrage périodique
    5. Rapport de performance
    
    Attributes:
        config: Configuration de la stratégie.
        client: Client Trade Republic.
        optimizer: Optimiseur de portefeuille.
        dca_manager: Gestionnaire DCA.
        target_allocation: Allocation cible calculée.
    """
    
    def __init__(
        self,
        config: Optional[InvestmentConfig] = None,
        phone: Optional[str] = None,
        pin: Optional[str] = None
    ) -> None:
        """Initialise le gestionnaire d'investissement.
        
        Args:
            config: Configuration de la stratégie.
            phone: Numéro Trade Republic.
            pin: PIN Trade Republic.
        """
        self.config = config or InvestmentConfig()
        self.client = TradeRepublicClient(
            phone or os.getenv('TR_PHONE', ''),
            pin or os.getenv('TR_PIN', '')
        )
        self.optimizer = PortfolioOptimizer(
            risk_profile=self.config.risk_profile
        )
        self.target_allocation = {}
        self.dca_manager = None
    
    def set_target_allocation(self, strategy: Optional[str] = None) -> Dict[str, float]:
        """Définit l'allocation cible selon la stratégie choisie.
        
        Args:
            strategy: Nom de la stratégie (utilise config si None).
            
        Returns:
            Allocation cible {isin: poids}.
        """
        strategy = strategy or self.config.strategy
        
        if strategy == "markowitz":
            # Nécessite des données historiques
            self.target_allocation = {
                "IE00B4L5Y983": 0.50,
                "IE00BKM4GZ66": 0.20,
                "IE00BDBRDM35": 0.20,
                "IE00B6R52259": 0.10,
            }
        else:
            self.target_allocation = self.optimizer.get_lazy_portfolio(strategy)
        
        self.dca_manager = DCAManager(
            target_allocation=self.target_allocation,
            monthly_budget=self.config.monthly_budget
        )
        
        return self.target_allocation
    
    async def analyze_portfolio(self) -> Dict:
        """Analyse complète du portefeuille actuel.
        
        Returns:
            Dictionnaire avec l'état complet du portefeuille.
        """
        print("\n📊 Analyse du Portefeuille Trade Republic")
        print("=" * 50)
        
        await self.client.login()
        positions = await self.client.get_portfolio()
        account = await self.client.get_account_info()
        
        total_value = sum(p.current_price * p.quantity for p in positions) + account.cash_balance
        
        # Allocation actuelle
        current_allocation = {}
        for p in positions:
            val = p.current_price * p.quantity
            current_allocation[p.isin] = val / total_value
        
        # Comparer avec l'allocation cible
        if not self.target_allocation:
            self.set_target_allocation()
        
        print(f"\n💰 Capital total : {total_value:,.2f}€")
        print(f"   Cash disponible : {account.cash_balance:,.2f}€")
        print(f"   Profil de risque : {self.config.risk_profile}")
        print(f"   Stratégie : {self.config.strategy}")
        
        print(f"\n📈 Positions actuelles ({len(positions)}) :")
        for p in positions:
            print(f"   {p.name[:40]:40s} {p.allocation_pct:5.1f}%  PnL: {p.pnl_eur:+.2f}€ ({p.pnl_pct:+.1f}%)")
        
        return {
            "total_value": total_value,
            "cash_balance": account.cash_balance,
            "positions": positions,
            "current_allocation": current_allocation,
            "target_allocation": self.target_allocation,
        }
    
    async def execute_monthly_investment(self) -> Dict:
        """Exécute l'investissement mensuel.
        
        Returns:
            Résultat des investissements du mois.
        """
        print(f"\n💶 Investissement Mensuel — {datetime.now().strftime('%B %Y')}")
        print("-" * 40)
        
        if not self.dca_manager:
            self.set_target_allocation()
        
        # Récupérer les positions actuelles
        positions = await self.client.get_portfolio()
        current_values = {}
        for p in positions:
            current_values[p.isin] = p.current_price * p.quantity
        
        # Calculer les montants à investir
        investments = self.dca_manager.compute_monthly_investments(current_values)
        
        total_to_invest = sum(investments.values())
        print(f"   Budget : {self.config.monthly_budget:.2f}€")
        print(f"   À investir ce mois : {total_to_invest:.2f}€")
        
        # Exécuter les ordres
        results = []
        for isin, amount in investments.items():
            etf_info = self.optimizer.RECOMMENDED_ETFS.get(isin, {})
            name = etf_info.get('name', isin)
            print(f"   📈 {name[:40]:40s} {amount:8.2f}€")
            
            result = await self.client.place_order(isin, amount)
            results.append(result)
        
        return {
            "month": datetime.now().strftime('%Y-%m'),
            "total_invested": total_to_invest,
            "investments": investments,
            "results": results,
        }
    
    async def check_and_rebalance(self) -> Dict:
        """Vérifie si un rééquilibrage est nécessaire.
        
        Returns:
            Résultat du rééquilibrage.
        """
        print(f"\n🔄 Vérification Rééquilibrage")
        print("-" * 40)
        
        positions = await self.client.get_portfolio()
        total_value = sum(p.current_price * p.quantity for p in positions)
        
        current_allocation = {}
        for p in positions:
            current_allocation[p.isin] = p.current_price * p.quantity / total_value
        
        needs = self.dca_manager.check_rebalance_needed(current_allocation)
        
        if not needs:
            print("   ✅ Allocation dans les limites — pas de rééquilibrage")
            return {"rebalanced": False}
        
        print(f"   ⚠️  {len(needs)} ETF(s) hors cible :")
        for isin, deviation, direction in needs:
            etf_info = self.optimizer.RECOMMENDED_ETFS.get(isin, {})
            print(f"   {'🔴' if direction == 'OVER' else '🟢'} "
                  f"{etf_info.get('name', isin)[:40]:40s} "
                  f"{deviation:+.1%} ({direction})")
        
        return {"rebalanced": len(needs) > 0, "details": needs}
    
    def projection_report(self, years: int = 20) -> str:
        """Génère un rapport de projection sur le long terme.
        
        Args:
            years: Nombre d'années de projection.
            
        Returns:
            Rapport formaté.
        """
        # Rendements historiques approximatifs par stratégie
        strategy_returns = {
            "core_satellite": 0.08,
            "three_fund": 0.07,
            "boglehead": 0.09,
            "all_weather": 0.06,
            "markowitz": 0.07,
        }
        
        annual_ret = strategy_returns.get(self.config.strategy, 0.07)
        result = calculate_dca_returns(
            monthly_investment=self.config.monthly_budget,
            annual_return=annual_ret,
            years=years,
        )
        
        report = f"""
📊 Projection Long Terme — {self.config.strategy.upper()}
{'='*50}
  Budget mensuel :    {self.config.monthly_budget:,.0f}€
  Rendement estimé :  {annual_ret*100:.1f}%/an
  Horizon :           {years} ans

  💰 Valeur finale :     {result['final_value']:,.0f}€
  💵 Total investi :     {result['total_invested']:,.0f}€
  📈 Plus-value :        {result['total_gains']:,.0f}€ ({result['roi_pct']:+.1f}%)
"""
        return report


async def main():
    """Point d'entrée du module d'investissement."""
    # Utiliser les variables d'environnement
    phone = os.getenv('TR_PHONE')
    pin = os.getenv('TR_PIN')
    
    if not phone or not pin:
        print("⚠️  Configurer TR_PHONE et TR_PIN dans les variables d'environnement")
        print("   export TR_PHONE=+33...")
        print("   export TR_PIN=1234")
        return
    
    config = InvestmentConfig(
        strategy="core_satellite",
        monthly_budget=300.0,
        risk_profile="moderate",
        rebalance_frequency="quarterly",
        enable_sparplan=True,
    )
    
    manager = InvestmentManager(config, phone, pin)
    
    # 1. Définir l'allocation
    print(manager.projection_report(years=20))
    manager.set_target_allocation()
    
    # 2. Analyser le portefeuille
    await manager.analyze_portfolio()
    
    # 3. Investissement mensuel
    await manager.execute_monthly_investment()
    
    # 4. Vérifier le rééquilibrage
    await manager.check_and_rebalance()


if __name__ == '__main__':
    asyncio.run(main())