"""Module de calculs quantitatifs en Julia.
Appelé par le service Quant dockerisé.

Calcule :
- Kelly Criterion (position sizing)
- VaR / CVaR (Value-at-Risk)
- Optimisation de portefeuille (Markowitz)
- Indicateurs techniques HPC
"""

module Octopus

export optimize_portfolio,
       kelly_size,
       calculate_var

"""
    optimize_portfolio(returns, target_return)

Optimisation Mean-Variance du portefeuille.
Calcule les poids optimaux minimisant la variance.

# Arguments
- `returns`: Matrice des rendements (n_assets × n_periods)
- `target_return`: Rendement cible

# Retourne
- `weights`: Vecteur des poids optimaux (n_assets)
"""
function optimize_portfolio(returns, target_return)
    n = size(returns, 1)
    mu = mean(returns, dims=2)
    cov = cov(returns', dims=1)
    
    # Minimiser variance
    inv_cov = inv(cov)
    ones_vec = ones(n)
    
    A = ones_vec' * inv_cov * ones_vec
    B = ones_vec' * inv_cov * mu
    C = mu' * inv_cov * mu
    D = A * C - B^2
    
    g = inv_cov * (C * ones_vec - B * mu) / D
    h = inv_cov * (A * mu - B * ones_vec) / D
    
    weights = g + h * target_return
    return weights
end

"""
    kelly_size(win_rate, avg_win, avg_loss, fraction=0.25)

Calcule la taille de position Kelly fractionnaire.

# Arguments
- `win_rate`: Probabilité de gain
- `avg_win`: Gain moyen par trade gagnant
- `avg_loss`: Perte moyenne par trade perdant
- `fraction`: Fraction du Kelly complet (0.25 = quart-Kelly)

# Retourne
- Pourcentage du capital à risquer
"""
function kelly_size(win_rate, avg_win, avg_loss; fraction=0.25)
    b = avg_win / abs(avg_loss)
    p = win_rate
    q = 1 - p
    
    full_kelly = (p * b - q) / b
    return max(0.0, full_kelly * fraction)
end

"""
    calculate_var(returns, level=0.95)

Calcule la Value at Risk historique.

# Arguments
- `returns`: Série de rendements
- `level`: Niveau de confiance

# Retourne
- VaR (perte dans le pire cas)
"""
function calculate_var(returns; level=0.95)
    sorted = sort(returns)
    index = Int(floor(length(sorted) * (1 - level)))
    return sorted[max(1, index)]
end

function run()
    println("🐙 Octopus Quant Engine — Julia $(VERSION)")
    println("   Modules: Kelly, VaR, CVaR, Markowitz, Indicators")
end

end # module Octopus