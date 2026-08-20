import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import shap

# ==========================================
# 1. RIGOROUS DATA SIMULATION (The DGP)
# ==========================================
np.random.seed(42)
n_days = 1000

# Simulate Market Factors (Standardized to mean 0, std 1 for realistic scaling)
X_1 = np.random.normal(0, 1, n_days) # Equities
X_2 = np.random.normal(0, 1, n_days) # Rates
X_3 = np.random.normal(0, 1, n_days) # Credit
# VIX is skewed (log-normal or similar), spiking occasionally
X_4 = np.random.exponential(1, n_days) - 1 

# Combine into a DataFrame
factors = pd.DataFrame({'Equity': X_1, 'Rates': X_2, 'Credit': X_3, 'VIX': X_4})

# Formulate Portfolios based on the Math DGP
noise_level = 0.5
# Fund A: Long Equity, Long Rates, Long Volatility Option (Max payoff)
P_A = 1.5 * factors['Equity'] + 0.8 * factors['Rates'] + 2.0 * np.maximum(0, factors['VIX'] - 1.5) + np.random.normal(0, noise_level, n_days)

# Portfolio B (Our Bank): Higher Equity, Long Credit, SHORT Volatility (Quadratic tail risk)
P_B = 2.0 * factors['Equity'] + 1.2 * factors['Credit'] - 3.0 * (factors['VIX'] ** 2) + np.random.normal(0, noise_level, n_days)

# Away Portfolio (Hidden at other prime brokers)
P_Away = P_A - P_B

# ==========================================
# 2. RIDGE REGRESSION: Structural Beta Drift
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(factors)

ridge_A = Ridge(alpha=1.0).fit(X_scaled, P_A)
ridge_B = Ridge(alpha=1.0).fit(X_scaled, P_B)
ridge_Away = Ridge(alpha=1.0).fit(X_scaled, P_Away)

# Plot 1: Linear Beta Comparison (For Jack's Macro View)
beta_df = pd.DataFrame({
    'Fund A (Total)': ridge_A.coef_,
    'Portfolio B (Ours)': ridge_B.coef_,
    'Away Portfolio (Hidden)': ridge_Away.coef_
}, index=factors.columns)

fig, ax = plt.subplots(figsize=(10, 6))
beta_df.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black')
plt.title("Structural Linear Beta Exposure (Ridge Regression)", fontsize=14, fontweight='bold')
plt.ylabel("Beta Coefficient")
plt.axhline(0, color='black', linewidth=1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ==========================================
# 3. XGBOOST & SHAP: Non-Linear Tail Risk Attribution
# ==========================================
# Train XGBoost on Portfolio B (The toxic book at our bank)
xgb_B = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
xgb_B.fit(factors, P_B)

# Calculate SHAP values for Portfolio B
explainer = shap.Explainer(xgb_B, factors)
shap_values = explainer(factors)

# Plot 2: SHAP Summary Plot (Beeswarm) - Showing Non-linear magnitude
plt.figure(figsize=(10, 6))
plt.title("Non-Linear Risk Attribution for Portfolio B (SHAP Summary)", fontsize=14, fontweight='bold')
shap.summary_plot(shap_values, factors, show=False)
plt.tight_layout()
plt.show()

# Plot 3: SHAP Waterfall Plot for the WORST day (Stress Testing)
# Find the day with the lowest PnL in Portfolio B (The crash day)
worst_day_idx = np.argmin(P_B)

plt.figure(figsize=(10, 6))
plt.title(f"Transaction-Level Crash Attribution (Day {worst_day_idx})", fontsize=14, fontweight='bold')
shap.plots.waterfall(shap_values[worst_day_idx], show=False)
plt.tight_layout()
plt.show()
