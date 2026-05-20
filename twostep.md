# Formal Specification

## Notation and Data

**Time index:** $t = 1, \dots, T$ business days, $T \approx 90$ (Jan 2 – May 19, 2026).

**Observables:**

- $R^{HF}_t \in \mathbb{R}$ — daily return of the hedge fund (reported, net of fees ideally gross; be explicit).
- $F_t = (F_{1,t}, \dots, F_{K,t})' \in \mathbb{R}^K$ — vector of $K$ market factor returns/changes on day $t$. Each $F_{k,t}$ is the *change* in the relevant risk driver (e.g., $\Delta y_t$ for rates in bp, $\Delta s_t$ for credit spreads in bp, $r^{eq}_t$ for equity index log-return, $\Delta \sigma_t$ for implied vol, etc.). Units must match the sensitivity definitions below.
- $S_t = (S_{1,t}, \dots, S_{K,t})' \in \mathbb{R}^K$ — vector of **end-of-day-$t$ sensitivities** of the bank-held slice of the fund's portfolio to each factor. Convention: $S_{k,t}$ is defined so that $S_{k,t} \cdot F_{k,t+1}$ is a P&L in currency units (e.g., DV01 in \$/bp times $\Delta y$ in bp; equity delta in \$ times equity return; etc.).

**Latent / unobserved:**

- $A^{bank}_t$ — market value of bank slice. $A^{HF}_t$ — total fund AUM.
- $w^{bank}_t = A^{bank}_t / A^{HF}_t \in (0,1]$ — bank's share of fund AUM (unknown, possibly drifting).
- $R^{other}_t$ — return on the non-bank slice.

## Identity

By accounting:
$$R^{HF}_t \;=\; w^{bank}_{t-1}\, R^{bank}_t \;+\; (1-w^{bank}_{t-1})\, R^{other}_t \tag{1}$$

where $R^{bank}_t = PnL^{bank}_t / A^{bank}_{t-1}$.

## Synthetic Bank-Slice P&L from Sensitivities

Define the **sensitivity-implied P&L** of the bank slice:
$$\widehat{PnL}^{bank}_t \;\equiv\; \sum_{k=1}^{K} S_{k,t-1} \, F_{k,t} \tag{2}$$

and the corresponding return proxy:
$$\widehat{R}^{bank}_t \;\equiv\; \widehat{PnL}^{bank}_t \,/\, A^{bank}_{t-1} \tag{3}$$

The true bank-slice return decomposes as:
$$R^{bank}_t \;=\; \widehat{R}^{bank}_t \;+\; \eta_t \tag{4}$$

where $\eta_t$ collects (i) higher-order terms (gamma, cross-gamma), (ii) idiosyncratic/specific risk not spanned by $F$, (iii) intraday rebalancing, (iv) carry/financing not in $F$.

## The Estimation Model

Substitute (4) into (1):

$$R^{HF}_t \;=\; w^{bank}_{t-1} \widehat{R}^{bank}_t \;+\; w^{bank}_{t-1}\eta_t \;+\; (1-w^{bank}_{t-1}) R^{other}_t$$

**Assumption A1 (constant bank share over the window):** $w^{bank}_{t-1} = w$ for $t = 1,\dots,T$.

**Assumption A2 (residual factor structure):** $R^{other}_t = \gamma' F_t + u_t$, with $u_t$ uncorrelated with $F_t$ and $\widehat{R}^{bank}_t$.

Combining, and absorbing $w \cdot \eta_t$ into an error term $\varepsilon_t$:

$$\boxed{\;R^{HF}_t \;=\; \alpha \;+\; \beta\, \widehat{R}^{bank}_t \;+\; \delta' F_t \;+\; \varepsilon_t\;} \tag{5}$$

where the structural interpretation is:

- $\beta = w$ (bank share of AUM),
- $\delta = (1-w)\gamma$ (residual factor exposure from the non-bank slice),
- $\alpha$ captures average carry/fees/specific drift.

Equivalently, since $\widehat{R}^{bank}_t = \sum_k (S_{k,t-1}/A^{bank}_{t-1}) F_{k,t}$, equation (5) can be written factor-by-factor; the coefficient on factor $k$ is $\beta \cdot S_{k,t-1}/A^{bank}_{t-1} + \delta_k$.

## Estimator (Small-Sample Regime, $T \approx 90$)

Because $K$ is moderate (say 8–15) and factors are collinear, OLS on (5) is ill-conditioned. Use a two-stage regularized estimator.

**Stage 1 — Identify the scale $\beta$:**
$$(\hat\alpha, \hat\beta) \;=\; \arg\min_{\alpha,\beta} \sum_{t=1}^{T} \big(R^{HF}_t - \alpha - \beta\,\widehat{R}^{bank}_t\big)^2 \tag{6}$$

This is a univariate regression; well-identified at $T=90$. Report $\hat\beta$, its standard error, and the $R^2_{stage1}$.

**Stage 2 — Residual factor decomposition** via Lasso (or grouped Lasso on factor blocks):

$$\hat\delta \;=\; \arg\min_{\delta} \;\;\frac{1}{T}\sum_{t=1}^{T}\Big(R^{HF}_t - \hat\alpha - \hat\beta\,\widehat{R}^{bank}_t - \delta' F_t\Big)^2 \;+\; \lambda \|\delta\|_1 \tag{7}$$

with $\lambda$ chosen by $K$-fold CV (e.g., $K=5$) on a small grid. Optionally use group Lasso where factors are pre-grouped into macro blocks $G_1,\dots,G_M$ (rates, credit, equity, FX, vol, commodities) with penalty $\lambda\sum_m \|\delta_{G_m}\|_2$.

## Decomposition (the Output)

The fitted prediction:
$$\widehat{R}^{HF}_t \;=\; \hat\alpha \;+\; \hat\beta \sum_{k=1}^K \frac{S_{k,t-1}}{A^{bank}_{t-1}} F_{k,t} \;+\; \sum_{k=1}^K \hat\delta_k F_{k,t}$$

decomposes into per-factor attribution:

$$\widehat{R}^{HF}_t \;=\; \hat\alpha \;+\; \sum_{k=1}^K \underbrace{\Big(\hat\beta \cdot \tfrac{S_{k,t-1}}{A^{bank}_{t-1}} + \hat\delta_k\Big)}_{\text{total loading on factor }k} F_{k,t} \tag{8}$$

with the split into "bank-slice-implied" $\hat\beta \cdot S_{k,t-1}/A^{bank}_{t-1}$ (low-uncertainty, varies daily with $S$) and "residual" $\hat\delta_k$ (estimated, constant over window).

## Diagnostics

D1. **Scale check:** is $\hat\beta \in (0,1]$? If $\hat\beta>1$ or $<0$, A1/A2 are violated — bank slice is not a clean scaled sample.

D2. **Stage-1 fit:** $R^2_{stage1}$ high (say $>0.5$) ⟹ bank slice is a strong proxy; proceed with confidence. Low ⟹ slice is biased or hedges live elsewhere; stop or interpret with caution.

D3. **Residual sparsity:** few non-zero $\hat\delta_k$ ⟹ slice already spans the fund's risk. Many non-zero ⟹ off-bank exposures matter.

D4. **Split-sample stability:** estimate $\hat\beta$ on Jan–Feb vs. Mar–May; large drift signals time-varying $w$ or strategy shift.

D5. **Sign coherence:** for factors with strong $S_{k,t-1}$, the sign of $\hat\delta_k$ should not flip the total loading vs. what the bank slice implies; if it does, the slice is misleading on factor $k$.

## What is and isn't identified

- **Identified:** $\beta$ (= bank AUM share under A1, A2), total loading on each factor (eq. 8), $\alpha$ (drift).
- **Not separately identified without extra info:** $w$ vs. $\eta_t$ scaling (we can't tell a small slice with big sensitivities from a big slice with small sensitivities — only the product matters); $\gamma$ separate from $(1-w)$.
- **Not identified at $T=90$:** time variation in $\beta$ or $\delta$, regime dependence, higher-order (gamma) terms.

## Implementation Checklist

1. Align units: every $S_{k,t} \cdot F_{k,t+1}$ must be in the same currency.
2. Lag sensitivities by one day: use $S_{k,t-1}$ with $F_{k,t}$.
3. Normalize: if $A^{bank}_t$ is unknown, regress on $\widehat{PnL}^{bank}_t$ directly; then $\hat\beta$ has units 1/currency and equals $w/A^{bank}$ on average.
4. Standardize $F_t$ before Lasso so $\lambda$ penalizes comparably across factors.
5. Report $\hat\beta$, $\hat\delta$, their CIs (bootstrap with $B=1000$), $R^2_{stage1}$, $R^2_{full}$, and the daily decomposition (8).