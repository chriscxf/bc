# Decomposing Hedge Fund Returns Using Bank Portfolio Sensitivities

**A Two-Stage Method for Partial Holdings Information**

---

## Notation Conventions

Throughout: *scalars* are written in plain italic ($R$, $w$, $\alpha$); *vectors and matrices* are written in **bold** ($\mathbf{F}$, $\mathbf{S}$, $\boldsymbol{\delta}$). All vectors are column vectors. The symbol $\top$ denotes transpose, so for two vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^{K}$ the product $\mathbf{a}^{\top}\mathbf{b} = \sum_{k=1}^{K} a_k b_k$ is a scalar. The operator $\operatorname{plim}$ denotes the probability limit.

---

## 1. Data and Dimensions

**Time index.** $t = 1, \dots, T$ business days, with $T \approx 90$ (Jan–mid-May 2026, daily).

**Observable scalars** (one number per day):

- $R^{\mathrm{HF}}_t \in \mathbb{R}$ — daily return of the hedge fund (reported). State explicitly whether gross or net of fees.
- $R^{\mathrm{bank}}_t \in \mathbb{R}$ — realized daily return of the bank-held slice (if available as a true realized number; see §6).

**Observable vectors** (dimension $K$ = number of market factors, e.g. $K \approx 8$–$15$):

- $\mathbf{F}_t = (F_{1,t}, \dots, F_{K,t})^{\top} \in \mathbb{R}^{K}$ — vector of market factor returns/changes on day $t$. Each entry $F_{k,t}$ is the *change* in risk driver $k$ (e.g. $\Delta y_t$ for rates in bp, $\Delta s_t$ for credit spread in bp, equity index log-return, $\Delta\sigma_t$ for implied vol). Units must match the sensitivities below.
- $\mathbf{S}_t = (S_{1,t}, \dots, S_{K,t})^{\top} \in \mathbb{R}^{K}$ — vector of end-of-day-$t$ sensitivities of the bank slice to each factor. Convention: $S_{k,t} \cdot F_{k,t+1}$ is a P&L in currency units (DV01 in \$/bp $\times\, \Delta y$ in bp; equity delta in \$ $\times$ equity return; etc.).

**Latent (unobserved) scalars:**

- $A^{\mathrm{bank}}_t \in \mathbb{R}$ — market value of the bank slice; $A^{\mathrm{HF}}_t \in \mathbb{R}$ — total fund AUM.
- $w^{\mathrm{bank}}_t = A^{\mathrm{bank}}_t / A^{\mathrm{HF}}_t \in (0, 1]$ — bank's share of fund AUM (unknown, possibly drifting).
- $R^{\mathrm{other}}_t \in \mathbb{R}$ — return on the non-bank slice.

---

## 2. Accounting Identity

By construction the fund return is the AUM-weighted average of the two slices:

$$
R^{\mathrm{HF}}_t = w^{\mathrm{bank}}_{t-1}\, R^{\mathrm{bank}}_t + \bigl(1 - w^{\mathrm{bank}}_{t-1}\bigr)\, R^{\mathrm{other}}_t,
\tag{1}
$$

where $R^{\mathrm{bank}}_t = \mathit{PnL}^{\mathrm{bank}}_t / A^{\mathrm{bank}}_{t-1}$ is the bank-slice return.

---

## 3. Sensitivity-Implied P&L of the Bank Slice

Define the *synthetic* (first-order, factor-explained) P&L using the sensitivity vector $\mathbf{S}_{t-1}$ and the factor vector $\mathbf{F}_t$:

$$
\widehat{\mathit{PnL}}^{\,\mathrm{bank}}_t \equiv \mathbf{S}_{t-1}^{\top}\mathbf{F}_t = \sum_{k=1}^{K} S_{k,t-1}\, F_{k,t} \qquad (\text{a scalar}),
\tag{2}
$$

and the corresponding synthetic return

$$
\widehat{R}^{\,\mathrm{bank}}_t \equiv \widehat{\mathit{PnL}}^{\,\mathrm{bank}}_t \,/\, A^{\mathrm{bank}}_{t-1}.
\tag{3}
$$

The true bank-slice return differs from the synthetic one by an approximation error:

$$
R^{\mathrm{bank}}_t = \widehat{R}^{\,\mathrm{bank}}_t + \eta_t,
\tag{4}
$$

where the scalar $\eta_t$ collects (i) higher-order terms (gamma, cross-gamma), (ii) idiosyncratic/specific risk not spanned by $\mathbf{F}_t$, (iii) intraday rebalancing, and (iv) carry/financing not in $\mathbf{F}_t$.

---

## 4. The Estimation Model

Substitute (4) into (1) and impose two assumptions.

**Assumption A1 (constant bank share over the window).** $w^{\mathrm{bank}}_{t-1} = w$ for all $t$, a scalar.

**Assumption A2 (residual factor structure).** The non-bank slice loads linearly on the same factors,

$$
R^{\mathrm{other}}_t = \boldsymbol{\gamma}^{\top}\mathbf{F}_t + u_t, \qquad \boldsymbol{\gamma} \in \mathbb{R}^{K},\quad u_t \in \mathbb{R},
$$

with $u_t$ uncorrelated with $\mathbf{F}_t$ and $\widehat{R}^{\,\mathrm{bank}}_t$. Here $\boldsymbol{\gamma} = (\gamma_1, \dots, \gamma_K)^{\top}$ is the vector of *off-bank* factor loadings.

Combining and absorbing $w\,\eta_t + (1-w)u_t$ into a scalar error $\varepsilon_t$ gives the estimated regression:

$$
R^{\mathrm{HF}}_t = \alpha + \beta\, \widehat{R}^{\,\mathrm{bank}}_t + \boldsymbol{\delta}^{\top}\mathbf{F}_t + \varepsilon_t
\tag{5}
$$

with the structural interpretation

$$
\beta = w \quad (\text{scalar, bank AUM share}), \qquad \boldsymbol{\delta} = (1-w)\,\boldsymbol{\gamma} \in \mathbb{R}^{K} \quad (\text{residual factor loadings}).
\tag{6}
$$

The scalar $\alpha$ captures average carry, fees, and specific drift. Note $\boldsymbol{\delta}$ is the vector actually estimated; $\boldsymbol{\gamma}$ is the structural object it represents, and $(1-w)$ cannot be separated from $\boldsymbol{\gamma}$ without extra information.

---

## 5. Estimator (Small-Sample Regime, $T \approx 90$)

Because $K$ is moderate and the entries of $\mathbf{F}_t$ are collinear, OLS on (5) is ill-conditioned. Use a two-stage regularized procedure.

### 5.1 Stage 0 — Reconciliation / span check (requires actual $R^{\mathrm{bank}}_t$)

Regress the actual slice return on the synthetic one:

$$
R^{\mathrm{bank}}_t = a + b\, \widehat{R}^{\,\mathrm{bank}}_t + \eta_t.
\tag{7}
$$

If $b \approx 1$ with high $R^2$, the sensitivity vector $\mathbf{S}_{t-1}$ spans the slice well and the factor decomposition is trustworthy. If $b$ is far from $1$ or $R^2$ is low, $\eta_t$ is large and the attribution will be incomplete.

### 5.2 Stage 1 — Identify the scale $\beta$ (one parameter)

Estimate the single scalar $\beta$ by univariate least squares. **Use the actual slice return $R^{\mathrm{bank}}_t$ as the regressor when available** (it removes the errors-in-variables attenuation that arises from using $\widehat{R}^{\,\mathrm{bank}}_t$):

$$
(\hat\alpha, \hat\beta) = \arg\min_{\alpha, \beta}\; \sum_{t=1}^{T}\bigl(R^{\mathrm{HF}}_t - \alpha - \beta\, R^{\mathrm{bank}}_t\bigr)^2.
\tag{8}
$$

If only the synthetic return is available, replacing $R^{\mathrm{bank}}_t$ by $\widehat{R}^{\,\mathrm{bank}}_t$ yields

$$
\operatorname{plim}\hat\beta = \beta \cdot \frac{\operatorname{Var}(\widehat{R}^{\,\mathrm{bank}})}{\operatorname{Var}(\widehat{R}^{\,\mathrm{bank}}) + \operatorname{Var}(\eta)} < \beta \quad (\text{attenuation toward zero}).
$$

Report $\hat\beta$, its standard error, and $R^2_{\text{stage 1}}$.

### 5.3 Stage 2 — Residual factor decomposition (regularized)

Holding $\hat\alpha, \hat\beta$ fixed, estimate the loading vector $\boldsymbol{\delta} \in \mathbb{R}^K$ by Lasso (or group Lasso on factor blocks):

$$
\hat{\boldsymbol{\delta}} = \arg\min_{\boldsymbol{\delta} \in \mathbb{R}^{K}}\; \frac{1}{T}\sum_{t=1}^{T}\Bigl(R^{\mathrm{HF}}_t - \hat\alpha - \hat\beta\, \widehat{R}^{\,\mathrm{bank}}_t - \boldsymbol{\delta}^{\top}\mathbf{F}_t\Bigr)^2 + \lambda\,\lVert\boldsymbol{\delta}\rVert_1,
\tag{9}
$$

with the scalar penalty $\lambda \ge 0$ chosen by 5-fold cross-validation on a small grid. With factors pre-grouped into $M$ macro blocks $G_1, \dots, G_M$ (rates, credit, equity, FX, vol, commodities), use the group penalty $\lambda\sum_{m=1}^{M}\lVert\boldsymbol{\delta}_{G_m}\rVert_2$, where $\boldsymbol{\delta}_{G_m}$ is the sub-vector of $\boldsymbol{\delta}$ for block $m$.

---

## 6. Decomposition (the Output)

Writing $\widehat{R}^{\,\mathrm{bank}}_t = \sum_{k} (S_{k,t-1}/A^{\mathrm{bank}}_{t-1})\, F_{k,t}$, the fitted return decomposes into per-factor contributions:

$$
\widehat{R}^{\mathrm{HF}}_t = \hat\alpha + \sum_{k=1}^{K} \Bigl(\hat\beta\, \tfrac{S_{k,t-1}}{A^{\mathrm{bank}}_{t-1}} + \hat\delta_k\Bigr) F_{k,t}.
\tag{10}
$$

Here the term $\hat\beta\, S_{k,t-1}/A^{\mathrm{bank}}_{t-1} + \hat\delta_k$ is the **total loading on factor $k$**. Each factor's loading splits into a *bank-slice-implied* part $\hat\beta\, S_{k,t-1}/A^{\mathrm{bank}}_{t-1}$ (low-uncertainty, varies daily with $\mathbf{S}_{t-1}$) and a *residual* part $\hat\delta_k$ (estimated, constant over the window). When working from actual $R^{\mathrm{bank}}_t$, present the unexplained piece $\eta_t$ as an explicit "specific/gamma/carry" line so the attribution adds up:

$$
R^{\mathrm{bank}}_t = \underbrace{\sum_{k=1}^{K}\tfrac{S_{k,t-1}}{A^{\mathrm{bank}}_{t-1}}\, F_{k,t}}_{\text{factor-explained}} + \underbrace{\eta_t}_{\text{specific / gamma / carry}}.
$$

---

## 7. Diagnostics

- **D1 (Scale check).** Is $\hat\beta \in (0, 1]$? Values $> 1$ or $< 0$ signal that A1/A2 fail — the bank slice is not a clean scaled sample.
- **D2 (Stage-1 fit).** $R^2_{\text{stage 1}}$ high ($> 0.5$) $\Rightarrow$ strong proxy; low $\Rightarrow$ biased slice or hedges held elsewhere.
- **D3 (Residual sparsity).** Few non-zero $\hat\delta_k$ $\Rightarrow$ the slice already spans the fund's risk; many non-zero $\Rightarrow$ off-bank exposures matter.
- **D4 (Split-sample stability).** Estimate $\hat\beta$ on Jan–Feb vs. Mar–May; large drift signals time-varying $w$ or a strategy shift.
- **D5 (Sign coherence).** For factors with large $S_{k,t-1}$, check that $\hat\delta_k$ does not flip the sign of the total loading in (10); if it does, the slice is misleading on factor $k$.

---

## 8. What Is and Is Not Identified

- **Identified:** $\beta$ ($= w$ under A1, A2); the total loading on each factor (10); the drift $\alpha$.
- **Not separately identified:** $w$ vs. the scaling of $\eta_t$ (only the product enters); $\boldsymbol{\gamma}$ apart from $(1-w)$.
- **Not identified at $T \approx 90$:** time variation in $\beta$ or $\boldsymbol{\delta}$, regime dependence, higher-order (gamma) terms.

---

## 9. Implementation Checklist

1. Align units: every product $S_{k,t} \cdot F_{k,t+1}$ must be in the same currency.
2. Lag sensitivities by one day: pair $\mathbf{S}_{t-1}$ with $\mathbf{F}_t$, never same-day with same-day.
3. If $A^{\mathrm{bank}}_t$ is unknown, regress on $\widehat{\mathit{PnL}}^{\,\mathrm{bank}}_t$ directly; then $\hat\beta$ has units $1/\text{currency}$ and equals $w/A^{\mathrm{bank}}$ on average.
4. Standardize the entries of $\mathbf{F}_t$ before Lasso so $\lambda$ penalizes loadings comparably.
5. Prefer actual $R^{\mathrm{bank}}_t$ for Stage 1 (no attenuation), but verify it is a true realized return on the *same* slice with valuation timestamps aligned to the fund's NAV strike.
6. Report $\hat\beta$, $\hat{\boldsymbol{\delta}}$, and bootstrap confidence intervals ($B = 1000$), plus $R^2_{\text{stage 1}}$, $R^2_{\text{full}}$, and the daily decomposition (10). With $T \approx 90$ the intervals will be wide — that is the honest answer.