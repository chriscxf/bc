# Soft Factor-Penalty Portfolio Optimization

Here’s a clean, fully-rigorous version of the **soft factor-penalty** formulation, with all notation defined and explicit solutions where we can get them in closed form.

---

## 1. Notation and Setup

We work in a single-period (one-day) setting; everything is for one rebalancing date.

### 1.1 Assets and Portfolio

- Number of assets (bonds):
  
  $$
  N \in \mathbb{N}.
  $$

- Portfolio weights (decision variable):
  
  $$
  w \in \mathbb{R}^N.
  $$
  
  Typically $w_i$ is the fraction of capital allocated to asset $i$.

- Budget (full-investment) constraint (optional):
  
  $$
  \mathbf{1}^\top w = 1,
  $$
  
  where $\mathbf{1} \in \mathbb{R}^N$ is the vector of ones.

### 1.2 Factor Model and Risk

- Number of risk factors:
  
  $$
  K \in \mathbb{N}.
  $$

- Exposure matrix:
  
  $$
  X \in \mathbb{R}^{N \times K},
  $$
  
  where row $i$ is the vector of factor exposures for asset $i$:
  
  $$
  X =
  \begin{bmatrix}
  - & x_1^\top & - \\
  - & x_2^\top & - \\
    & \vdots   &   \\
  - & x_N^\top & -
  \end{bmatrix}, 
  \quad x_i \in \mathbb{R}^K.
  $$

- Portfolio **factor exposure vector** (as a function of $w$):
  
  $$
  \beta(w) := X^\top w \in \mathbb{R}^K.
  $$

- Asset covariance matrix (from your BARRA model):
  
  $$
  \Sigma \in \mathbb{R}^{N \times N}, 
  \quad \Sigma \succ 0 \text{ (symmetric positive definite, in idealized theory)}.
  $$

### 1.3 Expected Returns and Risk Aversion

- Expected excess return vector (alpha):
  
  $$
  \hat{\mu} \in \mathbb{R}^N.
  $$

- Risk aversion parameter:
  
  $$
  \lambda > 0.
  $$
  
  Higher $\lambda$ means more penalty on variance.

### 1.4 Factor Target and Penalty Structure

- Target factor exposure vector (what you want your portfolio to look like in factor space):
  
  $$
  b^\ast \in \mathbb{R}^K.
  $$
  
  Examples:
  
  - $b^\ast = 0$ for factor neutrality.
  - $b^\ast = X^\top w^{\text{benchmark}}$ to match a benchmark.

- Factor penalty weight matrix:
  
  $$
  W \in \mathbb{R}^{K \times K}, 
  \quad W \succeq 0 \text{ (symmetric positive semidefinite)}.
  $$

  Special cases:

  - $W = I_K$: plain Euclidean penalty on exposure mismatch.
  - $W = \Sigma_f$: factor covariance; then the penalty is the **variance** of factor hedging error.
  - $W = \operatorname{diag}(w_1,\ldots,w_K)$ with custom weights.

- Penalty strength scalar:
  
  $$
  \gamma \ge 0,
  $$
  
  which scales how strongly we penalize deviations from $b^\ast$.

- Trading cost term (optional, may be non-smooth):
  
  $$
  TC(w, w_{\text{prev}}) \ge 0.
  $$
  
  For analytic derivation we will set $TC = 0$; in practice this term makes the problem a general QP rather than something we can solve in a single closed form.

---

## 2. Objective with Soft Factor Penalties

Ignoring trading costs for the moment, a standard **mean–variance + factor penalty** objective is:

$$
\max_{w \in \mathbb{R}^N}
\;
\underbrace{\hat{\mu}^\top w}_{\text{expected return}}
\;-\;
\underbrace{\frac{\lambda}{2} w^\top \Sigma w}_{\text{variance penalty}}
\;-\;
\underbrace{\gamma\,(\beta(w) - b^\ast)^\top W (\beta(w) - b^\ast)}_{\text{soft factor exposure penalty}}.
$$

Substituting $\beta(w) = X^\top w$, this is

$$
\max_{w \in \mathbb{R}^N}
\;
\hat{\mu}^\top w
\;-\;
\frac{\lambda}{2} w^\top \Sigma w
\;-\;
\gamma \big(X^\top w - b^\ast\big)^\top W \big(X^\top w - b^\ast\big).
$$

You can also write this as a **minimization** problem by flipping the sign:

$$
\min_{w \in \mathbb{R}^N}
\;
\frac{\lambda}{2} w^\top \Sigma w
\;+\;
\gamma \big(X^\top w - b^\ast\big)^\top W \big(X^\top w - b^\ast\big)
\;-\;
\hat{\mu}^\top w.
$$

We’ll derive the **unconstrained optimum**, then the optimum with just the **budget constraint** $\mathbf{1}^\top w = 1$. With trading costs and box constraints you typically use a numerical QP solver.

---

## 3. Unconstrained Solution (Closed Form)

We first solve

$$
\max_{w \in \mathbb{R}^N}
\;
\hat{\mu}^\top w
\;-\;
\frac{\lambda}{2} w^\top \Sigma w
\;-\;
\gamma \big(X^\top w - b^\ast\big)^\top W \big(X^\top w - b^\ast\big).
$$

### 3.1 Rewrite in Quadratic Form

Expand the factor penalty:

$$
\begin{aligned}
\big(X^\top w - b^\ast\big)^\top W \big(X^\top w - b^\ast\big)
&= (X^\top w)^\top W (X^\top w) - 2 (X^\top w)^\top W b^\ast + (b^\ast)^\top W b^\ast \\
&= w^\top X W X^\top w - 2 w^\top X W b^\ast + (b^\ast)^\top W b^\ast.
\end{aligned}
$$

Hence the objective is

$$
\begin{aligned}
J(w)
&= \hat{\mu}^\top w
- \frac{\lambda}{2} w^\top \Sigma w
- \gamma \Big[ w^\top X W X^\top w - 2 w^\top X W b^\ast + (b^\ast)^\top W b^\ast \Big].
\end{aligned}
$$

Drop the constant term $\gamma (b^\ast)^\top W b^\ast$ since it does not depend on $w$. We get

$$
\begin{aligned}
J(w)
&= \hat{\mu}^\top w
- \frac{\lambda}{2} w^\top \Sigma w
- \gamma w^\top X W X^\top w
+ 2 \gamma w^\top X W b^\ast.
\end{aligned}
$$

Group quadratic terms and linear terms:

- Quadratic term in $w$:

  $$
  -\frac{1}{2} w^\top \left( \lambda \Sigma + 2\gamma X W X^\top \right) w.
  $$

- Linear term in $w$:

  $$
  \left( \hat{\mu} + 2\gamma X W b^\ast \right)^\top w.
  $$

So we can write

$$
J(w)
=
-\frac{1}{2} w^\top A w + c^\top w + \text{constant},
$$

where

$$
A := \lambda \Sigma + 2\gamma X W X^\top \in \mathbb{R}^{N \times N},
$$

$$
c := \hat{\mu} + 2\gamma X W b^\ast \in \mathbb{R}^N.
$$

Assume $A$ is symmetric positive definite (this holds if $\Sigma \succ 0$ and $\lambda > 0$, plus mild conditions).

### 3.2 First-Order Condition

For a function of the form

$$
J(w) = -\frac{1}{2} w^\top A w + c^\top w + \text{const},
$$

the gradient is

$$
\nabla_w J(w) = -A w + c.
$$

Set the gradient to zero for the maximizer:

$$
-A w^\ast + c = 0
\quad \Rightarrow \quad
A w^\ast = c.
$$

Thus,

$$
\boxed{
w^\ast_{\text{unconstrained}} = A^{-1} c
= \left(\lambda \Sigma + 2\gamma X W X^\top \right)^{-1} \left(\hat{\mu} + 2\gamma X W b^\ast \right).
}
$$

That’s the **closed-form unconstrained solution**.

---

## 4. Solution with a Budget Constraint $\mathbf{1}^\top w = 1$

Now consider the problem:

$$
\begin{aligned}
\max_{w \in \mathbb{R}^N} \quad &
\hat{\mu}^\top w
- \frac{\lambda}{2} w^\top \Sigma w
- \gamma \big(X^\top w - b^\ast\big)^\top W \big(X^\top w - b^\ast\big) \\
\text{s.t.} \quad & \mathbf{1}^\top w = 1.
\end{aligned}
$$

We use the same $A$ and $c$:

$$
A = \lambda \Sigma + 2\gamma X W X^\top,
\quad c = \hat{\mu} + 2\gamma X W b^\ast.
$$

The objective is again

$$
J(w) = -\frac{1}{2} w^\top A w + c^\top w + \text{const}.
$$

### 4.1 Lagrangian

Introduce Lagrange multiplier $\eta \in \mathbb{R}$ for the budget constraint $\mathbf{1}^\top w = 1$:

$$
\mathcal{L}(w, \eta)
=
-\frac{1}{2} w^\top A w + c^\top w
+ \eta (\mathbf{1}^\top w - 1).
$$

### 4.2 First-Order Conditions

Take gradient with respect to $w$:

$$
\nabla_w \mathcal{L}(w, \eta) = -A w + c + \eta \mathbf{1}.
$$

Set to zero:

$$
-A w^\ast + c + \eta^\ast \mathbf{1} = 0
\quad \Rightarrow \quad
A w^\ast = c + \eta^\ast \mathbf{1}.
$$

So

$$
w^\ast = A^{-1} (c + \eta^\ast \mathbf{1}).
$$

Now impose the constraint $\mathbf{1}^\top w^\ast = 1$:

$$
\mathbf{1}^\top w^\ast
= \mathbf{1}^\top A^{-1} (c + \eta^\ast \mathbf{1})
= 1.
$$

Define

$$
u := A^{-1} c \in \mathbb{R}^N, 
\quad 
v := A^{-1} \mathbf{1} \in \mathbb{R}^N.
$$

Then

$$
w^\ast = u + \eta^\ast v,
$$

and the budget constraint becomes

$$
\mathbf{1}^\top w^\ast
= \mathbf{1}^\top u + \eta^\ast \mathbf{1}^\top v
= 1.
$$

So

$$
\eta^\ast
= \frac{1 - \mathbf{1}^\top u}{\mathbf{1}^\top v}
= \frac{1 - \mathbf{1}^\top A^{-1} c}{\mathbf{1}^\top A^{-1} \mathbf{1}}.
$$

Substitute back:

$$
\boxed{
w^\ast_{\text{budget}}
= A^{-1} c
+ \frac{1 - \mathbf{1}^\top A^{-1} c}{\mathbf{1}^\top A^{-1} \mathbf{1}} \, A^{-1} \mathbf{1},
}
$$

where

$$
A = \lambda \Sigma + 2\gamma X W X^\top,
\quad c = \hat{\mu} + 2\gamma X W b^\ast.
$$

This is the **closed-form solution** with a single linear equality constraint $\mathbf{1}^\top w = 1$.

---

## 5. Interpretation of the Solution

### 5.1 Effective Covariance and Effective Alpha

Compare to standard mean–variance:

- In classical Markowitz (no factor penalty), the matrix is just $\lambda \Sigma$, and the linear term is $\hat{\mu}$.
- With the factor penalty, you get:

  - **Effective risk matrix**:
    
    $$
    \Sigma_{\text{eff}} := \frac{1}{\lambda} A
    = \Sigma + \frac{2\gamma}{\lambda} X W X^\top.
    $$
  
  - **Effective alpha**:
    
    $$
    \hat{\mu}_{\text{eff}} := c
    = \hat{\mu} + 2\gamma X W b^\ast.
    $$

So:

- You are behaving **as if** you:
  - Increased covariance in directions spanned by the factor exposures $X$, proportional to $W$ and $\gamma$.
  - Added a drift term $2\gamma X W b^\ast$ that pulls the portfolio towards the target factor exposures.

This is why it is a **soft constraint**: the optimizer is allowed to deviate from $b^\ast$ if the expected return benefit is large enough to overcome the penalty.

### 5.2 Special Case: $b^\ast = 0$

If you want factor neutrality (target zero exposures), set $b^\ast = 0$. Then

$$
c = \hat{\mu} + 2\gamma X W b^\ast = \hat{\mu},
$$

but the risk matrix is still penalized:

$$
A = \lambda \Sigma + 2\gamma X W X^\top.
$$

So you:

- Do not add an extra “alpha” pull; you only make factor exposures “more expensive” (higher effective risk).
- The optimizer tends to avoid taking large exposures in factor directions that $W$ penalizes.

---

## 6. Trading Costs and Box Constraints

If you add:

- Trading costs $TC(w, w_{\text{prev}})$, usually convex (e.g. proportional or quadratic in $\lvert w - w_{\text{prev}} \rvert$),
- Box constraints $\underline{w} \le w \le \overline{w}$,

then the problem is still a **convex quadratic program** (QP) but:

- The closed-form $w^\ast = A^{-1} c + \dots$ no longer holds.
- You solve numerically with a QP solver, but the structure above is exactly what the solver is optimizing:
  mean–variance + soft factor penalty + costs + linear constraints.




## 7. With TC



### 1. Baseline (No Trading Costs)

We consider a single-period problem with:

- Number of assets: $N \in \mathbb{N}$.
- Portfolio weights: $w \in \mathbb{R}^N$.
- Expected excess returns: $\hat{\mu} \in \mathbb{R}^N$.
- Asset covariance matrix: $\Sigma \in \mathbb{R}^{N \times N}$, with $\Sigma \succ 0$.
- Number of factors: $K \in \mathbb{N}$.
- Exposure matrix: $X \in \mathbb{R}^{N \times K}$.
- Target factor exposure: $b^\ast \in \mathbb{R}^K$.
- Factor penalty weight matrix: $W \in \mathbb{R}^{K \times K}$, with $W \succeq 0$.
- Risk aversion: $\lambda > 0$.
- Factor penalty strength: $\gamma \ge 0$.

The portfolio factor exposure is

$$
\beta(w) := X^\top w \in \mathbb{R}^K.
$$

The **soft factor-penalty** objective (no trading costs) is

$$
J(w)
=
\hat{\mu}^\top w
- \frac{\lambda}{2} w^\top \Sigma w
- \gamma \big(X^\top w - b^\ast\big)^\top W \big(X^\top w - b^\ast\big).
$$

We can rewrite it in quadratic form as

$$
J(w)
=
-\frac{1}{2} w^\top A w + c^\top w + \text{const},
$$

where

$$
A := \lambda \Sigma + 2\gamma X W X^\top \in \mathbb{R}^{N \times N},
$$

$$
c := \hat{\mu} + 2\gamma X W b^\ast \in \mathbb{R}^N.
$$

Assuming $A$ is symmetric positive definite, the **unconstrained** maximizer is

$$
w^\ast_{\text{unconstrained}}
=
A^{-1} c.
$$

With a **budget constraint** $\mathbf{1}^\top w = 1$, where $\mathbf{1} \in \mathbb{R}^N$ is the vector of ones, the solution becomes

$$
w^\ast_{\text{budget}}
=
A^{-1} c
+ 
\frac{1 - \mathbf{1}^\top A^{-1} c}{\mathbf{1}^\top A^{-1} \mathbf{1}} \,
A^{-1} \mathbf{1}.
$$

---

### 2. Adding Quadratic Trading Costs

Let the previous-day weights be $w_{\text{prev}} \in \mathbb{R}^N$. Define a **quadratic** trading cost:

$$
TC(w, w_{\text{prev}})
=
\frac{1}{2} (w - w_{\text{prev}})^\top K (w - w_{\text{prev}}),
$$

where

- $K \in \mathbb{R}^{N \times N}$ is symmetric positive semidefinite ($K \succeq 0$).

The new objective is

$$
\tilde{J}(w)
=
\hat{\mu}^\top w
- \frac{\lambda}{2} w^\top \Sigma w
- \gamma \big(X^\top w - b^\ast\big)^\top W \big(X^\top w - b^\ast\big)
- \frac{1}{2} (w - w_{\text{prev}})^\top K (w - w_{\text{prev}}).
$$

### 2.1 Expanding the Trading Cost

Expand the last term:

$$
\begin{aligned}
(w - w_{\text{prev}})^\top K (w - w_{\text{prev}})
&= w^\top K w - 2 w^\top K w_{\text{prev}}
+ w_{\text{prev}}^\top K w_{\text{prev}}.
\end{aligned}
$$

Thus

$$
-\frac{1}{2} (w - w_{\text{prev}})^\top K (w - w_{\text{prev}})
=
-\frac{1}{2} w^\top K w
+ w^\top K w_{\text{prev}}
- \frac{1}{2} w_{\text{prev}}^\top K w_{\text{prev}}.
$$

Dropping constants that do not depend on $w$, we can write

$$
\tilde{J}(w)
=
-\frac{1}{2} w^\top A w + c^\top w
-\frac{1}{2} w^\top K w + w^\top K w_{\text{prev}} + \text{const}.
$$

Group terms:

- Quadratic term:

  $$
  -\frac{1}{2} w^\top (A + K) w.
  $$

- Linear term:

  $$
  (c + K w_{\text{prev}})^\top w.
  $$

Define

$$
\tilde{A} := A + K
= \lambda \Sigma + 2\gamma X W X^\top + K,
$$

$$
\tilde{c} := c + K w_{\text{prev}}
= \hat{\mu} + 2\gamma X W b^\ast + K w_{\text{prev}}.
$$

Then

$$
\tilde{J}(w)
=
-\frac{1}{2} w^\top \tilde{A} w + \tilde{c}^\top w + \text{const}.
$$

### 2.2 Unconstrained Optimum (Quadratic TC)

The gradient is

$$
\nabla_w \tilde{J}(w) = -\tilde{A} w + \tilde{c}.
$$

Setting this to zero,

$$
-\tilde{A} w^\ast + \tilde{c} = 0
\quad\Longrightarrow\quad
\tilde{A} w^\ast = \tilde{c}.
$$

Thus the **unconstrained** solution with quadratic trading costs is

$$
\boxed{
w^\ast_{\text{unconstrained, TC}}
=
\tilde{A}^{-1} \tilde{c}
=
\big(\lambda \Sigma + 2\gamma X W X^\top + K\big)^{-1}
\big(\hat{\mu} + 2\gamma X W b^\ast + K w_{\text{prev}}\big).
}
$$

### 2.3 Budget-Constrained Optimum (Quadratic TC)

Now add the budget constraint $\mathbf{1}^\top w = 1$.

The Lagrangian is

$$
\tilde{\mathcal{L}}(w, \eta)
=
-\frac{1}{2} w^\top \tilde{A} w + \tilde{c}^\top w
+ \eta (\mathbf{1}^\top w - 1),
$$

with Lagrange multiplier $\eta \in \mathbb{R}$.

First-order conditions:

$$
\nabla_w \tilde{\mathcal{L}}(w, \eta)
= -\tilde{A} w + \tilde{c} + \eta \mathbf{1} = 0,
$$

so

$$
\tilde{A} w^\ast = \tilde{c} + \eta^\ast \mathbf{1}.
$$

Hence

$$
w^\ast = \tilde{A}^{-1} (\tilde{c} + \eta^\ast \mathbf{1}).
$$

Impose the budget constraint:

$$
\mathbf{1}^\top w^\ast
=
\mathbf{1}^\top \tilde{A}^{-1} (\tilde{c} + \eta^\ast \mathbf{1})
= 1.
$$

Define

$$
\tilde{u} := \tilde{A}^{-1} \tilde{c}, 
\quad
\tilde{v} := \tilde{A}^{-1} \mathbf{1}.
$$

Then

$$
w^\ast = \tilde{u} + \eta^\ast \tilde{v},
$$

and the constraint becomes

$$
\mathbf{1}^\top \tilde{u} + \eta^\ast \mathbf{1}^\top \tilde{v} = 1.
$$

Solve for $\eta^\ast$:

$$
\eta^\ast
=
\frac{1 - \mathbf{1}^\top \tilde{u}}{\mathbf{1}^\top \tilde{v}}
=
\frac{1 - \mathbf{1}^\top \tilde{A}^{-1} \tilde{c}}{\mathbf{1}^\top \tilde{A}^{-1} \mathbf{1}}.
$$

Thus the **budget-constrained** solution with quadratic trading costs is

$$
\boxed{
w^\ast_{\text{budget, TC}}
=
\tilde{A}^{-1} \tilde{c}
+
\frac{1 - \mathbf{1}^\top \tilde{A}^{-1} \tilde{c}}{\mathbf{1}^\top \tilde{A}^{-1} \mathbf{1}}
\,
\tilde{A}^{-1} \mathbf{1},
}
$$

where

$$
\tilde{A} = \lambda \Sigma + 2\gamma X W X^\top + K,
\quad
\tilde{c} = \hat{\mu} + 2\gamma X W b^\ast + K w_{\text{prev}}.
$$

---

### 3. L1 / Turnover Costs (No Simple Closed Form)

If instead we use **L1** (absolute value) trading costs, e.g.

$$
TC_{\text{L1}}(w, w_{\text{prev}})
=
\sum_{i=1}^N c_i \,\big| w_i - (w_{\text{prev}})_i \big|,
$$

the optimization remains convex but becomes **non-smooth**. In the general correlated, constrained case, there is **no simple closed-form vector expression** for $w^\ast$; one usually solves it numerically (e.g. via QP or proximal gradient methods).

Only in very special cases (e.g. diagonal $\Sigma$, no factor penalty, no coupling constraints) does the problem decouple asset-by-asset and admit simple scalar “soft-thresholding” formulas.
