\documentclass[11pt]{article}
\usepackage[letterpaper,margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{bm}
\usepackage{booktabs}
\usepackage{array}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{mathtools}
\usepackage{parskip}

\definecolor{accent}{RGB}{30,70,120}
\titleformat{\section}{\large\bfseries\color{accent}}{\thesection}{0.6em}{}
\titleformat{\subsection}{\normalsize\bfseries\color{accent}}{\thesubsection}{0.6em}{}
\titlespacing*{\section}{0pt}{14pt}{6pt}
\titlespacing*{\subsection}{0pt}{10pt}{4pt}

% Notation conventions:
%  - scalars: plain italic (e.g., R, w, alpha, beta)
%  - vectors and matrices: bold (e.g., \bm{F}, \bm{S}, \bm{\delta})
\newcommand{\R}{\mathbb{R}}
\newcommand{\RHF}{R^{\mathrm{HF}}}
\newcommand{\Rbank}{R^{\mathrm{bank}}}
\newcommand{\Rhatbank}{\widehat{R}^{\,\mathrm{bank}}}
\newcommand{\Rother}{R^{\mathrm{other}}}
\newcommand{\PnLbank}{\mathit{PnL}^{\mathrm{bank}}}
\newcommand{\PnLhatbank}{\widehat{\mathit{PnL}}^{\,\mathrm{bank}}}
\newcommand{\Abank}{A^{\mathrm{bank}}}
\newcommand{\AHF}{A^{\mathrm{HF}}}
\newcommand{\wbank}{w^{\mathrm{bank}}}
\newcommand{\plim}{\operatorname{plim}}
\newcommand{\Var}{\operatorname{Var}}
\newcommand{\Cov}{\operatorname{Cov}}
\newcommand{\T}{^{\!\top}}

\title{\vspace{-1.5em}\textbf{\color{accent}Decomposing Hedge Fund Returns Using\\ Bank Portfolio Sensitivities}\\[0.3em]\large A Two-Stage Method for Partial Holdings Information}
\author{}
\date{}

\begin{document}
\maketitle
\vspace{-3em}

\section*{Notation Conventions}
Throughout: \emph{scalars} are written in plain italic ($R$, $w$, $\alpha$); \emph{vectors and matrices} are written in \textbf{bold} ($\bm{F}$, $\bm{S}$, $\bm{\delta}$). All vectors are column vectors. The symbol ${}\T$ denotes transpose, so for two vectors $\bm{a},\bm{b}\in\R^{K}$ the product $\bm{a}\T\bm{b}=\sum_{k=1}^{K} a_k b_k$ is a scalar. The operator $\plim$ denotes the probability limit.

\section{Data and Dimensions}

\textbf{Time index.} $t = 1,\dots,T$ business days, with $T\approx 90$ (Jan--mid-May 2026, daily).

\medskip
\noindent\textbf{Observable scalars} (one number per day):
\begin{itemize}[leftmargin=1.4em,itemsep=2pt,topsep=2pt]
\item $\RHF_t\in\R$ --- daily return of the hedge fund (reported). State explicitly whether gross or net of fees.
\item $\Rbank_t\in\R$ --- realized daily return of the bank-held slice (if available as a true realized number; see \S6).
\end{itemize}

\noindent\textbf{Observable vectors} (dimension $K$ = number of market factors, e.g.\ $K\approx 8$--$15$):
\begin{itemize}[leftmargin=1.4em,itemsep=2pt,topsep=2pt]
\item $\bm{F}_t = (F_{1,t},\dots,F_{K,t})\T \in \R^{K}$ --- vector of market factor returns/changes on day $t$. Each entry $F_{k,t}$ is the \emph{change} in risk driver $k$ (e.g.\ $\Delta y_t$ for rates in bp, $\Delta s_t$ for credit spread in bp, equity index log-return, $\Delta\sigma_t$ for implied vol). Units must match the sensitivities below.
\item $\bm{S}_t = (S_{1,t},\dots,S_{K,t})\T \in \R^{K}$ --- vector of end-of-day-$t$ sensitivities of the bank slice to each factor. Convention: $S_{k,t}\cdot F_{k,t+1}$ is a P\&L in currency units (DV01 in \$/bp $\times\,\Delta y$ in bp; equity delta in \$ $\times$ equity return; etc.).
\end{itemize}

\noindent\textbf{Latent (unobserved) scalars:}
\begin{itemize}[leftmargin=1.4em,itemsep=2pt,topsep=2pt]
\item $\Abank_t\in\R$ --- market value of the bank slice; $\AHF_t\in\R$ --- total fund AUM.
\item $\wbank_t = \Abank_t/\AHF_t \in (0,1]$ --- bank's share of fund AUM (unknown, possibly drifting).
\item $\Rother_t\in\R$ --- return on the non-bank slice.
\end{itemize}

\section{Accounting Identity}
By construction the fund return is the AUM-weighted average of the two slices:
\begin{equation}
\RHF_t \;=\; \wbank_{t-1}\,\Rbank_t \;+\; \bigl(1-\wbank_{t-1}\bigr)\,\Rother_t,
\label{eq:identity}
\end{equation}
where $\Rbank_t = \PnLbank_t/\Abank_{t-1}$ is the bank-slice return.

\section{Sensitivity-Implied P\&L of the Bank Slice}
Define the \emph{synthetic} (first-order, factor-explained) P\&L using the sensitivity vector $\bm{S}_{t-1}$ and the factor vector $\bm{F}_t$:
\begin{equation}
\PnLhatbank_t \;\equiv\; \bm{S}_{t-1}\T \bm{F}_t \;=\; \sum_{k=1}^{K} S_{k,t-1}\,F_{k,t}
\qquad(\text{a scalar}),
\label{eq:synthpnl}
\end{equation}
and the corresponding synthetic return
\begin{equation}
\Rhatbank_t \;\equiv\; \PnLhatbank_t \,/\, \Abank_{t-1}.
\label{eq:synthret}
\end{equation}
The true bank-slice return differs from the synthetic one by an approximation error:
\begin{equation}
\Rbank_t \;=\; \Rhatbank_t \;+\; \eta_t,
\label{eq:eta}
\end{equation}
where the scalar $\eta_t$ collects (i) higher-order terms (gamma, cross-gamma), (ii) idiosyncratic/specific risk not spanned by $\bm{F}_t$, (iii) intraday rebalancing, and (iv) carry/financing not in $\bm{F}_t$.

\section{The Estimation Model}
Substitute \eqref{eq:eta} into \eqref{eq:identity} and impose two assumptions.

\medskip
\noindent\textbf{Assumption A1 (constant bank share over the window).} $\wbank_{t-1}=w$ for all $t$, a scalar.

\medskip
\noindent\textbf{Assumption A2 (residual factor structure).} The non-bank slice loads linearly on the same factors,
\[
\Rother_t \;=\; \bm{\gamma}\T \bm{F}_t \;+\; u_t,
\qquad \bm{\gamma}\in\R^{K},\quad u_t\in\R,
\]
with $u_t$ uncorrelated with $\bm{F}_t$ and $\Rhatbank_t$. Here $\bm{\gamma}=(\gamma_1,\dots,\gamma_K)\T$ is the vector of \emph{off-bank} factor loadings.

\medskip
Combining and absorbing $w\,\eta_t + (1-w)u_t$ into a scalar error $\varepsilon_t$ gives the estimated regression:
\begin{equation}
\boxed{\;\RHF_t \;=\; \alpha \;+\; \beta\,\Rhatbank_t \;+\; \bm{\delta}\T\bm{F}_t \;+\; \varepsilon_t\;}
\label{eq:model}
\end{equation}
with the structural interpretation
\begin{equation}
\beta = w \quad(\text{scalar, bank AUM share}),\qquad
\bm{\delta}=(1-w)\,\bm{\gamma}\in\R^{K}\quad(\text{residual factor loadings}).
\label{eq:structural}
\end{equation}
The scalar $\alpha$ captures average carry, fees, and specific drift. Note $\bm{\delta}$ is the vector actually estimated; $\bm{\gamma}$ is the structural object it represents, and $(1-w)$ cannot be separated from $\bm{\gamma}$ without extra information.

\section{Estimator (Small-Sample Regime, $T\approx 90$)}
Because $K$ is moderate and the entries of $\bm{F}_t$ are collinear, OLS on \eqref{eq:model} is ill-conditioned. Use a two-stage regularized procedure.

\subsection{Stage 0 --- Reconciliation / span check (requires actual $\Rbank_t$)}
Regress the actual slice return on the synthetic one:
\begin{equation}
\Rbank_t \;=\; a \;+\; b\,\Rhatbank_t \;+\; \eta_t .
\label{eq:stage0}
\end{equation}
If $b\approx 1$ with high $R^2$, the sensitivity vector $\bm{S}_{t-1}$ spans the slice well and the factor decomposition is trustworthy. If $b$ is far from $1$ or $R^2$ is low, $\eta_t$ is large and the attribution will be incomplete.

\subsection{Stage 1 --- Identify the scale $\beta$ (one parameter)}
Estimate the single scalar $\beta$ by univariate least squares. \textbf{Use the actual slice return $\Rbank_t$ as the regressor when available} (it removes the errors-in-variables attenuation that arises from using $\Rhatbank_t$):
\begin{equation}
(\hat\alpha,\hat\beta) \;=\; \arg\min_{\alpha,\beta}\;\sum_{t=1}^{T}\bigl(\RHF_t - \alpha - \beta\,\Rbank_t\bigr)^2 .
\label{eq:stage1}
\end{equation}
If only the synthetic return is available, replacing $\Rbank_t$ by $\Rhatbank_t$ yields
\[
\plim\hat\beta \;=\; \beta\cdot\frac{\Var(\Rhatbank)}{\Var(\Rhatbank)+\Var(\eta)} \;<\; \beta
\quad(\text{attenuation toward zero}).
\]
Report $\hat\beta$, its standard error, and $R^2_{\text{stage 1}}$.

\subsection{Stage 2 --- Residual factor decomposition (regularized)}
Holding $\hat\alpha,\hat\beta$ fixed, estimate the loading vector $\bm{\delta}\in\R^K$ by Lasso (or group Lasso on factor blocks):
\begin{equation}
\hat{\bm{\delta}} \;=\; \arg\min_{\bm{\delta}\in\R^{K}}\;
\frac{1}{T}\sum_{t=1}^{T}\Bigl(\RHF_t - \hat\alpha - \hat\beta\,\Rhatbank_t - \bm{\delta}\T\bm{F}_t\Bigr)^2
\;+\; \lambda\,\lVert\bm{\delta}\rVert_1 ,
\label{eq:stage2}
\end{equation}
with the scalar penalty $\lambda\ge 0$ chosen by $5$-fold cross-validation on a small grid. With factors pre-grouped into $M$ macro blocks $G_1,\dots,G_M$ (rates, credit, equity, FX, vol, commodities), use the group penalty $\lambda\sum_{m=1}^{M}\lVert\bm{\delta}_{G_m}\rVert_2$, where $\bm{\delta}_{G_m}$ is the sub-vector of $\bm{\delta}$ for block $m$.

\section{Decomposition (the Output)}
Writing $\Rhatbank_t = \sum_{k} (S_{k,t-1}/\Abank_{t-1})\,F_{k,t}$, the fitted return decomposes into per-factor contributions:
\begin{equation}
\widehat{\RHF}_t \;=\; \hat\alpha \;+\; \sum_{k=1}^{K}
\underbrace{\Bigl(\hat\beta\,\tfrac{S_{k,t-1}}{\Abank_{t-1}} + \hat\delta_k\Bigr)}_{\text{total loading on factor }k}\,F_{k,t}.
\label{eq:decomp}
\end{equation}
Each factor's loading splits into a \emph{bank-slice-implied} part $\hat\beta\,S_{k,t-1}/\Abank_{t-1}$ (low-uncertainty, varies daily with $\bm{S}_{t-1}$) and a \emph{residual} part $\hat\delta_k$ (estimated, constant over the window). When working from actual $\Rbank_t$, present the unexplained piece $\eta_t$ as an explicit ``specific/gamma/carry'' line so the attribution adds up:
\[
\Rbank_t \;=\; \underbrace{\sum_{k=1}^{K}\tfrac{S_{k,t-1}}{\Abank_{t-1}}\,F_{k,t}}_{\text{factor-explained}} \;+\; \underbrace{\eta_t}_{\text{specific / gamma / carry}}.
\]

\subsection{Actual vs.\ synthetic slice: effect on the decomposition}
The factor pieces $S_{k,t-1}/\Abank_{t-1}$ in \eqref{eq:decomp} come from the sensitivity vector $\bm{S}_{t-1}$ regardless of which slice return enters Stage~1. \emph{What changes is the scalar $\hat\beta$ that multiplies them, and where the slice's non-factor P\&L $\eta_t$ is booked.} Let $\kappa\equiv\Var(\Rbank)/[\Var(\Rbank)+\Var(\eta)]\in(0,1]$ denote the reliability ratio (the signal share of the synthetic proxy).

\medskip
\noindent\textbf{(a) Using the actual slice return $\Rbank_t$ in Stage~1.} The scale is consistent, $\plim\hat\beta = w$, so each bank-slice-implied loading is correctly scaled:
\begin{equation}
\text{loading}^{\text{actual}}_k \;=\; \hat\beta\,\frac{S_{k,t-1}}{\Abank_{t-1}} \;+\; \hat\delta_k,
\qquad \plim\hat\beta = w.
\label{eq:loadingactual}
\end{equation}
Because $\Rbank_t = \Rhatbank_t + \eta_t$ \emph{contains} $\eta_t$ while the synthetic factor pieces do not, $\eta_t$ must be carried as an explicit residual line; the decomposition is then factor-explained (correctly scaled) plus a genuine, visible specific/gamma/carry bucket.

\medskip
\noindent\textbf{(b) Using the synthetic slice return $\Rhatbank_t$ in Stage~1.} The errors-in-variables attenuation gives $\plim\hat\beta = w\,\kappa < w$, so \emph{every} bank-slice-implied loading is uniformly shrunk by the same factor $\kappa$:
\begin{equation}
\text{loading}^{\text{synth}}_k \;=\; \hat\beta\,\frac{S_{k,t-1}}{\Abank_{t-1}} \;+\; \hat\delta_k^{\,\prime},
\qquad \plim\hat\beta = w\,\kappa.
\label{eq:loadingsynth}
\end{equation}
There is no $\eta_t$ to book (it was excluded from the regressor), so the specific bucket is empty \emph{by construction}, not because the fund lacks specific risk. The lost exposure does not vanish; it is reabsorbed into the residual factor loadings and the intercept. In the multivariate model this is the smearing result: writing $\sigma^2_{\Rhatbank\mid\bm{F}}$ for the variance of the synthetic slice return after partialling out $\bm{F}_t$,
\begin{equation}
\plim\hat\beta = w\cdot\frac{\sigma^2_{\Rhatbank\mid\bm{F}}}{\sigma^2_{\Rhatbank\mid\bm{F}}+\Var(\eta)},
\qquad
\plim\hat{\bm{\delta}}^{\,\prime} = \bm{\delta} + \underbrace{(w-\plim\hat\beta)\,\bm{\pi}}_{\text{absorbed slice exposure}},
\label{eq:smear}
\end{equation}
where $\bm{\pi}\in\R^{K}$ are the coefficients of projecting the bank-slice factor pieces onto $\bm{F}_t$. The attenuated portion of the slice's true exposure, $(w-\plim\hat\beta)$, is thus pushed into $\hat{\bm{\delta}}^{\,\prime}$ (and any unprojected remainder into $\hat\alpha$), contaminating exactly the residual-exposure and alpha terms one wants to interpret.

\medskip
\noindent\textbf{Net effect.}
\begin{center}
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{>{\raggedright\arraybackslash}p{0.30\textwidth} >{\raggedright\arraybackslash}p{0.30\textwidth} >{\raggedright\arraybackslash}p{0.30\textwidth}}
\toprule
\textbf{Aspect} & \textbf{Actual $\Rbank_t$} & \textbf{Synthetic $\Rhatbank_t$} \\
\midrule
Scale $\hat\beta$ & consistent ($\to w$) & attenuated ($\to w\kappa$, low) \\
Bank-slice factor loadings & correctly scaled & uniformly shrunk by $\kappa$ \\
Slice non-factor P\&L $\eta_t$ & explicit residual bucket & absent; leaks into $\hat{\bm{\delta}}$, $\hat\alpha$ \\
Multivariate side effect & none & $\hat{\bm{\delta}}$, $\hat\alpha$ contaminated \\
Net on decomposition & clean attribution, honest residual & factor attribution understated, alpha overstated \\
\bottomrule
\end{tabular}
\end{center}

\noindent In one line: \emph{the actual slice return attributes more of the fund's return to correctly-scaled factor exposure and isolates a true specific/gamma/carry residual; the synthetic slice return shrinks the factor attribution toward zero and pushes the lost exposure into the alpha and residual-factor terms, overstating apparent skill.} Hence use $\Rbank_t$ for Stage~1 whenever available, reserving $\Rhatbank_t$ for the per-factor split (\S6) and the span check (Stage~0).

\section{Diagnostics}
\begin{description}[leftmargin=2.6em,style=nextline,itemsep=2pt,topsep=2pt]
\item[D1 (Scale check).] Is $\hat\beta\in(0,1]$? Values $>1$ or $<0$ signal that A1/A2 fail --- the bank slice is not a clean scaled sample.
\item[D2 (Stage-1 fit).] $R^2_{\text{stage 1}}$ high ($>0.5$) $\Rightarrow$ strong proxy; low $\Rightarrow$ biased slice or hedges held elsewhere.
\item[D3 (Residual sparsity).] Few non-zero $\hat\delta_k$ $\Rightarrow$ the slice already spans the fund's risk; many non-zero $\Rightarrow$ off-bank exposures matter.
\item[D4 (Split-sample stability).] Estimate $\hat\beta$ on Jan--Feb vs.\ Mar--May; large drift signals time-varying $w$ or a strategy shift.
\item[D5 (Sign coherence).] For factors with large $S_{k,t-1}$, check that $\hat\delta_k$ does not flip the sign of the total loading in \eqref{eq:decomp}; if it does, the slice is misleading on factor $k$.
\end{description}

\section{What Is and Is Not Identified}
\begin{itemize}[leftmargin=1.4em,itemsep=2pt,topsep=2pt]
\item \textbf{Identified:} $\beta\,(=w$ under A1, A2$)$; the total loading on each factor \eqref{eq:decomp}; the drift $\alpha$.
\item \textbf{Not separately identified:} $w$ vs.\ the scaling of $\eta_t$ (only the product enters); $\bm{\gamma}$ apart from $(1-w)$.
\item \textbf{Not identified at $T\approx 90$:} time variation in $\beta$ or $\bm{\delta}$, regime dependence, higher-order (gamma) terms.
\end{itemize}

\section{Implementation Checklist}
\begin{enumerate}[leftmargin=1.6em,itemsep=2pt,topsep=2pt]
\item Align units: every product $S_{k,t}\cdot F_{k,t+1}$ must be in the same currency.
\item Lag sensitivities by one day: pair $\bm{S}_{t-1}$ with $\bm{F}_t$, never same-day with same-day.
\item If $\Abank_t$ is unknown, regress on $\PnLhatbank_t$ directly; then $\hat\beta$ has units $1/\text{currency}$ and equals $w/\Abank$ on average.
\item Standardize the entries of $\bm{F}_t$ before Lasso so $\lambda$ penalizes loadings comparably.
\item Prefer actual $\Rbank_t$ for Stage 1 (no attenuation), but verify it is a true realized return on the \emph{same} slice with valuation timestamps aligned to the fund's NAV strike.
\item Report $\hat\beta$, $\hat{\bm{\delta}}$, and bootstrap confidence intervals ($B=1000$), plus $R^2_{\text{stage 1}}$, $R^2_{\text{full}}$, and the daily decomposition \eqref{eq:decomp}. With $T\approx 90$ the intervals will be wide --- that is the honest answer.
\end{enumerate}

\end{document}