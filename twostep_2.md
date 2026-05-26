
6.1 The off-bank object (what the fund does beyond the slice)
To study what the fund does beyond your slice, strip out the bank-slice contribution. Define the off-bank return
Ot≡RtHF−β^ Rtbank  →  p    (1−w) Rtother,(10’)O_t \equiv R^{\mathrm{HF}}_t - \hat\beta\, R^{\mathrm{bank}}_t \;\xrightarrow{\;p\;}\; (1-w)\,R^{\mathrm{other}}_t,
\tag{10'}Ot​≡RtHF​−β^​Rtbank​p​(1−w)Rtother​,(10’)
the empirical counterpart of α+δ⊤Ft+εt\alpha + \boldsymbol{\delta}^\top\mathbf{F}_t + \varepsilon_t
α+δ⊤Ft​+εt​ in (5) — i.e. the off-bank slice's return scaled by its AUM share (1−w)(1-w)
(1−w). Its left side is observed; its right side is the factor attribution:
RtHF−β^ Rtbank⏟Ot (observed)=α^⏟off-bank drift+∑k=1Kδ^kFk,t⏟off-bank factor exposure+ε^t⏟idiosyncratic.(10”)\underbrace{R^{\mathrm{HF}}_t - \hat\beta\, R^{\mathrm{bank}}_t}_{O_t \text{ (observed)}} = \underbrace{\hat\alpha}_{\text{off-bank drift}} + \underbrace{\sum_{k=1}^{K}\hat\delta_k F_{k,t}}_{\text{off-bank factor exposure}} + \underbrace{\hat\varepsilon_t}_{\text{idiosyncratic}}.
\tag{10''}Ot​ (observed)RtHF​−β^​Rtbank​​​=off-bank driftα^​​+off-bank factor exposurek=1∑K​δ^k​Fk,t​​​+idiosyncraticε^t​​​.(10”)
Here δ^k→(1−w)γk\hat\delta_k \to (1-w)\gamma_k
δ^k​→(1−w)γk​ is the factor-kk
k exposure the fund carries that the slice does not; α^\hat\alpha
α^ is off-bank drift (carry, away-book skill, fees); ε^t\hat\varepsilon_t
ε^t​ is the idiosyncratic off-bank return. Two caveats: (i) OtO_t
Ot​ inherits any bias in β^\hat\beta
β^​ — an attenuated β^\hat\beta
β^​ (synthetic-slice case) leaves residual bank exposure in OtO_t
Ot​, so use actual RtbankR^{\mathrm{bank}}_t
Rtbank​; (ii) the (1−w)(1-w)
(1−w) scaling is not separately identified, so a large δ^k\hat\delta_k
δ^k​ may mean a big off-bank book with modest tilts or a small one with aggressive tilts.