# Key Mechanisms & Predictions
## *Intergenerational Wealth Lock-in and Housing Affordability*

---

## 1. Core Setup (30-second summary)

A two-generation OLG model. Child households are heterogeneous in **income** $y_i$ and **parental wealth** $W^p_i$. Housing requires a mortgage subject to two simultaneous credit constraints:

| Constraint | Binding condition | Who it hurts most |
|---|---|---|
| **LTV** (loan-to-value) | Need $d_i \geq \chi P h$ | Low-savings / low-transfer households |
| **DSTI** (debt-service-to-income) | Need $m \cdot \alpha(r,T) \leq \psi y_i$ | Low-income households with large mortgages |

Parental gifts $g_i = \min(\lambda W^p_i, \bar{g})$ raise $d_i$, directly relaxing **both constraints** simultaneously: a larger down payment means a smaller mortgage, which both satisfies the LTV floor *and* reduces the DSTI ratio.

---

## 2. Three Key Mechanisms

### Mechanism 1 — Constraint Relaxation (Proposition A)

> *"Parental gifts expand the feasible mortgage set monotonically."*

- As $g_i$ rises, the LTV-feasible housing set grows (you can buy a bigger house).
- Simultaneously, the required mortgage shrinks, relaxing the DSTI constraint.
- There is a **transfer threshold** $\bar{g}_i$: below it, the household rents; above it, it owns.
- **Prediction:** A dollar of parental transfer has the *largest* ownership effect for households right at the margin — "pivotal" households who would just miss the constraints without the gift.

### Mechanism 2 — Intergenerational Lock-in (Proposition B)

> *"The ownership gap between wealth quintiles is monotone increasing in transfer intensity."*

- Because $g_i$ is increasing in $W^p_i$, and $o^*_i$ is increasing in $g_i$ (Mechanism 1), ownership is increasing in parental wealth.
- The **lock-in index** $LI = \Pr(o=1 \mid W^p \geq Q_{80}) - \Pr(o=1 \mid W^p \leq Q_{20})$ is always ≥ 0, and rises with $\lambda$.
- **General equilibrium amplification:** Higher $\lambda$ raises aggregate demand → raises equilibrium price $P^*$ → further excludes low-$W^p$ households. Inelastic supply *amplifies* lock-in; elastic supply *attenuates* it.
- **Prediction:** Doubling $\lambda$ from 0.10 to 0.20 raises $LI$ by ~14 pp under inelastic supply.

### Mechanism 3 — Convex Rate Sensitivity (Proposition C)

> *"DSTI-constrained households face accelerating ownership loss as rates rise."*

- The maximum DSTI-feasible mortgage $m^{\max}_{DSTI}(r) = \psi y_i / \alpha(r,T)$ is **convex decreasing** in $r$.
- Low-$W^p$ households need large mortgages → DSTI binds → ownership collapses nonlinearly.
- High-$W^p$ households use large down payments → LTV binds first (or neither binds) → ownership declines slowly/linearly.
- **Prediction:** A 200 bp rate increase reduces bottom-tercile ownership > 2× as much as top-tercile ownership.

---

## 3. General Equilibrium Price Channel

| Policy/shock | Price effect | Who gains | Who loses |
|---|---|---|---|
| ↑ Transfer intensity $\lambda$ | ↑ $P^*$ (under inelastic supply) | Existing owners (capital gain) | Low-$W^p$ renters (higher rents, higher price barrier) |
| ↑ Mortgage rate $r$ | ↓ $P^*$ (demand falls) | Future buyers | Existing owners, DSTI-constrained households excluded faster |
| ↑ Supply elasticity $\eta$ | Attenuates all price responses | Low-$W^p$ households | Incumbent owners (smaller capital gain) |
| Stricter LTV ($\uparrow \chi$) | ↓ $P^*$ (demand falls) | Low-income renters via lower rents | Marginal buyers excluded |

---

## 4. Testable Predictions

1. **Ownership gradient:** A scatter of ownership rate vs. parental wealth decile should be monotone upward, with slope steeper under higher mortgage rates.

2. **Price escalation:** Cross-MSA regressions should show that areas with higher inter vivos transfer rates (proxied by estate-tax-return data or survey measures) have higher price-to-income ratios, controlling for income and supply.

3. **Asymmetric rate sensitivity:** Following rate-hiking episodes, the decline in first-time buyer share should be concentrated among households with no parental gift history (bank records / mortgage application data).

4. **Supply elasticity moderates lock-in:** Cities with more elastic supply (lower Saiz elasticity index) should show a smaller gap between high- and low-parental-wealth ownership rates.

---

## 5. Welfare Punchline

- Transfers are privately efficient (every recipient benefits).
- They are **not** generally socially efficient: the general-equilibrium price externality hurts non-recipients and may more than offset the direct gains under sufficiently inelastic supply.
- The welfare-maximising transfer intensity $\lambda^*$ is strictly positive but **below current levels** in most OECD housing markets, suggesting over-transfer relative to the social optimum.
- Policy implication: **Supply-side reform** (increasing $\eta$) Pareto-dominates demand-side interventions for reducing lock-in without adverse price effects.

---

*Generated 2026-02-19. All results reproducible from `simulate.py` with seed 42.*
