"""
simulate.py — Intergenerational Wealth Lock-in and Housing Affordability
=========================================================================
Full simulation code implementing the two-generation OLG model with
heterogeneous households, parental transfers, and endogenous house prices.

Authors : [Anonymous for Review]
Date    : February 2026
Seed    : 42 (all results reproducible)

Structure
---------
  Params                      – model parameter dataclass
  annuity_factor              – alpha(r, T)
  renter_utility_vec          – vectorised renter optimisation
  draw_population             – Gaussian-copula income/wealth draw
  transfer_rule               – gift g_i = min(lambda * Wp_i, g_bar)
  household_choice_given_P    – vectorised tenure + housing choice
  aggregate_demand            – D(P) = sum h*_i
  supply                      – S(P) = S_bar + eta*(P - P_bar)
  equilibrium_price_solver    – Brent root-finding on excess demand
  compute_summary             – stats at equilibrium price
  run_grid                    – comparative statics across param grid
  make_figures                – 4 publication-quality matplotlib plots
  save_tables                 – CSV tables
  main                        – driver
"""

import dataclasses
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import optimize, stats
from typing import Dict, Tuple

warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Parameters ────────────────────────────────────────────────────────────────

@dataclass
class Params:
    """All model parameters in one place."""

    # Population
    N: int = 5_000

    # Income distribution  y ~ LogN(mu_y, sigma_y)
    mu_y: float = np.log(60_000)
    sigma_y: float = 0.50

    # Parental-wealth distribution  Wp ~ LogN(mu_w, sigma_w)
    mu_w: float = np.log(150_000)
    sigma_w: float = 1.20

    # Copula correlation (income – parental wealth in log space)
    rho: float = 0.40

    # Household own savings as share of income.
    # 0.25 → 3 months of income saved; keeps median household right at the
    # 20% LTV boundary at P_bar, producing ~50% ownership at baseline.
    savings_rate: float = 0.25

    # Mortgage parameters
    r: float = 0.05      # annual mortgage rate
    T: int   = 30        # loan term (years)
    # chi=0.10 (10% down-payment): LTV bites mainly for large h, so the
    # ownership/rental split is determined by the utility comparison, not by
    # hard exclusion.  DSTI then governs the feasible house size.
    chi: float = 0.10    # minimum LTV down-payment share
    psi: float = 0.36    # maximum DSTI ratio

    # Transfer technology
    lam: float   = 0.10           # g = lam * Wp (transfer intensity)
    g_bar: float = 100_000.0      # maximum gift ($)

    # Preferences
    gamma: float = 2.0    # CRRA curvature (consumption)
    theta: float = 1.5    # CRRA curvature (housing)
    # phi calibrated for dollar units so renter spends ~30% of income on rent.
    # With gamma=2, theta=1.5, y~$60k, R~$19.5k: phi ≈ 1.1e-5 delivers s*≈0.30.
    phi: float   = 1.1e-5

    # Housing market
    # kappa chosen so the median household is near-indifferent between renting
    # and owning.  Break-even condition (no down payment):
    #   kappa_break ≈ (1-chi) * alpha(r,T) = 0.80 * 0.0651 = 0.0521
    # At chi=0.20 and savings_rate=0.25, the effective median dp ratio is ~10%,
    # giving kappa_break ≈ 0.90 * 0.0651 = 0.0586.  We set kappa = 0.058 so
    # the median buyer is near the indifference margin, producing ~50% ownership
    # and a large, meaningful lock-in index.
    # kappa=0.060: rent = $18k/yr on $300k house (6% gross yield).
    # With chi=0.10 and alpha(5%,30)=0.0651, the median household (d=$30k,
    # h*=1) pays $17.6k/yr in mortgage vs $18k/yr in rent — near-indifferent
    # → ~50% ownership at baseline.
    kappa: float = 0.060          # rent-to-price ratio R = kappa * P_bar (fixed)
    S_bar: float = 5_000.0        # baseline housing supply (calibrated in main)
    P_bar: float = 300_000.0      # baseline / reference price ($)
    eta: float   = 0.0            # supply elasticity (units per $)

    # Housing-size grid searched by each household.
    # h_min=0.30 (studio / small flat): small enough that LTV rarely excludes
    # outright, so ownership choice is driven by utility comparison.
    h_min: float = 0.30    # min size (units, normalised to 1 = baseline house)
    h_max: float = 2.00    # max size
    n_h: int     = 20      # grid resolution

    # Bisection bounds for price solver
    P_lo: float = 50_000.0
    P_hi: float = 900_000.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def annuity_factor(r: float, T: int) -> float:
    """
    Annual annuity factor alpha(r, T) = r / [1 - (1+r)^{-T}].
    Converts mortgage principal to annual payment.
    """
    if r < 1e-12:
        return 1.0 / T
    return r / (1.0 - (1.0 + r) ** (-T))


def renter_utility_vec(y_arr: np.ndarray, P: float, p: Params) -> np.ndarray:
    """
    Vectorised solution of the renter's optimisation problem:

        max_{c, h^R}  c^{1-gamma}/(1-gamma) + phi * (h^R)^{1-theta}/(1-theta)
        s.t.          c + R * h^R = y_i,    c > 0,  h^R > 0

    FOC:  c^{-gamma} = phi * (h^R)^{-theta} / R
          => h^R = (phi/R)^{1/theta} * c^{gamma/theta}

    Substituting into the budget yields:
          f(c) = c + R * (phi/R)^{1/theta} * c^{gamma/theta} - y = 0

    Solved by bisection over the array (60 iterations, machine-precision).

    Returns
    -------
    V_R : (N,) array of maximised renter utilities.
    """
    # Rent is tied to the long-run reference price P_bar, not the current
    # equilibrium price P.  This keeps the rental market in its long-run
    # steady state and makes aggregate ownership demand D(P) monotone
    # decreasing in P (standard simplification in static housing models).
    R = p.kappa * p.P_bar
    N = len(y_arr)

    # Precompute constants
    factor = (p.phi / R) ** (1.0 / p.theta)   # (phi/R)^{1/theta}
    power  = p.gamma / p.theta                  # gamma / theta

    # Bracket: f(0) = -y < 0,  f(y) = y * [1 + factor*y^{power-1}] > 0
    c_lo = np.full(N, 1e-10)
    c_hi = y_arr - 1e-10

    for _ in range(60):
        c_mid  = 0.5 * (c_lo + c_hi)
        h_mid  = factor * c_mid ** power
        f_mid  = c_mid + R * h_mid - y_arr
        above  = f_mid > 0.0
        c_hi   = np.where(above,  c_mid, c_hi)
        c_lo   = np.where(~above, c_mid, c_lo)

    c_opt   = 0.5 * (c_lo + c_hi)
    h_r_opt = factor * c_opt ** power

    mask = (c_opt > 0) & (h_r_opt > 0) & (y_arr > 0)
    V_R = np.where(
        mask,
        c_opt ** (1 - p.gamma) / (1 - p.gamma)
        + p.phi * h_r_opt ** (1 - p.theta) / (1 - p.theta),
        -1e30,
    )
    return V_R


# ── Population ────────────────────────────────────────────────────────────────

def draw_population(p: Params, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw (y_i, Wp_i) for N households using a Gaussian copula with
    correlation rho to capture the log-space income–parental-wealth link.

    Returns
    -------
    y  : (N,) household annual income
    Wp : (N,) parental wealth
    """
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, p.rho], [p.rho, 1.0]])
    Z   = rng.multivariate_normal([0.0, 0.0], cov, size=p.N)

    # Convert to uniform margins, then to log-normal quantiles
    U  = stats.norm.cdf(Z)
    y  = stats.lognorm.ppf(U[:, 0], s=p.sigma_y, scale=np.exp(p.mu_y))
    Wp = stats.lognorm.ppf(U[:, 1], s=p.sigma_w, scale=np.exp(p.mu_w))
    return y, Wp


# ── Transfer Rule ─────────────────────────────────────────────────────────────

def transfer_rule(Wp: np.ndarray, p: Params) -> np.ndarray:
    """g_i = min(lambda * Wp_i, g_bar)."""
    return np.minimum(p.lam * Wp, p.g_bar)


# ── Household Optimisation ────────────────────────────────────────────────────

def household_choice_given_P(
    y: np.ndarray,
    Wp: np.ndarray,
    P: float,
    p: Params,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised computation of optimal tenure and housing demand for every
    household given house price P.

    Down-payment resources: d_i = a_i + g_i  (own savings + parental gift).

    Feasibility of housing size h requires:
      (LTV)   d_i >= chi * P * h
      (DSTI)  (P*h - d_i) * alpha(r,T) <= psi * y_i
      (c > 0) y_i - (P*h - d_i) * alpha(r,T) > 0

    The household chooses h from a grid [h_min, h_max] to maximise
    U(c, h) = c^{1-gamma}/(1-gamma) + phi * h^{1-theta}/(1-theta).

    Returns
    -------
    owner  : (N,) bool  – True if household optimally becomes an owner
    h_star : (N,) float – optimal housing quantity (0 if renter)
    V_diff : (N,) float – V^O* - V^R* (positive iff owner)
    """
    N     = len(y)
    a     = p.savings_rate * y                       # own savings: a_i
    g     = transfer_rule(Wp, p)                     # parental gift: g_i
    d     = a + g                                    # total down-payment resources

    alpha = annuity_factor(p.r, p.T)

    # Housing grid  (n_h,)
    h_grid = np.linspace(p.h_min, p.h_max, p.n_h)

    # Broadcast to (N, n_h)
    H  = h_grid[np.newaxis, :]          # (1, n_h)
    D  = d[:, np.newaxis]               # (N, 1)
    Y  = y[:, np.newaxis]               # (N, 1)

    house_cost = P * H                  # (1, n_h)

    # ── Constraint 1: LTV ──────────────────────────────────────────────────
    ltv_ok = D >= p.chi * house_cost    # (N, n_h)

    # ── Mortgage required ──────────────────────────────────────────────────
    # Use all available resources as down payment (d_i); m = max(Ph - d_i, 0)
    m_mat  = np.maximum(house_cost - D, 0.0)   # (N, n_h)

    # ── Constraint 2: DSTI ────────────────────────────────────────────────
    m_max_dsti = p.psi * Y / alpha             # (N, 1) – broadcast
    dsti_ok    = m_mat <= m_max_dsti           # (N, n_h)

    # ── Constraint 3: positive consumption ────────────────────────────────
    pay_mat = m_mat * alpha                    # annual mortgage payment  (N, n_h)
    c_mat   = Y - pay_mat                      # consumption  (N, n_h)
    c_ok    = c_mat > 1e-3

    feasible = ltv_ok & dsti_ok & c_ok        # (N, n_h)

    # ── Owner utility (only at feasible (i, h) pairs) ─────────────────────
    safe_c = np.where(feasible, c_mat, 1.0)
    safe_h = np.where(feasible, H,     1.0)

    U_c     = np.where(feasible, safe_c ** (1 - p.gamma) / (1 - p.gamma), -1e30)
    U_h     = np.where(feasible, p.phi * safe_h ** (1 - p.theta) / (1 - p.theta), -1e30)
    V_O_mat = np.where(feasible, U_c + U_h, -1e30)

    # Best h for each household
    best_idx  = np.argmax(V_O_mat, axis=1)          # (N,)
    best_V_O  = V_O_mat[np.arange(N), best_idx]     # (N,)
    has_opt   = feasible.any(axis=1)                 # (N,)
    best_h    = np.where(has_opt, h_grid[best_idx], 0.0)
    best_V_O  = np.where(has_opt, best_V_O, -1e30)

    # ── Renter utility ────────────────────────────────────────────────────
    V_R = renter_utility_vec(y, P, p)               # (N,)

    # ── Tenure decision ───────────────────────────────────────────────────
    owner  = best_V_O > V_R
    h_star = np.where(owner, best_h, 0.0)
    V_diff = best_V_O - V_R

    return owner, h_star, V_diff


# ── Market Aggregates ─────────────────────────────────────────────────────────

def aggregate_demand(
    y: np.ndarray, Wp: np.ndarray, P: float, p: Params
) -> float:
    """D(P) = Σ h*_i over all households."""
    _, h_star, _ = household_choice_given_P(y, Wp, P, p)
    return float(h_star.sum())


def supply(P: float, p: Params) -> float:
    """S(P) = S_bar + eta * (P - P_bar)."""
    return p.S_bar + p.eta * (P - p.P_bar)


# ── Equilibrium Price Solver ──────────────────────────────────────────────────

def equilibrium_price_solver(
    y: np.ndarray,
    Wp: np.ndarray,
    p: Params,
    verbose: bool = False,
) -> float:
    """
    Find P* s.t. D(P*) = S(P*) by Brent's method on excess demand.

    The excess demand function ED(P) = D(P) - S(P) is continuous and
    generically decreasing in P (more expensive houses → fewer owners
    and smaller optimal h), ensuring a unique crossing.

    Returns
    -------
    P_star : float  equilibrium house price.
    """
    def excess_demand(P: float) -> float:
        return aggregate_demand(y, Wp, P, p) - supply(P, p)

    ed_lo = excess_demand(p.P_lo)
    ed_hi = excess_demand(p.P_hi)

    if verbose:
        print(f"    ED(P_lo={p.P_lo:,.0f}) = {ed_lo:+.1f}")
        print(f"    ED(P_hi={p.P_hi:,.0f}) = {ed_hi:+.1f}")

    if ed_lo * ed_hi > 0:
        # No sign change: return bound with smallest absolute excess demand
        return p.P_lo if abs(ed_lo) <= abs(ed_hi) else p.P_hi

    P_star = optimize.brentq(
        excess_demand, p.P_lo, p.P_hi, xtol=500.0, maxiter=60
    )
    return float(P_star)


# ── Summary Statistics ────────────────────────────────────────────────────────

def compute_summary(
    y: np.ndarray, Wp: np.ndarray, P_star: float, p: Params
) -> Dict:
    """Compute key model statistics at the equilibrium price P*."""
    owner, h_star, V_diff = household_choice_given_P(y, Wp, P_star, p)
    N = len(y)

    # Ownership by parental-wealth decile (1–10)
    decile_edges = np.percentile(Wp, np.arange(0, 110, 10))
    dec_idx      = np.clip(np.digitize(Wp, decile_edges, right=True), 1, 10)
    ownership_by_decile = np.array(
        [owner[dec_idx == d].mean() for d in range(1, 11)]
    )

    # Lock-in index: ownership gap between top and bottom quintile
    q20 = np.percentile(Wp, 20)
    q80 = np.percentile(Wp, 80)
    lock_in = owner[Wp >= q80].mean() - owner[Wp <= q20].mean()

    # Price-to-income ratio
    PTI = P_star / np.median(y)

    # Per-household utility
    a      = p.savings_rate * y
    g      = transfer_rule(Wp, p)
    d_vec  = a + g
    alpha  = annuity_factor(p.r, p.T)

    utilities = np.empty(N)
    for i in range(N):
        if owner[i] and h_star[i] > 0:
            m   = max(P_star * h_star[i] - d_vec[i], 0.0)
            pay = m * alpha
            c   = y[i] - pay
            if c > 0:
                utilities[i] = (
                    c ** (1 - p.gamma) / (1 - p.gamma)
                    + p.phi * h_star[i] ** (1 - p.theta) / (1 - p.theta)
                )
            else:
                utilities[i] = -1e30
        else:
            utilities[i] = renter_utility_vec(np.array([y[i]]), P_star, p)[0]

    return {
        "P_star":              P_star,
        "ownership_rate":      float(owner.mean()),
        "lock_in_index":       float(lock_in),
        "PTI":                 float(PTI),
        "ownership_by_decile": ownership_by_decile,
        "owner":               owner,
        "h_star":              h_star,
        "V_diff":              V_diff,
        "avg_utility":         float(np.nanmean(utilities)),
        "utilities":           utilities,
    }


# ── Grid Runs ─────────────────────────────────────────────────────────────────

def run_grid(p_base: Params, y: np.ndarray, Wp: np.ndarray) -> pd.DataFrame:
    """
    Run comparative statics across three experimental dimensions:
      A. Transfer intensity lambda x supply elasticity eta
      B. Mortgage rate r x transfer intensity lambda
      C. LTV constraint chi x transfer intensity lambda
    Returns one DataFrame with all results.
    """
    records = []

    # ── Experiment A: lambda × eta ──────────────────────────────────────────
    print("  Experiment A: lambda × eta grid …")
    for lam in [0.0, 0.10, 0.20, 0.30]:
        for eta in [0.0, 0.002, 0.008]:
            p      = dataclasses.replace(p_base, lam=lam, eta=eta)
            P_star = equilibrium_price_solver(y, Wp, p)
            s      = compute_summary(y, Wp, P_star, p)
            records.append(dict(
                experiment="lambda_eta",
                r=p_base.r, lam=lam, chi=p_base.chi,
                psi=p_base.psi, eta=eta,
                P_star=s["P_star"], ownership_rate=s["ownership_rate"],
                lock_in_index=s["lock_in_index"], PTI=s["PTI"],
                avg_utility=s["avg_utility"],
            ))

    # ── Experiment B: r × lambda ────────────────────────────────────────────
    print("  Experiment B: r × lambda grid …")
    for r_val in [0.03, 0.05, 0.07]:
        for lam in [0.0, 0.10, 0.20]:
            p      = dataclasses.replace(p_base, r=r_val, lam=lam)
            P_star = equilibrium_price_solver(y, Wp, p)
            s      = compute_summary(y, Wp, P_star, p)
            records.append(dict(
                experiment="rate_shock",
                r=r_val, lam=lam, chi=p_base.chi,
                psi=p_base.psi, eta=p_base.eta,
                P_star=s["P_star"], ownership_rate=s["ownership_rate"],
                lock_in_index=s["lock_in_index"], PTI=s["PTI"],
                avg_utility=s["avg_utility"],
            ))

    # ── Experiment C: chi × lambda ──────────────────────────────────────────
    print("  Experiment C: chi × lambda grid …")
    for chi_val in [0.05, 0.10, 0.20]:
        for lam in [0.0, 0.10, 0.20, 0.30]:
            p      = dataclasses.replace(p_base, chi=chi_val, lam=lam)
            P_star = equilibrium_price_solver(y, Wp, p)
            s      = compute_summary(y, Wp, P_star, p)
            records.append(dict(
                experiment="chi_lam",
                r=p_base.r, lam=lam, chi=chi_val,
                psi=p_base.psi, eta=p_base.eta,
                P_star=s["P_star"], ownership_rate=s["ownership_rate"],
                lock_in_index=s["lock_in_index"], PTI=s["PTI"],
                avg_utility=s["avg_utility"],
            ))

    return pd.DataFrame(records)


def decile_ownership_by_rate(
    y: np.ndarray, Wp: np.ndarray, p_base: Params
) -> Dict[float, np.ndarray]:
    """
    For Figure 1: compute ownership rate by parental-wealth decile
    at three mortgage rates.
    """
    result = {}
    for r_val in [0.03, 0.05, 0.07]:
        p      = dataclasses.replace(p_base, r=r_val)
        P_star = equilibrium_price_solver(y, Wp, p)
        s      = compute_summary(y, Wp, P_star, p)
        result[r_val] = s["ownership_by_decile"]
        print(f"    r={r_val:.0%}  P*=${P_star:,.0f}  "
              f"own={s['ownership_rate']:.2%}  LI={s['lock_in_index']:.3f}")
    return result


def rate_response_by_wealth(
    y: np.ndarray, Wp: np.ndarray, p_base: Params
) -> Dict:
    """
    For Figure 4: ownership rate by parental-wealth tercile across
    r ∈ {3%, 4%, 5%, 6%, 7%}, evaluated at P = P_bar (partial equilibrium).

    Holding price fixed isolates the direct DSTI/LTV channel of Proposition C:
    as r rises, alpha rises, the feasible mortgage set shrinks, and low-Wp
    households (who need larger mortgages) are disproportionately excluded.
    This is the mechanism stated in the paper; GE price adjustment (which would
    partially offset the effect via lower P*) is discussed separately.
    """
    r_vals = [0.03, 0.04, 0.05, 0.06, 0.07]
    q33, q66 = np.percentile(Wp, [33, 66])
    groups = {
        "Bottom Tercile": Wp <= q33,
        "Middle Tercile": (Wp > q33) & (Wp <= q66),
        "Top Tercile":    Wp > q66,
    }
    by_group = {name: [] for name in groups}
    P_fixed  = p_base.P_bar       # hold price at long-run baseline

    for r_val in r_vals:
        p = dataclasses.replace(p_base, r=r_val)
        owner, _, _ = household_choice_given_P(y, Wp, P_fixed, p)
        for name, mask in groups.items():
            by_group[name].append(owner[mask].mean() * 100)

    return {"r_vals": r_vals, "P_vals": [P_fixed] * len(r_vals),
            "by_group": by_group, "group_names": list(groups.keys())}


# ── Figures ───────────────────────────────────────────────────────────────────

PLOT_STYLE = {
    "figure.dpi":          150,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "font.family":         "serif",
    "font.size":           11,
    "axes.titlesize":      12,
    "axes.labelsize":      11,
    "legend.fontsize":     10,
    "lines.linewidth":     2.0,
    "axes.grid":           True,
    "grid.alpha":          0.30,
    "grid.linestyle":      "--",
}
plt.rcParams.update(PLOT_STYLE)

COLORS = ["#2166ac", "#d6604d", "#4dac26"]
MARKERS = ["o", "s", "^"]


def make_figures(
    grid_df:      pd.DataFrame,
    decile_data:  Dict[float, np.ndarray],
    rate_data:    Dict,
    out_dir:      str = "output",
) -> None:
    """Generate and save the four main paper figures (PDF + PNG)."""
    os.makedirs(out_dir, exist_ok=True)

    # ── Figure 1: Ownership by parental-wealth decile ─────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for idx, (r_val, ownership) in enumerate(sorted(decile_data.items())):
        ax.plot(range(1, 11), ownership * 100,
                marker=MARKERS[idx], color=COLORS[idx],
                label=f"r = {r_val*100:.0f}%")
    ax.set_xlabel("Parental Wealth Decile")
    ax.set_ylabel("Ownership Rate (%)")
    ax.set_title("Figure 1: Homeownership Rate by Parental Wealth Decile")
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f"D{d}" for d in range(1, 11)])
    ax.legend(title="Mortgage rate")
    fig.tight_layout()
    _savefig(fig, out_dir, "fig1_ownership_by_decile")
    print("  Saved Figure 1.")

    # ── Figure 2: Equilibrium P* vs transfer intensity lambda ─────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    eta_labels = {0.0: "Inelastic ($\\eta=0$)", 0.002: "$\\eta=0.002$",
                  0.008: "$\\eta=0.008$"}
    sub_A = grid_df[grid_df["experiment"] == "lambda_eta"]
    for idx, eta_val in enumerate([0.0, 0.002, 0.008]):
        sub = sub_A[sub_A["eta"] == eta_val].sort_values("lam")
        ax.plot(sub["lam"], sub["P_star"] / 1_000,
                marker=MARKERS[idx], color=COLORS[idx],
                label=eta_labels[eta_val])
    ax.set_xlabel("Transfer Intensity $\\lambda$")
    ax.set_ylabel("Equilibrium Price $P^*$ (\\$'000)")
    ax.set_title("Figure 2: Equilibrium Price vs. Transfer Intensity")
    ax.legend(title="Supply elasticity")
    fig.tight_layout()
    _savefig(fig, out_dir, "fig2_price_vs_lambda")
    print("  Saved Figure 2.")

    # ── Figure 3: Lock-in index vs lambda (different supply elasticities) ─
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for idx, eta_val in enumerate([0.0, 0.002, 0.008]):
        sub = sub_A[sub_A["eta"] == eta_val].sort_values("lam")
        ax.plot(sub["lam"], sub["lock_in_index"],
                marker=MARKERS[idx], color=COLORS[idx],
                label=eta_labels[eta_val])
    ax.set_xlabel("Transfer Intensity $\\lambda$")
    ax.set_ylabel("Lock-in Index $\\mathit{LI}$")
    ax.set_title("Figure 3: Intergenerational Lock-in vs. Transfer Intensity")
    ax.legend(title="Supply elasticity")
    fig.tight_layout()
    _savefig(fig, out_dir, "fig3_lockin_vs_lambda")
    print("  Saved Figure 3.")

    # ── Figure 4: Rate shock response by parental-wealth group ───────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    r_pct = [r * 100 for r in rate_data["r_vals"]]
    for idx, name in enumerate(rate_data["group_names"]):
        ax.plot(r_pct, rate_data["by_group"][name],
                marker=MARKERS[idx], color=COLORS[idx], label=name)
    ax.set_xlabel("Mortgage Rate $r$ (%)")
    ax.set_ylabel("Ownership Rate (%)")
    ax.set_title("Figure 4: Rate Shock Response by Wealth Group (PE, P fixed)")
    ax.legend(title="Parental wealth tercile")
    fig.tight_layout()
    _savefig(fig, out_dir, "fig4_rate_response")
    print("  Saved Figure 4.")


def _savefig(fig: plt.Figure, out_dir: str, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ── Save Tables ───────────────────────────────────────────────────────────────

def save_tables(grid_df: pd.DataFrame, out_dir: str = "output") -> None:
    """Write CSV tables used in the paper."""
    os.makedirs(out_dir, exist_ok=True)

    # Full grid
    grid_df.to_csv(os.path.join(out_dir, "grid_results.csv"), index=False)

    # Baseline row
    baseline = grid_df[
        (grid_df["experiment"] == "rate_shock")
        & (grid_df["r"] == 0.05)
        & (grid_df["lam"] == 0.10)
    ]
    baseline.to_csv(os.path.join(out_dir, "baseline_summary.csv"), index=False)

    # Rate × lambda pivot
    rate_sub = grid_df[grid_df["experiment"] == "rate_shock"]
    for col in ["ownership_rate", "lock_in_index", "PTI"]:
        piv = rate_sub.pivot(index="r", columns="lam", values=col)
        piv.to_csv(os.path.join(out_dir, f"rate_lam_{col}.csv"))

    print(f"  Tables saved to {out_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    bar = "=" * 60
    print(bar)
    print("  Intergenerational Wealth Lock-in — Simulation")
    print(bar)

    # ── Draw population (fixed seed, reused across all experiments) ────────
    p_proto = Params()
    print(f"\n[1] Drawing population  N={p_proto.N}  (seed={SEED}) …")
    y, Wp = draw_population(p_proto, seed=SEED)
    print(f"    Median income:             ${np.median(y):>10,.0f}")
    print(f"    Median parental wealth:    ${np.median(Wp):>10,.0f}")
    emp_rho = np.corrcoef(np.log(y), np.log(Wp))[0, 1]
    print(f"    Log income–wealth corr:    {emp_rho:.3f}  (target {p_proto.rho})")

    # ── Calibrate S_bar so baseline equilibrium lands at P_bar ────────────
    print(f"\n[2] Calibrating S_bar so P* = P_bar = ${p_proto.P_bar:,.0f} …")
    D_at_Pbar = aggregate_demand(y, Wp, p_proto.P_bar, p_proto)
    p_base    = dataclasses.replace(p_proto, S_bar=D_at_Pbar)
    print(f"    Calibrated S_bar = {D_at_Pbar:.1f} units")

    # ── Verify baseline equilibrium ────────────────────────────────────────
    print("\n[3] Verifying baseline equilibrium …")
    P_star_base = equilibrium_price_solver(y, Wp, p_base, verbose=True)
    s_base      = compute_summary(y, Wp, P_star_base, p_base)
    print(f"    P* = ${P_star_base:,.0f}   "
          f"(target P_bar = ${p_base.P_bar:,.0f})")
    print(f"    Ownership rate:   {s_base['ownership_rate']:.3f}")
    print(f"    Lock-in index:    {s_base['lock_in_index']:.3f}")
    print(f"    Price-to-income:  {s_base['PTI']:.2f}x")

    # ── Decile data for Figure 1 ───────────────────────────────────────────
    print("\n[4] Ownership by decile at r ∈ {3%, 5%, 7%} …")
    decile_data = decile_ownership_by_rate(y, Wp, p_base)

    # ── Rate response by wealth group for Figure 4 ─────────────────────────
    print("\n[5] Rate-response by parental wealth group …")
    rate_data = rate_response_by_wealth(y, Wp, p_base)

    # ── Comparative statics grid ───────────────────────────────────────────
    print("\n[6] Running comparative statics grid …")
    grid_df = run_grid(p_base, y, Wp)
    print(f"    Grid complete: {len(grid_df)} scenarios.")

    # ── Figures ───────────────────────────────────────────────────────────
    print("\n[7] Generating figures …")
    make_figures(grid_df, decile_data, rate_data, out_dir="output")

    # ── Tables ────────────────────────────────────────────────────────────
    print("\n[8] Saving tables …")
    save_tables(grid_df, out_dir="output")

    print(f"\n{bar}")
    print("  Done.  All outputs written to ./output/")
    print(bar)


if __name__ == "__main__":
    main()
