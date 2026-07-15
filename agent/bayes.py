"""Bayesian decision engine for early-development go/no-go.

Every model here is CONJUGATE, so every quantity is closed-form or deterministic numeric integration
on a fixed grid. There is NO Monte Carlo in this module: no seed to manage, results are
bit-reproducible across runs and platforms, and the tests assert exact values rather than tolerances.
A tool whose only job is to support a decision must not return a different verdict on re-run.

Endpoints:  binary (Beta-Binomial)  |  continuous mean, known SD (Normal-Normal)
Framings:   single-arm vs a performance goal (device)  |  two-arm vs a control (drug)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import brentq

_GRID = 2001          # quadrature points for the Beta-difference integral


@dataclass(frozen=True)
class Prior:
    name: str                       # "Phase-I informed" | "Vague" | "Skeptical" | "Enthusiastic"
    kind: str                       # "beta" (binary endpoint) | "normal" (continuous endpoint)
    params: tuple[float, float]     # beta: (a, b).  normal: (mu, sd).
    provenance: str                 # human-readable: where this prior came from


@dataclass(frozen=True)
class DecisionRule:
    """Dual-criterion (Lalonde) go/no-go. A device performance goal is the degenerate case tv == lrv."""
    tv: float                       # Target Value: the effect we hope for
    lrv: float                      # Lower Reference Value: the minimum worth pursuing
    gate_tv: float = 0.80           # required P(theta beyond tv)
    gate_lrv: float = 0.90          # required P(theta beyond lrv)
    stop_lrv: float = 0.10          # P(theta beyond lrv) below this -> STOP
    higher_is_better: bool = True


# ── conjugate updates ─────────────────────────────────────────────────────────────────────────────
def beta_posterior(a: float, b: float, x: int, n: int) -> tuple[float, float]:
    """Beta(a,b) prior + x successes in n trials -> Beta(a+x, b+n-x). Exact."""
    return float(a + x), float(b + n - x)


def normal_posterior(mu0: float, sd0: float, xbar: float, sd: float, n: int) -> tuple[float, float]:
    """Normal(mu0, sd0) prior + n observations with mean xbar and KNOWN sd -> normal posterior. Exact."""
    if n <= 0:
        return float(mu0), float(sd0)
    prec0, prec_d = 1.0 / sd0 ** 2, n / sd ** 2
    var = 1.0 / (prec0 + prec_d)
    return float(var * (prec0 * mu0 + prec_d * xbar)), float(np.sqrt(var))


# ── tail probabilities ────────────────────────────────────────────────────────────────────────────
def prob_exceeds(kind: str, p1, p2, threshold: float, higher_is_better: bool = True):
    """P(theta is BEYOND threshold on the good side). Vectorized over p1/p2 (numpy arrays welcome)."""
    if kind == "beta":
        sf = stats.beta.sf(threshold, p1, p2)
    else:
        sf = stats.norm.sf(threshold, loc=p1, scale=p2)
    return sf if higher_is_better else 1.0 - sf


def prob_diff_exceeds(kind: str, t: tuple[float, float], c: tuple[float, float],
                      threshold: float, higher_is_better: bool = True) -> float:
    """P(theta_treatment - theta_control is beyond threshold).

    normal: a difference of normals is normal -> closed form.
    beta:   no closed form, so 1-D quadrature on a fixed grid:
                P(T - C > d) = INT f_C(v) * sf_T(v + d) dv
            Deterministic and fast. NOT Monte Carlo.
    """
    if kind == "normal":
        mu = t[0] - c[0]
        sd = float(np.hypot(t[1], c[1]))
        sf = float(stats.norm.sf(threshold, loc=mu, scale=sd))
    else:
        v = np.linspace(0.0, 1.0, _GRID)
        f_c = stats.beta.pdf(v, c[0], c[1])
        sf_t = stats.beta.sf(np.clip(v + threshold, 0.0, 1.0), t[0], t[1])
        sf = float(np.trapezoid(f_c * sf_t, v))
    return sf if higher_is_better else 1.0 - sf


# ── the decision rule ─────────────────────────────────────────────────────────────────────────────
def decide(p_tv: float, p_lrv: float, rule: DecisionRule) -> tuple[str, str]:
    """Dual-criterion verdict. p_tv / p_lrv are probabilities of being on the GOOD side of each value."""
    side = "above" if rule.higher_is_better else "below"
    ev = (f"P({side} TV {rule.tv:g}) = {p_tv:.1%}, P({side} LRV {rule.lrv:g}) = {p_lrv:.1%}")
    if p_tv >= rule.gate_tv and p_lrv >= rule.gate_lrv:
        return "GO", (f"{ev}. Clears both pre-specified gates "
                      f"({rule.gate_tv:.0%} at the TV and {rule.gate_lrv:.0%} at the LRV).")
    if p_lrv < rule.stop_lrv:
        return "STOP", (f"{ev}. The effect is very unlikely to reach even the LRV "
                        f"({rule.lrv:g}), the minimum worth pursuing.")
    return "CONSIDER", (f"{ev}. Promising but short of the pre-specified GO gates "
                        f"({rule.gate_tv:.0%} at the TV, {rule.gate_lrv:.0%} at the LRV) -- "
                        "the evidence does not yet justify a commitment.")


# ── priors ────────────────────────────────────────────────────────────────────────────────────────
def prior_ess(prior: Prior) -> float:
    """Effective sample size. Beta(a,b) carries as much information as a+b observations."""
    return float(prior.params[0] + prior.params[1]) if prior.kind == "beta" else float("nan")


def prior_panel(informed: Prior, rule: DecisionRule) -> list[Prior]:
    """Four defensible priors. If the verdict flips across them it is FRAGILE, not an answer.

    This is FDA's prior-sensitivity requirement (Jan 2026 draft guidance): show that the trial's
    conclusion is robust across plausible alternative priors, not an artefact of one choice.
    """
    if informed.kind != "beta":
        _, sd = informed.params
        span = abs(rule.tv - rule.lrv) or 1.0
        return [
            informed,
            Prior("Vague", "normal", (rule.lrv, 10.0 * span), "Weakly informative, centred at the LRV."),
            Prior("Skeptical", "normal", (rule.lrv, span / 2), "Centred at the minimum worth pursuing."),
            Prior("Enthusiastic", "normal", (rule.tv, span / 2), "Centred at the target value."),
        ]
    ess = 10.0                                       # the reference priors carry ~10 observations
    skeptical = (rule.lrv * ess, (1 - rule.lrv) * ess)
    enthusiastic = (rule.tv * ess, (1 - rule.tv) * ess)
    return [
        informed,
        Prior("Vague", "beta", (1.0, 1.0), "Uniform on [0,1]: every response rate equally likely."),
        Prior("Skeptical", "beta", skeptical,
              f"Centred on the LRV ({rule.lrv:g}), the minimum worth pursuing; ESS {ess:g}."),
        Prior("Enthusiastic", "beta", enthusiastic,
              f"Centred on the TV ({rule.tv:g}), the hoped-for effect; ESS {ess:g}."),
    ]


# ── assurance + operating characteristics ─────────────────────────────────────────────────────────
def go_grid_binary(prior: Prior, n: int, rule: DecisionRule) -> np.ndarray:
    """go[x] == 1 iff observing x successes in n trials yields a GO. Computed ONCE, then reused by
    assurance, the operating characteristics, and the predictive probability -- all three are just
    different weightings of this same vector."""
    a, b = prior.params
    xs = np.arange(n + 1)
    post_a, post_b = a + xs, b + (n - xs)
    p_tv = prob_exceeds("beta", post_a, post_b, rule.tv, rule.higher_is_better)
    p_lrv = prob_exceeds("beta", post_a, post_b, rule.lrv, rule.higher_is_better)
    return ((p_tv >= rule.gate_tv) & (p_lrv >= rule.gate_lrv)).astype(int)


def _go_threshold_normal(prior: Prior, n: int, rule: DecisionRule, sd: float) -> float:
    """The observed sample mean at which the decision tips to GO. The posterior tail probability is
    monotone in the sample mean, so a single crossing exists; solve it EXACTLY with brentq rather
    than a grid search. (A grid sized in raw per-observation SD units, never rescaled by the
    standard error sd/sqrt(n), silently loses resolution relative to the standard error as n grows
    -- the search then always lands on the conservative side, biasing assurance/power low, with the
    bias growing with n. brentq is deterministic and machine-precision: not Monte Carlo.)"""
    mu0, sd0 = prior.params
    se = sd / np.sqrt(n)

    def g(xbar: float) -> float:
        """Signed margin by which the GO criterion is met at this sample mean. GO iff g >= 0."""
        pm, ps = normal_posterior(mu0, sd0, xbar, sd, n)
        p_tv = prob_exceeds("normal", pm, ps, rule.tv, rule.higher_is_better)
        p_lrv = prob_exceeds("normal", pm, ps, rule.lrv, rule.higher_is_better)
        return min(p_tv - rule.gate_tv, p_lrv - rule.gate_lrv)

    lo, hi = mu0 - (10 * sd0 + 10 * se), mu0 + (10 * sd0 + 10 * se)
    if rule.higher_is_better:
        if g(hi) < 0:
            return float("inf")     # met nowhere in the bracket -> GO rate 0
        if g(lo) >= 0:
            return lo               # met everywhere -> GO rate 1 (far end of the bracket)
        return float(brentq(g, lo, hi))
    if g(lo) < 0:
        return float("-inf")        # met nowhere in the bracket -> GO rate 0
    if g(hi) >= 0:
        return hi                   # met everywhere -> GO rate 1 (far end of the bracket)
    return float(brentq(g, lo, hi))


def assurance(prior: Prior, n_planned: int, rule: DecisionRule, sd: float | None = None) -> float:
    """P(the trial reaches GO), averaging over the prior uncertainty about the true effect.

    This is Bayesian power, and it is the honest number: classical power asks "what is the chance of
    success IF the effect is exactly X", which is a question nobody can answer. Assurance integrates
    over what you actually believe about X, and is usually lower.

    Binary: EXACT. assurance = SUM_x go[x] * BetaBinomial(x; n, a, b), because the prior-predictive
    distribution of the success count under a Beta prior IS the beta-binomial. No integration error.
    """
    if prior.kind == "beta":
        a, b = prior.params
        go = go_grid_binary(prior, n_planned, rule)
        xs = np.arange(n_planned + 1)
        return float(np.sum(stats.betabinom.pmf(xs, n_planned, a, b) * go))
    if sd is None or sd <= 0:
        raise ValueError("a continuous endpoint needs a positive known SD")
    mu0, sd0 = prior.params
    crit = _go_threshold_normal(prior, n_planned, rule, sd)
    se = sd / np.sqrt(n_planned)
    # theta ~ prior; xbar | theta ~ N(theta, se) -> xbar ~ N(mu0, sqrt(sd0^2 + se^2)) marginally
    marg = float(np.hypot(sd0, se))
    p = float(stats.norm.sf(crit, loc=mu0, scale=marg))
    return p if rule.higher_is_better else 1.0 - p


def operating_characteristics(prior: Prior, n_planned: int, rule: DecisionRule,
                              sd: float | None = None, grid=None) -> list[dict]:
    """The GO rate at each TRUE effect value. FDA's second pillar: show how the design behaves across
    a plausible range of truths, not just at the value you hope for.

    Read off this curve: the GO rate at the LRV is the type I error (declaring success when the effect
    is not worth pursuing); the GO rate at the TV is the power.
    """
    if grid is None:
        base = (np.linspace(0.01, 0.99, 99) if prior.kind == "beta"
                else np.linspace(rule.lrv - 2 * abs(rule.tv - rule.lrv),
                                 rule.tv + 2 * abs(rule.tv - rule.lrv), 99))
        # the plotted curve should visibly pass through the two thresholds that define the
        # decision, not just come close to them; union1d also sorts ascending and dedupes.
        grid = np.union1d(base, [rule.lrv, rule.tv])
    out = []
    if prior.kind == "beta":
        go = go_grid_binary(prior, n_planned, rule)
        xs = np.arange(n_planned + 1)
        for th in np.asarray(grid, dtype=float):
            out.append({"theta": float(th),
                        "go_rate": float(np.sum(stats.binom.pmf(xs, n_planned, th) * go))})
        return out
    if sd is None or sd <= 0:
        raise ValueError("a continuous endpoint needs a positive known SD")
    crit = _go_threshold_normal(prior, n_planned, rule, sd)
    se = sd / np.sqrt(n_planned)
    for th in np.asarray(grid, dtype=float):
        p = float(stats.norm.sf(crit, loc=th, scale=se))
        out.append({"theta": float(th), "go_rate": p if rule.higher_is_better else 1.0 - p})
    return out


def type_i_and_power(prior: Prior, n_planned: int, rule: DecisionRule,
                     sd: float | None = None) -> tuple[float, float]:
    """Type I error = GO rate at the LRV. Power = GO rate at the TV. Computed EXACTLY at those two
    thresholds (not read off the nearest point of a coarser plotting grid, which can be off by
    several percentage points when the thresholds don't land on a grid line)."""
    oc = operating_characteristics(prior, n_planned, rule, sd=sd, grid=np.array([rule.lrv, rule.tv]))
    return oc[0]["go_rate"], oc[1]["go_rate"]


# ── predictive probability of success (interim) ───────────────────────────────────────────────────
MAX_ENUM = 20_000     # documented cap on the enumeration; above it we bin rather than get slow


def predictive_prob_success(prior: Prior, x: int, n: int, n_planned: int, rule: DecisionRule) -> float:
    """P(the trial ENDS in GO | what we have seen so far). The interim question.

    EXACT for a binary endpoint. Enumerate every possible number of successes y among the
    n_planned - n patients not yet observed, weight each by the posterior-predictive (beta-binomial)
    probability of seeing exactly y, and check whether the FINAL total x + y would be a GO:

        PPoS = SUM_y  BetaBinom(y; m, a+x, b+n-x) * go_final[x + y]

    No simulation error. Low PPoS is the futility signal that stops a trial early and saves the money.
    """
    if prior.kind != "beta":
        raise ValueError("predictive_prob_success currently supports a binary endpoint only")
    if n > n_planned:
        raise ValueError(f"observed n ({n}) exceeds the planned n ({n_planned})")
    a, b = prior.params
    go_final = go_grid_binary(prior, n_planned, rule)
    m = n_planned - n
    if m == 0:                                        # trial complete: this IS the final decision
        return float(go_final[x])
    if m > MAX_ENUM:
        raise ValueError(f"{m:,} unobserved patients exceeds the {MAX_ENUM:,} enumeration cap")
    post_a, post_b = beta_posterior(a, b, x, n)
    ys = np.arange(m + 1)
    w = stats.betabinom.pmf(ys, m, post_a, post_b)     # posterior-predictive of the remaining successes
    return float(np.sum(w * go_final[x + ys]))
