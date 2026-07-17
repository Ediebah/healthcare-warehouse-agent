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
    """Four defensible priors for a prior-sensitivity analysis: the assurance under each shows how much
    the probability of success depends on the choice of prior.

    This is FDA's prior-sensitivity requirement (Jan 2026 draft guidance): show how the trial's
    probability of success varies across plausible alternative priors, not just under one choice.
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


# ── two-arm predictive probability of success ─────────────────────────────────────────────────────
MAX_ENUM_DIFF = 10_000    # cap on the (reachable) completion grid; above it, thin on a fixed stride

# trapezoid weights on the fixed [0,1] quadrature grid, computed once
_WV = np.full(_GRID, 1.0 / (_GRID - 1))
_WV[0] = _WV[-1] = 0.5 / (_GRID - 1)
_VGRID = np.linspace(0.0, 1.0, _GRID)


def _go_diff_block(t_ab, st, n_t: int, c_ab, sc, n_c: int, rule: DecisionRule) -> np.ndarray:
    """GO decision for every (final treatment count st, final control count sc). Vectorized: the
    beta-difference tail P(rate_t - rate_c > threshold) is one trapezoid integral per (st, sc), but it
    factorizes into (len_st, GRID) survival rows and (len_sc, GRID) density rows combined by a single
    matrix multiply -- no per-cell Python loop, no 3-D intermediate.

        P(t - c > d) = INT f_c(v) * sf_t(v + d) dv  ==  (sf_t * wv) @ f_c.T
    """
    at, bt = t_ab
    ac, bc = c_ab
    st = np.asarray(st, dtype=float)
    sc = np.asarray(sc, dtype=float)
    a_t, b_t = at + st, bt + (n_t - st)                       # final treatment posteriors  (len_st,)
    a_c, b_c = ac + sc, bc + (n_c - sc)                       # final control posteriors    (len_sc,)
    f_c = stats.beta.pdf(_VGRID[None, :], a_c[:, None], b_c[:, None])      # (len_sc, GRID)

    def block_p(threshold):
        thr = np.clip(_VGRID + threshold, 0.0, 1.0)
        sf_t = stats.beta.sf(thr[None, :], a_t[:, None], b_t[:, None])     # (len_st, GRID)
        p = (sf_t * _WV[None, :]) @ f_c.T                                  # (len_st, len_sc)
        return p if rule.higher_is_better else 1.0 - p

    p_tv = block_p(rule.tv)
    p_lrv = block_p(rule.lrv)
    return ((p_tv >= rule.gate_tv) & (p_lrv >= rule.gate_lrv)).astype(int)


def go_grid_diff(prior_t: Prior, prior_c: Prior, n_t: int, n_c: int, rule: DecisionRule) -> np.ndarray:
    """go[st, sc] == 1 iff a trial that finishes with st/n_t treatment and sc/n_c control responders is a
    GO. The two-arm analog of go_grid_binary; reused by the predictive probability and by the tests."""
    return _go_diff_block(prior_t.params, np.arange(n_t + 1), n_t,
                          prior_c.params, np.arange(n_c + 1), n_c, rule)


def predictive_prob_success_diff(prior_t: Prior, prior_c: Prior, x_t: int, n_t: int, x_c: int, n_c: int,
                                 n_planned_t: int, n_planned_c: int, rule: DecisionRule) -> float:
    """P(the randomized trial ENDS in GO | both arms' data so far). Exact for a binary endpoint.

    Enumerate every joint completion (y_t future treatment responders, y_c future control responders),
    weight by the PRODUCT of each arm's beta-binomial posterior-predictive, and check whether the FINAL
    difference clears the pre-specified gates:

        PPoS = SUM_{y_t, y_c} BetaBinom(y_t; m_t, ...) * BetaBinom(y_c; m_c, ...) * go[x_t+y_t, x_c+y_c]

    No simulation error. Above MAX_ENUM_DIFF reachable cells the completion grid is thinned on a fixed
    stride (deterministic, still no Monte Carlo)."""
    if prior_t.kind != "beta" or prior_c.kind != "beta":
        raise ValueError("the two-arm predictive probability supports a binary endpoint only")
    if n_t > n_planned_t or n_c > n_planned_c:
        raise ValueError("observed n exceeds the planned n in an arm")
    post_t = beta_posterior(*prior_t.params, x_t, n_t)        # observed posterior -> predictive weights
    post_c = beta_posterior(*prior_c.params, x_c, n_c)
    m_t, m_c = n_planned_t - n_t, n_planned_c - n_c
    y_t = np.arange(m_t + 1)
    y_c = np.arange(m_c + 1)
    if (m_t + 1) * (m_c + 1) > MAX_ENUM_DIFF:                 # thin: keep <= sqrt(cap) points per arm
        side = max(1, int(MAX_ENUM_DIFF ** 0.5))
        step_t = max(1, -(-(m_t + 1) // side))                # ceil division
        step_c = max(1, -(-(m_c + 1) // side))
        y_t, y_c = y_t[::step_t], y_c[::step_c]
    w_t = stats.betabinom.pmf(y_t, m_t, post_t[0], post_t[1])
    w_c = stats.betabinom.pmf(y_c, m_c, post_c[0], post_c[1])
    w_t = w_t / w_t.sum()                # renormalize (exact when unthinned; a subsample, not binning, when thinned)
    w_c = w_c / w_c.sum()
    go = _go_diff_block(prior_t.params, x_t + y_t, n_planned_t,
                        prior_c.params, x_c + y_c, n_planned_c, rule)
    return float(w_t @ go @ w_c)


# ── two-arm design-stage assurance ────────────────────────────────────────────────────────────────
def assurance_diff(prior_t: Prior, control_rate: float, n_planned_t: int, n_planned_c: int,
                   rule: DecisionRule) -> float:
    """P(a randomized trial ends in GO), before it runs. The treatment arm is averaged over its
    prior-predictive (beta-binomial); the control responds at the known control_rate and contributes
    sampling variability only (binomial). The analysis decision uses a vague control prior, exactly as
    the shipped two-arm interim decides. Exact -- no Monte Carlo.

        assurance = SUM_{s_t, s_c} BetaBinom(s_t; n_t, prior_t) * Binom(s_c; n_c, control_rate) * go[s_t, s_c]
    """
    if prior_t.kind != "beta":
        raise ValueError("two-arm assurance supports a binary endpoint only")
    if not 0.0 <= control_rate <= 1.0:
        raise ValueError("control_rate must be between 0 and 1")
    go = go_grid_diff(prior_t, Prior("Vague", "beta", (1.0, 1.0), ""), n_planned_t, n_planned_c, rule)
    w_t = stats.betabinom.pmf(np.arange(n_planned_t + 1), n_planned_t, *prior_t.params)
    w_c = stats.binom.pmf(np.arange(n_planned_c + 1), n_planned_c, control_rate)
    return float(w_t @ go @ w_c)


def operating_characteristics_diff(prior_t: Prior, control_rate: float, n_planned_t: int,
                                   n_planned_c: int, rule: DecisionRule, grid=None) -> list[dict]:
    """The GO rate at each TRUE risk difference, with control fixed at control_rate. `theta` carries the
    risk difference (treatment rate - control_rate) so the existing OC chart plots it correctly; `theta_t`
    is the treatment rate. Type I error = GO rate at difference = LRV; power = GO rate at difference = TV."""
    if prior_t.kind != "beta":
        raise ValueError("two-arm operating characteristics support a binary endpoint only")
    if not 0.0 <= control_rate <= 1.0:
        raise ValueError("control_rate must be between 0 and 1")
    go = go_grid_diff(prior_t, Prior("Vague", "beta", (1.0, 1.0), ""), n_planned_t, n_planned_c, rule)
    w_c = stats.binom.pmf(np.arange(n_planned_c + 1), n_planned_c, control_rate)
    if grid is None:
        span = abs(rule.tv - rule.lrv) or 0.1
        base = np.clip(np.linspace(control_rate + rule.lrv - 2 * span,
                                   control_rate + rule.tv + 2 * span, 99), 0.0, 1.0)
        grid = np.union1d(base, np.clip([control_rate + rule.lrv, control_rate + rule.tv], 0.0, 1.0))
    xs = np.arange(n_planned_t + 1)
    out = []
    for theta_t in np.asarray(grid, dtype=float):
        w_t = stats.binom.pmf(xs, n_planned_t, theta_t)
        out.append({"theta": float(theta_t - control_rate), "theta_t": float(theta_t),
                    "go_rate": float(w_t @ go @ w_c)})
    return out
