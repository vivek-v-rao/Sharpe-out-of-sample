
"""
linear predictive models: in-sample and out-of-sample sharpe ratios

implements formulas from "in-sample and out-of-sample sharpe ratios for linear predictive models"
through figure 10.

conventions:
- signals s_t are p-vectors
- returns r_{t+1} are m-vectors
- beta is m x p
- z is m x m (risk-normalization / weighting matrix)
- sigma_s is p x p (cov of s_t), mu_s is p-vector
- sigma_eps is m x m (cov of eps_{t+1})

all sharpe ratios returned are per-period unless annualized explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
from numpy.linalg import LinAlgError


def _as_1d(x, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1d, got shape {x.shape}")
    return x


def _as_2d(x, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2d, got shape {x.shape}")
    return x


def _sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def _trace(a: np.ndarray) -> float:
    return float(np.trace(a))


def _pinv(a: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    return np.linalg.pinv(a, rcond=rcond)


def annualize_sharpe(sr_per_period: float, periods_per_year: int = 252) -> float:
    return float(sr_per_period) * math.sqrt(periods_per_year)


def deannualize_sharpe(sr_annual: float, periods_per_year: int = 252) -> float:
    return float(sr_annual) / math.sqrt(periods_per_year)


@dataclass(frozen=True)
class PnLMoments:
    mean_is: float
    var_is: float
    mean_oos: float
    var_oos: float

    @property
    def sr_is(self) -> float:
        return self.mean_is / math.sqrt(self.var_is)

    @property
    def sr_oos(self) -> float:
        return self.mean_oos / math.sqrt(self.var_oos)

    @property
    def replication_ratio(self) -> float:
        if self.sr_is == 0.0:
            return float("nan")
        return self.sr_oos / self.sr_is


def true_sharpe_ratio(
    beta: np.ndarray,
    sigma_s: np.ndarray,
    sigma_eps: np.ndarray,
    z: Optional[np.ndarray] = None,
    mu_s: Optional[np.ndarray] = None,
) -> float:
    """
    true sharpe ratio for the oracle strategy w_t = z beta s_t.

    implements eq (2.2) in the paper.
    """
    beta = _as_2d(beta, "beta")
    sigma_s = _as_2d(sigma_s, "sigma_s")
    sigma_eps = _as_2d(sigma_eps, "sigma_eps")
    m, p = beta.shape
    if sigma_s.shape != (p, p):
        raise ValueError("sigma_s shape mismatch")
    if sigma_eps.shape != (m, m):
        raise ValueError("sigma_eps shape mismatch")
    if z is None:
        z = np.eye(m)
    z = _as_2d(z, "z")
    if z.shape != (m, m):
        raise ValueError("z shape mismatch")
    if mu_s is None:
        mu_s = np.zeros(p)
    mu_s = _as_1d(mu_s, "mu_s")
    if mu_s.shape[0] != p:
        raise ValueError("mu_s shape mismatch")

    g = beta.T @ z @ beta                      # p x p
    f = beta.T @ z @ sigma_eps @ z @ beta      # p x p

    num = _trace(g @ sigma_s) + float(mu_s.T @ g @ mu_s)

    gs = g @ sigma_s
    den = (
        2.0 * _trace(gs @ gs)
        + 4.0 * float(mu_s.T @ g @ sigma_s @ g @ mu_s)
        + _trace(f @ sigma_s)
        + float(mu_s.T @ f @ mu_s)
    )
    if den <= 0.0:
        return float("nan")
    return num / math.sqrt(den)


def true_sharpe_ratio_ar1_single(
    beta_scalar: float,
    phi: float,
) -> float:
    """
    eq (3.5) specialized to m=p=1, sigma_eps=1, u_t ~ n(0,1), st = phi st-1 + u_t.

    sr = beta / sqrt(1 + 2 beta^2 - phi^2)
    """
    b = float(beta_scalar)
    ph = float(phi)
    den = 1.0 + 2.0 * b * b - ph * ph
    if den <= 0.0:
        return float("nan")
    return b / math.sqrt(den)


def univariate_closed_forms(beta_scalar: float, t1: int) -> Dict[str, float]:
    """
    univariate case (m=p=1) with sigma_s = sigma_eps (after rescaling), mu_s=0.

    returns sr_true, sreis, sreoos, rr as in eq (3.3).
    """
    b2 = float(beta_scalar) ** 2
    t1 = int(t1)
    if t1 <= 2:
        raise ValueError("t1 must be > 2 for the univariate closed forms")

    sr = b2 / math.sqrt(2.0 * b2 * b2 + b2)

    sreis_num = b2 + 1.0 / t1
    sreis_den = (
        2.0 * b2 * b2
        + (1.0 + 15.0 / (t1 - 2) - 2.0 / t1) * b2
        + 4.0 / t1
        - 3.0 / (t1 + 2)
        - 1.0 / (t1 * t1)
    )
    sreis = sreis_num / math.sqrt(sreis_den)

    sreoos_num = b2
    sreoos_den = 2.0 * b2 * b2 + (1.0 + 2.0 / (t1 - 2)) * b2 + 1.0 / (t1 - 2)
    sreoos = sreoos_num / math.sqrt(sreoos_den)

    rr = sreoos / sreis if sreis != 0.0 else float("nan")
    return {"sr_true": sr, "sreis": sreis, "sreoos": sreoos, "rr": rr}


def prop31_constants(m: int, p: int, t1: int) -> Dict[str, float]:
    """
    constants from eq (3.1) / prop 3.1.

    mapping:
    - c1, c2 are out-of-sample variance constants
    - c1_tilde, c2_tilde are extra in-sample variance terms (the "tilde c" in the paper)
    """
    m = int(m)
    p = int(p)
    t1 = int(t1)
    if t1 <= p + 1:
        raise ValueError("need t1 > p + 1")
    denom = t1 - p - 1

    c1 = 1.0 + (p + 1.0) / denom
    c1_tilde = (2.0 * p + 5.0) / denom + 2.0 * m * (p * p + p + 2.0 * t1) / (t1 * denom)

    c2 = m * p / denom
    c2_tilde = (
        m * p * (2.0 * m + p + t1 + 4.0) / (t1 * (t1 + 2.0))
        - 2.0 * (m * m) * (p * p) / (t1 * t1 * (t1 + 2.0))
        - m * p / denom
    )
    return {"c1": c1, "c2": c2, "c1_tilde": c1_tilde, "c2_tilde": c2_tilde}


def expected_pnl_moments_centered_iid(
    tr_gamma: float,
    tr_gamma2: float,
    m: int,
    p: int,
    t1: int,
    eps0: float = 0.0,
    eps1: float = 0.0,
    eps2: float = 0.0,
) -> PnLMoments:
    """
    special case in prop 3.1:
    - mu_s = 0
    - sigma_s = i_p
    - z = sigma_eps^{-1}
    then expected moments depend on tr(gamma) and tr(gamma^2).

    eps0/eps1/eps2 correspond to the (typically small) epsilon terms in prop 3.1.
    """
    const = prop31_constants(m=m, p=p, t1=t1)
    c1 = const["c1"]
    c2 = const["c2"]
    c1t = const["c1_tilde"]
    c2t = const["c2_tilde"]

    mean_is = float(tr_gamma) + (m * p) / float(t1)
    mean_oos = float(tr_gamma)

    var_oos = 2.0 * float(tr_gamma2) + c1 * float(tr_gamma) + c2
    var_is = 2.0 * float(tr_gamma2) + (c1 + c1t) * float(tr_gamma) + (c2 + c2t)
    var_is += float(eps0) + float(eps1) * float(tr_gamma) + float(eps2) * float(tr_gamma2)

    return PnLMoments(mean_is=mean_is, var_is=var_is, mean_oos=mean_oos, var_oos=var_oos)


def solve_rank1_k_for_true_sr(
    sr_annual_target: float,
    m: int,
    p: int,
    periods_per_year: int = 252,
) -> float:
    """
    choose beta = k * 1_{m,p} with sigma_eps=i, sigma_s=i, z=i so that true sr matches target.

    for rank-1 gamma (beta=k ones), sr = tr_gamma / sqrt(2 tr_gamma^2 + tr_gamma)
    with tr_gamma2 = tr_gamma^2. solve for tr_gamma, then k = sqrt(tr_gamma/(m p)).
    """
    sr = deannualize_sharpe(sr_annual_target, periods_per_year=periods_per_year)
    sr2 = sr * sr
    if sr2 >= 0.5:
        raise ValueError("target sr too large for rank-1 model (needs sr^2 < 1/2 per period)")
    tr_gamma = sr2 / (1.0 - 2.0 * sr2)
    k2 = tr_gamma / (float(m) * float(p))
    if k2 <= 0.0:
        return float("nan")
    return math.sqrt(k2)


def ols_beta_hat(
    r: np.ndarray,  # (t, m)
    s: np.ndarray,  # (t, p)
    ridge: float = 0.0,
) -> np.ndarray:
    """
    beta_hat = argmin_b sum_t || r_t - b s_t ||^2, optionally with ridge.

    returns m x p.
    """
    r = _as_2d(r, "r")
    s = _as_2d(s, "s")
    t, m = r.shape
    t2, p = s.shape
    if t2 != t:
        raise ValueError("r and s must have same number of rows")
    xtx = s.T @ s
    if ridge > 0.0:
        xtx = xtx + float(ridge) * np.eye(p)
    try:
        xtx_inv = np.linalg.inv(xtx)
    except LinAlgError:
        xtx_inv = _pinv(xtx)
    return (r.T @ s) @ xtx_inv


def simulate_predictive_model(
    beta: np.ndarray,
    sigma_s: np.ndarray,
    sigma_eps: np.ndarray,
    t1: int,
    t2: int,
    z: Optional[np.ndarray] = None,
    mu_s: Optional[np.ndarray] = None,
    seed: int = 0,
    dist: str = "normal",
    df_t: float = 5.0,
    ar1_phi: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    monte-carlo one-path simulation of the model, returning realized sr_is and sr_oos.

    dist:
    - "normal": iid gaussian signals and residuals
    - "t": iid heavy tails using a multivariate t (elliptical) with df_t
    - "ar1": signals follow diagonal ar(1) with coefficients ar1_phi, residuals iid normal

    returns:
    - sr_is, sr_oos, mean_is, std_is, mean_oos, std_oos
    """
    beta = _as_2d(beta, "beta")
    sigma_s = _as_2d(sigma_s, "sigma_s")
    sigma_eps = _as_2d(sigma_eps, "sigma_eps")
    m, p = beta.shape
    if sigma_s.shape != (p, p) or sigma_eps.shape != (m, m):
        raise ValueError("sigma shapes mismatch")
    if z is None:
        z = np.eye(m)
    z = _as_2d(z, "z")
    if z.shape != (m, m):
        raise ValueError("z shape mismatch")
    if mu_s is None:
        mu_s = np.zeros(p)
    mu_s = _as_1d(mu_s, "mu_s")
    if mu_s.shape[0] != p:
        raise ValueError("mu_s shape mismatch")

    rng = np.random.default_rng(int(seed))

    def mvn(n: int, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        return rng.multivariate_normal(mean, cov, size=n, method="svd")

    def mvt(n: int, mean: np.ndarray, cov: np.ndarray, df: float) -> np.ndarray:
        # elliptical multivariate t with "cov" as the gaussian scale matrix.
        z0 = rng.multivariate_normal(np.zeros(mean.size), cov, size=n, method="svd")
        g = rng.chisquare(df, size=n) / df
        x = z0 / np.sqrt(g)[:, None]
        return x + mean[None, :]

    if dist == "normal":
        s1 = mvn(t1, mu_s, sigma_s)
        s2 = mvn(t2, mu_s, sigma_s)
        eps1 = mvn(t1, np.zeros(m), sigma_eps)
        eps2 = mvn(t2, np.zeros(m), sigma_eps)
    elif dist == "t":
        s1 = mvt(t1, mu_s, sigma_s, df=df_t)
        s2 = mvt(t2, mu_s, sigma_s, df=df_t)
        eps1 = mvt(t1, np.zeros(m), sigma_eps, df=df_t)
        eps2 = mvt(t2, np.zeros(m), sigma_eps, df=df_t)
    elif dist == "ar1":
        if ar1_phi is None:
            raise ValueError("ar1_phi required for dist='ar1'")
        phi = np.asarray(ar1_phi, dtype=float).ravel()
        if phi.size == 1:
            phi = np.full(p, float(phi[0]))
        if phi.size != p:
            raise ValueError("ar1_phi must have length p or be scalar")
        # innovations with cov chosen so that cov(s_t)=sigma_s if possible when sigma_s is diagonal
        # for general sigma_s this is not exact; intended for the figure 7 setup (p=1 or diagonal).
        if not np.allclose(sigma_s, np.diag(np.diag(sigma_s)), atol=1e-10):
            raise ValueError("dist='ar1' currently supports diagonal sigma_s only")
        sig_diag = np.diag(sigma_s)
        u_var = sig_diag * (1.0 - phi * phi)
        u_var = np.maximum(u_var, 0.0)
        u_cov = np.diag(u_var)
        s = np.zeros((t1 + t2, p))
        s[0] = mvn(1, mu_s, sigma_s)[0]
        for t in range(1, t1 + t2):
            u = mvn(1, np.zeros(p), u_cov)[0]
            s[t] = mu_s + phi * (s[t - 1] - mu_s) + u
        s1 = s[:t1]
        s2 = s[t1:]
        eps1 = mvn(t1, np.zeros(m), sigma_eps)
        eps2 = mvn(t2, np.zeros(m), sigma_eps)
    else:
        raise ValueError("unknown dist")

    r1 = (s1 @ beta.T) + eps1     # (t1, m), since beta is m x p
    r2 = (s2 @ beta.T) + eps2

    beta_hat = ols_beta_hat(r=r1, s=s1, ridge=0.0)  # m x p

    def pnl_from(r: np.ndarray, s: np.ndarray) -> np.ndarray:
        w = (s @ beta_hat.T) @ z.T   # (t, m) because (t,p)*(p,m) = (t,m); then times z'
        # but w_t should be z beta_hat s_t (m,), so w = s @ beta_hat.T @ z.T is correct if z symmetric
        pnl = np.sum(w * r, axis=1)
        return pnl

    pnl1 = pnl_from(r1, s1)
    pnl2 = pnl_from(r2, s2)

    mean1 = float(np.mean(pnl1))
    mean2 = float(np.mean(pnl2))
    std1 = float(np.std(pnl1, ddof=1))
    std2 = float(np.std(pnl2, ddof=1))
    sr1 = mean1 / std1 if std1 > 0.0 else float("nan")
    sr2 = mean2 / std2 if std2 > 0.0 else float("nan")
    return {
        "sr_is": sr1,
        "sr_oos": sr2,
        "mean_is": mean1,
        "std_is": std1,
        "mean_oos": mean2,
        "std_oos": std2,
    }


def expected_pnl_moments_general(
    beta: np.ndarray,
    sigma_s: np.ndarray,
    sigma_eps: np.ndarray,
    t1: int,
    z: Optional[np.ndarray] = None,
    mu_s: Optional[np.ndarray] = None,
    include_eps_terms: bool = False,
) -> PnLMoments:
    """
    proposition b.1: expected in-sample and out-of-sample mean and variance of pnl.

    this implements the formulas used in eq (3.2) (with eps terms set to 0 by default).
    """
    beta = _as_2d(beta, "beta")
    sigma_s = _as_2d(sigma_s, "sigma_s")
    sigma_eps = _as_2d(sigma_eps, "sigma_eps")
    t1 = int(t1)
    m, p = beta.shape
    if sigma_s.shape != (p, p):
        raise ValueError("sigma_s shape mismatch")
    if sigma_eps.shape != (m, m):
        raise ValueError("sigma_eps shape mismatch")
    if z is None:
        z = np.linalg.inv(sigma_eps)
    z = _as_2d(z, "z")
    if z.shape != (m, m):
        raise ValueError("z shape mismatch")
    if mu_s is None:
        mu_s = np.zeros(p)
    mu_s = _as_1d(mu_s, "mu_s")
    if mu_s.shape[0] != p:
        raise ValueError("mu_s shape mismatch")
    if t1 <= p + 1:
        raise ValueError("need t1 > p + 1")

    sigma_s = _sym(sigma_s)
    sigma_eps = _sym(sigma_eps)
    z = _sym(z)

    mu_outer = np.outer(mu_s, mu_s)
    sp = sigma_s + mu_outer

    try:
        sp_inv = np.linalg.inv(sp)
    except LinAlgError:
        sp_inv = _pinv(sp)

    g = beta.T @ z @ beta
    f = beta.T @ z @ sigma_eps @ z @ beta

    tr_zse = _trace(z @ sigma_eps)                 # tr(z sigma_eps)
    tr_seszse = _trace(sigma_eps @ z @ sigma_eps @ z)  # tr(sigma_eps z sigma_eps z)
    tr_zse_sq = tr_zse * tr_zse

    # means
    mean_oos = _trace(g @ sigma_s) + float(mu_s.T @ g @ mu_s)
    mean_is = mean_oos + (p / float(t1)) * tr_zse

    # base variance terms
    gs = g @ sigma_s
    var_base = (
        2.0 * _trace(gs @ gs)
        + 4.0 * float(mu_s.T @ g @ sigma_s @ g @ mu_s)
        + _trace(f @ sigma_s)
        + float(mu_s.T @ f @ mu_s)
    )

    # out-of-sample variance extra terms
    d = sp_inv / float(t1 - p - 1)

    # mu' [2 d sigma_s^t f + tr(f sigma_s^t) d + tr(sigma_s^t d) f] mu
    mu_term = float(mu_s.T @ (2.0 * d @ sigma_s.T @ f + _trace(f @ sigma_s.T) * d + _trace(sigma_s.T @ d) * f) @ mu_s)

    tr_terms = (
        _trace(d @ sigma_s.T @ f @ sigma_s)
        + _trace(d @ sigma_s) * _trace(f @ sigma_s.T)
        + _trace(z @ sigma_eps @ z @ sigma_eps) * (_trace(sigma_s.T @ d) + float(mu_s.T @ d @ mu_s))
    )

    var_oos = var_base + mu_term + tr_terms

    # in-sample variance extra terms (eps terms ignored by default)
    denom = float(t1 - p - 1)

    # term: tr( 2 f (sigma_s+mu mu') + (mu' f mu) (sigma_s+mu mu')^{-1} (sigma_s - mu mu') + p f sigma_s )
    term_f = _trace(
        2.0 * f @ sp
        + float(mu_s.T @ f @ mu_s) * (sp_inv @ (sigma_s - mu_outer))
        + float(p) * f @ sigma_s
    )
    var_is = var_base + (3.0 / denom) * term_f

    mu2 = float(mu_s.T @ mu_s)

    # tr(sigma_eps z sigma_eps z) * [ p(p+t1+4)/(t1(t1+2)) - 2 mu2 p (p-t1-2)(p-t1) / (t1^2 (t1+1)(t1+2)) ]
    a = float(p) * (float(p) + float(t1) + 4.0) / (float(t1) * (float(t1) + 2.0))
    b = 2.0 * mu2 * float(p) * (float(p) - float(t1) - 2.0) * (float(p) - float(t1)) / (float(t1) ** 2 * (float(t1) + 1.0) * (float(t1) + 2.0))
    var_is += tr_seszse * (a - b)

    # - tr(z sigma_eps)^2 * 2(p-t1) * ((p-2) mu2 + t1 ((p-1) mu2 + p)) / (t1^2 (t1+1)(t1+2))
    c_num = 2.0 * (float(p) - float(t1)) * ((float(p) - 2.0) * mu2 + float(t1) * ((float(p) - 1.0) * mu2 + float(p)))
    c_den = float(t1) ** 2 * (float(t1) + 1.0) * (float(t1) + 2.0)
    var_is -= tr_zse_sq * (c_num / c_den)

    # + 2/(t1-p-1) tr(z sigma_eps) tr(2 g (sigma_s+mu mu') + (mu' g mu) (sigma_s+mu mu')^{-1} (sigma_s - mu mu') + p g sigma_s)
    term_g = _trace(
        2.0 * g @ sp
        + float(mu_s.T @ g @ mu_s) * (sp_inv @ (sigma_s - mu_outer))
        + float(p) * g @ sigma_s
    )
    var_is += (2.0 / denom) * tr_zse * term_g

    # - (2p/t1) tr(z sigma_eps) (tr(g sigma_s) + mu' g mu)
    var_is -= (2.0 * float(p) / float(t1)) * tr_zse * mean_oos

    if include_eps_terms:
        # the eps terms in prop b.1 are covariances involving the inverse of s s^t.
        # the paper sets eps=0 for the analysis (see discussion near eq (3.2)).
        raise NotImplementedError("include_eps_terms=True is not implemented (paper takes eps=0)")

    return PnLMoments(mean_is=mean_is, var_is=var_is, mean_oos=mean_oos, var_oos=var_oos)

def kwz_expected_sharpes(theta, m, t):
    # theta: true Sharpe ratio per period (sqrt(mu' Sigma^{-1} mu))
    # m: number of assets
    # t: sample size (training length)

    from scipy.special import gammaln, hyp1f1

    z = -0.5 * t * theta * theta

    # E[theta_hat]
    log_hat = (gammaln((m + 1) / 2) + gammaln((t - m - 1) / 2)
               - gammaln(m / 2) - gammaln((t - m) / 2))
    e_hat = math.exp(float(log_hat)) * hyp1f1(-0.5, m / 2, z)

    # E[theta_tilde]
    if theta == 0.0:
        e_tilde = 0.0
    else:
        log_tilde = (2.0 * math.log(theta) + 0.5 * math.log(t)
                     + gammaln((m + 1) / 2) + gammaln((t - m + 2) / 2) + gammaln(t / 2)
                     - 0.5 * math.log(2.0) - gammaln((m + 2) / 2) - gammaln((t - m + 1) / 2) - gammaln((t + 1) / 2))
        e_tilde = math.exp(float(log_tilde)) * hyp1f1(0.5, (m + 2) / 2, z)

    return {"e_hat": float(e_hat), "e_tilde": float(e_tilde), "rr": float(e_tilde / e_hat)}

def replication_ratio_from_general_moments(mom: PnLMoments) -> float:
    return mom.replication_ratio
