"""
Replicate Fig. 1 of Kan, Wang, and Zheng (2024 preprint):
"In-sample and Out-of-sample Sharpe Ratios of Multi-factor Asset Pricing Models"

Implements eqs. (9)-(12):
- (9)-(10): stochastic representation of (theta_hat, theta_tilde)
- (11)-(12): exact expectations E[theta_hat] and E[theta_tilde]

Also includes:
- theta (population Sharpe) from true mu,Sigma
- theta_hat from sample mu_hat,Sigma_hat
- theta_tilde from true mu,Sigma and sample mu_hat,Sigma_hat (eq. (8))

Requirements:
  pandas, numpy, scipy, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import hyp1f1, gammaln

# -----------------------------
# Sharpe ratios from parameters
# -----------------------------

def theta_hat_from_sample(mu_hat, Sigma_hat):
    """theta_hat = sqrt(mu_hat' Sigma_hat^{-1} mu_hat)."""
    mu_hat = np.asarray(mu_hat, dtype=float).reshape(-1)
    Sigma_hat = np.asarray(Sigma_hat, dtype=float)
    return float(np.sqrt(mu_hat @ np.linalg.solve(Sigma_hat, mu_hat)))

def theta_tilde_from_true_and_sample(mu_true, Sigma_true, mu_hat, Sigma_hat):
    """
    theta_tilde (eq. (8)):
      (mu_hat' Sigma_hat^{-1} mu_true) / sqrt(mu_hat' Sigma_hat^{-1} Sigma_true Sigma_hat^{-1} mu_hat)
    """
    mu_true = np.asarray(mu_true, dtype=float).reshape(-1)
    Sigma_true = np.asarray(Sigma_true, dtype=float)
    mu_hat = np.asarray(mu_hat, dtype=float).reshape(-1)
    Sigma_hat = np.asarray(Sigma_hat, dtype=float)

    a = np.linalg.solve(Sigma_hat, mu_true)          # Sigma_hat^{-1} mu_true
    b = np.linalg.solve(Sigma_hat, mu_hat)           # Sigma_hat^{-1} mu_hat
    num = mu_hat @ a
    den2 = mu_hat @ (np.linalg.solve(Sigma_hat, Sigma_true @ b))
    den = np.sqrt(den2)
    return float(num / den)

# -----------------------------------------
# Eqs. (9)-(10): stochastic representation
# -----------------------------------------

def simulate_theta_hat_tilde_stochastic(theta, N, T, size=1, seed=0):
    """
    Draw (theta_hat, theta_tilde) using eqs. (9)-(10).

    u1 ~ chi2_{T-N}
    b  ~ Beta((T-N+1)/2, (N-1)/2)
    z  | b ~ N(sqrt(b)*sqrt(T)*theta, 1)
    u  | b ~ noncentral chi2_{N-1}((1-b)*T*theta^2)

    theta_hat  = sqrt(z^2 + u) / sqrt(u1)
    theta_tilde = theta*z / sqrt(z^2 + u)
    """
    if N < 2:
        raise ValueError("need N >= 2")
    if T <= N:
        raise ValueError("need T > N")

    rng = np.random.default_rng(seed)

    u1 = rng.chisquare(df=T - N, size=size)

    a = 0.5 * (T - N + 1.0)
    bpar = 0.5 * (N - 1.0)
    b = rng.beta(a=a, b=bpar, size=size)

    mean_z = np.sqrt(b) * np.sqrt(T) * theta
    z = rng.normal(loc=mean_z, scale=1.0, size=size)

    nc = (1.0 - b) * T * theta * theta
    u = rng.noncentral_chisquare(df=N - 1, nonc=nc, size=size)

    denom = np.sqrt(z * z + u)
    theta_hat = denom / np.sqrt(u1)
    theta_tilde = theta * z / denom
    return theta_hat, theta_tilde


# -----------------------------------------
# Eqs. (11)-(12): exact expectations
# -----------------------------------------

def E_theta_hat(theta, N, T):
    """
    Eq. (11): E[theta_hat] for T >= N+2
      E = gamma((N+1)/2)*gamma((T-N-1)/2) / (gamma(N/2)*gamma((T-N)/2))
          * 1F1(-1/2; N/2; -T*theta^2/2)
    """
    if T < N + 2:
        raise ValueError("need T >= N+2 for eq. (11)")
    theta = np.asarray(theta, dtype=float)

    log_coef = (gammaln((N + 1.0) / 2.0) +
                gammaln((T - N - 1.0) / 2.0) -
                gammaln(N / 2.0) -
                gammaln((T - N) / 2.0))
    coef = np.exp(log_coef)
    return coef * hyp1f1(-0.5, N / 2.0, -0.5 * T * theta * theta)


def E_theta_tilde(theta, N, T):
    """
    Eq. (12): E[theta_tilde] for T >= N+1
      E = theta^2 * sqrt(T) * gamma((N+1)/2)*gamma((T-N+2)/2)*gamma(T/2)
          / (sqrt(2)*gamma((N+2)/2)*gamma((T-N+1)/2)*gamma((T+1)/2))
          * 1F1(1/2; (N+2)/2; -T*theta^2/2)
    """
    if T < N + 1:
        raise ValueError("need T >= N+1 for eq. (12)")
    theta = np.asarray(theta, dtype=float)

    out = np.zeros_like(theta, dtype=float)
    mask = theta != 0.0
    if not np.any(mask):
        return out

    th = theta[mask]
    log_coef = (2.0 * np.log(th) +
                0.5 * np.log(T) +
                gammaln((N + 1.0) / 2.0) +
                gammaln((T - N + 2.0) / 2.0) +
                gammaln(T / 2.0) -
                0.5 * np.log(2.0) -
                gammaln((N + 2.0) / 2.0) -
                gammaln((T - N + 1.0) / 2.0) -
                gammaln((T + 1.0) / 2.0))
    out[mask] = np.exp(log_coef) * hyp1f1(0.5, (N + 2.0) / 2.0, -0.5 * T * th * th)
    return out


# -----------------------------
# Fig. 1 replication
# -----------------------------

def replicate_figure1(theta_max=0.6, ngrid=301, savepath="fig1_replication.png", show=True):
    theta_grid = np.linspace(0.0, float(theta_max), int(ngrid))

    cases = [
        (120, 3, "-",  "T = 120, N = 3"),
        (120, 6, ":",  "T = 120, N = 6"),
        (240, 3, "--", "T = 240, N = 3"),
        (240, 6, "-.", "T = 240, N = 6"),
    ]

    plt.figure(figsize=(7.5, 4.8))
    for T, N, ls, label in cases:
        y = E_theta_tilde(theta_grid, N, T) - E_theta_hat(theta_grid, N, T)
        plt.plot(theta_grid, y, linestyle=ls, label=label)

    plt.axhline(0.0, linewidth=1.0)
    plt.xlim(0.0, theta_max)
    plt.ylim(-0.25, 0.0)
    plt.xlabel("theta")
    plt.ylabel("E[theta_tilde - theta_hat]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    if show:
        plt.show()
    return savepath


def main():

    pd.set_option("display.float_format", "{:.4f}".format)
    do_plot = True
    if do_plot:
        # Replicate Fig. 1 using eqs. (11)-(12)
        out = replicate_figure1(show=True)
        print("wrote:", out)

    # Quick numerical check mentioned in the paper text:
    # when theta=0.1, T=120, N=6, E[theta_tilde - theta_hat] approx -0.204
    theta_vec = [0.1, 0.5] # population (true) Sharpe ratio of the factor model
    N_vec = [6, 10] # cross-sectional dimension
    T_vec = [120, 1000] # number of observations
    M = 200000 # number of Monte Carlo replications
    print("M:", M)
    for theta in theta_vec:
        for N in N_vec:
            for T in T_vec:
                print("\ntheta:", theta)
                print("T:", T)
                print("N:", N)

                # Monte Carlo check via stochastic representation (9)-(10)
                th_hat, th_tilde = simulate_theta_hat_tilde_stochastic(theta=theta, N=N, T=T, size=M, seed=1)
                df_sharpe = pd.DataFrame({"tilde":[E_theta_tilde(theta,N,T), np.mean(th_tilde)],
                    "hat":[E_theta_hat(theta,N,T), np.mean(th_hat)]}, index=["exact","simulated"],
                    dtype=float)
                df_sharpe["diff"] = df_sharpe["tilde"] - df_sharpe["hat"]
                print("\nSharpe ratios:\n" + df_sharpe.to_string())

if __name__ == "__main__":
    main()

