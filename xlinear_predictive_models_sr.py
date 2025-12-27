"""
generate numerical reproductions for figures 3-8 (and helpers for 9-10) of the paper
"In-Sample and Out-of-Sample Sharpe Ratios for Linear Predictive Models" by Mulligan
et al.

notes:
- this script depends only on numpy/matplotlib and the module linear_predictive_models_sr.py
- figures 9-10 in the paper depend on an empirical commodity dataset; here we provide simulation hooks
  and placeholders for plugging in real data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import linear_predictive_models_sr as lpm

save_figures = True
fig_dir = "figures"

def _savefig(name, dpi=200):
    if save_figures:
        os.makedirs(fig_dir, exist_ok=True)
        plt.gcf().savefig(os.path.join(fig_dir, f"{name}.png"), dpi=dpi)

def fig5_univariate_heatmap(t1_grid, sr_grid_annual, periods_per_year=252):
    # use univariate closed forms (3.3). we parameterize by true sr (annual) and t1.
    rr = np.zeros((len(t1_grid), len(sr_grid_annual)))
    for i, t1 in enumerate(t1_grid):
        for j, sr_ann in enumerate(sr_grid_annual):
            # solve for beta via rank-1 relationship (m=p=1)
            beta = lpm.solve_rank1_k_for_true_sr(sr_ann, m=1, p=1, periods_per_year=periods_per_year)
            out = lpm.univariate_closed_forms(beta_scalar=beta, t1=t1)
            rr[i, j] = out["rr"]
    plt.figure()
    plt.imshow(rr, origin="lower", aspect="auto",
               extent=[sr_grid_annual[0], sr_grid_annual[-1], t1_grid[0], t1_grid[-1]])
    plt.colorbar(label="replication ratio (oos/is)")
    plt.xlabel("true sharpe ratio (annual)")
    plt.ylabel("t1")
    plt.title("figure 5 style: univariate replication ratio heatmap (eq 3.3)")
    plt.tight_layout()
    _savefig("fig5_univariate_heatmap")
    plt.show()

def fig5_univariate_curves(t1_list, sr_grid_annual, periods_per_year=252):
    # figure 5 (panel 1) style: replication ratio vs true sharpe ratio, with separate curves by t1
    plt.figure()
    for t1 in t1_list:
        rr = np.zeros(len(sr_grid_annual))
        for j, sr_ann in enumerate(sr_grid_annual):
            beta = lpm.solve_rank1_k_for_true_sr(sr_ann, m=1, p=1, periods_per_year=periods_per_year)
            rr[j] = lpm.univariate_closed_forms(beta_scalar=beta, t1=t1)["rr"]
        plt.plot(sr_grid_annual, rr, label=f"T1 = {t1}")
    plt.margins(x=0.0)
    plt.xlim(0.0, float(sr_grid_annual[-1]))
    plt.xlabel("true sharpe ratio (annual)")
    plt.ylabel("replication ratio (oos/is)")
    plt.title("figure 5 panel 1 style: univariate replication ratio vs sharpe (eq 3.3)")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    _savefig("fig5_univariate_curves")
    plt.show()

def fig2_region_c2tilde_positive(m_vals, p_vals, t1_vals):
    # plots indicator where c2_tilde > 0 (eq 3.1)
    for t1 in t1_vals:
        mask = np.zeros((len(m_vals), len(p_vals)))
        for i, m in enumerate(m_vals):
            for j, p in enumerate(p_vals):
                if t1 <= p + 1:
                    mask[i, j] = np.nan
                    continue
                c = lpm.prop31_constants(m=m, p=p, t1=t1)
                mask[i, j] = 1.0 if (c["c2_tilde"] > 0.0) else 0.0
        plt.figure()
        plt.imshow(mask, origin="lower", aspect="auto",
                   extent=[p_vals[0], p_vals[-1], m_vals[0], m_vals[-1]])
        plt.colorbar(label="1 if c2_tilde>0 else 0")
        plt.xlabel("p: # of signals")
        plt.ylabel("m")
        plt.title(f"figure 2 style: region where c2_tilde>0 (t1={t1})")
        plt.tight_layout()
        _savefig(f"fig2_c2tilde_positive_t1_{t1}")
        plt.show()


def fig6_multivariate_replication_ratio(m_list, p_list, t1=2520, sr_is_annual=2.0):
    # use beta = k ones_{m,p}, sigma_s=i, sigma_eps=i, z=i, mu_s=0.
    rr = np.zeros((len(m_list), len(p_list)))
    for i, m in enumerate(m_list):
        for j, p in enumerate(p_list):
            # choose k to hit in-sample expected sharpe (annual) approximately by root find on k.
            # use monotonicity in tr_gamma = k^2 m p.
            target = lpm.deannualize_sharpe(sr_is_annual)
            # bracket k
            k_lo, k_hi = 0.0, 10.0
            for _ in range(60):
                k_mid = 0.5 * (k_lo + k_hi)
                trg = (k_mid * k_mid) * m * p
                mom = lpm.expected_pnl_moments_centered_iid(tr_gamma=trg, tr_gamma2=trg*trg, m=m, p=p, t1=t1)
                if mom.sr_is < target:
                    k_lo = k_mid
                else:
                    k_hi = k_mid
            k = k_hi
            trg = (k * k) * m * p
            mom = lpm.expected_pnl_moments_centered_iid(tr_gamma=trg, tr_gamma2=trg*trg, m=m, p=p, t1=t1)
            rr[i, j] = mom.replication_ratio
    plt.figure()
    plt.imshow(rr, origin="lower", aspect="auto",
               extent=[p_list[0], p_list[-1], m_list[0], m_list[-1]], vmin=0.0, vmax=1.0)
    plt.colorbar(label="replication ratio (oos/is)")
    plt.xlabel("p: # of signals")
    plt.ylabel("m")
    plt.title(f"figure 6 style: replication ratio by m,p (t1={t1}, sr_is={sr_is_annual} annual)")
    plt.tight_layout()
    _savefig("fig6_multivariate_replication_ratio")
    plt.show()


def fig7_ar1_simulation(phi_list, t1_list, nsim=2000, sr_true_annual=1.0, seed=0):
    # one asset, one signal. choose beta from true sr under ar(1) eq (3.5) via numeric solve.
    rng = np.random.default_rng(seed)
    out = {}
    for t1 in t1_list:
        out[t1] = {"phi": [], "rr_mc": []}
        for phi in phi_list:
            # solve for beta so that true sr matches target under eq (3.5)
            target = lpm.deannualize_sharpe(sr_true_annual)
            b_lo, b_hi = 0.0, 20.0
            for _ in range(80):
                b_mid = 0.5 * (b_lo + b_hi)
                sr_mid = lpm.true_sharpe_ratio_ar1_single(b_mid, phi)
                if sr_mid < target:
                    b_lo = b_mid
                else:
                    b_hi = b_mid
            beta = b_hi

            # simulation setup: sigma_eps=1, sigma_s determined by phi with u var=1
            sigma_s = np.array([[1.0 / (1.0 - phi * phi)]])
            sigma_eps = np.array([[1.0]])
            beta_mat = np.array([[beta]])

            rr_vals = []
            for k in range(nsim):
                sim = lpm.simulate_predictive_model(
                    beta=beta_mat, sigma_s=sigma_s, sigma_eps=sigma_eps,
                    t1=t1, t2=t1, seed=int(rng.integers(0, 2**31-1)),
                    dist="ar1", ar1_phi=np.array([phi])
                )
                rr_vals.append(sim["sr_oos"] / sim["sr_is"])
            out[t1]["phi"].append(phi)
            out[t1]["rr_mc"].append(float(np.mean(rr_vals)))

    # plot
    plt.figure()
    for t1 in t1_list:
        plt.plot(out[t1]["phi"], out[t1]["rr_mc"], label=f"t1={t1}")
    plt.xlabel("phi (signal persistence)")
    plt.ylabel("replication ratio (mc mean)")
    plt.title(f"figure 7 style: ar(1) replication ratio (sr_true={sr_true_annual} annual)")
    plt.legend()
    plt.tight_layout()
    _savefig("fig7_ar1_simulation")
    plt.show()

def fig8_kwz_comparison(m_list, t=2520, sr_true_annual=1.5, periods_per_year=252):
    # kwz: single infinitely persistent signal per asset.
    theta = lpm.deannualize_sharpe(sr_true_annual, periods_per_year=periods_per_year)
    rr_kwz = []
    rr_ours_p1 = []
    rr_ours_p10 = []

    for m in m_list:
        rr_kwz.append(lpm.kwz_expected_sharpes(theta=theta, m=m, t=t)["rr"])

        # ours, p=1 and p=10: use centered iid special case with rank-1 beta=k ones
        for p, vec in [(1, rr_ours_p1), (10, rr_ours_p10)]:
            k = lpm.solve_rank1_k_for_true_sr(sr_true_annual, m=m, p=p, periods_per_year=periods_per_year)
            trg = (k * k) * m * p
            mom = lpm.expected_pnl_moments_centered_iid(tr_gamma=trg, tr_gamma2=trg*trg, m=m, p=p, t1=t)
            vec.append(mom.replication_ratio)

    plt.figure()
    plt.plot(m_list, rr_kwz, label="kwz")
    plt.plot(m_list, rr_ours_p1, label="ours, p=1")
    plt.plot(m_list, rr_ours_p10, label="ours, p=10")
    plt.xlabel("m: # of assets")
    plt.ylabel("replication ratio (oos/is)")
    plt.title(f"figure 8 style: kwz vs ours (t={t}, true sr={sr_true_annual} annual)")
    plt.legend()
    plt.tight_layout()
    _savefig("fig8_kwz_comparison")
    plt.show()

def main():
    do_fig2 = True
    do_fig5 = True
    do_fig5_curves = True
    do_fig6 = True
    do_fig7 = False
    do_fig8 = True
    
    # figure 2 style
    if do_fig2:
        fig2_region_c2tilde_positive(m_vals=np.arange(1, 1001, 25), p_vals=np.arange(1, 1001, 25), t1_vals=[252, 1260, 2520])

    # figure 5 style
    if do_fig5:
        fig5_univariate_heatmap(t1_grid=np.array([252, 500, 750, 1250, 1750, 2500]), sr_grid_annual=np.linspace(0.0, 3.0, 61))

    if do_fig5_curves:
        fig5_univariate_curves(t1_list=[250, 750, 2500],
            sr_grid_annual=np.linspace(0.0, 3.0, 301))

    # figure 6 style
    if do_fig6:
        fig6_multivariate_replication_ratio(m_list=[1, 5, 10, 20, 30], p_list=list(range(1, 31)), t1=2520, sr_is_annual=2.0)

    # figure 7 style
    if do_fig7: # very slow
        fig7_ar1_simulation(phi_list=np.linspace(0.0, 0.95, 20), t1_list=[252, 1260], nsim=500, sr_true_annual=1.0)

    if do_fig8:
        fig8_kwz_comparison(m_list=list(range(1, 51)),  # adjust to match the paper's m-range if desired
            t=2520, sr_true_annual=1.5)

if __name__ == "__main__":
    main()
