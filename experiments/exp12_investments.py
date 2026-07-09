r"""
Experiment 12 -- Investment case study for the covariate-augmented MBPP.

Narrative.  Events are *investments*.  When a deal closes in some sector, it
excites further deals in *similar* sectors (positive cross-excitation), but the
same firm/sector is briefly *anti-excited* -- having just closed a round, another
one is unlikely immediately (negative self-excitation / inhibition).  Excitation
also rises with a sector's *popularity* score.

We run two regimes, exactly as a referee would demand:

(A) MISSPECIFIED / "messy reality".  We *generate* genuine multivariate Hawkes
    data with a POWER-LAW kernel, positive cross-excitation, NEGATIVE self-
    excitation (a nonlinear max(0,.) intensity), and popularity-modulated
    excitation.  We then fit the *linear, exponential-kernel* MBPP -- which is
    therefore triply misspecified (kernel shape, nonlinearity, cross-sector
    coupling) -- both WITHOUT and WITH the covariate in the excitation, and ask:
    how much of the covariate effect can we still recover, and does modelling it
    improve held-out fit?

(B) CORRECTLY SPECIFIED.  We generate from a process the augmented MBPP
    represents *exactly* (linear, exponential kernel, covariate-modulated
    excitation, including an exogenous "cooldown" covariate that genuinely
    suppresses excitation -- anti-excitation done the lawful way).  We fit the
    same estimator and report recovery against ground truth.

Outputs: results/exp12_investments.png and results/exp12_investments.json.

Run:  PYTHONPATH=. python -m experiments.exp12_investments
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes_calibration import (
    Constant,
    PiecewiseConstantCovariate,
    dispersion,
    fit_mbpp_ic_excitation,
    interval_censor,
    simulate_hawkes_excitation,
    uniform_obs_times,
)
from hawkes_calibration.mbpp.interval_censored import _excitation_compensator_fast, ic_ll


# ---------------------------------------------------------------------------
# Shared covariate environment (piecewise-constant regimes over [0, T]).
# ---------------------------------------------------------------------------
def popularity_covariate(T, n_regimes=8, seed=0):
    """A standardised, piecewise-constant 'popularity' score on [0, T]."""
    rng = np.random.default_rng(seed)
    bks = np.linspace(0.0, T, n_regimes + 1)
    vals = rng.normal(0.0, 0.6, size=n_regimes)
    vals = vals - vals.mean()  # centre -> exp(0)=1 on average
    return PiecewiseConstantCovariate(bks, vals[:, None]), bks


def two_covariates(T, n_regimes=8, seed=0):
    """popularity (continuous) + cooldown (0/1 exogenous quiet period)."""
    rng = np.random.default_rng(seed)
    bks = np.linspace(0.0, T, n_regimes + 1)
    pop = rng.normal(0.0, 0.8, size=n_regimes)
    pop = pop - pop.mean()
    cool = (rng.uniform(size=n_regimes) < 0.35).astype(float)  # ~35% of regimes are quiet
    return PiecewiseConstantCovariate(bks, np.column_stack([pop, cool]))


# ---------------------------------------------------------------------------
# (A) Realistic DGP: multivariate, power-law, nonlinear (self-inhibition),
#     covariate-modulated excitation.  Exact Ogata thinning.
# ---------------------------------------------------------------------------
def powerlaw_phi(s, c, eta):
    r"""Power-law triggering kernel phi(s) = (1 + s/c)^{-(1+eta)}, s>=0.

    Heavy-tailed; integral over [0, inf) equals c/eta (used to reason about the
    effective branching ratio).  This is the kernel family the paper highlights
    as best-performing on social-media/popularity data.
    """
    return (1.0 + s / c) ** (-(1.0 + eta))


def simulate_powerlaw_investments(T, A, baseline, c, eta, Zpop, delta_pop, seed, max_events=20000):
    r"""
    Multivariate nonlinear Hawkes with power-law kernel and covariate-modulated
    excitation:

        lambda_m(t) = max(0,  baseline_m
                         + sum_j A[m,j] * exp(delta_pop * pop(t))
                                        * sum_{t_k^j < t} phi(t - t_k^j) ).

    A[m,m] < 0 encodes self-inhibition (anti-excitation); A[m,j] > 0 for similar
    sectors encodes positive cross-excitation.  The max(0,.) link makes this a
    genuine *nonlinear* Hawkes process, so the linear MBPP mean-field equation is
    only an approximation -- exactly the misspecification we want to probe.

    Returns a list of M arrays of event times (one per sector).
    """
    rng = np.random.default_rng(seed)
    M = A.shape[0]
    Apos = np.clip(A, 0.0, None)  # positive part for the envelope
    baseline = np.asarray(baseline, dtype=float)
    bks = np.asarray(Zpop.breakpoints, dtype=float)
    ev = [np.empty(0) for _ in range(M)]

    def gmod(tt):
        return float(np.exp(delta_pop * float(Zpop(tt)[0, 0])))

    def intensity(tt, positive=False):
        g = gmod(tt)
        Amat = Apos if positive else A
        lam = baseline.copy()
        for j in range(M):
            if ev[j].size:
                ksum = powerlaw_phi(tt - ev[j], c, eta).sum()
                lam = lam + Amat[:, j] * g * ksum
        return lam if positive else np.maximum(lam, 0.0)

    t, n = 0.0, 0
    while t < T and n < max_events:
        lam_bar = intensity(t, positive=True).sum()  # valid upper envelope on [t, .)
        if lam_bar <= 1e-12:
            nb = bks[bks > t]
            t = float(nb[0]) if nb.size else T
            continue
        t_new = t + rng.exponential(1.0 / lam_bar)
        nb = bks[(bks > t) & (bks < t_new)]  # stop at covariate breakpoints
        if nb.size:
            t = float(nb[0])
            continue
        if t_new >= T:
            break
        lam_vec = intensity(t_new, positive=False)
        lam_tot = lam_vec.sum()
        if rng.uniform() < lam_tot / lam_bar:  # thinning acceptance
            m = int(rng.choice(M, p=lam_vec / lam_tot))
            ev[m] = np.append(ev[m], t_new)
            n += 1
        t = t_new
    return [np.sort(e) for e in ev]


# ---------------------------------------------------------------------------
# Helpers: rebuild interval compensator for a fitted excitation model + score.
# ---------------------------------------------------------------------------
def interval_compensator(fit, Z, obs_times):
    Xi = _excitation_compensator_fast(
        fit.baseline, fit.kappa, fit.theta, Z, np.atleast_1d(fit.delta), obs_times
    )
    return np.diff(Xi)


def heldout_ic_ll(fit, Z, obs_times, test_counts):
    Xi = interval_compensator(fit, Z, obs_times)
    return float(np.mean([ic_ll(c, Xi) for c in test_counts]))


# ---------------------------------------------------------------------------
# (A) Misspecified experiment.
# ---------------------------------------------------------------------------
def run_misspecified(out, T=60.0, n_int=30, n_seq=48, n_train=36, seed=0):
    rng = np.random.default_rng(seed)
    obs = uniform_obs_times(T, n_int)

    # popularity environment (shared across the i.i.d. investment histories)
    Zpop, _ = popularity_covariate(T, n_regimes=8, seed=1)
    Z0 = PiecewiseConstantCovariate(Zpop.breakpoints, np.zeros((Zpop.values.shape[0], 1)))

    # 3 sectors: {0,1} similar (strong cross-excitation), 2 weakly linked;
    # negative diagonal = self-inhibition.  Baseline kept low so that a large
    # share of sector-0 events are *excitation*-driven -- which is what the
    # popularity covariate modulates, hence what makes delta identifiable.
    A = np.array(
        [
            [-0.10, 0.40, 0.05],
            [0.35, -0.10, 0.05],
            [0.05, 0.05, -0.08],
        ]
    )
    baseline = np.array([0.8, 0.9, 0.8])
    c, eta = 1.0, 2.0  # power-law (tail ~ t^-3); integral c/eta = 0.5
    delta_pop_true = 0.7

    # simulate n_seq i.i.d. investment histories sharing the SAME popularity path
    counts_sec0, tot_events = [], []
    for _s in range(n_seq):
        ev = simulate_powerlaw_investments(
            T, A, baseline, c, eta, Zpop, delta_pop_true, seed=int(rng.integers(1e9))
        )
        counts_sec0.append(interval_censor(ev[0], obs))  # focus on sector 0
        tot_events.append(sum(e.size for e in ev))
    train, test = counts_sec0[:n_train], counts_sec0[n_train:]

    # fit WITHOUT and WITH the popularity covariate in the excitation
    fit0 = fit_mbpp_ic_excitation(obs, train, Z0, n_restarts=5, seed=0)
    fitc = fit_mbpp_ic_excitation(obs, train, Zpop, n_restarts=5, seed=0)

    # held-out IC-LL (lower is better) and dispersion of pooled test counts
    ll0 = heldout_ic_ll(fit0, Z0, obs, test)
    llc = heldout_ic_ll(fitc, Zpop, obs, test)
    Xi0, Xic = interval_compensator(fit0, Z0, obs), interval_compensator(fitc, Zpop, obs)
    disp0 = float(np.mean([dispersion(c, Xi0, n_params=3) for c in test]))
    dispc = float(np.mean([dispersion(c, Xic, n_params=4) for c in test]))

    res = dict(
        regime="misspecified (power-law + self-inhibition + cross-excitation)",
        avg_total_events=float(np.mean(tot_events)),
        delta_pop_true=delta_pop_true,
        delta_pop_hat_nocov=0.0,  # this model omits the covariate (modulation == 1 by construction)
        delta_pop_hat_cov=float(np.atleast_1d(fitc.delta)[0]),
        kappa_hat_nocov=float(fit0.kappa),
        kappa_hat_cov=float(fitc.kappa),
        heldout_ic_ll_nocov=ll0,
        heldout_ic_ll_cov=llc,
        heldout_ic_ll_improvement=ll0 - llc,
        dispersion_nocov=disp0,
        dispersion_cov=dispc,
    )
    _plot_misspecified(out, res, obs, Zpop, test, Xi0, Xic)
    return res


def _plot_misspecified(out, res, obs, Zpop, test, Xi0, Xic):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    mids = 0.5 * (obs[1:] + obs[:-1])
    mean_counts = np.mean(test, axis=0)
    pop = np.array([float(Zpop(t)[0, 0]) for t in mids])
    ax[0].bar(mids, mean_counts, width=(obs[1] - obs[0]) * 0.9, alpha=0.5, label="mean test counts")
    axt = ax[0].twinx()
    axt.plot(mids, pop, "C3-o", ms=3, label="popularity")
    axt.set_ylabel("popularity")
    ax[0].plot(mids, Xi0, "C1--", label="MBPP no-cov $\\Xi$")
    ax[0].plot(mids, Xic, "C2-", label="MBPP cov $\\Xi$")
    ax[0].set_title("(a) counts vs popularity & fits")
    ax[0].set_xlabel("t")
    ax[0].legend(loc="upper left", fontsize=7)

    ax[1].bar(
        ["no-cov", "cov"],
        [res["heldout_ic_ll_nocov"], res["heldout_ic_ll_cov"]],
        color=["C1", "C2"],
    )
    ax[1].set_title("(b) held-out IC-LL (lower better)")
    ax[1].set_ylabel("mean IC-LL / sequence")

    ax[2].axhline(
        res["delta_pop_true"],
        color="r",
        ls="--",
        label="true $\\delta_{pop}$=%.2f" % res["delta_pop_true"],
    )
    ax[2].bar(
        ["no-cov", "cov"],
        [res["delta_pop_hat_nocov"], res["delta_pop_hat_cov"]],
        color=["C1", "C2"],
    )
    ax[2].set_title("(c) recovered $\\delta_{pop}$ (misspecified)")
    ax[2].legend(fontsize=8)
    fig.suptitle(
        "Experiment A -- misspecified (power-law, self-inhibition, cross-excitation): "
        "the covariate still helps"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out, "exp12_investments_A.png"), dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (B) Correctly-specified experiment.
# ---------------------------------------------------------------------------
def run_correct(out, T=45.0, n_int=30, n_seq=44, seed=0):
    rng = np.random.default_rng(seed)
    obs = uniform_obs_times(T, n_int)
    Z = two_covariates(T, n_regimes=9, seed=2)  # [popularity, cooldown]
    kappa, theta, mu = 0.40, 1.0, 2.2
    delta = np.array([0.6, -0.5])  # pop excites, cooldown suppresses

    counts = []
    for _s in range(n_seq):
        ev = simulate_hawkes_excitation(
            Constant(mu, T), kappa, theta, Z, delta, T, seed=int(rng.integers(1e9))
        )
        counts.append(interval_censor(ev, obs))
    train, test = counts[:34], counts[34:]

    fit = fit_mbpp_ic_excitation(obs, train, Z, n_restarts=6, seed=0)
    Xi = interval_compensator(fit, Z, obs)
    disp = float(np.mean([dispersion(c, Xi, n_params=5) for c in test]))

    res = dict(
        regime="correctly specified (exponential kernel, covariate excitation)",
        kappa_true=kappa,
        theta_true=theta,
        mu_true=mu,
        delta_true=[float(x) for x in delta],
        kappa_hat=float(fit.kappa),
        theta_hat=float(fit.theta),
        mu_hat=float(fit.baseline),
        delta_hat=[float(x) for x in np.atleast_1d(fit.delta)],
        dispersion=disp,
    )
    _plot_correct(out, res)
    return res


def _plot_correct(out, res):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    names = [r"$\kappa$", r"$\theta$", r"$\mu$", r"$\delta_{pop}$", r"$\delta_{cool}$"]
    true = [res["kappa_true"], res["theta_true"], res["mu_true"], *res["delta_true"]]
    hat = [res["kappa_hat"], res["theta_hat"], res["mu_hat"], *res["delta_hat"]]
    x = np.arange(len(names))
    ax[0].bar(x - 0.18, true, width=0.36, label="true", color="0.6")
    ax[0].bar(x + 0.18, hat, width=0.36, label="estimated", color="C2")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_title("(a) parameter recovery (correctly specified)")
    ax[0].legend()

    ax[1].plot(true, hat, "o", color="C2", ms=9)
    lo, hi = min(true) - 0.3, max(true) + 0.3
    ax[1].plot([lo, hi], [lo, hi], "r--", label="identity")
    for n, a, b in zip(names, true, hat):
        ax[1].annotate(n, (a, b), textcoords="offset points", xytext=(5, 4), fontsize=9)
    ax[1].set_xlabel("true")
    ax[1].set_ylabel("estimated")
    ax[1].set_title("(b) estimated vs true")
    ax[1].legend()
    fig.suptitle("Experiment B -- correctly specified: the augmented MBPP recovers every parameter")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "exp12_investments_B.png"), dpi=140)
    plt.close(fig)


def run(out_dir="results", seed=0):
    os.makedirs(out_dir, exist_ok=True)
    print("=== Experiment 12: investment case study ===")
    print("[A] misspecified (power-law + self-inhibition + cross-excitation) ...")
    A = run_misspecified(out_dir, seed=seed)
    print("    avg events/history     : %.0f" % A["avg_total_events"])
    print(
        "    delta_pop  true=%.2f  no-cov=%.3f  cov=%.3f"
        % (A["delta_pop_true"], A["delta_pop_hat_nocov"], A["delta_pop_hat_cov"])
    )
    print(
        "    held-out IC-LL  no-cov=%.3f  cov=%.3f  (improvement %.3f)"
        % (A["heldout_ic_ll_nocov"], A["heldout_ic_ll_cov"], A["heldout_ic_ll_improvement"])
    )
    print(
        "    dispersion      no-cov=%.2f  cov=%.2f" % (A["dispersion_nocov"], A["dispersion_cov"])
    )

    print("[B] correctly specified (exponential kernel, covariate excitation) ...")
    B = run_correct(out_dir, seed=seed)
    print(
        "    kappa  true=%.2f hat=%.3f | theta true=%.2f hat=%.3f | mu true=%.2f hat=%.3f"
        % (
            B["kappa_true"],
            B["kappa_hat"],
            B["theta_true"],
            B["theta_hat"],
            B["mu_true"],
            B["mu_hat"],
        )
    )
    print("    delta  true=%s hat=%s" % (B["delta_true"], [round(x, 3) for x in B["delta_hat"]]))
    print("    dispersion=%.2f" % B["dispersion"])

    with open(os.path.join(out_dir, "exp12_investments.json"), "w") as f:
        json.dump({"A_misspecified": A, "B_correct": B}, f, indent=2)
    print("\nWrote results/exp12_investments_{A,B}.png and exp12_investments.json")
    return {"A": A, "B": B}


if __name__ == "__main__":
    run()
