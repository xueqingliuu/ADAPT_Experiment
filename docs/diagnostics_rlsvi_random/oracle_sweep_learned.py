"""Information-limit check: oracle v1map_t vs a learner on the same 10 columns.

The oracle (``v1map_t``) is the infinite-data map from ``oracle_sweep2.py``:
per-user least squares of the *true* myopic env CATE on

    φ = [1, Ê, CAE_{w-1}, yesterday steps, prior2h, activeDays7,
         burden, interact7, slot_pm, day]

The learner (``v1map_t_learned``) never sees that CATE. For each (uid, seed)
it first rolls Bern(0.5), then at week ``k`` fits a ridge of noisy weekly
CAE on week-level design

    z_w = Σ_{d,t} (A_{wdt} − 0.5) φ_{wdt},    y_w = CAE_{w+1}

using only that user's random-policy weeks ``0..k-1``. Week 0 is random.
The threshold is the median of φᵀβ on those same random slots so the send
rate stays near 1/2 (avoids confounding with the concave Bern(p) curve).

If the oracle stays ≈ +0.02 raw CAE/week on STE 0.8 and the learner stays
≈ 0, H2 (information) is binding: adding A×[t, d] to RLSVI will not rescue
a 36-week per-user learner.

Usage::

    python docs/diagnostics_rlsvi_random/oracle_sweep_learned.py \\
        env_para_ste0.8 /tmp/learned_ste08.npz 10
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import experiment as ex
from ewm_utils import gamma_from_n
from vani_env import denormalize_CAE

PD = "env_para_ste0.8" if len(sys.argv) < 2 else sys.argv[1]
OUT = os.path.expanduser(sys.argv[2]) if len(sys.argv) > 2 else os.path.expanduser("~/diag/learned_ste08.npz")
NSEED = int(sys.argv[3]) if len(sys.argv) > 3 else 10
U0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
U1 = int(sys.argv[5]) if len(sys.argv) > 5 else 28
RIDGE_LAM = float(os.getenv("ADAPR_LEARNED_RIDGE_LAM", "1.0"))
P_PHI = 10
UIDS_ALL = [13, 18, 22, 31, 33, 72, 75, 86, 106, 118, 128, 129, 143, 170,
            184, 195, 204, 239, 248, 252, 269, 277, 291, 299, 307, 327, 333, 339]
UIDS = UIDS_ALL[U0:U1]


def ewma_w(n):
    g = gamma_from_n(n)
    w = g ** np.arange(n - 1, -1, -1)
    return w / w.sum()


WF = ewma_w(14)
WA = ewma_w(7)


def delta_myopic(oenv, k, d, t):
    s = oenv.s
    env = oenv.env
    s["decisionTimeSlot"] = float(t)
    s["stepCountLast7DaysEma"] = oenv._stepCountLast7DaysEma_by_slot[int(t)]
    df = env.gen_fourSC_mean(s, 1.0) - env.gen_fourSC_mean(s, 0.0)
    if t == 0:
        da = env.gen_antic_mean(s, 1.0, 0.0) - env.gen_antic_mean(s, 0.0, 0.0)
    else:
        da = env.gen_antic_mean(s, 0.0, 1.0) - env.gen_antic_mean(s, 0.0, 0.0)
    th = env.cfg.theta_CAE
    return th[3] * WF[d * 2 + t] * df + th[4] * WA[d] * da


def phi_v1t(oenv, k, d, t):
    """Same 10 columns as oracle_sweep2.v1map_t."""
    s = oenv.s
    return np.array([
        1.0,
        float(oenv.E_known_all[k]),
        float(s["caeAverageLastWeek"]),
        float(s["yesterdayStepCount"]),
        float(s["prior2HourStepCount"]),
        float(s["activeDaysLast7Days"]),
        float(s["activitySuggestionsSentLast7Days"]),
        float(s["activitySuggestionInteractLast7Days"]),
        float(t),
        float(d),
    ], dtype=float)


def ridge_beta(Z, y, lam=RIDGE_LAM):
    """y ≈ α + Z β; intercept unpenalized. Z is (n, 10)."""
    n, p = Z.shape
    X = np.column_stack([np.ones(n), Z])
    A = X.T @ X + lam * np.eye(p + 1)
    A[0, 0] -= lam
    try:
        coef = np.linalg.solve(A, X.T @ y)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(A, X.T @ y, rcond=None)[0]
    return coef[1:]


def fit_online_from_random(phis, A, y_week):
    """Online β_k from random-policy weeks 0..k-1 of this episode.

    ``phis`` (W, 6, 2, 10), ``A`` (W, 6, 2), ``y_week`` (W,) = CAE after
    each RL week. Returns ``betas[k]`` (10,) or None, and ``tau[k]``.
    """
    nweek = int(A.shape[0])
    betas = [None] * nweek
    taus = [np.nan] * nweek
    Z_rows = []
    y_rows = []
    score_pool = []
    for w in range(nweek):
        if Z_rows:
            Z = np.asarray(Z_rows, dtype=float)
            y = np.asarray(y_rows, dtype=float)
            beta = ridge_beta(Z, y)
            betas[w] = beta
            taus[w] = float(np.median(score_pool)) if score_pool else 0.0
        z_w = np.zeros(P_PHI, dtype=float)
        for d in range(ex.N_RL_DAYS):
            for t in range(ex.N_RL_SLOTS):
                phi = phis[w, d, t]
                z_w += (float(A[w, d, t]) - 0.5) * phi
                if betas[w] is not None:
                    score_pool.append(float(phi @ betas[w]))
        Z_rows.append(z_w)
        y_rows.append(float(y_week[w]))
    return betas, taus


def run(uid, seed, policy, *, tau=None, beta_uid=None, online_betas=None, online_taus=None):
    ex._ensure_priors_configured(PD)
    cfg, env, oenv = ex._make_online_env(uid, seed=seed, params_dir=PD)
    nweek = cfg.nweek
    rng = ex._fixed_policy_rng_after_legacy_reset(seed)
    _w0, I_hist = ex.shared_episode_exogenous(seed, nweek)
    oenv._reset_episode_state()
    A_hist = np.zeros((nweek, ex.N_RL_DAYS, ex.N_RL_SLOTS), dtype=int)
    phis = np.zeros((nweek, ex.N_RL_DAYS, ex.N_RL_SLOTS, P_PHI), dtype=float)
    dls = np.zeros((nweek, ex.N_RL_DAYS, ex.N_RL_SLOTS), dtype=float)
    for k in range(nweek):
        I_w = int(I_hist[k])
        oenv.start_week(k, I_w)
        for d in range(ex.N_RL_DAYS):
            for t in range(ex.N_RL_SLOTS):
                d_global = oenv._day_idx(k, d)
                step_idx = oenv._step_idx(k, d, t)
                oenv._ensure_day_started(k, d, d_global)
                oenv._generate_prior2hour_for_slot(k, d, t, d_global, step_idx)
                phi = phi_v1t(oenv, k, d, t)
                phis[k, d, t] = phi
                dls[k, d, t] = delta_myopic(oenv, k, d, t)
                u = rng.binomial(1, 0.5)
                if policy == "bern0.5":
                    a = int(u)
                elif policy == "v1map_t":
                    score = float(phi @ beta_uid)
                    a = int(score > tau)
                elif policy == "v1map_t_learned":
                    beta = None if online_betas is None else online_betas[k]
                    if beta is None:
                        a = int(u)
                    else:
                        score = float(phi @ beta)
                        thr = 0.0 if online_taus is None or not np.isfinite(online_taus[k]) else float(online_taus[k])
                        a = int(score > thr)
                else:
                    raise ValueError(policy)
                A_hist[k, d, t] = a
                oenv.step_action(k, d, t, a, I_w)
        oenv._finalize_week(k)
    y_week = np.asarray(oenv.CAE_all[1:nweek + 1], dtype=float)
    return oenv.CAE_all.copy(), oenv.CAE_mean_all.copy(), A_hist, phis, dls, y_week


def paired_table(cae, lat, rate, names):
    ref = "bern0.5"
    raw_c = {p: denormalize_CAE(cae[p], params_dir=PD) for p in names}
    raw_l = {p: denormalize_CAE(lat[p], params_dir=PD) for p in names}
    print(f"\nparams_dir={PD}  n_users={len(UIDS)}  n_seeds={NSEED}  ridge_λ={RIDGE_LAM:g}")
    print(f"{'policy':20s} {'rawCAE':>8s} {'Δ vs bern.5':>12s} {'SE':>8s} {'z':>6s} "
          f"{'latentΔ':>9s} {'z_lat':>6s} {'rate':>6s}")
    for p in names:
        d = np.nanmean(raw_c[p][..., 1:] - raw_c[ref][..., 1:], axis=(0, 2))
        dl = np.nanmean(raw_l[p][..., 1:] - raw_l[ref][..., 1:], axis=(0, 2))
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        sel = float(dl.std(ddof=1) / np.sqrt(len(dl))) if len(dl) > 1 else float("nan")
        z = d.mean() / se if se and np.isfinite(se) and se > 0 else float("nan")
        zl = dl.mean() / sel if sel and np.isfinite(sel) and sel > 0 else float("nan")
        print(f"{p:20s} {np.nanmean(raw_c[p][..., 1:]):8.4f} {d.mean():+12.5f} {se:8.5f} "
              f"{z:6.2f} {dl.mean():+9.5f} {zl:6.2f} {np.nanmean(rate[p]):6.3f}")
    print(
        "\nRead: if v1map_t (oracle) is clearly above random and v1map_t_learned "
        "is ≈0, H2 binds — a 36-week per-user learner cannot use A×[t,d]. "
        "If the learner also wins, H1 was the bottleneck and a V1 + timing arm "
        "is worth a 200-seed job."
    )


def main():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    print(f"calibrating oracle v1map_t on {PD} ({len(UIDS_ALL)} users, seed 999)…", flush=True)
    FIT_beta = {}
    FIT_tau = {}
    t0 = time.time()
    for i, uid in enumerate(UIDS_ALL):
        _c, _l, _A, phis, dls, _y = run(uid, ex._episode_seed(999, i), "bern0.5")
        X = phis.reshape(-1, P_PHI)
        y = dls.reshape(-1)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        FIT_beta[uid] = beta
        FIT_tau[uid] = float(np.median(X @ beta))
    print(f"oracle calibration done {time.time() - t0:.0f}s", flush=True)

    POL = ["bern0.5", "v1map_t", "v1map_t_learned"]
    cae = {p: np.full((len(UIDS), NSEED, 37), np.nan) for p in POL}
    lat = {p: np.full((len(UIDS), NSEED, 37), np.nan) for p in POL}
    rate = {p: np.full((len(UIDS), NSEED), np.nan) for p in POL}

    for i, uid in enumerate(UIDS):
        for j in range(NSEED):
            seed = ex._episode_seed(j, i)
            c, l, A, phis, _dls, y_week = run(uid, seed, "bern0.5")
            cae["bern0.5"][i, j] = c
            lat["bern0.5"][i, j] = l
            rate["bern0.5"][i, j] = A.mean()
            betas, taus = fit_online_from_random(phis, A, y_week)
            c, l, A, *_ = run(
                uid, seed, "v1map_t",
                tau=FIT_tau[uid], beta_uid=FIT_beta[uid],
            )
            cae["v1map_t"][i, j] = c
            lat["v1map_t"][i, j] = l
            rate["v1map_t"][i, j] = A.mean()
            c, l, A, *_ = run(
                uid, seed, "v1map_t_learned",
                online_betas=betas, online_taus=taus,
            )
            cae["v1map_t_learned"][i, j] = c
            lat["v1map_t_learned"][i, j] = l
            rate["v1map_t_learned"][i, j] = A.mean()
        print(
            f"uid {uid} done {time.time() - t0:.0f}s  "
            f"oracle-rate={rate['v1map_t'][i].mean():.3f}  "
            f"learned-rate={rate['v1map_t_learned'][i].mean():.3f}",
            flush=True,
        )
        np.savez(
            OUT,
            uids=np.array(UIDS),
            pols=np.array(POL),
            params_dir=PD,
            ridge_lam=RIDGE_LAM,
            **{f"cae_{p}": cae[p] for p in POL},
            **{f"lat_{p}": lat[p] for p in POL},
            **{f"rate_{p}": rate[p] for p in POL},
        )

    paired_table(cae, lat, rate, POL)
    print("saved", OUT, f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
