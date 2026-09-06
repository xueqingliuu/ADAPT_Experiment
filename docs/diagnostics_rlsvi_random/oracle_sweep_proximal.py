"""Proximal-outcome timing learner: can a per-user learner recover the
AM/PM (and day) timing effect from the *observed per-slot mediators*
(4-hour step count, daily anticipated affect) instead of weekly CAE?

Motivation. ``oracle_sweep_learned.py`` showed that on ste0.8 the infinite-
data timing map gains ≈+0.02 raw CAE/week over Bern(0.5) while a ridge on
noisy *weekly CAE* gains ≈0 (H2: information limit). Weekly CAE is one noisy
number per 12 decisions. The env's timing effect, however, enters CAE only
through the mediators fourSC (per slot, always observed) and antic (per day,
observed when the survey is completed), with population weights
``theta_CAE[3]``, ``theta_CAE[4]`` and the within-week EWMA weights — and the
calibration itself identified the AM/PM coefficients from those proximal
outcomes. So a learner that regresses the observed mediators on
``A × [slot, ...]`` sees 12× (fourSC) / 6× (antic) the observations at far
higher SNR, with no pilot-data leakage.

Learner (per user, online, weeks 0..k-1 of that user's own random-policy
rollout — same off-policy construction as ``oracle_sweep_learned.py``):

    fourSC_wdt ≈ α + c·controls + A·(β0 + β_t·t [+ β_x·x])          (ridge)
    antic_wd   ≈ α + c·controls + ws_m·γ_m + ws_a·γ_a                (ridge)

    score(d,t | s) = θ3 · WF[d,t] · Δ̂f(t, s) + θ4 · WA[d] · Δ̂a(t)
    send iff score > running median of scores on the random slots seen so far

``theta_CAE`` (population reward model) and the EWMA weights are treated as
known — the PF reward prior in ``rl_priors.json`` estimates the former from
pooled pilot data, which the LOO protocol allows.

Two learners: ``prox_slot`` (A, A×t only) and ``prox_x`` (adds A×[burden,
prior2h, yesterday] for fourSC and ws×[burden, pu, cae, act7] for antic —
the env's own interaction set minus pv7/wear7/antic_y/E).

Reference arms in the same paired run: ``bern0.5`` and the infinite-data
oracle ``v1map_t`` from ``oracle_sweep_learned.py``.

Usage::

    python docs/diagnostics_rlsvi_random/oracle_sweep_proximal.py \
        env_para_ste0.8 ~/diag/prox_s8_0.npz 10 0 7
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import experiment as ex  # noqa: E402
from ewm_utils import gamma_from_n  # noqa: E402
from vani_env import denormalize_CAE  # noqa: E402

PD = sys.argv[1] if len(sys.argv) > 1 else "env_para_ste0.8"
OUT = os.path.expanduser(sys.argv[2]) if len(sys.argv) > 2 else os.path.expanduser("~/diag/prox.npz")
NSEED = int(sys.argv[3]) if len(sys.argv) > 3 else 10
U0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
U1 = int(sys.argv[5]) if len(sys.argv) > 5 else 28
LAM = float(os.getenv("ADAPR_PROX_RIDGE_LAM", "1.0"))
UIDS_ALL = [13, 18, 22, 31, 33, 72, 75, 86, 106, 118, 128, 129, 143, 170,
            184, 195, 204, 239, 248, 252, 269, 277, 291, 299, 307, 327, 333, 339]
UIDS = UIDS_ALL[U0:U1]
ND, NT = ex.N_RL_DAYS, ex.N_RL_SLOTS


def ewma_w(n):
    g = gamma_from_n(n)
    w = g ** np.arange(n - 1, -1, -1)
    return w / w.sum()


# Weekly CAE EWMA runs over 7 days × 2 slots (Sunday has no decisions and
# takes the heaviest, unused positions 12–13), as in oracle_sweep_learned.py.
WF = ewma_w(7 * NT)
WA = ewma_w(7)


# ----------------------------------------------------------------- features
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
    return th[3] * WF[d * NT + t] * df + th[4] * WA[d] * da


def phi_v1t(oenv, k, d, t):
    s = oenv.s
    return np.array([1.0, float(oenv.E_known_all[k]), float(s["caeAverageLastWeek"]),
                     float(s["yesterdayStepCount"]), float(s["prior2HourStepCount"]),
                     float(s["activeDaysLast7Days"]), float(s["activitySuggestionsSentLast7Days"]),
                     float(s["activitySuggestionInteractLast7Days"]), float(t), float(d)], dtype=float)


def slot_ctrl(oenv, k, d, t):
    """Decision-time controls for the fourSC regression (agent-visible)."""
    s = oenv.s
    return np.array([float(s["stepCountNext4HourLag1"]), float(s["yesterdayStepCount"]),
                     float(s["prior2HourStepCount"]), float(s["activitySuggestionsSentLast7Days"]),
                     float(s["isWeekend"]), float(t)], dtype=float)


def slot_x(oenv, k, d, t):
    """Extra A×x modifiers for ``prox_x`` (fourSC)."""
    s = oenv.s
    return np.array([float(s["activitySuggestionsSentLast7Days"]), float(s["prior2HourStepCount"]),
                     float(s["yesterdayStepCount"])], dtype=float)


def day_ctrl(oenv, k, d):
    """Morning-state controls for the antic regression (agent-visible)."""
    s = oenv.s
    return np.array([float(s.get("dailyAnticipatedAffectYesterdayAgent", 0.0)), float(s["activeDaysLast7Days"]),
                     float(s["isWeekend"]), float(oenv.E_known_all[k]), float(s["caeAverageLastWeek"]),
                     float(s["activitySuggestionsSentLast7Days"])], dtype=float)


def day_x(oenv, k, d):
    """Extra ws×x modifiers for ``prox_x`` (antic)."""
    s = oenv.s
    return np.array([float(s["activitySuggestionsSentLast7Days"]), float(oenv.E_known_all[k]),
                     float(s["caeAverageLastWeek"]), float(s["activeDaysLast7Days"])], dtype=float)


def ridge(X, y, lam=LAM):
    n, p = X.shape
    Xc = np.column_stack([np.ones(n), X])
    A = Xc.T @ Xc + lam * np.eye(p + 1)
    A[0, 0] -= lam
    try:
        coef = np.linalg.solve(A, Xc.T @ y)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(A, Xc.T @ y, rcond=None)[0]
    return coef  # includes intercept at [0]


class ProxModel:
    """Fitted per-user proximal model; ``score`` maps a decision to an
    estimated myopic CATE on weekly CAE."""

    def __init__(self, variant, bf, ba, th3, th4):
        self.variant, self.bf, self.ba, self.th3, self.th4 = variant, bf, ba, th3, th4

    def delta_f(self, t, x):
        # fourSC design: [ctrl(6), A, A*t, (A*x(3))]
        b = self.bf
        d = b[1 + 6] + b[1 + 7] * t
        if self.variant == "prox_x":
            d += float(b[1 + 8:1 + 11] @ x)
        return d

    def delta_a(self, t, xd):
        # antic design: [ctrl(6), ws_m, ws_a, (ws_m*xd(4), ws_a*xd(4))]
        b = self.ba
        if t == 0:
            d = b[1 + 6]
            if self.variant == "prox_x":
                d += float(b[1 + 8:1 + 12] @ xd)
        else:
            d = b[1 + 7]
            if self.variant == "prox_x":
                d += float(b[1 + 12:1 + 16] @ xd)
        return d

    def score(self, d, t, x, xd):
        return self.th3 * WF[d * NT + t] * self.delta_f(t, x) + self.th4 * WA[d] * self.delta_a(t, xd)


def fit_online(variant, log, th3, th4):
    """Return models[k] fitted on weeks 0..k-1 (None when no data) and
    running thresholds tau[k] (median of scores on random slots seen so far)."""
    nweek = log["A"].shape[0]
    models = [None] * nweek
    taus = np.full(nweek, np.nan)
    pool = []
    for k in range(nweek):
        if k >= 1:
            # ---- fourSC rows: weeks 0..k-1
            C = log["ctrl"][:k].reshape(-1, 6)
            A = log["A"][:k].reshape(-1).astype(float)
            T = log["ctrl"][:k].reshape(-1, 6)[:, 5]
            Y = log["fourSC"][:k].reshape(-1)
            cols = [C, A[:, None], (A * T)[:, None]]
            if variant == "prox_x":
                X3 = log["x"][:k].reshape(-1, 3)
                cols.append(A[:, None] * X3)
            Xf = np.column_stack(cols)
            bf = ridge(Xf, Y)
            # ---- antic rows: days in weeks 0..k-1 with observed antic
            Cd = log["dctrl"][:k].reshape(-1, 6)
            wm = log["A"][:k, :, 0].reshape(-1).astype(float)
            wa = log["A"][:k, :, 1].reshape(-1).astype(float)
            Ya = log["antic"][:k].reshape(-1)
            cols = [Cd, wm[:, None], wa[:, None]]
            if variant == "prox_x":
                Xd = log["dx"][:k].reshape(-1, 4)
                cols += [wm[:, None] * Xd, wa[:, None] * Xd]
            Xa = np.column_stack(cols)
            ok = np.isfinite(Ya)
            if ok.sum() >= Xa.shape[1] + 2:
                ba = ridge(Xa[ok], Ya[ok])
            else:
                ba = np.zeros(Xa.shape[1] + 1)
            models[k] = ProxModel(variant, bf, ba, th3, th4)
            taus[k] = float(np.median(pool)) if pool else 0.0
        # add this week's random slots to the score pool using the current model
        if models[k] is not None:
            for d in range(ND):
                for t in range(NT):
                    pool.append(models[k].score(d, t, log["x"][k, d, t], log["dx"][k, d]))
    return models, taus


# ----------------------------------------------------------------- rollout
def run(uid, seed, policy, *, tau=None, beta_uid=None, models=None, taus=None):
    ex._ensure_priors_configured(PD)
    cfg, env, oenv = ex._make_online_env(uid, seed=seed, params_dir=PD)
    nweek = cfg.nweek
    rng = ex._fixed_policy_rng_after_legacy_reset(seed)
    _w0, I_hist = ex.shared_episode_exogenous(seed, nweek)
    oenv._reset_episode_state()
    log = {
        "A": np.zeros((nweek, ND, NT), dtype=int),
        "ctrl": np.zeros((nweek, ND, NT, 6)),
        "x": np.zeros((nweek, ND, NT, 3)),
        "fourSC": np.zeros((nweek, ND, NT)),
        "dctrl": np.zeros((nweek, ND, 6)),
        "dx": np.zeros((nweek, ND, 4)),
        "antic": np.full((nweek, ND), np.nan),
        "phi": np.zeros((nweek, ND, NT, 10)),
        "dl": np.zeros((nweek, ND, NT)),
    }
    for k in range(nweek):
        I_w = int(I_hist[k])
        oenv.start_week(k, I_w)
        # Rate-preserving threshold for the proximal learners: median of the
        # current model's scores on this rollout's OWN past slots (causal;
        # tracks the state drift the policy itself induces, e.g. burden).
        thr = 0.0
        if policy in ("prox_slot", "prox_x") and models is not None and models[k] is not None and k >= 1:
            m = models[k]
            past = [m.score(dd, tt, log["x"][kk, dd, tt], log["dx"][kk, dd])
                    for kk in range(k) for dd in range(ND) for tt in range(NT)]
            thr = float(np.median(past))
        for d in range(ND):
            d_global = oenv._day_idx(k, d)
            for t in range(NT):
                step_idx = oenv._step_idx(k, d, t)
                oenv._ensure_day_started(k, d, d_global)
                oenv._generate_prior2hour_for_slot(k, d, t, d_global, step_idx)
                if t == 0:
                    log["dctrl"][k, d] = day_ctrl(oenv, k, d)
                    log["dx"][k, d] = day_x(oenv, k, d)
                log["ctrl"][k, d, t] = slot_ctrl(oenv, k, d, t)
                log["x"][k, d, t] = slot_x(oenv, k, d, t)
                log["phi"][k, d, t] = phi_v1t(oenv, k, d, t)
                log["dl"][k, d, t] = delta_myopic(oenv, k, d, t)
                u = rng.binomial(1, 0.5)
                if policy == "bern0.5":
                    a = int(u)
                elif policy == "v1map_t":
                    a = int(float(log["phi"][k, d, t] @ beta_uid) > tau)
                elif policy in ("prox_slot", "prox_x"):
                    m = models[k]
                    if m is None:
                        a = int(u)
                    else:
                        a = int(m.score(d, t, log["x"][k, d, t], log["dx"][k, d]) > thr)
                else:
                    raise ValueError(policy)
                log["A"][k, d, t] = a
                oenv.step_action(k, d, t, a, I_w)
                log["fourSC"][k, d, t] = float(oenv.stepCountNext4HourAll[step_idx])
            # antic for the day is generated at end of day inside step_action/_end_day
        oenv._finalize_week(k)
        for d in range(ND):
            d_global = oenv._day_idx(k, d)
            log["antic"][k, d] = float(oenv.dailyAnticipatedAffectObsAll[d_global])
    return oenv.CAE_all.copy(), oenv.CAE_mean_all.copy(), log


def paired_table(cae, lat, rate, names, extra=""):
    ref = "bern0.5"
    rc = {p: denormalize_CAE(cae[p], params_dir=PD) for p in names}
    rl = {p: denormalize_CAE(lat[p], params_dir=PD) for p in names}
    print(f"\nparams_dir={PD}  n_users={len(UIDS)}  n_seeds={NSEED}  ridge_λ={LAM:g} {extra}")
    print(f"{'policy':14s} {'rawCAE':>8s} {'Δ vs bern.5':>12s} {'SE':>8s} {'z':>6s} {'latentΔ':>9s} {'z_lat':>6s} {'rate':>6s}")
    for p in names:
        d = np.nanmean(rc[p][..., 1:] - rc[ref][..., 1:], axis=(0, 2))
        dl = np.nanmean(rl[p][..., 1:] - rl[ref][..., 1:], axis=(0, 2))
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        sel = dl.std(ddof=1) / np.sqrt(len(dl)) if len(dl) > 1 else np.nan
        print(f"{p:14s} {np.nanmean(rc[p][..., 1:]):8.4f} {d.mean():+12.5f} {se:8.5f} "
              f"{d.mean() / se if se > 0 else np.nan:6.2f} {dl.mean():+9.5f} "
              f"{dl.mean() / sel if sel > 0 else np.nan:6.2f} {np.nanmean(rate[p]):6.3f}")


def main():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    t0 = time.time()
    # oracle calibration (same as oracle_sweep_learned.py)
    FIT_beta, FIT_tau = {}, {}
    for i, uid in enumerate(UIDS_ALL):
        _c, _l, log = run(uid, ex._episode_seed(999, i), "bern0.5")
        X = log["phi"].reshape(-1, 10)
        y = log["dl"].reshape(-1)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        FIT_beta[uid], FIT_tau[uid] = beta, float(np.median(X @ beta))
    print(f"oracle calibration done {time.time() - t0:.0f}s", flush=True)

    POL = ["bern0.5", "v1map_t", "prox_slot", "prox_x"]
    cae = {p: np.full((len(UIDS), NSEED, 37), np.nan) for p in POL}
    lat = {p: np.full((len(UIDS), NSEED, 37), np.nan) for p in POL}
    rate = {p: np.full((len(UIDS), NSEED), np.nan) for p in POL}
    # diagnostics: sign agreement of the learned AM−PM contrast with the truth, by week
    signok = {v: np.full((len(UIDS), NSEED, 36), np.nan) for v in ("prox_slot", "prox_x")}
    truth_c = np.full(len(UIDS), np.nan)

    for i, uid in enumerate(UIDS):
        ex._ensure_priors_configured(PD)
        cfg, env, _ = ex._make_online_env(uid, seed=0, params_dir=PD)
        th3, th4 = float(env.cfg.theta_CAE[3]), float(env.cfg.theta_CAE[4])
        for j in range(NSEED):
            seed = ex._episode_seed(j, i)
            c, l, log = run(uid, seed, "bern0.5")
            cae["bern0.5"][i, j], lat["bern0.5"][i, j], rate["bern0.5"][i, j] = c, l, log["A"].mean()
            # truth: user's AM−PM myopic CATE contrast (mean over the random rollout)
            dl = log["dl"]
            tc = dl[:, :, 0].mean() - dl[:, :, 1].mean()
            if j == 0:
                truth_c[i] = tc
            c, l, lg = run(uid, seed, "v1map_t", tau=FIT_tau[uid], beta_uid=FIT_beta[uid])
            cae["v1map_t"][i, j], lat["v1map_t"][i, j], rate["v1map_t"][i, j] = c, l, lg["A"].mean()
            for v in ("prox_slot", "prox_x"):
                models, taus = fit_online(v, log, th3, th4)
                c, l, lg = run(uid, seed, v, models=models, taus=taus)
                cae[v][i, j], lat[v][i, j], rate[v][i, j] = c, l, lg["A"].mean()
                # learned AM−PM contrast at each week vs truth sign
                for k in range(1, 36):
                    m = models[k]
                    if m is None:
                        continue
                    xbar = log["x"][:k].reshape(-1, 3).mean(0)
                    xdbar = log["dx"][:k].reshape(-1, 4).mean(0)
                    sc = np.mean([m.score(d, 0, xbar, xdbar) - m.score(d, 1, xbar, xdbar) for d in range(ND)])
                    signok[v][i, j, k] = float(np.sign(sc) == np.sign(tc))
        print(f"uid {uid} done {time.time() - t0:.0f}s  truthAM-PM={truth_c[i]:+.4f}  "
              f"rates: slot={rate['prox_slot'][i].mean():.3f} x={rate['prox_x'][i].mean():.3f}  "
              f"sign-ok wk35: slot={np.nanmean(signok['prox_slot'][i, :, 35]):.2f} x={np.nanmean(signok['prox_x'][i, :, 35]):.2f}",
              flush=True)
        np.savez(OUT, uids=np.array(UIDS), pols=np.array(POL), params_dir=PD, ridge_lam=LAM,
                 truth_c=truth_c,
                 **{f"cae_{p}": cae[p] for p in POL}, **{f"lat_{p}": lat[p] for p in POL},
                 **{f"rate_{p}": rate[p] for p in POL}, **{f"signok_{v}": signok[v] for v in signok})
    paired_table(cae, lat, rate, POL)
    for v in signok:
        print(f"{v}: P(learned AM−PM sign == truth) by week 5/12/24/35: "
              + " ".join(f"{np.nanmean(signok[v][:, :, k]):.2f}" for k in (5, 12, 24, 35)))
    print("saved", OUT, f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
