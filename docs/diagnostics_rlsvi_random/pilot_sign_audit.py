"""Sign-agreement audit: how well do 11 weeks of pilot data identify each
user's AM-vs-PM timing preference (the effect an informative prior would
have to carry)?

For each of the 28 users, fit from ``env_para_vanilla/df_fit_11week.csv``:

    fourSC:  4hour_step_norm ~ ctrl + A + A*DecisionTime            (OLS, slot rows)
    antic:   anticipated_affect_norm ~ ctrl + ws_m + ws_a          (OLS, day rows)

and form the pilot estimate of the user's mean AM−PM myopic CATE contrast

    ĉ_u = θ3 (W̄F0 Δ̂f(0) − W̄F1 Δ̂f(1)) + θ4 W̄A (γ̂_m − γ̂_a)

with a delta-method SE. Compare with the env truth ``truth_c`` (mean over a
random rollout of Δ_myopic(t=0) − Δ_myopic(t=1)) saved by
``oracle_sweep_proximal.py``. Reports per-user ĉ, SE, z, truth, sign match,
and the overall agreement rate, plus the same for the pooled (all-user)
contrast.

Usage::

    python docs/diagnostics_rlsvi_random/pilot_sign_audit.py "~/diag/prox_v_*.npz" "~/diag/prox_s8_*.npz"
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import experiment as ex  # noqa: E402
from ewm_utils import gamma_from_n  # noqa: E402

UIDS = [13, 18, 22, 31, 33, 72, 75, 86, 106, 118, 128, 129, 143, 170,
        184, 195, 204, 239, 248, 252, 269, 277, 291, 299, 307, 327, 333, 339]


def ewma_w(n):
    g = gamma_from_n(n)
    w = g ** np.arange(n - 1, -1, -1)
    return w / w.sum()


WF = ewma_w(14)
WA = ewma_w(7)
WF0 = WF[[d * 2 for d in range(6)]].mean()
WF1 = WF[[d * 2 + 1 for d in range(6)]].mean()
WAm = WA[:6].mean()


def ols(X, y):
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    n, p = X.shape
    s2 = float(r @ r) / max(n - p, 1)
    cov = s2 * np.linalg.pinv(X.T @ X)
    return beta, cov, n


def load_truth(pattern):
    tc = {}
    for f in glob.glob(os.path.expanduser(pattern)):
        z = np.load(f, allow_pickle=True)
        for u, c in zip(z["uids"], z["truth_c"]):
            if np.isfinite(c):
                tc[int(u)] = float(c)
    return tc


def main():
    truth_v = load_truth(sys.argv[1] if len(sys.argv) > 1 else "~/diag/prox_v_*.npz")
    truth_s = load_truth(sys.argv[2] if len(sys.argv) > 2 else "~/diag/prox_s8_*.npz")
    df = pd.read_csv("env_para_vanilla/df_fit_11week.csv", low_memory=False)
    ex._ensure_priors_configured("env_para_vanilla")

    rows = []
    pooled = {"Xf": [], "yf": [], "Xa": [], "ya": []}
    for uid in UIDS:
        d = df[df.ParticipantIdentifier == uid].copy()
        cfg, env, _ = ex._make_online_env(uid, seed=0, params_dir="env_para_vanilla")
        th3, th4 = float(env.cfg.theta_CAE[3]), float(env.cfg.theta_CAE[4])
        # ---- fourSC slot rows
        A = d["WalkingSuggestion"].astype(float)
        t = d["DecisionTime"].astype(float)
        Xf = np.column_stack([
            d["FourSC_lag1"], d["YesterdayStepCount_norm"], d["prior2hour_step_norm"],
            d["recent_burden_norm"], d["is_weekend"], t, A, A * t,
        ]).astype(float)
        yf = d["4hour_step_norm"].astype(float).to_numpy()
        Xf = np.where(np.isfinite(Xf), Xf, 0.0)  # missing controls -> normalised mean (0)
        ok = np.isfinite(yf) & np.isfinite(Xf[:, 6])
        bf, cf, nf = ols(Xf[ok], yf[ok])
        pooled["Xf"].append(Xf[ok]); pooled["yf"].append(yf[ok])
        iA, iAt = 7, 8  # +1 for intercept below
        # ---- antic day rows
        ws = d.pivot_table(index="Date", columns="DecisionTime", values="WalkingSuggestion", aggfunc="first")
        g = d.sort_values(["Date", "DecisionTime"]).groupby("Date")
        day = pd.DataFrame({
            "ws_m": ws[0] if 0 in ws else np.nan,
            "ws_a": ws[1] if 1 in ws else np.nan,
            "y": g["anticipated_affect_norm"].first(),
            "ay": g["anticipated_affect_yesterday_norm"].first(),
            "act": g["active_status_fraction_7days"].first(),
            "we": g["is_weekend"].first(),
            "pu": g["perceived_utility_lastweek"].first(),
            "cae": g["CAE_avg_lastweek_norm"].first(),
            "rb": g["recent_burden_norm"].first(),
        })
        Xa = day[["ay", "act", "we", "pu", "cae", "rb", "ws_m", "ws_a"]].to_numpy(float)
        ya = day["y"].to_numpy(float)
        Xa[:, :6] = np.where(np.isfinite(Xa[:, :6]), Xa[:, :6], np.nanmean(Xa[:, :6], axis=0))
        Xa = np.where(np.isfinite(Xa), Xa, 0.0)
        ok = np.isfinite(ya) & np.isfinite(day["ws_m"].to_numpy(float)) & np.isfinite(day["ws_a"].to_numpy(float))
        ba, ca, na = ols(Xa[ok], ya[ok])
        pooled["Xa"].append(Xa[ok]); pooled["ya"].append(ya[ok])
        im, ia = 7, 8
        # ---- contrast and SE (delta method, fits independent)
        gf = np.zeros(len(bf)); gf[iA] = th3 * (WF0 - WF1); gf[iAt] = -th3 * WF1
        ga = np.zeros(len(ba)); ga[im] = th4 * WAm; ga[ia] = -th4 * WAm
        c_hat = float(gf @ bf + ga @ ba)
        se = float(np.sqrt(gf @ cf @ gf + ga @ ca @ ga))
        rows.append(dict(uid=uid, n_slots=nf, n_days=na, dF_AM=bf[iA], dF_PM=bf[iA] + bf[iAt],
                         g_m=ba[im], g_a=ba[ia], c_hat=c_hat, se=se, z=c_hat / se if se > 0 else np.nan,
                         truth_v=truth_v.get(uid, np.nan), truth_s=truth_s.get(uid, np.nan)))
    R = pd.DataFrame(rows)
    R["match_v"] = np.sign(R.c_hat) == np.sign(R.truth_v)
    R["match_s"] = np.sign(R.c_hat) == np.sign(R.truth_s)
    pd.set_option("display.width", 200)
    print(R.round(4).to_string(index=False))
    print("\nWeights: WF0=%.3f WF1=%.3f WAm=%.3f" % (WF0, WF1, WAm))
    print("pilot |z| of AM−PM contrast: median %.2f, share |z|>1.96: %.2f" % (R.z.abs().median(), (R.z.abs() > 1.96).mean()))
    for k, lab in (("v", "vanilla"), ("s", "ste0.8")):
        m = R[f"match_{k}"]; tr = R[f"truth_{k}"]
        okm = np.isfinite(tr)
        if okm.sum() == 0:
            print(f"{lab}: truth not available yet"); continue
        print(f"{lab}: sign agreement pilot vs env truth = {m[okm].mean():.2f} ({m[okm].sum()}/{okm.sum()});"
              f" Spearman(c_hat, truth) = {pd.Series(R.c_hat[okm]).corr(pd.Series(tr[okm]), method='spearman'):.2f};"
              f" env truth AM-preferred share = {(tr[okm] > 0).mean():.2f}")
        # what a prior arm could capture: truth-weighted agreement
        w = np.abs(tr[okm]); print(f"   |truth|-weighted agreement = {(m[okm] * w).sum() / w.sum():.2f}")
    # pooled contrast
    Xf = np.concatenate(pooled["Xf"]); yf = np.concatenate(pooled["yf"])
    Xa = np.concatenate(pooled["Xa"]); ya = np.concatenate(pooled["ya"])
    bf, cf, _ = ols(Xf, yf); ba, ca, _ = ols(Xa, ya)
    th3, th4 = 0.045, 0.40
    gf = np.zeros(len(bf)); gf[7] = th3 * (WF0 - WF1); gf[8] = -th3 * WF1
    ga = np.zeros(len(ba)); ga[7] = th4 * WAm; ga[8] = -th4 * WAm
    c = float(gf @ bf + ga @ ba); se = float(np.sqrt(gf @ cf @ gf + ga @ ca @ ga))
    print(f"\npooled (all users) pilot AM−PM contrast: {c:+.4f} (SE {se:.4f}, z {c / se:+.2f}); "
          f"fourSC A: AM {bf[7]:+.3f} PM {bf[7] + bf[8]:+.3f}; antic ws_m {ba[7]:+.3f} ws_a {ba[8]:+.3f}")
    R.to_csv(os.path.expanduser("~/diag/pilot_sign_audit.csv"), index=False)


if __name__ == "__main__":
    main()
