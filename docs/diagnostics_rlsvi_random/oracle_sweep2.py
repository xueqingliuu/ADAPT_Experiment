"""Bern(p) sweep + myopic env-CATE oracles, CRN-paired per (uid, seed).
Read-only w.r.t. the repo: uses OnlineEnv exactly as _run_fixed_policy_fast does."""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import experiment as ex
from ewm_utils import gamma_from_n

PD = "env_para_vanilla" if len(sys.argv) < 2 else sys.argv[1]
OUT = os.path.expanduser(sys.argv[2]) if len(sys.argv) > 2 else os.path.expanduser("~/diag/oracle_vanilla.npz")
NSEED = int(sys.argv[3]) if len(sys.argv) > 3 else 10
U0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
U1 = int(sys.argv[5]) if len(sys.argv) > 5 else 28
UIDS_ALL = [13,18,22,31,33,72,75,86,106,118,128,129,143,170,184,195,204,239,248,252,269,277,291,299,307,327,333,339]
UIDS = UIDS_ALL[U0:U1]

def ewma_w(n):
    g = gamma_from_n(n); w = g ** np.arange(n-1, -1, -1); return w / w.sum()
WF = ewma_w(14); WA = ewma_w(7)

def delta_myopic(oenv, k, d, t):
    """Env's immediate expected effect of A=1 vs 0 on this week's CAE mean (normalized scale)."""
    s = oenv.s; env = oenv.env
    s["decisionTimeSlot"] = float(t)
    s["stepCountLast7DaysEma"] = oenv._stepCountLast7DaysEma_by_slot[int(t)]
    df = env.gen_fourSC_mean(s, 1.0) - env.gen_fourSC_mean(s, 0.0)
    if t == 0: da = env.gen_antic_mean(s, 1.0, 0.0) - env.gen_antic_mean(s, 0.0, 0.0)
    else:      da = env.gen_antic_mean(s, 0.0, 1.0) - env.gen_antic_mean(s, 0.0, 0.0)
    th = env.cfg.theta_CAE
    return th[3]*WF[d*2+t]*df + th[4]*WA[d]*da, df, da

def run(uid, seed, policy, tau=None, log_states=False):
    ex._ensure_priors_configured(PD)
    cfg, env, oenv = ex._make_online_env(uid, seed=seed, params_dir=PD)
    nweek = cfg.nweek
    rng = ex._fixed_policy_rng_after_legacy_reset(seed)
    _w0, I_hist = ex.shared_episode_exogenous(seed, nweek)
    oenv._reset_episode_state()
    rows = []
    A_hist = np.zeros((nweek, ex.N_RL_DAYS, ex.N_RL_SLOTS))
    for k in range(nweek):
        I_w = int(I_hist[k]); oenv.start_week(k, I_w)
        for d in range(ex.N_RL_DAYS):
            for t in range(ex.N_RL_SLOTS):
                d_global = oenv._day_idx(k, d); step_idx = oenv._step_idx(k, d, t)
                oenv._ensure_day_started(k, d, d_global)
                oenv._generate_prior2hour_for_slot(k, d, t, d_global, step_idx)
                dl, df, da = delta_myopic(oenv, k, d, t)
                u = rng.binomial(1, 0.5)   # always consume same rng stream for CRN
                if policy.startswith("bern"):
                    p = float(policy[4:]); a = int(rng.random() < p) if p not in (0.0, 1.0) else int(p)
                    if p == 0.5: a = u
                elif policy == "sign": a = int(dl > 0)
                elif policy == "rank": a = int(dl > tau)
                elif policy == "antirank": a = int(dl < tau)
                elif policy == "antisign": a = int(dl <= 0)
                elif policy in ("timing","cov","v1map","v10map","v1map_t"):
                    s_ = oenv.s
                    x = np.array([1.0, oenv.E_known_all[k], s_["caeAverageLastWeek"], s_["yesterdayStepCount"], s_["prior2HourStepCount"],
                                  s_["activeDaysLast7Days"], s_["activitySuggestionsSentLast7Days"], s_["activitySuggestionInteractLast7Days"]])
                    if policy == "timing": score = FIT["timing"][uid][d, t]
                    elif policy == "cov":  score = dl - FIT["timing"][uid][d, t]
                    elif policy == "v1map": score = x @ FIT["v1"][uid]
                    elif policy == "v1map_t": score = np.concatenate([x,[t, d]]) @ FIT["v1t"][uid]
                    else: score = x @ FIT["v10"]
                    a = int(score > FIT["tau_"+policy][uid])
                else: raise ValueError(policy)
                A_hist[k, d, t] = a
                if log_states:
                    s = oenv.s
                    rows.append([uid, seed, k, d, t, dl, df, da, s["activitySuggestionsSentLast7Days"],
                                 s.get("morningFitbitWearLast7Days", np.nan), s["pageViewLast7DaysEma"],
                                 s["dailyAnticipatedAffectYesterday"], s["yesterdayStepCount"], s["prior2HourStepCount"],
                                 s["activitySuggestionInteractLast7Days"], s["activeDaysLast7Days"],
                                 s["perceivedUtilityLastWeek"], s["caeAverageLastWeek"], s["isWeekend"], oenv.E_known_all[k]])
                oenv.step_action(k, d, t, a, I_w)
        oenv._finalize_week(k)
    return oenv.CAE_all.copy(), oenv.CAE_mean_all.copy(), A_hist, np.asarray(rows)

POL = ["bern0.5","timing","cov","v1map","v1map_t","v10map"]
# ---- calibration pass over ALL users (random rollouts, seed 999) ----
FIT = {"timing":{}, "v1":{}, "v1t":{}, "tau_timing":{}, "tau_cov":{}, "tau_v1map":{}, "tau_v1map_t":{}, "tau_v10map":{}}
cal = {}
for i, uid in enumerate(UIDS_ALL):
    _, _, _, rows = run(uid, ex._episode_seed(999, i), "bern0.5", log_states=True)
    cal[uid] = rows
def xmat(rows): return np.column_stack([np.ones(len(rows)), rows[:,19], rows[:,17], rows[:,12], rows[:,13], rows[:,15], rows[:,8], rows[:,14]])
allrows = np.concatenate(list(cal.values()), 0)
FIT["v10"] = np.linalg.lstsq(xmat(allrows), allrows[:,5], rcond=None)[0]
for uid, rows in cal.items():
    dl_ = rows[:,5]; d_ = rows[:,3].astype(int); t_ = rows[:,4].astype(int)
    T = np.zeros((6,2))
    for dd in range(6):
        for tt in range(2): T[dd,tt] = dl_[(d_==dd)&(t_==tt)].mean()
    FIT["timing"][uid] = T
    X = xmat(rows); FIT["v1"][uid] = np.linalg.lstsq(X, dl_, rcond=None)[0]
    Xt = np.column_stack([X, t_, d_]); FIT["v1t"][uid] = np.linalg.lstsq(Xt, dl_, rcond=None)[0]
    FIT["tau_timing"][uid] = float(np.median(T[d_, t_]))
    FIT["tau_cov"][uid] = float(np.median(dl_ - T[d_, t_]))
    FIT["tau_v1map"][uid] = float(np.median(X @ FIT["v1"][uid]))
    FIT["tau_v1map_t"][uid] = float(np.median(Xt @ FIT["v1t"][uid]))
    FIT["tau_v10map"][uid] = float(np.median(X @ FIT["v10"]))
print("calibration done", flush=True)
cae = {p: np.full((len(UIDS), NSEED, 37), np.nan) for p in POL}
lat = {p: np.full((len(UIDS), NSEED, 37), np.nan) for p in POL}
rate = {p: np.full((len(UIDS), NSEED), np.nan) for p in POL}
state_rows = []
tau_u = {}
t0 = time.time()
for i, uid in enumerate(UIDS):
    # calibrate tau_u on a random rollout with a seed not in the eval set
    rows = cal[uid]; tau_u[uid] = float(np.median(rows[:, 5])); state_rows.append(rows)
    for j in range(NSEED):
        seed = ex._episode_seed(j, i)
        for p in POL:
            c, l, A, _ = run(uid, seed, p, tau=tau_u[uid])
            cae[p][i, j] = c; lat[p][i, j] = l; rate[p][i, j] = A.mean()
    print(f"uid {uid} done {time.time()-t0:.0f}s  tau={tau_u[uid]:.2e}  timing-rate={rate['timing'][i].mean():.3f}", flush=True)
    np.savez(OUT, uids=np.array(UIDS), pols=np.array(POL), tau=np.array([tau_u.get(u, np.nan) for u in UIDS]),
             states=np.concatenate(state_rows, axis=0),
             **{f"cae_{p}": cae[p] for p in POL}, **{f"lat_{p}": lat[p] for p in POL}, **{f"rate_{p}": rate[p] for p in POL})
print("DONE", time.time()-t0)
