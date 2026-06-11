"""Cloud-sizing + disk-snapshot profile for ``experiment.run_micro_query``.

Profiles a single algorithm (micro-query) across a list of (user, seed) runs
and captures three independent flavours of cost:

  RAM
    * ``peak_heap_mb``           – peak Python heap during the run (tracemalloc)
    * ``delta_max_rss_mb``       – increase in process RSS attributable to the run
    * ``cumulative_max_rss_mb``  – process-wide max RSS so far

  vCPU (per-week wall clock)
    * ``_per_week_wall``         – total wall per simulated week (begin + 12 decisions + finalize)
    * ``_per_week_begin_wall``   – heavy ``begin_week`` call (PF + posterior + RLSVI)
    * ``_per_decision_wall``     – light ``agent.act`` per (week, day, slot)

  DISK
    * Per-week snapshot ``.npz`` written at the end of every simulated week,
      containing every realised ``OnlineEnv`` trajectory array sliced to
      ``[:realised_portion]`` plus (optionally) the agent's posterior history.

Aggregates the rows and prints an extrapolation block to ``--n-trial-users``
real participants over 36 weeks, sized against a MiWaves-style spec
(6 vCPU / 18 GB RAM / 50 GB disk).

Usage:
    python profile_experiment.py                       # 1 user x 1 seed
    python profile_experiment.py --users 11 --seeds 3  # 33 runs
    python profile_experiment.py --no-snapshots        # skip disk-snapshot block
    python profile_experiment.py --no-agent-state      # snapshot only oenv trajectory
    python profile_experiment.py --keep                # keep snapshot dir for inspection
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import pickle
import platform
import resource
import shutil
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

import experiment
from experiment import OnlineEnv, FOURSC_SLOTS_PER_WEEK
from vani_env import PARAMS_DIR, EnvConfig, Env
from agents import MicroQueryAgent


# ---------------------------------------------------------------------------
# Per-week timing + snapshot patch
# ---------------------------------------------------------------------------
#
# ``bayesian_posterior_update`` and the RLSVI update both re-fit on the
# *cumulative* design matrix every week (X.T @ X with X growing linearly in
# weeks), so per-week wall time grows roughly linearly in week number and
# total wall is ~quadratic in ``nweek``. To size the production weekly batch
# we want the *peak* per-week cost (= week ``nweek-1``), not the average.
#
# Timed windayOfWeekNorm breakdayOfWeekNormn:
#
#   get_week_packet(k)                 ← t0_week
#   ▼   packet returned                ← t_after_pkt
#   agent.begin_week(k, packet)        ← ★ begin_week wall ★
#   start_week(k, I_w)
#   ▼ for d in 1..6, t in 1..2:
#       get_state(k, d, t)             ← t0_decision
#       agent.act(k, d, t, state)      ← ★ decision wall  ★
#       step_action(k, d, t, A, I_w)
#   _finalize_week(k)                  ← t_week recorded (per-week wall total)
#                                      ← also writes per-week DISK snapshot
#
# Per-week wall = get_week_packet wall + begin_week wall + start_week wall +
#                 12 × decision wall + finalize_week wall.
#
# ``_finalize_week`` is also where we write a per-week ``.npz`` snapshot
# containing every realised ``OnlineEnv`` trajectory + the agent posterior
# history through that week (when ``oenv._snapshot_dir`` is set).

_ORIG_RUN_EPISODE     = OnlineEnv.run_episode
_ORIG_GET_WEEK_PACKET = OnlineEnv.get_week_packet
_ORIG_START_WEEK      = OnlineEnv.start_week
_ORIG_GET_CONTEXT     = OnlineEnv.get_context
_ORIG_STEP_ACTION     = OnlineEnv.step_action
_ORIG_FINALIZE_WEEK   = OnlineEnv._finalize_week


def _truncated_oenv(oenv, k):
    """All oenv trajectory arrays sliced to the realised portion through week k."""
    n_slot = (k + 1) * FOURSC_SLOTS_PER_WEEK   # 14 * (k + 1)
    n_day  = (k + 1) * 7
    n_wk   = k + 2                              # baseline (idx 0) + weeks 0..k
    return {
        # weekly
        "CAE_all":            oenv.CAE_all[:n_wk].copy(),
        "pu_all":             oenv.pu_all[:n_wk].copy(),
        "wp_all":             oenv.wp_all[:n_wk].copy(),
        "CAE_short_all":      oenv.CAE_short_all[:n_wk].copy(),
        "U1_all":             oenv.U1_all[:n_wk].copy(),
        "U2_all":             oenv.U2_all[:n_wk].copy(),
        "E_known_all":        oenv.E_known_all[:n_wk].copy(),
        # daily
        "dailyAnticipatedAffectAll":              oenv.dailyAnticipatedAffectAll[:n_day].copy(),
        "dailyAnticipatedAffectObsAll":          oenv.dailyAnticipatedAffectObsAll[:n_day].copy(),
        "morningFitbitWearAll":             oenv.morningFitbitWearAll[:n_day].copy(),
        "dailySurveyCompleteAll":              oenv.dailySurveyCompleteAll[:n_day].copy(),
        "activityCompletedLast7DaysAll": oenv.activityCompletedLast7DaysAll[:n_day].copy(),
        "activityCompletedLast7DaysAll":   oenv.activityCompletedLast7DaysAll[:n_day].copy(),
        # slot-level
        "stepCountFourHourAll":             oenv.stepCountFourHourAll[:n_slot].copy(),
        "pageViewFourHourAll":           oenv.pageViewFourHourAll[:n_slot].copy(),
        "action_all":             oenv.action_all[:n_slot].copy(),
        "prior2HourStepCountAll":    oenv.prior2HourStepCountAll[:n_slot].copy(),
        "ws_interaction_all":     oenv.ws_interaction_all[:n_slot].copy(),
    }


def _agent_state(agent):
    """Best-effort grab of agent posterior history. Silently skips on failure."""
    out = {}
    try:
        res = agent.results()
        if "pi_A"   in res: out["agent_pi_A"]   = np.asarray(res["pi_A"]).copy()
        if "betas"  in res: out["agent_betas"]  = np.asarray(res["betas"]).copy()
        if "alphas" in res: out["agent_alphas"] = np.asarray(res["alphas"]).copy()
        if "eta"    in res: out["agent_eta"]    = np.asarray(res["eta"]).copy()
    except Exception:
        pass
    return out


def _timed_run_episode(self, agent, dataset):
    nw = self.nweek
    self._current_agent           = agent
    self._per_week_wall           = np.zeros(nw)
    self._per_week_begin_wall     = np.zeros(nw)
    self._per_decision_wall       = np.zeros((nw, 6, 2))
    self._per_week_snapshot_bytes = np.zeros(nw, dtype=np.int64)
    self._t0_week     = 0.0
    self._t_after_pkt = 0.0
    self._t0_decision = 0.0
    return _ORIG_RUN_EPISODE(self, agent, dataset)


def _timed_get_week_packet(self, k):
    self._t0_week = time.perf_counter()
    out = _ORIG_GET_WEEK_PACKET(self, k)
    self._t_after_pkt = time.perf_counter()
    return out


def _timed_start_week(self, k, I_w):
    pwb = getattr(self, "_per_week_begin_wall", None)
    if pwb is not None and 0 <= k < pwb.size:
        pwb[k] = time.perf_counter() - self._t_after_pkt
    return _ORIG_START_WEEK(self, k, I_w)


def _timed_get_context(self, k, d, t):
    self._t0_decision = time.perf_counter()
    return _ORIG_GET_CONTEXT(self, k, d, t)


def _timed_step_action(self, k, d, t, A_wdt, I_w):
    out = _ORIG_STEP_ACTION(self, k, d, t, A_wdt, I_w)
    pd = getattr(self, "_per_decision_wall", None)
    if (pd is not None
            and 0 <= k < pd.shape[0]
            and 1 <= d <= 6
            and 1 <= t <= 2):
        pd[k, d - 1, t - 1] = time.perf_counter() - self._t0_decision
    return out


def _timed_finalize_week(self, sim_w):
    out = _ORIG_FINALIZE_WEEK(self, sim_w)
    pw = getattr(self, "_per_week_wall", None)
    if pw is not None and 0 <= sim_w < pw.size:
        pw[sim_w] = time.perf_counter() - self._t0_week

    # Optional per-week disk snapshot.
    snap_dir = getattr(self, "_snapshot_dir", None)
    if snap_dir is not None:
        snap = _truncated_oenv(self, sim_w)
        if getattr(self, "_save_agent_state", True):
            snap.update(_agent_state(self._current_agent))
        path = snap_dir / f"week_{sim_w:02d}.npz"
        np.savez_compressed(path, **snap)
        self._per_week_snapshot_bytes[sim_w] = path.stat().st_size
    return out


OnlineEnv.run_episode     = _timed_run_episode
OnlineEnv.get_week_packet = _timed_get_week_packet
OnlineEnv.start_week      = _timed_start_week
OnlineEnv.get_context     = _timed_get_context
OnlineEnv.step_action     = _timed_step_action
OnlineEnv._finalize_week  = _timed_finalize_week


# ---------------------------------------------------------------------------
# Resource probes
# ---------------------------------------------------------------------------

def _ru_maxrss_mb() -> float:
    """Cumulative max RSS of this process in MB.

    ``ru_maxrss`` is in **bytes** on macOS and **kilobytes** on Linux/BSD.
    Returns a monotonically non-decreasing high-water mark.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if platform.system() == "Darwin" else rss / 1024


def _measure(fn, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` and return (output, metrics_dict)."""
    gc.collect()
    rss_before = _ru_maxrss_mb()
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    wall = time.perf_counter() - t0
    _, peak_heap = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _ru_maxrss_mb()
    return out, {
        "wall_s": wall,
        "peak_heap_mb": peak_heap / (1024 * 1024),
        "delta_max_rss_mb": max(0.0, rss_after - rss_before),
        "cumulative_max_rss_mb": rss_after,
    }


def _pickled_kb(*objs) -> float:
    blob = pickle.dumps(objs, protocol=pickle.HIGHEST_PROTOCOL)
    return len(blob) / 1024.0


def _params_kb(uid: int) -> float:
    path = PARAMS_DIR / f"params_env_{uid}.json"
    return path.stat().st_size / 1024.0 if path.exists() else float("nan")


# ---------------------------------------------------------------------------
# Runner — build the agent ourselves so we can stamp the snapshot dir onto
# ``oenv`` *before* ``run_episode`` is called.
# ---------------------------------------------------------------------------

def _run_micro_query(uid, seed, snapshot_dir=None, save_agent_state=True):
    cfg = EnvConfig(uid)
    nweek = cfg.nweek
    env = Env(cfg, noise="sequential")
    oenv = OnlineEnv(env, nweek=nweek, seed=seed)
    if snapshot_dir is not None:
        oenv._snapshot_dir = snapshot_dir
        oenv._save_agent_state = save_agent_state

    agent = MicroQueryAgent(
        W=nweek,
        J=experiment.J_PARTICLES, B=experiment.B_ENSEMBLES,
        epsilon_0=experiment.EPSILON_0,
        mu_0_rl=experiment.mu_0_micro,
        Sigma_0_rl=experiment.Sigma_0_micro,
        sigma2_rl=experiment.sigma2_rl_micro,
        gamma_dt=experiment._gamma_dt_micro(experiment.GAMMA_BAR),
        gamma_bar=experiment.GAMMA_BAR,
        target_update_C=experiment.TARGET_C,
        nu_0_MY=experiment.nu_0_MY,
        Gamma_0_MY=experiment.Gamma_0_MY,
        sigma2_MY=experiment.sigma2_MY,
        nu_0_Y=experiment.nu_0_Y,
        Gamma_0_Y=experiment.Gamma_0_Y,
        sigma2_Y=experiment.sigma2_Y,
        nu_0_tilde_Y=experiment.nu_0_tilde_Y,
        Gamma_0_tilde_Y=experiment.Gamma_0_tilde_Y,
        sigma2_tilde_Y=experiment.sigma2_tilde_Y,
        Y_1=float(oenv.CAE_all[0]),
        rng=np.random.default_rng(seed),
    )

    dataset = experiment.EpisodeDataset(nweek)
    result = oenv.run_episode(agent, dataset)
    return result, oenv


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--users", type=int, default=None,
                   help="Number of users to profile (default: all in user_ids.txt).")
    p.add_argument("--seeds", type=int, default=1,
                   help="Number of seeds (replications) per user. Default 1.")
    p.add_argument("--out", type=Path, default=Path("profile_experiment.csv"),
                   help="CSV output path. Default ./profile_experiment.csv")
    p.add_argument("--no-snapshots", action="store_true",
                   help="Skip per-week disk snapshots (RAM/CPU profile only).")
    p.add_argument("--no-agent-state", action="store_true",
                   help="Save only oenv trajectory, skip agent posterior history.")
    p.add_argument("--snapshot-dir", type=Path, default=None,
                   help="Where to write per-week snapshots; default: temp dir.")
    p.add_argument("--keep", action="store_true",
                   help="Don't delete snapshot dir when done.")
    p.add_argument("--n-trial-users", type=int, default=100,
                   help="Real-trial participant count for the extrapolation block.")
    p.add_argument("--n-trial-cores", type=int, default=6,
                   help="Cores assumed in the extrapolation block (MiWaves: 6).")
    return p.parse_args(argv)


def _user_list(args) -> list[int]:
    uids = np.loadtxt(PARAMS_DIR / "user_ids.txt", dtype=int).tolist()
    return uids[: args.users] if args.users is not None else uids


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"min": float("nan"), "median": float("nan"),
                "mean": float("nan"), "max": float("nan")}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    user_ids = _user_list(args)
    seeds = list(range(args.seeds))

    do_snapshots = not args.no_snapshots
    save_agent   = not args.no_agent_state
    if do_snapshots:
        if args.snapshot_dir is None:
            tmp = Path(tempfile.mkdtemp(prefix="weekly_snap_"))
            cleanup_at_end = not args.keep
        else:
            tmp = args.snapshot_dir
            tmp.mkdir(parents=True, exist_ok=True)
            cleanup_at_end = False
    else:
        tmp = None
        cleanup_at_end = False

    n_total = len(user_ids) * len(seeds)
    print(f"Profiling {len(user_ids)} user(s) x {len(seeds)} seed(s) = "
          f"{n_total} runs (algorithm: micro-query).")
    print(f"Platform: {platform.system()} {platform.release()}  "
          f"Python: {sys.version.split()[0]}  PID: {os.getpid()}")
    print(f"Snapshots: {do_snapshots}  agent_state={save_agent}  dir={tmp}")
    print()

    rows: list[dict] = []
    per_week_walls: list[np.ndarray] = []
    per_week_begin_walls: list[np.ndarray] = []
    per_decision_walls: list[np.ndarray] = []
    per_run_week_bytes: list[np.ndarray] = []   # only when do_snapshots

    t_loop = time.perf_counter()

    for uid in user_ids:
        params_kb = _params_kb(uid)
        for seed in seeds:
            tag = f"u={uid:<4} seed={seed:<2}"
            run_dir = (tmp / f"u{uid}_s{seed}") if do_snapshots else None
            if run_dir is not None:
                run_dir.mkdir(parents=True, exist_ok=True)

            try:
                (result, oenv), m = _measure(
                    _run_micro_query, uid, seed=seed,
                    snapshot_dir=run_dir, save_agent_state=save_agent,
                )
            except Exception as exc:
                print(f"  {tag}  FAILED: {exc!r}")
                continue

            pickled_kb = _pickled_kb(result, oenv)
            pw  = np.asarray(getattr(oenv, "_per_week_wall", []),       dtype=float)
            pwb = np.asarray(getattr(oenv, "_per_week_begin_wall", []), dtype=float)
            pdw = np.asarray(getattr(oenv, "_per_decision_wall", []),   dtype=float)
            pdw_last_ms = (pdw[-1].ravel() * 1000.0) if pdw.size else np.array([])
            pdw_all_ms  = (pdw.ravel()    * 1000.0) if pdw.size else np.array([])

            week_bytes = (
                np.asarray(oenv._per_week_snapshot_bytes, dtype=np.int64)
                if do_snapshots else np.zeros(oenv.nweek, dtype=np.int64)
            )

            row = {
                "user_id": uid,
                "seed": seed,
                "wall_s": m["wall_s"],
                "peak_heap_mb": m["peak_heap_mb"],
                "delta_max_rss_mb": m["delta_max_rss_mb"],
                "cumulative_max_rss_mb": m["cumulative_max_rss_mb"],
                "pickled_session_kb": pickled_kb,
                "params_json_kb": params_kb,
                "nweek": oenv.nweek,
                "wall_week_first_s":      float(pw[0])  if pw.size else float("nan"),
                "wall_week_last_s":       float(pw[-1]) if pw.size else float("nan"),
                "wall_week_max_s":        float(pw.max()) if pw.size else float("nan"),
                "wall_week_growth_ratio": (
                    float(pw[-1] / pw[0]) if pw.size and pw[0] > 0 else float("nan")
                ),
                "wall_begin_week_last_s": float(pwb[-1]) if pwb.size else float("nan"),
                "wall_begin_week_max_s":  float(pwb.max()) if pwb.size else float("nan"),
                "wall_decision_last_max_ms":    float(pdw_last_ms.max())          if pdw_last_ms.size else float("nan"),
                "wall_decision_last_median_ms": float(np.median(pdw_last_ms))     if pdw_last_ms.size else float("nan"),
                "wall_decision_overall_max_ms": float(pdw_all_ms.max())           if pdw_all_ms.size  else float("nan"),
                "snapshot_total_kb": float(week_bytes.sum()) / 1024.0,
                "snapshot_first_week_kb": float(week_bytes[0]) / 1024.0 if week_bytes.size else float("nan"),
                "snapshot_last_week_kb":  float(week_bytes[-1]) / 1024.0 if week_bytes.size else float("nan"),
                "snapshot_max_week_kb":   float(week_bytes.max()) / 1024.0 if week_bytes.size else float("nan"),
            }
            rows.append(row)
            if pw.size:  per_week_walls.append(pw)
            if pwb.size: per_week_begin_walls.append(pwb)
            if pdw.size: per_decision_walls.append(pdw)
            if do_snapshots:
                per_run_week_bytes.append(week_bytes)

            snap_str = (
                f"  snap_total={row['snapshot_total_kb']:7.1f}KB"
                if do_snapshots else ""
            )
            print(f"  {tag}  wall={m['wall_s']:6.2f}s  "
                  f"peak_heap={m['peak_heap_mb']:6.1f}MB  "
                  f"d_rss={m['delta_max_rss_mb']:5.1f}MB  "
                  f"pickle={pickled_kb:6.1f}KB  "
                  f"wk1→wk{oenv.nweek}: {row['wall_week_first_s']:.2f}s→"
                  f"{row['wall_week_last_s']:.2f}s  "
                  f"beginwk(max)={row['wall_begin_week_max_s']:.2f}s  "
                  f"dec(last,max)={row['wall_decision_last_max_ms']:.2f}ms"
                  + snap_str)
            del result, oenv

    total_loop_s = time.perf_counter() - t_loop
    print(f"\nLoop wall clock: {total_loop_s:.1f} s for {len(rows)} runs.\n")

    if not rows:
        print("No successful runs; nothing to summarize.")
        if cleanup_at_end:
            shutil.rmtree(tmp, ignore_errors=True)
        return 1

    # ── CSV ─────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {args.out}")

    # ── Per-run summary ─────────────────────────────────────────────────
    print("\n=== Per-run summary ===")
    metrics = [
        "wall_s", "peak_heap_mb",
        "delta_max_rss_mb", "pickled_session_kb",
        "wall_week_first_s", "wall_week_last_s", "wall_week_max_s",
        "wall_week_growth_ratio",
        "wall_begin_week_last_s", "wall_begin_week_max_s",
        "wall_decision_last_median_ms",
        "wall_decision_last_max_ms",
        "wall_decision_overall_max_ms",
    ]
    if do_snapshots:
        metrics += [
            "snapshot_total_kb", "snapshot_first_week_kb",
            "snapshot_last_week_kb", "snapshot_max_week_kb",
        ]
    print(f"  {'metric':<32}{'min':>10}{'median':>10}{'mean':>10}{'max':>10}")
    summary = {}
    for mname in metrics:
        s = _summarize([r[mname] for r in rows if r[mname] == r[mname]])
        summary[mname] = s
        print(f"  {mname:<32}{s['min']:>10.2f}{s['median']:>10.2f}"
              f"{s['mean']:>10.2f}{s['max']:>10.2f}")

    final_rss_mb = rows[-1]["cumulative_max_rss_mb"]
    nweek = rows[0]["nweek"]
    print(f"\n  Final process max RSS: {final_rss_mb:.1f} MB"
          f"   (nweek per run = {nweek})")

    # ── Per-week wall curve + growth fit ────────────────────────────────
    pw_median = pw_p90 = None
    if per_week_walls:
        pw_stack  = np.vstack(per_week_walls)
        pw_median = np.median(pw_stack, axis=0)
        pw_p90    = np.percentile(pw_stack, 90, axis=0)

        print("\n=== Per-week wall (median across runs) ===")
        sample_w = sorted(set([0, max(1, nweek // 4), max(2, nweek // 2),
                               max(3, 3 * nweek // 4), nweek - 1]))
        for w in sample_w:
            print(f"  week {w + 1:>3}: median={pw_median[w]:>5.2f}s  "
                  f"p90={pw_p90[w]:>5.2f}s")

        ws = np.arange(nweek, dtype=float)
        a, b = np.polyfit(ws, pw_median, 1)[::-1]
        c2, c1, c0 = np.polyfit(ws, pw_median, 2)
        print(f"\n  Linear fit:    wall(w) ≈ {a:.3f} + {b:.4f} * w")
        print(f"  Quadratic fit: wall(w) ≈ {c0:.3f} + {c1:.4f}*w + {c2:.5f}*w^2")
        print(f"  Peak per-week wall (= week {nweek}): "
              f"median={pw_median[-1]:.2f}s, p90={pw_p90[-1]:.2f}s")

    # ── Per-week disk curve ─────────────────────────────────────────────
    if do_snapshots and per_run_week_bytes:
        bw_stack  = np.vstack(per_run_week_bytes)
        bw_median = np.median(bw_stack, axis=0)
        bw_p90    = np.percentile(bw_stack, 90, axis=0)

        print("\n=== Per-week snapshot size (median across runs) ===")
        for w in sample_w:
            print(f"  week {w + 1:>3}: median={bw_median[w] / 1024:>7.1f} KB   "
                  f"p90={bw_p90[w] / 1024:>7.1f} KB")
        ws = np.arange(nweek, dtype=float)
        a_b, b_b = np.polyfit(ws, bw_median, 1)[::-1]
        print(f"\n  Linear fit: size(w) ≈ {a_b / 1024:.2f} KB + {b_b / 1024:.3f} KB · w")

        per_run_total_kb = bw_stack.sum(axis=1) / 1024.0
        print(f"\n  Per-run total disk: "
              f"median={np.median(per_run_total_kb) / 1024:.2f} MB   "
              f"p90={np.percentile(per_run_total_kb, 90) / 1024:.2f} MB   "
              f"max={per_run_total_kb.max() / 1024:.2f} MB")

    # ── Extrapolation ───────────────────────────────────────────────────
    N = args.n_trial_users
    K = args.n_trial_cores

    median_pkl = summary["pickled_session_kb"]["median"]
    median_par = statistics.median([
        r["params_json_kb"] for r in rows
        if r["params_json_kb"] == r["params_json_kb"]
    ])

    rss_worker_mb = final_rss_mb
    state_per_user_kb = median_pkl + median_par
    total_state_mb = N * state_per_user_kb / 1024

    if per_week_walls:
        per_run_week_max = np.max(np.vstack(per_week_walls), axis=1)
        peak_week_wall  = float(np.median(per_run_week_max))
        peak_week_p90   = float(np.percentile(per_run_week_max, 90))
        peak_week_worst = float(np.max(per_run_week_max))
    else:
        peak_week_wall = peak_week_p90 = peak_week_worst = float("nan")

    if per_week_begin_walls:
        per_run_begin_max = np.max(np.vstack(per_week_begin_walls), axis=1)
        peak_begin_wall = float(np.median(per_run_begin_max))
        peak_begin_p90  = float(np.percentile(per_run_begin_max, 90))
    else:
        peak_begin_wall = peak_begin_p90 = float("nan")

    if per_decision_walls:
        pdw_last_runs = np.stack(
            [pdw[-1].ravel() * 1000.0 for pdw in per_decision_walls]
        )
        peak_dec_median_ms = float(np.median(pdw_last_runs))
        peak_dec_p90_ms    = float(np.percentile(pdw_last_runs, 90))
        peak_dec_max_ms    = float(np.max(pdw_last_runs))
    else:
        peak_dec_median_ms = peak_dec_p90_ms = peak_dec_max_ms = float("nan")

    if do_snapshots and per_run_week_bytes:
        median_disk_per_run_mb = float(
            np.median(np.vstack(per_run_week_bytes).sum(axis=1)) / (1024 ** 2)
        )
        p90_disk_per_run_mb = float(
            np.percentile(np.vstack(per_run_week_bytes).sum(axis=1), 90)
            / (1024 ** 2)
        )
    else:
        median_disk_per_run_mb = p90_disk_per_run_mb = 0.0

    print(f"\n=== Extrapolation to N={N} participants x {nweek} weeks ===")

    print("\n  Memory model A — one Python process per participant:")
    print(f"    RAM ≈ {N} x {rss_worker_mb:.0f} MB = {N * rss_worker_mb / 1024:.2f} GB")

    print(f"\n  Memory model B — {K} worker processes:")
    print(f"    RAM ≈ {K} x {rss_worker_mb:.0f} MB = {K * rss_worker_mb / 1024:.2f} GB")
    print(f"    Stored algorithm state ≈ {total_state_mb:.1f} MB for {N} users")

    print(f"\n  CPU model A — once-a-week begin-of-week batch on {K} cores:")
    print(f"    (heavy: PF particle update + posterior updates + RLSVI sample)")
    print(f"    Median max-week begin: {N} x {peak_begin_wall:.2f}s / {K} cores "
          f"≈ {N * peak_begin_wall / K:.1f} s "
          f"= {N * peak_begin_wall / K / 60:.2f} min")
    print(f"    P90 max-week begin:    {N} x {peak_begin_p90:.2f}s / {K} cores "
          f"≈ {N * peak_begin_p90 / K:.1f} s "
          f"= {N * peak_begin_p90 / K / 60:.2f} min")

    print(f"\n  CPU model B — twice-a-day decision-time batch on {K} cores:")
    print(f"    (light: build phi(s,a), score with pre-sampled RLSVI beta)")
    print(f"    Median peak-week decision: {N} x {peak_dec_median_ms:.2f}ms / {K} cores "
          f"≈ {N * peak_dec_median_ms / K:.1f} ms")
    print(f"    P90 peak-week decision:    {N} x {peak_dec_p90_ms:.2f}ms / {K} cores "
          f"≈ {N * peak_dec_p90_ms / K:.1f} ms")
    print(f"    Worst observed:            {N} x {peak_dec_max_ms:.2f}ms / {K} cores "
          f"≈ {N * peak_dec_max_ms / K:.1f} ms")

    print(f"\n  CPU model C — full weekly batch on {K} cores (begin + 12 decisions + finalize):")
    print(f"    Median max-week: {N} x {peak_week_wall:.2f}s / {K} cores "
          f"≈ {N * peak_week_wall / K / 60:.2f} min")
    print(f"    P90 max-week:    {N} x {peak_week_p90:.2f}s / {K} cores "
          f"≈ {N * peak_week_p90 / K / 60:.2f} min")
    print(f"    Worst observed:  {N} x {peak_week_worst:.2f}s / {K} cores "
          f"≈ {N * peak_week_worst / K / 60:.2f} min")

    if do_snapshots:
        print(f"\n  Disk model — per-week snapshot of all past data:")
        print(f"    Per-participant: median={median_disk_per_run_mb:.2f} MB   "
              f"p90={p90_disk_per_run_mb:.2f} MB")
        print(f"    Total ({N} pts):  median={N * median_disk_per_run_mb / 1024:.2f} GB   "
              f"p90={N * p90_disk_per_run_mb / 1024:.2f} GB")

    print(f"\n=== Vs MiWaves spec ({K} cores / 18 GB RAM / 50 GB disk) ===")
    ram_workers_gb = K * rss_worker_mb / 1024
    print(f"  RAM with {K} workers: {ram_workers_gb:.2f} GB vs 18 GB "
          f"-> {18 - ram_workers_gb:+.2f} GB headroom")
    print(f"  Disk (algorithm state pickle): {total_state_mb:.1f} MB vs 50 GB")
    if do_snapshots:
        total_disk_gb = N * median_disk_per_run_mb / 1024
        print(f"  Disk (per-week snapshots,   {N} pts): {total_disk_gb:.2f} GB vs 50 GB "
              f"-> {50 - total_disk_gb:+.2f} GB headroom")

    if cleanup_at_end and tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"\nDeleted snapshot dir {tmp}")
    elif tmp is not None:
        print(f"\nSnapshots kept at {tmp}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
