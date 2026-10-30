"""Calibrate misspecification A and B strengths once per params dir.

After a refit::

    python calibrate_misspec.py --params-dir env_para_ste0.2
    python calibrate_misspec.py --params-dir env_para_ste0.5

or both STE folders::

    python calibrate_misspec.py --all-ste

Both A and B are calibrated on simulated data under random suggestion
(p = 0.5), 12 weeks, engagement OFF, with common random numbers between
the unmodified and modified environment.

A. Fix ``lam_Y = 0.5``. Solve ``lam_SC``, ``lam_AA`` so the coefficients of
``Mbar_SC`` and ``Mbar_AA`` in the pooled OLS
``Y_{w+1} ~ (1, Y_w, Mbar_SC, Mbar_AA)`` are within 5% of the same
regression on the unmodified environment.

B. ``Y_{w+1} = mu^Y + beta_D * f_w + eps`` (no −0.5 centering). Solve
``beta_D`` so the coefficient of ``f_w`` in
``Y_{w+1} ~ (1, Y_w, f_w)`` is within 5% of the unmodified environment.

Uses enough simulated participants that the Monte Carlo (cluster-robust)
SE of each target coefficient is < 2% of its value; prints the SEs.

Writes ``<params_dir>/misspec_calibration.json``. Does not retune STE.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiment import N_RL_DAYS, N_RL_SLOTS, _make_online_env
from misspec import (
    CONFOUNDER_LAM_Y,
    CONFOUNDER_RHO,
    save_calibration,
)
from testbed import STE_PARAM_DIRS
from vani_env import build_CAE_features, resolve_params_dir


SE_REL_MAX = 0.02
MATCH_TOL = 0.05
MIN_EPISODES = 4
DEFAULT_MAX_EPISODES = 256
A_SE_NAMES = ("Mbar_SC", "Mbar_AA")
B_SE_NAMES = ("f_w",)


def _user_ids(params_dir: Path) -> list[int]:
    return [int(x) for x in np.loadtxt(params_dir / "user_ids.txt", dtype=int)]


def ols_cluster_se(X: np.ndarray, y: np.ndarray, clusters: np.ndarray):
    """OLS coefficients and cluster-robust sandwich SEs."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(clusters):
        mask = clusters == c
        score = X[mask].T @ resid[mask]
        meat += np.outer(score, score)
    var = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(var), 0.0))
    return beta, se


def _relerr(got, target) -> float:
    if abs(target) < 1e-8:
        return abs(got - target)
    return abs(got / target - 1.0)


def _run_random_episode(
    params_dir,
    uid,
    seed,
    nweek,
    *,
    misspec: str,
    strengths: dict | None = None,
):
    """One random-0.5, activity-off episode. Returns rows for A and B OLS."""
    cfg, env, oenv = _make_online_env(
        int(uid),
        seed=seed,
        params_dir=params_dir,
        activity="off",
        misspec=misspec,
        nweek=nweek,
        misspec_strengths=strengths,
    )
    rec = []
    orig = env.gen_CAE

    def wrapped(cae_last, foursc_wk, antic_wk, week_idx):
        x = build_CAE_features(cae_last, foursc_wk, antic_wk)
        y = orig(cae_last, foursc_wk, antic_wk, week_idx)
        f_w = 0.0 if env._week_action_frac is None else float(env._week_action_frac)
        rec.append((float(y), np.asarray(x, dtype=float), f_w))
        return y

    env.gen_CAE = wrapped
    oenv._reset_episode_state()
    rng = np.random.default_rng(seed)
    for k in range(oenv.nweek):
        oenv.start_week(k, 1)
        for d in range(N_RL_DAYS):
            d_global = oenv._day_idx(k, d)
            oenv._ensure_day_started(k, d, d_global)
            for t in range(N_RL_SLOTS):
                a = 0 if oenv.dropped_out else int(rng.random() < 0.5)
                oenv.step_action(k, d, t, float(a), 1)
        oenv._finalize_week(k)
    return rec


def _collect_design(
    params_dir,
    user_ids,
    *,
    nweek,
    episodes,
    seed0,
    misspec,
    strengths=None,
):
    rows = []
    clusters = []
    cluster = 0
    for e in range(int(episodes)):
        for uid in user_ids:
            rec = _run_random_episode(
                params_dir, uid, int(seed0) + e, nweek,
                misspec=misspec, strengths=strengths,
            )
            for y, x, f_w in rec:
                rows.append((y, x, f_w))
                clusters.append(cluster)
            cluster += 1
    if not rows:
        raise RuntimeError("no simulated weeks collected")
    y = np.asarray([r[0] for r in rows], dtype=float)
    X_a = np.vstack([r[1] for r in rows])
    f = np.asarray([r[2] for r in rows], dtype=float)
    X_b = np.column_stack([np.ones(y.size), X_a[:, 1], f])
    return y, X_a, X_b, np.asarray(clusters, dtype=int)


def _se_met(coef: float, se: float) -> bool:
    if not np.isfinite(se):
        return False
    if abs(coef) < 1e-8:
        return float(se) < SE_REL_MAX
    return float(se) < SE_REL_MAX * abs(coef)


def _se_rows(beta, se, idxs, names) -> list[dict]:
    rows = []
    for i, name in zip(idxs, names):
        coef = float(beta[i])
        sei = float(se[i])
        rel = float("nan") if abs(coef) < 1e-8 else sei / abs(coef)
        rows.append(
            {
                "name": name,
                "coef": coef,
                "se": sei,
                "se_over_abs_coef": rel,
                "se_criterion_met": _se_met(coef, sei),
            }
        )
    return rows


def _enough_se(rows: list[dict]) -> bool:
    return all(bool(r["se_criterion_met"]) for r in rows)


def _print_se_misses(rows: list[dict], *, max_episodes: int) -> None:
    missed = [r for r in rows if not r["se_criterion_met"]]
    if not missed:
        return
    print(
        f"  [warn] MC SE criterion not met at max-episodes={max_episodes}",
        flush=True,
    )
    for r in missed:
        rel = r["se_over_abs_coef"]
        rel_txt = "nan" if not np.isfinite(rel) else f"{rel:.4f}"
        print(
            f"    {r['name']}  coef={r['coef']:.6f}  SE={r['se']:.6f}  "
            f"SE/|coef|={rel_txt}",
            flush=True,
        )


def _grow_unmodified(
    params_dir,
    user_ids,
    *,
    nweek,
    seed0,
    start_episodes,
    which: str,
    max_episodes: int = DEFAULT_MAX_EPISODES,
):
    """Increase episodes until cluster-robust SEs of the target coeffs are small."""
    episodes = int(start_episodes)
    cap = max(int(max_episodes), episodes)
    names = A_SE_NAMES if which == "A" else B_SE_NAMES
    while True:
        t0 = time.perf_counter()
        y, X_a, X_b, cl = _collect_design(
            params_dir, user_ids, nweek=nweek, episodes=episodes,
            seed0=seed0, misspec="none",
        )
        if which == "A":
            beta, se = ols_cluster_se(X_a, y, cl)
            idxs = (2, 3)
        else:
            beta, se = ols_cluster_se(X_b, y, cl)
            idxs = (2,)
        rows = _se_rows(beta, se, idxs, names)
        wall = time.perf_counter() - t0
        print(
            f"  unmodified {which} episodes={episodes}  wall={wall:.1f}s  "
            + "  ".join(
                f"{r['name']}={r['coef']:.4f} (SE {r['se']:.4f}, "
                f"SE/|coef|="
                f"{'nan' if not np.isfinite(r['se_over_abs_coef']) else f'{r['se_over_abs_coef']:.3f}'}"
                f")"
                for r in rows
            ),
            flush=True,
        )
        if _enough_se(rows) or episodes >= cap:
            if not _enough_se(rows):
                _print_se_misses(rows, max_episodes=cap)
            return episodes, beta, se, rows, y, X_a, X_b, cl
        episodes = min(cap, episodes * 2)


def _run_A_episode(params_dir, uid, seed, nweek, lam_sc, lam_aa):
    """Compatibility helper used by tests."""
    rec = _run_random_episode(
        params_dir, uid, seed, nweek,
        misspec="A",
        strengths={"lam_SC": lam_sc, "lam_AA": lam_aa},
    )
    if not rec:
        return np.empty((0, 4)), np.empty(0)
    X = np.vstack([r[1] for r in rec])
    y = np.asarray([r[0] for r in rec], dtype=float)
    return X, y


def calibrate_A(
    params_dir: Path,
    user_ids: list[int],
    *,
    nweek: int,
    episodes: int,
    seed0: int,
    tol: float = MATCH_TOL,
    max_episodes: int = DEFAULT_MAX_EPISODES,
) -> dict:
    episodes, target_beta, target_se, se_rows, *_ = _grow_unmodified(
        params_dir, user_ids, nweek=nweek, seed0=seed0,
        start_episodes=episodes, which="A", max_episodes=max_episodes,
    )
    target_sc, target_aa = float(target_beta[2]), float(target_beta[3])
    print(
        f"  A targets (unmodified OLS): SC={target_sc:.4f} (SE {target_se[2]:.4f})  "
        f"AA={target_aa:.4f} (SE {target_se[3]:.4f})  nweek={nweek}  "
        f"episodes={episodes}",
        flush=True,
    )

    def fit(lams):
        y, X_a, _X_b, cl = _collect_design(
            params_dir, user_ids, nweek=nweek, episodes=episodes,
            seed0=seed0, misspec="A",
            strengths={"lam_SC": lams[0], "lam_AA": lams[1]},
        )
        beta, se = ols_cluster_se(X_a, y, cl)
        e_sc, e_aa = _relerr(beta[2], target_sc), _relerr(beta[3], target_aa)
        print(
            f"    lam_SC={lams[0]:.3f} lam_AA={lams[1]:.3f}  "
            f"OLS SC={beta[2]:.4f} (SE {se[2]:.4f})  "
            f"AA={beta[3]:.4f} (SE {se[3]:.4f})  "
            f"relerr={e_sc:.3f}/{e_aa:.3f}",
            flush=True,
        )
        return beta, se, e_sc, e_aa

    grid = [0.15, 0.30, 0.45, 0.60]
    best = None
    best_pack = None
    for ls in grid:
        for la in grid:
            pack = fit((ls, la))
            score = abs(pack[0][2] - target_sc) + abs(pack[0][3] - target_aa)
            if best is None or score < best[0]:
                best = (score, ls, la)
                best_pack = pack
    lam_sc, lam_aa = best[1], best[2]
    beta, se, e_sc, e_aa = best_pack
    if e_sc > tol or e_aa > tol:
        try:
            from scipy.optimize import least_squares

            def fun(v):
                b, _se, _esc, _eaa = fit((float(v[0]), float(v[1])))
                return np.array([b[2] - target_sc, b[3] - target_aa], dtype=float)

            sol = least_squares(fun, [lam_sc, lam_aa], bounds=([0.0, 0.0], [0.95, 0.95]))
            lam_sc, lam_aa = float(sol.x[0]), float(sol.x[1])
            beta, se, e_sc, e_aa = fit((lam_sc, lam_aa))
        except Exception as exc:
            print(f"  [warn] refine failed ({exc}); keeping grid solution", flush=True)
    return {
        "lam_Y": CONFOUNDER_LAM_Y,
        "lam_SC": float(lam_sc),
        "lam_AA": float(lam_aa),
        "rho": CONFOUNDER_RHO,
        "target_coef_SC": target_sc,
        "target_coef_AA": target_aa,
        "target_se_SC": float(target_se[2]),
        "target_se_AA": float(target_se[3]),
        "ols_coef_SC": float(beta[2]),
        "ols_coef_AA": float(beta[3]),
        "ols_se_SC": float(se[2]),
        "ols_se_AA": float(se[3]),
        "relerr_SC": float(e_sc),
        "relerr_AA": float(e_aa),
        "within_5pct": bool(e_sc <= tol and e_aa <= tol),
        "se_criterion_met": _enough_se(se_rows),
        "se_targets": se_rows,
        "max_episodes": int(max_episodes),
        "nweek": int(nweek),
        "episodes": int(episodes),
        "seed0": int(seed0),
        "n_users": len(user_ids),
        "activity": "off",
        "policy": "random_0.5",
    }


def calibrate_B(
    params_dir: Path,
    user_ids: list[int],
    *,
    nweek: int,
    episodes: int,
    seed0: int,
    tol: float = MATCH_TOL,
    max_episodes: int = DEFAULT_MAX_EPISODES,
) -> dict:
    episodes, target_beta, target_se, se_rows, *_ = _grow_unmodified(
        params_dir, user_ids, nweek=nweek, seed0=seed0,
        start_episodes=episodes, which="B", max_episodes=max_episodes,
    )
    target_f = float(target_beta[2])
    print(
        f"  B target (unmodified OLS coef of f_w) = {target_f:.4f} "
        f"(SE {target_se[2]:.4f})  nweek={nweek}  episodes={episodes}",
        flush=True,
    )

    def fit(beta_d):
        y, _X_a, X_b, cl = _collect_design(
            params_dir, user_ids, nweek=nweek, episodes=episodes,
            seed0=seed0, misspec="B",
            strengths={"beta_D": float(beta_d)},
        )
        beta, se = ols_cluster_se(X_b, y, cl)
        err = float(beta[2] - target_f)
        rel = _relerr(beta[2], target_f)
        print(
            f"    beta_D={beta_d:.4f}  coef_f={beta[2]:.4f} (SE {se[2]:.4f})  "
            f"err={err:.4f}  relerr={rel:.3f}",
            flush=True,
        )
        return err, float(beta[2]), float(se[2]), rel

    lo, hi = -2.0, 2.0
    err_lo, got, se_f, rel = fit(lo)
    err_hi, _, _, _ = fit(hi)
    if err_lo * err_hi > 0:
        if abs(err_lo) < abs(err_hi):
            lo -= 2.0
            err_lo, got, se_f, rel = fit(lo)
        else:
            hi += 2.0
            err_hi, got, se_f, rel = fit(hi)
    beta_d = 0.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        err_mid, got, se_f, rel = fit(mid)
        beta_d = mid
        if rel <= tol:
            break
        if err_lo * err_mid <= 0:
            hi, err_hi = mid, err_mid
        else:
            lo, err_lo = mid, err_mid
    return {
        "beta_D": float(beta_d),
        "target_coef_f": target_f,
        "target_se_f": float(target_se[2]),
        "ols_coef_f": float(got),
        "ols_se_f": float(se_f),
        "relerr_f": float(rel),
        "within_5pct": bool(rel <= tol),
        "se_criterion_met": _enough_se(se_rows),
        "se_targets": se_rows,
        "max_episodes": int(max_episodes),
        "nweek": int(nweek),
        "episodes": int(episodes),
        "seed0": int(seed0),
        "n_users": len(user_ids),
        "activity": "off",
        "policy": "random_0.5",
        "centering": "none",
    }


def calibrate_dir(params_dir: Path, args) -> dict:
    params_dir = resolve_params_dir(params_dir)
    user_ids = _user_ids(params_dir)
    if args.max_users and args.max_users < len(user_ids):
        user_ids = user_ids[: int(args.max_users)]
    print(f"Calibrating {params_dir}  ({len(user_ids)} users)", flush=True)
    t0 = time.perf_counter()
    payload = {"params_dir": str(params_dir)}
    nweek_a = args.nweek
    nweek_b = args.nweek_b if args.nweek_b is not None else args.nweek
    if args.which in {"A", "both"}:
        payload["A"] = calibrate_A(
            params_dir, user_ids,
            nweek=nweek_a, episodes=args.episodes, seed0=args.seed,
            max_episodes=args.max_episodes,
        )
    if args.which in {"B", "both"}:
        payload["B"] = calibrate_B(
            params_dir, user_ids,
            nweek=nweek_b, episodes=args.episodes, seed0=args.seed,
            max_episodes=args.max_episodes,
        )
    path = save_calibration(params_dir, {k: payload[k] for k in payload if k in {"A", "B"}})
    print(f"  wrote {path}  ({time.perf_counter() - t0:.1f}s)", flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--params-dir", default=None)
    p.add_argument(
        "--all-ste",
        action="store_true",
        help="Calibrate env_para_ste0.2 / 0.5",
    )
    p.add_argument("--which", choices=("A", "B", "both"), default="both")
    p.add_argument(
        "--nweek",
        type=int,
        default=12,
        help="Weeks for A and (by default) B (random-0.5, activity off).",
    )
    p.add_argument(
        "--nweek-b",
        type=int,
        default=12,
        help="Weeks for misspec B (default 12, same as A).",
    )
    p.add_argument("--episodes", type=int, default=MIN_EPISODES)
    p.add_argument(
        "--max-episodes",
        type=int,
        default=DEFAULT_MAX_EPISODES,
        help="Cap on doubling episodes until SE < 2% of |coef| (default 256).",
    )
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--max-users", type=int, default=None)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    dirs = []
    if args.all_ste:
        dirs.extend(STE_PARAM_DIRS)
    if args.params_dir:
        dirs.append(args.params_dir)
    if not dirs:
        dirs.append(None)
    reports = []
    for d in dirs:
        reports.append(calibrate_dir(d, args))
    print(json.dumps(
        [{k: v for k, v in r.items() if k != "params_dir"} | {"params_dir": r.get("params_dir")}
         for r in reports],
        indent=2,
    ))


if __name__ == "__main__":
    main()
