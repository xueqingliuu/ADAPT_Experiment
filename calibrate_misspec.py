"""Calibrate misspecification A and B strengths once per params dir.

Default (after STE folders exist)::

    python calibrate_misspec.py

writes ``misspec_calibration.json`` into ``env_para_ste0.2`` and
``env_para_ste0.5``. Vanilla only with an explicit folder::

    python calibrate_misspec.py --params-dir env_para_vanilla

Both A and B use simulated data under random suggestion (p = 0.5), 12 weeks,
engagement OFF, with common random numbers between the unmodified and
modified environment.

Precision uses the paired difference of target coefficients, not each
environment's own coefficient. ``SE(diff)`` is a cluster bootstrap over
``(participant, episode)`` pairs (resampled jointly). Criterion:
``SE(diff) < 2%`` of ``|coef_unmodified|``. Episodes double up to
``--max-episodes`` (default 256).

A. Fix ``lam_Y = 0.5``. 2-D search on ``(lam_SC, lam_AA)`` so each paired
difference (modified − unmodified) of the ``Mbar_SC`` / ``Mbar_AA``
coefficients in ``Y_{w+1} ~ (1, Y_w, Mbar_SC, Mbar_AA)`` is within 5% of
``|coef_unmodified|``.

B. ``Y_{w+1} = mu^Y + beta_D * f_w + eps`` (no −0.5 centering). Fit
``Y_{w+1} ~ (1, Y_w, f_w)`` in the unmodified environment and in B with
``beta_D = 0`` (mediator action effects halved). Set
``beta_D = b_unmod - b_B(0)``. Verify on fresh seeds; one secant step if
the paired ``f_w`` difference is not within 5% of ``|b_unmod|``.

Does not retune STE.
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
DEFAULT_N_BOOT = 200
A_GROW_LAMS = (0.30, 0.30)
A_REFINE_MAX_EVALS = 8
A_NAMES = ("Mbar_SC", "Mbar_AA")
A_IDXS = (2, 3)
B_NAMES = ("f_w",)
B_IDXS = (2,)
VERIFY_SEED_SHIFT = 1_000_003


def _user_ids(params_dir: Path) -> list[int]:
    return [int(x) for x in np.loadtxt(params_dir / "user_ids.txt", dtype=int)]


def ols_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(np.asarray(X, dtype=float), np.asarray(y, dtype=float), rcond=None)
    return np.asarray(beta, dtype=float)


def ols_cluster_se(X: np.ndarray, y: np.ndarray, clusters: np.ndarray):
    """OLS coefficients and cluster-robust sandwich SEs (single environment)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = ols_beta(X, y)
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


def _cluster_from_rec(rec, *, uid: int, episode: int) -> dict:
    if not rec:
        raise RuntimeError("empty episode")
    y = np.asarray([r[0] for r in rec], dtype=float)
    xa = np.vstack([r[1] for r in rec])
    fw = np.asarray([r[2] for r in rec], dtype=float)
    xb = np.column_stack([np.ones(y.size), xa[:, 1], fw])
    return {"uid": int(uid), "episode": int(episode), "y": y, "Xa": xa, "Xb": xb}


def collect_clusters(
    params_dir,
    user_ids,
    *,
    nweek,
    episodes,
    seed0,
    misspec,
    strengths=None,
    episode_start: int = 0,
) -> list[dict]:
    clusters = []
    for e in range(int(episode_start), int(episodes)):
        for uid in user_ids:
            rec = _run_random_episode(
                params_dir, uid, int(seed0) + e, nweek,
                misspec=misspec, strengths=strengths,
            )
            clusters.append(_cluster_from_rec(rec, uid=uid, episode=e))
    if not clusters and int(episode_start) == 0:
        raise RuntimeError("no simulated weeks collected")
    return clusters


def stack_clusters(clusters: list[dict], xkey: str):
    y = np.concatenate([c["y"] for c in clusters])
    X = np.vstack([c[xkey] for c in clusters])
    cl = np.concatenate(
        [np.full(c["y"].size, i, dtype=int) for i, c in enumerate(clusters)]
    )
    return y, X, cl


def _stack_draw(clusters: list[dict], draw: np.ndarray, xkey: str):
    y = np.concatenate([clusters[int(i)]["y"] for i in draw])
    X = np.vstack([clusters[int(i)][xkey] for i in draw])
    return y, X


def bootstrap_diff_se(
    clu_u: list[dict],
    clu_m: list[dict],
    *,
    xkey: str,
    idxs: tuple[int, ...],
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Cluster bootstrap SE of ``beta_mod[idxs] - beta_unmod[idxs]``."""
    n = len(clu_u)
    if n != len(clu_m):
        raise ValueError("paired clusters must have the same (participant, episode) keys")
    idxs = tuple(int(i) for i in idxs)
    diffs = np.empty((int(n_boot), len(idxs)), dtype=float)
    for b in range(int(n_boot)):
        draw = rng.integers(0, n, size=n)
        yu, Xu = _stack_draw(clu_u, draw, xkey)
        ym, Xm = _stack_draw(clu_m, draw, xkey)
        bu = ols_beta(Xu, yu)
        bm = ols_beta(Xm, ym)
        diffs[b] = np.array([bm[i] - bu[i] for i in idxs], dtype=float)
    return np.std(diffs, axis=0, ddof=1)


def _se_diff_met(coef_unmod: float, se_diff: float) -> bool:
    if not np.isfinite(se_diff):
        return False
    if abs(coef_unmod) < 1e-8:
        return float(se_diff) < SE_REL_MAX
    return float(se_diff) < SE_REL_MAX * abs(coef_unmod)


def _match_met(diff: float, coef_unmod: float, tol: float = MATCH_TOL) -> bool:
    if abs(coef_unmod) < 1e-8:
        return abs(diff) <= tol
    return abs(diff) <= tol * abs(coef_unmod)


def paired_rows(
    names,
    idxs,
    beta_u,
    se_u,
    beta_m,
    se_m,
    se_diff,
) -> list[dict]:
    rows = []
    for k, (name, i) in enumerate(zip(names, idxs)):
        un = float(beta_u[i])
        mo = float(beta_m[i])
        diff = mo - un
        sd = float(se_diff[k])
        rel = float("nan") if abs(un) < 1e-8 else sd / abs(un)
        rows.append(
            {
                "name": name,
                "coef_unmodified": un,
                "coef_modified": mo,
                "diff": diff,
                "se_diff": sd,
                "se_diff_over_abs_unmod": rel,
                "se_unmodified": float(se_u[i]),
                "se_modified": float(se_m[i]),
                "se_criterion_met": _se_diff_met(un, sd),
                "within_5pct": _match_met(diff, un),
            }
        )
    return rows


def _fmt_rel(value) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{float(value):.4f}"


def _print_paired_rows(tag: str, rows: list[dict], *, wall: float | None = None) -> None:
    extra = f"  wall={wall:.1f}s" if wall is not None else ""
    print(f"  {tag}{extra}", flush=True)
    for r in rows:
        print(
            f"    {r['name']}: unmod={r['coef_unmodified']:.4f} (SE {r['se_unmodified']:.4f})  "
            f"mod={r['coef_modified']:.4f} (SE {r['se_modified']:.4f})  "
            f"diff={r['diff']:.4f}  SE(diff)={r['se_diff']:.4f}  "
            f"SE(diff)/|unmod|={_fmt_rel(r['se_diff_over_abs_unmod'])}",
            flush=True,
        )


def _print_se_misses(rows: list[dict], *, max_episodes: int) -> None:
    missed = [r for r in rows if not r["se_criterion_met"]]
    if not missed:
        return
    print(
        f"  [warn] paired SE(diff) criterion not met at max-episodes={max_episodes}",
        flush=True,
    )
    for r in missed:
        print(
            f"    {r['name']}  unmod={r['coef_unmodified']:.6f}  "
            f"diff={r['diff']:.6f}  SE(diff)={r['se_diff']:.6f}  "
            f"SE(diff)/|unmod|={_fmt_rel(r['se_diff_over_abs_unmod'])}",
            flush=True,
        )


def _enough_se(rows: list[dict]) -> bool:
    return all(bool(r["se_criterion_met"]) for r in rows)


def _projected_growth_s(episodes: int, cap: int, wall: float, *, added: int | None = None) -> float:
    """Remaining growth time if later doublings only simulate the new episodes."""
    if episodes >= cap or wall <= 0:
        return 0.0
    denom = float(added if added not in (None, 0) else episodes)
    return float(wall) * (float(cap) - float(episodes)) / denom


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return x if np.isfinite(x) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def fit_side(clusters: list[dict], xkey: str):
    y, X, cl = stack_clusters(clusters, xkey)
    return ols_cluster_se(X, y, cl)


def paired_fit(
    clu_u,
    clu_m,
    *,
    which: str,
    n_boot: int,
    rng: np.random.Generator,
) -> list[dict]:
    if which == "A":
        names, idxs, xkey = A_NAMES, A_IDXS, "Xa"
    else:
        names, idxs, xkey = B_NAMES, B_IDXS, "Xb"
    bu, seu = fit_side(clu_u, xkey)
    bm, sem = fit_side(clu_m, xkey)
    se_diff = bootstrap_diff_se(
        clu_u, clu_m, xkey=xkey, idxs=idxs, n_boot=n_boot, rng=rng,
    )
    return paired_rows(names, idxs, bu, seu, bm, sem, se_diff)


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


def _grow_paired(
    params_dir,
    user_ids,
    *,
    nweek,
    seed0,
    start_episodes,
    max_episodes,
    n_boot,
    which: str,
    strengths,
    rng: np.random.Generator,
):
    """Double episodes until paired SE(diff) meets the criterion (or the cap)."""
    cap = max(int(max_episodes), MIN_EPISODES)
    episodes = min(max(int(start_episodes), MIN_EPISODES), cap)
    last = None
    clu_u: list[dict] = []
    clu_m: list[dict] = []
    episode_start = 0
    while True:
        t0 = time.perf_counter()
        clu_u.extend(collect_clusters(
            params_dir, user_ids, nweek=nweek, episodes=episodes,
            seed0=seed0, misspec="none", episode_start=episode_start,
        ))
        clu_m.extend(collect_clusters(
            params_dir, user_ids, nweek=nweek, episodes=episodes,
            seed0=seed0, misspec=which, strengths=strengths,
            episode_start=episode_start,
        ))
        rows = paired_fit(clu_u, clu_m, which=which, n_boot=n_boot, rng=rng)
        wall = time.perf_counter() - t0
        added = episodes - episode_start
        proj = _projected_growth_s(episodes, cap, wall, added=added)
        _print_paired_rows(
            f"paired {which} episodes={episodes}  n_users={len(user_ids)}  "
            f"projected remaining growth ~{proj:.0f}s",
            rows,
            wall=wall,
        )
        last = (episodes, clu_u, clu_m, rows, wall)
        if _enough_se(rows) or episodes >= cap:
            if not _enough_se(rows):
                _print_se_misses(rows, max_episodes=cap)
            return last
        episode_start = episodes
        episodes = min(cap, episodes * 2)


def calibrate_A(
    params_dir: Path,
    user_ids: list[int],
    *,
    nweek: int,
    episodes: int,
    seed0: int,
    tol: float = MATCH_TOL,
    max_episodes: int = DEFAULT_MAX_EPISODES,
    n_boot: int = DEFAULT_N_BOOT,
) -> dict:
    rng = np.random.default_rng(int(seed0) + 17)
    episodes, clu_u, _clu_grow, grow_rows, grow_wall = _grow_paired(
        params_dir, user_ids, nweek=nweek, seed0=seed0,
        start_episodes=episodes, max_episodes=max_episodes, n_boot=n_boot,
        which="A", strengths={"lam_SC": A_GROW_LAMS[0], "lam_AA": A_GROW_LAMS[1]},
        rng=rng,
    )
    n_eval_est = 16
    print(
        f"  A search at episodes={episodes}: ~{n_eval_est}+ modified evals; "
        f"one eval ~{grow_wall / 2:.1f}s (unmodified cached)",
        flush=True,
    )
    target_sc = float(grow_rows[0]["coef_unmodified"])
    target_aa = float(grow_rows[1]["coef_unmodified"])

    def fit(lams):
        clu_m = collect_clusters(
            params_dir, user_ids, nweek=nweek, episodes=episodes,
            seed0=seed0, misspec="A",
            strengths={"lam_SC": lams[0], "lam_AA": lams[1]},
        )
        rows = paired_fit(clu_u, clu_m, which="A", n_boot=n_boot, rng=rng)
        _print_paired_rows(f"lam_SC={lams[0]:.3f} lam_AA={lams[1]:.3f}", rows)
        d_sc, d_aa = float(rows[0]["diff"]), float(rows[1]["diff"])
        return rows, d_sc, d_aa

    grid = [0.15, 0.30, 0.45, 0.60]
    best = None
    best_pack = None
    for ls in grid:
        for la in grid:
            pack = fit((ls, la))
            score = abs(pack[1]) + abs(pack[2])
            if best is None or score < best[0]:
                best = (score, ls, la)
                best_pack = pack
    lam_sc, lam_aa = best[1], best[2]
    rows, d_sc, d_aa = best_pack
    matched = _match_met(d_sc, target_sc, tol) and _match_met(d_aa, target_aa, tol)
    if not matched:
        class _Budget(RuntimeError):
            pass

        n_eval = [0]
        best_score, best_ls, best_la = best[0], lam_sc, lam_aa
        best_pack = (rows, d_sc, d_aa)
        try:
            from scipy.optimize import least_squares

            def fun(v):
                n_eval[0] += 1
                if n_eval[0] > A_REFINE_MAX_EVALS:
                    raise _Budget()
                ls = float(np.clip(v[0], 0.0, 0.95))
                la = float(np.clip(v[1], 0.0, 0.95))
                pack = fit((ls, la))
                score = abs(pack[1]) + abs(pack[2])
                nonlocal best_score, best_ls, best_la, best_pack
                if score < best_score:
                    best_score, best_ls, best_la, best_pack = score, ls, la, pack
                return np.array([pack[1], pack[2]], dtype=float)

            sol = least_squares(
                fun, [lam_sc, lam_aa],
                bounds=([0.0, 0.0], [0.95, 0.95]),
                max_nfev=A_REFINE_MAX_EVALS,
            )
            ls = float(np.clip(sol.x[0], 0.0, 0.95))
            la = float(np.clip(sol.x[1], 0.0, 0.95))
            pack = fit((ls, la))
            score = abs(pack[1]) + abs(pack[2])
            if score < best_score:
                best_ls, best_la, best_pack = ls, la, pack
        except _Budget:
            print(
                f"  A refine stopped at {A_REFINE_MAX_EVALS} evals (keeping best so far)",
                flush=True,
            )
        except Exception as exc:
            print(f"  [warn] A refine failed ({exc}); keeping grid solution", flush=True)
        lam_sc, lam_aa = float(best_ls), float(best_la)
        rows, d_sc, d_aa = best_pack
        matched = _match_met(d_sc, target_sc, tol) and _match_met(d_aa, target_aa, tol)
    return {
        "lam_Y": CONFOUNDER_LAM_Y,
        "lam_SC": float(lam_sc),
        "lam_AA": float(lam_aa),
        "rho": CONFOUNDER_RHO,
        "target_coef_SC": target_sc,
        "target_coef_AA": target_aa,
        "paired": rows,
        "relerr_SC": _relerr(rows[0]["coef_modified"], target_sc),
        "relerr_AA": _relerr(rows[1]["coef_modified"], target_aa),
        "within_5pct": bool(matched),
        "se_criterion_met": _enough_se(rows),
        "max_episodes": int(max_episodes),
        "nweek": int(nweek),
        "episodes": int(episodes),
        "n_boot": int(n_boot),
        "seed0": int(seed0),
        "n_users": len(user_ids),
        "activity": "off",
        "policy": "random_0.5",
        "precision": "paired_cluster_bootstrap",
    }


def _b_unmod_and_b0(clu_u, clu_m0) -> tuple[float, float]:
    bu, _ = fit_side(clu_u, "Xb")
    b0, _ = fit_side(clu_m0, "Xb")
    return float(bu[2]), float(b0[2])


def calibrate_B(
    params_dir: Path,
    user_ids: list[int],
    *,
    nweek: int,
    episodes: int,
    seed0: int,
    tol: float = MATCH_TOL,
    max_episodes: int = DEFAULT_MAX_EPISODES,
    n_boot: int = DEFAULT_N_BOOT,
) -> dict:
    rng = np.random.default_rng(int(seed0) + 23)
    episodes, clu_u, clu_m0, grow_rows, grow_wall = _grow_paired(
        params_dir, user_ids, nweek=nweek, seed0=seed0,
        start_episodes=episodes, max_episodes=max_episodes, n_boot=n_boot,
        which="B", strengths={"beta_D": 0.0},
        rng=rng,
    )
    b_unmod, b_0 = _b_unmod_and_b0(clu_u, clu_m0)
    beta_d = b_unmod - b_0
    used_secant = False
    print(
        f"  B direct: b_unmod={b_unmod:.4f}  b_B(0)={b_0:.4f}  "
        f"beta_D={beta_d:.4f}  verify eval ~{grow_wall / 2:.1f}s",
        flush=True,
    )

    def verify(beta, *, seed):
        clu_uv = collect_clusters(
            params_dir, user_ids, nweek=nweek, episodes=episodes,
            seed0=seed, misspec="none",
        )
        clu_mv = collect_clusters(
            params_dir, user_ids, nweek=nweek, episodes=episodes,
            seed0=seed, misspec="B", strengths={"beta_D": float(beta)},
        )
        rows = paired_fit(clu_uv, clu_mv, which="B", n_boot=n_boot, rng=rng)
        _print_paired_rows(f"verify beta_D={beta:.4f} seed={seed}", rows)
        return rows

    vseed = int(seed0) + VERIFY_SEED_SHIFT
    vrows = verify(beta_d, seed=vseed)
    vdiff = float(vrows[0]["diff"])
    v_unmod = float(vrows[0]["coef_unmodified"])
    if not _match_met(vdiff, v_unmod, tol):
        used_secant = True
        d0 = b_0 - b_unmod
        d1 = vdiff
        denom = d1 - d0
        if abs(denom) < 1e-12:
            print("  [warn] B secant denominator ~0; keeping direct beta_D", flush=True)
        else:
            beta_next = beta_d - d1 * (beta_d - 0.0) / denom
            print(
                f"  B secant: d0={d0:.4f} d1={d1:.4f}  beta_D {beta_d:.4f} -> {beta_next:.4f}",
                flush=True,
            )
            beta_d = float(beta_next)
            vrows = verify(beta_d, seed=vseed + 1)
            vdiff = float(vrows[0]["diff"])
            v_unmod = float(vrows[0]["coef_unmodified"])
    return {
        "beta_D": float(beta_d),
        "b_unmod": b_unmod,
        "b_B0": b_0,
        "method": "secant" if used_secant else "direct_paired_difference",
        "paired_grow": grow_rows,
        "paired_verify": vrows,
        "within_5pct": bool(_match_met(vdiff, v_unmod, tol)),
        "se_criterion_met": _enough_se(grow_rows),
        "max_episodes": int(max_episodes),
        "nweek": int(nweek),
        "episodes": int(episodes),
        "n_boot": int(n_boot),
        "seed0": int(seed0),
        "verify_seed0": vseed,
        "n_users": len(user_ids),
        "activity": "off",
        "policy": "random_0.5",
        "centering": "none",
        "precision": "paired_cluster_bootstrap",
    }


def beta_d_from_paired(
    params_dir,
    user_ids,
    *,
    nweek: int,
    episodes: int,
    seed0: int,
) -> tuple[float, list[dict], list[dict]]:
    """In-sample ``beta_D = b_unmod - b_B(0)`` and the paired ``f_w`` rows at that value."""
    clu_u = collect_clusters(
        params_dir, user_ids, nweek=nweek, episodes=episodes,
        seed0=seed0, misspec="none",
    )
    clu_0 = collect_clusters(
        params_dir, user_ids, nweek=nweek, episodes=episodes,
        seed0=seed0, misspec="B", strengths={"beta_D": 0.0},
    )
    b_unmod, b_0 = _b_unmod_and_b0(clu_u, clu_0)
    beta_d = b_unmod - b_0
    rng = np.random.default_rng(int(seed0) + 23)
    grow_rows = paired_fit(clu_u, clu_0, which="B", n_boot=80, rng=rng)
    clu_b = collect_clusters(
        params_dir, user_ids, nweek=nweek, episodes=episodes,
        seed0=seed0, misspec="B", strengths={"beta_D": float(beta_d)},
    )
    at_beta = paired_fit(clu_u, clu_b, which="B", n_boot=80, rng=rng)
    return float(beta_d), grow_rows, at_beta


def calibrate_dir(params_dir: Path, args) -> dict:
    params_dir = resolve_params_dir(params_dir)
    user_ids = _user_ids(params_dir)
    if args.max_users and args.max_users < len(user_ids):
        user_ids = user_ids[: int(args.max_users)]
    nweek_a = args.nweek
    nweek_b = args.nweek_b if args.nweek_b is not None else args.nweek
    print(
        f"Calibrating {params_dir}  n_users={len(user_ids)}  "
        f"doubling {args.episodes}..{args.max_episodes}  "
        f"nweek A={nweek_a} B={nweek_b}  n_boot={args.n_boot}",
        flush=True,
    )
    t0 = time.perf_counter()
    payload = {"params_dir": str(params_dir)}
    if args.which in {"A", "both"}:
        payload["A"] = calibrate_A(
            params_dir, user_ids,
            nweek=nweek_a, episodes=args.episodes, seed0=args.seed,
            max_episodes=args.max_episodes, n_boot=args.n_boot,
        )
    if args.which in {"B", "both"}:
        payload["B"] = calibrate_B(
            params_dir, user_ids,
            nweek=nweek_b, episodes=args.episodes, seed0=args.seed,
            max_episodes=args.max_episodes, n_boot=args.n_boot,
        )
    path = save_calibration(
        params_dir,
        _jsonable({k: payload[k] for k in payload if k in {"A", "B"}}),
    )
    print(f"  wrote {path}  ({time.perf_counter() - t0:.1f}s)", flush=True)
    return payload


def _target_dirs(args) -> list:
    if args.params_dir:
        return [args.params_dir]
    return list(STE_PARAM_DIRS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--params-dir",
        default=None,
        help="One folder. Default (no flag): env_para_ste0.2 and env_para_ste0.5.",
    )
    p.add_argument(
        "--all-ste",
        action="store_true",
        help="Calibrate env_para_ste0.2 / 0.5 (the default when --params-dir is omitted).",
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
        help="Cap on doubling until SE(diff) < 2% of |coef_unmodified| (default 256).",
    )
    p.add_argument(
        "--n-boot",
        type=int,
        default=DEFAULT_N_BOOT,
        help="Cluster-bootstrap replicates for SE(diff) (default 200).",
    )
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--max-users", type=int, default=None)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    dirs = _target_dirs(args)
    reports = []
    t_all = time.perf_counter()
    for d in dirs:
        reports.append(calibrate_dir(d, args))
    print(
        f"all environments wall={time.perf_counter() - t_all:.1f}s  "
        f"n_env={len(dirs)}",
        flush=True,
    )
    print(json.dumps(_jsonable(reports), indent=2))


if __name__ == "__main__":
    main()
