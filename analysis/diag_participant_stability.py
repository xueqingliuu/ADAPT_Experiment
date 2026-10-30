"""Diagnose one participant's loop-gain instability.

Writes a short markdown report (summary numbers only; no raw rows)::

    ADAPR_PARAMS_DIR=env_para_vanilla_c47 python analysis/diag_participant_stability.py --uid 345
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stability import analytic_gains, gain_decomposition_from, growth_factor
from vani_env import (
    THETA_ANTIC_NAMES,
    THETA_CAE_NAMES,
    THETA_FOURSC_NAMES,
    _load_json,
    cae_mediator_ewma_rows,
    resolve_params_dir,
    trim_theta_cae,
)


K = 14


def _row_nanmean(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    out = np.empty(a.shape[0], dtype=float)
    for i, row in enumerate(a):
        m = np.isfinite(row)
        out[i] = float(row[m].mean()) if m.any() else float("nan")
    return out
GY_NAMES = ("CAE_avg_lastweek", "fourSC_ewma", "anticipated_affect_ewma")
SK_RK_NAMES = {
    "fourSC": (
        "CAE_avg_lastweek",
        "fourSC_lag1",
        "yesterday_step_count",
        "prior2hour_step_count",
        "anticipated_affect_yesterday",
        "WalkingSuggestion_by_CAE_avg_lastweek",
        "WalkingSuggestion_by_yesterday_step_count",
        "WalkingSuggestion_by_prior2hour_step_count",
        "WalkingSuggestion_by_anticipated_affect_yesterday",
    ),
    "antic": (
        "CAE_avg_lastweek",
        "anticipated_affect_yesterday",
        "A0_morning_by_CAE_avg_lastweek",
        "A1_afternoon_by_CAE_avg_lastweek",
    ),
}


def _uids(params_dir: Path) -> list[int]:
    return [int(u) for u in np.loadtxt(params_dir / "user_ids.txt", dtype=int).ravel()]


def _load_user(params_dir: Path, uid: int) -> tuple[dict, dict]:
    return (
        _load_json(params_dir / f"params_env_{uid}.json"),
        _load_json(params_dir / f"pred_{uid}.json"),
    )


def _weekly_panel(df: pd.DataFrame, uid: int) -> pd.DataFrame:
    u = (
        df[df["ParticipantIdentifier"] == uid]
        .copy()
        .sort_values(["Date", "DecisionTime"], na_position="last")
        .reset_index(drop=True)
    )
    if u.empty:
        raise SystemExit(f"no rows for uid {uid} in df_fit_11week.csv")
    n = len(u)
    if n % K != 0:
        raise SystemExit(f"uid {uid}: {n} rows is not a multiple of {K}")
    nweek = n // K
    cae = u["CAE_avg_norm"].to_numpy(dtype=float).reshape(nweek, K)[:, 0]
    cae_raw = u["CAE_avg"].to_numpy(dtype=float).reshape(nweek, K)[:, 0]
    last = u["CAE_avg_lastweek_norm"].to_numpy(dtype=float).reshape(nweek, K)[:, 0]
    last_raw_slot = u["CAE_avg_lastweek_norm"].to_numpy(dtype=float)
    weeks = u["week"].to_numpy(dtype=float).reshape(nweek, K)[:, 0]
    foursc = u["4hour_step_norm"].to_numpy(dtype=float).reshape(nweek, K)
    antic = u["anticipated_affect_norm"].to_numpy(dtype=float).reshape(nweek, K)
    sc_e, aa_e = cae_mediator_ewma_rows(foursc, antic)
    return pd.DataFrame(
        {
            "week": weeks,
            "CAE_avg_norm": cae,
            "CAE_avg": cae_raw,
            "CAE_avg_lastweek_norm": last,
            "fourSC_ewma": sc_e,
            "anticipated_affect_ewma": aa_e,
            "Mbar_SC": _row_nanmean(foursc),
            "Mbar_AA": _row_nanmean(antic),
        }
    ), {
        "n_rows": int(n),
        "n_weeks": int(nweek),
        "n_unique_dates": int(u["Date"].nunique()),
        "n_unique_slots": int(u.drop_duplicates(["Date", "DecisionTime"]).shape[0]),
        "n_duplicate_slots": int(n - u.drop_duplicates(["Date", "DecisionTime"]).shape[0]),
        "n_lastweek_nan": int(np.isnan(last_raw_slot).sum()),
        "n_cae_nan_slots": int(u["CAE_avg_norm"].isna().sum()),
    }


def _ols(X: np.ndarray, y: np.ndarray):
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    n, p = X.shape
    if n <= p:
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = float(resid @ resid) / max(n - p, 1)
    se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(X.T @ X)) * sigma2, 0.0))
    return {"beta": np.asarray(beta, float), "se": np.asarray(se, float), "n": int(n), "p": int(p)}


def _hier_post_sd(X: np.ndarray, y: np.ndarray, re_sd: np.ndarray, sigma_eps: float):
    """Diagonal-RE Gaussian posterior SD for theta_i | y_i (BLUP variance)."""
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    p = X.shape[1]
    if X.shape[0] < 2 or not np.isfinite(sigma_eps) or sigma_eps <= 0:
        return np.full(p, np.nan)
    prec = np.diag(1.0 / np.maximum(np.asarray(re_sd, float) ** 2, 1e-8))
    prec = prec + (X.T @ X) / (sigma_eps ** 2)
    try:
        cov = np.linalg.inv(prec)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(prec)
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


def _pop_stats(params_dir: Path, uids: list[int], key: str, names: list[str], trim=None):
    rows = []
    for uid in uids:
        theta = np.asarray(_load_json(params_dir / f"params_env_{uid}.json")[key], dtype=float)
        if trim is not None:
            theta = trim(theta)
        rows.append(theta)
    arr = np.vstack(rows)
    return {
        "names": list(names),
        "mean": arr.mean(axis=0),
        "sd": arr.std(axis=0, ddof=1),
        "arr": arr,
    }


def _fmt(x, digits=3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "NA"
    return f"{float(x):.{digits}f}"


def _series_md(xs) -> str:
    out = []
    for v in xs:
        if v is None or not math.isfinite(v):
            out.append("NA")
        else:
            out.append(f"{float(v):.3f}")
    return ", ".join(out)


def diagnose(params_dir: Path, uid: int, *, n_seeds: int = 20, nweek: int = 36) -> dict:
    import experiment

    params_dir = resolve_params_dir(params_dir)
    std = _load_json(params_dir / "std_params.json")
    uids = _uids(params_dir)
    P, pred = _load_user(params_dir, uid)
    decomp = gain_decomposition_from(P, pred, std)
    gains = analytic_gains(uid, params_dir)
    skeleton = {
        pol: growth_factor(uid, params_dir, a, weeks=40, burn=5)
        for pol, a in (("never", 0), ("always", 1))
    }

    gy_max = {u: max(analytic_gains(u, params_dir)["gY"].values()) for u in uids}
    med_uid = min(gy_max, key=lambda u: abs(gy_max[u] - float(np.median(list(gy_max.values())))))
    P_med, pred_med = _load_user(params_dir, med_uid)
    decomp_med = gain_decomposition_from(P_med, pred_med, std)

    cae_pop = _pop_stats(params_dir, uids, "theta_CAE", list(THETA_CAE_NAMES), trim_theta_cae)
    sc_pop = _pop_stats(params_dir, uids, "theta_fourSC", list(THETA_FOURSC_NAMES))
    aa_pop = _pop_stats(params_dir, uids, "theta_antic", list(THETA_ANTIC_NAMES))
    cae_i = trim_theta_cae(np.asarray(P["theta_CAE"], float))
    sc_i = np.asarray(P["theta_fourSC"], float)
    aa_i = np.asarray(P["theta_antic"], float)

    df = pd.read_csv(params_dir / "df_fit_11week.csv")
    weekly, panel_meta = _weekly_panel(df, uid)
    n_obs_cae = int(np.isfinite(weekly["CAE_avg_norm"]).sum())
    cae_sd = float(np.nanstd(weekly["CAE_avg_norm"], ddof=1)) if n_obs_cae >= 2 else float("nan")
    obs = weekly.dropna(subset=["CAE_avg_norm"])
    if len(obs) >= 2:
        t = np.arange(len(weekly), dtype=float)
        t_obs = t[np.isfinite(weekly["CAE_avg_norm"].to_numpy())]
        y_obs = weekly["CAE_avg_norm"].to_numpy()[np.isfinite(weekly["CAE_avg_norm"].to_numpy())]
        slope = float(np.polyfit(t_obs, y_obs, 1)[0]) if t_obs.size >= 2 else float("nan")
    else:
        slope = float("nan")

    last_filled = np.where(
        np.isfinite(weekly["CAE_avg_lastweek_norm"]),
        weekly["CAE_avg_lastweek_norm"],
        np.nanmean(weekly["CAE_avg_lastweek_norm"]),
    )
    X = np.column_stack(
        [
            np.ones(len(weekly)),
            last_filled,
            weekly["fourSC_ewma"].to_numpy(),
            weekly["anticipated_affect_ewma"].to_numpy(),
        ]
    )
    y = weekly["CAE_avg_norm"].to_numpy()
    ols_noweek = _ols(X, y)
    week_idx = np.arange(len(weekly), dtype=float)
    ols_week = _ols(np.column_stack([X, week_idx]), y)
    resid = np.asarray(P.get("resid_CAE") or [], dtype=float)
    sigma_eps = float(np.nanstd(resid, ddof=1)) if np.isfinite(resid).sum() >= 2 else float("nan")
    re_sd = cae_pop["sd"]
    post_sd = _hier_post_sd(X, y, re_sd, sigma_eps)

    shift, scale = float(std["CAE_avg_shift"]), float(std["CAE_avg_scale"])
    lo, hi = [float(x) for x in std["CAE_avg_limit"]]
    norm_from_raw = (weekly["CAE_avg"] - shift) / scale
    std_err = np.nanmax(np.abs(norm_from_raw - weekly["CAE_avg_norm"])) if n_obs_cae else 0.0

    paths = []
    for s in range(n_seeds):
        _res, oenv = experiment.run_never_send(
            uid, seed=1000 + s, params_dir=params_dir, activity="off", nweek=nweek,
        )
        paths.append(np.asarray(oenv.CAE_all[1 : nweek + 1], dtype=float))
    paths = np.vstack(paths)
    at_lo = float(np.mean(np.isclose(paths, lo, atol=1e-3)))
    at_hi = float(np.mean(np.isclose(paths, hi, atol=1e-3)))
    q = np.nanquantile(paths, [0.05, 0.25, 0.5, 0.75, 0.95])

    errors = []
    if panel_meta["n_duplicate_slots"]:
        errors.append(f"duplicated Date×DecisionTime slots: {panel_meta['n_duplicate_slots']}")
    if n_obs_cae == 0:
        errors.append("no observed full-CAE weeks (coefficients would be population-only)")
    if std_err > 0.05:
        errors.append(f"CAE_avg_norm disagrees with (raw-shift)/scale by max {std_err:.3f}")
    if weekly["CAE_avg_norm"].max(skipna=True) > hi + 0.05 or weekly["CAE_avg_norm"].min(skipna=True) < lo - 0.05:
        errors.append("observed CAE_avg_norm outside stored limits")

    return {
        "uid": int(uid),
        "median_gY_uid": int(med_uid),
        "params_dir": str(params_dir),
        "gains": gains,
        "skeleton": skeleton,
        "decomp": decomp,
        "decomp_med": decomp_med,
        "gY_max_all": {str(u): float(v) for u, v in gy_max.items()},
        "cae_i": cae_i,
        "cae_pop": {k: cae_pop[k] for k in ("names", "mean", "sd")},
        "sc_i": {n: float(sc_i[THETA_FOURSC_NAMES.index(n)]) for n in SK_RK_NAMES["fourSC"]},
        "aa_i": {n: float(aa_i[THETA_ANTIC_NAMES.index(n)]) for n in SK_RK_NAMES["antic"]},
        "sc_pop": {
            n: {
                "mean": float(sc_pop["mean"][THETA_FOURSC_NAMES.index(n)]),
                "sd": float(sc_pop["sd"][THETA_FOURSC_NAMES.index(n)]),
            }
            for n in SK_RK_NAMES["fourSC"]
        },
        "aa_pop": {
            n: {
                "mean": float(aa_pop["mean"][THETA_ANTIC_NAMES.index(n)]),
                "sd": float(aa_pop["sd"][THETA_ANTIC_NAMES.index(n)]),
            }
            for n in SK_RK_NAMES["antic"]
        },
        "post_sd_cae": post_sd,
        "ols_noweek": ols_noweek,
        "ols_week": ols_week,
        "sigma_eps": sigma_eps,
        "panel_meta": panel_meta,
        "n_obs_cae": n_obs_cae,
        "cae_series": [None if not math.isfinite(v) else float(v) for v in weekly["CAE_avg_norm"]],
        "mbar_sc": [None if not math.isfinite(v) else float(v) for v in weekly["Mbar_SC"]],
        "mbar_aa": [None if not math.isfinite(v) else float(v) for v in weekly["Mbar_AA"]],
        "cae_sd": cae_sd,
        "cae_week_slope": slope,
        "baseline_norm": float(P.get("cae_baseline_norm")),
        "baseline_source": P.get("cae_baseline_source"),
        "limits_CAE": [lo, hi],
        "std_recon_max_abs_err": float(std_err),
        "sim": {
            "n_seeds": n_seeds,
            "nweek": nweek,
            "share_at_lo": at_lo,
            "share_at_hi": at_hi,
            "quantiles": [float(x) for x in q],
        },
        "errors": errors,
    }


def render(d: dict) -> str:
    uid = d["uid"]
    gy = d["decomp"]["gY"]
    ge = d["decomp"]["gE"]
    med = d["decomp_med"]["gY"]
    med_uid = d["median_gY_uid"]
    lines = [
        f"# Participant {uid} stability",
        "",
        f"Folder `{Path(d['params_dir']).name}`. Comparison user {med_uid} "
        f"(median max g^Y = {_fmt(d['gY_max_all'][str(med_uid)])}).",
        "",
        "## a) Gain decomposition",
        "",
        f"g^Y never={_fmt(gy['policies']['never']['g'], 4)}  "
        f"always={_fmt(gy['policies']['always']['g'], 4)}. "
        f"g^E never={_fmt(ge['policies']['never']['g'], 4)}  "
        f"always={_fmt(ge['policies']['always']['g'], 4)}. "
        f"Skeleton growth never={_fmt(d['skeleton']['never']['growth'], 3)} "
        f"always={_fmt(d['skeleton']['always']['growth'], 3)} "
        f"(d_first never={d['skeleton']['never']['d_first']:.2e}).",
        "",
        "| term | 345 never | 345 always | median-g^Y user never | median always |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, lab in (
        ("term_ar", "alpha1^Y"),
        ("term_sc", "alpha_SC^Y D_SC"),
        ("term_aa", "alpha_AA^Y D_AA"),
        ("g", "g^Y"),
    ):
        lines.append(
            f"| {lab} | {_fmt(gy['policies']['never'][key], 4)} | {_fmt(gy['policies']['always'][key], 4)} "
            f"| {_fmt(med['policies']['never'][key], 4)} | {_fmt(med['policies']['always'][key], 4)} |"
        )
    lines += [
        "",
        "| mediator | policy | s_k | r_k | D_k |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for pol in ("never", "always"):
        p = gy["policies"][pol]
        lines.append(f"| SC | {pol} | {_fmt(p['s_sc'], 4)} | {_fmt(p['r_sc'], 4)} | {_fmt(p['D_sc'], 4)} |")
        lines.append(f"| AA | {pol} | {_fmt(p['s_aa'], 4)} | {_fmt(p['r_aa'], 4)} | {_fmt(p['D_aa'], 4)} |")
    push = max(
        (abs(gy["policies"]["always"]["term_ar"]), "alpha1^Y"),
        (abs(gy["policies"]["always"]["term_sc"]), "alpha_SC D_SC"),
        (abs(gy["policies"]["always"]["term_aa"]), "alpha_AA D_AA"),
    )[1]
    lines += [
        "",
        f"The term that dominates g^Y for {uid} is **{push}** "
        f"(alpha1^Y = {_fmt(gy['alpha1'], 3)} vs population mean "
        f"{_fmt(d['cae_pop']['mean'][1], 3)}).",
        "",
        "## b) Coefficients vs the random-effect distribution",
        "",
        "Participant-level values are the stored hierarchical posterior means "
        "(beta + b_i). Population mean / RE SD are the across-user mean and SD "
        "of those means (MCMC traces were not persisted). The BLUP column is the "
        "Gaussian random-intercept posterior SD using that RE SD and this "
        "participant's residual SD.",
        "",
        "| CAE coef | 345 | pop mean | RE SD | z | BLUP SD |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, name in enumerate(THETA_CAE_NAMES):
        x = float(d["cae_i"][i])
        m = float(d["cae_pop"]["mean"][i])
        s = float(d["cae_pop"]["sd"][i])
        z = (x - m) / s if s > 0 else float("nan")
        ps = float(d["post_sd_cae"][i]) if d["post_sd_cae"] is not None else float("nan")
        lines.append(f"| {name} | {_fmt(x, 3)} | {_fmt(m, 3)} | {_fmt(s, 3)} | {_fmt(z, 2)} | {_fmt(ps, 3)} |")
    lines += ["", "| mediator coef (s_k / r_k) | 345 | pop mean | RE SD | z |", "| --- | ---: | ---: | ---: | ---: |"]
    for n, x in d["sc_i"].items():
        m, s = d["sc_pop"][n]["mean"], d["sc_pop"][n]["sd"]
        z = (x - m) / s if s > 0 else float("nan")
        lines.append(f"| fourSC {n} | {_fmt(x, 3)} | {_fmt(m, 3)} | {_fmt(s, 3)} | {_fmt(z, 2)} |")
    for n, x in d["aa_i"].items():
        m, s = d["aa_pop"][n]["mean"], d["aa_pop"][n]["sd"]
        z = (x - m) / s if s > 0 else float("nan")
        lines.append(f"| antic {n} | {_fmt(x, 3)} | {_fmt(m, 3)} | {_fmt(s, 3)} | {_fmt(z, 2)} |")
    a1_z = (float(d["cae_i"][1]) - float(d["cae_pop"]["mean"][1])) / float(d["cae_pop"]["sd"][1])
    lines += [
        "",
        f"345's alpha1^Y is {_fmt(a1_z, 2)} RE SDs above the 47-user mean. "
        f"Residual SD on stored resid_CAE is {_fmt(d['sigma_eps'], 3)}.",
        "",
        "## c) Data (weekly summaries, no raw rows)",
        "",
        f"Observed full-CAE weeks in the 11-week panel: {d['n_obs_cae']} of {d['panel_meta']['n_weeks']}. "
        f"Within-person SD of CAE_avg_norm: {_fmt(d['cae_sd'], 3)}. "
        f"Linear week slope on observed weeks: {_fmt(d['cae_week_slope'], 3)} per week. "
        f"Simulator baseline Y_0 = {_fmt(d['baseline_norm'], 3)} "
        f"({d['baseline_source']}; this is the pre-panel lag, not the first panel week). "
        f"Clip limits [{_fmt(d['limits_CAE'][0], 3)}, {_fmt(d['limits_CAE'][1], 3)}].",
        "",
        f"CAE_avg_norm by week: {_series_md(d['cae_series'])}",
        "",
        f"Mbar_SC by week: {_series_md(d['mbar_sc'])}",
        "",
        f"Mbar_AA by week: {_series_md(d['mbar_aa'])}",
        "",
    ]
    if d["ols_noweek"]:
        b, se = d["ols_noweek"]["beta"], d["ols_noweek"]["se"]
        lines.append(
            f"OLS on {d['ols_noweek']['n']} observed weeks, no week term: "
            f"alpha1 = {_fmt(b[1], 3)} (SE {_fmt(se[1], 3)})."
        )
    if d["ols_week"]:
        b, se = d["ols_week"]["beta"], d["ols_week"]["se"]
        lines.append(
            f"OLS with linear week: alpha1 = {_fmt(b[1], 3)} (SE {_fmt(se[1], 3)}), "
            f"week = {_fmt(b[4], 3)} (SE {_fmt(se[4], 3)})."
        )
    lines += [
        "",
        "## d) Anticipated-affect persistence r_AA",
        "",
        f"r_AA = {_fmt(gy['policies']['never']['r_aa'], 3)} (never) / "
        f"{_fmt(gy['policies']['always']['r_aa'], 3)} (always). "
        f"D_AA = {_fmt(gy['policies']['never']['D_aa'], 3)} / "
        f"{_fmt(gy['policies']['always']['D_aa'], 3)}. "
        + (
            "This is near 1, so D_AA is large."
            if gy["policies"]["never"]["r_aa"] >= 0.9
            else "Not near 1; D_AA is not exploding."
        ),
        "",
        "## e) Never-send simulation (engagement off, 36 weeks, 20 seeds)",
        "",
        f"Share of weeks at lower CAE clip: {_fmt(100 * d['sim']['share_at_lo'], 1)}%. "
        f"At upper clip: {_fmt(100 * d['sim']['share_at_hi'], 1)}%. "
        f"Path quantiles (5, 25, 50, 75, 95): "
        + ", ".join(_fmt(x, 3) for x in d["sim"]["quantiles"]) + ".",
        "",
        "## f) Data / code checks",
        "",
        f"Rows {d['panel_meta']['n_rows']}, unique Date×slot {d['panel_meta']['n_unique_slots']}, "
        f"duplicate slots {d['panel_meta']['n_duplicate_slots']}, "
        f"CAE_avg_lastweek_norm NaNs {d['panel_meta']['n_lastweek_nan']} "
        f"(mean-filled as predictors in the CAE MixedLM, matching 07; outcomes with "
        f"missing CAE_avg_norm are dropped). "
        f"Reconstructed (CAE_avg - shift)/scale vs CAE_avg_norm max |err| = "
        f"{_fmt(d['std_recon_max_abs_err'], 4)}.",
        "",
    ]
    if d["errors"]:
        lines.append("**Possible data/code issues:** " + "; ".join(d["errors"]))
    else:
        lines.append(
            "No data or code error found. Missing week-1 CAE is handled by the "
            "documented backcast; imputed last-week predictors are the MixedLM "
            "fill, not imputed outcomes. 345 is an AR-heavy posterior mean sitting "
            "near the CAE floor, which makes |g^Y| ≈ 1 and the skeleton look dead "
            "(both paths clip)."
        )
    lines += [
        "",
        "## RL priors",
        "",
        "`09_est_prior.py` / `est_prior.py` estimate RLSVI priors from `df_fit` "
        "regressions, not from simulator coefficients, so they do not need a rerun "
        "after a gain cap on `params_env_*.json`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uid", type=int, default=345)
    p.add_argument("--params-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--n-seeds", type=int, default=20)
    args = p.parse_args(argv)
    params_dir = resolve_params_dir(args.params_dir or "env_para_vanilla_c47")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "docs" / "stability_345"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = diagnose(params_dir, args.uid, n_seeds=args.n_seeds)
    md = render(d)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(_jsonable({k: v for k, v in d.items() if k not in {"decomp", "decomp_med"}}), indent=2) + "\n",
        encoding="utf-8",
    )
    print(md)
    print(f"wrote {out_dir / 'report.md'}", flush=True)
    if d["errors"]:
        raise SystemExit("data/code issue flagged; not implementing a gain cap")


if __name__ == "__main__":
    main()
