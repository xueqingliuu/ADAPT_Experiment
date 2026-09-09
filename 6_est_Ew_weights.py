"""Approximate filtered E_{w+1} with a pooled linear filter of weekly summaries.

Used at RCT time, when the latent E_w from script 4 is not available. The
outcome is the filtered trajectory in ``pred_<uid>.json``. Predictors follow
the script-4 SSM: last week's Ê (AR), this week's PV/FW/PJ (transition),
and this Sunday's ``J_{w+1}`` plus ``J_{w+1}(U_1+U_2)/14`` (measurement
update of the target E). ``E_lag`` is the previous week's *predicted* Ê
(week 1 uses the E_1 pin 2.0), not last Sunday's survey and not script 4's
latent filter.

PV/FW/PJ averages use the same fixed denominators as script 4 (14 slots /
7 days) with missing values as 0. FW therefore treats a missing wear flag
as not wearing, matching the simulator — not a wear-among-observed-days
mean.

Fits nonnegative within-user ridge (demean each person, shared slopes, then
one intercept on the raw scale), iterating ``E_lag`` until the recursive
predictions match the regressor. Writes
``env_para_vanilla/Ew_pooled_linear_coefs.json``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from algorithm_helpers import apply_pooled_coefs

PROJECT_ROOT = Path(
    os.getenv("ADAPR_PROJECT_ROOT", str(Path(__file__).resolve().parent))
).expanduser().resolve()
_combined = os.getenv("ADAPR_COMBINED_DIR", "").strip()
if not _combined:
    raise SystemExit(
        "Set ADAPR_COMBINED_DIR to the folder with extracted MRT tables."
    )
COMBINED_DIR = Path(_combined).expanduser().resolve()
WORK_DIR = PROJECT_ROOT / "env_para_vanilla"
EW_POOLED_COEF_JSON = WORK_DIR / "Ew_pooled_linear_coefs.json"
COEF_DECIMALS = 3
FEATURE_COLS = ["E_lag", "PV_sum", "FW_sum", "PJ_sum", "J_close", "half_J_close"]
OUTCOME_COL = "pred_penalized_filtered_Ew"
DEFAULT_RIDGE_ALPHA = 0.0
DEFAULT_NONNEGATIVE_SLOPES = True
DEFAULT_E_LAG0 = 2.0
MAX_ELAG_ITERS = 20
ELAG_TOL = 1e-6


def _round_decimals(x: float, digits: int = COEF_DECIMALS) -> float:
    return float(np.round(float(x), digits))


def _r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).ravel()
    yhat = np.asarray(yhat, dtype=float).ravel()
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def load_user_ids(work_dir: Path = WORK_DIR) -> set[int] | None:
    path = Path(work_dir) / "user_ids.txt"
    if not path.is_file():
        return None
    return {int(u) for u in np.loadtxt(path, dtype=int).ravel()}


def load_df_fit(path: Path | None = None, work_dir: Path = WORK_DIR) -> pd.DataFrame:
    p = path or (COMBINED_DIR / "df_fit.csv")
    df = pd.read_csv(p)
    df = df.loc[df["week"].between(1, 12)].copy()
    keep = load_user_ids(work_dir)
    if keep:
        df = df.loc[df["ParticipantIdentifier"].astype(int).isin(keep)].copy()
    return df


def _first_nonmissing(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[0]) if len(s) else np.nan


def weekly_response_imputation_means(df: pd.DataFrame) -> dict[str, float]:
    """Pooled means of the first observed U1/U2 response in each user-week."""
    values = {"U1": [], "U2": []}
    for _, g in df.groupby(["ParticipantIdentifier", "week"], sort=False):
        u1 = _first_nonmissing(g["Exp-tool-1"])
        u2 = _first_nonmissing(g["Exp-tool-2"])
        if np.isfinite(u1):
            values["U1"].append(u1)
        if np.isfinite(u2):
            values["U2"].append(u2)
    return {
        key: float(np.mean(observed)) if observed else 0.0
        for key, observed in values.items()
    }


def weekly_predictor_table(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    decision_col: str = "DecisionTime",
    response_imputation_means: dict[str, float] | None = None,
) -> pd.DataFrame:
    """One row per (ParticipantIdentifier, week) with predictors from ``df_fit``.

    ``PV_sum`` / ``FW_sum`` / ``PJ_sum`` are ``nansum / 14`` and ``nansum / 7``
    (missing → 0), matching ``4_perceived_utility.build_user_blocks``. A
    missing Fitbit wear flag is coded as not wearing, not dropped from the
    denominator. ``J_close`` / ``half_J_close`` are this Sunday's
    ``J_{w+1}`` and ``J_{w+1}(U_1+U_2)/14`` (emissions of ``E_{w+1}``).
    ``E_lag`` is filled later from recursive predictions.
    """
    if response_imputation_means is None:
        response_imputation_means = weekly_response_imputation_means(df)
    u1_mean = float(response_imputation_means["U1"])
    u2_mean = float(response_imputation_means["U2"])

    rows = []
    for (uid, wk), g in df.groupby(["ParticipantIdentifier", "week"], sort=True):
        g = g.sort_values([date_col, decision_col], na_position="last")
        j_w = _first_nonmissing(g["week_present"])
        u1 = _first_nonmissing(g["Exp-tool-1"])
        u2 = _first_nonmissing(g["Exp-tool-2"])
        if np.isnan(u1):
            u1 = u1_mean
        if np.isnan(u2):
            u2 = u2_mean
        if np.isnan(j_w):
            j_w = 0.0
        j_w = float(j_w)
        half_j_close = j_w * (u1 + u2) / 14.0
        pv_arr = g["HourlyPageviewCount_norm"].to_numpy(dtype=float)
        pv_arr = np.where(np.isfinite(pv_arr), pv_arr, 0.0)
        pv_sum = float(np.sum(pv_arr) / 14.0) if pv_arr.size else 0.0

        fw_daily, pj_daily = [], []
        for _, g_day in g.groupby(date_col, sort=True):
            g_day = g_day.sort_values(decision_col, na_position="last")
            row0 = g_day.iloc[0]
            vfw = row0["nextday_wearing"]
            vpj = row0["daily_present"]
            fw_daily.append(float(vfw) if pd.notna(vfw) else 0.0)
            pj_daily.append(float(vpj) if pd.notna(vpj) else 0.0)
        # Missing wear → 0 / 7, not mean over observed days (script 4 / vani_env).
        fw_arr = np.asarray(fw_daily, dtype=float)
        pj_arr = np.asarray(pj_daily, dtype=float)
        fw_sum = float(np.sum(fw_arr) / 7.0) if fw_arr.size else 0.0
        pj_sum = float(np.sum(pj_arr) / 7.0) if pj_arr.size else 0.0

        rows.append(
            {
                "ParticipantIdentifier": int(uid) if isinstance(uid, (int, np.integer)) else uid,
                "week": int(wk),
                "J_close": j_w,
                "half_J_close": half_j_close,
                "PV_sum": pv_sum,
                "FW_sum": fw_sum,
                "PJ_sum": pj_sum,
            }
        )
    return pd.DataFrame(rows)


def load_pred_ew_series(work_dir: Path, user_id: int | str) -> list[float] | None:
    uid = int(user_id) if isinstance(user_id, (int, np.integer)) else user_id
    p = work_dir / f"pred_{uid}.json"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    e = d.get("pred_penalized_filtered_Ew")
    if e is None:
        # Backward compatibility with outputs produced before the naming fix.
        e = d.get("pred_ml_filtered_Ew")
    if e is None:
        return None
    return [float(x) for x in e]


def align_ew_to_weeks(ew_list: list[float]) -> dict[int, float]:
    """
    Map study week index -> exported \\hat E_{w+1}. ``ew_list[i]`` is for
    study week ``i + 1`` (matches ``perceived_utility`` in ``df_fit``: week w
    holds \\hat E_{w+1}, with the final entry being the predictive E_13).
    """
    return {i + 1: ew_list[i] for i in range(len(ew_list))}


def _raw_coefs(intercept: float, coef: np.ndarray, feature_cols: list[str]) -> dict[str, float]:
    out = {"intercept": float(intercept)}
    out.update(
        {key: float(value) for key, value in zip(feature_cols, np.asarray(coef, dtype=float).ravel())}
    )
    return out


def _predict_row(coefs: dict[str, float], e_lag: float, row: pd.Series) -> float:
    return apply_pooled_coefs(
        coefs,
        e_lag,
        row["PV_sum"],
        row["FW_sum"],
        row["PJ_sum"],
        row["J_close"],
        row["half_J_close"],
    )


def fill_recursive_E_lag(
    frame: pd.DataFrame,
    coefs: dict[str, float],
    *,
    e_lag0: float = DEFAULT_E_LAG0,
) -> pd.DataFrame:
    """Set ``E_lag`` to the rolled Ê from ``coefs`` (week 1 = ``e_lag0``)."""
    out = frame.copy()
    out["E_lag"] = np.nan
    for _, idx in out.groupby("ParticipantIdentifier", sort=False).groups.items():
        idx_sorted = out.loc[idx].sort_values("week").index
        e = float(e_lag0)
        for i in idx_sorted:
            out.at[i, "E_lag"] = e
            e = _predict_row(coefs, e, out.loc[i])
    return out


def _warm_start_E_lag(
    frame: pd.DataFrame,
    *,
    e_lag0: float = DEFAULT_E_LAG0,
) -> pd.DataFrame:
    """Week 1 = pin; later weeks = previous row's filtered outcome."""
    out = frame.sort_values(["ParticipantIdentifier", "week"]).copy()
    out["E_lag"] = out.groupby("ParticipantIdentifier", sort=False)[OUTCOME_COL].shift(1)
    first = out.groupby("ParticipantIdentifier", sort=False).cumcount() == 0
    out.loc[first, "E_lag"] = float(e_lag0)
    out["E_lag"] = out["E_lag"].fillna(float(e_lag0))
    return out


def _fit_slopes(
    m: pd.DataFrame,
    feature_cols: list[str],
    *,
    within_user: bool,
    ridge_alpha: float,
    nonnegative_slopes: bool,
):
    if within_user:
        return fit_within_user_linear_Ew(
            m,
            feature_cols=feature_cols,
            ridge_alpha=ridge_alpha,
            nonnegative_slopes=nonnegative_slopes,
        )
    y = m[OUTCOME_COL].to_numpy(dtype=float)
    X = m[feature_cols].to_numpy(dtype=float)
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    raw = _raw_coefs(float(reg.intercept_), reg.coef_, feature_cols)
    yhat = raw["intercept"] + X @ np.asarray(reg.coef_, dtype=float).ravel()
    return {
        "coefficients": {
            "intercept": _round_decimals(raw["intercept"], COEF_DECIMALS),
            **{key: _round_decimals(raw[key], COEF_DECIMALS) for key in feature_cols},
        },
        "coef_raw": np.asarray(reg.coef_, dtype=float).ravel(),
        "intercept_raw": float(reg.intercept_),
        "r2_within": None,
        "r2_level": _r2_score(y, yhat),
        "ridge_alpha": None,
        "nonnegative_slopes": False,
    }


def iterate_recursive_E_lag(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    within_user: bool,
    ridge_alpha: float,
    nonnegative_slopes: bool,
    e_lag0: float = DEFAULT_E_LAG0,
    max_iters: int = MAX_ELAG_ITERS,
    tol: float = ELAG_TOL,
):
    """Alternate slope fits and recursive Ê until ``E_lag`` is consistent."""
    m = _warm_start_E_lag(frame, e_lag0=e_lag0)
    last_elag = m["E_lag"].to_numpy(dtype=float)
    n_iters = 0
    for n_iters in range(1, int(max_iters) + 1):
        fit = _fit_slopes(
            m,
            feature_cols,
            within_user=within_user,
            ridge_alpha=ridge_alpha,
            nonnegative_slopes=nonnegative_slopes,
        )
        raw = _raw_coefs(fit["intercept_raw"], fit["coef_raw"], feature_cols)
        m = fill_recursive_E_lag(m, raw, e_lag0=e_lag0)
        new_elag = m["E_lag"].to_numpy(dtype=float)
        if float(np.max(np.abs(new_elag - last_elag))) < float(tol):
            break
        last_elag = new_elag
    fit = _fit_slopes(
        m,
        feature_cols,
        within_user=within_user,
        ridge_alpha=ridge_alpha,
        nonnegative_slopes=nonnegative_slopes,
    )
    return fit, m, n_iters


def build_pooled_regression_frame(
    df_fit: pd.DataFrame,
    work_dir: Path = WORK_DIR,
    *,
    response_imputation_means: dict[str, float] | None = None,
) -> pd.DataFrame:
    pred_tbl = weekly_predictor_table(
        df_fit,
        response_imputation_means=response_imputation_means,
    ).copy()
    out_rows = []
    for uid, sub in pred_tbl.groupby("ParticipantIdentifier"):
        ew = load_pred_ew_series(work_dir, uid)
        if ew is None:
            continue
        ew_by_w = align_ew_to_weeks(ew)
        for _, r in sub.iterrows():
            wk = int(r["week"])
            if wk not in ew_by_w:
                continue
            out_rows.append(
                {
                    "ParticipantIdentifier": r["ParticipantIdentifier"],
                    "week": wk,
                    "pred_penalized_filtered_Ew": ew_by_w[wk],
                    "J_close": r["J_close"],
                    "half_J_close": r["half_J_close"],
                    "PV_sum": r["PV_sum"],
                    "FW_sum": r["FW_sum"],
                    "PJ_sum": r["PJ_sum"],
                }
            )
    return pd.DataFrame(out_rows)


def fit_within_user_linear_Ew(
    m: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    *,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    nonnegative_slopes: bool = DEFAULT_NONNEGATIVE_SLOPES,
):
    """
    Shared within-user slopes via person-mean demeaning (+ optional ridge), plus
    a grand-mean intercept so predictions apply to raw weekly predictors.

    ``ridge_alpha=0`` is within-user OLS; ``ridge_alpha>0`` (default 5) is ridge
    on the demeaned predictors in their original units. When
    ``nonnegative_slopes`` is true, every slope is constrained to be at least
    zero; the reconstructed intercept remains unconstrained.
    """
    ycol = OUTCOME_COL
    md = m.copy()
    for c in feature_cols + [ycol]:
        md[c] = m[c] - m.groupby("ParticipantIdentifier")[c].transform("mean")

    X_dm = md[feature_cols].to_numpy(dtype=float)
    y_dm = md[ycol].to_numpy(dtype=float)
    alpha = float(ridge_alpha)
    if alpha < 0.0:
        raise ValueError(f"ridge_alpha must be >= 0 (got {ridge_alpha})")

    if alpha == 0.0:
        reg = LinearRegression(
            fit_intercept=False,
            positive=bool(nonnegative_slopes),
        )
    else:
        reg = Ridge(
            alpha=alpha,
            fit_intercept=False,
            positive=bool(nonnegative_slopes),
        )
    reg.fit(X_dm, y_dm)
    coef = np.asarray(reg.coef_, dtype=float).ravel()

    X = m[feature_cols].to_numpy(dtype=float)
    y = m[ycol].to_numpy(dtype=float)
    intercept = float(np.mean(y) - np.mean(X, axis=0) @ coef)
    yhat_level = intercept + X @ coef
    yhat_within = X_dm @ coef

    out = {"intercept": _round_decimals(intercept, COEF_DECIMALS)}
    out.update(
        {
            key: _round_decimals(value, COEF_DECIMALS)
            for key, value in zip(feature_cols, coef)
        }
    )
    return {
        "coefficients": out,
        "coef_raw": coef,
        "intercept_raw": intercept,
        "r2_within": _r2_score(y_dm, yhat_within),
        "r2_level": _r2_score(y, yhat_level),
        "ridge_alpha": alpha,
        "nonnegative_slopes": bool(nonnegative_slopes),
    }


def fit_pooled_linear_Ew(
    df_fit: pd.DataFrame | None = None,
    work_dir: Path = WORK_DIR,
    *,
    within_user: bool = True,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    nonnegative_slopes: bool = DEFAULT_NONNEGATIVE_SLOPES,
):
    """
    Fit the agent-visible linear E_w proxy.

    Parameters
    ----------
    within_user :
        If True (default), estimate shared slopes after within-user demeaning and
        reconstruct a grand-mean intercept for raw predictors. If False, use
        ordinary pooled OLS on raw levels (``ridge_alpha`` ignored).
    ridge_alpha :
        Ridge penalty used after within-user demeaning (original predictor units).
        Default 5. Set to 0 for within-user OLS.
    nonnegative_slopes :
        If True (default), constrain every within-user slope to be nonnegative.
        Ignored for pooled OLS.
    """
    if df_fit is None:
        df_fit = load_df_fit(work_dir=work_dir)
    response_means = weekly_response_imputation_means(df_fit)
    frame = build_pooled_regression_frame(
        df_fit,
        work_dir=work_dir,
        response_imputation_means=response_means,
    )
    feature_cols = list(FEATURE_COLS)
    static_cols = [c for c in feature_cols if c != "E_lag"]
    m = frame.dropna(subset=[OUTCOME_COL] + static_cols)
    if len(m) < len(feature_cols):
        raise ValueError(
            f"Not enough complete rows for regression (n={len(m)}). "
            "Check ``pred_penalized_filtered_Ew`` and ``df_fit``."
        )

    n_users = int(m["ParticipantIdentifier"].nunique())
    if within_user and n_users < 2:
        raise ValueError(
            "Within-user demeaning needs at least 2 participants "
            f"(found {n_users})."
        )
    fit, m, n_elag_iters = iterate_recursive_E_lag(
        m,
        feature_cols,
        within_user=within_user,
        ridge_alpha=ridge_alpha,
        nonnegative_slopes=nonnegative_slopes,
    )
    return {
        "coefficients": fit["coefficients"],
        "response_imputation_means": response_means,
        "r2": fit["r2_within"] if within_user else fit["r2_level"],
        "r2_within": fit["r2_within"],
        "r2_level": fit["r2_level"],
        "within_user": bool(within_user),
        "ridge_alpha": fit["ridge_alpha"],
        "nonnegative_slopes": fit["nonnegative_slopes"] if within_user else False,
        "n": len(m),
        "n_users": n_users,
        "n_elag_iters": n_elag_iters,
        "e_lag0": DEFAULT_E_LAG0,
        "frame": m,
    }


if __name__ == "__main__":
    result = fit_pooled_linear_Ew(
        within_user=True,
        ridge_alpha=DEFAULT_RIDGE_ALPHA,
        nonnegative_slopes=DEFAULT_NONNEGATIVE_SLOPES,
    )
    if result["within_user"]:
        alpha = result["ridge_alpha"]
        mode = (
            f"within-user ridge (alpha={alpha:g})"
            if alpha and alpha > 0
            else "within-user OLS"
        )
        if result["nonnegative_slopes"]:
            mode = f"nonnegative {mode}"
    else:
        mode = "pooled OLS"
    print(f"mode = {mode}")
    print(f'n = {result["n"]}, n_users = {result["n_users"]}')
    print(f'E_lag iters = {result["n_elag_iters"]}  (E_lag0 = {result["e_lag0"]:g})')
    if result["within_user"]:
        print(f'R^2_within = {result["r2_within"]:.4f}')
        print(f'R^2_level  = {result["r2_level"]:.4f}  (raw X with grand-mean intercept)')
    else:
        print(f'R^2 = {result["r2"]:.4f}')
    for k, v in result["coefficients"].items():
        print(f"  {k}: {v:.3f}")
    print("Weekly-response imputation means:")
    for k, v in result["response_imputation_means"].items():
        print(f"  {k}: {v:.6f}")
    output = {
        **result["coefficients"],
        "U1_impute_mean": float(result["response_imputation_means"]["U1"]),
        "U2_impute_mean": float(result["response_imputation_means"]["U2"]),
        "E_lag0": float(result["e_lag0"]),
    }
    EW_POOLED_COEF_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(EW_POOLED_COEF_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved coefficients to {EW_POOLED_COEF_JSON}")
