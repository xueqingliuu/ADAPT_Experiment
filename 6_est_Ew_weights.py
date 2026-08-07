# %%
"""
Pooled linear regression of quadrature-filtered \\hat E_w on weekly summaries from ``df_fit``.

Outcome: ``pred_penalized_filtered_Ew`` from ``env_para_vanilla/pred_<ParticipantIdentifier>.json``
(one value per study week 1..12: entry i is \\hat E_{i+1}, i.e. E_2..E_13
filtered plus a predictive E_13 for the last study week, matching
``perceived_utility`` in ``df_fit``).

Predictors (from ``df_fit`` only, aggregated per participant-week):
  1. J_w — ``week_present`` (constant within week; first non-missing).
  2. 0.5 * J_w * ((U1+1) + (U2+1)) / 8 — U1, U2 raw ``Exp-tool-1``, ``Exp-tool-2``.
  3. PV_sum — (1/14) \\times sum of ``HourlyPageviewCount_norm`` in the week.
  4. FW_sum — (1/7) \\times sum of ``nextday_wearing`` over **calendar days**,
        using only the **first ``DecisionTime`` row per day** (matches ``build_user_blocks``;
        avoids double-counting the two daily RCT decision times).
  5. PJ_sum — (1/7) \\times sum of ``daily_present`` with the same day / first-row rule.

Scaling: 14 hourly PV slots / week; 7 daily FW and PJ values after de-duplication.

Missing PV, FW, and PJ measurements are imputed as zero before weekly aggregation.
Missing U1 and U2 weekly responses are imputed with their pooled observed weekly means.
Rows with missing outcomes are dropped before pooling across users.

By default the slopes are **within-user ridge**: outcome and predictors are
demeaned within ``ParticipantIdentifier``, then shared ridge (no intercept) is
fit on the demeaned predictors in their original units. A single grand-mean
intercept is then chosen so ``E ≈ intercept + β'X`` still applies to raw weekly
predictors (``intercept = mean(E) - β'mean(X)``), matching ``apply_pooled_coefs``.

Default ``ridge_alpha=3`` shrinks the collinear ``J_w`` / ``half_J_tool8``
pair enough that both slopes stay nonnegative, with little loss of within-user
fit relative to OLS.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

PROJECT_ROOT = Path(
    os.getenv("ADAPR_PROJECT_ROOT", str(Path(__file__).resolve().parent))
).expanduser().resolve()
COMBINED_DIR = Path(
    os.getenv(
        "ADAPR_COMBINED_DIR",
        "/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPT_MRT/Xueqing",
    )
).expanduser().resolve()
WORK_DIR = PROJECT_ROOT / "env_para_vanilla"
EW_POOLED_COEF_JSON = WORK_DIR / "Ew_pooled_linear_coefs.json"
COEF_DECIMALS = 3
FEATURE_COLS = ["J_w", "half_J_tool8", "PV_sum", "FW_sum", "PJ_sum"]
OUTCOME_COL = "pred_penalized_filtered_Ew"
DEFAULT_RIDGE_ALPHA = 3.0


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


def load_df_fit(path: Path | None = None) -> pd.DataFrame:
    p = path or (COMBINED_DIR / "df_fit.csv")
    df = pd.read_csv(p)
    return df.loc[df["week"].between(1, 12)].copy()


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
    """One row per (ParticipantIdentifier, week) with predictors from ``df_fit``."""
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
        tool_combo = 0.5 * j_w * ((u1 + 1.0) + (u2 + 1.0)) / 8.0
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
        fw_arr = np.asarray(fw_daily, dtype=float)
        pj_arr = np.asarray(pj_daily, dtype=float)
        fw_sum = float(np.sum(fw_arr) / 7.0) if fw_arr.size else 0.0
        pj_sum = float(np.sum(pj_arr) / 7.0) if pj_arr.size else 0.0

        rows.append(
            {
                "ParticipantIdentifier": int(uid) if isinstance(uid, (int, np.integer)) else uid,
                "week": int(wk),
                "J_w": j_w,
                "half_J_tool8": tool_combo,
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


def build_pooled_regression_frame(
    df_fit: pd.DataFrame,
    work_dir: Path = WORK_DIR,
    *,
    response_imputation_means: dict[str, float] | None = None,
) -> pd.DataFrame:
    pred_tbl = weekly_predictor_table(
        df_fit,
        response_imputation_means=response_imputation_means,
    )
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
                    "J_w": r["J_w"],
                    "half_J_tool8": r["half_J_tool8"],
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
):
    """
    Shared within-user slopes via person-mean demeaning (+ optional ridge), plus a
    grand-mean intercept so predictions apply to raw weekly predictors.

    ``ridge_alpha=0`` is within-user OLS; ``ridge_alpha>0`` is ridge on the
    demeaned predictors in their original units.
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
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_dm, y_dm)
        coef = np.asarray(reg.coef_, dtype=float).ravel()
    else:
        reg = Ridge(alpha=alpha, fit_intercept=False)
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
    }


def fit_pooled_linear_Ew(
    df_fit: pd.DataFrame | None = None,
    work_dir: Path = WORK_DIR,
    *,
    within_user: bool = True,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
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
        Default 3. Set to 0 for within-user OLS.
    """
    if df_fit is None:
        df_fit = load_df_fit()
    response_means = weekly_response_imputation_means(df_fit)
    frame = build_pooled_regression_frame(
        df_fit,
        work_dir=work_dir,
        response_imputation_means=response_means,
    )
    feature_cols = list(FEATURE_COLS)
    m = frame.dropna(subset=[OUTCOME_COL] + feature_cols)
    if len(m) < len(feature_cols):
        raise ValueError(
            f"Not enough complete rows for regression (n={len(m)}). "
            "Check ``pred_penalized_filtered_Ew`` and ``df_fit``."
        )

    if within_user:
        n_users = int(m["ParticipantIdentifier"].nunique())
        if n_users < 2:
            raise ValueError(
                "Within-user demeaning needs at least 2 participants "
                f"(found {n_users})."
            )
        fit = fit_within_user_linear_Ew(
            m,
            feature_cols=feature_cols,
            ridge_alpha=ridge_alpha,
        )
        return {
            "coefficients": fit["coefficients"],
            "response_imputation_means": response_means,
            "r2": fit["r2_within"],
            "r2_within": fit["r2_within"],
            "r2_level": fit["r2_level"],
            "within_user": True,
            "ridge_alpha": fit["ridge_alpha"],
            "n": len(m),
            "n_users": n_users,
            "frame": m,
        }

    y = m[OUTCOME_COL].to_numpy(dtype=float)
    X = m[feature_cols].to_numpy(dtype=float)
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    out = {"intercept": _round_decimals(reg.intercept_, COEF_DECIMALS)}
    out.update(
        {
            key: _round_decimals(value, COEF_DECIMALS)
            for key, value in zip(feature_cols, reg.coef_.ravel())
        }
    )
    r2 = float(reg.score(X, y))
    return {
        "coefficients": out,
        "response_imputation_means": response_means,
        "r2": r2,
        "r2_within": None,
        "r2_level": r2,
        "within_user": False,
        "ridge_alpha": None,
        "n": len(m),
        "n_users": int(m["ParticipantIdentifier"].nunique()),
        "frame": m,
    }


if __name__ == "__main__":
    result = fit_pooled_linear_Ew(within_user=True, ridge_alpha=DEFAULT_RIDGE_ALPHA)
    if result["within_user"]:
        alpha = result["ridge_alpha"]
        mode = (
            f"within-user ridge (alpha={alpha:g})"
            if alpha and alpha > 0
            else "within-user OLS"
        )
    else:
        mode = "pooled OLS"
    print(f"mode = {mode}")
    print(f'n = {result["n"]}, n_users = {result["n_users"]}')
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
    }
    EW_POOLED_COEF_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(EW_POOLED_COEF_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved coefficients to {EW_POOLED_COEF_JSON}")
