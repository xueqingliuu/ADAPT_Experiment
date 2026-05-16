# %%
"""
Pooled linear regression of quadrature-filtered \\hat E_w on weekly summaries from ``df_fit``.

Outcome: ``pred_ml_filtered_Ew`` from ``env_para_vanilla/pred_<ParticipantIdentifier>.json``
(one value per calendar week, aligned with ``week`` 1..12 as in ``perceived_utility``).

Predictors (from ``df_fit`` only, aggregated per participant-week):
  1. J_w — ``week_present`` (constant within week; first non-missing).
  2. 0.5 * J_w * ((U1+1) + (U2+1)) / 8 — U1, U2 raw ``Exp-tool-1``, ``Exp-tool-2``.
  3. PV_sum — (1/14) \\times sum of ``HourlyPageviewCount_norm`` in the week.
  4. FW_sum — (1/7) \\times sum of ``nextday_wearing`` over **calendar days**,
        using only the **first ``DecisionTime`` row per day** (matches ``build_user_blocks``;
        avoids double-counting the two daily RCT decision times).
  5. PJ_sum — (1/7) \\times sum of ``daily_present`` with the same day / first-row rule.

Scaling: 14 hourly PV slots / week; 7 daily FW and PJ values after de-duplication.

Rows with missing outcome or any predictor are dropped before pooling across users.
The regression has **no intercept** (``fit_intercept=False``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPR-MRT-Testbed")
COMBINED_DIR = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPT_MRT/rawdata/_combined")
WORK_DIR = PROJECT_ROOT / "env_para_vanilla"
EW_POOLED_COEF_JSON = WORK_DIR / "Ew_pooled_linear_coefs.json"


def _week_fix_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Match ``perceived_utility`` week relabel and ``week < 13`` filter."""
    df = df.copy()
    for userid in df["ParticipantIdentifier"].unique():
        vc = df.loc[df["ParticipantIdentifier"] == userid, "week"].value_counts()
        if vc.get(0, 0) > 2:
            m = df["ParticipantIdentifier"] == userid
            df.loc[m, "week"] = df.loc[m, "week"] + 1
    return df.loc[df["week"] < 13].copy()


def load_df_fit(path: Path | None = None) -> pd.DataFrame:
    p = path or (COMBINED_DIR / "df_fit.csv")
    return _week_fix_and_filter(pd.read_csv(p))


def _first_nonmissing(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[0]) if len(s) else np.nan


def weekly_predictor_table(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    decision_col: str = "DecisionTime",
) -> pd.DataFrame:
    """One row per (ParticipantIdentifier, week) with predictors from ``df_fit``."""
    rows = []
    for (uid, wk), g in df.groupby(["ParticipantIdentifier", "week"], sort=True):
        g = g.sort_values([date_col, decision_col], na_position="last")
        j_w = _first_nonmissing(g["week_present"])
        u1 = _first_nonmissing(g["Exp-tool-1"])
        u2 = _first_nonmissing(g["Exp-tool-2"])
        if np.isnan(u1):
            u1 = 0.0
        if np.isnan(u2):
            u2 = 0.0
        if np.isnan(j_w):
            j_w = 0.0
        j_w = float(j_w)
        tool_combo = 0.5 * j_w * ((u1 + 1.0) + (u2 + 1.0)) / 8.0
        pv_arr = g["HourlyPageviewCount_norm"].to_numpy(dtype=float)
        pv_sum = float(np.nansum(pv_arr) / 14.0) if pv_arr.size else np.nan

        fw_daily, pj_daily = [], []
        for _, g_day in g.groupby(date_col, sort=True):
            g_day = g_day.sort_values(decision_col, na_position="last")
            row0 = g_day.iloc[0]
            vfw = row0["nextday_wearing"]
            vpj = row0["daily_present"]
            fw_daily.append(float(vfw) if pd.notna(vfw) else np.nan)
            pj_daily.append(float(vpj) if pd.notna(vpj) else np.nan)
        fw_arr = np.asarray(fw_daily, dtype=float)
        pj_arr = np.asarray(pj_daily, dtype=float)
        fw_sum = float(np.nansum(fw_arr) / 7.0) if fw_arr.size else np.nan
        pj_sum = float(np.nansum(pj_arr) / 7.0) if pj_arr.size else np.nan

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
    e = d.get("pred_ml_filtered_Ew")
    if e is None:
        return None
    return [float(x) for x in e]


def align_ew_to_weeks(ew_list: list[float]) -> dict[int, float]:
    """
    Map week index -> E_w. ``perceived_utility`` uses ``full_weeks=range(1, 13)``:
    ``ew_list[i]`` is for week ``i + 1``.
    """
    return {i + 1: ew_list[i] for i in range(len(ew_list))}


def build_pooled_regression_frame(
    df_fit: pd.DataFrame,
    work_dir: Path = WORK_DIR,
) -> pd.DataFrame:
    pred_tbl = weekly_predictor_table(df_fit)
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
                    "pred_ml_filtered_Ew": ew_by_w[wk],
                    "J_w": r["J_w"],
                    "half_J_tool8": r["half_J_tool8"],
                    "PV_sum": r["PV_sum"],
                    "FW_sum": r["FW_sum"],
                    "PJ_sum": r["PJ_sum"],
                }
            )
    return pd.DataFrame(out_rows)


FEATURE_COLS = ("J_w", "half_J_tool8", "PV_sum", "FW_sum", "PJ_sum")


def load_pooled_coefs(work_dir: Path = WORK_DIR) -> dict[str, float]:
    """Load the pooled linear-approximation coefficients written by ``__main__``."""
    p = (work_dir or WORK_DIR) / "Ew_pooled_linear_coefs.json"
    with open(p, encoding="utf-8") as f:
        return {k: float(v) for k, v in json.load(f).items()}


def apply_pooled_coefs(
    coefs: dict,
    J_w: float,
    half_J_tool8: float,
    PV_sum: float,
    FW_sum: float,
    PJ_sum: float,
) -> float:
    """Linear combination ``\\hat E_w = sum_k w_k x_k`` (no intercept)."""
    return float(
        coefs["J_w"] * float(J_w)
        + coefs["half_J_tool8"] * float(half_J_tool8)
        + coefs["PV_sum"] * float(PV_sum)
        + coefs["FW_sum"] * float(FW_sum)
        + coefs["PJ_sum"] * float(PJ_sum)
    )


def initial_Ew_hat_for_user(
    user_id,
    df_fit: pd.DataFrame | None = None,
    work_dir: Path = WORK_DIR,
    coefs: dict | None = None,
) -> float:
    """``\\hat E_w`` for the last pre-RL week of ``user_id`` via the pooled formula.

    Returns ``0.0`` if no eligible week exists in ``df_fit`` for the participant.
    """
    if df_fit is None:
        df_fit = load_df_fit()
    if coefs is None:
        coefs = load_pooled_coefs(work_dir)

    sub = df_fit[df_fit["ParticipantIdentifier"] == user_id]
    if sub.empty:
        return 0.0
    tbl = weekly_predictor_table(sub).dropna(subset=list(FEATURE_COLS))
    if tbl.empty:
        return 0.0
    last = tbl.sort_values("week").iloc[-1]
    return apply_pooled_coefs(
        coefs,
        last["J_w"],
        last["half_J_tool8"],
        last["PV_sum"],
        last["FW_sum"],
        last["PJ_sum"],
    )


def fit_pooled_linear_Ew(
    df_fit: pd.DataFrame | None = None,
    work_dir: Path = WORK_DIR,
):
    if df_fit is None:
        df_fit = load_df_fit()
    frame = build_pooled_regression_frame(df_fit, work_dir=work_dir)
    feature_cols = ["J_w", "half_J_tool8", "PV_sum", "FW_sum", "PJ_sum"]
    m = frame.dropna(subset=["pred_ml_filtered_Ew"] + feature_cols)
    if len(m) < len(feature_cols):
        raise ValueError(
            f"Not enough complete rows for regression (n={len(m)}). "
            "Check ``pred_ml_filtered_Ew`` and ``df_fit``."
        )
    y = m["pred_ml_filtered_Ew"].to_numpy(dtype=float)
    X = m[feature_cols].to_numpy(dtype=float)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)
    names = feature_cols
    coefs = reg.coef_.ravel()
    out = {k: float(c) for k, c in zip(names, coefs)}
    r2 = reg.score(X, y)
    return {"coefficients": out, "r2": r2, "n": len(m), "frame": m}


if __name__ == "__main__":
    result = fit_pooled_linear_Ew()
    print(f'n = {result["n"]}, R^2 = {result["r2"]:.4f}')
    for k, v in result["coefficients"].items():
        print(f'  {k}: {v:.6f}')
    EW_POOLED_COEF_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(EW_POOLED_COEF_JSON, "w", encoding="utf-8") as f:
        json.dump(result["coefficients"], f, indent=2)
    print(f"Saved coefficients to {EW_POOLED_COEF_JSON}")
