# Standardize the merged panel
#
# Reads `df_merged.csv`, log-transforms most counts in place, z-scores
# continuous features, maps Likert items to [-1, 1], and writes `df_fit.csv`
# plus `std_params.json` (shifts, scales, clip limits) under
# `env_para_vanilla/`. ``HourlyPageviewCount`` (and its lag) stay raw.
# ``HourlyPageviewCount_norm`` is ``log(x)`` then z-scored among positives
# (no ``+1``); count 0 sits at ``log(0.5)`` so PV_sum / lag1 stay below 1.
# Next: ``4_perceived_utility.py``.

# 0. Setup

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(
    os.environ.get("ADAPR_PROJECT_ROOT", Path(__file__).resolve().parent)
).expanduser().resolve()
_combined = os.environ.get("ADAPR_COMBINED_DIR", "").strip()
if not _combined:
    raise SystemExit(
        "Set ADAPR_COMBINED_DIR to the folder with extracted MRT tables."
    )
COMBINED_DIR = Path(_combined).expanduser().resolve()
WORK_DIR = PROJECT_ROOT / "env_para_vanilla"
WORK_DIR.mkdir(parents=True, exist_ok=True)

folder = COMBINED_DIR
work_folder = WORK_DIR
DIGITS = 3

DAY_RANGE = 84
WEEK_RANGE = 12

# Ordinal survey items use a 1..7 scale. Script 2 corrects invalid zeros to one
# before this standardization step.
LIKERT_MIN = 1
LIKERT_MAX = 7

# Columns expected in df_merged (missing cols are skipped with a warning)
FIT_COLUMNS = [
    "ParticipantIdentifier",
    "Date",
    "DecisionTime",
    "week",
    "day",
    "dow",
    "is_weekend",
    # interventions
    "WalkingSuggestion",
    "Interacted_walk",
    "Interacted_7d_walk",
    "planning_prompt",
    "yesterday_planning_prompt",
    "recent_burden",
    # step counts
    "4hour_step",
    "EMA_StepCount",
    "TodayStepCount",
    "YesterdayStepCount",
    "prior2hour_step",
    "EMA_Prior2HourStepCount",
    # wearables / activity
    "morning_wearing",
    "nextday_wearing",
    "past7days_morning_wearing",
    # pageviews
    "DailyPageviewCount",
    "Past7DaysPageviewEMA",
    "HourlyPageviewCount",
    "HourlyPageviewCount_lag1",
    "Past7DaysHourlyPageviewEMA",
    # surveys
    "week_present",
    "week_present_lastweek",
    "daily_present",
    "daily_present_yesterday",
    "affective_reflection",
    "affective_reflection_yesterday",
    "anticipated_affect",
    "anticipated_affect_yesterday",
    "active_status",
    "active_status_yesterday",
    "active_status_fraction_7days",
    "active_status_fraction_7days_yesterday",
    # weekly CAE / tools
    "CAE_avg",
    "CAE_avg_lastweek",
    "CAE_short_avg",
    "Exp-tool-1",
    "Exp-tool-2",
]

# In-place ``log(x+1)`` overwrite. Hourly pageview is excluded: the raw
# integer count is kept, and log-then-z is written only to ``*_norm``.
LOG_COLUMNS = [
    "4hour_step",
    "TodayStepCount",
    "YesterdayStepCount",
    "prior2hour_step",
    "DailyPageviewCount",
    "Past7DaysPageviewEMA",
    "Past7DaysHourlyPageviewEMA",
]

# Source columns whose z-score is of ``log(x)`` for x>0, without mutating ``x``.
# Count 0 is mapped to ``PV_ZERO_ON_LOG_AXIS`` (not ``log(x+1)`` on positives).
KEEP_RAW_LOG_THEN_Z = frozenset({"HourlyPageviewCount"})
PV_ZERO_ON_LOG_AXIS = 0.5

ZSCORE_SPECS = [
    # (source_col, norm_col, json_shift_key, json_scale_key, json_limit_key)
    ("4hour_step", "4hour_step_norm", "4hour_step_count_shift", "4hour_step_count_scale", "4hour_step_count_limit"),
    ("TodayStepCount", "TodayStepCount_norm", "today_step_count_shift", "today_step_count_scale", "TodayStepCount_limit"),
    ("YesterdayStepCount", "YesterdayStepCount_norm", "yesterday_step_count_shift", "yesterday_step_count_scale", "YesterdayStepCount_limit"),
    ("prior2hour_step", "prior2hour_step_norm", "prior2hour_step_count_shift", "prior2hour_step_count_scale", "prior2hour_step_count_limit"),
    ("EMA_StepCount", "EMA_StepCount_norm", "EMA_step_count_shift", "EMA_step_count_scale", "EMA_StepCount_limit"),
    ("EMA_Prior2HourStepCount", "EMA_Prior2HourStepCount_norm", "EMA_prior2hour_step_count_shift", "EMA_prior2hour_step_count_scale", "EMA_Prior2HourStepCount_limit"),
    ("DailyPageviewCount", "DailyPageviewCount_norm", "DailyPageviewCount_shift", "DailyPageviewCount_scale", "DailyPageviewCount_limit"),
    ("Past7DaysPageviewEMA", "Past7DaysPageviewEMA_norm", "Past7DaysPageviewEMA_shift", "Past7DaysPageviewEMA_scale", "Past7DaysPageviewEMA_limit"),
    ("HourlyPageviewCount", "HourlyPageviewCount_norm", "HourlyPageviewCount_shift", "HourlyPageviewCount_scale", "HourlyPageviewCount_limit"),
    ("Past7DaysHourlyPageviewEMA", "Past7DaysHourlyPageviewEMA_norm", "Past7DaysHourlyPageviewEMA_shift", "Past7DaysHourlyPageviewEMA_scale", "Past7DaysHourlyPageviewEMA_limit"),
    ("CAE_avg", "CAE_avg_norm", "CAE_avg_shift", "CAE_avg_scale", "CAE_avg_limit"),
    ("CAE_avg_lastweek", "CAE_avg_lastweek_norm", "CAE_avg_lastweek_shift", "CAE_avg_lastweek_scale", "CAE_avg_lastweek_limit"),
    ("CAE_short_avg", "CAE_short_avg_norm", "CAE_short_avg_shift", "CAE_short_avg_scale", "CAE_short_avg_limit"),
    ("recent_burden", "recent_burden_norm", "recent_burden_shift", "recent_burden_scale", "recent_burden_limit"),
]

# Fractions in [0, 1] — kept raw (no z-score / Likert); only limits for env clipping.
# Binary {0, 1} columns (e.g. active_status, morning_wearing) are left alone.
RAW_UNIT_INTERVAL_COLS = [
    # ActivityCheck rolling fractions
    ("active_status_fraction_7days", "active_status_fraction_7days_limit"),
    ("active_status_fraction_7days_yesterday", "active_status_fraction_7days_yesterday_limit"),
    # Rolling interaction / wear fractions. Interacted_7d_walk is the
    # delivered-only fraction over the prior 14 slots (NaN if none sent).
    ("Interacted_7d_walk", "Interacted_7d_walk_limit"),
    ("past7days_morning_wearing", "past7days_morning_wearing_limit"),
]

LIKERT_SPECS = [
    ("affective_reflection", "affective_reflection_norm", "affective_reflection_limit"),
    ("anticipated_affect", "anticipated_affect_norm", "anticipated_affect_limit"),
    ("affective_reflection_yesterday", "affective_reflection_yesterday_norm", "affective_reflection_yesterday_limit"),
    ("anticipated_affect_yesterday", "anticipated_affect_yesterday_norm", "anticipated_affect_yesterday_limit"),
    ("Exp-tool-1", "Exp-tool-1_norm", "exp1_limit"),
    ("Exp-tool-2", "Exp-tool-2_norm", "exp2_limit"),
]


def _zscore(series, digits=DIGITS):
    vals = pd.to_numeric(series, errors="coerce")
    shift = float(np.round(np.nanmean(vals), digits))
    scale = float(np.round(np.nanstd(vals, ddof=0), digits))
    if scale == 0 or np.isnan(scale):
        scale = 1.0
    norm = (vals - shift) / scale
    limit = [
        float(np.round(np.nanmin(norm), digits)),
        float(np.round(np.nanmax(norm), digits)),
    ]
    return norm, shift, scale, limit


def _likert_norm(
    series,
    digits=DIGITS,
    lo=LIKERT_MIN,
    hi=LIKERT_MAX,
):
    """
    Map ordinal Likert responses on {lo,...,hi} (default 1..7) to [-1, 1].

    Uses:  2 * (x - lo) / (hi - lo) - 1
    so 1 → -1, midpoint 4 → 0, and 7 → 1.
    Values outside [lo, hi] are set to NaN.
    """
    vals = pd.to_numeric(series, errors="coerce")
    invalid = vals.notna() & ((vals < lo) | (vals > hi))
    if invalid.any():
        n_bad = int(invalid.sum())
        print(
            f"Warning: {n_bad} Likert values outside [{lo}, {hi}] "
            f"set to NaN before normalization"
        )
        vals = vals.mask(invalid)

    norm = 2.0 * (vals - lo) / (hi - lo) - 1.0
    if vals.notna().any():
        limit = [
            float(np.round(np.nanmin(norm), digits)),
            float(np.round(np.nanmax(norm), digits)),
        ]
    else:
        limit = [-1.0, 1.0]
    return norm, limit


def _hourly_pv_log_axis(raw, zero_count=PV_ZERO_ON_LOG_AXIS):
    """Log axis for hourly pageview: ``log(x)`` if ``x>0``, else ``log(zero_count)``."""
    vals = pd.to_numeric(raw, errors="coerce")
    out = np.full(len(vals), np.nan, dtype=float)
    v = vals.to_numpy(dtype=float, copy=False)
    pos = np.isfinite(v) & (v > 0.0)
    zero = np.isfinite(v) & (v <= 0.0)
    out[pos] = np.log(v[pos])
    out[zero] = np.log(float(zero_count))
    return out, v, pos


# 1. Load merged panel

df_merged = pd.read_csv(COMBINED_DIR / "df_merged.csv")
print("COMBINED_DIR:", COMBINED_DIR.resolve())
print("WORK_DIR:   ", WORK_DIR.resolve())
print("df_merged columns:", len(df_merged.columns))

missing = [c for c in FIT_COLUMNS if c not in df_merged.columns]
if missing:
    print("Warning: missing columns (skipped):", missing)

use_cols = [c for c in FIT_COLUMNS if c in df_merged.columns]
df_fit = df_merged[use_cols].copy()
print(df_fit.shape)

# 2. Calendar normalization

df_fit["day_norm"] = (df_fit["day"] - (1 + DAY_RANGE) / 2) / ((DAY_RANGE - 1) / 2)
df_fit["week_norm"] = (df_fit["week"] - (1 + WEEK_RANGE) / 2) / ((WEEK_RANGE - 1) / 2)
df_fit["dow_norm"] = (df_fit["dow"] - (1 + 7) / 2) / ((7 - 1) / 2)

# 3. Log transforms (counts / pageviews)
#
# Most count columns are overwritten with ``log(x+1)``. Hourly pageview
# stays raw. Its ``*_norm`` is ``log(x)`` for positive counts (hurdle
# intensity does not need ``+1``); zeros sit at ``log(0.5)`` so the weekly
# PV summary still has a point below count 1.

for col in LOG_COLUMNS:
    if col in df_fit.columns:
        df_fit[col] = np.log(df_fit[col] + 1)

# 4. Z-score + fixed-scale ordinal normalization

std_params = {}

for src, norm, shift_key, scale_key, limit_key in ZSCORE_SPECS:
    if src not in df_fit.columns:
        continue
    if src in KEEP_RAW_LOG_THEN_Z:
        log_axis, _, pos = _hourly_pv_log_axis(df_fit[src])
        log_pos = log_axis[pos]
        shift = float(np.round(np.nanmean(log_pos), DIGITS)) if pos.any() else 0.0
        scale = float(np.round(np.nanstd(log_pos, ddof=0), DIGITS)) if pos.sum() > 1 else 1.0
        if scale == 0 or np.isnan(scale):
            scale = 1.0
        series_norm = (log_axis - shift) / scale
        finite = series_norm[np.isfinite(series_norm)]
        limit = [
            float(np.round(np.nanmin(finite), DIGITS)) if finite.size else -1.0,
            float(np.round(np.nanmax(finite), DIGITS)) if finite.size else 1.0,
        ]
        df_fit[norm] = series_norm
        std_params[shift_key] = shift
        std_params[scale_key] = scale
        std_params[limit_key] = limit
        std_params["HourlyPageviewCount_zero_count"] = PV_ZERO_ON_LOG_AXIS
        continue
    df_fit[norm], shift, scale, limit = _zscore(df_fit[src])
    std_params[shift_key] = shift
    std_params[scale_key] = scale
    std_params[limit_key] = limit

# Keep ``HourlyPageviewCount_lag1`` as a raw count and put it on the same
# log-then-z axis as the outcome (``log(x)`` if x>0, else ``log(0.5)``).
if "HourlyPageviewCount_lag1" in df_fit.columns:
    lag_log, _, _ = _hourly_pv_log_axis(df_fit["HourlyPageviewCount_lag1"])
    df_fit["hourly_pageview_count_lag1"] = (
        lag_log - std_params["HourlyPageviewCount_shift"]
    ) / std_params["HourlyPageviewCount_scale"]

for src, norm, limit_key in LIKERT_SPECS:
    if src not in df_fit.columns:
        continue
    df_fit[norm], limit = _likert_norm(df_fit[src])
    std_params[limit_key] = limit

for col, limit_key in RAW_UNIT_INTERVAL_COLS:
    if col in df_fit.columns:
        std_params[limit_key] = [
            np.round(df_fit[col].min(), DIGITS),
            np.round(df_fit[col].max(), DIGITS),
        ]

# 5. Save `std_params.json`

std_params_path = work_folder / "std_params.json"
with open(std_params_path, "w") as f:
    json.dump(std_params, f)
print(f"Wrote {std_params_path}")

# 6. Decision-slot lags + save `df_fit.csv`

df_fit = df_fit.sort_values(
    ["ParticipantIdentifier", "Date", "DecisionTime"],
    kind="mergesort",
).reset_index(drop=True)

# shift(1) = previous decision slot; shift(2) ≈ prior calendar day for day-level vars
lag_specs = [
    ("4hour_step_norm", "FourSC_lag1", 1),
    ("prior2hour_step_norm", "prior2hour_step_count_lag1", 1),
]
for src, out, n in lag_specs:
    if src in df_fit.columns:
        df_fit[out] = df_fit.groupby("ParticipantIdentifier", sort=False)[src].shift(n)

df_fit_path = folder / "df_fit.csv"
df_fit.to_csv(df_fit_path, index=False)
print(f"Wrote {df_fit_path} ({len(df_fit)} rows)")
