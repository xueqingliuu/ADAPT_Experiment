# %% [markdown]
# # Standardize `df_merged` → `df_fit.csv` + `std_params.json`
#
# Reads `df_merged.csv` from `2_combine_data_frame.py`.
#
# ## Pipeline
# 1. Select analysis columns
# 2. Calendar covariates → `*_norm` in [-1, 1]
# 3. Log-transform count outcomes / pageviews
# 4. Z-score continuous features; ordinal 0–7 surveys → [-1, 1]
# 5. Write `std_params.json` (shifts, scales, limits) to `env_para_vanilla/`
# 6. Add decision-slot lags; save `df_fit.csv`

# %% [markdown]
# ## 0. Setup

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPR-MRT-Testbed")
COMBINED_DIR = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPT_MRT/Xueqing")
WORK_DIR = PROJECT_ROOT / "env_para_vanilla"
WORK_DIR.mkdir(parents=True, exist_ok=True)

folder = COMBINED_DIR
work_folder = WORK_DIR
DIGITS = 3

DAY_RANGE = 84
WEEK_RANGE = 12

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
    "SalienceMessage",
    "Interacted_salience",
    "yesterday_SalienceMessage",
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
    "RecordedPhysicalActivity",
    "Previous7DaysRPA",
    "morning_wearing",
    "nextday_wearing",
    "past7days_morning_wearing",
    # pageviews
    "DailyPageviewCount",
    "Past7DaysPageviewEMA",
    "HourlyPageviewCount",
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

LOG_COLUMNS = [
    "4hour_step",
    "TodayStepCount",
    "YesterdayStepCount",
    "prior2hour_step",
    "DailyPageviewCount",
    "Past7DaysPageviewEMA",
    "HourlyPageviewCount",
    "Past7DaysHourlyPageviewEMA",
]

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
    # Rolling interaction / wear / RPA fractions
    ("Interacted_7d_walk", "Interacted_7d_walk_limit"),
    ("Previous7DaysRPA", "Previous7DaysRPA_limit"),
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
    shift = np.round(np.mean(series), digits)
    scale = np.round(np.std(series), digits)
    if scale == 0:
        scale = 1.0
    norm = (series - shift) / scale
    limit = [np.round(norm.min(), digits), np.round(norm.max(), digits)]
    return norm, shift, scale, limit


def _likert_norm(series, digits=DIGITS):
    # Fixed-scale normalization for ordinal survey items coded on 0..7.
    # Maps scale midpoint to 0 and keeps outputs comparable to other
    # standardized predictors on approximately [-1, 1].
    norm = 2 * series / 7 - 1
    limit = [np.round(norm.min(), digits), np.round(norm.max(), digits)]
    return norm, limit


# %% [markdown]
# ## 1. Load merged panel

# %%
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

# %% [markdown]
# ## 2. Calendar normalization

# %%
df_fit["day_norm"] = (df_fit["day"] - (1 + DAY_RANGE) / 2) / ((DAY_RANGE - 1) / 2)
df_fit["week_norm"] = (df_fit["week"] - (1 + WEEK_RANGE) / 2) / ((WEEK_RANGE - 1) / 2)
df_fit["dow_norm"] = (df_fit["dow"] - (1 + 7) / 2) / ((7 - 1) / 2)

# %% [markdown]
# ## 3. Log transforms (counts / pageviews)

# %%
for col in LOG_COLUMNS:
    if col in df_fit.columns:
        df_fit[col] = np.log(df_fit[col] + 1)

# %% [markdown]
# ## 4. Z-score + fixed-scale ordinal normalization

# %%
std_params = {}

for src, norm, shift_key, scale_key, limit_key in ZSCORE_SPECS:
    if src not in df_fit.columns:
        continue
    df_fit[norm], shift, scale, limit = _zscore(df_fit[src])
    std_params[shift_key] = shift
    std_params[scale_key] = scale
    std_params[limit_key] = limit

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

# %% [markdown]
# ## 5. Save `std_params.json`

# %%
std_params_path = work_folder / "std_params.json"
with open(std_params_path, "w") as f:
    json.dump(std_params, f)
print(f"Wrote {std_params_path}")

# %% [markdown]
# ## 6. Decision-slot lags + save `df_fit.csv`

# %%
df_fit = df_fit.sort_values(
    ["ParticipantIdentifier", "Date", "DecisionTime"],
    kind="mergesort",
).reset_index(drop=True)

# shift(1) = previous decision slot; shift(2) ≈ prior calendar day for day-level vars
lag_specs = [
    ("4hour_step_norm", "FourSC_lag1", 1),
    ("prior2hour_step_norm", "prior2hour_step_count_lag1", 1),
    ("HourlyPageviewCount_norm", "hourly_pageview_count_lag1", 1),
    ("RecordedPhysicalActivity", "recorded_physical_activity_lag1", 2),
]
for src, out, n in lag_specs:
    if src in df_fit.columns:
        df_fit[out] = df_fit.groupby("ParticipantIdentifier", sort=False)[src].shift(n)

df_fit_path = folder / "df_fit.csv"
df_fit.to_csv(df_fit_path, index=False)
print(f"Wrote {df_fit_path} ({len(df_fit)} rows)")
