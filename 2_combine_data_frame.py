# %% [markdown]
# # Merge extraction CSVs into one decision-time panel
#
# Reads the tables from `1_data_extraction.py`, builds two rows per day
# (morning / afternoon), merges surveys, wear, page views, and interventions,
# drops burn-in and weeks 0/13, and writes `df_merged.csv`.
# Next: `3_standardization.py`.

# %% [markdown]
# ## 0. Setup

# %%
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FOLDER = Path(
    "/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPT_MRT/Xueqing"
)
folder = DATA_FOLDER

# Drop first BURN_IN_DAYS calendar days after each participant's earliest
# decision-panel date (Date >= min_date + BURN_IN_DAYS). With BURN_IN_DAYS=6,
# that removes 6 days; remaining calendar + week filters yield 84 analysis days.
BURN_IN_DAYS = 6
EXCLUDED_WEEKS = (0, 13)

DAILY_SURVEY_COLS = [
    "daily_present",
    "affective_reflection",
    "anticipated_affect",
    "active_status",
    "active_status_fraction_7days",
]

# These questionnaire responses use a 1--7 scale. A small number of extracted
# records encode the minimum response as 0; correct those records before
# constructing weekly/daily lags or summary scores. Binary presence/status
# indicators are intentionally excluded because zero is valid for them.
WEEKLY_ONE_TO_SEVEN_COLS = [
    "AffectiveValuation",
    "Exp-tool-1",
    "Exp-tool-2",
    *[f"CAE-{i}" for i in range(1, 13)],
]
DAILY_ONE_TO_SEVEN_COLS = [
    "affective_reflection",
    "anticipated_affect",
]


def _as_str_id(df, col="ParticipantIdentifier"):
    if col in df.columns:
        df[col] = df[col].astype(str)
    return df


def _correct_one_to_seven_zeros(df, columns, *, source):
    """Replace invalid zero responses with the minimum valid response, one."""
    corrected = {}
    for col in columns:
        if col not in df.columns:
            continue
        zero_mask = df[col].eq(0)
        count = int(zero_mask.sum())
        if count:
            df.loc[zero_mask, col] = 1
            corrected[col] = count
    if corrected:
        details = ", ".join(f"{col}={count}" for col, count in corrected.items())
        print(f"Corrected zero-valued 1--7 responses in {source}: {details}")
    return df


# %% [markdown]
# ## 1. Load extracted tables

# %%
df_4hour_step = _as_str_id(pd.read_csv(folder / "hourly_step_counts.csv"))
df_today_step = _as_str_id(pd.read_csv(folder / "today_step_counts.csv"))
df_prior2hours_step = _as_str_id(pd.read_csv(folder / "prior_2hours_step_counts.csv"))

df_weekly = _as_str_id(pd.read_csv(folder / "df_weekly_filled.csv"))
df_daily = _as_str_id(pd.read_csv(folder / "df_daily_filled.csv"))
df_weekly = _correct_one_to_seven_zeros(
    df_weekly,
    WEEKLY_ONE_TO_SEVEN_COLS,
    source="df_weekly_filled.csv",
)
df_daily = _correct_one_to_seven_zeros(
    df_daily,
    DAILY_ONE_TO_SEVEN_COLS,
    source="df_daily_filled.csv",
)

df_daily_pageview = _as_str_id(pd.read_csv(folder / "df_daily_pageview.csv"))
df_hourly_pageview = _as_str_id(pd.read_csv(folder / "hourly_pageview.csv"))

df_gif = _as_str_id(pd.read_csv(folder / "df_gif_all.csv"))
df_planning = _as_str_id(pd.read_csv(folder / "df_end_all.csv"))

df_notwearing = _as_str_id(pd.read_csv(folder / "missing_days.csv"))
df_wearing_morning = _as_str_id(pd.read_csv(folder / "wear_day.csv"))

# %% [markdown]
# ## 2. Parse dates

# %%
df_4hour_step["Date"] = pd.to_datetime(df_4hour_step["Date"])
df_today_step["Date"] = pd.to_datetime(df_today_step["Date"])
df_prior2hours_step["Date"] = pd.to_datetime(df_prior2hours_step["Date"])

df_weekly["Date"] = pd.to_datetime(df_weekly["date"])
df_daily["Date"] = pd.to_datetime(df_daily["date"])

df_daily_pageview["Date"] = pd.to_datetime(df_daily_pageview["Date"])
df_hourly_pageview["Date"] = pd.to_datetime(df_hourly_pageview["Date"])

df_gif["Date"] = pd.to_datetime(df_gif["Date"])
df_planning["Date"] = pd.to_datetime(df_planning["date"])

df_notwearing["Date"] = pd.to_datetime(df_notwearing["Date"])
df_wearing_morning["Date"] = pd.to_datetime(df_wearing_morning["Date"])

# %% [markdown]
# ## 3. Base panel + wearable / engagement merges

# %%
print(f"Before merging: {len(df_4hour_step)} decision-time rows")

df_merged = df_4hour_step.copy()

df_merged = df_merged.merge(df_today_step, on=["ParticipantIdentifier", "Date"], how="left")
df_merged = df_merged.merge(
    df_prior2hours_step,
    on=["ParticipantIdentifier", "Date", "DecisionTime"],
    how="left",
)
df_merged = df_merged.merge(df_notwearing, on=["ParticipantIdentifier", "Date"], how="left")
df_merged = df_merged.merge(df_wearing_morning, on=["ParticipantIdentifier", "Date"], how="left")
df_merged = df_merged.merge(df_daily_pageview, on=["ParticipantIdentifier", "Date"], how="left")
df_merged = df_merged.merge(
    df_hourly_pageview,
    on=["ParticipantIdentifier", "Date", "DecisionTime"],
    how="left",
)

print(df_merged.head())

# %% [markdown]
# ## 4. Weekly survey (ISO week join + `_lastweek` lags)

# %%
# ISO week/year keys for joining weekly survey to daily decision rows
df_weekly["iso_week"] = pd.to_datetime(df_weekly["date"]).dt.isocalendar().week
df_weekly["iso_year"] = pd.to_datetime(df_weekly["date"]).dt.isocalendar().year

df_weekly_unique = df_weekly.groupby(
    ["ParticipantIdentifier", "iso_week", "iso_year"], as_index=False
).first()

df_merged["iso_week"] = df_merged["Date"].dt.isocalendar().week
df_merged["iso_year"] = df_merged["Date"].dt.isocalendar().year

# Keep survey slot index as survey_week; study calendar `week` is built later.
if "week" in df_weekly_unique.columns:
    df_weekly_unique = df_weekly_unique.rename(columns={"week": "survey_week"})

weekly_cols = [
    c
    for c in df_weekly_unique.columns
    if c
    not in [
        "ParticipantIdentifier",
        "date",
        "Date",
        "iso_week",
        "iso_year",
        "observed_date",
    ]
]

df_merged = df_merged.merge(
    df_weekly_unique[["ParticipantIdentifier", "iso_week", "iso_year"] + weekly_cols],
    on=["ParticipantIdentifier", "iso_week", "iso_year"],
    how="left",
)

# Resolve duplicate Date columns from weekly merge
if "Date_y" in df_merged.columns:
    df_merged = df_merged.drop(columns=["Date_y"])
if "Date_x" in df_merged.columns:
    df_merged = df_merged.rename(columns={"Date_x": "Date"})

# Lag weekly columns within participant (prior survey week)
for col in weekly_cols:
    if col in df_weekly_unique.columns:
        df_weekly_unique[f"{col}_lastweek"] = (
            df_weekly_unique.groupby("ParticipantIdentifier")[col].shift(1)
        )

lastweek_cols = [c for c in df_weekly_unique.columns if c.endswith("_lastweek")]
df_merged = df_merged.merge(
    df_weekly_unique[["ParticipantIdentifier", "iso_week", "iso_year"] + lastweek_cols],
    on=["ParticipantIdentifier", "iso_week", "iso_year"],
    how="left",
    suffixes=("", "_dup"),
)

print(f"After weekly merge: {len(df_merged)} rows")

# %% [markdown]
# ## 5. Daily EOD survey (+ yesterday lags)

# %%
for col in DAILY_SURVEY_COLS:
    if col in df_daily.columns:
        df_daily[f"{col}_yesterday"] = df_daily.groupby("ParticipantIdentifier")[col].shift(1)

daily_merge_cols = ["ParticipantIdentifier", "Date"]
for col in DAILY_SURVEY_COLS:
    if col in df_daily.columns:
        daily_merge_cols.extend([col, f"{col}_yesterday"])

df_merged = df_merged.merge(df_daily[daily_merge_cols], on=["ParticipantIdentifier", "Date"], how="left")

# An unmatched participant-date means no daily survey response. This includes
# eligible participants with no daily-survey result rows at all.
df_merged["daily_present"] = df_merged["daily_present"].fillna(0).astype(int)

# %% [markdown]
# ## 6. Interventions (planning, walking suggestions)

# %%
planning_cols = [
    c for c in df_planning.columns if c not in ["ParticipantIdentifier", "Date", "time", "date"]
]

df_merged = df_merged.merge(
    df_planning[["ParticipantIdentifier", "Date"] + planning_cols],
    on=["ParticipantIdentifier", "Date"],
    how="left",
)

df_planning_yesterday = df_planning[["ParticipantIdentifier", "Date"] + planning_cols].copy()
df_planning_yesterday["Date"] = df_planning_yesterday["Date"] + pd.Timedelta(days=1)
df_planning_yesterday.columns = ["ParticipantIdentifier", "Date"] + [
    f"yesterday_{c}" for c in planning_cols
]
df_merged = df_merged.merge(df_planning_yesterday, on=["ParticipantIdentifier", "Date"], how="left")

# Walking suggestions (includes recent_burden from extraction)
df_merged = df_merged.merge(
    df_gif,
    on=["ParticipantIdentifier", "Date", "DecisionTime"],
    how="left",
)

# %% [markdown]
# ## 7. Column cleanup after overlapping merges

# %%
print(df_merged.columns)

drop_cols = [c for c in ["date", "Time_x", "Time_y", "iso_week", "iso_year"] if c in df_merged.columns]
df_merged = df_merged.drop(columns=drop_cols)

rename_map = {
    "Interacted": "Interacted_walk",
    "Interacted_7d": "Interacted_7d_walk",
    "Interacted_x": "Interacted_walk",
    "Interacted_7d_x": "Interacted_7d_walk",
    "StepCount_x": "4hour_step",
    "StepCount_y": "prior2hour_step",
    "CheckStatus_x": "CheckStatus_4hour",
    "CheckStatus_y": "CheckStatus_prior2hour",
}
df_merged = df_merged.rename(columns={k: v for k, v in rename_map.items() if k in df_merged.columns})

# Fail loudly if merge suffixes were left unresolved
_leftover = [c for c in df_merged.columns if c.endswith("_x") or c.endswith("_y")]
if _leftover:
    print("WARNING: unresolved merge-suffix columns:", _leftover)

# %% [markdown]
# ## 8. Burn-in filter + calendar indices

# %%
df_merged = df_merged.copy()
df_merged["Date"] = pd.to_datetime(df_merged["Date"], errors="coerce").dt.normalize()

min_date = df_merged.groupby("ParticipantIdentifier")["Date"].transform("min")
cutoff = min_date + pd.Timedelta(days=BURN_IN_DAYS)
df_merged = df_merged[df_merged["Date"] >= cutoff].reset_index(drop=True)

# Day index from first retained calendar date (shared by AM/PM slots)
_d = df_merged["Date"]
df_merged["day"] = (
    df_merged.assign(_dn=_d)
    .sort_values(["ParticipantIdentifier", "Date"])
    .groupby("ParticipantIdentifier")["_dn"]
    .transform(lambda d: (d - d.min()).dt.days)
)

# Monday=1 … Sunday=7
df_merged["dow"] = df_merged["Date"].dt.dayofweek + 1

# Study week index (Mon–Sun blocks); overwrites ISO week used only for joining
df_merged = df_merged.sort_values(["ParticipantIdentifier", "Date"])
_start_dow = df_merged.groupby("ParticipantIdentifier", sort=False)["Date"].transform("min").dt.dayofweek
_wk = ((df_merged["day"] + _start_dow) // 7).astype(int)
_starts_monday = _start_dow.eq(0)
df_merged["week"] = _wk + _starts_monday.astype(int)

df_merged["is_weekend"] = (df_merged["dow"] >= 6).astype(int)

sort_cols = [c for c in ["ParticipantIdentifier", "Date", "DateTimeStart"] if c in df_merged.columns]
df_merged = df_merged.sort_values(sort_cols)

print(df_merged.groupby("ParticipantIdentifier")["Date"].nunique())
print(df_merged.groupby("ParticipantIdentifier").size())

# %% [markdown]
# ## 9. Derived weekly summaries + week filter

# %%
cae_cols = [f"CAE-{i}" for i in range(1, 13)]
cae_lastweek_cols = [f"CAE-{i}_lastweek" for i in range(1, 13)]

df_merged["CAE_avg"] = df_merged[cae_cols].mean(axis=1)
df_merged["CAE_avg_lastweek"] = df_merged[cae_lastweek_cols].mean(axis=1)
df_merged["CAE_short_avg"] = df_merged[["CAE-3", "CAE-5", "CAE-11"]].mean(axis=1)

df_merged = df_merged[~df_merged["week"].isin(EXCLUDED_WEEKS)].reset_index(drop=True)

# Re-index days after dropping partial edge weeks so downstream day_norm uses 1..84.
df_merged = df_merged.sort_values(
    ["ParticipantIdentifier", "Date", "DecisionTime"],
    kind="mergesort",
).reset_index(drop=True)
df_merged["day"] = (
    df_merged.groupby("ParticipantIdentifier")["Date"]
    .transform(lambda s: s.rank(method="dense").astype(int))
)

print(df_merged.groupby("ParticipantIdentifier")["week"].unique())
print(df_merged.groupby("ParticipantIdentifier").size())

# %% [markdown]
# ## 10. Save

# %%
df_merged.to_csv(folder / "df_merged.csv", index=False)
print(f"Wrote {folder / 'df_merged.csv'} ({len(df_merged)} rows)")
