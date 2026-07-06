# %% [markdown]
# # ADAPTS MRT — data extraction
#
# Notebook-style script (`# %%` cells). Outputs CSVs under `DATA_FOLDER`.
#
# ## Pipeline
# 1. Setup — imports, paths, timezone lookup
# 2. Cohort — testers removed; ≥83-day active span; manual exclusions
# 3. Surveys — weekly (12) and daily (85) filled panels
# 4. Schedule — wakeup/bedtime (+ push-timing imputation)
# 5. Engagement — page views
# 6. Interventions — walking suggestions, salience, planning prompts
# 7. Wearables — HR/steps, wear flags, step features
# 8. Fitbit activity log — recorded physical activity

# %% [markdown]
# ## 0. Setup

# %%
import datetime
import json
import os
import re
from itertools import zip_longest
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

plt.ion()

DATA_FOLDER = Path(
    "/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPT_MRT/Xueqing"
)
folder = DATA_FOLDER

MIN_ACTIVE_SPAN_DAYS = 83
# Manual wearable-quality exclusions (see wearable_step_quality_audit.csv after extraction):
# 112: no usable HR and all-zero steps in the active window
# 117: no usable HR even though step rows are present
# 219: no usable HR and very sparse nonzero step data
# 13:  step/HR rows present but HourWearing never flagged during decision windows
# 22:  sparse prior2hour but zero post-decision hourly/FourSC step counts
# 248, 291: no usable step counts in prior2hour / hourly / today decision windows
EXCLUDED_PARTICIPANT_IDS = ("112", "117", "219", "13", "22", "248", "291")
EXTRA_TESTER_IDS = ("test-Yuxuan",)

# %%
# Device metadata (timezone lookup source)

project_device_data = pd.read_csv(folder / "ProjectDeviceData_selected_fields_combined.csv")

# %% [markdown]
# ## 1a. Timezone lookup (UTC → participant local)

# %%
# project_device_data should already contain:
# ParticipantIdentifier (or participantidentifier), date, timeZone, utcOffset

tz_lookup = (
    project_device_data.rename(columns={"participantidentifier": "ParticipantIdentifier"})
    .copy()
)

# keep only needed columns
tz_lookup = tz_lookup[["ParticipantIdentifier", "date", "timeZone", "utcOffset"]].copy()

# normalize date dtype
tz_lookup["date"] = pd.to_datetime(tz_lookup["date"], errors="coerce").dt.date

# pick one timezone record per participant/date (first non-null values)
tz_lookup = (
    tz_lookup.sort_values(["ParticipantIdentifier", "date"])
             .groupby(["ParticipantIdentifier", "date"], as_index=False)
             .agg({
                 "timeZone": lambda s: next((x for x in s if pd.notna(x) and str(x).strip() != ""), np.nan),
                 "utcOffset": lambda s: next((x for x in s if pd.notna(x) and str(x).strip() != ""), np.nan),
             })
)

# -----------------------------
# 2) Helpers: convert UTC -> local per row
# -----------------------------
def _offset_to_timedelta(offset_str):
    # supports formats like -07:00:00 or +05:30:00
    if pd.isna(offset_str):
        return pd.NaT
    m = re.match(r"^([+-])(\d{2}):(\d{2})(?::(\d{2}))?$", str(offset_str).strip())
    if not m:
        return pd.NaT
    sign = -1 if m.group(1) == "-" else 1
    hh = int(m.group(2))
    mm = int(m.group(3))
    ss = int(m.group(4) or 0)
    return sign * pd.Timedelta(hours=hh, minutes=mm, seconds=ss)

def _convert_one_utc_to_local(ts_utc, tz_name, utc_offset):
    if pd.isna(ts_utc):
        return pd.NaT

    ts_utc = pd.to_datetime(ts_utc, errors="coerce", utc=True)
    if pd.isna(ts_utc):
        return pd.NaT

    # Preferred: IANA timezone (handles DST correctly)
    if pd.notna(tz_name) and str(tz_name).strip() != "":
        try:
            return ts_utc.tz_convert(str(tz_name)).tz_localize(None)
        except Exception:
            pass

    # Fallback: static UTC offset (no DST logic)
    delta = _offset_to_timedelta(utc_offset)
    if pd.isna(delta):
        return ts_utc.tz_localize(None)  # fallback to UTC naive
    return (ts_utc + delta).tz_localize(None)

def convert_utc_columns_to_user_local(
    df,
    datetime_cols,
    participant_col="ParticipantIdentifier",
    join_date_col="InsertedDate",  # use "Timestamp" for pageview
):
    out = df.copy()

    out["_join_date"] = (
        pd.to_datetime(out[join_date_col], errors="coerce", utc=True).dt.date
    )

    out = out.merge(
        tz_lookup,
        how="left",
        left_on=[participant_col, "_join_date"],
        right_on=["ParticipantIdentifier", "date"],
        suffixes=("", "_tz"),
    )

    for col in datetime_cols:
        out[col] = [
            _convert_one_utc_to_local(ts, tz, off)
            for ts, tz, off in zip(out[col], out["timeZone"], out["utcOffset"])
        ]

    out = out.drop(
        columns=["_join_date", "ParticipantIdentifier_tz", "date", "timeZone", "utcOffset"],
        errors="ignore",
    )
    return out


# %% [markdown]
# ## 1b. Participant cohort

# %%
testers = pd.read_csv(
    folder / "Testers.csv"
)

testers_id = testers.ParticipantIdentifier.values

# add 'test-Yuxuan' to the testers_id
testers_id = np.concatenate([testers_id, list(EXTRA_TESTER_IDS)])

all_participant_ids = project_device_data.participantidentifier.unique()

print(all_participant_ids)

real_participant_ids = all_participant_ids[~np.isin(all_participant_ids, testers_id)]

print(real_participant_ids)



# %% [markdown]
# ## 2. Survey tasks & results

# %%
# survey task and survey question are linked by surveykey (surveytask.surveykey = surveyquestion.surveykey)
# but this does not distinguish between different days and different participants and different questions
# participantidentifier
# resultidentifier
# answers
# question_startdate
# question_enddate

surveytask = pd.read_csv(
    folder / "SurveyTasks.csv"
)
surveyquestionresults = pd.read_csv(
    folder / "SurveyResults.csv"
)

# add a date column to the dataframe and remove duplicate rows
surveytask["date"] = pd.to_datetime(surveytask["InsertedDate"]).dt.date

surveyquestionresults["date"] = pd.to_datetime(surveyquestionresults["InsertedDate"]).dt.date

# surveytask
# surveytask = convert_utc_columns_to_user_local(
#     surveytask,
#     datetime_cols=["InsertedDate", "DueDate"],
#     participant_col="ParticipantIdentifier"
# )

# surveyquestionresults (adjust participant_col if column name differs)
# surveyquestionresults = convert_utc_columns_to_user_local(
#     surveyquestionresults,
#     datetime_cols=["StartDate", "EndDate", "InsertedDate"],  # keep only columns that exist
#     participant_col="ParticipantIdentifier"
# )

survey_key_weekly = surveytask[surveytask.SurveyName == 'MRT - Weekly Check-in survey'].SurveyKey.values[0]
survey_key_monthly = surveytask[surveytask.SurveyName == 'MRT - Monthly check-in survey and goal setting'].SurveyKey.values[0]
survey_key_last_monthly = surveytask[surveytask.SurveyName == 'MRT - Monthly check-in survey (Final month)'].SurveyKey.values[0]
survey_key_daily = surveytask[surveytask.SurveyName == 'MRT - Daily End of day survey and Planning Exercise'].SurveyKey.values[0]

# transform survey keys to lowercase to match the surveyquestionresults.surveykey
survey_key_weekly = survey_key_weekly.lower()
survey_key_monthly = survey_key_monthly.lower()
survey_key_last_monthly = survey_key_last_monthly.lower()
survey_key_daily = survey_key_daily.lower()

print(survey_key_weekly, survey_key_monthly, survey_key_last_monthly, survey_key_daily)

# remove testers from surveytask and surveyquestionresults
surveytask = surveytask[~surveytask.ParticipantIdentifier.isin(testers_id)]
surveyquestionresults = surveyquestionresults[~surveyquestionresults.ParticipantIdentifier.isin(testers_id)]

print(surveyquestionresults.head())
# %% [markdown]
# ## 1c. Active-phase span (daily EOD survey tasks)

# %%
surveytask['date'] = pd.to_datetime(surveytask['InsertedDate']).dt.date
# survey_task_active = surveytask[surveytask.surveyname == 'MRT - Salience - Message Display']
survey_task_active = surveytask[surveytask.SurveyName == 'MRT - Daily End of day survey and Planning Exercise']
summary_surveytask = (
    survey_task_active
    .groupby('ParticipantIdentifier')['date']
    .agg(date_min='min', date_max='max', n_days='nunique')
    .assign(span_days=lambda df: (pd.to_datetime(df.date_max) -
                                  pd.to_datetime(df.date_min)).dt.days + 1)
)


# make participantidentifier a column name
summary_surveytask.reset_index(inplace=True)
summary_surveytask.rename(columns={'ParticipantIdentifier': 'ParticipantIdentifier'}, inplace=True)



# correct for missing day 0 for participant 22
# summary_surveytask.loc[summary_surveytask['ParticipantIdentifier'] == 22, 'date_min'] = pd.to_datetime('2025-06-29').date()
# summary_surveytask['date_min'] = pd.to_datetime(summary_surveytask['date_min']).dt.normalize()

print(summary_surveytask)

# filter out participants who have span_days longer than 84 days
span_qualified_participant_ids = summary_surveytask[
    summary_surveytask['span_days'] >= MIN_ACTIVE_SPAN_DAYS
]['ParticipantIdentifier'].unique()
print(span_qualified_participant_ids, len(span_qualified_participant_ids))

complete_participant_ids = np.setdiff1d(
    span_qualified_participant_ids, list(EXCLUDED_PARTICIPANT_IDS)
)
print(complete_participant_ids, len(complete_participant_ids))

# filter out incomplete participants from surveytask and surveyquestionresults
surveytask = surveytask[surveytask['ParticipantIdentifier'].isin(complete_participant_ids)]
surveyquestionresults = surveyquestionresults[surveyquestionresults['ParticipantIdentifier'].isin(complete_participant_ids)]

# print(surveytask.ParticipantIdentifier.unique())
# print(surveyquestionresults.ParticipantIdentifier.unique())

# sort surveytask and surveyquestionresults by participantidentifier and date
surveytask = surveytask.sort_values(by=['ParticipantIdentifier', 'date'])
surveyquestionresults = surveyquestionresults.sort_values(by=['ParticipantIdentifier', 'date'])

# print(surveytask.head())
# print(surveyquestionresults.head())


# %% [markdown]
# ## Extract four variables from weekly survey: affective valuation, CAE, perceived helpfulness, perceived pleasantness
# - keep participant identifier
# - keep delivery date and time
# - when the surveys are not responded, we use NA to indicate missingness
# - each participant should have exactly 12 rows of four variables
# 
# - add adherence indicator and column for participant identifier (j=1 if respond)
# - according to the date, the gap should be about 7 days (+/- 1 day)
# - use NANs to fill the missing values
# 
# 
# - time zone issue: since it's daily 6pm, even though due to the use of UTC, it will be around 10pm. This will not flow to the next day, so it's fine for now!!!!!!!!

# %% [markdown]
# ### 3a. Flatten nested SurveyResults JSON

# %%

def _to_obj(x):
    if isinstance(x, (list, dict)):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return []
    return []

flat_rows = []

for _, row in surveyquestionresults.iterrows():
    pid = row.get("ParticipantIdentifier")
    survey_key = row.get("SurveyKey")
    inserted = row.get("InsertedDate")

    # IMPORTANT: parse SurveyResults here
    survey_results = _to_obj(row.get("StepResults"))
    if isinstance(survey_results, dict):
        survey_results = [survey_results]

    for step in survey_results:
        if not isinstance(step, dict):
            continue

        step_id = step.get("StepIdentifier")
        step_start = step.get("StartDate")
        step_end = step.get("EndDate")

        results = _to_obj(step.get("Results"))
        if isinstance(results, dict):
            results = [results]

        for r in results:
            if not isinstance(r, dict):
                continue

            ans = r.get("Answers")
            if isinstance(ans, list):
                ans_first = ans[0] if ans else np.nan
                ans_raw = "|".join(map(str, ans))
            else:
                ans_first = ans
                ans_raw = str(ans) if ans is not None else np.nan

            flat_rows.append({
                "ParticipantIdentifier": pid,
                "SurveyKey": survey_key,
                "InsertedDate": inserted,
                "StepIdentifier": step_id,
                "StepStartDate": step_start,
                "StepEndDate": step_end,
                "ResultType": r.get("Type"),
                "ResultIdentifier": r.get("ResultIdentifier"),
                "AnswerFirst": ans_first,
                "AnswersRaw": ans_raw,
                "QuestionStartDate": r.get("StartDate"),
                "QuestionEndDate": r.get("EndDate"),
            })

survey_results_flat = pd.DataFrame(flat_rows)
print(survey_results_flat.shape)
survey_results_flat.head()

# %%
questions = ["AffectiveValuation", "Exp-tool-1", "Exp-tool-2"] + [f"CAE-{i}" for i in range(1, 13)]

tmp = (
    survey_results_flat
    .loc[survey_results_flat["ResultIdentifier"].isin(questions)]
    .copy()
)

# Keep parsed datetime for ordering/debug (UTC-normalized)
tmp["datetime"] = pd.to_datetime(tmp["QuestionStartDate"], errors="coerce", utc=True)

print(tmp["datetime"].dtype)

# Numeric answer
tmp["value"] = pd.to_numeric(tmp["AnswerFirst"], errors="coerce")

# Local date from original offset timestamp string (YYYY-MM-DD part)
tmp["date"] = pd.to_datetime(
    tmp["QuestionStartDate"].astype(str).str.slice(0, 10),
    errors="coerce"
).dt.date

tmp = tmp.dropna(subset=["datetime", "date"])

df_weekly_survey = (
    tmp.pivot_table(
        index=["ParticipantIdentifier", "date"],
        columns="ResultIdentifier",
        values="value",
        aggfunc="last"
    )
    .reset_index()
)

df_weekly_survey.head()

# %%
# extract weekly survey from survey task
survey_task_weekly = surveytask.loc[
    surveytask["SurveyName"].isin([
        "MRT - Weekly Check-in survey",
        "MRT - Monthly check-in survey and goal setting",
        "MRT - Monthly check-in survey (Final month)"
    ])
].copy()

survey_task_weekly.loc[:, "date"] = pd.to_datetime(
    survey_task_weekly["InsertedDate"], errors="coerce"
)

print(survey_task_weekly.head())

# %%
# force both to pandas datetime64[ns] (naive midnight)
df_weekly_survey["date"] = pd.to_datetime(df_weekly_survey["date"], errors="coerce").dt.normalize()
survey_task_weekly["date"] = pd.to_datetime(survey_task_weekly["date"], errors="coerce").dt.normalize()

print(df_weekly_survey["date"].dtype)
print(survey_task_weekly["date"].dtype)

# %%
def fill_weekly_12(df1, df2, id_col="ParticipantIdentifier", date_col="date",
                         tolerance_days=2, weeks=12):
    """
    df1: wide weekly-level survey frame with [id_col, date_col, value columns...]
    df2: wide weekly-level survey task frame with [id_col, date_col, survey task...]
    Returns a frame with exactly `weeks` rows per participant, anchored at the
    participant's earliest observed date, spaced at 7-day intervals (± tolerance).
    """
    # ensure datetime (normalized to date)
    df1 = df1.copy()
    df1[date_col] = pd.to_datetime(df1[date_col], errors="coerce").dt.normalize()

    df2 = df2.copy()
    df2[date_col] = (
        pd.to_datetime(df2[date_col], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )

    anchor_lookup = (
        df2.groupby(id_col)[date_col]
           .min()  # earliest task date per participant
    )

    # value columns to carry through
    value_cols = [c for c in df1.columns if c not in [id_col, date_col]]

    out_parts = []
    for pid, g in df1.groupby(id_col, sort=False):

        anchor = anchor_lookup.get(pid)  # earliest observed date

        # expected weekly slots
        slots = pd.DataFrame({
            "idx": np.arange(weeks, dtype=int),
            id_col: pid,
            "date": [anchor + pd.Timedelta(days=7*i) for i in range(weeks)]
        })

        # assign each observed date to the nearest weekly slot
        g = g.assign(
            idx=np.round((g[date_col] - anchor).dt.days / 7.0).astype(int)
        )
        # distance from its slot in absolute days
        g = g.assign(
            _slot_date=lambda d: anchor + pd.to_timedelta(d["idx"]*7, unit="D"),
            _abs_diff=lambda d: (d[date_col] - d["_slot_date"]).abs()
        )
        # keep only rows that fall within tolerance and within slot range
        g = g[(g["idx"] >= 0) & (g["idx"] < weeks) &
                (g["_abs_diff"] <= pd.Timedelta(days=tolerance_days))]

        # if multiple observed dates map to the same slot, keep the nearest
        g = (g.sort_values(["_abs_diff", date_col])
                .drop_duplicates(subset=["idx"], keep="first"))

        # prepare for merge
        keep_cols = [id_col, "idx", date_col] + value_cols
        g = g[keep_cols].rename(columns={date_col: "observed_date"})

        # merge slots with matched observations
        merged = slots.merge(g, on=[id_col, "idx"], how="left")

        # present indicator: 1 if we matched a row to this slot, else 0
        merged["week_present"] = merged["observed_date"].notna().astype(int)

        # ensure all value cols exist (NaN if missing)
        for c in value_cols:
            if c not in merged.columns:
                merged[c] = np.nan

        # tidy columns
        merged = merged.drop(columns=["idx"]).sort_values(["date"])
        out_parts.append(merged)

    out = pd.concat(out_parts, ignore_index=True)

    # order columns: id, date (expected slot), observed_date, present, then values
    ordered = [id_col, "date", "observed_date", "week_present"] + value_cols
    out = out[ordered]

    return out


df_weekly_filled = fill_weekly_12(df_weekly_survey, survey_task_weekly)

# add week numbers as well:
df_weekly_filled["week"] = (
    df_weekly_filled.groupby("ParticipantIdentifier").cumcount() + 1
)

# print(df_weekly_filled)

# print the number of adherence=1 for each participant
all_counts = (
    df_weekly_filled.groupby("ParticipantIdentifier")["week_present"]
    .sum()
    .astype(int)
)
print(all_counts)
# print the number of survey results for each participant in the original dataframe
print(df_weekly_survey.groupby("ParticipantIdentifier").size())

# # check what happens to participant 33
# print(df_weekly_survey[df_weekly_survey.ParticipantIdentifier == "117"])

# # check what happens to participant 33 in the filled dataframe
# print(df_weekly_filled[df_weekly_filled.ParticipantIdentifier == "117"])

# participant 33 completed a survey on 2025-10-09 which is not within the 12 weeks

# save the filled dataframe
df_weekly_filled.to_csv(folder / "df_weekly_filled.csv", index=False)

# %% [markdown]
# ## Extract variables from end-of-day survey: affective reflection, anticipated affect
# 
# - keep participant identifier
# - keep delivery date and time
# - when the surveys are not responded, we use NA to indicate missingness
# - each participant should have about 84 rows of 2 variables
# - rows after 84 are removed..
# - **Important**: the start date should be decided according to survey task not survey question results, as the latter only has data when the user answers the survey!!!
# 
# - add missingness indicator and column for participant identifier
# - according to the date, the gap should be about 84 days (+/- 1 day)
# - use NANs to fill the missing values
# - missingness indicator: 1 for present, 0 for missing
# 
# - Time zone issue: it's bedtime minus 2 or 4, so sometimes it can flow to next day.........
# - Let's do a minus 4 for all users 
# - ignore winter time or other time zones in addition to eastern time.

# %%
questions = ["AffectiveReflection", "AnticipatedAffect", "ActivityCheck"]    

tmp = (
    survey_results_flat
    .loc[survey_results_flat["ResultIdentifier"].isin(questions)]
    .copy()
)

# Same pattern as weekly block
tmp["datetime"] = pd.to_datetime(tmp["QuestionStartDate"], errors="coerce", utc=True)
tmp["value"] = pd.to_numeric(tmp["AnswerFirst"], errors="coerce")
# ActivityCheck: "wasActive" -> 1, any other response -> 0
activity_mask = tmp["ResultIdentifier"] == "ActivityCheck"
tmp.loc[activity_mask, "value"] = (
    tmp.loc[activity_mask, "AnswerFirst"]
    .astype(str)
    .str.strip()
    .eq("wasActive")
    .astype(float)
)
tmp["date"] = pd.to_datetime(
    tmp["QuestionStartDate"].astype(str).str.slice(0, 10),
    errors="coerce"
).dt.date

tmp = tmp.dropna(subset=["datetime", "date"])

df_daily_survey = (
    tmp.pivot_table(
        index=["ParticipantIdentifier", "date"],
        columns="ResultIdentifier",
        values="value",
        aggfunc="last"
    )
    .reset_index()
)

rename_map = {
    "AffectiveReflection": "affective_reflection",
    "AnticipatedAffect": "anticipated_affect",
    "ActivityCheck": "active_status",
}
df_daily_survey = df_daily_survey.rename(columns=rename_map)

desired_cols = ["ParticipantIdentifier", "date", "affective_reflection", "anticipated_affect", "active_status"]
for c in desired_cols:
    if c not in df_daily_survey.columns:
        df_daily_survey[c] = np.nan

df_daily_survey = df_daily_survey[desired_cols].sort_values(["ParticipantIdentifier", "date"])
print(df_daily_survey.head())

# %%
# extract daily survey from survey task
survey_task_eod = surveytask.loc[surveytask.SurveyName == 'MRT - Daily End of day survey and Planning Exercise'].copy()
survey_task_eod['date'] = pd.to_datetime(survey_task_eod['InsertedDate'])

print(survey_task_eod)

# %%
def fill_daily_85(df1, id_col="ParticipantIdentifier", date_col="date", days=85):
    """
    df1: wide daily-level survey question results frame with [id_col, date_col, value columns...]
    df2: wide daily-level survey task frame with [id_col, date_col, survey task...]
    Returns a frame with exactly `days` rows per participant, anchored at the
    participant's earliest observed date, spaced at 1-day intervals (± tolerance).
    """
    # ensure datetime (normalized to date)
    df1 = df1.copy()
    df1[date_col] = pd.to_datetime(df1[date_col]).dt.normalize()

    # df2 = df2.copy()
    # df2[date_col] = pd.to_datetime(df2[date_col]).dt.normalize()

    # anchor_lookup = (
    #     df2.groupby(id_col)[date_col]
    #        .min()  # earliest task date per participant
    # )

    # value columns to carry through
    value_cols = [c for c in df1.columns if c not in [id_col, date_col]]

    out_parts = []
    for pid, g in df1.groupby(id_col, sort=False):

        # anchor = anchor_lookup.get(pid)  # earliest observed date
        anchor = summary_surveytask.loc[summary_surveytask['ParticipantIdentifier'] == pid, 'date_min'].iloc[0]
        anchor = pd.to_datetime(anchor).normalize()
        
        # expected daily slots
        slots = pd.DataFrame({
            "idx": np.arange(days, dtype=int),
            id_col: pid,
            "date": [anchor + pd.Timedelta(days=1*i) for i in range(days)]
        })

        # assign each observed date to the nearest daily slot
        g = g.assign(
            idx=np.round((g[date_col] - anchor).dt.days / 1.0).astype(int)
        )
        # distance from its slot in absolute days
        g = g.assign(
            _slot_date=lambda d: anchor + pd.to_timedelta(d["idx"]*1, unit="D"),
            _abs_diff=lambda d: (d[date_col] - d["_slot_date"]).abs()
        )
        # keep only rows that fall within tolerance and within slot range
        g = g[(g["idx"] >= 0) & (g["idx"] < days)]

        # if multiple observed dates map to the same slot, keep the nearest
        g = (g.sort_values(["_abs_diff", date_col])
                .drop_duplicates(subset=["idx"], keep="first"))

        # prepare for merge
        keep_cols = [id_col, "idx", date_col] + value_cols
        g = g[keep_cols].rename(columns={date_col: "observed_date"})

        # merge slots with matched observations
        merged = slots.merge(g, on=[id_col, "idx"], how="left")

        # present indicator: 1 if we matched a row to this slot, else 0
        merged["daily_present"] = merged["observed_date"].notna().astype(int)

        # ensure all value cols exist (NaN if missing)
        for c in value_cols:
            if c not in merged.columns:
                merged[c] = np.nan

        # tidy columns
        merged = merged.drop(columns=["idx"]).sort_values(["date"])
        out_parts.append(merged)

    out = pd.concat(out_parts, ignore_index=True)

    # order columns: id, date (expected slot), observed_date, present, then values
    ordered = [id_col, "date", "observed_date", "daily_present"] + value_cols
    out = out[ordered]

    return out


df_daily_filled = fill_daily_85(df_daily_survey)
# , survey_task_eod)


def _wearable_exclude_reason(n_prior, n_hourly, n_today):
    """Return a short reason string when a participant should be flagged for exclusion."""
    if n_prior == 0 and n_hourly == 0 and n_today == 0:
        return "no_wearable_steps_any_window"
    if n_hourly == 0:
        return "no_hourly_fourSC"
    return ""


def audit_wearable_step_quality(participant_ids, df_prior_2hours, df_hourly, df_today):
    """
    Summarize non-NaN wearable step outcomes per participant.

    Returns (audit_df, exclude_candidates) where exclude_candidates are active
    participants lacking usable FourSC (hourly) step counts, including cases
    with sparse prior2hour/today data but zero post-decision 4-hour windows.
    """
    excluded = {str(pid) for pid in EXCLUDED_PARTICIPANT_IDS}
    rows = []
    for pid in participant_ids:
        pid_key = str(pid)
        p_prior = df_prior_2hours[
            df_prior_2hours["ParticipantIdentifier"].astype(str) == pid_key
        ]
        p_hourly = df_hourly[
            df_hourly["ParticipantIdentifier"].astype(str) == pid_key
        ]
        p_today = df_today[
            df_today["ParticipantIdentifier"].astype(str) == pid_key
        ]
        n_prior = int(p_prior["StepCount"].notna().sum())
        n_hourly = int(p_hourly["StepCount"].notna().sum())
        n_today = int(p_today["TodayStepCount"].notna().sum())
        reason = _wearable_exclude_reason(n_prior, n_hourly, n_today)
        rows.append(
            {
                "ParticipantIdentifier": pid_key,
                "prior2hour_non_nan": n_prior,
                "hourly_non_nan": n_hourly,
                "today_non_nan": n_today,
                "manually_excluded": pid_key in excluded,
                "recommend_exclude": bool(reason),
                "exclude_reason": reason,
            }
        )
    audit_df = pd.DataFrame(rows)
    exclude_candidates = audit_df[
        audit_df["recommend_exclude"] & ~audit_df["manually_excluded"]
    ].copy()
    return audit_df, exclude_candidates


def _mean_prior_rows(series, window=7, min_periods=1):
    """Mean over prior `window` calendar rows; excludes the current row."""
    return series.shift(1).rolling(window=window, min_periods=min_periods).mean()


def _ewm_prior_rows(series, gamma=6/7, window=7, min_periods=1):
    """
    Exponentially weighted average over at most the prior `window` rows only.
    Excludes the current row.

    Computes:
        sum_{j=1}^k gamma^{j-1} x_{t-j}
        ---------------------------------
        sum_{j=1}^k gamma^{j-1}

    where j=1 is the most recent prior row.
    """
    alpha = 1 - gamma

    def _ewm_last(window_values):
        w = pd.Series(window_values, dtype=float).dropna()
        if w.empty:
            return np.nan

        return w.ewm(alpha=alpha, adjust=True).mean().iloc[-1]

    return (
        series.shift(1)
        .rolling(window=window, min_periods=min_periods)
        .apply(_ewm_last, raw=True)
    )


# Fraction wasActive among prior ≤7 days with ActivityCheck answered (excludes today)
def _active_frac_last7_answered(g):
    g = g.sort_values("date")
    hist = []
    out = np.full(len(g), np.nan)
    for i, status in enumerate(g["active_status"]):
        if hist:
            out[i] = np.mean(hist[-7:])
        if pd.notna(status):
            hist.append(float(status))
    return pd.Series(out, index=g.index)


df_daily_filled = df_daily_filled.sort_values(
    ["ParticipantIdentifier", "date"], kind="mergesort"
)
df_daily_filled["active_status_fraction_7days"] = (
    df_daily_filled.groupby("ParticipantIdentifier", sort=False)
    .apply(_active_frac_last7_answered)
    .reset_index(level=0, drop=True)
)


# add day numbers as well:
df_daily_filled["day"] = (
    df_daily_filled.groupby("ParticipantIdentifier").cumcount() + 1
)



# print the number of missingness=1 for each participant
print(df_daily_filled[df_daily_filled.daily_present == 1].groupby("ParticipantIdentifier").size())

# print the number of survey results for each participant in the original dataframe
print(df_daily_survey.groupby("ParticipantIdentifier").size())

# check what happens to participant 33
# print(df_daily_survey[df_daily_survey.ParticipantIdentifier == 75])

# check what happens to participant 33 in the filled dataframe
# print(df_daily_filled[df_daily_filled.participantidentifier == 75])

df_daily_filled.to_csv(folder / "df_daily_filled.csv", index=False)


# %% [markdown]
# ## 4. Wakeup and bedtime schedule
# - this is useful for filling in the walking suggestions delivery time!

# %%
# get baseline surveykey
# survey_key_baseline = surveytask[surveytask.surveyname == 'MRT - Personalize HeartSteps'].surveykey.values[0]

# get wakeup and bedtime for each user
# key fields: resultidentifier: Wakeday Wakeup, Wakeday Bedtime, Weekend Wakeup, Weekend Bedtime
weekday_wakeup_all = []
weekday_bedtime_all = []
weekend_wakeup_all = []
weekend_bedtime_all = []

question_enddate_all = []

for participant_id in complete_participant_ids:
    # print(participant_id)
    subset = survey_results_flat[survey_results_flat.ParticipantIdentifier == participant_id]
    wake_weekday = (
        subset[subset.ResultIdentifier == 'Weekday Wakeup']['AnswersRaw']
        .str.strip('[]')
        .pipe(pd.to_datetime, format='%I:%M %p', errors='coerce')
    )
    weekday_wakeup_all.append(wake_weekday.dt.time.to_list())

    question_enddate = pd.to_datetime(
        subset.loc[subset["ResultIdentifier"] == "Weekday Wakeup", "QuestionStartDate"],
        errors="coerce",
        utc=True
    )
    question_enddate_all.append(question_enddate.dt.date.to_list())


    bed_weekday = (
        subset[subset.ResultIdentifier == 'Weekday Bedtime']['AnswersRaw']
        .str.strip('[]')
        .pipe(pd.to_datetime, format='%I:%M %p', errors='coerce')
    )
    weekday_bedtime_all.append(bed_weekday.dt.time.to_list())

    wake_weekend = (
        subset[subset.ResultIdentifier == 'Weekend Wakeup']['AnswersRaw']
        .str.strip('[]')
        .pipe(pd.to_datetime, format='%I:%M %p', errors='coerce')
    )
    weekend_wakeup_all.append(wake_weekend.dt.time.to_list())

    bed_weekend = (
        subset[subset.ResultIdentifier == 'Weekend Bedtime']['AnswersRaw']
        .str.strip('[]')
        .pipe(pd.to_datetime, format='%I:%M %p', errors='coerce')
    )
    weekend_bedtime_all.append(bed_weekend.dt.time.to_list())       

rows = []
for pid, time1s, time2s, time3s, time4s, dates in zip(
    complete_participant_ids,
    weekday_wakeup_all,
    weekday_bedtime_all,
    weekend_wakeup_all,
    weekend_bedtime_all,
    question_enddate_all
):
    for time1, time2, time3, time4, date in zip_longest(
        time1s, time2s, time3s, time4s, dates
    ):
        rows.append({
            "ParticipantIdentifier": pid,
            "WeekdayWakeup": time1,
            "WeekdayBedtime": time2,
            "WeekendWakeup": time3,
            "WeekendBedtime": time4,
            "QuestionStartDate": date,
        })

df_wakeup_bedtime = pd.DataFrame(rows)

# recover wakeup and bedtime for users with missing survey data
def _time_to_minutes(t):
    if pd.isna(t):
        return np.nan
    return t.hour * 60 + t.minute + t.second / 60


def _minutes_to_time(total_minutes):
    total_minutes = int(round(total_minutes)) % (24 * 60)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return pd.to_datetime(f"{hours:02d}:{minutes:02d}:00").time()


def _round_minutes(total_minutes, step=30):
    total_minutes = int(round(total_minutes)) % (24 * 60)
    return int(step * round(total_minutes / step)) % (24 * 60)


def _canonical_time_from_daily_estimates(daily_estimates):
    if len(daily_estimates) == 0:
        return pd.NaT
    rounded = [_round_minutes(v, step=30) for v in daily_estimates]
    mode_vals = pd.Series(rounded).mode()
    if len(mode_vals):
        return _minutes_to_time(mode_vals.iloc[0])
    return _minutes_to_time(_round_minutes(np.median(rounded), step=30))


def _median_time(series):
    vals = [v for v in (_time_to_minutes(x) for x in series) if not pd.isna(v)]
    if len(vals) == 0:
        return pd.NaT
    return _minutes_to_time(np.median(vals))


def _infer_wakeup_from_gif(gif_participant):
    if gif_participant.empty:
        return pd.NaT, pd.NaT

    gif_participant = gif_participant.copy()
    gif_participant["Timestamp"] = pd.to_datetime(gif_participant["Timestamp"])
    gif_participant["date"] = gif_participant["Timestamp"].dt.date
    gif_participant["minute_of_day"] = (
        gif_participant["Timestamp"].dt.hour * 60 + gif_participant["Timestamp"].dt.minute
    )

    weekday_candidates = []
    weekend_candidates = []
    for date, day_rows in gif_participant.groupby("date"):
        mins = np.sort(day_rows["minute_of_day"].to_numpy())
        if len(mins) == 0:
            continue
        # Protocol rule: first daily gif is wakeup + 1h; second is wakeup + 6h.
        # We prioritize first delivery when present.
        first_delivery = mins[0]
        wake_est = (first_delivery - 60) % (24 * 60)

        if pd.to_datetime(date).weekday() < 5:
            weekday_candidates.append(wake_est)
        else:
            weekend_candidates.append(wake_est)

    weekday_wakeup = _canonical_time_from_daily_estimates(weekday_candidates)
    weekend_wakeup = _canonical_time_from_daily_estimates(weekend_candidates)
    return weekday_wakeup, weekend_wakeup


def _infer_bedtime_from_end(end_participant):
    if end_participant.empty:
        return pd.NaT, pd.NaT

    end_participant = end_participant.copy()
    end_participant["Timestamp"] = pd.to_datetime(end_participant["Timestamp"])
    end_participant["date"] = end_participant["Timestamp"].dt.date
    bedtime_minute = (
        end_participant["Timestamp"].dt.hour * 60
        + end_participant["Timestamp"].dt.minute
        + 120  # end-of-day delivery is bedtime - 2h
    ) % (24 * 60)
    end_participant["bedtime_minute"] = bedtime_minute

    weekday_vals = end_participant.loc[
        end_participant["date"].map(lambda d: pd.to_datetime(d).weekday() < 5),
        "bedtime_minute",
    ].to_numpy()
    weekend_vals = end_participant.loc[
        end_participant["date"].map(lambda d: pd.to_datetime(d).weekday() >= 5),
        "bedtime_minute",
    ].to_numpy()

    weekday_bedtime = _canonical_time_from_daily_estimates(weekday_vals.tolist())
    weekend_bedtime = _canonical_time_from_daily_estimates(weekend_vals.tolist())
    return weekday_bedtime, weekend_bedtime


required_cols = ["WeekdayWakeup", "WeekendWakeup", "WeekdayBedtime", "WeekendBedtime"]
participant_has_data = (
    df_wakeup_bedtime.assign(ParticipantIdentifier=df_wakeup_bedtime["ParticipantIdentifier"].astype(str))
    .groupby("ParticipantIdentifier")[required_cols]
    .apply(lambda x: x.notna().any().any())
)
participant_without_wakeup_bedtime = [
    str(pid) for pid in complete_participant_ids if not bool(participant_has_data.get(str(pid), False))
]
print("Participants without wakeup/bedtime from survey extraction:")
print(participant_without_wakeup_bedtime)

# infer missing wakeup/bedtime from local-time delivery schedules
push_sent_for_impute = pd.read_csv(folder / "AnalyticsEvents_PushNotificationSent.csv")
push_sent_for_impute = convert_utc_columns_to_user_local(
    push_sent_for_impute,
    datetime_cols=["Timestamp"],
    participant_col="ParticipantIdentifier",
    join_date_col="Timestamp",
)
gif_rows_for_impute = push_sent_for_impute.loc[
    push_sent_for_impute["Properties.NotificationIdentifier"].str.startswith("gif", na=False)
].copy()
end_rows_for_impute = push_sent_for_impute.loc[
    push_sent_for_impute["Properties.NotificationIdentifier"].str.startswith("endOfDay", na=False)
].copy()

cohort_fallback = {
    col: _median_time(df_wakeup_bedtime[col]) for col in required_cols
}

recovered_from_delivery = []
fallback_used = []
still_missing = []
for pid in participant_without_wakeup_bedtime:
    gif_participant = gif_rows_for_impute.loc[
        gif_rows_for_impute["ParticipantIdentifier"].astype(str) == pid
    ].copy()
    end_participant = end_rows_for_impute.loc[
        end_rows_for_impute["ParticipantIdentifier"].astype(str) == pid
    ].copy()

    weekday_wakeup, weekend_wakeup = _infer_wakeup_from_gif(gif_participant)
    weekday_bedtime, weekend_bedtime = _infer_bedtime_from_end(end_participant)

    # if one of weekday/weekend is unavailable, borrow from the other side first
    if pd.isna(weekday_wakeup) and not pd.isna(weekend_wakeup):
        weekday_wakeup = weekend_wakeup
    if pd.isna(weekend_wakeup) and not pd.isna(weekday_wakeup):
        weekend_wakeup = weekday_wakeup
    if pd.isna(weekday_bedtime) and not pd.isna(weekend_bedtime):
        weekday_bedtime = weekend_bedtime
    if pd.isna(weekend_bedtime) and not pd.isna(weekday_bedtime):
        weekend_bedtime = weekday_bedtime

    inferred_any = any(
        not pd.isna(v)
        for v in [weekday_wakeup, weekend_wakeup, weekday_bedtime, weekend_bedtime]
    )
    row = {
        "ParticipantIdentifier": pid,
        "QuestionStartDate": pd.NaT,
        "WeekdayWakeup": weekday_wakeup if not pd.isna(weekday_wakeup) else cohort_fallback["WeekdayWakeup"],
        "WeekendWakeup": weekend_wakeup if not pd.isna(weekend_wakeup) else cohort_fallback["WeekendWakeup"],
        "WeekdayBedtime": weekday_bedtime if not pd.isna(weekday_bedtime) else cohort_fallback["WeekdayBedtime"],
        "WeekendBedtime": weekend_bedtime if not pd.isna(weekend_bedtime) else cohort_fallback["WeekendBedtime"],
    }

    if inferred_any:
        recovered_from_delivery.append(pid)
    elif all(not pd.isna(row[col]) for col in required_cols):
        fallback_used.append(pid)
    else:
        still_missing.append(pid)

    existing_mask = df_wakeup_bedtime["ParticipantIdentifier"].astype(str) == pid
    if existing_mask.any():
        idx = df_wakeup_bedtime.index[existing_mask][0]
        for col in ["QuestionStartDate"] + required_cols:
            df_wakeup_bedtime.loc[idx, col] = row[col]
    else:
        df_wakeup_bedtime = pd.concat(
            [df_wakeup_bedtime, pd.DataFrame([row])],
            ignore_index=True,
        )

# Manual bedtime override for participants with clearly implausible inferred bedtime from push timing.
manual_bedtime_override_ids = {"118", "138", "143", "151"}
manual_bedtime = pd.to_datetime("23:00:00").time()
override_mask = df_wakeup_bedtime["ParticipantIdentifier"].astype(str).isin(manual_bedtime_override_ids)
df_wakeup_bedtime.loc[override_mask, "WeekdayBedtime"] = manual_bedtime
df_wakeup_bedtime.loc[override_mask, "WeekendBedtime"] = manual_bedtime

print("Recovered from gif/endOfDay delivery timing:", recovered_from_delivery)
print("Used cohort fallback wake/bed medians:", fallback_used)
print("Still missing after recovery:", still_missing)
print(df_wakeup_bedtime)


def resolve_schedule_for_date(df_wb_participant, date):
    """Weekday/weekend wake and bed times effective on `date` → (wakeup_time, bedtime_time)."""
    # Align with QuestionStartDate (datetime.date) so multi-row schedules compare safely.
    date = pd.Timestamp(date).date()
    wb = df_wb_participant
    if wb.shape[0] == 1:
        weekday_wakeup = wb["WeekdayWakeup"].iloc[0]
        weekend_wakeup = wb["WeekendWakeup"].iloc[0]
        weekday_bedtime = wb["WeekdayBedtime"].iloc[0]
        weekend_bedtime = wb["WeekendBedtime"].iloc[0]
    else:
        weekday_wakeup = wb["WeekdayWakeup"].iloc[0]
        weekend_wakeup = wb["WeekendWakeup"].iloc[0]
        weekday_bedtime = wb["WeekdayBedtime"].iloc[0]
        weekend_bedtime = wb["WeekendBedtime"].iloc[0]
        for j in range(len(wb)):
            change_date = wb["QuestionStartDate"].iloc[j]
            if j < len(wb) - 1:
                next_change_date = wb["QuestionStartDate"].iloc[j + 1]
                if change_date <= date < next_change_date:
                    weekday_wakeup = wb["WeekdayWakeup"].iloc[j]
                    weekend_wakeup = wb["WeekendWakeup"].iloc[j]
                    weekday_bedtime = wb["WeekdayBedtime"].iloc[j]
                    weekend_bedtime = wb["WeekendBedtime"].iloc[j]
                    break
            elif change_date <= date:
                weekday_wakeup = wb["WeekdayWakeup"].iloc[j]
                weekend_wakeup = wb["WeekendWakeup"].iloc[j]
                weekday_bedtime = wb["WeekdayBedtime"].iloc[j]
                weekend_bedtime = wb["WeekendBedtime"].iloc[j]
                break
    is_weekday = pd.Timestamp(date).weekday() < 5
    wakeup_time = weekday_wakeup if is_weekday else weekend_wakeup
    bedtime_time = weekday_bedtime if is_weekday else weekend_bedtime
    return wakeup_time, bedtime_time


def resolve_wakeup_for_date(df_wb_participant, date):
    """Wakeup time only (walking-suggestion default timing)."""
    wakeup_time, _ = resolve_schedule_for_date(df_wb_participant, date)
    return wakeup_time


# %% [markdown]
# ## 5. Daily engagement (page views)
# 
# - for engagement, we may have yesterday's engagement data for day 1
# - this is different from survey variables...

# %%
# Select date range based on daily survey start and end date
pageview = pd.read_csv(folder / 'AnalyticsEvents_ViewViewed.csv')
pageview = convert_utc_columns_to_user_local(
    pageview,
    datetime_cols=["Timestamp"],
    participant_col="ParticipantIdentifier",
    join_date_col="Timestamp",
)
pageview_selected = []
for participant_id in complete_participant_ids:
    pageview_participant = pageview[pageview['ParticipantIdentifier'] == participant_id].copy()
    pageview_participant['Date'] = pageview_participant['Timestamp'].dt.date
    summary_row = summary_surveytask.loc[
        summary_surveytask['ParticipantIdentifier'] == participant_id
    ].iloc[0]  # assumes one row per participant

   
    start_date = pd.to_datetime(summary_row['date_min']).date() - pd.Timedelta(days=1)
    end_date = start_date + pd.Timedelta(days=85)
    pageview_participant = pageview_participant[pageview_participant.Date >= start_date]
    pageview_participant = pageview_participant[pageview_participant.Date <= end_date]
    pageview_selected.append(pageview_participant)

pageview_selected = pd.concat(pageview_selected)
pageview_selected = pageview_selected[["ParticipantIdentifier", "Timestamp"]]
# print(pageview_selected[pageview_selected['ParticipantIdentifier'] == 37])

# %%

daily_pageview_list = []
for participant_id in complete_participant_ids:
    pageview_participant = pageview_selected[(pageview_selected['ParticipantIdentifier'] == participant_id)].copy()
    df_wakeup_bedtime_participant = df_wakeup_bedtime.loc[df_wakeup_bedtime['ParticipantIdentifier'] == participant_id].copy()

    min_date = summary_surveytask.loc[
        summary_surveytask['ParticipantIdentifier'] == participant_id
    ].iloc[0].date_min - pd.Timedelta(days=7)
    date_range_length = 92

    for i in range(date_range_length):
        date = min_date + pd.Timedelta(days=i)
        # print(date)
        wakeup_time, bedtime_time = resolve_schedule_for_date(df_wakeup_bedtime_participant, date)


        # filter out the days with less than 8 hours of wearing fitbit (more than 8 hours of heart rate =0 or nan
        # within the wakeup and bedtime)
        pageview_participant_date = pageview_participant[pageview_participant.Timestamp.dt.date == date]
        pageview_participant_date = pageview_participant_date[pageview_participant_date.Timestamp.dt.time >= wakeup_time]
        pageview_participant_date = pageview_participant_date[pageview_participant_date.Timestamp.dt.time <= bedtime_time]
        

        # Calculate time span from first to last valid reading
        if pageview_participant_date.shape[0] > 0:
            valid_count = pageview_participant_date.shape[0] 
        else:
            valid_count = 0
        # Mark as missing if less than 8 hours of valid data

        daily_pageview_list.append({
            'ParticipantIdentifier': participant_id,
            'Date': pd.to_datetime(date),
            'DailyPageviewCount': valid_count
        })


df_daily_pageview = pd.DataFrame(daily_pageview_list)

# get yesterday's pageview count
df_daily_pageview['YesterdayPageviewCount'] = (
    df_daily_pageview
    .sort_values(['ParticipantIdentifier', 'Date'])
    .groupby('ParticipantIdentifier')['DailyPageviewCount']
    .shift(1)
    .fillna(0)            # or np.nan, or dropna() later
    .astype(int)
)

# EWM (gamma=6/7) over prior ≤7 calendar days of pageviews (excludes today)
df_daily_pageview['Past7DaysPageviewEMA'] = (
    df_daily_pageview
    .sort_values(['ParticipantIdentifier', 'Date'], kind='mergesort')
    .groupby('ParticipantIdentifier', sort=False)['DailyPageviewCount']
    .transform(lambda s: _ewm_prior_rows(s, gamma=6/7))
)

# get tomorrow's pageview count
# df_daily_pageview['TomorrowPageviewCount'] = (
#     df_daily_pageview['DailyPageviewCount'].shift(-1)
# )


df_daily_pageview.to_csv(os.path.join(folder, 'df_daily_pageview.csv'), index=False)
# print(df_daily_pageview[df_daily_pageview['participantidentifier'] == 31])



# remove the last row for each participant
df_daily_pageview = df_daily_pageview.groupby('ParticipantIdentifier').apply(lambda x: x.iloc[:-1])

print(df_daily_pageview)


# %%


hourly_pageview_list = []

for participant_id in complete_participant_ids:
    pageview_participant = pageview_selected[pageview_selected['ParticipantIdentifier'] == participant_id].copy()
    df_wakeup_bedtime_participant = df_wakeup_bedtime.loc[df_wakeup_bedtime['ParticipantIdentifier'] == participant_id].copy()

    # we minus 1 day because we want to include yesterday's data of day 1
    # TODO: figured out we may not need to minus 1 day because we start modeling step counts after the first day of end of day survey
    min_date = summary_surveytask.loc[
        summary_surveytask['ParticipantIdentifier'] == participant_id
    ].iloc[0].date_min 
    date_range_length = 85

    for i in range(date_range_length):
        date = min_date + pd.Timedelta(days=i)
        
        wakeup_time, bedtime_time = resolve_schedule_for_date(df_wakeup_bedtime_participant, date)

        # Get all heart rate data for this date
        pageview_participant_date = pageview_participant[pageview_participant.Timestamp.dt.date == date]
        pageview_participant_date = pageview_participant_date[pageview_participant_date.Timestamp.dt.time >= wakeup_time]
        pageview_participant_date = pageview_participant_date[pageview_participant_date.Timestamp.dt.time <= bedtime_time]
        
        # Create hourly bins from wakeup to bedtime
        wakeup_datetime = pd.Timestamp.combine(date, wakeup_time)
        bedtime_datetime = pd.Timestamp.combine(date, bedtime_time)
        
        # Generate hourly time bins
        num_decisions = 2 # 2 decision points per day wakeup + 1, wakeup + 6
        current_decision = 0
        while current_decision < num_decisions:
            start_window = wakeup_datetime + pd.Timedelta(hours=1) if current_decision == 0 else wakeup_datetime + pd.Timedelta(hours=6)
            end_window = start_window + pd.Timedelta(hours=4)
            
            # Filter data for this hour
            view_data = pageview_participant_date[
                (pageview_participant_date['Timestamp'] >= start_window) &
                (pageview_participant_date['Timestamp'] < end_window)
            ]
            

            
            # Calculate time span of valid data in this hour
            if view_data.shape[0] > 0:
                valid_count = view_data.shape[0]
            else:
                valid_count = 0

            
            hourly_pageview_list.append({
                'ParticipantIdentifier': participant_id,
                'Date': date,
                'DecisionTime': current_decision,
                'DatetimeStart': start_window,
                'HourlyPageviewCount': valid_count
            })
            
            current_decision += 1

df_hourly_pageview = pd.DataFrame(hourly_pageview_list)


# EWM (gamma=6/7) over prior ≤7 same-slot pageviews (excludes current slot)
df_hourly_pageview['Past7DaysHourlyPageviewEMA'] = (
    df_hourly_pageview
    .sort_values(['ParticipantIdentifier', 'Date', 'DecisionTime'], kind='mergesort')
    .groupby(['ParticipantIdentifier', 'DecisionTime'], sort=False)['HourlyPageviewCount']
    .transform(lambda s: _ewm_prior_rows(s, gamma=6/7))
)

# save the dataframe
df_hourly_pageview.to_csv(folder / 'hourly_pageview.csv', index=False)


# %% [markdown]
# ## 6. Interventions (push notifications & survey display)
# - twice daily walking suggestions map to hourly, twice-daily, daily, and weekly
# - daily planning prompts map to hourly, twice-daily, daily, and weekly
# - daily salience message map to hourly, twice-daily, daily, and weekly
# 
# 
# - Time zone: timestamp minus 4 hours for now!!!
# 
# ## Notes on click data
# - for walking suggestions/salience messages, an important variable is whether the user opens the notification
# - for planning prompts, an important variable is whether the user fills in the survey!
# 
# - maybe the users saw the notifictaion but didn't open it... so perhaps we can also ignore this for now!!!!!!!
# 
# ### Missing data problem:
# - for user 13, end-of-day survey + planning action delivery information is missing on day 7/26 and 8/3. This is weired. 
# - TODO: check project device data, which is more accurate than notification sent csv...

# %%
# load analytics_event_push data
push_sent = pd.read_csv(folder / 'AnalyticsEvents_PushNotificationSent.csv')

# to local time
push_sent = convert_utc_columns_to_user_local(
    push_sent,
    datetime_cols=["Timestamp"],
    participant_col="ParticipantIdentifier",
    join_date_col="Timestamp",
)


push_open = pd.read_csv(folder / 'AnalyticsEvents_PushNotificationOpened.csv')

# to local time
push_open = convert_utc_columns_to_user_local(
    push_open,
    datetime_cols=["Timestamp"],
    participant_col="ParticipantIdentifier",
    join_date_col="Timestamp",
)


# %%
gif_rows      = push_sent.loc[push_sent['Properties.NotificationIdentifier'].str.startswith('gif', na=False)].copy()
end_rows      = push_sent.loc[push_sent['Properties.NotificationIdentifier'].str.startswith('endOfDay', na=False)].copy()
salience_rows = push_sent.loc[push_sent['Properties.NotificationIdentifier'].str.startswith('salience', na=False)].copy()

# delete repeated rows
gif_rows = gif_rows.drop_duplicates()
end_rows = end_rows.drop_duplicates()
salience_rows = salience_rows.drop_duplicates()

# print(end_rows[end_rows['ParticipantIdentifier'] == "117"])
# print(end_rows.head())
# print(salience_rows.head())


gif_rows_open      = push_open.loc[push_open['Properties.NotificationIdentifier'].str.startswith('gif', na=False)].copy()
# end_rows_open      = push_open.loc[push_open['notification_id'].str.startswith('endOfDay', na=False)].copy()
salience_rows_open = push_open.loc[push_open['Properties.NotificationIdentifier'].str.startswith('salience', na=False)].copy()

# print(gif_rows_open[gif_rows_open['participantidentifier'] == 75])
# print(salience_rows_open.head())


# %%
# check the date range of the end of day survey for each participant
min_date_list = []
max_date_list = []
date_range_length_list = []
# complete_participant_ids = [complete_participant_ids, "251"]
for participant_id in complete_participant_ids:
    end_rows_participant = end_rows.loc[end_rows['ParticipantIdentifier'] == participant_id].copy()
    end_rows_participant['Timestamp'] = pd.to_datetime(end_rows_participant['Timestamp'])

    min_date = end_rows_participant['Timestamp'].min()
    max_date = end_rows_participant['Timestamp'].max()

    print(f"Participant {participant_id} end of day survey date range: {min_date} to {max_date}")
    min_date_list.append(min_date)
    max_date_list.append(max_date)
    # get the length of the date range
    date_range_length = (max_date - min_date).days + 1
    date_range_length_list.append(date_range_length)

print(min_date_list)
print(max_date_list)
print(date_range_length_list)

end_rows_251 = end_rows.loc[end_rows['ParticipantIdentifier'] == "251"].copy()
end_rows_251['Timestamp'] = pd.to_datetime(end_rows_251['Timestamp'])
min_date_251 = end_rows_251['Timestamp'].min()
max_date_251 = end_rows_251['Timestamp'].max()
print(f"Participant 251 end of day survey date range: {min_date_251} to {max_date_251}")
print(end_rows_251)



# %%
# print walkings suggestions range
# check the date range of the end of day survey for each participant
min_date_list_gif = []
max_date_list_gif = []
date_range_length_list_gif = []
for participant_id in complete_participant_ids:
    gif_rows_participant = gif_rows.loc[gif_rows['ParticipantIdentifier'] == participant_id].copy()
    gif_rows_participant['Timestamp'] = pd.to_datetime(gif_rows_participant['Timestamp'])

    min_date = gif_rows_participant['Timestamp'].min()
    max_date = gif_rows_participant['Timestamp'].max()

    # print(f"Participant {participant_id} gif date range: {min_date} to {max_date}")
    min_date_list_gif.append(min_date)
    max_date_list_gif.append(max_date)

    # get the length of the date range
    date_range_length_gif = (max_date - min_date).days + 1
    date_range_length_list_gif.append(date_range_length_gif)

# print(min_date_list_gif)
# print(max_date_list_gif)
# print(date_range_length_list_gif)

gif_rows_251 = gif_rows.loc[gif_rows['ParticipantIdentifier'] == "251"].copy()
gif_rows_251['Timestamp'] = pd.to_datetime(gif_rows_251['Timestamp'])
min_date_251 = gif_rows_251['Timestamp'].min()
max_date_251 = gif_rows_251['Timestamp'].max()
print(f"Participant 251 gif date range: {min_date_251} to {max_date_251}")
print(gif_rows_251)


# print(gif_rows_251.head())

# print(gif_rows_251.tail())



# %%
end_rows['planning_prompt'] = 0
end_rows.loc[end_rows['Properties.NotificationIdentifier'].str.startswith('endOfDay_Planning', na=False), 'planning_prompt'] = 1
# print(end_rows.head())

end_rows['Timestamp'] = pd.to_datetime(end_rows['Timestamp'])
end_rows['date'] = end_rows['Timestamp'].dt.date
end_rows['time'] = end_rows['Timestamp'].dt.time
df_end_all = end_rows[['ParticipantIdentifier', 'date', 'time', 'planning_prompt']]


# filter out completed participants
df_end_all = df_end_all[df_end_all['ParticipantIdentifier'].isin(complete_participant_ids)]

# drop duplicates(one row per participant per day)
df_end_all = df_end_all.groupby(['ParticipantIdentifier', 'date']).first().reset_index()


# fill missing (participant, date) in [date_min, date_max] with planning_prompt=0
# (must check per participant — date alone is wrong: another participant may already have that day)
fills = []
for participant_id in complete_participant_ids:
    min_date = summary_surveytask.loc[summary_surveytask['ParticipantIdentifier'] == participant_id, 'date_min'].iloc[0]
    max_date = summary_surveytask.loc[summary_surveytask['ParticipantIdentifier'] == participant_id, 'date_max'].iloc[0]
    existing = set(
        pd.to_datetime(
            df_end_all.loc[df_end_all['ParticipantIdentifier'] == participant_id, 'date'],
            errors='coerce',
        ).dt.date.dropna()
    )
    for ts in pd.date_range(pd.to_datetime(min_date), pd.to_datetime(max_date), freq='D'):
        d = ts.date()
        if d not in existing:
            fills.append(
                {'ParticipantIdentifier': participant_id, 'date': d, 'planning_prompt': 0}
            )
if fills:
    df_end_all = pd.concat([df_end_all, pd.DataFrame(fills)], ignore_index=True)
    df_end_all = df_end_all.sort_values(['ParticipantIdentifier', 'date']).reset_index(drop=True)

# print(df_end_all[df_end_all['participantidentifier'] == 31])
print(df_end_all)
# save the dataframe
df_end_all.to_csv(os.path.join(folder, 'df_end_all.csv'), index=False)



# %%
# the above rows do not contain no delivery data, so we need to complete the data
# first, complete walking suggestions

questions = ["MessageDisplay With Template"]

tmp = (
    survey_results_flat
    .loc[survey_results_flat["ResultIdentifier"].isin(questions)]
    .copy()
)


tmp["datetime"] = pd.to_datetime(tmp["QuestionStartDate"], errors="coerce", utc=True)
s = tmp["QuestionStartDate"].astype(str).str.replace(
    r"([+-]\d{2}:\d{2}|Z)$", "", regex=True
)
tmp["datetime_local"] = pd.to_datetime(s, errors="coerce")


# Local date from original offset timestamp string (YYYY-MM-DD part)
tmp["date"] = tmp["datetime_local"].dt.date

tmp['time'] = tmp["datetime_local"].dt.time
print(len(tmp))


# check the date range of the walking suggestions for each participant
df_gif_all = []  # Use list for efficient appending

for n in range(len(complete_participant_ids)):
    participant_id = complete_participant_ids[n]
    gif_rows_participant = gif_rows.loc[gif_rows['ParticipantIdentifier'] == participant_id].copy()
    print(len(gif_rows_participant))
    gif_rows_participant['Timestamp'] = pd.to_datetime(gif_rows_participant['Timestamp'])    
    # gif_rows_open_participant = gif_rows_open.loc[gif_rows_open['ParticipantIdentifier'] == participant_id].copy()
    # gif_rows_open_participant['Timestamp'] = pd.to_datetime(gif_rows_open_participant['Timestamp'])
    tmp_participant = tmp.loc[tmp['ParticipantIdentifier'] == participant_id].copy()

    # extract the date and time from the timestamp
    gif_rows_participant['date'] = gif_rows_participant['Timestamp'].dt.date
    gif_rows_participant['time'] = gif_rows_participant['Timestamp'].dt.time
    # gif_rows_open_participant['date'] = gif_rows_open_participant['Timestamp'].dt.date
    # gif_rows_open_participant['time'] = gif_rows_open_participant['Timestamp'].dt.time

    min_date = summary_surveytask.loc[summary_surveytask['ParticipantIdentifier'] == participant_id, 'date_min'].iloc[0]
    min_date = pd.to_datetime(min_date).date()
    
    # max_date = max_date_list[n].date()
    date_range_length = 85
    # date_range_length_list[n]

    # read wakeup and bedtime data for the participant
    df_wakeup_bedtime_participant = df_wakeup_bedtime.loc[df_wakeup_bedtime['ParticipantIdentifier'] == participant_id].copy()


    for i in range(date_range_length):
        date = min_date + pd.Timedelta(days=i)
        # print(date)
        wakeup_time = resolve_wakeup_for_date(df_wakeup_bedtime_participant, date)

        wake_td = pd.Timedelta(hours=wakeup_time.hour, minutes=wakeup_time.minute, seconds=wakeup_time.second)
        m_time = wake_td + pd.Timedelta(minutes=60)   # Morning: 1 hour after wake
        a_time = wake_td + pd.Timedelta(minutes=360)  # Afternoon: 6 hours after wake

        # search for date and time in gif_rows_participant
        gif_row = gif_rows_participant.loc[(gif_rows_participant['date'] == date)].copy()
        # gif_row_open = gif_rows_open_participant.loc[(gif_rows_open_participant['date'] == date)]
        tmp_par_date = tmp_participant.loc[(tmp_participant['date'] == date)].copy()
        # print(tmp_par_date.head())

        
        # valid_count = [
        #     df_hourly_pageview.loc[(df_hourly_pageview['participantidentifier'] == participant_id) &
        #                         (df_hourly_pageview['date'] == date) &
        #                         (df_hourly_pageview['decision_time'] == dt),
        #                         'valid_count'].iloc[0]
        #     for dt in (0, 1)
        # ]

        if gif_row.shape[0] == 2:
            decision_time = 0
            for time_val, timestamp_val in zip(gif_row['time'].values, gif_row['Timestamp'].values):
                # Check if opened within 1 hour
                # open_status = 0
                # for open_timestamp in gif_row_open['timestamp'].values:
                #     time_diff = abs((pd.to_datetime(open_timestamp) - pd.to_datetime(timestamp_val)).total_seconds() / 60)
                #     if time_diff <= 60:  # Within 60 minutes
                #         open_status = 1
                #         break

                # check if the user interacted with the notification
                
                if len(tmp_par_date['datetime_local'].values) >= 2:
                    interacted = 1
                else:
                    interacted = 0
                    for interaction in tmp_par_date['datetime_local'].values:
                        if interaction < timestamp_val:
                            continue
                        time_diff = pd.Timedelta(interaction - timestamp_val).total_seconds() / 60
                        # print(time_diff)
                        if decision_time == 0 and 0 <= time_diff <= 300:
                            interacted = 1
                            break
                        elif decision_time == 1 and 0 <= time_diff <= 600:
                            interacted = 1
                            break

                df_gif_all.append({
                    'ParticipantIdentifier': participant_id,
                    'Date': date,
                    'Time': time_val,
                    'DecisionTime': decision_time,
                    'WalkingSuggestion': 1,
                    'Interacted': interacted
                    # 'ViewStatus': min(open_status + int(valid_count[decision_time] > 0), 1)
                })
                decision_time += 1
        elif gif_row.shape[0] == 1:
            # One delivery - determine if it's morning or afternoon
            delivered_time = gif_row['time'].iloc[0]
            delivered_timestamp = gif_row['Timestamp'].iloc[0]

            # Check if opened within 1 hour
            # TODO: check if this is correct
            # open_status = 0
            # for open_timestamp in gif_row_open['timestamp'].values:
            #     time_diff = abs((pd.to_datetime(open_timestamp) - pd.to_datetime(delivered_timestamp)).total_seconds() / 60)
            #     if time_diff <= 60:  # Within 60 minutes
            #         open_status = 1
            #         break

            # check if the user interacted with the notification
            
            if len(tmp_par_date['datetime_local'].values) >= 1:
                interacted = 1
            else:
                interacted = 0
                for interaction in tmp_par_date['datetime_local'].values:
                    if interaction < timestamp_val:
                        continue
                    time_diff = pd.Timedelta(interaction - timestamp_val).total_seconds() / 60
                    if decision_time == 0 and 0 <= time_diff <= 300:
                        interacted = 1
                        break
                    elif decision_time == 1 and 0 <= time_diff <= 600:
                        interacted = 1
                        break

        
            # Convert time to timedelta for comparison
            delivered_td = pd.Timedelta(
                hours=delivered_time.hour, 
                minutes=delivered_time.minute, 
                seconds=delivered_time.second
            )
            
            # Check which slot it's closer to
            is_morning = abs((delivered_td - m_time).total_seconds()) < abs((delivered_td - a_time).total_seconds())
            
            if is_morning:
                # Delivered in morning, not in afternoon
                df_gif_all.append({
                    'ParticipantIdentifier': participant_id,
                    'Date': date,
                    'Time': delivered_time,
                    'DecisionTime': 0,
                    'WalkingSuggestion': 1,
                    'Interacted': interacted
                    # 'ViewStatus': min(open_status + int(valid_count[0] > 0), 1)
                })
                df_gif_all.append({
                    'ParticipantIdentifier': participant_id,
                    'Date': date,
                    'Time': (pd.Timestamp('2000-01-01') + a_time).time(),
                    'DecisionTime': 1,
                    'WalkingSuggestion': 0,
                    'Interacted': 0
                    # 'ViewStatus': int(valid_count[1] > 0)
                })
            else:
                # Delivered in afternoon, not in morning
                df_gif_all.append({
                    'ParticipantIdentifier': participant_id,
                    'Date': date,
                    'Time': (pd.Timestamp('2000-01-01') + m_time).time(),
                    'DecisionTime': 0,
                    'WalkingSuggestion': 0,
                    'Interacted': 0
                    # 'ViewStatus': int(valid_count[0] > 0)
                })
                df_gif_all.append({
                    'ParticipantIdentifier': participant_id,
                    'Date': date,
                    'Time': delivered_time,
                    'DecisionTime': 1,
                    'WalkingSuggestion': 1,
                    'Interacted': interacted
                    # 'ViewStatus': min(open_status + int(valid_count[1] > 0), 1)
                })
        else:
            # No deliveries - both slots empty
            df_gif_all.append({
                'ParticipantIdentifier': participant_id,
                'Date': date,
                'Time': (pd.Timestamp('2000-01-01') + m_time).time(),
                'DecisionTime': 0,
                'WalkingSuggestion': 0,
                'Interacted': 0
                # 'ViewStatus': int(valid_count[0] > 0)
            })
            df_gif_all.append({
                'ParticipantIdentifier': participant_id,
                'Date': date,
                'Time': (pd.Timestamp('2000-01-01') + a_time).time(),
                'DecisionTime': 1,
                'WalkingSuggestion': 0,
                'Interacted': 0
                # 'ViewStatus': int(valid_count[1] > 0)
            })

df_gif_all = pd.DataFrame(df_gif_all)

# Fraction interacted over prior 14 walking-suggestion slots (~7 days × 2); excludes current slot
df_gif_all = df_gif_all.sort_values(['ParticipantIdentifier', 'Date', 'DecisionTime'], kind='mergesort')
df_gif_all['Interacted_7d'] = (
    df_gif_all.groupby('ParticipantIdentifier', sort=False)['Interacted']
    .transform(lambda s: _mean_prior_rows(s, window=14, min_periods=7))
)

# recent_burden: X_d = daily walking-suggestion count (sum over AM/PM slots), then
#   _ewm_prior_rows on the daily series (gamma=6/7, window=7):
#   (X_{d-1} + r X_{d-2} + ... + r^6 X_{d-7}) / (1 + r + ... + r^6),  r = 6/7
_daily_panel = (
    df_gif_all.groupby(['ParticipantIdentifier', 'Date'], sort=False)['WalkingSuggestion']
    .sum()
    .reset_index(name='_daily_suggestions')
)
_daily_panel['recent_burden'] = (
    _daily_panel.groupby('ParticipantIdentifier', sort=False)['_daily_suggestions']
    .transform(lambda s: _ewm_prior_rows(s, gamma=6 / 7, window=7, min_periods=1))
)
df_gif_all = df_gif_all.merge(
    _daily_panel[['ParticipantIdentifier', 'Date', 'recent_burden']],
    on=['ParticipantIdentifier', 'Date'],
    how='left',
)

print(df_gif_all)
print(df_gif_all.Interacted.value_counts())
# print(df_gif_all[df_gif_all['participantidentifier'] == 75].head(50))
# save the dataframe
df_gif_all.to_csv(os.path.join(folder, 'df_gif_all.csv'), index=False)


# %%
# second, complete salience messages
questions = ["Salience Message Display"]

tmp = (
    survey_results_flat
    .loc[survey_results_flat["ResultIdentifier"].isin(questions)]
    .copy()
)


tmp["datetime"] = pd.to_datetime(tmp["QuestionStartDate"], errors="coerce", utc=True)
s = tmp["QuestionStartDate"].astype(str).str.replace(
    r"([+-]\d{2}:\d{2}|Z)$", "", regex=True
)
tmp["datetime_local"] = pd.to_datetime(s, errors="coerce")


# Local date from original offset timestamp string (YYYY-MM-DD part)
tmp["date"] = tmp["datetime_local"].dt.date

tmp['time'] = tmp["datetime_local"].dt.time
print(len(tmp))
df_salience_all = []
# check the date range of the salience messages
for n in range(len(complete_participant_ids)):
    participant_id = complete_participant_ids[n]
    salience_rows_participant = salience_rows.loc[salience_rows['ParticipantIdentifier'] == participant_id].copy()
    salience_rows_participant['Timestamp'] = pd.to_datetime(salience_rows_participant['Timestamp'])
    # salience_rows_open_participant = salience_rows_open.loc[salience_rows_open['ParticipantIdentifier'] == participant_id].copy()
    # salience_rows_open_participant['Timestamp'] = pd.to_datetime(salience_rows_open_participant['Timestamp'])
    # print(f"Participant {participant_id} salience messages date range: {salience_rows_participant['timestamp'].min()} to {salience_rows_participant['timestamp'].max()}")
    tmp_participant = tmp.loc[tmp['ParticipantIdentifier'] == participant_id].copy()
    # extract the date and time from the timestamp
    salience_rows_participant['date'] = salience_rows_participant['Timestamp'].dt.date
    salience_rows_participant['time'] = salience_rows_participant['Timestamp'].dt.time
    # salience_rows_open_participant['date'] = salience_rows_open_participant['timestamp'].dt.date
    # salience_rows_open_participant['time'] = salience_rows_open_participant['timestamp'].dt.time

    min_date = summary_surveytask.loc[summary_surveytask['ParticipantIdentifier'] == participant_id, 'date_min'].iloc[0]
    min_date = pd.to_datetime(min_date).date()

    date_range_length = 85
    # date_range_length_list[n]

    for i in range(date_range_length):
        date = min_date + pd.Timedelta(days=i)
        salience_row = salience_rows_participant.loc[(salience_rows_participant['date'] == date)]
        tmp_par_date = tmp_participant.loc[(tmp_participant['date'] == date)].copy()
        # salience_row_open = salience_rows_open_participant.loc[(salience_rows_open_participant['date'] == date)]

        # open_status = 0
        # for open_timestamp in salience_row_open['timestamp'].values:
        #     time_diff = abs((pd.to_datetime(open_timestamp) - pd.to_datetime(salience_row['timestamp'].values[0])).total_seconds() / 60)
        #     if time_diff <= 60:  # Within 60 minutes
        #         open_status = 1
        #         break

        if len(tmp_par_date['datetime_local'].values) >= 1:
            interacted = 1
        else:
            interacted = 0

        if salience_row.shape[0] == 1:
            df_salience_all.append({
                'ParticipantIdentifier': participant_id,
                'Date': date,
                'Time': salience_row['time'].iloc[0],
                'SalienceMessage': 1,
                'Interacted': interacted
                # 'open': open_status
            })
        else:
            df_salience_all.append({
                'ParticipantIdentifier': participant_id,
                'Date': date,
                'Time': (pd.Timestamp('2000-01-01') + pd.Timedelta(hours=11, minutes=45)).time(),
                'SalienceMessage': 0,
                'Interacted': interacted
                # 'open': 0
            })

df_salience_all = pd.DataFrame(df_salience_all)
print(df_salience_all)
print(df_salience_all.Interacted.value_counts())

# Fraction interacted over prior 7 calendar days; excludes today
df_salience_all = df_salience_all.sort_values(['ParticipantIdentifier', 'Date'], kind='mergesort')
df_salience_all['Interacted_7d'] = (
    df_salience_all.groupby('ParticipantIdentifier', sort=False)['Interacted']
    .transform(lambda s: _mean_prior_rows(s, window=7, min_periods=7))
)

# print(df_salience_all.head(85))
# print(df_salience_all.open.value_counts())

# save the dataframe
df_salience_all.to_csv(os.path.join(folder, 'df_salience_all.csv'), index=False)


# %% [markdown]
# ## 7. Wearables — heart rate, steps, wear flags
# - case 1- missing hours: user receive walking suggestions at 9am, but only started wearing fitbit until 10 am 
#   treatment: 
# - case 2- missing days: user didn't wear fitbit for most hours between the wakeup and bedtime

# %%
# using heartrate to define drop out...etc
heartratebymin = pd.read_csv(folder/ 'filtered_activities-heart.csv')
print(heartratebymin.head())

#change to local time
heartratebymin = convert_utc_columns_to_user_local(
    heartratebymin,
    datetime_cols=["DateTime"],
    participant_col="ParticipantIdentifier",
    join_date_col="DateTime",
)
#filter out completed users

# %%
# filter out completed users
heartratebymin = heartratebymin[heartratebymin['ParticipantIdentifier'].isin(complete_participant_ids)].copy()
print(heartratebymin.head())

# %%
# for the active phase, we require 12*7 = 84 days of step count data
heartratebymin['DateTime'] = pd.to_datetime(heartratebymin['DateTime'])
heartratebymin['Date'] = heartratebymin['DateTime'].dt.date

# sort by participantidentifier and date
heartratebymin = heartratebymin.sort_values(by=['ParticipantIdentifier', 'DateTime'])

summary = (
    heartratebymin
    .groupby('ParticipantIdentifier')['Date']
    .agg(date_min='min', date_max='max', n_days='nunique')
    .assign(span_days=lambda df: (pd.to_datetime(df.date_max) -
                                  pd.to_datetime(df.date_min)).dt.days + 1)
)



# make participantidentifier a column name
summary.reset_index(inplace=True)
summary.rename(columns={'ParticipantIdentifier': 'ParticipantIdentifier'}, inplace=True)

print(summary)

# %%
# Select date range based on daily survey start and end date
heartratebymin_selected = []
for participant_id in complete_participant_ids:
    heartratebymin_participant = heartratebymin[heartratebymin['ParticipantIdentifier'] == participant_id]
    summary_row = summary_surveytask.loc[
        summary_surveytask['ParticipantIdentifier'] == participant_id
    ].iloc[0]  # assumes one row per participant

   
    start_date = pd.to_datetime(summary_row['date_min']).date() - pd.Timedelta(days=7)
    end_date = start_date + pd.Timedelta(days=91)
    heartratebymin_participant = heartratebymin_participant[heartratebymin_participant.Date >= start_date]
    heartratebymin_participant = heartratebymin_participant[heartratebymin_participant.Date <= end_date]
    heartratebymin_selected.append(heartratebymin_participant)

heartratebymin_selected = pd.concat(heartratebymin_selected)
# print(heartratebymin_selected.head())

heartratebymin_selected = heartratebymin_selected[["ParticipantIdentifier", "DateTime", "Value", "Date"]]
print(heartratebymin_selected[heartratebymin_selected['ParticipantIdentifier'] == "118"])

# %%
# read step count data by minute
stepcountbymin = pd.read_csv(folder /'filtered_activities-steps.csv')
stepcountbymin.head()

# change to local time
stepcountbymin = convert_utc_columns_to_user_local(
    stepcountbymin,
    datetime_cols=["DateTime"],
    participant_col="ParticipantIdentifier",
    join_date_col="DateTime",
)

# %%
# for the active phase, we require 12*7 = 84 days of step count data
stepcountbymin['DateTime'] = pd.to_datetime(stepcountbymin['DateTime'])
stepcountbymin['Date'] = stepcountbymin['DateTime'].dt.date

# filter out the data for complete participants
stepcountbymin = stepcountbymin[stepcountbymin['ParticipantIdentifier'].isin(complete_participant_ids)]

summary = (
    stepcountbymin
    .groupby('ParticipantIdentifier')['Date']
    .agg(date_min='min', date_max='max', n_days='nunique')
    .assign(span_days=lambda df: (pd.to_datetime(df.date_max) -
                                  pd.to_datetime(df.date_min)).dt.days + 1)
)



# make participantidentifier a column name
summary.reset_index(inplace=True)
summary.rename(columns={'ParticipantIdentifier': 'ParticipantIdentifier'}, inplace=True)

print(summary)

# %%
# Select date range based on daily survey start and end date
stepcountbymin_selected = []
for participant_id in complete_participant_ids:
    print(participant_id)
    stepcountbymin_participant = stepcountbymin[stepcountbymin['ParticipantIdentifier'] == participant_id]
    summary_row = summary_surveytask.loc[
        summary_surveytask['ParticipantIdentifier'] == participant_id
    ].iloc[0]  # assumes one row per participant

    start_date = pd.to_datetime(summary_row['date_min']).date() - pd.Timedelta(days=7)
    end_date = start_date + pd.Timedelta(days=91)
    print(start_date, end_date)
    stepcountbymin_participant = stepcountbymin_participant[stepcountbymin_participant.Date >= start_date]
    stepcountbymin_participant = stepcountbymin_participant[stepcountbymin_participant.Date <= end_date]
    stepcountbymin_selected.append(stepcountbymin_participant)

stepcountbymin_selected = pd.concat(stepcountbymin_selected)
stepcountbymin_selected = stepcountbymin_selected[["ParticipantIdentifier", "DateTime", "Value", "Date"]]
print(stepcountbymin_selected[stepcountbymin_selected['ParticipantIdentifier'] == "118"])

# %%
# checck missing days by filtering out the days with less than 8 hours of wearing fitbit (more than 8 hours of heart rate =0 or nan
# within the wakeup and bedtime)

missing_days_list = []
for participant_id in complete_participant_ids:
    heartratebymin_participant = heartratebymin_selected.loc[heartratebymin_selected['ParticipantIdentifier'] == participant_id].copy()
    stepcountbymin_participant = stepcountbymin_selected.loc[stepcountbymin_selected['ParticipantIdentifier'] == participant_id].copy()

    df_wakeup_bedtime_participant = df_wakeup_bedtime.loc[df_wakeup_bedtime['ParticipantIdentifier'] == participant_id].copy()

    # we minus 7 day becasue we want to include past 7 days data of day 1
    min_date = summary_surveytask.loc[
        summary_surveytask['ParticipantIdentifier'] == participant_id
    ].iloc[0].date_min - pd.Timedelta(days=7)
    date_range_length = 92

    for i in range(date_range_length):
        date = min_date + pd.Timedelta(days=i)
        # print(date)
        wakeup_time, bedtime_time = resolve_schedule_for_date(df_wakeup_bedtime_participant, date)


        # filter out the days with less than 8 hours of wearing fitbit (more than 8 hours of heart rate =0 or nan
        # within the wakeup and bedtime)
        heartratebymin_participant_date = heartratebymin_participant[heartratebymin_participant.Date == date]
        heartratebymin_participant_date = heartratebymin_participant_date[heartratebymin_participant_date.DateTime.dt.time >= wakeup_time]
        heartratebymin_participant_date = heartratebymin_participant_date[heartratebymin_participant_date.DateTime.dt.time <= bedtime_time]
        
        stepcountbymin_participant_date = stepcountbymin_participant[stepcountbymin_participant.Date == date]
        stepcountbymin_participant_date = stepcountbymin_participant_date[stepcountbymin_participant_date.DateTime.dt.time >= wakeup_time]
        stepcountbymin_participant_date = stepcountbymin_participant_date[stepcountbymin_participant_date.DateTime.dt.time <= bedtime_time]

        # Filter valid heart rate readings (not 0 and not NaN)
        # Build the mask on heartrate rows only to avoid index misalignment warnings.
        step_active_times = stepcountbymin_participant_date.loc[
            (stepcountbymin_participant_date['Value'] > 0) &
            (stepcountbymin_participant_date['Value'].notna()),
            'DateTime'
        ]
        valid_hr = heartratebymin_participant_date[
            ((heartratebymin_participant_date['Value'] != 0) &
             (heartratebymin_participant_date['Value'].notna())) |
            (heartratebymin_participant_date['DateTime'].isin(step_active_times))
        ]
        
        # Calculate time span from first to last valid reading
        if len(valid_hr) > 0:
            first_timestamp = valid_hr['DateTime'].min()
            last_timestamp = valid_hr['DateTime'].max()
            time_span = last_timestamp - first_timestamp
            valid_hours = time_span.total_seconds() / 3600
        else:
            valid_hours = 0

        wearing = 0 if valid_hours < 8 else 1


        # Mark as missing if less than 8 hours of valid data

        missing_days_list.append({
            'ParticipantIdentifier': participant_id,
            'Date': date,
            'DayWearing': wearing,
            'ValidHours': valid_hours
        })

df_missing_days = pd.DataFrame(missing_days_list)

# Proportion of prior ≤7 calendar days with sufficient wear (excludes today)
df_missing_days['past7days_daywearing'] = (
    df_missing_days
    .sort_values(['ParticipantIdentifier', 'Date'], kind='mergesort')
    .groupby('ParticipantIdentifier', sort=False)['DayWearing']
    .transform(lambda s: _mean_prior_rows(s, window=7, min_periods=1))
)

# add a column of next day wearing
# df_missing_days['nextday_wearing'] = df_missing_days['DayWearing'].shift(-1)

df_missing_days.to_csv(os.path.join(folder, 'missing_days.csv'), index=False)


# %%
print(df_missing_days[df_missing_days['ParticipantIdentifier'] == "219"][51:100])

# %%
## extract hourly level missingness

# check missing hours by filtering out hours with less than a certain threshold of valid heart rate data
# within the wakeup and bedtime)

missing_hours_list = []

for participant_id in complete_participant_ids:
    heartratebymin_participant = heartratebymin_selected[heartratebymin_selected['ParticipantIdentifier'] == participant_id].copy()
    stepcountbymin_participant = stepcountbymin_selected.loc[stepcountbymin_selected['ParticipantIdentifier'] == participant_id].copy()
    df_wakeup_bedtime_participant = df_wakeup_bedtime.loc[df_wakeup_bedtime['ParticipantIdentifier'] == participant_id].copy()

    # we minus 7 day because we want to include past 7 days data of day 1
    # TODO: figured out we may not need to minus 1 day because we start modeling step counts after the first day of end of day survey
    min_date = summary_surveytask.loc[
        summary_surveytask['ParticipantIdentifier'] == participant_id
    ].iloc[0].date_min - pd.Timedelta(days=7)
    date_range_length = 92

    for i in range(date_range_length):
        date = min_date + pd.Timedelta(days=i)
        
        wakeup_time, bedtime_time = resolve_schedule_for_date(df_wakeup_bedtime_participant, date)

        # Get all heart rate data for this date
        heartratebymin_participant_date = heartratebymin_participant[heartratebymin_participant.Date == date]
        heartratebymin_participant_date = heartratebymin_participant_date[heartratebymin_participant_date.DateTime.dt.time >= wakeup_time]
        heartratebymin_participant_date = heartratebymin_participant_date[heartratebymin_participant_date.DateTime.dt.time <= bedtime_time]

        stepcountbymin_participant_date = stepcountbymin_participant[stepcountbymin_participant.Date == date]
        stepcountbymin_participant_date = stepcountbymin_participant_date[stepcountbymin_participant_date.DateTime.dt.time >= wakeup_time]
        stepcountbymin_participant_date = stepcountbymin_participant_date[stepcountbymin_participant_date.DateTime.dt.time <= bedtime_time]
        
        # Create hourly bins from wakeup to bedtime
        wakeup_datetime = pd.Timestamp.combine(date, wakeup_time)
        bedtime_datetime = pd.Timestamp.combine(date, bedtime_time)
        
        # wake up set to floor
        # anchor_date = pd.Timestamp('2000-01-01')  
        # wakeup_floor = (
        # pd.Timestamp.combine(anchor_date, wakeup_time)
        # .floor('H')                       
        # .time()
        # )

        # bedtime set to ceil
        # bedtime_ceil = (
        # pd.Timestamp.combine(anchor_date, bedtime_time)
        # .ceil('H')                       
        # .time()
        # )
        
        # Generate hourly time bins
        num_decisions = 2 # 2 decision points per day wakeup + 1, wakeup + 6
        current_decision = 0
        while current_decision < num_decisions:
            start_window = wakeup_datetime + pd.Timedelta(hours=1) if current_decision == 0 else wakeup_datetime + pd.Timedelta(hours=6)
            end_window = start_window + pd.Timedelta(hours=4)
            
            # Filter data for this hour
            hour_data = heartratebymin_participant_date[
                (heartratebymin_participant_date['DateTime'] >= start_window) &
                (heartratebymin_participant_date['DateTime'] < end_window)
            ]
            
            step_active_times = stepcountbymin_participant_date.loc[
                (stepcountbymin_participant_date['Value'] > 0) &
                (stepcountbymin_participant_date['Value'].notna()),
                'DateTime'
            ]

            # Filter valid heart rate readings (not 0 and not NaN)
            valid_hr = hour_data[
                (hour_data['Value'] != 0) & 
                (hour_data['Value'].notna()) | 
                (hour_data['DateTime'].isin(step_active_times))
            ]
            
            # Calculate time span of valid data in this hour
            if len(valid_hr) > 0:
                first_timestamp = valid_hr['DateTime'].min()
                last_timestamp = valid_hr['DateTime'].max()
                time_span = last_timestamp - first_timestamp
                valid_minutes = time_span.total_seconds() / 60
            else:
                valid_minutes = 0
            
            # Mark as missing if less than a threshold (e.g., 30 minutes of valid data in the hour)
            # You can adjust this threshold as needed
            wearing = 0 if valid_minutes < 200 else 1
            
            missing_hours_list.append({
                'ParticipantIdentifier': participant_id,
                'Date': date,
                'DecisionTime': current_decision,
                'DateTimeStart': start_window,
                'HourWearing': wearing
            })
            
            current_decision += 1

df_missing_hours = pd.DataFrame(missing_hours_list)
print(df_missing_hours[df_missing_hours['ParticipantIdentifier'] == "219"])

# save the dataframe
df_missing_hours.to_csv(os.path.join(folder, 'missing_hours.csv'), index=False)


# %%
## extract 2 hours level missingness

# check missing hours by filtering out hours with less than a certain threshold of valid heart rate data
# within the wakeup and bedtime)

missing_2hours_list = []

for participant_id in complete_participant_ids:
    heartratebymin_participant = heartratebymin_selected[heartratebymin_selected['ParticipantIdentifier'] == participant_id].copy()
    stepcountbymin_participant = stepcountbymin_selected.loc[stepcountbymin_selected['ParticipantIdentifier'] == participant_id].copy()

    df_wakeup_bedtime_participant = df_wakeup_bedtime.loc[df_wakeup_bedtime['ParticipantIdentifier'] == participant_id].copy()

    # we minus 1 day because we want to include yesterday's data of day 1
    # TODO: figured out we may not need to minus 1 day because we start modeling step counts after the first day of end of day survey
    min_date = summary_surveytask.loc[
        summary_surveytask['ParticipantIdentifier'] == participant_id
    ].iloc[0].date_min 
    date_range_length = 85

    for i in range(date_range_length):
        date = min_date + pd.Timedelta(days=i)
        
        wakeup_time, bedtime_time = resolve_schedule_for_date(df_wakeup_bedtime_participant, date)

        # Get all heart rate data for this date
        heartratebymin_participant_date = heartratebymin_participant[heartratebymin_participant.Date == date]
        # heartratebymin_participant_date = heartratebymin_participant_date[heartratebymin_participant_date.DateTime.dt.time >= wakeup_time]
        heartratebymin_participant_date = heartratebymin_participant_date[heartratebymin_participant_date.DateTime.dt.time <= bedtime_time]

        stepcountbymin_participant_date = stepcountbymin_participant[stepcountbymin_participant.Date == date]
        # stepcountbymin_participant_date = stepcountbymin_participant_date[stepcountbymin_participant_date.DateTime.dt.time >= wakeup_time]
        stepcountbymin_participant_date = stepcountbymin_participant_date[stepcountbymin_participant_date.DateTime.dt.time <= bedtime_time]
        
        # Create hourly bins from wakeup to bedtime
        wakeup_datetime = pd.Timestamp.combine(date, wakeup_time)
        bedtime_datetime = pd.Timestamp.combine(date, bedtime_time)
        
        
        # Generate hourly time bins
        num_decisions = 2 # 2 decision points per day wakeup + 1, wakeup + 6
        current_decision = 0
        while current_decision < num_decisions:
            end_window = wakeup_datetime + pd.Timedelta(hours=1) if current_decision == 0 else wakeup_datetime + pd.Timedelta(hours=6)
            start_window = end_window - pd.Timedelta(hours=2)

            # print(start_window, end_window)
            
            # Filter data for this hour
            hour_data = heartratebymin_participant_date[
                (heartratebymin_participant_date['DateTime'] >= start_window) &
                (heartratebymin_participant_date['DateTime'] < end_window)
            ]
            
            step_active_times = stepcountbymin_participant_date.loc[
                (stepcountbymin_participant_date['Value'] > 0) &
                (stepcountbymin_participant_date['Value'].notna()),
                'DateTime'
            ]

            # Filter valid heart rate readings (not 0 and not NaN)
            valid_hr = hour_data[
                (hour_data['Value'] != 0) & 
                (hour_data['Value'].notna()) |
                (hour_data['DateTime'].isin(step_active_times))
            ]
            # print(valid_hr)
            
            # Calculate time span of valid data in this hour
            if len(valid_hr) > 0:
                first_timestamp = valid_hr['DateTime'].min()
                last_timestamp = valid_hr['DateTime'].max()
                time_span = last_timestamp - first_timestamp
                valid_minutes = time_span.total_seconds() / 60
            else:
                valid_minutes = 0
            
            # Mark as missing if less than a threshold (e.g., 30 minutes of valid data in the hour)
            # You can adjust this threshold as needed
            wearing = 0 if valid_minutes <= 100 else 1
            
            missing_2hours_list.append({
                'ParticipantIdentifier': participant_id,
                'Date': date,
                'DecisionTime': current_decision,
                # 'DateTimeStart': start_window,
                'HourWearing': wearing
            })
            
            current_decision += 1

df_2hours = pd.DataFrame(missing_2hours_list)
# print(df_30_minutes[df_30_minutes['ParticipantIdentifier'] == 13].head())

print(df_2hours)

# print the proportion of Hourwearing =0
prop = df_2hours['HourWearing'].value_counts() / len(df_2hours)
print(prop)

# save the dataframe
df_2hours.to_csv(os.path.join(folder, 'missing_2hours.csv'), index=False)


# %%
# generate a 2x2 table of wearing in the morning (wakeup-1 hour to wakeup+1 hour)
# vs wearing in the rest of that day (after wakeup+1 hour to bedtime)

def _to_time(x):
    if pd.isna(x):
        return None
    if isinstance(x, pd.Timestamp):
        return x.time()
    if hasattr(x, 'hour') and hasattr(x, 'minute'):
        return x
    t = pd.to_datetime(str(x), errors='coerce')
    return None if pd.isna(t) else t.time()


def _participant_mask(series, participant_id):
    """Match participant IDs across int/float/str storage."""
    return series.astype(str) == str(participant_id)


def _get_wakeup_bedtime(df_wb_participant, date):
    if df_wb_participant.empty:
        return None, None
    try:
        wakeup_time, bedtime_time = resolve_schedule_for_date(df_wb_participant, date)
    except Exception:
        return None, None
    if pd.isna(wakeup_time) or pd.isna(bedtime_time):
        return None, None
    return wakeup_time, bedtime_time


def _valid_minutes(hr_df, step_df):
    # A minute is treated as wearable if HR is valid, or step count is positive.
    step_active_times = step_df.loc[
        (step_df['Value'] > 0) & (step_df['Value'].notna()),
        'DateTime'
    ]
    valid_hr = hr_df.loc[
        ((hr_df['Value'] != 0) & (hr_df['Value'].notna())) |
        (hr_df['DateTime'].isin(step_active_times))
    ]

    if len(valid_hr) > 0:
        first_timestamp = valid_hr['DateTime'].min()
        last_timestamp = valid_hr['DateTime'].max()
        time_span = last_timestamp - first_timestamp
        return time_span.total_seconds() / 60
    return 0


# Tune this threshold if needed
RESTDAY_MIN_VALID_MINUTES = 200  # rest-of-day window

if 'df_2hours' not in globals():
    raise ValueError('Run the first block that creates df_2hours before this 2x2 table block.')

# Use morning wearing directly from block 1 (DecisionTime == 0)
morning_lookup = (
    df_2hours.loc[df_2hours['DecisionTime'] == 0, ['ParticipantIdentifier', 'Date', 'HourWearing']]
    .copy()
)
morning_lookup['Date'] = pd.to_datetime(morning_lookup['Date'], errors='coerce').dt.normalize()
morning_lookup = (
    morning_lookup.dropna(subset=['Date'])
    .groupby(['ParticipantIdentifier', 'Date'], as_index=False)['HourWearing']
    .max()
    .rename(columns={'HourWearing': 'morning_wearing'})
)
morning_lookup['ParticipantIdentifier'] = morning_lookup['ParticipantIdentifier'].astype(str)
morning_lookup = morning_lookup.set_index(['ParticipantIdentifier', 'Date'])['morning_wearing']

wear_rows = []

for participant_id in complete_participant_ids:
    hr_p = heartratebymin_selected.loc[
        heartratebymin_selected['ParticipantIdentifier'] == participant_id
    ].copy()
    step_p = stepcountbymin_selected.loc[
        stepcountbymin_selected['ParticipantIdentifier'] == participant_id
    ].copy()
    wb_p = df_wakeup_bedtime.loc[
        _participant_mask(df_wakeup_bedtime['ParticipantIdentifier'], participant_id)
    ].copy()

    if hr_p.empty:
        continue

    hr_p['Date'] = pd.to_datetime(hr_p['Date']).dt.normalize()
    step_p['Date'] = pd.to_datetime(step_p['Date']).dt.normalize()

    # Match missing_2hours / df_2hours calendar grid (date_min as date, not normalized Timestamp).
    min_date = summary_surveytask.loc[
        summary_surveytask['ParticipantIdentifier'] == participant_id
    ].iloc[0].date_min
    if pd.isna(min_date):
        continue

    date_range_length = 85

    for i in range(date_range_length):
        date = min_date + pd.Timedelta(days=i)
        wakeup_time, bedtime_time = _get_wakeup_bedtime(wb_p, date)
        if wakeup_time is None or bedtime_time is None:
            continue

        wakeup_dt = pd.Timestamp.combine(date, wakeup_time)
        morning_end = wakeup_dt + pd.Timedelta(hours=1)
        bedtime_dt = pd.Timestamp.combine(date, bedtime_time)

        # Rest-of-day: after wakeup+1 hour to bedtime
        hr_rest = hr_p.loc[
            (hr_p['DateTime'] >= morning_end) &
            (hr_p['DateTime'] <= bedtime_dt)
        ]
        step_rest = step_p.loc[
            (step_p['DateTime'] >= morning_end) &
            (step_p['DateTime'] <= bedtime_dt)
        ]

        restday_valid_minutes = _valid_minutes(hr_rest, step_rest)
        lookup_date = pd.Timestamp(date).normalize()
        morning_wearing = int(
            morning_lookup.get((str(participant_id), lookup_date), 0)
        )

        wear_rows.append({
            'ParticipantIdentifier': participant_id,
            'Date': date,
            'restday_valid_minutes': restday_valid_minutes,
            'morning_wearing': morning_wearing,
            'restday_wearing': int(restday_valid_minutes > RESTDAY_MIN_VALID_MINUTES)
        })

wear_day = pd.DataFrame(wear_rows)

# Safety net: if the HR/rest-day loop skipped a user, still emit morning_wearing from df_2hours.
if not wear_day.empty:
    have_wear = set(
        zip(
            wear_day['ParticipantIdentifier'].astype(str),
            pd.to_datetime(wear_day['Date']).dt.normalize(),
        )
    )
else:
    have_wear = set()

backfill_rows = []
for participant_id in complete_participant_ids:
    morning_rows = df_2hours.loc[
        (df_2hours['DecisionTime'] == 0)
        & _participant_mask(df_2hours['ParticipantIdentifier'], participant_id)
    ].copy()
    if morning_rows.empty:
        continue
    morning_rows['Date'] = pd.to_datetime(morning_rows['Date'], errors='coerce').dt.normalize()
    for _, mrow in morning_rows.iterrows():
        key = (str(participant_id), mrow['Date'])
        if key in have_wear:
            continue
        backfill_rows.append({
            'ParticipantIdentifier': participant_id,
            'Date': mrow['Date'],
            'restday_valid_minutes': np.nan,
            'morning_wearing': int(mrow['HourWearing']),
            'restday_wearing': np.nan,
        })
        have_wear.add(key)

if backfill_rows:
    wear_day = pd.concat([wear_day, pd.DataFrame(backfill_rows)], ignore_index=True)

wear_2x2 = pd.crosstab(
    wear_day['morning_wearing'].astype(int),
    wear_day['restday_wearing'].astype(int),
    rownames=['Morning wearing (0/1)'],
    colnames=['Rest-of-day wearing (0/1)'],
    dropna=False
).reindex(index=[0, 1], columns=[0, 1], fill_value=0)

print('2x2 count table:')
print(wear_2x2)

print('\n2x2 proportion table:')
print((wear_2x2 / wear_2x2.to_numpy().sum()).round(4))

# Next morning wearing: morning_wearing on the following calendar day (per participant).
wear_day['nextday_wearing'] = (
    wear_day
    .sort_values(['ParticipantIdentifier', 'Date'], kind='mergesort')
    .groupby('ParticipantIdentifier', sort=False)['morning_wearing']
    .shift(-1)
)

# Proportion of prior ≤7 calendar mornings with wear (excludes today)
wear_day['past7days_morning_wearing'] = (
    wear_day
    .sort_values(['ParticipantIdentifier', 'Date'], kind='mergesort')
    .groupby('ParticipantIdentifier', sort=False)['morning_wearing']
    .transform(lambda s: _mean_prior_rows(s, window=7, min_periods=1))
)

# save the next morning wearing data
wear_day.to_csv(folder/ 'wear_day.csv', index=False)

# %%
# extract hourly step counts in between wakeup and bedtime

hourly_step_counts = []

for participant_id in complete_participant_ids:
    stepcountbymin_participant = stepcountbymin_selected.loc[stepcountbymin_selected['ParticipantIdentifier'] == participant_id].copy()
    df_wakeup_bedtime_participant = df_wakeup_bedtime.loc[df_wakeup_bedtime['ParticipantIdentifier'] == participant_id].copy()

    min_date = summary_surveytask.loc[
        summary_surveytask['ParticipantIdentifier'] == participant_id
    ].iloc[0].date_min - pd.Timedelta(days=7)
    # print(min_date)
    date_range_length = 92

    for i in range(date_range_length):
        date = min_date + pd.Timedelta(days=i)
        
        wakeup_time, bedtime_time = resolve_schedule_for_date(df_wakeup_bedtime_participant, date)

        # print(participant_id)
        # print(date, wakeup_time, bedtime_time)

        # wake up set to floor
        # anchor_date = pd.Timestamp('2000-01-01')  
        # wakeup_floor = (
        # pd.Timestamp.combine(anchor_date, wakeup_time)
        # .floor('h')                       
        # .time()
        # )

        # bedtime set to ceil
        # bedtime_ceil = (
        # pd.Timestamp.combine(anchor_date, bedtime_time)
        # .ceil('h')                       
        # .time()
        # )

        # Get all heart rate data for this date
        stepcountbymin_participant_date = stepcountbymin_participant[stepcountbymin_participant.Date == date]
        stepcountbymin_participant_date = stepcountbymin_participant_date[stepcountbymin_participant_date.DateTime.dt.time >= wakeup_time]
        stepcountbymin_participant_date = stepcountbymin_participant_date[stepcountbymin_participant_date.DateTime.dt.time <= bedtime_time]
        
        # Create hourly bins from wakeup to bedtime
        wakeup_datetime = pd.Timestamp.combine(date, wakeup_time)
        bedtime_datetime = pd.Timestamp.combine(date, bedtime_time)
        

        
        # Generate hourly time bins
        decision_time = 0
        num_decisions = 2 # 2 decision points per day wakeup + 1, wakeup + 6
        while decision_time < num_decisions:
            start_window = wakeup_datetime + pd.Timedelta(hours=1) if decision_time == 0 else wakeup_datetime + pd.Timedelta(hours=6)
            end_window = start_window + pd.Timedelta(hours=4)

            # read heart rate data for this hour
            missing_hours_participant_date = df_missing_hours[
                (df_missing_hours['ParticipantIdentifier'] == participant_id) &
                (df_missing_hours['Date'] == date) &
                (df_missing_hours['DecisionTime'] == decision_time)
            ]

            wearing = missing_hours_participant_date['HourWearing'].iloc[0]
            
            # Filter data for this hour
            hour_data = stepcountbymin_participant_date[
                (stepcountbymin_participant_date['DateTime'] >= start_window) &
                (stepcountbymin_participant_date['DateTime'] < end_window)
            ]
            
            # Filter valid step count readings (not 0 and not NaN)
            valid_sc_init = hour_data[
                (hour_data['Value'].notna())
            ]
            
            # Calculate number of valid step counts in this hour
            if (len(valid_sc_init) > 0) and (wearing == 1):
                valid_sc = sum(valid_sc_init['Value'])
            else:
                valid_sc = np.nan

            # print the proportion of wearing == 1 but valid_sc !=0
            if len(valid_sc_init) > 0:
                check_valid_sc = sum(valid_sc_init['Value'])
                check_status = int((check_valid_sc > 0) & (wearing == 0))
            else:
                check_status = np.nan


            hourly_step_counts.append({
                'ParticipantIdentifier': participant_id,
                'Date': date,
                'DecisionTime': decision_time,
                'DateTimeStart': start_window,
                'StepCount': valid_sc,
                'CheckStatus': check_status
            })
            
            decision_time += 1

df_hourly_step_counts = pd.DataFrame(hourly_step_counts)

# EWM (gamma=6/7) over prior ≤7 same-slot step counts (excludes current slot)
df_hourly_step_counts = df_hourly_step_counts.sort_values(
    ["ParticipantIdentifier", "DecisionTime", "Date"],
    kind="mergesort",
).reset_index(drop=True)
df_hourly_step_counts["EMA_StepCount"] = (
    df_hourly_step_counts.groupby(["ParticipantIdentifier", "DecisionTime"], sort=False)[
        "StepCount"
    ].transform(lambda s: _ewm_prior_rows(s, gamma=6/7))
)

# print(df_hourly_step_counts[df_hourly_step_counts['participantidentifier'] == 22])
# print((df_hourly_step_counts[df_hourly_step_counts['ParticipantIdentifier'] == 31]))

# print the proportion of wearing == 0 but valid_sc !=0
print(len(df_hourly_step_counts[df_hourly_step_counts['CheckStatus'] == 1]) / len(df_hourly_step_counts))
nan_ratio = (
    df_hourly_step_counts['CheckStatus'].isna().sum()
    / len(df_hourly_step_counts)
)
print(nan_ratio)

# %%
print(df_hourly_step_counts[df_hourly_step_counts['ParticipantIdentifier'] == "219"][101:150])

# %%

today_step_counts = []

for participant_id in complete_participant_ids:
    stepcountbymin_participant = stepcountbymin_selected.loc[stepcountbymin_selected['ParticipantIdentifier'] == participant_id].copy()
    df_wakeup_bedtime_participant = df_wakeup_bedtime.loc[df_wakeup_bedtime['ParticipantIdentifier'] == participant_id].copy()

    min_date = summary_surveytask.loc[
        summary_surveytask['ParticipantIdentifier'] == participant_id
    ].iloc[0].date_min - pd.Timedelta(days=7)
    # print(min_date)
    date_range_length = 92

    for i in range(date_range_length):
        date = min_date + pd.Timedelta(days=i)
        
        wakeup_time, bedtime_time = resolve_schedule_for_date(df_wakeup_bedtime_participant, date)


        # Get all heart rate data for this date
        stepcountbymin_participant_date = stepcountbymin_participant[stepcountbymin_participant.Date == date]
 
        # Create hourly bins from wakeup to bedtime
        wakeup_datetime = pd.Timestamp.combine(date, wakeup_time)
        bedtime_datetime = pd.Timestamp.combine(date, bedtime_time)
        

        
 
            # read heart rate data for this hour
        missing_days_participant_date = df_missing_days[
            (df_missing_days['ParticipantIdentifier'] == participant_id) &
            (df_missing_days['Date'] == date)
        ]

        wearing = missing_days_participant_date['DayWearing'].iloc[0]
            
            # Filter data for this hour
        daily_data = stepcountbymin_participant_date[
            (stepcountbymin_participant_date['DateTime'] >= wakeup_datetime) &
            (stepcountbymin_participant_date['DateTime'] < bedtime_datetime)
        ]
            
            # Filter valid step count readings (not 0 and not NaN)
        valid_sc = daily_data[
            (daily_data['Value'].notna())
        ]
            
        # 
            # Calculate number of valid step counts in this hour
        if (len(valid_sc) > 0) and (wearing == 1):
            valid_sc = sum(valid_sc['Value'])
        else:
            valid_sc = np.nan
            

        today_step_counts.append({
            'ParticipantIdentifier': participant_id,
            'Date': date + pd.Timedelta(days=1),
            'TodayStepCount': valid_sc
        })
        

df_today_step_counts = pd.DataFrame(today_step_counts)

# Previous calendar row's TodayStepCount per participant (chronological by Date).
df_today_step_counts['YesterdayStepCount'] = (
    df_today_step_counts
    .sort_values(['ParticipantIdentifier', 'Date'], kind='mergesort')
    .groupby('ParticipantIdentifier', sort=False)['TodayStepCount']
    .shift(1)
)

#print ratio of wearing == 1 but valid_sc !=0
print(len(df_today_step_counts[df_today_step_counts['TodayStepCount'] > 0]) / len(df_today_step_counts))

print(df_today_step_counts[df_today_step_counts['ParticipantIdentifier'] == "151"])

#print rate of nan today step counts
print(len(df_today_step_counts[df_today_step_counts['TodayStepCount'].isna()]) / len(df_today_step_counts))


# %%
# extract hourly step counts in between wakeup and bedtime

prior_2hours_step_counts = []

for participant_id in complete_participant_ids:
    stepcountbymin_participant = stepcountbymin_selected.loc[stepcountbymin_selected['ParticipantIdentifier'] == participant_id].copy()
    df_wakeup_bedtime_participant = df_wakeup_bedtime.loc[df_wakeup_bedtime['ParticipantIdentifier'] == participant_id].copy()

    min_date = summary_surveytask.loc[
        summary_surveytask['ParticipantIdentifier'] == participant_id
    ].iloc[0].date_min
    # print(min_date)
    date_range_length = 85

    for i in range(date_range_length):
        date = min_date + pd.Timedelta(days=i)
        
        wakeup_time, bedtime_time = resolve_schedule_for_date(df_wakeup_bedtime_participant, date)


        # Get all heart rate data for this date
        stepcountbymin_participant_date = stepcountbymin_participant[stepcountbymin_participant.Date == date]
        stepcountbymin_participant_date = stepcountbymin_participant_date[stepcountbymin_participant_date.DateTime.dt.time >= wakeup_time]
        stepcountbymin_participant_date = stepcountbymin_participant_date[stepcountbymin_participant_date.DateTime.dt.time <= bedtime_time]
        
        # Create hourly bins from wakeup to bedtime
        wakeup_datetime = pd.Timestamp.combine(date, wakeup_time)
        bedtime_datetime = pd.Timestamp.combine(date, bedtime_time)
        

        
        # Generate hourly time bins
        decision_time = 0
        num_decisions = 2 # 2 decision points per day wakeup + 1, wakeup + 6
        while decision_time < num_decisions:
            end_window = wakeup_datetime + pd.Timedelta(hours=1) if decision_time == 0 else wakeup_datetime + pd.Timedelta(hours=6)
            start_window = end_window - pd.Timedelta(hours=0.5)

            # read heart rate data for this hour
            missing_2hours_participant_date = df_2hours[
                (df_2hours['ParticipantIdentifier'] == participant_id) &
                (df_2hours['Date'] == date) &
                (df_2hours['DecisionTime'] == decision_time)
            ]

            wearing = missing_2hours_participant_date['HourWearing'].iloc[0]
            
            # Filter data for this hour
            hour_data = stepcountbymin_participant_date[
                (stepcountbymin_participant_date['DateTime'] >= start_window) &
                (stepcountbymin_participant_date['DateTime'] < end_window)
            ]
            
            # Filter valid step count readings (not 0 and not NaN)
            valid_sc_init = hour_data[
                (hour_data['Value'].notna())
            ]
            
            # Calculate number of valid step counts in this hour
            if (len(valid_sc_init) > 0) and (wearing == 1):
                valid_sc = sum(valid_sc_init['Value'])
            else:
                valid_sc = np.nan

            # print the proportion of wearing == 1 but valid_sc !=0
            if len(valid_sc_init) > 0:
                check_valid_sc = sum(valid_sc_init['Value'])
                check_status = int((check_valid_sc > 0) & (wearing == 0))
            else:
                check_status = np.nan


            prior_2hours_step_counts.append({
                'ParticipantIdentifier': participant_id,
                'Date': date,
                'DecisionTime': decision_time,
                # 'DateTimeStart': start_window,
                'StepCount': valid_sc,
                'CheckStatus': check_status
            })
            
            decision_time += 1

df_prior_2hours_step_counts = pd.DataFrame(prior_2hours_step_counts)

# EWM (gamma=6/7) over prior ≤7 same-slot prior-2h step counts (excludes current slot)
df_prior_2hours_step_counts = df_prior_2hours_step_counts.sort_values(
    ["ParticipantIdentifier", "DecisionTime", "Date"],
    kind="mergesort",
).reset_index(drop=True)
df_prior_2hours_step_counts["EMA_Prior2HourStepCount"] = (
    df_prior_2hours_step_counts.groupby(["ParticipantIdentifier", "DecisionTime"], sort=False)[
        "StepCount"
    ].transform(lambda s: _ewm_prior_rows(s, gamma=6 / 7))
)

# print(df_hourly_step_counts[df_hourly_step_counts['participantidentifier'] == 22])
print((df_prior_2hours_step_counts[df_prior_2hours_step_counts['ParticipantIdentifier'] == 31]))

# print the proportion of wearing == 1 but valid_sc !=0
print(len(df_prior_2hours_step_counts[df_prior_2hours_step_counts['CheckStatus'] == 1]) / len(df_prior_2hours_step_counts))
nan_ratio = (
    df_prior_2hours_step_counts['CheckStatus'].isna().sum()
    / len(df_prior_2hours_step_counts)
)
print(nan_ratio)

# Wearable step quality audit (flags participants to add to EXCLUDED_PARTICIPANT_IDS).
_wearable_audit_df, _wearable_exclude_candidates = audit_wearable_step_quality(
    complete_participant_ids,
    df_prior_2hours_step_counts,
    df_hourly_step_counts,
    df_today_step_counts,
)
_wearable_audit_path = os.path.join(folder, "wearable_step_quality_audit.csv")
_wearable_audit_df.to_csv(_wearable_audit_path, index=False)
print(f"Wrote wearable step quality audit to {_wearable_audit_path}")

_excluded_from_span = [
    str(pid)
    for pid in span_qualified_participant_ids
    if str(pid) in {str(x) for x in EXCLUDED_PARTICIPANT_IDS}
]
if _excluded_from_span:
    print(
        "Wearable step quality: manually excluded span-qualified participants "
        f"({len(_excluded_from_span)}): "
        + ", ".join(sorted(_excluded_from_span))
    )

if _wearable_exclude_candidates.empty:
    print(
        "Wearable step quality: no new exclusion candidates among active participants."
    )
else:
    _candidate_path = os.path.join(folder, "wearable_exclude_candidates.txt")
    candidate_ids = _wearable_exclude_candidates["ParticipantIdentifier"].tolist()
    with open(_candidate_path, "w", encoding="utf-8") as f:
        f.write("\n".join(candidate_ids))
        f.write("\n")
    print(
        "Wearable step quality: NEW participants with no usable FourSC (hourly) "
        "step counts (add to EXCLUDED_PARTICIPANT_IDS):"
    )
    for _, row in _wearable_exclude_candidates.iterrows():
        print(
            f"  {row['ParticipantIdentifier']}: "
            f"prior2hour={row['prior2hour_non_nan']}, "
            f"hourly={row['hourly_non_nan']}, "
            f"today={row['today_non_nan']}, "
            f"reason={row['exclude_reason']}"
        )
    print(f"Wrote candidate IDs to {_candidate_path}")

# %%
# save the dataframe
df_today_step_counts.to_csv(os.path.join(folder, 'today_step_counts.csv'), index=False)
df_hourly_step_counts.to_csv(os.path.join(folder, 'hourly_step_counts.csv'), index=False)
df_prior_2hours_step_counts.to_csv(os.path.join(folder, 'prior_2hours_step_counts.csv'), index=False)

# %% [markdown]
# ## 8. Fitbit recorded physical activity

# %%
fitbit_log_data = pd.read_csv(folder / 'FitbitActivityLogs.csv')
fitbit_log_data['Date'] = pd.to_datetime(fitbit_log_data['EndDate']).dt.date
recorded_physical_activity = []
for participant_id in complete_participant_ids:
    fitbit_log_participant = fitbit_log_data.loc[fitbit_log_data['ParticipantIdentifier'] == participant_id].copy()

    min_date = summary_surveytask.loc[
        summary_surveytask['ParticipantIdentifier'] == participant_id
    ].iloc[0].date_min - pd.Timedelta(days=7)
    # print(min_date)
    date_range_length = 92

    for i in range(date_range_length):
        date = min_date + pd.Timedelta(days=i)

        day_log = fitbit_log_participant[fitbit_log_participant["Date"] == date]
        if day_log.empty:
            rpa = 0
        else:
            types = day_log["ActivityName"].dropna().unique()
            rpa = 0 if (len(types) == 1 and types[0] == "Walk") else 1

        recorded_physical_activity.append({
            "ParticipantIdentifier": participant_id,
            "Date": date,
            "RecordedPhysicalActivity": rpa,
        })

df_recorded_physical_activity = pd.DataFrame(recorded_physical_activity)

# Fraction of prior 7 calendar days with non-walk RPA logged (excludes today)
df_recorded_physical_activity = df_recorded_physical_activity.sort_values(
    ['ParticipantIdentifier', 'Date'], kind='mergesort'
)
df_recorded_physical_activity['Previous7DaysRPA'] = (
    df_recorded_physical_activity
    .groupby('ParticipantIdentifier', sort=False)['RecordedPhysicalActivity']
    .transform(lambda s: _mean_prior_rows(s, window=7, min_periods=1))
)

print(df_recorded_physical_activity)


#save the dataframe
df_recorded_physical_activity.to_csv(os.path.join(folder, 'recorded_physical_activity.csv'), index=False)


# %%
# Active minutes data

