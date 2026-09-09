# Extract MRT raw exports into analysis CSVs
#
# Reads the ADAPTS MRT source tables under `DATA_FOLDER` and writes one CSV per
# stream (surveys, schedule, page views, walking suggestions, wearables, steps).
# Cohort rules (completers, wear/FourSC, weekly/daily response, ≥1 CAE week,
# high-CAE ceiling) are the constants in the setup cell below — not a
# hand-picked ID list.
# Next: `2_combine_data_frame.py`.

# 0. Setup

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

from ewm_utils import ewma_gamma, mean_prior_delivered_fraction

plt.ion()

_combined = os.environ.get("ADAPR_COMBINED_DIR", "").strip()
if not _combined:
    raise SystemExit(
        "Set ADAPR_COMBINED_DIR to the folder with extracted MRT tables."
    )
DATA_FOLDER = Path(_combined).expanduser().resolve()
folder = DATA_FOLDER

# ---------------------------------------------------------------------------
# A. Protocol eligibility (inclusion into the study observation window)
# ---------------------------------------------------------------------------
# Planned active phase = 12 weeks of daily decision points ≈ 84 calendar days.
# Require EOD-survey *task* span covering nearly the full planned length so the
# participant was still in the active intervention window long enough to be a
# completer for the designed MRT (not early dropout / never-started).
MIN_ACTIVE_SPAN_DAYS = 83

# Staff / engineering accounts are not analysis units.
EXTRA_TESTER_IDS = ("test-Yuxuan",)

# When ProjectDeviceData has no IANA zone or UTC offset (empty placeholder
# rows, or the participant is missing from that table), treat stamps as
# US Eastern. Do not use UTC wall clock (shifts evening local dates) and
# do not leave NaT (drops otherwise span-eligible completers).
DEFAULT_IANA_TIMEZONE = "America/New_York"

# ---------------------------------------------------------------------------
# B. Analysis-sample restriction: wearable primary-outcome availability
# ---------------------------------------------------------------------------
# Primary proximal outcome = 4-hour post-decision step count (FourSC).
# FourSC is filled only when HourWearing==1 *and* step rows exist in the
# window; HR-based wear can be 1 while StepCount stays NaN. Both checks are
# required:
#   B1. sum(HourWearing) over 4h post-decision windows
#       ≥ MIN_DECISION_WINDOW_WEAR_SUM
#   B2. count of non-missing FourSC (hourly StepCount)
#       ≥ MIN_FOURSC_NON_NAN
#
# Scientific rationale: available-case / completers for the sensor outcome —
# need enough wear-valid decision windows *and* usable proximal step outcomes.
MIN_DECISION_WINDOW_WEAR_SUM = 20  # require ≥ this many wear-valid 4h windows
MIN_FOURSC_NON_NAN = 20           # require ≥ this many non-missing FourSC

# ---------------------------------------------------------------------------
# C. Survey-response availability (self-report moderators / outcomes)
# ---------------------------------------------------------------------------
# Exclude participants with zero observed weekly surveys (all week_present==0)
# or zero observed daily EOD surveys (all daily_present==0). With no responses
# they contribute no information to survey-based analyses / missingness models
# for those panels.
# Also require ≥1 week with a non-missing CAE item. week_present can be 1 from
# a check-in that skipped CAE; a all-NaN CAE_avg makes the script-5 CAE model
# a fallback-zero and was the Jun-2026 "no CAE obs" drop (users 251, 257).
EXCLUDE_IF_ALL_WEEKLY_SURVEYS_MISSING = True
EXCLUDE_IF_ALL_DAILY_SURVEYS_MISSING = True
EXCLUDE_IF_ALL_CAE_MISSING = True
MIN_WEEKLY_SURVEYS_PRESENT = 1
MIN_DAILY_SURVEYS_PRESENT = 1
MIN_CAE_WEEKS_PRESENT = 1
_CAE_ITEM_COLS = tuple(f"CAE-{i}" for i in range(1, 13))

# High-CAE ceiling on the raw 1–7 scale (12 filled weekly slots). Users who
# never leave the top of the scale do not inform the hierarchical E_w / CAE /
# mediator pools in scripts 4–5. Drop if the 12-week minimum weekly CAE_avg
# is ≥ 6, or week-1 CAE_avg equals 7. User 18 stays (week 1 = 5.75, min = 5.75).
EXCLUDE_IF_HIGH_CAE = True
MIN_WEEKLY_CAE_THRESHOLD = 6.0
WEEK1_CAE_CEILING = 7.0

# Rare protocol overrides only (empty by default).
MANUAL_EXCLUSION_OVERRIDE_IDS = ()

# Backward-compatible name: filled later by rule-based wearable exclusions.
# Do not hand-edit participant IDs here.
EXCLUDED_PARTICIPANT_IDS: tuple[str, ...] = ()

# Analysis sample as of 2026-09-01 (Eastern default for missing device TZ):
# 43 span-eligible (≥83-day EOD-task span) → 15 excluded → 28 fitted.
# Wear/FourSC < 20: 112, 117, 138, 219.
# No weekly and/or no daily (and/or no CAE week): 82, 112, 251, 257, 259.
# Both: 112.
# High-CAE ceiling (min weekly CAE_avg ≥ 6 or week-1 CAE_avg = 7):
# 37, 141, 151, 160, 188, 225, 285.
# July 26 extract had 31 after also dropping 248, 259, 291, 327, 339 for
# FourSC < 20 under UTC wall-clock conversion; 259 still fails daily.

# Device metadata (timezone lookup source)

project_device_data = pd.read_csv(folder / "ProjectDeviceData_selected_fields_combined.csv")

# 1a. Timezone lookup (UTC → participant local)

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
def _first_nonempty(s):
    for x in s:
        if pd.notna(x) and str(x).strip() != "":
            return x
    return np.nan


tz_lookup = (
    tz_lookup.sort_values(["ParticipantIdentifier", "date"])
             .groupby(["ParticipantIdentifier", "date"], as_index=False)
             .agg({"timeZone": _first_nonempty, "utcOffset": _first_nonempty})
)

# Precompute static UTC-offset timedeltas once (vectorized conversion path)
_OFFSET_RE = re.compile(r"^([+-])(\d{2}):(\d{2})(?::(\d{2}))?$")


def _offset_to_timedelta(offset_str):
    # supports formats like -07:00:00 or +05:30:00
    if pd.isna(offset_str):
        return pd.NaT
    m = _OFFSET_RE.match(str(offset_str).strip())
    if not m:
        return pd.NaT
    sign = -1 if m.group(1) == "-" else 1
    hh = int(m.group(2))
    mm = int(m.group(3))
    ss = int(m.group(4) or 0)
    return sign * pd.Timedelta(hours=hh, minutes=mm, seconds=ss)


_offset_cache: dict[str, pd.Timedelta] = {}
for _off in tz_lookup["utcOffset"].dropna().astype(str).unique():
    _td = _offset_to_timedelta(_off)
    if not pd.isna(_td):
        _offset_cache[_off] = _td


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
    delta = _offset_cache.get(str(utc_offset), _offset_to_timedelta(utc_offset))
    if pd.isna(delta):
        return ts_utc.tz_convert(DEFAULT_IANA_TIMEZONE).tz_localize(None)
    return (ts_utc + delta).tz_localize(None)


def _as_calendar_dates(s):
    """Python ``datetime.date`` values; missing timestamps stay pandas ``NaT``.

    ``Series.dt.date`` on NaT becomes float NaN, which cannot be compared with
    ``datetime.date`` in ``groupby.min``. Keep a uniform object column instead.
    """
    ts = pd.to_datetime(s, errors="coerce")
    out = pd.Series(pd.NaT, index=ts.index, dtype=object)
    ok = ts.notna()
    if ok.any():
        out.loc[ok] = ts.loc[ok].dt.date
    return out


def convert_utc_columns_to_user_local(
    df,
    datetime_cols,
    participant_col="ParticipantIdentifier",
    join_date_col="InsertedDate",  # use "Timestamp" for pageview
):
    """Vectorized UTC→local conversion (groupby IANA tz / offset)."""
    out = df.copy()
    out["_pid"] = out[participant_col].astype(str)
    out["_join_date"] = (
        pd.to_datetime(out[join_date_col], errors="coerce", utc=True).dt.date
    )

    tz = tz_lookup.copy()
    tz["ParticipantIdentifier"] = tz["ParticipantIdentifier"].astype(str)

    out = out.merge(
        tz,
        how="left",
        left_on=["_pid", "_join_date"],
        right_on=["ParticipantIdentifier", "date"],
        suffixes=("", "_tz"),
    )

    # UTC calendar date is often the next local day for US evening stamps.
    miss = out["timeZone"].isna() & out["utcOffset"].isna()
    if miss.any():
        prev_date = pd.to_datetime(out.loc[miss, "_join_date"], errors="coerce") - pd.Timedelta(
            days=1
        )
        prev = pd.DataFrame(
            {
                "_pid": out.loc[miss, "_pid"].to_numpy(),
                "date": prev_date.dt.date.to_numpy(),
            },
            index=out.index[miss],
        )
        prev = prev.merge(
            tz,
            how="left",
            left_on=["_pid", "date"],
            right_on=["ParticipantIdentifier", "date"],
        )
        prev.index = out.index[miss]
        out.loc[miss, "timeZone"] = prev["timeZone"]
        out.loc[miss, "utcOffset"] = prev["utcOffset"]

    # Last resort: this participant's usual zone, then study Eastern.
    miss = out["timeZone"].isna() & out["utcOffset"].isna()
    if miss.any():
        pid_fb = (
            tz.sort_values(["ParticipantIdentifier", "date"])
            .groupby("ParticipantIdentifier", as_index=False)
            .agg({"timeZone": _first_nonempty, "utcOffset": _first_nonempty})
            .rename(columns={"ParticipantIdentifier": "_pid"})
        )
        fb = out.loc[miss, ["_pid"]].merge(pid_fb, on="_pid", how="left")
        fb.index = out.index[miss]
        out.loc[miss, "timeZone"] = fb["timeZone"]
        out.loc[miss, "utcOffset"] = fb["utcOffset"]

    miss = out["timeZone"].isna() & out["utcOffset"].isna()
    if miss.any():
        n_default = int(miss.sum())
        n_pid = int(out.loc[miss, "_pid"].nunique())
        print(
            f"WARNING: {n_default} timestamps ({n_pid} participants) have no "
            f"device timezone/offset; using {DEFAULT_IANA_TIMEZONE}."
        )
        out.loc[miss, "timeZone"] = DEFAULT_IANA_TIMEZONE

    for col in datetime_cols:
        ts = pd.to_datetime(out[col], errors="coerce", utc=True)
        local = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
        converted = pd.Series(False, index=out.index)

        # Group only non-empty IANA zones (NaN categories break some pandas groupbys)
        tz_ok = out["timeZone"].notna() & (out["timeZone"].astype(str).str.strip() != "")
        if tz_ok.any():
            for tz_name, idx in out.loc[tz_ok].groupby("timeZone", sort=False).groups.items():
                try:
                    local.loc[idx] = (
                        ts.loc[idx].dt.tz_convert(str(tz_name)).dt.tz_localize(None)
                    )
                    converted.loc[idx] = True
                except Exception:
                    continue

        need_offset = ~converted & ts.notna()
        if need_offset.any():
            off = out.loc[need_offset, "utcOffset"].map(
                lambda x: str(x) if pd.notna(x) else ""
            )
            deltas = off.map(_offset_cache)
            known = deltas.notna()
            if known.any():
                idx_k = deltas.index[known]
                local.loc[idx_k] = (
                    ts.loc[idx_k] + deltas.loc[idx_k]
                ).dt.tz_localize(None)
                converted.loc[idx_k] = True
            still = need_offset & ~converted
            if still.any():
                n_unknown = int(still.sum())
                print(
                    f"WARNING: {n_unknown} timestamps still unconverted after "
                    f"offset lookup; using {DEFAULT_IANA_TIMEZONE}."
                )
                local.loc[still] = (
                    ts.loc[still].dt.tz_convert(DEFAULT_IANA_TIMEZONE).dt.tz_localize(None)
                )
                converted.loc[still] = True

        out[col] = local

    out = out.drop(
        columns=[
            "_join_date",
            "_pid",
            "ParticipantIdentifier_tz",
            "date",
            "timeZone",
            "utcOffset",
        ],
        errors="ignore",
    )
    return out


def _filter_wearable_to_participant_windows(df, participant_ids, summary_df, pad_before=7, span_days=91):
    """Restrict minute-level wearable rows to each participant's analysis window."""
    if df.empty:
        return df
    windows = summary_df.loc[
        summary_df["ParticipantIdentifier"].isin(participant_ids),
        ["ParticipantIdentifier", "date_min"],
    ].copy()
    windows["start_date"] = pd.to_datetime(windows["date_min"]) - pd.Timedelta(days=pad_before)
    windows["end_date"] = windows["start_date"] + pd.Timedelta(days=span_days)
    windows["start_date"] = windows["start_date"].dt.date
    windows["end_date"] = windows["end_date"].dt.date

    out = df[df["ParticipantIdentifier"].isin(participant_ids)].merge(
        windows[["ParticipantIdentifier", "start_date", "end_date"]],
        on="ParticipantIdentifier",
        how="inner",
    )
    out = out[out["Date"].notna()]
    out = out[(out["Date"] >= out["start_date"]) & (out["Date"] <= out["end_date"])]
    return out.drop(columns=["start_date", "end_date"])


def _groupby_pid_frames(df, pid_col="ParticipantIdentifier"):
    """One DataFrame per participant (avoids repeated full-table boolean masks)."""
    if df is None or df.empty:
        return {}
    return {pid: g for pid, g in df.groupby(pid_col, sort=False)}


# 1b. Participant cohort

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


# 2. Survey tasks & results

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

# Survey timestamps are UTC in the export. Convert before taking the calendar
# date; otherwise evening tasks (e.g. 21:00 ET = 02:00 UTC next day) shift the
# entire participant analysis window forward by one day.
surveytask = convert_utc_columns_to_user_local(
    surveytask,
    datetime_cols=["InsertedDate", "DueDate"],
    participant_col="ParticipantIdentifier",
)
surveyquestionresults = convert_utc_columns_to_user_local(
    surveyquestionresults,
    datetime_cols=["StartDate", "EndDate", "InsertedDate"],
    participant_col="ParticipantIdentifier",
)

# Add local calendar dates after timezone conversion.
surveytask["date"] = _as_calendar_dates(surveytask["InsertedDate"])
surveyquestionresults["date"] = _as_calendar_dates(
    surveyquestionresults["InsertedDate"]
)

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
# 1c. Active-phase span (daily EOD survey tasks)

# survey_task_active = surveytask[surveytask.surveyname == 'MRT - Salience - Message Display']
survey_task_active = surveytask[
    (surveytask.SurveyName == 'MRT - Daily End of day survey and Planning Exercise')
    & surveytask['date'].notna()
]
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

# Protocol eligibility: observed active span covers planned ~12-week MRT
span_qualified_participant_ids = summary_surveytask[
    summary_surveytask['span_days'] >= MIN_ACTIVE_SPAN_DAYS
]['ParticipantIdentifier'].unique()
print(span_qualified_participant_ids, len(span_qualified_participant_ids))

# Analysis cohort starts as span-eligible minus rare manual overrides.
# Rule-based wearable exclusions (W1–W3) are applied after feature construction.
# Normalize IDs to str so merges/filters stay consistent across files.
span_qualified_participant_ids = np.array(
    [str(x) for x in span_qualified_participant_ids], dtype=object
)
complete_participant_ids = np.setdiff1d(
    span_qualified_participant_ids,
    np.array([str(x) for x in MANUAL_EXCLUSION_OVERRIDE_IDS], dtype=object),
)
print(
    "Span-eligible analysis starters (wearable rules applied later):",
    complete_participant_ids,
    len(complete_participant_ids),
)

# Harmonize ParticipantIdentifier dtypes used in later filters
surveytask["ParticipantIdentifier"] = surveytask["ParticipantIdentifier"].astype(str)
surveyquestionresults["ParticipantIdentifier"] = surveyquestionresults[
    "ParticipantIdentifier"
].astype(str)
summary_surveytask["ParticipantIdentifier"] = summary_surveytask[
    "ParticipantIdentifier"
].astype(str)

# filter out incomplete participants from surveytask and surveyquestionresults
surveytask = surveytask[surveytask['ParticipantIdentifier'].isin(complete_participant_ids)]
surveyquestionresults = surveyquestionresults[surveyquestionresults['ParticipantIdentifier'].isin(complete_participant_ids)]


# sort surveytask and surveyquestionresults by participantidentifier and date
surveytask = surveytask.sort_values(by=['ParticipantIdentifier', 'date'])
surveyquestionresults = surveyquestionresults.sort_values(by=['ParticipantIdentifier', 'date'])


# Extract four variables from weekly survey: affective valuation, CAE, perceived helpfulness, perceived pleasantness
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

# 3a. Flatten nested SurveyResults JSON


def _to_obj(x):
    if isinstance(x, (list, dict)):
        return x
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return []
    try:
        if pd.isna(x):
            return []
    except (TypeError, ValueError):
        pass
    return []


def flatten_survey_results(df):
    """Flatten nested StepResults JSON via itertuples (faster than iterrows)."""
    flat_rows = []
    for row in df.itertuples(index=False):
        pid = getattr(row, "ParticipantIdentifier", None)
        survey_key = getattr(row, "SurveyKey", None)
        inserted = getattr(row, "InsertedDate", None)

        survey_results = _to_obj(getattr(row, "StepResults", None))
        if isinstance(survey_results, dict):
            survey_results = [survey_results]
        if not isinstance(survey_results, list):
            continue

        for step in survey_results:
            if not isinstance(step, dict):
                continue

            step_id = step.get("StepIdentifier")
            step_start = step.get("StartDate")
            step_end = step.get("EndDate")

            results = _to_obj(step.get("Results"))
            if isinstance(results, dict):
                results = [results]
            if not isinstance(results, list):
                continue

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

                flat_rows.append((
                    pid,
                    survey_key,
                    inserted,
                    step_id,
                    step_start,
                    step_end,
                    r.get("Type"),
                    r.get("ResultIdentifier"),
                    ans_first,
                    ans_raw,
                    r.get("StartDate"),
                    r.get("EndDate"),
                ))

    return pd.DataFrame(
        flat_rows,
        columns=[
            "ParticipantIdentifier",
            "SurveyKey",
            "InsertedDate",
            "StepIdentifier",
            "StepStartDate",
            "StepEndDate",
            "ResultType",
            "ResultIdentifier",
            "AnswerFirst",
            "AnswersRaw",
            "QuestionStartDate",
            "QuestionEndDate",
        ],
    )


survey_results_flat = flatten_survey_results(surveyquestionresults)
print(survey_results_flat.shape)
survey_results_flat.head()

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

# force both to pandas datetime64[ns] (naive midnight)
df_weekly_survey["date"] = pd.to_datetime(df_weekly_survey["date"], errors="coerce").dt.normalize()
survey_task_weekly["date"] = pd.to_datetime(survey_task_weekly["date"], errors="coerce").dt.normalize()

print(df_weekly_survey["date"].dtype)
print(survey_task_weekly["date"].dtype)

def fill_weekly_12(df1, df2, id_col="ParticipantIdentifier", date_col="date",
                         tolerance_days=2, weeks=12):
    """Map observed weekly surveys onto a 12-week Sunday slot grid.

    ``df1`` is the wide weekly survey; ``df2`` is the weekly task calendar.
    Each participant gets ``weeks`` rows, anchored at their earliest task
    date and spaced every 7 days. An observation attaches to the nearest
    slot if it is within ``tolerance_days`` (default 2). In this MRT the
    slots are Sundays, so Monday/Tuesday completions still fill that
    Sunday. Script 2 then joins on the slot's ISO week, copying the
    emission onto all seven days — not a decision-time covariate.
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


# print the number of adherence=1 for each participant
all_counts = (
    df_weekly_filled.groupby("ParticipantIdentifier")["week_present"]
    .sum()
    .astype(int)
)
print(all_counts)
# print the number of survey results for each participant in the original dataframe
print(df_weekly_survey.groupby("ParticipantIdentifier").size())

# check what happens to participant 33

# check what happens to participant 33 in the filled dataframe

# participant 33 completed a survey on 2025-10-09 which is not within the 12 weeks

# save the filled dataframe
df_weekly_filled.to_csv(folder / "df_weekly_filled.csv", index=False)

# Extract variables from end-of-day survey: affective reflection, anticipated affect
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

# extract daily survey from survey task
survey_task_eod = surveytask.loc[surveytask.SurveyName == 'MRT - Daily End of day survey and Planning Exercise'].copy()
survey_task_eod['date'] = pd.to_datetime(survey_task_eod['InsertedDate'])

print(survey_task_eod)

def fill_daily_84(
    df1,
    participant_ids,
    anchor_summary,
    id_col="ParticipantIdentifier",
    date_col="date",
    days=84,
):
    """
    df1: wide daily-level survey question results frame with [id_col, date_col, value columns...]
    participant_ids: full eligible cohort, including participants with no survey responses
    anchor_summary: frame containing [id_col, date_min] from daily survey tasks
    Returns a frame with exactly `days` rows per participant, anchored at the
    participant's earliest survey-task date and spaced at 1-day intervals.
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

    anchor_lookup = (
        anchor_summary
        .drop_duplicates(id_col)
        .set_index(id_col)["date_min"]
    )

    out_parts = []
    for pid in participant_ids:
        # Selecting from the full cohort ensures participants with no responses
        # still receive 84 rows with daily_present=0.
        g = df1.loc[df1[id_col] == pid].copy()

        anchor = anchor_lookup.loc[pid]
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


df_daily_filled = fill_daily_84(
    df_daily_survey,
    participant_ids=complete_participant_ids,
    anchor_summary=summary_surveytask,
)


def _weekly_cae_ceiling_stats(df_weekly_filled):
    """Per-user min and week-1 ``CAE_avg`` on the 12 filled weekly slots.

    ``CAE_avg`` is the row mean of CAE-1..12, after recoding invalid 0s to 1
    (same as ``2_combine_data_frame._correct_one_to_seven_zeros``). NaN weeks
    are ignored, so a missing week 1 still flags the user if every observed
    week is ≥ ``MIN_WEEKLY_CAE_THRESHOLD``.
    """
    empty = pd.Series(dtype=float), pd.Series(dtype=float)
    if (
        df_weekly_filled is None
        or len(df_weekly_filled) == 0
        or "week" not in df_weekly_filled.columns
    ):
        return empty
    cae_cols = [c for c in _CAE_ITEM_COLS if c in df_weekly_filled.columns]
    if not cae_cols:
        return empty
    items = df_weekly_filled[cae_cols].apply(pd.to_numeric, errors="coerce")
    items = items.mask(items.eq(0), 1.0)
    panel = pd.DataFrame(
        {
            "pid": df_weekly_filled["ParticipantIdentifier"].astype(str),
            "week": df_weekly_filled["week"].astype(int),
            "cae_avg": items.mean(axis=1),
        }
    )
    panel = panel.loc[panel["week"].between(1, 12)]
    weekly = panel.groupby(["pid", "week"], sort=False)["cae_avg"].first()
    mins = weekly.groupby(level="pid").min()
    try:
        w1 = weekly.xs(1, level="week")
    except KeyError:
        w1 = pd.Series(dtype=float)
    return mins, w1


def audit_analysis_sample_exclusions(
    participant_ids,
    df_prior_2hours,
    df_hourly,
    df_today,
    df_missing_hours,
    df_weekly_filled=None,
    df_daily_filled=None,
):
    """
    Apply analysis-sample exclusion rules and return an audit table.

    Rules:
      1) HourWearing: sum over 4h post-decision windows
         < MIN_DECISION_WINDOW_WEAR_SUM
      2) FourSC: count of non-missing hourly StepCount
         < MIN_FOURSC_NON_NAN
      3) Weekly surveys: sum(week_present) < MIN_WEEKLY_SURVEYS_PRESENT
      4) Daily surveys: sum(daily_present) < MIN_DAILY_SURVEYS_PRESENT
      5) CAE: weeks with any non-missing CAE-1..12 < MIN_CAE_WEEKS_PRESENT
      6) High-CAE ceiling: 12-week min CAE_avg ≥ MIN_WEEKLY_CAE_THRESHOLD
         or week-1 CAE_avg = WEEK1_CAE_CEILING
    """
    override = {str(pid) for pid in MANUAL_EXCLUSION_OVERRIDE_IDS}

    def _non_nan_counts(df, value_col):
        if df is None or len(df) == 0 or value_col not in df.columns:
            return pd.Series(dtype=int)
        return (
            df.groupby(df["ParticipantIdentifier"].astype(str))[value_col]
            .apply(lambda s: int(s.notna().sum()))
        )

    def _present_sums(df, present_col):
        if df is None or len(df) == 0 or present_col not in df.columns:
            return pd.Series(dtype=int)
        return (
            df.groupby(df["ParticipantIdentifier"].astype(str))[present_col]
            .sum()
            .astype(int)
        )

    prior_ok = _non_nan_counts(df_prior_2hours, "StepCount")
    hourly_ok = _non_nan_counts(df_hourly, "StepCount")
    today_ok = _non_nan_counts(df_today, "TodayStepCount")
    weekly_present = _present_sums(df_weekly_filled, "week_present")
    daily_present = _present_sums(df_daily_filled, "daily_present")
    cae_weeks = pd.Series(dtype=int)
    if df_weekly_filled is not None and len(df_weekly_filled) > 0:
        cae_cols = [c for c in _CAE_ITEM_COLS if c in df_weekly_filled.columns]
        if cae_cols:
            cae_weeks = (
                df_weekly_filled.assign(
                    _pid=df_weekly_filled["ParticipantIdentifier"].astype(str),
                    _cae_any=df_weekly_filled[cae_cols].notna().any(axis=1),
                )
                .groupby("_pid")["_cae_any"]
                .sum()
                .astype(int)
            )
    min_weekly_cae, week1_cae = _weekly_cae_ceiling_stats(df_weekly_filled)

    if df_missing_hours is None or len(df_missing_hours) == 0:
        raise ValueError(
            "df_missing_hours is required for the unified HourWearing exclusion rule."
        )
    wear_sum = (
        df_missing_hours.groupby(
            df_missing_hours["ParticipantIdentifier"].astype(str)
        )["HourWearing"]
        .sum()
        .astype(int)
    )
    wear_n = (
        df_missing_hours.groupby(
            df_missing_hours["ParticipantIdentifier"].astype(str)
        )["HourWearing"]
        .size()
        .astype(int)
    )

    rows = []
    for pid in participant_ids:
        pid_key = str(pid)
        n_wear = int(wear_sum.get(pid_key, 0))
        n_windows = int(wear_n.get(pid_key, 0))
        n_foursc = int(hourly_ok.get(pid_key, 0))
        n_weekly = int(weekly_present.get(pid_key, 0))
        n_daily = int(daily_present.get(pid_key, 0))
        n_cae = int(cae_weeks.get(pid_key, 0))
        min_cae = min_weekly_cae.get(pid_key, np.nan)
        w1_cae = week1_cae.get(pid_key, np.nan)

        reasons = []
        if n_wear < MIN_DECISION_WINDOW_WEAR_SUM:
            reasons.append("insufficient_decision_window_wear")
        if n_foursc < MIN_FOURSC_NON_NAN:
            reasons.append("no_foursc_step_outcomes")
        if (
            EXCLUDE_IF_ALL_WEEKLY_SURVEYS_MISSING
            and n_weekly < MIN_WEEKLY_SURVEYS_PRESENT
        ):
            reasons.append("all_weekly_surveys_missing")
        if (
            EXCLUDE_IF_ALL_DAILY_SURVEYS_MISSING
            and n_daily < MIN_DAILY_SURVEYS_PRESENT
        ):
            reasons.append("all_daily_surveys_missing")
        if EXCLUDE_IF_ALL_CAE_MISSING and n_cae < MIN_CAE_WEEKS_PRESENT:
            reasons.append("all_cae_missing")
        if EXCLUDE_IF_HIGH_CAE:
            if np.isfinite(min_cae) and float(min_cae) >= float(
                MIN_WEEKLY_CAE_THRESHOLD
            ):
                reasons.append("min_weekly_cae_ge_6")
            if np.isfinite(w1_cae) and np.isclose(
                float(w1_cae), float(WEEK1_CAE_CEILING)
            ):
                reasons.append("week1_cae_eq_7")
        if pid_key in override and not reasons:
            reasons.append("manual_override")

        reason = "|".join(reasons)
        rows.append(
            {
                "ParticipantIdentifier": pid_key,
                "decision_window_wear_sum": n_wear,
                "decision_window_n": n_windows,
                "weekly_present_sum": n_weekly,
                "daily_present_sum": n_daily,
                "cae_weeks_present": n_cae,
                "min_weekly_cae": min_cae,
                "week1_cae": w1_cae,
                "prior2hour_non_nan": int(prior_ok.get(pid_key, 0)),
                "hourly_non_nan": n_foursc,
                "today_non_nan": int(today_ok.get(pid_key, 0)),
                "manual_override": pid_key in override,
                "recommend_exclude": bool(reason),
                "exclude_reason": reason,
            }
        )
    audit_df = pd.DataFrame(rows)
    exclude_df = audit_df[audit_df["recommend_exclude"]].copy()
    return audit_df, exclude_df


# Backward-compatible alias
audit_wearable_step_quality = audit_analysis_sample_exclusions


def _mean_prior_rows(series, window=7, min_periods=1):
    """Mean over prior `window` calendar rows; excludes the current row."""
    return series.shift(1).rolling(window=window, min_periods=min_periods).mean()


def _ewm_prior_rows(series, gamma=None, window=7, min_periods=1):
    """
    Exponentially weighted average over at most the prior `window` rows only.
    Excludes the current row.

    Computes:
        sum_{j=1}^n gamma^{j-1} x_{t-j} I_j
        -----------------------------------
        sum_{j=1}^n gamma^{j-1} I_j

    where j=1 is the most recent prior row and ``I_j`` is 0 if that row is
    NaN. Same implementation as ``ewm_utils.ewma_gamma``. NaN rows stay in
    the window so the decay follows calendar spacing (pandas
    ``ewm(..., ignore_na=False)`` on the same window, for a fixed gamma).

    ``gamma=None`` (default) derives the decay from ``n``, the calendar
    length of this rolling window (:func:`ewm_utils.gamma_from_n`), not the
    finite count. A full 7-row window gets ``gamma=6/7`` even if some days
    are missing; a partial window near the start (e.g. ``n=3``) gets
    ``gamma=2/3``. Pass a numeric ``gamma`` to force a fixed decay.
    """

    def _ewm_last(window_values):
        return ewma_gamma(window_values, gamma, empty=np.nan)

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

# check what happens to participant 33 in the filled dataframe

df_daily_filled.to_csv(folder / "df_daily_filled.csv", index=False)


# 4. Wakeup and bedtime schedule
# - this is useful for filling in the walking suggestions delivery time!

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
df_wakeup_bedtime["ParticipantIdentifier"] = df_wakeup_bedtime[
    "ParticipantIdentifier"
].astype(str)
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


def _wake_bed_interval(date, wakeup_time, bedtime_time):
    """Naive [wakeup, bedtime] timestamps; bedtime after midnight rolls to the next day.

    ``datetime.time`` has no date, so ``Timestamp.combine(same_date, 00:30)``
    is 12:30am *that* morning and the wear window is empty whenever bedtime
    is past midnight. Roll the end forward one day in that case.
    """
    d = pd.Timestamp(date).date()
    start = pd.Timestamp.combine(d, wakeup_time)
    end = pd.Timestamp.combine(d, bedtime_time)
    if pd.isna(start) or pd.isna(end):
        return start, end
    if end <= start:
        end = end + pd.Timedelta(days=1)
    return start, end


def _wear_frames_for_interval(hr_by_date, step_by_date, date_key, end_dt):
    """HR/step tables covering ``date_key`` through ``end_dt`` (next day if needed)."""
    keys = [date_key]
    end_date = pd.Timestamp(end_dt).date()
    day0 = pd.Timestamp(date_key).date()
    if end_date > day0:
        keys.append(day0 + datetime.timedelta(days=1))

    def _cat(store):
        parts = []
        for k in keys:
            g = store.get(k)
            if g is None:
                g = store.get(pd.Timestamp(k))
            if g is None or (hasattr(g, "empty") and g.empty):
                continue
            parts.append(g)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    return _cat(hr_by_date), _cat(step_by_date)


# Post-delivery MessageDisplay windows (minutes). A display strictly before
# the gif send is not an interaction with that notification.
_INTERACT_WINDOW_MIN = {0: 300.0, 1: 600.0}


def _as_naive_timestamp(val):
    """Parse to a tz-naive Timestamp so local delivery/display clocks compare."""
    ts = pd.to_datetime(val, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_localize(None)
    return ts


def _interacted_after_delivery(displays, delivery_ts, decision_time):
    """1 iff a MessageDisplay falls in this slot's post-delivery window.

    Morning (decision_time=0): [0, 300] minutes after ``delivery_ts``.
    Afternoon (decision_time=1): [0, 600] minutes after ``delivery_ts``.
    Displays before the send do not count. Applied per delivered slot — do
    not substitute a same-day display *count* for the timing test.
    """
    window_min = _INTERACT_WINDOW_MIN.get(int(decision_time))
    if window_min is None:
        return 0
    delivery_ts = _as_naive_timestamp(delivery_ts)
    if pd.isna(delivery_ts):
        return 0
    for interaction in displays:
        interaction = _as_naive_timestamp(interaction)
        if pd.isna(interaction) or interaction < delivery_ts:
            continue
        time_diff = (interaction - delivery_ts).total_seconds() / 60.0
        if 0.0 <= time_diff <= window_min:
            return 1
    return 0


def _time_to_timedelta(t):
    """Clock time → timedelta since midnight (for distance to slot anchors)."""
    if isinstance(t, pd.Timedelta):
        return t
    if not hasattr(t, "hour"):
        t = pd.Timestamp(t)
    return pd.Timedelta(
        hours=int(t.hour),
        minutes=int(t.minute),
        seconds=int(getattr(t, "second", 0) or 0),
    )


def _nearest_decision_time(delivered_td, m_time, a_time):
    """0 = morning, 1 = afternoon. Exact ties go to afternoon (old `<` test)."""
    d_m = abs((delivered_td - m_time).total_seconds())
    d_a = abs((delivered_td - a_time).total_seconds())
    return 0 if d_m < d_a else 1


def _gif_slot_record(participant_id, date, time_val, decision_time, walking_suggestion, interacted):
    return {
        "ParticipantIdentifier": participant_id,
        "Date": date,
        "Time": time_val,
        "DecisionTime": decision_time,
        "WalkingSuggestion": walking_suggestion,
        "Interacted": interacted,
    }


def _deliveries_by_nearest_slot(times, timestamps, m_time, a_time):
    """Map each send to the closer of morning / afternoon.

    Within a slot the first pair is the send closest to that slot's
    scheduled time (canonical delivery used for ``Time``).
    """
    buckets = {0: [], 1: []}
    for time_val, ts in zip(times, timestamps):
        td = _time_to_timedelta(time_val)
        dt = _nearest_decision_time(td, m_time, a_time)
        slot_td = m_time if dt == 0 else a_time
        dist = abs((td - slot_td).total_seconds())
        buckets[dt].append((dist, time_val, ts))
    for dt in buckets:
        buckets[dt].sort(key=lambda item: item[0])
        buckets[dt] = [(time_val, ts) for _, time_val, ts in buckets[dt]]
    return buckets


def _slot_rows_nearest_deliveries(
    participant_id, date, times, timestamps, m_time, a_time, displays,
):
    """Two slot records; ``WalkingSuggestion=1`` iff a send mapped to that slot.

    Used for one-delivery days and for ≥3 sends (resends / extras). Empty
    ``times`` yields both slots untreated (true no-delivery days).
    """
    default_m = (pd.Timestamp("2000-01-01") + m_time).time()
    default_a = (pd.Timestamp("2000-01-01") + a_time).time()
    by_slot = _deliveries_by_nearest_slot(times, timestamps, m_time, a_time)
    rows = []
    for dt, default_t in ((0, default_m), (1, default_a)):
        occ = by_slot[dt]
        if not occ:
            rows.append(_gif_slot_record(participant_id, date, default_t, dt, 0, 0))
            continue
        interacted = 0
        for _t, ts in occ:
            if _interacted_after_delivery(displays, ts, dt):
                interacted = 1
                break
        rows.append(
            _gif_slot_record(participant_id, date, occ[0][0], dt, 1, interacted)
        )
    return rows


# 5. Daily engagement (page views)
# 
# - for engagement, we may have yesterday's engagement data for day 1
# - this is different from survey variables...

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

   
    # Keep the same seven-day passive-data pad as the step panel so an
    # analysis-window first-slot lag can use pageviews observed before date_min.
    start_date = pd.to_datetime(summary_row['date_min']).date() - pd.Timedelta(days=7)
    end_date = start_date + pd.Timedelta(days=91)
    pageview_participant = pageview_participant[pageview_participant.Date >= start_date]
    pageview_participant = pageview_participant[pageview_participant.Date <= end_date]
    pageview_selected.append(pageview_participant)

pageview_selected = pd.concat(pageview_selected)
pageview_selected = pageview_selected[["ParticipantIdentifier", "Timestamp"]]


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
        wakeup_time, bedtime_time = resolve_schedule_for_date(df_wakeup_bedtime_participant, date)
        start_dt, end_dt = _wake_bed_interval(date, wakeup_time, bedtime_time)

        # filter out the days with less than 8 hours of wearing fitbit (more than 8 hours of heart rate =0 or nan
        # within the wakeup and bedtime)
        pageview_participant_date = pageview_participant[
            (pageview_participant.Timestamp >= start_dt)
            & (pageview_participant.Timestamp <= end_dt)
        ]
        

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

# EWM (gamma derived from k, the number of prior days available; 6/7 for a
# full 7-day window) over prior ≤7 calendar days of pageviews (excludes today)
df_daily_pageview['Past7DaysPageviewEMA'] = (
    df_daily_pageview
    .sort_values(['ParticipantIdentifier', 'Date'], kind='mergesort')
    .groupby('ParticipantIdentifier', sort=False)['DailyPageviewCount']
    .transform(lambda s: _ewm_prior_rows(s))
)

# get tomorrow's pageview count
# df_daily_pageview['TomorrowPageviewCount'] = (
#     df_daily_pageview['DailyPageviewCount'].shift(-1)
# )


df_daily_pageview.to_csv(os.path.join(folder, 'df_daily_pageview.csv'), index=False)


# remove the last row for each participant
df_daily_pageview = df_daily_pageview.groupby('ParticipantIdentifier').apply(lambda x: x.iloc[:-1])

print(df_daily_pageview)


hourly_pageview_list = []

for participant_id in complete_participant_ids:
    pageview_participant = pageview_selected[pageview_selected['ParticipantIdentifier'] == participant_id].copy()
    df_wakeup_bedtime_participant = df_wakeup_bedtime.loc[df_wakeup_bedtime['ParticipantIdentifier'] == participant_id].copy()

    # Retain the seven passive-data days preceding the intervention anchor.
    min_date = summary_surveytask.loc[
        summary_surveytask['ParticipantIdentifier'] == participant_id
    ].iloc[0].date_min - pd.Timedelta(days=7)
    date_range_length = 92

    for i in range(date_range_length):
        date = min_date + pd.Timedelta(days=i)
        
        wakeup_time, bedtime_time = resolve_schedule_for_date(df_wakeup_bedtime_participant, date)
        wakeup_datetime, bedtime_datetime = _wake_bed_interval(
            date, wakeup_time, bedtime_time
        )

        pageview_participant_date = pageview_participant[
            (pageview_participant.Timestamp >= wakeup_datetime)
            & (pageview_participant.Timestamp <= bedtime_datetime)
        ]
        
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

# Previous decision-slot pageview is constructed on the padded 92-day panel,
# before the downstream 84-day analysis filter.
df_hourly_pageview = df_hourly_pageview.sort_values(
    ['ParticipantIdentifier', 'Date', 'DecisionTime'],
    kind='mergesort',
)
df_hourly_pageview['HourlyPageviewCount_lag1'] = (
    df_hourly_pageview
    .groupby('ParticipantIdentifier', sort=False)['HourlyPageviewCount']
    .shift(1)
)

# EWM (gamma derived from k, the number of prior same-slot rows available)
# over prior ≤7 same-slot pageviews (excludes current slot)
df_hourly_pageview['Past7DaysHourlyPageviewEMA'] = (
    df_hourly_pageview
    .sort_values(['ParticipantIdentifier', 'Date', 'DecisionTime'], kind='mergesort')
    .groupby(['ParticipantIdentifier', 'DecisionTime'], sort=False)['HourlyPageviewCount']
    .transform(lambda s: _ewm_prior_rows(s))
)

# save the dataframe
df_hourly_pageview.to_csv(folder / 'hourly_pageview.csv', index=False)


# 6. Interventions (push notifications & survey display)
# - twice daily walking suggestions map to hourly, twice-daily, daily, and weekly
# - daily planning prompts map to hourly, twice-daily, daily, and weekly
# 
# 
# - Time zone: timestamp minus 4 hours for now!!!
# 
# Notes on click data
# - for walking suggestions, an important variable is whether the user opens the notification
# - for planning prompts, an important variable is whether the user fills in the survey!
# 
# - maybe the users saw the notifictaion but didn't open it... so perhaps we can also ignore this for now!!!!!!!
# 
# Missing data problem:
# - for user 13, end-of-day survey + planning action delivery information is missing on day 7/26 and 8/3. This is weired. 
# - TODO: check project device data, which is more accurate than notification sent csv...

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


gif_rows      = push_sent.loc[push_sent['Properties.NotificationIdentifier'].str.startswith('gif', na=False)].copy()
end_rows      = push_sent.loc[push_sent['Properties.NotificationIdentifier'].str.startswith('endOfDay', na=False)].copy()

# delete repeated rows
gif_rows = gif_rows.drop_duplicates()
end_rows = end_rows.drop_duplicates()


gif_rows_open      = push_open.loc[push_open['Properties.NotificationIdentifier'].str.startswith('gif', na=False)].copy()


# check the date range of the end of day survey for each participant
min_date_list = []
max_date_list = []
date_range_length_list = []
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

    min_date_list_gif.append(min_date)
    max_date_list_gif.append(max_date)

    # get the length of the date range
    date_range_length_gif = (max_date - min_date).days + 1
    date_range_length_list_gif.append(date_range_length_gif)


gif_rows_251 = gif_rows.loc[gif_rows['ParticipantIdentifier'] == "251"].copy()
gif_rows_251['Timestamp'] = pd.to_datetime(gif_rows_251['Timestamp'])
min_date_251 = gif_rows_251['Timestamp'].min()
max_date_251 = gif_rows_251['Timestamp'].max()
print(f"Participant 251 gif date range: {min_date_251} to {max_date_251}")
print(gif_rows_251)


end_rows['planning_prompt'] = 0
end_rows.loc[end_rows['Properties.NotificationIdentifier'].str.startswith('endOfDay_Planning', na=False), 'planning_prompt'] = 1

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

print(df_end_all)
# save the dataframe
df_end_all.to_csv(os.path.join(folder, 'df_end_all.csv'), index=False)


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
        wakeup_time = resolve_wakeup_for_date(df_wakeup_bedtime_participant, date)

        wake_td = pd.Timedelta(hours=wakeup_time.hour, minutes=wakeup_time.minute, seconds=wakeup_time.second)
        m_time = wake_td + pd.Timedelta(minutes=60)   # Morning: 1 hour after wake
        a_time = wake_td + pd.Timedelta(minutes=360)  # Afternoon: 6 hours after wake

        # search for date and time in gif_rows_participant
        gif_row = gif_rows_participant.loc[(gif_rows_participant['date'] == date)].copy()
        # gif_row_open = gif_rows_open_participant.loc[(gif_rows_open_participant['date'] == date)]
        tmp_par_date = tmp_participant.loc[(tmp_participant['date'] == date)].copy()

        
        # valid_count = [
        #     df_hourly_pageview.loc[(df_hourly_pageview['participantidentifier'] == participant_id) &
        #                         (df_hourly_pageview['date'] == date) &
        #                         (df_hourly_pageview['decision_time'] == dt),
        #                         'valid_count'].iloc[0]
        #     for dt in (0, 1)
        # ]

        displays = tmp_par_date["datetime_local"].values
        gif_row = gif_row.sort_values("Timestamp")
        n_sent = int(gif_row.shape[0])

        if n_sent == 0:
            # True no-delivery day only — n>=3 is *not* treated as untreated.
            df_gif_all.extend(
                _slot_rows_nearest_deliveries(
                    participant_id, date, [], [], m_time, a_time, displays,
                )
            )
        elif n_sent == 2:
            decision_time = 0
            for time_val, timestamp_val in zip(gif_row['time'].values, gif_row['Timestamp'].values):
                interacted = _interacted_after_delivery(
                    displays, timestamp_val, decision_time
                )
                df_gif_all.append(_gif_slot_record(
                    participant_id, date, time_val, decision_time, 1, interacted,
                ))
                decision_time += 1
        else:
            # n == 1 (nearest slot) or n >= 3 (resends / extras → nearest slot).
            if n_sent >= 3:
                print(
                    f"Warning: {n_sent} gif deliveries for participant "
                    f"{participant_id} on {date}; assigning each send to the nearest slot"
                )
            df_gif_all.extend(
                _slot_rows_nearest_deliveries(
                    participant_id,
                    date,
                    gif_row["time"].values,
                    gif_row["Timestamp"].values,
                    m_time,
                    a_time,
                    displays,
                )
            )

df_gif_all = pd.DataFrame(df_gif_all)

# Delivered-only interaction fraction over prior 14 slots (~7 days × 2);
# excludes current slot. Non-deliveries are not coded 0 in the denominator.
# Windows with no deliveries stay NaN (imputed later when a complete vector
# is required).
df_gif_all = df_gif_all.sort_values(['ParticipantIdentifier', 'Date', 'DecisionTime'], kind='mergesort')
_gif_parts = []
for _, _g in df_gif_all.groupby('ParticipantIdentifier', sort=False):
    _g = _g.copy()
    _g['Interacted_7d'] = mean_prior_delivered_fraction(
        _g['Interacted'], _g['WalkingSuggestion'], window=14, min_periods=7,
    )
    _gif_parts.append(_g)
df_gif_all = pd.concat(_gif_parts, ignore_index=True)

# recent_burden: X_d = daily walking-suggestion count (sum over AM/PM slots), then
#   _ewm_prior_rows on the daily series (window=7, gamma from the calendar
#   window length; r = 6/7 once a full 7-day window exists). Missing days
#   keep their slot in the decay and are omitted from the average.
_daily_panel = (
    df_gif_all.groupby(['ParticipantIdentifier', 'Date'], sort=False)['WalkingSuggestion']
    .sum()
    .reset_index(name='_daily_suggestions')
)
_daily_panel['recent_burden'] = (
    _daily_panel.groupby('ParticipantIdentifier', sort=False)['_daily_suggestions']
    .transform(lambda s: _ewm_prior_rows(s, window=7, min_periods=1))
)
df_gif_all = df_gif_all.merge(
    _daily_panel[['ParticipantIdentifier', 'Date', 'recent_burden']],
    on=['ParticipantIdentifier', 'Date'],
    how='left',
)

print(df_gif_all)
print(df_gif_all.Interacted.value_counts())
# save the dataframe
df_gif_all.to_csv(os.path.join(folder, 'df_gif_all.csv'), index=False)


# 7. Wearables — heart rate, steps, wear flags
# - case 1- missing hours: user receive walking suggestions at 9am, but only started wearing fitbit until 10 am 
#   treatment: 
# - case 2- missing days: user didn't wear fitbit for most hours between the wakeup and bedtime

# using heartrate to define drop out...etc
# Filter to cohort BEFORE timezone conversion (huge win on multi-GB files).
_hr_usecols = ["DateTime", "ParticipantIdentifier", "Value"]
heartratebymin = pd.read_csv(folder / "filtered_activities-heart.csv", usecols=_hr_usecols)
heartratebymin["ParticipantIdentifier"] = heartratebymin["ParticipantIdentifier"].astype(str)
heartratebymin = heartratebymin[
    heartratebymin["ParticipantIdentifier"].isin(complete_participant_ids)
].copy()
print(heartratebymin.head())

# change to local time
heartratebymin = convert_utc_columns_to_user_local(
    heartratebymin,
    datetime_cols=["DateTime"],
    participant_col="ParticipantIdentifier",
    join_date_col="DateTime",
)
print(heartratebymin.head())

# for the active phase, we require 12*7 = 84 days of step count data
heartratebymin["DateTime"] = pd.to_datetime(heartratebymin["DateTime"], errors="coerce")
heartratebymin["Date"] = _as_calendar_dates(heartratebymin["DateTime"])
heartratebymin = heartratebymin.sort_values(by=["ParticipantIdentifier", "DateTime"])

summary = (
    heartratebymin.dropna(subset=["Date"])
    .groupby("ParticipantIdentifier")["Date"]
    .agg(date_min="min", date_max="max", n_days="nunique")
    .assign(span_days=lambda df: (pd.to_datetime(df.date_max) -
                                  pd.to_datetime(df.date_min)).dt.days + 1)
    .reset_index()
)
print(summary)

# Select date range based on daily survey start and end date
heartratebymin_selected = _filter_wearable_to_participant_windows(
    heartratebymin, complete_participant_ids, summary_surveytask, pad_before=7, span_days=91
)
heartratebymin_selected = heartratebymin_selected[
    ["ParticipantIdentifier", "DateTime", "Value", "Date"]
]
print(heartratebymin_selected[heartratebymin_selected["ParticipantIdentifier"] == "118"])

# read step count data by minute — filter cohort before TZ conversion
_step_usecols = ["DateTime", "ParticipantIdentifier", "Value"]
stepcountbymin = pd.read_csv(folder / "filtered_activities-steps.csv", usecols=_step_usecols)
stepcountbymin["ParticipantIdentifier"] = stepcountbymin["ParticipantIdentifier"].astype(str)
stepcountbymin = stepcountbymin[
    stepcountbymin["ParticipantIdentifier"].isin(complete_participant_ids)
].copy()
print(stepcountbymin.head())

# change to local time
stepcountbymin = convert_utc_columns_to_user_local(
    stepcountbymin,
    datetime_cols=["DateTime"],
    participant_col="ParticipantIdentifier",
    join_date_col="DateTime",
)

stepcountbymin["DateTime"] = pd.to_datetime(stepcountbymin["DateTime"], errors="coerce")
stepcountbymin["Date"] = _as_calendar_dates(stepcountbymin["DateTime"])
stepcountbymin = stepcountbymin.sort_values(by=["ParticipantIdentifier", "DateTime"])

summary = (
    stepcountbymin.dropna(subset=["Date"])
    .groupby("ParticipantIdentifier")["Date"]
    .agg(date_min="min", date_max="max", n_days="nunique")
    .assign(span_days=lambda df: (pd.to_datetime(df.date_max) -
                                  pd.to_datetime(df.date_min)).dt.days + 1)
    .reset_index()
)
print(summary)

stepcountbymin_selected = _filter_wearable_to_participant_windows(
    stepcountbymin, complete_participant_ids, summary_surveytask, pad_before=7, span_days=91
)
stepcountbymin_selected = stepcountbymin_selected[
    ["ParticipantIdentifier", "DateTime", "Value", "Date"]
]
print(stepcountbymin_selected[stepcountbymin_selected["ParticipantIdentifier"] == "118"])

# check missing days by filtering out the days with less than 8 hours of wearing fitbit
# within the wakeup and bedtime)


# Prior-to-decision step covariate and its wear gate share this lookback.
# Decision times are wakeup+1h (AM) and wakeup+6h (PM); the window is
# half-open [end - 2h, end). HourWearing=1 iff the valid-wear span exceeds
# 100 minutes of those 120.
PRIOR_2HOUR_LOOKBACK = pd.Timedelta(hours=2)
PRIOR_2HOUR_WEAR_MINUTES = 100.0


def _prior_2hour_window(wakeup_datetime, decision_time):
    """Half-open [start, end) ending at the AM/PM decision time."""
    end_window = (
        wakeup_datetime + pd.Timedelta(hours=1)
        if int(decision_time) == 0
        else wakeup_datetime + pd.Timedelta(hours=6)
    )
    return end_window - PRIOR_2HOUR_LOOKBACK, end_window


def _sum_steps_in_window(step_df, start_window, end_window):
    """Sum finite minute-level steps on [start_window, end_window); NaN if none."""
    if step_df is None or len(step_df) == 0:
        return np.nan
    vals = pd.to_numeric(
        step_df.loc[
            (step_df["DateTime"] >= start_window)
            & (step_df["DateTime"] < end_window),
            "Value",
        ],
        errors="coerce",
    ).dropna()
    return float(vals.sum()) if len(vals) else np.nan


def _valid_wear_span_seconds(hr_df, step_df, start_dt, end_dt, end_inclusive=True):
    """First-to-last valid wearable minute span (seconds) in [start_dt, end_dt]."""
    if hr_df is None or hr_df.empty:
        return 0.0
    if end_inclusive:
        hr_win = hr_df[(hr_df["DateTime"] >= start_dt) & (hr_df["DateTime"] <= end_dt)]
        step_win = (
            step_df[(step_df["DateTime"] >= start_dt) & (step_df["DateTime"] <= end_dt)]
            if step_df is not None and not step_df.empty
            else None
        )
    else:
        hr_win = hr_df[(hr_df["DateTime"] >= start_dt) & (hr_df["DateTime"] < end_dt)]
        step_win = (
            step_df[(step_df["DateTime"] >= start_dt) & (step_df["DateTime"] < end_dt)]
            if step_df is not None and not step_df.empty
            else None
        )
    if hr_win.empty:
        return 0.0

    dt_vals = hr_win["DateTime"].to_numpy()
    vals = pd.to_numeric(hr_win["Value"], errors="coerce").to_numpy()
    hr_ok = (~np.isnan(vals)) & (vals != 0)

    if step_win is not None and not step_win.empty:
        step_active = step_win.loc[
            (step_win["Value"] > 0) & step_win["Value"].notna(), "DateTime"
        ].to_numpy()
        if len(step_active):
            step_set = set(step_active)
            in_step = np.fromiter(
                (t in step_set for t in dt_vals), dtype=bool, count=len(dt_vals)
            )
            valid_dt = dt_vals[hr_ok | in_step]
        else:
            valid_dt = dt_vals[hr_ok]
    else:
        valid_dt = dt_vals[hr_ok]

    if len(valid_dt) == 0:
        return 0.0
    return float((valid_dt.max() - valid_dt.min()) / np.timedelta64(1, "s"))


# Pre-split once (avoids O(N_participants) full-table scans)
_hr_by_pid = _groupby_pid_frames(heartratebymin_selected)
_step_by_pid = _groupby_pid_frames(stepcountbymin_selected)
_wb_by_pid = _groupby_pid_frames(df_wakeup_bedtime)
_anchor_date_min = summary_surveytask.set_index("ParticipantIdentifier")["date_min"]

missing_days_list = []
for participant_id in complete_participant_ids:
    heartratebymin_participant = _hr_by_pid.get(participant_id, pd.DataFrame())
    stepcountbymin_participant = _step_by_pid.get(participant_id, pd.DataFrame())
    df_wakeup_bedtime_participant = _wb_by_pid.get(participant_id, pd.DataFrame())

    hr_by_date = (
        {d: g for d, g in heartratebymin_participant.groupby("Date", sort=False)}
        if len(heartratebymin_participant)
        else {}
    )
    step_by_date = (
        {d: g for d, g in stepcountbymin_participant.groupby("Date", sort=False)}
        if len(stepcountbymin_participant)
        else {}
    )

    try:
        anchor = _anchor_date_min.loc[participant_id]
    except KeyError:
        anchor = _anchor_date_min.loc[str(participant_id)]
    min_date = pd.to_datetime(anchor).date() - datetime.timedelta(days=7)
    date_range_length = 92

    for i in range(date_range_length):
        date_key = min_date + datetime.timedelta(days=i)

        wakeup_time, bedtime_time = resolve_schedule_for_date(
            df_wakeup_bedtime_participant, date_key
        )
        start_dt, end_dt = _wake_bed_interval(date_key, wakeup_time, bedtime_time)
        hr_day, step_day = _wear_frames_for_interval(
            hr_by_date, step_by_date, date_key, end_dt
        )

        valid_hours = (
            _valid_wear_span_seconds(
                hr_day, step_day, start_dt, end_dt, end_inclusive=True
            )
            / 3600.0
        )
        wearing = 0 if valid_hours < 8 else 1

        missing_days_list.append({
            "ParticipantIdentifier": participant_id,
            "Date": date_key,
            "DayWearing": wearing,
            "ValidHours": valid_hours,
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


print(df_missing_days[df_missing_days['ParticipantIdentifier'] == "219"][51:100])

## extract hourly level missingness (4h post-decision windows)

missing_hours_list = []

for participant_id in complete_participant_ids:
    heartratebymin_participant = _hr_by_pid.get(participant_id, pd.DataFrame())
    stepcountbymin_participant = _step_by_pid.get(participant_id, pd.DataFrame())
    df_wakeup_bedtime_participant = _wb_by_pid.get(participant_id, pd.DataFrame())

    hr_by_date = (
        {d: g for d, g in heartratebymin_participant.groupby("Date", sort=False)}
        if len(heartratebymin_participant)
        else {}
    )
    step_by_date = (
        {d: g for d, g in stepcountbymin_participant.groupby("Date", sort=False)}
        if len(stepcountbymin_participant)
        else {}
    )

    try:
        anchor = _anchor_date_min.loc[participant_id]
    except KeyError:
        anchor = _anchor_date_min.loc[str(participant_id)]
    min_date = pd.to_datetime(anchor).date() - datetime.timedelta(days=7)
    date_range_length = 92

    for i in range(date_range_length):
        date_key = min_date + datetime.timedelta(days=i)

        wakeup_time, bedtime_time = resolve_schedule_for_date(
            df_wakeup_bedtime_participant, date_key
        )
        wakeup_datetime = pd.Timestamp.combine(date_key, wakeup_time)

        hr_day = hr_by_date.get(date_key, pd.DataFrame())
        step_day = step_by_date.get(date_key, pd.DataFrame())

        for current_decision in (0, 1):
            start_window = (
                wakeup_datetime + pd.Timedelta(hours=1)
                if current_decision == 0
                else wakeup_datetime + pd.Timedelta(hours=6)
            )
            end_window = start_window + pd.Timedelta(hours=4)
            valid_minutes = (
                _valid_wear_span_seconds(
                    hr_day, step_day, start_window, end_window, end_inclusive=False
                )
                / 60.0
            )
            wearing = 0 if valid_minutes < 200 else 1

            missing_hours_list.append({
                "ParticipantIdentifier": participant_id,
                "Date": date_key,
                "DecisionTime": current_decision,
                "DateTimeStart": start_window,
                "HourWearing": wearing,
            })

df_missing_hours = pd.DataFrame(missing_hours_list)
print(df_missing_hours[df_missing_hours["ParticipantIdentifier"] == "219"])

# save the dataframe
df_missing_hours.to_csv(os.path.join(folder, "missing_hours.csv"), index=False)


## extract 2 hours level missingness (prior-to-decision windows)

missing_2hours_list = []

for participant_id in complete_participant_ids:
    heartratebymin_participant = _hr_by_pid.get(participant_id, pd.DataFrame())
    stepcountbymin_participant = _step_by_pid.get(participant_id, pd.DataFrame())
    df_wakeup_bedtime_participant = _wb_by_pid.get(participant_id, pd.DataFrame())

    hr_by_date = (
        {d: g for d, g in heartratebymin_participant.groupby("Date", sort=False)}
        if len(heartratebymin_participant)
        else {}
    )
    step_by_date = (
        {d: g for d, g in stepcountbymin_participant.groupby("Date", sort=False)}
        if len(stepcountbymin_participant)
        else {}
    )

    try:
        anchor = _anchor_date_min.loc[participant_id]
    except KeyError:
        anchor = _anchor_date_min.loc[str(participant_id)]
    min_date = pd.to_datetime(anchor).date()
    date_range_length = 85

    for i in range(date_range_length):
        date_key = min_date + datetime.timedelta(days=i)

        wakeup_time, bedtime_time = resolve_schedule_for_date(
            df_wakeup_bedtime_participant, date_key
        )
        wakeup_datetime = pd.Timestamp.combine(date_key, wakeup_time)

        hr_day = hr_by_date.get(date_key, pd.DataFrame())
        step_day = step_by_date.get(date_key, pd.DataFrame())

        for current_decision in (0, 1):
            start_window, end_window = _prior_2hour_window(
                wakeup_datetime, current_decision
            )
            valid_minutes = (
                _valid_wear_span_seconds(
                    hr_day, step_day, start_window, end_window, end_inclusive=False
                )
                / 60.0
            )
            wearing = 0 if valid_minutes <= PRIOR_2HOUR_WEAR_MINUTES else 1

            missing_2hours_list.append({
                "ParticipantIdentifier": participant_id,
                "Date": date_key,
                "DecisionTime": current_decision,
                "HourWearing": wearing,
            })

df_2hours = pd.DataFrame(missing_2hours_list)

print(df_2hours)

# print the proportion of Hourwearing =0
prop = df_2hours['HourWearing'].value_counts() / len(df_2hours)
print(prop)

# save the dataframe
df_2hours.to_csv(os.path.join(folder, 'missing_2hours.csv'), index=False)


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

        wakeup_dt, bedtime_dt = _wake_bed_interval(date, wakeup_time, bedtime_time)
        morning_end = wakeup_dt + pd.Timedelta(hours=1)

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

# Normalize dtypes (main loop uses date; backfill uses Timestamp)
wear_day["ParticipantIdentifier"] = wear_day["ParticipantIdentifier"].astype(str)
wear_day["Date"] = pd.to_datetime(wear_day["Date"], errors="coerce").dt.normalize()

# 2x2 table uses only days with both morning and rest-of-day wear defined
# (backfill rows may have restday_wearing=NaN when HR/rest-day data are missing).
wear_2x2_df = wear_day.dropna(subset=["morning_wearing", "restday_wearing"]).copy()
wear_2x2_df["morning_wearing"] = wear_2x2_df["morning_wearing"].astype(int)
wear_2x2_df["restday_wearing"] = wear_2x2_df["restday_wearing"].astype(int)
_n_2x2_excluded = len(wear_day) - len(wear_2x2_df)
if _n_2x2_excluded:
    print(
        f"2x2 wear table: excluded {_n_2x2_excluded} rows with undefined restday_wearing"
    )

wear_2x2 = pd.crosstab(
    wear_2x2_df["morning_wearing"],
    wear_2x2_df["restday_wearing"],
    rownames=["Morning wearing (0/1)"],
    colnames=["Rest-of-day wearing (0/1)"],
    dropna=False,
).reindex(index=[0, 1], columns=[0, 1], fill_value=0)

print('2x2 count table:')
print(wear_2x2)

print('\n2x2 proportion table:')
print((wear_2x2 / wear_2x2.to_numpy().sum()).round(4))

# Next morning wearing: explicitly look up the DecisionTime==0 wear value on
# calendar date D+1. Unlike shift(-1), this cannot jump across a missing date.
next_morning = (
    df_2hours.loc[
        df_2hours['DecisionTime'] == 0,
        ['ParticipantIdentifier', 'Date', 'HourWearing'],
    ]
    .copy()
)
next_morning['ParticipantIdentifier'] = next_morning[
    'ParticipantIdentifier'
].astype(str)
next_morning['Date'] = (
    pd.to_datetime(next_morning['Date'], errors='coerce').dt.normalize()
    - pd.Timedelta(days=1)
)
next_morning = (
    next_morning
    .dropna(subset=['Date'])
    .groupby(['ParticipantIdentifier', 'Date'], as_index=False)['HourWearing']
    .max()
    .rename(columns={'HourWearing': 'nextday_wearing'})
)
wear_day = wear_day.merge(
    next_morning,
    on=['ParticipantIdentifier', 'Date'],
    how='left',
    validate='one_to_one',
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

# extract hourly step counts in between wakeup and bedtime

hourly_step_counts = []

for participant_id in complete_participant_ids:
    stepcountbymin_participant = stepcountbymin_selected.loc[stepcountbymin_selected['ParticipantIdentifier'] == participant_id].copy()
    df_wakeup_bedtime_participant = df_wakeup_bedtime.loc[df_wakeup_bedtime['ParticipantIdentifier'] == participant_id].copy()

    min_date = summary_surveytask.loc[
        summary_surveytask['ParticipantIdentifier'] == participant_id
    ].iloc[0].date_min - pd.Timedelta(days=7)
    date_range_length = 92

    for i in range(date_range_length):
        date = min_date + pd.Timedelta(days=i)
        
        wakeup_time, bedtime_time = resolve_schedule_for_date(df_wakeup_bedtime_participant, date)


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

        wakeup_datetime, bedtime_datetime = _wake_bed_interval(
            date, wakeup_time, bedtime_time
        )

        stepcountbymin_participant_date = stepcountbymin_participant[
            (stepcountbymin_participant.DateTime >= wakeup_datetime)
            & (stepcountbymin_participant.DateTime <= bedtime_datetime)
        ]
        

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

# EWM (gamma derived from k, the number of prior same-slot rows available)
# over prior ≤7 same-slot step counts (excludes current slot)
df_hourly_step_counts = df_hourly_step_counts.sort_values(
    ["ParticipantIdentifier", "DecisionTime", "Date"],
    kind="mergesort",
).reset_index(drop=True)
df_hourly_step_counts["EMA_StepCount"] = (
    df_hourly_step_counts.groupby(["ParticipantIdentifier", "DecisionTime"], sort=False)[
        "StepCount"
    ].transform(lambda s: _ewm_prior_rows(s))
)


# print the proportion of wearing == 0 but valid_sc !=0
print(len(df_hourly_step_counts[df_hourly_step_counts['CheckStatus'] == 1]) / len(df_hourly_step_counts))
nan_ratio = (
    df_hourly_step_counts['CheckStatus'].isna().sum()
    / len(df_hourly_step_counts)
)
print(nan_ratio)

print(df_hourly_step_counts[df_hourly_step_counts['ParticipantIdentifier'] == "219"][101:150])


today_step_counts = []

for participant_id in complete_participant_ids:
    stepcountbymin_participant = stepcountbymin_selected.loc[stepcountbymin_selected['ParticipantIdentifier'] == participant_id].copy()
    df_wakeup_bedtime_participant = df_wakeup_bedtime.loc[df_wakeup_bedtime['ParticipantIdentifier'] == participant_id].copy()

    min_date = summary_surveytask.loc[
        summary_surveytask['ParticipantIdentifier'] == participant_id
    ].iloc[0].date_min - pd.Timedelta(days=7)
    date_range_length = 92

    for i in range(date_range_length):
        date = min_date + pd.Timedelta(days=i)
        
        wakeup_time, bedtime_time = resolve_schedule_for_date(df_wakeup_bedtime_participant, date)
        wakeup_datetime, bedtime_datetime = _wake_bed_interval(
            date, wakeup_time, bedtime_time
        )

        stepcountbymin_participant_date = stepcountbymin_participant[
            (stepcountbymin_participant.DateTime >= wakeup_datetime)
            & (stepcountbymin_participant.DateTime < bedtime_datetime)
        ]
        

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
            'Date': date,
            'TodayStepCount': valid_sc
        })
        

df_today_step_counts = pd.DataFrame(today_step_counts)

# Yesterday at D = TodayStepCount of the previous calendar row (steps on D-1).
# Date is the measurement day — do not store under D+1; that plus shift(1)
# would make YesterdayStepCount a two-day lag.
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


# Prior-2h step counts: same [decision-2h, decision) window as HourWearing.

prior_2hours_step_counts = []

for participant_id in complete_participant_ids:
    stepcountbymin_participant = stepcountbymin_selected.loc[stepcountbymin_selected['ParticipantIdentifier'] == participant_id].copy()
    df_wakeup_bedtime_participant = df_wakeup_bedtime.loc[df_wakeup_bedtime['ParticipantIdentifier'] == participant_id].copy()

    min_date = summary_surveytask.loc[
        summary_surveytask['ParticipantIdentifier'] == participant_id
    ].iloc[0].date_min
    date_range_length = 85

    for i in range(date_range_length):
        date = min_date + pd.Timedelta(days=i)
        wakeup_time, bedtime_time = resolve_schedule_for_date(
            df_wakeup_bedtime_participant, date
        )
        wakeup_datetime = pd.Timestamp.combine(pd.Timestamp(date).date(), wakeup_time)

        for decision_time in (0, 1):
            start_window, end_window = _prior_2hour_window(
                wakeup_datetime, decision_time
            )
            missing_2hours_participant_date = df_2hours[
                (df_2hours['ParticipantIdentifier'] == participant_id) &
                (df_2hours['Date'] == date) &
                (df_2hours['DecisionTime'] == decision_time)
            ]
            wearing = missing_2hours_participant_date['HourWearing'].iloc[0]
            window_sum = _sum_steps_in_window(
                stepcountbymin_participant, start_window, end_window
            )
            valid_sc = window_sum if wearing == 1 else np.nan
            check_status = (
                np.nan if not np.isfinite(window_sum)
                else int((window_sum > 0) and (wearing == 0))
            )
            prior_2hours_step_counts.append({
                'ParticipantIdentifier': participant_id,
                'Date': date,
                'DecisionTime': decision_time,
                'StepCount': valid_sc,
                'CheckStatus': check_status
            })

df_prior_2hours_step_counts = pd.DataFrame(prior_2hours_step_counts)

# EWM (gamma derived from k, the number of prior same-slot rows available)
# over prior ≤7 same-slot prior-2h step counts (excludes current slot)
df_prior_2hours_step_counts = df_prior_2hours_step_counts.sort_values(
    ["ParticipantIdentifier", "DecisionTime", "Date"],
    kind="mergesort",
).reset_index(drop=True)
df_prior_2hours_step_counts["EMA_Prior2HourStepCount"] = (
    df_prior_2hours_step_counts.groupby(["ParticipantIdentifier", "DecisionTime"], sort=False)[
        "StepCount"
    ].transform(lambda s: _ewm_prior_rows(s))
)

print((df_prior_2hours_step_counts[df_prior_2hours_step_counts['ParticipantIdentifier'] == 31]))

# print the proportion of wearing == 1 but valid_sc !=0
print(len(df_prior_2hours_step_counts[df_prior_2hours_step_counts['CheckStatus'] == 1]) / len(df_prior_2hours_step_counts))
nan_ratio = (
    df_prior_2hours_step_counts['CheckStatus'].isna().sum()
    / len(df_prior_2hours_step_counts)
)
print(nan_ratio)

# Analysis-sample audit: HourWearing + all-missing weekly/daily surveys.
_analysis_audit_df, _analysis_exclude_df = audit_analysis_sample_exclusions(
    complete_participant_ids,
    df_prior_2hours_step_counts,
    df_hourly_step_counts,
    df_today_step_counts,
    df_missing_hours,
    df_weekly_filled=df_weekly_filled,
    df_daily_filled=df_daily_filled,
)
_wearable_audit_path = os.path.join(folder, "wearable_step_quality_audit.csv")
_analysis_audit_df.to_csv(_wearable_audit_path, index=False)
print(f"Wrote analysis-sample audit to {_wearable_audit_path}")

EXCLUDED_PARTICIPANT_IDS = tuple(
    sorted(_analysis_exclude_df["ParticipantIdentifier"].astype(str).tolist())
)
_exclusion_path = os.path.join(folder, "analysis_sample_exclusions.csv")
_analysis_exclude_df.to_csv(_exclusion_path, index=False)

if _analysis_exclude_df.empty:
    print("Analysis-sample rules: no participants excluded.")
else:
    print(
        f"Analysis-sample rules excluded {len(EXCLUDED_PARTICIPANT_IDS)} "
        f"span-eligible participants "
        f"(wear≥{MIN_DECISION_WINDOW_WEAR_SUM}; "
        f"FourSC≥{MIN_FOURSC_NON_NAN}; "
        f"weekly≥{MIN_WEEKLY_SURVEYS_PRESENT}; "
        f"daily≥{MIN_DAILY_SURVEYS_PRESENT}; "
        f"CAE≥{MIN_CAE_WEEKS_PRESENT}; "
        f"drop min CAE≥{MIN_WEEKLY_CAE_THRESHOLD:g} or "
        f"week-1 CAE={WEEK1_CAE_CEILING:g}):"
    )
    for _, row in _analysis_exclude_df.iterrows():
        print(
            f"  {row['ParticipantIdentifier']}: "
            f"wear_sum={row['decision_window_wear_sum']}/{row['decision_window_n']}, "
            f"fourSC={row['hourly_non_nan']}, "
            f"weekly={row['weekly_present_sum']}, "
            f"daily={row['daily_present_sum']}, "
            f"cae={row['cae_weeks_present']}, "
            f"min_cae={row['min_weekly_cae']}, "
            f"week1_cae={row['week1_cae']}, "
            f"reason={row['exclude_reason']}"
        )
    print(f"Wrote exclusion table to {_exclusion_path}")

analysis_participant_ids = np.setdiff1d(
    complete_participant_ids,
    list(EXCLUDED_PARTICIPANT_IDS),
)
print(
    f"Final analysis sample: {len(analysis_participant_ids)} participants "
    f"(from {len(complete_participant_ids)} span-eligible)"
)

# Restrict outputs to the rule-based analysis sample
def _keep_analysis(df, id_col="ParticipantIdentifier"):
    return df[df[id_col].astype(str).isin(set(map(str, analysis_participant_ids)))].copy()


df_today_step_counts = _keep_analysis(df_today_step_counts)
df_hourly_step_counts = _keep_analysis(df_hourly_step_counts)
df_prior_2hours_step_counts = _keep_analysis(df_prior_2hours_step_counts)
df_missing_days = _keep_analysis(df_missing_days)
df_missing_hours = _keep_analysis(df_missing_hours)
df_2hours = _keep_analysis(df_2hours)
df_weekly_filled = _keep_analysis(df_weekly_filled)
df_daily_filled = _keep_analysis(df_daily_filled)
if "wear_day" in globals():
    wear_day = _keep_analysis(wear_day)

# Prefer analysis sample for any remaining cohort loops
complete_participant_ids = analysis_participant_ids

# save the dataframe
df_today_step_counts.to_csv(os.path.join(folder, "today_step_counts.csv"), index=False)
df_hourly_step_counts.to_csv(os.path.join(folder, "hourly_step_counts.csv"), index=False)
df_prior_2hours_step_counts.to_csv(
    os.path.join(folder, "prior_2hours_step_counts.csv"), index=False
)
df_missing_days.to_csv(os.path.join(folder, "missing_days.csv"), index=False)
df_missing_hours.to_csv(os.path.join(folder, "missing_hours.csv"), index=False)
df_2hours.to_csv(os.path.join(folder, "missing_2hours.csv"), index=False)
df_weekly_filled.to_csv(folder / "df_weekly_filled.csv", index=False)
df_daily_filled.to_csv(folder / "df_daily_filled.csv", index=False)

# Align earlier cohort exports with the final rule-based analysis sample
_analysis_id_set = set(map(str, complete_participant_ids))
_cohort_csv_names = [
    "df_weekly_filled.csv",
    "df_daily_filled.csv",
    "df_daily_pageview.csv",
    "hourly_pageview.csv",
    "df_end_all.csv",
    "df_gif_all.csv",
    "wear_day.csv",
]
for _name in _cohort_csv_names:
    _path = folder / _name
    if not _path.exists():
        continue
    _df = pd.read_csv(_path)
    if "ParticipantIdentifier" not in _df.columns:
        continue
    _n0 = len(_df)
    _df = _df[_df["ParticipantIdentifier"].astype(str).isin(_analysis_id_set)]
    _df.to_csv(_path, index=False)
    print(f"Aligned {_name}: {_n0} -> {len(_df)} rows")
