# %%
# 0. import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.pyplot as plt
plt.ion()
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import datetime
import json
from pathlib import Path

# %%
# read cleaned data set
folder = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPT_MRT/rawdata/_combined")

df_4hour_step = pd.read_csv(folder / 'hourly_step_counts.csv')
df_today_step = pd.read_csv(folder / 'today_step_counts.csv')
df_prior2hours_step = pd.read_csv(folder / 'prior_2hours_step_counts.csv')
df_otherPA = pd.read_csv(folder / 'recorded_physical_activity.csv')

df_weekly = pd.read_csv(folder / 'df_weekly_filled.csv')
df_daily = pd.read_csv(folder / 'df_daily_filled.csv')

df_daily_pageview = pd.read_csv(folder / 'df_daily_pageview.csv')
df_hourly_pageview = pd.read_csv(folder / 'hourly_pageview.csv')


df_gif = pd.read_csv(folder / 'df_gif_all.csv')
df_salience = pd.read_csv(folder / 'df_salience_all.csv')
df_planning = pd.read_csv(folder / 'df_end_all.csv')

df_notwearing = pd.read_csv(folder / 'missing_days.csv')
df_wearing_morning = pd.read_csv(folder / 'wear_day.csv')

# %%
print(df_weekly)

# %%
# create a data frame to contain outcome and predictors for hourly step counts
df_4hour_step['Date'] = pd.to_datetime(df_4hour_step['Date'])
df_today_step['Date'] = pd.to_datetime(df_today_step['Date'])
df_prior2hours_step['Date'] = pd.to_datetime(df_prior2hours_step['Date'])
df_otherPA['Date'] = pd.to_datetime(df_otherPA['Date'])

df_weekly['Date'] = pd.to_datetime(df_weekly['date'])
df_daily['Date'] = pd.to_datetime(df_daily['date'])

df_daily_pageview['Date'] = pd.to_datetime(df_daily_pageview['Date'])
df_hourly_pageview['Date'] = pd.to_datetime(df_hourly_pageview['Date'])

df_gif['Date'] = pd.to_datetime(df_gif['Date'])
df_salience['Date'] = pd.to_datetime(df_salience['Date'])
df_planning['Date'] = pd.to_datetime(df_planning['date'])

df_notwearing['Date'] = pd.to_datetime(df_notwearing['Date'])
df_wearing_morning['Date'] = pd.to_datetime(df_wearing_morning['Date'])


# %%
print(f"Before merging with other data: {len(df_4hour_step)} rows")

# %%
# Start with hourly step data
df_merged = df_4hour_step.copy()
print(df_merged[150:200])

# add lag columns


# %%
# merge with yesterday step counts
df_merged = df_merged.merge(
    df_today_step,
    on=['ParticipantIdentifier', 'Date'],
    how='left'
)

print(df_merged.head(50))

# %%
# merge with prior 2hours step counts
df_merged = df_merged.merge(
    df_prior2hours_step,
    on=['ParticipantIdentifier', 'Date', 'DecisionTime'],
    how='left'
)

print(df_merged)

# %%
# merge with other PA data
df_merged = df_merged.merge(
    df_otherPA,
    on=['ParticipantIdentifier', 'Date'],
    how='left'
)


# %%
# merge with not wearing data
df_merged = df_merged.merge(
    df_notwearing,
    on=['ParticipantIdentifier', 'Date'],
    how='left'
)


# %%
# merge with wearing data
df_merged = df_merged.merge(
    df_wearing_morning,
    on=['ParticipantIdentifier', 'Date'],
    how='left'
)


# %%
# merge with daily pageview counts
df_merged = df_merged.merge(
    df_daily_pageview,
    on=['ParticipantIdentifier', 'Date'],
    how='left'
)

print(df_merged)

# %%
# merge with hourly pageview counts
df_merged = df_merged.merge(
    df_hourly_pageview,
    on=['ParticipantIdentifier', 'Date', 'DecisionTime'],
    how='left'
)


# %%
# 2) Merge weekly data - use only one date per week to avoid duplicates
df_weekly['week'] = pd.to_datetime(df_weekly['date']).dt.isocalendar().week
df_weekly['year'] = pd.to_datetime(df_weekly['date']).dt.isocalendar().year

# Keep only one row per participant-week-year (the first date of each week)
df_weekly_unique = df_weekly.groupby(
    ['ParticipantIdentifier', 'week', 'year'], as_index=False
).first()

df_merged['week'] = df_merged['Date'].dt.isocalendar().week
df_merged['year'] = df_merged['Date'].dt.isocalendar().year

# print(df_merged[df_merged['ParticipantIdentifier'] == "118"])

weekly_cols = [col for col in df_weekly_unique.columns 
               if col not in ['ParticipantIdentifier', 'date', 'week', 'year', 'observed_date']]

df_merged = df_merged.merge(
    df_weekly_unique[['ParticipantIdentifier', 'week', 'year'] + weekly_cols],
    on=['ParticipantIdentifier', 'week', 'year'],
    how='left'
)
print(f"After weekly merge: {len(df_merged)} rows")

print(df_weekly.week.unique())
print(df_merged.week.unique())
# print(df_merged.iloc[5100:5150, :26])

# %%


# remove duplicate Date columns
df_merged = df_merged.drop(columns=['Date_y'])

# change column names Date_x and Date_y to Date
df_merged = df_merged.rename(columns={'Date_x': 'Date'})

print(df_merged[df_merged['ParticipantIdentifier'] == 141]['week'].unique())



# %%
# Define columns to lag
lag_columns = weekly_cols

# Create lagged columns
for col in lag_columns:
    if col in df_weekly_unique.columns:
        df_weekly_unique[f'{col}_lastweek'] = df_weekly_unique.groupby('ParticipantIdentifier')[col].shift(1)


# Now merge the lagged columns into df_merged
lastweek_cols = [col for col in df_weekly_unique.columns if 'lastweek' in col]

df_merged = df_merged.merge(
    df_weekly_unique[['ParticipantIdentifier', 'week', 'year'] + lastweek_cols],
    on=['ParticipantIdentifier', 'week', 'year'],
    how='left',
    suffixes=('', '_dup')
)

print(df_merged)
# print(df_merged[df_merged['participantidentifier'] == 13].iloc[:50, :26])


# %%
# merge with daily survey data
# df_merged = df_merged.merge(
#     df_daily,
#     on=['ParticipantIdentifier', 'Date'],
#     how='left'
# )

# add lag columns
lag_columns = ['daily_present', 'affective_reflection', 'anticipated_affect']

for col in lag_columns:
    df_daily[f'{col}_yesterday'] = df_daily.groupby('ParticipantIdentifier')[col].shift(1)

df_merged = df_merged.merge(
    df_daily[['ParticipantIdentifier', 'Date'] + lag_columns + [f'{col}_yesterday' for col in lag_columns]],
    on=['ParticipantIdentifier', 'Date'],
    how='left'
)

print(df_merged)

# %%
# merging with action delivery data

# 1) Merge planning prompts- map to all hours of the same date
planning_cols = [col for col in df_planning.columns if col not in ['ParticipantIdentifier', 'Date', 'time']]
# df_merged["ParticipantIdentifier"] = pd.to_numeric(df_merged["ParticipantIdentifier"], errors="coerce").astype("Int64")
df_planning["ParticipantIdentifier"] = pd.to_numeric(df_planning["ParticipantIdentifier"], errors="coerce").astype("Int64")
print(df_planning.ParticipantIdentifier.unique())
df_merged = df_merged.merge(
    df_planning[['ParticipantIdentifier', 'Date'] + planning_cols],
    on=['ParticipantIdentifier', 'Date'],
    how='left'
)
print(df_merged.shape)

# # 2) add yesterday's end of day survey
df_yesterday = df_planning[['ParticipantIdentifier', 'Date'] + planning_cols].copy()
df_yesterday['Date'] = df_yesterday['Date'] + pd.Timedelta(days=1)
df_yesterday.columns = ['ParticipantIdentifier', 'Date'] + [f'yesterday_{col}' for col in planning_cols]

df_merged = df_merged.merge(df_yesterday, on=['ParticipantIdentifier', 'Date'], how='left')

# print(df_merged[df_merged['participantidentifier'] == 31])

# %%
print(df_merged)

# %%
# Merge with walking data
# print(df_walking.head())

# 1. Build an hourly view of the walking data
# df_walking['datetime'] = pd.to_datetime(
#     df_walking['date'].dt.strftime('%Y-%m-%d') + ' ' + df_walking['time']
# )

# walking_hourly = (
#     df_walking
#     .sort_values('datetime')
#     .groupby(['participantidentifier', 'date', 'hour'], as_index=False)
#     .agg({'walking_suggestion': 'max', 'open': 'max'})  # max==1 if any event in that hour
# )

# print(walking_hourly.head())

# 2. Merge into the existing hourly panel
df_merged = df_merged.merge(
    df_gif,
    on=['ParticipantIdentifier', 'Date', 'DecisionTime'],
    how='left'
)

# 3. Fill missing hours with 0 (no walking suggestion delivered in that hour)
# df_merged['walking_suggestion'] = df_merged['walking_suggestion'].fillna(0).astype(int)
# df_merged['open'] = df_merged['open'].fillna(0).astype(int)

print(df_merged.head(60))

# %%
# Merge with salience data
print(df_salience.head())

# 1. Build an hourly view of the salience data
salience_cols = [col for col in df_salience.columns if col not in ['ParticipantIdentifier', 'Date', 'time']]
df_merged = df_merged.merge(
    df_salience[['ParticipantIdentifier', 'Date'] + salience_cols],
    on=['ParticipantIdentifier', 'Date'],
    how='left'
)

# 2. add yesterday's salience data
df_yesterday = df_salience[['ParticipantIdentifier', 'Date'] + salience_cols].copy()
df_yesterday['Date'] = df_yesterday['Date'] + pd.Timedelta(days=1)
df_yesterday.columns = ['ParticipantIdentifier', 'Date'] + [f'yesterday_{col}' for col in salience_cols]

df_merged = df_merged.merge(df_yesterday, on=['ParticipantIdentifier', 'Date'], how='left')

print(df_merged[df_merged['ParticipantIdentifier'] == 118])


# %%
# check variables
print(df_merged.columns)

# %%
# drop columns date_x, date_y, Time_x, Time_y
df_merged = df_merged.drop(columns=['date', 'Time_x', 'Time_y'])

# modify the column names
df_merged = df_merged.rename(columns={'Interacted_x': 'Interacted_walk', 'Interacted_y': 'Interacted_salience', 
                                     'Interacted_7d_x': 'Interacted_7d_walk', 'Interacted_7d_y': 'Interacted_7d_salience',
                                     'StepCount_x': '4hour_step', 'StepCount_y': 'prior2hour_step'})


# %%
# check the number of dates per participant
print(df_merged.groupby('ParticipantIdentifier')['Date'].nunique())

# %%
# Remove the first 7 calendar days of data for each participant
df_merged = df_merged.copy()
df_merged["Date"] = pd.to_datetime(df_merged["Date"], errors="coerce").dt.normalize()

min_date = df_merged.groupby("ParticipantIdentifier")["Date"].transform("min")
cutoff = min_date + pd.Timedelta(days=6)

df_merged = df_merged[df_merged["Date"] >= cutoff].reset_index(drop=True)

# check the number of rows per participant
print(df_merged.groupby('ParticipantIdentifier').size())


# %%
# start from 0 for each participant
# Calendar days since first study date (normalize so two same-day decisions share one `day`)
_d = pd.to_datetime(df_merged['Date']).dt.normalize()
df_merged['day'] = (
    df_merged.assign(_dn=_d)
    .sort_values(['ParticipantIdentifier', 'Date'])
    .groupby('ParticipantIdentifier')['_dn']
    .transform(lambda d: (d - d.min()).dt.days)
)
# print(df_merged.day)



# 1) Day-of-week number: Monday=1 … Sunday=7
df_merged['dow'] = df_merged['Date'].dt.dayofweek + 1

# Study week index aligned to Mon–Sun calendar weeks (increments each Monday).
# If first analysis day is Monday (dow==1): 1-based weeks (1, 2, 3, …).
# Otherwise: 0-based weeks (0, 1, 2, …).
# _wk = (day + start_dow) // 7 with start_dow from pandas (Mon=0 … Sun=6).
df_merged = df_merged.sort_values(['ParticipantIdentifier', 'Date'])
_start_dow = df_merged.groupby('ParticipantIdentifier', sort=False)['Date'].transform('min').dt.dayofweek
_wk = ((df_merged['day'] + _start_dow) // 7).astype(int)
_starts_monday = _start_dow.eq(0)  # matches dow==1 after +1
df_merged['week'] = _wk + _starts_monday.astype(int)
_ex = (
    df_merged[df_merged['ParticipantIdentifier'] == 141][['Date', 'day', 'week', 'dow']]
    .head(172)
    .reset_index(drop=True)
)
print(_ex)
# check week column for each participant
print(df_merged.groupby('ParticipantIdentifier')['week'].unique())


#%%

# 2) Weekend flag (1 = Saturday/Sunday, else 0)
df_merged['is_weekend'] = (df_merged['dow'] >= 6).astype(int)


df_merged = df_merged.sort_values(
    ['ParticipantIdentifier', 'Date', 'DateTimeStart']
)

# df_merged['hour_in_day'] = (
#     df_merged.groupby(['participantidentifier', 'date'])
#              .cumcount() + 1
# )

# creat CAE_avg that averages CAE_1 to CAE_12 
# and CAE_avg_lastweek that averages CAE_1_lastweek to CAE_12_lastweek of the last week
df_merged['CAE_avg'] = df_merged[['CAE-1', 'CAE-2', 'CAE-3', 'CAE-4', 'CAE-5', 'CAE-6', 'CAE-7', 'CAE-8', 
                                 'CAE-9', 'CAE-10', 'CAE-11', 'CAE-12']].mean(axis=1)
df_merged['CAE_avg_lastweek'] = df_merged[['CAE-1_lastweek', 
                                          'CAE-2_lastweek', 'CAE-3_lastweek', 'CAE-4_lastweek', 
                                          'CAE-5_lastweek', 'CAE-6_lastweek', 'CAE-7_lastweek', 
                                          'CAE-8_lastweek', 'CAE-9_lastweek', 'CAE-10_lastweek', 
                                          'CAE-11_lastweek', 'CAE-12_lastweek']].mean(axis=1)
df_merged['CAE_short_avg'] = df_merged[['CAE-3', 'CAE-5', 'CAE-11']].mean(axis=1)

# df_merged['Perceived_utility'] = df_merged[['Exp-tool-1', 'Exp-tool-2']].mean(axis=1)
# df_merged['Perceived_utility_lastweek'] = df_merged[['Exp-tool-1_lastweek', 'Exp-tool-2_lastweek']].mean(axis=1)

# create a new column called Exponentially weighted average of 4 hour step counts over the past 7 days
window = 7  # seven mornings (or afternoons)

df_merged = df_merged.sort_values(["ParticipantIdentifier", "Date", "DecisionTime"])

# recent burden: row-wise mean of walk / xsalience / planning interaction flags, then exponentially
# weighted *within each participant* (rows must be time-ordered). span=14 rows ≈ 7 calendar days
# if there are 2 decision rows per day; use span=7 if you mean 7 decision rows instead.
_burden_cols = ["WalkingSuggestion",  "yesterday_SalienceMessage", "yesterday_planning_prompt"]
df_merged["recent_burden"] = (
    df_merged.assign(_instant_burden=df_merged[_burden_cols].mean(axis=1))
    .groupby("ParticipantIdentifier")["_instant_burden"]
    .transform(lambda s: s.ewm(span=14, adjust=False, min_periods=1).mean())
)


# remove week 0 and week 13
df_merged = df_merged[df_merged['week'] != 0]
df_merged = df_merged[df_merged['week'] != 13]
print(df_merged.groupby('ParticipantIdentifier')['week'].unique())
#check the number of rows per participant
print(df_merged.groupby('ParticipantIdentifier').size())

# %%
# save the merged data
df_merged.to_csv(folder / 'df_merged.csv', index=False)


