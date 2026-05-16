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
import statsmodels.api as sm
from patsy import dmatrix
from pathlib import Path


# %%
# read data — paths do not depend on os.getcwd()
PROJECT_ROOT = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPR-MRT-Testbed")
COMBINED_DIR = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPT_MRT/rawdata/_combined")
WORK_DIR = PROJECT_ROOT / "env_para_vanilla"
WORK_DIR.mkdir(parents=True, exist_ok=True)

df_merged = pd.read_csv(COMBINED_DIR / "df_merged.csv")

print("COMBINED_DIR:", COMBINED_DIR.resolve())
print("WORK_DIR:   ", WORK_DIR.resolve(), "| exists:", WORK_DIR.is_dir())

# Aliases for later cells that use `folder` / `work_folder` (Path so `/` joins work)
folder = COMBINED_DIR
work_folder = Path(WORK_DIR)

# %%
# display all columns
print(df_merged.columns)

# %%
# select columns
df_fit = df_merged[['ParticipantIdentifier', 'Date', 'DecisionTime', 'week', 'day', 'dow', 'is_weekend',
                    'WalkingSuggestion', 'Interacted_walk', 'Interacted_7d_walk',
                    'SalienceMessage', 'Interacted_salience', 'Interacted_7d_salience', 
                    'yesterday_SalienceMessage', 
                    'planning_prompt', 'yesterday_planning_prompt',
                    '4hour_step', 'EMA_StepCount', 
                    'TodayStepCount', 'YesterdayStepCount', 'prior2hour_step', 
                    'RecordedPhysicalActivity','Previous7DaysRPA', 
                    'morning_wearing', 'nextday_wearing', 'past7days_morning_wearing', 
                    'DailyPageviewCount', 'Past7DaysPageviewEMA', 'HourlyPageviewCount', 'Past7DaysHourlyPageviewEMA',
                    'week_present', 'week_present_lastweek', 'daily_present', 'daily_present_yesterday',
                    'affective_reflection', 'affective_reflection_yesterday', 
                    'anticipated_affect', 'anticipated_affect_yesterday',
                    'CAE_avg', 'CAE_avg_lastweek', 'CAE_short_avg', 
                    'recent_burden',
                    'Exp-tool-1', 'Exp-tool-2',
                  ]]

print(df_fit)

# %%
# normalize day and week to be within [-1,1] 
# TODO: based on the expected range of RCT
# TODO: I'll use the MRT range for now, 84 and 12
df_fit = df_fit.copy()
df_fit['day_norm'] = (df_fit['day'] - (1+84)/2) / ((84-1)/2)
df_fit['week_norm'] = (df_fit['week'] - (1+12)/2) / ((12-1)/2)
df_fit['dow_norm'] = (df_fit['dow'] - (1+7)/2) / ((7-1)/2)

print(df_fit)

# remove the first day of data for each participant


# %%
# log transform the step count data
df_fit['4hour_step'] = np.log(df_fit['4hour_step'] + 1)
df_fit['TodayStepCount'] = np.log(df_fit['TodayStepCount'] + 1)
df_fit['YesterdayStepCount'] = np.log(df_fit['YesterdayStepCount'] + 1)
df_fit['prior2hour_step'] = np.log(df_fit['prior2hour_step'] + 1)

# log transform the pageview count data
df_fit['DailyPageviewCount'] = np.log(df_fit['DailyPageviewCount'] + 1)
df_fit['Past7DaysPageviewEMA'] = np.log(df_fit['Past7DaysPageviewEMA'] + 1)
df_fit['HourlyPageviewCount'] = np.log(df_fit['HourlyPageviewCount'] + 1)
df_fit['Past7DaysHourlyPageviewEMA'] = np.log(df_fit['Past7DaysHourlyPageviewEMA'] + 1)

# %%
# standardize the rest of the data
digits = 3
df_fit = df_fit.copy()
step_count_shift = np.round(np.mean(df_fit['4hour_step']), digits)
step_count_scale = np.round(np.std(df_fit['4hour_step']), digits)
df_fit['4hour_step_norm'] = (df_fit['4hour_step'] - step_count_shift) / step_count_scale

today_step_count_shift = np.round(np.mean(df_fit['TodayStepCount']), digits)
today_step_count_scale = np.round(np.std(df_fit['TodayStepCount']), digits)
df_fit['TodayStepCount_norm'] = (df_fit['TodayStepCount'] - today_step_count_shift) / today_step_count_scale

yesterday_step_count_shift = np.round(np.mean(df_fit['YesterdayStepCount']), digits)
yesterday_step_count_scale = np.round(np.std(df_fit['YesterdayStepCount']), digits)
df_fit['YesterdayStepCount_norm'] = (df_fit['YesterdayStepCount'] - yesterday_step_count_shift) / yesterday_step_count_scale

prior2hour_step_count_shift = np.round(np.mean(df_fit['prior2hour_step']), digits)
prior2hour_step_count_scale = np.round(np.std(df_fit['prior2hour_step']), digits)
df_fit['prior2hour_step_norm'] = (df_fit['prior2hour_step'] - prior2hour_step_count_shift) / prior2hour_step_count_scale

EMA_StepCount_shift = np.round(np.mean(df_fit['EMA_StepCount']), digits)
EMA_StepCount_scale = np.round(np.std(df_fit['EMA_StepCount']), digits)
df_fit['EMA_StepCount_norm'] = (df_fit['EMA_StepCount'] - EMA_StepCount_shift) / EMA_StepCount_scale

DailyPageviewCount_shift = np.round(np.mean(df_fit['DailyPageviewCount']), digits)
DailyPageviewCount_scale = np.round(np.std(df_fit['DailyPageviewCount']), digits)
df_fit['DailyPageviewCount_norm'] = (df_fit['DailyPageviewCount'] - DailyPageviewCount_shift) / DailyPageviewCount_scale

Past7DaysPageviewEMA_shift = np.round(np.mean(df_fit['Past7DaysPageviewEMA']), digits)
Past7DaysPageviewEMA_scale = np.round(np.std(df_fit['Past7DaysPageviewEMA']), digits)
df_fit['Past7DaysPageviewEMA_norm'] = (df_fit['Past7DaysPageviewEMA'] - Past7DaysPageviewEMA_shift) / Past7DaysPageviewEMA_scale

HourlyPageviewCount_shift = np.round(np.mean(df_fit['HourlyPageviewCount']), digits)
HourlyPageviewCount_scale = np.round(np.std(df_fit['HourlyPageviewCount']), digits)
df_fit['HourlyPageviewCount_norm'] = (df_fit['HourlyPageviewCount'] - HourlyPageviewCount_shift) / HourlyPageviewCount_scale

Past7DaysHourlyPageviewEMA_shift = np.round(np.mean(df_fit['Past7DaysHourlyPageviewEMA']), digits)
Past7DaysHourlyPageviewEMA_scale = np.round(np.std(df_fit['Past7DaysHourlyPageviewEMA']), digits)
df_fit['Past7DaysHourlyPageviewEMA_norm'] = (df_fit['Past7DaysHourlyPageviewEMA'] - Past7DaysHourlyPageviewEMA_shift) / Past7DaysHourlyPageviewEMA_scale


df_fit['affective_reflection_norm'] = (df_fit['affective_reflection'] + 1) / 8

df_fit['anticipated_affect_norm'] = (df_fit['anticipated_affect'] + 1) / 8

df_fit['affective_reflection_yesterday_norm'] = (df_fit['affective_reflection_yesterday'] + 1) / 8

df_fit['anticipated_affect_yesterday_norm'] = (df_fit['anticipated_affect_yesterday'] + 1) / 8

CAE_avg_shift = np.round(np.mean(df_fit['CAE_avg']), digits)
CAE_avg_scale = np.round(np.std(df_fit['CAE_avg']), digits)
df_fit['CAE_avg_norm'] = (df_fit['CAE_avg'] - CAE_avg_shift) / CAE_avg_scale

CAE_avg_lastweek_shift = np.round(np.mean(df_fit['CAE_avg_lastweek']), digits)
CAE_avg_lastweek_scale = np.round(np.std(df_fit['CAE_avg_lastweek']), digits)
df_fit['CAE_avg_lastweek_norm'] = (df_fit['CAE_avg_lastweek'] - CAE_avg_lastweek_shift) / CAE_avg_lastweek_scale

CAE_short_avg_shift = np.round(np.mean(df_fit['CAE_short_avg']), digits)
CAE_short_avg_scale = np.round(np.std(df_fit['CAE_short_avg']), digits)
df_fit['CAE_short_avg_norm'] = (df_fit['CAE_short_avg'] - CAE_short_avg_shift) / CAE_short_avg_scale


recent_burden_shift = np.round(np.mean(df_fit['recent_burden']), digits)
recent_burden_scale = np.round(np.std(df_fit['recent_burden']), digits)
df_fit['recent_burden_norm'] = (df_fit['recent_burden'] - recent_burden_shift) / recent_burden_scale

df_fit['Exp-tool-1_norm'] = (df_fit['Exp-tool-1'] + 1) / 8
df_fit['Exp-tool-2_norm'] = (df_fit['Exp-tool-2'] + 1) / 8

# perceived utility need to be normalized later because we will tune it (what is true vs what we use in the algorithm)


# check the range of the normalized data
hour4_step_count_limit = [np.round(np.min(df_fit['4hour_step_norm']), digits), np.round(np.max(df_fit['4hour_step_norm']), digits)]
today_step_count_limit = [np.round(np.min(df_fit['TodayStepCount_norm']), digits), np.round(np.max(df_fit['TodayStepCount_norm']), digits)]
yesterday_step_count_limit = [np.round(np.min(df_fit['YesterdayStepCount_norm']), digits), np.round(np.max(df_fit['YesterdayStepCount_norm']), digits)]
prior2hour_step_count_limit = [np.round(np.min(df_fit['prior2hour_step_norm']), digits), np.round(np.max(df_fit['prior2hour_step_norm']), digits)]
EMA_step_count_limit = [np.round(np.min(df_fit['EMA_StepCount_norm']), digits), np.round(np.max(df_fit['EMA_StepCount_norm']), digits)]
daily_pageview_count_limit = [np.round(np.min(df_fit['DailyPageviewCount_norm']), digits), np.round(np.max(df_fit['DailyPageviewCount_norm']), digits)]
past7days_pageview_count_limit = [np.round(np.min(df_fit['Past7DaysPageviewEMA_norm']), digits), np.round(np.max(df_fit['Past7DaysPageviewEMA_norm']), digits)]
hourly_pageview_count_limit = [np.round(np.min(df_fit['HourlyPageviewCount_norm']), digits), np.round(np.max(df_fit['HourlyPageviewCount_norm']), digits)]
past7days_hourly_pageview_count_limit = [np.round(np.min(df_fit['Past7DaysHourlyPageviewEMA_norm']), digits), np.round(np.max(df_fit['Past7DaysHourlyPageviewEMA_norm']), digits)]
affective_reflection_limit = [np.round(np.min(df_fit['affective_reflection_norm']), digits), np.round(np.max(df_fit['affective_reflection_norm']), digits)]
anticipated_affect_limit = [np.round(np.min(df_fit['anticipated_affect_norm']), digits), np.round(np.max(df_fit['anticipated_affect_norm']), digits)]
affective_reflection_yesterday_limit = [np.round(np.min(df_fit['affective_reflection_yesterday_norm']), digits), np.round(np.max(df_fit['affective_reflection_yesterday_norm']), digits)]
anticipated_affect_yesterday_limit = [np.round(np.min(df_fit['anticipated_affect_yesterday_norm']), digits), np.round(np.max(df_fit['anticipated_affect_yesterday_norm']), digits)]
CAE_avg_limit = [np.round(np.min(df_fit['CAE_avg_norm']), digits), np.round(np.max(df_fit['CAE_avg_norm']), digits)]
CAE_avg_lastweek_limit = [np.round(np.min(df_fit['CAE_avg_lastweek_norm']), digits), np.round(np.max(df_fit['CAE_avg_lastweek_norm']), digits)]
CAE_short_avg_limit = [np.round(np.min(df_fit['CAE_short_avg_norm']), digits), np.round(np.max(df_fit['CAE_short_avg_norm']), digits)]
recent_burden_limit = [np.round(np.min(df_fit['recent_burden_norm']), digits), np.round(np.max(df_fit['recent_burden_norm']), digits)]
exp1_limit = [np.round(np.min(df_fit['Exp-tool-1_norm']), digits), np.round(np.max(df_fit['Exp-tool-1_norm']), digits)]
exp2_limit = [np.round(np.min(df_fit['Exp-tool-2_norm']), digits), np.round(np.max(df_fit['Exp-tool-2_norm']), digits)]

# save the shift and scales into a json file
std_params = {
    '4hour_step_count_shift': step_count_shift,
    '4hour_step_count_scale': step_count_scale,
    'today_step_count_shift': today_step_count_shift,
    'today_step_count_scale': today_step_count_scale,
    'yesterday_step_count_shift': yesterday_step_count_shift,
    'yesterday_step_count_scale': yesterday_step_count_scale,
    'prior2hour_step_count_shift': prior2hour_step_count_shift,
    'prior2hour_step_count_scale': prior2hour_step_count_scale,
    'EMA_step_count_shift': EMA_StepCount_shift,
    'EMA_step_count_scale': EMA_StepCount_scale,
    'DailyPageviewCount_shift': DailyPageviewCount_shift,
    'DailyPageviewCount_scale': DailyPageviewCount_scale,
    'Past7DaysPageviewEMA_shift': Past7DaysPageviewEMA_shift,
    'Past7DaysPageviewEMA_scale': Past7DaysPageviewEMA_scale,
    'HourlyPageviewCount_shift': HourlyPageviewCount_shift,
    'HourlyPageviewCount_scale': HourlyPageviewCount_scale,
    'Past7DaysHourlyPageviewEMA_shift': Past7DaysHourlyPageviewEMA_shift,
    'Past7DaysHourlyPageviewEMA_scale': Past7DaysHourlyPageviewEMA_scale,
    'CAE_avg_shift': CAE_avg_shift,
    'CAE_avg_scale': CAE_avg_scale,
    'CAE_avg_lastweek_shift': CAE_avg_lastweek_shift,
    'CAE_avg_lastweek_scale': CAE_avg_lastweek_scale,
    'CAE_short_avg_shift': CAE_short_avg_shift,
    'CAE_short_avg_scale': CAE_short_avg_scale,
    'recent_burden_shift': recent_burden_shift,
    'recent_burden_scale': recent_burden_scale,
    '4hour_step_count_limit': hour4_step_count_limit,   
    'TodayStepCount_limit': today_step_count_limit,
    'YesterdayStepCount_limit': yesterday_step_count_limit,
    'prior2hour_step_count_limit': prior2hour_step_count_limit,
    'EMA_StepCount_limit': EMA_step_count_limit,
    'DailyPageviewCount_limit': daily_pageview_count_limit,
    'Past7DaysPageviewEMA_limit': past7days_pageview_count_limit,
    'HourlyPageviewCount_limit': hourly_pageview_count_limit,
    'affective_reflection_limit': affective_reflection_limit,
    'anticipated_affect_limit': anticipated_affect_limit,
    'affective_reflection_yesterday_limit': affective_reflection_yesterday_limit,
    'anticipated_affect_yesterday_limit': anticipated_affect_yesterday_limit,
    'CAE_avg_limit': CAE_avg_limit,
    'CAE_avg_lastweek_limit': CAE_avg_lastweek_limit,
    'CAE_short_avg_limit': CAE_short_avg_limit,
    'recent_burden_limit': recent_burden_limit,
    'exp1_limit': exp1_limit,
    'exp2_limit': exp2_limit,
}

_work_dir = Path(work_folder)
output_path = _work_dir / "std_params.json"
os.makedirs(_work_dir, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(std_params, f)





# %%
# only save the normalized data
# remove the unnormalized data
# df_fit = df_fit.drop(columns=['day', 'week',
#                               'step_count', 'yesterday_step_count', 'prior30_step_count',
#                               'yesterday_pageview_count', 
#                               'AA_avg', 'perceived_utility',
#                               'affective_valuation',
#                               'perceived_utility_lastweek', 'AA_avg_lastweek',
#                               '7day_step_count_avg', '7day_step_count_std',
#                               'week_step_count_avg', 'week_walking_suggestion', 'week_view_status'])

# Shifts are row-based: expect two rows per calendar day (DecisionTime 0 then 1), chronological.
df_fit = df_fit.sort_values(
    ['ParticipantIdentifier', 'Date', 'DecisionTime'],
    kind='mergesort',
).reset_index(drop=True)

# shift(1) = previous decision slot; shift(2) = one calendar day back if day-level vars repeat on AM/PM.
df_fit['FourSC_lag1'] = df_fit.groupby('ParticipantIdentifier', sort=False)['4hour_step_norm'].shift(1)
df_fit['prior2hour_step_count_lag1'] = df_fit.groupby('ParticipantIdentifier', sort=False)['prior2hour_step_norm'].shift(1)
df_fit['hourly_pageview_count_lag1'] = df_fit.groupby('ParticipantIdentifier', sort=False)['HourlyPageviewCount_norm'].shift(1)
df_fit['recorded_physical_activity_lag1'] = df_fit.groupby('ParticipantIdentifier', sort=False)['RecordedPhysicalActivity'].shift(2)


df_fit.to_csv(folder / 'df_fit.csv', index=False)


# %%
# print(df_fit.loc[df_fit['ParticipantIdentifier'] == 219, '4hour_step_norm'])


