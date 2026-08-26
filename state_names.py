"""
Canonical simulation / trajectory / covariate names (camelCase).

Maps legacy snake_case keys to readable names used in ``vani_env`` state dict
``s``, trajectory logs on ``OnlineEnv``, and feature-builder keyword arguments.
"""

# Simulation state ``s`` (generative chain in vani_env / OnlineEnv)
STATE_PRIOR2HOUR_STEP_COUNT = "prior2HourStepCount"
STATE_PRIOR2HOUR_STEP_COUNT_LAG1 = "prior2HourStepCountLag1"
STATE_PRIOR2HOUR_STEP_COUNT_EMA_7D = "prior2HourStepCountEma7d"
STATE_PRIOR2HOUR_STEP_COUNT_AGENT = STATE_PRIOR2HOUR_STEP_COUNT

STATE_STEP_COUNT_FOUR_HOUR_LAG1 = "stepCountNext4HourLag1"
STATE_PAGE_VIEW_FOUR_HOUR_LAG1 = "pageViewNext4HourLag1"

STATE_YESTERDAY_STEP_COUNT = "yesterdayStepCount"
STATE_TODAY_STEP_COUNT = "todayStepCount"
STATE_STEP_COUNT_LAST7_DAYS_EMA = "stepCountLast7DaysEma"
STATE_PAGE_VIEW_LAST7_DAYS_EMA = "pageViewLast7DaysEma"

STATE_ACTIVE_DAYS_LAST7_DAYS = "activeDaysLast7Days"
STATE_ACTIVITY_SUGGESTION_INTERACT_LAST7_DAYS = "activitySuggestionInteractLast7Days"

STATE_MORNING_FITBIT_WEAR = "morningFitbitWear"
STATE_MORNING_FITBIT_WEAR_YESTERDAY = "morningFitbitWearYesterday"
STATE_MORNING_FITBIT_WEAR_LAST7_DAYS = "morningFitbitWearLast7Days"
STATE_MORNING_FITBIT_WEAR_LAST7_DAYS_ALT = "morningFitbitWearLast7DaysAlt"

STATE_DAILY_SURVEY_COMPLETE = "dailySurveyComplete"
STATE_DAILY_SURVEY_COMPLETE_YESTERDAY = "dailySurveyCompleteYesterday"

STATE_DAILY_ANTICIPATED_AFFECT = "dailyAnticipatedAffect"
STATE_DAILY_ANTICIPATED_AFFECT_YESTERDAY = "dailyAnticipatedAffectYesterday"
STATE_DAILY_ANTICIPATED_AFFECT_YESTERDAY_AGENT = "dailyAnticipatedAffectYesterdayAgent"

STATE_ACTIVITY_STATUS_TODAY = "activityStatusToday"

STATE_ACTIVITY_SUGGESTIONS_SENT_LAST7_DAYS = "activitySuggestionsSentLast7Days"
STATE_DAY_OF_WEEK_NORM = "dayOfWeekNorm"
STATE_DAY_OF_WEEK_NORM_LAG1 = "dayOfWeekNormLag1"
STATE_IS_WEEKEND = "isWeekend"
STATE_DECISION_TIME_SLOT = "decisionTimeSlot"

STATE_PERCEIVED_UTILITY_LAST_WEEK = "perceivedUtilityLastWeek"
STATE_PERCEIVED_UTILITY = "perceivedUtility"
STATE_CAE_AVERAGE_LAST_WEEK = "caeAverageLastWeek"
STATE_CAE_AVERAGE = "caeAverage"

STATE_PAGE_VIEW_MORNING_YESTERDAY = "pageViewMorningYesterday"
STATE_PAGE_VIEW_AFTERNOON_YESTERDAY = "pageViewAfternoonYesterday"
STATE_PAGE_VIEW_MORNING_TODAY = "pageViewMorningToday"
STATE_PAGE_VIEW_AFTERNOON_TODAY = "pageViewAfternoonToday"

# Slot- and day-level trajectory logs (OnlineEnv)
TRAJ_STEP_COUNT_FOUR_HOUR = "stepCountNext4HourAll"
TRAJ_STEP_COUNT_FOUR_HOUR_OBS = TRAJ_STEP_COUNT_FOUR_HOUR
TRAJ_STEP_COUNT_FOUR_HOUR_AGENT = TRAJ_STEP_COUNT_FOUR_HOUR
TRAJ_PAGE_VIEW_FOUR_HOUR = "pageViewNext4HourAll"
TRAJ_PRIOR2HOUR_STEP_COUNT = "prior2HourStepCountAll"
TRAJ_PRIOR2HOUR_STEP_COUNT_OBS = TRAJ_PRIOR2HOUR_STEP_COUNT
TRAJ_PRIOR2HOUR_STEP_COUNT_AGENT = TRAJ_PRIOR2HOUR_STEP_COUNT
TRAJ_DAILY_ANTICIPATED_AFFECT = "dailyAnticipatedAffectAll"
TRAJ_DAILY_ANTICIPATED_AFFECT_AGENT = "dailyAnticipatedAffectAgentAll"
TRAJ_DAILY_ANTICIPATED_AFFECT_OBS = "dailyAnticipatedAffectObsAll"
TRAJ_MORNING_FITBIT_WEAR = "morningFitbitWearAll"
TRAJ_DAILY_SURVEY_COMPLETE = "dailySurveyCompleteAll"
TRAJ_ACTIVITY_SUGGESTIONS_SENT_LAST7_DAYS = "activitySuggestionsSentLast7DaysAll"

# Per-day covariate logs (PF / RL; not in ``s`` at use time)
LOG_DAY_OF_WEEK_NORM = "logDayOfWeekNorm"
LOG_YESTERDAY_STEP_COUNT = "logYesterdayStepCount"
LOG_STEP_COUNT_LAST7_DAYS_EMA = "logStepCountLast7DaysEma"
LOG_PAGE_VIEW_LAST7_DAYS_EMA = "logPageViewLast7DaysEma"
LOG_MORNING_FITBIT_WEAR_LAST7_DAYS = "logMorningFitbitWearLast7Days"
LOG_DAILY_ANTICIPATED_AFFECT_YESTERDAY = "logDailyAnticipatedAffectYesterday"
LOG_TODAY_STEP_COUNT = "logTodayStepCount"

# Legacy → canonical (longest keys first for safe replace)
LEGACY_TO_CANONICAL = [
    ("anticipated_affect_yesterday_agent", STATE_DAILY_ANTICIPATED_AFFECT_YESTERDAY_AGENT),
    ("anticipated_affect_yesterday", STATE_DAILY_ANTICIPATED_AFFECT_YESTERDAY),
    ("prior2hour_step_count_lag1", STATE_PRIOR2HOUR_STEP_COUNT_LAG1),
    ("EMA_Prior2HourStepCount_norm", STATE_PRIOR2HOUR_STEP_COUNT_EMA_7D),
    ("EMA_Prior2HourStepCount", STATE_PRIOR2HOUR_STEP_COUNT_EMA_7D),
    ("prior2hour_step_agent_all", TRAJ_PRIOR2HOUR_STEP_COUNT_AGENT),
    ("prior2hour_step_agent", STATE_PRIOR2HOUR_STEP_COUNT_AGENT),
    ("yesterday_fitbitwearing_morning", STATE_MORNING_FITBIT_WEAR_YESTERDAY),
    ("yesterday_pageview_afternoon", STATE_PAGE_VIEW_AFTERNOON_YESTERDAY),
    ("yesterday_pageview_morning", STATE_PAGE_VIEW_MORNING_YESTERDAY),
    ("active_status", STATE_ACTIVITY_STATUS_TODAY),
    ("today_pageview_afternoon", STATE_PAGE_VIEW_AFTERNOON_TODAY),
    ("today_pageview_morning", STATE_PAGE_VIEW_MORNING_TODAY),
    ("perceived_utility_lastweek", STATE_PERCEIVED_UTILITY_LAST_WEEK),
    ("seven_day_step_count_avg", STATE_STEP_COUNT_LAST7_DAYS_EMA),
    ("past7days_morning_wearing", STATE_MORNING_FITBIT_WEAR_LAST7_DAYS),
    ("past7days_daywearing", STATE_MORNING_FITBIT_WEAR_LAST7_DAYS_ALT),
    ("active_status_fraction_7days", STATE_ACTIVE_DAYS_LAST7_DAYS),
    ("activeDaysLast7Days", STATE_ACTIVE_DAYS_LAST7_DAYS),
    ("Interacted_7d_walk", STATE_ACTIVITY_SUGGESTION_INTERACT_LAST7_DAYS),
    ("hourly_pageview_lag1", STATE_PAGE_VIEW_FOUR_HOUR_LAG1),
    ("prior2hour_step_all", TRAJ_PRIOR2HOUR_STEP_COUNT),
    ("prior2hour_obs_all", TRAJ_PRIOR2HOUR_STEP_COUNT_OBS),
    ("fourSC_agent_all", TRAJ_STEP_COUNT_FOUR_HOUR_AGENT),
    ("fourSC_obs_all", TRAJ_STEP_COUNT_FOUR_HOUR_OBS),
    ("fourSC_lag1", STATE_STEP_COUNT_FOUR_HOUR_LAG1),
    ("fourSC_all", TRAJ_STEP_COUNT_FOUR_HOUR),
    ("pageview_all", TRAJ_PAGE_VIEW_FOUR_HOUR),
    ("antic_agent_all", TRAJ_DAILY_ANTICIPATED_AFFECT_AGENT),
    ("antic_obs_all", TRAJ_DAILY_ANTICIPATED_AFFECT_OBS),
    ("antic_all", TRAJ_DAILY_ANTICIPATED_AFFECT),
    ("fitbit_all", TRAJ_MORNING_FITBIT_WEAR),
    ("daily_all", TRAJ_DAILY_SURVEY_COMPLETE),
    ("recent_burden_all", TRAJ_ACTIVITY_SUGGESTIONS_SENT_LAST7_DAYS),
    ("activitySuggestionsSentLast7DaysAll", TRAJ_ACTIVITY_SUGGESTIONS_SENT_LAST7_DAYS),
    ("log_antic_yesterday_latent", LOG_DAILY_ANTICIPATED_AFFECT_YESTERDAY),
    ("log_seven_day_pageview", LOG_PAGE_VIEW_LAST7_DAYS_EMA),
    ("log_yesterday_step", LOG_YESTERDAY_STEP_COUNT),
    ("log_seven_day_step", LOG_STEP_COUNT_LAST7_DAYS_EMA),
    ("log_past7_wear", LOG_MORNING_FITBIT_WEAR_LAST7_DAYS),
    ("log_today_step", LOG_TODAY_STEP_COUNT),
    ("log_dow", LOG_DAY_OF_WEEK_NORM),
    ("prior2hour_step", STATE_PRIOR2HOUR_STEP_COUNT),
    ("yesterday_step", STATE_YESTERDAY_STEP_COUNT),
    ("today_step", STATE_TODAY_STEP_COUNT),
    ("seven_day_pageview", STATE_PAGE_VIEW_LAST7_DAYS_EMA),
    ("yesterday_present", STATE_DAILY_SURVEY_COMPLETE_YESTERDAY),
    ("fitbitwearing_morning", STATE_MORNING_FITBIT_WEAR),
    ("daily_present", STATE_DAILY_SURVEY_COMPLETE),
    ("anticipated_affect", STATE_DAILY_ANTICIPATED_AFFECT),
    ("recent_burden", STATE_ACTIVITY_SUGGESTIONS_SENT_LAST7_DAYS),
    ("activitySuggestionsSentLast7Days", STATE_ACTIVITY_SUGGESTIONS_SENT_LAST7_DAYS),
    ("decision_time", STATE_DECISION_TIME_SLOT),
    ("perceived_utility", STATE_PERCEIVED_UTILITY),
    ("CAE_avg_lastweek", STATE_CAE_AVERAGE_LAST_WEEK),
    ("CAE_avg", STATE_CAE_AVERAGE),
    ("dow_lag1", STATE_DAY_OF_WEEK_NORM_LAG1),
    ("is_weekend", STATE_IS_WEEKEND),
    ("dow", STATE_DAY_OF_WEEK_NORM),
]

# Feature-builder keyword aliases (algorithm.py)
KW_FOURSC_LAG1 = "stepCountNext4HourLag1"
KW_YESTERDAY_STEP = "yesterdayStepCount"
KW_STEP_COUNT_LAST7 = "stepCountLast7DaysEma"
KW_PRIOR2HOUR = "prior2HourStepCount"
KW_PRIOR2HOUR_AGENT = KW_PRIOR2HOUR
KW_ACTIVITY_SUGGESTIONS_SENT_LAST7 = "activitySuggestionsSentLast7Days"
KW_PAGE_VIEW_LAST7 = "pageViewLast7DaysEma"
KW_MORNING_FITBIT_LAST7 = "morningFitbitWearLast7Days"
KW_ACTIVITY_SUGGEST_INTERACT = "activitySuggestionInteractLast7Days"
KW_ANTIC_YESTERDAY = "dailyAnticipatedAffectYesterday"
KW_ANTIC_YESTERDAY_AGENT = "dailyAnticipatedAffectYesterdayAgent"
KW_DAY_OF_WEEK = "dayOfWeekNorm"
KW_DECISION_TIME = "decisionTimeSlot"
KW_PERCEIVED_UTILITY = "perceivedUtility"
KW_CAE_LAST_WEEK = "caeAverageLastWeek"
