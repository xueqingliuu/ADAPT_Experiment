# Prior Mean and Variance Summary

`prior_variance` is the diagonal entry of the prior covariance. For `q_td_modify_joint`, the full covariance remains in `rl_priors.json`; this table reports marginal variances.

## PF Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| PF | fourSC | theta | 0 | intercept | -0.064155 | 0.051124 |
| PF | fourSC | theta | 1 | stepCountNext4HourLag1 | 0.485640 | 0.044035 |
| PF | fourSC | theta | 2 | yesterdayStepCount | -0.148401 | 0.011163 |
| PF | fourSC | theta | 3 | stepCountLast7DaysEma | 0.358050 | 0.034993 |
| PF | fourSC | theta | 4 | prior2HourStepCount | 0.097754 | 0.034164 |
| PF | fourSC | theta | 5 | activitySuggestionsSentLast7Days | -0.037946 | 0.026999 |
| PF | fourSC | theta | 6 | activitySuggestionInteractLast7Days | 0.037768 | 0.061428 |
| PF | fourSC | theta | 7 | activeDaysLast7Days | 0.158754 | 0.043973 |
| PF | fourSC | theta | 8 | isWeekend | -0.034373 | 0.046135 |
| PF | fourSC | theta | 9 | decisionTimeSlot | -0.066423 | 0.123913 |
| PF | fourSC | theta | 10 | perceivedUtilityLastWeek | 0.021186 | 0.014038 |
| PF | fourSC | theta | 11 | caeAverageLastWeek | -0.053808 | 0.097074 |
| PF | fourSC | theta | 12 | Ah | -0.142819 | 0.077520 |
| PF | fourSC | theta | 13 | Ah*yesterdayStepCount | 0.005743 | 0.019237 |
| PF | fourSC | theta | 14 | Ah*prior2HourStepCount | 0.010735 | 0.024257 |
| PF | fourSC | theta | 15 | Ah*activitySuggestionsSentLast7Days | 0.030233 | 0.024848 |
| PF | fourSC | theta | 16 | Ah*activitySuggestionInteractLast7Days | 0.083931 | 0.098780 |
| PF | fourSC | theta | 17 | Ah*perceivedUtilityLastWeek | 0.020380 | 0.021050 |
| PF | fourSC | theta | 18 | Ah*caeAverageLastWeek | -0.013623 | 0.076366 |
| PF | fourSC | theta | 19 | Ah*decisionTimeSlot | 0.064549 | 0.021185 |
| PF | antic | theta | 0 | intercept | 0.340290 | 0.012202 |
| PF | antic | theta | 1 | anticipated_affect_yesterday | 0.216039 | 0.005106 |
| PF | antic | theta | 2 | active_status_fraction_7days | 0.087343 | 0.008163 |
| PF | antic | theta | 3 | is_weekend | -0.027256 | 0.004637 |
| PF | antic | theta | 4 | perceived_utility_lastweek | 0.012722 | 0.007367 |
| PF | antic | theta | 5 | CAE_avg_lastweek | 0.206524 | 0.007625 |
| PF | antic | theta | 6 | recent_burden | 0.014275 | 0.004193 |
| PF | antic | theta | 7 | A0_morning | 0.011321 | 0.002879 |
| PF | antic | theta | 8 | A1_afternoon | 0.023782 | 0.001518 |
| PF | antic | theta | 9 | A0_morning_by_perceived_utility_lastweek | -0.004974 | 0.001292 |
| PF | antic | theta | 10 | A1_afternoon_by_perceived_utility_lastweek | -0.002954 | 0.000588 |
| PF | antic | theta | 11 | A0_morning_by_CAE_avg_lastweek | -0.000253 | 0.005307 |
| PF | antic | theta | 12 | A1_afternoon_by_CAE_avg_lastweek | -0.024172 | 0.002645 |
| PF | antic | theta | 13 | A0_morning_by_recent_burden | -0.008892 | 0.001489 |
| PF | antic | theta | 14 | A1_afternoon_by_recent_burden | -0.003619 | 0.005200 |
| PF | CAE | theta | 0 | intercept | -0.097854 | 0.110520 |
| PF | CAE | theta | 1 | CAE_avg_lastweek | 0.888991 | 0.059786 |
| PF | CAE | theta | 2 | fourSC_ewma | -0.039333 | 0.021144 |
| PF | CAE | theta | 3 | anticipated_affect_ewma | 0.194762 | 0.024286 |
| PF | CAE_short | theta | 0 | intercept | 0.008297 | 0.067244 |
| PF | CAE_short | theta | 1 | caeAverage | 0.962049 | 0.060039 |

## RL Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| RL | reward_shaping | eta | 0 | intercept | -0.035989 | 0.030184 |
| RL | reward_shaping | eta | 1 | weekday_vs_weekend | -0.005998 | 0.000838 |
| RL | reward_shaping | eta | 2 | slot_pm | -0.017994 | 0.007546 |
| RL | reward_shaping | eta | 3 | E_w | 0.067562 | 0.003160 |
| RL | reward_shaping | eta | 4 | b_hat | 0.067837 | 0.030268 |
| RL | reward_shaping | eta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | reward_shaping | eta | 6 | M_Y_anticipated_affect_ewma | 0.078880 | 0.010501 |
| RL | reward_shaping | eta | 7 | M_Y_fourSC_ewma | 0.012711 | 0.021856 |
| RL | reward_shaping | eta | 8 | M_E_pageview_ewma | -0.005174 | 0.004223 |
| RL | reward_shaping | eta | 9 | M_E_fitbit_wear_ewma | -0.007982 | 0.005312 |
| RL | reward_shaping | eta | 10 | M_E_survey_complete_ewma | -0.000370 | 0.011361 |
| RL | reward_shaping | eta | 11 | yesterday_step_count | 0.000689 | 0.013522 |
| RL | reward_shaping | eta | 12 | prior2hour_step_count | -0.021887 | 0.005554 |
| RL | reward_shaping | eta | 13 | active_status_fraction_7days | 0.006800 | 0.034555 |
| RL | reward_shaping | eta | 14 | recent_burden | -0.000133 | 0.003091 |
| RL | reward_shaping | eta | 15 | walk_interaction_7d | 0.016573 | 0.037141 |
| RL | redistribution_stage1_AA | eta | 0 | intercept | 0.222969 | 0.007320 |
| RL | redistribution_stage1_AA | eta | 1 | E_w | 0.001377 | 0.001126 |
| RL | redistribution_stage1_AA | eta | 2 | b_hat | 0.084304 | 0.002502 |
| RL | redistribution_stage1_AA | eta | 3 | b_tilde | 0.000000 | 1.000000 |
| RL | redistribution_stage1_AA | eta | 4 | M_Y_anticipated_affect_ewma | 0.174362 | 0.001935 |
| RL | redistribution_stage1_AA | eta | 5 | M_Y_fourSC_ewma | -0.011469 | 0.002822 |
| RL | redistribution_stage1_AA | eta | 6 | M_E_pageview_ewma | 0.013249 | 0.000549 |
| RL | redistribution_stage1_AA | eta | 7 | M_E_fitbit_wear_ewma | -0.033483 | 0.001424 |
| RL | redistribution_stage1_AA | eta | 8 | M_E_survey_complete_ewma | -0.027182 | 0.001595 |
| RL | redistribution_stage1_AA | eta | 9 | yesterday_step_count | 0.007297 | 0.000216 |
| RL | redistribution_stage1_AA | eta | 10 | prior2hour_step_count | 0.010348 | 0.000478 |
| RL | redistribution_stage1_AA | eta | 11 | active_status_fraction_7days | 0.025958 | 0.004648 |
| RL | redistribution_stage1_AA | eta | 12 | recent_burden | 0.000336 | 0.000118 |
| RL | redistribution_stage1_AA | eta | 13 | walk_interaction_7d | -0.003652 | 0.002932 |
| RL | redistribution_stage1_AA | eta | 14 | A | 0.004928 | 0.001389 |
| RL | redistribution_stage1_AA | eta | 15 | A*E_w | 4.743482e-05 | 0.000420 |
| RL | redistribution_stage1_AA | eta | 16 | A*b_hat | -0.007451 | 0.002134 |
| RL | redistribution_stage1_AA | eta | 17 | A*b_tilde | 0.000000 | 1.000000 |
| RL | redistribution_stage1_AA | eta | 18 | A*yesterday_step_count | -0.000406 | 0.000582 |
| RL | redistribution_stage1_AA | eta | 19 | A*prior2hour_step_count | 0.008620 | 0.000482 |
| RL | redistribution_stage1_AA | eta | 20 | A*active_status_fraction_7days | -0.008784 | 0.002604 |
| RL | redistribution_stage1_AA | eta | 21 | A*recent_burden | 0.003156 | 0.000297 |
| RL | redistribution_stage1_AA | eta | 22 | A*walk_interaction_7d | -0.000755 | 0.000842 |
| RL | redistribution_stage1_AA | eta | 23 | A*slot_pm | 0.012964 | 0.001697 |
| RL | redistribution_stage1_FW | eta | 0 | intercept | 0.363061 | 0.024519 |
| RL | redistribution_stage1_FW | eta | 1 | E_w | 0.006563 | 0.005689 |
| RL | redistribution_stage1_FW | eta | 2 | b_hat | 0.001032 | 0.008942 |
| RL | redistribution_stage1_FW | eta | 3 | b_tilde | 0.000000 | 1.000000 |
| RL | redistribution_stage1_FW | eta | 4 | M_Y_anticipated_affect_ewma | -0.123811 | 0.027769 |
| RL | redistribution_stage1_FW | eta | 5 | M_Y_fourSC_ewma | -0.029052 | 0.011063 |
| RL | redistribution_stage1_FW | eta | 6 | M_E_pageview_ewma | 0.028266 | 0.001493 |
| RL | redistribution_stage1_FW | eta | 7 | M_E_fitbit_wear_ewma | 0.160578 | 0.018363 |
| RL | redistribution_stage1_FW | eta | 8 | M_E_survey_complete_ewma | -0.012227 | 0.009850 |
| RL | redistribution_stage1_FW | eta | 9 | yesterday_step_count | 0.004136 | 0.003544 |
| RL | redistribution_stage1_FW | eta | 10 | prior2hour_step_count | 0.005224 | 0.002698 |
| RL | redistribution_stage1_FW | eta | 11 | active_status_fraction_7days | 0.044195 | 0.015556 |
| RL | redistribution_stage1_FW | eta | 12 | recent_burden | -0.010694 | 0.003197 |
| RL | redistribution_stage1_FW | eta | 13 | walk_interaction_7d | 0.028904 | 0.025930 |
| RL | redistribution_stage1_FW | eta | 14 | A | 0.034705 | 0.010074 |
| RL | redistribution_stage1_FW | eta | 15 | A*E_w | -0.002168 | 0.006426 |
| RL | redistribution_stage1_FW | eta | 16 | A*b_hat | -0.005751 | 0.014031 |
| RL | redistribution_stage1_FW | eta | 17 | A*b_tilde | 0.000000 | 1.000000 |
| RL | redistribution_stage1_FW | eta | 18 | A*yesterday_step_count | -0.000833 | 0.007621 |
| RL | redistribution_stage1_FW | eta | 19 | A*prior2hour_step_count | -0.003484 | 0.005169 |
| RL | redistribution_stage1_FW | eta | 20 | A*active_status_fraction_7days | -0.023458 | 0.024079 |
| RL | redistribution_stage1_FW | eta | 21 | A*recent_burden | 0.006203 | 0.004120 |
| RL | redistribution_stage1_FW | eta | 22 | A*walk_interaction_7d | -0.012811 | 0.023791 |
| RL | redistribution_stage1_FW | eta | 23 | A*slot_pm | -0.031203 | 0.012399 |
| RL | redistribution_stage1_PJ | eta | 0 | intercept | 0.103925 | 0.035849 |
| RL | redistribution_stage1_PJ | eta | 1 | E_w | 0.004667 | 0.008531 |
| RL | redistribution_stage1_PJ | eta | 2 | b_hat | 0.020412 | 0.019403 |
| RL | redistribution_stage1_PJ | eta | 3 | b_tilde | 0.000000 | 1.000000 |
| RL | redistribution_stage1_PJ | eta | 4 | M_Y_anticipated_affect_ewma | -0.140528 | 0.030398 |
| RL | redistribution_stage1_PJ | eta | 5 | M_Y_fourSC_ewma | -0.016472 | 0.022762 |
| RL | redistribution_stage1_PJ | eta | 6 | M_E_pageview_ewma | 0.012888 | 0.002279 |
| RL | redistribution_stage1_PJ | eta | 7 | M_E_fitbit_wear_ewma | 0.015800 | 0.007100 |
| RL | redistribution_stage1_PJ | eta | 8 | M_E_survey_complete_ewma | 0.159951 | 0.011044 |
| RL | redistribution_stage1_PJ | eta | 9 | yesterday_step_count | 0.001069 | 0.007391 |
| RL | redistribution_stage1_PJ | eta | 10 | prior2hour_step_count | 0.024372 | 0.002801 |
| RL | redistribution_stage1_PJ | eta | 11 | active_status_fraction_7days | 0.073571 | 0.034628 |
| RL | redistribution_stage1_PJ | eta | 12 | recent_burden | 0.014304 | 0.005100 |
| RL | redistribution_stage1_PJ | eta | 13 | walk_interaction_7d | 0.188163 | 0.023156 |
| RL | redistribution_stage1_PJ | eta | 14 | A | 0.051847 | 0.012479 |
| RL | redistribution_stage1_PJ | eta | 15 | A*E_w | 0.005603 | 0.009282 |
| RL | redistribution_stage1_PJ | eta | 16 | A*b_hat | 0.008547 | 0.023838 |
| RL | redistribution_stage1_PJ | eta | 17 | A*b_tilde | 0.000000 | 1.000000 |
| RL | redistribution_stage1_PJ | eta | 18 | A*yesterday_step_count | 0.012052 | 0.013079 |
| RL | redistribution_stage1_PJ | eta | 19 | A*prior2hour_step_count | -0.003130 | 0.012208 |
| RL | redistribution_stage1_PJ | eta | 20 | A*active_status_fraction_7days | -0.058701 | 0.032340 |
| RL | redistribution_stage1_PJ | eta | 21 | A*recent_burden | -0.006634 | 0.010734 |
| RL | redistribution_stage1_PJ | eta | 22 | A*walk_interaction_7d | -0.049878 | 0.019150 |
| RL | redistribution_stage1_PJ | eta | 23 | A*slot_pm | -0.020927 | 0.018829 |
| RL | redistribution_stage1_SC | eta | 0 | intercept | -0.705132 | 0.351709 |
| RL | redistribution_stage1_SC | eta | 1 | weekday_vs_weekend | -0.007309 | 0.025683 |
| RL | redistribution_stage1_SC | eta | 2 | slot_pm | -0.153867 | 0.090372 |
| RL | redistribution_stage1_SC | eta | 3 | E_w | 0.144278 | 0.018131 |
| RL | redistribution_stage1_SC | eta | 4 | b_hat | -0.232069 | 0.070862 |
| RL | redistribution_stage1_SC | eta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | redistribution_stage1_SC | eta | 6 | yesterday_step_count | 0.057615 | 0.003956 |
| RL | redistribution_stage1_SC | eta | 7 | prior2hour_step_count | 0.302017 | 0.020929 |
| RL | redistribution_stage1_SC | eta | 8 | active_status_fraction_7days | 0.879395 | 0.042327 |
| RL | redistribution_stage1_SC | eta | 9 | recent_burden | -0.063057 | 0.002461 |
| RL | redistribution_stage1_SC | eta | 10 | walk_interaction_7d | 0.100917 | 0.055888 |
| RL | redistribution_stage1_SC | eta | 11 | A | -0.144828 | 0.015284 |
| RL | redistribution_stage1_SC | eta | 12 | A*slot_pm | 0.124689 | 0.023721 |
| RL | redistribution_stage1_PV | eta | 0 | intercept | -2.789009 | 0.681921 |
| RL | redistribution_stage1_PV | eta | 1 | weekday_vs_weekend | -0.165289 | 0.172679 |
| RL | redistribution_stage1_PV | eta | 2 | slot_pm | -0.160459 | 0.143064 |
| RL | redistribution_stage1_PV | eta | 3 | E_w | 0.046370 | 0.193018 |
| RL | redistribution_stage1_PV | eta | 4 | b_hat | 0.001551 | 0.203445 |
| RL | redistribution_stage1_PV | eta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | redistribution_stage1_PV | eta | 6 | yesterday_step_count | -0.071943 | 0.027039 |
| RL | redistribution_stage1_PV | eta | 7 | prior2hour_step_count | 0.097826 | 0.029345 |
| RL | redistribution_stage1_PV | eta | 8 | active_status_fraction_7days | 0.089289 | 0.541693 |
| RL | redistribution_stage1_PV | eta | 9 | recent_burden | -0.046387 | 0.035630 |
| RL | redistribution_stage1_PV | eta | 10 | walk_interaction_7d | 0.991033 | 0.222607 |
| RL | redistribution_stage1_PV | eta | 11 | A | 0.532358 | 0.497144 |
| RL | redistribution_stage1_PV | eta | 12 | A*slot_pm | 0.116056 | 0.117668 |
| RL | redistribution_stage2_v2 | eta | 0 | redistribution_stage2_v2_coef_0 | -0.746000 | 0.001348 |
| RL | redistribution_stage2_v2 | eta | 1 | redistribution_stage2_v2_coef_1 | -0.124333 | 3.744789e-05 |
| RL | redistribution_stage2_v2 | eta | 2 | redistribution_stage2_v2_coef_2 | 0.052255 | 0.000941 |
| RL | redistribution_stage2_v2 | eta | 3 | redistribution_stage2_v2_coef_3 | -0.104549 | 0.001634 |
| RL | redistribution_stage2_v2 | eta | 4 | redistribution_stage2_v2_coef_4 | 0.000000 | 1.000000 |
| RL | redistribution_stage2_v2 | eta | 5 | redistribution_stage2_v2_coef_5 | 0.716815 | 0.003279 |
| RL | redistribution_stage2_v2 | eta | 6 | redistribution_stage2_v2_coef_6 | 0.022342 | 0.003028 |
| RL | redistribution_stage2_v2 | eta | 7 | redistribution_stage2_v2_coef_7 | -0.089218 | 0.002292 |
| RL | redistribution_stage2_v2 | eta | 8 | redistribution_stage2_v2_coef_8 | -0.164891 | 0.001886 |
| RL | redistribution_stage2_v2 | eta | 9 | redistribution_stage2_v2_coef_9 | -0.159278 | 0.002538 |
| RL | redistribution_stage2_v2 | eta | 10 | redistribution_stage2_v2_coef_10 | 0.043484 | 0.002711 |
| RL | redistribution_stage2_v2 | eta | 11 | redistribution_stage2_v2_coef_11 | -0.019601 | 0.001276 |
| RL | redistribution_stage2_v2 | eta | 12 | redistribution_stage2_v2_coef_12 | 0.066967 | 0.001260 |
| RL | redistribution_stage2_v2 | eta | 13 | redistribution_stage2_v2_coef_13 | 0.000734 | 0.000365 |
| RL | redistribution_stage2_v2 | eta | 14 | redistribution_stage2_v2_coef_14 | -0.047275 | 0.004041 |
| RL | redistribution_stage2_v2 | eta | 15 | redistribution_stage2_v2_coef_15 | 0.044416 | 0.002375 |
| RL | redistribution_stage2_v2 | eta | 16 | redistribution_stage2_v2_coef_16 | 0.127519 | 0.000201 |
| RL | redistribution_stage2_v2 | eta | 17 | redistribution_stage2_v2_coef_17 | 0.709298 | 0.000540 |
| RL | redistribution_stage2_v2 | eta | 18 | redistribution_stage2_v2_coef_18 | 0.996091 | 0.000328 |
| RL | redistribution_stage2_v2 | eta | 19 | redistribution_stage2_v2_coef_19 | -0.141645 | 0.000347 |
| RL | redistribution_stage2_v4 | eta | 0 | redistribution_stage2_v4_coef_0 | 0.100663 | 0.006338 |
| RL | redistribution_stage2_v4 | eta | 1 | redistribution_stage2_v4_coef_1 | 0.016777 | 0.000176 |
| RL | redistribution_stage2_v4 | eta | 2 | redistribution_stage2_v4_coef_2 | -0.117818 | 0.002298 |
| RL | redistribution_stage2_v4 | eta | 3 | redistribution_stage2_v4_coef_3 | 0.148301 | 0.013714 |
| RL | redistribution_stage2_v4 | eta | 4 | redistribution_stage2_v4_coef_4 | 0.000000 | 1.000000 |
| RL | redistribution_stage2_v4 | eta | 5 | redistribution_stage2_v4_coef_5 | 0.310320 | 0.008028 |
| RL | redistribution_stage2_v4 | eta | 6 | redistribution_stage2_v4_coef_6 | 0.052028 | 0.017660 |
| RL | redistribution_stage2_v4 | eta | 7 | redistribution_stage2_v4_coef_7 | -0.051810 | 0.005886 |
| RL | redistribution_stage2_v4 | eta | 8 | redistribution_stage2_v4_coef_8 | -0.069528 | 0.005247 |
| RL | redistribution_stage2_v4 | eta | 9 | redistribution_stage2_v4_coef_9 | -0.253482 | 0.008457 |
| RL | redistribution_stage2_v4 | eta | 10 | redistribution_stage2_v4_coef_10 | -0.045762 | 0.015378 |
| RL | redistribution_stage2_v4 | eta | 11 | redistribution_stage2_v4_coef_11 | -0.253583 | 0.004759 |
| RL | redistribution_stage2_v4 | eta | 12 | redistribution_stage2_v4_coef_12 | -0.612637 | 0.005273 |
| RL | redistribution_stage2_v4 | eta | 13 | redistribution_stage2_v4_coef_13 | 0.033090 | 0.002823 |
| RL | redistribution_stage2_v4 | eta | 14 | redistribution_stage2_v4_coef_14 | -0.326770 | 0.016208 |
| RL | redistribution_stage2_v4 | eta | 15 | redistribution_stage2_v4_coef_15 | 0.020750 | 0.011054 |
| RL | redistribution_stage2_v4 | eta | 16 | redistribution_stage2_v4_coef_16 | 0.254772 | 0.000921 |
| RL | redistribution_stage2_v4 | eta | 17 | redistribution_stage2_v4_coef_17 | 0.289924 | 0.001400 |
| RL | redistribution_stage2_v4 | eta | 18 | redistribution_stage2_v4_coef_18 | 1.636309 | 0.000957 |
| RL | redistribution_stage2_v4 | eta | 19 | redistribution_stage2_v4_coef_19 | 0.608859 | 0.002845 |
| RL | q_no_td_modify | beta | 0 | intercept | 0.113675 | 0.084997 |
| RL | q_no_td_modify | beta | 1 | weekday_vs_weekend | -0.071093 | 0.102456 |
| RL | q_no_td_modify | beta | 2 | slot_pm | -0.008039 | 0.002440 |
| RL | q_no_td_modify | beta | 3 | E_w | 0.020835 | 0.039664 |
| RL | q_no_td_modify | beta | 4 | b_hat | 1.149582 | 0.037594 |
| RL | q_no_td_modify | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify | beta | 6 | M_Y_anticipated_affect_ewma | 0.291744 | 0.010067 |
| RL | q_no_td_modify | beta | 7 | M_Y_fourSC_ewma | -0.024196 | 0.006828 |
| RL | q_no_td_modify | beta | 8 | M_E_pageview_ewma | 0.001350 | 0.000228 |
| RL | q_no_td_modify | beta | 9 | M_E_fitbit_wear_ewma | -0.132628 | 0.004399 |
| RL | q_no_td_modify | beta | 10 | M_E_survey_complete_ewma | -0.040520 | 0.004034 |
| RL | q_no_td_modify | beta | 11 | yesterday_step_count | -0.005486 | 0.001592 |
| RL | q_no_td_modify | beta | 12 | prior2hour_step_count | -0.009700 | 0.000751 |
| RL | q_no_td_modify | beta | 13 | active_status_fraction_7days | 0.006992 | 0.024485 |
| RL | q_no_td_modify | beta | 14 | recent_burden | -0.011190 | 0.002337 |
| RL | q_no_td_modify | beta | 15 | walk_interaction_7d | -0.065415 | 0.026574 |
| RL | q_no_td_modify | beta | 16 | A | 0.002052 | 0.001255 |
| RL | q_no_td_modify | beta | 17 | A*E_w | 0.002054 | 0.001100 |
| RL | q_no_td_modify | beta | 18 | A*b_hat | 0.000230 | 0.002239 |
| RL | q_no_td_modify | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify | beta | 20 | A*yesterday_step_count | -0.003479 | 0.003409 |
| RL | q_no_td_modify | beta | 21 | A*prior2hour_step_count | 0.006049 | 0.001273 |
| RL | q_no_td_modify | beta | 22 | A*active_status_fraction_7days | -0.012574 | 0.001346 |
| RL | q_no_td_modify | beta | 23 | A*recent_burden | -0.003053 | 0.000984 |
| RL | q_no_td_modify | beta | 24 | A*walk_interaction_7d | -0.004083 | 0.002206 |
| RL | q_no_td_modify | beta | 25 | A*weekday_vs_weekend | 0.040212 | 0.004830 |
| RL | q_no_td_modify | beta | 26 | A*slot_pm | 0.008674 | 0.001847 |
| RL | q_no_td_modify_g09 | beta | 0 | intercept | 0.138555 | 0.149838 |
| RL | q_no_td_modify_g09 | beta | 1 | weekday_vs_weekend | -0.065512 | 0.249483 |
| RL | q_no_td_modify_g09 | beta | 2 | slot_pm | -0.006419 | 0.008029 |
| RL | q_no_td_modify_g09 | beta | 3 | E_w | 0.038245 | 0.067596 |
| RL | q_no_td_modify_g09 | beta | 4 | b_hat | 1.551511 | 0.060741 |
| RL | q_no_td_modify_g09 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g09 | beta | 6 | M_Y_anticipated_affect_ewma | 0.480382 | 0.023600 |
| RL | q_no_td_modify_g09 | beta | 7 | M_Y_fourSC_ewma | -0.034393 | 0.011422 |
| RL | q_no_td_modify_g09 | beta | 8 | M_E_pageview_ewma | 0.004011 | 0.000435 |
| RL | q_no_td_modify_g09 | beta | 9 | M_E_fitbit_wear_ewma | -0.210532 | 0.007868 |
| RL | q_no_td_modify_g09 | beta | 10 | M_E_survey_complete_ewma | -0.058451 | 0.007966 |
| RL | q_no_td_modify_g09 | beta | 11 | yesterday_step_count | -0.001844 | 0.003350 |
| RL | q_no_td_modify_g09 | beta | 12 | prior2hour_step_count | -0.015708 | 0.001207 |
| RL | q_no_td_modify_g09 | beta | 13 | active_status_fraction_7days | 0.041455 | 0.052210 |
| RL | q_no_td_modify_g09 | beta | 14 | recent_burden | -0.016466 | 0.005391 |
| RL | q_no_td_modify_g09 | beta | 15 | walk_interaction_7d | -0.083348 | 0.046839 |
| RL | q_no_td_modify_g09 | beta | 16 | A | 0.002667 | 0.004088 |
| RL | q_no_td_modify_g09 | beta | 17 | A*E_w | 0.002832 | 0.003146 |
| RL | q_no_td_modify_g09 | beta | 18 | A*b_hat | 0.000121 | 0.004670 |
| RL | q_no_td_modify_g09 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g09 | beta | 20 | A*yesterday_step_count | -0.006318 | 0.005818 |
| RL | q_no_td_modify_g09 | beta | 21 | A*prior2hour_step_count | 0.009695 | 0.002041 |
| RL | q_no_td_modify_g09 | beta | 22 | A*active_status_fraction_7days | -0.022035 | 0.004096 |
| RL | q_no_td_modify_g09 | beta | 23 | A*recent_burden | -0.004925 | 0.002056 |
| RL | q_no_td_modify_g09 | beta | 24 | A*walk_interaction_7d | 0.000545 | 0.004489 |
| RL | q_no_td_modify_g09 | beta | 25 | A*weekday_vs_weekend | 0.053499 | 0.007139 |
| RL | q_no_td_modify_g09 | beta | 26 | A*slot_pm | 0.013438 | 0.005035 |
| RL | q_no_td_modify_g099 | beta | 0 | intercept | 0.142570 | 0.172997 |
| RL | q_no_td_modify_g099 | beta | 1 | weekday_vs_weekend | -0.061689 | 0.309976 |
| RL | q_no_td_modify_g099 | beta | 2 | slot_pm | -0.005581 | 0.010556 |
| RL | q_no_td_modify_g099 | beta | 3 | E_w | 0.043923 | 0.076374 |
| RL | q_no_td_modify_g099 | beta | 4 | b_hat | 1.665509 | 0.068730 |
| RL | q_no_td_modify_g099 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g099 | beta | 6 | M_Y_anticipated_affect_ewma | 0.538293 | 0.030058 |
| RL | q_no_td_modify_g099 | beta | 7 | M_Y_fourSC_ewma | -0.037351 | 0.012877 |
| RL | q_no_td_modify_g099 | beta | 8 | M_E_pageview_ewma | 0.004953 | 0.000535 |
| RL | q_no_td_modify_g099 | beta | 9 | M_E_fitbit_wear_ewma | -0.234054 | 0.009951 |
| RL | q_no_td_modify_g099 | beta | 10 | M_E_survey_complete_ewma | -0.063766 | 0.009850 |
| RL | q_no_td_modify_g099 | beta | 11 | yesterday_step_count | -0.000454 | 0.004075 |
| RL | q_no_td_modify_g099 | beta | 12 | prior2hour_step_count | -0.017609 | 0.001386 |
| RL | q_no_td_modify_g099 | beta | 13 | active_status_fraction_7days | 0.053356 | 0.063837 |
| RL | q_no_td_modify_g099 | beta | 14 | recent_burden | -0.018053 | 0.006858 |
| RL | q_no_td_modify_g099 | beta | 15 | walk_interaction_7d | -0.087896 | 0.054523 |
| RL | q_no_td_modify_g099 | beta | 16 | A | 0.002877 | 0.005504 |
| RL | q_no_td_modify_g099 | beta | 17 | A*E_w | 0.003057 | 0.004185 |
| RL | q_no_td_modify_g099 | beta | 18 | A*b_hat | 2.566757e-05 | 0.005718 |
| RL | q_no_td_modify_g099 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g099 | beta | 20 | A*yesterday_step_count | -0.007239 | 0.006611 |
| RL | q_no_td_modify_g099 | beta | 21 | A*prior2hour_step_count | 0.010833 | 0.002359 |
| RL | q_no_td_modify_g099 | beta | 22 | A*active_status_fraction_7days | -0.025016 | 0.005621 |
| RL | q_no_td_modify_g099 | beta | 23 | A*recent_burden | -0.005566 | 0.002475 |
| RL | q_no_td_modify_g099 | beta | 24 | A*walk_interaction_7d | 0.002200 | 0.005356 |
| RL | q_no_td_modify_g099 | beta | 25 | A*weekday_vs_weekend | 0.056981 | 0.007991 |
| RL | q_no_td_modify_g099 | beta | 26 | A*slot_pm | 0.014820 | 0.006780 |
| RL | q_redistribution_v2 | beta | 0 | intercept | 0.393847 | 0.274909 |
| RL | q_redistribution_v2 | beta | 1 | weekday_vs_weekend | -0.293985 | 0.099593 |
| RL | q_redistribution_v2 | beta | 2 | slot_pm | -0.094329 | 0.012712 |
| RL | q_redistribution_v2 | beta | 3 | E_w | 0.837382 | 0.703841 |
| RL | q_redistribution_v2 | beta | 4 | b_hat | 1.119051 | 0.087346 |
| RL | q_redistribution_v2 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v2 | beta | 6 | M_Y_anticipated_affect_ewma | 0.703490 | 0.037229 |
| RL | q_redistribution_v2 | beta | 7 | M_Y_fourSC_ewma | -0.073244 | 0.034858 |
| RL | q_redistribution_v2 | beta | 8 | M_E_pageview_ewma | -0.009390 | 0.007606 |
| RL | q_redistribution_v2 | beta | 9 | M_E_fitbit_wear_ewma | -0.355365 | 0.021440 |
| RL | q_redistribution_v2 | beta | 10 | M_E_survey_complete_ewma | -0.024785 | 0.025393 |
| RL | q_redistribution_v2 | beta | 11 | yesterday_step_count | 0.065013 | 0.010131 |
| RL | q_redistribution_v2 | beta | 12 | prior2hour_step_count | 0.035191 | 0.004009 |
| RL | q_redistribution_v2 | beta | 13 | active_status_fraction_7days | 0.698258 | 0.205946 |
| RL | q_redistribution_v2 | beta | 14 | recent_burden | 0.132252 | 0.015293 |
| RL | q_redistribution_v2 | beta | 15 | walk_interaction_7d | 1.176970 | 0.116917 |
| RL | q_redistribution_v2 | beta | 16 | A | 0.014239 | 0.010261 |
| RL | q_redistribution_v2 | beta | 17 | A*E_w | 0.005523 | 0.001818 |
| RL | q_redistribution_v2 | beta | 18 | A*b_hat | -0.025965 | 0.003480 |
| RL | q_redistribution_v2 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v2 | beta | 20 | A*yesterday_step_count | 0.032249 | 0.001640 |
| RL | q_redistribution_v2 | beta | 21 | A*prior2hour_step_count | 0.010646 | 0.003268 |
| RL | q_redistribution_v2 | beta | 22 | A*active_status_fraction_7days | 0.006078 | 0.009083 |
| RL | q_redistribution_v2 | beta | 23 | A*recent_burden | 0.028866 | 0.001378 |
| RL | q_redistribution_v2 | beta | 24 | A*walk_interaction_7d | -0.031602 | 0.012141 |
| RL | q_redistribution_v2 | beta | 25 | A*weekday_vs_weekend | -0.020054 | 0.005258 |
| RL | q_redistribution_v2 | beta | 26 | A*slot_pm | 0.110712 | 0.010328 |
| RL | q_redistribution_v4 | beta | 0 | intercept | 0.225510 | 0.537248 |
| RL | q_redistribution_v4 | beta | 1 | weekday_vs_weekend | -0.113020 | 0.047482 |
| RL | q_redistribution_v4 | beta | 2 | slot_pm | -0.025272 | 0.014591 |
| RL | q_redistribution_v4 | beta | 3 | E_w | -0.274086 | 0.156554 |
| RL | q_redistribution_v4 | beta | 4 | b_hat | 1.695745 | 0.209762 |
| RL | q_redistribution_v4 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v4 | beta | 6 | M_Y_anticipated_affect_ewma | 0.298098 | 0.048099 |
| RL | q_redistribution_v4 | beta | 7 | M_Y_fourSC_ewma | 0.015017 | 0.065484 |
| RL | q_redistribution_v4 | beta | 8 | M_E_pageview_ewma | -0.001824 | 0.020659 |
| RL | q_redistribution_v4 | beta | 9 | M_E_fitbit_wear_ewma | -0.136178 | 0.018604 |
| RL | q_redistribution_v4 | beta | 10 | M_E_survey_complete_ewma | 0.015157 | 0.036720 |
| RL | q_redistribution_v4 | beta | 11 | yesterday_step_count | -0.051534 | 0.042463 |
| RL | q_redistribution_v4 | beta | 12 | prior2hour_step_count | 0.011497 | 0.007922 |
| RL | q_redistribution_v4 | beta | 13 | active_status_fraction_7days | 0.381110 | 0.322484 |
| RL | q_redistribution_v4 | beta | 14 | recent_burden | 0.013454 | 0.092297 |
| RL | q_redistribution_v4 | beta | 15 | walk_interaction_7d | 0.305166 | 0.476499 |
| RL | q_redistribution_v4 | beta | 16 | A | 0.074267 | 0.015245 |
| RL | q_redistribution_v4 | beta | 17 | A*E_w | -0.004824 | 0.005973 |
| RL | q_redistribution_v4 | beta | 18 | A*b_hat | 0.004548 | 0.024421 |
| RL | q_redistribution_v4 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v4 | beta | 20 | A*yesterday_step_count | 0.041675 | 0.005573 |
| RL | q_redistribution_v4 | beta | 21 | A*prior2hour_step_count | 0.021925 | 0.007248 |
| RL | q_redistribution_v4 | beta | 22 | A*active_status_fraction_7days | -0.104843 | 0.015332 |
| RL | q_redistribution_v4 | beta | 23 | A*recent_burden | -0.004650 | 0.002337 |
| RL | q_redistribution_v4 | beta | 24 | A*walk_interaction_7d | -0.096529 | 0.044130 |
| RL | q_redistribution_v4 | beta | 25 | A*weekday_vs_weekend | 0.064317 | 0.010790 |
| RL | q_redistribution_v4 | beta | 26 | A*slot_pm | 0.038850 | 0.030459 |
| RL | q_residual_g09 | beta | 0 | intercept | 0.143200 | 0.034386 |
| RL | q_residual_g09 | beta | 1 | weekday_vs_weekend | -0.024504 | 0.003929 |
| RL | q_residual_g09 | beta | 2 | slot_pm | 0.006656 | 0.000355 |
| RL | q_residual_g09 | beta | 3 | E_w | -0.002540 | 0.017946 |
| RL | q_residual_g09 | beta | 4 | b_hat | -0.081496 | 0.088876 |
| RL | q_residual_g09 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_residual_g09 | beta | 6 | M_Y_anticipated_affect_ewma | 0.077410 | 0.009838 |
| RL | q_residual_g09 | beta | 7 | M_Y_fourSC_ewma | -0.013060 | 0.003326 |
| RL | q_residual_g09 | beta | 8 | M_E_pageview_ewma | 0.000928 | 0.000140 |
| RL | q_residual_g09 | beta | 9 | M_E_fitbit_wear_ewma | -0.012875 | 0.009253 |
| RL | q_residual_g09 | beta | 10 | M_E_survey_complete_ewma | -0.039118 | 0.003913 |
| RL | q_residual_g09 | beta | 11 | yesterday_step_count | -0.017469 | 0.000365 |
| RL | q_residual_g09 | beta | 12 | prior2hour_step_count | -0.005104 | 0.000245 |
| RL | q_residual_g09 | beta | 13 | active_status_fraction_7days | 0.069813 | 0.016219 |
| RL | q_residual_g09 | beta | 14 | recent_burden | -0.038079 | 0.007193 |
| RL | q_residual_g09 | beta | 15 | walk_interaction_7d | 0.028455 | 0.017337 |
| RL | q_residual_g09 | beta | 16 | A | -0.003463 | 0.002078 |
| RL | q_residual_g09 | beta | 17 | A*E_w | 0.001418 | 0.000438 |
| RL | q_residual_g09 | beta | 18 | A*b_hat | -0.000146 | 0.002736 |
| RL | q_residual_g09 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_residual_g09 | beta | 20 | A*yesterday_step_count | 0.003477 | 0.000543 |
| RL | q_residual_g09 | beta | 21 | A*prior2hour_step_count | 0.000870 | 0.000311 |
| RL | q_residual_g09 | beta | 22 | A*active_status_fraction_7days | 0.000975 | 0.002200 |
| RL | q_residual_g09 | beta | 23 | A*recent_burden | -0.004584 | 0.000415 |
| RL | q_residual_g09 | beta | 24 | A*walk_interaction_7d | -0.009756 | 0.003933 |
| RL | q_residual_g09 | beta | 25 | A*weekday_vs_weekend | 0.041147 | 0.011890 |
| RL | q_residual_g09 | beta | 26 | A*slot_pm | -0.010617 | 0.002479 |
| RL | q_adv_m_g09 | beta | 0 | intercept | 0.161478 | 0.147886 |
| RL | q_adv_m_g09 | beta | 1 | weekday_vs_weekend | -0.069805 | 0.254945 |
| RL | q_adv_m_g09 | beta | 2 | slot_pm | -0.006616 | 0.007829 |
| RL | q_adv_m_g09 | beta | 3 | E_w | 0.037028 | 0.071764 |
| RL | q_adv_m_g09 | beta | 4 | b_hat | 1.560549 | 0.068137 |
| RL | q_adv_m_g09 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_adv_m_g09 | beta | 6 | M_Y_anticipated_affect_ewma | 0.471878 | 0.020238 |
| RL | q_adv_m_g09 | beta | 7 | M_Y_fourSC_ewma | -0.035731 | 0.013773 |
| RL | q_adv_m_g09 | beta | 8 | M_E_pageview_ewma | 0.004950 | 0.000956 |
| RL | q_adv_m_g09 | beta | 9 | M_E_fitbit_wear_ewma | -0.209975 | 0.010651 |
| RL | q_adv_m_g09 | beta | 10 | M_E_survey_complete_ewma | -0.039774 | 0.008506 |
| RL | q_adv_m_g09 | beta | 11 | yesterday_step_count | -0.000542 | 0.002837 |
| RL | q_adv_m_g09 | beta | 12 | prior2hour_step_count | -0.015368 | 0.000734 |
| RL | q_adv_m_g09 | beta | 13 | active_status_fraction_7days | 0.068560 | 0.049885 |
| RL | q_adv_m_g09 | beta | 14 | recent_burden | -0.015774 | 0.005330 |
| RL | q_adv_m_g09 | beta | 15 | walk_interaction_7d | -0.092735 | 0.059576 |
| RL | q_adv_m_g09 | beta | 16 | A | 0.006199 | 0.004165 |
| RL | q_adv_m_g09 | beta | 17 | A*E_w | 0.003991 | 0.004247 |
| RL | q_adv_m_g09 | beta | 18 | A*b_hat | -0.007169 | 0.003778 |
| RL | q_adv_m_g09 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_adv_m_g09 | beta | 20 | A*yesterday_step_count | -0.005667 | 0.005836 |
| RL | q_adv_m_g09 | beta | 21 | A*prior2hour_step_count | 0.009663 | 0.001038 |
| RL | q_adv_m_g09 | beta | 22 | A*active_status_fraction_7days | -0.023573 | 0.004286 |
| RL | q_adv_m_g09 | beta | 23 | A*recent_burden | -0.004755 | 0.001851 |
| RL | q_adv_m_g09 | beta | 24 | A*walk_interaction_7d | 0.020064 | 0.006511 |
| RL | q_adv_m_g09 | beta | 25 | A*weekday_vs_weekend | 0.054744 | 0.005309 |
| RL | q_adv_m_g09 | beta | 26 | A*slot_pm | 0.013297 | 0.005865 |
| RL | q_adv_m_g09 | beta | 27 | A*M_Y_anticipated_affect_ewma | 0.036784 | 0.005475 |
| RL | q_adv_m_g09 | beta | 28 | A*M_Y_fourSC_ewma | 0.002040 | 0.003886 |
| RL | q_adv_m_g09 | beta | 29 | A*M_E_pageview_ewma | -0.000762 | 0.001495 |
| RL | q_adv_m_g09 | beta | 30 | A*M_E_fitbit_wear_ewma | -0.021481 | 0.002335 |
| RL | q_adv_m_g09 | beta | 31 | A*M_E_survey_complete_ewma | -0.037157 | 0.004437 |

## Joint Modified-TD Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| Joint Modified TD | q_td_modify_joint | eta | 0 | eta_intercept | 0.625875 | 0.172404 |
| Joint Modified TD | q_td_modify_joint | eta | 1 | eta_E_w | -0.008020 | 0.076765 |
| Joint Modified TD | q_td_modify_joint | eta | 2 | eta_b_hat | -0.502976 | 0.077097 |
| Joint Modified TD | q_td_modify_joint | eta | 3 | eta_b_tilde | 0.000000 | 1.000001 |
| Joint Modified TD | q_td_modify_joint | beta | 4 | beta_intercept | 0.572090 | 0.117165 |
| Joint Modified TD | q_td_modify_joint | beta | 5 | beta_weekday_vs_weekend | -0.091517 | 0.211256 |
| Joint Modified TD | q_td_modify_joint | beta | 6 | beta_slot_pm | 0.094518 | 0.004545 |
| Joint Modified TD | q_td_modify_joint | beta | 7 | beta_E_w | -0.001418 | 0.058372 |
| Joint Modified TD | q_td_modify_joint | beta | 8 | beta_b_hat | -0.040826 | 0.041384 |
| Joint Modified TD | q_td_modify_joint | beta | 9 | beta_b_tilde | 0.000000 | 1.000001 |
| Joint Modified TD | q_td_modify_joint | beta | 10 | beta_M_Y_anticipated_affect_ewma | 0.711917 | 0.023900 |
| Joint Modified TD | q_td_modify_joint | beta | 11 | beta_M_Y_fourSC_ewma | -0.007089 | 0.010320 |
| Joint Modified TD | q_td_modify_joint | beta | 12 | beta_M_E_pageview_ewma | 0.008955 | 0.000962 |
| Joint Modified TD | q_td_modify_joint | beta | 13 | beta_M_E_fitbit_wear_ewma | -0.342879 | 0.006177 |
| Joint Modified TD | q_td_modify_joint | beta | 14 | beta_M_E_survey_complete_ewma | -0.043786 | 0.007098 |
| Joint Modified TD | q_td_modify_joint | beta | 15 | beta_yesterday_step_count | 0.005910 | 0.002107 |
| Joint Modified TD | q_td_modify_joint | beta | 16 | beta_prior2hour_step_count | -0.007861 | 0.000795 |
| Joint Modified TD | q_td_modify_joint | beta | 17 | beta_active_status_fraction_7days | -0.016537 | 0.029912 |
| Joint Modified TD | q_td_modify_joint | beta | 18 | beta_recent_burden | -0.004574 | 0.001894 |
| Joint Modified TD | q_td_modify_joint | beta | 19 | beta_walk_interaction_7d | 0.004353 | 0.010320 |
| Joint Modified TD | q_td_modify_joint | beta | 20 | beta_A | 0.010737 | 0.002805 |
| Joint Modified TD | q_td_modify_joint | beta | 21 | beta_A*E_w | -0.000827 | 0.002089 |
| Joint Modified TD | q_td_modify_joint | beta | 22 | beta_A*b_hat | -0.070734 | 0.003057 |
| Joint Modified TD | q_td_modify_joint | beta | 23 | beta_A*b_tilde | 0.000000 | 1.000001 |
| Joint Modified TD | q_td_modify_joint | beta | 24 | beta_A*yesterday_step_count | -0.004032 | 0.004002 |
| Joint Modified TD | q_td_modify_joint | beta | 25 | beta_A*prior2hour_step_count | 0.001257 | 0.001611 |
| Joint Modified TD | q_td_modify_joint | beta | 26 | beta_A*active_status_fraction_7days | -0.001813 | 0.002714 |
| Joint Modified TD | q_td_modify_joint | beta | 27 | beta_A*recent_burden | -0.002120 | 0.002203 |
| Joint Modified TD | q_td_modify_joint | beta | 28 | beta_A*walk_interaction_7d | 0.012966 | 0.002735 |
| Joint Modified TD | q_td_modify_joint | beta | 29 | beta_A*weekday_vs_weekend | -0.002051 | 0.006244 |
| Joint Modified TD | q_td_modify_joint | beta | 30 | beta_A*slot_pm | -0.009897 | 0.003284 |

## Pooled PF Regression Coefficients

Coefficients are from the all-user stacked ridge fits used as PF prior means. `p_value` uses ridge sandwich standard errors and is approximate because ridge shrinks coefficients. `identified=False` marks zero-variance design columns; SE / p-values are omitted for those rows.

| model | outcome | index | feature | coefficient | identified | feature_std | std_error | t_stat | p_value | significant_0.05 | significant_0.01 | n_obs | n_features | ridge_alpha | residual_sigma2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fourSC | 4hour_step_norm | 0 | intercept | -0.064155 | False | 0.000000 |  |  |  | False | False | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 1 | stepCountNext4HourLag1 | 0.485640 | True | 0.915568 | 0.017115 | 28.375627 | 2.446067e-156 | True | True | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 2 | yesterdayStepCount | -0.148401 | True | 0.930518 | 0.021697 | -6.839575 | 9.668360e-12 | True | True | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 3 | stepCountLast7DaysEma | 0.358050 | True | 0.954106 | 0.016239 | 22.049005 | 1.169953e-99 | True | True | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 4 | prior2HourStepCount | 0.097754 | True | 0.935469 | 0.020107 | 4.861600 | 1.227090e-06 | True | True | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 5 | activitySuggestionsSentLast7Days | -0.037946 | True | 0.968302 | 0.019069 | -1.989881 | 0.046699 | True | False | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 6 | activitySuggestionInteractLast7Days | 0.037768 | True | 0.353473 | 0.052805 | 0.715233 | 0.474523 | False | False | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 7 | activeDaysLast7Days | 0.158754 | True | 0.260938 | 0.054972 | 2.887887 | 0.003907 | True | True | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 8 | isWeekend | -0.034373 | True | 0.369659 | 0.034733 | -0.989662 | 0.322423 | False | False | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 9 | decisionTimeSlot | -0.066423 | True | 0.497078 | 0.037195 | -1.785782 | 0.074240 | False | False | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 10 | perceivedUtilityLastWeek | 0.021186 | True | 2.361545 | 0.008803 | 2.406709 | 0.016160 | True | False | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 11 | caeAverageLastWeek | -0.053808 | True | 0.947049 | 0.021086 | -2.551842 | 0.010767 | True | False | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 12 | Ah | -0.142819 | True | 0.499925 | 0.048229 | -2.961273 | 0.003089 | True | True | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 13 | Ah*yesterdayStepCount | 0.005743 | True | 0.661722 | 0.028867 | 0.198928 | 0.842333 | False | False | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 14 | Ah*prior2HourStepCount | 0.010735 | True | 0.655759 | 0.028235 | 0.380205 | 0.703821 | False | False | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 15 | Ah*activitySuggestionsSentLast7Days | 0.030233 | True | 0.684165 | 0.026836 | 1.126565 | 0.260021 | False | False | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 16 | Ah*activitySuggestionInteractLast7Days | 0.083931 | True | 0.343672 | 0.074824 | 1.121716 | 0.262077 | False | False | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 17 | Ah*perceivedUtilityLastWeek | 0.020380 | True | 1.784542 | 0.012033 | 1.693638 | 0.090443 | False | False | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 18 | Ah*caeAverageLastWeek | -0.013623 | True | 0.644148 | 0.029020 | -0.469430 | 0.638798 | False | False | 2890 | 20 | 1.000000 | 0.475782 |
| fourSC | 4hour_step_norm | 19 | Ah*decisionTimeSlot | 0.064549 | True | 0.413354 | 0.051488 | 1.253674 | 0.210063 | False | False | 2890 | 20 | 1.000000 | 0.475782 |
| antic | anticipated_affect_norm | 0 | intercept | 0.340290 | False | 0.000000 |  |  |  | False | False | 817 | 15 | 1.000000 | 0.040464 |
| antic | anticipated_affect_norm | 1 | anticipated_affect_yesterday | 0.216039 | True | 0.329197 | 0.024954 | 8.657563 | 2.621496e-17 | True | True | 817 | 15 | 1.000000 | 0.040464 |
| antic | anticipated_affect_norm | 2 | active_status_fraction_7days | 0.087343 | True | 0.224466 | 0.032858 | 2.658208 | 0.008012 | True | True | 817 | 15 | 1.000000 | 0.040464 |
| antic | anticipated_affect_norm | 3 | is_weekend | -0.027256 | True | 0.370289 | 0.019152 | -1.423165 | 0.155077 | False | False | 817 | 15 | 1.000000 | 0.040464 |
| antic | anticipated_affect_norm | 4 | perceived_utility_lastweek | 0.012722 | True | 2.244579 | 0.005444 | 2.336767 | 0.019697 | True | False | 817 | 15 | 1.000000 | 0.040464 |
| antic | anticipated_affect_norm | 5 | CAE_avg_lastweek | 0.206524 | True | 0.895117 | 0.015024 | 13.746591 | 9.144558e-39 | True | True | 817 | 15 | 1.000000 | 0.040464 |
| antic | anticipated_affect_norm | 6 | recent_burden | 0.014275 | True | 0.956169 | 0.010820 | 1.319339 | 0.187432 | False | False | 817 | 15 | 1.000000 | 0.040464 |
| antic | anticipated_affect_norm | 7 | A0_morning | 0.011321 | True | 0.499865 | 0.017534 | 0.645633 | 0.518702 | False | False | 817 | 15 | 1.000000 | 0.040464 |
| antic | anticipated_affect_norm | 8 | A1_afternoon | 0.023782 | True | 0.499937 | 0.017373 | 1.368892 | 0.171416 | False | False | 817 | 15 | 1.000000 | 0.040464 |
| antic | anticipated_affect_norm | 9 | A0_morning_by_perceived_utility_lastweek | -0.004974 | True | 1.765888 | 0.006817 | -0.729681 | 0.465799 | False | False | 817 | 15 | 1.000000 | 0.040464 |
| antic | anticipated_affect_norm | 10 | A1_afternoon_by_perceived_utility_lastweek | -0.002954 | True | 1.731512 | 0.006769 | -0.436332 | 0.662714 | False | False | 817 | 15 | 1.000000 | 0.040464 |
| antic | anticipated_affect_norm | 11 | A0_morning_by_CAE_avg_lastweek | -0.000253 | True | 0.641238 | 0.016738 | -0.015124 | 0.987937 | False | False | 817 | 15 | 1.000000 | 0.040464 |
| antic | anticipated_affect_norm | 12 | A1_afternoon_by_CAE_avg_lastweek | -0.024172 | True | 0.615386 | 0.016663 | -1.450621 | 0.147276 | False | False | 817 | 15 | 1.000000 | 0.040464 |
| antic | anticipated_affect_norm | 13 | A0_morning_by_recent_burden | -0.008892 | True | 0.646184 | 0.016519 | -0.538277 | 0.590535 | False | False | 817 | 15 | 1.000000 | 0.040464 |
| antic | anticipated_affect_norm | 14 | A1_afternoon_by_recent_burden | -0.003619 | True | 0.628143 | 0.016552 | -0.218668 | 0.826964 | False | False | 817 | 15 | 1.000000 | 0.040464 |
| CAE | CAE_avg_norm | 0 | intercept | -0.097854 | False | 0.000000 |  |  |  | False | False | 213 | 4 | 1.000000 | 0.166376 |
| CAE | CAE_avg_norm | 1 | CAE_avg_lastweek | 0.888991 | True | 0.979802 | 0.038303 | 23.209550 | 9.194734e-60 | True | True | 213 | 4 | 1.000000 | 0.166376 |
| CAE | CAE_avg_norm | 2 | fourSC_ewma | -0.039333 | True | 0.522369 | 0.053322 | -0.737652 | 0.461553 | False | False | 213 | 4 | 1.000000 | 0.166376 |
| CAE | CAE_avg_norm | 3 | anticipated_affect_ewma | 0.194762 | True | 0.332321 | 0.106835 | 1.823008 | 0.069731 | False | False | 213 | 4 | 1.000000 | 0.166376 |

## Pooled PF GEE Coefficients

Population-averaged GEE fits on the same stacked PF designs, clustered by `ParticipantIdentifier`. `p_value` uses GEE sandwich standard errors with exchangeable working correlation. `identified=False` marks zero-variance design columns; SE / p-values are omitted for those rows. These are for inference/audit only; PF priors still use ridge.

| coefficient | robust_se | z_stat | p_value | significant_0.05 | significant_0.01 | model | outcome | index | feature | identified | feature_std | n_obs | n_clusters | n_features | working_correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -0.091531 |  |  |  | False | False | fourSC | 4hour_step_norm | 0 | intercept | False | 0.000000 | 2890 | 28 | 20 | exchangeable |
| -0.038615 | 0.040912 | -0.943849 | 0.345247 | False | False | fourSC | 4hour_step_norm | 1 | stepCountNext4HourLag1 | True | 0.915568 | 2890 | 28 | 20 | exchangeable |
| -0.016611 | 0.029781 | -0.557773 | 0.576999 | False | False | fourSC | 4hour_step_norm | 2 | yesterdayStepCount | True | 0.930518 | 2890 | 28 | 20 | exchangeable |
| 0.185851 | 0.034825 | 5.336776 | 9.461398e-08 | True | True | fourSC | 4hour_step_norm | 3 | stepCountLast7DaysEma | True | 0.954106 | 2890 | 28 | 20 | exchangeable |
| 0.128739 | 0.031225 | 4.122967 | 3.740232e-05 | True | True | fourSC | 4hour_step_norm | 4 | prior2HourStepCount | True | 0.935469 | 2890 | 28 | 20 | exchangeable |
| 0.005287 | 0.016131 | 0.327785 | 0.743074 | False | False | fourSC | 4hour_step_norm | 5 | activitySuggestionsSentLast7Days | True | 0.968302 | 2890 | 28 | 20 | exchangeable |
| 0.046202 | 0.064890 | 0.712002 | 0.476463 | False | False | fourSC | 4hour_step_norm | 6 | activitySuggestionInteractLast7Days | True | 0.353473 | 2890 | 28 | 20 | exchangeable |
| 0.030258 | 0.068237 | 0.443420 | 0.657462 | False | False | fourSC | 4hour_step_norm | 7 | activeDaysLast7Days | True | 0.260938 | 2890 | 28 | 20 | exchangeable |
| -0.006581 | 0.042534 | -0.154726 | 0.877037 | False | False | fourSC | 4hour_step_norm | 8 | isWeekend | True | 0.369659 | 2890 | 28 | 20 | exchangeable |
| -0.073765 | 0.060467 | -1.219923 | 0.222494 | False | False | fourSC | 4hour_step_norm | 9 | decisionTimeSlot | True | 0.497078 | 2890 | 28 | 20 | exchangeable |
| 0.025613 | 0.018503 | 1.384258 | 0.166279 | False | False | fourSC | 4hour_step_norm | 10 | perceivedUtilityLastWeek | True | 2.361545 | 2890 | 28 | 20 | exchangeable |
| 0.000199 | 0.031994 | 0.006212 | 0.995043 | False | False | fourSC | 4hour_step_norm | 11 | caeAverageLastWeek | True | 0.947049 | 2890 | 28 | 20 | exchangeable |
| -0.051081 | 0.054384 | -0.939266 | 0.347594 | False | False | fourSC | 4hour_step_norm | 12 | Ah | True | 0.499925 | 2890 | 28 | 20 | exchangeable |
| 0.014488 | 0.033100 | 0.437714 | 0.661593 | False | False | fourSC | 4hour_step_norm | 13 | Ah*yesterdayStepCount | True | 0.661722 | 2890 | 28 | 20 | exchangeable |
| -0.018528 | 0.028255 | -0.655738 | 0.511993 | False | False | fourSC | 4hour_step_norm | 14 | Ah*prior2HourStepCount | True | 0.655759 | 2890 | 28 | 20 | exchangeable |
| 0.007360 | 0.016516 | 0.445647 | 0.655852 | False | False | fourSC | 4hour_step_norm | 15 | Ah*activitySuggestionsSentLast7Days | True | 0.684165 | 2890 | 28 | 20 | exchangeable |
| 0.048140 | 0.081787 | 0.588601 | 0.556129 | False | False | fourSC | 4hour_step_norm | 16 | Ah*activitySuggestionInteractLast7Days | True | 0.343672 | 2890 | 28 | 20 | exchangeable |
| 0.005944 | 0.012298 | 0.483287 | 0.628892 | False | False | fourSC | 4hour_step_norm | 17 | Ah*perceivedUtilityLastWeek | True | 1.784542 | 2890 | 28 | 20 | exchangeable |
| -0.014947 | 0.025416 | -0.588117 | 0.556454 | False | False | fourSC | 4hour_step_norm | 18 | Ah*caeAverageLastWeek | True | 0.644148 | 2890 | 28 | 20 | exchangeable |
| -9.247129e-05 | 0.046372 | -0.001994 | 0.998409 | False | False | fourSC | 4hour_step_norm | 19 | Ah*decisionTimeSlot | True | 0.413354 | 2890 | 28 | 20 | exchangeable |
| 0.387381 |  |  |  | False | False | antic | anticipated_affect_norm | 0 | intercept | False | 0.000000 | 817 | 28 | 15 | exchangeable |
| 0.077584 | 0.032578 | 2.381505 | 0.017242 | True | False | antic | anticipated_affect_norm | 1 | anticipated_affect_yesterday | True | 0.329197 | 817 | 28 | 15 | exchangeable |
| 0.125006 | 0.038548 | 3.242842 | 0.001183 | True | True | antic | anticipated_affect_norm | 2 | active_status_fraction_7days | True | 0.224466 | 817 | 28 | 15 | exchangeable |
| -0.003913 | 0.021393 | -0.182930 | 0.854853 | False | False | antic | anticipated_affect_norm | 3 | is_weekend | True | 0.370289 | 817 | 28 | 15 | exchangeable |
| 0.009045 | 0.008035 | 1.125716 | 0.260286 | False | False | antic | anticipated_affect_norm | 4 | perceived_utility_lastweek | True | 2.244579 | 817 | 28 | 15 | exchangeable |
| 0.117986 | 0.036596 | 3.223987 | 0.001264 | True | True | antic | anticipated_affect_norm | 5 | CAE_avg_lastweek | True | 0.895117 | 817 | 28 | 15 | exchangeable |
| 0.011392 | 0.006039 | 1.886287 | 0.059256 | False | False | antic | anticipated_affect_norm | 6 | recent_burden | True | 0.956169 | 817 | 28 | 15 | exchangeable |
| 0.008079 | 0.009928 | 0.813759 | 0.415783 | False | False | antic | anticipated_affect_norm | 7 | A0_morning | True | 0.499865 | 817 | 28 | 15 | exchangeable |
| 0.028478 | 0.009705 | 2.934333 | 0.003343 | True | True | antic | anticipated_affect_norm | 8 | A1_afternoon | True | 0.499937 | 817 | 28 | 15 | exchangeable |
| -0.003603 | 0.003376 | -1.067068 | 0.285941 | False | False | antic | anticipated_affect_norm | 9 | A0_morning_by_perceived_utility_lastweek | True | 1.765888 | 817 | 28 | 15 | exchangeable |
| -0.006580 | 0.004202 | -1.565806 | 0.117394 | False | False | antic | anticipated_affect_norm | 10 | A1_afternoon_by_perceived_utility_lastweek | True | 1.731512 | 817 | 28 | 15 | exchangeable |
| 0.017136 | 0.011413 | 1.501438 | 0.133242 | False | False | antic | anticipated_affect_norm | 11 | A0_morning_by_CAE_avg_lastweek | True | 0.641238 | 817 | 28 | 15 | exchangeable |
| -0.011453 | 0.014823 | -0.772652 | 0.439729 | False | False | antic | anticipated_affect_norm | 12 | A1_afternoon_by_CAE_avg_lastweek | True | 0.615386 | 817 | 28 | 15 | exchangeable |
| -0.005207 | 0.011147 | -0.467088 | 0.640437 | False | False | antic | anticipated_affect_norm | 13 | A0_morning_by_recent_burden | True | 0.646184 | 817 | 28 | 15 | exchangeable |
| -0.014790 | 0.012470 | -1.186046 | 0.235604 | False | False | antic | anticipated_affect_norm | 14 | A1_afternoon_by_recent_burden | True | 0.628143 | 817 | 28 | 15 | exchangeable |
| -0.079791 |  |  |  | False | False | CAE | CAE_avg_norm | 0 | intercept | False | 0.000000 | 213 | 28 | 4 | exchangeable |
| 0.935921 | 0.041480 | 22.563226 | 9.958951e-113 | True | True | CAE | CAE_avg_norm | 1 | CAE_avg_lastweek | True | 0.979802 | 213 | 28 | 4 | exchangeable |
| -0.033925 | 0.030529 | -1.111241 | 0.266465 | False | False | CAE | CAE_avg_norm | 2 | fourSC_ewma | True | 0.522369 | 213 | 28 | 4 | exchangeable |
| 0.144196 | 0.118838 | 1.213380 | 0.224984 | False | False | CAE | CAE_avg_norm | 3 | anticipated_affect_ewma | True | 0.332321 | 213 | 28 | 4 | exchangeable |

## Pooled RL Q GEE Coefficients

Gaussian GEE on the final fitted-Q regression from pooled FQI (``phi_obs`` vs bootstrap targets), clustered by participant. ``identified=False`` marks structurally unused features with zero design variance (e.g. ``b_tilde``, masked day-6 mediators); SE / p-values are omitted for those rows. RL priors still use ridge-FQI for ``mu_0_micro``; all-zero design columns get ``UNIDENTIFIED_PRIOR_VAR`` instead of the ``MIN_SIGMA2`` floor.

| coefficient | robust_se | z_stat | p_value | significant_0.05 | significant_0.01 | model | outcome | index | feature | identified | feature_std | n_obs | n_clusters | n_features | working_correlation | fqi_iters | ridge_alpha | residual_sigma2 | block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.119423 |  |  |  | False | False | q_no_td_modify | fqi_target | 0 | intercept | False | 0.000000 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| -0.071457 | 0.036984 | -1.932121 | 0.053345 | False | False | q_no_td_modify | fqi_target | 1 | weekday_vs_weekend | True | 0.372678 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| -0.009129 | 0.014166 | -0.644434 | 0.519294 | False | False | q_no_td_modify | fqi_target | 2 | slot_pm | True | 0.500000 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| 0.020973 | 0.001251 | 16.769529 | 4.077255e-63 | True | True | q_no_td_modify | fqi_target | 3 | E_w | True | 2.350134 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| 1.158349 | 0.005906 | 196.147265 | 0.000000 | True | True | q_no_td_modify | fqi_target | 4 | b_hat | True | 0.921918 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| -2.839751e-16 |  |  |  | False | False | q_no_td_modify | fqi_target | 5 | b_tilde | False | 0.000000 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| 0.290343 | 0.015041 | 19.303118 | 5.056430e-83 | True | True | q_no_td_modify | fqi_target | 6 | M_Y_anticipated_affect_ewma | True | 0.311549 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| -0.023274 | 0.001954 | -11.908768 | 1.065403e-32 | True | True | q_no_td_modify | fqi_target | 7 | M_Y_fourSC_ewma | True | 0.948911 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| -0.000129 | 0.001993 | -0.064584 | 0.948505 | False | False | q_no_td_modify | fqi_target | 8 | M_E_pageview_ewma | True | 1.050150 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| -0.131137 | 0.006629 | -19.781461 | 4.300507e-87 | True | True | q_no_td_modify | fqi_target | 9 | M_E_fitbit_wear_ewma | True | 0.412251 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| -0.042466 | 0.005724 | -7.419342 | 1.177037e-13 | True | True | q_no_td_modify | fqi_target | 10 | M_E_survey_complete_ewma | True | 0.411996 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| -0.007210 | 0.004864 | -1.482184 | 0.138291 | False | False | q_no_td_modify | fqi_target | 11 | yesterday_step_count | True | 0.918323 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| -0.013071 | 0.004326 | -3.021692 | 0.002514 | True | True | q_no_td_modify | fqi_target | 12 | prior2hour_step_count | True | 0.923902 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| -0.003204 | 0.011978 | -0.267506 | 0.789080 | False | False | q_no_td_modify | fqi_target | 13 | active_status_fraction_7days | True | 0.266148 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| -0.011825 | 0.004167 | -2.837605 | 0.004545 | True | True | q_no_td_modify | fqi_target | 14 | recent_burden | True | 0.931218 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| -0.068351 | 0.009877 | -6.920569 | 4.498330e-12 | True | True | q_no_td_modify | fqi_target | 15 | walk_interaction_7d | True | 0.354304 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| -0.000958 | 0.020094 | -0.047690 | 0.961964 | False | False | q_no_td_modify | fqi_target | 16 | A | True | 0.499915 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| 0.000915 | 0.002065 | 0.443027 | 0.657746 | False | False | q_no_td_modify | fqi_target | 17 | A*E_w | True | 1.764758 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| 0.000379 | 0.010754 | 0.035234 | 0.971893 | False | False | q_no_td_modify | fqi_target | 18 | A*b_hat | True | 0.635088 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| -1.822752e-16 |  |  |  | False | False | q_no_td_modify | fqi_target | 19 | A*b_tilde | False | 0.000000 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| -0.004016 | 0.007597 | -0.528610 | 0.597076 | False | False | q_no_td_modify | fqi_target | 20 | A*yesterday_step_count | True | 0.654833 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| 0.006328 | 0.004401 | 1.437707 | 0.150517 | False | False | q_no_td_modify | fqi_target | 21 | A*prior2hour_step_count | True | 0.644358 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| -0.006657 | 0.020732 | -0.321119 | 0.748120 | False | False | q_no_td_modify | fqi_target | 22 | A*active_status_fraction_7days | True | 0.375177 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| -0.001899 | 0.005138 | -0.369686 | 0.711617 | False | False | q_no_td_modify | fqi_target | 23 | A*recent_burden | True | 0.662545 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| -0.004046 | 0.017607 | -0.229805 | 0.818243 | False | False | q_no_td_modify | fqi_target | 24 | A*walk_interaction_7d | True | 0.334802 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| 0.040297 | 0.031478 | 1.280154 | 0.200491 | False | False | q_no_td_modify | fqi_target | 25 | A*weekday_vs_weekend | True | 0.281695 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
| 0.009116 | 0.011661 | 0.781803 | 0.434330 | False | False | q_no_td_modify | fqi_target | 26 | A*slot_pm | True | 0.432496 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.030153 | beta |
