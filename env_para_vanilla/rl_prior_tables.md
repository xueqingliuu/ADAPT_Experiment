# Prior Mean and Variance Summary

`prior_variance` is the diagonal entry of the prior covariance. For `q_td_modify_joint`, the full covariance remains in `rl_priors.json`; this table reports marginal variances.

## PF Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| PF | fourSC | theta | 0 | intercept | -0.054691 | 0.053447 |
| PF | fourSC | theta | 1 | stepCountNext4HourLag1 | 0.478842 | 0.042910 |
| PF | fourSC | theta | 2 | yesterdayStepCount | -0.168025 | 0.011548 |
| PF | fourSC | theta | 3 | stepCountLast7DaysEma | 0.522006 | 0.034859 |
| PF | fourSC | theta | 4 | prior2HourStepCount | 0.104427 | 0.033859 |
| PF | fourSC | theta | 5 | activitySuggestionsSentLast7Days | -0.027834 | 0.027891 |
| PF | fourSC | theta | 6 | activitySuggestionInteractLast7Days | 0.062358 | 0.074129 |
| PF | fourSC | theta | 7 | activeDaysLast7Days | 0.149411 | 0.045480 |
| PF | fourSC | theta | 8 | isWeekend | -0.033029 | 0.049200 |
| PF | fourSC | theta | 9 | decisionTimeSlot | -0.049603 | 0.152311 |
| PF | fourSC | theta | 10 | perceivedUtilityLastWeek | 0.012075 | 0.014799 |
| PF | fourSC | theta | 11 | caeAverageLastWeek | -0.048148 | 0.101409 |
| PF | fourSC | theta | 12 | Ah | -0.093421 | 0.064264 |
| PF | fourSC | theta | 13 | Ah*yesterdayStepCount | 0.004651 | 0.020010 |
| PF | fourSC | theta | 14 | Ah*prior2HourStepCount | 0.012977 | 0.024124 |
| PF | fourSC | theta | 15 | Ah*activitySuggestionsSentLast7Days | 0.024239 | 0.024753 |
| PF | fourSC | theta | 16 | Ah*activitySuggestionInteractLast7Days | 0.106010 | 0.107903 |
| PF | fourSC | theta | 17 | Ah*perceivedUtilityLastWeek | 0.008482 | 0.023184 |
| PF | fourSC | theta | 18 | Ah*caeAverageLastWeek | 0.016509 | 0.092960 |
| PF | antic | theta | 0 | intercept | 0.446849 | 0.014107 |
| PF | antic | theta | 1 | anticipated_affect_yesterday | 0.181563 | 0.005331 |
| PF | antic | theta | 2 | active_status_fraction_7days | 0.073015 | 0.008829 |
| PF | antic | theta | 3 | is_weekend | -0.023439 | 0.004320 |
| PF | antic | theta | 4 | perceived_utility_lastweek | 0.004530 | 0.009429 |
| PF | antic | theta | 5 | CAE_avg_lastweek | 0.247484 | 0.009014 |
| PF | antic | theta | 6 | recent_burden | 0.020060 | 0.003703 |
| PF | antic | theta | 7 | A0_morning | 0.008973 | 0.002500 |
| PF | antic | theta | 8 | A1_afternoon | 0.019286 | 0.001367 |
| PF | antic | theta | 9 | A0_morning_by_perceived_utility_lastweek | -0.001815 | 0.001397 |
| PF | antic | theta | 10 | A1_afternoon_by_perceived_utility_lastweek | -0.005232 | 0.000758 |
| PF | antic | theta | 11 | A0_morning_by_CAE_avg_lastweek | 0.000862 | 0.005257 |
| PF | antic | theta | 12 | A1_afternoon_by_CAE_avg_lastweek | -0.025406 | 0.002887 |
| PF | antic | theta | 13 | A0_morning_by_recent_burden | -0.013469 | 0.001339 |
| PF | antic | theta | 14 | A1_afternoon_by_recent_burden | 0.000301 | 0.004066 |
| PF | CAE | theta | 0 | intercept | -0.103666 | 0.094922 |
| PF | CAE | theta | 1 | CAE_avg_lastweek | 0.903687 | 0.056058 |
| PF | CAE | theta | 2 | fourSC_ewma | -0.018991 | 0.013880 |
| PF | CAE | theta | 3 | anticipated_affect_ewma | 0.162856 | 0.023575 |
| PF | CAE_short | theta | 0 | intercept | 0.003822 | 0.074460 |
| PF | CAE_short | theta | 1 | caeAverage | 0.968580 | 0.059765 |

## RL Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| RL | reward_shaping | eta | 0 | intercept | -0.049157 | 0.026333 |
| RL | reward_shaping | eta | 1 | weekday_vs_weekend | -0.008193 | 0.000731 |
| RL | reward_shaping | eta | 2 | slot_pm | -0.024578 | 0.006583 |
| RL | reward_shaping | eta | 3 | E_w | 0.067738 | 0.002772 |
| RL | reward_shaping | eta | 4 | b_hat | 0.062050 | 0.044578 |
| RL | reward_shaping | eta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | reward_shaping | eta | 6 | M_Y_anticipated_affect_ewma | 0.075245 | 0.008477 |
| RL | reward_shaping | eta | 7 | M_Y_fourSC_ewma | 0.009537 | 0.024468 |
| RL | reward_shaping | eta | 8 | M_E_pageview_ewma | -0.008676 | 0.003816 |
| RL | reward_shaping | eta | 9 | M_E_fitbit_wear_ewma | -0.011294 | 0.007240 |
| RL | reward_shaping | eta | 10 | M_E_survey_complete_ewma | 0.006704 | 0.013047 |
| RL | reward_shaping | eta | 11 | yesterday_step_count | -0.003268 | 0.015238 |
| RL | reward_shaping | eta | 12 | prior2hour_step_count | -0.017366 | 0.005625 |
| RL | reward_shaping | eta | 13 | active_status_fraction_7days | 0.006740 | 0.039757 |
| RL | reward_shaping | eta | 14 | recent_burden | -0.000966 | 0.004392 |
| RL | reward_shaping | eta | 15 | walk_interaction_7d | 0.012112 | 0.035146 |
| RL | redistribution_stage1_AA | eta | 0 | redistribution_stage1_AA_coef_0 | 0.253426 | 0.007558 |
| RL | redistribution_stage1_AA | eta | 1 | redistribution_stage1_AA_coef_1 | 0.000182 | 0.001271 |
| RL | redistribution_stage1_AA | eta | 2 | redistribution_stage1_AA_coef_2 | 0.097357 | 0.003118 |
| RL | redistribution_stage1_AA | eta | 3 | redistribution_stage1_AA_coef_3 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_AA | eta | 4 | redistribution_stage1_AA_coef_4 | 0.160654 | 0.002037 |
| RL | redistribution_stage1_AA | eta | 5 | redistribution_stage1_AA_coef_5 | -0.009476 | 0.002966 |
| RL | redistribution_stage1_AA | eta | 6 | redistribution_stage1_AA_coef_6 | 0.011249 | 0.000540 |
| RL | redistribution_stage1_AA | eta | 7 | redistribution_stage1_AA_coef_7 | -0.038080 | 0.001548 |
| RL | redistribution_stage1_AA | eta | 8 | redistribution_stage1_AA_coef_8 | -0.024668 | 0.001331 |
| RL | redistribution_stage1_AA | eta | 9 | redistribution_stage1_AA_coef_9 | 0.007318 | 0.000195 |
| RL | redistribution_stage1_AA | eta | 10 | redistribution_stage1_AA_coef_10 | 0.010482 | 0.000348 |
| RL | redistribution_stage1_AA | eta | 11 | redistribution_stage1_AA_coef_11 | 0.024917 | 0.004639 |
| RL | redistribution_stage1_AA | eta | 12 | redistribution_stage1_AA_coef_12 | 0.001968 | 0.000114 |
| RL | redistribution_stage1_AA | eta | 13 | redistribution_stage1_AA_coef_13 | 0.005027 | 0.002810 |
| RL | redistribution_stage1_AA | eta | 14 | redistribution_stage1_AA_coef_14 | 0.014206 | 0.001237 |
| RL | redistribution_stage1_AA | eta | 15 | redistribution_stage1_AA_coef_15 | 0.000890 | 0.000571 |
| RL | redistribution_stage1_AA | eta | 16 | redistribution_stage1_AA_coef_16 | -0.006646 | 0.002138 |
| RL | redistribution_stage1_AA | eta | 17 | redistribution_stage1_AA_coef_17 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_AA | eta | 18 | redistribution_stage1_AA_coef_18 | 0.001192 | 0.000591 |
| RL | redistribution_stage1_AA | eta | 19 | redistribution_stage1_AA_coef_19 | 0.007257 | 0.000480 |
| RL | redistribution_stage1_AA | eta | 20 | redistribution_stage1_AA_coef_20 | -0.015962 | 0.003085 |
| RL | redistribution_stage1_AA | eta | 21 | redistribution_stage1_AA_coef_21 | 0.003264 | 0.000280 |
| RL | redistribution_stage1_AA | eta | 22 | redistribution_stage1_AA_coef_22 | 0.002496 | 0.001002 |
| RL | redistribution_stage1_FW | eta | 0 | redistribution_stage1_FW_coef_0 | 0.371782 | 0.022976 |
| RL | redistribution_stage1_FW | eta | 1 | redistribution_stage1_FW_coef_1 | 0.007157 | 0.006044 |
| RL | redistribution_stage1_FW | eta | 2 | redistribution_stage1_FW_coef_2 | 0.013281 | 0.009366 |
| RL | redistribution_stage1_FW | eta | 3 | redistribution_stage1_FW_coef_3 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_FW | eta | 4 | redistribution_stage1_FW_coef_4 | -0.127282 | 0.026394 |
| RL | redistribution_stage1_FW | eta | 5 | redistribution_stage1_FW_coef_5 | -0.022729 | 0.008944 |
| RL | redistribution_stage1_FW | eta | 6 | redistribution_stage1_FW_coef_6 | 0.024439 | 0.001378 |
| RL | redistribution_stage1_FW | eta | 7 | redistribution_stage1_FW_coef_7 | 0.162780 | 0.016139 |
| RL | redistribution_stage1_FW | eta | 8 | redistribution_stage1_FW_coef_8 | -0.008621 | 0.008775 |
| RL | redistribution_stage1_FW | eta | 9 | redistribution_stage1_FW_coef_9 | 0.007918 | 0.003149 |
| RL | redistribution_stage1_FW | eta | 10 | redistribution_stage1_FW_coef_10 | 0.008765 | 0.002329 |
| RL | redistribution_stage1_FW | eta | 11 | redistribution_stage1_FW_coef_11 | 0.031998 | 0.015378 |
| RL | redistribution_stage1_FW | eta | 12 | redistribution_stage1_FW_coef_12 | -0.006602 | 0.002950 |
| RL | redistribution_stage1_FW | eta | 13 | redistribution_stage1_FW_coef_13 | 0.032297 | 0.020367 |
| RL | redistribution_stage1_FW | eta | 14 | redistribution_stage1_FW_coef_14 | 0.006586 | 0.009969 |
| RL | redistribution_stage1_FW | eta | 15 | redistribution_stage1_FW_coef_15 | -0.006946 | 0.005733 |
| RL | redistribution_stage1_FW | eta | 16 | redistribution_stage1_FW_coef_16 | -0.000561 | 0.013207 |
| RL | redistribution_stage1_FW | eta | 17 | redistribution_stage1_FW_coef_17 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_FW | eta | 18 | redistribution_stage1_FW_coef_18 | -0.000858 | 0.006931 |
| RL | redistribution_stage1_FW | eta | 19 | redistribution_stage1_FW_coef_19 | -0.002671 | 0.004850 |
| RL | redistribution_stage1_FW | eta | 20 | redistribution_stage1_FW_coef_20 | -0.018634 | 0.021222 |
| RL | redistribution_stage1_FW | eta | 21 | redistribution_stage1_FW_coef_21 | 0.005473 | 0.004336 |
| RL | redistribution_stage1_FW | eta | 22 | redistribution_stage1_FW_coef_22 | 0.011590 | 0.022783 |
| RL | redistribution_stage1_PJ | eta | 0 | redistribution_stage1_PJ_coef_0 | 0.115059 | 0.038631 |
| RL | redistribution_stage1_PJ | eta | 1 | redistribution_stage1_PJ_coef_1 | 0.006675 | 0.006843 |
| RL | redistribution_stage1_PJ | eta | 2 | redistribution_stage1_PJ_coef_2 | 0.018764 | 0.019950 |
| RL | redistribution_stage1_PJ | eta | 3 | redistribution_stage1_PJ_coef_3 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_PJ | eta | 4 | redistribution_stage1_PJ_coef_4 | -0.131692 | 0.031634 |
| RL | redistribution_stage1_PJ | eta | 5 | redistribution_stage1_PJ_coef_5 | -0.011948 | 0.019283 |
| RL | redistribution_stage1_PJ | eta | 6 | redistribution_stage1_PJ_coef_6 | 0.008832 | 0.002149 |
| RL | redistribution_stage1_PJ | eta | 7 | redistribution_stage1_PJ_coef_7 | 0.010663 | 0.009392 |
| RL | redistribution_stage1_PJ | eta | 8 | redistribution_stage1_PJ_coef_8 | 0.154537 | 0.011279 |
| RL | redistribution_stage1_PJ | eta | 9 | redistribution_stage1_PJ_coef_9 | 0.000972 | 0.007661 |
| RL | redistribution_stage1_PJ | eta | 10 | redistribution_stage1_PJ_coef_10 | 0.020865 | 0.003086 |
| RL | redistribution_stage1_PJ | eta | 11 | redistribution_stage1_PJ_coef_11 | 0.053417 | 0.031360 |
| RL | redistribution_stage1_PJ | eta | 12 | redistribution_stage1_PJ_coef_12 | 0.016608 | 0.004689 |
| RL | redistribution_stage1_PJ | eta | 13 | redistribution_stage1_PJ_coef_13 | 0.192442 | 0.025532 |
| RL | redistribution_stage1_PJ | eta | 14 | redistribution_stage1_PJ_coef_14 | 0.050302 | 0.010974 |
| RL | redistribution_stage1_PJ | eta | 15 | redistribution_stage1_PJ_coef_15 | 0.002572 | 0.008864 |
| RL | redistribution_stage1_PJ | eta | 16 | redistribution_stage1_PJ_coef_16 | 0.027540 | 0.022400 |
| RL | redistribution_stage1_PJ | eta | 17 | redistribution_stage1_PJ_coef_17 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_PJ | eta | 18 | redistribution_stage1_PJ_coef_18 | 0.017035 | 0.012879 |
| RL | redistribution_stage1_PJ | eta | 19 | redistribution_stage1_PJ_coef_19 | 0.002255 | 0.009797 |
| RL | redistribution_stage1_PJ | eta | 20 | redistribution_stage1_PJ_coef_20 | -0.059764 | 0.030052 |
| RL | redistribution_stage1_PJ | eta | 21 | redistribution_stage1_PJ_coef_21 | -0.002778 | 0.009392 |
| RL | redistribution_stage1_PJ | eta | 22 | redistribution_stage1_PJ_coef_22 | -0.024371 | 0.021003 |
| RL | redistribution_stage2_v2 | eta | 0 | redistribution_stage2_v2_coef_0 | -0.176881 | 0.000991 |
| RL | redistribution_stage2_v2 | eta | 1 | redistribution_stage2_v2_coef_1 | -0.029480 | 2.752997e-05 |
| RL | redistribution_stage2_v2 | eta | 2 | redistribution_stage2_v2_coef_2 | 0.038636 | 0.000596 |
| RL | redistribution_stage2_v2 | eta | 3 | redistribution_stage2_v2_coef_3 | -0.067550 | 0.001462 |
| RL | redistribution_stage2_v2 | eta | 4 | redistribution_stage2_v2_coef_4 | 0.000000 | 1.000000 |
| RL | redistribution_stage2_v2 | eta | 5 | redistribution_stage2_v2_coef_5 | 0.308234 | 0.002251 |
| RL | redistribution_stage2_v2 | eta | 6 | redistribution_stage2_v2_coef_6 | -0.020696 | 0.002401 |
| RL | redistribution_stage2_v2 | eta | 7 | redistribution_stage2_v2_coef_7 | -0.024073 | 0.001399 |
| RL | redistribution_stage2_v2 | eta | 8 | redistribution_stage2_v2_coef_8 | 0.066243 | 0.001598 |
| RL | redistribution_stage2_v2 | eta | 9 | redistribution_stage2_v2_coef_9 | -0.019562 | 0.002384 |
| RL | redistribution_stage2_v2 | eta | 10 | redistribution_stage2_v2_coef_10 | 0.032711 | 0.001960 |
| RL | redistribution_stage2_v2 | eta | 11 | redistribution_stage2_v2_coef_11 | -0.023149 | 0.000993 |
| RL | redistribution_stage2_v2 | eta | 12 | redistribution_stage2_v2_coef_12 | -0.025654 | 0.000930 |
| RL | redistribution_stage2_v2 | eta | 13 | redistribution_stage2_v2_coef_13 | 0.008923 | 0.000221 |
| RL | redistribution_stage2_v2 | eta | 14 | redistribution_stage2_v2_coef_14 | 0.121707 | 0.002799 |
| RL | redistribution_stage2_v2 | eta | 15 | redistribution_stage2_v2_coef_15 | 0.006327 | 0.000954 |
| RL | redistribution_stage2_v2 | eta | 16 | redistribution_stage2_v2_coef_16 | 0.007306 | 0.001864 |
| RL | redistribution_stage2_v2 | eta | 17 | redistribution_stage2_v2_coef_17 | 0.494699 | 0.000122 |
| RL | redistribution_stage2_v2 | eta | 18 | redistribution_stage2_v2_coef_18 | -0.488972 | 0.000528 |
| RL | redistribution_stage2_v2 | eta | 19 | redistribution_stage2_v2_coef_19 | 0.040290 | 0.000389 |
| RL | redistribution_stage2_v4 | eta | 0 | redistribution_stage2_v4_coef_0 | -0.096746 | 0.004892 |
| RL | redistribution_stage2_v4 | eta | 1 | redistribution_stage2_v4_coef_1 | -0.016124 | 0.000136 |
| RL | redistribution_stage2_v4 | eta | 2 | redistribution_stage2_v4_coef_2 | -0.023864 | 0.002255 |
| RL | redistribution_stage2_v4 | eta | 3 | redistribution_stage2_v4_coef_3 | 0.002895 | 0.021000 |
| RL | redistribution_stage2_v4 | eta | 4 | redistribution_stage2_v4_coef_4 | 0.000000 | 1.000000 |
| RL | redistribution_stage2_v4 | eta | 5 | redistribution_stage2_v4_coef_5 | 0.122643 | 0.009232 |
| RL | redistribution_stage2_v4 | eta | 6 | redistribution_stage2_v4_coef_6 | 0.080666 | 0.033829 |
| RL | redistribution_stage2_v4 | eta | 7 | redistribution_stage2_v4_coef_7 | -0.029602 | 0.004681 |
| RL | redistribution_stage2_v4 | eta | 8 | redistribution_stage2_v4_coef_8 | 0.068655 | 0.005202 |
| RL | redistribution_stage2_v4 | eta | 9 | redistribution_stage2_v4_coef_9 | -0.175055 | 0.020285 |
| RL | redistribution_stage2_v4 | eta | 10 | redistribution_stage2_v4_coef_10 | -0.010731 | 0.019433 |
| RL | redistribution_stage2_v4 | eta | 11 | redistribution_stage2_v4_coef_11 | -0.046293 | 0.006598 |
| RL | redistribution_stage2_v4 | eta | 12 | redistribution_stage2_v4_coef_12 | -0.013898 | 0.006331 |
| RL | redistribution_stage2_v4 | eta | 13 | redistribution_stage2_v4_coef_13 | -0.021524 | 0.004085 |
| RL | redistribution_stage2_v4 | eta | 14 | redistribution_stage2_v4_coef_14 | -0.192866 | 0.015772 |
| RL | redistribution_stage2_v4 | eta | 15 | redistribution_stage2_v4_coef_15 | -0.061917 | 0.025081 |
| RL | redistribution_stage2_v4 | eta | 16 | redistribution_stage2_v4_coef_16 | 0.022637 | 0.012265 |
| RL | redistribution_stage2_v4 | eta | 17 | redistribution_stage2_v4_coef_17 | 0.292583 | 0.000881 |
| RL | redistribution_stage2_v4 | eta | 18 | redistribution_stage2_v4_coef_18 | -0.514000 | 0.001470 |
| RL | redistribution_stage2_v4 | eta | 19 | redistribution_stage2_v4_coef_19 | 1.206672 | 0.002577 |
| RL | q_no_td_modify | beta | 0 | intercept | 0.060995 | 0.067700 |
| RL | q_no_td_modify | beta | 1 | weekday_vs_weekend | -0.088658 | 0.084778 |
| RL | q_no_td_modify | beta | 2 | slot_pm | -0.012127 | 0.001956 |
| RL | q_no_td_modify | beta | 3 | E_w | 0.000735 | 0.028790 |
| RL | q_no_td_modify | beta | 4 | b_hat | 1.169780 | 0.042603 |
| RL | q_no_td_modify | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify | beta | 6 | M_Y_anticipated_affect_ewma | 0.224245 | 0.006904 |
| RL | q_no_td_modify | beta | 7 | M_Y_fourSC_ewma | -0.011125 | 0.006137 |
| RL | q_no_td_modify | beta | 8 | M_E_pageview_ewma | 0.002983 | 0.000206 |
| RL | q_no_td_modify | beta | 9 | M_E_fitbit_wear_ewma | -0.112261 | 0.003062 |
| RL | q_no_td_modify | beta | 10 | M_E_survey_complete_ewma | -0.039276 | 0.004260 |
| RL | q_no_td_modify | beta | 11 | yesterday_step_count | -0.000345 | 0.000923 |
| RL | q_no_td_modify | beta | 12 | prior2hour_step_count | -0.008811 | 0.000691 |
| RL | q_no_td_modify | beta | 13 | active_status_fraction_7days | 0.009858 | 0.024503 |
| RL | q_no_td_modify | beta | 14 | recent_burden | -0.003384 | 0.001695 |
| RL | q_no_td_modify | beta | 15 | walk_interaction_7d | -0.013197 | 0.021276 |
| RL | q_no_td_modify | beta | 16 | A | -0.001170 | 0.001279 |
| RL | q_no_td_modify | beta | 17 | A*E_w | 0.000884 | 0.000907 |
| RL | q_no_td_modify | beta | 18 | A*b_hat | -0.005901 | 0.001851 |
| RL | q_no_td_modify | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify | beta | 20 | A*yesterday_step_count | -0.006130 | 0.002236 |
| RL | q_no_td_modify | beta | 21 | A*prior2hour_step_count | 0.004554 | 0.001064 |
| RL | q_no_td_modify | beta | 22 | A*active_status_fraction_7days | 0.002712 | 0.001133 |
| RL | q_no_td_modify | beta | 23 | A*recent_burden | -0.003878 | 0.000779 |
| RL | q_no_td_modify | beta | 24 | A*walk_interaction_7d | 0.002433 | 0.001709 |
| RL | q_no_td_modify_g09 | beta | 0 | intercept | 0.104814 | 0.118842 |
| RL | q_no_td_modify_g09 | beta | 1 | weekday_vs_weekend | -0.135660 | 0.224355 |
| RL | q_no_td_modify_g09 | beta | 2 | slot_pm | -0.021428 | 0.006837 |
| RL | q_no_td_modify_g09 | beta | 3 | E_w | 0.001297 | 0.047244 |
| RL | q_no_td_modify_g09 | beta | 4 | b_hat | 1.595390 | 0.067837 |
| RL | q_no_td_modify_g09 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g09 | beta | 6 | M_Y_anticipated_affect_ewma | 0.367878 | 0.015434 |
| RL | q_no_td_modify_g09 | beta | 7 | M_Y_fourSC_ewma | -0.016787 | 0.010672 |
| RL | q_no_td_modify_g09 | beta | 8 | M_E_pageview_ewma | 0.005965 | 0.000361 |
| RL | q_no_td_modify_g09 | beta | 9 | M_E_fitbit_wear_ewma | -0.180975 | 0.006041 |
| RL | q_no_td_modify_g09 | beta | 10 | M_E_survey_complete_ewma | -0.058079 | 0.006829 |
| RL | q_no_td_modify_g09 | beta | 11 | yesterday_step_count | 0.005650 | 0.001684 |
| RL | q_no_td_modify_g09 | beta | 12 | prior2hour_step_count | -0.014170 | 0.001251 |
| RL | q_no_td_modify_g09 | beta | 13 | active_status_fraction_7days | 0.040799 | 0.042703 |
| RL | q_no_td_modify_g09 | beta | 14 | recent_burden | -0.003532 | 0.003488 |
| RL | q_no_td_modify_g09 | beta | 15 | walk_interaction_7d | -0.005326 | 0.032437 |
| RL | q_no_td_modify_g09 | beta | 16 | A | -0.009092 | 0.004059 |
| RL | q_no_td_modify_g09 | beta | 17 | A*E_w | 0.000936 | 0.002026 |
| RL | q_no_td_modify_g09 | beta | 18 | A*b_hat | -0.011186 | 0.003545 |
| RL | q_no_td_modify_g09 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g09 | beta | 20 | A*yesterday_step_count | -0.011553 | 0.004201 |
| RL | q_no_td_modify_g09 | beta | 21 | A*prior2hour_step_count | 0.006649 | 0.001805 |
| RL | q_no_td_modify_g09 | beta | 22 | A*active_status_fraction_7days | 0.009238 | 0.002621 |
| RL | q_no_td_modify_g09 | beta | 23 | A*recent_burden | -0.006856 | 0.001423 |
| RL | q_no_td_modify_g09 | beta | 24 | A*walk_interaction_7d | 0.010419 | 0.003340 |
| RL | q_no_td_modify_g099 | beta | 0 | intercept | 0.118619 | 0.135699 |
| RL | q_no_td_modify_g099 | beta | 1 | weekday_vs_weekend | -0.147352 | 0.280948 |
| RL | q_no_td_modify_g099 | beta | 2 | slot_pm | -0.024190 | 0.009063 |
| RL | q_no_td_modify_g099 | beta | 3 | E_w | 0.001527 | 0.053050 |
| RL | q_no_td_modify_g099 | beta | 4 | b_hat | 1.717365 | 0.075569 |
| RL | q_no_td_modify_g099 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g099 | beta | 6 | M_Y_anticipated_affect_ewma | 0.412956 | 0.019018 |
| RL | q_no_td_modify_g099 | beta | 7 | M_Y_fourSC_ewma | -0.018400 | 0.011994 |
| RL | q_no_td_modify_g099 | beta | 8 | M_E_pageview_ewma | 0.006980 | 0.000412 |
| RL | q_no_td_modify_g099 | beta | 9 | M_E_fitbit_wear_ewma | -0.201758 | 0.007416 |
| RL | q_no_td_modify_g099 | beta | 10 | M_E_survey_complete_ewma | -0.063722 | 0.007857 |
| RL | q_no_td_modify_g099 | beta | 11 | yesterday_step_count | 0.007780 | 0.001992 |
| RL | q_no_td_modify_g099 | beta | 12 | prior2hour_step_count | -0.015861 | 0.001452 |
| RL | q_no_td_modify_g099 | beta | 13 | active_status_fraction_7days | 0.050804 | 0.048762 |
| RL | q_no_td_modify_g099 | beta | 14 | recent_burden | -0.003529 | 0.004206 |
| RL | q_no_td_modify_g099 | beta | 15 | walk_interaction_7d | -0.002616 | 0.036374 |
| RL | q_no_td_modify_g099 | beta | 16 | A | -0.011742 | 0.005355 |
| RL | q_no_td_modify_g099 | beta | 17 | A*E_w | 0.000927 | 0.002492 |
| RL | q_no_td_modify_g099 | beta | 18 | A*b_hat | -0.012874 | 0.004162 |
| RL | q_no_td_modify_g099 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g099 | beta | 20 | A*yesterday_step_count | -0.013293 | 0.004829 |
| RL | q_no_td_modify_g099 | beta | 21 | A*prior2hour_step_count | 0.007293 | 0.002076 |
| RL | q_no_td_modify_g099 | beta | 22 | A*active_status_fraction_7days | 0.011353 | 0.003371 |
| RL | q_no_td_modify_g099 | beta | 23 | A*recent_burden | -0.007840 | 0.001650 |
| RL | q_no_td_modify_g099 | beta | 24 | A*walk_interaction_7d | 0.013079 | 0.003982 |
| RL | q_redistribution_v2 | beta | 0 | intercept | 0.673154 | 0.232453 |
| RL | q_redistribution_v2 | beta | 1 | weekday_vs_weekend | -0.218664 | 0.070462 |
| RL | q_redistribution_v2 | beta | 2 | slot_pm | -0.013618 | 0.002038 |
| RL | q_redistribution_v2 | beta | 3 | E_w | 0.732784 | 0.479471 |
| RL | q_redistribution_v2 | beta | 4 | b_hat | 1.159861 | 0.084893 |
| RL | q_redistribution_v2 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v2 | beta | 6 | M_Y_anticipated_affect_ewma | 0.569038 | 0.025141 |
| RL | q_redistribution_v2 | beta | 7 | M_Y_fourSC_ewma | -0.040373 | 0.021551 |
| RL | q_redistribution_v2 | beta | 8 | M_E_pageview_ewma | -0.009888 | 0.004903 |
| RL | q_redistribution_v2 | beta | 9 | M_E_fitbit_wear_ewma | -0.316984 | 0.014320 |
| RL | q_redistribution_v2 | beta | 10 | M_E_survey_complete_ewma | -0.024397 | 0.017579 |
| RL | q_redistribution_v2 | beta | 11 | yesterday_step_count | 0.061786 | 0.007835 |
| RL | q_redistribution_v2 | beta | 12 | prior2hour_step_count | 0.036328 | 0.002619 |
| RL | q_redistribution_v2 | beta | 13 | active_status_fraction_7days | -0.043670 | 0.169684 |
| RL | q_redistribution_v2 | beta | 14 | recent_burden | 0.144826 | 0.008624 |
| RL | q_redistribution_v2 | beta | 15 | walk_interaction_7d | 1.027268 | 0.097458 |
| RL | q_redistribution_v2 | beta | 16 | A | 0.093737 | 0.008603 |
| RL | q_redistribution_v2 | beta | 17 | A*E_w | 0.006594 | 0.001179 |
| RL | q_redistribution_v2 | beta | 18 | A*b_hat | -0.022914 | 0.002328 |
| RL | q_redistribution_v2 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v2 | beta | 20 | A*yesterday_step_count | 0.030831 | 0.001229 |
| RL | q_redistribution_v2 | beta | 21 | A*prior2hour_step_count | 0.001670 | 0.002704 |
| RL | q_redistribution_v2 | beta | 22 | A*active_status_fraction_7days | -0.024571 | 0.006159 |
| RL | q_redistribution_v2 | beta | 23 | A*recent_burden | 0.023146 | 0.000838 |
| RL | q_redistribution_v2 | beta | 24 | A*walk_interaction_7d | -0.034968 | 0.008077 |
| RL | q_redistribution_v4 | beta | 0 | intercept | 0.037433 | 0.406974 |
| RL | q_redistribution_v4 | beta | 1 | weekday_vs_weekend | -0.096408 | 0.067532 |
| RL | q_redistribution_v4 | beta | 2 | slot_pm | -0.002327 | 0.001891 |
| RL | q_redistribution_v4 | beta | 3 | E_w | -0.283889 | 0.118975 |
| RL | q_redistribution_v4 | beta | 4 | b_hat | 1.660794 | 0.121849 |
| RL | q_redistribution_v4 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v4 | beta | 6 | M_Y_anticipated_affect_ewma | 0.154857 | 0.034835 |
| RL | q_redistribution_v4 | beta | 7 | M_Y_fourSC_ewma | -0.149053 | 0.093849 |
| RL | q_redistribution_v4 | beta | 8 | M_E_pageview_ewma | -0.020250 | 0.022403 |
| RL | q_redistribution_v4 | beta | 9 | M_E_fitbit_wear_ewma | -0.076416 | 0.018071 |
| RL | q_redistribution_v4 | beta | 10 | M_E_survey_complete_ewma | -0.007444 | 0.077652 |
| RL | q_redistribution_v4 | beta | 11 | yesterday_step_count | -0.030263 | 0.063990 |
| RL | q_redistribution_v4 | beta | 12 | prior2hour_step_count | 0.054305 | 0.014637 |
| RL | q_redistribution_v4 | beta | 13 | active_status_fraction_7days | 0.341830 | 0.214391 |
| RL | q_redistribution_v4 | beta | 14 | recent_burden | 0.030603 | 0.084057 |
| RL | q_redistribution_v4 | beta | 15 | walk_interaction_7d | 0.318308 | 0.296357 |
| RL | q_redistribution_v4 | beta | 16 | A | 0.084304 | 0.024821 |
| RL | q_redistribution_v4 | beta | 17 | A*E_w | -0.003151 | 0.004625 |
| RL | q_redistribution_v4 | beta | 18 | A*b_hat | -0.006382 | 0.006743 |
| RL | q_redistribution_v4 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v4 | beta | 20 | A*yesterday_step_count | 0.046783 | 0.005007 |
| RL | q_redistribution_v4 | beta | 21 | A*prior2hour_step_count | -0.019107 | 0.021893 |
| RL | q_redistribution_v4 | beta | 22 | A*active_status_fraction_7days | -0.075118 | 0.011639 |
| RL | q_redistribution_v4 | beta | 23 | A*recent_burden | -0.005940 | 0.002400 |
| RL | q_redistribution_v4 | beta | 24 | A*walk_interaction_7d | -0.084588 | 0.055796 |

## Joint Modified-TD Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| Joint Modified TD | q_td_modify_joint | eta | 0 | eta_intercept | 0.204206 | 0.135688 |
| Joint Modified TD | q_td_modify_joint | eta | 1 | eta_E_w | -0.016052 | 0.044906 |
| Joint Modified TD | q_td_modify_joint | eta | 2 | eta_b_hat | -0.485075 | 0.072986 |
| Joint Modified TD | q_td_modify_joint | eta | 3 | eta_b_tilde | 0.000000 | 1.000000 |
| Joint Modified TD | q_td_modify_joint | beta | 4 | beta_intercept | 0.220181 | 0.093568 |
| Joint Modified TD | q_td_modify_joint | beta | 5 | beta_weekday_vs_weekend | -0.181288 | 0.189888 |
| Joint Modified TD | q_td_modify_joint | beta | 6 | beta_slot_pm | -0.012871 | 0.003662 |
| Joint Modified TD | q_td_modify_joint | beta | 7 | beta_E_w | -0.017832 | 0.033925 |
| Joint Modified TD | q_td_modify_joint | beta | 8 | beta_b_hat | -0.021339 | 0.042785 |
| Joint Modified TD | q_td_modify_joint | beta | 9 | beta_b_tilde | 0.000000 | 1.000000 |
| Joint Modified TD | q_td_modify_joint | beta | 10 | beta_M_Y_anticipated_affect_ewma | 0.625428 | 0.014710 |
| Joint Modified TD | q_td_modify_joint | beta | 11 | beta_M_Y_fourSC_ewma | 0.023640 | 0.009457 |
| Joint Modified TD | q_td_modify_joint | beta | 12 | beta_M_E_pageview_ewma | 0.008153 | 0.000845 |
| Joint Modified TD | q_td_modify_joint | beta | 13 | beta_M_E_fitbit_wear_ewma | -0.324366 | 0.005111 |
| Joint Modified TD | q_td_modify_joint | beta | 14 | beta_M_E_survey_complete_ewma | -0.032859 | 0.006620 |
| Joint Modified TD | q_td_modify_joint | beta | 15 | beta_yesterday_step_count | 0.003819 | 0.000959 |
| Joint Modified TD | q_td_modify_joint | beta | 16 | beta_prior2hour_step_count | -0.006082 | 0.000872 |
| Joint Modified TD | q_td_modify_joint | beta | 17 | beta_active_status_fraction_7days | 0.011865 | 0.022070 |
| Joint Modified TD | q_td_modify_joint | beta | 18 | beta_recent_burden | -0.000765 | 0.000921 |
| Joint Modified TD | q_td_modify_joint | beta | 19 | beta_walk_interaction_7d | 0.076278 | 0.010927 |
| Joint Modified TD | q_td_modify_joint | beta | 20 | beta_A | -0.010418 | 0.003245 |
| Joint Modified TD | q_td_modify_joint | beta | 21 | beta_A*E_w | 0.000918 | 0.001624 |
| Joint Modified TD | q_td_modify_joint | beta | 22 | beta_A*b_hat | -0.071431 | 0.002707 |
| Joint Modified TD | q_td_modify_joint | beta | 23 | beta_A*b_tilde | 0.000000 | 1.000000 |
| Joint Modified TD | q_td_modify_joint | beta | 24 | beta_A*yesterday_step_count | -0.002578 | 0.003635 |
| Joint Modified TD | q_td_modify_joint | beta | 25 | beta_A*prior2hour_step_count | -0.002197 | 0.001252 |
| Joint Modified TD | q_td_modify_joint | beta | 26 | beta_A*active_status_fraction_7days | 0.003019 | 0.001701 |
| Joint Modified TD | q_td_modify_joint | beta | 27 | beta_A*recent_burden | -0.002056 | 0.001496 |
| Joint Modified TD | q_td_modify_joint | beta | 28 | beta_A*walk_interaction_7d | 0.002096 | 0.002268 |

## Pooled PF Regression Coefficients

Coefficients are from the all-user stacked ridge fits used as PF prior means. `p_value` uses ridge sandwich standard errors and is approximate because ridge shrinks coefficients. `identified=False` marks zero-variance design columns; SE / p-values are omitted for those rows.

| model | outcome | index | feature | coefficient | identified | feature_std | std_error | t_stat | p_value | significant_0.05 | significant_0.01 | n_obs | n_features | ridge_alpha | residual_sigma2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fourSC | 4hour_step_norm | 0 | intercept | -0.054691 | False | 0.000000 |  |  |  | False | False | 3257 | 19 | 1.000000 | 0.504144 |
| fourSC | 4hour_step_norm | 1 | stepCountNext4HourLag1 | 0.478842 | True | 0.934447 | 0.015738 | 30.425409 | 4.708529e-179 | True | True | 3257 | 19 | 1.000000 | 0.504144 |
| fourSC | 4hour_step_norm | 2 | yesterdayStepCount | -0.168025 | True | 0.905403 | 0.021412 | -7.847238 | 5.732136e-15 | True | True | 3257 | 19 | 1.000000 | 0.504144 |
| fourSC | 4hour_step_norm | 3 | stepCountLast7DaysEma | 0.522006 | True | 0.737848 | 0.020058 | 26.024380 | 9.647544e-136 | True | True | 3257 | 19 | 1.000000 | 0.504144 |
| fourSC | 4hour_step_norm | 4 | prior2HourStepCount | 0.104427 | True | 0.938155 | 0.019752 | 5.286970 | 1.326067e-07 | True | True | 3257 | 19 | 1.000000 | 0.504144 |
| fourSC | 4hour_step_norm | 5 | activitySuggestionsSentLast7Days | -0.027834 | True | 0.985431 | 0.018266 | -1.523839 | 0.127646 | False | False | 3257 | 19 | 1.000000 | 0.504144 |
| fourSC | 4hour_step_norm | 6 | activitySuggestionInteractLast7Days | 0.062358 | True | 0.354638 | 0.051315 | 1.215210 | 0.224374 | False | False | 3257 | 19 | 1.000000 | 0.504144 |
| fourSC | 4hour_step_norm | 7 | activeDaysLast7Days | 0.149411 | True | 0.260113 | 0.053315 | 2.802402 | 0.005103 | True | True | 3257 | 19 | 1.000000 | 0.504144 |
| fourSC | 4hour_step_norm | 8 | isWeekend | -0.033029 | True | 0.368554 | 0.033788 | -0.977544 | 0.328373 | False | False | 3257 | 19 | 1.000000 | 0.504144 |
| fourSC | 4hour_step_norm | 9 | decisionTimeSlot | -0.049603 | True | 0.497626 | 0.026753 | -1.854125 | 0.063812 | False | False | 3257 | 19 | 1.000000 | 0.504144 |
| fourSC | 4hour_step_norm | 10 | perceivedUtilityLastWeek | 0.012075 | True | 2.477410 | 0.007778 | 1.552467 | 0.120648 | False | False | 3257 | 19 | 1.000000 | 0.504144 |
| fourSC | 4hour_step_norm | 11 | caeAverageLastWeek | -0.048148 | True | 0.901679 | 0.020516 | -2.346908 | 0.018990 | True | False | 3257 | 19 | 1.000000 | 0.504144 |
| fourSC | 4hour_step_norm | 12 | Ah | -0.093421 | True | 0.499778 | 0.043498 | -2.147700 | 0.031812 | True | False | 3257 | 19 | 1.000000 | 0.504144 |
| fourSC | 4hour_step_norm | 13 | Ah*yesterdayStepCount | 0.004651 | True | 0.645325 | 0.028606 | 0.162584 | 0.870856 | False | False | 3257 | 19 | 1.000000 | 0.504144 |
| fourSC | 4hour_step_norm | 14 | Ah*prior2HourStepCount | 0.012977 | True | 0.662143 | 0.027485 | 0.472131 | 0.636865 | False | False | 3257 | 19 | 1.000000 | 0.504144 |
| fourSC | 4hour_step_norm | 15 | Ah*activitySuggestionsSentLast7Days | 0.024239 | True | 0.699765 | 0.025717 | 0.942524 | 0.345995 | False | False | 3257 | 19 | 1.000000 | 0.504144 |
| fourSC | 4hour_step_norm | 16 | Ah*activitySuggestionInteractLast7Days | 0.106010 | True | 0.354630 | 0.071356 | 1.485639 | 0.137472 | False | False | 3257 | 19 | 1.000000 | 0.504144 |
| fourSC | 4hour_step_norm | 17 | Ah*perceivedUtilityLastWeek | 0.008482 | True | 1.828147 | 0.010585 | 0.801280 | 0.423028 | False | False | 3257 | 19 | 1.000000 | 0.504144 |
| fourSC | 4hour_step_norm | 18 | Ah*caeAverageLastWeek | 0.016509 | True | 0.628671 | 0.028183 | 0.585781 | 0.558063 | False | False | 3257 | 19 | 1.000000 | 0.504144 |
| antic | anticipated_affect_norm | 0 | intercept | 0.446849 | False | 0.000000 |  |  |  | False | False | 935 | 15 | 1.000000 | 0.038802 |
| antic | anticipated_affect_norm | 1 | anticipated_affect_yesterday | 0.181563 | True | 0.352394 | 0.021851 | 8.309260 | 3.428992e-16 | True | True | 935 | 15 | 1.000000 | 0.038802 |
| antic | anticipated_affect_norm | 2 | active_status_fraction_7days | 0.073015 | True | 0.223154 | 0.030298 | 2.409879 | 0.016153 | True | False | 935 | 15 | 1.000000 | 0.038802 |
| antic | anticipated_affect_norm | 3 | is_weekend | -0.023439 | True | 0.371879 | 0.017472 | -1.341561 | 0.180070 | False | False | 935 | 15 | 1.000000 | 0.038802 |
| antic | anticipated_affect_norm | 4 | perceived_utility_lastweek | 0.004530 | True | 2.516622 | 0.004364 | 1.038087 | 0.299502 | False | False | 935 | 15 | 1.000000 | 0.038802 |
| antic | anticipated_affect_norm | 5 | CAE_avg_lastweek | 0.247484 | True | 0.867434 | 0.013719 | 18.038987 | 1.620189e-62 | True | True | 935 | 15 | 1.000000 | 0.038802 |
| antic | anticipated_affect_norm | 6 | recent_burden | 0.020060 | True | 0.985297 | 0.009913 | 2.023518 | 0.043308 | True | False | 935 | 15 | 1.000000 | 0.038802 |
| antic | anticipated_affect_norm | 7 | A0_morning | 0.008973 | True | 0.499519 | 0.014487 | 0.619366 | 0.535829 | False | False | 935 | 15 | 1.000000 | 0.038802 |
| antic | anticipated_affect_norm | 8 | A1_afternoon | 0.019286 | True | 0.499917 | 0.014448 | 1.334832 | 0.182261 | False | False | 935 | 15 | 1.000000 | 0.038802 |
| antic | anticipated_affect_norm | 9 | A0_morning_by_perceived_utility_lastweek | -0.001815 | True | 1.882937 | 0.005476 | -0.331444 | 0.740385 | False | False | 935 | 15 | 1.000000 | 0.038802 |
| antic | anticipated_affect_norm | 10 | A1_afternoon_by_perceived_utility_lastweek | -0.005232 | True | 1.886348 | 0.005493 | -0.952512 | 0.341088 | False | False | 935 | 15 | 1.000000 | 0.038802 |
| antic | anticipated_affect_norm | 11 | A0_morning_by_CAE_avg_lastweek | 0.000862 | True | 0.636251 | 0.015276 | 0.056443 | 0.955001 | False | False | 935 | 15 | 1.000000 | 0.038802 |
| antic | anticipated_affect_norm | 12 | A1_afternoon_by_CAE_avg_lastweek | -0.025406 | True | 0.610267 | 0.015260 | -1.664893 | 0.096275 | False | False | 935 | 15 | 1.000000 | 0.038802 |
| antic | anticipated_affect_norm | 13 | A0_morning_by_recent_burden | -0.013469 | True | 0.681543 | 0.014998 | -0.898073 | 0.369382 | False | False | 935 | 15 | 1.000000 | 0.038802 |
| antic | anticipated_affect_norm | 14 | A1_afternoon_by_recent_burden | 0.000301 | True | 0.672785 | 0.014961 | 0.020138 | 0.983937 | False | False | 935 | 15 | 1.000000 | 0.038802 |
| CAE | CAE_avg_norm | 0 | intercept | -0.103666 | False | 0.000000 |  |  |  | False | False | 241 | 4 | 1.000000 | 0.120903 |
| CAE | CAE_avg_norm | 1 | CAE_avg_lastweek | 0.903687 | True | 0.927401 | 0.032987 | 27.395467 | 2.125965e-75 | True | True | 241 | 4 | 1.000000 | 0.120903 |
| CAE | CAE_avg_norm | 2 | fourSC_ewma | -0.018991 | True | 0.528050 | 0.042028 | -0.451870 | 0.651776 | False | False | 241 | 4 | 1.000000 | 0.120903 |
| CAE | CAE_avg_norm | 3 | anticipated_affect_ewma | 0.162856 | True | 0.345908 | 0.085078 | 1.914193 | 0.056800 | False | False | 241 | 4 | 1.000000 | 0.120903 |

## Pooled PF GEE Coefficients

Population-averaged GEE fits on the same stacked PF designs, clustered by `ParticipantIdentifier`. `p_value` uses GEE sandwich standard errors with exchangeable working correlation. `identified=False` marks zero-variance design columns; SE / p-values are omitted for those rows. These are for inference/audit only; PF priors still use ridge.

| coefficient | robust_se | z_stat | p_value | significant_0.05 | significant_0.01 | model | outcome | index | feature | identified | feature_std | n_obs | n_clusters | n_features | working_correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -0.095067 |  |  |  | False | False | fourSC | 4hour_step_norm | 0 | intercept | False | 0.000000 | 3257 | 31 | 19 | exchangeable |
| -0.047291 | 0.040887 | -1.156637 | 0.247421 | False | False | fourSC | 4hour_step_norm | 1 | stepCountNext4HourLag1 | True | 0.934447 | 3257 | 31 | 19 | exchangeable |
| -0.021553 | 0.029250 | -0.736865 | 0.461205 | False | False | fourSC | 4hour_step_norm | 2 | yesterdayStepCount | True | 0.905403 | 3257 | 31 | 19 | exchangeable |
| 0.270050 | 0.046013 | 5.869031 | 4.383504e-09 | True | True | fourSC | 4hour_step_norm | 3 | stepCountLast7DaysEma | True | 0.737848 | 3257 | 31 | 19 | exchangeable |
| 0.138831 | 0.031081 | 4.466768 | 7.941028e-06 | True | True | fourSC | 4hour_step_norm | 4 | prior2HourStepCount | True | 0.938155 | 3257 | 31 | 19 | exchangeable |
| 0.006945 | 0.014860 | 0.467390 | 0.640221 | False | False | fourSC | 4hour_step_norm | 5 | activitySuggestionsSentLast7Days | True | 0.985431 | 3257 | 31 | 19 | exchangeable |
| 0.059435 | 0.062815 | 0.946195 | 0.344049 | False | False | fourSC | 4hour_step_norm | 6 | activitySuggestionInteractLast7Days | True | 0.354638 | 3257 | 31 | 19 | exchangeable |
| 0.018708 | 0.069440 | 0.269410 | 0.787614 | False | False | fourSC | 4hour_step_norm | 7 | activeDaysLast7Days | True | 0.260113 | 3257 | 31 | 19 | exchangeable |
| -0.002585 | 0.040738 | -0.063443 | 0.949414 | False | False | fourSC | 4hour_step_norm | 8 | isWeekend | True | 0.368554 | 3257 | 31 | 19 | exchangeable |
| -0.086794 | 0.057249 | -1.516092 | 0.129496 | False | False | fourSC | 4hour_step_norm | 9 | decisionTimeSlot | True | 0.497626 | 3257 | 31 | 19 | exchangeable |
| 0.022067 | 0.014651 | 1.506152 | 0.132028 | False | False | fourSC | 4hour_step_norm | 10 | perceivedUtilityLastWeek | True | 2.477410 | 3257 | 31 | 19 | exchangeable |
| 0.005144 | 0.032483 | 0.158345 | 0.874185 | False | False | fourSC | 4hour_step_norm | 11 | caeAverageLastWeek | True | 0.901679 | 3257 | 31 | 19 | exchangeable |
| -0.033778 | 0.050480 | -0.669131 | 0.503412 | False | False | fourSC | 4hour_step_norm | 12 | Ah | True | 0.499778 | 3257 | 31 | 19 | exchangeable |
| 0.014468 | 0.032036 | 0.451607 | 0.651552 | False | False | fourSC | 4hour_step_norm | 13 | Ah*yesterdayStepCount | True | 0.645325 | 3257 | 31 | 19 | exchangeable |
| -0.016140 | 0.027701 | -0.582671 | 0.560115 | False | False | fourSC | 4hour_step_norm | 14 | Ah*prior2HourStepCount | True | 0.662143 | 3257 | 31 | 19 | exchangeable |
| 0.005741 | 0.014226 | 0.403569 | 0.686530 | False | False | fourSC | 4hour_step_norm | 15 | Ah*activitySuggestionsSentLast7Days | True | 0.699765 | 3257 | 31 | 19 | exchangeable |
| 0.038880 | 0.075356 | 0.515956 | 0.605885 | False | False | fourSC | 4hour_step_norm | 16 | Ah*activitySuggestionInteractLast7Days | True | 0.354630 | 3257 | 31 | 19 | exchangeable |
| 0.000738 | 0.009040 | 0.081623 | 0.934947 | False | False | fourSC | 4hour_step_norm | 17 | Ah*perceivedUtilityLastWeek | True | 1.828147 | 3257 | 31 | 19 | exchangeable |
| 0.000243 | 0.025790 | 0.009409 | 0.992493 | False | False | fourSC | 4hour_step_norm | 18 | Ah*caeAverageLastWeek | True | 0.628671 | 3257 | 31 | 19 | exchangeable |
| 0.468464 |  |  |  | False | False | antic | anticipated_affect_norm | 0 | intercept | False | 0.000000 | 935 | 31 | 15 | exchangeable |
| 0.052940 | 0.028780 | 1.839485 | 0.065844 | False | False | antic | anticipated_affect_norm | 1 | anticipated_affect_yesterday | True | 0.352394 | 935 | 31 | 15 | exchangeable |
| 0.112895 | 0.035917 | 3.143265 | 0.001671 | True | True | antic | anticipated_affect_norm | 2 | active_status_fraction_7days | True | 0.223154 | 935 | 31 | 15 | exchangeable |
| 3.301554e-05 | 0.019022 | 0.001736 | 0.998615 | False | False | antic | anticipated_affect_norm | 3 | is_weekend | True | 0.371879 | 935 | 31 | 15 | exchangeable |
| 0.002094 | 0.006660 | 0.314371 | 0.753239 | False | False | antic | anticipated_affect_norm | 4 | perceived_utility_lastweek | True | 2.516622 | 935 | 31 | 15 | exchangeable |
| 0.133828 | 0.039467 | 3.390866 | 0.000697 | True | True | antic | anticipated_affect_norm | 5 | CAE_avg_lastweek | True | 0.867434 | 935 | 31 | 15 | exchangeable |
| 0.013044 | 0.005457 | 2.390213 | 0.016839 | True | False | antic | anticipated_affect_norm | 6 | recent_burden | True | 0.985297 | 935 | 31 | 15 | exchangeable |
| 0.005645 | 0.008511 | 0.663317 | 0.507127 | False | False | antic | anticipated_affect_norm | 7 | A0_morning | True | 0.499519 | 935 | 31 | 15 | exchangeable |
| 0.013991 | 0.009110 | 1.535825 | 0.124581 | False | False | antic | anticipated_affect_norm | 8 | A1_afternoon | True | 0.499917 | 935 | 31 | 15 | exchangeable |
| 0.000592 | 0.002550 | 0.232155 | 0.816417 | False | False | antic | anticipated_affect_norm | 9 | A0_morning_by_perceived_utility_lastweek | True | 1.882937 | 935 | 31 | 15 | exchangeable |
| -0.002871 | 0.003506 | -0.819045 | 0.412761 | False | False | antic | anticipated_affect_norm | 10 | A1_afternoon_by_perceived_utility_lastweek | True | 1.886348 | 935 | 31 | 15 | exchangeable |
| 0.014540 | 0.009975 | 1.457639 | 0.144940 | False | False | antic | anticipated_affect_norm | 11 | A0_morning_by_CAE_avg_lastweek | True | 0.636251 | 935 | 31 | 15 | exchangeable |
| -0.022176 | 0.013583 | -1.632665 | 0.102540 | False | False | antic | anticipated_affect_norm | 12 | A1_afternoon_by_CAE_avg_lastweek | True | 0.610267 | 935 | 31 | 15 | exchangeable |
| -0.011357 | 0.010821 | -1.049567 | 0.293917 | False | False | antic | anticipated_affect_norm | 13 | A0_morning_by_recent_burden | True | 0.681543 | 935 | 31 | 15 | exchangeable |
| -0.013490 | 0.011033 | -1.222659 | 0.221459 | False | False | antic | anticipated_affect_norm | 14 | A1_afternoon_by_recent_burden | True | 0.672785 | 935 | 31 | 15 | exchangeable |
| -0.072511 |  |  |  | False | False | CAE | CAE_avg_norm | 0 | intercept | False | 0.000000 | 241 | 31 | 4 | exchangeable |
| 0.956309 | 0.033278 | 28.736984 | 1.317294e-181 | True | True | CAE | CAE_avg_norm | 1 | CAE_avg_lastweek | True | 0.927401 | 241 | 31 | 4 | exchangeable |
| -0.015923 | 0.024038 | -0.662399 | 0.507716 | False | False | CAE | CAE_avg_norm | 2 | fourSC_ewma | True | 0.528050 | 241 | 31 | 4 | exchangeable |
| 0.103799 | 0.088167 | 1.177301 | 0.239075 | False | False | CAE | CAE_avg_norm | 3 | anticipated_affect_ewma | True | 0.345908 | 241 | 31 | 4 | exchangeable |

## Pooled RL Q GEE Coefficients

Gaussian GEE on the final fitted-Q regression from pooled FQI (``phi_obs`` vs bootstrap targets), clustered by participant. ``identified=False`` marks structurally unused features with zero design variance (e.g. ``b_tilde``, masked day-6 mediators); SE / p-values are omitted for those rows. RL priors still use ridge-FQI for ``mu_0_micro``; all-zero design columns get ``UNIDENTIFIED_PRIOR_VAR`` instead of the ``MIN_SIGMA2`` floor.

| coefficient | robust_se | z_stat | p_value | significant_0.05 | significant_0.01 | model | outcome | index | feature | identified | feature_std | n_obs | n_clusters | n_features | working_correlation | fqi_iters | ridge_alpha | residual_sigma2 | block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.068450 |  |  |  | False | False | q_no_td_modify | fqi_target | 0 | intercept | False | 0.000000 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| -0.088999 | 0.030215 | -2.945501 | 0.003224 | True | True | q_no_td_modify | fqi_target | 1 | weekday_vs_weekend | True | 0.372678 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| -0.012982 | 0.011577 | -1.121357 | 0.262136 | False | False | q_no_td_modify | fqi_target | 2 | slot_pm | True | 0.500000 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| 0.001285 | 0.001174 | 1.094566 | 0.273707 | False | False | q_no_td_modify | fqi_target | 3 | E_w | True | 2.454586 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| 1.178422 | 0.006748 | 174.645340 | 0.000000 | True | True | q_no_td_modify | fqi_target | 4 | b_hat | True | 0.871522 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| 1.158839e-16 |  |  |  | False | False | q_no_td_modify | fqi_target | 5 | b_tilde | False | 0.000000 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| 0.220115 | 0.012965 | 16.978296 | 1.188887e-64 | True | True | q_no_td_modify | fqi_target | 6 | M_Y_anticipated_affect_ewma | True | 0.325774 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| -0.010094 | 0.001377 | -7.328714 | 2.323721e-13 | True | True | q_no_td_modify | fqi_target | 7 | M_Y_fourSC_ewma | True | 0.963986 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| 0.001600 | 0.001728 | 0.926176 | 0.354354 | False | False | q_no_td_modify | fqi_target | 8 | M_E_pageview_ewma | True | 1.038492 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| -0.108541 | 0.005003 | -21.694653 | 2.304581e-104 | True | True | q_no_td_modify | fqi_target | 9 | M_E_fitbit_wear_ewma | True | 0.410788 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| -0.041561 | 0.005311 | -7.825034 | 5.075206e-15 | True | True | q_no_td_modify | fqi_target | 10 | M_E_survey_complete_ewma | True | 0.415970 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| -0.001747 | 0.004871 | -0.358689 | 0.719827 | False | False | q_no_td_modify | fqi_target | 11 | yesterday_step_count | True | 0.896241 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| -0.011718 | 0.004377 | -2.677262 | 0.007423 | True | True | q_no_td_modify | fqi_target | 12 | prior2hour_step_count | True | 0.928929 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| -0.000425 | 0.010417 | -0.040838 | 0.967425 | False | False | q_no_td_modify | fqi_target | 13 | active_status_fraction_7days | True | 0.265779 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| -0.003214 | 0.003471 | -0.925860 | 0.354519 | False | False | q_no_td_modify | fqi_target | 14 | recent_burden | True | 0.950687 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| -0.017116 | 0.007902 | -2.165933 | 0.030316 | True | False | q_no_td_modify | fqi_target | 15 | walk_interaction_7d | True | 0.357352 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| -0.005234 | 0.015798 | -0.331294 | 0.740422 | False | False | q_no_td_modify | fqi_target | 16 | A | True | 0.499695 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| -0.000312 | 0.001701 | -0.183476 | 0.854424 | False | False | q_no_td_modify | fqi_target | 17 | A*E_w | True | 1.810391 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| -0.004974 | 0.011352 | -0.438218 | 0.661228 | False | False | q_no_td_modify | fqi_target | 18 | A*b_hat | True | 0.614356 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| -2.249112e-17 |  |  |  | False | False | q_no_td_modify | fqi_target | 19 | A*b_tilde | False | 0.000000 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| -0.006799 | 0.007695 | -0.883584 | 0.376921 | False | False | q_no_td_modify | fqi_target | 20 | A*yesterday_step_count | True | 0.642142 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| 0.004992 | 0.004877 | 1.023570 | 0.306038 | False | False | q_no_td_modify | fqi_target | 21 | A*prior2hour_step_count | True | 0.655751 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| 0.008096 | 0.018887 | 0.428675 | 0.668160 | False | False | q_no_td_modify | fqi_target | 22 | A*active_status_fraction_7days | True | 0.385058 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| -0.003554 | 0.004040 | -0.879833 | 0.378950 | False | False | q_no_td_modify | fqi_target | 23 | A*recent_burden | True | 0.681276 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
| 0.004886 | 0.012975 | 0.376605 | 0.706467 | False | False | q_no_td_modify | fqi_target | 24 | A*walk_interaction_7d | True | 0.347052 | 3720 | 31 | 25 | exchangeable | 25 | 1.000000 | 0.024650 | beta |
