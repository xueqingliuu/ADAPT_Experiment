# Prior Mean and Variance Summary

`prior_variance` is the diagonal entry of the prior covariance. For `q_td_modify_joint`, the full covariance remains in `rl_priors.json`; this table reports marginal variances.

## PF Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| PF | fourSC | theta | 0 | intercept | -0.077843 | 0.052657 |
| PF | fourSC | theta | 1 | stepCountNext4HourLag1 | 0.486631 | 0.044052 |
| PF | fourSC | theta | 2 | yesterdayStepCount | -0.148510 | 0.010915 |
| PF | fourSC | theta | 3 | stepCountLast7DaysEma | 0.358334 | 0.034164 |
| PF | fourSC | theta | 4 | prior2HourStepCount | 0.097988 | 0.033684 |
| PF | fourSC | theta | 5 | activitySuggestionsSentLast7Days | -0.037568 | 0.027602 |
| PF | fourSC | theta | 6 | activitySuggestionInteractLast7Days | 0.037534 | 0.064859 |
| PF | fourSC | theta | 7 | activeDaysLast7Days | 0.157462 | 0.042702 |
| PF | fourSC | theta | 8 | isWeekend | -0.034017 | 0.046565 |
| PF | fourSC | theta | 9 | decisionTimeSlot | -0.034384 | 0.143728 |
| PF | fourSC | theta | 10 | perceivedUtilityLastWeek | 0.020972 | 0.014344 |
| PF | fourSC | theta | 11 | caeAverageLastWeek | -0.053702 | 0.098342 |
| PF | fourSC | theta | 12 | Ah | -0.116266 | 0.075658 |
| PF | fourSC | theta | 13 | Ah*yesterdayStepCount | 0.005743 | 0.018772 |
| PF | fourSC | theta | 14 | Ah*prior2HourStepCount | 0.009660 | 0.023679 |
| PF | fourSC | theta | 15 | Ah*activitySuggestionsSentLast7Days | 0.029985 | 0.024949 |
| PF | fourSC | theta | 16 | Ah*activitySuggestionInteractLast7Days | 0.087900 | 0.103011 |
| PF | fourSC | theta | 17 | Ah*perceivedUtilityLastWeek | 0.020755 | 0.022559 |
| PF | fourSC | theta | 18 | Ah*caeAverageLastWeek | -0.014121 | 0.076718 |
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
| RL | redistribution_stage1_AA | eta | 0 | redistribution_stage1_AA_coef_0 | 0.222874 | 0.007288 |
| RL | redistribution_stage1_AA | eta | 1 | redistribution_stage1_AA_coef_1 | 0.001377 | 0.001146 |
| RL | redistribution_stage1_AA | eta | 2 | redistribution_stage1_AA_coef_2 | 0.084417 | 0.002518 |
| RL | redistribution_stage1_AA | eta | 3 | redistribution_stage1_AA_coef_3 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_AA | eta | 4 | redistribution_stage1_AA_coef_4 | 0.173730 | 0.002020 |
| RL | redistribution_stage1_AA | eta | 5 | redistribution_stage1_AA_coef_5 | -0.011266 | 0.003613 |
| RL | redistribution_stage1_AA | eta | 6 | redistribution_stage1_AA_coef_6 | 0.013090 | 0.000507 |
| RL | redistribution_stage1_AA | eta | 7 | redistribution_stage1_AA_coef_7 | -0.033435 | 0.001449 |
| RL | redistribution_stage1_AA | eta | 8 | redistribution_stage1_AA_coef_8 | -0.027310 | 0.001645 |
| RL | redistribution_stage1_AA | eta | 9 | redistribution_stage1_AA_coef_9 | 0.007289 | 0.000214 |
| RL | redistribution_stage1_AA | eta | 10 | redistribution_stage1_AA_coef_10 | 0.010540 | 0.000374 |
| RL | redistribution_stage1_AA | eta | 11 | redistribution_stage1_AA_coef_11 | 0.025573 | 0.004570 |
| RL | redistribution_stage1_AA | eta | 12 | redistribution_stage1_AA_coef_12 | 0.000339 | 0.000128 |
| RL | redistribution_stage1_AA | eta | 13 | redistribution_stage1_AA_coef_13 | -0.003126 | 0.002941 |
| RL | redistribution_stage1_AA | eta | 14 | redistribution_stage1_AA_coef_14 | 0.011233 | 0.001336 |
| RL | redistribution_stage1_AA | eta | 15 | redistribution_stage1_AA_coef_15 | 3.110337e-05 | 0.000425 |
| RL | redistribution_stage1_AA | eta | 16 | redistribution_stage1_AA_coef_16 | -0.007456 | 0.002126 |
| RL | redistribution_stage1_AA | eta | 17 | redistribution_stage1_AA_coef_17 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_AA | eta | 18 | redistribution_stage1_AA_coef_18 | -0.000295 | 0.000612 |
| RL | redistribution_stage1_AA | eta | 19 | redistribution_stage1_AA_coef_19 | 0.008142 | 0.000402 |
| RL | redistribution_stage1_AA | eta | 20 | redistribution_stage1_AA_coef_20 | -0.008475 | 0.002592 |
| RL | redistribution_stage1_AA | eta | 21 | redistribution_stage1_AA_coef_21 | 0.003168 | 0.000300 |
| RL | redistribution_stage1_AA | eta | 22 | redistribution_stage1_AA_coef_22 | -0.000629 | 0.000762 |
| RL | redistribution_stage1_FW | eta | 0 | redistribution_stage1_FW_coef_0 | 0.363291 | 0.024664 |
| RL | redistribution_stage1_FW | eta | 1 | redistribution_stage1_FW_coef_1 | 0.006563 | 0.005946 |
| RL | redistribution_stage1_FW | eta | 2 | redistribution_stage1_FW_coef_2 | 0.000760 | 0.008520 |
| RL | redistribution_stage1_FW | eta | 3 | redistribution_stage1_FW_coef_3 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_FW | eta | 4 | redistribution_stage1_FW_coef_4 | -0.122290 | 0.028653 |
| RL | redistribution_stage1_FW | eta | 5 | redistribution_stage1_FW_coef_5 | -0.029540 | 0.009999 |
| RL | redistribution_stage1_FW | eta | 6 | redistribution_stage1_FW_coef_6 | 0.028648 | 0.001500 |
| RL | redistribution_stage1_FW | eta | 7 | redistribution_stage1_FW_coef_7 | 0.160463 | 0.017493 |
| RL | redistribution_stage1_FW | eta | 8 | redistribution_stage1_FW_coef_8 | -0.011918 | 0.009944 |
| RL | redistribution_stage1_FW | eta | 9 | redistribution_stage1_FW_coef_9 | 0.004154 | 0.003503 |
| RL | redistribution_stage1_FW | eta | 10 | redistribution_stage1_FW_coef_10 | 0.004761 | 0.002673 |
| RL | redistribution_stage1_FW | eta | 11 | redistribution_stage1_FW_coef_11 | 0.045122 | 0.016601 |
| RL | redistribution_stage1_FW | eta | 12 | redistribution_stage1_FW_coef_12 | -0.010701 | 0.003106 |
| RL | redistribution_stage1_FW | eta | 13 | redistribution_stage1_FW_coef_13 | 0.027637 | 0.025439 |
| RL | redistribution_stage1_FW | eta | 14 | redistribution_stage1_FW_coef_14 | 0.019529 | 0.010590 |
| RL | redistribution_stage1_FW | eta | 15 | redistribution_stage1_FW_coef_15 | -0.002129 | 0.006144 |
| RL | redistribution_stage1_FW | eta | 16 | redistribution_stage1_FW_coef_16 | -0.005741 | 0.014831 |
| RL | redistribution_stage1_FW | eta | 17 | redistribution_stage1_FW_coef_17 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_FW | eta | 18 | redistribution_stage1_FW_coef_18 | -0.001101 | 0.007624 |
| RL | redistribution_stage1_FW | eta | 19 | redistribution_stage1_FW_coef_19 | -0.002334 | 0.005002 |
| RL | redistribution_stage1_FW | eta | 20 | redistribution_stage1_FW_coef_20 | -0.024201 | 0.022744 |
| RL | redistribution_stage1_FW | eta | 21 | redistribution_stage1_FW_coef_21 | 0.006174 | 0.003980 |
| RL | redistribution_stage1_FW | eta | 22 | redistribution_stage1_FW_coef_22 | -0.013116 | 0.023700 |
| RL | redistribution_stage1_PJ | eta | 0 | redistribution_stage1_PJ_coef_0 | 0.104080 | 0.036917 |
| RL | redistribution_stage1_PJ | eta | 1 | redistribution_stage1_PJ_coef_1 | 0.004667 | 0.007817 |
| RL | redistribution_stage1_PJ | eta | 2 | redistribution_stage1_PJ_coef_2 | 0.020230 | 0.020094 |
| RL | redistribution_stage1_PJ | eta | 3 | redistribution_stage1_PJ_coef_3 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_PJ | eta | 4 | redistribution_stage1_PJ_coef_4 | -0.139508 | 0.031186 |
| RL | redistribution_stage1_PJ | eta | 5 | redistribution_stage1_PJ_coef_5 | -0.016799 | 0.022434 |
| RL | redistribution_stage1_PJ | eta | 6 | redistribution_stage1_PJ_coef_6 | 0.013145 | 0.002327 |
| RL | redistribution_stage1_PJ | eta | 7 | redistribution_stage1_PJ_coef_7 | 0.015723 | 0.007224 |
| RL | redistribution_stage1_PJ | eta | 8 | redistribution_stage1_PJ_coef_8 | 0.160158 | 0.011900 |
| RL | redistribution_stage1_PJ | eta | 9 | redistribution_stage1_PJ_coef_9 | 0.001081 | 0.007660 |
| RL | redistribution_stage1_PJ | eta | 10 | redistribution_stage1_PJ_coef_10 | 0.024062 | 0.002571 |
| RL | redistribution_stage1_PJ | eta | 11 | redistribution_stage1_PJ_coef_11 | 0.074192 | 0.033846 |
| RL | redistribution_stage1_PJ | eta | 12 | redistribution_stage1_PJ_coef_12 | 0.014299 | 0.005121 |
| RL | redistribution_stage1_PJ | eta | 13 | redistribution_stage1_PJ_coef_13 | 0.187313 | 0.024023 |
| RL | redistribution_stage1_PJ | eta | 14 | redistribution_stage1_PJ_coef_14 | 0.041669 | 0.011210 |
| RL | redistribution_stage1_PJ | eta | 15 | redistribution_stage1_PJ_coef_15 | 0.005630 | 0.008859 |
| RL | redistribution_stage1_PJ | eta | 16 | redistribution_stage1_PJ_coef_16 | 0.008554 | 0.023867 |
| RL | redistribution_stage1_PJ | eta | 17 | redistribution_stage1_PJ_coef_17 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_PJ | eta | 18 | redistribution_stage1_PJ_coef_18 | 0.011872 | 0.013299 |
| RL | redistribution_stage1_PJ | eta | 19 | redistribution_stage1_PJ_coef_19 | -0.002359 | 0.010828 |
| RL | redistribution_stage1_PJ | eta | 20 | redistribution_stage1_PJ_coef_20 | -0.059199 | 0.031600 |
| RL | redistribution_stage1_PJ | eta | 21 | redistribution_stage1_PJ_coef_21 | -0.006653 | 0.010222 |
| RL | redistribution_stage1_PJ | eta | 22 | redistribution_stage1_PJ_coef_22 | -0.050083 | 0.020368 |
| RL | redistribution_stage2_v2 | eta | 0 | redistribution_stage2_v2_coef_0 | -0.574256 | 0.001245 |
| RL | redistribution_stage2_v2 | eta | 1 | redistribution_stage2_v2_coef_1 | -0.095709 | 3.457090e-05 |
| RL | redistribution_stage2_v2 | eta | 2 | redistribution_stage2_v2_coef_2 | 0.034147 | 0.000922 |
| RL | redistribution_stage2_v2 | eta | 3 | redistribution_stage2_v2_coef_3 | -0.085433 | 0.001618 |
| RL | redistribution_stage2_v2 | eta | 4 | redistribution_stage2_v2_coef_4 | 0.000000 | 1.000000 |
| RL | redistribution_stage2_v2 | eta | 5 | redistribution_stage2_v2_coef_5 | 0.627728 | 0.003086 |
| RL | redistribution_stage2_v2 | eta | 6 | redistribution_stage2_v2_coef_6 | 0.043908 | 0.002630 |
| RL | redistribution_stage2_v2 | eta | 7 | redistribution_stage2_v2_coef_7 | -0.084152 | 0.002041 |
| RL | redistribution_stage2_v2 | eta | 8 | redistribution_stage2_v2_coef_8 | -0.121924 | 0.001843 |
| RL | redistribution_stage2_v2 | eta | 9 | redistribution_stage2_v2_coef_9 | -0.128397 | 0.002562 |
| RL | redistribution_stage2_v2 | eta | 10 | redistribution_stage2_v2_coef_10 | 0.036104 | 0.002357 |
| RL | redistribution_stage2_v2 | eta | 11 | redistribution_stage2_v2_coef_11 | -0.060965 | 0.001160 |
| RL | redistribution_stage2_v2 | eta | 12 | redistribution_stage2_v2_coef_12 | -0.048044 | 0.001228 |
| RL | redistribution_stage2_v2 | eta | 13 | redistribution_stage2_v2_coef_13 | 0.010080 | 0.000345 |
| RL | redistribution_stage2_v2 | eta | 14 | redistribution_stage2_v2_coef_14 | -0.030650 | 0.004101 |
| RL | redistribution_stage2_v2 | eta | 15 | redistribution_stage2_v2_coef_15 | -0.028293 | 0.001255 |
| RL | redistribution_stage2_v2 | eta | 16 | redistribution_stage2_v2_coef_16 | 0.045306 | 0.002303 |
| RL | redistribution_stage2_v2 | eta | 17 | redistribution_stage2_v2_coef_17 | 0.340279 | 0.000181 |
| RL | redistribution_stage2_v2 | eta | 18 | redistribution_stage2_v2_coef_18 | 0.482604 | 0.000431 |
| RL | redistribution_stage2_v2 | eta | 19 | redistribution_stage2_v2_coef_19 | 0.832170 | 0.000353 |
| RL | redistribution_stage2_v4 | eta | 0 | redistribution_stage2_v4_coef_0 | -0.373170 | 0.005790 |
| RL | redistribution_stage2_v4 | eta | 1 | redistribution_stage2_v4_coef_1 | -0.062195 | 0.000161 |
| RL | redistribution_stage2_v4 | eta | 2 | redistribution_stage2_v4_coef_2 | -0.031061 | 0.002459 |
| RL | redistribution_stage2_v4 | eta | 3 | redistribution_stage2_v4_coef_3 | 0.021286 | 0.018858 |
| RL | redistribution_stage2_v4 | eta | 4 | redistribution_stage2_v4_coef_4 | 0.000000 | 1.000000 |
| RL | redistribution_stage2_v4 | eta | 5 | redistribution_stage2_v4_coef_5 | 0.373639 | 0.010722 |
| RL | redistribution_stage2_v4 | eta | 6 | redistribution_stage2_v4_coef_6 | 0.098946 | 0.039202 |
| RL | redistribution_stage2_v4 | eta | 7 | redistribution_stage2_v4_coef_7 | -0.050662 | 0.005354 |
| RL | redistribution_stage2_v4 | eta | 8 | redistribution_stage2_v4_coef_8 | -0.086523 | 0.004936 |
| RL | redistribution_stage2_v4 | eta | 9 | redistribution_stage2_v4_coef_9 | -0.290615 | 0.020952 |
| RL | redistribution_stage2_v4 | eta | 10 | redistribution_stage2_v4_coef_10 | -0.011133 | 0.016220 |
| RL | redistribution_stage2_v4 | eta | 11 | redistribution_stage2_v4_coef_11 | -0.068862 | 0.005327 |
| RL | redistribution_stage2_v4 | eta | 12 | redistribution_stage2_v4_coef_12 | -0.083275 | 0.006437 |
| RL | redistribution_stage2_v4 | eta | 13 | redistribution_stage2_v4_coef_13 | -0.013496 | 0.003039 |
| RL | redistribution_stage2_v4 | eta | 14 | redistribution_stage2_v4_coef_14 | -0.294472 | 0.016041 |
| RL | redistribution_stage2_v4 | eta | 15 | redistribution_stage2_v4_coef_15 | -0.042654 | 0.028818 |
| RL | redistribution_stage2_v4 | eta | 16 | redistribution_stage2_v4_coef_16 | 0.016370 | 0.011146 |
| RL | redistribution_stage2_v4 | eta | 17 | redistribution_stage2_v4_coef_17 | 0.023226 | 0.001051 |
| RL | redistribution_stage2_v4 | eta | 18 | redistribution_stage2_v4_coef_18 | 0.312929 | 0.001608 |
| RL | redistribution_stage2_v4 | eta | 19 | redistribution_stage2_v4_coef_19 | 1.838330 | 0.002006 |
| RL | q_no_td_modify | beta | 0 | intercept | 0.095386 | 0.086030 |
| RL | q_no_td_modify | beta | 1 | weekday_vs_weekend | -0.047426 | 0.090180 |
| RL | q_no_td_modify | beta | 2 | slot_pm | -0.002001 | 0.002037 |
| RL | q_no_td_modify | beta | 3 | E_w | 0.022913 | 0.040842 |
| RL | q_no_td_modify | beta | 4 | b_hat | 1.151216 | 0.039430 |
| RL | q_no_td_modify | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify | beta | 6 | M_Y_anticipated_affect_ewma | 0.296152 | 0.008652 |
| RL | q_no_td_modify | beta | 7 | M_Y_fourSC_ewma | -0.023823 | 0.006627 |
| RL | q_no_td_modify | beta | 8 | M_E_pageview_ewma | 0.002285 | 0.000247 |
| RL | q_no_td_modify | beta | 9 | M_E_fitbit_wear_ewma | -0.134112 | 0.003994 |
| RL | q_no_td_modify | beta | 10 | M_E_survey_complete_ewma | -0.043636 | 0.004812 |
| RL | q_no_td_modify | beta | 11 | yesterday_step_count | -0.005347 | 0.001208 |
| RL | q_no_td_modify | beta | 12 | prior2hour_step_count | -0.010379 | 0.000664 |
| RL | q_no_td_modify | beta | 13 | active_status_fraction_7days | -0.027710 | 0.025667 |
| RL | q_no_td_modify | beta | 14 | recent_burden | -0.012425 | 0.002421 |
| RL | q_no_td_modify | beta | 15 | walk_interaction_7d | -0.061136 | 0.023232 |
| RL | q_no_td_modify | beta | 16 | A | 0.012236 | 0.001439 |
| RL | q_no_td_modify | beta | 17 | A*E_w | 0.002134 | 0.001055 |
| RL | q_no_td_modify | beta | 18 | A*b_hat | -0.000243 | 0.002456 |
| RL | q_no_td_modify | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify | beta | 20 | A*yesterday_step_count | -0.003693 | 0.002665 |
| RL | q_no_td_modify | beta | 21 | A*prior2hour_step_count | 0.006629 | 0.001128 |
| RL | q_no_td_modify | beta | 22 | A*active_status_fraction_7days | -0.011960 | 0.001427 |
| RL | q_no_td_modify | beta | 23 | A*recent_burden | -0.003028 | 0.000854 |
| RL | q_no_td_modify | beta | 24 | A*walk_interaction_7d | -0.004677 | 0.002166 |
| RL | q_no_td_modify_g09 | beta | 0 | intercept | 0.112094 | 0.152331 |
| RL | q_no_td_modify_g09 | beta | 1 | weekday_vs_weekend | -0.044791 | 0.250970 |
| RL | q_no_td_modify_g09 | beta | 2 | slot_pm | 0.001357 | 0.007562 |
| RL | q_no_td_modify_g09 | beta | 3 | E_w | 0.040714 | 0.067453 |
| RL | q_no_td_modify_g09 | beta | 4 | b_hat | 1.555100 | 0.062612 |
| RL | q_no_td_modify_g09 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g09 | beta | 6 | M_Y_anticipated_affect_ewma | 0.483698 | 0.021707 |
| RL | q_no_td_modify_g09 | beta | 7 | M_Y_fourSC_ewma | -0.034287 | 0.011359 |
| RL | q_no_td_modify_g09 | beta | 8 | M_E_pageview_ewma | 0.005238 | 0.000485 |
| RL | q_no_td_modify_g09 | beta | 9 | M_E_fitbit_wear_ewma | -0.212861 | 0.007967 |
| RL | q_no_td_modify_g09 | beta | 10 | M_E_survey_complete_ewma | -0.062399 | 0.009343 |
| RL | q_no_td_modify_g09 | beta | 11 | yesterday_step_count | -0.001705 | 0.002603 |
| RL | q_no_td_modify_g09 | beta | 12 | prior2hour_step_count | -0.016653 | 0.001153 |
| RL | q_no_td_modify_g09 | beta | 13 | active_status_fraction_7days | -0.017481 | 0.051035 |
| RL | q_no_td_modify_g09 | beta | 14 | recent_burden | -0.017942 | 0.004518 |
| RL | q_no_td_modify_g09 | beta | 15 | walk_interaction_7d | -0.066107 | 0.037074 |
| RL | q_no_td_modify_g09 | beta | 16 | A | 0.016998 | 0.004640 |
| RL | q_no_td_modify_g09 | beta | 17 | A*E_w | 0.002957 | 0.003094 |
| RL | q_no_td_modify_g09 | beta | 18 | A*b_hat | -0.000594 | 0.005287 |
| RL | q_no_td_modify_g09 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g09 | beta | 20 | A*yesterday_step_count | -0.006732 | 0.005107 |
| RL | q_no_td_modify_g09 | beta | 21 | A*prior2hour_step_count | 0.010436 | 0.001997 |
| RL | q_no_td_modify_g09 | beta | 22 | A*active_status_fraction_7days | -0.020881 | 0.003762 |
| RL | q_no_td_modify_g09 | beta | 23 | A*recent_burden | -0.004927 | 0.001830 |
| RL | q_no_td_modify_g09 | beta | 24 | A*walk_interaction_7d | -0.000254 | 0.004399 |
| RL | q_no_td_modify_g099 | beta | 0 | intercept | 0.114106 | 0.174228 |
| RL | q_no_td_modify_g099 | beta | 1 | weekday_vs_weekend | -0.042474 | 0.318175 |
| RL | q_no_td_modify_g099 | beta | 2 | slot_pm | 0.002642 | 0.010170 |
| RL | q_no_td_modify_g099 | beta | 3 | E_w | 0.046442 | 0.075911 |
| RL | q_no_td_modify_g099 | beta | 4 | b_hat | 1.669690 | 0.069990 |
| RL | q_no_td_modify_g099 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g099 | beta | 6 | M_Y_anticipated_affect_ewma | 0.541223 | 0.027725 |
| RL | q_no_td_modify_g099 | beta | 7 | M_Y_fourSC_ewma | -0.037330 | 0.012771 |
| RL | q_no_td_modify_g099 | beta | 8 | M_E_pageview_ewma | 0.006258 | 0.000584 |
| RL | q_no_td_modify_g099 | beta | 9 | M_E_fitbit_wear_ewma | -0.236628 | 0.009947 |
| RL | q_no_td_modify_g099 | beta | 10 | M_E_survey_complete_ewma | -0.067896 | 0.011445 |
| RL | q_no_td_modify_g099 | beta | 11 | yesterday_step_count | -0.000318 | 0.003240 |
| RL | q_no_td_modify_g099 | beta | 12 | prior2hour_step_count | -0.018615 | 0.001336 |
| RL | q_no_td_modify_g099 | beta | 13 | active_status_fraction_7days | -0.012662 | 0.060751 |
| RL | q_no_td_modify_g099 | beta | 14 | recent_burden | -0.019579 | 0.005493 |
| RL | q_no_td_modify_g099 | beta | 15 | walk_interaction_7d | -0.066464 | 0.042498 |
| RL | q_no_td_modify_g099 | beta | 16 | A | 0.018387 | 0.006224 |
| RL | q_no_td_modify_g099 | beta | 17 | A*E_w | 0.003203 | 0.004048 |
| RL | q_no_td_modify_g099 | beta | 18 | A*b_hat | -0.000751 | 0.006398 |
| RL | q_no_td_modify_g099 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g099 | beta | 20 | A*yesterday_step_count | -0.007701 | 0.005926 |
| RL | q_no_td_modify_g099 | beta | 21 | A*prior2hour_step_count | 0.011610 | 0.002359 |
| RL | q_no_td_modify_g099 | beta | 22 | A*active_status_fraction_7days | -0.023695 | 0.005047 |
| RL | q_no_td_modify_g099 | beta | 23 | A*recent_burden | -0.005574 | 0.002243 |
| RL | q_no_td_modify_g099 | beta | 24 | A*walk_interaction_7d | 0.001262 | 0.005324 |
| RL | q_redistribution_v2 | beta | 0 | intercept | 0.253985 | 0.262921 |
| RL | q_redistribution_v2 | beta | 1 | weekday_vs_weekend | -0.292363 | 0.096410 |
| RL | q_redistribution_v2 | beta | 2 | slot_pm | -0.020584 | 0.002712 |
| RL | q_redistribution_v2 | beta | 3 | E_w | 0.838941 | 0.651008 |
| RL | q_redistribution_v2 | beta | 4 | b_hat | 1.067305 | 0.081728 |
| RL | q_redistribution_v2 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v2 | beta | 6 | M_Y_anticipated_affect_ewma | 0.674251 | 0.036108 |
| RL | q_redistribution_v2 | beta | 7 | M_Y_fourSC_ewma | -0.086744 | 0.028413 |
| RL | q_redistribution_v2 | beta | 8 | M_E_pageview_ewma | -0.010419 | 0.007149 |
| RL | q_redistribution_v2 | beta | 9 | M_E_fitbit_wear_ewma | -0.344546 | 0.023510 |
| RL | q_redistribution_v2 | beta | 10 | M_E_survey_complete_ewma | -0.017008 | 0.022904 |
| RL | q_redistribution_v2 | beta | 11 | yesterday_step_count | 0.066885 | 0.008805 |
| RL | q_redistribution_v2 | beta | 12 | prior2hour_step_count | 0.045389 | 0.003978 |
| RL | q_redistribution_v2 | beta | 13 | active_status_fraction_7days | 0.756894 | 0.199391 |
| RL | q_redistribution_v2 | beta | 14 | recent_burden | 0.145970 | 0.014964 |
| RL | q_redistribution_v2 | beta | 15 | walk_interaction_7d | 1.164098 | 0.128093 |
| RL | q_redistribution_v2 | beta | 16 | A | 0.066755 | 0.012273 |
| RL | q_redistribution_v2 | beta | 17 | A*E_w | 0.004343 | 0.001752 |
| RL | q_redistribution_v2 | beta | 18 | A*b_hat | -0.025422 | 0.003489 |
| RL | q_redistribution_v2 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v2 | beta | 20 | A*yesterday_step_count | 0.032924 | 0.001476 |
| RL | q_redistribution_v2 | beta | 21 | A*prior2hour_step_count | 0.000229 | 0.003653 |
| RL | q_redistribution_v2 | beta | 22 | A*active_status_fraction_7days | 0.013942 | 0.009875 |
| RL | q_redistribution_v2 | beta | 23 | A*recent_burden | 0.026229 | 0.001325 |
| RL | q_redistribution_v2 | beta | 24 | A*walk_interaction_7d | -0.023852 | 0.010973 |
| RL | q_redistribution_v4 | beta | 0 | intercept | 0.072770 | 0.428540 |
| RL | q_redistribution_v4 | beta | 1 | weekday_vs_weekend | -0.129793 | 0.070315 |
| RL | q_redistribution_v4 | beta | 2 | slot_pm | -0.007674 | 0.001599 |
| RL | q_redistribution_v4 | beta | 3 | E_w | -0.257600 | 0.148975 |
| RL | q_redistribution_v4 | beta | 4 | b_hat | 1.693123 | 0.107758 |
| RL | q_redistribution_v4 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v4 | beta | 6 | M_Y_anticipated_affect_ewma | 0.233679 | 0.045414 |
| RL | q_redistribution_v4 | beta | 7 | M_Y_fourSC_ewma | -0.165017 | 0.110457 |
| RL | q_redistribution_v4 | beta | 8 | M_E_pageview_ewma | -0.023703 | 0.022494 |
| RL | q_redistribution_v4 | beta | 9 | M_E_fitbit_wear_ewma | -0.107422 | 0.021927 |
| RL | q_redistribution_v4 | beta | 10 | M_E_survey_complete_ewma | 0.005130 | 0.080435 |
| RL | q_redistribution_v4 | beta | 11 | yesterday_step_count | -0.049234 | 0.047662 |
| RL | q_redistribution_v4 | beta | 12 | prior2hour_step_count | 0.060204 | 0.010386 |
| RL | q_redistribution_v4 | beta | 13 | active_status_fraction_7days | 0.414567 | 0.242057 |
| RL | q_redistribution_v4 | beta | 14 | recent_burden | 0.009462 | 0.079428 |
| RL | q_redistribution_v4 | beta | 15 | walk_interaction_7d | 0.334916 | 0.325039 |
| RL | q_redistribution_v4 | beta | 16 | A | 0.076047 | 0.026897 |
| RL | q_redistribution_v4 | beta | 17 | A*E_w | -0.005204 | 0.004908 |
| RL | q_redistribution_v4 | beta | 18 | A*b_hat | 0.001173 | 0.007372 |
| RL | q_redistribution_v4 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v4 | beta | 20 | A*yesterday_step_count | 0.049476 | 0.004303 |
| RL | q_redistribution_v4 | beta | 21 | A*prior2hour_step_count | -0.016503 | 0.014283 |
| RL | q_redistribution_v4 | beta | 22 | A*active_status_fraction_7days | -0.073437 | 0.014240 |
| RL | q_redistribution_v4 | beta | 23 | A*recent_burden | -0.005582 | 0.002817 |
| RL | q_redistribution_v4 | beta | 24 | A*walk_interaction_7d | -0.074872 | 0.043561 |

## Joint Modified-TD Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| Joint Modified TD | q_td_modify_joint | eta | 0 | eta_intercept | 0.578288 | 0.164050 |
| Joint Modified TD | q_td_modify_joint | eta | 1 | eta_E_w | -0.005213 | 0.068416 |
| Joint Modified TD | q_td_modify_joint | eta | 2 | eta_b_hat | -0.528452 | 0.074128 |
| Joint Modified TD | q_td_modify_joint | eta | 3 | eta_b_tilde | 0.000000 | 1.000001 |
| Joint Modified TD | q_td_modify_joint | beta | 4 | beta_intercept | 0.557278 | 0.111962 |
| Joint Modified TD | q_td_modify_joint | beta | 5 | beta_weekday_vs_weekend | -0.080675 | 0.208742 |
| Joint Modified TD | q_td_modify_joint | beta | 6 | beta_slot_pm | -0.005826 | 0.003851 |
| Joint Modified TD | q_td_modify_joint | beta | 7 | beta_E_w | 0.000951 | 0.052789 |
| Joint Modified TD | q_td_modify_joint | beta | 8 | beta_b_hat | -0.069668 | 0.035087 |
| Joint Modified TD | q_td_modify_joint | beta | 9 | beta_b_tilde | 0.000000 | 1.000001 |
| Joint Modified TD | q_td_modify_joint | beta | 10 | beta_M_Y_anticipated_affect_ewma | 0.721069 | 0.020361 |
| Joint Modified TD | q_td_modify_joint | beta | 11 | beta_M_Y_fourSC_ewma | -0.007655 | 0.009986 |
| Joint Modified TD | q_td_modify_joint | beta | 12 | beta_M_E_pageview_ewma | 0.006913 | 0.001060 |
| Joint Modified TD | q_td_modify_joint | beta | 13 | beta_M_E_fitbit_wear_ewma | -0.338350 | 0.005558 |
| Joint Modified TD | q_td_modify_joint | beta | 14 | beta_M_E_survey_complete_ewma | -0.040540 | 0.008286 |
| Joint Modified TD | q_td_modify_joint | beta | 15 | beta_yesterday_step_count | 0.001234 | 0.001461 |
| Joint Modified TD | q_td_modify_joint | beta | 16 | beta_prior2hour_step_count | -0.006964 | 0.000830 |
| Joint Modified TD | q_td_modify_joint | beta | 17 | beta_active_status_fraction_7days | -0.015056 | 0.028000 |
| Joint Modified TD | q_td_modify_joint | beta | 18 | beta_recent_burden | -0.009281 | 0.001560 |
| Joint Modified TD | q_td_modify_joint | beta | 19 | beta_walk_interaction_7d | 0.024365 | 0.012118 |
| Joint Modified TD | q_td_modify_joint | beta | 20 | beta_A | 0.005332 | 0.003248 |
| Joint Modified TD | q_td_modify_joint | beta | 21 | beta_A*E_w | -0.000515 | 0.002352 |
| Joint Modified TD | q_td_modify_joint | beta | 22 | beta_A*b_hat | -0.068869 | 0.003193 |
| Joint Modified TD | q_td_modify_joint | beta | 23 | beta_A*b_tilde | 0.000000 | 1.000001 |
| Joint Modified TD | q_td_modify_joint | beta | 24 | beta_A*yesterday_step_count | -0.000448 | 0.003477 |
| Joint Modified TD | q_td_modify_joint | beta | 25 | beta_A*prior2hour_step_count | -0.002119 | 0.001401 |
| Joint Modified TD | q_td_modify_joint | beta | 26 | beta_A*active_status_fraction_7days | -0.003036 | 0.002329 |
| Joint Modified TD | q_td_modify_joint | beta | 27 | beta_A*recent_burden | 0.000770 | 0.001896 |
| Joint Modified TD | q_td_modify_joint | beta | 28 | beta_A*walk_interaction_7d | 0.003025 | 0.002371 |

## Pooled PF Regression Coefficients

Coefficients are from the all-user stacked ridge fits used as PF prior means. `p_value` uses ridge sandwich standard errors and is approximate because ridge shrinks coefficients. `identified=False` marks zero-variance design columns; SE / p-values are omitted for those rows.

| model | outcome | index | feature | coefficient | identified | feature_std | std_error | t_stat | p_value | significant_0.05 | significant_0.01 | n_obs | n_features | ridge_alpha | residual_sigma2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fourSC | 4hour_step_norm | 0 | intercept | -0.077843 | False | 0.000000 |  |  |  | False | False | 2890 | 19 | 1.000000 | 0.475879 |
| fourSC | 4hour_step_norm | 1 | stepCountNext4HourLag1 | 0.486631 | True | 0.915568 | 0.017098 | 28.461457 | 3.585629e-157 | True | True | 2890 | 19 | 1.000000 | 0.475879 |
| fourSC | 4hour_step_norm | 2 | yesterdayStepCount | -0.148510 | True | 0.930518 | 0.021699 | -6.843953 | 9.381263e-12 | True | True | 2890 | 19 | 1.000000 | 0.475879 |
| fourSC | 4hour_step_norm | 3 | stepCountLast7DaysEma | 0.358334 | True | 0.954106 | 0.016239 | 22.066327 | 8.384093e-100 | True | True | 2890 | 19 | 1.000000 | 0.475879 |
| fourSC | 4hour_step_norm | 4 | prior2HourStepCount | 0.097988 | True | 0.935469 | 0.020109 | 4.872959 | 1.159096e-06 | True | True | 2890 | 19 | 1.000000 | 0.475879 |
| fourSC | 4hour_step_norm | 5 | activitySuggestionsSentLast7Days | -0.037568 | True | 0.968302 | 0.019069 | -1.970129 | 0.048920 | True | False | 2890 | 19 | 1.000000 | 0.475879 |
| fourSC | 4hour_step_norm | 6 | activitySuggestionInteractLast7Days | 0.037534 | True | 0.353473 | 0.052810 | 0.710748 | 0.477298 | False | False | 2890 | 19 | 1.000000 | 0.475879 |
| fourSC | 4hour_step_norm | 7 | activeDaysLast7Days | 0.157462 | True | 0.260938 | 0.054967 | 2.864649 | 0.004205 | True | True | 2890 | 19 | 1.000000 | 0.475879 |
| fourSC | 4hour_step_norm | 8 | isWeekend | -0.034017 | True | 0.369659 | 0.034735 | -0.979346 | 0.327492 | False | False | 2890 | 19 | 1.000000 | 0.475879 |
| fourSC | 4hour_step_norm | 9 | decisionTimeSlot | -0.034384 | True | 0.497078 | 0.027074 | -1.269994 | 0.204190 | False | False | 2890 | 19 | 1.000000 | 0.475879 |
| fourSC | 4hour_step_norm | 10 | perceivedUtilityLastWeek | 0.020972 | True | 2.361545 | 0.008802 | 2.382592 | 0.017256 | True | False | 2890 | 19 | 1.000000 | 0.475879 |
| fourSC | 4hour_step_norm | 11 | caeAverageLastWeek | -0.053702 | True | 0.947049 | 0.021088 | -2.546610 | 0.010929 | True | False | 2890 | 19 | 1.000000 | 0.475879 |
| fourSC | 4hour_step_norm | 12 | Ah | -0.116266 | True | 0.499925 | 0.043379 | -2.680242 | 0.007399 | True | True | 2890 | 19 | 1.000000 | 0.475879 |
| fourSC | 4hour_step_norm | 13 | Ah*yesterdayStepCount | 0.005743 | True | 0.661722 | 0.028870 | 0.198939 | 0.842324 | False | False | 2890 | 19 | 1.000000 | 0.475879 |
| fourSC | 4hour_step_norm | 14 | Ah*prior2HourStepCount | 0.009660 | True | 0.655759 | 0.028224 | 0.342255 | 0.732184 | False | False | 2890 | 19 | 1.000000 | 0.475879 |
| fourSC | 4hour_step_norm | 15 | Ah*activitySuggestionsSentLast7Days | 0.029985 | True | 0.684165 | 0.026838 | 1.117260 | 0.263976 | False | False | 2890 | 19 | 1.000000 | 0.475879 |
| fourSC | 4hour_step_norm | 16 | Ah*activitySuggestionInteractLast7Days | 0.087900 | True | 0.343672 | 0.074760 | 1.175762 | 0.239787 | False | False | 2890 | 19 | 1.000000 | 0.475879 |
| fourSC | 4hour_step_norm | 17 | Ah*perceivedUtilityLastWeek | 0.020755 | True | 1.784542 | 0.012031 | 1.725147 | 0.084609 | False | False | 2890 | 19 | 1.000000 | 0.475879 |
| fourSC | 4hour_step_norm | 18 | Ah*caeAverageLastWeek | -0.014121 | True | 0.644148 | 0.029020 | -0.486577 | 0.626595 | False | False | 2890 | 19 | 1.000000 | 0.475879 |
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
| -0.091498 |  |  |  | False | False | fourSC | 4hour_step_norm | 0 | intercept | False | 0.000000 | 2890 | 28 | 19 | exchangeable |
| -0.038643 | 0.040875 | -0.945394 | 0.344458 | False | False | fourSC | 4hour_step_norm | 1 | stepCountNext4HourLag1 | True | 0.915568 | 2890 | 28 | 19 | exchangeable |
| -0.016604 | 0.029792 | -0.557322 | 0.577308 | False | False | fourSC | 4hour_step_norm | 2 | yesterdayStepCount | True | 0.930518 | 2890 | 28 | 19 | exchangeable |
| 0.185840 | 0.034816 | 5.337807 | 9.407734e-08 | True | True | fourSC | 4hour_step_norm | 3 | stepCountLast7DaysEma | True | 0.954106 | 2890 | 28 | 19 | exchangeable |
| 0.128739 | 0.031263 | 4.117944 | 3.822669e-05 | True | True | fourSC | 4hour_step_norm | 4 | prior2HourStepCount | True | 0.935469 | 2890 | 28 | 19 | exchangeable |
| 0.005290 | 0.016240 | 0.325712 | 0.744642 | False | False | fourSC | 4hour_step_norm | 5 | activitySuggestionsSentLast7Days | True | 0.968302 | 2890 | 28 | 19 | exchangeable |
| 0.046203 | 0.064877 | 0.712174 | 0.476357 | False | False | fourSC | 4hour_step_norm | 6 | activitySuggestionInteractLast7Days | True | 0.353473 | 2890 | 28 | 19 | exchangeable |
| 0.030243 | 0.068187 | 0.443530 | 0.657383 | False | False | fourSC | 4hour_step_norm | 7 | activeDaysLast7Days | True | 0.260938 | 2890 | 28 | 19 | exchangeable |
| -0.006580 | 0.042500 | -0.154827 | 0.876958 | False | False | fourSC | 4hour_step_norm | 8 | isWeekend | True | 0.369659 | 2890 | 28 | 19 | exchangeable |
| -0.073814 | 0.059019 | -1.250686 | 0.211049 | False | False | fourSC | 4hour_step_norm | 9 | decisionTimeSlot | True | 0.497078 | 2890 | 28 | 19 | exchangeable |
| 0.025606 | 0.018481 | 1.385495 | 0.165901 | False | False | fourSC | 4hour_step_norm | 10 | perceivedUtilityLastWeek | True | 2.361545 | 2890 | 28 | 19 | exchangeable |
| 0.000210 | 0.031915 | 0.006580 | 0.994750 | False | False | fourSC | 4hour_step_norm | 11 | caeAverageLastWeek | True | 0.947049 | 2890 | 28 | 19 | exchangeable |
| -0.051115 | 0.051440 | -0.993683 | 0.320377 | False | False | fourSC | 4hour_step_norm | 12 | Ah | True | 0.499925 | 2890 | 28 | 19 | exchangeable |
| 0.014488 | 0.033086 | 0.437891 | 0.661466 | False | False | fourSC | 4hour_step_norm | 13 | Ah*yesterdayStepCount | True | 0.661722 | 2890 | 28 | 19 | exchangeable |
| -0.018527 | 0.028262 | -0.655563 | 0.512105 | False | False | fourSC | 4hour_step_norm | 14 | Ah*prior2HourStepCount | True | 0.655759 | 2890 | 28 | 19 | exchangeable |
| 0.007359 | 0.016613 | 0.442960 | 0.657795 | False | False | fourSC | 4hour_step_norm | 15 | Ah*activitySuggestionsSentLast7Days | True | 0.684165 | 2890 | 28 | 19 | exchangeable |
| 0.048133 | 0.081452 | 0.590937 | 0.554562 | False | False | fourSC | 4hour_step_norm | 16 | Ah*activitySuggestionInteractLast7Days | True | 0.343672 | 2890 | 28 | 19 | exchangeable |
| 0.005942 | 0.012354 | 0.480952 | 0.630551 | False | False | fourSC | 4hour_step_norm | 17 | Ah*perceivedUtilityLastWeek | True | 1.784542 | 2890 | 28 | 19 | exchangeable |
| -0.014947 | 0.025355 | -0.589516 | 0.555515 | False | False | fourSC | 4hour_step_norm | 18 | Ah*caeAverageLastWeek | True | 0.644148 | 2890 | 28 | 19 | exchangeable |
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
| 0.101639 |  |  |  | False | False | q_no_td_modify | fqi_target | 0 | intercept | False | 0.000000 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| -0.047786 | 0.032612 | -1.465309 | 0.142837 | False | False | q_no_td_modify | fqi_target | 1 | weekday_vs_weekend | True | 0.372678 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| -0.002810 | 0.013104 | -0.214470 | 0.830181 | False | False | q_no_td_modify | fqi_target | 2 | slot_pm | True | 0.500000 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| 0.023085 | 0.001263 | 18.278057 | 1.237578e-74 | True | True | q_no_td_modify | fqi_target | 3 | E_w | True | 2.350134 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| 1.159896 | 0.005881 | 197.239473 | 0.000000 | True | True | q_no_td_modify | fqi_target | 4 | b_hat | True | 0.921918 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| 2.790276e-17 |  |  |  | False | False | q_no_td_modify | fqi_target | 5 | b_tilde | False | 0.000000 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| 0.295910 | 0.015087 | 19.613402 | 1.188134e-85 | True | True | q_no_td_modify | fqi_target | 6 | M_Y_anticipated_affect_ewma | True | 0.311549 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| -0.022627 | 0.001946 | -11.626719 | 3.014634e-31 | True | True | q_no_td_modify | fqi_target | 7 | M_Y_fourSC_ewma | True | 0.948911 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| 0.000883 | 0.001933 | 0.456683 | 0.647899 | False | False | q_no_td_modify | fqi_target | 8 | M_E_pageview_ewma | True | 1.050150 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| -0.132573 | 0.006427 | -20.627111 | 1.567393e-94 | True | True | q_no_td_modify | fqi_target | 9 | M_E_fitbit_wear_ewma | True | 0.412251 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| -0.046121 | 0.005620 | -8.206283 | 2.281419e-16 | True | True | q_no_td_modify | fqi_target | 10 | M_E_survey_complete_ewma | True | 0.411996 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| -0.007099 | 0.004625 | -1.534940 | 0.124799 | False | False | q_no_td_modify | fqi_target | 11 | yesterday_step_count | True | 0.918323 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| -0.013743 | 0.004362 | -3.150521 | 0.001630 | True | True | q_no_td_modify | fqi_target | 12 | prior2hour_step_count | True | 0.923902 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| -0.039260 | 0.012360 | -3.176234 | 0.001492 | True | True | q_no_td_modify | fqi_target | 13 | active_status_fraction_7days | True | 0.266148 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| -0.013095 | 0.004157 | -3.150414 | 0.001630 | True | True | q_no_td_modify | fqi_target | 14 | recent_burden | True | 0.931218 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| -0.064043 | 0.009466 | -6.765549 | 1.328045e-11 | True | True | q_no_td_modify | fqi_target | 15 | walk_interaction_7d | True | 0.354304 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| 0.008988 | 0.016310 | 0.551056 | 0.581595 | False | False | q_no_td_modify | fqi_target | 16 | A | True | 0.499915 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| 0.000856 | 0.002032 | 0.421547 | 0.673356 | False | False | q_no_td_modify | fqi_target | 17 | A*E_w | True | 1.764758 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| 3.948423e-05 | 0.010581 | 0.003732 | 0.997023 | False | False | q_no_td_modify | fqi_target | 18 | A*b_hat | True | 0.635088 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| 7.424461e-17 |  |  |  | False | False | q_no_td_modify | fqi_target | 19 | A*b_tilde | False | 0.000000 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| -0.004196 | 0.007377 | -0.568827 | 0.569473 | False | False | q_no_td_modify | fqi_target | 20 | A*yesterday_step_count | True | 0.654833 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| 0.007060 | 0.004454 | 1.585254 | 0.112909 | False | False | q_no_td_modify | fqi_target | 21 | A*prior2hour_step_count | True | 0.644358 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| -0.005233 | 0.020854 | -0.250934 | 0.801865 | False | False | q_no_td_modify | fqi_target | 22 | A*active_status_fraction_7days | True | 0.375177 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| -0.001908 | 0.005186 | -0.367815 | 0.713011 | False | False | q_no_td_modify | fqi_target | 23 | A*recent_burden | True | 0.662545 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
| -0.004410 | 0.017293 | -0.255033 | 0.798697 | False | False | q_no_td_modify | fqi_target | 24 | A*walk_interaction_7d | True | 0.334802 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.030243 | beta |
