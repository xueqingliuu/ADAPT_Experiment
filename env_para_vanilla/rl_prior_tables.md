# Prior Mean and Variance Summary

`prior_variance` is the diagonal entry of the prior covariance. For `q_td_modify_joint`, the full covariance remains in `rl_priors.json`; this table reports marginal variances.

## PF Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| PF | fourSC | theta | 0 | intercept | -0.048286 | 0.058055 |
| PF | fourSC | theta | 1 | stepCountNext4HourLag1 | 0.497640 | 0.045673 |
| PF | fourSC | theta | 2 | yesterdayStepCount | -0.161932 | 0.012278 |
| PF | fourSC | theta | 3 | stepCountLast7DaysEma | 0.499996 | 0.038208 |
| PF | fourSC | theta | 4 | prior2HourStepCount | 0.099240 | 0.036862 |
| PF | fourSC | theta | 5 | activitySuggestionsSentLast7Days | -0.037892 | 0.030240 |
| PF | fourSC | theta | 6 | activitySuggestionInteractLast7Days | 0.037644 | 0.074924 |
| PF | fourSC | theta | 7 | activeDaysLast7Days | 0.131898 | 0.047088 |
| PF | fourSC | theta | 8 | isWeekend | -0.036638 | 0.051733 |
| PF | fourSC | theta | 9 | decisionTimeSlot | -0.038970 | 0.158486 |
| PF | fourSC | theta | 10 | perceivedUtilityLastWeek | 0.017864 | 0.015846 |
| PF | fourSC | theta | 11 | caeAverageLastWeek | -0.052762 | 0.109445 |
| PF | fourSC | theta | 12 | Ah | -0.109031 | 0.070290 |
| PF | fourSC | theta | 13 | Ah*yesterdayStepCount | -0.002172 | 0.021199 |
| PF | fourSC | theta | 14 | Ah*prior2HourStepCount | 0.010920 | 0.025855 |
| PF | fourSC | theta | 15 | Ah*activitySuggestionsSentLast7Days | 0.028748 | 0.027249 |
| PF | fourSC | theta | 16 | Ah*activitySuggestionInteractLast7Days | 0.110169 | 0.119036 |
| PF | fourSC | theta | 17 | Ah*perceivedUtilityLastWeek | 0.007230 | 0.025151 |
| PF | fourSC | theta | 18 | Ah*caeAverageLastWeek | -0.004544 | 0.101238 |
| PF | antic | theta | 0 | intercept | 0.422129 | 0.013421 |
| PF | antic | theta | 1 | anticipated_affect_yesterday | 0.218613 | 0.005263 |
| PF | antic | theta | 2 | active_status_fraction_7days | 0.075727 | 0.008608 |
| PF | antic | theta | 3 | is_weekend | -0.028077 | 0.004774 |
| PF | antic | theta | 4 | perceived_utility_lastweek | 0.007743 | 0.009738 |
| PF | antic | theta | 5 | CAE_avg_lastweek | 0.236253 | 0.006699 |
| PF | antic | theta | 6 | recent_burden | 0.014369 | 0.004008 |
| PF | antic | theta | 7 | A0_morning | 0.004252 | 0.002681 |
| PF | antic | theta | 8 | A1_afternoon | 0.014142 | 0.001403 |
| PF | antic | theta | 9 | A0_morning_by_perceived_utility_lastweek | -0.001464 | 0.001329 |
| PF | antic | theta | 10 | A1_afternoon_by_perceived_utility_lastweek | -0.001489 | 0.000719 |
| PF | antic | theta | 11 | A0_morning_by_CAE_avg_lastweek | -0.004551 | 0.005764 |
| PF | antic | theta | 12 | A1_afternoon_by_CAE_avg_lastweek | -0.027552 | 0.002811 |
| PF | antic | theta | 13 | A0_morning_by_recent_burden | -0.009632 | 0.001321 |
| PF | antic | theta | 14 | A1_afternoon_by_recent_burden | -0.003091 | 0.004237 |
| PF | CAE | theta | 0 | intercept | -0.125413 | 0.086648 |
| PF | CAE | theta | 1 | CAE_avg_lastweek | 0.883442 | 0.059847 |
| PF | CAE | theta | 2 | fourSC_ewma | -0.033341 | 0.014975 |
| PF | CAE | theta | 3 | anticipated_affect_ewma | 0.175007 | 0.019823 |
| PF | CAE_short | theta | 0 | intercept | 0.010005 | 0.063876 |
| PF | CAE_short | theta | 1 | caeAverage | 0.975894 | 0.063795 |

## RL Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| RL | reward_shaping | eta | 0 | intercept | -0.053316 | 0.029170 |
| RL | reward_shaping | eta | 1 | weekday_vs_weekend | -0.008886 | 0.000810 |
| RL | reward_shaping | eta | 2 | slot_pm | -0.026658 | 0.007293 |
| RL | reward_shaping | eta | 3 | E_w | 0.066117 | 0.002858 |
| RL | reward_shaping | eta | 4 | b_hat | 0.061279 | 0.049271 |
| RL | reward_shaping | eta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | reward_shaping | eta | 6 | M_Y_anticipated_affect_ewma | 0.088694 | 0.009084 |
| RL | reward_shaping | eta | 7 | M_Y_fourSC_ewma | 0.010784 | 0.020365 |
| RL | reward_shaping | eta | 8 | M_E_pageview_ewma | -0.004156 | 0.004132 |
| RL | reward_shaping | eta | 9 | M_E_fitbit_wear_ewma | -0.008695 | 0.005135 |
| RL | reward_shaping | eta | 10 | M_E_survey_complete_ewma | 0.008685 | 0.011135 |
| RL | reward_shaping | eta | 11 | yesterday_step_count | -0.002343 | 0.015261 |
| RL | reward_shaping | eta | 12 | prior2hour_step_count | -0.020731 | 0.005069 |
| RL | reward_shaping | eta | 13 | active_status_fraction_7days | 0.014045 | 0.044129 |
| RL | reward_shaping | eta | 14 | recent_burden | -0.000487 | 0.003778 |
| RL | reward_shaping | eta | 15 | walk_interaction_7d | 0.018068 | 0.038146 |
| RL | redistribution_stage1_AA | eta | 0 | redistribution_stage1_AA_coef_0 | 0.250424 | 0.007256 |
| RL | redistribution_stage1_AA | eta | 1 | redistribution_stage1_AA_coef_1 | 0.001380 | 0.001326 |
| RL | redistribution_stage1_AA | eta | 2 | redistribution_stage1_AA_coef_2 | 0.094461 | 0.002593 |
| RL | redistribution_stage1_AA | eta | 3 | redistribution_stage1_AA_coef_3 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_AA | eta | 4 | redistribution_stage1_AA_coef_4 | 0.172739 | 0.001988 |
| RL | redistribution_stage1_AA | eta | 5 | redistribution_stage1_AA_coef_5 | -0.010439 | 0.003267 |
| RL | redistribution_stage1_AA | eta | 6 | redistribution_stage1_AA_coef_6 | 0.012372 | 0.000584 |
| RL | redistribution_stage1_AA | eta | 7 | redistribution_stage1_AA_coef_7 | -0.033790 | 0.001518 |
| RL | redistribution_stage1_AA | eta | 8 | redistribution_stage1_AA_coef_8 | -0.029003 | 0.001416 |
| RL | redistribution_stage1_AA | eta | 9 | redistribution_stage1_AA_coef_9 | 0.007512 | 0.000214 |
| RL | redistribution_stage1_AA | eta | 10 | redistribution_stage1_AA_coef_10 | 0.010700 | 0.000374 |
| RL | redistribution_stage1_AA | eta | 11 | redistribution_stage1_AA_coef_11 | 0.024711 | 0.004523 |
| RL | redistribution_stage1_AA | eta | 12 | redistribution_stage1_AA_coef_12 | 0.000486 | 0.000120 |
| RL | redistribution_stage1_AA | eta | 13 | redistribution_stage1_AA_coef_13 | -0.001317 | 0.003098 |
| RL | redistribution_stage1_AA | eta | 14 | redistribution_stage1_AA_coef_14 | 0.012571 | 0.001325 |
| RL | redistribution_stage1_AA | eta | 15 | redistribution_stage1_AA_coef_15 | 0.001673 | 0.000599 |
| RL | redistribution_stage1_AA | eta | 16 | redistribution_stage1_AA_coef_16 | -0.008056 | 0.002329 |
| RL | redistribution_stage1_AA | eta | 17 | redistribution_stage1_AA_coef_17 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_AA | eta | 18 | redistribution_stage1_AA_coef_18 | 0.000487 | 0.000641 |
| RL | redistribution_stage1_AA | eta | 19 | redistribution_stage1_AA_coef_19 | 0.007425 | 0.000442 |
| RL | redistribution_stage1_AA | eta | 20 | redistribution_stage1_AA_coef_20 | -0.010492 | 0.002631 |
| RL | redistribution_stage1_AA | eta | 21 | redistribution_stage1_AA_coef_21 | 0.003436 | 0.000285 |
| RL | redistribution_stage1_AA | eta | 22 | redistribution_stage1_AA_coef_22 | -0.004554 | 0.000846 |
| RL | redistribution_stage1_FW | eta | 0 | redistribution_stage1_FW_coef_0 | 0.367931 | 0.025215 |
| RL | redistribution_stage1_FW | eta | 1 | redistribution_stage1_FW_coef_1 | 0.010367 | 0.006417 |
| RL | redistribution_stage1_FW | eta | 2 | redistribution_stage1_FW_coef_2 | 0.001137 | 0.009085 |
| RL | redistribution_stage1_FW | eta | 3 | redistribution_stage1_FW_coef_3 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_FW | eta | 4 | redistribution_stage1_FW_coef_4 | -0.125709 | 0.028988 |
| RL | redistribution_stage1_FW | eta | 5 | redistribution_stage1_FW_coef_5 | -0.026630 | 0.009920 |
| RL | redistribution_stage1_FW | eta | 6 | redistribution_stage1_FW_coef_6 | 0.026766 | 0.001488 |
| RL | redistribution_stage1_FW | eta | 7 | redistribution_stage1_FW_coef_7 | 0.159830 | 0.016846 |
| RL | redistribution_stage1_FW | eta | 8 | redistribution_stage1_FW_coef_8 | -0.015303 | 0.009089 |
| RL | redistribution_stage1_FW | eta | 9 | redistribution_stage1_FW_coef_9 | 0.005042 | 0.003490 |
| RL | redistribution_stage1_FW | eta | 10 | redistribution_stage1_FW_coef_10 | 0.004735 | 0.002570 |
| RL | redistribution_stage1_FW | eta | 11 | redistribution_stage1_FW_coef_11 | 0.038489 | 0.016372 |
| RL | redistribution_stage1_FW | eta | 12 | redistribution_stage1_FW_coef_12 | -0.009536 | 0.003129 |
| RL | redistribution_stage1_FW | eta | 13 | redistribution_stage1_FW_coef_13 | 0.022557 | 0.022325 |
| RL | redistribution_stage1_FW | eta | 14 | redistribution_stage1_FW_coef_14 | 0.014595 | 0.010817 |
| RL | redistribution_stage1_FW | eta | 15 | redistribution_stage1_FW_coef_15 | -0.007000 | 0.006203 |
| RL | redistribution_stage1_FW | eta | 16 | redistribution_stage1_FW_coef_16 | -0.003479 | 0.014633 |
| RL | redistribution_stage1_FW | eta | 17 | redistribution_stage1_FW_coef_17 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_FW | eta | 18 | redistribution_stage1_FW_coef_18 | -0.000746 | 0.007659 |
| RL | redistribution_stage1_FW | eta | 19 | redistribution_stage1_FW_coef_19 | -0.002091 | 0.004944 |
| RL | redistribution_stage1_FW | eta | 20 | redistribution_stage1_FW_coef_20 | -0.022001 | 0.023328 |
| RL | redistribution_stage1_FW | eta | 21 | redistribution_stage1_FW_coef_21 | 0.005185 | 0.003899 |
| RL | redistribution_stage1_FW | eta | 22 | redistribution_stage1_FW_coef_22 | -0.002336 | 0.024275 |
| RL | redistribution_stage1_PJ | eta | 0 | redistribution_stage1_PJ_coef_0 | 0.114062 | 0.039345 |
| RL | redistribution_stage1_PJ | eta | 1 | redistribution_stage1_PJ_coef_1 | 0.007312 | 0.007330 |
| RL | redistribution_stage1_PJ | eta | 2 | redistribution_stage1_PJ_coef_2 | 0.022720 | 0.021867 |
| RL | redistribution_stage1_PJ | eta | 3 | redistribution_stage1_PJ_coef_3 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_PJ | eta | 4 | redistribution_stage1_PJ_coef_4 | -0.143301 | 0.031177 |
| RL | redistribution_stage1_PJ | eta | 5 | redistribution_stage1_PJ_coef_5 | -0.015829 | 0.020931 |
| RL | redistribution_stage1_PJ | eta | 6 | redistribution_stage1_PJ_coef_6 | 0.009917 | 0.002267 |
| RL | redistribution_stage1_PJ | eta | 7 | redistribution_stage1_PJ_coef_7 | 0.014974 | 0.007313 |
| RL | redistribution_stage1_PJ | eta | 8 | redistribution_stage1_PJ_coef_8 | 0.153474 | 0.012013 |
| RL | redistribution_stage1_PJ | eta | 9 | redistribution_stage1_PJ_coef_9 | 0.001745 | 0.007861 |
| RL | redistribution_stage1_PJ | eta | 10 | redistribution_stage1_PJ_coef_10 | 0.024919 | 0.002510 |
| RL | redistribution_stage1_PJ | eta | 11 | redistribution_stage1_PJ_coef_11 | 0.069315 | 0.033787 |
| RL | redistribution_stage1_PJ | eta | 12 | redistribution_stage1_PJ_coef_12 | 0.015195 | 0.005073 |
| RL | redistribution_stage1_PJ | eta | 13 | redistribution_stage1_PJ_coef_13 | 0.188649 | 0.024694 |
| RL | redistribution_stage1_PJ | eta | 14 | redistribution_stage1_PJ_coef_14 | 0.054905 | 0.011789 |
| RL | redistribution_stage1_PJ | eta | 15 | redistribution_stage1_PJ_coef_15 | 0.007035 | 0.008752 |
| RL | redistribution_stage1_PJ | eta | 16 | redistribution_stage1_PJ_coef_16 | 0.013576 | 0.023827 |
| RL | redistribution_stage1_PJ | eta | 17 | redistribution_stage1_PJ_coef_17 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_PJ | eta | 18 | redistribution_stage1_PJ_coef_18 | 0.014711 | 0.013295 |
| RL | redistribution_stage1_PJ | eta | 19 | redistribution_stage1_PJ_coef_19 | -0.003542 | 0.010488 |
| RL | redistribution_stage1_PJ | eta | 20 | redistribution_stage1_PJ_coef_20 | -0.065602 | 0.032771 |
| RL | redistribution_stage1_PJ | eta | 21 | redistribution_stage1_PJ_coef_21 | -0.006280 | 0.010074 |
| RL | redistribution_stage1_PJ | eta | 22 | redistribution_stage1_PJ_coef_22 | -0.054588 | 0.020693 |
| RL | redistribution_stage2_v2 | eta | 0 | redistribution_stage2_v2_coef_0 | -0.451223 | 0.001056 |
| RL | redistribution_stage2_v2 | eta | 1 | redistribution_stage2_v2_coef_1 | -0.075204 | 2.932921e-05 |
| RL | redistribution_stage2_v2 | eta | 2 | redistribution_stage2_v2_coef_2 | 0.027103 | 0.000659 |
| RL | redistribution_stage2_v2 | eta | 3 | redistribution_stage2_v2_coef_3 | -0.101671 | 0.001612 |
| RL | redistribution_stage2_v2 | eta | 4 | redistribution_stage2_v2_coef_4 | 0.000000 | 1.000000 |
| RL | redistribution_stage2_v2 | eta | 5 | redistribution_stage2_v2_coef_5 | 0.464009 | 0.002478 |
| RL | redistribution_stage2_v2 | eta | 6 | redistribution_stage2_v2_coef_6 | 0.016869 | 0.002636 |
| RL | redistribution_stage2_v2 | eta | 7 | redistribution_stage2_v2_coef_7 | -0.053226 | 0.001532 |
| RL | redistribution_stage2_v2 | eta | 8 | redistribution_stage2_v2_coef_8 | -0.002916 | 0.001638 |
| RL | redistribution_stage2_v2 | eta | 9 | redistribution_stage2_v2_coef_9 | -0.127377 | 0.002079 |
| RL | redistribution_stage2_v2 | eta | 10 | redistribution_stage2_v2_coef_10 | 0.029139 | 0.002021 |
| RL | redistribution_stage2_v2 | eta | 11 | redistribution_stage2_v2_coef_11 | -0.050063 | 0.000998 |
| RL | redistribution_stage2_v2 | eta | 12 | redistribution_stage2_v2_coef_12 | -0.075365 | 0.001026 |
| RL | redistribution_stage2_v2 | eta | 13 | redistribution_stage2_v2_coef_13 | 0.004549 | 0.000235 |
| RL | redistribution_stage2_v2 | eta | 14 | redistribution_stage2_v2_coef_14 | -0.002182 | 0.003097 |
| RL | redistribution_stage2_v2 | eta | 15 | redistribution_stage2_v2_coef_15 | -0.005930 | 0.000984 |
| RL | redistribution_stage2_v2 | eta | 16 | redistribution_stage2_v2_coef_16 | 0.020135 | 0.001987 |
| RL | redistribution_stage2_v2 | eta | 17 | redistribution_stage2_v2_coef_17 | 0.592462 | 0.000134 |
| RL | redistribution_stage2_v2 | eta | 18 | redistribution_stage2_v2_coef_18 | -0.101263 | 0.000573 |
| RL | redistribution_stage2_v2 | eta | 19 | redistribution_stage2_v2_coef_19 | 0.825025 | 0.000362 |
| RL | redistribution_stage2_v4 | eta | 0 | redistribution_stage2_v4_coef_0 | -0.407234 | 0.005262 |
| RL | redistribution_stage2_v4 | eta | 1 | redistribution_stage2_v4_coef_1 | -0.067872 | 0.000146 |
| RL | redistribution_stage2_v4 | eta | 2 | redistribution_stage2_v4_coef_2 | -0.039650 | 0.002376 |
| RL | redistribution_stage2_v4 | eta | 3 | redistribution_stage2_v4_coef_3 | -0.022891 | 0.023228 |
| RL | redistribution_stage2_v4 | eta | 4 | redistribution_stage2_v4_coef_4 | 0.000000 | 1.000000 |
| RL | redistribution_stage2_v4 | eta | 5 | redistribution_stage2_v4_coef_5 | 0.325883 | 0.010164 |
| RL | redistribution_stage2_v4 | eta | 6 | redistribution_stage2_v4_coef_6 | 0.108442 | 0.036190 |
| RL | redistribution_stage2_v4 | eta | 7 | redistribution_stage2_v4_coef_7 | -0.047792 | 0.004561 |
| RL | redistribution_stage2_v4 | eta | 8 | redistribution_stage2_v4_coef_8 | -0.035729 | 0.004960 |
| RL | redistribution_stage2_v4 | eta | 9 | redistribution_stage2_v4_coef_9 | -0.278102 | 0.021152 |
| RL | redistribution_stage2_v4 | eta | 10 | redistribution_stage2_v4_coef_10 | -0.019151 | 0.021298 |
| RL | redistribution_stage2_v4 | eta | 11 | redistribution_stage2_v4_coef_11 | -0.075101 | 0.006413 |
| RL | redistribution_stage2_v4 | eta | 12 | redistribution_stage2_v4_coef_12 | -0.062282 | 0.006960 |
| RL | redistribution_stage2_v4 | eta | 13 | redistribution_stage2_v4_coef_13 | -0.020319 | 0.003924 |
| RL | redistribution_stage2_v4 | eta | 14 | redistribution_stage2_v4_coef_14 | -0.303658 | 0.016767 |
| RL | redistribution_stage2_v4 | eta | 15 | redistribution_stage2_v4_coef_15 | -0.058047 | 0.026687 |
| RL | redistribution_stage2_v4 | eta | 16 | redistribution_stage2_v4_coef_16 | 0.025274 | 0.012289 |
| RL | redistribution_stage2_v4 | eta | 17 | redistribution_stage2_v4_coef_17 | 0.299796 | 0.000972 |
| RL | redistribution_stage2_v4 | eta | 18 | redistribution_stage2_v4_coef_18 | 0.048238 | 0.001603 |
| RL | redistribution_stage2_v4 | eta | 19 | redistribution_stage2_v4_coef_19 | 1.944382 | 0.002276 |
| RL | q_no_td_modify | beta | 0 | intercept | 0.034986 | 0.053024 |
| RL | q_no_td_modify | beta | 1 | weekday_vs_weekend | -0.112223 | 0.086463 |
| RL | q_no_td_modify | beta | 2 | slot_pm | -0.018101 | 0.001873 |
| RL | q_no_td_modify | beta | 3 | E_w | 0.009175 | 0.029568 |
| RL | q_no_td_modify | beta | 4 | b_hat | 1.143865 | 0.034769 |
| RL | q_no_td_modify | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify | beta | 6 | M_Y_anticipated_affect_ewma | 0.240865 | 0.007541 |
| RL | q_no_td_modify | beta | 7 | M_Y_fourSC_ewma | -0.018059 | 0.006733 |
| RL | q_no_td_modify | beta | 8 | M_E_pageview_ewma | 0.002751 | 0.000191 |
| RL | q_no_td_modify | beta | 9 | M_E_fitbit_wear_ewma | -0.124686 | 0.003074 |
| RL | q_no_td_modify | beta | 10 | M_E_survey_complete_ewma | -0.037735 | 0.004683 |
| RL | q_no_td_modify | beta | 11 | yesterday_step_count | -0.005086 | 0.000819 |
| RL | q_no_td_modify | beta | 12 | prior2hour_step_count | -0.008950 | 0.000759 |
| RL | q_no_td_modify | beta | 13 | active_status_fraction_7days | -0.006717 | 0.017864 |
| RL | q_no_td_modify | beta | 14 | recent_burden | -0.010246 | 0.001690 |
| RL | q_no_td_modify | beta | 15 | walk_interaction_7d | -0.038719 | 0.020589 |
| RL | q_no_td_modify | beta | 16 | A | 0.008041 | 0.000811 |
| RL | q_no_td_modify | beta | 17 | A*E_w | 0.001490 | 0.000938 |
| RL | q_no_td_modify | beta | 18 | A*b_hat | -0.001366 | 0.001799 |
| RL | q_no_td_modify | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify | beta | 20 | A*yesterday_step_count | -0.003678 | 0.002464 |
| RL | q_no_td_modify | beta | 21 | A*prior2hour_step_count | 0.005164 | 0.001170 |
| RL | q_no_td_modify | beta | 22 | A*active_status_fraction_7days | -0.007013 | 0.001236 |
| RL | q_no_td_modify | beta | 23 | A*recent_burden | -0.003328 | 0.000844 |
| RL | q_no_td_modify | beta | 24 | A*walk_interaction_7d | -0.001593 | 0.001603 |
| RL | q_no_td_modify_g09 | beta | 0 | intercept | 0.053454 | 0.088192 |
| RL | q_no_td_modify_g09 | beta | 1 | weekday_vs_weekend | -0.202038 | 0.212634 |
| RL | q_no_td_modify_g09 | beta | 2 | slot_pm | -0.036957 | 0.006212 |
| RL | q_no_td_modify_g09 | beta | 3 | E_w | 0.015053 | 0.047928 |
| RL | q_no_td_modify_g09 | beta | 4 | b_hat | 1.551120 | 0.050885 |
| RL | q_no_td_modify_g09 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g09 | beta | 6 | M_Y_anticipated_affect_ewma | 0.381104 | 0.016634 |
| RL | q_no_td_modify_g09 | beta | 7 | M_Y_fourSC_ewma | -0.027290 | 0.011801 |
| RL | q_no_td_modify_g09 | beta | 8 | M_E_pageview_ewma | 0.005315 | 0.000323 |
| RL | q_no_td_modify_g09 | beta | 9 | M_E_fitbit_wear_ewma | -0.200460 | 0.005146 |
| RL | q_no_td_modify_g09 | beta | 10 | M_E_survey_complete_ewma | -0.053457 | 0.007352 |
| RL | q_no_td_modify_g09 | beta | 11 | yesterday_step_count | -0.002239 | 0.001475 |
| RL | q_no_td_modify_g09 | beta | 12 | prior2hour_step_count | -0.014120 | 0.001320 |
| RL | q_no_td_modify_g09 | beta | 13 | active_status_fraction_7days | 0.026442 | 0.030574 |
| RL | q_no_td_modify_g09 | beta | 14 | recent_burden | -0.013937 | 0.003314 |
| RL | q_no_td_modify_g09 | beta | 15 | walk_interaction_7d | -0.030195 | 0.030399 |
| RL | q_no_td_modify_g09 | beta | 16 | A | 0.006977 | 0.002532 |
| RL | q_no_td_modify_g09 | beta | 17 | A*E_w | 0.001939 | 0.001922 |
| RL | q_no_td_modify_g09 | beta | 18 | A*b_hat | -0.003583 | 0.003188 |
| RL | q_no_td_modify_g09 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g09 | beta | 20 | A*yesterday_step_count | -0.007056 | 0.004648 |
| RL | q_no_td_modify_g09 | beta | 21 | A*prior2hour_step_count | 0.007346 | 0.001939 |
| RL | q_no_td_modify_g09 | beta | 22 | A*active_status_fraction_7days | -0.008268 | 0.002733 |
| RL | q_no_td_modify_g09 | beta | 23 | A*recent_burden | -0.005537 | 0.001525 |
| RL | q_no_td_modify_g09 | beta | 24 | A*walk_interaction_7d | 0.004177 | 0.002887 |
| RL | q_no_td_modify_g099 | beta | 0 | intercept | 0.059838 | 0.099469 |
| RL | q_no_td_modify_g099 | beta | 1 | weekday_vs_weekend | -0.228108 | 0.261271 |
| RL | q_no_td_modify_g099 | beta | 2 | slot_pm | -0.043038 | 0.008162 |
| RL | q_no_td_modify_g099 | beta | 3 | E_w | 0.016946 | 0.053651 |
| RL | q_no_td_modify_g099 | beta | 4 | b_hat | 1.667455 | 0.055537 |
| RL | q_no_td_modify_g099 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g099 | beta | 6 | M_Y_anticipated_affect_ewma | 0.424625 | 0.020379 |
| RL | q_no_td_modify_g099 | beta | 7 | M_Y_fourSC_ewma | -0.029967 | 0.013271 |
| RL | q_no_td_modify_g099 | beta | 8 | M_E_pageview_ewma | 0.006160 | 0.000368 |
| RL | q_no_td_modify_g099 | beta | 9 | M_E_fitbit_wear_ewma | -0.223363 | 0.006094 |
| RL | q_no_td_modify_g099 | beta | 10 | M_E_survey_complete_ewma | -0.058035 | 0.008402 |
| RL | q_no_td_modify_g099 | beta | 11 | yesterday_step_count | -0.001086 | 0.001754 |
| RL | q_no_td_modify_g099 | beta | 12 | prior2hour_step_count | -0.015732 | 0.001505 |
| RL | q_no_td_modify_g099 | beta | 13 | active_status_fraction_7days | 0.038241 | 0.035100 |
| RL | q_no_td_modify_g099 | beta | 14 | recent_burden | -0.015011 | 0.003953 |
| RL | q_no_td_modify_g099 | beta | 15 | walk_interaction_7d | -0.026549 | 0.033988 |
| RL | q_no_td_modify_g099 | beta | 16 | A | 0.006577 | 0.003328 |
| RL | q_no_td_modify_g099 | beta | 17 | A*E_w | 0.002061 | 0.002303 |
| RL | q_no_td_modify_g099 | beta | 18 | A*b_hat | -0.004283 | 0.003665 |
| RL | q_no_td_modify_g099 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g099 | beta | 20 | A*yesterday_step_count | -0.008125 | 0.005342 |
| RL | q_no_td_modify_g099 | beta | 21 | A*prior2hour_step_count | 0.008011 | 0.002200 |
| RL | q_no_td_modify_g099 | beta | 22 | A*active_status_fraction_7days | -0.008688 | 0.003480 |
| RL | q_no_td_modify_g099 | beta | 23 | A*recent_burden | -0.006257 | 0.001757 |
| RL | q_no_td_modify_g099 | beta | 24 | A*walk_interaction_7d | 0.006065 | 0.003379 |
| RL | q_redistribution_v2 | beta | 0 | intercept | 0.563288 | 0.232343 |
| RL | q_redistribution_v2 | beta | 1 | weekday_vs_weekend | -0.230681 | 0.077807 |
| RL | q_redistribution_v2 | beta | 2 | slot_pm | -0.015677 | 0.002236 |
| RL | q_redistribution_v2 | beta | 3 | E_w | 0.753504 | 0.522403 |
| RL | q_redistribution_v2 | beta | 4 | b_hat | 1.215632 | 0.075286 |
| RL | q_redistribution_v2 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v2 | beta | 6 | M_Y_anticipated_affect_ewma | 0.652694 | 0.027716 |
| RL | q_redistribution_v2 | beta | 7 | M_Y_fourSC_ewma | -0.047029 | 0.023149 |
| RL | q_redistribution_v2 | beta | 8 | M_E_pageview_ewma | -0.010218 | 0.005424 |
| RL | q_redistribution_v2 | beta | 9 | M_E_fitbit_wear_ewma | -0.327615 | 0.015683 |
| RL | q_redistribution_v2 | beta | 10 | M_E_survey_complete_ewma | -0.022393 | 0.017756 |
| RL | q_redistribution_v2 | beta | 11 | yesterday_step_count | 0.064243 | 0.006159 |
| RL | q_redistribution_v2 | beta | 12 | prior2hour_step_count | 0.039783 | 0.002771 |
| RL | q_redistribution_v2 | beta | 13 | active_status_fraction_7days | 0.054441 | 0.176931 |
| RL | q_redistribution_v2 | beta | 14 | recent_burden | 0.137419 | 0.009399 |
| RL | q_redistribution_v2 | beta | 15 | walk_interaction_7d | 1.188349 | 0.106682 |
| RL | q_redistribution_v2 | beta | 16 | A | 0.082350 | 0.009477 |
| RL | q_redistribution_v2 | beta | 17 | A*E_w | 0.007324 | 0.001282 |
| RL | q_redistribution_v2 | beta | 18 | A*b_hat | -0.022906 | 0.002501 |
| RL | q_redistribution_v2 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v2 | beta | 20 | A*yesterday_step_count | 0.031767 | 0.001081 |
| RL | q_redistribution_v2 | beta | 21 | A*prior2hour_step_count | 0.005006 | 0.002762 |
| RL | q_redistribution_v2 | beta | 22 | A*active_status_fraction_7days | -0.011091 | 0.006825 |
| RL | q_redistribution_v2 | beta | 23 | A*recent_burden | 0.025450 | 0.000908 |
| RL | q_redistribution_v2 | beta | 24 | A*walk_interaction_7d | -0.030086 | 0.008847 |
| RL | q_redistribution_v4 | beta | 0 | intercept | -0.048738 | 0.411964 |
| RL | q_redistribution_v4 | beta | 1 | weekday_vs_weekend | -0.089584 | 0.066454 |
| RL | q_redistribution_v4 | beta | 2 | slot_pm | -0.006581 | 0.001475 |
| RL | q_redistribution_v4 | beta | 3 | E_w | -0.285285 | 0.118197 |
| RL | q_redistribution_v4 | beta | 4 | b_hat | 1.682694 | 0.108214 |
| RL | q_redistribution_v4 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v4 | beta | 6 | M_Y_anticipated_affect_ewma | 0.201490 | 0.038387 |
| RL | q_redistribution_v4 | beta | 7 | M_Y_fourSC_ewma | -0.161756 | 0.103921 |
| RL | q_redistribution_v4 | beta | 8 | M_E_pageview_ewma | -0.015249 | 0.023049 |
| RL | q_redistribution_v4 | beta | 9 | M_E_fitbit_wear_ewma | -0.097074 | 0.020058 |
| RL | q_redistribution_v4 | beta | 10 | M_E_survey_complete_ewma | 0.012804 | 0.083268 |
| RL | q_redistribution_v4 | beta | 11 | yesterday_step_count | -0.036744 | 0.065111 |
| RL | q_redistribution_v4 | beta | 12 | prior2hour_step_count | 0.059460 | 0.015872 |
| RL | q_redistribution_v4 | beta | 13 | active_status_fraction_7days | 0.454521 | 0.211925 |
| RL | q_redistribution_v4 | beta | 14 | recent_burden | 0.017338 | 0.086183 |
| RL | q_redistribution_v4 | beta | 15 | walk_interaction_7d | 0.341976 | 0.321835 |
| RL | q_redistribution_v4 | beta | 16 | A | 0.092797 | 0.027380 |
| RL | q_redistribution_v4 | beta | 17 | A*E_w | -0.002787 | 0.004612 |
| RL | q_redistribution_v4 | beta | 18 | A*b_hat | 0.002684 | 0.007103 |
| RL | q_redistribution_v4 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v4 | beta | 20 | A*yesterday_step_count | 0.048576 | 0.005209 |
| RL | q_redistribution_v4 | beta | 21 | A*prior2hour_step_count | -0.014020 | 0.021528 |
| RL | q_redistribution_v4 | beta | 22 | A*active_status_fraction_7days | -0.088274 | 0.012470 |
| RL | q_redistribution_v4 | beta | 23 | A*recent_burden | -0.006156 | 0.002361 |
| RL | q_redistribution_v4 | beta | 24 | A*walk_interaction_7d | -0.077618 | 0.061771 |

## Joint Modified-TD Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| Joint Modified TD | q_td_modify_joint | eta | 0 | eta_intercept | -0.120555 | 0.102985 |
| Joint Modified TD | q_td_modify_joint | eta | 1 | eta_E_w | -0.005215 | 0.044061 |
| Joint Modified TD | q_td_modify_joint | eta | 2 | eta_b_hat | -0.579779 | 0.034258 |
| Joint Modified TD | q_td_modify_joint | eta | 3 | eta_b_tilde | 0.000000 | 1.000001 |
| Joint Modified TD | q_td_modify_joint | beta | 4 | beta_intercept | -0.023970 | 0.066740 |
| Joint Modified TD | q_td_modify_joint | beta | 5 | beta_weekday_vs_weekend | -0.242897 | 0.178682 |
| Joint Modified TD | q_td_modify_joint | beta | 6 | beta_slot_pm | -0.026398 | 0.003264 |
| Joint Modified TD | q_td_modify_joint | beta | 7 | beta_E_w | -0.001986 | 0.034473 |
| Joint Modified TD | q_td_modify_joint | beta | 8 | beta_b_hat | -0.117701 | 0.020200 |
| Joint Modified TD | q_td_modify_joint | beta | 9 | beta_b_tilde | 0.000000 | 1.000001 |
| Joint Modified TD | q_td_modify_joint | beta | 10 | beta_M_Y_anticipated_affect_ewma | 0.596332 | 0.015489 |
| Joint Modified TD | q_td_modify_joint | beta | 11 | beta_M_Y_fourSC_ewma | -0.006580 | 0.010258 |
| Joint Modified TD | q_td_modify_joint | beta | 12 | beta_M_E_pageview_ewma | 0.006821 | 0.000772 |
| Joint Modified TD | q_td_modify_joint | beta | 13 | beta_M_E_fitbit_wear_ewma | -0.317007 | 0.003506 |
| Joint Modified TD | q_td_modify_joint | beta | 14 | beta_M_E_survey_complete_ewma | -0.036297 | 0.007277 |
| Joint Modified TD | q_td_modify_joint | beta | 15 | beta_yesterday_step_count | -0.001770 | 0.000877 |
| Joint Modified TD | q_td_modify_joint | beta | 16 | beta_prior2hour_step_count | -0.005227 | 0.000915 |
| Joint Modified TD | q_td_modify_joint | beta | 17 | beta_active_status_fraction_7days | -0.002620 | 0.016288 |
| Joint Modified TD | q_td_modify_joint | beta | 18 | beta_recent_burden | -0.007608 | 0.000864 |
| Joint Modified TD | q_td_modify_joint | beta | 19 | beta_walk_interaction_7d | 0.029190 | 0.009020 |
| Joint Modified TD | q_td_modify_joint | beta | 20 | beta_A | -0.012729 | 0.001942 |
| Joint Modified TD | q_td_modify_joint | beta | 21 | beta_A*E_w | 0.000478 | 0.001680 |
| Joint Modified TD | q_td_modify_joint | beta | 22 | beta_A*b_hat | -0.068789 | 0.002319 |
| Joint Modified TD | q_td_modify_joint | beta | 23 | beta_A*b_tilde | 0.000000 | 1.000001 |
| Joint Modified TD | q_td_modify_joint | beta | 24 | beta_A*yesterday_step_count | 0.002071 | 0.003998 |
| Joint Modified TD | q_td_modify_joint | beta | 25 | beta_A*prior2hour_step_count | -0.004236 | 0.001329 |
| Joint Modified TD | q_td_modify_joint | beta | 26 | beta_A*active_status_fraction_7days | -0.001847 | 0.001738 |
| Joint Modified TD | q_td_modify_joint | beta | 27 | beta_A*recent_burden | 0.000609 | 0.001647 |
| Joint Modified TD | q_td_modify_joint | beta | 28 | beta_A*walk_interaction_7d | 0.002010 | 0.001663 |

## Pooled PF Regression Coefficients

Coefficients are from the all-user stacked ridge fits used as PF prior means. `p_value` uses ridge sandwich standard errors and is approximate because ridge shrinks coefficients. `identified=False` marks zero-variance design columns; SE / p-values are omitted for those rows.

| model | outcome | index | feature | coefficient | identified | feature_std | std_error | t_stat | p_value | significant_0.05 | significant_0.01 | n_obs | n_features | ridge_alpha | residual_sigma2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fourSC | 4hour_step_norm | 0 | intercept | -0.048286 | False | 0.000000 |  |  |  | False | False | 2890 | 19 | 1.000000 | 0.529944 |
| fourSC | 4hour_step_norm | 1 | stepCountNext4HourLag1 | 0.497640 | True | 0.965022 | 0.016734 | 29.738128 | 1.205992e-169 | True | True | 2890 | 19 | 1.000000 | 0.529944 |
| fourSC | 4hour_step_norm | 2 | yesterdayStepCount | -0.161932 | True | 0.928522 | 0.022839 | -7.090030 | 1.681016e-12 | True | True | 2890 | 19 | 1.000000 | 0.529944 |
| fourSC | 4hour_step_norm | 3 | stepCountLast7DaysEma | 0.499996 | True | 0.733979 | 0.022104 | 22.620352 | 2.161301e-104 | True | True | 2890 | 19 | 1.000000 | 0.529944 |
| fourSC | 4hour_step_norm | 4 | prior2HourStepCount | 0.099240 | True | 0.945854 | 0.020960 | 4.734748 | 2.299649e-06 | True | True | 2890 | 19 | 1.000000 | 0.529944 |
| fourSC | 4hour_step_norm | 5 | activitySuggestionsSentLast7Days | -0.037892 | True | 0.970948 | 0.020064 | -1.888545 | 0.059054 | False | False | 2890 | 19 | 1.000000 | 0.529944 |
| fourSC | 4hour_step_norm | 6 | activitySuggestionInteractLast7Days | 0.037644 | True | 0.353473 | 0.056101 | 0.671006 | 0.502271 | False | False | 2890 | 19 | 1.000000 | 0.529944 |
| fourSC | 4hour_step_norm | 7 | activeDaysLast7Days | 0.131898 | True | 0.260938 | 0.057432 | 2.296597 | 0.021713 | True | False | 2890 | 19 | 1.000000 | 0.529944 |
| fourSC | 4hour_step_norm | 8 | isWeekend | -0.036638 | True | 0.369659 | 0.036657 | -0.999496 | 0.317639 | False | False | 2890 | 19 | 1.000000 | 0.529944 |
| fourSC | 4hour_step_norm | 9 | decisionTimeSlot | -0.038970 | True | 0.497078 | 0.028569 | -1.364072 | 0.172652 | False | False | 2890 | 19 | 1.000000 | 0.529944 |
| fourSC | 4hour_step_norm | 10 | perceivedUtilityLastWeek | 0.017864 | True | 2.465144 | 0.008717 | 2.049393 | 0.040514 | True | False | 2890 | 19 | 1.000000 | 0.529944 |
| fourSC | 4hour_step_norm | 11 | caeAverageLastWeek | -0.052762 | True | 0.850337 | 0.023921 | -2.205697 | 0.027484 | True | False | 2890 | 19 | 1.000000 | 0.529944 |
| fourSC | 4hour_step_norm | 12 | Ah | -0.109031 | True | 0.499925 | 0.046789 | -2.330261 | 0.019861 | True | False | 2890 | 19 | 1.000000 | 0.529944 |
| fourSC | 4hour_step_norm | 13 | Ah*yesterdayStepCount | -0.002172 | True | 0.662751 | 0.030563 | -0.071073 | 0.943345 | False | False | 2890 | 19 | 1.000000 | 0.529944 |
| fourSC | 4hour_step_norm | 14 | Ah*prior2HourStepCount | 0.010920 | True | 0.664658 | 0.029424 | 0.371141 | 0.710560 | False | False | 2890 | 19 | 1.000000 | 0.529944 |
| fourSC | 4hour_step_norm | 15 | Ah*activitySuggestionsSentLast7Days | 0.028748 | True | 0.684392 | 0.028241 | 1.017945 | 0.308790 | False | False | 2890 | 19 | 1.000000 | 0.529944 |
| fourSC | 4hour_step_norm | 16 | Ah*activitySuggestionInteractLast7Days | 0.110169 | True | 0.343672 | 0.079457 | 1.386516 | 0.165697 | False | False | 2890 | 19 | 1.000000 | 0.529944 |
| fourSC | 4hour_step_norm | 17 | Ah*perceivedUtilityLastWeek | 0.007230 | True | 1.818825 | 0.011949 | 0.605046 | 0.545197 | False | False | 2890 | 19 | 1.000000 | 0.529944 |
| fourSC | 4hour_step_norm | 18 | Ah*caeAverageLastWeek | -0.004544 | True | 0.597408 | 0.033242 | -0.136685 | 0.891289 | False | False | 2890 | 19 | 1.000000 | 0.529944 |
| antic | anticipated_affect_norm | 0 | intercept | 0.422129 | False | 0.000000 |  |  |  | False | False | 817 | 15 | 1.000000 | 0.040611 |
| antic | anticipated_affect_norm | 1 | anticipated_affect_yesterday | 0.218613 | True | 0.329197 | 0.024955 | 8.760409 | 1.144484e-17 | True | True | 817 | 15 | 1.000000 | 0.040611 |
| antic | anticipated_affect_norm | 2 | active_status_fraction_7days | 0.075727 | True | 0.224466 | 0.032478 | 2.331651 | 0.019966 | True | False | 817 | 15 | 1.000000 | 0.040611 |
| antic | anticipated_affect_norm | 3 | is_weekend | -0.028077 | True | 0.370289 | 0.019176 | -1.464154 | 0.143544 | False | False | 817 | 15 | 1.000000 | 0.040611 |
| antic | anticipated_affect_norm | 4 | perceived_utility_lastweek | 0.007743 | True | 2.397863 | 0.004970 | 1.557915 | 0.119648 | False | False | 817 | 15 | 1.000000 | 0.040611 |
| antic | anticipated_affect_norm | 5 | CAE_avg_lastweek | 0.236253 | True | 0.803708 | 0.016255 | 14.534163 | 1.177905e-42 | True | True | 817 | 15 | 1.000000 | 0.040611 |
| antic | anticipated_affect_norm | 6 | recent_burden | 0.014369 | True | 0.958781 | 0.010815 | 1.328586 | 0.184362 | False | False | 817 | 15 | 1.000000 | 0.040611 |
| antic | anticipated_affect_norm | 7 | A0_morning | 0.004252 | True | 0.499865 | 0.017044 | 0.249468 | 0.803062 | False | False | 817 | 15 | 1.000000 | 0.040611 |
| antic | anticipated_affect_norm | 8 | A1_afternoon | 0.014142 | True | 0.499937 | 0.016918 | 0.835905 | 0.403457 | False | False | 817 | 15 | 1.000000 | 0.040611 |
| antic | anticipated_affect_norm | 9 | A0_morning_by_perceived_utility_lastweek | -0.001464 | True | 1.810012 | 0.006156 | -0.237824 | 0.812078 | False | False | 817 | 15 | 1.000000 | 0.040611 |
| antic | anticipated_affect_norm | 10 | A1_afternoon_by_perceived_utility_lastweek | -0.001489 | True | 1.790500 | 0.006154 | -0.241935 | 0.808892 | False | False | 817 | 15 | 1.000000 | 0.040611 |
| antic | anticipated_affect_norm | 11 | A0_morning_by_CAE_avg_lastweek | -0.004551 | True | 0.588850 | 0.018092 | -0.251538 | 0.801462 | False | False | 817 | 15 | 1.000000 | 0.040611 |
| antic | anticipated_affect_norm | 12 | A1_afternoon_by_CAE_avg_lastweek | -0.027552 | True | 0.561687 | 0.018084 | -1.523575 | 0.128009 | False | False | 817 | 15 | 1.000000 | 0.040611 |
| antic | anticipated_affect_norm | 13 | A0_morning_by_recent_burden | -0.009632 | True | 0.645439 | 0.016488 | -0.584205 | 0.559247 | False | False | 817 | 15 | 1.000000 | 0.040611 |
| antic | anticipated_affect_norm | 14 | A1_afternoon_by_recent_burden | -0.003091 | True | 0.628065 | 0.016530 | -0.187016 | 0.851695 | False | False | 817 | 15 | 1.000000 | 0.040611 |
| CAE | CAE_avg_norm | 0 | intercept | -0.125413 | False | 0.000000 |  |  |  | False | False | 213 | 4 | 1.000000 | 0.132797 |
| CAE | CAE_avg_norm | 1 | CAE_avg_lastweek | 0.883442 | True | 0.879745 | 0.037771 | 23.389160 | 2.866918e-60 | True | True | 213 | 4 | 1.000000 | 0.132797 |
| CAE | CAE_avg_norm | 2 | fourSC_ewma | -0.033341 | True | 0.550348 | 0.045257 | -0.736702 | 0.462130 | False | False | 213 | 4 | 1.000000 | 0.132797 |
| CAE | CAE_avg_norm | 3 | anticipated_affect_ewma | 0.175007 | True | 0.332321 | 0.094746 | 1.847123 | 0.066143 | False | False | 213 | 4 | 1.000000 | 0.132797 |

## Pooled PF GEE Coefficients

Population-averaged GEE fits on the same stacked PF designs, clustered by `ParticipantIdentifier`. `p_value` uses GEE sandwich standard errors with exchangeable working correlation. `identified=False` marks zero-variance design columns; SE / p-values are omitted for those rows. These are for inference/audit only; PF priors still use ridge.

| coefficient | robust_se | z_stat | p_value | significant_0.05 | significant_0.01 | model | outcome | index | feature | identified | feature_std | n_obs | n_clusters | n_features | working_correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -0.134203 |  |  |  | False | False | fourSC | 4hour_step_norm | 0 | intercept | False | 0.000000 | 2890 | 28 | 19 | exchangeable |
| -0.037452 | 0.041288 | -0.907076 | 0.364366 | False | False | fourSC | 4hour_step_norm | 1 | stepCountNext4HourLag1 | True | 0.965022 | 2890 | 28 | 19 | exchangeable |
| -0.017112 | 0.031581 | -0.541839 | 0.587929 | False | False | fourSC | 4hour_step_norm | 2 | yesterdayStepCount | True | 0.928522 | 2890 | 28 | 19 | exchangeable |
| 0.255376 | 0.047870 | 5.334733 | 9.568528e-08 | True | True | fourSC | 4hour_step_norm | 3 | stepCountLast7DaysEma | True | 0.733979 | 2890 | 28 | 19 | exchangeable |
| 0.133578 | 0.032503 | 4.109661 | 3.962398e-05 | True | True | fourSC | 4hour_step_norm | 4 | prior2HourStepCount | True | 0.945854 | 2890 | 28 | 19 | exchangeable |
| 0.005191 | 0.016798 | 0.309013 | 0.757312 | False | False | fourSC | 4hour_step_norm | 5 | activitySuggestionsSentLast7Days | True | 0.970948 | 2890 | 28 | 19 | exchangeable |
| 0.047910 | 0.068678 | 0.697602 | 0.485426 | False | False | fourSC | 4hour_step_norm | 6 | activitySuggestionInteractLast7Days | True | 0.353473 | 2890 | 28 | 19 | exchangeable |
| 0.031780 | 0.072378 | 0.439086 | 0.660600 | False | False | fourSC | 4hour_step_norm | 7 | activeDaysLast7Days | True | 0.260938 | 2890 | 28 | 19 | exchangeable |
| -0.007329 | 0.044807 | -0.163561 | 0.870076 | False | False | fourSC | 4hour_step_norm | 8 | isWeekend | True | 0.369659 | 2890 | 28 | 19 | exchangeable |
| -0.077987 | 0.062160 | -1.254606 | 0.209622 | False | False | fourSC | 4hour_step_norm | 9 | decisionTimeSlot | True | 0.497078 | 2890 | 28 | 19 | exchangeable |
| 0.025394 | 0.017162 | 1.479720 | 0.138948 | False | False | fourSC | 4hour_step_norm | 10 | perceivedUtilityLastWeek | True | 2.465144 | 2890 | 28 | 19 | exchangeable |
| -0.001273 | 0.035567 | -0.035789 | 0.971450 | False | False | fourSC | 4hour_step_norm | 11 | caeAverageLastWeek | True | 0.850337 | 2890 | 28 | 19 | exchangeable |
| -0.054024 | 0.054707 | -0.987520 | 0.323388 | False | False | fourSC | 4hour_step_norm | 12 | Ah | True | 0.499925 | 2890 | 28 | 19 | exchangeable |
| 0.011392 | 0.035076 | 0.324771 | 0.745354 | False | False | fourSC | 4hour_step_norm | 13 | Ah*yesterdayStepCount | True | 0.662751 | 2890 | 28 | 19 | exchangeable |
| -0.018574 | 0.029198 | -0.636155 | 0.524676 | False | False | fourSC | 4hour_step_norm | 14 | Ah*prior2HourStepCount | True | 0.664658 | 2890 | 28 | 19 | exchangeable |
| 0.007669 | 0.016886 | 0.454185 | 0.649696 | False | False | fourSC | 4hour_step_norm | 15 | Ah*activitySuggestionsSentLast7Days | True | 0.684392 | 2890 | 28 | 19 | exchangeable |
| 0.060762 | 0.085144 | 0.713644 | 0.475448 | False | False | fourSC | 4hour_step_norm | 16 | Ah*activitySuggestionInteractLast7Days | True | 0.343672 | 2890 | 28 | 19 | exchangeable |
| -0.001601 | 0.010535 | -0.151951 | 0.879225 | False | False | fourSC | 4hour_step_norm | 17 | Ah*perceivedUtilityLastWeek | True | 1.818825 | 2890 | 28 | 19 | exchangeable |
| -0.011767 | 0.026738 | -0.440087 | 0.659874 | False | False | fourSC | 4hour_step_norm | 18 | Ah*caeAverageLastWeek | True | 0.597408 | 2890 | 28 | 19 | exchangeable |
| 0.432566 |  |  |  | False | False | antic | anticipated_affect_norm | 0 | intercept | False | 0.000000 | 817 | 28 | 15 | exchangeable |
| 0.078868 | 0.032984 | 2.391074 | 0.016799 | True | False | antic | anticipated_affect_norm | 1 | anticipated_affect_yesterday | True | 0.329197 | 817 | 28 | 15 | exchangeable |
| 0.124503 | 0.038454 | 3.237682 | 0.001205 | True | True | antic | anticipated_affect_norm | 2 | active_status_fraction_7days | True | 0.224466 | 817 | 28 | 15 | exchangeable |
| -0.003703 | 0.021609 | -0.171375 | 0.863929 | False | False | antic | anticipated_affect_norm | 3 | is_weekend | True | 0.370289 | 817 | 28 | 15 | exchangeable |
| 0.005175 | 0.007081 | 0.730763 | 0.464924 | False | False | antic | anticipated_affect_norm | 4 | perceived_utility_lastweek | True | 2.397863 | 817 | 28 | 15 | exchangeable |
| 0.135078 | 0.041112 | 3.285586 | 0.001018 | True | True | antic | anticipated_affect_norm | 5 | CAE_avg_lastweek | True | 0.803708 | 817 | 28 | 15 | exchangeable |
| 0.011389 | 0.006030 | 1.888585 | 0.058948 | False | False | antic | anticipated_affect_norm | 6 | recent_burden | True | 0.958781 | 817 | 28 | 15 | exchangeable |
| 0.009620 | 0.009035 | 1.064736 | 0.286996 | False | False | antic | anticipated_affect_norm | 7 | A0_morning | True | 0.499865 | 817 | 28 | 15 | exchangeable |
| 0.016824 | 0.009464 | 1.777699 | 0.075453 | False | False | antic | anticipated_affect_norm | 8 | A1_afternoon | True | 0.499937 | 817 | 28 | 15 | exchangeable |
| -0.001552 | 0.002583 | -0.600732 | 0.548019 | False | False | antic | anticipated_affect_norm | 9 | A0_morning_by_perceived_utility_lastweek | True | 1.810012 | 817 | 28 | 15 | exchangeable |
| -0.002423 | 0.003964 | -0.611349 | 0.540969 | False | False | antic | anticipated_affect_norm | 10 | A1_afternoon_by_perceived_utility_lastweek | True | 1.790500 | 817 | 28 | 15 | exchangeable |
| 0.017288 | 0.012067 | 1.432709 | 0.151941 | False | False | antic | anticipated_affect_norm | 11 | A0_morning_by_CAE_avg_lastweek | True | 0.588850 | 817 | 28 | 15 | exchangeable |
| -0.016652 | 0.015605 | -1.067077 | 0.285937 | False | False | antic | anticipated_affect_norm | 12 | A1_afternoon_by_CAE_avg_lastweek | True | 0.561687 | 817 | 28 | 15 | exchangeable |
| -0.005571 | 0.011355 | -0.490599 | 0.623710 | False | False | antic | anticipated_affect_norm | 13 | A0_morning_by_recent_burden | True | 0.645439 | 817 | 28 | 15 | exchangeable |
| -0.014224 | 0.012667 | -1.122895 | 0.261482 | False | False | antic | anticipated_affect_norm | 14 | A1_afternoon_by_recent_burden | True | 0.628065 | 817 | 28 | 15 | exchangeable |
| -0.095100 |  |  |  | False | False | CAE | CAE_avg_norm | 0 | intercept | False | 0.000000 | 213 | 28 | 4 | exchangeable |
| 0.931242 | 0.041343 | 22.524531 | 2.386713e-112 | True | True | CAE | CAE_avg_norm | 1 | CAE_avg_lastweek | True | 0.879745 | 213 | 28 | 4 | exchangeable |
| -0.028528 | 0.025368 | -1.124605 | 0.260757 | False | False | CAE | CAE_avg_norm | 2 | fourSC_ewma | True | 0.550348 | 213 | 28 | 4 | exchangeable |
| 0.128522 | 0.106270 | 1.209390 | 0.226513 | False | False | CAE | CAE_avg_norm | 3 | anticipated_affect_ewma | True | 0.332321 | 213 | 28 | 4 | exchangeable |

## Pooled RL Q GEE Coefficients

Gaussian GEE on the final fitted-Q regression from pooled FQI (``phi_obs`` vs bootstrap targets), clustered by participant. ``identified=False`` marks structurally unused features with zero design variance (e.g. ``b_tilde``, masked day-6 mediators); SE / p-values are omitted for those rows. RL priors still use ridge-FQI for ``mu_0_micro``; all-zero design columns get ``UNIDENTIFIED_PRIOR_VAR`` instead of the ``MIN_SIGMA2`` floor.

| coefficient | robust_se | z_stat | p_value | significant_0.05 | significant_0.01 | model | outcome | index | feature | identified | feature_std | n_obs | n_clusters | n_features | working_correlation | fqi_iters | ridge_alpha | residual_sigma2 | block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.042361 |  |  |  | False | False | q_no_td_modify | fqi_target | 0 | intercept | False | 0.000000 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| -0.112627 | 0.030160 | -3.734287 | 0.000188 | True | True | q_no_td_modify | fqi_target | 1 | weekday_vs_weekend | True | 0.372678 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| -0.018832 | 0.011895 | -1.583182 | 0.113380 | False | False | q_no_td_modify | fqi_target | 2 | slot_pm | True | 0.500000 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| 0.009551 | 0.001153 | 8.285562 | 1.175526e-16 | True | True | q_no_td_modify | fqi_target | 3 | E_w | True | 2.447340 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| 1.153071 | 0.005720 | 201.570355 | 0.000000 | True | True | q_no_td_modify | fqi_target | 4 | b_hat | True | 0.827772 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| -7.639714e-17 |  |  |  | False | False | q_no_td_modify | fqi_target | 5 | b_tilde | False | 0.000000 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| 0.239116 | 0.013224 | 18.081734 | 4.438980e-73 | True | True | q_no_td_modify | fqi_target | 6 | M_Y_anticipated_affect_ewma | True | 0.311549 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| -0.017040 | 0.001482 | -11.500450 | 1.312302e-30 | True | True | q_no_td_modify | fqi_target | 7 | M_Y_fourSC_ewma | True | 1.001192 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| 0.001398 | 0.001767 | 0.791193 | 0.428831 | False | False | q_no_td_modify | fqi_target | 8 | M_E_pageview_ewma | True | 1.043961 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| -0.122595 | 0.005562 | -22.041564 | 1.150935e-107 | True | True | q_no_td_modify | fqi_target | 9 | M_E_fitbit_wear_ewma | True | 0.412251 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| -0.040027 | 0.005360 | -7.467144 | 8.195389e-14 | True | True | q_no_td_modify | fqi_target | 10 | M_E_survey_complete_ewma | True | 0.411996 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| -0.006484 | 0.004326 | -1.498780 | 0.133931 | False | False | q_no_td_modify | fqi_target | 11 | yesterday_step_count | True | 0.916963 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| -0.012143 | 0.004074 | -2.980628 | 0.002877 | True | True | q_no_td_modify | fqi_target | 12 | prior2hour_step_count | True | 0.934159 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| -0.016655 | 0.010913 | -1.526201 | 0.126960 | False | False | q_no_td_modify | fqi_target | 13 | active_status_fraction_7days | True | 0.266148 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| -0.010888 | 0.004033 | -2.699379 | 0.006947 | True | True | q_no_td_modify | fqi_target | 14 | recent_burden | True | 0.933762 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| -0.041303 | 0.008066 | -5.120479 | 3.047602e-07 | True | True | q_no_td_modify | fqi_target | 15 | walk_interaction_7d | True | 0.354304 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| 0.004321 | 0.015200 | 0.284294 | 0.776185 | False | False | q_no_td_modify | fqi_target | 16 | A | True | 0.499915 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| 0.000418 | 0.001723 | 0.242585 | 0.808327 | False | False | q_no_td_modify | fqi_target | 17 | A*E_w | True | 1.794385 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| -0.001653 | 0.010920 | -0.151325 | 0.879719 | False | False | q_no_td_modify | fqi_target | 18 | A*b_hat | True | 0.584148 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| -6.945407e-18 |  |  |  | False | False | q_no_td_modify | fqi_target | 19 | A*b_tilde | False | 0.000000 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| -0.004448 | 0.006577 | -0.676216 | 0.498904 | False | False | q_no_td_modify | fqi_target | 20 | A*yesterday_step_count | True | 0.656410 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| 0.005508 | 0.004355 | 1.264821 | 0.205935 | False | False | q_no_td_modify | fqi_target | 21 | A*prior2hour_step_count | True | 0.656112 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| -0.000696 | 0.019436 | -0.035820 | 0.971426 | False | False | q_no_td_modify | fqi_target | 22 | A*active_status_fraction_7days | True | 0.375177 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| -0.002336 | 0.004972 | -0.469842 | 0.638468 | False | False | q_no_td_modify | fqi_target | 23 | A*recent_burden | True | 0.662888 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
| -0.001403 | 0.014754 | -0.095067 | 0.924262 | False | False | q_no_td_modify | fqi_target | 24 | A*walk_interaction_7d | True | 0.334802 | 3360 | 28 | 25 | exchangeable | 25 | 1.000000 | 0.025254 | beta |
