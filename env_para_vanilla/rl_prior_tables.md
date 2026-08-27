# Prior Mean and Variance Summary

`prior_variance` is the diagonal entry of the prior covariance. For `q_td_modify_joint`, the full covariance remains in `rl_priors.json`; this table reports marginal variances.

## PF Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| PF | fourSC | theta | 0 | intercept | -0.032248 | 0.052740 |
| PF | fourSC | theta | 1 | stepCountNext4HourLag1 | 0.463390 | 0.038741 |
| PF | fourSC | theta | 2 | yesterdayStepCount | -0.127085 | 0.011851 |
| PF | fourSC | theta | 3 | stepCountLast7DaysEma | 0.369354 | 0.031748 |
| PF | fourSC | theta | 4 | prior2HourStepCount | 0.131687 | 0.031603 |
| PF | fourSC | theta | 5 | activitySuggestionsSentLast7Days | -0.025906 | 0.025422 |
| PF | fourSC | theta | 6 | activitySuggestionInteractLast7Days | 0.024006 | 0.071312 |
| PF | fourSC | theta | 7 | activeDaysLast7Days | 0.107989 | 0.052841 |
| PF | fourSC | theta | 8 | isWeekend | -0.046799 | 0.146605 |
| PF | fourSC | theta | 9 | decisionTimeSlot | -0.102159 | 0.154286 |
| PF | fourSC | theta | 10 | perceivedUtilityLastWeek | 0.033568 | 0.013093 |
| PF | fourSC | theta | 11 | caeAverageLastWeek | -0.053376 | 0.108784 |
| PF | fourSC | theta | 12 | Ah | -0.069907 | 0.055967 |
| PF | fourSC | theta | 13 | Ah*yesterdayStepCount | 0.005826 | 0.020915 |
| PF | fourSC | theta | 14 | Ah*prior2HourStepCount | 0.017884 | 0.021673 |
| PF | fourSC | theta | 15 | Ah*activitySuggestionsSentLast7Days | 0.023945 | 0.024920 |
| PF | fourSC | theta | 16 | Ah*activitySuggestionInteractLast7Days | 0.048477 | 0.099938 |
| PF | fourSC | theta | 17 | Ah*perceivedUtilityLastWeek | 0.020124 | 0.015127 |
| PF | fourSC | theta | 18 | Ah*caeAverageLastWeek | 0.030309 | 0.087403 |
| PF | antic | theta | 0 | intercept | 0.472889 | 0.017052 |
| PF | antic | theta | 1 | anticipated_affect_yesterday | 0.157048 | 0.004984 |
| PF | antic | theta | 2 | active_status_fraction_7days | 0.064866 | 0.007922 |
| PF | antic | theta | 3 | is_weekend | -0.018996 | 0.003880 |
| PF | antic | theta | 4 | perceived_utility_lastweek | -0.003941 | 0.006974 |
| PF | antic | theta | 5 | CAE_avg_lastweek | 0.252866 | 0.014589 |
| PF | antic | theta | 6 | recent_burden | 0.016217 | 0.003776 |
| PF | antic | theta | 7 | A0_morning | -0.007192 | 0.002691 |
| PF | antic | theta | 8 | A1_afternoon | 0.037005 | 0.002095 |
| PF | antic | theta | 9 | A0_morning_by_perceived_utility_lastweek | 0.006378 | 0.002695 |
| PF | antic | theta | 10 | A1_afternoon_by_perceived_utility_lastweek | -0.012111 | 0.001510 |
| PF | antic | theta | 11 | A0_morning_by_CAE_avg_lastweek | -0.004741 | 0.003950 |
| PF | antic | theta | 12 | A1_afternoon_by_CAE_avg_lastweek | -0.023529 | 0.002813 |
| PF | antic | theta | 13 | A0_morning_by_recent_burden | -0.011432 | 0.002055 |
| PF | antic | theta | 14 | A1_afternoon_by_recent_burden | 0.002631 | 0.002979 |
| PF | CAE | theta | 0 | intercept | -0.085373 | 0.100058 |
| PF | CAE | theta | 1 | CAE_avg_lastweek | 0.920348 | 0.059327 |
| PF | CAE | theta | 2 | fourSC_ewma | -0.007744 | 0.012901 |
| PF | CAE | theta | 3 | anticipated_affect_ewma | 0.145659 | 0.024469 |
| PF | CAE_short | theta | 0 | intercept | 0.003419 | 0.079414 |
| PF | CAE_short | theta | 1 | caeAverage | 0.967518 | 0.056877 |

## RL Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| RL | reward_shaping | eta | 0 | intercept | -0.026115 | 0.024801 |
| RL | reward_shaping | eta | 1 | weekday_vs_weekend | -0.004353 | 0.000689 |
| RL | reward_shaping | eta | 2 | slot_pm | -0.013058 | 0.006200 |
| RL | reward_shaping | eta | 3 | E_w | 0.041334 | 0.003747 |
| RL | reward_shaping | eta | 4 | b_hat | 0.058843 | 0.064524 |
| RL | reward_shaping | eta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | reward_shaping | eta | 6 | M_Y_anticipated_affect_ewma | 0.076077 | 0.128897 |
| RL | reward_shaping | eta | 7 | M_Y_fourSC_ewma | 0.026490 | 0.052640 |
| RL | reward_shaping | eta | 8 | M_E_pageview_ewma | 0.019050 | 0.014724 |
| RL | reward_shaping | eta | 9 | M_E_fitbit_wear_ewma | -0.001531 | 0.037097 |
| RL | reward_shaping | eta | 10 | M_E_survey_complete_ewma | 0.051034 | 0.068661 |
| RL | reward_shaping | eta | 11 | yesterday_step_count | -0.007175 | 0.060909 |
| RL | reward_shaping | eta | 12 | prior2hour_step_count | -0.039751 | 0.054568 |
| RL | reward_shaping | eta | 13 | active_status_fraction_7days | 0.031466 | 0.122514 |
| RL | reward_shaping | eta | 14 | recent_burden | -0.003091 | 0.007887 |
| RL | reward_shaping | eta | 15 | walk_interaction_7d | 0.021717 | 0.102127 |
| RL | redistribution_stage1_AA | eta | 0 | redistribution_stage1_AA_coef_0 | 0.268200 | 0.009403 |
| RL | redistribution_stage1_AA | eta | 1 | redistribution_stage1_AA_coef_1 | -0.001292 | 0.000555 |
| RL | redistribution_stage1_AA | eta | 2 | redistribution_stage1_AA_coef_2 | 0.104069 | 0.005149 |
| RL | redistribution_stage1_AA | eta | 3 | redistribution_stage1_AA_coef_3 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_AA | eta | 4 | redistribution_stage1_AA_coef_4 | 0.136994 | 0.002309 |
| RL | redistribution_stage1_AA | eta | 5 | redistribution_stage1_AA_coef_5 | -0.008871 | 0.002378 |
| RL | redistribution_stage1_AA | eta | 6 | redistribution_stage1_AA_coef_6 | 0.011655 | 0.000487 |
| RL | redistribution_stage1_AA | eta | 7 | redistribution_stage1_AA_coef_7 | -0.037348 | 0.001479 |
| RL | redistribution_stage1_AA | eta | 8 | redistribution_stage1_AA_coef_8 | -0.018813 | 0.001195 |
| RL | redistribution_stage1_AA | eta | 9 | redistribution_stage1_AA_coef_9 | 0.005858 | 0.000176 |
| RL | redistribution_stage1_AA | eta | 10 | redistribution_stage1_AA_coef_10 | 0.011101 | 0.000311 |
| RL | redistribution_stage1_AA | eta | 11 | redistribution_stage1_AA_coef_11 | 0.021025 | 0.003165 |
| RL | redistribution_stage1_AA | eta | 12 | redistribution_stage1_AA_coef_12 | 0.001725 | 9.356100e-05 |
| RL | redistribution_stage1_AA | eta | 13 | redistribution_stage1_AA_coef_13 | 0.002782 | 0.002238 |
| RL | redistribution_stage1_AA | eta | 14 | redistribution_stage1_AA_coef_14 | 0.008590 | 0.001229 |
| RL | redistribution_stage1_AA | eta | 15 | redistribution_stage1_AA_coef_15 | 0.000260 | 0.000363 |
| RL | redistribution_stage1_AA | eta | 16 | redistribution_stage1_AA_coef_16 | -0.009949 | 0.002101 |
| RL | redistribution_stage1_AA | eta | 17 | redistribution_stage1_AA_coef_17 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_AA | eta | 18 | redistribution_stage1_AA_coef_18 | 0.001326 | 0.000489 |
| RL | redistribution_stage1_AA | eta | 19 | redistribution_stage1_AA_coef_19 | 0.004818 | 0.000436 |
| RL | redistribution_stage1_AA | eta | 20 | redistribution_stage1_AA_coef_20 | -0.009465 | 0.001773 |
| RL | redistribution_stage1_AA | eta | 21 | redistribution_stage1_AA_coef_21 | 0.002358 | 0.000253 |
| RL | redistribution_stage1_AA | eta | 22 | redistribution_stage1_AA_coef_22 | 0.002876 | 0.000654 |
| RL | redistribution_stage1_FW | eta | 0 | redistribution_stage1_FW_coef_0 | 0.390033 | 0.020615 |
| RL | redistribution_stage1_FW | eta | 1 | redistribution_stage1_FW_coef_1 | 0.010806 | 0.004899 |
| RL | redistribution_stage1_FW | eta | 2 | redistribution_stage1_FW_coef_2 | 0.026303 | 0.011340 |
| RL | redistribution_stage1_FW | eta | 3 | redistribution_stage1_FW_coef_3 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_FW | eta | 4 | redistribution_stage1_FW_coef_4 | -0.123155 | 0.021559 |
| RL | redistribution_stage1_FW | eta | 5 | redistribution_stage1_FW_coef_5 | -0.021697 | 0.009235 |
| RL | redistribution_stage1_FW | eta | 6 | redistribution_stage1_FW_coef_6 | 0.015243 | 0.001257 |
| RL | redistribution_stage1_FW | eta | 7 | redistribution_stage1_FW_coef_7 | 0.162895 | 0.013326 |
| RL | redistribution_stage1_FW | eta | 8 | redistribution_stage1_FW_coef_8 | -0.032741 | 0.010593 |
| RL | redistribution_stage1_FW | eta | 9 | redistribution_stage1_FW_coef_9 | 0.009738 | 0.003292 |
| RL | redistribution_stage1_FW | eta | 10 | redistribution_stage1_FW_coef_10 | 0.017381 | 0.002488 |
| RL | redistribution_stage1_FW | eta | 11 | redistribution_stage1_FW_coef_11 | -0.006548 | 0.013087 |
| RL | redistribution_stage1_FW | eta | 12 | redistribution_stage1_FW_coef_12 | -0.004288 | 0.003192 |
| RL | redistribution_stage1_FW | eta | 13 | redistribution_stage1_FW_coef_13 | 0.027892 | 0.015836 |
| RL | redistribution_stage1_FW | eta | 14 | redistribution_stage1_FW_coef_14 | 0.006189 | 0.008230 |
| RL | redistribution_stage1_FW | eta | 15 | redistribution_stage1_FW_coef_15 | -0.000713 | 0.004389 |
| RL | redistribution_stage1_FW | eta | 16 | redistribution_stage1_FW_coef_16 | -0.003605 | 0.012467 |
| RL | redistribution_stage1_FW | eta | 17 | redistribution_stage1_FW_coef_17 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_FW | eta | 18 | redistribution_stage1_FW_coef_18 | 0.002765 | 0.007283 |
| RL | redistribution_stage1_FW | eta | 19 | redistribution_stage1_FW_coef_19 | -0.009297 | 0.004826 |
| RL | redistribution_stage1_FW | eta | 20 | redistribution_stage1_FW_coef_20 | -0.010217 | 0.017768 |
| RL | redistribution_stage1_FW | eta | 21 | redistribution_stage1_FW_coef_21 | 0.002973 | 0.004886 |
| RL | redistribution_stage1_FW | eta | 22 | redistribution_stage1_FW_coef_22 | -0.000426 | 0.019425 |
| RL | redistribution_stage1_PJ | eta | 0 | redistribution_stage1_PJ_coef_0 | 0.110139 | 0.015907 |
| RL | redistribution_stage1_PJ | eta | 1 | redistribution_stage1_PJ_coef_1 | 0.023907 | 0.009408 |
| RL | redistribution_stage1_PJ | eta | 2 | redistribution_stage1_PJ_coef_2 | 0.008449 | 0.017223 |
| RL | redistribution_stage1_PJ | eta | 3 | redistribution_stage1_PJ_coef_3 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_PJ | eta | 4 | redistribution_stage1_PJ_coef_4 | -0.087098 | 0.028323 |
| RL | redistribution_stage1_PJ | eta | 5 | redistribution_stage1_PJ_coef_5 | -0.025189 | 0.020786 |
| RL | redistribution_stage1_PJ | eta | 6 | redistribution_stage1_PJ_coef_6 | 0.008908 | 0.001923 |
| RL | redistribution_stage1_PJ | eta | 7 | redistribution_stage1_PJ_coef_7 | -0.013648 | 0.008095 |
| RL | redistribution_stage1_PJ | eta | 8 | redistribution_stage1_PJ_coef_8 | 0.139413 | 0.010041 |
| RL | redistribution_stage1_PJ | eta | 9 | redistribution_stage1_PJ_coef_9 | 0.001363 | 0.007555 |
| RL | redistribution_stage1_PJ | eta | 10 | redistribution_stage1_PJ_coef_10 | 0.020392 | 0.003128 |
| RL | redistribution_stage1_PJ | eta | 11 | redistribution_stage1_PJ_coef_11 | 0.063867 | 0.039285 |
| RL | redistribution_stage1_PJ | eta | 12 | redistribution_stage1_PJ_coef_12 | 0.017175 | 0.004058 |
| RL | redistribution_stage1_PJ | eta | 13 | redistribution_stage1_PJ_coef_13 | 0.142792 | 0.023084 |
| RL | redistribution_stage1_PJ | eta | 14 | redistribution_stage1_PJ_coef_14 | 0.032203 | 0.011108 |
| RL | redistribution_stage1_PJ | eta | 15 | redistribution_stage1_PJ_coef_15 | 0.008905 | 0.007403 |
| RL | redistribution_stage1_PJ | eta | 16 | redistribution_stage1_PJ_coef_16 | 0.013836 | 0.021049 |
| RL | redistribution_stage1_PJ | eta | 17 | redistribution_stage1_PJ_coef_17 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_PJ | eta | 18 | redistribution_stage1_PJ_coef_18 | 0.017771 | 0.012007 |
| RL | redistribution_stage1_PJ | eta | 19 | redistribution_stage1_PJ_coef_19 | -0.000722 | 0.009457 |
| RL | redistribution_stage1_PJ | eta | 20 | redistribution_stage1_PJ_coef_20 | -0.036244 | 0.030774 |
| RL | redistribution_stage1_PJ | eta | 21 | redistribution_stage1_PJ_coef_21 | -0.003115 | 0.008305 |
| RL | redistribution_stage1_PJ | eta | 22 | redistribution_stage1_PJ_coef_22 | -0.033370 | 0.020580 |
| RL | redistribution_stage2_v2 | eta | 0 | redistribution_stage2_v2_coef_0 | -0.067161 | 0.001061 |
| RL | redistribution_stage2_v2 | eta | 1 | redistribution_stage2_v2_coef_1 | -0.011193 | 2.947162e-05 |
| RL | redistribution_stage2_v2 | eta | 2 | redistribution_stage2_v2_coef_2 | 0.038847 | 0.000932 |
| RL | redistribution_stage2_v2 | eta | 3 | redistribution_stage2_v2_coef_3 | 0.049806 | 0.000733 |
| RL | redistribution_stage2_v2 | eta | 4 | redistribution_stage2_v2_coef_4 | 0.000000 | 1.000000 |
| RL | redistribution_stage2_v2 | eta | 5 | redistribution_stage2_v2_coef_5 | 0.102411 | 0.002785 |
| RL | redistribution_stage2_v2 | eta | 6 | redistribution_stage2_v2_coef_6 | 0.073940 | 0.002059 |
| RL | redistribution_stage2_v2 | eta | 7 | redistribution_stage2_v2_coef_7 | 0.049514 | 0.001193 |
| RL | redistribution_stage2_v2 | eta | 8 | redistribution_stage2_v2_coef_8 | 0.046894 | 0.001838 |
| RL | redistribution_stage2_v2 | eta | 9 | redistribution_stage2_v2_coef_9 | 0.187112 | 0.001316 |
| RL | redistribution_stage2_v2 | eta | 10 | redistribution_stage2_v2_coef_10 | -0.093669 | 0.000924 |
| RL | redistribution_stage2_v2 | eta | 11 | redistribution_stage2_v2_coef_11 | 0.037199 | 0.001271 |
| RL | redistribution_stage2_v2 | eta | 12 | redistribution_stage2_v2_coef_12 | 0.040626 | 0.001899 |
| RL | redistribution_stage2_v2 | eta | 13 | redistribution_stage2_v2_coef_13 | 0.003574 | 0.000236 |
| RL | redistribution_stage2_v2 | eta | 14 | redistribution_stage2_v2_coef_14 | 0.154669 | 0.003834 |
| RL | redistribution_stage2_v2 | eta | 15 | redistribution_stage2_v2_coef_15 | -0.101302 | 0.000867 |
| RL | redistribution_stage2_v2 | eta | 16 | redistribution_stage2_v2_coef_16 | -0.040341 | 0.001274 |
| RL | redistribution_stage2_v2 | eta | 17 | redistribution_stage2_v2_coef_17 | 0.034410 | 0.000118 |
| RL | redistribution_stage2_v2 | eta | 18 | redistribution_stage2_v2_coef_18 | -0.058409 | 0.000445 |
| RL | redistribution_stage2_v2 | eta | 19 | redistribution_stage2_v2_coef_19 | -0.615820 | 0.000687 |
| RL | redistribution_stage2_v4 | eta | 0 | redistribution_stage2_v4_coef_0 | -0.040123 | 0.011060 |
| RL | redistribution_stage2_v4 | eta | 1 | redistribution_stage2_v4_coef_1 | -0.006687 | 0.000307 |
| RL | redistribution_stage2_v4 | eta | 2 | redistribution_stage2_v4_coef_2 | -0.044849 | 0.003542 |
| RL | redistribution_stage2_v4 | eta | 3 | redistribution_stage2_v4_coef_3 | 0.077735 | 0.032797 |
| RL | redistribution_stage2_v4 | eta | 4 | redistribution_stage2_v4_coef_4 | 0.000000 | 1.000000 |
| RL | redistribution_stage2_v4 | eta | 5 | redistribution_stage2_v4_coef_5 | 0.141998 | 0.125474 |
| RL | redistribution_stage2_v4 | eta | 6 | redistribution_stage2_v4_coef_6 | 0.034437 | 0.018973 |
| RL | redistribution_stage2_v4 | eta | 7 | redistribution_stage2_v4_coef_7 | 0.021222 | 0.029349 |
| RL | redistribution_stage2_v4 | eta | 8 | redistribution_stage2_v4_coef_8 | -0.037407 | 0.045186 |
| RL | redistribution_stage2_v4 | eta | 9 | redistribution_stage2_v4_coef_9 | 0.050338 | 0.038073 |
| RL | redistribution_stage2_v4 | eta | 10 | redistribution_stage2_v4_coef_10 | -0.008072 | 0.043806 |
| RL | redistribution_stage2_v4 | eta | 11 | redistribution_stage2_v4_coef_11 | -0.038993 | 0.056468 |
| RL | redistribution_stage2_v4 | eta | 12 | redistribution_stage2_v4_coef_12 | 0.035790 | 0.077619 |
| RL | redistribution_stage2_v4 | eta | 13 | redistribution_stage2_v4_coef_13 | -0.002289 | 0.005011 |
| RL | redistribution_stage2_v4 | eta | 14 | redistribution_stage2_v4_coef_14 | 0.013993 | 0.087470 |
| RL | redistribution_stage2_v4 | eta | 15 | redistribution_stage2_v4_coef_15 | -0.003310 | 0.044351 |
| RL | redistribution_stage2_v4 | eta | 16 | redistribution_stage2_v4_coef_16 | -0.001460 | 0.041998 |
| RL | redistribution_stage2_v4 | eta | 17 | redistribution_stage2_v4_coef_17 | -0.242055 | 0.001039 |
| RL | redistribution_stage2_v4 | eta | 18 | redistribution_stage2_v4_coef_18 | 0.167056 | 0.031186 |
| RL | redistribution_stage2_v4 | eta | 19 | redistribution_stage2_v4_coef_19 | 0.039680 | 0.026639 |
| RL | q_no_td_modify | beta | 0 | intercept | 0.145166 | 0.129849 |
| RL | q_no_td_modify | beta | 1 | weekday_vs_weekend | -0.058816 | 0.076789 |
| RL | q_no_td_modify | beta | 2 | slot_pm | -0.004116 | 0.002084 |
| RL | q_no_td_modify | beta | 3 | E_w | -0.007189 | 0.021551 |
| RL | q_no_td_modify | beta | 4 | b_hat | 1.224367 | 0.090535 |
| RL | q_no_td_modify | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify | beta | 6 | M_Y_anticipated_affect_ewma | 0.203032 | 0.005331 |
| RL | q_no_td_modify | beta | 7 | M_Y_fourSC_ewma | 0.004876 | 0.007628 |
| RL | q_no_td_modify | beta | 8 | M_E_pageview_ewma | 0.001726 | 0.000326 |
| RL | q_no_td_modify | beta | 9 | M_E_fitbit_wear_ewma | -0.106523 | 0.002634 |
| RL | q_no_td_modify | beta | 10 | M_E_survey_complete_ewma | -0.029988 | 0.004321 |
| RL | q_no_td_modify | beta | 11 | yesterday_step_count | 0.004249 | 0.000567 |
| RL | q_no_td_modify | beta | 12 | prior2hour_step_count | -0.008123 | 0.000378 |
| RL | q_no_td_modify | beta | 13 | active_status_fraction_7days | -0.002527 | 0.044678 |
| RL | q_no_td_modify | beta | 14 | recent_burden | 0.000882 | 0.003244 |
| RL | q_no_td_modify | beta | 15 | walk_interaction_7d | -0.048739 | 0.016356 |
| RL | q_no_td_modify | beta | 16 | A | -0.007060 | 0.001551 |
| RL | q_no_td_modify | beta | 17 | A*E_w | -0.002804 | 0.000697 |
| RL | q_no_td_modify | beta | 18 | A*b_hat | -0.007105 | 0.000857 |
| RL | q_no_td_modify | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify | beta | 20 | A*yesterday_step_count | -0.009345 | 0.002109 |
| RL | q_no_td_modify | beta | 21 | A*prior2hour_step_count | 0.005163 | 0.000657 |
| RL | q_no_td_modify | beta | 22 | A*active_status_fraction_7days | 0.011654 | 0.001812 |
| RL | q_no_td_modify | beta | 23 | A*recent_burden | -0.003269 | 0.001572 |
| RL | q_no_td_modify | beta | 24 | A*walk_interaction_7d | 0.007657 | 0.002096 |
| RL | q_no_td_modify_g09 | beta | 0 | intercept | 0.264758 | 0.224628 |
| RL | q_no_td_modify_g09 | beta | 1 | weekday_vs_weekend | -0.045214 | 0.243751 |
| RL | q_no_td_modify_g09 | beta | 2 | slot_pm | 0.000361 | 0.008310 |
| RL | q_no_td_modify_g09 | beta | 3 | E_w | -0.010200 | 0.042404 |
| RL | q_no_td_modify_g09 | beta | 4 | b_hat | 1.698549 | 0.160259 |
| RL | q_no_td_modify_g09 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g09 | beta | 6 | M_Y_anticipated_affect_ewma | 0.361022 | 0.014948 |
| RL | q_no_td_modify_g09 | beta | 7 | M_Y_fourSC_ewma | 0.008948 | 0.014967 |
| RL | q_no_td_modify_g09 | beta | 8 | M_E_pageview_ewma | 0.003749 | 0.000763 |
| RL | q_no_td_modify_g09 | beta | 9 | M_E_fitbit_wear_ewma | -0.175390 | 0.006399 |
| RL | q_no_td_modify_g09 | beta | 10 | M_E_survey_complete_ewma | -0.050173 | 0.008469 |
| RL | q_no_td_modify_g09 | beta | 11 | yesterday_step_count | 0.014264 | 0.001147 |
| RL | q_no_td_modify_g09 | beta | 12 | prior2hour_step_count | -0.013966 | 0.000934 |
| RL | q_no_td_modify_g09 | beta | 13 | active_status_fraction_7days | -0.003287 | 0.083824 |
| RL | q_no_td_modify_g09 | beta | 14 | recent_burden | 0.004894 | 0.007740 |
| RL | q_no_td_modify_g09 | beta | 15 | walk_interaction_7d | -0.087126 | 0.030462 |
| RL | q_no_td_modify_g09 | beta | 16 | A | -0.020342 | 0.003671 |
| RL | q_no_td_modify_g09 | beta | 17 | A*E_w | -0.005928 | 0.001533 |
| RL | q_no_td_modify_g09 | beta | 18 | A*b_hat | -0.014719 | 0.001989 |
| RL | q_no_td_modify_g09 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g09 | beta | 20 | A*yesterday_step_count | -0.018731 | 0.004003 |
| RL | q_no_td_modify_g09 | beta | 21 | A*prior2hour_step_count | 0.008354 | 0.001503 |
| RL | q_no_td_modify_g09 | beta | 22 | A*active_status_fraction_7days | 0.026598 | 0.004763 |
| RL | q_no_td_modify_g09 | beta | 23 | A*recent_burden | -0.005133 | 0.003467 |
| RL | q_no_td_modify_g09 | beta | 24 | A*walk_interaction_7d | 0.020563 | 0.004558 |
| RL | q_no_td_modify_g099 | beta | 0 | intercept | 0.301514 | 0.254564 |
| RL | q_no_td_modify_g099 | beta | 1 | weekday_vs_weekend | -0.035894 | 0.316545 |
| RL | q_no_td_modify_g099 | beta | 2 | slot_pm | 0.002486 | 0.011304 |
| RL | q_no_td_modify_g099 | beta | 3 | E_w | -0.010932 | 0.049912 |
| RL | q_no_td_modify_g099 | beta | 4 | b_hat | 1.835960 | 0.183539 |
| RL | q_no_td_modify_g099 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g099 | beta | 6 | M_Y_anticipated_affect_ewma | 0.411828 | 0.018994 |
| RL | q_no_td_modify_g099 | beta | 7 | M_Y_fourSC_ewma | 0.010217 | 0.017309 |
| RL | q_no_td_modify_g099 | beta | 8 | M_E_pageview_ewma | 0.004512 | 0.000932 |
| RL | q_no_td_modify_g099 | beta | 9 | M_E_fitbit_wear_ewma | -0.196498 | 0.008259 |
| RL | q_no_td_modify_g099 | beta | 10 | M_E_survey_complete_ewma | -0.056554 | 0.010391 |
| RL | q_no_td_modify_g099 | beta | 11 | yesterday_step_count | 0.017732 | 0.001457 |
| RL | q_no_td_modify_g099 | beta | 12 | prior2hour_step_count | -0.015887 | 0.001149 |
| RL | q_no_td_modify_g099 | beta | 13 | active_status_fraction_7days | -0.004356 | 0.097314 |
| RL | q_no_td_modify_g099 | beta | 14 | recent_burden | 0.006237 | 0.009514 |
| RL | q_no_td_modify_g099 | beta | 15 | walk_interaction_7d | -0.100028 | 0.035728 |
| RL | q_no_td_modify_g099 | beta | 16 | A | -0.024840 | 0.004631 |
| RL | q_no_td_modify_g099 | beta | 17 | A*E_w | -0.006965 | 0.001947 |
| RL | q_no_td_modify_g099 | beta | 18 | A*b_hat | -0.017313 | 0.002474 |
| RL | q_no_td_modify_g099 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g099 | beta | 20 | A*yesterday_step_count | -0.021840 | 0.004658 |
| RL | q_no_td_modify_g099 | beta | 21 | A*prior2hour_step_count | 0.009378 | 0.001865 |
| RL | q_no_td_modify_g099 | beta | 22 | A*active_status_fraction_7days | 0.031533 | 0.006047 |
| RL | q_no_td_modify_g099 | beta | 23 | A*recent_burden | -0.005717 | 0.004100 |
| RL | q_no_td_modify_g099 | beta | 24 | A*walk_interaction_7d | 0.024903 | 0.005559 |
| RL | q_redistribution_v2 | beta | 0 | intercept | 0.336495 | 0.320105 |
| RL | q_redistribution_v2 | beta | 1 | weekday_vs_weekend | -0.179770 | 0.091902 |
| RL | q_redistribution_v2 | beta | 2 | slot_pm | -0.011468 | 0.001344 |
| RL | q_redistribution_v2 | beta | 3 | E_w | 0.526385 | 0.824087 |
| RL | q_redistribution_v2 | beta | 4 | b_hat | 1.722617 | 0.192837 |
| RL | q_redistribution_v2 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v2 | beta | 6 | M_Y_anticipated_affect_ewma | 0.176953 | 0.008917 |
| RL | q_redistribution_v2 | beta | 7 | M_Y_fourSC_ewma | -0.155900 | 0.016707 |
| RL | q_redistribution_v2 | beta | 8 | M_E_pageview_ewma | 0.030626 | 0.003566 |
| RL | q_redistribution_v2 | beta | 9 | M_E_fitbit_wear_ewma | -0.165542 | 0.014495 |
| RL | q_redistribution_v2 | beta | 10 | M_E_survey_complete_ewma | 0.222146 | 0.016350 |
| RL | q_redistribution_v2 | beta | 11 | yesterday_step_count | -0.146198 | 0.005754 |
| RL | q_redistribution_v2 | beta | 12 | prior2hour_step_count | -0.035641 | 0.002931 |
| RL | q_redistribution_v2 | beta | 13 | active_status_fraction_7days | -0.240384 | 0.126120 |
| RL | q_redistribution_v2 | beta | 14 | recent_burden | -0.045952 | 0.008459 |
| RL | q_redistribution_v2 | beta | 15 | walk_interaction_7d | 2.065831 | 0.091931 |
| RL | q_redistribution_v2 | beta | 16 | A | 0.059114 | 0.004117 |
| RL | q_redistribution_v2 | beta | 17 | A*E_w | 0.013703 | 0.000573 |
| RL | q_redistribution_v2 | beta | 18 | A*b_hat | 0.001595 | 0.002764 |
| RL | q_redistribution_v2 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v2 | beta | 20 | A*yesterday_step_count | -0.001952 | 0.001164 |
| RL | q_redistribution_v2 | beta | 21 | A*prior2hour_step_count | -0.021507 | 0.002287 |
| RL | q_redistribution_v2 | beta | 22 | A*active_status_fraction_7days | -0.066950 | 0.006365 |
| RL | q_redistribution_v2 | beta | 23 | A*recent_burden | -0.000931 | 0.000858 |
| RL | q_redistribution_v2 | beta | 24 | A*walk_interaction_7d | -0.064862 | 0.003929 |
| RL | q_redistribution_v4 | beta | 0 | intercept | 0.069163 | 0.673950 |
| RL | q_redistribution_v4 | beta | 1 | weekday_vs_weekend | 0.028331 | 0.266900 |
| RL | q_redistribution_v4 | beta | 2 | slot_pm | 0.068571 | 0.015861 |
| RL | q_redistribution_v4 | beta | 3 | E_w | -0.653036 | 0.156835 |
| RL | q_redistribution_v4 | beta | 4 | b_hat | 1.533758 | 0.399626 |
| RL | q_redistribution_v4 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v4 | beta | 6 | M_Y_anticipated_affect_ewma | 0.335811 | 0.098390 |
| RL | q_redistribution_v4 | beta | 7 | M_Y_fourSC_ewma | 0.070562 | 0.168385 |
| RL | q_redistribution_v4 | beta | 8 | M_E_pageview_ewma | 0.084969 | 0.052765 |
| RL | q_redistribution_v4 | beta | 9 | M_E_fitbit_wear_ewma | -0.097717 | 0.275442 |
| RL | q_redistribution_v4 | beta | 10 | M_E_survey_complete_ewma | -0.019333 | 0.248793 |
| RL | q_redistribution_v4 | beta | 11 | yesterday_step_count | 0.035815 | 0.087558 |
| RL | q_redistribution_v4 | beta | 12 | prior2hour_step_count | -0.001364 | 0.051177 |
| RL | q_redistribution_v4 | beta | 13 | active_status_fraction_7days | 0.632705 | 0.614197 |
| RL | q_redistribution_v4 | beta | 14 | recent_burden | -0.054144 | 0.194505 |
| RL | q_redistribution_v4 | beta | 15 | walk_interaction_7d | 0.728390 | 0.444393 |
| RL | q_redistribution_v4 | beta | 16 | A | 0.117164 | 0.058357 |
| RL | q_redistribution_v4 | beta | 17 | A*E_w | 0.021956 | 0.005968 |
| RL | q_redistribution_v4 | beta | 18 | A*b_hat | 0.039043 | 0.045670 |
| RL | q_redistribution_v4 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v4 | beta | 20 | A*yesterday_step_count | 0.012315 | 0.021514 |
| RL | q_redistribution_v4 | beta | 21 | A*prior2hour_step_count | 0.003379 | 0.018657 |
| RL | q_redistribution_v4 | beta | 22 | A*active_status_fraction_7days | -0.152215 | 0.068106 |
| RL | q_redistribution_v4 | beta | 23 | A*recent_burden | 0.043593 | 0.011872 |
| RL | q_redistribution_v4 | beta | 24 | A*walk_interaction_7d | -0.085365 | 0.096313 |

## Joint Modified-TD Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| Joint Modified TD | q_td_modify_joint | eta | 0 | eta_intercept | 0.615555 | 0.229765 |
| Joint Modified TD | q_td_modify_joint | eta | 1 | eta_E_w | -0.021220 | 0.032799 |
| Joint Modified TD | q_td_modify_joint | eta | 2 | eta_b_hat | -0.332832 | 0.173151 |
| Joint Modified TD | q_td_modify_joint | eta | 3 | eta_b_tilde | 0.000000 | 1.000000 |
| Joint Modified TD | q_td_modify_joint | beta | 4 | beta_intercept | 0.606379 | 0.166346 |
| Joint Modified TD | q_td_modify_joint | beta | 5 | beta_weekday_vs_weekend | -0.092199 | 0.202916 |
| Joint Modified TD | q_td_modify_joint | beta | 6 | beta_slot_pm | -0.001367 | 0.004315 |
| Joint Modified TD | q_td_modify_joint | beta | 7 | beta_E_w | -0.023113 | 0.026776 |
| Joint Modified TD | q_td_modify_joint | beta | 8 | beta_b_hat | 0.120138 | 0.123367 |
| Joint Modified TD | q_td_modify_joint | beta | 9 | beta_b_tilde | 0.000000 | 1.000000 |
| Joint Modified TD | q_td_modify_joint | beta | 10 | beta_M_Y_anticipated_affect_ewma | 0.697418 | 0.013933 |
| Joint Modified TD | q_td_modify_joint | beta | 11 | beta_M_Y_fourSC_ewma | 0.059855 | 0.011470 |
| Joint Modified TD | q_td_modify_joint | beta | 12 | beta_M_E_pageview_ewma | 0.005036 | 0.001467 |
| Joint Modified TD | q_td_modify_joint | beta | 13 | beta_M_E_fitbit_wear_ewma | -0.342864 | 0.007015 |
| Joint Modified TD | q_td_modify_joint | beta | 14 | beta_M_E_survey_complete_ewma | -0.082834 | 0.007048 |
| Joint Modified TD | q_td_modify_joint | beta | 15 | beta_yesterday_step_count | 0.009212 | 0.000872 |
| Joint Modified TD | q_td_modify_joint | beta | 16 | beta_prior2hour_step_count | -0.005132 | 0.000620 |
| Joint Modified TD | q_td_modify_joint | beta | 17 | beta_active_status_fraction_7days | -0.047995 | 0.053587 |
| Joint Modified TD | q_td_modify_joint | beta | 18 | beta_recent_burden | 0.002386 | 0.001423 |
| Joint Modified TD | q_td_modify_joint | beta | 19 | beta_walk_interaction_7d | 0.052526 | 0.013324 |
| Joint Modified TD | q_td_modify_joint | beta | 20 | beta_A | -0.016956 | 0.002627 |
| Joint Modified TD | q_td_modify_joint | beta | 21 | beta_A*E_w | -0.002318 | 0.000793 |
| Joint Modified TD | q_td_modify_joint | beta | 22 | beta_A*b_hat | -0.068788 | 0.001700 |
| Joint Modified TD | q_td_modify_joint | beta | 23 | beta_A*b_tilde | 0.000000 | 1.000000 |
| Joint Modified TD | q_td_modify_joint | beta | 24 | beta_A*yesterday_step_count | -0.000831 | 0.003891 |
| Joint Modified TD | q_td_modify_joint | beta | 25 | beta_A*prior2hour_step_count | -0.002935 | 0.001134 |
| Joint Modified TD | q_td_modify_joint | beta | 26 | beta_A*active_status_fraction_7days | 0.025932 | 0.002261 |
| Joint Modified TD | q_td_modify_joint | beta | 27 | beta_A*recent_burden | -0.002613 | 0.002672 |
| Joint Modified TD | q_td_modify_joint | beta | 28 | beta_A*walk_interaction_7d | 0.009649 | 0.002892 |

## Pooled PF Regression Coefficients

Coefficients are from the all-user stacked ridge fits used as PF prior means. `p_value` uses ridge sandwich standard errors and is approximate because ridge shrinks coefficients. `identified=False` marks zero-variance design columns; SE / p-values are omitted for those rows.

| model | outcome | index | feature | coefficient | identified | feature_std | std_error | t_stat | p_value | significant_0.05 | significant_0.01 | n_obs | n_features | ridge_alpha | residual_sigma2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fourSC | 4hour_step_norm | 0 | intercept | -0.032248 | False | 0.000000 |  |  |  | False | False | 3735 | 19 | 1.000000 | 0.521280 |
| fourSC | 4hour_step_norm | 1 | stepCountNext4HourLag1 | 0.463390 | True | 0.924861 | 0.015212 | 30.461482 | 3.963449e-182 | True | True | 3735 | 19 | 1.000000 | 0.521280 |
| fourSC | 4hour_step_norm | 2 | yesterdayStepCount | -0.127085 | True | 0.948113 | 0.019296 | -6.585936 | 5.154724e-11 | True | True | 3735 | 19 | 1.000000 | 0.521280 |
| fourSC | 4hour_step_norm | 3 | stepCountLast7DaysEma | 0.369354 | True | 1.005044 | 0.013815 | 26.735607 | 3.552898e-144 | True | True | 3735 | 19 | 1.000000 | 0.521280 |
| fourSC | 4hour_step_norm | 4 | prior2HourStepCount | 0.131687 | True | 0.941170 | 0.018983 | 6.937172 | 4.698361e-12 | True | True | 3735 | 19 | 1.000000 | 0.521280 |
| fourSC | 4hour_step_norm | 5 | activitySuggestionsSentLast7Days | -0.025906 | True | 0.967404 | 0.017651 | -1.467691 | 0.142273 | False | False | 3735 | 19 | 1.000000 | 0.521280 |
| fourSC | 4hour_step_norm | 6 | activitySuggestionInteractLast7Days | 0.024006 | True | 0.347380 | 0.054324 | 0.441904 | 0.658585 | False | False | 3735 | 19 | 1.000000 | 0.521280 |
| fourSC | 4hour_step_norm | 7 | activeDaysLast7Days | 0.107989 | True | 0.277433 | 0.044778 | 2.411638 | 0.015929 | True | False | 3735 | 19 | 1.000000 | 0.521280 |
| fourSC | 4hour_step_norm | 8 | isWeekend | -0.046799 | True | 0.364457 | 0.032453 | -1.442046 | 0.149374 | False | False | 3735 | 19 | 1.000000 | 0.521280 |
| fourSC | 4hour_step_norm | 9 | decisionTimeSlot | -0.102159 | True | 0.498173 | 0.025905 | -3.943639 | 8.173423e-05 | True | True | 3735 | 19 | 1.000000 | 0.521280 |
| fourSC | 4hour_step_norm | 10 | perceivedUtilityLastWeek | 0.033568 | True | 1.967084 | 0.009690 | 3.464293 | 0.000538 | True | True | 3735 | 19 | 1.000000 | 0.521280 |
| fourSC | 4hour_step_norm | 11 | caeAverageLastWeek | -0.053376 | True | 0.992053 | 0.017183 | -3.106393 | 0.001908 | True | True | 3735 | 19 | 1.000000 | 0.521280 |
| fourSC | 4hour_step_norm | 12 | Ah | -0.069907 | True | 0.499771 | 0.040964 | -1.706531 | 0.087993 | False | False | 3735 | 19 | 1.000000 | 0.521280 |
| fourSC | 4hour_step_norm | 13 | Ah*yesterdayStepCount | 0.005826 | True | 0.672639 | 0.025912 | 0.224830 | 0.822124 | False | False | 3735 | 19 | 1.000000 | 0.521280 |
| fourSC | 4hour_step_norm | 14 | Ah*prior2HourStepCount | 0.017884 | True | 0.666699 | 0.026222 | 0.682024 | 0.495266 | False | False | 3735 | 19 | 1.000000 | 0.521280 |
| fourSC | 4hour_step_norm | 15 | Ah*activitySuggestionsSentLast7Days | 0.023945 | True | 0.688627 | 0.024697 | 0.969563 | 0.332328 | False | False | 3735 | 19 | 1.000000 | 0.521280 |
| fourSC | 4hour_step_norm | 16 | Ah*activitySuggestionInteractLast7Days | 0.048477 | True | 0.349684 | 0.075931 | 0.638437 | 0.523229 | False | False | 3735 | 19 | 1.000000 | 0.521280 |
| fourSC | 4hour_step_norm | 17 | Ah*perceivedUtilityLastWeek | 0.020124 | True | 1.470923 | 0.013703 | 1.468618 | 0.142021 | False | False | 3735 | 19 | 1.000000 | 0.521280 |
| fourSC | 4hour_step_norm | 18 | Ah*caeAverageLastWeek | 0.030309 | True | 0.692755 | 0.024235 | 1.250616 | 0.211153 | False | False | 3735 | 19 | 1.000000 | 0.521280 |
| antic | anticipated_affect_norm | 0 | intercept | 0.472889 | False | 0.000000 |  |  |  | False | False | 1019 | 15 | 1.000000 | 0.036553 |
| antic | anticipated_affect_norm | 1 | anticipated_affect_yesterday | 0.157048 | True | 0.366210 | 0.019786 | 7.937396 | 5.496247e-15 | True | True | 1019 | 15 | 1.000000 | 0.036553 |
| antic | anticipated_affect_norm | 2 | active_status_fraction_7days | 0.064866 | True | 0.229784 | 0.028027 | 2.314354 | 0.020849 | True | False | 1019 | 15 | 1.000000 | 0.036553 |
| antic | anticipated_affect_norm | 3 | is_weekend | -0.018996 | True | 0.366570 | 0.016449 | -1.154903 | 0.248405 | False | False | 1019 | 15 | 1.000000 | 0.036553 |
| antic | anticipated_affect_norm | 4 | perceived_utility_lastweek | -0.003941 | True | 1.621897 | 0.006164 | -0.639382 | 0.522720 | False | False | 1019 | 15 | 1.000000 | 0.036553 |
| antic | anticipated_affect_norm | 5 | CAE_avg_lastweek | 0.252866 | True | 0.928551 | 0.012140 | 20.828757 | 2.292026e-80 | True | True | 1019 | 15 | 1.000000 | 0.036553 |
| antic | anticipated_affect_norm | 6 | recent_burden | 0.016217 | True | 0.974256 | 0.009569 | 1.694727 | 0.090437 | False | False | 1019 | 15 | 1.000000 | 0.036553 |
| antic | anticipated_affect_norm | 7 | A0_morning | -0.007192 | True | 0.499217 | 0.017910 | -0.401583 | 0.688077 | False | False | 1019 | 15 | 1.000000 | 0.036553 |
| antic | anticipated_affect_norm | 8 | A1_afternoon | 0.037005 | True | 0.499873 | 0.017875 | 2.070221 | 0.038687 | True | False | 1019 | 15 | 1.000000 | 0.036553 |
| antic | anticipated_affect_norm | 9 | A0_morning_by_perceived_utility_lastweek | 0.006378 | True | 1.408726 | 0.007626 | 0.836312 | 0.403178 | False | False | 1019 | 15 | 1.000000 | 0.036553 |
| antic | anticipated_affect_norm | 10 | A1_afternoon_by_perceived_utility_lastweek | -0.012111 | True | 1.410691 | 0.007622 | -1.589016 | 0.112372 | False | False | 1019 | 15 | 1.000000 | 0.036553 |
| antic | anticipated_affect_norm | 11 | A0_morning_by_CAE_avg_lastweek | -0.004741 | True | 0.686065 | 0.013164 | -0.360131 | 0.718825 | False | False | 1019 | 15 | 1.000000 | 0.036553 |
| antic | anticipated_affect_norm | 12 | A1_afternoon_by_CAE_avg_lastweek | -0.023529 | True | 0.657745 | 0.013145 | -1.789937 | 0.073765 | False | False | 1019 | 15 | 1.000000 | 0.036553 |
| antic | anticipated_affect_norm | 13 | A0_morning_by_recent_burden | -0.011432 | True | 0.682435 | 0.013847 | -0.825605 | 0.409225 | False | False | 1019 | 15 | 1.000000 | 0.036553 |
| antic | anticipated_affect_norm | 14 | A1_afternoon_by_recent_burden | 0.002631 | True | 0.665827 | 0.013781 | 0.190880 | 0.848658 | False | False | 1019 | 15 | 1.000000 | 0.036553 |
| CAE | CAE_avg_norm | 0 | intercept | -0.085373 | False | 0.000000 |  |  |  | False | False | 262 | 4 | 1.000000 | 0.115176 |
| CAE | CAE_avg_norm | 1 | CAE_avg_lastweek | 0.920348 | True | 0.981409 | 0.029292 | 31.420050 | 3.613334e-90 | True | True | 262 | 4 | 1.000000 | 0.115176 |
| CAE | CAE_avg_norm | 2 | fourSC_ewma | -0.007744 | True | 0.527652 | 0.039465 | -0.196220 | 0.844593 | False | False | 262 | 4 | 1.000000 | 0.115176 |
| CAE | CAE_avg_norm | 3 | anticipated_affect_ewma | 0.145659 | True | 0.356150 | 0.078324 | 1.859703 | 0.064066 | False | False | 262 | 4 | 1.000000 | 0.115176 |

## Pooled PF GEE Coefficients

Population-averaged GEE fits on the same stacked PF designs, clustered by `ParticipantIdentifier`. `p_value` uses GEE sandwich standard errors with exchangeable working correlation. `identified=False` marks zero-variance design columns; SE / p-values are omitted for those rows. These are for inference/audit only; PF priors still use ridge.

| coefficient | robust_se | z_stat | p_value | significant_0.05 | significant_0.01 | model | outcome | index | feature | identified | feature_std | n_obs | n_clusters | n_features | working_correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -0.081452 |  |  |  | False | False | fourSC | 4hour_step_norm | 0 | intercept | False | 0.000000 | 3735 | 35 | 19 | exchangeable |
| -0.059381 | 0.038792 | -1.530734 | 0.125835 | False | False | fourSC | 4hour_step_norm | 1 | stepCountNext4HourLag1 | True | 0.924861 | 3735 | 35 | 19 | exchangeable |
| -0.012206 | 0.022301 | -0.547335 | 0.584149 | False | False | fourSC | 4hour_step_norm | 2 | yesterdayStepCount | True | 0.948113 | 3735 | 35 | 19 | exchangeable |
| 0.196902 | 0.042341 | 4.650337 | 3.313927e-06 | True | True | fourSC | 4hour_step_norm | 3 | stepCountLast7DaysEma | True | 1.005044 | 3735 | 35 | 19 | exchangeable |
| 0.154926 | 0.030468 | 5.084821 | 3.679721e-07 | True | True | fourSC | 4hour_step_norm | 4 | prior2HourStepCount | True | 0.941170 | 3735 | 35 | 19 | exchangeable |
| 0.002565 | 0.014515 | 0.176713 | 0.859734 | False | False | fourSC | 4hour_step_norm | 5 | activitySuggestionsSentLast7Days | True | 0.967404 | 3735 | 35 | 19 | exchangeable |
| 0.101741 | 0.063269 | 1.608054 | 0.107823 | False | False | fourSC | 4hour_step_norm | 6 | activitySuggestionInteractLast7Days | True | 0.347380 | 3735 | 35 | 19 | exchangeable |
| 0.017911 | 0.061405 | 0.291691 | 0.770523 | False | False | fourSC | 4hour_step_norm | 7 | activeDaysLast7Days | True | 0.277433 | 3735 | 35 | 19 | exchangeable |
| -0.030262 | 0.039868 | -0.759047 | 0.447824 | False | False | fourSC | 4hour_step_norm | 8 | isWeekend | True | 0.364457 | 3735 | 35 | 19 | exchangeable |
| -0.121616 | 0.061468 | -1.978536 | 0.047868 | True | False | fourSC | 4hour_step_norm | 9 | decisionTimeSlot | True | 0.498173 | 3735 | 35 | 19 | exchangeable |
| 0.018391 | 0.009818 | 1.873154 | 0.061047 | False | False | fourSC | 4hour_step_norm | 10 | perceivedUtilityLastWeek | True | 1.967084 | 3735 | 35 | 19 | exchangeable |
| 0.015305 | 0.034331 | 0.445803 | 0.655739 | False | False | fourSC | 4hour_step_norm | 11 | caeAverageLastWeek | True | 0.992053 | 3735 | 35 | 19 | exchangeable |
| -0.019766 | 0.044656 | -0.442628 | 0.658035 | False | False | fourSC | 4hour_step_norm | 12 | Ah | True | 0.499771 | 3735 | 35 | 19 | exchangeable |
| 0.018629 | 0.028795 | 0.646949 | 0.517665 | False | False | fourSC | 4hour_step_norm | 13 | Ah*yesterdayStepCount | True | 0.672639 | 3735 | 35 | 19 | exchangeable |
| -0.005428 | 0.030513 | -0.177882 | 0.858816 | False | False | fourSC | 4hour_step_norm | 14 | Ah*prior2HourStepCount | True | 0.666699 | 3735 | 35 | 19 | exchangeable |
| 0.011343 | 0.017484 | 0.648786 | 0.516476 | False | False | fourSC | 4hour_step_norm | 15 | Ah*activitySuggestionsSentLast7Days | True | 0.688627 | 3735 | 35 | 19 | exchangeable |
| 0.032624 | 0.076324 | 0.427437 | 0.669061 | False | False | fourSC | 4hour_step_norm | 16 | Ah*activitySuggestionInteractLast7Days | True | 0.349684 | 3735 | 35 | 19 | exchangeable |
| -0.001242 | 0.010319 | -0.120321 | 0.904229 | False | False | fourSC | 4hour_step_norm | 17 | Ah*perceivedUtilityLastWeek | True | 1.470923 | 3735 | 35 | 19 | exchangeable |
| 0.008142 | 0.021403 | 0.380389 | 0.703657 | False | False | fourSC | 4hour_step_norm | 18 | Ah*caeAverageLastWeek | True | 0.692755 | 3735 | 35 | 19 | exchangeable |
| 0.502819 |  |  |  | False | False | antic | anticipated_affect_norm | 0 | intercept | False | 0.000000 | 1019 | 35 | 15 | exchangeable |
| 0.044663 | 0.025022 | 1.784936 | 0.074272 | False | False | antic | anticipated_affect_norm | 1 | anticipated_affect_yesterday | True | 0.366210 | 1019 | 35 | 15 | exchangeable |
| 0.096338 | 0.033381 | 2.885985 | 0.003902 | True | True | antic | anticipated_affect_norm | 2 | active_status_fraction_7days | True | 0.229784 | 1019 | 35 | 15 | exchangeable |
| -6.678765e-05 | 0.018698 | -0.003572 | 0.997150 | False | False | antic | anticipated_affect_norm | 3 | is_weekend | True | 0.366570 | 1019 | 35 | 15 | exchangeable |
| 0.003923 | 0.007898 | 0.496746 | 0.619368 | False | False | antic | anticipated_affect_norm | 4 | perceived_utility_lastweek | True | 1.621897 | 1019 | 35 | 15 | exchangeable |
| 0.145400 | 0.036033 | 4.035175 | 5.456162e-05 | True | True | antic | anticipated_affect_norm | 5 | CAE_avg_lastweek | True | 0.928551 | 1019 | 35 | 15 | exchangeable |
| 0.012861 | 0.005777 | 2.226281 | 0.025995 | True | False | antic | anticipated_affect_norm | 6 | recent_burden | True | 0.974256 | 1019 | 35 | 15 | exchangeable |
| -0.011929 | 0.009899 | -1.205142 | 0.228148 | False | False | antic | anticipated_affect_norm | 7 | A0_morning | True | 0.499217 | 1019 | 35 | 15 | exchangeable |
| 0.023678 | 0.012561 | 1.885050 | 0.059423 | False | False | antic | anticipated_affect_norm | 8 | A1_afternoon | True | 0.499873 | 1019 | 35 | 15 | exchangeable |
| 0.008507 | 0.004217 | 2.017488 | 0.043645 | True | False | antic | anticipated_affect_norm | 9 | A0_morning_by_perceived_utility_lastweek | True | 1.408726 | 1019 | 35 | 15 | exchangeable |
| -0.006067 | 0.005751 | -1.054968 | 0.291440 | False | False | antic | anticipated_affect_norm | 10 | A1_afternoon_by_perceived_utility_lastweek | True | 1.410691 | 1019 | 35 | 15 | exchangeable |
| 0.008286 | 0.009125 | 0.908043 | 0.363855 | False | False | antic | anticipated_affect_norm | 11 | A0_morning_by_CAE_avg_lastweek | True | 0.686065 | 1019 | 35 | 15 | exchangeable |
| -0.019820 | 0.011635 | -1.703464 | 0.088481 | False | False | antic | anticipated_affect_norm | 12 | A1_afternoon_by_CAE_avg_lastweek | True | 0.657745 | 1019 | 35 | 15 | exchangeable |
| -0.010093 | 0.010121 | -0.997246 | 0.318645 | False | False | antic | anticipated_affect_norm | 13 | A0_morning_by_recent_burden | True | 0.682435 | 1019 | 35 | 15 | exchangeable |
| -0.011512 | 0.010137 | -1.135642 | 0.256107 | False | False | antic | anticipated_affect_norm | 14 | A1_afternoon_by_recent_burden | True | 0.665827 | 1019 | 35 | 15 | exchangeable |
| -0.060492 |  |  |  | False | False | CAE | CAE_avg_norm | 0 | intercept | False | 0.000000 | 262 | 35 | 4 | exchangeable |
| 0.970574 | 0.027811 | 34.898388 | 7.865086e-267 | True | True | CAE | CAE_avg_norm | 1 | CAE_avg_lastweek | True | 0.981409 | 262 | 35 | 4 | exchangeable |
| -0.008924 | 0.022906 | -0.389597 | 0.696834 | False | False | CAE | CAE_avg_norm | 2 | fourSC_ewma | True | 0.527652 | 262 | 35 | 4 | exchangeable |
| 0.086312 | 0.076550 | 1.127520 | 0.259523 | False | False | CAE | CAE_avg_norm | 3 | anticipated_affect_ewma | True | 0.356150 | 262 | 35 | 4 | exchangeable |

## Pooled RL Q GEE Coefficients

Gaussian GEE on the final fitted-Q regression from pooled FQI (``phi_obs`` vs bootstrap targets), clustered by participant. ``identified=False`` marks structurally unused features with zero design variance (e.g. ``b_tilde``, masked day-6 mediators); SE / p-values are omitted for those rows. RL priors still use ridge-FQI for ``mu_0_micro``; all-zero design columns get ``UNIDENTIFIED_PRIOR_VAR`` instead of the ``MIN_SIGMA2`` floor.

| coefficient | robust_se | z_stat | p_value | significant_0.05 | significant_0.01 | model | outcome | index | feature | identified | feature_std | n_obs | n_clusters | n_features | working_correlation | fqi_iters | ridge_alpha | residual_sigma2 | block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.151462 |  |  |  | False | False | q_no_td_modify | fqi_target | 0 | intercept | False | 0.000000 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| -0.058796 | 0.029190 | -2.014275 | 0.043981 | True | False | q_no_td_modify | fqi_target | 1 | weekday_vs_weekend | True | 0.372678 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| -0.004832 | 0.011045 | -0.437457 | 0.661780 | False | False | q_no_td_modify | fqi_target | 2 | slot_pm | True | 0.500000 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| -0.007272 | 0.001761 | -4.129372 | 3.637550e-05 | True | True | q_no_td_modify | fqi_target | 3 | E_w | True | 2.024864 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| 1.230626 | 0.005492 | 224.092036 | 0.000000 | True | True | q_no_td_modify | fqi_target | 4 | b_hat | True | 0.960113 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| 1.782810e-16 |  |  |  | False | False | q_no_td_modify | fqi_target | 5 | b_tilde | False | 0.000000 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| 0.199013 | 0.012256 | 16.238089 | 2.712418e-59 | True | True | q_no_td_modify | fqi_target | 6 | M_Y_anticipated_affect_ewma | True | 0.347580 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| 0.006406 | 0.001541 | 4.156420 | 3.232738e-05 | True | True | q_no_td_modify | fqi_target | 7 | M_Y_fourSC_ewma | True | 0.936432 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| 0.001467 | 0.001878 | 0.781281 | 0.434637 | False | False | q_no_td_modify | fqi_target | 8 | M_E_pageview_ewma | True | 1.042579 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| -0.103822 | 0.005164 | -20.106250 | 6.506029e-90 | True | True | q_no_td_modify | fqi_target | 9 | M_E_fitbit_wear_ewma | True | 0.409227 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| -0.029273 | 0.005337 | -5.485131 | 4.131643e-08 | True | True | q_no_td_modify | fqi_target | 10 | M_E_survey_complete_ewma | True | 0.415096 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| 0.002106 | 0.004029 | 0.522733 | 0.601160 | False | False | q_no_td_modify | fqi_target | 11 | yesterday_step_count | True | 0.936481 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| -0.011443 | 0.004050 | -2.825540 | 0.004720 | True | True | q_no_td_modify | fqi_target | 12 | prior2hour_step_count | True | 0.937702 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| -0.008787 | 0.009233 | -0.951686 | 0.341256 | False | False | q_no_td_modify | fqi_target | 13 | active_status_fraction_7days | True | 0.281500 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| 0.001152 | 0.002948 | 0.390700 | 0.696019 | False | False | q_no_td_modify | fqi_target | 14 | recent_burden | True | 0.941316 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| -0.054637 | 0.008732 | -6.257208 | 3.919300e-10 | True | True | q_no_td_modify | fqi_target | 15 | walk_interaction_7d | True | 0.349884 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| -0.010559 | 0.014616 | -0.722402 | 0.470048 | False | False | q_no_td_modify | fqi_target | 16 | A | True | 0.499646 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| -0.002029 | 0.002856 | -0.710230 | 0.477562 | False | False | q_no_td_modify | fqi_target | 17 | A*E_w | True | 1.499567 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| -0.005695 | 0.008827 | -0.645125 | 0.518846 | False | False | q_no_td_modify | fqi_target | 18 | A*b_hat | True | 0.678372 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| 1.597252e-17 |  |  |  | False | False | q_no_td_modify | fqi_target | 19 | A*b_tilde | False | 0.000000 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| -0.008888 | 0.006731 | -1.320550 | 0.186652 | False | False | q_no_td_modify | fqi_target | 20 | A*yesterday_step_count | True | 0.667043 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| 0.005175 | 0.004706 | 1.099536 | 0.271534 | False | False | q_no_td_modify | fqi_target | 21 | A*prior2hour_step_count | True | 0.664418 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| 0.016086 | 0.016932 | 0.950016 | 0.342104 | False | False | q_no_td_modify | fqi_target | 22 | A*active_status_fraction_7days | True | 0.387150 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| -0.003066 | 0.003857 | -0.794865 | 0.426692 | False | False | q_no_td_modify | fqi_target | 23 | A*recent_burden | True | 0.676751 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
| 0.007652 | 0.014421 | 0.530659 | 0.595655 | False | False | q_no_td_modify | fqi_target | 24 | A*walk_interaction_7d | True | 0.342632 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024031 | beta |
