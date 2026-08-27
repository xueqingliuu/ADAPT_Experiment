# Prior Mean and Variance Summary

`prior_variance` is the diagonal entry of the prior covariance. For `q_td_modify_joint`, the full covariance remains in `rl_priors.json`; this table reports marginal variances.

## PF Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| PF | fourSC | theta | 0 | intercept | -0.045670 | 0.047352 |
| PF | fourSC | theta | 1 | stepCountNext4HourLag1 | 0.474239 | 0.039214 |
| PF | fourSC | theta | 2 | yesterdayStepCount | -0.131156 | 0.012200 |
| PF | fourSC | theta | 3 | stepCountLast7DaysEma | 0.368042 | 0.032000 |
| PF | fourSC | theta | 4 | prior2HourStepCount | 0.129246 | 0.031148 |
| PF | fourSC | theta | 5 | activitySuggestionsSentLast7Days | -0.026926 | 0.027655 |
| PF | fourSC | theta | 6 | activitySuggestionInteractLast7Days | 0.078786 | 0.072474 |
| PF | fourSC | theta | 7 | activeDaysLast7Days | 0.124281 | 0.046821 |
| PF | fourSC | theta | 8 | isWeekend | -0.044306 | 0.145085 |
| PF | fourSC | theta | 9 | decisionTimeSlot | -0.102752 | 0.153274 |
| PF | fourSC | theta | 10 | perceivedUtilityLastWeek | 0.014894 | 0.013554 |
| PF | fourSC | theta | 11 | caeAverageLastWeek | -0.063918 | 0.094642 |
| PF | fourSC | theta | 12 | Ah | -0.075317 | 0.059964 |
| PF | fourSC | theta | 13 | Ah*yesterdayStepCount | 0.006655 | 0.019807 |
| PF | fourSC | theta | 14 | Ah*prior2HourStepCount | 0.018185 | 0.021101 |
| PF | fourSC | theta | 15 | Ah*activitySuggestionsSentLast7Days | 0.030062 | 0.026739 |
| PF | fourSC | theta | 16 | Ah*activitySuggestionInteractLast7Days | 0.088429 | 0.101369 |
| PF | fourSC | theta | 17 | Ah*perceivedUtilityLastWeek | 0.008842 | 0.023719 |
| PF | fourSC | theta | 18 | Ah*caeAverageLastWeek | 0.034895 | 0.082946 |
| PF | antic | theta | 0 | intercept | 0.464118 | 0.012414 |
| PF | antic | theta | 1 | anticipated_affect_yesterday | 0.157081 | 0.005049 |
| PF | antic | theta | 2 | active_status_fraction_7days | 0.062426 | 0.008443 |
| PF | antic | theta | 3 | is_weekend | -0.020114 | 0.003879 |
| PF | antic | theta | 4 | perceived_utility_lastweek | 0.003949 | 0.011332 |
| PF | antic | theta | 5 | CAE_avg_lastweek | 0.252418 | 0.012833 |
| PF | antic | theta | 6 | recent_burden | 0.018843 | 0.003799 |
| PF | antic | theta | 7 | A0_morning | 0.006314 | 0.002533 |
| PF | antic | theta | 8 | A1_afternoon | 0.021020 | 0.001437 |
| PF | antic | theta | 9 | A0_morning_by_perceived_utility_lastweek | -0.002127 | 0.001244 |
| PF | antic | theta | 10 | A1_afternoon_by_perceived_utility_lastweek | -0.005076 | 0.000743 |
| PF | antic | theta | 11 | A0_morning_by_CAE_avg_lastweek | -0.006890 | 0.004901 |
| PF | antic | theta | 12 | A1_afternoon_by_CAE_avg_lastweek | -0.022416 | 0.002647 |
| PF | antic | theta | 13 | A0_morning_by_recent_burden | -0.012200 | 0.001274 |
| PF | antic | theta | 14 | A1_afternoon_by_recent_burden | 0.001261 | 0.003713 |
| PF | CAE | theta | 0 | intercept | -0.085373 | 0.100058 |
| PF | CAE | theta | 1 | CAE_avg_lastweek | 0.920348 | 0.059327 |
| PF | CAE | theta | 2 | fourSC_ewma | -0.007744 | 0.012901 |
| PF | CAE | theta | 3 | anticipated_affect_ewma | 0.145659 | 0.024469 |
| PF | CAE_short | theta | 0 | intercept | 0.003419 | 0.079414 |
| PF | CAE_short | theta | 1 | caeAverage | 0.967518 | 0.056877 |

## RL Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| RL | reward_shaping | eta | 0 | intercept | -0.045636 | 0.024229 |
| RL | reward_shaping | eta | 1 | weekday_vs_weekend | -0.007606 | 0.000673 |
| RL | reward_shaping | eta | 2 | slot_pm | -0.022818 | 0.006057 |
| RL | reward_shaping | eta | 3 | E_w | 0.068707 | 0.002594 |
| RL | reward_shaping | eta | 4 | b_hat | 0.061793 | 0.040423 |
| RL | reward_shaping | eta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | reward_shaping | eta | 6 | M_Y_anticipated_affect_ewma | 0.072054 | 0.010183 |
| RL | reward_shaping | eta | 7 | M_Y_fourSC_ewma | 0.007634 | 0.022987 |
| RL | reward_shaping | eta | 8 | M_E_pageview_ewma | -0.007425 | 0.003787 |
| RL | reward_shaping | eta | 9 | M_E_fitbit_wear_ewma | -0.019244 | 0.007202 |
| RL | reward_shaping | eta | 10 | M_E_survey_complete_ewma | -0.000278 | 0.014493 |
| RL | reward_shaping | eta | 11 | yesterday_step_count | -0.000387 | 0.014483 |
| RL | reward_shaping | eta | 12 | prior2hour_step_count | -0.015983 | 0.005869 |
| RL | reward_shaping | eta | 13 | active_status_fraction_7days | 0.014516 | 0.035478 |
| RL | reward_shaping | eta | 14 | recent_burden | 0.000900 | 0.003883 |
| RL | reward_shaping | eta | 15 | walk_interaction_7d | 0.019577 | 0.031359 |
| RL | redistribution_stage1_AA | eta | 0 | redistribution_stage1_AA_coef_0 | 0.268183 | 0.007655 |
| RL | redistribution_stage1_AA | eta | 1 | redistribution_stage1_AA_coef_1 | -9.383158e-05 | 0.001148 |
| RL | redistribution_stage1_AA | eta | 2 | redistribution_stage1_AA_coef_2 | 0.103943 | 0.005235 |
| RL | redistribution_stage1_AA | eta | 3 | redistribution_stage1_AA_coef_3 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_AA | eta | 4 | redistribution_stage1_AA_coef_4 | 0.138298 | 0.001805 |
| RL | redistribution_stage1_AA | eta | 5 | redistribution_stage1_AA_coef_5 | -0.009642 | 0.002623 |
| RL | redistribution_stage1_AA | eta | 6 | redistribution_stage1_AA_coef_6 | 0.010950 | 0.000534 |
| RL | redistribution_stage1_AA | eta | 7 | redistribution_stage1_AA_coef_7 | -0.038348 | 0.001385 |
| RL | redistribution_stage1_AA | eta | 8 | redistribution_stage1_AA_coef_8 | -0.020700 | 0.001181 |
| RL | redistribution_stage1_AA | eta | 9 | redistribution_stage1_AA_coef_9 | 0.006130 | 0.000174 |
| RL | redistribution_stage1_AA | eta | 10 | redistribution_stage1_AA_coef_10 | 0.011650 | 0.000309 |
| RL | redistribution_stage1_AA | eta | 11 | redistribution_stage1_AA_coef_11 | 0.019718 | 0.004266 |
| RL | redistribution_stage1_AA | eta | 12 | redistribution_stage1_AA_coef_12 | 0.001992 | 0.000103 |
| RL | redistribution_stage1_AA | eta | 13 | redistribution_stage1_AA_coef_13 | 0.001294 | 0.002410 |
| RL | redistribution_stage1_AA | eta | 14 | redistribution_stage1_AA_coef_14 | 0.009111 | 0.001245 |
| RL | redistribution_stage1_AA | eta | 15 | redistribution_stage1_AA_coef_15 | 0.000779 | 0.000522 |
| RL | redistribution_stage1_AA | eta | 16 | redistribution_stage1_AA_coef_16 | -0.009888 | 0.002073 |
| RL | redistribution_stage1_AA | eta | 17 | redistribution_stage1_AA_coef_17 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_AA | eta | 18 | redistribution_stage1_AA_coef_18 | 0.001556 | 0.000531 |
| RL | redistribution_stage1_AA | eta | 19 | redistribution_stage1_AA_coef_19 | 0.004527 | 0.000489 |
| RL | redistribution_stage1_AA | eta | 20 | redistribution_stage1_AA_coef_20 | -0.009762 | 0.002749 |
| RL | redistribution_stage1_AA | eta | 21 | redistribution_stage1_AA_coef_21 | 0.002447 | 0.000260 |
| RL | redistribution_stage1_AA | eta | 22 | redistribution_stage1_AA_coef_22 | 0.002501 | 0.000933 |
| RL | redistribution_stage1_FW | eta | 0 | redistribution_stage1_FW_coef_0 | 0.392256 | 0.019438 |
| RL | redistribution_stage1_FW | eta | 1 | redistribution_stage1_FW_coef_1 | 0.006349 | 0.005112 |
| RL | redistribution_stage1_FW | eta | 2 | redistribution_stage1_FW_coef_2 | 0.025933 | 0.010581 |
| RL | redistribution_stage1_FW | eta | 3 | redistribution_stage1_FW_coef_3 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_FW | eta | 4 | redistribution_stage1_FW_coef_4 | -0.133291 | 0.026046 |
| RL | redistribution_stage1_FW | eta | 5 | redistribution_stage1_FW_coef_5 | -0.017838 | 0.009114 |
| RL | redistribution_stage1_FW | eta | 6 | redistribution_stage1_FW_coef_6 | 0.018275 | 0.001548 |
| RL | redistribution_stage1_FW | eta | 7 | redistribution_stage1_FW_coef_7 | 0.169178 | 0.015784 |
| RL | redistribution_stage1_FW | eta | 8 | redistribution_stage1_FW_coef_8 | -0.021972 | 0.008342 |
| RL | redistribution_stage1_FW | eta | 9 | redistribution_stage1_FW_coef_9 | 0.008559 | 0.003577 |
| RL | redistribution_stage1_FW | eta | 10 | redistribution_stage1_FW_coef_10 | 0.015106 | 0.002486 |
| RL | redistribution_stage1_FW | eta | 11 | redistribution_stage1_FW_coef_11 | -0.002209 | 0.014435 |
| RL | redistribution_stage1_FW | eta | 12 | redistribution_stage1_FW_coef_12 | -0.004961 | 0.003255 |
| RL | redistribution_stage1_FW | eta | 13 | redistribution_stage1_FW_coef_13 | 0.034996 | 0.018072 |
| RL | redistribution_stage1_FW | eta | 14 | redistribution_stage1_FW_coef_14 | -0.000689 | 0.009788 |
| RL | redistribution_stage1_FW | eta | 15 | redistribution_stage1_FW_coef_15 | -0.006604 | 0.005119 |
| RL | redistribution_stage1_FW | eta | 16 | redistribution_stage1_FW_coef_16 | -0.001538 | 0.012649 |
| RL | redistribution_stage1_FW | eta | 17 | redistribution_stage1_FW_coef_17 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_FW | eta | 18 | redistribution_stage1_FW_coef_18 | 0.001163 | 0.007409 |
| RL | redistribution_stage1_FW | eta | 19 | redistribution_stage1_FW_coef_19 | -0.007921 | 0.004708 |
| RL | redistribution_stage1_FW | eta | 20 | redistribution_stage1_FW_coef_20 | -0.005390 | 0.017850 |
| RL | redistribution_stage1_FW | eta | 21 | redistribution_stage1_FW_coef_21 | 0.002547 | 0.004728 |
| RL | redistribution_stage1_FW | eta | 22 | redistribution_stage1_FW_coef_22 | 0.008740 | 0.020585 |
| RL | redistribution_stage1_PJ | eta | 0 | redistribution_stage1_PJ_coef_0 | 0.107145 | 0.034016 |
| RL | redistribution_stage1_PJ | eta | 1 | redistribution_stage1_PJ_coef_1 | 0.007840 | 0.006278 |
| RL | redistribution_stage1_PJ | eta | 2 | redistribution_stage1_PJ_coef_2 | 0.007333 | 0.021458 |
| RL | redistribution_stage1_PJ | eta | 3 | redistribution_stage1_PJ_coef_3 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_PJ | eta | 4 | redistribution_stage1_PJ_coef_4 | -0.112711 | 0.030685 |
| RL | redistribution_stage1_PJ | eta | 5 | redistribution_stage1_PJ_coef_5 | -0.015955 | 0.018437 |
| RL | redistribution_stage1_PJ | eta | 6 | redistribution_stage1_PJ_coef_6 | 0.016262 | 0.002308 |
| RL | redistribution_stage1_PJ | eta | 7 | redistribution_stage1_PJ_coef_7 | 0.002199 | 0.009587 |
| RL | redistribution_stage1_PJ | eta | 8 | redistribution_stage1_PJ_coef_8 | 0.167638 | 0.010759 |
| RL | redistribution_stage1_PJ | eta | 9 | redistribution_stage1_PJ_coef_9 | -0.001788 | 0.007680 |
| RL | redistribution_stage1_PJ | eta | 10 | redistribution_stage1_PJ_coef_10 | 0.014371 | 0.003027 |
| RL | redistribution_stage1_PJ | eta | 11 | redistribution_stage1_PJ_coef_11 | 0.081125 | 0.033749 |
| RL | redistribution_stage1_PJ | eta | 12 | redistribution_stage1_PJ_coef_12 | 0.013475 | 0.004459 |
| RL | redistribution_stage1_PJ | eta | 13 | redistribution_stage1_PJ_coef_13 | 0.168107 | 0.022712 |
| RL | redistribution_stage1_PJ | eta | 14 | redistribution_stage1_PJ_coef_14 | 0.029995 | 0.010172 |
| RL | redistribution_stage1_PJ | eta | 15 | redistribution_stage1_PJ_coef_15 | 0.002518 | 0.008325 |
| RL | redistribution_stage1_PJ | eta | 16 | redistribution_stage1_PJ_coef_16 | 0.022018 | 0.020352 |
| RL | redistribution_stage1_PJ | eta | 17 | redistribution_stage1_PJ_coef_17 | 0.000000 | 1.000000 |
| RL | redistribution_stage1_PJ | eta | 18 | redistribution_stage1_PJ_coef_18 | 0.014433 | 0.012064 |
| RL | redistribution_stage1_PJ | eta | 19 | redistribution_stage1_PJ_coef_19 | 0.002626 | 0.009298 |
| RL | redistribution_stage1_PJ | eta | 20 | redistribution_stage1_PJ_coef_20 | -0.036878 | 0.027422 |
| RL | redistribution_stage1_PJ | eta | 21 | redistribution_stage1_PJ_coef_21 | 0.000983 | 0.008557 |
| RL | redistribution_stage1_PJ | eta | 22 | redistribution_stage1_PJ_coef_22 | -0.019569 | 0.020061 |
| RL | redistribution_stage2_v2 | eta | 0 | redistribution_stage2_v2_coef_0 | -0.226590 | 0.000920 |
| RL | redistribution_stage2_v2 | eta | 1 | redistribution_stage2_v2_coef_1 | -0.037765 | 2.555308e-05 |
| RL | redistribution_stage2_v2 | eta | 2 | redistribution_stage2_v2_coef_2 | 0.035526 | 0.000531 |
| RL | redistribution_stage2_v2 | eta | 3 | redistribution_stage2_v2_coef_3 | -0.091144 | 0.001398 |
| RL | redistribution_stage2_v2 | eta | 4 | redistribution_stage2_v2_coef_4 | 0.000000 | 1.000000 |
| RL | redistribution_stage2_v2 | eta | 5 | redistribution_stage2_v2_coef_5 | 0.280265 | 0.002103 |
| RL | redistribution_stage2_v2 | eta | 6 | redistribution_stage2_v2_coef_6 | 0.001054 | 0.002190 |
| RL | redistribution_stage2_v2 | eta | 7 | redistribution_stage2_v2_coef_7 | -0.028359 | 0.001240 |
| RL | redistribution_stage2_v2 | eta | 8 | redistribution_stage2_v2_coef_8 | 0.075600 | 0.001422 |
| RL | redistribution_stage2_v2 | eta | 9 | redistribution_stage2_v2_coef_9 | -0.000265 | 0.002172 |
| RL | redistribution_stage2_v2 | eta | 10 | redistribution_stage2_v2_coef_10 | 0.016947 | 0.001794 |
| RL | redistribution_stage2_v2 | eta | 11 | redistribution_stage2_v2_coef_11 | -0.011915 | 0.000892 |
| RL | redistribution_stage2_v2 | eta | 12 | redistribution_stage2_v2_coef_12 | -0.060212 | 0.000945 |
| RL | redistribution_stage2_v2 | eta | 13 | redistribution_stage2_v2_coef_13 | 0.008373 | 0.000199 |
| RL | redistribution_stage2_v2 | eta | 14 | redistribution_stage2_v2_coef_14 | 0.126455 | 0.002479 |
| RL | redistribution_stage2_v2 | eta | 15 | redistribution_stage2_v2_coef_15 | -0.008503 | 0.000824 |
| RL | redistribution_stage2_v2 | eta | 16 | redistribution_stage2_v2_coef_16 | 0.004153 | 0.001677 |
| RL | redistribution_stage2_v2 | eta | 17 | redistribution_stage2_v2_coef_17 | 0.765572 | 0.000111 |
| RL | redistribution_stage2_v2 | eta | 18 | redistribution_stage2_v2_coef_18 | -0.465720 | 0.000487 |
| RL | redistribution_stage2_v2 | eta | 19 | redistribution_stage2_v2_coef_19 | -0.091354 | 0.000351 |
| RL | redistribution_stage2_v4 | eta | 0 | redistribution_stage2_v4_coef_0 | 0.033593 | 0.005022 |
| RL | redistribution_stage2_v4 | eta | 1 | redistribution_stage2_v4_coef_1 | 0.005599 | 0.000140 |
| RL | redistribution_stage2_v4 | eta | 2 | redistribution_stage2_v4_coef_2 | -0.023439 | 0.002102 |
| RL | redistribution_stage2_v4 | eta | 3 | redistribution_stage2_v4_coef_3 | 0.039315 | 0.019020 |
| RL | redistribution_stage2_v4 | eta | 4 | redistribution_stage2_v4_coef_4 | 0.000000 | 1.000000 |
| RL | redistribution_stage2_v4 | eta | 5 | redistribution_stage2_v4_coef_5 | 0.082661 | 0.011278 |
| RL | redistribution_stage2_v4 | eta | 6 | redistribution_stage2_v4_coef_6 | 0.079053 | 0.030641 |
| RL | redistribution_stage2_v4 | eta | 7 | redistribution_stage2_v4_coef_7 | -0.032691 | 0.004576 |
| RL | redistribution_stage2_v4 | eta | 8 | redistribution_stage2_v4_coef_8 | 0.099690 | 0.005199 |
| RL | redistribution_stage2_v4 | eta | 9 | redistribution_stage2_v4_coef_9 | -0.202779 | 0.020842 |
| RL | redistribution_stage2_v4 | eta | 10 | redistribution_stage2_v4_coef_10 | -0.001173 | 0.018181 |
| RL | redistribution_stage2_v4 | eta | 11 | redistribution_stage2_v4_coef_11 | -0.027491 | 0.006411 |
| RL | redistribution_stage2_v4 | eta | 12 | redistribution_stage2_v4_coef_12 | -0.064431 | 0.006235 |
| RL | redistribution_stage2_v4 | eta | 13 | redistribution_stage2_v4_coef_13 | -0.018165 | 0.003618 |
| RL | redistribution_stage2_v4 | eta | 14 | redistribution_stage2_v4_coef_14 | -0.140768 | 0.014339 |
| RL | redistribution_stage2_v4 | eta | 15 | redistribution_stage2_v4_coef_15 | -0.061441 | 0.023055 |
| RL | redistribution_stage2_v4 | eta | 16 | redistribution_stage2_v4_coef_16 | 0.019414 | 0.010997 |
| RL | redistribution_stage2_v4 | eta | 17 | redistribution_stage2_v4_coef_17 | 0.195396 | 0.000840 |
| RL | redistribution_stage2_v4 | eta | 18 | redistribution_stage2_v4_coef_18 | -0.694646 | 0.001819 |
| RL | redistribution_stage2_v4 | eta | 19 | redistribution_stage2_v4_coef_19 | 1.152319 | 0.003037 |
| RL | q_no_td_modify | beta | 0 | intercept | 0.146850 | 0.082330 |
| RL | q_no_td_modify | beta | 1 | weekday_vs_weekend | -0.055564 | 0.083249 |
| RL | q_no_td_modify | beta | 2 | slot_pm | -0.003540 | 0.001998 |
| RL | q_no_td_modify | beta | 3 | E_w | -0.002330 | 0.029401 |
| RL | q_no_td_modify | beta | 4 | b_hat | 1.224083 | 0.075076 |
| RL | q_no_td_modify | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify | beta | 6 | M_Y_anticipated_affect_ewma | 0.208509 | 0.006123 |
| RL | q_no_td_modify | beta | 7 | M_Y_fourSC_ewma | 0.000732 | 0.005518 |
| RL | q_no_td_modify | beta | 8 | M_E_pageview_ewma | 0.001660 | 0.000256 |
| RL | q_no_td_modify | beta | 9 | M_E_fitbit_wear_ewma | -0.106839 | 0.002944 |
| RL | q_no_td_modify | beta | 10 | M_E_survey_complete_ewma | -0.035125 | 0.004231 |
| RL | q_no_td_modify | beta | 11 | yesterday_step_count | 0.004656 | 0.000857 |
| RL | q_no_td_modify | beta | 12 | prior2hour_step_count | -0.008077 | 0.000635 |
| RL | q_no_td_modify | beta | 13 | active_status_fraction_7days | -0.028332 | 0.029068 |
| RL | q_no_td_modify | beta | 14 | recent_burden | 0.000900 | 0.002104 |
| RL | q_no_td_modify | beta | 15 | walk_interaction_7d | -0.049784 | 0.020335 |
| RL | q_no_td_modify | beta | 16 | A | -0.002465 | 0.001257 |
| RL | q_no_td_modify | beta | 17 | A*E_w | 0.000469 | 0.000934 |
| RL | q_no_td_modify | beta | 18 | A*b_hat | -0.006915 | 0.001762 |
| RL | q_no_td_modify | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify | beta | 20 | A*yesterday_step_count | -0.008790 | 0.002016 |
| RL | q_no_td_modify | beta | 21 | A*prior2hour_step_count | 0.005102 | 0.000981 |
| RL | q_no_td_modify | beta | 22 | A*active_status_fraction_7days | 0.006566 | 0.001045 |
| RL | q_no_td_modify | beta | 23 | A*recent_burden | -0.002787 | 0.000740 |
| RL | q_no_td_modify | beta | 24 | A*walk_interaction_7d | -0.000357 | 0.001537 |
| RL | q_no_td_modify_g09 | beta | 0 | intercept | 0.258238 | 0.150450 |
| RL | q_no_td_modify_g09 | beta | 1 | weekday_vs_weekend | -0.044753 | 0.253714 |
| RL | q_no_td_modify_g09 | beta | 2 | slot_pm | 0.000847 | 0.008097 |
| RL | q_no_td_modify_g09 | beta | 3 | E_w | -0.004920 | 0.050265 |
| RL | q_no_td_modify_g09 | beta | 4 | b_hat | 1.697096 | 0.145527 |
| RL | q_no_td_modify_g09 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g09 | beta | 6 | M_Y_anticipated_affect_ewma | 0.366484 | 0.013850 |
| RL | q_no_td_modify_g09 | beta | 7 | M_Y_fourSC_ewma | 0.002886 | 0.009907 |
| RL | q_no_td_modify_g09 | beta | 8 | M_E_pageview_ewma | 0.003912 | 0.000580 |
| RL | q_no_td_modify_g09 | beta | 9 | M_E_fitbit_wear_ewma | -0.175134 | 0.006533 |
| RL | q_no_td_modify_g09 | beta | 10 | M_E_survey_complete_ewma | -0.057013 | 0.008608 |
| RL | q_no_td_modify_g09 | beta | 11 | yesterday_step_count | 0.015201 | 0.001550 |
| RL | q_no_td_modify_g09 | beta | 12 | prior2hour_step_count | -0.013698 | 0.001200 |
| RL | q_no_td_modify_g09 | beta | 13 | active_status_fraction_7days | -0.046834 | 0.056270 |
| RL | q_no_td_modify_g09 | beta | 14 | recent_burden | 0.004787 | 0.005529 |
| RL | q_no_td_modify_g09 | beta | 15 | walk_interaction_7d | -0.076198 | 0.033014 |
| RL | q_no_td_modify_g09 | beta | 16 | A | -0.012014 | 0.004013 |
| RL | q_no_td_modify_g09 | beta | 17 | A*E_w | 9.706707e-05 | 0.002374 |
| RL | q_no_td_modify_g09 | beta | 18 | A*b_hat | -0.014296 | 0.003851 |
| RL | q_no_td_modify_g09 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g09 | beta | 20 | A*yesterday_step_count | -0.017825 | 0.004037 |
| RL | q_no_td_modify_g09 | beta | 21 | A*prior2hour_step_count | 0.008150 | 0.001835 |
| RL | q_no_td_modify_g09 | beta | 22 | A*active_status_fraction_7days | 0.017436 | 0.002509 |
| RL | q_no_td_modify_g09 | beta | 23 | A*recent_burden | -0.004435 | 0.001441 |
| RL | q_no_td_modify_g09 | beta | 24 | A*walk_interaction_7d | 0.004975 | 0.003056 |
| RL | q_no_td_modify_g099 | beta | 0 | intercept | 0.292246 | 0.173226 |
| RL | q_no_td_modify_g099 | beta | 1 | weekday_vs_weekend | -0.037063 | 0.330529 |
| RL | q_no_td_modify_g099 | beta | 2 | slot_pm | 0.002852 | 0.011128 |
| RL | q_no_td_modify_g099 | beta | 3 | E_w | -0.005714 | 0.057055 |
| RL | q_no_td_modify_g099 | beta | 4 | b_hat | 1.834091 | 0.170051 |
| RL | q_no_td_modify_g099 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g099 | beta | 6 | M_Y_anticipated_affect_ewma | 0.417004 | 0.017162 |
| RL | q_no_td_modify_g099 | beta | 7 | M_Y_fourSC_ewma | 0.003618 | 0.011403 |
| RL | q_no_td_modify_g099 | beta | 8 | M_E_pageview_ewma | 0.004741 | 0.000706 |
| RL | q_no_td_modify_g099 | beta | 9 | M_E_fitbit_wear_ewma | -0.196060 | 0.008325 |
| RL | q_no_td_modify_g099 | beta | 10 | M_E_survey_complete_ewma | -0.063841 | 0.010773 |
| RL | q_no_td_modify_g099 | beta | 11 | yesterday_step_count | 0.018852 | 0.001851 |
| RL | q_no_td_modify_g099 | beta | 12 | prior2hour_step_count | -0.015532 | 0.001407 |
| RL | q_no_td_modify_g099 | beta | 13 | active_status_fraction_7days | -0.053538 | 0.066013 |
| RL | q_no_td_modify_g099 | beta | 14 | recent_burden | 0.006104 | 0.007057 |
| RL | q_no_td_modify_g099 | beta | 15 | walk_interaction_7d | -0.084498 | 0.037651 |
| RL | q_no_td_modify_g099 | beta | 16 | A | -0.015316 | 0.005356 |
| RL | q_no_td_modify_g099 | beta | 17 | A*E_w | -5.993542e-05 | 0.003025 |
| RL | q_no_td_modify_g099 | beta | 18 | A*b_hat | -0.016797 | 0.004755 |
| RL | q_no_td_modify_g099 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g099 | beta | 20 | A*yesterday_step_count | -0.020835 | 0.004794 |
| RL | q_no_td_modify_g099 | beta | 21 | A*prior2hour_step_count | 0.009127 | 0.002180 |
| RL | q_no_td_modify_g099 | beta | 22 | A*active_status_fraction_7days | 0.021096 | 0.003244 |
| RL | q_no_td_modify_g099 | beta | 23 | A*recent_burden | -0.004960 | 0.001708 |
| RL | q_no_td_modify_g099 | beta | 24 | A*walk_interaction_7d | 0.006826 | 0.003709 |
| RL | q_redistribution_v2 | beta | 0 | intercept | 1.217639 | 0.244039 |
| RL | q_redistribution_v2 | beta | 1 | weekday_vs_weekend | -0.217407 | 0.065201 |
| RL | q_redistribution_v2 | beta | 2 | slot_pm | -0.016712 | 0.001895 |
| RL | q_redistribution_v2 | beta | 3 | E_w | 0.653003 | 0.443194 |
| RL | q_redistribution_v2 | beta | 4 | b_hat | 1.290128 | 0.139510 |
| RL | q_redistribution_v2 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v2 | beta | 6 | M_Y_anticipated_affect_ewma | 0.511466 | 0.022760 |
| RL | q_redistribution_v2 | beta | 7 | M_Y_fourSC_ewma | -0.013381 | 0.020316 |
| RL | q_redistribution_v2 | beta | 8 | M_E_pageview_ewma | -0.013453 | 0.004398 |
| RL | q_redistribution_v2 | beta | 9 | M_E_fitbit_wear_ewma | -0.322185 | 0.013216 |
| RL | q_redistribution_v2 | beta | 10 | M_E_survey_complete_ewma | -0.026404 | 0.016697 |
| RL | q_redistribution_v2 | beta | 11 | yesterday_step_count | 0.041779 | 0.007180 |
| RL | q_redistribution_v2 | beta | 12 | prior2hour_step_count | 0.028246 | 0.002325 |
| RL | q_redistribution_v2 | beta | 13 | active_status_fraction_7days | -0.640567 | 0.165066 |
| RL | q_redistribution_v2 | beta | 14 | recent_burden | 0.113186 | 0.007841 |
| RL | q_redistribution_v2 | beta | 15 | walk_interaction_7d | 0.766184 | 0.087209 |
| RL | q_redistribution_v2 | beta | 16 | A | 0.095172 | 0.008429 |
| RL | q_redistribution_v2 | beta | 17 | A*E_w | 0.007012 | 0.001547 |
| RL | q_redistribution_v2 | beta | 18 | A*b_hat | -0.019407 | 0.002809 |
| RL | q_redistribution_v2 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v2 | beta | 20 | A*yesterday_step_count | 0.031402 | 0.001474 |
| RL | q_redistribution_v2 | beta | 21 | A*prior2hour_step_count | 0.003320 | 0.002432 |
| RL | q_redistribution_v2 | beta | 22 | A*active_status_fraction_7days | -0.041907 | 0.006210 |
| RL | q_redistribution_v2 | beta | 23 | A*recent_burden | 0.018506 | 0.000776 |
| RL | q_redistribution_v2 | beta | 24 | A*walk_interaction_7d | -0.036530 | 0.007387 |
| RL | q_redistribution_v4 | beta | 0 | intercept | 0.017415 | 0.396208 |
| RL | q_redistribution_v4 | beta | 1 | weekday_vs_weekend | -0.097591 | 0.066602 |
| RL | q_redistribution_v4 | beta | 2 | slot_pm | 0.000311 | 0.001863 |
| RL | q_redistribution_v4 | beta | 3 | E_w | -0.288617 | 0.108409 |
| RL | q_redistribution_v4 | beta | 4 | b_hat | 1.685040 | 0.186931 |
| RL | q_redistribution_v4 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v4 | beta | 6 | M_Y_anticipated_affect_ewma | 0.131957 | 0.038600 |
| RL | q_redistribution_v4 | beta | 7 | M_Y_fourSC_ewma | -0.126291 | 0.084575 |
| RL | q_redistribution_v4 | beta | 8 | M_E_pageview_ewma | -0.020350 | 0.021043 |
| RL | q_redistribution_v4 | beta | 9 | M_E_fitbit_wear_ewma | -0.079253 | 0.017136 |
| RL | q_redistribution_v4 | beta | 10 | M_E_survey_complete_ewma | -0.010776 | 0.083113 |
| RL | q_redistribution_v4 | beta | 11 | yesterday_step_count | -0.015923 | 0.059582 |
| RL | q_redistribution_v4 | beta | 12 | prior2hour_step_count | 0.049517 | 0.013690 |
| RL | q_redistribution_v4 | beta | 13 | active_status_fraction_7days | 0.363717 | 0.207133 |
| RL | q_redistribution_v4 | beta | 14 | recent_burden | 0.031089 | 0.076542 |
| RL | q_redistribution_v4 | beta | 15 | walk_interaction_7d | 0.311200 | 0.269605 |
| RL | q_redistribution_v4 | beta | 16 | A | 0.076206 | 0.023908 |
| RL | q_redistribution_v4 | beta | 17 | A*E_w | -0.002435 | 0.006462 |
| RL | q_redistribution_v4 | beta | 18 | A*b_hat | -0.007112 | 0.010615 |
| RL | q_redistribution_v4 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v4 | beta | 20 | A*yesterday_step_count | 0.041498 | 0.006644 |
| RL | q_redistribution_v4 | beta | 21 | A*prior2hour_step_count | -0.012988 | 0.019414 |
| RL | q_redistribution_v4 | beta | 22 | A*active_status_fraction_7days | -0.066639 | 0.011377 |
| RL | q_redistribution_v4 | beta | 23 | A*recent_burden | -0.008234 | 0.002609 |
| RL | q_redistribution_v4 | beta | 24 | A*walk_interaction_7d | -0.081494 | 0.051043 |

## Joint Modified-TD Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| Joint Modified TD | q_td_modify_joint | eta | 0 | eta_intercept | 0.609502 | 0.153040 |
| Joint Modified TD | q_td_modify_joint | eta | 1 | eta_E_w | -0.024217 | 0.052833 |
| Joint Modified TD | q_td_modify_joint | eta | 2 | eta_b_hat | -0.330927 | 0.154121 |
| Joint Modified TD | q_td_modify_joint | eta | 3 | eta_b_tilde | 0.000000 | 1.000000 |
| Joint Modified TD | q_td_modify_joint | beta | 4 | beta_intercept | 0.601780 | 0.117532 |
| Joint Modified TD | q_td_modify_joint | beta | 5 | beta_weekday_vs_weekend | -0.092860 | 0.218039 |
| Joint Modified TD | q_td_modify_joint | beta | 6 | beta_slot_pm | -0.002441 | 0.004389 |
| Joint Modified TD | q_td_modify_joint | beta | 7 | beta_E_w | -0.024357 | 0.036665 |
| Joint Modified TD | q_td_modify_joint | beta | 8 | beta_b_hat | 0.123531 | 0.106868 |
| Joint Modified TD | q_td_modify_joint | beta | 9 | beta_b_tilde | 0.000000 | 1.000000 |
| Joint Modified TD | q_td_modify_joint | beta | 10 | beta_M_Y_anticipated_affect_ewma | 0.694704 | 0.013379 |
| Joint Modified TD | q_td_modify_joint | beta | 11 | beta_M_Y_fourSC_ewma | 0.062201 | 0.008681 |
| Joint Modified TD | q_td_modify_joint | beta | 12 | beta_M_E_pageview_ewma | 0.005406 | 0.001329 |
| Joint Modified TD | q_td_modify_joint | beta | 13 | beta_M_E_fitbit_wear_ewma | -0.340807 | 0.006589 |
| Joint Modified TD | q_td_modify_joint | beta | 14 | beta_M_E_survey_complete_ewma | -0.081343 | 0.008296 |
| Joint Modified TD | q_td_modify_joint | beta | 15 | beta_yesterday_step_count | 0.009720 | 0.000895 |
| Joint Modified TD | q_td_modify_joint | beta | 16 | beta_prior2hour_step_count | -0.005016 | 0.000843 |
| Joint Modified TD | q_td_modify_joint | beta | 17 | beta_active_status_fraction_7days | -0.044311 | 0.033832 |
| Joint Modified TD | q_td_modify_joint | beta | 18 | beta_recent_burden | 0.000691 | 0.001096 |
| Joint Modified TD | q_td_modify_joint | beta | 19 | beta_walk_interaction_7d | 0.045227 | 0.011228 |
| Joint Modified TD | q_td_modify_joint | beta | 20 | beta_A | -0.012564 | 0.003223 |
| Joint Modified TD | q_td_modify_joint | beta | 21 | beta_A*E_w | -7.541318e-05 | 0.001849 |
| Joint Modified TD | q_td_modify_joint | beta | 22 | beta_A*b_hat | -0.069161 | 0.002835 |
| Joint Modified TD | q_td_modify_joint | beta | 23 | beta_A*b_tilde | 0.000000 | 1.000000 |
| Joint Modified TD | q_td_modify_joint | beta | 24 | beta_A*yesterday_step_count | -0.001911 | 0.003433 |
| Joint Modified TD | q_td_modify_joint | beta | 25 | beta_A*prior2hour_step_count | -0.003335 | 0.001219 |
| Joint Modified TD | q_td_modify_joint | beta | 26 | beta_A*active_status_fraction_7days | 0.017931 | 0.001585 |
| Joint Modified TD | q_td_modify_joint | beta | 27 | beta_A*recent_burden | -0.002404 | 0.001366 |
| Joint Modified TD | q_td_modify_joint | beta | 28 | beta_A*walk_interaction_7d | 0.004085 | 0.002078 |

## Pooled PF Regression Coefficients

Coefficients are from the all-user stacked ridge fits used as PF prior means. `p_value` uses ridge sandwich standard errors and is approximate because ridge shrinks coefficients. `identified=False` marks zero-variance design columns; SE / p-values are omitted for those rows.

| model | outcome | index | feature | coefficient | identified | feature_std | std_error | t_stat | p_value | significant_0.05 | significant_0.01 | n_obs | n_features | ridge_alpha | residual_sigma2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fourSC | 4hour_step_norm | 0 | intercept | -0.045670 | False | 0.000000 |  |  |  | False | False | 3735 | 19 | 1.000000 | 0.524725 |
| fourSC | 4hour_step_norm | 1 | stepCountNext4HourLag1 | 0.474239 | True | 0.924861 | 0.015053 | 31.505186 | 2.667964e-193 | True | True | 3735 | 19 | 1.000000 | 0.524725 |
| fourSC | 4hour_step_norm | 2 | yesterdayStepCount | -0.131156 | True | 0.948113 | 0.019345 | -6.779700 | 1.395049e-11 | True | True | 3735 | 19 | 1.000000 | 0.524725 |
| fourSC | 4hour_step_norm | 3 | stepCountLast7DaysEma | 0.368042 | True | 1.005044 | 0.013885 | 26.507043 | 5.929145e-142 | True | True | 3735 | 19 | 1.000000 | 0.524725 |
| fourSC | 4hour_step_norm | 4 | prior2HourStepCount | 0.129246 | True | 0.941170 | 0.019053 | 6.783455 | 1.359681e-11 | True | True | 3735 | 19 | 1.000000 | 0.524725 |
| fourSC | 4hour_step_norm | 5 | activitySuggestionsSentLast7Days | -0.026926 | True | 0.967404 | 0.017738 | -1.518037 | 0.129090 | False | False | 3735 | 19 | 1.000000 | 0.524725 |
| fourSC | 4hour_step_norm | 6 | activitySuggestionInteractLast7Days | 0.078786 | True | 0.347380 | 0.050891 | 1.548131 | 0.121676 | False | False | 3735 | 19 | 1.000000 | 0.524725 |
| fourSC | 4hour_step_norm | 7 | activeDaysLast7Days | 0.124281 | True | 0.277433 | 0.044851 | 2.770971 | 0.005617 | True | True | 3735 | 19 | 1.000000 | 0.524725 |
| fourSC | 4hour_step_norm | 8 | isWeekend | -0.044306 | True | 0.364457 | 0.032560 | -1.360737 | 0.173679 | False | False | 3735 | 19 | 1.000000 | 0.524725 |
| fourSC | 4hour_step_norm | 9 | decisionTimeSlot | -0.102752 | True | 0.498173 | 0.025983 | -3.954555 | 7.810595e-05 | True | True | 3735 | 19 | 1.000000 | 0.524725 |
| fourSC | 4hour_step_norm | 10 | perceivedUtilityLastWeek | 0.014894 | True | 2.511183 | 0.007312 | 2.036979 | 0.041723 | True | False | 3735 | 19 | 1.000000 | 0.524725 |
| fourSC | 4hour_step_norm | 11 | caeAverageLastWeek | -0.063918 | True | 0.992053 | 0.017162 | -3.724419 | 0.000199 | True | True | 3735 | 19 | 1.000000 | 0.524725 |
| fourSC | 4hour_step_norm | 12 | Ah | -0.075317 | True | 0.499771 | 0.041118 | -1.831721 | 0.067073 | False | False | 3735 | 19 | 1.000000 | 0.524725 |
| fourSC | 4hour_step_norm | 13 | Ah*yesterdayStepCount | 0.006655 | True | 0.672639 | 0.026007 | 0.255914 | 0.798031 | False | False | 3735 | 19 | 1.000000 | 0.524725 |
| fourSC | 4hour_step_norm | 14 | Ah*prior2HourStepCount | 0.018185 | True | 0.666699 | 0.026319 | 0.690930 | 0.489653 | False | False | 3735 | 19 | 1.000000 | 0.524725 |
| fourSC | 4hour_step_norm | 15 | Ah*activitySuggestionsSentLast7Days | 0.030062 | True | 0.688627 | 0.024885 | 1.208017 | 0.227118 | False | False | 3735 | 19 | 1.000000 | 0.524725 |
| fourSC | 4hour_step_norm | 16 | Ah*activitySuggestionInteractLast7Days | 0.088429 | True | 0.349684 | 0.070416 | 1.255806 | 0.209265 | False | False | 3735 | 19 | 1.000000 | 0.524725 |
| fourSC | 4hour_step_norm | 17 | Ah*perceivedUtilityLastWeek | 0.008842 | True | 1.845777 | 0.009970 | 0.886889 | 0.375196 | False | False | 3735 | 19 | 1.000000 | 0.524725 |
| fourSC | 4hour_step_norm | 18 | Ah*caeAverageLastWeek | 0.034895 | True | 0.692755 | 0.024264 | 1.438137 | 0.150479 | False | False | 3735 | 19 | 1.000000 | 0.524725 |
| antic | anticipated_affect_norm | 0 | intercept | 0.464118 | False | 0.000000 |  |  |  | False | False | 1019 | 15 | 1.000000 | 0.036734 |
| antic | anticipated_affect_norm | 1 | anticipated_affect_yesterday | 0.157081 | True | 0.366210 | 0.019852 | 7.912493 | 6.637034e-15 | True | True | 1019 | 15 | 1.000000 | 0.036734 |
| antic | anticipated_affect_norm | 2 | active_status_fraction_7days | 0.062426 | True | 0.229784 | 0.027990 | 2.230272 | 0.025950 | True | False | 1019 | 15 | 1.000000 | 0.036734 |
| antic | anticipated_affect_norm | 3 | is_weekend | -0.020114 | True | 0.366570 | 0.016520 | -1.217549 | 0.223681 | False | False | 1019 | 15 | 1.000000 | 0.036734 |
| antic | anticipated_affect_norm | 4 | perceived_utility_lastweek | 0.003949 | True | 2.460228 | 0.004181 | 0.944421 | 0.345182 | False | False | 1019 | 15 | 1.000000 | 0.036734 |
| antic | anticipated_affect_norm | 5 | CAE_avg_lastweek | 0.252418 | True | 0.928551 | 0.012185 | 20.715211 | 1.197556e-79 | True | True | 1019 | 15 | 1.000000 | 0.036734 |
| antic | anticipated_affect_norm | 6 | recent_burden | 0.018843 | True | 0.974256 | 0.009471 | 1.989470 | 0.046920 | True | False | 1019 | 15 | 1.000000 | 0.036734 |
| antic | anticipated_affect_norm | 7 | A0_morning | 0.006314 | True | 0.499217 | 0.013576 | 0.465094 | 0.641965 | False | False | 1019 | 15 | 1.000000 | 0.036734 |
| antic | anticipated_affect_norm | 8 | A1_afternoon | 0.021020 | True | 0.499873 | 0.013515 | 1.555301 | 0.120190 | False | False | 1019 | 15 | 1.000000 | 0.036734 |
| antic | anticipated_affect_norm | 9 | A0_morning_by_perceived_utility_lastweek | -0.002127 | True | 1.866335 | 0.005209 | -0.408264 | 0.683167 | False | False | 1019 | 15 | 1.000000 | 0.036734 |
| antic | anticipated_affect_norm | 10 | A1_afternoon_by_perceived_utility_lastweek | -0.005076 | True | 1.863397 | 0.005220 | -0.972335 | 0.331118 | False | False | 1019 | 15 | 1.000000 | 0.036734 |
| antic | anticipated_affect_norm | 11 | A0_morning_by_CAE_avg_lastweek | -0.006890 | True | 0.686065 | 0.013250 | -0.519957 | 0.603208 | False | False | 1019 | 15 | 1.000000 | 0.036734 |
| antic | anticipated_affect_norm | 12 | A1_afternoon_by_CAE_avg_lastweek | -0.022416 | True | 0.657745 | 0.013257 | -1.690827 | 0.091180 | False | False | 1019 | 15 | 1.000000 | 0.036734 |
| antic | anticipated_affect_norm | 13 | A0_morning_by_recent_burden | -0.012200 | True | 0.682435 | 0.013880 | -0.879024 | 0.379599 | False | False | 1019 | 15 | 1.000000 | 0.036734 |
| antic | anticipated_affect_norm | 14 | A1_afternoon_by_recent_burden | 0.001261 | True | 0.665827 | 0.013829 | 0.091159 | 0.927384 | False | False | 1019 | 15 | 1.000000 | 0.036734 |
| CAE | CAE_avg_norm | 0 | intercept | -0.085373 | False | 0.000000 |  |  |  | False | False | 262 | 4 | 1.000000 | 0.115176 |
| CAE | CAE_avg_norm | 1 | CAE_avg_lastweek | 0.920348 | True | 0.981409 | 0.029292 | 31.420050 | 3.613334e-90 | True | True | 262 | 4 | 1.000000 | 0.115176 |
| CAE | CAE_avg_norm | 2 | fourSC_ewma | -0.007744 | True | 0.527652 | 0.039465 | -0.196220 | 0.844593 | False | False | 262 | 4 | 1.000000 | 0.115176 |
| CAE | CAE_avg_norm | 3 | anticipated_affect_ewma | 0.145659 | True | 0.356150 | 0.078324 | 1.859703 | 0.064066 | False | False | 262 | 4 | 1.000000 | 0.115176 |

## Pooled PF GEE Coefficients

Population-averaged GEE fits on the same stacked PF designs, clustered by `ParticipantIdentifier`. `p_value` uses GEE sandwich standard errors with exchangeable working correlation. `identified=False` marks zero-variance design columns; SE / p-values are omitted for those rows. These are for inference/audit only; PF priors still use ridge.

| coefficient | robust_se | z_stat | p_value | significant_0.05 | significant_0.01 | model | outcome | index | feature | identified | feature_std | n_obs | n_clusters | n_features | working_correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -0.069087 |  |  |  | False | False | fourSC | 4hour_step_norm | 0 | intercept | False | 0.000000 | 3735 | 35 | 19 | exchangeable |
| -0.059784 | 0.038748 | -1.542901 | 0.122855 | False | False | fourSC | 4hour_step_norm | 1 | stepCountNext4HourLag1 | True | 0.924861 | 3735 | 35 | 19 | exchangeable |
| -0.012286 | 0.021887 | -0.561349 | 0.574560 | False | False | fourSC | 4hour_step_norm | 2 | yesterdayStepCount | True | 0.948113 | 3735 | 35 | 19 | exchangeable |
| 0.196592 | 0.042049 | 4.675258 | 2.935843e-06 | True | True | fourSC | 4hour_step_norm | 3 | stepCountLast7DaysEma | True | 1.005044 | 3735 | 35 | 19 | exchangeable |
| 0.154906 | 0.030582 | 5.065339 | 4.076737e-07 | True | True | fourSC | 4hour_step_norm | 4 | prior2HourStepCount | True | 0.941170 | 3735 | 35 | 19 | exchangeable |
| 0.002048 | 0.014808 | 0.138311 | 0.889995 | False | False | fourSC | 4hour_step_norm | 5 | activitySuggestionsSentLast7Days | True | 0.967404 | 3735 | 35 | 19 | exchangeable |
| 0.093453 | 0.063618 | 1.468967 | 0.141842 | False | False | fourSC | 4hour_step_norm | 6 | activitySuggestionInteractLast7Days | True | 0.347380 | 3735 | 35 | 19 | exchangeable |
| 0.013726 | 0.065967 | 0.208080 | 0.835166 | False | False | fourSC | 4hour_step_norm | 7 | activeDaysLast7Days | True | 0.277433 | 3735 | 35 | 19 | exchangeable |
| -0.030288 | 0.039681 | -0.763289 | 0.445291 | False | False | fourSC | 4hour_step_norm | 8 | isWeekend | True | 0.364457 | 3735 | 35 | 19 | exchangeable |
| -0.121325 | 0.061425 | -1.975175 | 0.048248 | True | False | fourSC | 4hour_step_norm | 9 | decisionTimeSlot | True | 0.498173 | 3735 | 35 | 19 | exchangeable |
| 0.024755 | 0.013354 | 1.853787 | 0.063770 | False | False | fourSC | 4hour_step_norm | 10 | perceivedUtilityLastWeek | True | 2.511183 | 3735 | 35 | 19 | exchangeable |
| 0.015867 | 0.030958 | 0.512519 | 0.608288 | False | False | fourSC | 4hour_step_norm | 11 | caeAverageLastWeek | True | 0.992053 | 3735 | 35 | 19 | exchangeable |
| -0.019863 | 0.044487 | -0.446479 | 0.655251 | False | False | fourSC | 4hour_step_norm | 12 | Ah | True | 0.499771 | 3735 | 35 | 19 | exchangeable |
| 0.020771 | 0.028846 | 0.720053 | 0.471492 | False | False | fourSC | 4hour_step_norm | 13 | Ah*yesterdayStepCount | True | 0.672639 | 3735 | 35 | 19 | exchangeable |
| -0.004910 | 0.030390 | -0.161557 | 0.871655 | False | False | fourSC | 4hour_step_norm | 14 | Ah*prior2HourStepCount | True | 0.666699 | 3735 | 35 | 19 | exchangeable |
| 0.012515 | 0.016986 | 0.736800 | 0.461244 | False | False | fourSC | 4hour_step_norm | 15 | Ah*activitySuggestionsSentLast7Days | True | 0.688627 | 3735 | 35 | 19 | exchangeable |
| 0.029014 | 0.071334 | 0.406732 | 0.684205 | False | False | fourSC | 4hour_step_norm | 16 | Ah*activitySuggestionInteractLast7Days | True | 0.349684 | 3735 | 35 | 19 | exchangeable |
| -0.000493 | 0.008540 | -0.057787 | 0.953918 | False | False | fourSC | 4hour_step_norm | 17 | Ah*perceivedUtilityLastWeek | True | 1.845777 | 3735 | 35 | 19 | exchangeable |
| 0.009354 | 0.020125 | 0.464801 | 0.642074 | False | False | fourSC | 4hour_step_norm | 18 | Ah*caeAverageLastWeek | True | 0.692755 | 3735 | 35 | 19 | exchangeable |
| 0.501409 |  |  |  | False | False | antic | anticipated_affect_norm | 0 | intercept | False | 0.000000 | 1019 | 35 | 15 | exchangeable |
| 0.045783 | 0.025361 | 1.805207 | 0.071042 | False | False | antic | anticipated_affect_norm | 1 | anticipated_affect_yesterday | True | 0.366210 | 1019 | 35 | 15 | exchangeable |
| 0.100711 | 0.035124 | 2.867290 | 0.004140 | True | True | antic | anticipated_affect_norm | 2 | active_status_fraction_7days | True | 0.229784 | 1019 | 35 | 15 | exchangeable |
| 0.000505 | 0.018306 | 0.027605 | 0.977977 | False | False | antic | anticipated_affect_norm | 3 | is_weekend | True | 0.366570 | 1019 | 35 | 15 | exchangeable |
| 0.001684 | 0.006206 | 0.271428 | 0.786062 | False | False | antic | anticipated_affect_norm | 4 | perceived_utility_lastweek | True | 2.460228 | 1019 | 35 | 15 | exchangeable |
| 0.148363 | 0.036773 | 4.034586 | 5.469854e-05 | True | True | antic | anticipated_affect_norm | 5 | CAE_avg_lastweek | True | 0.928551 | 1019 | 35 | 15 | exchangeable |
| 0.012700 | 0.005527 | 2.297914 | 0.021567 | True | False | antic | anticipated_affect_norm | 6 | recent_burden | True | 0.974256 | 1019 | 35 | 15 | exchangeable |
| 0.002719 | 0.008352 | 0.325595 | 0.744731 | False | False | antic | anticipated_affect_norm | 7 | A0_morning | True | 0.499217 | 1019 | 35 | 15 | exchangeable |
| 0.014462 | 0.009429 | 1.533840 | 0.125069 | False | False | antic | anticipated_affect_norm | 8 | A1_afternoon | True | 0.499873 | 1019 | 35 | 15 | exchangeable |
| 0.000268 | 0.002483 | 0.107924 | 0.914056 | False | False | antic | anticipated_affect_norm | 9 | A0_morning_by_perceived_utility_lastweek | True | 1.866335 | 1019 | 35 | 15 | exchangeable |
| -0.001966 | 0.003524 | -0.557702 | 0.577048 | False | False | antic | anticipated_affect_norm | 10 | A1_afternoon_by_perceived_utility_lastweek | True | 1.863397 | 1019 | 35 | 15 | exchangeable |
| 0.007835 | 0.009290 | 0.843394 | 0.399008 | False | False | antic | anticipated_affect_norm | 11 | A0_morning_by_CAE_avg_lastweek | True | 0.686065 | 1019 | 35 | 15 | exchangeable |
| -0.018761 | 0.011819 | -1.587312 | 0.112442 | False | False | antic | anticipated_affect_norm | 12 | A1_afternoon_by_CAE_avg_lastweek | True | 0.657745 | 1019 | 35 | 15 | exchangeable |
| -0.009965 | 0.010171 | -0.979782 | 0.327194 | False | False | antic | anticipated_affect_norm | 13 | A0_morning_by_recent_burden | True | 0.682435 | 1019 | 35 | 15 | exchangeable |
| -0.012420 | 0.010287 | -1.207322 | 0.227308 | False | False | antic | anticipated_affect_norm | 14 | A1_afternoon_by_recent_burden | True | 0.665827 | 1019 | 35 | 15 | exchangeable |
| -0.060492 |  |  |  | False | False | CAE | CAE_avg_norm | 0 | intercept | False | 0.000000 | 262 | 35 | 4 | exchangeable |
| 0.970574 | 0.027811 | 34.898388 | 7.865086e-267 | True | True | CAE | CAE_avg_norm | 1 | CAE_avg_lastweek | True | 0.981409 | 262 | 35 | 4 | exchangeable |
| -0.008924 | 0.022906 | -0.389597 | 0.696834 | False | False | CAE | CAE_avg_norm | 2 | fourSC_ewma | True | 0.527652 | 262 | 35 | 4 | exchangeable |
| 0.086312 | 0.076550 | 1.127520 | 0.259523 | False | False | CAE | CAE_avg_norm | 3 | anticipated_affect_ewma | True | 0.356150 | 262 | 35 | 4 | exchangeable |

## Pooled RL Q GEE Coefficients

Gaussian GEE on the final fitted-Q regression from pooled FQI (``phi_obs`` vs bootstrap targets), clustered by participant. ``identified=False`` marks structurally unused features with zero design variance (e.g. ``b_tilde``, masked day-6 mediators); SE / p-values are omitted for those rows. RL priors still use ridge-FQI for ``mu_0_micro``; all-zero design columns get ``UNIDENTIFIED_PRIOR_VAR`` instead of the ``MIN_SIGMA2`` floor.

| coefficient | robust_se | z_stat | p_value | significant_0.05 | significant_0.01 | model | outcome | index | feature | identified | feature_std | n_obs | n_clusters | n_features | working_correlation | fqi_iters | ridge_alpha | residual_sigma2 | block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.153216 |  |  |  | False | False | q_no_td_modify | fqi_target | 0 | intercept | False | 0.000000 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| -0.055675 | 0.029131 | -1.911229 | 0.055975 | False | False | q_no_td_modify | fqi_target | 1 | weekday_vs_weekend | True | 0.372678 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| -0.004289 | 0.011093 | -0.386641 | 0.699022 | False | False | q_no_td_modify | fqi_target | 2 | slot_pm | True | 0.500000 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| -0.001773 | 0.001150 | -1.542351 | 0.122988 | False | False | q_no_td_modify | fqi_target | 3 | E_w | True | 2.478283 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| 1.230461 | 0.005790 | 212.500630 | 0.000000 | True | True | q_no_td_modify | fqi_target | 4 | b_hat | True | 0.960113 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| -9.115192e-17 |  |  |  | False | False | q_no_td_modify | fqi_target | 5 | b_tilde | False | 0.000000 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| 0.204373 | 0.012311 | 16.601364 | 6.812472e-62 | True | True | q_no_td_modify | fqi_target | 6 | M_Y_anticipated_affect_ewma | True | 0.347580 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| 0.001835 | 0.001370 | 1.338966 | 0.180582 | False | False | q_no_td_modify | fqi_target | 7 | M_Y_fourSC_ewma | True | 0.936432 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| 0.001180 | 0.001699 | 0.694554 | 0.487335 | False | False | q_no_td_modify | fqi_target | 8 | M_E_pageview_ewma | True | 1.042579 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| -0.103805 | 0.004937 | -21.027691 | 3.660362e-98 | True | True | q_no_td_modify | fqi_target | 9 | M_E_fitbit_wear_ewma | True | 0.409227 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| -0.035058 | 0.005274 | -6.647883 | 2.973381e-11 | True | True | q_no_td_modify | fqi_target | 10 | M_E_survey_complete_ewma | True | 0.415096 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| 0.002794 | 0.004050 | 0.689959 | 0.490220 | False | False | q_no_td_modify | fqi_target | 11 | yesterday_step_count | True | 0.936481 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| -0.011203 | 0.004157 | -2.694953 | 0.007040 | True | True | q_no_td_modify | fqi_target | 12 | prior2hour_step_count | True | 0.937702 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| -0.035260 | 0.008541 | -4.128325 | 3.654149e-05 | True | True | q_no_td_modify | fqi_target | 13 | active_status_fraction_7days | True | 0.281500 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| 0.001329 | 0.003084 | 0.431149 | 0.666360 | False | False | q_no_td_modify | fqi_target | 14 | recent_burden | True | 0.941316 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| -0.056148 | 0.008125 | -6.910150 | 4.841429e-12 | True | True | q_no_td_modify | fqi_target | 15 | walk_interaction_7d | True | 0.349884 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| -0.008055 | 0.013169 | -0.611654 | 0.540767 | False | False | q_no_td_modify | fqi_target | 16 | A | True | 0.499646 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| -0.000372 | 0.001710 | -0.217359 | 0.827929 | False | False | q_no_td_modify | fqi_target | 17 | A*E_w | True | 1.824666 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| -0.005697 | 0.009387 | -0.606932 | 0.543896 | False | False | q_no_td_modify | fqi_target | 18 | A*b_hat | True | 0.678372 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| 3.308339e-18 |  |  |  | False | False | q_no_td_modify | fqi_target | 19 | A*b_tilde | False | 0.000000 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| -0.008694 | 0.006578 | -1.321675 | 0.186276 | False | False | q_no_td_modify | fqi_target | 20 | A*yesterday_step_count | True | 0.667043 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| 0.005161 | 0.004726 | 1.091852 | 0.274898 | False | False | q_no_td_modify | fqi_target | 21 | A*prior2hour_step_count | True | 0.664418 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| 0.012879 | 0.015339 | 0.839651 | 0.401104 | False | False | q_no_td_modify | fqi_target | 22 | A*active_status_fraction_7days | True | 0.387150 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| -0.002769 | 0.003814 | -0.725966 | 0.467860 | False | False | q_no_td_modify | fqi_target | 23 | A*recent_burden | True | 0.676751 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
| 0.003359 | 0.013216 | 0.254139 | 0.799388 | False | False | q_no_td_modify | fqi_target | 24 | A*walk_interaction_7d | True | 0.342632 | 4200 | 35 | 25 | exchangeable | 25 | 1.000000 | 0.024020 | beta |
