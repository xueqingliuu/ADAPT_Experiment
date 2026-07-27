# Prior Mean and Variance Summary

`prior_variance` is the diagonal entry of the prior covariance. For `q_td_modify_joint`, the full covariance remains in `rl_priors.json`; this table reports marginal variances.

## PF Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| PF | fourSC | theta | 0 | intercept | -0.131659 | 0.111724 |
| PF | fourSC | theta | 1 | yesterdayStepCount | 0.191097 | 0.037446 |
| PF | fourSC | theta | 2 | stepCountLast7DaysEma | 0.294415 | 0.044134 |
| PF | fourSC | theta | 3 | prior2HourStepCount | 0.222562 | 0.049141 |
| PF | fourSC | theta | 5 | activitySuggestionsSentLast7Days | -0.086075 | 0.028043 |
| PF | fourSC | theta | 6 | morningFitbitWearLast7Days | 0.071333 | 0.088006 |
| PF | fourSC | theta | 7 | salienceMessageSentYesterday | -0.013428 | 0.034389 |
| PF | fourSC | theta | 8 | activitySuggestionInteractLast7Days | 0.392326 | 0.071268 |
| PF | fourSC | theta | 9 | activeDaysLast7Days | 0.234786 | 0.073747 |
| PF | fourSC | theta | 10 | dayOfWeekNorm | -0.005288 | 0.048813 |
| PF | fourSC | theta | 11 | decisionTimeSlot | -0.188756 | 0.196322 |
| PF | fourSC | theta | 12 | perceivedUtilityLastWeek | -0.031225 | 0.062243 |
| PF | fourSC | theta | 13 | caeAverageLastWeek | -0.018646 | 0.041335 |
| PF | fourSC | theta | 14 | Ah | 0.057038 | 0.048703 |
| PF | fourSC | theta | 15 | Ah*yesterdayStepCount | -0.015600 | 0.031585 |
| PF | fourSC | theta | 16 | Ah*prior2HourStepCount | 0.011591 | 0.078955 |
| PF | fourSC | theta | 17 | Ah*activitySuggestionsSentLast7Days | 0.071797 | 0.030709 |
| PF | fourSC | theta | 18 | Ah*morningFitbitWearLast7Days | -0.178951 | 0.092588 |
| PF | fourSC | theta | 19 | Ah*salienceMessageSentYesterday | 0.058491 | 0.099085 |
| PF | fourSC | theta | 20 | Ah*activitySuggestionInteractLast7Days | 0.199530 | 0.057382 |
| PF | fourSC | theta | 21 | Ah*dayOfWeekNorm | 0.037550 | 0.069464 |
| PF | fourSC | theta | 22 | Ah*decisionTimeSlot | 0.003662 | 0.140221 |
| PF | fourSC | theta | 23 | Ah*perceivedUtilityLastWeek | 0.003270 | 0.032355 |
| PF | fourSC | theta | 24 | Ah*caeAverageLastWeek | 0.018902 | 0.042914 |
| PF | antic | theta | 0 | intercept | 0.576919 | 0.013276 |
| PF | antic | theta | 1 | dailyAnticipatedAffectYesterday | 0.164033 | 0.003687 |
| PF | antic | theta | 2 | todayStepCount | 0.007128 | 0.004573 |
| PF | antic | theta | 4 | activityStatusToday | 0.026028 | 0.008440 |
| PF | antic | theta | 5 | salienceMessageSentToday | -0.044461 | 0.002568 |
| PF | antic | theta | 6 | dayOfWeekNorm | -0.065572 | 0.003680 |
| PF | antic | theta | 7 | perceivedUtilityLastWeek | -0.005629 | 0.005001 |
| PF | antic | theta | 8 | caeAverageLastWeek | 0.243590 | 0.012484 |
| PF | antic | theta | 9 | ws_morning | 0.046889 | 0.004056 |
| PF | antic | theta | 10 | ws_afternoon | -0.041902 | 0.003288 |
| PF | antic | theta | 11 | ws_morning*salienceMessageSentToday | 0.001011 | 0.002167 |
| PF | antic | theta | 12 | ws_afternoon*salienceMessageSentToday | 0.039842 | 0.002199 |
| PF | antic | theta | 13 | ws_morning*dayOfWeekNorm | -0.001076 | 0.002085 |
| PF | antic | theta | 14 | ws_afternoon*dayOfWeekNorm | 0.032351 | 0.003299 |
| PF | antic | theta | 15 | ws_morning*perceivedUtilityLastWeek | -0.009619 | 0.001655 |
| PF | antic | theta | 16 | ws_afternoon*perceivedUtilityLastWeek | 0.001093 | 0.002597 |
| PF | antic | theta | 17 | ws_morning*caeAverageLastWeek | 0.002336 | 0.003763 |
| PF | antic | theta | 18 | ws_afternoon*caeAverageLastWeek | -0.023105 | 0.002437 |
| PF | CAE | theta | 0 | intercept | -0.406692 | 0.017661 |
| PF | CAE | theta | 1 | caeAverageLastWeek | 0.806559 | 0.033216 |
| PF | CAE | theta | 2 | week_norm | 0.002936 | 0.013933 |
| PF | CAE | theta | 3 | fourSC_slot_0 | -0.004979 | 0.006041 |
| PF | CAE | theta | 4 | fourSC_slot_1 | -0.016836 | 0.001067 |
| PF | CAE | theta | 5 | fourSC_slot_2 | -0.012104 | 0.006099 |
| PF | CAE | theta | 6 | fourSC_slot_3 | -0.003986 | 0.005774 |
| PF | CAE | theta | 7 | fourSC_slot_4 | -0.036306 | 0.002934 |
| PF | CAE | theta | 8 | fourSC_slot_5 | 0.025052 | 0.003477 |
| PF | CAE | theta | 9 | fourSC_slot_6 | 0.012014 | 0.002747 |
| PF | CAE | theta | 10 | fourSC_slot_7 | 0.027332 | 0.003179 |
| PF | CAE | theta | 11 | fourSC_slot_8 | -0.011972 | 0.002007 |
| PF | CAE | theta | 12 | fourSC_slot_9 | 0.006710 | 0.002868 |
| PF | CAE | theta | 13 | fourSC_slot_10 | 0.003332 | 0.002846 |
| PF | CAE | theta | 14 | fourSC_slot_11 | 0.018821 | 0.004846 |
| PF | CAE | theta | 15 | fourSC_slot_12 | -0.014065 | 0.004712 |
| PF | CAE | theta | 16 | fourSC_slot_13 | -0.043196 | 0.002485 |
| PF | CAE | theta | 17 | antic_day_0 | 0.136524 | 0.003478 |
| PF | CAE | theta | 18 | antic_day_1 | -0.214721 | 0.003843 |
| PF | CAE | theta | 19 | antic_day_2 | -0.045320 | 0.002743 |
| PF | CAE | theta | 20 | antic_day_3 | 0.264817 | 0.002472 |
| PF | CAE | theta | 21 | antic_day_4 | 0.275072 | 0.004611 |
| PF | CAE | theta | 22 | antic_day_5 | 0.054528 | 0.004016 |
| PF | CAE | theta | 23 | antic_day_6 | 0.170617 | 0.002606 |
| PF | CAE_short | theta | 0 | intercept | 0.007340 | 0.092943 |
| PF | CAE_short | theta | 1 | caeAverage | 0.970645 | 0.054155 |

## RL Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| RL | reward_shaping | eta | 0 | intercept | -0.022020 | 5.453467e-05 |
| RL | reward_shaping | eta | 1 | weekday_vs_weekend | -0.002697 | 1.000000e-06 |
| RL | reward_shaping | eta | 2 | slot_pm | -0.010692 | 1.285774e-05 |
| RL | reward_shaping | eta | 3 | E_w | 0.000120 | 2.108174e-05 |
| RL | reward_shaping | eta | 4 | weekday_vs_weekend*E_w | 1.469266e-05 | 1.000000e-06 |
| RL | reward_shaping | eta | 5 | slot_pm*E_w | 5.825656e-05 | 4.970480e-06 |
| RL | reward_shaping | eta | 6 | b_hat | 0.035670 | 0.000109 |
| RL | reward_shaping | eta | 7 | weekday_vs_weekend*b_hat | 0.004368 | 1.636128e-06 |
| RL | reward_shaping | eta | 8 | slot_pm*b_hat | 0.017320 | 2.572209e-05 |
| RL | reward_shaping | eta | 9 | b_tilde | 0.000000 | 1.000000e-06 |
| RL | reward_shaping | eta | 10 | M_Y_day1_fourSC_morning | 0.000928 | 1.441751e-05 |
| RL | reward_shaping | eta | 11 | M_Y_day1_fourSC_afternoon | -0.000857 | 7.270020e-06 |
| RL | reward_shaping | eta | 12 | M_Y_day1_anticipated_affect | 0.012856 | 2.263007e-05 |
| RL | reward_shaping | eta | 13 | M_Y_day2_fourSC_morning | -2.816514e-05 | 2.259355e-05 |
| RL | reward_shaping | eta | 14 | M_Y_day2_fourSC_afternoon | 1.060107e-05 | 1.555242e-05 |
| RL | reward_shaping | eta | 15 | M_Y_day2_anticipated_affect | -0.010313 | 2.467431e-05 |
| RL | reward_shaping | eta | 16 | M_Y_day3_fourSC_morning | -0.001031 | 1.644660e-05 |
| RL | reward_shaping | eta | 17 | M_Y_day3_fourSC_afternoon | 0.002057 | 5.858065e-06 |
| RL | reward_shaping | eta | 18 | M_Y_day3_anticipated_affect | -0.012207 | 3.764841e-06 |
| RL | reward_shaping | eta | 19 | M_Y_day4_fourSC_morning | 0.003966 | 1.385541e-05 |
| RL | reward_shaping | eta | 20 | M_Y_day4_fourSC_afternoon | 0.004327 | 8.348649e-06 |
| RL | reward_shaping | eta | 21 | M_Y_day4_anticipated_affect | 0.101678 | 2.395578e-06 |
| RL | reward_shaping | eta | 22 | M_Y_day5_fourSC_morning | -0.002955 | 3.274450e-06 |
| RL | reward_shaping | eta | 23 | M_Y_day5_fourSC_afternoon | -0.002906 | 2.348965e-06 |
| RL | reward_shaping | eta | 24 | M_Y_day5_anticipated_affect | 0.062629 | 1.000000e-06 |
| RL | reward_shaping | eta | 25 | M_Y_day6_fourSC_morning | -0.002400 | 1.465696e-05 |
| RL | reward_shaping | eta | 26 | M_Y_day6_fourSC_afternoon | -0.012788 | 1.000000e-06 |
| RL | reward_shaping | eta | 27 | M_Y_day6_anticipated_affect | 0.094653 | 1.000000e-06 |
| RL | reward_shaping | eta | 28 | M_E_day1_pageview_morning | -0.000153 | 1.032824e-05 |
| RL | reward_shaping | eta | 29 | M_E_day1_pageview_afternoon | -0.000410 | 2.695009e-05 |
| RL | reward_shaping | eta | 30 | M_E_day1_morning_fitbit_wear | -0.000836 | 1.681055e-05 |
| RL | reward_shaping | eta | 31 | M_E_day1_daily_survey_complete | 0.003520 | 0.000101 |
| RL | reward_shaping | eta | 32 | M_E_day2_pageview_morning | 0.002139 | 1.538955e-05 |
| RL | reward_shaping | eta | 33 | M_E_day2_pageview_afternoon | 0.001281 | 2.903072e-05 |
| RL | reward_shaping | eta | 34 | M_E_day2_morning_fitbit_wear | 0.000310 | 1.133212e-05 |
| RL | reward_shaping | eta | 35 | M_E_day2_daily_survey_complete | -0.000177 | 3.189922e-05 |
| RL | reward_shaping | eta | 36 | M_E_day3_pageview_morning | 0.001177 | 8.561118e-06 |
| RL | reward_shaping | eta | 37 | M_E_day3_pageview_afternoon | -0.002410 | 2.507029e-05 |
| RL | reward_shaping | eta | 38 | M_E_day3_morning_fitbit_wear | 0.001223 | 7.192961e-06 |
| RL | reward_shaping | eta | 39 | M_E_day3_daily_survey_complete | -0.006232 | 9.358174e-06 |
| RL | reward_shaping | eta | 40 | M_E_day4_pageview_morning | 0.001381 | 3.477508e-06 |
| RL | reward_shaping | eta | 41 | M_E_day4_pageview_afternoon | 0.007344 | 2.754137e-05 |
| RL | reward_shaping | eta | 42 | M_E_day4_morning_fitbit_wear | -0.012967 | 1.388470e-05 |
| RL | reward_shaping | eta | 43 | M_E_day4_daily_survey_complete | -0.007687 | 4.059668e-06 |
| RL | reward_shaping | eta | 44 | M_E_day5_pageview_morning | -0.001679 | 6.670694e-06 |
| RL | reward_shaping | eta | 45 | M_E_day5_pageview_afternoon | -1.027396e-05 | 8.877195e-06 |
| RL | reward_shaping | eta | 46 | M_E_day5_morning_fitbit_wear | 0.009864 | 5.425800e-06 |
| RL | reward_shaping | eta | 47 | M_E_day5_daily_survey_complete | -0.001364 | 1.675531e-05 |
| RL | reward_shaping | eta | 48 | M_E_day6_pageview_morning | -0.006146 | 2.739861e-06 |
| RL | reward_shaping | eta | 49 | M_E_day6_pageview_afternoon | -0.030693 | 1.000000e-06 |
| RL | reward_shaping | eta | 50 | M_E_day6_morning_fitbit_wear | -0.012475 | 1.000000e-06 |
| RL | reward_shaping | eta | 51 | M_E_day6_daily_survey_complete | 0.047536 | 1.000000e-06 |
| RL | reward_shaping | eta | 52 | yesterday_step_count | -0.005882 | 2.092755e-05 |
| RL | reward_shaping | eta | 53 | ema_step_count | 0.003667 | 4.913628e-05 |
| RL | reward_shaping | eta | 54 | prior2hour_step_count | -0.002664 | 1.031427e-05 |
| RL | reward_shaping | eta | 56 | active_status_fraction_7days | -0.004486 | 3.485054e-05 |
| RL | reward_shaping | eta | 57 | recent_burden | 0.000305 | 1.992214e-05 |
| RL | reward_shaping | eta | 58 | salience_yesterday | 0.003675 | 2.941391e-05 |
| RL | reward_shaping | eta | 59 | walk_interaction_7d | -0.007257 | 3.465680e-06 |
| RL | reward_shaping | eta | 60 | reserved_context_zero | 0.000000 | 1.000000e-06 |
| RL | q_no_td_modify | beta | 0 | intercept | 0.130004 | 0.028085 |
| RL | q_no_td_modify | beta | 1 | weekday_vs_weekend | -0.362518 | 0.001844 |
| RL | q_no_td_modify | beta | 2 | slot_pm | 0.020715 | 0.001711 |
| RL | q_no_td_modify | beta | 3 | E_w | 0.025310 | 0.007729 |
| RL | q_no_td_modify | beta | 4 | weekday_vs_weekend*E_w | 0.009325 | 0.001512 |
| RL | q_no_td_modify | beta | 5 | slot_pm*E_w | -0.000215 | 0.000174 |
| RL | q_no_td_modify | beta | 6 | b_hat | 0.780360 | 0.027295 |
| RL | q_no_td_modify | beta | 7 | weekday_vs_weekend*b_hat | 0.412250 | 0.004551 |
| RL | q_no_td_modify | beta | 8 | slot_pm*b_hat | 0.097196 | 0.000651 |
| RL | q_no_td_modify | beta | 9 | b_tilde | 0.000000 | 1.000000e-06 |
| RL | q_no_td_modify | beta | 10 | M_Y_day1_fourSC_morning | -0.013956 | 0.001531 |
| RL | q_no_td_modify | beta | 11 | M_Y_day1_fourSC_afternoon | 0.000701 | 0.001026 |
| RL | q_no_td_modify | beta | 12 | M_Y_day1_anticipated_affect | 0.153716 | 0.005083 |
| RL | q_no_td_modify | beta | 13 | M_Y_day2_fourSC_morning | 0.021192 | 0.001228 |
| RL | q_no_td_modify | beta | 14 | M_Y_day2_fourSC_afternoon | -0.003004 | 0.002150 |
| RL | q_no_td_modify | beta | 15 | M_Y_day2_anticipated_affect | 0.231390 | 0.003837 |
| RL | q_no_td_modify | beta | 16 | M_Y_day3_fourSC_morning | 0.006615 | 0.000810 |
| RL | q_no_td_modify | beta | 17 | M_Y_day3_fourSC_afternoon | 0.034603 | 0.001429 |
| RL | q_no_td_modify | beta | 18 | M_Y_day3_anticipated_affect | 0.242772 | 0.004796 |
| RL | q_no_td_modify | beta | 19 | M_Y_day4_fourSC_morning | 0.027301 | 0.001865 |
| RL | q_no_td_modify | beta | 20 | M_Y_day4_fourSC_afternoon | 0.021343 | 0.008197 |
| RL | q_no_td_modify | beta | 21 | M_Y_day4_anticipated_affect | 0.339733 | 0.004941 |
| RL | q_no_td_modify | beta | 22 | M_Y_day5_fourSC_morning | -0.005326 | 0.002519 |
| RL | q_no_td_modify | beta | 23 | M_Y_day5_fourSC_afternoon | -0.064163 | 0.000764 |
| RL | q_no_td_modify | beta | 24 | M_Y_day5_anticipated_affect | 0.202651 | 0.001690 |
| RL | q_no_td_modify | beta | 25 | M_Y_day6_fourSC_morning | -0.014506 | 0.002248 |
| RL | q_no_td_modify | beta | 26 | M_Y_day6_fourSC_afternoon | 0.000000 | 1.000000e-06 |
| RL | q_no_td_modify | beta | 27 | M_Y_day6_anticipated_affect | 0.000000 | 1.000000e-06 |
| RL | q_no_td_modify | beta | 28 | M_E_day1_pageview_morning | -0.007012 | 0.000947 |
| RL | q_no_td_modify | beta | 29 | M_E_day1_pageview_afternoon | -0.006522 | 0.003185 |
| RL | q_no_td_modify | beta | 30 | M_E_day1_morning_fitbit_wear | -0.069811 | 0.006173 |
| RL | q_no_td_modify | beta | 31 | M_E_day1_daily_survey_complete | -0.008787 | 0.006213 |
| RL | q_no_td_modify | beta | 32 | M_E_day2_pageview_morning | 0.049114 | 0.001241 |
| RL | q_no_td_modify | beta | 33 | M_E_day2_pageview_afternoon | 0.010486 | 0.003150 |
| RL | q_no_td_modify | beta | 34 | M_E_day2_morning_fitbit_wear | -0.114277 | 0.007493 |
| RL | q_no_td_modify | beta | 35 | M_E_day2_daily_survey_complete | -0.112927 | 0.002736 |
| RL | q_no_td_modify | beta | 36 | M_E_day3_pageview_morning | 0.059234 | 0.001457 |
| RL | q_no_td_modify | beta | 37 | M_E_day3_pageview_afternoon | -0.020868 | 0.004586 |
| RL | q_no_td_modify | beta | 38 | M_E_day3_morning_fitbit_wear | -0.079408 | 0.008257 |
| RL | q_no_td_modify | beta | 39 | M_E_day3_daily_survey_complete | -0.152134 | 0.002748 |
| RL | q_no_td_modify | beta | 40 | M_E_day4_pageview_morning | 0.085551 | 0.001500 |
| RL | q_no_td_modify | beta | 41 | M_E_day4_pageview_afternoon | 0.089038 | 0.004032 |
| RL | q_no_td_modify | beta | 42 | M_E_day4_morning_fitbit_wear | -0.210389 | 0.010960 |
| RL | q_no_td_modify | beta | 43 | M_E_day4_daily_survey_complete | -0.206485 | 0.004125 |
| RL | q_no_td_modify | beta | 44 | M_E_day5_pageview_morning | 0.049412 | 0.004237 |
| RL | q_no_td_modify | beta | 45 | M_E_day5_pageview_afternoon | -0.038623 | 0.000433 |
| RL | q_no_td_modify | beta | 46 | M_E_day5_morning_fitbit_wear | 0.173185 | 0.000862 |
| RL | q_no_td_modify | beta | 47 | M_E_day5_daily_survey_complete | 0.103065 | 0.001509 |
| RL | q_no_td_modify | beta | 48 | M_E_day6_pageview_morning | -0.029344 | 0.002127 |
| RL | q_no_td_modify | beta | 49 | M_E_day6_pageview_afternoon | 0.000000 | 1.000000e-06 |
| RL | q_no_td_modify | beta | 50 | M_E_day6_morning_fitbit_wear | 0.000000 | 1.000000e-06 |
| RL | q_no_td_modify | beta | 51 | M_E_day6_daily_survey_complete | 0.000000 | 1.000000e-06 |
| RL | q_no_td_modify | beta | 52 | yesterday_step_count | -0.004644 | 0.000964 |
| RL | q_no_td_modify | beta | 53 | ema_step_count | 0.006759 | 0.000492 |
| RL | q_no_td_modify | beta | 54 | prior2hour_step_count | -0.008797 | 0.000112 |
| RL | q_no_td_modify | beta | 56 | active_status_fraction_7days | -0.089906 | 0.009830 |
| RL | q_no_td_modify | beta | 57 | recent_burden | 0.009091 | 0.000578 |
| RL | q_no_td_modify | beta | 58 | salience_yesterday | -0.011360 | 0.000442 |
| RL | q_no_td_modify | beta | 59 | walk_interaction_7d | -0.009682 | 0.003823 |
| RL | q_no_td_modify | beta | 60 | reserved_context_zero | 0.000000 | 1.000000e-06 |
| RL | q_no_td_modify | beta | 61 | A | 0.009949 | 0.000636 |
| RL | q_no_td_modify | beta | 62 | A*E_w | 0.009301 | 0.000304 |
| RL | q_no_td_modify | beta | 63 | A*b_hat | -0.002151 | 0.000444 |
| RL | q_no_td_modify | beta | 64 | A*weekday_vs_weekend | -0.020456 | 0.000327 |
| RL | q_no_td_modify | beta | 65 | A*slot_pm | -0.015246 | 0.000463 |
| RL | q_no_td_modify | beta | 66 | A*weekday_vs_weekend*E_w | -0.035841 | 0.001629 |
| RL | q_no_td_modify | beta | 67 | A*slot_pm*E_w | -0.006462 | 0.000323 |
| RL | q_no_td_modify | beta | 68 | A*weekday_vs_weekend*b_hat | 0.033307 | 0.000592 |
| RL | q_no_td_modify | beta | 69 | A*slot_pm*b_hat | 0.000592 | 0.001016 |
| RL | q_no_td_modify | beta | 70 | A*yesterday_step_count | -0.005680 | 0.000412 |
| RL | q_no_td_modify | beta | 71 | A*ema_step_count | -0.000182 | 0.000225 |
| RL | q_no_td_modify | beta | 72 | A*prior2hour_step_count | 0.015661 | 0.000360 |
| RL | q_no_td_modify | beta | 74 | A*active_status_fraction_7days | -0.029465 | 0.001637 |
| RL | q_no_td_modify | beta | 75 | A*recent_burden | -7.054652e-05 | 0.000175 |
| RL | q_no_td_modify | beta | 76 | A*salience_yesterday | 0.035974 | 0.000675 |
| RL | q_no_td_modify | beta | 77 | A*walk_interaction_7d | 0.024040 | 0.001709 |
| RL | q_no_td_modify | beta | 78 | A*reserved_context_zero | 0.000000 | 1.000000e-06 |

## Joint Modified-TD Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| Joint Modified TD | q_td_modify_joint | eta | 0 | eta_intercept | 0.243988 | 0.027453 |
| Joint Modified TD | q_td_modify_joint | eta | 1 | eta_E_w | 0.010654 | 0.015867 |
| Joint Modified TD | q_td_modify_joint | eta | 2 | eta_b_hat | 0.401266 | 0.033938 |
| Joint Modified TD | q_td_modify_joint | eta | 3 | eta_b_tilde | 0.000000 | 1.000000e-06 |
| Joint Modified TD | q_td_modify_joint | beta | 4 | beta_intercept | 0.197915 | 0.021974 |
| Joint Modified TD | q_td_modify_joint | beta | 5 | beta_weekday_vs_weekend | -0.259142 | 0.001459 |
| Joint Modified TD | q_td_modify_joint | beta | 6 | beta_slot_pm | 0.034320 | 0.001467 |
| Joint Modified TD | q_td_modify_joint | beta | 7 | beta_E_w | 0.011578 | 0.010299 |
| Joint Modified TD | q_td_modify_joint | beta | 8 | beta_weekday_vs_weekend*E_w | 0.006196 | 0.001101 |
| Joint Modified TD | q_td_modify_joint | beta | 9 | beta_slot_pm*E_w | -0.002041 | 0.000172 |
| Joint Modified TD | q_td_modify_joint | beta | 10 | beta_b_hat | 0.531722 | 0.022869 |
| Joint Modified TD | q_td_modify_joint | beta | 11 | beta_weekday_vs_weekend*b_hat | 0.382469 | 0.002954 |
| Joint Modified TD | q_td_modify_joint | beta | 12 | beta_slot_pm*b_hat | 0.060032 | 0.000594 |
| Joint Modified TD | q_td_modify_joint | beta | 13 | beta_b_tilde | 0.000000 | 1.000000e-06 |
| Joint Modified TD | q_td_modify_joint | beta | 14 | beta_M_Y_day1_fourSC_morning | -0.001203 | 0.002017 |
| Joint Modified TD | q_td_modify_joint | beta | 15 | beta_M_Y_day1_fourSC_afternoon | 0.002739 | 0.001054 |
| Joint Modified TD | q_td_modify_joint | beta | 16 | beta_M_Y_day1_anticipated_affect | 0.204505 | 0.004566 |
| Joint Modified TD | q_td_modify_joint | beta | 17 | beta_M_Y_day2_fourSC_morning | 0.029077 | 0.000966 |
| Joint Modified TD | q_td_modify_joint | beta | 18 | beta_M_Y_day2_fourSC_afternoon | -0.007465 | 0.002176 |
| Joint Modified TD | q_td_modify_joint | beta | 19 | beta_M_Y_day2_anticipated_affect | 0.227993 | 0.003537 |
| Joint Modified TD | q_td_modify_joint | beta | 20 | beta_M_Y_day3_fourSC_morning | 0.003681 | 0.000651 |
| Joint Modified TD | q_td_modify_joint | beta | 21 | beta_M_Y_day3_fourSC_afternoon | 0.032498 | 0.001189 |
| Joint Modified TD | q_td_modify_joint | beta | 22 | beta_M_Y_day3_anticipated_affect | 0.226038 | 0.004130 |
| Joint Modified TD | q_td_modify_joint | beta | 23 | beta_M_Y_day4_fourSC_morning | 0.021973 | 0.001516 |
| Joint Modified TD | q_td_modify_joint | beta | 24 | beta_M_Y_day4_fourSC_afternoon | 0.020326 | 0.005192 |
| Joint Modified TD | q_td_modify_joint | beta | 25 | beta_M_Y_day4_anticipated_affect | 0.309063 | 0.004502 |
| Joint Modified TD | q_td_modify_joint | beta | 26 | beta_M_Y_day5_fourSC_morning | -0.009668 | 0.001831 |
| Joint Modified TD | q_td_modify_joint | beta | 27 | beta_M_Y_day5_fourSC_afternoon | -0.054959 | 0.000616 |
| Joint Modified TD | q_td_modify_joint | beta | 28 | beta_M_Y_day5_anticipated_affect | 0.093796 | 0.001019 |
| Joint Modified TD | q_td_modify_joint | beta | 29 | beta_M_Y_day6_fourSC_morning | -0.012980 | 0.001559 |
| Joint Modified TD | q_td_modify_joint | beta | 30 | beta_M_Y_day6_fourSC_afternoon | 0.000000 | 1.000000e-06 |
| Joint Modified TD | q_td_modify_joint | beta | 31 | beta_M_Y_day6_anticipated_affect | 0.000000 | 1.000000e-06 |
| Joint Modified TD | q_td_modify_joint | beta | 32 | beta_M_E_day1_pageview_morning | 0.001925 | 0.002012 |
| Joint Modified TD | q_td_modify_joint | beta | 33 | beta_M_E_day1_pageview_afternoon | -0.004868 | 0.002793 |
| Joint Modified TD | q_td_modify_joint | beta | 34 | beta_M_E_day1_morning_fitbit_wear | -0.099590 | 0.006204 |
| Joint Modified TD | q_td_modify_joint | beta | 35 | beta_M_E_day1_daily_survey_complete | -0.032203 | 0.005621 |
| Joint Modified TD | q_td_modify_joint | beta | 36 | beta_M_E_day2_pageview_morning | 0.048632 | 0.001338 |
| Joint Modified TD | q_td_modify_joint | beta | 37 | beta_M_E_day2_pageview_afternoon | 0.008215 | 0.002877 |
| Joint Modified TD | q_td_modify_joint | beta | 38 | beta_M_E_day2_morning_fitbit_wear | -0.114068 | 0.007213 |
| Joint Modified TD | q_td_modify_joint | beta | 39 | beta_M_E_day2_daily_survey_complete | -0.118107 | 0.002871 |
| Joint Modified TD | q_td_modify_joint | beta | 40 | beta_M_E_day3_pageview_morning | 0.057346 | 0.000924 |
| Joint Modified TD | q_td_modify_joint | beta | 41 | beta_M_E_day3_pageview_afternoon | -0.020614 | 0.004258 |
| Joint Modified TD | q_td_modify_joint | beta | 42 | beta_M_E_day3_morning_fitbit_wear | -0.068442 | 0.007796 |
| Joint Modified TD | q_td_modify_joint | beta | 43 | beta_M_E_day3_daily_survey_complete | -0.142358 | 0.002938 |
| Joint Modified TD | q_td_modify_joint | beta | 44 | beta_M_E_day4_pageview_morning | 0.085259 | 0.001432 |
| Joint Modified TD | q_td_modify_joint | beta | 45 | beta_M_E_day4_pageview_afternoon | 0.077125 | 0.003791 |
| Joint Modified TD | q_td_modify_joint | beta | 46 | beta_M_E_day4_morning_fitbit_wear | -0.186321 | 0.010488 |
| Joint Modified TD | q_td_modify_joint | beta | 47 | beta_M_E_day4_daily_survey_complete | -0.189498 | 0.004253 |
| Joint Modified TD | q_td_modify_joint | beta | 48 | beta_M_E_day5_pageview_morning | 0.048673 | 0.003243 |
| Joint Modified TD | q_td_modify_joint | beta | 49 | beta_M_E_day5_pageview_afternoon | -0.036425 | 0.000405 |
| Joint Modified TD | q_td_modify_joint | beta | 50 | beta_M_E_day5_morning_fitbit_wear | 0.171931 | 0.000827 |
| Joint Modified TD | q_td_modify_joint | beta | 51 | beta_M_E_day5_daily_survey_complete | 0.097991 | 0.001454 |
| Joint Modified TD | q_td_modify_joint | beta | 52 | beta_M_E_day6_pageview_morning | -0.025607 | 0.001583 |
| Joint Modified TD | q_td_modify_joint | beta | 53 | beta_M_E_day6_pageview_afternoon | 0.000000 | 1.000000e-06 |
| Joint Modified TD | q_td_modify_joint | beta | 54 | beta_M_E_day6_morning_fitbit_wear | 0.000000 | 1.000000e-06 |
| Joint Modified TD | q_td_modify_joint | beta | 55 | beta_M_E_day6_daily_survey_complete | 0.000000 | 1.000000e-06 |
| Joint Modified TD | q_td_modify_joint | beta | 56 | beta_yesterday_step_count | 0.001690 | 0.000789 |
| Joint Modified TD | q_td_modify_joint | beta | 57 | beta_ema_step_count | 0.005903 | 0.000360 |
| Joint Modified TD | q_td_modify_joint | beta | 58 | beta_prior2hour_step_count | -0.007122 | 9.910073e-05 |
| Joint Modified TD | q_td_modify_joint | beta | 60 | beta_active_status_fraction_7days | -0.035701 | 0.006288 |
| Joint Modified TD | q_td_modify_joint | beta | 61 | beta_recent_burden | 0.004313 | 0.000204 |
| Joint Modified TD | q_td_modify_joint | beta | 62 | beta_salience_yesterday | -0.006322 | 0.000367 |
| Joint Modified TD | q_td_modify_joint | beta | 63 | beta_walk_interaction_7d | -0.003174 | 0.001653 |
| Joint Modified TD | q_td_modify_joint | beta | 64 | beta_reserved_context_zero | 0.000000 | 1.000000e-06 |
| Joint Modified TD | q_td_modify_joint | beta | 65 | beta_A | 0.028061 | 0.000505 |
| Joint Modified TD | q_td_modify_joint | beta | 66 | beta_A*E_w | 0.004806 | 0.000439 |
| Joint Modified TD | q_td_modify_joint | beta | 67 | beta_A*b_hat | -0.040710 | 0.000778 |
| Joint Modified TD | q_td_modify_joint | beta | 68 | beta_A*weekday_vs_weekend | -0.027532 | 0.000300 |
| Joint Modified TD | q_td_modify_joint | beta | 69 | beta_A*slot_pm | -0.037255 | 0.000383 |
| Joint Modified TD | q_td_modify_joint | beta | 70 | beta_A*weekday_vs_weekend*E_w | -0.029983 | 0.001301 |
| Joint Modified TD | q_td_modify_joint | beta | 71 | beta_A*slot_pm*E_w | -0.002386 | 0.000547 |
| Joint Modified TD | q_td_modify_joint | beta | 72 | beta_A*weekday_vs_weekend*b_hat | 0.050839 | 0.000455 |
| Joint Modified TD | q_td_modify_joint | beta | 73 | beta_A*slot_pm*b_hat | 0.037511 | 0.001350 |
| Joint Modified TD | q_td_modify_joint | beta | 74 | beta_A*yesterday_step_count | -0.003256 | 0.000346 |
| Joint Modified TD | q_td_modify_joint | beta | 75 | beta_A*ema_step_count | 0.003165 | 0.000199 |
| Joint Modified TD | q_td_modify_joint | beta | 76 | beta_A*prior2hour_step_count | 0.012595 | 0.000302 |
| Joint Modified TD | q_td_modify_joint | beta | 78 | beta_A*active_status_fraction_7days | -0.022308 | 0.000619 |
| Joint Modified TD | q_td_modify_joint | beta | 79 | beta_A*recent_burden | -0.003058 | 0.000178 |
| Joint Modified TD | q_td_modify_joint | beta | 80 | beta_A*salience_yesterday | 0.027186 | 0.000593 |
| Joint Modified TD | q_td_modify_joint | beta | 81 | beta_A*walk_interaction_7d | 0.030249 | 0.000630 |
| Joint Modified TD | q_td_modify_joint | beta | 82 | beta_A*reserved_context_zero | 0.000000 | 1.000000e-06 |
