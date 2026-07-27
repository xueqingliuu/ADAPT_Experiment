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
| PF | CAE | theta | 0 | intercept | -0.406692 | 0.017661 |
| PF | CAE | theta | 1 | caeAverageLastWeek | 0.806559 | 0.033216 |
| PF | CAE | theta | 2 | week_norm | 0.002936 | 0.013933 |
| PF | CAE | theta | 3 | fourSC_slot_0 | -0.004979 | 0.006041 |
| PF | CAE | theta | 4 | fourSC_slot_1 | -0.016836 | 0.001067 |
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
| RL | q_no_td_modify | beta | 61 | A*E_w | 0.009949 | 0.000636 |
| RL | q_no_td_modify | beta | 62 | A*b_hat | 0.009301 | 0.000304 |
| RL | q_no_td_modify | beta | 63 | A*weekday_vs_weekend | -0.002151 | 0.000444 |
| RL | q_no_td_modify | beta | 64 | A*slot_pm | -0.020456 | 0.000327 |
| RL | q_no_td_modify | beta | 65 | A*weekday_vs_weekend*E_w | -0.015246 | 0.000463 |
| RL | q_no_td_modify | beta | 66 | A*slot_pm*E_w | -0.035841 | 0.001629 |
| RL | q_no_td_modify | beta | 67 | A*weekday_vs_weekend*b_hat | -0.006462 | 0.000323 |
| RL | q_no_td_modify | beta | 68 | A*slot_pm*b_hat | 0.033307 | 0.000592 |
| RL | q_no_td_modify | beta | 69 | A*yesterday_step_count | 0.000592 | 0.001016 |
| RL | q_no_td_modify | beta | 70 | A*ema_step_count | -0.005680 | 0.000412 |
| RL | q_no_td_modify | beta | 71 | A*prior2hour_step_count | -0.000182 | 0.000225 |
| RL | q_no_td_modify | beta | 73 | A*active_status_fraction_7days | -0.004788 | 0.000355 |
| RL | q_no_td_modify | beta | 74 | A*recent_burden | -0.029465 | 0.001637 |
| RL | q_no_td_modify | beta | 75 | A*salience_yesterday | -7.054652e-05 | 0.000175 |
| RL | q_no_td_modify | beta | 76 | A*walk_interaction_7d | 0.035974 | 0.000675 |
| RL | q_no_td_modify | beta | 77 | q_no_td_modify_coef_77 | 0.024040 | 0.001709 |

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
| Joint Modified TD | q_td_modify_joint | beta | 65 | beta_A*E_w | 0.028061 | 0.000505 |
| Joint Modified TD | q_td_modify_joint | beta | 66 | beta_A*b_hat | 0.004806 | 0.000439 |
| Joint Modified TD | q_td_modify_joint | beta | 67 | beta_A*weekday_vs_weekend | -0.040710 | 0.000778 |
| Joint Modified TD | q_td_modify_joint | beta | 68 | beta_A*slot_pm | -0.027532 | 0.000300 |
| Joint Modified TD | q_td_modify_joint | beta | 69 | beta_A*weekday_vs_weekend*E_w | -0.037255 | 0.000383 |
| Joint Modified TD | q_td_modify_joint | beta | 70 | beta_A*slot_pm*E_w | -0.029983 | 0.001301 |
| Joint Modified TD | q_td_modify_joint | beta | 71 | beta_A*weekday_vs_weekend*b_hat | -0.002386 | 0.000547 |
| Joint Modified TD | q_td_modify_joint | beta | 72 | beta_A*slot_pm*b_hat | 0.050839 | 0.000455 |
| Joint Modified TD | q_td_modify_joint | beta | 73 | beta_A*yesterday_step_count | 0.037511 | 0.001350 |
| Joint Modified TD | q_td_modify_joint | beta | 74 | beta_A*ema_step_count | -0.003256 | 0.000346 |
| Joint Modified TD | q_td_modify_joint | beta | 75 | beta_A*prior2hour_step_count | 0.003165 | 0.000199 |
| Joint Modified TD | q_td_modify_joint | beta | 77 | beta_A*active_status_fraction_7days | 0.001501 | 0.000275 |
| Joint Modified TD | q_td_modify_joint | beta | 78 | beta_A*recent_burden | -0.022308 | 0.000619 |
| Joint Modified TD | q_td_modify_joint | beta | 79 | beta_A*salience_yesterday | -0.003058 | 0.000178 |
| Joint Modified TD | q_td_modify_joint | beta | 80 | beta_A*walk_interaction_7d | 0.027186 | 0.000593 |
| Joint Modified TD | q_td_modify_joint | beta | 81 | beta_beta_coef_77 | 0.030249 | 0.000630 |

## Pooled PF Regression Coefficients

Coefficients are from the all-user stacked ridge fits used as PF prior means. `p_value` uses ridge sandwich standard errors and is approximate because ridge shrinks coefficients.

| model | outcome | index | feature | coefficient | std_error | t_stat | p_value | significant_0.05 | significant_0.01 | n_obs | n_features | ridge_alpha | residual_sigma2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fourSC | 4hour_step_norm | 0 | intercept | -0.131659 | 0.092865 | -1.417747 | 0.156397 | False | False | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 1 | yesterdayStepCount | 0.191097 | 0.027758 | 6.884292 | 7.415007e-12 | True | True | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 2 | stepCountLast7DaysEma | 0.294415 | 0.020476 | 14.378785 | 5.269104e-45 | True | True | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 3 | prior2HourStepCount | 0.222562 | 0.025622 | 8.686480 | 6.849065e-18 | True | True | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 5 | activitySuggestionsSentLast7Days | -0.086075 | 0.027629 | -3.115387 | 0.001859 | True | True | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 6 | morningFitbitWearLast7Days | 0.071333 | 0.083538 | 0.853896 | 0.393249 | False | False | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 7 | salienceMessageSentYesterday | -0.013428 | 0.047292 | -0.283938 | 0.776483 | False | False | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 8 | activitySuggestionInteractLast7Days | 0.392326 | 0.134852 | 2.909308 | 0.003656 | True | True | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 9 | activeDaysLast7Days | 0.234786 | 0.059357 | 3.955492 | 7.862603e-05 | True | True | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 10 | dayOfWeekNorm | -0.005288 | 0.041164 | -0.128463 | 0.897794 | False | False | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 11 | decisionTimeSlot | -0.188756 | 0.048991 | -3.852901 | 0.000120 | True | True | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 12 | perceivedUtilityLastWeek | -0.031225 | 0.019961 | -1.564316 | 0.117877 | False | False | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 13 | caeAverageLastWeek | -0.018646 | 0.022707 | -0.821158 | 0.411639 | False | False | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 14 | Ah | 0.057038 | 0.122695 | 0.464872 | 0.642066 | False | False | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 15 | Ah*yesterdayStepCount | -0.015600 | 0.037188 | -0.419494 | 0.674894 | False | False | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 16 | Ah*prior2HourStepCount | 0.011591 | 0.036607 | 0.316645 | 0.751541 | False | False | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 17 | Ah*activitySuggestionsSentLast7Days | 0.071797 | 0.039614 | 1.812431 | 0.070046 | False | False | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 18 | Ah*morningFitbitWearLast7Days | -0.178951 | 0.115167 | -1.553841 | 0.120356 | False | False | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 19 | Ah*salienceMessageSentYesterday | 0.058491 | 0.067160 | 0.870928 | 0.383882 | False | False | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 20 | Ah*activitySuggestionInteractLast7Days | 0.199530 | 0.185775 | 1.074045 | 0.282912 | False | False | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 21 | Ah*dayOfWeekNorm | 0.037550 | 0.058491 | 0.641976 | 0.520951 | False | False | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 22 | Ah*decisionTimeSlot | 0.003662 | 0.067341 | 0.054381 | 0.956636 | False | False | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 23 | Ah*perceivedUtilityLastWeek | 0.003270 | 0.026944 | 0.121344 | 0.903429 | False | False | 2388 | 25 | 1.000000 | 0.665714 |
| fourSC | 4hour_step_norm | 24 | Ah*caeAverageLastWeek | 0.018902 | 0.032978 | 0.573168 | 0.566585 | False | False | 2388 | 25 | 1.000000 | 0.665714 |
| antic | anticipated_affect_norm | 0 | intercept | 0.576919 | 0.027734 | 20.801531 | 2.897665e-76 | True | True | 771 | 19 | 1.000000 | 0.026333 |
| antic | anticipated_affect_norm | 1 | dailyAnticipatedAffectYesterday | 0.164033 | 0.022019 | 7.449684 | 2.572770e-13 | True | True | 771 | 19 | 1.000000 | 0.026333 |
| antic | anticipated_affect_norm | 2 | todayStepCount | 0.007128 | 0.007112 | 1.002281 | 0.316530 | False | False | 771 | 19 | 1.000000 | 0.026333 |
| antic | anticipated_affect_norm | 4 | activityStatusToday | 0.026028 | 0.017135 | 1.519026 | 0.129176 | False | False | 771 | 19 | 1.000000 | 0.026333 |
| antic | anticipated_affect_norm | 5 | salienceMessageSentToday | -0.044461 | 0.019064 | -2.332208 | 0.019953 | True | False | 771 | 19 | 1.000000 | 0.026333 |
| antic | anticipated_affect_norm | 6 | dayOfWeekNorm | -0.065572 | 0.017948 | -3.653342 | 0.000277 | True | True | 771 | 19 | 1.000000 | 0.026333 |
| antic | anticipated_affect_norm | 7 | perceivedUtilityLastWeek | -0.005629 | 0.009659 | -0.582761 | 0.560229 | False | False | 771 | 19 | 1.000000 | 0.026333 |
| antic | anticipated_affect_norm | 8 | caeAverageLastWeek | 0.243590 | 0.011794 | 20.654447 | 2.025902e-75 | True | True | 771 | 19 | 1.000000 | 0.026333 |
| antic | anticipated_affect_norm | 9 | ws_morning | 0.046889 | 0.027104 | 1.730010 | 0.084039 | False | False | 771 | 19 | 1.000000 | 0.026333 |
| antic | anticipated_affect_norm | 10 | ws_afternoon | -0.041902 | 0.027106 | -1.545882 | 0.122554 | False | False | 771 | 19 | 1.000000 | 0.026333 |
| antic | anticipated_affect_norm | 11 | ws_morning*salienceMessageSentToday | 0.001011 | 0.023935 | 0.042219 | 0.966336 | False | False | 771 | 19 | 1.000000 | 0.026333 |
| antic | anticipated_affect_norm | 12 | ws_afternoon*salienceMessageSentToday | 0.039842 | 0.023526 | 1.693506 | 0.090773 | False | False | 771 | 19 | 1.000000 | 0.026333 |
| antic | anticipated_affect_norm | 13 | ws_morning*dayOfWeekNorm | -0.001076 | 0.020463 | -0.052590 | 0.958073 | False | False | 771 | 19 | 1.000000 | 0.026333 |
| antic | anticipated_affect_norm | 14 | ws_afternoon*dayOfWeekNorm | 0.032351 | 0.020381 | 1.587341 | 0.112856 | False | False | 771 | 19 | 1.000000 | 0.026333 |
| antic | anticipated_affect_norm | 15 | ws_morning*perceivedUtilityLastWeek | -0.009619 | 0.011970 | -0.803576 | 0.421896 | False | False | 771 | 19 | 1.000000 | 0.026333 |
| antic | anticipated_affect_norm | 16 | ws_afternoon*perceivedUtilityLastWeek | 0.001093 | 0.011966 | 0.091306 | 0.927274 | False | False | 771 | 19 | 1.000000 | 0.026333 |
| antic | anticipated_affect_norm | 17 | ws_morning*caeAverageLastWeek | 0.002336 | 0.013043 | 0.179068 | 0.857932 | False | False | 771 | 19 | 1.000000 | 0.026333 |
| antic | anticipated_affect_norm | 18 | ws_afternoon*caeAverageLastWeek | -0.023105 | 0.012842 | -1.799141 | 0.072397 | False | False | 771 | 19 | 1.000000 | 0.026333 |
| CAE | CAE_avg_norm | 0 | intercept | -0.406692 | 0.090461 | -4.495778 | 1.242286e-05 | True | True | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 1 | caeAverageLastWeek | 0.806559 | 0.042258 | 19.086405 | 5.075466e-45 | True | True | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 2 | week_norm | 0.002936 | 0.032599 | 0.090070 | 0.928333 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 3 | fourSC_slot_0 | -0.004979 | 0.033577 | -0.148282 | 0.882287 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 4 | fourSC_slot_1 | -0.016836 | 0.035390 | -0.475719 | 0.634855 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 5 | fourSC_slot_2 | -0.012104 | 0.033438 | -0.361985 | 0.717791 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 6 | fourSC_slot_3 | -0.003986 | 0.037306 | -0.106853 | 0.915025 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 7 | fourSC_slot_4 | -0.036306 | 0.036348 | -0.998846 | 0.319218 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 8 | fourSC_slot_5 | 0.025052 | 0.037218 | 0.673115 | 0.501742 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 9 | fourSC_slot_6 | 0.012014 | 0.032927 | 0.364880 | 0.715631 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 10 | fourSC_slot_7 | 0.027332 | 0.037051 | 0.737687 | 0.461671 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 11 | fourSC_slot_8 | -0.011972 | 0.030904 | -0.387380 | 0.698935 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 12 | fourSC_slot_9 | 0.006710 | 0.036079 | 0.185970 | 0.852679 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 13 | fourSC_slot_10 | 0.003332 | 0.032545 | 0.102376 | 0.918573 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 14 | fourSC_slot_11 | 0.018821 | 0.032612 | 0.577136 | 0.564572 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 15 | fourSC_slot_12 | -0.014065 | 0.035778 | -0.393123 | 0.694696 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 16 | fourSC_slot_13 | -0.043196 | 0.037675 | -1.146535 | 0.253104 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 17 | antic_day_0 | 0.136524 | 0.132551 | 1.029970 | 0.304414 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 18 | antic_day_1 | -0.214721 | 0.122262 | -1.756236 | 0.080758 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 19 | antic_day_2 | -0.045320 | 0.125739 | -0.360428 | 0.718952 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 20 | antic_day_3 | 0.264817 | 0.130398 | 2.030839 | 0.043751 | True | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 21 | antic_day_4 | 0.275072 | 0.129752 | 2.119987 | 0.035385 | True | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 22 | antic_day_5 | 0.054528 | 0.124996 | 0.436236 | 0.663191 | False | False | 203 | 24 | 1.000000 | 0.080889 |
| CAE | CAE_avg_norm | 23 | antic_day_6 | 0.170617 | 0.114814 | 1.486036 | 0.139029 | False | False | 203 | 24 | 1.000000 | 0.080889 |

## Pooled PF GEE Coefficients

Population-averaged GEE fits on the same stacked PF designs, clustered by `ParticipantIdentifier`. `p_value` uses GEE sandwich standard errors with exchangeable working correlation. These are for inference/audit only; PF priors still use ridge.

| coefficient | robust_se | z_stat | p_value | significant_0.05 | significant_0.01 | model | outcome | index | feature | identified | feature_std | n_obs | n_clusters | n_features | working_correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -0.106544 |  |  |  | False | False | fourSC | 4hour_step_norm | 0 | intercept | False | 0.000000 | 2388 | 29 | 25 | exchangeable |
| 0.049442 | 0.031971 | 1.546467 | 0.121992 | False | False | fourSC | 4hour_step_norm | 1 | yesterdayStepCount | True | 0.964362 | 2388 | 29 | 25 | exchangeable |
| 0.248809 | 0.077974 | 3.190932 | 0.001418 | True | True | fourSC | 4hour_step_norm | 2 | stepCountLast7DaysEma | True | 0.999697 | 2388 | 29 | 25 | exchangeable |
| 0.194896 | 0.043251 | 4.506219 | 6.599301e-06 | True | True | fourSC | 4hour_step_norm | 3 | prior2HourStepCount | True | 0.946605 | 2388 | 29 | 25 | exchangeable |
| -0.047899 | 0.034550 | -1.386351 | 0.165640 | False | False | fourSC | 4hour_step_norm | 5 | activitySuggestionsSentLast7Days | True | 0.907489 | 2388 | 29 | 25 | exchangeable |
| 0.148940 | 0.068037 | 2.189104 | 0.028589 | True | False | fourSC | 4hour_step_norm | 6 | morningFitbitWearLast7Days | True | 0.280658 | 2388 | 29 | 25 | exchangeable |
| -0.004134 | 0.047891 | -0.086324 | 0.931209 | False | False | fourSC | 4hour_step_norm | 7 | salienceMessageSentYesterday | True | 0.495878 | 2388 | 29 | 25 | exchangeable |
| 0.084254 | 0.147264 | 0.572127 | 0.567236 | False | False | fourSC | 4hour_step_norm | 8 | activitySuggestionInteractLast7Days | True | 0.186427 | 2388 | 29 | 25 | exchangeable |
| 0.149232 | 0.152923 | 0.975868 | 0.329130 | False | False | fourSC | 4hour_step_norm | 9 | activeDaysLast7Days | True | 0.303019 | 2388 | 29 | 25 | exchangeable |
| 0.002833 | 0.052176 | 0.054298 | 0.956698 | False | False | fourSC | 4hour_step_norm | 10 | dayOfWeekNorm | True | 0.568928 | 2388 | 29 | 25 | exchangeable |
| -0.259070 | 0.114308 | -2.266420 | 0.023426 | True | False | fourSC | 4hour_step_norm | 11 | decisionTimeSlot | True | 0.499809 | 2388 | 29 | 25 | exchangeable |
| -0.010366 | 0.021639 | -0.479023 | 0.631922 | False | False | fourSC | 4hour_step_norm | 12 | perceivedUtilityLastWeek | True | 1.362100 | 2388 | 29 | 25 | exchangeable |
| 0.051431 | 0.036364 | 1.414331 | 0.157265 | False | False | fourSC | 4hour_step_norm | 13 | caeAverageLastWeek | True | 1.043337 | 2388 | 29 | 25 | exchangeable |
| 0.022548 | 0.096789 | 0.232963 | 0.815790 | False | False | fourSC | 4hour_step_norm | 14 | Ah | True | 0.499961 | 2388 | 29 | 25 | exchangeable |
| -0.014315 | 0.033307 | -0.429775 | 0.667360 | False | False | fourSC | 4hour_step_norm | 15 | Ah*yesterdayStepCount | True | 0.682505 | 2388 | 29 | 25 | exchangeable |
| 0.016656 | 0.038233 | 0.435633 | 0.663103 | False | False | fourSC | 4hour_step_norm | 16 | Ah*prior2HourStepCount | True | 0.659713 | 2388 | 29 | 25 | exchangeable |
| 0.071692 | 0.055291 | 1.296626 | 0.194760 | False | False | fourSC | 4hour_step_norm | 17 | Ah*activitySuggestionsSentLast7Days | True | 0.626980 | 2388 | 29 | 25 | exchangeable |
| -0.113324 | 0.126209 | -0.897906 | 0.369236 | False | False | fourSC | 4hour_step_norm | 18 | Ah*morningFitbitWearLast7Days | True | 0.444954 | 2388 | 29 | 25 | exchangeable |
| 0.021982 | 0.080116 | 0.274375 | 0.783796 | False | False | fourSC | 4hour_step_norm | 19 | Ah*salienceMessageSentYesterday | True | 0.409532 | 2388 | 29 | 25 | exchangeable |
| 0.166784 | 0.231854 | 0.719348 | 0.471927 | False | False | fourSC | 4hour_step_norm | 20 | Ah*activitySuggestionInteractLast7Days | True | 0.195631 | 2388 | 29 | 25 | exchangeable |
| 0.030703 | 0.061302 | 0.500842 | 0.616483 | False | False | fourSC | 4hour_step_norm | 21 | Ah*dayOfWeekNorm | True | 0.409448 | 2388 | 29 | 25 | exchangeable |
| 0.025447 | 0.078003 | 0.326233 | 0.744248 | False | False | fourSC | 4hour_step_norm | 22 | Ah*decisionTimeSlot | True | 0.444660 | 2388 | 29 | 25 | exchangeable |
| 0.005308 | 0.022363 | 0.237371 | 0.812369 | False | False | fourSC | 4hour_step_norm | 23 | Ah*perceivedUtilityLastWeek | True | 1.241546 | 2388 | 29 | 25 | exchangeable |
| -0.005725 | 0.029488 | -0.194156 | 0.846054 | False | False | fourSC | 4hour_step_norm | 24 | Ah*caeAverageLastWeek | True | 0.712362 | 2388 | 29 | 25 | exchangeable |
| 0.599835 |  |  |  | False | False | antic | anticipated_affect_norm | 0 | intercept | False | 0.000000 | 771 | 28 | 19 | exchangeable |
| 0.052763 | 0.027824 | 1.896311 | 0.057919 | False | False | antic | anticipated_affect_norm | 1 | dailyAnticipatedAffectYesterday | True | 0.369763 | 771 | 28 | 19 | exchangeable |
| -0.005632 | 0.006742 | -0.835363 | 0.403514 | False | False | antic | anticipated_affect_norm | 2 | todayStepCount | True | 0.873584 | 771 | 28 | 19 | exchangeable |
| 0.040104 | 0.020476 | 1.958585 | 0.050161 | False | False | antic | anticipated_affect_norm | 4 | activityStatusToday | True | 0.363950 | 771 | 28 | 19 | exchangeable |
| -0.029320 | 0.015741 | -1.862725 | 0.062501 | False | False | antic | anticipated_affect_norm | 5 | salienceMessageSentToday | True | 0.496505 | 771 | 28 | 19 | exchangeable |
| -0.022895 | 0.020589 | -1.111972 | 0.266150 | False | False | antic | anticipated_affect_norm | 6 | dayOfWeekNorm | True | 0.573071 | 771 | 28 | 19 | exchangeable |
| 0.007864 | 0.020386 | 0.385745 | 0.699686 | False | False | antic | anticipated_affect_norm | 7 | perceivedUtilityLastWeek | True | 0.994628 | 771 | 28 | 19 | exchangeable |
| 0.150917 | 0.043218 | 3.492018 | 0.000479 | True | True | antic | anticipated_affect_norm | 8 | caeAverageLastWeek | True | 0.933883 | 771 | 28 | 19 | exchangeable |
| 0.040519 | 0.034947 | 1.159428 | 0.246282 | False | False | antic | anticipated_affect_norm | 9 | ws_morning | True | 0.494209 | 771 | 28 | 19 | exchangeable |
| -0.026190 | 0.032669 | -0.801684 | 0.422736 | False | False | antic | anticipated_affect_norm | 10 | ws_afternoon | True | 0.495691 | 771 | 28 | 19 | exchangeable |
| -0.008751 | 0.022083 | -0.396270 | 0.691906 | False | False | antic | anticipated_affect_norm | 11 | ws_morning*salienceMessageSentToday | True | 0.381155 | 771 | 28 | 19 | exchangeable |
| 0.046279 | 0.017805 | 2.599269 | 0.009342 | True | True | antic | anticipated_affect_norm | 12 | ws_afternoon*salienceMessageSentToday | True | 0.430168 | 771 | 28 | 19 | exchangeable |
| 0.008826 | 0.011369 | 0.776312 | 0.437565 | False | False | antic | anticipated_affect_norm | 13 | ws_morning*dayOfWeekNorm | True | 0.387399 | 771 | 28 | 19 | exchangeable |
| 0.015361 | 0.018549 | 0.828103 | 0.407612 | False | False | antic | anticipated_affect_norm | 14 | ws_afternoon*dayOfWeekNorm | True | 0.440234 | 771 | 28 | 19 | exchangeable |
| -0.008592 | 0.012359 | -0.695181 | 0.486942 | False | False | antic | anticipated_affect_norm | 15 | ws_morning*perceivedUtilityLastWeek | True | 1.141978 | 771 | 28 | 19 | exchangeable |
| -0.000476 | 0.013614 | -0.034978 | 0.972097 | False | False | antic | anticipated_affect_norm | 16 | ws_afternoon*perceivedUtilityLastWeek | True | 1.203026 | 771 | 28 | 19 | exchangeable |
| 0.006642 | 0.021100 | 0.314785 | 0.752925 | False | False | antic | anticipated_affect_norm | 17 | ws_morning*caeAverageLastWeek | True | 0.600451 | 771 | 28 | 19 | exchangeable |
| -0.021756 | 0.018721 | -1.162103 | 0.245194 | False | False | antic | anticipated_affect_norm | 18 | ws_afternoon*caeAverageLastWeek | True | 0.705847 | 771 | 28 | 19 | exchangeable |
| -0.772799 |  |  |  | False | False | CAE | CAE_avg_norm | 0 | intercept | False | 0.000000 | 203 | 29 | 24 | exchangeable |
| 0.611836 | 0.064347 | 9.508395 | 1.936276e-21 | True | True | CAE | CAE_avg_norm | 1 | caeAverageLastWeek | True | 0.982080 | 203 | 29 | 24 | exchangeable |
| -0.000886 | 0.034736 | -0.025516 | 0.979644 | False | False | CAE | CAE_avg_norm | 2 | week_norm | True | 0.620577 | 203 | 29 | 24 | exchangeable |
| -0.009590 | 0.030987 | -0.309490 | 0.756949 | False | False | CAE | CAE_avg_norm | 3 | fourSC_slot_0 | True | 0.898890 | 203 | 29 | 24 | exchangeable |
| -0.005887 | 0.032076 | -0.183533 | 0.854380 | False | False | CAE | CAE_avg_norm | 4 | fourSC_slot_1 | True | 0.830783 | 203 | 29 | 24 | exchangeable |
| -0.002492 | 0.038847 | -0.064156 | 0.948846 | False | False | CAE | CAE_avg_norm | 5 | fourSC_slot_2 | True | 0.848201 | 203 | 29 | 24 | exchangeable |
| -0.016899 | 0.035482 | -0.476275 | 0.633879 | False | False | CAE | CAE_avg_norm | 6 | fourSC_slot_3 | True | 0.746729 | 203 | 29 | 24 | exchangeable |
| -0.037877 | 0.026729 | -1.417107 | 0.156452 | False | False | CAE | CAE_avg_norm | 7 | fourSC_slot_4 | True | 0.819992 | 203 | 29 | 24 | exchangeable |
| 0.034527 | 0.032861 | 1.050700 | 0.293396 | False | False | CAE | CAE_avg_norm | 8 | fourSC_slot_5 | True | 0.753022 | 203 | 29 | 24 | exchangeable |
| 0.016033 | 0.020173 | 0.794770 | 0.426747 | False | False | CAE | CAE_avg_norm | 9 | fourSC_slot_6 | True | 0.902600 | 203 | 29 | 24 | exchangeable |
| 0.015182 | 0.020756 | 0.731470 | 0.464492 | False | False | CAE | CAE_avg_norm | 10 | fourSC_slot_7 | True | 0.759013 | 203 | 29 | 24 | exchangeable |
| -0.003332 | 0.024488 | -0.136049 | 0.891783 | False | False | CAE | CAE_avg_norm | 11 | fourSC_slot_8 | True | 0.855158 | 203 | 29 | 24 | exchangeable |
| -0.004679 | 0.030597 | -0.152917 | 0.878464 | False | False | CAE | CAE_avg_norm | 12 | fourSC_slot_9 | True | 0.801225 | 203 | 29 | 24 | exchangeable |
| 0.010646 | 0.034200 | 0.311277 | 0.755590 | False | False | CAE | CAE_avg_norm | 13 | fourSC_slot_10 | True | 0.844007 | 203 | 29 | 24 | exchangeable |
| -0.001146 | 0.026801 | -0.042770 | 0.965885 | False | False | CAE | CAE_avg_norm | 14 | fourSC_slot_11 | True | 0.822304 | 203 | 29 | 24 | exchangeable |
| -0.012531 | 0.040586 | -0.308766 | 0.757499 | False | False | CAE | CAE_avg_norm | 15 | fourSC_slot_12 | True | 0.753779 | 203 | 29 | 24 | exchangeable |
| -0.056782 | 0.032051 | -1.771635 | 0.076455 | False | False | CAE | CAE_avg_norm | 16 | fourSC_slot_13 | True | 0.782485 | 203 | 29 | 24 | exchangeable |
| 0.279526 | 0.197424 | 1.415870 | 0.156814 | False | False | CAE | CAE_avg_norm | 17 | antic_day_0 | True | 0.289879 | 203 | 29 | 24 | exchangeable |
| -0.238822 | 0.245019 | -0.974708 | 0.329705 | False | False | CAE | CAE_avg_norm | 18 | antic_day_1 | True | 0.303869 | 203 | 29 | 24 | exchangeable |
| 0.036264 | 0.193582 | 0.187330 | 0.851402 | False | False | CAE | CAE_avg_norm | 19 | antic_day_2 | True | 0.306768 | 203 | 29 | 24 | exchangeable |
| 0.427313 | 0.170838 | 2.501281 | 0.012375 | True | False | CAE | CAE_avg_norm | 20 | antic_day_3 | True | 0.294023 | 203 | 29 | 24 | exchangeable |
| 0.305880 | 0.172143 | 1.776898 | 0.075585 | False | False | CAE | CAE_avg_norm | 21 | antic_day_4 | True | 0.296584 | 203 | 29 | 24 | exchangeable |
| 0.101434 | 0.153948 | 0.658885 | 0.509970 | False | False | CAE | CAE_avg_norm | 22 | antic_day_5 | True | 0.313285 | 203 | 29 | 24 | exchangeable |
| 0.288084 | 0.151994 | 1.895362 | 0.058045 | False | False | CAE | CAE_avg_norm | 23 | antic_day_6 | True | 0.338504 | 203 | 29 | 24 | exchangeable |

## Pooled RL Q GEE Coefficients

Gaussian GEE on the final fitted-Q regression from pooled FQI (``phi_obs`` vs bootstrap targets), clustered by participant. ``identified=False`` marks structurally unused features with zero design variance (e.g. ``b_tilde``, masked day-6 mediators); SE / p-values are omitted for those rows. RL priors still use ridge-FQI for ``mu_0_micro``.

| coefficient | robust_se | z_stat | p_value | significant_0.05 | significant_0.01 | model | outcome | index | feature | identified | feature_std | n_obs | n_clusters | n_features | working_correlation | fqi_iters | ridge_alpha | residual_sigma2 | block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.131942 |  |  |  | False | False | q_no_td_modify | fqi_target | 0 | intercept | False | 0.000000 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.394475 | 0.069958 | -5.638731 | 1.713078e-08 | True | True | q_no_td_modify | fqi_target | 1 | weekday_vs_weekend | True | 0.372678 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.017978 | 0.009646 | 1.863789 | 0.062351 | False | False | q_no_td_modify | fqi_target | 2 | slot_pm | True | 0.500000 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.026525 | 0.002213 | 11.986906 | 4.161595e-33 | True | True | q_no_td_modify | fqi_target | 3 | E_w | True | 1.304254 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.010014 | 0.012407 | 0.807130 | 0.419592 | False | False | q_no_td_modify | fqi_target | 4 | weekday_vs_weekend*E_w | True | 0.817377 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.001597 | 0.003559 | 0.448779 | 0.653591 | False | False | q_no_td_modify | fqi_target | 5 | slot_pm*E_w | True | 1.242097 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.789520 | 0.006791 | 116.255012 | 0.000000 | True | True | q_no_td_modify | fqi_target | 6 | b_hat | True | 0.973386 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.405972 | 0.031829 | 12.754964 | 2.925099e-37 | True | True | q_no_td_modify | fqi_target | 7 | weekday_vs_weekend*b_hat | True | 0.397656 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.095261 | 0.007354 | 12.953084 | 2.257002e-38 | True | True | q_no_td_modify | fqi_target | 8 | slot_pm*b_hat | True | 0.688572 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -2.339899e-15 |  |  |  | False | False | q_no_td_modify | fqi_target | 9 | b_tilde | False | 0.000000 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.012492 | 0.003437 | -3.634600 | 0.000278 | True | True | q_no_td_modify | fqi_target | 10 | M_Y_day1_fourSC_morning | True | 0.904548 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.001356 | 0.004811 | -0.281953 | 0.777980 | False | False | q_no_td_modify | fqi_target | 11 | M_Y_day1_fourSC_afternoon | True | 0.823339 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.148913 | 0.014012 | 10.627299 | 2.224628e-26 | True | True | q_no_td_modify | fqi_target | 12 | M_Y_day1_anticipated_affect | True | 0.363285 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.017526 | 0.003144 | 5.574355 | 2.484485e-08 | True | True | q_no_td_modify | fqi_target | 13 | M_Y_day2_fourSC_morning | True | 0.821105 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.000136 | 0.005615 | -0.024183 | 0.980707 | False | False | q_no_td_modify | fqi_target | 14 | M_Y_day2_fourSC_afternoon | True | 0.651816 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.224669 | 0.020329 | 11.051365 | 2.159066e-28 | True | True | q_no_td_modify | fqi_target | 15 | M_Y_day2_anticipated_affect | True | 0.388591 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.006758 | 0.004200 | 1.609223 | 0.107568 | False | False | q_no_td_modify | fqi_target | 16 | M_Y_day3_fourSC_morning | True | 0.687032 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.038695 | 0.008674 | 4.460906 | 8.161390e-06 | True | True | q_no_td_modify | fqi_target | 17 | M_Y_day3_fourSC_afternoon | True | 0.590632 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.238232 | 0.015871 | 15.010046 | 6.310406e-51 | True | True | q_no_td_modify | fqi_target | 18 | M_Y_day3_anticipated_affect | True | 0.382124 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.028018 | 0.006969 | 4.020657 | 5.803596e-05 | True | True | q_no_td_modify | fqi_target | 19 | M_Y_day4_fourSC_morning | True | 0.618347 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.025839 | 0.012995 | 1.988341 | 0.046774 | True | False | q_no_td_modify | fqi_target | 20 | M_Y_day4_fourSC_afternoon | True | 0.466581 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.340992 | 0.047410 | 7.192352 | 6.368441e-13 | True | True | q_no_td_modify | fqi_target | 21 | M_Y_day4_anticipated_affect | True | 0.344182 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.009690 | 0.017267 | -0.561165 | 0.574685 | False | False | q_no_td_modify | fqi_target | 22 | M_Y_day5_fourSC_morning | True | 0.451750 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.064485 | 0.022709 | -2.839672 | 0.004516 | True | True | q_no_td_modify | fqi_target | 23 | M_Y_day5_fourSC_afternoon | True | 0.348963 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.236848 | 0.120192 | 1.970581 | 0.048772 | True | False | q_no_td_modify | fqi_target | 24 | M_Y_day5_anticipated_affect | True | 0.265232 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.014814 | 0.031438 | -0.471199 | 0.637499 | False | False | q_no_td_modify | fqi_target | 25 | M_Y_day6_fourSC_morning | True | 0.260353 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -4.938048e-16 |  |  |  | False | False | q_no_td_modify | fqi_target | 26 | M_Y_day6_fourSC_afternoon | False | 0.000000 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 5.546376e-17 |  |  |  | False | False | q_no_td_modify | fqi_target | 27 | M_Y_day6_anticipated_affect | False | 0.000000 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.007635 | 0.003510 | -2.175470 | 0.029595 | True | False | q_no_td_modify | fqi_target | 28 | M_E_day1_pageview_morning | True | 0.968985 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.003042 | 0.003327 | -0.914232 | 0.360595 | False | False | q_no_td_modify | fqi_target | 29 | M_E_day1_pageview_afternoon | True | 1.002472 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.070387 | 0.008900 | -7.908673 | 2.601461e-15 | True | True | q_no_td_modify | fqi_target | 30 | M_E_day1_morning_fitbit_wear | True | 0.500000 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.006947 | 0.006127 | -1.133817 | 0.256871 | False | False | q_no_td_modify | fqi_target | 31 | M_E_day1_daily_survey_complete | True | 0.484492 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.046229 | 0.002759 | 16.752965 | 5.387321e-63 | True | True | q_no_td_modify | fqi_target | 32 | M_E_day2_pageview_morning | True | 0.731130 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.010362 | 0.002846 | 3.641359 | 0.000271 | True | True | q_no_td_modify | fqi_target | 33 | M_E_day2_pageview_afternoon | True | 0.929593 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.106949 | 0.010954 | -9.763323 | 1.617659e-22 | True | True | q_no_td_modify | fqi_target | 34 | M_E_day2_morning_fitbit_wear | True | 0.491255 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.110096 | 0.008429 | -13.060884 | 5.508440e-39 | True | True | q_no_td_modify | fqi_target | 35 | M_E_day2_daily_survey_complete | True | 0.451442 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.054373 | 0.003925 | 13.851833 | 1.240068e-43 | True | True | q_no_td_modify | fqi_target | 36 | M_E_day3_pageview_morning | True | 0.649803 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.023658 | 0.004800 | -4.928381 | 8.291370e-07 | True | True | q_no_td_modify | fqi_target | 37 | M_E_day3_pageview_afternoon | True | 0.762895 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.069376 | 0.011902 | -5.828879 | 5.580091e-09 | True | True | q_no_td_modify | fqi_target | 38 | M_E_day3_morning_fitbit_wear | True | 0.453602 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.154841 | 0.009219 | -16.796202 | 2.601776e-63 | True | True | q_no_td_modify | fqi_target | 39 | M_E_day3_daily_survey_complete | True | 0.409982 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.084931 | 0.007128 | 11.915461 | 9.832123e-33 | True | True | q_no_td_modify | fqi_target | 40 | M_E_day4_pageview_morning | True | 0.609669 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.090011 | 0.007538 | 11.940539 | 7.274987e-33 | True | True | q_no_td_modify | fqi_target | 41 | M_E_day4_pageview_afternoon | True | 0.667160 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.213293 | 0.039257 | -5.433216 | 5.534741e-08 | True | True | q_no_td_modify | fqi_target | 42 | M_E_day4_morning_fitbit_wear | True | 0.394733 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.203036 | 0.026544 | -7.649116 | 2.023651e-14 | True | True | q_no_td_modify | fqi_target | 43 | M_E_day4_daily_survey_complete | True | 0.336177 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.044657 | 0.010564 | 4.227104 | 2.367183e-05 | True | True | q_no_td_modify | fqi_target | 44 | M_E_day5_pageview_morning | True | 0.427859 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.038943 | 0.012631 | -3.083222 | 0.002048 | True | True | q_no_td_modify | fqi_target | 45 | M_E_day5_pageview_afternoon | True | 0.457491 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.176705 | 0.031830 | 5.551458 | 2.832960e-08 | True | True | q_no_td_modify | fqi_target | 46 | M_E_day5_morning_fitbit_wear | True | 0.295346 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.110081 | 0.029767 | 3.698021 | 0.000217 | True | True | q_no_td_modify | fqi_target | 47 | M_E_day5_daily_survey_complete | True | 0.241281 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.026521 | 0.024593 | -1.078393 | 0.280858 | False | False | q_no_td_modify | fqi_target | 48 | M_E_day6_pageview_morning | True | 0.293760 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 2.495337e-18 |  |  |  | False | False | q_no_td_modify | fqi_target | 49 | M_E_day6_pageview_afternoon | False | 0.000000 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -1.170659e-16 |  |  |  | False | False | q_no_td_modify | fqi_target | 50 | M_E_day6_morning_fitbit_wear | False | 0.000000 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 1.429978e-16 |  |  |  | False | False | q_no_td_modify | fqi_target | 51 | M_E_day6_daily_survey_complete | False | 0.000000 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.010245 | 0.005237 | -1.956419 | 0.050416 | False | False | q_no_td_modify | fqi_target | 52 | yesterday_step_count | True | 0.951195 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.010822 | 0.003800 | 2.848062 | 0.004399 | True | True | q_no_td_modify | fqi_target | 53 | ema_step_count | True | 0.926023 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.011013 | 0.004922 | -2.237355 | 0.025263 | True | False | q_no_td_modify | fqi_target | 54 | prior2hour_step_count | True | 0.803693 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.093598 | 0.009261 | -10.106472 | 5.171303e-24 | True | True | q_no_td_modify | fqi_target | 56 | active_status_fraction_7days | True | 0.317262 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.011058 | 0.003229 | 3.424805 | 0.000615 | True | True | q_no_td_modify | fqi_target | 57 | recent_burden | True | 0.894208 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.009261 | 0.022090 | -0.419221 | 0.675054 | False | False | q_no_td_modify | fqi_target | 59 | walk_interaction_7d | True | 0.195461 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 9.186864e-17 |  |  |  | False | False | q_no_td_modify | fqi_target | 60 | A | False | 0.000000 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.003898 | 0.016499 | 0.236240 | 0.813247 | False | False | q_no_td_modify | fqi_target | 61 | A*E_w | True | 0.499952 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.010853 | 0.003382 | 3.209570 | 0.001329 | True | True | q_no_td_modify | fqi_target | 62 | A*b_hat | True | 1.238003 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.006199 | 0.007148 | -0.867257 | 0.385801 | False | False | q_no_td_modify | fqi_target | 63 | A*weekday_vs_weekend | True | 0.671890 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.015634 | 0.045833 | -0.341121 | 0.733012 | False | False | q_no_td_modify | fqi_target | 64 | A*slot_pm | True | 0.274645 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.004973 | 0.013381 | -0.371621 | 0.710175 | False | False | q_no_td_modify | fqi_target | 65 | A*weekday_vs_weekend*E_w | True | 0.449644 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.035552 | 0.020013 | -1.776470 | 0.075656 | False | False | q_no_td_modify | fqi_target | 66 | A*slot_pm*E_w | True | 0.595351 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.011352 | 0.005112 | -2.220841 | 0.026362 | True | False | q_no_td_modify | fqi_target | 67 | A*weekday_vs_weekend*b_hat | True | 1.029780 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.031520 | 0.043242 | 0.728927 | 0.466047 | False | False | q_no_td_modify | fqi_target | 68 | A*slot_pm*b_hat | True | 0.266467 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.006879 | 0.013265 | 0.518607 | 0.604035 | False | False | q_no_td_modify | fqi_target | 69 | A*yesterday_step_count | True | 0.512879 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.005262 | 0.007803 | -0.674334 | 0.500099 | False | False | q_no_td_modify | fqi_target | 70 | A*ema_step_count | True | 0.668643 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.000329 | 0.004840 | -0.068002 | 0.945784 | False | False | q_no_td_modify | fqi_target | 71 | A*prior2hour_step_count | True | 0.638505 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.005500 | 0.024259 | -0.226729 | 0.820634 | False | False | q_no_td_modify | fqi_target | 73 | A*active_status_fraction_7days | True | 0.189277 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| -0.026148 | 0.014588 | -1.792360 | 0.073075 | False | False | q_no_td_modify | fqi_target | 74 | A*recent_burden | True | 0.384224 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
| 0.000652 | 0.005375 | 0.121234 | 0.903506 | False | False | q_no_td_modify | fqi_target | 75 | A*salience_yesterday | True | 0.621052 | 3480 | 29 | 79 | exchangeable | 25 | 1.000000 | 0.031989 | beta |
