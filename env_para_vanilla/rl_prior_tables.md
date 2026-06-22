# RL Prior Mean and Feature Variance Tables

Variance is the diagonal entry of the prior covariance matrix (`Gamma_0` for PF, `Sigma_0` for RL priors). Off-diagonal covariance terms are not shown in these tables.

## PF

| model | feature | mean | variance |
| --- | --- | --- | --- |
| fourSC | intercept | 0.194753 | 0.138972 |
| fourSC | yesterdayStepCount | 0.174644 | 0.005573 |
| fourSC | stepCountLast7DaysEma | 0.307883 | 0.017336 |
| fourSC | prior2HourStepCount | 0.194106 | 0.031018 |
| fourSC | activityCompletedLast7Days | -0.190223 | 0.049279 |
| fourSC | activitySuggestionsSentLast7Days | -0.091630 | 0.026071 |
| fourSC | morningFitbitWearLast7Days | -0.041087 | 0.122548 |
| fourSC | salienceMessageSentYesterday | 0.006459 | 0.032031 |
| fourSC | activitySuggestionInteractLast7Days | 0.452949 | 0.138251 |
| fourSC | activeDaysLast7Days | -0.077278 | 0.177183 |
| fourSC | dayOfWeekNorm | 0.060370 | 0.079064 |
| fourSC | decisionTimeSlot | -0.291529 | 0.213020 |
| fourSC | perceivedUtilityLastWeek | -0.042385 | 0.090355 |
| fourSC | caeAverageLastWeek | -0.023047 | 0.066690 |
| fourSC | Ah | 0.139455 | 0.037014 |
| fourSC | Ah*yesterdayStepCount | -0.047952 | 0.023847 |
| fourSC | Ah*prior2HourStepCount | 0.034672 | 0.019963 |
| fourSC | Ah*activitySuggestionsSentLast7Days | 0.070455 | 0.014922 |
| fourSC | Ah*morningFitbitWearLast7Days | 0.084767 | 0.070636 |
| fourSC | Ah*salienceMessageSentYesterday | -0.139707 | 0.077303 |
| fourSC | Ah*activitySuggestionInteractLast7Days | -0.075206 | 0.072351 |
| fourSC | Ah*dayOfWeekNorm | -0.059135 | 0.063611 |
| fourSC | Ah*decisionTimeSlot | 0.002902 | 0.100250 |
| fourSC | Ah*perceivedUtilityLastWeek | -0.044606 | 0.013215 |
| fourSC | Ah*caeAverageLastWeek | 0.013474 | 0.052263 |
| antic | intercept | 0.629315 | 0.024035 |
| antic | dailyAnticipatedAffectYesterday | 0.132914 | 0.003136 |
| antic | todayStepCount | 0.011892 | 0.003346 |
| antic | recordedPhysicalActivityToday | -0.032619 | 0.004595 |
| antic | activityStatusToday | 0.092372 | 0.003957 |
| antic | salienceMessageSentToday | -0.016374 | 0.002356 |
| antic | dayOfWeekNorm | -0.032409 | 0.003078 |
| antic | perceivedUtilityLastWeek | -0.014044 | 0.004440 |
| antic | caeAverageLastWeek | 0.217485 | 0.017499 |
| antic | ws_morning | 0.020965 | 0.002975 |
| antic | ws_afternoon | -0.005174 | 0.002747 |
| antic | ws_morning*salienceMessageSentToday | 0.015488 | 0.002631 |
| antic | ws_afternoon*salienceMessageSentToday | 0.008882 | 0.000732 |
| antic | ws_morning*dayOfWeekNorm | -0.000598 | 0.000763 |
| antic | ws_afternoon*dayOfWeekNorm | 0.014776 | 0.003444 |
| antic | ws_morning*perceivedUtilityLastWeek | -0.009295 | 0.001347 |
| antic | ws_afternoon*perceivedUtilityLastWeek | -0.004993 | 0.001443 |
| antic | ws_morning*caeAverageLastWeek | -0.013977 | 0.001486 |
| antic | ws_afternoon*caeAverageLastWeek | -7.136197e-05 | 0.001598 |
| CAE | intercept | -0.478568 | 0.011953 |
| CAE | caeAverageLastWeek | 0.769174 | 0.029882 |
| CAE | week_norm | 0.016622 | 0.004875 |
| CAE | fourSC_slot_0 | 0.017897 | 0.010546 |
| CAE | fourSC_slot_1 | 0.017724 | 0.002456 |
| CAE | fourSC_slot_2 | 0.029814 | 0.011167 |
| CAE | fourSC_slot_3 | 0.039531 | 0.013879 |
| CAE | fourSC_slot_4 | -0.057148 | 0.003938 |
| CAE | fourSC_slot_5 | 0.035330 | 0.005028 |
| CAE | fourSC_slot_6 | 0.054746 | 0.002402 |
| CAE | fourSC_slot_7 | 0.026846 | 0.005966 |
| CAE | fourSC_slot_8 | -0.012704 | 0.002611 |
| CAE | fourSC_slot_9 | -0.018847 | 0.003623 |
| CAE | fourSC_slot_10 | -0.035142 | 0.002228 |
| CAE | fourSC_slot_11 | -0.019777 | 0.003978 |
| CAE | fourSC_slot_12 | -0.047684 | 0.006351 |
| CAE | fourSC_slot_13 | -0.034275 | 0.001215 |
| CAE | antic_day_0 | 0.008222 | 0.004190 |
| CAE | antic_day_1 | 0.286272 | 0.003629 |
| CAE | antic_day_2 | 0.099307 | 0.003121 |
| CAE | antic_day_3 | 0.126081 | 0.002063 |
| CAE | antic_day_4 | 0.138427 | 0.003463 |
| CAE | antic_day_5 | 0.028415 | 0.003213 |
| CAE | antic_day_6 | 0.023084 | 0.002536 |
| CAE_short | intercept | -2.930550e-05 | 0.105669 |
| CAE_short | caeAverage | 0.975493 | 0.043058 |

## Q

| feature | mean | variance |
| --- | --- | --- |
| intercept | -0.224403 | 0.019406 |
| day_norm | -0.258527 | 0.009092 |
| slot_pm | -0.001764 | 0.003762 |
| E_w | 0.014941 | 0.006344 |
| day_norm*E_w | 0.004757 | 0.001676 |
| slot_pm*E_w | 0.002452 | 0.000435 |
| b_hat | 0.606309 | 0.019402 |
| day_norm*b_hat | 0.251210 | 0.008447 |
| slot_pm*b_hat | 0.090336 | 0.001701 |
| b_tilde | 0.000000 | 1.000000e-06 |
| M_Y_day1_fourSC_morning | 0.016092 | 0.004023 |
| M_Y_day1_fourSC_afternoon | 0.030375 | 0.001446 |
| M_Y_day1_anticipated_affect | 0.259609 | 0.006355 |
| M_Y_day2_fourSC_morning | 0.010913 | 0.001084 |
| M_Y_day2_fourSC_afternoon | -0.001928 | 0.000391 |
| M_Y_day2_anticipated_affect | 0.179670 | 0.000733 |
| M_Y_day3_fourSC_morning | -0.021304 | 0.000159 |
| M_Y_day3_fourSC_afternoon | 0.017173 | 0.000274 |
| M_Y_day3_anticipated_affect | 0.121280 | 0.001057 |
| M_Y_day4_fourSC_morning | 0.038913 | 0.000999 |
| M_Y_day4_fourSC_afternoon | -0.015913 | 0.000506 |
| M_Y_day4_anticipated_affect | 0.150819 | 0.001317 |
| M_Y_day5_fourSC_morning | -0.009296 | 0.001015 |
| M_Y_day5_fourSC_afternoon | -0.008943 | 0.000336 |
| M_Y_day5_anticipated_affect | 0.180307 | 0.001987 |
| M_Y_day6_fourSC_morning | -0.032943 | 0.000775 |
| M_Y_day6_fourSC_afternoon | -0.022333 | 0.000277 |
| M_Y_day6_anticipated_affect | 0.147094 | 0.005675 |
| M_E_day1_pageview_morning | 0.013535 | 0.002109 |
| M_E_day1_pageview_afternoon | -0.009741 | 0.003474 |
| M_E_day1_morning_fitbit_wear | 0.030053 | 0.003204 |
| M_E_day1_daily_survey_complete | 0.060255 | 0.002720 |
| M_E_day2_pageview_morning | 0.009437 | 0.002157 |
| M_E_day2_pageview_afternoon | 0.012915 | 0.000626 |
| M_E_day2_morning_fitbit_wear | 0.007175 | 0.000580 |
| M_E_day2_daily_survey_complete | -0.050595 | 0.001776 |
| M_E_day3_pageview_morning | 0.035216 | 0.000563 |
| M_E_day3_pageview_afternoon | -0.058863 | 0.000336 |
| M_E_day3_morning_fitbit_wear | 0.002345 | 0.000556 |
| M_E_day3_daily_survey_complete | 0.006956 | 0.000570 |
| M_E_day4_pageview_morning | 0.019232 | 0.000264 |
| M_E_day4_pageview_afternoon | 0.033793 | 0.000629 |
| M_E_day4_morning_fitbit_wear | 0.003494 | 0.000516 |
| M_E_day4_daily_survey_complete | -0.019567 | 0.001086 |
| M_E_day5_pageview_morning | -0.031211 | 0.001663 |
| M_E_day5_pageview_afternoon | -0.049107 | 8.140448e-05 |
| M_E_day5_morning_fitbit_wear | -0.028787 | 0.001113 |
| M_E_day5_daily_survey_complete | 0.034647 | 0.000263 |
| M_E_day6_pageview_morning | -0.036105 | 0.000882 |
| M_E_day6_pageview_afternoon | -0.009047 | 0.000199 |
| M_E_day6_morning_fitbit_wear | 0.007036 | 0.001031 |
| M_E_day6_daily_survey_complete | 0.011044 | 0.002297 |
| yesterday_step_count | 0.009712 | 0.000109 |
| ema_step_count | -0.000224 | 0.000160 |
| prior2hour_step_count | -0.008957 | 1.393291e-05 |
| previous7days_rpa | 0.044094 | 0.000638 |
| active_status_fraction_7days | -0.036035 | 0.007075 |
| recent_burden | -0.009976 | 0.000575 |
| salience_yesterday | -0.021224 | 0.000426 |
| walk_interaction_7d | -0.165345 | 0.001251 |
| reserved_context_zero | 0.000000 | 1.000000e-06 |
| A | 0.012163 | 0.000294 |
| A*E_w | 0.001783 | 0.000294 |
| A*b_hat | -0.010714 | 0.000795 |
| A*day_norm | -0.004088 | 0.001863 |
| A*slot_pm | -0.006781 | 0.000331 |
| A*day_norm*E_w | -0.004642 | 0.000521 |
| A*slot_pm*E_w | -0.002883 | 0.000148 |
| A*day_norm*b_hat | 0.020796 | 0.000380 |
| A*slot_pm*b_hat | 0.008520 | 0.000426 |
| A*yesterday_step_count | -0.010608 | 0.000193 |
| A*ema_step_count | 0.000318 | 0.000131 |
| A*prior2hour_step_count | 0.003179 | 0.000160 |
| A*previous7days_rpa | -0.018435 | 0.000497 |
| A*active_status_fraction_7days | -0.022352 | 0.000181 |
| A*recent_burden | -0.002840 | 0.000229 |
| A*salience_yesterday | 0.009505 | 0.000370 |
| A*walk_interaction_7d | 0.010764 | 0.000300 |
| A*reserved_context_zero | 0.000000 | 1.000000e-06 |

## RS

| feature | mean | variance |
| --- | --- | --- |
| intercept | -0.008207 | 1.879193e-05 |
| day_norm | -0.002186 | 1.332823e-06 |
| slot_pm | -0.012192 | 4.147188e-05 |
| E_w | 9.949174e-05 | 2.900312e-06 |
| day_norm*E_w | 2.649645e-05 | 1.000000e-06 |
| slot_pm*E_w | 0.000148 | 6.400694e-06 |
| b_hat | 0.013965 | 1.787524e-05 |
| day_norm*b_hat | 0.003719 | 1.267807e-06 |
| slot_pm*b_hat | 0.020746 | 3.944884e-05 |
| b_tilde | 0.000000 | 1.000000e-06 |
| M_Y_day1_fourSC_morning | 0.002842 | 1.876109e-05 |
| M_Y_day1_fourSC_afternoon | 0.003761 | 6.873691e-06 |
| M_Y_day1_anticipated_affect | 0.003746 | 4.268385e-06 |
| M_Y_day2_fourSC_morning | 0.002137 | 1.379801e-05 |
| M_Y_day2_fourSC_afternoon | 0.000494 | 6.559771e-06 |
| M_Y_day2_anticipated_affect | 0.025269 | 4.645605e-06 |
| M_Y_day3_fourSC_morning | 0.001336 | 7.142862e-06 |
| M_Y_day3_fourSC_afternoon | 0.003584 | 4.430009e-06 |
| M_Y_day3_anticipated_affect | -0.006399 | 2.542365e-06 |
| M_Y_day4_fourSC_morning | 0.007510 | 4.255016e-06 |
| M_Y_day4_fourSC_afternoon | 0.000328 | 6.655193e-06 |
| M_Y_day4_anticipated_affect | 0.023138 | 1.556798e-06 |
| M_Y_day5_fourSC_morning | -0.001092 | 6.358438e-06 |
| M_Y_day5_fourSC_afternoon | -0.001579 | 6.971129e-06 |
| M_Y_day5_anticipated_affect | 0.019099 | 1.000000e-06 |
| M_Y_day6_fourSC_morning | -0.007512 | 2.281197e-06 |
| M_Y_day6_fourSC_afternoon | -0.009571 | 1.713608e-06 |
| M_Y_day6_anticipated_affect | -0.013786 | 1.000000e-06 |
| M_E_day1_pageview_morning | 0.001479 | 1.040492e-05 |
| M_E_day1_pageview_afternoon | -0.000225 | 1.174666e-05 |
| M_E_day1_morning_fitbit_wear | 0.004686 | 2.749412e-06 |
| M_E_day1_daily_survey_complete | 0.002135 | 8.312353e-06 |
| M_E_day2_pageview_morning | 0.001466 | 1.677005e-05 |
| M_E_day2_pageview_afternoon | 0.001389 | 8.782851e-06 |
| M_E_day2_morning_fitbit_wear | 0.001395 | 2.171104e-06 |
| M_E_day2_daily_survey_complete | -0.003530 | 4.751867e-06 |
| M_E_day3_pageview_morning | 0.003017 | 6.717780e-06 |
| M_E_day3_pageview_afternoon | -0.005056 | 5.761374e-06 |
| M_E_day3_morning_fitbit_wear | -0.007633 | 2.022505e-06 |
| M_E_day3_daily_survey_complete | -0.002797 | 1.834510e-06 |
| M_E_day4_pageview_morning | -0.000124 | 1.096528e-05 |
| M_E_day4_pageview_afternoon | 0.004186 | 8.499417e-06 |
| M_E_day4_morning_fitbit_wear | -0.007657 | 1.454928e-06 |
| M_E_day4_daily_survey_complete | 4.582548e-05 | 7.568777e-06 |
| M_E_day5_pageview_morning | -0.005747 | 4.516669e-06 |
| M_E_day5_pageview_afternoon | -0.006152 | 1.083555e-05 |
| M_E_day5_morning_fitbit_wear | 0.003087 | 1.000000e-06 |
| M_E_day5_daily_survey_complete | 0.002719 | 1.000000e-06 |
| M_E_day6_pageview_morning | -0.008505 | 1.380061e-05 |
| M_E_day6_pageview_afternoon | -0.003179 | 2.398375e-06 |
| M_E_day6_morning_fitbit_wear | 0.010247 | 1.000000e-06 |
| M_E_day6_daily_survey_complete | 0.017552 | 4.755885e-06 |
| yesterday_step_count | -0.007060 | 1.952694e-05 |
| ema_step_count | -0.001028 | 5.409846e-06 |
| prior2hour_step_count | -0.003610 | 5.137705e-06 |
| previous7days_rpa | -0.006666 | 4.516986e-06 |
| active_status_fraction_7days | -0.003965 | 1.195383e-05 |
| recent_burden | -0.001270 | 6.373641e-06 |
| salience_yesterday | -0.000175 | 4.650236e-06 |
| walk_interaction_7d | -0.004671 | 1.000000e-06 |
| reserved_context_zero | 0.000000 | 1.000000e-06 |

## Joint Q

| block | feature | mean | variance |
| --- | --- | --- | --- |
| eta | eta_intercept | 0.179179 | 0.033966 |
| eta | eta_E_w | 0.018628 | 0.011341 |
| eta | eta_b_hat | 0.317210 | 0.026988 |
| eta | eta_b_tilde | 0.000000 | 1.000000e-06 |
| beta | beta_intercept | -0.047567 | 0.013566 |
| beta | beta_day_norm | -0.229040 | 0.008190 |
| beta | beta_slot_pm | -0.001465 | 0.003604 |
| beta | beta_E_w | 0.010178 | 0.006570 |
| beta | beta_day_norm*E_w | 0.013302 | 0.002096 |
| beta | beta_slot_pm*E_w | 0.003111 | 0.000416 |
| beta | beta_b_hat | 0.498704 | 0.008873 |
| beta | beta_day_norm*b_hat | 0.260849 | 0.007829 |
| beta | beta_slot_pm*b_hat | 0.067979 | 0.001422 |
| beta | beta_b_tilde | 0.000000 | 1.000000e-06 |
| beta | beta_M_Y_day1_fourSC_morning | 0.005673 | 0.001471 |
| beta | beta_M_Y_day1_fourSC_afternoon | 0.009530 | 0.000915 |
| beta | beta_M_Y_day1_anticipated_affect | 0.138424 | 0.005232 |
| beta | beta_M_Y_day2_fourSC_morning | 0.006951 | 0.000882 |
| beta | beta_M_Y_day2_fourSC_afternoon | 0.003513 | 0.001435 |
| beta | beta_M_Y_day2_anticipated_affect | 0.210818 | 0.001191 |
| beta | beta_M_Y_day3_fourSC_morning | -0.011914 | 0.001493 |
| beta | beta_M_Y_day3_fourSC_afternoon | 0.019879 | 0.000657 |
| beta | beta_M_Y_day3_anticipated_affect | 0.101271 | 0.000954 |
| beta | beta_M_Y_day4_fourSC_morning | 0.028041 | 0.000552 |
| beta | beta_M_Y_day4_fourSC_afternoon | -0.013723 | 0.000538 |
| beta | beta_M_Y_day4_anticipated_affect | 0.115196 | 0.001115 |
| beta | beta_M_Y_day5_fourSC_morning | -0.015359 | 0.001198 |
| beta | beta_M_Y_day5_fourSC_afternoon | -0.002997 | 0.000449 |
| beta | beta_M_Y_day5_anticipated_affect | 0.132188 | 0.001844 |
| beta | beta_M_Y_day6_fourSC_morning | -0.025916 | 0.000526 |
| beta | beta_M_Y_day6_fourSC_afternoon | -0.016555 | 0.000154 |
| beta | beta_M_Y_day6_anticipated_affect | 0.069640 | 0.004101 |
| beta | beta_M_E_day1_pageview_morning | 0.008146 | 0.000934 |
| beta | beta_M_E_day1_pageview_afternoon | -0.003435 | 0.002086 |
| beta | beta_M_E_day1_morning_fitbit_wear | 0.013346 | 0.002896 |
| beta | beta_M_E_day1_daily_survey_complete | 0.029255 | 0.001167 |
| beta | beta_M_E_day2_pageview_morning | 0.007851 | 0.001210 |
| beta | beta_M_E_day2_pageview_afternoon | 0.007530 | 0.000666 |
| beta | beta_M_E_day2_morning_fitbit_wear | -0.028268 | 0.000905 |
| beta | beta_M_E_day2_daily_survey_complete | -0.085304 | 0.000969 |
| beta | beta_M_E_day3_pageview_morning | 0.022884 | 0.000917 |
| beta | beta_M_E_day3_pageview_afternoon | -0.051076 | 0.000385 |
| beta | beta_M_E_day3_morning_fitbit_wear | 0.003667 | 0.000472 |
| beta | beta_M_E_day3_daily_survey_complete | -0.001900 | 0.000549 |
| beta | beta_M_E_day4_pageview_morning | 0.017110 | 0.000307 |
| beta | beta_M_E_day4_pageview_afternoon | 0.032931 | 0.000768 |
| beta | beta_M_E_day4_morning_fitbit_wear | 0.018006 | 0.000615 |
| beta | beta_M_E_day4_daily_survey_complete | -0.007467 | 0.001098 |
| beta | beta_M_E_day5_pageview_morning | -0.020723 | 0.000991 |
| beta | beta_M_E_day5_pageview_afternoon | -0.044264 | 0.000113 |
| beta | beta_M_E_day5_morning_fitbit_wear | -0.011881 | 0.001209 |
| beta | beta_M_E_day5_daily_survey_complete | 0.051232 | 0.000540 |
| beta | beta_M_E_day6_pageview_morning | -0.031661 | 0.000638 |
| beta | beta_M_E_day6_pageview_afternoon | -0.005934 | 0.000247 |
| beta | beta_M_E_day6_morning_fitbit_wear | 0.032858 | 0.001112 |
| beta | beta_M_E_day6_daily_survey_complete | 0.055258 | 0.001899 |
| beta | beta_yesterday_step_count | 0.006173 | 0.000234 |
| beta | beta_ema_step_count | 0.000814 | 0.000119 |
| beta | beta_prior2hour_step_count | -0.007951 | 3.254885e-05 |
| beta | beta_previous7days_rpa | 0.031280 | 0.000601 |
| beta | beta_active_status_fraction_7days | -0.025407 | 0.005157 |
| beta | beta_recent_burden | -0.007672 | 0.000340 |
| beta | beta_salience_yesterday | -0.007134 | 0.000245 |
| beta | beta_walk_interaction_7d | -0.082427 | 0.000906 |
| beta | beta_reserved_context_zero | 0.000000 | 1.000000e-06 |
| beta | beta_A | 0.014527 | 0.000284 |
| beta | beta_A*E_w | 0.006454 | 0.000519 |
| beta | beta_A*b_hat | -0.078328 | 0.002075 |
| beta | beta_A*day_norm | 0.003645 | 0.002338 |
| beta | beta_A*slot_pm | -0.008586 | 0.000802 |
| beta | beta_A*day_norm*E_w | -0.009217 | 0.000300 |
| beta | beta_A*slot_pm*E_w | -0.005372 | 0.000132 |
| beta | beta_A*day_norm*b_hat | 0.040876 | 0.000278 |
| beta | beta_A*slot_pm*b_hat | 0.037018 | 0.000653 |
| beta | beta_A*yesterday_step_count | -0.007236 | 0.000292 |
| beta | beta_A*ema_step_count | -0.000840 | 0.000169 |
| beta | beta_A*prior2hour_step_count | 0.001540 | 0.000253 |
| beta | beta_A*previous7days_rpa | -0.008822 | 5.235692e-05 |
| beta | beta_A*active_status_fraction_7days | -0.019403 | 0.000242 |
| beta | beta_A*recent_burden | -0.001359 | 0.000256 |
| beta | beta_A*salience_yesterday | 0.012339 | 0.000181 |
| beta | beta_A*walk_interaction_7d | 0.022057 | 9.595987e-05 |
| beta | beta_A*reserved_context_zero | 0.000000 | 1.000000e-06 |

## Scalar Noise Variances

| prior | noise_variance |
| --- | --- |
| pf.fourSC | 0.473659 |
| pf.antic | 0.080328 |
| pf.CAE | 0.077142 |
| pf.CAE_short | 0.098195 |
| q_no_td_modify | 0.001748 |
| reward | 1.013745e-06 |
| q_td_modify_joint | 0.002221 |
