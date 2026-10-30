# Participant 345 stability

Folder `env_para_vanilla_c47`. Comparison user 388 (median max g^Y = 0.709).

## a) Gain decomposition

g^Y never=1.0002  always=1.0033. g^E never=-0.0445  always=0.0308. Skeleton growth never=0.000 always=0.000 (d_first never=8.83e-09).

| term | 345 never | 345 always | median-g^Y user never | median always |
| --- | ---: | ---: | ---: | ---: |
| alpha1^Y | 0.8920 | 0.8920 | 0.6380 | 0.6380 |
| alpha_SC^Y D_SC | -0.0089 | -0.0088 | 0.0030 | 0.0029 |
| alpha_AA^Y D_AA | 0.1171 | 0.1200 | 0.0589 | 0.0680 |
| g^Y | 1.0002 | 1.0033 | 0.6999 | 0.7089 |

| mediator | policy | s_k | r_k | D_k |
| --- | --- | ---: | ---: | ---: |
| SC | never | 0.1012 | 0.6354 | 0.2775 |
| AA | never | 0.2360 | 0.3690 | 0.3740 |
| SC | always | 0.1113 | 0.5935 | 0.2738 |
| AA | always | 0.2420 | 0.3690 | 0.3835 |

The term that dominates g^Y for 345 is **alpha1^Y** (alpha1^Y = 0.892 vs population mean 0.639).

## b) Coefficients vs the random-effect distribution

Participant-level values are the stored hierarchical posterior means (beta + b_i). Population mean / RE SD are the across-user mean and SD of those means (MCMC traces were not persisted). The BLUP column is the Gaussian random-intercept posterior SD using that RE SD and this participant's residual SD.

| CAE coef | 345 | pop mean | RE SD | z | BLUP SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| intercept | -0.174 | -0.096 | 0.102 | -0.77 | 0.098 |
| CAE_avg_lastweek | 0.892 | 0.639 | 0.133 | 1.90 | 0.057 |
| fourSC_ewma | -0.032 | 0.029 | 0.031 | -1.99 | 0.028 |
| anticipated_affect_ewma | 0.313 | 0.306 | 0.034 | 0.22 | 0.034 |

| mediator coef (s_k / r_k) | 345 | pop mean | RE SD | z |
| --- | ---: | ---: | ---: | ---: |
| fourSC CAE_avg_lastweek | 0.023 | 0.025 | 0.037 | -0.05 |
| fourSC fourSC_lag1 | 0.485 | 0.055 | 0.169 | 2.55 |
| fourSC yesterday_step_count | 0.049 | 0.077 | 0.088 | -0.32 |
| fourSC prior2hour_step_count | 0.177 | 0.148 | 0.110 | 0.27 |
| fourSC anticipated_affect_yesterday | 0.209 | 0.175 | 0.024 | 1.41 |
| fourSC WalkingSuggestion_by_CAE_avg_lastweek | 0.039 | 0.020 | 0.012 | 1.49 |
| fourSC WalkingSuggestion_by_yesterday_step_count | -0.080 | -0.038 | 0.061 | -0.70 |
| fourSC WalkingSuggestion_by_prior2hour_step_count | 0.034 | -0.014 | 0.028 | 1.72 |
| fourSC WalkingSuggestion_by_anticipated_affect_yesterday | -0.077 | -0.097 | 0.012 | 1.63 |
| antic CAE_avg_lastweek | 0.236 | 0.128 | 0.059 | 1.83 |
| antic anticipated_affect_yesterday | 0.369 | 0.324 | 0.036 | 1.27 |
| antic A0_morning_by_CAE_avg_lastweek | -0.017 | 0.010 | 0.015 | -1.88 |
| antic A1_afternoon_by_CAE_avg_lastweek | 0.024 | 0.005 | 0.019 | 0.99 |

345's alpha1^Y is 1.90 RE SDs above the 47-user mean. Residual SD on stored resid_CAE is 0.344.

## c) Data (weekly summaries, no raw rows)

Observed full-CAE weeks in the 11-week panel: 7 of 11. Within-person SD of CAE_avg_norm: 0.352. Linear week slope on observed weeks: 0.024 per week. Simulator baseline Y_0 = -3.327 (backcast_1w; this is the pre-panel lag, not the first panel week). Clip limits [-3.378, 1.886]. The first panel week is observed at the lower clip.

CAE_avg_norm by week: -3.378, NA, -2.568, -2.649, NA, NA, -2.568, -2.568, -2.568, -3.216, NA

Mbar_SC by week: 0.470, 0.588, 0.340, -0.085, NA, NA, -4.848, NA, NA, NA, NA

Mbar_AA by week: -1.000, -0.333, -0.667, -0.667, NA, NA, -0.667, -0.667, -0.833, -0.333, -0.833

OLS on 7 observed weeks, no week term: alpha1 = 1.143 (SE 1.914).
OLS with linear week: alpha1 = 1.300 (SE 3.399), week = -0.008 (SE 0.126).
The week coefficient is consistent with zero; a linear trend is not absorbing persistence. Seven weeks cannot identify alpha1 (SE > 1), so the hierarchical 0.892 is a noisy individual likelihood shrunk toward the population 0.639, still 1.9 RE SDs high.

## d) Anticipated-affect persistence r_AA

r_AA = 0.369 (never) / 0.369 (always). D_AA = 0.374 / 0.384. Not near 1; D_AA is not exploding.

## e) Never-send simulation (engagement off, 36 weeks, 20 seeds)

Share of weeks at lower CAE clip: 26.0%. At upper clip: 0.0%. Path quantiles (5, 25, 50, 75, 95): -3.378, -3.378, -3.191, -3.016, -2.711.

## f) Data / code checks

Rows 154, unique Date×slot 154, duplicate slots 0, CAE_avg_lastweek_norm NaNs 56 (mean-filled as predictors in the CAE MixedLM, matching 07; outcomes with missing CAE_avg_norm are dropped). Reconstructed (CAE_avg - shift)/scale vs CAE_avg_norm max |err| = 0.0000.

No data or code error found. Missing week-1 CAE is handled by the documented backcast; imputed last-week predictors are the MixedLM fill, not imputed outcomes. 345 is an AR-heavy posterior mean sitting near the CAE floor, which makes |g^Y| ≈ 1 and the skeleton look dead (both paths clip).

## RL priors

`09_est_prior.py` / `est_prior.py` estimate RLSVI priors from `df_fit` regressions, not from simulator coefficients, so they do not need a rerun after a gain cap on `params_env_*.json`.

