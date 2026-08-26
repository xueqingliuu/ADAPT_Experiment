# ADAPR-MRT-Testbed — Senior Engineering Code Review (2026-08-25)

Scope: full trace of config → data extraction → simulator fitting → prior estimation → online RL experiment → aggregation. No files were modified. Line numbers refer to the current working tree. Every HIGH finding was verified against the quoted source; pipeline and prior/STE findings were additionally cross-checked empirically against `env_para_vanilla/df_fit_11week.csv` and the JSON outputs.

---

## 1. Architecture and execution flow

**Fitting pipeline (real MRT data → simulator).** `0_combine_json.py` merges raw exports; `1_data_extraction.py` builds the per-participant decision panel (2 slots/day × 84 days: 4-hour steps, prior-2h steps, pageviews, wear, surveys, walking-suggestion deliveries/interactions, EWMAs); `2_combine_data_frame.py` merges into a two-row-per-day panel with Mon–Sun weeks and drops burn-in/edge weeks; `3_standardization.py` log-transforms and pools z-scores (`std_params.json`); `4_perceived_utility.py` fits the 1-D latent engagement state E_w (weekly emissions J/U1/U2; within-week mediators PV/FW/PJ; transition E_{w+1}=f(E_w, weekly means)) by penalized MLE, and imputes a query-effect block; `5_fit_vanilla_testbed.py` fits Bayesian hierarchical models for 4h steps (fourSC), anticipated affect, weekly CAE/CAE-short, plus per-user ridge/logistic models, writing `env_para_vanilla/params_env_<uid>.json`, `df_fit_11week.csv`, residual pools; `6_est_Ew_weights.py` fits the pooled linear Ê_w the agent can compute from observables.

**Priors.** `est_prior.py` estimates PF mediator/CAE priors, reward-shaping and Q priors (via offline FQI at γ̄∈{0.5,0.9,0.99}), V2/V4 redistribution priors, and a joint (η,β) modified-TD prior; `--loo` writes per-participant held-out bundles.

**Online experiment.** `run_array.sh` → `experiment.py` (300 seeds × 75 participant draws with replacement from 31 fitted users, `nweek=36`). Per draw, all 11 policies (RL V1–V8 variants + never/always/random) run on the same uid with the same `draw_seed`; `vani_env.Env` generates outcomes (AR(1) residual bootstrap, per-user fitted coefficients, closed-loop proxy translation for fourSC, PV hurdle); `OnlineEnv` steps slot-by-slot, maintains agent-visible vs latent variables, and builds weekly PF packets; `algorithm_helpers.py` implements the particle filter (particle learning with per-particle sufficient statistics), RLSVI with AR(1)-discounted exploration noise, reward shaping/redistribution, and the modified-TD bottleneck joint posterior; `agents/` are thin policy wrappers.

**Aggregation.** `aggregate.py` loads per-seed `.npz` files, denormalizes CAE, pairs each policy against never-send on aligned (seed, draw) axes, and reports means with MC SEs clustered by seed (sd of per-seed means / √n_exp).

**STE tuning.** `tune_ste.py` rescales coefficient pathways to hit a target standardized treatment effect (paired-seed MC proxy); `ste_vanilla.py` validates with per-user DiscreteCQL policies.

---

## 2. Findings

Severity legend: C = critical, H = high, M = medium, L = low. "Verified" = code read directly and/or checked empirically.

### H1. Stale `rl_priors.json` everywhere + silent fallback to zero priors (verified)
- **File:** `experiment.py:1465-1476` (`_load_priors`), `env_para_*/rl_priors.json` (identical md5 in every folder, mtime 2026-07-27), `tune_ste.py:130-137` (copies it verbatim into every tuned folder).
- **What it does:** The shared prior file predates the August feature-map rewrite (dims fourSC=24/antic=18/CAE=24/reward=61/q=79 vs current 19/15/4-5/17/26, and it lacks `q_no_td_modify_g09/g099`, `q_redistribution`, `reward_redistribution`). Loading it raises inside `trim_pf_cae_prior`; `_load_priors` catches **any** exception (`except Exception`) and silently returns defaults.
- **Why risky:** Any run with `PRIOR_MODE=saved, USE_ESTIMATED_PRIORS=1` since the rewrite actually used zero/identity priors while `config.json` implies estimated priors were requested. Cross-condition comparisons of "informative-prior" agents in saved mode are invalid.
- **Failure example:** `USE_ESTIMATED_PRIORS=1 python experiment.py --prior-mode saved` → one log line `load_estimated_priors failed: ValueError(...)`, then a full 300-seed batch on zero priors.
- **Fix:** Regenerate `rl_priors.json` per folder with the current `est_prior.py`; narrow the except to `FileNotFoundError`/`JSONDecodeError`; make dimension mismatches fatal.
- **Test:** For each `env_para_*/rl_priors.json`, assert `len(pf.fourSC.nu_0)==len(PF_THETA_FOURSC_NAMES)`, `len(q.mu_0)==P_RL_MICRO`, etc.; assert `_load_priors` raises (not falls back) on a truncated file.

### H2. Walking-suggestion "Interacted" label is a broken heuristic (verified)
- **File:** `1_data_extraction.py:1886-1900` (two-delivery days), `:1928-1941` (one-delivery days).
- **What it does:** Two deliveries: `if len(displays) >= 2: interacted = 1` for **both** slots, regardless of timing. One delivery: `if len(displays) >= 1: interacted = 1`; the `else` branch's timing loop (300/600-min windows) iterates over an empty array — dead code — and references `timestamp_val` leaked from the previous branch.
- **Why risky:** `Interacted` feeds `Interacted_7d_walk`, a fourSC covariate and action-interaction, and the `theta_ws_interaction` model — a mislabeled treatment-engagement mediator propagates into the simulator's action effects.
- **Failure example:** Message displayed 7 a.m., gif delivered 5 p.m. → afternoon slot labeled `Interacted=1`.
- **Fix:** Hoist the timing-window test into a helper applied per delivered slot in all branches.
- **Test:** Synthetic day, one 17:00 delivery + one 07:00 display → assert `Interacted == 0`.

### H3. Days with ≥3 deliveries coded as "no deliveries" (verified)
- **File:** `1_data_extraction.py:1873` (`== 2`), `:1912` (`elif == 1`), `:1994` (`else: # No deliveries`).
- **What it does:** ≥3 rows for a day (resends / near-duplicate timestamps) fall into the no-delivery branch: `WalkingSuggestion = 0` on both slots.
- **Why risky:** Corrupts the treatment indicator itself, silently, exactly on the anomalous days.
- **Fix:** `else` should assert `shape[0] == 0`; handle ≥3 by nearest-slot assignment or raise.
- **Test:** 3 sent rows in one day → both slots `WalkingSuggestion == 1` (or a hard error), never 0.

### H4. `YesterdayStepCount` is actually a two-day lag (verified)
- **File:** `1_data_extraction.py:2846-2849` stores day d's total under `Date = d+1`; `:2856-2861` then applies `shift(1)`, so the column at date D holds steps(D−2).
- **Why risky:** `5_fit_vanilla_testbed.py` fits the fourSC model (`yesterday_step_count` + `WalkingSuggestion×yesterday_step_count`) against a 2-day lag, while the simulator state (`experiment.py:905`) is a true 1-day lag. `vani_env`'s closed-loop proxy translation (`_ridge_preserve_mean`, `vani_env.py:730-820`) re-expresses θ on the 1-day-lag sum-of-z-scores proxy while preserving the *fitted means* — so the simulator is internally consistent, but those fitted means themselves came from a mistimed covariate, so the fourSC dynamics (especially the action×yesterday interaction) are estimated against the wrong day.
- **Fix:** Store with `'Date': date` (drop the `+1`) and keep `shift(1)`; refit 5 → proxies → priors.
- **Test:** 3 synthetic days with distinct totals → merged row at day D has `YesterdayStepCount == steps(D−1)`.

### H5. est_prior antic design uses the wrong regressor (verified)
- **File:** `est_prior.py:664`: `act = _fill_nan(dat_am["active_status"]...)` (daily binary), while the runtime PF design, the env model (`vani_env.gen_antic_mean`), and the original fit all use `active_status_fraction_7days` (smooth [0,1]).
- **Why risky:** The antic prior's mean/covariance for that coordinate — and via ridge non-orthogonality every other antic coefficient — is estimated under a mis-specified design; contaminates the shared file and all 31 LOO bundles.
- **Fix:** One-line change to `active_status_fraction_7days`; regenerate priors + LOO bundles.
- **Test:** Assert design column 2 equals the CSV's `active_status_fraction_7days` AM values.

### H6. One-day panel shift: burn-in of 6, not 7 (superseded)
- **File:** `2_combine_data_frame.py` (`BURN_IN_DAYS`).
- **Original claim:** burn-in of 6 kept a pre-study day and dropped the last study day into week 13.
- **Now:** `BURN_IN_DAYS = 7` cuts at study `date_min`. The 92-day extraction pad is unchanged; week 13 still drops `date_min+84`. `_starts_monday` + `EXCLUDED_WEEKS=(0,13)` already produced Monday-start 84-day panels in current `df_merged.csv`. Re-run script 2 on the next pipeline refit.

### M1. est_prior reward-shaping prior trains on a final week with `E_{w+1}` zeroed
- **File:** `est_prior.py:1053-1056`: `E_next = zeros; E_next[:-1] = E_w_start[1:]; y = R_week + γ̄·E_next` keeps week n_w−1 with a target missing its `γ̄·E_{w+1}` term (mean ≈ 0.73 in the data), unlike the runtime regression (`algorithm_helpers.reward_shaping_week_targets`) and the V2/V4 fits in the same file, which drop it.
- **Fix:** Use `n_w − 1` rows. **Test:** constant-E synthetic user → intercept matches analytic value.

### M2. Tuned STE folders run with vanilla-DGP priors; docs claim otherwise
- **File:** `tune_ste.py:130-137` copies `rl_priors.json`/`df_fit_11week.csv` verbatim; `est_prior.py:121-126` comment claims tuned folders get "their own" priors (impossible — est_prior only reads the unchanged RCT CSV); `env_para_ste0.5_bs_large_bf/loo_priors` is an absolute machine-specific symlink (broken locally); `env_para_ste0.2/0.5/0.8` have no `loo_priors` at all, so `PRIOR_MODE=loo` fails there.
- **Why risky:** In a κ=2.25-scaled environment, priors encode vanilla-size action effects; the prior/DGP mismatch grows with κ and confounds cross-STE comparisons of prior-informed agents.
- **Fix:** Document the "priors from real data" stance and delete the misleading comment, or add a mode that refits priors from tuned-env simulated data; copy `loo_priors` with relative paths.

### M3. Structurally unidentified prior coordinates get near-dogmatic zero priors
- **File:** `est_prior.py:563` (`var = max(var, 1e-6)` with all-zero per-user θ for `b_tilde` columns, which offline designs always set to 0).
- **Why risky:** Online, RLSVI receives N(0, 1e-6) on the belief-uncertainty coefficient — the opposite of the diffuse prior an unidentified coordinate needs; the agent effectively cannot learn a `b_tilde` effect within the horizon.
- **Fix:** Detect zero-variance design columns and assign a diffuse variance (e.g. 1.0).
- **Test:** Assert `diag(Sigma_0)[b_tilde indices] >= 1.0` in generated priors.

### M4. Proxy STE estimand is upward-biased (max over noisy arms, truncated at 0)
- **File:** `tune_ste.py:627-632` (`best = max(arms, key=delta)`, `delta = max(0, ...)`) vs `ste_vanilla.py:995-999` (gate-then-Δ, can be negative).
- **Why risky:** Winner's curse + truncation inflate the calibration target, so `env_para_ste0.5` will validate below 0.5 on average; `--proxy-to-true` defaults to 1.0 and is never set in `run_tune_ste.sh`.
- **Fix:** Split-seed arm selection/estimation, or calibrate against the exact `aggregate_ste` estimand.
- **Test:** Null environment (κ→0 on action pathways) → proxy mean STE ≈ 0 within MC error.

### M5. LOO prior installation skips the dimension asserts
- **File:** `experiment.py:1619-1687` (`_apply_fitted_loo_priors`) vs the assert block in `_configure_priors` (`:1576-1605`).
- **Why risky:** A LOO bundle generated under a different `ACTION_BLOCK_C` or feature map dies later in a matmul deep inside RLSVI instead of failing with a clear message at load — or, worse, broadcasts.
- **Fix:** Factor the asserts into a helper called from both paths. **Test:** truncated `q_no_td_modify.mu_0` in a held-out JSON → clear `ValueError` at configure time.

### M6. "prior2hour" step count is a 30-minute window gated by a 2-hour wear check
- **File:** `1_data_extraction.py:2908-2909` (`start_window = end_window − 0.5h`) vs the 2-hour wear gate (`:2389-2396`); every downstream name says 2 hours.
- **Why risky:** Internally consistent (fit and simulation share the column), but the scientific label is wrong by 4×, and the mismatched gate produces NaNs when the outcome window itself was fully observed.
- **Fix:** Change to 2h or rename the chain and align the gate. **Test:** constant 10 steps/min → assert 300 vs 1200 against the intended spec.

### M7. Fit-vs-simulate imputation mismatch in CAE mediator EWMAs
- **File:** `5_fit_vanilla_testbed.py:1464-1472` fills missing slots with the whole-trajectory mean (uses future rows) before the within-week EWMA; the simulator/PF helper `vani_env.py:340-346` fills with the week-local mean. Same observed week ⇒ different `fourSC_ewma` at fit vs simulation time; biases the CAE transition for sparse-wear users.
- **Fix:** Adopt the week-local fill in script 5 and refit. **Test:** week `[1, NaN×13]` in a trajectory with global mean ≠ 1 → both paths equal.

### M8. Historical PF design row 0 reads current mutable state (my finding, verified)
- **File:** `experiment.py:1107-1113` (`_pf_foursc_row`): for `step_idx == 0` the AR-1 lag falls back to `self.s["stepCountNext4HourLag1"]` — but `self.s` is mutated every slot, so when cumulative PF rows are rebuilt at weeks k≥2, the week-0 Monday-AM row's lag column contains the **most recently generated** 4-hour step count, not the initial lag.
- **Why risky:** A historical design row changes value from week to week and injects a future outcome into a past covariate. One row among hundreds, so the PF posterior bias is small — but it is a genuine use-of-future-information defect and makes the cumulative design non-reproducible across weeks.
- **Failure example:** At k=10, `fourSC_row_fn(0)` uses Saturday-week-9's fourSC as week-0 Monday's lag.
- **Fix:** Cache `self._initial_foursc_lag1` at `_reset_episode_state` and use it for `step_idx == 0`.
- **Test:** Call `_pf_foursc_row(0)` at k=2 and k=10 within one episode; assert identical rows.

### M9. Positional `reshape(-1, 14)` week-chunking with no alignment guards
- **File:** `5_fit_vanilla_testbed.py:1458-1463`, `:1615`, `:1105`; `est_prior.py:631`, `:712`, `:869` (silent `len//14` truncation).
- **Why risky:** Today every (user, week) has exactly 14 rows (verified), but a mid-panel partial week would either crash (script 5) or silently shift every subsequent week's slot alignment (est_prior), misassigning Sundays and weekly rewards.
- **Fix:** Group by the actual `week`/`Date` columns, or `assert (df.groupby([uid,'week']).size() == 14).all()` in every loader.
- **Test:** Delete one CSV row → loader raises.

### M10. Weekly ISO-week join attaches current-week survey values to earlier days
- **File:** `2_combine_data_frame.py:171-203`; plus the ±2-day slot tolerance in `fill_weekly_12` (`1_data_extraction.py:648-649`).
- **Why risky:** No decision-time model currently consumes `CAE_avg`/`week_present` at the daily level, so no leakage materializes today — but the per-slot columns exported in `df_fit.csv` are a leakage trap for any future decision-time model.
- **Fix:** Document the emission-only contract or join weekly values only from the observed date onward.

### M11. Fixed-denominator weekly averages code missingness as zero engagement
- **File:** `4_perceived_utility.py:645-647`, `6_est_Ew_weights.py:107-123`, mirrored in `vani_env._week_means_from_arrays` and `agents/ew_hat.py`.
- **Why risky:** Documented convention and self-consistent in simulation, but "missing = not engaged" is wrong for FW (sensor missingness ≠ not wearing); systematically deflates E_w inputs for sparsely observed users at fit time.
- **Fix/decide:** Confirm intent; at minimum document that FW inherits it knowingly.

### M12. `cleansed_simulator.py` / `aligned_base_v1.py` cannot run from this repo (my finding)
- **File:** `cleansed_simulator.py:8-13` imports `high_fidelity_simulator` (not in the repo) and hardcodes `PACKAGE_ROOT/xq_original/ADAPT_RCT_Experiment-main`.
- **Why risky:** The "cleansed/aligned reference simulator" used for fidelity diagnosis is not reproducible from this repository; nobody can re-verify the alignment claims.
- **Fix:** Vendor `high_fidelity_simulator.py` (and the path base) or move these files out with a README pointer.

### M13. `day_norm` re-derivation off-scale in script 5
- **File:** `5_fit_vanilla_testbed.py:73-82`: 0-based `day` on a 77-day panel against an 84-day normalizer → `day_norm ∈ [−1.024, 0.807]` (verified). Not in any current design matrix, but exported via `df_fit_11week.csv` for downstream consumers.
- **Fix:** Correct the formula or drop the column. **Test:** `between(-1, 1).all()` and symmetric range.

### L1. Contradictory documentation of the never/always baselines
- **File:** `algorithm_helpers.py:609-612` ("clipped always/never baselines... same policy class"), `experiment.py:1976-1983` (docstrings "π_A = ε / 1−ε") vs actual hard 0/1 (`experiment.py:1935-1940`, comment at `:1353-1354`, aggregate labels).
- **Risk:** Readers/paper text may claim the baselines lie in the ε-clipped policy class; they do not (they bound performance more loosely). Fix the docstrings or actually clip.

### L2. Dead E_w bootstrap path + wrong comment
- **File:** `experiment.py:279-280` says the fallback is 0.0; `agents/ew_hat.py:19` is 2.0. `df_fit_full` is never passed by any runner, so `initial_Ew_hat_for_user`'s table logic is dead and every participant starts at Ê₀ = 2.0. Fix comment; either wire `df_fit_full` or delete the path.

### L3. "MRT-end" seeding docstrings are wrong; mixed start-vs-end conventions
- **File:** `experiment.py:54-56`, `:1069-1075`, `:1086-1092` claim EWM windows are left-padded with the *MRT-end* value; `make_initial_state`/`slot_ema_initials_from_df_fit` (`vani_env.py:2142-2228`) load the **first** row / first finite value. Meanwhile `expTool1/2` are read from the **last** row (`vani_env.py:2224-2227`). Behavior (restart at study entry) looks intended — the docs and the tool-survey seed convention should be reconciled.

### L4. Dead/incorrect statistics code in aggregate.py
- **File:** `aggregate.py:284-294` (`compute_stats["se"]` = sd/√(n_exp·n_users), ignores participant clustering — unused), `:307-318` (`_per_uid_mean` unused), `:567-626` (`make_gamma09_minus_never_plot` never called; its SE also uses √(n_exp·n_users)). All *produced* figures/tables use the correct seed-clustered `_se_across_replications`. Delete the dead code before someone resurrects it. Also: `_se_clustered_by_user` (`:629`) clusters by experiment despite its name; `DEFAULT_RESULTS_ROOT` (`:17`) is `results_vanilla_loo` while the help text (`:255`) says `results_vanilla`.

### L5. `run_uids` loading tied to the first algorithm's file presence
- **File:** `aggregate.py:764` collects `run_uids` only while looping the first algorithm; a folder missing that one `.npz` misaligns `all_uids` with other algorithms (the shape check catches most, not all, cases). Load `run_uids.npy` from the run dir instead.

### L6. `ewma_gamma` is calendar-blind under missingness; docstring overclaims
- **File:** `ewm_utils.py:3-4`, `:51-60`. Dropping NaNs before positional weights time-warps the decay and changes γ with the missing count; the claimed pandas equivalence holds only for fixed γ and no NaNs. Used symmetrically on fit and sim sides, so impact is interpretability.

### L7. Legacy duplicate tree `RL_agents/`
Contains its own copies of `algorithm_helpers.py`, `rl_priors.json`, `std_params.json`, not imported by the experiment. Guaranteed to drift; archive or delete.

### L8. Misc (verified by delegated review)
`est_prior.py:38` stale docstring (`mu_0_reward_unbiased` never produced); `est_prior.py:266` ridge σ² uses centered variance with ddof=1 instead of RSS/(n−p); `tune_ste.py:170-184` burden_shift subtracts κ from A×E_w interactions so the "fatigue" shift changes sign for E_w < −1; `ste_vanilla.py:764-777` eval metadata does not record the params dir (cross-env train/eval passes silently); `tune_ste.py:692-703` NaN-STE users silently dropped from the mean but counted in `n_users`; timezone fallback to UTC wall clock and past-midnight bedtimes → empty wear windows (`1_data_extraction.py:220-224`, `:1470-1471`, `:2237-2238`); `0_combine_json.py:384` `v.get("value") or ...` drops literal-0 heart-rate values; `impute_query_effect.py:75` unseeded default RNG (current callers pass one); `vani_env.gen_pageview` hurdle consumes a data-dependent number of global-RNG draws (weakens within-episode common-random-numbers alignment across policies; comparisons remain valid — episode-level pairing is what the aggregation relies on).

---

## 3. Prioritized findings table

| # | Sev | File:Line | Issue |
|---|-----|-----------|-------|
| H1 | High | experiment.py:1465; env_para_*/rl_priors.json; tune_ste.py:130 | Stale prior files + silent broad-except fallback to zero priors |
| H2 | High | 1_data_extraction.py:1886, 1928 | Interacted label: dead timing logic, any-display ⇒ interacted |
| H3 | High | 1_data_extraction.py:1873-1994 | ≥3 deliveries/day coded as WalkingSuggestion=0 |
| H4 | High | 1_data_extraction.py:2846-2861 | YesterdayStepCount is a 2-day lag; fit vs simulator timing mismatch |
| H5 | High | est_prior.py:664 | Antic prior fit on `active_status` instead of 7-day fraction |
| H6 | High/Med | 2_combine_data_frame.py | BURN_IN_DAYS=7 (was 6); week 13 still drops pad day `date_min+84` |
| M1 | Med | est_prior.py:1053-1056 | Reward prior off-by-one (last week with E_next=0) |
| M2 | Med | tune_ste.py:130; est_prior.py:121 | Tuned-STE folders reuse vanilla priors; comment claims otherwise; broken/missing loo_priors |
| M3 | Med | est_prior.py:563 | b_tilde prior variance floored at 1e-6 (dogmatic zero) |
| M4 | Med | tune_ste.py:627-632 | Proxy STE = truncated max over arms (upward bias) |
| M5 | Med | experiment.py:1619-1687 | LOO prior install skips dimension asserts |
| M6 | Med | 1_data_extraction.py:2908 | "prior2hour" is a 30-min window; wear gate mismatched |
| M7 | Med | 5_fit:1464 vs vani_env.py:340 | Fit/sim EWMA NaN-fill mismatch for CAE mediators |
| M8 | Med | experiment.py:1107-1113 | Historical PF row 0 lag reads current state (future info) |
| M9 | Med | 5_fit:1458; est_prior.py:631,869 | Unguarded reshape(-1,14) week chunking |
| M10 | Med | 2_combine_data_frame.py:171-203 | Weekly values joined to pre-survey days (leakage trap) |
| M11 | Med | 4:645; 6:107 | Missing = zero engagement in weekly averages (FW dubious) |
| M12 | Med | cleansed_simulator.py:8-13 | Reference simulator not importable from this repo |
| M13 | Med | 5_fit:73-82 | day_norm off-scale in exported CSV |
| L1–L8 | Low | (see §2) | Doc contradictions, dead stats code, duplicates, misc fragilities |

## 4. Five highest-value fixes

1. **Regenerate `rl_priors.json` in every `env_para_*` folder and make `_load_priors` fail loudly on dimension mismatch** (H1) — until then, no "saved-mode estimated prior" result is what it claims to be.
2. **Fix the Interacted/WalkingSuggestion extraction (H2 + H3) and refit the pipeline** — these corrupt the treatment variable and a treatment-interaction mediator at the source.
3. **H4 (YesterdayStepCount) and H6 (`BURN_IN_DAYS=7`) are in code; refit 1 → 2 → 3 → 4 → 5 → 6 → est_prior** so the CSVs match.
4. **`est_prior.py`: `active_status` → `active_status_fraction_7days` (H5), drop the zero-E_next week (M1), diffuse-prior unidentified coordinates (M3); regenerate shared + LOO bundles.**
5. **Cache the initial fourSC lag for PF row 0 (M8) and factor the prior-dimension asserts into the LOO path (M5)** — small changes that remove the only future-information defect I found in the online loop and the largest silent-failure surface.

## 5. Components inspected and found sound

- **Temporal ordering of the online loop** (`experiment.py`): `get_context` exposes only strictly-past slot mediators and previous-day daily mediators; week-start states carry only Ê_k computed from week k−1 observables + that Sunday's survey; RLSVI refits after day 0 use only recorded snapshots. No use of latent E_w/CAE by any agent; latent vs observed vs agent-visible (LOCF) anticipated-affect streams are kept cleanly separate.
- **Reward/return identities**: both reward-shaping variants satisfy Σ Δ_{d,t}(r + 1_terminal·R_add) = Δ_{5,1}·Y_w exactly (verified algebraically for `build_rl_training_data_with_rewardshaping` and the bottleneck twin); V2/V4 redistribution adds the terminal leftover so weeks sum to the weekly target; V1/V3 bump lands on the correct terminal row index.
- **Discount placement**: γ̄ only on the terminal slot (`_gamma_dt_micro`), matching est_prior's FQI and ste_vanilla's transition picker.
- **PF particle-learning update** (`algorithm_helpers.ParticleFilterRuntime`): per-particle sufficient statistics, correct lagged/contemporaneous CAE substitution, J/I gating of Y vs tilde-Y likelihoods, ESS-triggered resampling with weight reset, per-particle (not latent-truth) responses for inactive weeks — explicitly avoiding leakage of unobserved CAE.
- **Reward alignment**: terminal reward `b_hat[kp+1]` is the belief of week kp's CAE (b_hat[k] indexes the belief formed Monday of week k about week k−1) — consistent everywhere it is consumed.
- **Posterior sampling / action probabilities**: RLSVI closed-form posteriors with AR(1)-discounted exploration noise; ensemble softmax/vote → `clip_prob` to [ε, 1−ε]; double-Q target-net argmax.
- **Pairing and seeds**: uid table pre-drawn with `default_rng(0)` for all 300 experiments; every policy runs the same (uid, draw_seed); week-0 actions and query paths shared via a salted SeedSequence independent of agent RNG consumption; env noise on the reseeded global stream, agents on private Generators.
- **Aggregation statistics actually produced**: paired differences on aligned (seed, draw) axes; SE = sd of per-seed means / √n_exp (a correct MC SE treating the 31 users as fixed); week indexing and π_A week-0 exclusion are consistent.
- **LOO priors**: genuine exclusion of the held-out participant (verified by the delegated review down to file diffs); per-draw installation order in the main loop is correct.
- **`vani_env` transforms**: standardize/invert round-trips, name-validated coefficient loading with explicit legacy-layout trims, residual AR(1) bootstrap with population-pool blending for tiny residual pools.

## 6. Questions / assumptions needing your confirmation

1. **Saved-mode runs since ~Aug 14**: were any published results produced with `PRIOR_MODE=saved, USE_ESTIMATED_PRIORS=1`? If yes, they used zero priors (H1) and need rerunning. LOO-mode results (`results_*_loo`) are unaffected.
2. **Is "priors fitted from the real RCT data even for tuned-STE environments" the intended design** (M2), or should tuned folders get priors matched to their DGP?
3. **The simulation restarts each participant at study entry** (first-row states, Ê₀=2.0, pre-study CAE baseline) while docstrings say "MRT-end" and `expTool1/2` come from the last row (L3) — which convention is intended?
4. **Never/always baselines**: hard 0/1 or ε-clipped (L1)? The paper text should match whichever the code does.
5. **FW missing = not worn** (M11): deliberate convention or should sensor-missing days be excluded from the FW denominator?
6. **`fittedQ.py` and the `ipynb/` analyses were excluded** from this review by agreement with the stated scope — say the word if they feed any reported result.
7. **`aligned_base_v1.py` freeze**: is `high_fidelity_simulator.py` supposed to live in this repo (M12)?
