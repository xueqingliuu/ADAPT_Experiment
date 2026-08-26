# Re-review after today's changes (2026-08-25, evening tree)

Scope: re-verification of every finding from the morning review against the current working tree (files modified 19:48–20:28), plus a deep pass on the reworked query-effect chain (`impute_query_effect.py`, `vani_env` J-gating, `experiment._jw_week`, φ-map changes). Excluded per your instruction: `fittedQ.py`, `aligned_base_v1.py` / `high_fidelity_simulator`, `RL_agents/`. No files modified.

---

## 1. Status of all prior findings

| # | Prior finding | Status now |
|---|---|---|
| H1 | Silent broad-except fallback to zero priors | **OPEN — and hotter** (see N4 below): `experiment.py:1487` unchanged, and today's φ change re-staled *every* prior bundle |
| H2 | Interacted label heuristic | **FIXED** — `_interacted_after_delivery` applies the 300/600-min window per delivered slot in all branches (`1_data_extraction.py`) |
| H3 | ≥3 deliveries coded as untreated | **FIXED** — nearest-slot assignment with a printed warning (`n_sent >= 3` branch, ~`1_data_extraction.py:2042-2078`) |
| H4 | YesterdayStepCount 2-day lag | **FIXED** — stored under the measurement day (`'Date': date`, `:2941`) with a comment forbidding the old D+1 convention |
| H5 | est_prior antic on binary `active_status` | **FIXED** — `active_status_fraction_7days` (`est_prior.py:765`) |
| H6 | BURN_IN_DAYS=6 one-day panel shift | **FIXED** — `BURN_IN_DAYS = 7` so the combine cutoff is study `date_min`, not `date_min−1`. Extraction is still a 92-day pad (`date_min−7 … +84`); week 13 drops the extra Monday at `date_min+84`. Current `df_merged.csv` (31 users) is already Monday-start, 84 days / weeks 1–12 — the old “pre-study Sunday kept, last study day in week 13” claim was already undone by `_starts_monday` + `EXCLUDED_WEEKS=(0,13)`. |
| M1 | Reward prior trained with zeroed E_next | **FIXED** — `n = n_w − 1`, docstring documents the drop (`est_prior.py` `fit_reward_prior`) |
| M2 | Tuned-STE folders reuse vanilla priors, misleading comment | **RESOLVED as documented design** — `tune_ste.py:145-147` now states it explicitly; `loo_priors` installed as a relative symlink. Your Q2 decision (simulate-then-fit vs keep) still pending |
| M3 | b_tilde prior variance floored at 1e-6 | **FIXED** — `UNIDENTIFIED_PRIOR_VAR = 1.0` applied to structurally all-zero design columns (`est_prior.py:166-175, 295-330`) |
| M4 | Proxy STE = truncated max over arms | **OPEN** — `tune_ste.py:677-681` unchanged |
| M5 | LOO prior install skips dim asserts | **OPEN** — `_apply_fitted_loo_priors` (`experiment.py:1631-1699`) still has no shape checks; consequence upgraded by N4 |
| M6 | "prior2hour" was a 30-min window | **FIXED** — `PRIOR_2HOUR_LOOKBACK = 2h`, half-open `[end−2h, end)` shared with the wear gate (`1_data_extraction.py:2225-2236`) |
| M7 | Fit/sim EWMA NaN-fill mismatch | **FIXED** — script 5 now uses the shared `vani_env.cae_mediator_ewma_rows` (week-local fill); old global-mean fill commented out |
| M8 | PF week-0 row read current mutable lag | **FIXED** — frozen `_initial_foursc_lag1` captured at init and used for `step_idx == 0` (`experiment.py:412-413, 1133-1137`) |
| M9 | Unguarded `reshape(-1, 14)` | **FIXED** — `assert_complete_week_slots` (defined `vani_env.py:310`) called in the script-5 loaders and five est_prior builders |
| M10 | Weekly ISO join puts survey values on pre-survey days | **OPEN (unchanged)** — note it is now *load-bearing*: the new J-effect estimation reads `week_present` off every row of the week. Fine as a week-level attribute; remains a leakage trap for any decision-time model |
| M11 | Missing FW = not worn | **CLOSED per your decision** — now documented in `4_perceived_utility.build_user_blocks` and `6_est_Ew_weights.py` |
| M12 | cleansed/aligned simulators not importable | **Dropped from scope** per your instruction |
| M13 | `day_norm` off-scale | **OPEN** — `5_fit_vanilla_testbed.py:74-83` still normalizes a 0-based 77-day index by the 84-day constants; range ≈ [−1.02, 0.81] |
| L1 | never/always docstring contradictions | **FIXED** (`experiment.py:1365`, `algorithm_helpers.py:610`) |
| L2 | Ê₀ fallback comment 0.0 vs 2.0; dead path | **FIXED** — comments now say 2.0 and note no runner passes `df_fit_full` |
| L3 | "MRT-end" seeds; expTool from last row | **FIXED** — `make_initial_state` reads Exp-tool-1/2 from the *first* row (`vani_env.py:2288-2292`); study-entry convention consistent |
| L4/L5 | Dead unclustered-SE code; run_uids tied to first algo | **FIXED** — dead functions removed; `run_uids.npy` loaded per run dir (`aggregate.py:532-540`). Remaining nit: `DEFAULT_RESULTS_ROOT` is `results_vanilla_loo` while the `--results-root` help says `results_vanilla` |
| L6 | `ewma_gamma` calendar-blind under NaN | **FIXED** — NaN slots keep calendar position, excluded from num/denom; docstring corrected (`ewm_utils.py:51-71`) |
| L8 (misc) | ridge σ² centered/ddof; PV hurdle CRN draw count | **FIXED** — `RSS/(n−p)` (`est_prior.py:277-289`); `gen_pageview` always consumes one uniform + one choice (`vani_env.py:1720-1724`). Still present (low): `gen_ws_interaction` consumes no draw when `Ah=0`; `_interacted` two-delivery branch assigns AM/PM by sort order rather than nearest slot |

Everything marked FIXED was verified by reading the new code (and, where cheap, empirically).

---

## 2. The reworked query effect — findings

The mechanism changed from "sampled/imputed I-gated query block" to: **estimate a J_w (week_present) contrast on PV/FW/PJ from the MRT data** (`impute_query_effect.py`, two-step on base-model residuals, per-user where the user has J variation, pooled otherwise) and **apply `+ J_w·(q@z)` in the simulator whenever the simulated week's J = 1** (`vani_env._ml_query_applied_shift`, `experiment._jw_week`; I_w no longer gates mediators; the I·J feature `query_x_weekly_present` was removed from all φ maps). The chain is coherent end-to-end in structure — same z = [1, E, rb], same arms in estimation and application, hurdle Δp translation preserved, idempotent suffix rewrite, est_prior uses the shared φ builders so priors will match after regeneration. Four substantive issues:

### N1. HIGH — The two-step estimator recovers only ≈ (1−p̄) of the J contrast and inflates the J=0 arm
- **Files:** `impute_query_effect.py:145-150` (`z_row` = `[J, J·E, J·rb]` — no un-gated block), `:92-106` (`_ridge_solve` on base-model residuals), `:109-125` (same structure for the FW/PJ one-step logit).
- **What happens:** The base PV/FW/PJ fits (script 4) contain no J term, so the base θ absorbs ≈ p̄·(J effect) into the intercept/E coefficients (p̄ = P(J=1)). The residuals are then ≈ (1−p̄)·effect on J=1 rows and −p̄·effect on J=0 rows. Because Z has **no un-gated [1, E, rb] block**, the J=0 rows contribute nothing to the fit, so q̂ ≈ (1−p̄)·effect. In simulation: the J=1 arm comes out right (base + q̂ ≈ truth), but the J=0 arm keeps the inflated base — biased up by p̄·effect — so the simulated J-contrast is attenuated to ≈ (1−p̄) of the identified contrast, and the marginal engagement level is inflated by ≈ p̄(1−p̄)·effect.
- **Magnitude:** pooled p̄ = 0.663 (226/341 user-weeks in `df_fit_11week.csv`) → the simulator carries only ≈ **34%** of the identified contrast; per-user fits attenuate by (1−p̄ᵤ) with median p̄ᵤ = 0.73 (8 of 31 users have p̄ᵤ = 1 and fall back to pooled).
- **Verified numerically** with your exact estimator: true contrast (0.30, 0.10, 0.0) → q̂ = (0.102, 0.031, 0.004) ≈ (1−p̄)·e; J=0-arm bias +0.194 ≈ p̄·e. Adding the un-gated main-effect block to Z recovers (0.297, 0.092, 0.011).
- **Fix (small):** augment the two-step design to `Z = [J·1, J·E, J·rb, 1, E, rb]`; keep the J-block as the query suffix and **fold the un-gated block's coefficients into the base θ** (intercept, E, recent-burden coefficients all exist in the PV/FW/PJ bases). Same one-step form works for the logits. Alternative (cleaner, more work): put J·z into the script-4 emission models and fit jointly.
- **Test:** synthetic DGP `med = Xβ + J·e + ε` with J ~ Bern(0.66); assert the estimated suffix recovers `e` within tolerance and the reconstructed J=0 mean equals `Xβ`.

### N2. MEDIUM — One-week timing mismatch between estimation and application of the J gate
- **Files:** `impute_query_effect.py:137,146` (regressor = `week_present` **on the same week's rows**, i.e. the survey at the END of week w, per the 2_combine join); `experiment.py:553-558, 799, 889, 966` (`_jw_week(sim_w)` = `wp_all[sim_w]`, the survey drawn at the END of week sim_w−1).
- **Why it matters:** Physically, the estimation measures "week-w engagement vs completing week-w's own Sunday survey" (contemporaneous, both emissions of E_w); the simulator applies q to week-k mediators gated on the *previous* Sunday's completion. A contemporaneous end-of-week gate is causally impossible in a sequential simulator, so this had to lag somewhere — but then the estimand should match: the honest estimator for what the simulator applies is the **lagged** contrast (regress week-w mediators on `week_present_{w−1}`).
- **Interaction with N3:** the code partially compensates by drawing that Sunday's J from the *post-transition* E (see N3), which makes sim-J as informative about the coming week as data-J is about its own week. The pair (N2, N3) is internally consistent under the code's labeling, but then the CAE gate and the Ê bundle (which follow the physical labeling — see below) sit one convention apart. Pick one story.
- **Fix options:** (a) re-estimate q on lagged `week_present` (one-line change: shift within participant) and keep the sim as is; or (b) document the emission-labeling convention explicitly and keep contemporaneous q, accepting that E's autocorrelation carries the difference.
- **Test:** in a long simulated trajectory, regress simulated week-k mediator residuals on `wp_all[k]`; the recovered coefficient should match the q used to generate — then repeat the same regression on the MRT layout (J of the same calendar week) and confirm which estimand your paper claims.

### N3. MEDIUM (pre-existing, now more load-bearing) — Sunday J/U1/U2 are emitted from the post-transition E
- **File:** `experiment.py:1035-1056` (`_finalize_week`): `pu = gen_perceivedUtility(...)` computes E_{w+1} from week-w mediators, and *then* `wp_all[w+1] = gen_week_present(pu)`, `u1/u2 = gen_tool_*(pu)` — the survey physically completed at the end of week w is drawn from E_{w+1}, whereas script 4 fits J_w, U_w ~ E_w (`4_perceived_utility.py:761-767`, emission block).
- **Consequences:** θ_J and θ_U were estimated as emissions of the within-week state but are applied to the freshly transitioned one. It also props up N2 (makes the lagged gate behave like a contemporaneous one). Under the *physical* mapping everything else is right: this Sunday's J correctly gates CAE_w (`get_week_packet`, `wp_all[w+1]` gating `CAE_all[w+1]` = CAE of week w) and correctly enters Ê_{w+1} with week-w mediators (`compute_Ew_hat_from_week`, `survey_idx = sim_w+1`).
- **Fix:** draw J/U1/U2 from the pre-transition `s["perceivedUtilityLastWeek"]` (E_w) in `_finalize_week` before overwriting it with the new pu. That makes emission (script 4), CAE gating, and the Ê bundle all consistent under the single physical mapping — and then N2's fix (a), lagged-J estimation, matches the mediator gate exactly.
- **Test:** simulate long trajectories and check corr(J at end of week w, E_w) vs corr(J, E_{w+1}) against the fitted θ_J emission structure.

### N4. HIGH (operational) — today's φ change re-stales every prior bundle; the failure modes are the bad ones
- **What changed:** removing `query_x_weekly_present` shrank the feature maps: current dims are **q=25, reward=16** (verified by import); the Aug-23 LOO bundles are 26/17 and the Jul-27 shared `rl_priors.json` is 79/61.
- **Consequences with the current code:** `--prior-mode loo` (the CLI default) loads 26-dim priors without any shape check (M5 still open) and dies mid-episode in a matmul at week 1, after the PF has already run; `--prior-mode saved` with `USE_ESTIMATED_PRIORS=1` silently falls back to zero priors (H1's `except Exception`, `experiment.py:1487`).
- **Fix:** after the pipeline refit, rerun `python est_prior.py` and `--loo` for every folder you keep; narrow the except to `FileNotFoundError`/`JSONDecodeError`; factor the `_configure_priors` assert block into a helper and call it at the end of `_apply_fitted_loo_priors`.
- **Test:** load each bundle and assert `len(q_no_td_modify.mu_0) == P_RL_MICRO` and `len(reward.mu_0) == P_RL_REWARDSHAPING` at configure time, not at first matmul.

### N5. LOW — stale metadata inside the regenerated params JSONs
- All 31 `env_para_vanilla/params_env_*.json` carry `_query_Jw_effect.apply = "subtract when I_w=0"` — an intermediate convention — while the current code adds when J_w=1 (`impute_query_effect.py:385` now writes `"add J_w * (q @ z); I_w not in model"`). The suffix *values* look like current-estimator output (23/31 per-user contrasts used, 22 distinct suffixes), so this is metadata drift, but regenerate so file provenance matches behavior — especially since N1's fix changes the values anyway.

### N6. LOW — per-user q is nearly unregularized
- `ridge = 1e-2` (`impute_query_effect.py:294`) on per-user two-step fits with as few as ~2–10 contrast weeks; the hard per-user-vs-pooled switch (`:357-361`) adds variance discontinuously. Consider precision-weighted shrinkage of per-user q toward pooled instead of the hard switch.

### Question (please confirm the protocol)
`I_w` now affects **only** the PF observation branch (`outcome_from_packet`: I=1 → observe full weekly CAE, I=0 → observe CAE-short), while J (survey completion) is drawn and applied independent of I. That encodes: *the Sunday check-in happens regardless of the query decision; the query only selects the long vs short battery.* Consistent with that, the PF's cumulative CAE-short rows include all J=1 weeks — including I=1 weeks — which is only correct if the short measure is embedded in the full battery (observed whenever J=1). If instead "no query → no weekly survey at all," the env should force J=0 when I=0 and both the mediator gate and the CAE gate should return to I·J.

---

## 3. What to run, in order

1. Fix N1 (un-gated block in Z, folded into base θ) and decide N2/N3 (lagged-J estimation + pre-transition emission draw is the self-consistent combination).
2. Decide M13 (`day_norm`) so the refit only happens once.
3. Refit: 1 → 2 → 3 → 4 (incl. new impute) → 5 → 6.
4. Regenerate priors: `est_prior.py` + `--loo` per folder (clears N4); add the LOO dim asserts (M5) and narrow H1's except while you're there.
5. Only then re-run experiments; `tune_ste` folders need re-tuning after the env refit (and M4's proxy bias is still worth fixing before recalibrating).

---

## 4. Query-effect fix verification (21:22 tree)

You replaced the two-step with the **single-model joint fit** and I verified the full chain. Everything substantive is resolved:

| Issue | Status |
|---|---|
| N1 (two-step attenuation, ≈(1−p̄) recovery, J=0-arm inflation) | **FIXED by design** — `impute_query_effect.py` deleted (no dangling imports anywhere); the lagged-J block `J_{w−1}·[q0 + qE·E_w + qrb·rb]` is inside all three emission likelihoods (`_jw_query_shift`, applied on the quadrature grid at `4_perceived_utility.py:888/923/958`, and identically in the prediction/residual paths `:1257/1325/1394` and `week_loglik_at_point`). Fit jointly with the E-filter → no baseline absorption, and the Ê trajectories are filtered under the correct emission model. |
| N2 (contemporaneous-vs-lagged timing) | **FIXED** — regressor is `week_present_lastweek` (`J_lag_col`, `:569`), created in `2_combine` as a within-participant weekly shift on the filled 12-slot weekly frame; fallback = previous block's `J_week`, else 0. Week-1's `J_lag = 0` is the physically correct boundary (no prior Sunday intervention at study entry). Matches the simulator exactly: week-k mediators read `wp_all[k]` = last Sunday's completion. |
| N3 (emission from post-transition E) | **FIXED** — `_finalize_week` now captures `e_w` before the transition and draws `gen_tool_U1/U2(e_w)`, `gen_week_present(e_w)` (`experiment.py:1025-1046`), then computes `pu = E_{w+1}`. The whole mapping is now coherent: the Sunday-of-week-w draw is emitted from E_w (script-4's `J_w ~ E_w`), gates CAE_w, pairs with week-w mediators in Ê_{w+1}, and shifts week-(w+1) mediators — one physical story everywhere. |
| N5 (stale apply metadata) | **FIXED in the writer** — `penalized_json_export` emits the correct provenance and `"add J_w * (q @ z); I_w not in model"`. |
| N6 (unshrunk per-user q, hard pooled switch) | **FIXED by design** — q is now three ordinary coefficients per mediator in the per-user penalized fit with pooled warm start and zero ridge centers (`build_penalized_prior_center` explicitly leaves `*_q*` at 0); no special-casing. |

Implementation details verified: packed θ is 50-dim (`_THETA_DIM_BASE`), `unpack_theta` count checks out (6+2+3+3+12+12+12); `make_bounds`' hard-coded log-σ indices [5, 10, 13, 21] are **still correct** because the q coefficients were appended after each block's σ/AR entries (walked the packing by hand); prior centers use the probe-vector method so they survive reordering; the JSON suffix layout (base 9 + `query_Jw_*` 3) is exactly what `vani_env._split_ml_*` splits, so the env, `est_prior`, and the hurdle Δp translation need no changes; old 9-dim files degrade to q = 0 gracefully; `tune_ste` deliberately excludes `query_Jw_*` from every scaling pathway and documents it (`tune_ste.py:167-168`); z-features match between fit and sim (E on the grid ↔ `s["perceivedUtilityLastWeek"]`; row/day `recent_burden` ↔ the burden state, constant within day in both).

Remaining items — all small:

1. **Artifacts still pending (expected).** The `params_env_*.json` on disk were written by an intermediate lagged-J run (meta says "last Sunday J … on this week's PV/FW/PJ", not the current writer's "joint lagged-J in script-4 …"), so the joint-fit output is not on disk yet. Run 4 → 5 → 6 → `est_prior` (+`--loo`) before any experiment (N4/H1/M5 as before).
2. **`wp_all[0] = 1.0` bootstrap** (`experiment.py:335`): simulated week 0 gets the q boost, but the fitted week-1 has `J_lag = 0` (study entry, no prior intervention). Its only remaining role is the week-0 mediator gate, so `0.0` is the faithful choice — unless the RL deployment actually opens with a delivered weekly intervention; your call, one character either way.
3. **Loop-gain guards omit the q path** (low): `_loop_gain_penalty` (`4_perceived_utility.py:1851-1875`) and `tune_ste.loop_gain_Ew` (`:770-809`) compute d(med)/dE without the `qE·J` contribution (and there is now a second, two-week feedback path E_w → J_w → med_{w+1}). With |q| small this barely moves g, but the barrier claims to bound the compound gain — add J∈{0,1} arms (dPV/dE = α₁ + A·α₄ + J·α_qE, etc.) or note the omission next to the formula.
4. The PF-side question from §2 stands (CAE-short assumed observed on all J=1 weeks — fine iff the short measure is embedded in the opened intervention, which your protocol description supports).

---

## 5. STE tuning review — tune_ste.py, ste_vanilla.py, run_tune_ste.sh (21:00 tree)

Scope: the two commands you ran — `apply burden_shift κ=0.4 → env_para_burden_shift_large`, then `calibrate foursc_to_y_shift targets 0.5/0.8 on that folder`. Full read of all three files plus empirical checks against the folders on disk.

### T1. HIGH — the κ=0.4 folder on disk (and on the cluster) was built with the OLD burden_shift semantics; re-apply is NOT optional
- **Evidence:** `env_para_burden_shift_large/ste_tuning.json` records `multipliers: {A_to_ME: 0.4, E_to_MY: −0.4}` with 372 scaled coefficients (12/user) — the old pathway that subtracts κ from the **A×E_w interactions too** (verified per-user: `alpha4_action_by_Ew` shifted −0.188→−0.588 etc.). The current knob is `{A_to_ME_main: κ, E_to_MY: −κ}` (mains only, 9/user), introduced precisely because the old version makes the fatigue addend −κ(1+E_w), which **flips sign for E_w < −1** — sending *raises* engagement in exactly the low-engagement states that matter.
- **Blast radius:** everything stacked on that folder inherits it — `env_para_ste0.5_bs_large_bf`, `env_para_ste0.8_bs_large_bf`, and the `results_ste0.5/0.8_bs_large_bf_loo` experiment outputs.
- **Fix:** after the pipeline refit, rerun command 1 with the current code (`--overwrite` will be needed: same knob name, so `refuse_knob_overwrite` won't block; but the stale `ste_tuning.json` will be replaced correctly), then re-calibrate the stacked folders.

### T2. HIGH — `_pv_hurdle_action_scales` translates subtractive knobs multiplicatively; ratios explode and flip sign
- **File:** `vani_env.py:1081-1111`: `s_a = a3_now / a3_van` (guarded only against |a3_van|<1e-12 and non-finite), then hurdle logistic `coef[5] *= s_a`, `coef[6] *= s_ae`.
- **Measured on the κ=0.4 folder:** s_a ∈ [−6.3, +58.1], median ≈ 0; **|s_a| > 5 for 4 users and sign-flipped for 15 of 31**. uid 333: a3 = −0.007 → −0.407 ⇒ hurdle action slope ×58 — P(PV>0 | A) saturates; uid 33: s_a = −6.3 — direction and magnitude both distorted. The pathology is inherent: an *additive* shift on a Gaussian mean coefficient (a3 ← a3 − κ) cannot be represented as a *multiplicative* rescaling of a logistic slope, and it persists under the current mains-only knob (s_a = (a3v−κ)/a3v still blows up whenever the fitted a3v ≈ 0, which is typical).
- **Consequence:** in every burden_shift-derived folder, the PV-occurrence response to the action is not the intended −κ z-units — for some users it is enormous, for half it points the wrong way. This distorts PV → E_w dynamics differently per arm and per user, so both the proxy STE and the confirmation STE are measured in a DGP that does not match the documented knob.
- **Fix:** translate additive knobs additively, using the machinery already there for the J shift: compute the intended Gaussian action-effect change Δμ_A = (a3_now − a3_van) + (a4_now − a4_van)·E_w and add `A · Δμ_A / (z̄₊ − z₀)` to `p_adj` inside `_pv_hurdle_occurrence` (exactly like `intended`), keeping the fitted hurdle slopes untouched. Keep the ratio path only for multiplicative knobs (`action`, `benefit`, `burden`, `benefit_foursc`'s A→MY part), or clip s_a and warn loudly.
- **Test:** for a subtract knob, assert `gen_pageview_mean(s, A=1) − gen_pageview_mean(s, A=0)` in the tuned env ≈ (vanilla difference − κ) within the hurdle's representable range, per user.

### T3. MEDIUM — your second command is documented (twice) to be infeasible
`foursc_to_y_shift` calibrate to 0.5/0.8 on the burden folder: the module docstring (`tune_ste.py:55-58`) and the shell header (`run_tune_ste.sh:27`) both record that this knob caps at proxy STE ≈ 0.36 (job 41048118; CAE hits its clip limits, and the CAE loop-gain cap binds). `calibrate` will print "closest achievable", skip both targets, exit 1, and submit nothing. The recorded working recipe is `TUNE_KNOB=benefit_foursc` (which is exactly what produced your existing `_bs_large_bf` folders). If you specifically want the foursc-shift mechanism, targets ≤ ~0.3 are the realistic range — verify with `TUNE_PHASE=scan` first.

### T4. MEDIUM (statistical, known & now documented) — the calibration proxy is an upward-biased, different estimand
Proxy = per-user max over arms (Bernoulli 0.5/1.0 ± transferred CQL) of the paired Δ̂ **on the same episodes used to pick the arm**, truncated at 0, ÷ σ̂(never-suggest). Same-sample max ⇒ winner's curse; truncation ⇒ E[max(0,noise)] > 0 near null. The docstrings now state this clearly and the confirmation pipeline measures the honest estimand, so this is an accepted design — but remember that "env_para_ste0.5" names the *proxy* target; papers should quote `ste_vanilla aggregate`. Cheap tightening if you ever want it: split the 100 paired episodes — pick the arm on half, estimate Δ on the other half (removes the max bias at the cost of √2 noise).

### T5. MEDIUM — the transferred-CQL arm (TUNE_DQN_EXP=5 default) predates the pipeline refit
`run_tune_ste.sh:116` defaults the extra proxy arm to `d3rlpy_logs/ste_exp_5` checkpoints, which were trained under the **old** vanilla DGP. After the current refit (query rework + H-fixes) those policies come from a different environment than the "source env" the docstring intends. Retrain exp 5 on the refit vanilla before recalibrating, or set `TUNE_DQN_EXP=''` and calibrate on the Bernoulli grid only (the compat checker will not catch this — it validates the observation map, deliberately not the env parameters).

### T6–T9. LOW
- **T6:** both loop-gain guards (`tune_ste.loop_gain_Ew`, script-4 barrier) omit the `qE·J` contribution to d(med)/dE and the new two-week E→J→med path (see §4.3).
- **T7:** calibrate/scan scratch dirs (`.ste_tune_scratch_*`, one subfolder per κ, each with a full params copy + `df_fit_11week.csv`) are never deleted.
- **T8:** `aggregate_ste` reports the mean user STE with no uncertainty; sd across users/√n (plus the per-user paired SEs already in metadata) would cost one line.
- **T9:** `SUPPORTING_FILES` copies the (currently stale) `rl_priors.json` into every tuned folder — same H1/N4 regeneration requirement applies to tuned folders.

### Verified sound (checked, no issues)
- **ste_vanilla.py procedure:** seed hygiene is clean and asserted disjoint (train 2024+jobid+ep / selection 100000+ / gate 150000+ / test 300000+); checkpoint selection on val seeds, deployment gate on independent seeds (`Δ − c·SE > 0`, paired SE = sd(D)/√n — correct), test on fresh seeds; gated-out users contribute STE exactly 0 via `out[n,1]=out[n,0]` — an honest estimate of the deployed decision rule, with no winner's-curse leakage into the test estimate.
- **CRN pairing:** every arm re-seeds the global RNG per episode (`rd.seed(seed)` in `rollout_total_cae`); `prepare_ste_state_vector` is called on non-DQN arms too, so all arms consume the same pre-decision draws; tune_ste's zero-arm cache is correctly restricted to `sigma_invariant` knobs (burden_shift/foursc shift correctly re-simulate the control arm at every κ).
- **Reward/discount construction:** weekly CAE reward lands on the terminal Saturday slot after `_finalize_week` (matching `run_episode` ordering); `WeeklyDiscountTransitionPicker` (γ within week 1, 0.5 at the boundary) matches `_gamma_dt_micro` and est_prior's FQI; the timeout sentinel with one look-ahead week is the correct d3rlpy encoding of a truncated continuing task.
- **No future information:** `known_weekly_cae(k)` = CAE of week k−1 (finalized before week k); the frozen 20-d STE observation is built from `get_context` (strictly-past mediators) after generating the current slot's pre-decision covariates.
- **Metadata gates:** eval now refuses a params_dir mismatch (round-1 L4 fixed); the shell's checkpoint compat checker validates algo + frozen state_dim before enabling the CQL arm; `folder_is_stable` gates auto-submission on `stable: true`; the knob stack is recorded through stacked folders.
- **Stacking mechanics:** `apply` → `calibrate --params-dir <applied folder>` composes correctly (pathway sets disjoint; supporting files and the loo_priors symlink propagate; `refuse_knob_overwrite` and per-target suffixed output dirs prevent clobbering).
- **Root finder:** secant + Illinois false-position with a shared response curve across targets, stability-capped bracket, and same-seed evaluations (smooth f(κ)) — sensible and robust; `largest_stable_kappa` bisection is valid because every gain is affine in κ (the stable set is an interval containing 0).

### Suggested order of operations for the large-fatigue stack
1. Finish the pipeline refit (4→5→6, priors) — everything below re-derives from it.
2. Fix T2 (hurdle translation for subtract knobs); optionally add the J arm to the loop-gain guards (T6).
3. Re-run command 1 (`apply burden_shift κ=0.4`) with the current code — do not reuse the cluster folder (T1).
4. Retrain the vanilla CQL arm (exp 5) or set `TUNE_DQN_EXP=''` (T5).
5. Calibrate with `TUNE_KNOB=benefit_foursc` for 0.5/0.8 (T3); use `foursc_to_y_shift` only via `scan` if you want its mechanism at lower targets.
6. Let the auto-submitted `run_ste.sh` confirmation runs define the reported STE (T4).
